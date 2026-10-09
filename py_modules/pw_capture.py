"""Standalone gamescope PipeWire capture.

Runs on the host Python (native even when Decky itself is emulated) and writes raw RGB
frames of WIDTH x HEIGHT to stdout, the same contract as the GStreamer pipeline.
Usage: pw_capture.py TARGET WIDTH HEIGHT FPS
Exit codes: 0 stdout closed, 2 no source, 3 libpipewire unavailable.

gamescope paints a capture frame on every vblank, at full output size, whenever it
can dequeue a free buffer. This client asks gamescope for a small frame
(SPA_FORMAT_VIDEO_requested_size, a gamescope extension) and keeps every buffer but
one, handing that one back once per frame interval: gamescope then paints FPS small
frames per second instead of a full-size frame per refresh.
"""
import ctypes
import ctypes.util
import operator
import struct
import sys
import threading
import time
import traceback

SPA_TYPE_ID = 3
SPA_TYPE_RECTANGLE = 10
SPA_TYPE_FRACTION = 11
SPA_TYPE_OBJECT = 15
SPA_TYPE_CHOICE = 19
SPA_CHOICE_RANGE = 1
SPA_CHOICE_ENUM = 3
SPA_TYPE_OBJECT_FORMAT = 0x40003
SPA_PARAM_ENUM_FORMAT = 3
SPA_PARAM_FORMAT = 4
SPA_FORMAT_MEDIA_TYPE = 1
SPA_FORMAT_MEDIA_SUBTYPE = 2
SPA_FORMAT_VIDEO_FORMAT = 0x20001
SPA_FORMAT_VIDEO_SIZE = 0x20003
SPA_FORMAT_VIDEO_FRAMERATE = 0x20004
SPA_FORMAT_VIDEO_REQUESTED_SIZE = 0x70000  # gamescope pipewire_gamescope.hpp
SPA_MEDIA_TYPE_VIDEO = 2
SPA_MEDIA_SUBTYPE_RAW = 1

# spa_video_format value -> byte offsets of (r, g, b) inside a 4-byte pixel
PIXEL_LAYOUTS = {
    7: (0, 1, 2),   # RGBx
    8: (2, 1, 0),   # BGRx
    9: (1, 2, 3),   # xRGB
    10: (3, 2, 1),  # xBGR
    11: (0, 1, 2),  # RGBA
    12: (2, 1, 0),  # BGRA
    13: (1, 2, 3),  # ARGB
    14: (3, 2, 1),  # ABGR
}

PW_DIRECTION_INPUT = 0
PW_ID_ANY = 0xFFFFFFFF
PW_STREAM_FLAG_AUTOCONNECT = 1 << 0
PW_STREAM_FLAG_MAP_BUFFERS = 1 << 2
PW_STREAM_STATE_ERROR = -1
PW_STREAM_STATE_UNCONNECTED = 0
PW_STREAM_STATE_STREAMING = 3

SAMPLES = 2
CONNECT_TIMEOUT = 5.0


def _pod(type_, body):
    pad = (-len(body)) % 8
    return struct.pack("<II", len(body), type_) + body + b"\0" * pad


def _choice(kind, child_type, child_size, values):
    body = struct.pack("<IIII", kind, 0, child_size, child_type) + b"".join(values)
    return _pod(SPA_TYPE_CHOICE, body)


def _prop(key, value):
    return struct.pack("<II", key, 0) + value


def enum_format_pod(requested=(64, 40)):
    formats = (8, 7, 12, 11, 9, 10, 13, 14)
    ids = [struct.pack("<I", fmt) for fmt in (formats[0],) + formats]
    rect = [struct.pack("<II", *size) for size in ((64, 36), (1, 1), (8192, 8192))]
    frac = [struct.pack("<II", *rate) for rate in ((30, 1), (0, 1), (1000, 1))]
    props = b"".join((
        _prop(SPA_FORMAT_MEDIA_TYPE, _pod(SPA_TYPE_ID, struct.pack("<I", SPA_MEDIA_TYPE_VIDEO))),
        _prop(SPA_FORMAT_MEDIA_SUBTYPE, _pod(SPA_TYPE_ID, struct.pack("<I", SPA_MEDIA_SUBTYPE_RAW))),
        _prop(SPA_FORMAT_VIDEO_FORMAT, _choice(SPA_CHOICE_ENUM, SPA_TYPE_ID, 4, ids)),
        _prop(SPA_FORMAT_VIDEO_SIZE, _choice(SPA_CHOICE_RANGE, SPA_TYPE_RECTANGLE, 8, rect)),
        _prop(SPA_FORMAT_VIDEO_FRAMERATE, _choice(SPA_CHOICE_RANGE, SPA_TYPE_FRACTION, 8, frac)),
        _prop(SPA_FORMAT_VIDEO_REQUESTED_SIZE, _pod(SPA_TYPE_RECTANGLE, struct.pack("<II", *requested))),
    ))
    return _pod(SPA_TYPE_OBJECT, struct.pack("<II", SPA_TYPE_OBJECT_FORMAT, SPA_PARAM_ENUM_FORMAT) + props)


def _pod_value(data, offset):
    size, type_ = struct.unpack_from("<II", data, offset)
    body = offset + 8
    if type_ == SPA_TYPE_CHOICE:
        _, _, child_size, child_type = struct.unpack_from("<IIII", data, body)
        return child_type, data[body + 16:body + 16 + child_size]
    return type_, data[body:body + size]


def parse_format(data):
    """Returns (video_format, width, height) from a negotiated Format object pod."""
    size, type_ = struct.unpack_from("<II", data, 0)
    if type_ != SPA_TYPE_OBJECT:
        return None
    end = 8 + size
    offset = 16
    found = {}
    while offset + 8 <= end:
        key, _ = struct.unpack_from("<II", data, offset)
        value_size, _ = struct.unpack_from("<II", data, offset + 8)
        found[key] = _pod_value(data, offset + 8)
        offset += 16 + value_size + ((-value_size) % 8)
    fmt = found.get(SPA_FORMAT_VIDEO_FORMAT)
    rect = found.get(SPA_FORMAT_VIDEO_SIZE)
    if not fmt or not rect or len(fmt[1]) < 4 or len(rect[1]) < 8:
        return None
    width, height = struct.unpack_from("<II", rect[1])
    return struct.unpack_from("<I", fmt[1])[0], width, height


def _taps(size, out, samples):
    return [
        [min(size - 1, int((cell + (s + 0.5) / samples) * size / out)) for s in range(samples)]
        for cell in range(out)
    ]


def sample_plan(stride, width, height, out_w, out_h, samples=SAMPLES):
    rows = [[y * stride for y in ys] for ys in _taps(height, out_h, samples)]
    cols = [[x * 4 for x in xs] for xs in _taps(width, out_w, samples)]
    return [[row + col for row in row_taps for col in col_taps] for row_taps in rows for col_taps in cols]


def pick_plan(stride, width, height, out_w, out_h, layout):
    """One source pixel per output cell, as (row offsets, getter of the R, G, B bytes).

    gamescope already scaled the frame down on the GPU, so the cell centres are enough.
    """
    rows = [min(height - 1, int((y + 0.5) * height / out_h)) * stride for y in range(out_h)]
    cols = [min(width - 1, int((x + 0.5) * width / out_w)) * 4 for x in range(out_w)]
    indices = [col + channel for col in cols for channel in layout]
    return rows, operator.itemgetter(*indices), (max(cols) + 4)


def pick(frame, plan):
    rows, getter, span = plan
    return b"".join(bytes(getter(frame[row:row + span])) for row in rows)


def downsample(frame, plan, layout):
    ri, gi, bi = layout
    out = bytearray(len(plan) * 3)
    o = 0
    for points in plan:
        n = len(points)
        out[o] = sum(frame[p + ri] for p in points) // n
        out[o + 1] = sum(frame[p + gi] for p in points) // n
        out[o + 2] = sum(frame[p + bi] for p in points) // n
        o += 3
    return bytes(out)


class _Chunk(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("stride", ctypes.c_int32), ("flags", ctypes.c_int32)]


class _Data(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("fd", ctypes.c_int64),
                ("mapoffset", ctypes.c_uint32), ("maxsize", ctypes.c_uint32),
                ("data", ctypes.c_void_p), ("chunk", ctypes.POINTER(_Chunk))]


class _SpaBuffer(ctypes.Structure):
    _fields_ = [("n_metas", ctypes.c_uint32), ("n_datas", ctypes.c_uint32),
                ("metas", ctypes.c_void_p), ("datas", ctypes.POINTER(_Data))]


class _PwBuffer(ctypes.Structure):
    _fields_ = [("buffer", ctypes.POINTER(_SpaBuffer)), ("user_data", ctypes.c_void_p),
                ("size", ctypes.c_uint64), ("requested", ctypes.c_uint64)]


_VOID = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_STATE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
_PARAM = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
_BUFFER = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class _StreamEvents(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("destroy", _VOID), ("state_changed", _STATE),
                ("control_info", ctypes.c_void_p), ("io_changed", ctypes.c_void_p),
                ("param_changed", _PARAM), ("add_buffer", ctypes.c_void_p),
                ("remove_buffer", _BUFFER), ("process", _VOID), ("drained", ctypes.c_void_p),
                ("command", ctypes.c_void_p), ("trigger_done", ctypes.c_void_p)]


def _load():
    for name in ("libpipewire-0.3.so.0", ctypes.util.find_library("pipewire-0.3")):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def capture(target, out_w, out_h, fps, out=None):
    pw = _load()
    if pw is None:
        return 3
    out = out or sys.stdout.buffer
    vp = ctypes.c_void_p
    pw.pw_thread_loop_new.restype = vp
    pw.pw_thread_loop_new.argtypes = [ctypes.c_char_p, vp]
    for name in ("start", "lock", "unlock", "stop", "destroy"):
        getattr(pw, "pw_thread_loop_" + name).argtypes = [vp]
    pw.pw_thread_loop_get_loop.restype = vp
    pw.pw_thread_loop_get_loop.argtypes = [vp]
    pw.pw_properties_new.restype = vp
    pw.pw_properties_new.argtypes = [vp]
    pw.pw_properties_set.argtypes = [vp, ctypes.c_char_p, ctypes.c_char_p]
    pw.pw_stream_new_simple.restype = vp
    pw.pw_stream_new_simple.argtypes = [vp, ctypes.c_char_p, vp, vp, vp]
    pw.pw_stream_connect.argtypes = [vp, ctypes.c_int, ctypes.c_uint32, ctypes.c_int, vp, ctypes.c_uint32]
    pw.pw_stream_dequeue_buffer.restype = ctypes.POINTER(_PwBuffer)
    pw.pw_stream_dequeue_buffer.argtypes = [vp]
    pw.pw_stream_queue_buffer.argtypes = [vp, vp]
    pw.pw_stream_disconnect.argtypes = [vp]
    pw.pw_stream_destroy.argtypes = [vp]

    pw.pw_init(None, None)
    loop = pw.pw_thread_loop_new(b"colores-ambilight", None)
    props = pw.pw_properties_new(None)
    for key, value in (
        ("media.type", "Video"), ("media.category", "Capture"), ("media.role", "Screen"),
        ("target.object", target), ("node.dont-reconnect", "true"),
    ):
        pw.pw_properties_set(props, key.encode(), value.encode())

    # Shared between the loop thread (callbacks) and the main thread (ticks);
    # the main thread only touches it with the loop lock held.
    state = {"format": None, "code": 0, "streaming": False, "plan": (None, None), "held": []}
    done = threading.Event()
    interval = 1.0 / max(1, fps)

    def quit(code):
        if state["code"] == 0:
            state["code"] = code
        done.set()

    def on_state(_data, _old, new, error):
        if new == PW_STREAM_STATE_ERROR or (new == PW_STREAM_STATE_UNCONNECTED and state["streaming"]):
            quit(2)
        elif new == PW_STREAM_STATE_STREAMING:
            state["streaming"] = True

    def guarded(callback):
        def call(*args):
            try:
                callback(*args)
            except Exception:  # noqa: BLE001
                if state["code"] == 0:
                    traceback.print_exc()
                    quit(2)
        return call

    @guarded
    def on_param(_data, param_id, param):
        if param_id != SPA_PARAM_FORMAT:
            return
        if not param:
            state["format"] = None
            return
        size = struct.unpack("<I", ctypes.string_at(param, 4))[0]
        state["format"] = parse_format(ctypes.string_at(param, size + 8))

    @guarded
    def on_remove_buffer(_data, buf):
        # Renegotiation or teardown takes buffers back: drop the ones we hold.
        state["held"] = [held for held in state["held"] if ctypes.addressof(held.contents) != buf]

    def on_process(_data):
        # Frames are taken on the main thread's tick; filled buffers wait in the stream.
        pass

    def convert(buf):
        fmt = state["format"]
        layout = PIXEL_LAYOUTS.get(fmt[0]) if fmt else None
        spa = buf.contents.buffer.contents
        if layout is None or spa.n_datas < 1:
            return None
        data = spa.datas[0]
        if not data.data or not data.chunk:
            return None
        chunk = data.chunk.contents
        _, width, height = fmt
        stride = chunk.stride or width * 4
        needed = stride * height
        if (chunk.size and chunk.size < needed) or chunk.offset + needed > data.maxsize:
            return None
        frame = (ctypes.c_ubyte * needed).from_address(data.data + chunk.offset)
        key = (stride, width, height, layout)
        if state["plan"][0] != key:
            state["plan"] = (key, pick_plan(stride, width, height, out_w, out_h, layout))
        return pick(memoryview(frame).cast("B"), state["plan"][1])

    def tick(stream):
        # Take every filled buffer, convert the newest and hand back only the oldest:
        # gamescope can paint one more frame before the next tick.
        while True:
            buf = pw.pw_stream_dequeue_buffer(stream)
            if not buf:
                break
            state["held"].append(buf)
        if not state["held"]:
            return None
        pixels = convert(state["held"][-1])
        pw.pw_stream_queue_buffer(stream, state["held"].pop(0))
        return pixels

    events = _StreamEvents()
    events.version = 2
    events.state_changed = _STATE(on_state)
    events.param_changed = _PARAM(on_param)
    events.remove_buffer = _BUFFER(on_remove_buffer)
    events.process = _VOID(on_process)
    pw.pw_thread_loop_lock(loop)
    stream = pw.pw_stream_new_simple(
        pw.pw_thread_loop_get_loop(loop), b"colores-ambilight", props, ctypes.byref(events), None
    )
    pod = ctypes.create_string_buffer(enum_format_pod(requested_size(out_w, out_h)))
    params = (ctypes.c_void_p * 1)(ctypes.addressof(pod))
    connected = pw.pw_stream_connect(
        stream, PW_DIRECTION_INPUT, PW_ID_ANY,
        PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS, params, 1,
    )
    pw.pw_thread_loop_unlock(loop)
    if connected < 0:
        pw.pw_stream_destroy(stream)
        pw.pw_thread_loop_destroy(loop)
        return 2
    pw.pw_thread_loop_start(loop)

    started = time.monotonic()
    while not done.wait(interval):
        pixels = None
        pw.pw_thread_loop_lock(loop)
        try:
            # A still screen sends no frames, so only a stream that never starts counts as no source.
            if state["streaming"]:
                pixels = tick(stream)
            elif time.monotonic() - started > CONNECT_TIMEOUT:
                quit(2)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            quit(2)
        finally:
            pw.pw_thread_loop_unlock(loop)
        if pixels is None:
            continue
        try:
            out.write(pixels)
            out.flush()
        except OSError:
            quit(0)

    pw.pw_thread_loop_lock(loop)
    for buf in state["held"]:
        pw.pw_stream_queue_buffer(stream, buf)
    state["held"] = []
    pw.pw_stream_disconnect(stream)
    pw.pw_stream_destroy(stream)
    pw.pw_thread_loop_unlock(loop)
    pw.pw_thread_loop_stop(loop)
    pw.pw_thread_loop_destroy(loop)
    return state["code"]


def requested_size(out_w, out_h):
    # A few source pixels per output cell keep the averages smooth; gamescope keeps
    # the output aspect ratio inside this box.
    return (out_w * 2, out_h * 2)

if __name__ == "__main__":
    target, width, height, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    sys.exit(capture(target, width, height, fps))
