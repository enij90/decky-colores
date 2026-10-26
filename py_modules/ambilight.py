import asyncio
import colorsys
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time
from dataclasses import replace

from run_as_user import user_env, user_cred

logger = logging.getLogger("colores.ambilight")

GAMESCOPE_NODE = "gamescope"
CAP_W = 32
CAP_H = 18
ODIN2_CAP_W = 64
ODIN2_CAP_H = 36

# Seconds between reconnect attempts when the gamescope source is missing or the
# stream drops. On a cold boot the user's PipeWire/gamescope session isn't ready when
# the (root) plugin loads, so the capture must keep retrying instead of giving up —
# otherwise ambient mode stays dark until the user manually re-selects it.
RETRY_INTERVAL = 3.0
MAX_RETRY_INTERVAL = 30.0
STABLE_CAPTURE_INTERVAL = 15.0
MAX_CAPTURE_FPS = 60
# gamescope exposes one PipeWire stream and sizes it for its consumers: while
# our small capture is linked, another reader (Steam Remote Play) gets our tiny
# frames. Step aside while anyone else reads the node, and come back
# SHARED_RESUME_DELAY seconds after the last one leaves.
OWN_NODE_NAME = "colores-ambilight"
SHARED_CHECK_INTERVAL = 1.0
SHARED_RESUME_DELAY = 2.0
# pw_capture.py hands the stream back at full size on SIGTERM (see there).
RELEASE_TIMEOUT = 2.0

_NATIVE_ENV_KEYS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "GST_PLUGIN_PATH",
    "GST_PLUGIN_SYSTEM_PATH",
)
_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "arm64"
# pw_capture.py asks gamescope for small frames at the ambilight rate; pipewiresrc
# takes a full-size frame on every refresh. It needs the host Python and libpipewire.
PW_CAPTURE = Path(__file__).resolve().parent / "pw_capture.py"
_PIPEWIRE_LIBS = ("lib64/libpipewire-0.3.so.0", "lib/libpipewire-0.3.so.0",
                  "lib/aarch64-linux-gnu/libpipewire-0.3.so.0")
_VENDOR_PW_DUMP = _VENDOR_ROOT / "bin" / "pw-dump"
_VENDOR_GST_DIR = _VENDOR_ROOT / "lib64" / "gstreamer-1.0"
_VENDOR_HASHES = {
    _VENDOR_PW_DUMP: "3a2b9a13842e0338b3551b5a9dbe4a89a544f6e7df8e242d971097a0a6bdda63",
    _VENDOR_GST_DIR / "libgstpipewire.so": "b1227d19e9bcb40941d9fcfebed662e08db6234c16aa1ed1ab74c74e26d5220b",
}

_FULL_REGION = [0.0, 0.0, 1.0, 1.0]
# Optional live tuning of the shared-edge (side light) sampling, re-read when
# it changes: {"width": 0.2, "top": 0.45, "bottom": 1.0, "method": "dominant",
# "falloff": 0.0, "min_share": 0.05}. "falloff" > 0 weighs pixels near the
# screen edge more ((1 - distance) ** falloff across the band).
_TUNE_PATH = Path("/var/tmp/colores-tune.json")
_tune_cache = {"mtime": None, "checked": 0.0, "values": {}}


def edge_tuning():
    now = time.monotonic()
    if now - _tune_cache["checked"] < 1.0:
        return _tune_cache["values"]
    _tune_cache["checked"] = now
    try:
        mtime = _TUNE_PATH.stat().st_mtime
    except OSError:
        _tune_cache.update(mtime=None, values={})
        return {}
    if mtime != _tune_cache["mtime"]:
        try:
            values = json.loads(_TUNE_PATH.read_text())
            if not isinstance(values, dict):
                values = {}
        except (OSError, ValueError):
            values = {}
        _tune_cache.update(mtime=mtime, values=values)
        logger.info("ambilight edge tuning: %s", values)
    return _tune_cache["values"]


def tuned_edge_region(region, tune):
    """Apply width/top/bottom overrides to a left or right edge region."""
    x0, y0, x1, y1 = region
    width = tune.get("width")
    if isinstance(width, (int, float)) and 0 < width <= 0.5:
        if x0 <= 0.0:
            x0, x1 = 0.0, float(width)
        else:
            x0, x1 = 1.0 - float(width), 1.0
    top = tune.get("top")
    if isinstance(top, (int, float)) and 0 <= top < 1:
        y0 = float(top)
    bottom = tune.get("bottom")
    if isinstance(bottom, (int, float)) and y0 < bottom <= 1:
        y1 = float(bottom)
    return [x0, y0, x1, y1]


@dataclass(frozen=True)
class CaptureBackend:
    name: str
    pw_dump: str
    gst_launch: str
    gst_plugin_path: str | None = None
    python: str | None = None


def _native_bin(name):
    host_bin = Path("/run/host/usr/bin")
    if host_bin.is_dir():
        host_path = host_bin / name
        if host_path.is_file() and os.access(host_path, os.X_OK):
            return str(host_path)
        return None
    local_path = Path("/usr/bin") / name
    if local_path.is_file() and os.access(local_path, os.X_OK):
        return str(local_path)
    return None


def _native_env(runtime_dir=None, gst_plugin_path=None):
    env = user_env(runtime_dir) if runtime_dir else os.environ.copy()
    for key in _NATIVE_ENV_KEYS:
        env.pop(key, None)
    if gst_plugin_path:
        env["GST_PLUGIN_PATH"] = str(gst_plugin_path)
    return env


def _host_architecture():
    uname = _native_bin("uname")
    if uname:
        try:
            result = subprocess.run(
                [uname, "-m"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                env=_native_env(),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().lower()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.machine().lower()


def _valid_vendor_files():
    for path, expected in _VENDOR_HASHES.items():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if digest != expected:
            logger.error("ambilight fallback checksum mismatch: %s", path)
            return False
    return os.access(_VENDOR_PW_DUMP, os.X_OK)


def _has_pipewiresrc(gst_inspect, plugin_path=None):
    try:
        result = subprocess.run(
            [gst_inspect, "pipewiresrc"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            env=_native_env(gst_plugin_path=plugin_path),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pw_capture_python():
    python = _native_bin("python3")
    if not python or not PW_CAPTURE.is_file():
        return None
    usr = Path(python).parent.parent
    if not any((usr / lib).exists() for lib in _PIPEWIRE_LIBS):
        return None
    return python


def resolve_capture_backend():
    """Select one coherent native backend, preferring the host over the bundle."""
    backend = _resolve_gst_backend()
    python = _pw_capture_python() if backend else None
    if python:
        return replace(backend, name=backend.name + "+pw_capture", python=python)
    return backend


def _resolve_gst_backend():
    gst_launch = _native_bin("gst-launch-1.0")
    gst_inspect = _native_bin("gst-inspect-1.0")
    system_pw_dump = _native_bin("pw-dump")
    if not gst_launch or not gst_inspect:
        return None

    if system_pw_dump and _has_pipewiresrc(gst_inspect):
        return CaptureBackend("system", system_pw_dump, gst_launch)

    if (
        _host_architecture() in {"aarch64", "arm64"}
        and _valid_vendor_files()
        and _has_pipewiresrc(gst_inspect, _VENDOR_GST_DIR)
    ):
        return CaptureBackend(
            "bundled-arm64",
            str(_VENDOR_PW_DUMP),
            gst_launch,
            str(_VENDOR_GST_DIR),
        )
    return None


def capture_available():
    return resolve_capture_backend() is not None


def subdivide(region, count):
    x0, y0, x1, y1 = region
    if count <= 1:
        return [tuple(region)]
    width = (x1 - x0) / count
    return [(x0 + i * width, y0, x0 + (i + 1) * width, y1) for i in range(count)]


def subdivide_rows(region, count):
    """Split a region into count bands from top to bottom."""
    x0, y0, x1, y1 = region
    if count <= 1:
        return [tuple(region)]
    height = (y1 - y0) / count
    return [(x0, y0 + i * height, x1, y0 + (i + 1) * height) for i in range(count)]


def avg_region(frame, width, height, region):
    x0, y0, x1, y1 = region
    cx0 = max(0, int(x0 * width))
    cx1 = min(width, max(cx0 + 1, int(x1 * width)))
    cy0 = max(0, int(y0 * height))
    cy1 = min(height, max(cy0 + 1, int(y1 * height)))
    r = g = b = n = 0
    for y in range(cy0, cy1):
        base = y * width * 3
        for x in range(cx0, cx1):
            i = base + x * 3
            r += frame[i]
            g += frame[i + 1]
            b += frame[i + 2]
            n += 1
    if n == 0:
        return (0, 0, 0)
    return (r // n, g // n, b // n)


def dominant_region(frame, width, height, region, falloff=0.0, min_share=0.05):
    """Pick the prevalent visible hue instead of washing mixed colors to gray."""
    x0, y0, x1, y1 = region
    cx0 = max(0, int(x0 * width))
    cx1 = min(width, max(cx0 + 1, int(x1 * width)))
    cy0 = max(0, int(y0 * height))
    cy1 = min(height, max(cy0 + 1, int(y1 * height)))
    # Distance from the screen edge the band touches, 0 at the edge.
    from_left = x0 <= 0.0
    span = max(1, cx1 - cx0)
    buckets = [[] for _ in range(12)]
    total_weight = 0.0
    for x in range(cx0, cx1):
        distance = ((x - cx0) if from_left else (cx1 - 1 - x)) / span
        weight = (1.0 - distance) ** falloff if falloff > 0 else 1.0
        for y in range(cy0, cy1):
            total_weight += weight
            i = (y * width + x) * 3
            color = (frame[i], frame[i + 1], frame[i + 2])
            highest = max(color)
            chroma = highest - min(color)
            if highest < 24 or chroma < 32:
                continue
            hue, _, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))
            buckets[min(11, int(hue * 12))].append((color, chroma * weight, weight))

    winner = max(buckets, key=lambda bucket: sum(w for _, _, w in bucket))
    winner_weight = sum(w for _, _, w in winner)
    if len(winner) < 3 or winner_weight < total_weight * min_share:
        return avg_region(frame, width, height, region)

    weight = sum(chroma for _, chroma, _ in winner)
    return tuple(
        round(sum(color[channel] * chroma for color, chroma, _ in winner) / weight)
        for channel in range(3)
    )


def boost_saturation(color, factor):
    r, g, b = color
    gray = r * 0.299 + g * 0.587 + b * 0.114
    return tuple(int(max(0, min(255, gray + (c - gray) * factor))) for c in (r, g, b))


def lerp(current, target, alpha):
    return tuple(int(c + (t - c) * alpha) for c, t in zip(current, target))


def alpha_for(smoothing):
    s = max(0, min(100, smoothing))
    return max(0.04, 1.0 - s / 100.0)


def adaptive_alpha(current, target, base_alpha):
    delta = max(abs(current[index] - target[index]) for index in range(3))
    return max(base_alpha, min(1.0, delta / 128.0))


def _gst_command(gst_launch, node, width, height, max_fps):
    # gamescope sends a full-size frame for every frame it composites (up to
    # the panel refresh rate). Drop the extra frames first and shrink before
    # converting, so only a few small frames per second reach videoconvert.
    size = f"width={width},height={height}"
    return [
        gst_launch, "-q", "pipewiresrc", f"path={int(node)}",
        "!", "videorate", "drop-only=true", f"max-rate={max(1, int(max_fps))}",
        "!", "queue", "leaky=downstream", "max-size-buffers=2",
        "!", "videoscale", "!", f"video/x-raw,{size}",
        "!", "videoconvert", "!", f"video/x-raw,format=RGB,{size}",
        "!", "fdsink", "fd=1",
    ]


def _replace_with_latest(queue, item):
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(item)


async def _read_latest_frames(reader, frame_bytes, queue):
    try:
        while True:
            frame = await reader.readexactly(frame_bytes)
            _replace_with_latest(queue, frame)
    except Exception as error:
        _replace_with_latest(queue, error)


class SourceConsumers:
    """Who reads which PipeWire node, kept up to date from `pw-dump --monitor`."""

    def __init__(self):
        self.links = {}
        self.nodes = {}
        self.clients = {}

    def update(self, objects):
        for obj in objects:
            oid = obj.get("id")
            info = obj.get("info")
            if info is None:
                for table in (self.links, self.nodes, self.clients):
                    table.pop(oid, None)
                continue
            kind = obj.get("type") or ""
            props = info.get("props")
            if kind.endswith(":Link"):
                self.links[oid] = (info.get("output-node-id"), info.get("input-node-id"))
            elif kind.endswith(":Node") and props is not None:
                self.nodes[oid] = (props.get("node.name"), props.get("client.id"))
            elif kind.endswith(":Client") and props is not None:
                self.clients[oid] = props.get("application.process.id")

    def _is_own(self, node, own_pid):
        name, client = self.nodes.get(node, (None, None))
        if name == OWN_NODE_NAME:
            return True
        pid = self.clients.get(client)
        return own_pid is not None and pid is not None and int(pid) == own_pid

    def others(self, source, own_pid=None):
        """Nodes other than our capture that are linked to `source`."""
        return sorted({
            dst for src, dst in self.links.values()
            if src == source and not self._is_own(dst, own_pid)
        })


def _decode_json_stream(buf, decoder, on_value):
    """Feeds every complete top-level JSON value in buf to on_value; returns the rest."""
    while True:
        rest = buf.lstrip()
        if not rest:
            return ""
        try:
            value, end = decoder.raw_decode(rest)
        except ValueError:
            return rest
        on_value(value)
        buf = rest[end:]


class Ambilight:
    def __init__(self, apply_zones, zones, runtime_dir, uid=None, gid=None, layout=None, max_fps=None):
        self._apply = apply_zones
        self._zones = max(1, zones)
        self._runtime_dir = runtime_dir
        self._uid = uid
        self._gid = gid
        self._max_fps = max_fps
        self._layout = layout or [{"name": "Lights", "region": _FULL_REGION, "zones": list(range(self._zones))}]
        self._adaptive_zones = {
            zone
            for group in self._layout
            if group.get("kind") in ("shared-edge", "split-edge")
            for zone in group["zones"]
        }
        self._capture_width = ODIN2_CAP_W if self._adaptive_zones else CAP_W
        self._capture_height = ODIN2_CAP_H if self._adaptive_zones else CAP_H
        self._task = None
        self._proc = None
        self._watch_task = None
        self._consumers = SourceConsumers()
        self._options = {}
        self._active = False
        self.status = "idle"
        self._current = [(0, 0, 0)] * self._zones
        self._targets = [(0, 0, 0)] * self._zones
        self._backend = resolve_capture_backend()
        if self._backend:
            logger.info("ambilight capture backend: %s", self._backend.name)
        else:
            logger.warning("ambilight capture backend unavailable")

    @property
    def running(self):
        return self._task is not None and not self._task.done()

    def _fallback(self):
        # Shown when there's no game source to sample (e.g. the Steam home screen, or a
        # cold boot before the session is up) so the LEDs hold the user's last solid color
        # instead of going dark. Routes through _apply, so brightness/power still apply.
        color = self._options.get("fallback") or (0, 0, 0)
        return [tuple(color)] * self._zones

    def _env(self):
        plugin_path = self._backend.gst_plugin_path if self._backend else None
        return _native_env(self._runtime_dir, plugin_path)

    def _cred(self):
        return user_cred(self._uid, self._gid)

    def _capture_interval(self):
        fps = max(1, int(self._options.get("fps", 10)))
        if self._max_fps is not None:
            fps = min(fps, max(1, int(self._max_fps)))
        return 1.0 / fps

    async def _find_node(self):
        # Async so the retry loop never blocks the event loop while waiting on pw-dump
        # (it runs every RETRY_INTERVAL while the source is missing).
        proc = None
        if self._backend is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self._backend.pw_dump,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._env(),
                **self._cred(),
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            data = json.loads(out)
        except (OSError, ValueError, asyncio.TimeoutError) as error:
            logger.warning("pw-dump failed: %s", error)
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await asyncio.gather(proc.wait(), return_exceptions=True)
            return None
        except asyncio.CancelledError:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await asyncio.gather(proc.wait(), return_exceptions=True)
            raise
        for obj in data:
            props = (obj.get("info") or {}).get("props") or {}
            if props.get("node.name") == GAMESCOPE_NODE and "Video" in str(
                props.get("media.class", "")
            ):
                return obj.get("id")
        return None

    def start(self, options):
        self._options = options or {}
        if self.running:
            self.set_active(True)
            return
        self.stop()
        self.set_active(True)
        logger.info("ambilight worker starting")
        self._task = asyncio.get_event_loop().create_task(self._run())

    def set_active(self, active):
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        if self.running:
            if active:
                self.status = (
                    "running" if self._proc is not None
                    else "no_source" if self._backend is not None
                    else "unavailable"
                )
            else:
                self.status = "paused"
        logger.info("ambilight rendering %s", "resumed" if active else "paused")

    def stop(self):
        self.set_active(False)
        self.status = "idle"
        if self._task is not None:
            logger.info("ambilight worker stop requested")
            self._task.cancel()
            self._task = None
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        self._kill()

    async def stop_and_wait(self):
        task = self._task
        proc = self._proc
        if task is not None:
            logger.info("ambilight worker stopping")
        self.stop()
        pending = [task] if task is not None else []
        if proc is not None:
            pending.append(proc.wait())
        await asyncio.gather(*pending, return_exceptions=True)
        if task is not None or proc is not None:
            logger.info("ambilight worker stopped; capture process reaped")

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            self._proc = None

    async def _watch_consumers(self):
        # One long-lived `pw-dump --monitor`: it prints only what changes, so
        # watching the links costs nothing while the graph is idle.
        while True:
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._backend.pw_dump, "--monitor", "--no-colors",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=self._env(),
                    **self._cred(),
                )
                decoder = json.JSONDecoder()
                buf = ""
                # A new monitor starts with a dump of the whole graph: rebuild from
                # it instead of adding to what an earlier monitor saw. Objects that
                # went away while no monitor ran (our own capture, stopped together
                # with the monitor at suspend) would otherwise stay forever.
                fresh = SourceConsumers()

                def _on_value(value):
                    self._consumers = fresh
                    fresh.update(value if isinstance(value, list) else [value])

                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf = _decode_json_stream(
                        buf + chunk.decode(errors="replace"), decoder, _on_value)
            except OSError as error:
                logger.warning("pw-dump monitor failed: %s", error)
            finally:
                if proc is not None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await asyncio.gather(proc.wait(), return_exceptions=True)
            self._consumers = SourceConsumers()
            await asyncio.sleep(RETRY_INTERVAL)

    def _shared(self, node, proc=None):
        return self._consumers.others(node, proc.pid if proc is not None else None)

    async def _release(self, proc):
        if not self._backend.python or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), RELEASE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("ambilight capture did not release the stream in time")

    async def _wait_until_unshared(self, node):
        logger.info("ambilight paused: gamescope node %s also read by %s",
                    node, self._shared(node))
        self.status = "shared" if self._active else "paused"
        if self._active:
            self._apply(self._fallback())
        quiet_since = None
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(SHARED_CHECK_INTERVAL)
            if self._shared(node):
                quiet_since = None
            elif quiet_since is None:
                quiet_since = loop.time()
            elif loop.time() - quiet_since >= SHARED_RESUME_DELAY:
                logger.info("ambilight resuming: gamescope node %s is ours again", node)
                return

    async def _run(self):
        # Outer reconnect loop: keep trying to find the gamescope source and capture it
        # until stop() cancels us. The source can be absent at boot (session not up yet)
        # or vanish (leaving Game Mode) and reappear — we recover from both automatically.
        frame_bytes = self._capture_width * self._capture_height * 3
        retry_delay = RETRY_INTERVAL
        while True:
            if self._backend is None:
                self._backend = resolve_capture_backend()
                if self._backend is None:
                    self.status = "unavailable" if self._active else "paused"
                    if self._active:
                        self._apply(self._fallback())
                    await asyncio.sleep(RETRY_INTERVAL)
                    continue
                logger.info("ambilight capture backend: %s", self._backend.name)
            if self._watch_task is None and self._backend.pw_dump:
                self._watch_task = asyncio.create_task(self._watch_consumers())
            node = await self._find_node()
            if node is None:
                logger.warning("gamescope PipeWire node not found; retrying")
                self.status = "no_source" if self._active else "paused"
                if self._active:
                    self._apply(self._fallback())
                await asyncio.sleep(RETRY_INTERVAL)
                continue

            if self._shared(node):
                await self._wait_until_unshared(node)
                continue

            interval = self._capture_interval()
            if self._backend.python:
                command = [
                    self._backend.python, str(PW_CAPTURE), str(node),
                    str(self._capture_width), str(self._capture_height),
                    str(round(1.0 / interval)),
                ]
            else:
                command = _gst_command(
                    self._backend.gst_launch,
                    node,
                    self._capture_width,
                    self._capture_height,
                    self._max_fps or MAX_CAPTURE_FPS,
                )
            proc = None
            reader_task = None
            shared = False
            logger.info("ambilight start: node=%s fps=%.0f", node, 1.0 / interval)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._env(),
                    **self._cred(),
                )
                self._proc = proc
                self.status = "running" if self._active else "paused"
                frames = asyncio.Queue(maxsize=1)
                reader_task = asyncio.create_task(
                    _read_latest_frames(proc.stdout, frame_bytes, frames)
                )
                capture_started = asyncio.get_running_loop().time()
                shared = False
                while True:
                    try:
                        frame = await asyncio.wait_for(frames.get(), SHARED_CHECK_INTERVAL)
                    except asyncio.TimeoutError:
                        frame = None
                    if self._shared(node, proc):
                        shared = True
                        await self._release(proc)
                        break
                    if frame is None:
                        # gamescope sends no frame while the screen is still: keep
                        # applying the last colors, so the lights finish their fade
                        # and a device that came back dark (replugged, reflashed)
                        # gets them again.
                        if self._active:
                            self._tick()
                        continue
                    if isinstance(frame, Exception):
                        raise frame
                    if (
                        retry_delay > RETRY_INTERVAL
                        and asyncio.get_running_loop().time() - capture_started
                        >= STABLE_CAPTURE_INTERVAL
                    ):
                        retry_delay = RETRY_INTERVAL
                    if self._active:
                        self._update_targets(frame)
                        self._tick()
                    await asyncio.sleep(self._capture_interval())
            except asyncio.IncompleteReadError:
                self.status = "no_source" if self._active else "paused"
                await self._log_exit(proc)
                if self._active:
                    self._apply(self._fallback())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ambilight loop failed")
            finally:
                if reader_task is not None:
                    reader_task.cancel()
                    await asyncio.gather(reader_task, return_exceptions=True)
                if proc is not None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await asyncio.gather(proc.wait(), return_exceptions=True)
                if self._proc is proc:
                    self._proc = None
            if shared:
                await self._wait_until_unshared(node)
                continue
            logger.warning("ambilight reconnect scheduled in %.1fs", retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_INTERVAL)

    async def _log_exit(self, proc):
        if proc is None:
            return
        err = b""
        try:
            err = await proc.stderr.read()
        except (OSError, ValueError):
            pass
        logger.warning(
            "ambilight stream ended (rc=%s): %s",
            proc.returncode,
            err.decode(errors="replace")[:300],
        )

    def _update_targets(self, frame):
        sat = float(self._options.get("saturation", 1.4))
        if self._options.get("global_color"):
            target = boost_saturation(
                avg_region(frame, self._capture_width, self._capture_height, _FULL_REGION),
                sat,
            )
            self._targets = [target] * self._zones
            return
        bottom_edge = self._options.get("sampling") == "bottom_edge"
        for group in self._layout:
            indices = group["zones"]
            region = group["region"]
            if group.get("kind") == "shared-full":
                target = boost_saturation(
                    avg_region(
                        frame,
                        self._capture_width,
                        self._capture_height,
                        region,
                    ),
                    sat,
                )
                for zone in indices:
                    if 0 <= zone < self._zones:
                        self._targets[zone] = target
                continue
            if group.get("kind") in ("shared-edge", "split-edge"):
                tune = edge_tuning()
                region = tuned_edge_region(region, tune)
                # One band for the whole strip, or one per LED down the edge.
                if group.get("kind") == "split-edge":
                    bands = list(zip(subdivide_rows(region, len(indices)), indices))
                else:
                    bands = [(region, None)]
                for band, band_zone in bands:
                    if tune.get("method") == "average":
                        sampled = avg_region(
                            frame, self._capture_width, self._capture_height, band
                        )
                    else:
                        sampled = dominant_region(
                            frame,
                            self._capture_width,
                            self._capture_height,
                            band,
                            falloff=float(tune.get("falloff", group.get("falloff", 0.0)) or 0.0),
                            min_share=float(tune.get("min_share", 0.05)),
                        )
                    target = boost_saturation(sampled, sat)
                    for zone in (indices if band_zone is None else [band_zone]):
                        if 0 <= zone < self._zones:
                            self._targets[zone] = target
                continue
            if bottom_edge:
                x0, y0, x1, y1 = region
                region = (x0, y1 - (y1 - y0) * 0.28, x1, y1)
            for sub, zone in zip(subdivide(region, len(indices)), indices):
                if 0 <= zone < self._zones:
                    self._targets[zone] = boost_saturation(
                        avg_region(
                            frame,
                            self._capture_width,
                            self._capture_height,
                            sub,
                        ),
                        sat,
                    )

    def _tick(self):
        if not self._active:
            return
        base_alpha = alpha_for(self._options.get("smoothing", 75))
        self._current = [
            lerp(
                current,
                target,
                adaptive_alpha(current, target, base_alpha)
                if zone in self._adaptive_zones
                else base_alpha,
            )
            for zone, (current, target) in enumerate(zip(self._current, self._targets))
        ]
        self._apply(list(self._current))
