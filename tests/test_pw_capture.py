import struct

import pw_capture


def _frame(width, height, stride):
    frame = bytearray(stride * height)
    for y in range(height):
        for x in range(width):
            offset = y * stride + x * 4
            frame[offset:offset + 4] = bytes((x, y, 200, 0))
    return frame


def test_pick_takes_the_cell_centres_in_rgb_order():
    width, height = 116, 72
    stride = width * 4 + 16
    plan = pw_capture.pick_plan(stride, width, height, 64, 36, pw_capture.PIXEL_LAYOUTS[8])
    out = pw_capture.pick(memoryview(_frame(width, height, stride)), plan)
    assert len(out) == 64 * 36 * 3
    assert out[:3] == bytes((200, 1, 0))
    assert out[-3:] == bytes((200, 71, 115))


def test_enum_format_requests_a_small_frame_from_gamescope():
    pod = pw_capture.enum_format_pod((128, 72))
    key = struct.pack("<II", pw_capture.SPA_FORMAT_VIDEO_REQUESTED_SIZE, 0)
    at = pod.index(key) + len(key)
    size, type_ = struct.unpack_from("<II", pod, at)
    assert type_ == pw_capture.SPA_TYPE_RECTANGLE
    assert struct.unpack_from("<II", pod, at + 8) == (128, 72)


def test_requested_size_is_twice_the_output():
    assert pw_capture.requested_size(64, 36) == (128, 72)
