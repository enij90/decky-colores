import json
import sys
import threading
import time
import types

import pytest

import armada_rgb
from armada_rgb import (
    ArmadaProfileError,
    ArmadaRgbDevice,
    GcmHidRgbDevice,
    build_armada_device,
    gcm_key_packet,
    gcm_strip_packet,
    catalog_for_root,
    layout_for_profile,
    profile_for_model,
    validate_catalog,
)


def _make_node(root, name, *, multicolor=False, order="red green blue", maximum=255):
    node = root / name
    node.mkdir(parents=True)
    (node / "brightness").write_text("0")
    (node / "max_brightness").write_text(str(maximum))
    if multicolor:
        (node / "multi_index").write_text(order)
        (node / "multi_intensity").write_text("0 0 0")
    return node


def _channel_backend(zone_count=1):
    targets = []
    for channel in ("red", "green", "blue"):
        targets.extend(f"{channel}={channel}:indicator-{index}" for index in range(zone_count))
    return {"type": "channels", "targets": targets}


def test_bundled_catalog_covers_all_current_armada_models():
    expected = {
        "AYN Odin 2 Portal",
        "AYN Odin 2",
        "AYN Odin 3",
        "AYN Thor",
        "AYN Thor Lite",
        "Retroid Pocket 5",
        "Retroid Pocket 5 Visionox",
        "Retroid Pocket Flip2",
        "Retroid Pocket Flip2 Visionox",
        "Retroid Pocket 6",
        "Retroid Pocket 6 TOP-DPAD",
        "Retroid Pocket Nova",
        "KONKR Pocket FIT Elite",
        "MANGMI Air Y Pro",
        "MANGMI Pocket Max",
        "Lenovo Legion Y700 (2025) / TB321FU",
    }
    catalog = catalog_for_root("/path/that/does/not/exist")
    actual = {model for profile in catalog["profiles"] for model in profile["models"]}
    assert actual == expected


def test_system_catalog_takes_precedence(tmp_path):
    path = tmp_path / "usr/share/armada-rgb/profiles.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": 1,
        "profiles": [{
            "models": ["Future Handheld"],
            "backend": {"type": "multicolor", "targets": ["future:rgb"]},
        }],
    }))

    assert profile_for_model("Future Handheld", str(tmp_path))["models"] == ["Future Handheld"]
    assert profile_for_model("AYN Odin 3", str(tmp_path)) is None


def test_invalid_system_catalog_falls_back_to_bundled(tmp_path, caplog):
    path = tmp_path / "usr/share/armada-rgb/profiles.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 2, "profiles": []}')

    assert profile_for_model("AYN Odin 3", str(tmp_path)) is not None
    assert "Ignoring invalid Armada RGB catalog" in caplog.text


@pytest.mark.parametrize(
    "catalog",
    [
        {"version": 2, "profiles": []},
        {"version": True, "profiles": []},
        {"version": 1, "profiles": [], "extra": True},
        {
            "version": 1,
            "profiles": [{
                "models": ["Unsafe"],
                "backend": {"type": "multicolor", "targets": ["../brightness"]},
            }],
        },
        {
            "version": 1,
            "profiles": [{
                "models": ["Uneven"],
                "backend": {
                    "type": "channels",
                    "targets": ["red=r:1", "green=g:1", "blue=b:1", "red=r:2"],
                },
            }],
        },
        {
            "version": 1,
            "profiles": [
                {"models": ["Duplicate"], "backend": {"type": "multicolor", "targets": ["rgb:a"]}},
                {"models": ["Duplicate"], "backend": {"type": "multicolor", "targets": ["rgb:b"]}},
            ],
        },
    ],
)
def test_catalog_validation_rejects_unsafe_or_ambiguous_data(catalog):
    with pytest.raises(ArmadaProfileError):
        validate_catalog(catalog)


def test_channel_backend_matches_armada_gamma_and_brightness_vector(tmp_path):
    backend = _channel_backend()
    for target in backend["targets"]:
        _make_node(tmp_path, target.split("=", 1)[1])
    device = ArmadaRgbDevice(str(tmp_path), backend)

    assert device.available is True
    assert device.apply_solid((255, 128, 0), 25, True) is True
    assert (tmp_path / "red:indicator-0/brightness").read_text().strip() == "64"
    assert (tmp_path / "green:indicator-0/brightness").read_text().strip() == "14"
    assert (tmp_path / "blue:indicator-0/brightness").read_text().strip() == "0"


def test_channel_backend_preserves_independent_zones(tmp_path):
    backend = _channel_backend(zone_count=2)
    for target in backend["targets"]:
        _make_node(tmp_path, target.split("=", 1)[1])
    device = ArmadaRgbDevice(str(tmp_path), backend)

    assert device.apply_zones([(255, 0, 0), (0, 255, 0)], 100, True) is True
    assert (tmp_path / "red:indicator-0/brightness").read_text().strip() == "255"
    assert (tmp_path / "green:indicator-0/brightness").read_text().strip() == "0"
    assert (tmp_path / "red:indicator-1/brightness").read_text().strip() == "0"
    assert (tmp_path / "green:indicator-1/brightness").read_text().strip() == "255"


def test_multicolor_backend_uses_node_order_maximum_and_per_zone_color(tmp_path):
    _make_node(tmp_path, "rgb:left", multicolor=True, order="blue green red", maximum=100)
    _make_node(tmp_path, "rgb:right", multicolor=True, order="red green blue", maximum=200)
    backend = {"type": "multicolor", "targets": ["rgb:left", "rgb:right"]}
    device = ArmadaRgbDevice(str(tmp_path), backend)

    assert device.apply_zones([(255, 0, 0), (0, 255, 0)], 50, True) is True
    assert (tmp_path / "rgb:left/multi_intensity").read_text().strip() == "0 0 100"
    assert (tmp_path / "rgb:left/brightness").read_text().strip() == "50"
    assert (tmp_path / "rgb:right/multi_intensity").read_text().strip() == "0 200 0"
    assert (tmp_path / "rgb:right/brightness").read_text().strip() == "100"


def test_konkr_correction_only_triggers_when_red_is_present(tmp_path):
    _make_node(tmp_path, "konkr:rgb:joysticks", multicolor=True)
    profile = profile_for_model("KONKR Pocket FIT Elite", str(tmp_path))
    device = ArmadaRgbDevice(str(tmp_path), profile["backend"], profile["correction"])

    assert device.apply_solid((255, 255, 255), 100, True) is True
    assert (tmp_path / "konkr:rgb:joysticks/multi_intensity").read_text().strip() == "255 154 154"
    assert device.apply_solid((0, 255, 255), 100, True) is True
    assert (tmp_path / "konkr:rgb:joysticks/multi_intensity").read_text().strip() == "0 255 255"


def test_partial_write_failure_blanks_every_multicolor_target(tmp_path, monkeypatch):
    for name in ("rgb:left", "rgb:right"):
        node = _make_node(tmp_path, name, multicolor=True)
        (node / "brightness").write_text("255")
    device = ArmadaRgbDevice(
        str(tmp_path),
        {"type": "multicolor", "targets": ["rgb:left", "rgb:right"]},
    )
    real_write = armada_rgb._write_open_file
    failed = False

    def fail_once(handle, value):
        nonlocal failed
        if value == "0 255 0" and not failed:
            failed = True
            raise OSError("simulated sysfs failure")
        real_write(handle, value)

    monkeypatch.setattr(armada_rgb, "_write_open_file", fail_once)

    assert device.apply_zones([(255, 0, 0), (0, 255, 0)], 100, True) is False
    assert failed is True
    assert all((tmp_path / name / "brightness").read_text().strip() == "0" for name in ("rgb:left", "rgb:right"))


def test_power_off_blanks_channel_targets_without_touching_unrelated_leds(tmp_path):
    backend = _channel_backend()
    for target in backend["targets"]:
        node = _make_node(tmp_path, target.split("=", 1)[1])
        (node / "brightness").write_text("255")
    power_led = _make_node(tmp_path, "power-led")
    (power_led / "brightness").write_text("1")
    device = ArmadaRgbDevice(str(tmp_path), backend)

    assert device.apply_solid((255, 255, 255), 100, False) is True
    for target in backend["targets"]:
        name = target.split("=", 1)[1]
        assert (tmp_path / name / "brightness").read_text().strip() == "0"
    assert (power_led / "brightness").read_text() == "1"


def test_known_side_layouts_share_ambilight_by_side():
    profile = profile_for_model("AYN Odin 3")
    layout = layout_for_profile("AYN Odin 3", profile, 8)

    assert layout[0]["zones"] == [0, 1, 2, 3]
    assert layout[1]["zones"] == [4, 5, 6, 7]
    assert {group["kind"] for group in layout} == {"shared-edge"}
    assert layout[0]["region"] == [0.0, 0.45, 0.20, 1.0]
    assert layout[1]["region"] == [0.80, 0.45, 1.0, 1.0]


def test_tb321fu_samples_a_narrow_band_weighted_to_the_edge():
    model = "Lenovo Legion Y700 (2025) / TB321FU"
    layout = layout_for_profile(model, profile_for_model(model), 2)

    assert layout[0]["region"] == [0.0, 0.45, 0.10, 1.0]
    assert layout[1]["region"] == [0.90, 0.45, 1.0, 1.0]
    assert {group["falloff"] for group in layout} == {1.5}


def test_odin3_eight_zone_channel_frames_are_serialized(tmp_path, monkeypatch):
    profile = profile_for_model("AYN Odin 3")
    for target in profile["backend"]["targets"]:
        _make_node(tmp_path, target.split("=", 1)[1])
    device = ArmadaRgbDevice(str(tmp_path), profile["backend"])
    second_device = ArmadaRgbDevice(str(tmp_path), profile["backend"])
    assert device.zone_count == 8
    assert len(profile["backend"]["targets"]) == 24
    original_write = armada_rgb._write_open_file
    guard = threading.Lock()
    active = 0
    maximum_active = 0

    def tracked_write(handle, value):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.001)
        try:
            original_write(handle, value)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(armada_rgb, "_write_open_file", tracked_write)
    frame_a = [(255, 0, 0)] * 8
    frame_b = [(0, 0, 255)] * 8
    results = []
    workers = [
        threading.Thread(target=lambda frame=frame: results.append(
        (device if frame is frame_a else second_device).apply_zones(frame, 100, True)
        ))
        for frame in (frame_a, frame_b)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert results == [True, True]
    assert maximum_active == 1


def test_unknown_geometry_uses_safe_global_ambilight_layout():
    profile = profile_for_model("MANGMI Pocket Max")
    layout = layout_for_profile("MANGMI Pocket Max", profile, 16)

    assert layout == [{
        "name": "Lights",
        "region": [0.0, 0.0, 1.0, 1.0],
        "zones": list(range(16)),
        "kind": "shared-full",
    }]


def test_unsupported_backend_skips_only_that_profile(tmp_path):
    path = tmp_path / "usr/share/armada-rgb/profiles.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": 1,
        "profiles": [
            {"models": ["Serial Handheld"], "backend": {"type": "serial", "device": "ttyHS2"}},
            {"models": ["Future Handheld"], "backend": {"type": "multicolor", "targets": ["future:rgb"]}},
        ],
    }))

    assert profile_for_model("Serial Handheld", str(tmp_path)) is None
    assert profile_for_model("Future Handheld", str(tmp_path))["models"] == ["Future Handheld"]


@pytest.mark.parametrize(
    "backend",
    [
        {"type": "gcmhid", "vendor": "3537", "product": "1134"},
        {"type": "gcmhid", "vendor": "35x7", "product": "1134", "strips": [1]},
        {"type": "gcmhid", "vendor": "3537", "product": "1134", "strips": []},
        {"type": "gcmhid", "vendor": "3537", "product": "1134", "strips": [1, 1]},
        {"type": "gcmhid", "vendor": "3537", "product": "1134", "strips": [255]},
    ],
)
def test_gcmhid_backend_validation(backend):
    with pytest.raises(ArmadaProfileError):
        validate_catalog({"version": 1, "profiles": [{"models": ["G9"], "backend": backend}]})


def _gcm_sysfs(tmp_path, name="hidraw1", hid_id="HID_ID=0003:00003537:00001134", descriptor=b"\x06\x7a\xff\x09\x01"):
    device = tmp_path / "sys/class/hidraw" / name / "device"
    device.mkdir(parents=True)
    (device / "uevent").write_text(f"DRIVER=hid-generic\n{hid_id}\nHID_NAME=Legion Gaming Controller G9\n")
    (device / "report_descriptor").write_bytes(descriptor)
    node = tmp_path / "dev" / name
    node.parent.mkdir(exist_ok=True)
    node.write_bytes(b"")
    return node


_G9 = {"type": "gcmhid", "vendor": "3537", "product": "1134", "strips": [1, 2]}


def test_gcm_packet_matches_the_controller_protocol():
    # Left strip red, solid: the packet read back from a G9.
    assert gcm_strip_packet(1, [0, 100, 100], 0x01) == bytes(
        [0x00, 0x05, 0x0C, 0x0C, 0x01, 0x01, 0x00, 0x64, 0x64, 0x01, 0x80, 0xFF, 0x67]
    )


def test_gcmhid_finds_the_vendor_interface_and_writes_each_strip(tmp_path):
    _gcm_sysfs(tmp_path, "hidraw0", descriptor=b"\x05\x01\x09\x05")  # gamepad interface
    node = _gcm_sysfs(tmp_path, "hidraw1")
    device = GcmHidRgbDevice(_G9, str(tmp_path))

    assert device.available and device.zone_count == 2 and device.supports_per_zone()
    assert device.apply_zones([(255, 0, 0), (0, 0, 255)], 100, True)
    data = node.read_bytes()
    assert data[:13] == gcm_strip_packet(1, [0, 100, 100], 0x01)
    assert data[13:26] == gcm_strip_packet(2, [170, 100, 100], 0x01)
    assert data[26:] == gcm_key_packet(0xC2, [170, 100, 100])


def test_gcm_key_packet_matches_the_legion_app():
    # 05 08 0A 01 C2 H S B 01 plus the byte sum, after the report ID.
    assert gcm_key_packet(0xC2, [0, 100, 100]) == bytes(
        [0x00, 0x05, 0x08, 0x0A, 0x01, 0xC2, 0x00, 0x64, 0x64, 0x01, 0xA3]
    )


def test_gcmhid_legion_key_follows_the_right_strip_when_off_and_in_effects(tmp_path):
    node = _gcm_sysfs(tmp_path)
    device = GcmHidRgbDevice(_G9, str(tmp_path))
    assert device.apply_zones([(255, 0, 0)], 100, False)
    assert node.read_bytes()[26:] == gcm_key_packet(0xC2, [0, 0, 0])
    node.write_bytes(b"")
    assert device.apply_hardware_effect("rainbow", (255, 0, 0), 0, 50, True)
    assert node.read_bytes()[26:] == gcm_key_packet(0xC2, [0, 100, 50])


def test_gcmhid_brightness_off_and_effects(tmp_path):
    node = _gcm_sysfs(tmp_path)
    device = GcmHidRgbDevice(_G9, str(tmp_path))

    device.apply_solid((0, 255, 0), 50, True)
    assert node.read_bytes()[:13] == gcm_strip_packet(1, [85, 100, 50], 0x01)
    node.write_bytes(b"")
    device.apply_solid((0, 255, 0), 50, False)
    assert node.read_bytes()[:13] == gcm_strip_packet(1, [0, 0, 0], 0xFF)
    node.write_bytes(b"")
    device.apply_hardware_effect("breathing", (255, 0, 0), 100, 100, True)
    assert node.read_bytes()[:13] == gcm_strip_packet(1, [0, 100, 100], 0x02, 0xFF, 0xFF)
    node.write_bytes(b"")
    device.apply_hardware_effect("rainbow", (255, 0, 0), 0, 50, True)
    assert node.read_bytes()[:13] == gcm_strip_packet(1, [0, 100, 50], 0x04, 0x00, 0x80)


def test_gcmhid_reports_a_missing_controller(tmp_path):
    _gcm_sysfs(tmp_path, hid_id="HID_ID=0003:0000045E:0000028E")
    device = GcmHidRgbDevice(_G9, str(tmp_path))

    assert not device.available
    assert not device.apply_solid((255, 255, 255), 100, True)
    assert device.last_error == "controller not connected"


def test_build_armada_device_returns_gcmhid_device(tmp_path):
    path = tmp_path / "usr/share/armada-rgb/profiles.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "profiles": [{"models": ["G9 Tablet"], "backend": _G9}]}))

    profile, device = build_armada_device("G9 Tablet", str(tmp_path / "sys/class/leds"), str(tmp_path))
    assert profile["backend"]["type"] == "gcmhid"
    assert isinstance(device, GcmHidRgbDevice)


def test_gcm_perled_packet_sets_one_led_by_speed_byte():
    # Channel 9 (middle of the left grip) green: strip 0, speed 0xE0 + channel.
    payload = [0x05, 0x0C, 0x0C, 0x01, 0x00, 85, 100, 100, 0x01, 0xE9, 0xFF]
    assert armada_rgb.gcm_perled_packet(9, (85, 100, 100)) == bytes([0x00, *payload, sum(payload) & 0xFF])
    # B=0 is ignored by the firmware, so a dark LED is sent as B=1.
    assert armada_rgb.gcm_perled_packet(0, (0, 0, 0))[8] == 1


def test_g9_perled_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("COLORES_G9_PERLED", raising=False)
    monkeypatch.delenv("DECKY_PLUGIN_SETTINGS_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "decky", types.SimpleNamespace(DECKY_PLUGIN_SETTINGS_DIR=str(tmp_path / "none")))
    assert not armada_rgb.g9_perled_enabled()
    monkeypatch.setitem(sys.modules, "decky", types.SimpleNamespace(DECKY_PLUGIN_SETTINGS_DIR=str(tmp_path)))
    assert not armada_rgb.g9_perled_enabled(str(tmp_path))
    (tmp_path / "g9-perled").write_text("")
    assert armada_rgb.g9_perled_enabled(str(tmp_path))
    assert armada_rgb.g9_perled_enabled()  # through decky.DECKY_PLUGIN_SETTINGS_DIR
    monkeypatch.setenv("COLORES_G9_PERLED", "1")
    assert armada_rgb.g9_perled_enabled()


def test_gcmhid_perled_sends_only_changed_leds_and_refreshes(tmp_path, monkeypatch):
    node = _gcm_sysfs(tmp_path, "hidraw1")
    clock = [100.0]
    monkeypatch.setattr(armada_rgb.time, "monotonic", lambda: clock[0])
    device = GcmHidRgbDevice(_G9, str(tmp_path), perled=True)
    assert device.zone_count == 10 and device.supports_per_zone()

    colors = [(255, 0, 0)] * 10
    colors[2] = (0, 255, 0)
    assert device.apply_zones(colors, 100, True)
    data = node.read_bytes()
    assert len(data) == 10 * 13
    # Zone 0 is the top of the left grip (channel 7), zone 5 the top of the right one (channel 4).
    assert data[:13] == armada_rgb.gcm_perled_packet(7, (0, 100, 100))
    assert data[2 * 13:3 * 13] == armada_rgb.gcm_perled_packet(9, (85, 100, 100))
    assert data[5 * 13:6 * 13] == armada_rgb.gcm_perled_packet(4, (0, 100, 100))

    clock[0] += 0.1
    assert device.apply_zones(colors, 100, True)
    assert len(node.read_bytes()) == 10 * 13  # unchanged: nothing sent

    colors[9] = (0, 0, 255)
    clock[0] += 0.1
    assert device.apply_zones(colors, 100, True)
    data = node.read_bytes()
    assert data[10 * 13:] == armada_rgb.gcm_perled_packet(0, (170, 100, 100))

    clock[0] += 3.0  # past the refresh interval: every LED again
    assert device.apply_zones(colors, 100, True)
    assert len(node.read_bytes()) == 21 * 13


def test_layout_splits_the_edge_per_led_only_with_perled():
    model = "Lenovo Legion Y700 (2025) / TB321FU"
    perled = armada_rgb.layout_for_profile(model, {"perled": True}, 10)
    assert [g["kind"] for g in perled] == ["split-edge", "split-edge"]
    assert perled[0]["zones"] == [0, 1, 2, 3, 4] and perled[1]["zones"] == [5, 6, 7, 8, 9]
    stock = armada_rgb.layout_for_profile(model, {}, 2)
    assert [g["kind"] for g in stock] == ["shared-edge", "shared-edge"]
