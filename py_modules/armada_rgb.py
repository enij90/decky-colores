"""Armada OS RGB profile loading and sysfs backends.

The catalog format intentionally mirrors armada-rgb v1. Colores reads the
system copy when available and keeps a bundled copy so older Armada images can
use the same hardware support without modifying the OS configuration.
"""

import json
import logging
import math
import os
import re
import threading
import time

from led_device import LedDevice


logger = logging.getLogger(__name__)

CATALOG_VERSION = 1
SYSTEM_CATALOG = "usr/share/armada-rgb/profiles.json"
BUNDLED_CATALOG = os.path.join(os.path.dirname(__file__), "armada_rgb_profiles.json")
_SAFE_TARGET = re.compile(r"^[A-Za-z0-9:_.-]+$")
_CHANNELS = ("red", "green", "blue")
_CORRECTION_TRIGGERS = {"always", *_CHANNELS}
_ARMADA_WRITE_LOCK = threading.RLock()


class ArmadaProfileError(ValueError):
    pass


def _exact_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ArmadaProfileError(f"invalid {context} fields")


def _validate_name(name, context="target"):
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or not _SAFE_TARGET.fullmatch(name)
    ):
        raise ArmadaProfileError(f"invalid {context} name")


def _validate_correction(value):
    _exact_keys(value, {"trigger", *_CHANNELS}, "correction")
    if value["trigger"] not in _CORRECTION_TRIGGERS:
        raise ArmadaProfileError("invalid correction trigger")
    for channel in _CHANNELS:
        reduction = value[channel]
        if isinstance(reduction, bool) or not isinstance(reduction, int) or not 0 <= reduction <= 100:
            raise ArmadaProfileError(f"invalid {channel} correction")
    return dict(value)


_HEX_ID = re.compile(r"^[0-9a-fA-F]{4}$")


def _validate_gcmhid(backend):
    _exact_keys(backend, {"type", "vendor", "product", "strips"}, "gcmhid backend")
    for key in ("vendor", "product"):
        if not isinstance(backend[key], str) or not _HEX_ID.fullmatch(backend[key]):
            raise ArmadaProfileError(f"invalid gcmhid {key}")
    strips = backend["strips"]
    if (
        not isinstance(strips, list)
        or not strips
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 < s < 255 for s in strips)
        or len(strips) != len(set(strips))
    ):
        raise ArmadaProfileError("invalid gcmhid strips")
    return {
        "type": "gcmhid",
        "vendor": backend["vendor"].lower(),
        "product": backend["product"].lower(),
        "strips": list(strips),
    }


def validate_catalog(raw):
    _exact_keys(raw, {"version", "profiles"}, "catalog")
    if (
        type(raw["version"]) is not int
        or raw["version"] != CATALOG_VERSION
        or not isinstance(raw["profiles"], list)
    ):
        raise ArmadaProfileError("unsupported Armada RGB catalog version")

    profiles = []
    seen_models = set()
    for raw_profile in raw["profiles"]:
        allowed = {"models", "backend", "correction"}
        if not isinstance(raw_profile, dict) or not set(raw_profile).issubset(allowed):
            raise ArmadaProfileError("invalid profile fields")
        if not {"models", "backend"}.issubset(raw_profile):
            raise ArmadaProfileError("profile is missing required fields")

        models = raw_profile["models"]
        if not isinstance(models, list) or not models:
            raise ArmadaProfileError("profile model list is empty")
        for model in models:
            if not isinstance(model, str) or not model.strip() or model != model.strip():
                raise ArmadaProfileError("invalid model name")
            if model in seen_models:
                raise ArmadaProfileError(f"duplicate model '{model}'")
            seen_models.add(model)

        backend = raw_profile["backend"]
        if not isinstance(backend, dict) or not isinstance(backend.get("type"), str):
            raise ArmadaProfileError("invalid backend fields")
        backend_type = backend["type"]
        if backend_type == "gcmhid":
            profile = {"models": list(models), "backend": _validate_gcmhid(backend)}
            if "correction" in raw_profile:
                profile["correction"] = _validate_correction(raw_profile["correction"])
            profiles.append(profile)
            continue
        if backend_type not in {"multicolor", "channels"}:
            # armada-rgb grows backends Colores does not drive (e.g. serial):
            # skip those devices instead of dropping the whole system catalog.
            logger.info("Skipping Armada RGB profile %s: unsupported backend %r", models, backend_type)
            continue
        _exact_keys(backend, {"type", "targets"}, "backend")
        targets = backend["targets"]
        if not isinstance(targets, list) or not targets:
            raise ArmadaProfileError("RGB target list is empty")

        names = []
        channel_names = {channel: [] for channel in _CHANNELS}
        for target in targets:
            if backend_type == "channels":
                if not isinstance(target, str) or target.count("=") != 1:
                    raise ArmadaProfileError("invalid channel target")
                channel, name = target.split("=", 1)
                if channel not in channel_names:
                    raise ArmadaProfileError("invalid RGB channel")
                _validate_name(name)
                channel_names[channel].append(name)
            else:
                name = target
                _validate_name(name)
            names.append(name)

        if len(names) != len(set(names)):
            raise ArmadaProfileError("duplicate RGB target")
        if backend_type == "channels":
            counts = {len(values) for values in channel_names.values()}
            if len(counts) != 1 or counts == {0}:
                raise ArmadaProfileError("RGB channel lists must have equal lengths")

        profile = {
            "models": list(models),
            "backend": {"type": backend_type, "targets": list(targets)},
        }
        if "correction" in raw_profile:
            profile["correction"] = _validate_correction(raw_profile["correction"])
        profiles.append(profile)
    return {"version": CATALOG_VERSION, "profiles": profiles}


def load_catalog(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return validate_catalog(json.load(handle))
    except (OSError, UnicodeError, json.JSONDecodeError, ArmadaProfileError) as error:
        raise ArmadaProfileError(str(error)) from error


def catalog_for_root(sysfs_root="/"):
    system_path = os.path.join(sysfs_root, SYSTEM_CATALOG)
    if os.path.isfile(system_path):
        try:
            return load_catalog(system_path)
        except ArmadaProfileError as error:
            logger.warning("Ignoring invalid Armada RGB catalog %s: %s", system_path, error)
    return load_catalog(BUNDLED_CATALOG)


def profile_for_model(model, sysfs_root="/"):
    if not model:
        return None
    try:
        catalog = catalog_for_root(sysfs_root)
    except ArmadaProfileError as error:
        logger.error("Cannot load bundled Armada RGB catalog: %s", error)
        return None
    for profile in catalog["profiles"]:
        if model in profile["models"]:
            return profile
    return None


def _read_positive(path):
    with open(path, encoding="ascii") as handle:
        value = int(handle.read().strip())
    if value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def _gamma(channel, maximum):
    value = max(0, min(255, int(channel))) / 255
    linear = value / 12.92 if value <= 0.04045 else math.pow((value + 0.055) / 1.055, 2.4)
    return math.floor(linear * maximum + 0.5)


def _scale(percent, maximum):
    return (max(0, min(100, int(percent))) * maximum + 50) // 100


def _correct(color, correction):
    rgb = [max(0, min(255, int(channel))) for channel in color]
    if not correction:
        return tuple(rgb)
    trigger = correction["trigger"]
    triggered = trigger == "always" or rgb[_CHANNELS.index(trigger)] > 0
    if not triggered:
        return tuple(rgb)
    return tuple((value * (100 - correction[channel]) + 50) // 100 for channel, value in zip(_CHANNELS, rgb))


def _write_open_file(handle, value):
    data = f"{value}\n".encode("ascii")
    offset = 0
    while offset < len(data):
        written = os.write(handle, data[offset:])
        if written <= 0:
            raise OSError("short sysfs write")
        offset += written
    try:
        os.ftruncate(handle, len(data))
    except OSError:
        # sysfs attributes do not support truncation; regular files in tests do.
        pass


def _open_write(path):
    return os.open(path, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))


def _close_all(prepared):
    for item in prepared:
        for handle in item.get("handles", ()):
            try:
                os.close(handle)
            except OSError:
                pass


class ArmadaRgbDevice(LedDevice):
    def __init__(self, leds_dir, backend, correction=None):
        self._leds_dir = leds_dir
        self._backend = backend["type"]
        self._targets = list(backend["targets"])
        self._correction = correction
        self._zones = self._zone_count()
        self._max_brightness = 255
        self.last_error = None
        # A channels profile (Odin 3) is one logical frame spread over 24
        # sysfs attributes. Keep each frame/blank transaction indivisible if
        # callers ever reach this device from more than one worker thread.
        self._write_lock = _ARMADA_WRITE_LOCK

    def _zone_count(self):
        if self._backend == "multicolor":
            return len(self._targets)
        return sum(target.startswith("red=") for target in self._targets)

    @property
    def available(self):
        try:
            self._inspect()
            return True
        except (OSError, UnicodeError, ValueError):
            return False

    @property
    def led_path(self):
        if not self._targets:
            return None
        name = self._targets[0].split("=", 1)[-1]
        return os.path.join(self._leds_dir, name)

    @property
    def zone_count(self):
        return self._zones

    def supports_per_zone(self):
        return self._zones > 1

    def reconnect(self):
        return self.available

    def _inspect(self):
        if self._backend == "channels":
            for target in self._targets:
                _, name = target.split("=", 1)
                node = os.path.join(self._leds_dir, name)
                _read_positive(os.path.join(node, "max_brightness"))
                if not os.path.isfile(os.path.join(node, "brightness")):
                    raise OSError(f"missing {name} brightness")
            return
        for name in self._targets:
            node = os.path.join(self._leds_dir, name)
            with open(os.path.join(node, "multi_index"), encoding="ascii") as handle:
                order = handle.read().lower().split()
            if len(order) != 3 or set(order) != set(_CHANNELS):
                raise ValueError(f"invalid RGB order for {name}")
            _read_positive(os.path.join(node, "max_brightness"))
            for attribute in ("multi_intensity", "brightness"):
                if not os.path.isfile(os.path.join(node, attribute)):
                    raise OSError(f"missing {name} {attribute}")

    def _channel_zones(self):
        channels = {channel: [] for channel in _CHANNELS}
        for target in self._targets:
            channel, name = target.split("=", 1)
            channels[channel].append(name)
        return list(zip(*(channels[channel] for channel in _CHANNELS)))

    def _prepare_channels(self, colors, brightness, power):
        prepared = []
        try:
            for names, color in zip(self._channel_zones(), colors):
                corrected = _correct(color, self._correction)
                for channel_index, name in enumerate(names):
                    path = os.path.join(self._leds_dir, name)
                    value = 0
                    if power:
                        maximum = _read_positive(os.path.join(path, "max_brightness"))
                        value = _scale(brightness, _gamma(corrected[channel_index], maximum))
                    handle = _open_write(os.path.join(path, "brightness"))
                    prepared.append({
                        "name": name,
                        "brightness_path": os.path.join(path, "brightness"),
                        "handles": [handle],
                        "writes": [(handle, str(value))],
                    })
            return prepared
        except Exception:
            _close_all(prepared)
            raise

    def _prepare_multicolor(self, colors, brightness, power):
        prepared = []
        try:
            for name, color in zip(self._targets, colors):
                path = os.path.join(self._leds_dir, name)
                brightness_path = os.path.join(path, "brightness")
                handles = []
                writes = []
                try:
                    blank_handle = _open_write(brightness_path)
                    handles.append(blank_handle)
                    if power:
                        maximum = _read_positive(os.path.join(path, "max_brightness"))
                        with open(os.path.join(path, "multi_index"), encoding="ascii") as handle:
                            order = handle.read().lower().split()
                        if len(order) != 3 or set(order) != set(_CHANNELS):
                            raise ValueError(f"invalid RGB order for {name}")
                        corrected = dict(zip(_CHANNELS, _correct(color, self._correction)))
                        intensity = " ".join(
                            str(_gamma(corrected[channel], maximum)) for channel in order
                        )
                        intensity_handle = _open_write(os.path.join(path, "multi_intensity"))
                        brightness_handle = _open_write(brightness_path)
                        handles.extend([intensity_handle, brightness_handle])
                        writes.extend([
                            (intensity_handle, intensity),
                            (brightness_handle, str(_scale(brightness, maximum))),
                        ])
                    else:
                        writes.append((blank_handle, "0"))
                    prepared.append({
                        "name": name,
                        "brightness_path": brightness_path,
                        "handles": handles,
                        "writes": writes,
                    })
                except Exception:
                    for handle in handles:
                        os.close(handle)
                    raise
            return prepared
        except Exception:
            _close_all(prepared)
            raise

    @staticmethod
    def _blank(prepared):
        for item in prepared:
            try:
                handle = _open_write(item["brightness_path"])
                try:
                    _write_open_file(handle, "0")
                finally:
                    os.close(handle)
            except OSError:
                pass

    def apply_zones(self, zone_colors, brightness, power):
        with self._write_lock:
            self.last_error = None
            colors = self._fit(zone_colors)
            prepared = []
            try:
                prepared = (
                    self._prepare_channels(colors, brightness, power)
                    if self._backend == "channels"
                    else self._prepare_multicolor(colors, brightness, power)
                )
                for item in prepared:
                    for handle, value in item["writes"]:
                        _write_open_file(handle, value)
                return True
            except (OSError, UnicodeError, ValueError) as error:
                self.last_error = str(error)
                self._blank(prepared)
                return False
            finally:
                _close_all(prepared)

    def apply_solid(self, color, brightness, power):
        return self.apply_zones([tuple(color)] * self._zones, brightness, power)


# GameSir "GCM" lighting over a controller's vendor HID interface: the Lenovo
# Legion G9 that docks to the Legion Tab Gen 3 (TB321FU). Each strip takes
# 05 0C 0C 01 <strip> H S B <effect> <speed> <brightness> plus a byte sum, as a
# report-ID-less output report; colors are HSB (hue scaled to a byte,
# saturation/brightness in percent). Effects: 1 solid, 2 breathing,
# 4 rainbow, 0xFF off. The controller animates the effects itself.
_GCM_SOLID = 0x01
_GCM_BREATHING = 0x02
_GCM_RAINBOW = 0x04
_GCM_OFF = 0xFF
# The Legion button LED (key 0xC2, 05 08 0A 01 <key> H S B <effect>) sits next
# to the right strip and takes its color. Keys only do a solid color.
_GCM_LEGION_KEY = 0xC2
_GCM_LEGION_KEY_STRIP = 2
_GCM_DESCRIPTOR_PREFIX = b"\x06\x7a\xff"  # usage page 0xFF7A
_GCM_EFFECTS = {
    "breathing": _GCM_BREATHING,
    "rainbow": _GCM_RAINBOW,
    "cycle": _GCM_RAINBOW,
    "spiral": _GCM_RAINBOW,
}


def _gcm_hsb(color):
    r, g, b = (max(0, min(255, int(c))) for c in color)
    high, low = max(r, g, b), min(r, g, b)
    delta = high - low
    if delta == 0:
        hue = 0.0
    elif high == r:
        hue = 60.0 * (((g - b) / delta) % 6)
    elif high == g:
        hue = 60.0 * ((b - r) / delta + 2)
    else:
        hue = 60.0 * ((r - g) / delta + 4)
    saturation = 0 if high == 0 else round(delta / high * 100)
    return [round(hue / 360 * 255) % 256, saturation, round(high / 255 * 100)]


def _gcm_byte(percent):
    return (max(0, min(100, int(percent))) * 255 + 50) // 100


def gcm_strip_packet(strip, hsb, effect, speed=0x80, brightness=0xFF):
    payload = [0x05, 0x0C, 0x0C, 0x01, strip, *hsb, effect, speed, brightness]
    return bytes([0x00, *payload, sum(payload) & 0xFF])


def gcm_key_packet(key, hsb):
    payload = [0x05, 0x08, 0x0A, 0x01, key, *hsb, _GCM_SOLID]
    return bytes([0x00, *payload, sum(payload) & 0xFF])


# Per-LED lights on a G9 running the custom per-LED firmware (not stock): a
# solid strip-0 command whose speed byte is 0xE0 + channel sets that one LED,
# hue/saturation as usual and the B byte as that LED's own brightness. B=0 is
# ignored by the firmware (the LED keeps its color), so "off" is sent as B=1.
# Any normal command (speed 0x80, effects, off) leaves per-LED mode.
# Channels top-down along each grip; 5 and 6 are the Legion button and status.
_GCM_PERLED_STRIP = 0
_GCM_PERLED_BASE = 0xE0
_G9_PERLED_LEFT = (7, 8, 9, 10, 11)
_G9_PERLED_RIGHT = (4, 3, 2, 1, 0)  # the right grip runs bottom-up (checked on the G9)
_G9_PERLED_FLAG = "g9-perled"
# Resend every LED this often even if unchanged, to recover after the
# controller reconnects or leaves per-LED mode.
_G9_PERLED_REFRESH_S = 2.0
# A LED this dim shows as a faint white, also for black and dark greys (no hue left):
# treat everything up to this value (percent) as off and stretch the rest, so dark
# colors still fade in smoothly. Off needs the per-LED firmware 0.75 or later.
_G9_PERLED_BLACK = 12


def _perled_value(value):
    return round(max(0, value - _G9_PERLED_BLACK) * 100 / (100 - _G9_PERLED_BLACK))


def gcm_perled_packet(channel, hsb):
    hue, saturation, value = hsb
    payload = [0x05, 0x0C, 0x0C, 0x01, _GCM_PERLED_STRIP, hue, saturation,
               max(1, value), _GCM_SOLID, _GCM_PERLED_BASE + channel, 0xFF]
    return bytes([0x00, *payload, sum(payload) & 0xFF])


def g9_perled_enabled(settings_dir=None):
    """Opt-in only: the stock G9 firmware paints whole strips with these commands."""
    if os.environ.get("COLORES_G9_PERLED", "").strip().lower() in ("1", "true", "yes"):
        return True
    if settings_dir is None:
        try:
            import decky  # only inside the Decky plugin runtime
            settings_dir = decky.DECKY_PLUGIN_SETTINGS_DIR
        except (ImportError, AttributeError):
            settings_dir = os.environ.get("DECKY_PLUGIN_SETTINGS_DIR")
    return bool(settings_dir) and os.path.exists(os.path.join(settings_dir, _G9_PERLED_FLAG))


class GcmHidRgbDevice(LedDevice):
    """Legion G9 light strips through armada-rgb's gcmhid catalog entry."""

    def __init__(self, backend, sysfs_root="/", correction=None, perled=False):
        self._root = sysfs_root
        self._hid_id = "HID_ID=0003:0000{}:0000{}".format(
            backend["vendor"].upper(), backend["product"].upper()
        )
        self._strips = list(backend["strips"])
        self.perled = bool(perled)
        # Per-LED: zones 0-4 down the left grip, 5-9 down the right one.
        self._channels = list(_G9_PERLED_LEFT + _G9_PERLED_RIGHT) if self.perled else []
        self._zones = len(self._channels) if self.perled else len(self._strips)
        self._sent = {}
        self._sent_at = 0.0
        self._correction = correction
        self._max_brightness = 100
        self.last_error = None
        self._write_lock = _ARMADA_WRITE_LOCK
        # Kept open between frames: reopening wakes the USB interface and costs
        # far more than the write itself (8 ms per report on the G9).
        self._fd = None

    def _find(self):
        hidraw = os.path.join(self._root, "sys/class/hidraw")
        try:
            names = sorted(n for n in os.listdir(hidraw) if n.startswith("hidraw"))
        except OSError:
            return None
        for name in names:
            device = os.path.join(hidraw, name, "device")
            try:
                with open(os.path.join(device, "uevent"), encoding="ascii", errors="replace") as handle:
                    if self._hid_id.upper() not in (line.strip().upper() for line in handle):
                        continue
                with open(os.path.join(device, "report_descriptor"), "rb") as handle:
                    if not handle.read(3).startswith(_GCM_DESCRIPTOR_PREFIX):
                        continue
            except OSError:
                continue
            return os.path.join(self._root, "dev", name)
        return None

    @property
    def available(self):
        return self._find() is not None

    @property
    def led_path(self):
        return self._find()

    @property
    def zone_count(self):
        return self._zones

    def supports_per_zone(self):
        return self._zones > 1

    def supports_hardware_effects(self):
        return True

    def reconnect(self):
        return self.available

    def _close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _write_all(self, packets):
        if self._fd is None:
            path = self._find()
            if path is None:
                raise FileNotFoundError("controller not connected")
            self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0))
        for packet in packets:
            os.write(self._fd, packet)

    def _send(self, packets):
        with self._write_lock:
            try:
                self._write_all(packets)
            except OSError:
                # Detached, powered off or re-enumerated: find it again once.
                self._close()
                try:
                    self._write_all(packets)
                except FileNotFoundError as error:
                    self.last_error = str(error)
                    return False
                except OSError as error:
                    self._close()
                    self.last_error = str(error)
                    return False
        self.last_error = None
        return True

    def invalidate(self):
        with self._write_lock:
            self._close()
        self._sent = {}

    def _with_legion_key(self, packets, strip, hsb):
        if strip == _GCM_LEGION_KEY_STRIP:
            packets.append(gcm_key_packet(_GCM_LEGION_KEY, hsb))
        return packets

    def _off(self):
        self._sent = {}
        packets = []
        for strip in self._strips:
            packets.append(gcm_strip_packet(strip, [0, 0, 0], _GCM_OFF))
            self._with_legion_key(packets, strip, [0, 0, 0])
        return self._send(packets)

    def _apply_perled(self, zone_colors, brightness):
        now = time.monotonic()
        refresh = now - self._sent_at >= _G9_PERLED_REFRESH_S
        packets = []
        for channel, color in zip(self._channels, self._fit(zone_colors)):
            hue, saturation, value = _gcm_hsb(_correct(color, self._correction))
            hsb = (hue, saturation, _scale(brightness, _perled_value(value)))
            if refresh or self._sent.get(channel) != hsb:
                packets.append((channel, hsb, gcm_perled_packet(channel, hsb)))
        if not packets:
            return True
        if not self._send([packet for _, _, packet in packets]):
            self._sent = {}
            return False
        for channel, hsb, _ in packets:
            self._sent[channel] = hsb
        if refresh:
            self._sent_at = now
        return True

    def apply_zones(self, zone_colors, brightness, power):
        if not power:
            return self._off()
        if self.perled:
            return self._apply_perled(zone_colors, brightness)
        packets = []
        for strip, color in zip(self._strips, self._fit(zone_colors)):
            hue, saturation, value = _gcm_hsb(_correct(color, self._correction))
            hsb = [hue, saturation, _scale(brightness, value)]
            packets.append(gcm_strip_packet(strip, hsb, _GCM_SOLID))
            self._with_legion_key(packets, strip, hsb)
        return self._send(packets)

    def apply_solid(self, color, brightness, power):
        return self.apply_zones([tuple(color)] * self._zones, brightness, power)

    def apply_hardware_effect(self, effect_id, color, speed, brightness, power):
        self._sent = {}
        effect = _GCM_EFFECTS.get(effect_id)
        if not power:
            return self._off()
        if effect is None:
            return self.apply_solid(color, brightness, power)
        hue, saturation, value = _gcm_hsb(_correct(color, self._correction))
        hsb = [hue, saturation, _scale(brightness, value)]
        packets = []
        for strip in self._strips:
            packets.append(gcm_strip_packet(strip, hsb, effect, _gcm_byte(speed), _gcm_byte(brightness)))
            # The key cannot breathe or cycle: it keeps the effect's base color.
            self._with_legion_key(packets, strip, hsb)
        return self._send(packets)


def build_armada_device(model, leds_dir, sysfs_root="/"):
    profile = profile_for_model(model, sysfs_root)
    if profile is None:
        return None
    if profile["backend"]["type"] == "gcmhid":
        perled = g9_perled_enabled()
        device = GcmHidRgbDevice(profile["backend"], sysfs_root, profile.get("correction"), perled=perled)
        if perled:
            profile = dict(profile, perled=True)
        return profile, device
    device = ArmadaRgbDevice(
        leds_dir,
        profile["backend"],
        correction=profile.get("correction"),
    )
    return profile, device


_SIDE_REGION_TOP = 0.45
_SIDE_REGION_WIDTH = 0.20
# Per model: a narrower band weighted towards the screen edge, so the strips
# follow what is next to them rather than the middle of the scene. Checked on
# the TB321FU with the Legion G9 (a 5 % colored edge is picked up).
_SIDE_TUNING = {
    "Lenovo Legion Y700 (2025) / TB321FU": {"width": 0.10, "falloff": 1.5},
}


def layout_for_profile(model, profile, zones):
    """Return calibrated Ambilight grouping only where physical sides are known."""
    side_models = {
        "AYN Odin 2 Portal",
        "AYN Thor",
        "AYN Thor Lite",
        "Retroid Pocket 5",
        "Retroid Pocket 5 Visionox",
        "Retroid Pocket Flip2",
        "Retroid Pocket Flip2 Visionox",
        "Retroid Pocket 6",
        "Retroid Pocket 6 TOP-DPAD",
        "Retroid Pocket Nova",
        "AYN Odin 3",
        "Lenovo Legion Y700 (2025) / TB321FU",
    }
    if model in side_models and zones > 1 and zones % 2 == 0:
        half = zones // 2
        tuning = _SIDE_TUNING.get(model, {})
        width = tuning.get("width", _SIDE_REGION_WIDTH)
        falloff = tuning.get("falloff", 0.0)
        # Per-LED strips get one sample per LED down the edge band.
        edge_kind = "split-edge" if (profile or {}).get("perled") else "shared-edge"
        # The side lights sit on the lower half of the grips, so sample only the
        # lower part of each screen edge instead of its full height.
        return [
            {
                "name": "Left lights",
                "region": [0.0, _SIDE_REGION_TOP, width, 1.0],
                "zones": list(range(half)),
                "kind": edge_kind,
                "falloff": falloff,
            },
            {
                "name": "Right lights",
                "region": [1.0 - width, _SIDE_REGION_TOP, 1.0, 1.0],
                "zones": list(range(half, zones)),
                "kind": edge_kind,
                "falloff": falloff,
            },
        ]
    return [
        {
            "name": "Lights",
            "region": [0.0, 0.0, 1.0, 1.0],
            "zones": list(range(zones)),
            "kind": "shared-full",
        }
    ] if zones else []
