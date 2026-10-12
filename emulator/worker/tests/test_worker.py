import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "worker.py"
SPEC = importlib.util.spec_from_file_location("colores_emulator_worker", MODULE_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class VirtualSysfsTests(unittest.TestCase):
    def test_multicolor_state(self):
        profile = {"backend": {"type": "multicolor", "targets": ["rgb:test"]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.create_virtual_sysfs(root, "Test", profile)
            node = root / "sys/class/leds/rgb:test"
            (node / "multi_intensity").write_text("255 0 0\n")
            (node / "brightness").write_text("255\n")
            self.assertEqual(worker.read_led_state(root, profile)["colors"], [[255, 0, 0]])

    def test_channel_zones_are_grouped(self):
        profile = {"backend": {"type": "channels", "targets": ["red=r0", "red=r1", "green=g0", "green=g1", "blue=b0", "blue=b1"]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.create_virtual_sysfs(root, "Test", profile)
            for name in ("r0", "g1"):
                (root / "sys/class/leds" / name / "brightness").write_text("255\n")
            self.assertEqual(worker.read_led_state(root, profile)["colors"], [[255, 0, 0], [0, 255, 0]])

    def test_gcmhid_strips_and_key(self):
        profile = {"backend": {"type": "gcmhid", "vendor": "3537", "product": "1134", "strips": [1, 2]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.create_virtual_sysfs(root, "Test", profile)
            uevent = (root / "sys/class/hidraw/hidraw0/device/uevent").read_text()
            self.assertIn("HID_ID=0003:00003537:00001134", uevent)

            def strip(number, hsb, effect=0x01):
                payload = [0x05, 0x0C, 0x0C, 0x01, number, *hsb, effect, 0x80, 0xFF]
                return bytes([0x00, *payload, sum(payload) & 0xFF])

            key = [0x05, 0x08, 0x0A, 0x01, 0xC2, 85, 100, 100, 0x01]
            with open(root / "dev/hidraw0", "ab") as handle:
                handle.write(strip(1, [0, 100, 100]) + strip(2, [85, 100, 100]) + bytes([0x00, *key, sum(key) & 0xFF]))
            state = worker.read_led_state(root, profile)
            self.assertEqual(state["colors"], [[255, 0, 0], [0, 255, 0]])
            self.assertEqual(state["raw"][-1], {"key": "legion", "rgb": [0, 255, 0]})

            with open(root / "dev/hidraw0", "ab") as handle:
                handle.write(strip(1, [0, 0, 0], 0xFF))
            self.assertEqual(worker.read_led_state(root, profile)["colors"][0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
