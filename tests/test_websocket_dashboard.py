import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web import server
from ws_kline import KlineBuffer


class TestWebsocketFrame(unittest.TestCase):
    def test_encodes_large_unicode_snapshot(self):
        payload = {"price": 2497.11, "summary": "做多" * 100}
        frame = server.websocket_frame(payload)

        self.assertEqual(frame[0], 0x81)
        self.assertEqual(frame[1], 126)
        size = struct.unpack("!H", frame[2:4])[0]
        self.assertEqual(json.loads(frame[4:4 + size]), payload)


class TestSnapshotThrottle(unittest.TestCase):
    def test_active_bar_is_saved_after_one_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = KlineBuffer(os.path.join(tmp, "snapshot.json"))
            buffer._last_saved = 10
            with mock.patch("ws_kline.time.time", return_value=11):
                buffer.upsert("ETHUSDC:15m", 1, 100, 101, 99, 100.5, 20, 0, False)

            self.assertTrue(os.path.exists(buffer.snapshot_path))

    def test_save_preserves_other_process_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            with open(path, "w") as existing:
                json.dump({"ETHUSDT:15m": [[1, 1, 1, 1, 1, 1, 0]]}, existing)
            buffer = KlineBuffer(path)
            buffer.bars = {"ETHUSDC:15m": [[2, 2, 2, 2, 2, 2, 0]]}

            buffer.save()

            with open(path) as saved:
                keys = json.load(saved).keys()
            self.assertEqual(set(keys), {"ETHUSDT:15m", "ETHUSDC:15m"})


if __name__ == "__main__":
    unittest.main()
