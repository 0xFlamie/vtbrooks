import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from brooks_failed_range_shadow import (MinuteStream, audit, frame_from_bars, initialize, prospective_signal,
                                        record_candidates, settle, write_event)


def message(kind, time, identity=1, price=100., size=1.):
    result = {"type": kind, "product_id": "ETH-USD", "time": time.isoformat()}
    result["last_trade_id" if kind == "heartbeat" else "trade_id"] = identity
    if kind == "match":
        result.update(price=str(price), size=str(size))
    return result


def setup_stream():
    start = pd.Timestamp("2026-09-10T00:00:10Z")
    stream = MinuteStream(start)
    stream.ingest(message("last_match", start, 10), start)
    return stream, start


def snapshot_fixture():
    start = pd.Timestamp("2026-09-10T00:00:00Z")
    bars = [{"stamp": (start + pd.Timedelta(minutes=i)).isoformat(), "open": 100., "high": 100.5, "low": 99.9,
             "close": 100.4, "volume": 1., "live": True,
             "received_at": (start + pd.Timedelta(minutes=i + 1, seconds=1)).isoformat()} for i in range(65)]
    signal = {"bar_start": start.isoformat(), "available_at": (start + pd.Timedelta(minutes=3)).isoformat(),
              "direction": 1, "atr": 1.}
    snap = prospective_signal(signal, bars[:3], start, start + pd.Timedelta(minutes=3, seconds=2))
    return start, bars, snap


class TestFailedRangeShadow(unittest.TestCase):
    def test_startup_partial_minute_ignored_not_faked(self):
        stream, start = setup_stream()
        self.assertEqual(stream.ingest(message("match", start + pd.Timedelta(seconds=1), 11), start + pd.Timedelta(seconds=1)), [])
        self.assertEqual(stream.trades, {})

    def test_heartbeat_seals_only_completed_minutes(self):
        stream, start = setup_stream()
        t = start.ceil("min")
        stream.ingest(message("match", t, 11, 100., 2.), t)
        stream.ingest(message("match", t + pd.Timedelta(seconds=10), 12, 101., 3.), t + pd.Timedelta(seconds=10))
        self.assertEqual(stream.ingest(message("heartbeat", t + pd.Timedelta(seconds=30), 12), t + pd.Timedelta(seconds=30)), [])
        closed = stream.ingest(message("heartbeat", t + pd.Timedelta(minutes=1), 12), t + pd.Timedelta(minutes=1, seconds=1))
        self.assertEqual(len(closed), 1)
        self.assertEqual((closed[0]["open"], closed[0]["close"], closed[0]["volume"]), (100., 101., 5.))
        self.assertEqual(len(closed[0]["trades"]), 2)

    def test_trade_gap_duplicate_and_heartbeat_mismatch_stop(self):
        for kind, identity in (("match", 12), ("match", 10), ("heartbeat", 12)):
            stream, start = setup_stream()
            with self.assertRaises(ValueError):
                stream.ingest(message(kind, start.ceil("min"), identity), start.ceil("min"))

    def test_empty_minutes_not_filled(self):
        stream, start = setup_stream()
        boundary = start.ceil("min") + pd.Timedelta(minutes=2)
        self.assertEqual(stream.ingest(message("heartbeat", boundary, 10), boundary), [])

    def test_late_trade_and_clock_rollback_stop(self):
        stream, start = setup_stream()
        with self.assertRaises(ValueError):
            stream.ingest(message("match", start.ceil("min"), 11), start.ceil("min") + pd.Timedelta(minutes=3))
        stream, start = setup_stream()
        with self.assertRaises(ValueError):
            stream.ingest(message("heartbeat", start, 10), start - pd.Timedelta(seconds=1))

    def test_wrong_symbol_stops(self):
        stream, start = setup_stream()
        wrong = {**message("heartbeat", start, 10), "product_id": "BTC-USD"}
        with self.assertRaises(ValueError):
            stream.ingest(wrong, start)

    def test_context_or_late_prefix_not_prospective(self):
        start, bars, snap = snapshot_fixture()
        self.assertTrue(snap["prospective_eligible"])
        bars[0]["live"] = False
        candidate = prospective_signal(snap["signal"], bars[:3], start, start + pd.Timedelta(minutes=3, seconds=2))
        self.assertFalse(candidate["prospective_eligible"])
        self.assertIsNone(candidate["user_delivered_at"])
        self.assertIsNone(candidate["entry_price"])

    def test_delayed_generation_not_prospective(self):
        start, bars, snap = snapshot_fixture()
        candidate = prospective_signal(snap["signal"], bars[:3], start, start + pd.Timedelta(minutes=6))
        self.assertFalse(candidate["prospective_eligible"])

    def test_future_entry_and_original_deadline(self):
        start, _, snap = snapshot_fixture()
        self.assertEqual(pd.Timestamp(snap["entry_not_before"]), start + pd.Timedelta(minutes=5))
        self.assertEqual(pd.Timestamp(snap["deadline"]), start + pd.Timedelta(minutes=63))

    def test_pending_and_missing_future_are_not_wins(self):
        start, bars, snap = snapshot_fixture()
        self.assertEqual(settle(snap, bars, start + pd.Timedelta(minutes=63))["status"], "pending")
        self.assertEqual(settle(snap, bars[:20], start + pd.Timedelta(minutes=65))["status"], "unverified")
        result = settle(snap, bars, start + pd.Timedelta(minutes=65))
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["cases"]["0.25/3"]["forward"]["held"], 0)

    def test_late_future_remains_unknown(self):
        start, bars, snap = snapshot_fixture()
        bars[8]["received_at"] = (start + pd.Timedelta(minutes=20)).isoformat()
        self.assertEqual(settle(snap, bars, start + pd.Timedelta(minutes=65))["status"], "unverified")

    def test_duplicate_minute_rejected(self):
        _, bars, _ = snapshot_fixture()
        with self.assertRaises(ValueError):
            frame_from_bars(bars + [bars[0]])

    def test_ledger_idempotence_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadow.sqlite3"
            initialize(path, {"version": "test"})
            self.assertTrue(write_event(path, "one", "minute", {"x": 1}))
            self.assertFalse(write_event(path, "one", "minute", {"x": 1}))
            self.assertEqual(len(audit(path)), 2)
            with sqlite3.connect(path) as db:
                db.execute("UPDATE events SET payload=? WHERE event_key='one'", (json.dumps({"x": 2}),))
            with self.assertRaises((ValueError, KeyError)):
                audit(path)

    def test_ledger_cannot_change_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadow.sqlite3"
            initialize(path, {"version": "one"})
            with self.assertRaises(ValueError):
                initialize(path, {"version": "two"})

    def test_changed_duplicate_and_clock_rollback_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadow.sqlite3"
            initial = pd.Timestamp("2026-09-10T00:00Z")
            with patch("brooks_failed_range_shadow.now", return_value=initial):
                initialize(path, {"version": "test"})
                write_event(path, "one", "minute", {"x": 1})
            with self.assertRaises(ValueError):
                write_event(path, "one", "minute", {"x": 2})
            with patch("brooks_failed_range_shadow.now", return_value=initial - pd.Timedelta(seconds=1)):
                with self.assertRaises(ValueError):
                    write_event(path, "two", "minute", {"x": 2})
            self.assertEqual(len(audit(path)), 2)

    def test_missing_prefix_does_not_get_prospective_status(self):
        start, bars, snap = snapshot_fixture()
        candidate = prospective_signal(snap["signal"], bars[1:3], start, start + pd.Timedelta(minutes=3, seconds=2))
        self.assertFalse(candidate["prospective_eligible"])

    def test_engine_to_ledger_once_before_entry(self):
        start = pd.Timestamp("2026-09-08T00:00Z")
        prices = [100 + .4 * (-1)**i for i in range(15 * 124)] + [99., 99.6, 99.8]
        bars = [{"stamp": (start + pd.Timedelta(minutes=i)).isoformat(), "open": p, "high": p + .1,
                 "low": p - .1, "close": p, "volume": 1., "live": True,
                 "received_at": (start + pd.Timedelta(minutes=i + 1, seconds=1)).isoformat()} for i, p in enumerate(prices)]
        live = start + pd.Timedelta(minutes=15 * 124)
        generated = live + pd.Timedelta(minutes=3, seconds=2)
        with tempfile.TemporaryDirectory() as temp, patch("brooks_failed_range_shadow.now", return_value=generated):
            path = Path(temp) / "shadow.sqlite3"
            initialize(path, {"version": "test"})
            signals = record_candidates(path, bars, live)
            self.assertEqual(len(signals), 1)
            self.assertTrue(signals[0]["prospective_eligible"])
            self.assertLess(generated, pd.Timestamp(signals[0]["entry_not_before"]))
            self.assertEqual(record_candidates(path, bars, live), [])
            self.assertEqual(len(audit(path)), 2)

    def test_positive_path_is_evaluated_only_after_deadline(self):
        start, bars, snap = snapshot_fixture()
        for i in range(6, 63):
            bars[i].update(open=100.3, high=100.5, low=100.3, close=100.4)
        result = settle(snap, bars, start + pd.Timedelta(minutes=65))
        self.assertEqual(result["cases"]["0.25/3"]["forward"]["held"], 1)
        self.assertIsNone(snap["entry_price"])

    def test_write_after_entry_or_slow_transaction_rolls_back(self):
        start, _, snap = snapshot_fixture()
        deadline = pd.Timestamp(snap["entry_not_before"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadow.sqlite3"
            with patch("brooks_failed_range_shadow.now", return_value=start):
                initialize(path, {"version": "test"})
            with patch("brooks_failed_range_shadow.now", return_value=deadline):
                with self.assertRaises(ValueError):
                    write_event(path, "candidate", "signal", snap)
            with patch("brooks_failed_range_shadow.now", side_effect=[deadline - pd.Timedelta(seconds=1), deadline]):
                with self.assertRaises(ValueError):
                    write_event(path, "candidate", "signal", snap)
            self.assertEqual(len(audit(path)), 1)


if __name__ == "__main__":
    unittest.main()
