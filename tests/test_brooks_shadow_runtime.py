import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

from brooks_failed_range_shadow import audit, initialize, prospective_signal, record_candidates, write_event
from brooks_shadow_runtime import (Journal, RUNTIME_POLICY, ROOT, SOURCES, coverage, exclusive, file_hash,
                                    capture_session, merge_context, parse_context, register, run, verify_header)


def fixture():
    start = pd.Timestamp("2026-09-10T00:00Z")
    bars = [{"stamp": (start + pd.Timedelta(minutes=i)).isoformat(), "open": 100., "high": 100.4,
             "low": 99.9, "close": 100.2, "volume": 1., "live": True,
             "received_at": (start + pd.Timedelta(minutes=i + 1, seconds=1)).isoformat()} for i in range(65)]
    signal = {"bar_start": start.isoformat(), "available_at": (start + pd.Timedelta(minutes=3)).isoformat(),
              "direction": 1, "atr": 1., "episode_id": "failed_range:" + start.isoformat()}
    snapshot = prospective_signal(signal, bars[:3], start, start + pd.Timedelta(minutes=3, seconds=2))
    return start, bars, snapshot


class TestShadowRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.start, self.bars, self.snap = fixture()
        with patch("brooks_failed_range_shadow.now", return_value=self.start):
            initialize(self.path, {"registered_at": self.start.isoformat()})

    def write_signal(self, journal):
        with patch("brooks_failed_range_shadow.now", return_value=self.start + pd.Timedelta(minutes=3, seconds=3)):
            journal.write("signal:test", "signal", self.snap)

    def test_new_registration_rejects_old_policy_and_overwrite(self):
        manifest = {"policy": RUNTIME_POLICY, "implementation_sha256": {n: file_hash(ROOT / n) for n in SOURCES}}
        verify_header(manifest)
        with self.assertRaises(ValueError):
            verify_header({**manifest, "policy": {"version": "v1"}})
        with self.assertRaises(FileExistsError):
            register(self.path, manifest)
        self.assertEqual(len(audit(self.path)), 1)

    def test_manifest_rejects_extra_path_and_changed_hash(self):
        manifest = {"policy": RUNTIME_POLICY, "implementation_sha256": {n: file_hash(ROOT / n) for n in SOURCES}}
        for hashes in ({**manifest["implementation_sha256"], "../outside.py": "x"},
                       {**manifest["implementation_sha256"], SOURCES[0]: "changed"}):
            with self.assertRaises(ValueError):
                verify_header({**manifest, "implementation_sha256": hashes})

    def test_exclusive_lock_rejects_second_writer_and_releases(self):
        with exclusive(self.path):
            with self.assertRaises(RuntimeError):
                with exclusive(self.path):
                    self.fail("second writer")
        with exclusive(self.path):
            pass

    def test_reopen_preserves_pending_and_deduplicates_signal(self):
        with closing(Journal(self.path)) as journal:
            self.write_signal(journal)
        with closing(Journal(self.path)) as journal:
            self.assertEqual(len(journal.pending()), 1)
            self.assertFalse(journal.write("signal:test", "signal", self.snap))
            self.assertEqual(len(journal.pending()), 1)

    def test_recovery_records_unknown_disconnect_once(self):
        with closing(Journal(self.path)) as journal:
            journal.write("start:crashed", "capture_start", {"at": self.start.isoformat()})
        with closing(Journal(self.path)) as journal:
            journal.recover(self.start + pd.Timedelta(hours=1))
            journal.recover(self.start + pd.Timedelta(hours=2))
        recovered = [v for v in audit(self.path) if v["kind"] == "recovery_detected"]
        self.assertEqual(len(recovered), 1)
        self.assertIsNone(recovered[0]["payload"]["actual_disconnect_at"])
        self.assertFalse(recovered[0]["payload"]["gap_filled_as_live"])

    def test_clean_stop_not_reported_as_crash(self):
        with closing(Journal(self.path)) as journal:
            journal.write("start:done", "capture_start", {})
            journal.write("stop:done", "capture_stopped", {})
            journal.recover(self.start)
        self.assertFalse(any(v["kind"] == "recovery_detected" for v in audit(self.path)))

    def test_missing_future_settles_unknown_after_restart_once(self):
        with closing(Journal(self.path)) as journal:
            self.write_signal(journal)
            journal.finalize(self.start + pd.Timedelta(minutes=64))
            self.assertEqual(len(journal.pending()), 1)
        with closing(Journal(self.path)) as journal:
            journal.finalize(self.start + pd.Timedelta(minutes=65))
            journal.finalize(self.start + pd.Timedelta(minutes=90))
            self.assertEqual(journal.pending(), [])
        settlements = [v for v in audit(self.path) if v["kind"] == "settlement"]
        self.assertEqual(len(settlements), 1)
        self.assertEqual(settlements[0]["payload"]["status"], "unverified")

    def test_restart_settlement_reads_archived_minutes_not_rest(self):
        with closing(Journal(self.path)) as journal:
            self.write_signal(journal)
            for bar in self.bars:
                journal.write("minute:" + bar["stamp"], "minute", {**bar, "trades": []})
        with closing(Journal(self.path)) as journal:
            journal.finalize(self.start + pd.Timedelta(days=2))
            recovered = journal.minutes(self.start, self.start + pd.Timedelta(minutes=1))
            self.assertNotIn("trades", recovered[0])
        self.assertEqual(audit(self.path)[-1]["payload"]["status"], "evaluated")

    def test_rest_context_cannot_fill_missing_future(self):
        with closing(Journal(self.path)) as journal:
            self.write_signal(journal)
            journal.write("context:all", "context", {"bars": self.bars, "live": False})
            journal.finalize(self.start + pd.Timedelta(minutes=65))
        self.assertEqual(audit(self.path)[-1]["payload"]["status"], "unverified")

    def test_foreign_tail_write_and_changed_duplicate_rejected(self):
        with closing(Journal(self.path)) as journal:
            journal.write("minute:one", "minute", {"n": 1})
            with self.assertRaises(ValueError):
                journal.write("minute:one", "minute", {"n": 2})
            write_event(self.path, "other", "test", {})
            with self.assertRaises(ValueError):
                journal.write("mine", "test", {})

    def test_startup_detects_middle_tampering(self):
        write_event(self.path, "one", "test", {})
        write_event(self.path, "two", "test", {})
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE events SET sha256='changed' WHERE event_key='one'")
        with self.assertRaises(ValueError):
            Journal(self.path)

    def test_late_signal_transaction_still_rolls_back(self):
        with closing(Journal(self.path)) as journal:
            due = pd.Timestamp(self.snap["entry_not_before"])
            with patch("brooks_failed_range_shadow.now", side_effect=[due - pd.Timedelta(seconds=1), due]):
                with self.assertRaises(ValueError):
                    journal.write("signal:test", "signal", self.snap)
            self.assertNotIn("signal:test", journal)
            journal.write("after", "test", {})
        self.assertEqual(len(audit(self.path)), 2)

    def test_merge_keeps_live_identity_over_revised_context(self):
        bar = self.bars[0]
        context = [{**bar, "close": 100.3, "live": False}, {**self.bars[1], "live": False}]
        merged = merge_context(context, [bar], self.start)
        self.assertEqual(merged[0], bar)
        self.assertFalse(merged[1]["live"])
        self.assertEqual(len(merge_context(context, [bar], self.start + pd.Timedelta(minutes=1))), 1)

    def test_context_parser_boundaries_empty_and_invalid_data(self):
        epoch = self.start.timestamp()
        payload = [[epoch + 60, 99, 101, 100, 100, 1], [epoch, 99, 101, 100, 100, 1]]
        self.assertEqual(len(parse_context(payload, self.start, self.start + pd.Timedelta(minutes=1))), 1)
        self.assertEqual(parse_context([], self.start, self.start), [])
        for invalid in (payload + [payload[0]], [[epoch + 1, 99, 101, 100, 100, 1]],
                        [[epoch, 99, 101, 100, 100, -1]], [[epoch, 99, 101, 100, float("nan"), 1]], {}):
            with self.assertRaises(ValueError):
                parse_context(invalid, self.start, self.start + pd.Timedelta(minutes=2))

    def test_coverage_includes_disconnected_and_empty_minutes(self):
        with closing(Journal(self.path)) as journal:
            journal.write("connected:1", "connected", {"first_full_minute": self.start.isoformat()})
            journal.write("minute:" + self.bars[0]["stamp"], "minute", self.bars[0])
            late = {**self.bars[1], "received_at": (self.start + pd.Timedelta(minutes=9)).isoformat()}
            journal.write("minute:" + late["stamp"], "minute", late)
        result = coverage(self.path, self.start + pd.Timedelta(minutes=10))
        self.assertEqual((result["expected_elapsed_minutes"], result["timely_live_minutes"]), (10, 1))
        self.assertEqual(result["missing_or_unverified_minutes"], 9)
        self.assertEqual(result["running"], "not_inferred_from_ledger")

    def test_registration_alone_has_no_observation_coverage(self):
        result = coverage(self.path, self.start + pd.Timedelta(days=30))
        self.assertIsNone(result["coverage_ratio"])
        self.assertEqual(result["expected_elapsed_minutes"], 0)

    def test_network_retry_retains_failure_then_new_session(self):
        async def scenario(journal):
            with patch("brooks_shadow_runtime.capture_session", new=AsyncMock(side_effect=[OSError(), ValueError()])) as capture:
                with patch("brooks_shadow_runtime.asyncio.sleep", new=AsyncMock()) as pause:
                    with self.assertRaises(ValueError):
                        await run(journal, 60, True)
                    self.assertEqual(capture.await_count, 2)
                    pause.assert_awaited_once_with(30)
        with closing(Journal(self.path)) as journal:
            asyncio.run(scenario(journal))
        failures = [v["payload"]["retryable"] for v in audit(self.path) if v["kind"] == "capture_failed"]
        self.assertEqual(failures, [True, False])

    def test_real_engine_with_persistent_dedup_after_restart(self):
        base = self.start - pd.Timedelta(minutes=15 * 124)
        prices = [100 + .4 * (-1)**i for i in range(15 * 124)] + [99., 99.6, 99.8]
        bars = [{"stamp": (base + pd.Timedelta(minutes=i)).isoformat(), "open": p, "high": p + .1,
                 "low": p - .1, "close": p, "volume": 1., "live": True,
                 "received_at": (base + pd.Timedelta(minutes=i + 1, seconds=1)).isoformat()} for i, p in enumerate(prices)]
        generated = self.start + pd.Timedelta(minutes=3, seconds=2)
        with patch("brooks_failed_range_shadow.now", return_value=generated):
            with closing(Journal(self.path)) as journal:
                found = record_candidates(self.path, bars, self.start, existing=journal, writer=journal.write)
                self.assertEqual(len(found), 1)
                self.assertTrue(found[0]["prospective_eligible"])
            with closing(Journal(self.path)) as journal:
                self.assertEqual(record_candidates(self.path, bars, self.start, existing=journal, writer=journal.write), [])

    def test_capture_fetches_startup_gap_only_after_it_is_closed(self):
        clock = [self.start + pd.Timedelta(seconds=10)]
        calls = []
        messages = [
            {"type": "last_match", "time": "2026-09-10T00:00:10Z", "trade_id": 10},
            {"type": "heartbeat", "time": "2026-09-10T00:00:30Z", "last_trade_id": 10},
            {"type": "match", "time": "2026-09-10T00:01:05Z", "trade_id": 11, "price": "100", "size": "1"},
            {"type": "heartbeat", "time": "2026-09-10T00:02:00Z", "last_trade_id": 11}]

        async def receive():
            if not messages:
                raise OSError("test disconnect")
            item = messages.pop(0)
            clock[0] = pd.Timestamp(item["time"]) + pd.Timedelta(seconds=1)
            return json.dumps({**item, "product_id": "ETH-USD"})

        def context(journal, start, end):
            self.assertLessEqual(end, clock[0].floor("min"))
            calls.append((start, end))
            return []

        ws = AsyncMock()
        ws.recv.side_effect = receive
        ws.__aenter__.return_value = ws
        with closing(Journal(self.path)) as journal, patch("brooks_shadow_runtime.connect", return_value=ws):
            with patch("brooks_shadow_runtime.now", side_effect=lambda: clock[0]):
                with patch("brooks_shadow_runtime.history", side_effect=context):
                    with patch("brooks_shadow_runtime.record_candidates", return_value=[]):
                        with self.assertRaises(OSError):
                            asyncio.run(capture_session(journal, 20))
        self.assertEqual(len(calls), 2)
        live = [v for v in audit(self.path) if v["kind"] == "minute"]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["payload"]["stamp"], (self.start + pd.Timedelta(minutes=1)).isoformat())


if __name__ == "__main__":
    unittest.main()
