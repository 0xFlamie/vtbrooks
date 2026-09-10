from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import macro_expectations as expectations
import vt_vote_bot as bot
from web import server


class TestMacroDashboard(unittest.TestCase):
    def test_snapshot_and_websocket_include_readonly_macro_data(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        expected = {"events": [], "collector_status": "not_collected"}
        with ExitStack() as stack:
            for name, value in (("compute_4h_context", {}), ("fetch_klines", empty),
                                ("compute_levels", {}), ("fetch_fast_price", 2500.), ("compute_vwap", None)):
                stack.enter_context(patch.object(server.bot, name, return_value=value))
            stack.enter_context(patch.object(server, "websocket_price", return_value=2500.))
            stack.enter_context(patch.object(server, "read_json", side_effect=lambda name, fallback: fallback))
            stack.enter_context(patch.object(server, "research_snapshot", return_value={}))
            stack.enter_context(patch.object(server, "macro_snapshot", return_value=expected))
            result = server.snapshot()
        self.assertEqual(result["macro_events"], expected)
        self.assertEqual(result["price"], 2500.)
        self.assertIsNone(result["4h"]["direction"])
        self.assertIn(b'"macro_events"', server.websocket_frame(result))

    def test_collector_does_not_run_when_reading_web_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite3"
            with patch.object(expectations.requests, "get") as get:
                expectations.snapshot(path)
            get.assert_not_called()
            self.assertFalse(path.exists())

    def test_ai_gets_real_pre_release_forecasts_and_comparison(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expectations.sqlite3"
            expectations.capture(path, [{"title": "Core CPI m/m", "country": "USD",
                                        "date": "2026-09-11T08:30:00-04:00", "forecast": "0.2%", "previous": "0.3%"}], now)
            brief = expectations.preparation_brief(path, now)
        self.assertIn("预期0.2%", brief)
        self.assertIn("前值0.3%", brief)
        self.assertIn("-0.1个百分点", brief)
        self.assertIn("不证明市场已经充分定价", brief)
        with patch.object(bot.expectations, "preparation_brief", return_value=brief):
            self.assertIn("预期0.2%", bot._macro_line())

    def test_no_future_event_is_added_only_by_forecast_source(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
        for row in ({"title": "Core CPI m/m", "country": "USD", "date": "2026-09-12T08:30:00-04:00"},
                    {"title": "Core CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                expectations.normalise([row], now)

    def test_normalising_does_not_mutate_original_payload(self):
        row = {"title": "Core CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "forecast": "0.0%", "previous": ""}
        original = json.dumps(row, sort_keys=True)
        expectations.normalise([row], datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(json.dumps(row, sort_keys=True), original)

    def test_public_feed_failure_is_visible_in_ai_material(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
        view = {"events": [], "note": "事前预期", "collector_status": "error"}
        with patch.object(expectations, "snapshot", return_value=view):
            self.assertIn("error", expectations.preparation_brief(now=now))


if __name__ == "__main__":
    unittest.main()
