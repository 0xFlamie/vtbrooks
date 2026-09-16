from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

import requests
import macro_expectations as expectations


class TestMacroExpectations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "quotes.sqlite3"
        self.release = datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc)
        self.before = self.release - timedelta(minutes=30)
        self.rows = [{"title": "CPI m/m", "country": "USD", "date": self.release.isoformat(),
                      "forecast": "0.4%", "previous": "0.1%"}]

    def test_only_verified_us_events_and_separate_core_fields(self):
        rows = self.rows + [{**self.rows[0], "country": "EUR"},
                            {**self.rows[0], "title": "Core CPI m/m", "forecast": "0.2%"}]
        quotes = expectations.normalise(rows, self.before)
        self.assertEqual([q["label"] for q in quotes], ["总体环比", "核心环比"])

    def test_busy_day_does_not_truncate_high_priority_events(self):
        now = datetime.fromisoformat("2026-09-30T08:00:00-04:00")
        view = expectations.snapshot(self.path, now, limit=1)
        self.assertEqual({e["kind"] for e in view["events"]}, {"adp", "gdp", "pce"})

    def test_meeting_stages_remain_together_until_last_stage_expires(self):
        now = datetime.fromisoformat("2026-09-16T16:15:00-04:00")
        events = expectations.snapshot(self.path, now)["events"]
        meeting = [e for e in events if e["kind"] in ("fomc", "projections", "press_conference")]
        self.assertEqual(len(meeting), 3)
        self.assertEqual(len({e["group_id"] for e in meeting}), 1)
        self.assertTrue(all(e["phase"] == "awaiting_actual" for e in meeting))
        expired = expectations.snapshot(self.path, now + timedelta(minutes=16))["events"]
        self.assertFalse(any(e["kind"] == "fomc" for e in expired))

    def test_qualitative_stage_does_not_claim_missing_numeric_consensus(self):
        now = datetime.fromisoformat("2026-09-16T10:00:00-04:00")
        events = expectations.snapshot(self.path, now)["events"]
        speech = next(e for e in events if e["kind"] == "press_conference")
        decision = next(e for e in events if e["kind"] == "fomc")
        self.assertEqual(speech["expectation_status"], "qualitative")
        self.assertEqual(decision["expectation_status"], "missing")
        self.assertIn("问答", speech["focus"])

    def test_policy_playbook_covers_action_path_conflict_and_invalidation(self):
        decision = " ".join(expectations.scenario_notes("fomc"))
        for word in ("加息", "维持", "降息", "缩表", "预期"):
            self.assertIn(word, decision)
        self.assertIn("承诺", " ".join(expectations.scenario_notes("projections")))
        self.assertIn("冲突", " ".join(expectations.scenario_notes("press_conference")))
        self.assertIn("旧会议", " ".join(expectations.scenario_notes("minutes")))

    def test_employment_cost_uses_wages_and_benefits_instead_of_cpi_components(self):
        notes = " ".join(expectations.scenario_notes("eci"))
        self.assertIn("工资与福利", notes)
        self.assertIn("季度", notes)
        self.assertNotIn("能源/住房", notes)

    def test_ai_material_uses_same_playbook_and_only_nearby_events(self):
        now = datetime.fromisoformat("2026-09-16T14:01:00-04:00")
        brief = expectations.preparation_brief(self.path, now)
        for word in ("加息", "点阵图", "问答", "失效", "来源", "待核实", "已到计划公布时间"):
            self.assertIn(word, brief)
        distant = expectations.preparation_brief(self.path, datetime.fromisoformat("2026-09-11T10:01:00-04:00"))
        self.assertNotIn("零售销售", distant)

    def test_preparation_version_changes_when_playbook_changes(self):
        from unittest.mock import patch
        version = expectations.preparation_version(self.path, self.before)
        with patch.object(expectations, "scenario_notes", return_value=["新的解读条件"]):
            self.assertNotEqual(version, expectations.preparation_version(self.path, self.before))

    def test_post_release_fetch_cannot_backfill_expectation(self):
        expectations.capture(self.path, self.rows, self.release)
        event = expectations.snapshot(self.path, self.release + timedelta(seconds=1))["events"][0]
        self.assertEqual(event["metrics"], [])
        self.assertEqual(event["expectation_status"], "missing")

    def test_pre_release_revisions_are_archived_and_frozen(self):
        expectations.capture(self.path, self.rows, self.before)
        changed = [{**self.rows[0], "forecast": "0.5%"}]
        expectations.capture(self.path, changed, self.before + timedelta(minutes=10))
        expectations.capture(self.path, [{**self.rows[0], "forecast": "9.9%"}], self.release + timedelta(minutes=1))
        event = expectations.snapshot(self.path, self.release + timedelta(minutes=2))["events"][0]
        self.assertEqual(event["metrics"][0]["forecast"], "0.5%")
        self.assertEqual(event["expectation_status"], "frozen")
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM expectation_batches").fetchone()[0], 3)

    def test_batch_at_exact_release_does_not_erase_prior_record(self):
        expectations.capture(self.path, self.rows, self.before)
        expectations.capture(self.path, self.rows, self.release)
        event = expectations.snapshot(self.path, self.release)["events"][0]
        self.assertEqual(event["metrics"][0]["forecast"], "0.4%")

    def test_known_official_previous_conflict_is_not_used_for_comparison(self):
        rows = [{"title": "Prelim UoM Consumer Sentiment", "country": "USD", "date": "2026-09-11T10:00:00-04:00",
                 "forecast": "51.0", "previous": "51.0"}]
        expectations.capture(self.path, rows, self.before)
        event = next(e for e in expectations.snapshot(self.path, self.before)["events"] if e["kind"] == "michigan")
        self.assertEqual(event["expectation_status"], "conflict")
        self.assertIn("51.7", event["metrics"][0]["comparison"])
        self.assertIn("前值待核对", expectations.preparation_brief(self.path, self.before))

    def test_future_fetch_not_visible_to_historical_query(self):
        expectations.capture(self.path, self.rows, self.before)
        view = expectations.snapshot(self.path, self.before - timedelta(minutes=1))
        self.assertEqual(view["events"][0]["metrics"], [])

    def test_official_time_mismatch_and_duplicate_conflict_fail(self):
        bad = [{**self.rows[0], "date": (self.release + timedelta(hours=1)).isoformat()}]
        for rows in (bad, self.rows + [{**self.rows[0], "forecast": "9%"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                expectations.capture(self.path, rows, self.before)

    def test_missing_values_not_zero_and_zero_is_retained(self):
        quotes = expectations.normalise([{**self.rows[0], "forecast": "", "previous": "0.0%"}], self.before)
        self.assertIsNone(quotes[0]["forecast"])
        self.assertEqual(quotes[0]["previous"], "0.0%")
        self.assertIn("缺少", expectations.delta_text(None, "0.0%"))

    def test_unit_comparison_uses_percentage_points(self):
        self.assertEqual(expectations.delta_text("0.4%", "0.1%"), "预期较前值+0.3个百分点")
        self.assertIn("口径不同", expectations.delta_text("200K", "0.2M"))

    def test_missing_storage_read_does_not_create_file(self):
        view = expectations.snapshot(self.path, self.before)
        self.assertFalse(self.path.exists())
        self.assertEqual(view["collector_status"], "not_collected")

    def test_stale_pre_release_quote_cannot_look_fresh(self):
        expectations.capture(self.path, self.rows, self.before - timedelta(hours=3))
        self.assertEqual(expectations.snapshot(self.path, self.before)["events"][0]["expectation_status"], "stale")

    def test_withdrawn_quote_is_not_silently_reused(self):
        expectations.capture(self.path, self.rows, self.before)
        expectations.capture(self.path, [], self.before + timedelta(minutes=1))
        self.assertEqual(expectations.snapshot(self.path, self.before + timedelta(minutes=2))["events"][0]["metrics"], [])

    def test_http_error_is_recorded_and_throttled(self):
        http = Mock()
        http.get.side_effect = requests.Timeout("do not expose transport details")
        self.assertFalse(expectations.refresh(self.path, http, lambda: self.before))
        self.assertFalse(expectations.refresh(self.path, http, lambda: self.before + timedelta(minutes=1)))
        self.assertEqual(http.get.call_count, 1)
        self.assertEqual(expectations.snapshot(self.path, self.before)["collector_status"], "error")

    def test_response_arriving_after_release_is_not_pre_release(self):
        response = Mock(content=b"[]")
        response.json.return_value = self.rows
        http = Mock()
        http.get.return_value = response
        times = iter([self.release - timedelta(seconds=1), self.release + timedelta(seconds=1)])
        expectations.refresh(self.path, http, lambda: next(times))
        self.assertEqual(expectations.snapshot(self.path, self.release + timedelta(seconds=2))["events"][0]["metrics"], [])

    def test_version_changes_when_consensus_changes(self):
        expectations.capture(self.path, self.rows, self.before)
        version = expectations.preparation_version(self.path, self.before)
        expectations.capture(self.path, [{**self.rows[0], "forecast": "0.5%"}], self.before + timedelta(seconds=1))
        self.assertNotEqual(version, expectations.preparation_version(self.path, self.before + timedelta(seconds=2)))

    def test_untrusted_forecast_text_is_rejected(self):
        with self.assertRaises(ValueError):
            expectations.capture(self.path, [{**self.rows[0], "forecast": "<script>"}], self.before)

    def test_raw_source_payload_and_hash_are_preserved(self):
        expectations.capture(self.path, self.rows, self.before)
        with sqlite3.connect(self.path) as db:
            data, digest = db.execute("SELECT payload,sha256 FROM expectation_batches").fetchone()
        self.assertEqual(json.loads(data)["raw"], self.rows)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
