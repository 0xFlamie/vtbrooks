import unittest
import json
from pathlib import Path
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import macro_events as macro
import vt_vote_bot as bot


def at(value):
    return datetime.fromisoformat(value)


class TestMacroEvents(unittest.TestCase):
    def test_priority_and_components_are_explicit(self):
        catalog = macro.CATALOG
        for key in ("cpi", "ppi", "pce", "employment", "fomc", "projections", "press_conference"):
            self.assertEqual(catalog[key]["priority"], "A")
        self.assertIn("失业率", catalog["employment"]["components"])
        self.assertIn("核心环比", catalog["ppi"]["components"])
        self.assertEqual(catalog["claims"]["priority"], "B")
        self.assertEqual(catalog["liquidity"]["priority"], "C")

    def test_ppi_is_present_and_october_cpi_date_corrected(self):
        self.assertIn("PPI", macro.day_label("2026-09-10"))
        self.assertNotIn("CPI", macro.day_label("2026-10-13"))
        self.assertIn("CPI", macro.day_label("2026-10-14"))

    def test_dst_uses_new_york_not_fixed_beijing_hour(self):
        summer = next(e for e in macro.events_on("2026-09-11") if e["kind"] == "cpi")
        winter = next(e for e in macro.events_on("2026-12-10") if e["kind"] == "cpi")
        self.assertEqual(summer["release_at"].astimezone(macro.BEIJING).hour, 20)
        self.assertEqual(winter["release_at"].astimezone(macro.BEIJING).hour, 21)

    def test_same_day_does_not_overwrite_retail_or_fomc(self):
        kinds = {e["kind"] for e in macro.events_on("2026-09-16")}
        self.assertTrue({"retail", "fomc", "projections", "press_conference"} <= kinds)
        after_statement = macro.pending(at("2026-09-16T18:01:00+00:00"))
        self.assertEqual([e["kind"] for e in after_statement], ["press_conference"])

    def test_pce_and_gdp_are_independent_events(self):
        kinds = {e["kind"] for e in macro.events_on("2026-09-30")}
        self.assertTrue({"adp", "gdp", "pce"} <= kinds)

    def test_preview_expires_exactly_at_release_not_two_hours_later(self):
        before = at("2026-09-11T12:29:59+00:00")
        preview = {"date": "2026-09-11", "event_ids": macro.pending_ids(before)}
        self.assertTrue(macro.preview_is_current(preview, before))
        self.assertFalse(macro.preview_is_current(preview, at("2026-09-11T12:30:00+00:00")))
        self.assertFalse(macro.preview_is_current({"date": "2026-09-11"}, before))

    def test_preparation_is_keyed_to_next_events_and_half_hour_refresh(self):
        early = at("2026-09-11T10:00:00+00:00")
        late = at("2026-09-11T12:00:00+00:00")
        self.assertNotEqual(macro.preparation_key(early), macro.preparation_key(late))
        self.assertEqual(macro.preparation_key(late), macro.preparation_key(at("2026-09-11T12:10:00+00:00")))

    def test_brief_distinguishes_scheduled_time_from_actual_arrival(self):
        before = macro.brief(at("2026-09-11T12:29:00+00:00"))
        after = macro.brief(at("2026-09-11T12:31:00+00:00"))
        self.assertIn("未到公布时刻", before)
        self.assertIn("已到计划公布时间", after)
        self.assertIn("不代表已收到实际值", after)
        self.assertIn("市场预期以单独事前记录为准", after)
        self.assertIn("不能把方向偏空等同于现在追空", after)

    def test_unknown_schedules_and_expired_coverage_are_visible(self):
        self.assertIn("treasury_refunding", macro.coverage()["unscheduled"])
        self.assertIn("日历覆盖期外", macro.brief(at("2027-01-05T12:00:00+00:00")))
        self.assertIn("需重新核验", macro.brief(at("2026-12-10T12:00:00+00:00")))

    def test_event_window_checks_all_releases(self):
        self.assertTrue(macro.in_window(at("2026-09-16T17:45:00+00:00")))
        self.assertTrue(macro.in_window(at("2026-09-10T12:30:00+00:00")))
        self.assertFalse(macro.in_window(at("2026-09-10T06:00:00+00:00")))

    def test_claims_respects_thanksgiving_exception(self):
        self.assertIn("claims", {e["kind"] for e in macro.events_on("2026-11-25")})
        self.assertNotIn("claims", {e["kind"] for e in macro.events_on("2026-11-26")})

    def test_michigan_preliminary_and_final_are_not_mixed(self):
        self.assertIn("初值", macro.day_label("2026-09-11"))
        self.assertIn("终值", macro.day_label("2026-09-25"))

    def test_one_event_expiring_invalidates_whole_old_preview(self):
        before = at("2026-09-11T12:29:00+00:00")
        after = at("2026-09-11T12:31:00+00:00")
        preview = {"date": "2026-09-11", "event_ids": macro.pending_ids(before)}
        self.assertFalse(macro.preview_is_current(preview, after))
        self.assertEqual([e["kind"] for e in macro.pending(after)], ["michigan"])

    def test_corrupt_snapshot_is_not_silently_an_empty_calendar(self):
        payload = json.loads(json.dumps(macro.CALENDAR))
        payload["groups"].append(payload["groups"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "重复"):
                macro.load_calendar(path)

    def test_data_feed_status_cannot_claim_realtime(self):
        coverage = macro.coverage()
        self.assertEqual(coverage["actual_feed"], "not_connected")
        self.assertEqual(coverage["consensus_feed"], "not_connected")

    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            macro.pending(at("2026-09-11T12:00:00"))

    def test_bot_preview_never_calls_ai_after_all_releases(self):
        with patch.object(bot.macro, "pending", return_value=[]), patch.object(bot._http, "post") as post:
            self.assertIsNone(bot.event_preview("ETHUSDC"))
        post.assert_not_called()

    def test_bot_uses_calendar_for_ppi_window(self):
        with patch.object(bot.pd.Timestamp, "now", return_value=at("2026-09-10T12:30:00+00:00")):
            self.assertTrue(bot.in_event_window())
            self.assertIn("PPI", bot._macro_line())

    def test_bot_compatibility_time_never_assumes_unknown_is_fomc(self):
        with patch.object(bot.pd.Timestamp, "now", return_value=at("2026-09-10T12:30:00+00:00")):
            release = bot._event_release_et("PPI生产者通胀")
            self.assertEqual((release.hour, release.minute), (8, 30))
            with self.assertRaises(ValueError):
                bot._event_release_et("未知事件")

    def test_cached_preparation_avoids_repeated_ai_calls(self):
        now = at("2026-09-11T08:00:00-04:00")
        key = macro.preparation_key(now)
        key["consensus_version"] = bot.expectations.preparation_version(now=now)
        preview = {"date": "2026-09-11", "event_ids": key["event_ids"], "preparation_key": key, "text": "条件预案"}
        with patch.object(bot.pd.Timestamp, "now", return_value=now), \
                patch.object(bot, "load_news_memory", return_value={"event_preview": preview}), \
                patch.object(bot._http, "post") as post:
            self.assertEqual(bot.event_preview("ETHUSDC"), "条件预案")
        post.assert_not_called()

    def test_slow_ai_result_crossing_release_is_not_saved(self):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "条件预案"}}]}
        with patch.object(bot.pd.Timestamp, "now", return_value=at("2026-09-11T08:29:59-04:00")), \
                patch.object(bot, "load_news_memory", return_value={}), \
                patch.object(bot, "compute_levels", return_value=None), \
                patch.object(bot, "polymarket_odds", return_value=[]), \
                patch.object(bot, "fetch_fast_price", return_value=2400), \
                patch.object(bot._http, "post", return_value=response), \
                patch.object(bot.macro, "preview_is_current", side_effect=[False, False]), \
                patch.object(bot, "save_news_memory") as save:
            self.assertIsNone(bot.event_preview("ETHUSDC"))
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
