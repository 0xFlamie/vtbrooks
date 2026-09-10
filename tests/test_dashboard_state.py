import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dashboard_state as state
import vt_vote_bot as bot


class TestDashboardState(unittest.TestCase):
    def test_old_combined_reasons_do_not_cross_timeframes(self):
        value = state.display_layers({}, {"reasons": ["4-A", "4-B", "15-C"]}, "ETHUSDC")
        self.assertEqual(value["4h"]["latest_reasons"], ["4-A", "4-B"])
        self.assertEqual(value["15m"]["latest_reasons"], ["15-C"])
        self.assertEqual(state.display_layers({}, {}, "ETHUSDC")["15m"]["latest_reasons"], [])

    def test_ambiguous_short_legacy_reason_list_is_not_assigned_to_a_period(self):
        value = state.display_layers({}, {"reasons": ["可能4h", "可能15m"]}, "ETHUSDC")
        self.assertEqual(value["4h"]["latest_reasons"], [])
        self.assertEqual(value["15m"]["latest_reasons"], [])

    def test_newest_display_uses_post_veto_not_technical_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.json"
            judge = {"direction": "SHORT", "confidence": 70, "summary": "闸门否决", "reasons": ["风险原因"],
                     "decision_meta": {"id": "technical-1", "available_at": "2026-09-11T00:00:00Z"}}
            state.save_current("ETHUSDC", judge, judge, path)
            data = json.loads(path.read_text())
            view = state.display_layers(data, {"dir4h": "LONG"}, "ETHUSDC")
            self.assertEqual(view["4h"]["direction"], "SHORT")
            self.assertEqual(view["4h"]["decision_id"], "technical-1")
            self.assertEqual(view["4h"]["decision_at"], "2026-09-11T00:00:00Z")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_same_direction_new_reasons_are_saved_and_previous_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.json"
            old = {"direction": "LONG", "summary": "守住100", "reasons": ["缩量"]}
            new = {**old, "summary": "突破102", "reasons": ["放量"]}
            state.save_current("ETHUSDC", old, old, path)
            state.save_current("ETHUSDC", new, new, path)
            state.save_current("ETHUSDC", new, new, path)
            view = state.display_layers(json.loads(path.read_text()), {}, "ETHUSDC")
            self.assertEqual(view["15m"]["previous_summary"], "守住100")
            self.assertEqual(view["15m"]["latest_reasons"], ["放量"])
            self.assertIn("依据已更新", view["15m"]["change"])
            self.assertFalse(list(Path(directory).glob(".dashboard-*")))

    def test_damaged_snapshot_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.json"
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                state.save_current("ETHUSDC", {}, {}, path)
            self.assertEqual(path.read_text(), '{broken')

    def test_current_snapshot_written_even_when_hourly_journal_deduplicates(self):
        import pandas as pd
        result = {"symbol": "ETHUSDC", "signal": "LONG"}
        judge = {"direction": "LONG", "summary": "新依据"}
        journal = {"entries": [{"symbol": "ETHUSDC", "time": pd.Timestamp.now().isoformat(),
                                "dir4h": "LONG", "dir15m": "LONG"}]}
        with patch.object(bot, "save_current") as save, patch.object(bot, "compute_levels", return_value={}), \
             patch.object(bot, "load_journal", return_value=journal), patch.object(bot, "save_journal") as history:
            bot.record_judge(result, judge, judge)
        save.assert_called_once_with("ETHUSDC", judge, judge)
        history.assert_not_called()

    def test_previous_neutral_view_and_timestamp_are_available_to_manager(self):
        text = bot.previous_decision_context({"direction": None, "summary": "等突破", "reasons": ["区间中部"],
                                              "decision_meta": {"available_at": "2026-09-11T00:00:00Z"}})
        for value in ("等突破", "区间中部", "2026-09-11T00:00:00Z", "旧观点"):
            self.assertIn(value, text)
        self.assertIn("不把confidence写成盈利概率", bot.MANAGER_SYSTEM)

    def test_invalid_current_shapes_fall_back_to_legacy(self):
        for value in ([], {"version": 2}, {"version": 1, "symbols": []},
                      {"version": 1, "symbols": {"ETHUSDC": []}}):
            with self.subTest(value=value):
                result = state.display_layers(value, {"dir4h": "SHORT"}, "ETHUSDC")
                self.assertEqual(result["4h"]["direction"], "SHORT")
                self.assertEqual(result["4h"]["source"], "legacy_journal")

    def test_15m_manager_receives_previous_thesis_not_only_direction(self):
        previous = {"direction": None, "summary": "上次等回踩", "reasons": ["上次量能不足"]}
        with patch.object(bot, "build_market_brief", return_value="当前已回踩"), \
             patch.object(bot, "multi_agent_judge", return_value={"direction": None}) as judge, \
             patch.object(bot, "audited_decision", side_effect=lambda symbol, result, tf, brief: result):
            bot.ai_judge_15m({"symbol": "ETHUSDC"}, prev=previous)
        brief = judge.call_args.args[0]
        for value in ("当前已回踩", "上次等回踩", "上次量能不足", "旧观点"):
            self.assertIn(value, brief)


if __name__ == "__main__":
    unittest.main()
