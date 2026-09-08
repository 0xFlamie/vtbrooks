import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.evaluate_judges import direction_return, evaluate, factor_alignment


class TestJudgeEvaluation(unittest.TestCase):
    def test_factor_alignment_uses_recorded_direction(self):
        entry = {"factor_evidence": [
            {"name": "RSI", "direction": "🟢"},
            {"name": "量能", "direction": "🔴"},
        ]}
        self.assertEqual(factor_alignment(entry, "SHORT"), ["量能"])

    def test_observation_has_no_alignment(self):
        self.assertEqual(factor_alignment({}, None), [])

    def test_one_bar_exit_is_entry_bar_close(self):
        frame = pd.DataFrame({"open": [100, 100, 100], "close": [101, 102, 110]},
                             index=pd.date_range("2024-01-01", periods=3, freq="4h"))
        self.assertAlmostEqual(direction_return(frame, "2024-01-01T00:01Z", "LONG", 1), 2)
        self.assertAlmostEqual(direction_return(frame, "2024-01-01T04:00Z", "SHORT", 1), -2)
        self.assertAlmostEqual(direction_return(frame, "2024-01-01T00:01Z", "LONG", 2), 10)

    def test_gap_cannot_be_counted_as_contiguous_window(self):
        index = pd.date_range("2024-01-01", periods=7, freq="4h").delete(2)
        frame = pd.DataFrame({"open": 100, "close": 101}, index=index)
        self.assertIsNone(direction_return(frame, "2024-01-01T00:01Z", "LONG", 3))

    def test_cached_judge_is_not_counted_twice_and_legacy_defaults_excluded(self):
        frame = pd.DataFrame({"open": 100, "close": 101}, index=pd.date_range("2024-01-01", periods=10, freq="4h"))
        entry = {"symbol": "ETHUSDC", "time": "2024-01-01T00:01Z", "dir4h": "LONG",
                 "decision_meta": {"4h": {"id": "once", "available_at": "2024-01-01T00:01Z"}}}
        self.assertEqual(len(evaluate([entry, {**entry, "time": "2024-01-01T01:01Z"}], frame, frame)), 2)
        self.assertEqual(evaluate([{**entry, "decision_meta": {}}], frame, frame), [])

    def test_invalid_direction_is_not_assumed_short(self):
        frame = pd.DataFrame({"open": [100, 100], "close": [101, 102]}, index=pd.date_range("2024-01-01", periods=2, freq="4h"))
        self.assertIsNone(direction_return(frame, frame.index[0], "UNKNOWN", 1))


if __name__ == "__main__":
    unittest.main()
