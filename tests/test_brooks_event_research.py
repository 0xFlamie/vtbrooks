import unittest

import numpy as np
import pandas as pd

from analysis.evaluate_brooks_events_v2 import evaluate_fold, trade_summary
from brooks_event_research import (
    MODEL_FEATURES, candidates, event_context, event_masks, non_overlapping,
    purged_window, trade_outcome, validate_bars,
)


def fixture():
    rng = np.random.default_rng(8)
    close = 100 + np.cumsum(rng.normal(0, .8, 400))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + .4,
                         "low": np.minimum(opening, close) - .4, "close": close, "volume": 1000.0},
                        index=pd.date_range("2024-01-01", periods=400, freq="4h"))


def trade_bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


class TestExecution(unittest.TestCase):
    def test_next_open_not_signal_close_is_entry(self):
        bars = trade_bars([(90, 90, 90, 90), (100, 101.1, 99.5, 100)])
        outcome = trade_outcome(bars, 0, 1, 1, horizon=1)
        self.assertEqual(outcome["entry"], 100)
        self.assertEqual(outcome["outcome"], "target")
        self.assertAlmostEqual(outcome["net_return_pct"], .86)

    def test_ambiguous_bar_is_stop_not_later_profit(self):
        bars = trade_bars([(100, 100, 100, 100), (100, 102, 98, 100), (100, 105, 99, 104)])
        for direction in (1, -1):
            result = trade_outcome(bars, 0, direction, 1, horizon=2)
            self.assertEqual(result["outcome"], "ambiguous_stop")
            self.assertEqual(result["exit_index"], 1)
            self.assertAlmostEqual(result["net_return_pct"], -1.14)

    def test_stop_gap_fills_at_open(self):
        bars = trade_bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 98, 96, 97)])
        result = trade_outcome(bars, 0, 1, 1, horizon=2)
        self.assertEqual(result["outcome"], "stop_gap")
        self.assertAlmostEqual(result["net_return_pct"], -3.14)

    def test_short_stop_gap_fills_at_open(self):
        bars = trade_bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (103, 104, 102, 103)])
        self.assertAlmostEqual(trade_outcome(bars, 0, -1, 1, horizon=2)["net_return_pct"], -3.14)

    def test_timeout_keeps_actual_signed_return(self):
        bars = trade_bars([(100, 100, 100, 100), (100, 100.5, 99.5, 100.4)])
        for direction in (1, -1):
            result = trade_outcome(bars, 0, direction, 1, horizon=1)
            self.assertEqual(result["outcome"], "timeout")
            self.assertAlmostEqual(result["net_return_pct"], direction * .4 - .14)

    def test_rejects_incomplete_future(self):
        with self.assertRaises(ValueError):
            trade_outcome(trade_bars([(100, 100, 100, 100)]), 0, 1, 1)


class TestEvents(unittest.TestCase):
    def test_future_changes_do_not_change_events_or_features(self):
        frame = fixture()
        before = candidates(frame)
        self.assertGreater(len(before[before.signal_index <= 280]), 0)
        changed = frame.copy()
        changed.iloc[281:, :4] *= 2
        after = candidates(changed)
        pd.testing.assert_frame_equal(before[before.signal_index <= 280].reset_index(drop=True),
                                      after[after.signal_index <= 280].reset_index(drop=True))

    def test_state_excludes_signal_bar(self):
        frame = fixture()
        before = event_context(frame).iloc[250][["state", "trend_direction"]]
        frame.iloc[250, frame.columns.get_loc("close")] *= 1.5
        frame.iloc[250, frame.columns.get_loc("high")] *= 1.5
        after = event_context(frame).iloc[250][["state", "trend_direction"]]
        pd.testing.assert_series_equal(before, after)

    def test_range_failure_needs_range_environment(self):
        frame = fixture()
        facts = event_context(frame)
        facts["state"] = "trend"
        masks = event_masks(frame, facts)
        self.assertFalse(masks[("range_failed_breakout", 1)].any())
        self.assertFalse(masks[("range_failed_breakout", -1)].any())

    def test_rule_matches_are_symmetric_under_price_reflection(self):
        frame = fixture()
        original = event_masks(frame, event_context(frame))
        mirror = frame.copy()
        mirror["open"], mirror["close"] = 300 - frame.open, 300 - frame.close
        mirror["high"], mirror["low"] = 300 - frame.low, 300 - frame.high
        reversed_masks = event_masks(mirror, event_context(mirror))
        for (setup, direction), mask in original.items():
            pd.testing.assert_series_equal(mask, reversed_masks[(setup, -direction)], check_names=False)

    def test_data_gaps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "连续4H"):
            validate_bars(fixture().drop(fixture().index[30]))

    def test_invalid_ohlc_is_rejected(self):
        frame = fixture()
        frame.iloc[20, frame.columns.get_loc("high")] = 1
        with self.assertRaisesRegex(ValueError, "OHLC"):
            validate_bars(frame)

    def test_full_label_window_cannot_cross_split(self):
        events = pd.DataFrame({"signal_index": [95, 96, 97, 100], "label_end_index": [98, 99, 100, 103]})
        self.assertEqual(purged_window(events, 0, 100).signal_index.tolist(), [95, 96])

    def test_overlapping_signals_cannot_open_multiple_positions(self):
        events = pd.DataFrame({"signal_index": [10, 11, 13], "exit_index": [13, 12, 16],
                               "setup": "trend_pullback", "direction": 1})
        self.assertEqual(non_overlapping(events).signal_index.tolist(), [10, 13])


class TestEvaluation(unittest.TestCase):
    def test_profit_and_target_rates_are_distinct(self):
        events = pd.DataFrame({"signal_index": [10, 20], "exit_index": [13, 23], "direction": 1,
                               "setup": "trend_pullback", "signal_time": pd.date_range("2024-01-01", periods=2),
                               "target_hit": [1, 0], "net_return_pct": [1.0, .2],
                               "outcome": ["target", "timeout"]})
        summary = trade_summary(events)
        self.assertEqual(summary["target_rate"], .5)
        self.assertEqual(summary["net_win_rate"], 1)

    def test_calibrated_training_and_empty_selection_are_supported(self):
        rng = np.random.default_rng(9)
        events = pd.DataFrame(rng.normal(size=(500, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
        events["signal_index"] = np.arange(500) * 5
        events["label_end_index"] = events.signal_index + 3
        events["exit_index"] = events.label_end_index
        events["setup"], events["direction"] = "trend_pullback", 1
        events["signal_time"] = pd.date_range("2024-01-01", periods=500, freq="20h")
        events["target_hit"] = (np.arange(500) % 3 == 0).astype(int)
        events["outcome"] = np.where(events.target_hit == 1, "target", "stop")
        events["net_return_pct"] = np.where(events.target_hit == 1, .86, -1.14)
        summary, chosen = evaluate_fold(events, "trend_pullback", (1250, 1750, 2500), 1)
        self.assertEqual(summary["status"], "evaluated")
        self.assertEqual(summary["train_n"], 250)
        self.assertEqual(summary["calibration_n"], 100)
        self.assertTrue(np.isfinite(summary["brier_calibrated"]))
        self.assertTrue(chosen.empty or chosen.probability.ge(.6).all())


if __name__ == "__main__":
    unittest.main()
