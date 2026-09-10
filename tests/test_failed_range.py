import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from brooks_failed_range import (advance_attempt, match_range_controls, observe_range_bar,
                                 scan_failed_range, summarize_process, validate_minutes)


def fixture(closes):
    stamp = pd.Timestamp("2024-01-01T00:00:00Z")
    prices = np.array(closes, float)
    frame = pd.DataFrame({"open": prices, "high": prices + .1, "low": prices - .1,
                          "close": prices, "volume": 1.}, index=pd.date_range(stamp, periods=len(prices), freq="min"))
    return stamp, frame


class TestFailedRange(unittest.TestCase):
    def test_three_observations_long(self):
        stamp, group = fixture([89., 91., 92.])
        signal, attempt = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertEqual(attempt["reason"], "signal")
        self.assertEqual(signal["direction"], 1)
        self.assertEqual(signal["trigger_reference"], 91.1)
        self.assertEqual(signal["structure_reference"], 88.9)
        self.assertLess(signal["breakout_at"], signal["reclaim_at"])
        self.assertLess(signal["reclaim_at"], signal["available_at"])
        self.assertEqual(signal["lead_minutes"], 12)

    def test_short_mirror(self):
        stamp, group = fixture([89., 91., 92.])
        long, _ = observe_range_bar(group, stamp, 90., 110., 2.)
        mirrored = group.assign(open=200-group.open, high=200-group.low, low=200-group.high, close=200-group.close)
        short, _ = observe_range_bar(mirrored, stamp, 90., 110., 2.)
        self.assertEqual(short["direction"], -1)
        self.assertEqual(short["available_at"], long["available_at"])
        self.assertAlmostEqual(short["trigger_reference"], 200-long["trigger_reference"])

    def test_wick_not_observed_breakout(self):
        stamp, group = fixture([91., 92., 93.])
        group.iloc[0, group.columns.get_loc("low")] = 89.
        signal, attempt = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertIsNone(signal)
        self.assertIsNone(attempt["state"])

    def test_reclaim_wick_cannot_confirm_same_minute(self):
        stamp, group = fixture([89., 91.])
        group.iloc[1, group.columns.get_loc("high")] = 95.
        signal, attempt = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertIsNone(signal)
        self.assertEqual(attempt["state"]["stage"], "inside")

    def test_return_outside_cancels_no_resurrection(self):
        stamp, group = fixture([89., 91., 90., 93., 94.])
        signal, attempt = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertIsNone(signal)
        self.assertEqual(attempt["reason"], "returned_outside")

    def test_center_or_opposite_cross_cancels_before_signal(self):
        for closes in ([89., 100., 99.], [89., 91., 100.], [89., 111., 92.]):
            stamp, group = fixture(closes)
            signal, attempt = observe_range_bar(group, stamp, 90., 110., 2.)
            self.assertIsNone(signal)
            self.assertEqual(attempt["reason"], "midpoint_consumed")

    def test_prefix_pending_and_complete_expiry(self):
        stamp, group = fixture([89.] * 15)
        self.assertEqual(observe_range_bar(group.iloc[:3], stamp, 90., 110., 2.)[1]["reason"], "pending")
        self.assertEqual(observe_range_bar(group, stamp, 90., 110., 2.)[1]["reason"], "expired")

    def test_gap_and_zero_before_signal_fail_closed(self):
        stamp, group = fixture([89., 91., 92., 93.])
        broken = group.drop(group.index[1])
        zero = group.copy()
        zero.iloc[1, zero.columns.get_loc("volume")] = 0.
        for frame in (broken, zero):
            signal, attempt = observe_range_bar(frame, stamp, 90., 110., 2.)
            self.assertIsNone(signal)
            self.assertEqual(attempt["reason"], "quality_reset")

    def test_post_signal_failure_does_not_erase(self):
        stamp, group = fixture([89., 91., 92., 88., 87.])
        expected = observe_range_bar(group.iloc[:3], stamp, 90., 110., 2.)
        self.assertEqual(observe_range_bar(group, stamp, 90., 110., 2.), expected)
        group.iloc[-1, group.columns.get_loc("volume")] = 0.
        self.assertEqual(observe_range_bar(group, stamp, 90., 110., 2.), expected)

    def test_only_one_attempt_per_bar(self):
        stamp, group = fixture([89., 91., 92., 111., 109., 108.])
        signal, _ = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertEqual(signal["direction"], 1)

    def test_equal_trigger_not_confirmation(self):
        minute = SimpleNamespace(close=91., high=91.1, low=90.9)
        time = pd.Timestamp("2024-01-01T00:01Z")
        state = {"direction": 1, "stage": "inside", "edge": 90., "trigger_reference": 91.}
        self.assertEqual(advance_attempt(state, minute, time, 90., 110.)[1], "waiting")

    def test_confirmation_must_cross_inward_not_outward_extreme(self):
        stamp, group = fixture([89., 91., 91.05, 91.2])
        signal, _ = observe_range_bar(group, stamp, 90., 110., 2.)
        self.assertEqual(signal["available_at"], (stamp + pd.Timedelta(minutes=4)).isoformat())

    def test_end_is_exclusive_and_missing_current_tail_does_not_leak(self):
        stamp, frame = fixture([100 + .4 * (-1)**i for i in range(15 * 125)])
        live = stamp + pd.Timedelta(minutes=15 * 124)
        frame.loc[live, ["open", "high", "low", "close"]] = [99., 99.1, 98.9, 99.]
        frame.loc[live + pd.Timedelta(minutes=1), ["open", "high", "low", "close"]] = [99.6, 99.7, 99.55, 99.6]
        frame.loc[live + pd.Timedelta(minutes=2), ["open", "high", "low", "close"]] = [99.8, 99.9, 99.75, 99.8]
        time = live + pd.Timedelta(minutes=3)
        self.assertEqual(scan_failed_range(frame, live, time)[0], [])
        full = scan_failed_range(frame, live, time + pd.Timedelta(nanoseconds=1))[0]
        missing = frame.drop(frame.index[-2:])
        self.assertEqual(scan_failed_range(missing, live, time + pd.Timedelta(nanoseconds=1))[0], full)
        frame.iloc[10, frame.columns.get_loc("volume")] = 0.
        # 前120根内的质量问题才应阻断，早于窗口的缺量不应永久污染。
        self.assertEqual(scan_failed_range(frame, live, time + pd.Timedelta(nanoseconds=1))[0], full)
        frame.loc[live - pd.Timedelta(minutes=1), "volume"] = 0.
        self.assertEqual(scan_failed_range(frame, live, time + pd.Timedelta(nanoseconds=1))[0], [])

    def test_empty_cohort(self):
        _, frame = fixture([])
        self.assertEqual(scan_failed_range(frame, frame.index.min(), frame.index.max()), ([], []))
        result = summarize_process([], [], [0, 1])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["forward_minus_reverse"])

    def test_invalid_inputs(self):
        _, frame = fixture([91., 92., 93.])
        for column, value in (("close", np.nan), ("high", 1.), ("volume", -1.)):
            broken = frame.copy()
            broken.iloc[0, broken.columns.get_loc(column)] = value
            with self.assertRaises(ValueError):
                validate_minutes(broken)
        with self.assertRaises(ValueError):
            validate_minutes(frame.iloc[::-1])
        with self.assertRaises(ValueError):
            validate_minutes(pd.concat([frame, frame]))

    def test_matching_requires_range_not_just_neutral_context(self):
        stamp, minutes = fixture([91., 92., 93.])
        frame = pd.DataFrame({"eligible": [True, True]}, index=[stamp, stamp + pd.Timedelta(minutes=15)])
        ranges = pd.DataFrame({"is_range": [True, False]}, index=frame.index)
        signal = {"available_at": (stamp + pd.Timedelta(minutes=3)).isoformat()}
        with patch("brooks_failed_range.backgrounds", return_value=(frame, {})), \
                patch("brooks_failed_range.range_background", return_value=(ranges, {})), \
                patch("brooks_failed_range.choose_control", return_value=None) as choose:
            self.assertEqual(match_range_controls(minutes, [signal], stamp, stamp + pd.Timedelta(days=1)), [None])
        self.assertEqual(choose.call_args.args[0].eligible.tolist(), [True, False])

    def test_scan_prefix_with_previous_bars_only(self):
        stamp, frame = fixture([100 + .4 * (-1)**i for i in range(15 * 125)])
        live = stamp + pd.Timedelta(minutes=15 * 124)
        frame.loc[live:live + pd.Timedelta(minutes=2), ["open", "high", "low", "close"]] = [99., 99.1, 98.9, 99.]
        frame.loc[live + pd.Timedelta(minutes=1), ["open", "high", "low", "close"]] = [99.6, 99.7, 99.55, 99.6]
        frame.loc[live + pd.Timedelta(minutes=2), ["open", "high", "low", "close"]] = [99.8, 99.9, 99.75, 99.8]
        time = live + pd.Timedelta(minutes=3)
        full, _ = scan_failed_range(frame, live, live + pd.Timedelta(minutes=15))
        prefix, _ = scan_failed_range(frame.loc[frame.index < time], live, time + pd.Timedelta(nanoseconds=1))
        self.assertEqual(len(full), 1)
        self.assertEqual(full, prefix)
        self.assertEqual(full[0]["range_low"], 99.5)


if __name__ == "__main__":
    unittest.main()
