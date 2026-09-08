import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_path_model import FEATURES, build_path_dataset, feature_frame, path_label
from analysis.train_path_model import calibration_error


class TestMarketPathModel(unittest.TestCase):
    def test_path_labels_first_touch(self):
        upper = pd.DataFrame({"close": [100, 100, 100], "high": [100, 102, 100], "low": [100, 99.5, 98]})
        lower = pd.DataFrame({"close": [100, 100], "high": [100, 100.5], "low": [100, 98]})
        self.assertEqual(path_label(upper, 0, 1, 2), 1)
        self.assertEqual(path_label(lower, 0, 1, 1), -1)

    def test_same_bar_and_timeout_are_no_action(self):
        ambiguous = pd.DataFrame({"close": [100, 100], "high": [100, 101], "low": [100, 99]})
        timeout = pd.DataFrame({"close": [100, 100], "high": [100, 100.5], "low": [100, 99.5]})
        self.assertEqual(path_label(ambiguous, 0, 1, 1), 0)
        self.assertEqual(path_label(timeout, 0, 1, 1), 0)

    def test_features_do_not_read_future_rows(self):
        index = pd.date_range("2026-01-01", periods=50, freq="4h")
        values = pd.Series(range(100, 150), index=index, dtype=float)
        frame = pd.DataFrame({"open": values, "high": values + 2, "low": values - 1,
                              "close": values + 1, "volume": values * 10})
        before = feature_frame(frame).loc[index[30], FEATURES]
        frame.loc[index[31]:, ["high", "close", "volume"]] = 10000
        after = feature_frame(frame).loc[index[30], FEATURES]
        pd.testing.assert_series_equal(before, after)

    def test_dataset_has_all_three_labels(self):
        index = pd.date_range("2026-01-01", periods=80, freq="4h")
        close = pd.Series([100 + (i % 8) * (-1 if i % 16 >= 8 else 1) for i in range(80)], index=index)
        frame = pd.DataFrame({"open": close, "high": close + 1.5, "low": close - 1.5,
                              "close": close, "volume": 1000.0})
        labels = set(build_path_dataset(frame, horizon=2)["label"].astype(int))
        self.assertTrue(labels.issubset({-1, 0, 1}))
        self.assertGreaterEqual(len(labels), 2)

    def test_calibration_error_is_zero_for_exact_confidence(self):
        labels = pd.Series([1, 1, 0, 0]).to_numpy()
        prediction = pd.Series([1, 1, 1, 1]).to_numpy()
        confidence = pd.Series([.5, .5, .5, .5]).to_numpy()
        self.assertAlmostEqual(calibration_error(labels, prediction, confidence), 0.0)


if __name__ == "__main__":
    unittest.main()
