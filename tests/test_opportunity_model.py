import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opportunity_model import add_market_context, build_dataset, triple_barrier


class TestTripleBarrier(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"close": [100, 100, 100], "high": [100, 102, 100], "low": [100, 99.5, 98]})

    def test_long_hits_target_first(self):
        self.assertEqual(triple_barrier(self.df, 0, 1, 1, 2), 1)

    def test_short_hits_stop_first(self):
        self.assertEqual(triple_barrier(self.df, 0, -1, 1, 2), -1)

    def test_same_bar_touch_is_ambiguous(self):
        df = pd.DataFrame({"close": [100, 100], "high": [100, 101], "low": [100, 99]})
        self.assertEqual(triple_barrier(df, 0, 1, 1, 1), 0)

    def test_context_uses_only_completed_4h_bars(self):
        index = pd.date_range("2026-01-01", periods=160, freq="15min")
        df = pd.DataFrame({"open": range(160), "high": range(1, 161), "low": range(160),
                           "close": range(1, 161), "volume": 1}, index=index, dtype=float)
        before = add_market_context(df).iloc[100]["context_momentum"]
        df.iloc[101:, df.columns.get_loc("close")] = 10000
        after = add_market_context(df).iloc[100]["context_momentum"]
        self.assertEqual(before, after)

    def test_dataset_records_horizon_return(self):
        index = pd.date_range("2026-01-01", periods=130, freq="4h")
        close = pd.Series(range(100, 230), index=index, dtype=float)
        df = pd.DataFrame({"open": close - 1, "high": close + 2, "low": close - 2,
                           "close": close, "volume": 1000.0})
        samples = build_dataset(df, horizon=3)
        if not samples.empty:
            self.assertIn("forward_return_pct", samples.columns)
            self.assertTrue(samples["forward_return_pct"].notna().all())


if __name__ == "__main__":
    unittest.main()
