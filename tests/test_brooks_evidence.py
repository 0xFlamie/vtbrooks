import unittest

import numpy as np
import pandas as pd

import vt_vote_bot as bot
from brooks_evidence import evidence_brief, inspect_evidence, quality_check


def bars():
    values = 100 + np.sin(np.arange(130) / 3) + np.arange(130) / 20
    return pd.DataFrame({"open": values-.1, "high": values+.3, "low": values-.3,
                         "close": values, "volume": 100.}, index=pd.date_range("2024-01-01", periods=130, freq="15min"))


class TestBrooksEvidence(unittest.TestCase):
    def test_normal_bars_keep_confirmations_available(self):
        q = quality_check(bars())
        self.assertEqual(q["status"], "ok")
        self.assertFalse(q["blocked"])

    def test_historical_flat_bar_warns_without_blocking_current_bar(self):
        frame = bars()
        frame.iloc[-5, :4] = frame.close.iloc[-5]
        quality = quality_check(frame)
        self.assertEqual(quality["status"], "degraded")
        self.assertEqual(quality["flat_bars"], 1)
        self.assertFalse(quality["blocked"])

    def test_latest_flat_bar_cannot_confirm_entry(self):
        frame = bars()
        frame.iloc[-1, :4] = frame.close.iloc[-1]
        result = bot.brooks_analyze(frame)
        self.assertTrue(result["evidence"]["quality"]["blocked"])
        self.assertEqual(result["setups"], [])
        self.assertTrue(all(name in ("BROOKS_市场状态", "BROOKS_AlwaysIn") for name, _ in result["votes"]))

    def test_latest_zero_volume_blocks_even_with_price_movement(self):
        frame = bars()
        frame.loc[frame.index[-1], "volume"] = 0
        self.assertTrue(quality_check(frame)["blocked"])

    def test_missing_volume_is_explicit(self):
        self.assertIn("成交量缺失", quality_check(bars().drop(columns="volume"))["message"])

    def test_missing_bar_blocks_continuous_pattern_confirmation(self):
        quality = quality_check(bars().drop(bars().index[-5]))
        self.assertTrue(quality["blocked"])
        self.assertEqual(quality["gaps"], 1)

    def test_invalid_prices_do_not_produce_swings(self):
        frame = bars()
        frame.iloc[-1, frame.columns.get_loc("high")] = 1
        result = bot.brooks_analyze(frame)
        self.assertEqual(result["evidence"]["quality"]["status"], "invalid")
        self.assertEqual(result["votes"], [])

    def test_future_cannot_confirm_current_swing(self):
        frame = bars()
        before = inspect_evidence(frame.iloc[:120])
        frame.iloc[120:, :4] *= 2
        self.assertEqual(before, inspect_evidence(frame.iloc[:120]))
        self.assertTrue(all(pd.Timestamp(p["confirmed_at"]) <= frame.index[119] for p in before["swings"]))

    def test_brief_has_quality_and_structure_not_probability(self):
        brief = evidence_brief(inspect_evidence(bars()))
        self.assertIn("数据质量", brief)
        self.assertIn("已确认摆动", brief)
        self.assertIn("不是经回测", brief)


if __name__ == "__main__":
    unittest.main()
