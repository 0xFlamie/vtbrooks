"""交易员观点段 + 每日复盘信单测(2026-08-20 新增):
- position_size 仓位映射: 幅度档×共振×置信修正, 封顶50%
- build_analyst_block: 观点/计划/失效/上次观点四行, 观望/背离/无窗口分支
- daily_review: 近24h战绩行, 样本<3不打扰
运行: python3 tests/test_analyst.py
"""
import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vt_vote_bot as V

RESULT = {"symbol": "ETHUSDC", "price": 1900.0, "signal": "LONG", "bullish": 5, "bearish": 3, "brooks": {}}
LV = {"swing_high": 1950.0, "swing_low": 1850.0, "ema20": 1890.0, "ema50": 1880.0, "atr14": 30.0}
EW = {"zone": "1890-1900", "type": "回调", "invalid": 1885.0, "dist": 0.79, "dir": "LONG", "basis": "贴近摆动低"}
PLAN = {"entry": 1900.0, "sl": 1888.0, "tp1": 1912.0, "tp2": 1945.0, "rr": 2.6, "sl_label": "ATR止损(按近期波动幅度)"}


class TestPositionSize(unittest.TestCase):
    def test_tier_basis(self):
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75), 40)   # 35+5共振
        self.assertEqual(V.position_size("LONG", "LONG", 1, 60), 20)   # 15+5共振
        self.assertEqual(V.position_size("LONG", "LONG", 0, 60), 10)   # 5+5共振

    def test_single_layer_and_confidence(self):
        self.assertEqual(V.position_size("LONG", None, 3, 75), 35)     # 单层无共振
        self.assertEqual(V.position_size("LONG", "LONG", 3, 85), 45)   # 高置信+5
        self.assertEqual(V.position_size("LONG", "LONG", 3, 55), 35)   # 低置信-5
        self.assertEqual(V.position_size(None, None, 3, 75), 0)        # 无方向0

    def test_cap_50(self):
        self.assertEqual(V.position_size("LONG", "LONG", 3, 90), 45)   # 35+5+5=45
        self.assertLessEqual(V.position_size("LONG", "LONG", 3, 100), 50)

    def test_atr_correction(self):
        # 基准ATR 3%: 高波动减仓, 低波动加仓
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=6.0), 20)   # 40×3/6
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=3.0), 40)   # 40×1
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=1.5), 50)   # 40×2→cap

    def test_squeeze_and_event(self):
        # 挤压≥70(波动已释放)减仓; <20可加仓(2026-08-22 回测)
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=3.0, squeeze_pct=85), 28)
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=3.0, squeeze_pct=10), 44)
        # 事件日七折
        self.assertEqual(V.position_size("LONG", "LONG", 3, 75, atr_pct=3.0, event_day=True), 28)


class TestAnalystBlock(unittest.TestCase):
    def test_full_view(self):
        j4 = {"direction": "LONG", "confidence": 80, "mag_tier": 3, "reasons": ["费率极端+放量突破", "x"]}
        j15 = {"direction": "LONG", "confidence": 75}
        L = V.build_analyst_block(RESULT, j4, j15, EW, LV, PLAN)
        self.assertTrue(any(x.startswith("🎯 观点: 做多(80%)") for x in L))
        # ATR修正: atr14=30/px1900=1.58% → 45×3/1.58 → cap 50
        self.assertTrue(any(x.startswith("📌 计划: 仓位50%") for x in L))
        self.assertTrue(any("止损$1888.00(-0.6%)" in x for x in L))
        self.assertTrue(any("RR 1:2.6" in x for x in L))
        self.assertTrue(any("风险0.3%" in x for x in L))
        self.assertTrue(any("🚫 失效: 收破$1885.00" in x for x in L))

    def test_event_day(self):
        j4 = {"direction": "LONG", "confidence": 80, "mag_tier": 3, "reasons": ["x"]}
        j15 = {"direction": "LONG", "confidence": 75}
        L = V.build_analyst_block(RESULT, j4, j15, EW, LV, PLAN, event_day=True)
        self.assertTrue(any("今日宏观事件日" in x for x in L))

    def test_watch_no_plan(self):
        j4 = {"direction": None, "confidence": 40, "mag_tier": None, "reasons": []}
        j15 = {"direction": None, "confidence": 45}
        L = V.build_analyst_block(RESULT, j4, j15, None, LV, None)
        self.assertTrue(any("🎯 观点: 观望" in x for x in L))
        self.assertFalse(any(x.startswith("📋 计划") for x in L))
        self.assertFalse(any(x.startswith("🚫") for x in L))

    def test_divergence(self):
        j4 = {"direction": "LONG", "confidence": 70, "mag_tier": 2, "reasons": []}
        j15 = {"direction": "SHORT", "confidence": 70}
        L = V.build_analyst_block(RESULT, j4, j15, None, LV, None)
        self.assertTrue(any("背离" in x for x in L))

    def test_no_window_fallback_sl(self):
        j4 = {"direction": "SHORT", "confidence": 70, "mag_tier": 2, "reasons": []}
        j15 = {"direction": "SHORT", "confidence": 70}
        plan = {"entry": 1900.0, "sl": 1912.0, "tp1": 1888.0, "tp2": 1860.0, "rr": 2.0}
        L = V.build_analyst_block(RESULT, j4, j15, None, LV, plan)
        self.assertTrue(any("入场等回调贴结构位" in x for x in L))
        self.assertTrue(any("🚫 失效: 4h收破$1950.00" in x for x in L))  # 空头看摆动高

    def test_prev_continuity(self):
        j4 = {"direction": "LONG", "confidence": 80, "mag_tier": 3, "reasons": ["费率极端值+放量突破摆动高", "x"]}
        j15 = {"direction": "LONG", "confidence": 75}
        L = V.build_analyst_block(RESULT, j4, j15, EW, LV, PLAN, prev4={"direction": "SHORT", "confidence": 70})
        self.assertTrue(any("🔁 上次判空 → 翻多" in x for x in L))
        L2 = V.build_analyst_block(RESULT, j4, j15, EW, LV, PLAN, prev4={"direction": "LONG", "confidence": 75})
        self.assertTrue(any("🔁 上次: 维持" in x for x in L2))


class TestDailyReview(unittest.TestCase):
    def _entry(self, verdict, direction, judgment=None, tier_correct=None, watch=None, ts_days_ago=0.5, score=None):
        et = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=ts_days_ago * 24)
        return {"time": et.isoformat(), "symbol": "ETHUSDC", "direction": direction, "verdict": verdict,
                "entry_px": 1900.0, "sl": 1888.0, "tp2": 1945.0, "atr": 30.0, "outcome": "loss" if judgment is False else "win",
                "judgment": judgment, "dir4h": direction, "dir15m": direction, "tier_correct": tier_correct,
                "mag_tier": 2, "ai_direction": direction, "watch_quality": watch,
                "dir_score_4h": score, "dir_score_2h": None, "mfe_long_12h": None,
                "mfe_short_12h": None, "mfe_against_12h": None}

    def test_review_lines(self):
        entries = [self._entry("执行", "LONG", score=0.8),
                   self._entry("执行", "SHORT", score=0.4),
                   self._entry("观望", None, watch=0.2)]
        with mock.patch.object(V, "load_journal", return_value={"entries": entries}):
            txt = V.daily_review()
        self.assertIn("推单 3 (执行2/观望1)", txt)
        self.assertIn("方向分: 平均+0.60 ATR", txt)
        self.assertIn("观望质量: 平均0.2 ATR", txt)
        self.assertNotIn("判对率", txt)
        self.assertNotIn("错题", txt)

    def test_too_few_samples(self):
        entries = [self._entry("执行", "LONG", judgment=True)]
        with mock.patch.object(V, "load_journal", return_value={"entries": entries}):
            self.assertIsNone(V.daily_review())

    def test_old_entries_excluded(self):
        entries = [self._entry("执行", "LONG", judgment=True, ts_days_ago=3)]
        with mock.patch.object(V, "load_journal", return_value={"entries": entries}):
            self.assertIsNone(V.daily_review())


if __name__ == "__main__":
    unittest.main()
