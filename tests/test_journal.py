"""verify_journal 结算逻辑单测(2026-08-20 新增):
- 观望判例: 不评方向/计划/判对, 只补 watch_quality
- 执行判例幅度档: 方向侧 MFE ≥ 档位下限 且 > 反方向侧 MFE (过滤震荡市假兑现)
mock fetch_klines, 不碰网络。运行: python3 -m unittest tests.test_journal -v
"""
import json
import os
import sys
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vt_vote_bot as V

ET = pd.Timestamp("2020-01-01T00:00:00", tz="UTC")  # 远早于现在, age>12h 满足结算条件
N = 60  # 15h 的 15m bars
IDX = pd.date_range(ET, periods=N, freq="15min")


def make_df(highs, lows, closes):
    """构造 15m DataFrame: 给定 high/low/close 序列(长度 N, 缺省补齐为最后值)"""
    highs = list(highs) + [highs[-1]] * (N - len(highs))
    lows = list(lows) + [lows[-1]] * (N - len(lows))
    closes = list(closes) + [closes[-1]] * (N - len(closes))
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": 1.0}, index=IDX)


def mock_klines(df):
    return mock.patch.object(V, "fetch_klines",
                             lambda sym, interval, limit, start_ms=None, drop_incomplete=True: df)


def base_entry(**kw):
    e = {"time": ET.isoformat(), "symbol": "ETHUSDC", "direction": "LONG", "verdict": "执行",
         "entry_px": 100.0, "sl": 98.0, "tp2": 110.0, "atr": 1.0,
         "dir4h": "LONG", "conf4h": 70, "dir15m": "LONG", "conf15m": 65, "mag_tier": 1,
         "ai_direction": "LONG", "outcome": None, "judgment": None,
         "dir_score_2h": None, "dir_score_4h": None,
         "mfe_long_12h": None, "mfe_short_12h": None, "mfe_against_12h": None,
         "watch_quality": None, "tier_correct": None}
    e.update(kw)
    return e


def run_verify(entry, df):
    with mock_klines(df):
        with mock.patch.object(V, "load_journal", return_value={"entries": [entry]}):
            with mock.patch.object(V, "save_journal", lambda j: None):
                V.verify_journal()
    return entry


class TestWatchQuality(unittest.TestCase):
    def test_watch_keeps_outside_verdict_fields(self):
        """观望判例: 补 watch_quality, 不碰方向分/计划/判对"""
        e = base_entry(direction=None, verdict="观望", sl=None, tp2=None,
                       dir4h=None, dir15m=None, ai_direction=None, mag_tier=None)
        # 4h 后 close 偏离 1.2×ATR → 观望错过行情, quality 应大
        closes = [100.0] * 16 + [101.2] * 44
        highs = [x + 0.3 for x in closes]
        lows = [x - 0.3 for x in closes]
        run_verify(e, make_df(highs, lows, closes))
        self.assertAlmostEqual(e["watch_quality"], 1.2, places=2)
        self.assertIsNone(e["judgment"])
        self.assertIsNone(e["outcome"])
        self.assertIsNone(e["dir_score_4h"])

    def test_watch_zero_when_flat(self):
        """横盘观望: quality≈0, 观望正确"""
        e = base_entry(direction=None, verdict="观望", sl=None, tp2=None,
                       dir4h=None, dir15m=None, ai_direction=None, mag_tier=None)
        closes = [100.0] * N
        run_verify(e, make_df([100.3] * N, [99.7] * N, closes))
        self.assertLessEqual(e["watch_quality"], 0.1)


class TestTierCorrect(unittest.TestCase):
    def test_trend_passes(self):
        """单边上涨: 方向侧 MFE 2.5% ≥ 档位下限 1%, 且 > 反向 0.5% → 兑现"""
        e = base_entry()
        highs = [100.0 + i * 0.05 for i in range(N)]  # 12h 窗口内最高 102.35
        lows = [x - 0.5 for x in highs]               # 反向 MFE ≈ 0.5%
        closes = highs
        run_verify(e, make_df(highs, lows, closes))
        self.assertGreaterEqual(e["mfe_long_12h"], 2.3)
        self.assertAlmostEqual(e["mfe_against_12h"], 0.5, places=1)
        self.assertTrue(e["tier_correct"])

    def test_choppy_fails(self):
        """震荡市: 方向侧 MFE 1.5% 达标但反向 2% 更大 → 假兑现被过滤"""
        e = base_entry()
        closes = [100.0] * 12 + [101.5] * 12 + [99.0] * 12 + [100.0] * 24
        highs = [101.5] * N  # 方向侧 MFE 1.5% ≥ 档位下限
        lows = [98.0] * N    # 反向 MFE 2.0% 更大 → 震荡假兑现
        run_verify(e, make_df(highs, lows, closes))
        self.assertGreaterEqual(e["mfe_long_12h"], 1.4)
        self.assertGreaterEqual(e["mfe_against_12h"], 1.9)
        self.assertFalse(e["tier_correct"])

    def test_undershoot_fails(self):
        """幅度不足: 方向侧 MFE 0.6% < 档位下限 1% → 不兑现"""
        e = base_entry()
        run_verify(e, make_df([100.6] * N, [99.7] * N, [100.0] * N))
        self.assertLess(e["mfe_long_12h"], 1.0)
        self.assertFalse(e["tier_correct"])


class TestExecSettlement(unittest.TestCase):
    def test_dir_score_and_judgment(self):
        """执行判例: 方向分 + 判对照常结算"""
        e = base_entry()
        closes = [100.0] * 16 + [101.5] * 44  # 4h 后 +1.5×ATR
        highs = [x + 0.3 for x in closes]
        lows = [x - 0.3 for x in closes]
        run_verify(e, make_df(highs, lows, closes))
        self.assertGreaterEqual(e["dir_score_4h"], 1.5)
        self.assertTrue(e["judgment"])


if __name__ == "__main__":
    unittest.main()
