"""WaveTrend Oscillator (LazyBear, TradingView 原版逻辑) 有效性回测。

原版公式（WaveTrend Oscillator [LazyBear]）：
    ap  = hlc3
    esa = ema(ap, CHLEN)
    d   = ema(|ap - esa|, CHLEN)
    ci  = (ap - esa) / (0.015 * d)
    tci = ema(ci, AVG_LEN)
    wt1 = tci ; wt2 = sma(wt1, MA_LEN)

信号（收线确认，下一根 open 入场）：
    做多 = wt1 上穿 wt2 且交叉时 wt1 < OS（超卖区金叉）
    做空 = wt1 下穿 wt2 且交叉时 wt1 > OB（超买区死叉）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import load, fwd_returns, evaluate, fmt

# TradingView 原版默认参数
CHLEN = 10      # channel length
AVG_LEN = 21    # average length
MA_LEN = 4      # signal MA length
OB = 60.0       # 超买
OS = -60.0      # 超卖

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def wavetrend(df):
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=CHLEN, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=CHLEN, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=AVG_LEN, adjust=False).mean()
    wt2 = wt1.rolling(MA_LEN).mean()
    return wt1, wt2


def run(csv, horizons, hlabels):
    df = load(os.path.join(DATA, csv))
    wt1, wt2 = wavetrend(df)
    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_dn = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    long_sig = (cross_up & (wt1 < OS)).fillna(False)
    short_sig = (cross_dn & (wt1 > OB)).fillna(False)
    fwd = fwd_returns(df, horizons)
    res_l = evaluate("WT超卖金叉多", long_sig, fwd, df, horizons, side=1)
    res_s = evaluate("WT超买死叉空", short_sig, fwd, df, horizons, side=-1)
    print(f"\n=== {csv} ({len(df)} bars, {df.index[0]} ~ {df.index[-1]}) ===")
    print(fmt(res_l, hlabels))
    print(fmt(res_s, hlabels))
    return pd.concat([res_l, res_s]).assign(tf=csv)


if __name__ == "__main__":
    r4 = run("okx_4h.csv", [1, 3, 6], {1: "4h", 3: "12h", 6: "24h"})
    r1 = run("okx_1h.csv", [4, 12, 24], {4: "4h", 12: "12h", 24: "24h"})
