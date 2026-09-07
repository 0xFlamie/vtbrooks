"""CM_Williams_Vix_Fix (ChrisMoody, TradingView 原版逻辑) 底部信号有效性回测。

原版公式（CM_Williams_Vix_Fix Finds Market Bottoms [ChrisMoody]）：
    wvf = (highest(close, PD) - low) / highest(close, PD) * 100
    midLine   = sma(wvf, BBL)
    upperBand = midLine + MULT * stdev(wvf, BBL)
    rangeHigh = highest(wvf, LB) * PH
    底部信号(绿柱) up = wvf >= upperBand or wvf >= rangeHigh

测法：绿柱收线确认，下一根 open 入场做多，看未来 4h/12h/24h。
绿柱常成簇出现，同时给出去簇（簇内首根）统计，样本更独立。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import load, fwd_returns, evaluate, fmt

# TradingView 原版默认参数
PD = 22     # LookBack Period Standard Deviation High
BBL = 20    # Bollinger Band Length
MULT = 2.0  # Bollinger Band Standard Deviation Up
LB = 50     # Look Back Period Percentile High
PH = 0.85   # Highest Percentile

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def vix_fix(df):
    highest_close = df["close"].rolling(PD).max()
    wvf = (highest_close - df["low"]) / highest_close * 100.0
    mid = wvf.rolling(BBL).mean()
    upper = mid + MULT * wvf.rolling(BBL).std(ddof=0)
    range_high = wvf.rolling(LB).max() * PH
    up = ((wvf >= upper) | (wvf >= range_high)).fillna(False).astype(bool)
    return up


def run(csv, horizons, hlabels):
    df = load(os.path.join(DATA, csv))
    up = vix_fix(df)
    first = up & ~up.shift(1, fill_value=False).astype(bool)  # 簇内首根
    fwd = fwd_returns(df, horizons)
    res_all = evaluate("VF底部绿柱(全部)", up, fwd, df, horizons, side=1)
    res_first = evaluate("VF底部绿柱(去簇)", first, fwd, df, horizons, side=1)
    print(f"\n=== {csv} ({len(df)} bars, {df.index[0]} ~ {df.index[-1]}) ===")
    print(fmt(res_all, hlabels))
    print(fmt(res_first, hlabels))
    return pd.concat([res_all, res_first]).assign(tf=csv)


if __name__ == "__main__":
    r4 = run("okx_4h.csv", [1, 3, 6], {1: "4h", 3: "12h", 6: "24h"})
    r1 = run("okx_1h.csv", [4, 12, 24], {4: "4h", 12: "12h", 24: "24h"})
