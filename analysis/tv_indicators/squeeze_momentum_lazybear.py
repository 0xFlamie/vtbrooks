"""Squeeze Momentum Indicator [LazyBear] 原版逻辑回测。

原版参数（TradingView LazyBear 公开版）:
  BB length=20, mult=2.0; KC length=20, mult=1.5; useTrueRange=True
  动量值 val = linreg(close - avg((highest(high,20)+lowest(low,20))/2, sma(close,20)), 20, 0)
  sqzOn  = lowerBB > lowerKC and upperBB < upperKC
  sqzOff = lowerBB < lowerKC and upperBB > upperKC
  柱色: val>0 且递增=亮绿, val>0 且递减=暗绿, val<0 且递减=亮红, val<0 且递增=暗红
注: Pine stdev 为总体标准差(biased), 这里 ddof=0。

信号（收线确认，下一根 open 入场）:
  亮绿做多 / 亮红做空，入场时机 = 挤压释放第一根(sqzOn -> sqzOff)；
  另测: 释放点只看动量符号(不验递增/递减)、以及纯动量柱色状态(不限释放点)作参照。

运行: python3 analysis/tv_indicators/squeeze_momentum_lazybear.py
"""
import numpy as np
import pandas as pd

from stats_common import load, evaluate, print_report

BB_LEN, BB_MULT = 20, 2.0
KC_LEN, KC_MULT = 20, 1.5
MOM_LEN = 20


def linreg_last(y: np.ndarray, length: int) -> np.ndarray:
    """滚动线性回归在最后一点的拟合值（Pine linreg(y, length, 0)）。"""
    sw = np.lib.stride_tricks.sliding_window_view(y, length)
    x = np.arange(length)
    xm = x.mean()
    var = ((x - xm) ** 2).mean()
    slope = ((sw - sw.mean(1, keepdims=True)) * (x - xm)).sum(1) / length / var
    return slope * (length - 1 - xm) + sw.mean(1)


def squeeze_momentum(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    basis = close.rolling(BB_LEN).mean()
    dev = close.rolling(BB_LEN).std(ddof=0) * BB_MULT
    upper_bb, lower_bb = basis + dev, basis - dev
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    rangema = tr.rolling(KC_LEN).mean()
    upper_kc, lower_kc = basis + KC_MULT * rangema, basis - KC_MULT * rangema
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    highest = high.rolling(MOM_LEN).max()
    lowest = low.rolling(MOM_LEN).min()
    mid = ((highest + lowest) / 2 + close.rolling(MOM_LEN).mean()) / 2
    src = close - mid
    val = pd.Series(np.nan, index=df.index)
    lv = linreg_last(src.fillna(0).to_numpy(), MOM_LEN)
    val.iloc[MOM_LEN - 1:] = lv
    val[src.isna()] = np.nan
    return pd.DataFrame({"val": val, "sqz_on": sqz_on, "sqz_off": sqz_off})


def main() -> None:
    rows = []
    for tf in ("4h", "1h"):
        df = load(tf)
        sm = squeeze_momentum(df)
        val, prev_val = sm["val"], sm["val"].shift(1)
        bright_green = (val > 0) & (val > prev_val)   # 亮绿
        bright_red = (val < 0) & (val < prev_val)     # 亮红
        release = sm["sqz_off"] & sm["sqz_on"].shift(1).fillna(False)  # 挤压释放第一根
        signals = [
            ("sqz_release_bright_green", release & bright_green, +1),
            ("sqz_release_bright_red", release & bright_red, -1),
            ("sqz_release_val>0", release & (val > 0), +1),
            ("sqz_release_val<0", release & (val < 0), -1),
            ("bright_green_any_bar", bright_green, +1),
            ("bright_red_any_bar", bright_red, -1),
        ]
        for name, mask, direction in signals:
            evaluate(df, name, mask, direction, tf, rows)
    print_report("Squeeze Momentum [LazyBear] BB20/2.0 KC20/1.5 linreg20", rows)


if __name__ == "__main__":
    main()
