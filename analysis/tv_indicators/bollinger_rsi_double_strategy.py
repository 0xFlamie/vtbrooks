"""Bollinger + RSI, Double Strategy (by ChartArt) v1.1 原版逻辑回测。

原版参数（TradingView ChartArt v1.1 公开源码）:
  RSI length=6, overSold=50, overBought=50（注意：阈值就是 50，不是 30/70）
  BB length=200, mult=2（close, 总体标准差）
  原始策略入场（同一根 K 线两个穿越同时发生才算）:
    long:  crossover(RSI6, 50)  AND crossover(close, BBlower)
    short: crossunder(RSI6, 50) AND crossunder(close, BBupper)
    （原策略用 stop 单挂在 BB 轨，这里按项目标准改为收线确认 + 下一根 open 入场）
  附带测 TrendColor 变体（原版 barcolor 条件，多了 BB basis 斜率过滤）:
    long:  close 上穿 BBlower 且 BBbasis 上行
    short: close 下穿 BBupper 且 BBbasis 下行

运行: python3 analysis/tv_indicators/bollinger_rsi_double_strategy.py
"""
import numpy as np
import pandas as pd

from stats_common import load, evaluate, print_report

RSI_LEN = 6
RSI_OS, RSI_OB = 50.0, 50.0
BB_LEN, BB_MULT = 200, 2.0


def rma(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.rma / RSI 用的 Wilder 平滑。"""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def crossover(a: pd.Series, level) -> pd.Series:
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=a.index)
    return (a > lvl) & (a.shift(1) <= lvl.shift(1))


def crossunder(a: pd.Series, level) -> pd.Series:
    lvl = level if isinstance(level, pd.Series) else pd.Series(level, index=a.index)
    return (a < lvl) & (a.shift(1) >= lvl.shift(1))


def bb_rsi(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    delta = close.diff()
    gain = rma(delta.clip(lower=0), RSI_LEN)
    loss = rma((-delta).clip(lower=0), RSI_LEN)
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    basis = close.rolling(BB_LEN).mean()
    dev = close.rolling(BB_LEN).std(ddof=0) * BB_MULT
    return pd.DataFrame({"rsi": rsi, "basis": basis,
                         "upper": basis + dev, "lower": basis - dev})


def main() -> None:
    rows = []
    for tf in ("4h", "1h"):
        df = load(tf)
        ind = bb_rsi(df)
        close = df["close"]
        x_up_lower = crossover(close, ind["lower"])
        x_dn_upper = crossunder(close, ind["upper"])
        raw_long = crossover(ind["rsi"], RSI_OS) & x_up_lower
        raw_short = crossunder(ind["rsi"], RSI_OB) & x_dn_upper
        color_long = x_up_lower & (ind["basis"] > ind["basis"].shift(1))
        color_short = x_dn_upper & (ind["basis"] < ind["basis"].shift(1))
        signals = [
            ("raw_rsi50_x_bblower", raw_long, +1),
            ("raw_rsi50_x_bbupper", raw_short, -1),
            ("trendcolor_bblower_up", color_long, +1),
            ("trendcolor_bbupper_dn", color_short, -1),
        ]
        for name, mask, direction in signals:
            evaluate(df, name, mask, direction, tf, rows)
    print_report("Bollinger+RSI Double Strategy [ChartArt v1.1] RSI6/50 BB200/2", rows)


if __name__ == "__main__":
    main()
