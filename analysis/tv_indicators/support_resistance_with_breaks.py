"""Support/Resistance with Breaks [LuxAlgo] 有效性回测: 关键位放量突破后的延续概率
指标逻辑(忠于 TradingView 公开原版):
  - 关键位 = pivot high/low (LEFT=RIGHT=15, ta.pivothigh 语义), 收线确认(滞后15根, 无未来函数),
    水平位保持到新 pivot 确认
  - 量能确认: volume oscillator = 100*(EMA(vol,5)-EMA(vol,10))/EMA(vol,10) > VOL_THRESHOLD(20)
  - 突破信号: close 上穿阻力 / 下穿支撑 (ta.crossover 语义, 沿触发一次) 且量振达标
运行: python3 analysis/tv_indicators/support_resistance_with_breaks.py
"""
import sys

import numpy as np

sys.path.insert(0, "analysis/tv_indicators")
from smc_common import HORIZONS, confirmed_pivots, fwd_ret, load, signal_table

LEFT = 15            # 原版默认 left bars
RIGHT = 15           # 原版默认 right bars
VOL_FAST = 5         # 量振快 EMA
VOL_SLOW = 10        # 量振慢 EMA
VOL_THRESHOLD = 20   # 原版默认量振阈值


def sr_breaks(df):
    ph, pl = confirmed_pivots(df, RIGHT)
    c = df["close"].values
    v = df["vol"].values.astype(float)
    ef = pd_ema(v, VOL_FAST)
    es = pd_ema(v, VOL_SLOW)
    vosc = 100 * (ef - es) / es
    res = sup = None
    sig_idx, sig_dir = [], []
    for j in range(len(df)):
        if not np.isnan(ph[j]):
            res = ph[j]
        if not np.isnan(pl[j]):
            sup = pl[j]
        if j == 0:
            continue
        if res is not None and c[j - 1] <= res and c[j] > res and vosc[j] > VOL_THRESHOLD:
            sig_idx.append(j); sig_dir.append(1)
            res = None  # 突破后该位失效, 等新 pivot
        elif sup is not None and c[j - 1] >= sup and c[j] < sup and vosc[j] > VOL_THRESHOLD:
            sig_idx.append(j); sig_dir.append(-1)
            sup = None
    return sig_idx, sig_dir


def pd_ema(x, span):
    alpha = 2 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def run_tf(tf):
    df = load(tf)
    print(f"\n{'='*78}\n周期 {tf}: {len(df)} 根 ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})")
    base_info = []
    for hz, k in HORIZONS[tf].items():
        r = fwd_ret(df, k).dropna()
        base_info.append(f"{hz}: P(涨)={100*(r > 0).mean():.1f}% 均{r.mean()*100:+.3f}%")
    print("基线(下一根open入场): " + " | ".join(base_info))
    idx, dirs = sr_breaks(df)
    print(f"  放量突破信号: {len(idx)} (多 {sum(1 for d in dirs if d==1)} / 空 {sum(1 for d in dirs if d==-1)})")
    for d, dn in [(1, "放量破阻力做多"), (-1, "放量破支撑做空")]:
        sel = [(i, dd) for i, dd in zip(idx, dirs) if dd == d]
        if sel:
            signal_table(df, tf, [s[0] for s in sel], [s[1] for s in sel], dn)


def main():
    for tf in ["4h", "1h"]:
        run_tf(tf)


if __name__ == "__main__":
    main()
