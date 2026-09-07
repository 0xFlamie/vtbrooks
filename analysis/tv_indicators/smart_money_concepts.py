"""Smart Money Concepts [LuxAlgo] + Market Structure Break & Order Blocks [EmreKb] 有效性回测
指标逻辑(忠于 TradingView 公开原版):
  - 摆动结构: ta.pivothigh/pivotlow(SWING_LEN, SWING_LEN), 收线确认(滞后 SWING_LEN 根, 无未来函数)
    LuxAlgo SMC 默认 swing=50 / internal=5; EmreKb MSB&OB 默认 swing=5 —— 两档都测
  - BOS = 顺趋势突破最近未破摆动极值; CHoCH = 逆趋势突破(含首次突破)
  - 订单块: 突破前最后一根反向K线 [low, high] 区间 (LuxAlgo/EmreKb 同款定义)
    回踩=价格进入区间; 失效=收线击穿区间另一端 (LuxAlgo mitigation=close 默认)
  - FVG: 三根K线缺口, 看涨 low[i] > high[i-2], 看跌 high[i] < low[i-2]
运行: python3 analysis/tv_indicators/smart_money_concepts.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis/tv_indicators")
from smc_common import HORIZONS, MIN_N, confirmed_pivots, fwd_ret, load, signal_table

SWING_LENS = [5, 50]   # internal / swing (LuxAlgo 默认 5 和 50)
OB_LOOKBACK = 100      # 订单块向前扫描上限(根)
OB_MAX_WAIT = 720      # 订单块等待回踩上限(根), 超时弃样
FVG_FILL_WINDOWS = {"24h": None, "5d": None}  # 按周期填充


def structure_events(df, L):
    """因果版 BOS/CHoCH: 逐 bar 推进, pivot 收线确认后才可用, 突破判定用当根 close。"""
    ph, pl = confirmed_pivots(df, L)
    c = df["close"].values
    sh = sl = None  # 最近未破摆动高/低
    trend = 0
    events = []  # (bar, kind, dir)
    for j in range(len(df)):
        if not np.isnan(ph[j]):
            sh = ph[j]
        if not np.isnan(pl[j]):
            sl = pl[j]
        if sh is not None and c[j] > sh:
            events.append((j, "BOS" if trend == 1 else "CHoCH", 1))
            trend, sh = 1, None
        elif sl is not None and c[j] < sl:
            events.append((j, "BOS" if trend == -1 else "CHoCH", -1))
            trend, sl = -1, None
    return events


def order_blocks(df, events):
    """每个突破事件对应订单块: 突破 bar 往前最后一根反向K线的 [low, high]。
    返回 (形成bar, 方向, 下沿, 上沿)"""
    o = df["open"].values
    c = df["close"].values
    lo = df["low"].values
    hi = df["high"].values
    obs = []
    prev = 0
    for j, _kind, d in events:
        start = max(prev, j - OB_LOOKBACK)
        for i in range(j - 1, start - 1, -1):
            if (d == 1 and c[i] < o[i]) or (d == -1 and c[i] > o[i]):
                obs.append((j, d, lo[i], hi[i]))
                break
        prev = j
    return obs


def ob_reactions(df, obs):
    """回踩反应: 形成后首根进入区间的 bar 为信号; 先收穿另一端=失效不计。
    返回 (信号bar列表, 方向列表, 失效数, 超时数)"""
    lo = df["low"].values
    hi = df["high"].values
    c = df["close"].values
    idx, dirs = [], []
    n_mitigated = n_expired = 0
    for j, d, zlo, zhi in obs:
        end = min(len(df), j + 1 + OB_MAX_WAIT)
        outcome = None
        for t in range(j + 1, end):
            if (d == 1 and c[t] < zlo) or (d == -1 and c[t] > zhi):
                outcome = "mitigated"
                break
            if (d == 1 and lo[t] <= zhi) or (d == -1 and hi[t] >= zlo):
                outcome = "touch"
                break
        if outcome == "touch":
            idx.append(t)
            dirs.append(d)
        elif outcome == "mitigated":
            n_mitigated += 1
        else:
            n_expired += 1
    return idx, dirs, n_mitigated, n_expired


def fvgs(df):
    """FVG: 信号 bar = 缺口第三根 (收线确认)。返回 (bars, dirs, 下沿, 上沿)"""
    h = df["high"].values
    l = df["low"].values
    idx, dirs, zlo, zhi = [], [], [], []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:  # 看涨缺口 [h[i-2], l[i]]
            idx.append(i); dirs.append(1); zlo.append(h[i - 2]); zhi.append(l[i])
        elif h[i] < l[i - 2]:  # 看跌缺口 [l[i-2], h[i]]
            idx.append(i); dirs.append(-1); zlo.append(l[i - 2]); zhi.append(h[i])
    return idx, dirs, zlo, zhi


def fvg_fill_stats(df, idx, dirs, zlo, zhi, tf):
    """填充率: 完全回补(看涨: low <= 缺口下沿; 看跌: high >= 缺口上沿)在 24h / 5d 内的比例"""
    l = df["low"].values
    h = df["high"].values
    bars_24h = 6 if tf == "4h" else 24
    bars_5d = 30 if tf == "4h" else 120
    res = {}
    for name, w in [("24h", bars_24h), ("5d", bars_5d)]:
        filled = 0
        tot = 0
        for i, d, a, b in zip(idx, dirs, zlo, zhi):
            end = min(len(df), i + 1 + w)
            if end <= i + 1:
                continue
            tot += 1
            seg_l = l[i + 1:end]
            seg_h = h[i + 1:end]
            if (d == 1 and (seg_l <= a).any()) or (d == -1 and (seg_h >= b).any()):
                filled += 1
        res[name] = (filled, tot)
    return res


def run_tf(tf):
    df = load(tf)
    print(f"\n{'='*78}\n周期 {tf}: {len(df)} 根 ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})")
    # 全样本无条件基线(各 horizon)
    base_info = []
    for hz, k in HORIZONS[tf].items():
        r = fwd_ret(df, k).dropna()
        base_info.append(f"{hz}: P(涨)={100*(r > 0).mean():.1f}% 均{r.mean()*100:+.3f}%")
    print("基线(下一根open入场): " + " | ".join(base_info))

    for L in SWING_LENS:
        ev = structure_events(df, L)
        for kind in ["BOS", "CHoCH"]:
            for d, dn in [(1, "多"), (-1, "空")]:
                sel = [(j, dd) for j, k_, dd in ev if k_ == kind and dd == d]
                if sel:
                    signal_table(df, tf, [s[0] for s in sel], [s[1] for s in sel],
                                 f"[swing={L}] {kind} 做{dn}")
        obs = order_blocks(df, ev)
        idx, dirs, n_mit, n_exp = ob_reactions(df, obs)
        print(f"\n  [swing={L}] 订单块: 形成 {len(obs)} | 回踩成信号 {len(idx)} | "
              f"未回踩先失效 {n_mit} | 超时 {n_exp}")
        for d, dn in [(1, "看涨OB回踩做多"), (-1, "看跌OB回踩做空")]:
            sel = [(i, dd) for i, dd in zip(idx, dirs) if dd == d]
            if sel:
                signal_table(df, tf, [s[0] for s in sel], [s[1] for s in sel], f"[swing={L}] {dn}")

    idx, dirs, zlo, zhi = fvgs(df)
    sizes = [(b - a) / df["close"].values[i] * 100 for i, a, b in zip(idx, zlo, zhi)]
    print(f"\n  FVG: 共 {len(idx)} 个 | 缺口中位大小 {np.median(sizes):.3f}%")
    fill = fvg_fill_stats(df, idx, dirs, zlo, zhi, tf)
    for name, (f, tot) in fill.items():
        print(f"  FVG {name} 内完全填充率: {f}/{tot} = {100*f/max(tot,1):.1f}%")
    for d, dn in [(1, "看涨FVG后做多"), (-1, "看跌FVG后做空")]:
        sel = [(i, dd) for i, dd in zip(idx, dirs) if dd == d]
        if sel:
            signal_table(df, tf, [s[0] for s in sel], [s[1] for s in sel], dn)


def main():
    for tf in ["4h", "1h"]:
        run_tf(tf)


if __name__ == "__main__":
    main()
