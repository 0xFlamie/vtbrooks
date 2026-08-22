"""第三轮深挖(5年 OKX 4h): EMA趋势排列/跨周期动量/量价背离/区间位置/距摆动极值
生产系统核心信号(EMA7/25/99 趋势排列)从未被独立验证——这一轮补上
运行: python3 analysis/third_explore.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from feature_explore import fetch_okx_4h


def pct_up(d, m, label):
    if m.sum() < 100:
        print(f"  {label}: n={int(m.sum()):4d} 样本不足")
        return
    up = d["fwd_up"][m].mean() * 100
    avg = d["fwd_ret"][m].mean()
    print(f"  {label}: n={int(m.sum()):4d} 未来4h涨 {up:.1f}% | 平均 {avg:+.3f}%")


def main():
    print("拉取 OKX 4h 5年...", flush=True)
    h4 = fetch_okx_4h(days=1825)
    h4["ret"] = h4["close"].pct_change() * 100
    h4["fwd_up"] = (h4["close"].pct_change(-1) > 0).astype(int)
    h4["fwd_ret"] = h4["close"].pct_change(-1) * 100
    h4["fwd_up_12h"] = (h4["close"].shift(-3) > h4["close"]).astype(int)
    print(f"基线: 未来4h涨 {h4['fwd_up'].mean()*100:.1f}% | 未来12h涨 {h4['fwd_up_12h'].mean()*100:.1f}%")

    # A. EMA 趋势排列 (生产核心信号)
    e7 = h4["close"].ewm(span=7, adjust=False).mean()
    e25 = h4["close"].ewm(span=25, adjust=False).mean()
    e99 = h4["close"].ewm(span=99, adjust=False).mean()
    print("\n=== A. EMA 趋势排列 → 未来方向(生产核心信号验证) ===")
    pct_up(h4, (e7 > e25) & (e25 > e99), "多头排列")
    pct_up(h4, (e7 < e25) & (e25 < e99), "空头排列")
    pct_up(h4, (e7 > e25) & (e25 < e99), "纠结(7>25但25<99)")
    # 分两段稳定性
    cut = h4.index[int(len(h4) * 0.6)]
    print("  -- 分段 --")
    for seg, d in [("前60%", h4[h4.index < cut]), ("后40%", h4[h4.index >= cut])]:
        e7s, e25s, e99s = (x.reindex(d.index) for x in (e7, e25, e99))
        m_up = (e7s > e25s) & (e25s > e99s)
        m_dn = (e7s < e25s) & (e25s < e99s)
        if m_up.sum() > 100:
            print(f"  {seg}: 多头 n={int(m_up.sum()):4d} 未来涨 {d['fwd_up'][m_up].mean()*100:.1f}% | "
                  f"空头 n={int(m_dn.sum()):4d} 未来涨 {d['fwd_up'][m_dn].mean()*100:.1f}%")

    # B. 跨周期动量: 1日/7日涨跌 → 4h方向
    print("\n=== B. 跨周期动量 → 未来4h方向 ===")
    h4["mom_1d"] = (h4["close"] / h4["close"].shift(6) - 1) * 100
    h4["mom_7d"] = (h4["close"] / h4["close"].shift(42) - 1) * 100
    for col, bins in [("mom_1d", [(-99, -3), (-3, -1), (-1, 1), (1, 3), (3, 99)]),
                      ("mom_7d", [(-99, -10), (-10, -3), (-3, 3), (3, 10), (10, 99)])]:
        print(f"  -- {col} --")
        for lo, hi in bins:
            m = (h4[col] >= lo) & (h4[col] < hi)
            pct_up(h4, m, f"[{lo:>4},{hi:>4})")

    # C. 量价背离: 20期新高 + 量缩
    print("\n=== C. 量价背离(20期新高/新低 + 量能) ===")
    hi20 = h4["high"].rolling(20).max().shift(1)
    lo20 = h4["low"].rolling(20).min().shift(1)
    vol_ma = h4["volume"].rolling(20).mean()
    new_hi = h4["close"] > hi20
    new_lo = h4["close"] < lo20
    shrink = h4["volume"] < vol_ma
    pct_up(h4, new_hi & shrink, "20期新高+缩量(背离)")
    pct_up(h4, new_hi & ~shrink, "20期新高+放量")
    pct_up(h4, new_lo & shrink, "20期新低+缩量")
    pct_up(h4, new_lo & ~shrink, "20期新低+放量")

    # D. 收盘位置: close 在 4h 高低区间分位
    print("\n=== D. 收盘位置(4h内) → 未来4h方向 ===")
    pos = (h4["close"] - h4["low"]) / (h4["high"] - h4["low"])
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        m = (pos >= lo) & (pos < hi)
        pct_up(h4, m, f"收盘位[{lo:.1f},{hi:.1f})")

    # E. 距20期摆动极值距离 → 未来方向
    print("\n=== E. 距摆动极值距离 → 未来4h方向 ===")
    h4["dist_hi"] = (hi20 / h4["close"] - 1) * 100   # 距20期高点距离%(正=在下)
    h4["dist_lo"] = (h4["close"] / lo20 - 1) * 100   # 距20期低点距离%
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]:
        m = (h4["dist_hi"] >= lo) & (h4["dist_hi"] < hi)
        pct_up(h4, m, f"距20期高点[{lo:>2},{hi:>2})%")
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]:
        m = (h4["dist_lo"] >= lo) & (h4["dist_lo"] < hi)
        pct_up(h4, m, f"距20期低点[{lo:>2},{hi:>2})%")


if __name__ == "__main__":
    main()
