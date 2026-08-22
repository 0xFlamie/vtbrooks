"""宏观事件日 vs 普通日: ETH 波动/方向统计(独立分析, 2026-08-22)
验证: FOMC决议日/CPI发布日的 4h窗口波动是否放大、方向有无偏差
数据: Hyperliquid 4h (2024-05起, 5000根)
运行: python3 analysis/event_test.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import vt_vote_bot as V

# FOMC 决议日(美东), UTC 日期; 决议 14:00 ET = 18:00/19:00 UTC
FOMC_DATES = [
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29",
]
# CPI 发布日(美东 8:30 ET): BLS 通常在每月 10-15 日的某工作日, 用近似宽窗口
CPI_MONTHS = [(y, m) for y in range(2021, 2027) for m in range(1, 13)]


def cpi_dates():
    """每月 10-15 日中的工作日(UTC 日期), 近似 CPI 发布日"""
    out = []
    for y, m in CPI_MONTHS:
        for d in range(10, 16):
            ts = pd.Timestamp(y, m, d)
            if ts.dayofweek < 5:
                out.append(str(ts.date()))
                break
    return out


def main():
    from feature_explore import fetch_cb_interval, resample_4h
    h4 = resample_4h(fetch_cb_interval(3600, days=1825))
    c, hi, lo = h4["close"], h4["high"], h4["low"]

    # 事件日标记: 当天 UTC 的任何 4h bar
    ev_days = set(FOMC_DATES + cpi_dates())
    is_ev = pd.Series([str(ts.date()) in ev_days for ts in h4.index], index=h4.index)

    # 每个事件日窗口: 事件日00:00 UTC 起 24h 的振幅 (用4h bar拼接)
    def day_amp(day_idx):
        day = h4[h4.index.date == day_idx.date()]
        if len(day) < 3:
            return None
        return (day["high"].max() - day["low"].min()) / day["close"].iloc[0] * 100

    days = h4.groupby(h4.index.date).apply(
        lambda g: pd.Series({"amp": (g["high"].max() - g["low"].min()) / g["close"].iloc[0] * 100,
                             "ret": (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100}))
    days["is_ev"] = [str(d) in ev_days for d in days.index]
    days = days.dropna()
    print(f"总交易日 {len(days)} | 事件日 {days['is_ev'].sum()} (FOMC+CPI)")

    g_amp = days["amp"].mean()
    g_ret = days["ret"].mean()
    print(f"\n全局: 日振幅 {g_amp:.2f}% | 日收益 {g_ret:+.2f}% | 上涨天数 {(days['ret'] > 0).mean()*100:.1f}%")

    ev = days[days["is_ev"]]
    print(f"\n=== 事件日 vs 普通日 ===")
    print(f"事件日: n={len(ev)} 平均振幅 {ev['amp'].mean():.2f}% ({(ev['amp'].mean()/g_amp-1)*100:+.0f}%) | "
          f"平均收益 {ev['ret'].mean():+.2f}% | 上涨 {(ev['ret'] > 0).mean()*100:.0f}%")
    print(f"普通日: n={len(days[~days['is_ev']])} 平均振幅 {days[~days['is_ev']]['amp'].mean():.2f}% | "
          f"平均收益 {days[~days['is_ev']]['ret'].mean():+.2f}%")

    # 分 FOMC / CPI 看
    fomc = days[[str(d) in set(FOMC_DATES) for d in days.index]]
    cpi = days[[str(d) in set(cpi_dates()) for d in days.index]]
    print(f"\nFOMC日: n={len(fomc)} 振幅 {fomc['amp'].mean():.2f}% ({(fomc['amp'].mean()/g_amp-1)*100:+.0f}%) | "
          f"收益 {fomc['ret'].mean():+.2f}% | 上涨 {(fomc['ret'] > 0).mean()*100:.0f}%")
    print(f"CPI日: n={len(cpi)} 振幅 {cpi['amp'].mean():.2f}% ({(cpi['amp'].mean()/g_amp-1)*100:+.0f}%) | "
          f"收益 {cpi['ret'].mean():+.2f}% | 上涨 {(cpi['ret'] > 0).mean()*100:.0f}%")

    # 事件日前后的波动传导: 事件日振幅 vs 前一日 (days.index 为 date 对象)
    ev_list = sorted(set(FOMC_DATES + cpi_dates()))
    ev_dates = [pd.Timestamp(d).date() for d in ev_list if pd.Timestamp(d).date() in days.index]
    prev_amp, ev_amp, next_amp = [], [], []
    for d in ev_dates:
        i = list(days.index).index(d)
        if i >= 1:
            prev_amp.append(days.iloc[i - 1]["amp"])
        ev_amp.append(days.iloc[i]["amp"])
        if i + 1 < len(days):
            next_amp.append(days.iloc[i + 1]["amp"])
    if prev_amp:
        print(f"\n事件日传导: 前一日 {np.mean(prev_amp):.2f}% → 事件日 {np.mean(ev_amp):.2f}% → 次日 {np.mean(next_amp):.2f}%")


if __name__ == "__main__":
    main()
