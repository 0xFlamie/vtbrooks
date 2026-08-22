"""挤压分位 → 未来12h振幅 验证(独立分析): 有单调关系且样本外稳定才算 edge
方法: BB宽度20期 → 200期分位 → 未来12h(3根4h)振幅, 分组对比全局, 前2/3 vs 后1/3 切分
运行: python3 analysis/squeeze_test.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import vt_vote_bot as V


def main():
    from feature_explore import fetch_cb_interval, resample_4h
    h4 = resample_4h(fetch_cb_interval(3600, days=1825))
    print(f"4h {len(h4)}根 ({h4.index[0]:%Y-%m-%d} 起, Coinbase 5年)")

    c = h4["close"]
    bbw = 4 * c.rolling(20).std() / c.rolling(20).mean()
    # 挤压分位: 每根对过去200根的分位
    from numpy.lib.stride_tricks import sliding_window_view
    arr = bbw.values
    pct = pd.Series(np.nan, index=h4.index)
    if len(arr) >= 200:
        w = sliding_window_view(arr, 200)
        valid = ~np.isnan(w).any(axis=1)
        pct.iloc[199:] = np.where(valid, (w < w[:, -1:]).mean(axis=1) * 100, np.nan)

    # 未来12h(3根4h)振幅: (high.max - low.min) / close
    hi3 = h4["high"].shift(-1).rolling(3).max().shift(-2)
    lo3 = h4["low"].shift(-1).rolling(3).min().shift(-2)
    fwd_amp = (hi3 - lo3) / c * 100

    df = pd.DataFrame({"pct": pct, "amp": fwd_amp}).dropna()
    df = df[df.index >= h4.index[0] + pd.Timedelta(days=60)]  # 去掉 rolling 预热
    print(f"有效样本 {len(df)}")

    global_amp = df["amp"].mean()
    global_p2 = (df["amp"] >= 2).mean() * 100
    print(f"\n全局: 平均12h振幅 {global_amp:.2f}% | ≥2%概率 {global_p2:.1f}%")

    # 时间切分: 前2/3训练期规律 vs 后1/3
    cut = df.index[int(len(df) * 2 / 3)]
    print(f"切分点 {cut:%Y-%m-%d} | 前段 {len(df[df.index < cut])} 后段 {len(df[df.index >= cut])}")

    print("\n=== 挤压分位 → 12h振幅(全样本) ===")
    for lo_, hi_ in [(0, 20), (20, 40), (40, 70), (70, 100)]:
        m = (df["pct"] >= lo_) & (df["pct"] < hi_)
        sub = df[m]
        if len(sub) < 50:
            print(f"  [{lo_:>3},{hi_:>3}) n={len(sub):4d} 样本不足")
            continue
        amp = sub["amp"].mean()
        p2 = (sub["amp"] >= 2).mean() * 100
        lift = (amp / global_amp - 1) * 100
        print(f"  [{lo_:>3},{hi_:>3}) n={len(sub):4d} 平均{amp:5.2f}% (vs全局{lift:+5.0f}%) | ≥2% {p2:4.1f}% (vs全局{p2-global_p2:+4.1f}pp)")

    print("\n=== 样本外稳定性(分两段分别看) ===")
    for seg_name, seg in [("前段", df[df.index < cut]), ("后段", df[df.index >= cut])]:
        g = seg["amp"].mean()
        print(f"  -- {seg_name} (全局 {g:.2f}%) --")
        for lo_, hi_ in [(0, 20), (20, 40), (40, 70), (70, 100)]:
            m = (seg["pct"] >= lo_) & (seg["pct"] < hi_)
            sub = seg[m]
            if len(sub) < 30:
                continue
            print(f"    [{lo_:>3},{hi_:>3}) n={len(sub):4d} 平均{sub['amp'].mean():5.2f}% (lift {(sub['amp'].mean()/g-1)*100:+5.0f}%)")


if __name__ == "__main__":
    main()
