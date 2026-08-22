"""第四轮深挖: 资金费率历史(HL 3.3年)/BTC联动/波动率×方向交互/突破延续
生产系统把费率当"拥挤度反向信号"注入简报——从未用历史验证, 这一轮补上
运行: python3 analysis/fourth_explore.py
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from feature_explore import fetch_okx_4h


def fetch_hl_funding(coin="ETH", start_ms=1640995200000):
    """Hyperliquid 资金费率历史(每2h结算), 分页拉全"""
    rows = []
    cur = start_ms
    while len(rows) < 25000:
        try:
            r = requests.post("https://api.hyperliquid.xyz/info",
                              json={"type": "fundingHistory", "coin": coin, "startTime": cur},
                              timeout=(5, 15))
            d = r.json()
        except Exception:
            break
        if not isinstance(d, list) or not d:
            break
        rows.extend(d)
        last = int(d[-1]["time"])
        if last <= cur:
            break
        cur = last + 1
        time.sleep(0.1)
    df = pd.DataFrame([{"ts": pd.to_datetime(int(x["time"]), unit="ms"), "fr": float(x["fundingRate"])} for x in rows])
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def pct_up(d, m, label):
    if m.sum() < 100:
        print(f"  {label}: n={int(m.sum()):4d} 样本不足")
        return
    up = d["fwd_up"][m].mean() * 100
    avg = d["fwd_ret"][m].mean()
    print(f"  {label}: n={int(m.sum()):4d} 未来4h涨 {up:.1f}% | 平均 {avg:+.3f}%")


def main():
    print("拉取 ETH/BTC 4h 5年 + HL 费率 3.3年...", flush=True)
    h4 = fetch_okx_4h(days=1825)
    btc = fetch_okx_4h("BTC-USDT", days=1825)
    fund = fetch_hl_funding()
    print(f"ETH {len(h4)} | BTC {len(btc)} | 费率 {len(fund)} ({fund.index[0]:%Y-%m-%d} 起)", flush=True)

    h4["ret"] = h4["close"].pct_change() * 100
    h4["fwd_up"] = (h4["close"].pct_change(-1) > 0).astype(int)
    h4["fwd_ret"] = h4["close"].pct_change(-1) * 100
    h4["fwd_up_12h"] = (h4["close"].shift(-3) > h4["close"]).astype(int)
    h4["fwd_ret_12h"] = (h4["close"].shift(-3) / h4["close"] - 1) * 100
    base = h4["fwd_up"].mean() * 100
    print(f"基线: 未来4h涨 {base:.1f}% | 未来12h涨 {h4['fwd_up_12h'].mean()*100:.1f}%")

    # A. 资金费率 → 未来方向 (每根4h取最后一个费率, 对齐)
    fr4 = fund["fr"].resample("4h").last()
    h4["fr"] = fr4.reindex(h4.index).ffill()
    print("\n=== A. 资金费率(4h) → 未来方向 ===")
    for label, m in [("多付空>0.05%(拥挤)", h4["fr"] > 0.0005),
                     ("多付空>0.03%", h4["fr"] > 0.0003),
                     ("空付多<-0.05%(反向拥挤)", h4["fr"] < -0.0005),
                     ("中性|fr|<0.01%", h4["fr"].abs() < 0.0001)]:
        pct_up(h4, m, label)
    # 分段稳定性(拥挤度反向信号)
    cut = h4.index[int(len(h4) * 0.6)]
    print("  -- 多付空>0.05% 分段 --")
    for seg, d in [("前60%", h4[h4.index < cut]), ("后40%", h4[h4.index >= cut])]:
        m = d["fr"] > 0.0005
        if m.sum() > 50:
            print(f"  {seg}: n={int(m.sum()):4d} 未来涨 {d['fwd_up'][m].mean()*100:.1f}%")

    # B. BTC 联动: BTC 前4h涨跌 → ETH 未来4h方向
    print("\n=== B. BTC 前4h涨跌 → ETH 未来4h方向 ===")
    h4["btc_ret"] = btc["close"].pct_change().reindex(h4.index) * 100
    for lo, hi in [(-99, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, 99)]:
        m = (h4["btc_ret"] >= lo) & (h4["btc_ret"] < hi)
        pct_up(h4, m, f"BTC前4h[{lo:>4},{hi:>4})%")

    # C. 波动率×方向交互: 前4h振幅分档 → 未来方向
    print("\n=== C. 波动率 × 方向交互 ===")
    h4["range_pct"] = (h4["high"] - h4["low"]) / h4["open"] * 100
    for lo, hi in [(0, 1), (1, 2), (2, 3.5), (3.5, 99)]:
        m = (h4["range_pct"] >= lo) & (h4["range_pct"] < hi)
        pct_up(h4, m, f"前4h振幅[{lo:>3},{hi:>3})%")
    # 高波动+大跌 → 反弹?
    hi_vol = h4["range_pct"] > 3.5
    big_dn = h4["ret"] <= -2
    print("  -- 高波动×大跌组合 --")
    pct_up(h4, hi_vol & big_dn, "高波动且4h大跌≤-2%")
    pct_up(h4, hi_vol & (h4["ret"] >= 2), "高波动且4h大涨≥2%")

    # D. 突破延续: 收破20期高点/低点 → 未来12h
    print("\n=== D. 20期突破延续 → 未来12h ===")
    hi20 = h4["high"].rolling(20).max().shift(1)
    lo20 = h4["low"].rolling(20).min().shift(1)
    brk_up = h4["close"] > hi20
    brk_dn = h4["close"] < lo20
    for label, m in [("突破20期高点", brk_up), ("跌破20期低点", brk_dn)]:
        if m.sum() < 100:
            print(f"  {label}: n={int(m.sum()):4d} 样本不足")
            continue
        up12 = h4["fwd_up_12h"][m].mean() * 100
        avg12 = h4["fwd_ret_12h"][m].mean()
        print(f"  {label}: n={int(m.sum()):4d} 未来12h涨 {up12:.1f}% | 平均 {avg12:+.3f}%")
    # 突破后回踩 vs 继续: 突破当根量能
    vol_ma = h4["volume"].rolling(20).mean()
    print("  -- 突破 + 量能 --")
    pct_up(h4, brk_up & (h4["volume"] > 1.5 * vol_ma), "突破20期高点+放量1.5x")
    pct_up(h4, brk_up & (h4["volume"] < vol_ma), "突破20期高点+缩量")
    pct_up(h4, brk_dn & (h4["volume"] > 1.5 * vol_ma), "跌破20期低点+放量1.5x")
    pct_up(h4, brk_dn & (h4["volume"] < vol_ma), "跌破20期低点+缩量")


if __name__ == "__main__":
    main()
