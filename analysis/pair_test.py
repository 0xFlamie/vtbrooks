"""ETH/BTC 相对强弱 → ETH 未来4h方向(5年): 相对动量是否有独立预测力
运行: python3 analysis/pair_test.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from feature_explore import fetch_okx_4h


def main():
    print("拉取 ETH/BTC 4h 5年...", flush=True)
    eth = fetch_okx_4h("ETH-USDT", days=1825)
    btc = fetch_okx_4h("BTC-USDT", days=1825)
    print(f"ETH {len(eth)}根 | BTC {len(btc)}根", flush=True)

    df = pd.DataFrame({"eth": eth["close"], "btc": btc["close"].reindex(eth.index).ffill()}).dropna()
    df["pair"] = df["eth"] / df["btc"]  # ETH/BTC 比率
    df["ret"] = df["eth"].pct_change() * 100
    df["fwd_ret"] = df["eth"].pct_change(-1) * 100  # 未来4h ETH收益
    df["fwd_up"] = (df["fwd_ret"] > 0).astype(int)
    # 相对动量: 过去12h(3根) ETH 相对 BTC 的强弱
    df["mom_rel"] = (df["eth"].pct_change(3) - df["btc"].pct_change(3)) * 100
    df["pair_mom"] = df["pair"].pct_change(3) * 100  # 等价

    base = df["fwd_up"].mean() * 100
    print(f"\n基线: 未来4h上涨 {base:.1f}% (n={len(df)})")

    print("\n=== ETH/BTC 相对动量(12h) → ETH 未来4h方向 ===")
    for lo_, hi_ in [(-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)]:
        m = (df["mom_rel"] >= lo_) & (df["mom_rel"] < hi_)
        if m.sum() < 100:
            continue
        up = df["fwd_up"][m].mean() * 100
        avg_ret = df["fwd_ret"][m].mean()
        print(f"  相对动量[{lo_:>5},{hi_:>5}) n={int(m.sum()):4d} 未来涨 {up:.1f}% | 平均收益 {avg_ret:+.3f}%")

    # 时间切分稳定性: 前60% vs 后40%
    cut = df.index[int(len(df) * 0.6)]
    print("\n=== 分段稳定性(相对动量±0.5 分界) ===")
    for seg, d in [("前60%", df[df.index < cut]), ("后40%", df[df.index >= cut])]:
        strong = d[d["mom_rel"] > 0.5]
        weak = d[d["mom_rel"] < -0.5]
        print(f"  {seg}: ETH强于BTC n={len(strong)} 未来涨 {strong['fwd_up'].mean()*100:.1f}% | "
              f"ETH弱于BTC n={len(weak)} 未来涨 {weak['fwd_up'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
