"""第二轮深挖(5年数据): 均值回归止损回测/RSI极值/时段效应/振幅自相关/ETH-BTC相对强弱
运行: python3 analysis/deeper_explore.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import vt_vote_bot as V
from feature_explore import fetch_cb_interval, resample_4h


def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = d.clip(upper=0).abs().ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-10))


def meanrev_backtest(h4, cond, sl_atr, tp_atr, look=6):
    """均值回归逐bar模拟: 信号后下一根4h开盘入场, 未来look根内先碰止损/目标"""
    atr = (h4["high"] - h4["low"]).rolling(14).mean()
    res = []
    for ts in h4.index[cond]:
        i = h4.index.get_loc(ts)
        if i + 2 >= len(h4):
            break
        entry = h4["open"].iloc[i + 1]
        a = atr.iloc[i + 1]
        if np.isnan(a) or a <= 0:
            continue
        sl, tp = entry - sl_atr * a, entry + tp_atr * a
        win = None
        for j in range(i + 1, min(i + 1 + look, len(h4))):
            hi, lo = h4["high"].iloc[j], h4["low"].iloc[j]
            if lo <= sl:
                win = 0
                break
            if hi >= tp:
                win = 1
                break
        if win is not None:
            res.append(win)
    if not res:
        return None
    n = len(res)
    wr = sum(res) / n
    # 盈亏比 sl:tp → 期望 = wr*tp - (1-wr)*sl (ATR单位)
    ev = wr * tp_atr - (1 - wr) * sl_atr
    return {"n": n, "win%": round(wr * 100, 1), "EV(ATR)": round(ev, 3),
            "盈亏比": f"{tp_atr}:{sl_atr}"}


def main():
    print("拉取数据(OKX 4h 5年)...", flush=True)
    from feature_explore import fetch_okx_4h
    h4 = fetch_okx_4h(days=1825)
    print(f"ETH {len(h4)}根 ({h4.index[0]:%Y-%m-%d} 起)", flush=True)

    # 基础特征
    h4["ret"] = h4["close"].pct_change() * 100
    h4["atr_pct"] = (h4["high"] - h4["low"]).rolling(14).mean() / h4["close"] * 100
    h4["rsi"] = rsi_wilder(h4["close"])
    h4["range_pct"] = (h4["high"] - h4["low"]) / h4["open"] * 100
    # 未来12h振幅
    hi3 = h4["high"].shift(-1).rolling(3).max().shift(-2)
    lo3 = h4["low"].shift(-1).rolling(3).min().shift(-2)
    h4["fwd_amp"] = (hi3 - lo3) / h4["close"] * 100

    print(f"\n=== A. RSI 极值 → 未来方向(5年) ===")
    base = h4["close"].shift(-1).gt(h4["close"]).mean() * 100
    print(f"基线: 未来4h上涨 {base:.1f}%")
    for lo_, hi_ in [(0, 30), (30, 40), (60, 70), (70, 100)]:
        m = (h4["rsi"] >= lo_) & (h4["rsi"] < hi_)
        if m.sum() < 50:
            continue
        up = h4["close"].shift(-1)[m].gt(h4["close"][m]).mean() * 100
        print(f"  RSI[{lo_:>3},{hi_:>3}) n={int(m.sum()):4d} 未来4h涨 {up:.1f}%")

    print(f"\n=== B. 时段效应(4h bar 起点 UTC) ===")
    for h in [0, 4, 8, 12, 16, 20]:
        m = h4.index.hour == h
        amp = h4["range_pct"][m].mean()
        up = h4["close"].shift(-1)[m].gt(h4["close"][m]).mean() * 100
        print(f"  UTC{h:02d}点: n={int(m.sum()):4d} 4h振幅{amp:.2f}% 未来涨{up:.1f}%")

    print(f"\n=== C. 振幅自相关(前4h振幅 → 未来12h振幅) ===")
    g = h4["fwd_amp"].mean()
    for lo_, hi_ in [(0, 1), (1, 2), (2, 3.5), (3.5, 99)]:
        m = (h4["range_pct"] >= lo_) & (h4["range_pct"] < hi_)
        if m.sum() < 100:
            continue
        lift = (h4["fwd_amp"][m].mean() / g - 1) * 100
        print(f"  前4h振幅[{lo_:>3},{hi_:>3}) n={int(m.sum()):4d} 未来12h振幅 {(h4['fwd_amp'][m].mean()):.2f}% (lift {lift:+.0f}%)")

    print(f"\n=== D. 均值回归止损回测(大跌后做多 / 大涨后做空, 不同止损目标) ===")
    big_dn = h4["ret"] <= -2
    big_up = h4["ret"] >= 2
    for name, cond, sl, tp in [("大跌后做多", big_dn, 1.0, 1.0), ("大跌后做多", big_dn, 1.0, 0.5),
                               ("大跌后做多", big_dn, 0.5, 1.0), ("大涨后做空", big_up, 1.0, 1.0),
                               ("大涨后做空", big_up, 0.5, 1.0)]:
        r = meanrev_backtest(h4, cond, sl, tp)
        if r:
            print(f"  {name} 止损{sl}目标{tp}: n={r['n']} 胜率{r['win%']}% EV={r['EV(ATR)']} ATR")

    print(f"\n=== E(拆出): ETH/BTC 相对强弱单独跑 pair_test.py ===")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
