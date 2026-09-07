# -*- coding: utf-8 -*-
"""ADX and DI (Wilder, TradingView 原版 period=14) 有效性回测

TV 原版: +DM/-DM 方向运动, TR/+DM/-DM 均用 RMA(Wilder) 平滑,
         DX=100*|+DI-DI|/(+DI+-DI), ADX=RMA(DX,14)
信号全部收线确认, 入场用下一根 open

测试:
  A. 趋势跟随: ADX>25 且 +DI>-DI -> 做多; ADX>25 且 -DI>+DI -> 做空
     条件方向概率(未来4h/12h/24h) + 机械持仓(条件破坏下一根open出场) 扣费EV
  B. 启动信号: ADX 上穿 25 且近 5 根内曾 <20 -> 顺 DI 方向入场
  C. ADX 作为预测因子: ADX 分位(全样本五分位) vs 未来 4h/12h 收益均值/胜率 和未来 12h 振幅
费用: 双边 taker 各 5bp, 共 10bp; 时间切分: 前2/3 train, 后1/3 test
"""
import numpy as np
import pandas as pd

ADX_PERIOD = 14
ADX_TREND = 25.0        # 趋势阈值
ADX_LOW = 20.0          # 启动信号前置低值
START_LOOKBACK = 5      # 上穿前多少根内须 <ADX_LOW
MAX_HOLD = 30           # A 机械持仓上限(根)
FEE_SIDE = 0.0005
HORIZONS_H = (4, 12, 24)

DATA = {
    "4h": "analysis/tv_indicators/data/okx_4h.csv",
    "1h": "analysis/tv_indicators/data/okx_1h.csv",
}
BAR_H = {"4h": 4, "1h": 1}


def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def compute_atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def adx_di(df, n=ADX_PERIOD):
    h, l = df["high"], df["low"]
    up = h.diff()
    down = -l.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = compute_atr(df, n)
    pdi = 100.0 * rma(pdm, n) / atr_
    mdi = 100.0 * rma(mdm, n) / atr_
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    return rma(dx, n), pdi, mdi


def regime_trades(df, cond_long, cond_short):
    """条件为真期间持仓: 条件成立根收线确认 -> 下一根 open 进;
    条件破坏 -> 下一根 open 出; 上限 MAX_HOLD 根"""
    open_ = df["open"].values
    close = df["close"].values
    cl = cond_long.fillna(False).values
    cs = cond_short.fillna(False).values
    m = len(df)
    trades = []
    i = 1
    while i < m - 1:
        d = 0
        if cl[i] and not cl[i - 1]:
            d = 1
        elif cs[i] and not cs[i - 1]:
            d = -1
        if d == 0:
            i += 1
            continue
        ei = i + 1
        entry = open_[ei]
        cond = cl if d == 1 else cs
        xi = None
        j = ei
        while j < m - 1:
            if not cond[j] or j - ei >= MAX_HOLD:
                xi = j + 1
                break
            j += 1
        if xi is None or xi >= m:
            xi = m - 1
        ret = (open_[xi] / entry - 1.0) * d * 100.0
        trades.append((ei, d, ret))
        i = xi + 1
    return trades


def trade_stats(trades):
    if not trades:
        return None
    rets = np.array([t[2] for t in trades])
    rets_net = rets - 2 * FEE_SIDE * 100.0
    return {
        "n": len(rets),
        "win": float((rets > 0).mean() * 100),
        "avg_gross": float(rets.mean()),
        "avg_net": float(rets_net.mean()),
        "ev_pos": bool(rets_net.mean() > 0),
        "tail_dep": float(np.sort(rets)[-max(1, len(rets) // 10):].sum() / rets.sum()) if rets.sum() != 0 else np.nan,
    }


def fwd_stats(close, idx, dirs, hb, base):
    out = []
    for i, d in zip(idx, dirs):
        if i + hb < len(close):
            out.append((close[i + hb] / close[i] - 1.0) * d * 100.0)
    out = np.array(out)
    if len(out) == 0:
        return None
    return len(out), float((out > 0).mean() * 100), float((out > 0).mean() * 100 - base), float(out.mean())


def run(tf):
    df = pd.read_csv(DATA[tf], index_col=0, parse_dates=True)
    adx_v, pdi, mdi = adx_di(df)
    split = int(len(df) * 2 / 3)
    bar_h = BAR_H[tf]
    close = df["close"].values

    cond_long = (adx_v > ADX_TREND) & (pdi > mdi)
    cond_short = (adx_v > ADX_TREND) & (mdi > pdi)

    print(f"\n{'='*78}\n周期: {tf}  bars={len(df)}  split={df.index[split]}  "
          f"long条件覆盖率={cond_long.mean()*100:.1f}%  short={cond_short.mean()*100:.1f}%\n{'='*78}")

    # --- A. 趋势跟随: 条件方向概率(按根计) + 机械持仓(按笔计) ---
    print("A. ADX>25 趋势跟随")
    li = np.where(cond_long.fillna(False).values)[0]
    si = np.where(cond_short.fillna(False).values)[0]
    for hh in HORIZONS_H:
        hb = hh // bar_h
        fwd = (close[hb:] / close[:-hb] - 1.0) * 100.0
        base = float((fwd > 0).mean() * 100)
        for name, idx in (("多", li), ("空", si)):
            d = 1 if name == "多" else -1
            for label, sub in (("train", idx[idx < split]), ("test", idx[idx >= split]),
                               ("full", idx)):
                r = fwd_stats(close, sub, np.full(len(sub), d), hb, base)
                if r is None:
                    continue
                n, win, dpp, mean = r
                note = " 样本不足" if n < 100 else ""
                print(f"  {name} {hh}h [{label:5s}] n={n:5d} 胜率={win:5.1f}% (基线{base:.1f}%, {dpp:+.1f}pp) 均值={mean:+.3f}%{note}")
    all_t = regime_trades(df, cond_long, cond_short)
    for label, ts in (("train", [t for t in all_t if t[0] < split]),
                      ("test", [t for t in all_t if t[0] >= split]),
                      ("full", all_t)):
        s = trade_stats(ts)
        if s is None:
            print(f"  机械持仓 [{label}] n=0")
            continue
        note = " 样本不足" if s["n"] < 100 else ""
        print(f"  机械持仓 [{label:5s}] n={s['n']:4d} 胜率={s['win']:5.1f}% 毛均={s['avg_gross']:+.3f}% "
              f"扣费EV={s['avg_net']:+.3f}% EV正={s['ev_pos']} 右尾依赖={s['tail_dep']:.2f}{note}")

    # --- B. 启动信号: ADX 上穿 25 且近 START_LOOKBACK 根内曾 <20 ---
    av = adx_v.values
    cross = (av[1:] >= ADX_TREND) & (av[:-1] < ADX_TREND)
    cidx = np.where(cross)[0] + 1
    start_idx, start_dir = [], []
    pdiv, mdiv = pdi.values, mdi.values
    for i in cidx:
        lo = max(0, i - START_LOOKBACK)
        if np.nanmin(av[lo:i]) < ADX_LOW:
            start_idx.append(i)
            start_dir.append(1 if pdiv[i] > mdiv[i] else -1)
    start_idx = np.array(start_idx)
    start_dir = np.array(start_dir)
    print(f"\nB. ADX 上穿25启动信号 (上穿前{START_LOOKBACK}根内曾<20)  n_total={len(start_idx)}")
    for hh in HORIZONS_H:
        hb = hh // bar_h
        fwd = (close[hb:] / close[:-hb] - 1.0) * 100.0
        base = float((fwd > 0).mean() * 100)
        for label, m_ in (("train", start_idx < split), ("test", start_idx >= split),
                          ("full", np.ones(len(start_idx), bool))):
            r = fwd_stats(close, start_idx[m_], start_dir[m_], hb, base)
            if r is None:
                continue
            n, win, dpp, mean = r
            note = " 样本不足" if n < 100 else ""
            print(f"  {hh}h [{label:5s}] n={n:4d} 胜率={win:5.1f}% (基线{base:.1f}%, {dpp:+.1f}pp) 均值={mean:+.3f}%{note}")

    # --- C. ADX 分位 vs 未来收益和振幅 ---
    print("\nC. ADX 分位因子 (全样本五分位)")
    high = df["high"].values
    low = df["low"].values
    qs = np.nanquantile(av, [0.2, 0.4, 0.6, 0.8])
    for hh, hb in (("4h", 4 // bar_h), ("12h", 12 // bar_h)):
        hb = max(hb, 1)
        fwd = (close[hb:] / close[:-hb] - 1.0) * 100.0
        base = float((fwd > 0).mean() * 100)
        print(f"  未来{hh} (n={len(fwd)}根/桶):")
        for q in range(5):
            lo_q = -np.inf if q == 0 else qs[q - 1]
            hi_q = np.inf if q == 4 else qs[q]
            mask = (av[:-hb] >= lo_q) & (av[:-hb] < hi_q)
            seg = fwd[mask]
            if len(seg) == 0:
                continue
            print(f"    Q{q+1} ADX[{lo_q if q>0 else 0:5.1f},{hi_q if q<4 else 999:5.1f}) "
                  f"n={len(seg):5d} 涨概率={float((seg>0).mean()*100):5.1f}% (基线{base:.1f}%) 均收益={seg.mean():+.3f}%")
    hb = max(12 // bar_h, 1)
    amp = np.array([(np.max(high[i + 1:i + 1 + hb]) - np.min(low[i + 1:i + 1 + hb])) / close[i] * 100.0
                    if i + hb < len(close) else np.nan for i in range(len(close))])
    base_amp = np.nanmean(amp)
    print(f"  未来12h振幅 (全样本均值={base_amp:.2f}%):")
    for q in range(5):
        lo_q = -np.inf if q == 0 else qs[q - 1]
        hi_q = np.inf if q == 4 else qs[q]
        mask = (av >= lo_q) & (av < hi_q) & ~np.isnan(amp)
        seg = amp[mask]
        if len(seg) == 0:
            continue
        print(f"    Q{q+1} n={len(seg):5d} 均振幅={seg.mean():.2f}% ({(seg.mean()/base_amp-1)*100:+.0f}% vs 基线)")


if __name__ == "__main__":
    for tf in ("4h", "1h"):
        run(tf)
