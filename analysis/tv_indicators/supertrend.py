# -*- coding: utf-8 -*-
"""SuperTrend (KivancOzbilgic, TradingView 原版) 有效性回测

TV 原版参数: ATR period=10, multiplier=3, src=hl2, ATR=ta.atr(RMA/Wilder)
方向判定: close > 上轨(dn) -> 1; close < 下轨(up) -> -1
信号: 翻转收线确认, 下一根 open 入场, 杜绝未来函数

测试:
  A. 翻转信号入场: 出场=反向翻转(下一根open) 或 1.5xATR 止损(盘中触及, 按止损价成交)
  B. A + 过滤: 信号根 ADX(14)>25 才入场
  C. 翻转后条件方向概率: 未来 4h/12h/24h 顺信号方向收涨概率 vs 全样本基线
费用: 双边 taker 各 5bp, 共 10bp; 时间切分: 前2/3 train, 后1/3 test
"""
import numpy as np
import pandas as pd

ATR_PERIOD = 10
MULT = 3.0
STOP_ATR = 1.5          # ATR 止损倍数
ADX_PERIOD = 14         # 仅用于过滤 B
ADX_FILTER = 25.0
FEE_SIDE = 0.0005       # 单边 taker 5bp
HORIZONS_H = (4, 12, 24)

DATA = {
    "4h": "analysis/tv_indicators/data/okx_4h.csv",
    "1h": "analysis/tv_indicators/data/okx_1h.csv",
}
BAR_H = {"4h": 4, "1h": 1}


def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def compute_atr(df, n=ATR_PERIOD):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def supertrend(df, n=ATR_PERIOD, mult=MULT):
    """TV 原版 SuperTrend 迭代逻辑, 返回 trend(+1/-1) 序列"""
    hl2 = (df["high"] + df["low"]) / 2.0
    atr = compute_atr(df, n)
    up_raw = (hl2 - mult * atr).values   # 下轨(多头线)
    dn_raw = (hl2 + mult * atr).values   # 上轨(空头线)
    close = df["close"].values
    m = len(df)
    up = np.full(m, np.nan)
    dn = np.full(m, np.nan)
    trend = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(up_raw[i]):
            continue
        if i == 0 or np.isnan(up[i - 1]):
            up[i] = up_raw[i]
            dn[i] = dn_raw[i]
            trend[i] = 1
            continue
        up[i] = max(up_raw[i], up[i - 1]) if close[i - 1] > up[i - 1] else up_raw[i]
        dn[i] = min(dn_raw[i], dn[i - 1]) if close[i - 1] < dn[i - 1] else dn_raw[i]
        if close[i] > dn[i - 1]:
            trend[i] = 1
        elif close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


def adx(df, n=ADX_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = compute_atr(df, n)
    pdi = 100.0 * rma(pdm, n) / atr_
    mdi = 100.0 * rma(mdm, n) / atr_
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    return rma(dx, n), pdi, mdi


def simulate_trades(df, trend, atr_, allow):
    """翻转入场机械回测; allow[i]=True 才允许在该信号入场(用于 ADX 过滤)
    返回每笔 (入场bar, 方向, 收益%)"""
    close = df["close"].values
    open_ = df["open"].values
    low = df["low"].values
    high = df["high"].values
    atr_v = atr_.values
    tv = trend.values
    m = len(df)
    trades = []
    i = 1
    while i < m - 1:
        # 收线确认的翻转: trend[i] != trend[i-1]
        if not np.isnan(tv[i]) and tv[i] != tv[i - 1] and allow.iloc[i]:
            d = int(tv[i])
            ei = i + 1                      # 下一根 open 入场
            if ei >= m:
                break
            entry = open_[ei]
            stop = entry - d * STOP_ATR * atr_v[i]
            xi = None
            exit_price = None
            j = ei
            while j < m - 1:
                hit_stop = (low[j] <= stop) if d == 1 else (high[j] >= stop)
                if hit_stop:
                    xi, exit_price = j, stop
                    break
                if not np.isnan(tv[j]) and tv[j] == -d and tv[j - 1] == d:
                    xi = j + 1              # 反向翻转收线确认, 下一根 open 出
                    if xi >= m:
                        xi = m - 1
                    exit_price = open_[xi]
                    break
                j += 1
            if xi is None:
                xi, exit_price = m - 1, close[m - 1]
            ret = (exit_price / entry - 1.0) * d * 100.0
            trades.append((ei, d, ret))
            i = xi + 1
        else:
            i += 1
    return trades


def cond_prob(df, sig_idx, sig_dir, horizon_bars, baseline):
    """信号根 close 后未来 horizon 根, 顺信号方向收益"""
    close = df["close"].values
    out = []
    for i, d in zip(sig_idx, sig_dir):
        if i + horizon_bars < len(close):
            out.append((close[i + horizon_bars] / close[i] - 1.0) * d * 100.0)
    out = np.array(out)
    if len(out) == 0:
        return None
    return {
        "n": len(out),
        "win": float((out > 0).mean() * 100),
        "vs_base": float((out > 0).mean() * 100 - baseline),
        "mean": float(out.mean()),
        "median": float(np.median(out)),
        "tail_dep": float(out[out > 0].sum() / np.abs(out).sum()) if np.abs(out).sum() > 0 else np.nan,
    }


def trade_stats(trades):
    if not trades:
        return None
    rets = np.array([t[2] for t in trades])
    rets_net = rets - 2 * FEE_SIDE * 100.0
    wins = rets[rets > 0]
    return {
        "n": len(rets),
        "win": float((rets > 0).mean() * 100),
        "avg_gross": float(rets.mean()),
        "avg_net": float(rets_net.mean()),
        "ev_pos": bool(rets_net.mean() > 0),
        "tail_dep": float(np.sort(rets)[-max(1, len(rets) // 10):].sum() / rets.sum()) if rets.sum() != 0 else np.nan,
    }


def run(tf):
    df = pd.read_csv(DATA[tf], index_col=0, parse_dates=True)
    atr_ = compute_atr(df)
    trend = supertrend(df)
    adx_v, _, _ = adx(df)
    split = int(len(df) * 2 / 3)
    bar_h = BAR_H[tf]

    flips = np.where((trend.values[1:] != trend.values[:-1]) & ~np.isnan(trend.values[1:]))[0] + 1
    sig_dir = trend.values[flips].astype(int)

    allow_all = pd.Series(True, index=df.index)
    allow_adx = (adx_v > ADX_FILTER).fillna(False)

    print(f"\n{'='*78}\n周期: {tf}  bars={len(df)}  翻转信号总数={len(flips)}  split={df.index[split]}\n{'='*78}")

    # --- 机械入场回测 A / B ---
    for name, allow in (("A. 裸翻转", allow_all), ("B. 翻转+ADX>25过滤", allow_adx)):
        all_t = simulate_trades(df, trend, atr_, allow)
        tr = [t for t in all_t if t[0] < split]
        te = [t for t in all_t if t[0] >= split]
        for label, ts in (("train", tr), ("test", te), ("full", all_t)):
            s = trade_stats(ts)
            if s is None:
                print(f"{name} [{label}] n=0")
                continue
            print(f"{name} [{label:5s}] n={s['n']:4d} 胜率={s['win']:5.1f}% "
                  f"毛均={s['avg_gross']:+.3f}% 扣费EV={s['avg_net']:+.3f}% "
                  f"EV正={s['ev_pos']} 右尾依赖={s['tail_dep']:.2f}")
        if len(all_t) < 100:
            print(f"  !! {name} 全样本 n<100, 样本不足")

    # --- 条件方向概率 C ---
    close = df["close"].values
    print(f"\nC. 翻转后条件方向概率 (顺信号方向, vs 全样本无条件基线)")
    for hh in HORIZONS_H:
        hb = hh // bar_h
        fwd = (close[hb:] / close[:-hb] - 1.0) * 100.0
        base = float((fwd > 0).mean() * 100)
        for label, mask in (("train", flips < split), ("test", flips >= split), ("full", np.ones(len(flips), bool))):
            r = cond_prob(df, flips[mask], sig_dir[mask], hb, base)
            if r is None:
                continue
            note = " 样本不足" if r["n"] < 100 else ""
            print(f"  {hh}h [{label:5s}] n={r['n']:4d} 胜率={r['win']:5.1f}% (基线{base:.1f}%, {r['vs_base']:+.1f}pp) "
                  f"均值={r['mean']:+.3f}% 右尾依赖={r['tail_dep']:.2f}{note}")


if __name__ == "__main__":
    for tf in ("4h", "1h"):
        run(tf)
