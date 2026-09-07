#!/usr/bin/env python3
"""Brooks 价格行为学模式在 BTC/ETH K 线上的历史验证（独立脚本，不改生产代码）。

方法：
- 拉最近 N 天 5m/15m K 线（币安 fapi，经 fetch_klines 复用）
- 滑动窗口(120根，与生产一致)逐根调用 brooks_analyze，收集 6 票模式的触发
- 每个触发点评估未来 horizon 根：
    a) R 胜率: 先触及 +1ATR(赢) 还是 -1ATR(输)（同方向做多/做空）
    b) 方向命中率: horizon 后收盘价方向
    c) 平均收益 bp
- 同一 (模式,方向) 5 根内去重，避免连续触发重复计数
"""
import sys
import time
import numpy as np
import pandas as pd
import vt_vote_bot as v

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = {"5m": {"ms": 300_000, "horizon": 48},  # 4h 观察窗
             "15m": {"ms": 900_000, "horizon": 24}}  # 6h 观察窗
DAYS = 90
COOLDOWN = 5  # 同模式同方向 5 根内只记一次
R_MULT = 0.5  # 赢/输判定: ±0.5ATR
ATR_N = 14


def load_history(symbol, interval, days):
    """直连 fapi 分批拉历史 K 线（绕开 fetch_klines 的 binance.us 超时回退）"""
    ms = INTERVALS[interval]["ms"]
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    frames, t = [], start
    while t < end:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
               f"&interval={interval}&limit=1000&startTime={t}")
        data = v.http_get_json(url, timeout=10)
        if not isinstance(data, list) or not data:
            print(f"  WARN: {symbol} {interval} 在 {t} 拉到空, 提前结束")
            break
        df = v._parse_binance(data, interval, drop_incomplete=True)
        if df is None or df.empty:
            break
        frames.append(df)
        last = df.index[-1]
        t = int(last.timestamp() * 1000) + ms
        if len(df) < 1000:
            break
        time.sleep(0.1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index()
    print(f"  {symbol} {interval}: {len(df)} 根 ({df.index[0]} ~ {df.index[-1]})")
    return df


def atr_at(df, i):
    """触发点 ATR = 前 ATR_N 根 mean(high-low)"""
    return float(np.mean(df["high"].values[i - ATR_N:i] - df["low"].values[i - ATR_N:i]))


def eval_trade(df, i, direction, horizon, atr):
    """触发后未来 horizon 根: 返回 (r_result, dir_hit, ret_bp)
    r_result: 1=先到+1ATR, -1=先到-1ATR, 0=都没到
    dir_hit: 收盘方向是否与 direction 一致 (bool/None)
    ret_bp: horizon 后相对收益 bp"""
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    p0 = c[i]
    j = i + 1
    end = min(i + horizon + 1, len(c))
    while j < end:
        if direction == 1:
            if h[j] >= p0 + R_MULT * atr:
                return 1, (c[min(end - 1, len(c) - 1)] > p0), (c[min(end - 1, len(c) - 1)] / p0 - 1) * 1e4
            if l[j] <= p0 - R_MULT * atr:
                return -1, (c[min(end - 1, len(c) - 1)] > p0), (c[min(end - 1, len(c) - 1)] / p0 - 1) * 1e4
        else:
            if l[j] <= p0 - R_MULT * atr:
                return 1, (c[min(end - 1, len(c) - 1)] < p0), (c[min(end - 1, len(c) - 1)] / p0 - 1) * 1e4
            if h[j] >= p0 + R_MULT * atr:
                return -1, (c[min(end - 1, len(c) - 1)] < p0), (c[min(end - 1, len(c) - 1)] / p0 - 1) * 1e4
        j += 1
    return 0, (c[min(end - 1, len(c) - 1)] > p0) if direction == 1 else (c[min(end - 1, len(c) - 1)] < p0), \
        (c[min(end - 1, len(c) - 1)] / p0 - 1) * (1e4 if direction == 1 else -1e4)


def run(symbol, interval):
    df = load_history(symbol, interval, DAYS)
    if df.empty or len(df) < 200:
        return None
    n = len(df)
    stats = {}  # mode -> {"n":0, "wins":0, "dir_hit":0, "ret":[]}
    last_seen = {}  # (mode, dir) -> index
    t0 = time.time()
    for i in range(120, n):
        if i % 3000 == 0:
            print(f"    {i}/{n} ({time.time()-t0:.0f}s)")
        window = df.iloc[i - 119:i + 1]
        ba = v.brooks_analyze(window)
        atr = atr_at(df, i)
        if atr <= 0:
            continue
        for name, direction in ba["votes"]:
            key = (name, direction)
            if key in last_seen and i - last_seen[key] < COOLDOWN:
                continue
            last_seen[key] = i
            r, hit, ret = eval_trade(df, i, direction, INTERVALS[interval]["horizon"], atr)
            s = stats.setdefault(name, {"n": 0, "wins": 0, "dir_hit": 0, "decided": 0, "ret": []})
            s["n"] += 1
            if r == 1:
                s["wins"] += 1
            if r != 0:
                s["decided"] += 1
            if hit:
                s["dir_hit"] += 1
            s["ret"].append(ret)
    return stats


def report(symbol, interval, stats):
    print(f"\n===== {symbol} {interval} =====")
    if not stats:
        print("  无触发")
        return
    print(f"{'模式':<20} {'触发':>4} {'决胜':>4} {'决胜胜率':>7} {'方向命中':>7} {'平均收益bp':>9}")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        rw = 100 * s["wins"] / s["n"]
        dw = 100 * s["wins"] / s["decided"] if s["decided"] else 0
        dh = 100 * s["dir_hit"] / s["n"]
        ar = np.mean(s["ret"])
        print(f"{name:<20} {s['n']:>4} {s['decided']:>4} {dw:>6.0f}% {dh:>6.0f}% {ar:>+8.0f}")


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else [f"{s}:{iv}" for s in SYMBOLS for iv in INTERVALS]
    for combo in which:
        sym, iv = combo.split(":")
        t0 = time.time()
        st = run(sym, iv)
        report(sym, iv, st)
        print(f"  耗时 {time.time()-t0:.0f}s")
