"""4h窗口内15m线特征 vs 未来涨跌: 关系探索 + 时间切分回测(独立分析, 不改生产)

特征(每根4h内的16根15m): 涨根占比/累计涨跌幅/量能分配/窗口振幅/末根方向/最大连涨
标签: 未来4h/12h收盘涨跌
方法: 单特征条件概率表 → IC → 时间切分回测(train找规则, test验证, 防过拟合)
基线: 无脑做多/无脑做空
运行: python3 analysis/feature_explore.py
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import vt_vote_bot as V

COIN = "ETH"
INTERVAL_MS = {"15m": 900000, "4h": 14400000}


def fetch_cb_interval(gran, days, symbol="ETH-USD"):
    """Coinbase 分页拉K线(gran: 900=15m / 3600=1h), 历史 2021+ 完整"""
    rows = []
    end = pd.Timestamp.now(tz="UTC").floor(f"{gran // 60}min")
    start_all = end - pd.Timedelta(days=days)
    cur = end
    while cur > start_all:
        start = max(cur - pd.Timedelta("3 days"), start_all)
        url = (f"https://api.exchange.coinbase.com/products/{symbol}/candles"
               f"?granularity={gran}&start={start.isoformat()}&end={cur.isoformat()}")
        d = V.http_get_json(url)
        if not d:
            break
        rows.extend(d)
        oldest = min(int(r[0]) for r in d)  # 返回降序, 取最旧时间戳往前推
        cur = pd.Timestamp(oldest, unit="s", tz="UTC")
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates("date").set_index("date").sort_index()


def resample_4h(h1):
    """1h → 4h (UTC 0/4/8/12/16/20 对齐, 与生产4h一致)"""
    return h1.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "volume": "sum"}).dropna()


def build_features(h4, m15):
    """每根4h窗口内的15m线统计特征, 对齐未来标签。O(n)构建窗口表, O(1)查询"""
    feat_cols = ["up_ratio", "up_sum", "dn_sum", "vol_up_ratio", "range_pct",
                 "last_dir", "max_streak", "mom_4h", "vol_ratio_4h"]
    rows = []
    # 15m 按 4h bucket 分组, 避免 5年×万窗口 的 O(n²) 切片
    m15b = m15.copy()
    m15b["bucket"] = m15b.index.floor("4h")
    groups = {ts: g for ts, g in m15b.groupby("bucket")}
    vol_ma = h4["volume"].rolling(20).mean()
    for ts, r in h4.iterrows():
        w = groups.get(ts)
        if w is None or len(w) < 10:
            continue
        w = w.copy()
        w["dir"] = np.sign(w["close"] - w["open"])
        ups = w[w["dir"] > 0]
        dns = w[w["dir"] < 0]
        up_sum = ups["close"].sub(ups["open"]).sum() / ups["open"].iloc[0] * 100 if len(ups) else 0.0
        dn_sum = dns["close"].sub(dns["open"]).sum() / dns["open"].iloc[0] * 100 if len(dns) else 0.0
        vol_up = ups["volume"].mean() if len(ups) else 0.0
        vol_dn = dns["volume"].mean() if len(dns) else 0.0
        sgn = w["dir"].values
        streak = 1
        max_s = 0
        for i in range(1, len(sgn)):
            if sgn[i] == sgn[i - 1] and sgn[i] != 0:
                streak += 1
            else:
                streak = 1
            max_s = max(max_s, streak if sgn[i] > 0 else 0)
        fwd_4h = h4["close"].get(ts + pd.Timedelta("4h"), np.nan)
        fwd_12h = h4["close"].get(ts + pd.Timedelta("12h"), np.nan)
        rows.append({
            "ts": ts,
            "up_ratio": len(ups) / max(len(w), 1),          # 涨根占比
            "up_sum": round(up_sum, 4),                      # 涨的15m累计涨幅%
            "dn_sum": round(dn_sum, 4),                      # 跌的15m累计跌幅%(负数)
            "vol_up_ratio": round(vol_up / max(vol_dn, 1e-9), 4),  # 涨均量/跌均量
            "range_pct": round((w["high"].max() - w["low"].min()) / r["open"] * 100, 4),  # 窗口振幅%
            "last_dir": 1 if w["dir"].iloc[-1] > 0 else (-1 if w["dir"].iloc[-1] < 0 else 0),  # 末根方向
            "max_streak": max_s,                             # 最大连涨根数
            "mom_4h": round((r["close"] / r["open"] - 1) * 100, 4),   # 4h涨跌幅%
            "vol_ratio_4h": round(r["volume"] / max(vol_ma.get(ts, np.nan), 1e-9), 4),
            "fwd_4h": 1 if not np.isnan(fwd_4h) and fwd_4h > r["close"] else 0,
            "fwd_12h": 1 if not np.isnan(fwd_12h) and fwd_12h > r["close"] else 0,
            "ret_4h": round((fwd_4h / r["close"] - 1) * 100, 4) if not np.isnan(fwd_4h) else np.nan,
        })
    df = pd.DataFrame(rows)
    return df.set_index("ts") if len(rows) else pd.DataFrame()


def cond_prob(df, col, lo, hi, label="fwd_4h"):
    """条件概率: lo<=col<hi 时未来涨的概率"""
    m = (df[col] >= lo) & (df[col] < hi)
    if m.sum() < 30:
        return None
    p = df[m][label].mean()
    n = int(m.sum())
    se = 1.96 * np.sqrt(p * (1 - p) / n)  # 95% Wald 区间
    return p, n, se


def ic(df, col, label="fwd_4h"):
    """Spearman 秩相关: 特征与未来方向的预测力"""
    from scipy.stats import spearmanr
    r, p = spearmanr(df[col], df[label])
    return r, p


def backtest(df, col, lo, hi, side, cost=0.001):
    """按条件开仓的回测: 每次持4h, 交易成本0.1%, 输出test期表现"""
    m = (df[col] >= lo) & (df[col] < hi)
    sub = df[m].copy()
    sub["ret"] = sub["ret_4h"] * side - cost
    n = len(sub)
    if n == 0:
        return None
    win = (sub["ret"] > 0).mean()
    avg = sub["ret"].mean()
    cum = (1 + sub["ret"] / 100).prod() - 1
    return {"n": n, "win": round(win * 100, 1), "avg%": round(avg, 3),
            "cum%": round(cum * 100, 1), "side": side}


def main():
    print("拉取数据(Coinbase 5年: 1h重采样4h + 15m)...")
    h1 = fetch_cb_interval(3600, days=1825)
    h4 = resample_4h(h1)
    m15 = fetch_cb_interval(900, days=1825)
    print(f"1h {len(h1)}根 → 4h {len(h4)}根 ({h4.index[0]:%Y-%m-%d} 起) | 15m {len(m15)}根")

    df = build_features(h4, m15)
    df.to_csv("analysis/features.csv")  # 缓存供 train_model.py 复用
    print(f"窗口特征 {len(df)} 个 (4h窗口), 已缓存 analysis/features.csv")
    if len(df) < 500:
        print("样本不足, 退出"); return
    # 时间切分: 前2/3训练找规律, 后1/3验证
    cut = df.index[int(len(df) * 2 / 3)]
    train, test = df[df.index < cut], df[df.index >= cut]
    print(f"train {len(train)} | test {len(test)} | 切分点 {cut:%Y-%m-%d}")

    base = test["fwd_4h"].mean()
    print(f"\n基线: test期未来4h上涨概率 {base*100:.1f}% (无脑做多胜率)")

    print("\n=== 单特征IC(全样本) ===")
    cols = ["up_ratio", "up_sum", "dn_sum", "vol_up_ratio", "range_pct", "last_dir", "max_streak", "mom_4h", "vol_ratio_4h"]
    for c in cols:
        r, p = ic(df, c)
        print(f"  {c:14s} IC={r:+.3f} p={p:.3f}")

    print("\n=== 关键特征条件概率(train期) ===")
    for c, bins in [("up_ratio", [(0.3, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.01)]),
                    ("mom_4h", [(-10, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, 10)]),
                    ("vol_up_ratio", [(0, 0.6), (0.6, 1.2), (1.2, 2.5), (2.5, 99)]),
                    ("range_pct", [(0, 1), (1, 2), (2, 3.5), (3.5, 99)])]:
        print(f"  -- {c} → 未来4h上涨概率 --")
        for lo, hi in bins:
            r = cond_prob(train, c, lo, hi)
            if r:
                print(f"    [{lo:>4},{hi:>4}) n={r[1]:4d}  P(涨)={r[0]*100:5.1f}% ±{r[2]*100:.1f}%")

    print("\n=== 时间切分回测(test期, 规则来自train期发现) ===")
    print("基线-无脑做多: ", end="")
    b = backtest(test.assign(_=0), "_", 0, 1, 1)
    print(f"n={b['n']} 胜率{b['win']}% 每笔{b['avg%']}% 累计{b['cum%']}%")
    rules = [
        ("涨根占比高(≥0.7)", "up_ratio", 0.7, 1.01, 1),
        ("涨根占比低(<0.4)", "up_ratio", 0, 0.4, -1),
        ("4h大涨(≥2%)", "mom_4h", 2, 10, 1),
        ("4h大跌(≤-2%)", "mom_4h", -10, -2, -1),
        ("涨的量明显多(vol_up≥1.2)", "vol_up_ratio", 1.2, 99, 1),
        ("跌的量明显多(vol_up<0.8)", "vol_up_ratio", 0, 0.8, -1),
        ("高振幅(≥3.5%)", "range_pct", 3.5, 99, 1),
        ("低振幅(<1%)", "range_pct", 0, 1, -1),
    ]
    for name, c, lo, hi, side in rules:
        r = backtest(test, c, lo, hi, side)
        if r:
            print(f"  {name:24s} n={r['n']:4d} 胜率{r['win']}% 每笔{r['avg%']}% 累计{r['cum%']}%")
        else:
            print(f"  {name:24s} 样本不足")


if __name__ == "__main__":
    main()
