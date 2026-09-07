"""公共工具: 数据加载 / 因果pivot / 前向收益 / 统计表 (analysis/tv_indicators 下脚本复用)
方法论约定(项目既定): 信号收线确认, 下一根 open 入场, 双边 taker 各 5bp (共 10bp)。
"""
import numpy as np
import pandas as pd

FEE_RT = 0.0010  # 双边 taker 共 10bp
# 各周期下 4h/12h/24h 对应的 K 线根数
HORIZONS = {"4h": {"4h": 1, "12h": 3, "24h": 6},
            "1h": {"4h": 4, "12h": 12, "24h": 24}}
MIN_N = 100  # 低于此样本数结论标注"样本不足"


def load(tf):
    df = pd.read_csv(f"analysis/tv_indicators/data/okx_{tf}.csv", index_col=0, parse_dates=True)
    return df


def fwd_ret(df, k):
    """收线 t 确认信号 → open[t+1] 入场 → 持有 k 根 → close[t+k] 离场"""
    return df["close"].shift(-k) / df["open"].shift(-1) - 1


def confirmed_pivots(df, L):
    """严格 pivot (ta.pivothigh/pivotlow 语义), 输出按确认时刻对齐:
    ph[j] = pivot 值, 当且仅当 pivot 位于 j-L (收线 j 时确认, 无未来函数)"""
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(L, n - L):
        if h[i] > h[i - L:i].max() and h[i] > h[i + 1:i + L + 1].max():
            ph[i + L] = h[i]
        if l[i] < l[i - L:i].min() and l[i] < l[i + 1:i + L + 1].min():
            pl[i + L] = l[i]
    return ph, pl


def tail_share(rets):
    """右尾依赖度: 收益最高 10% 样本的盈亏占全部样本总盈亏的比例 (>1 或占比极高=靠右尾)"""
    r = np.sort(np.asarray(rets, dtype=float))
    total = r.sum()
    if len(r) < 10 or total <= 0:
        return np.nan
    k = max(1, len(r) // 10)
    return r[-k:].sum() / total


def signal_table(df, tf, sig_idx, sig_dir, label, cut_frac=2 / 3):
    """sig_idx/sig_dir: 信号 bar 位置(整数索引)与方向(+1 多 / -1 空)。
    逐 horizon 输出: n / 方向胜率 vs 全样本无条件基线 / 平均收益 / 中位数 / 右尾依赖 /
    前2/3 vs 后1/3 一致性 / 扣 10bp 后 EV。"""
    sig_idx = np.asarray(sig_idx)
    sig_dir = np.asarray(sig_dir)
    cut_pos = int(len(df) * cut_frac)
    print(f"\n### {label}  (信号数 {len(sig_idx)})")
    if len(sig_idx) == 0:
        print("  无信号")
        return {}
    out = {}
    for hz, k in HORIZONS[tf].items():
        r = fwd_ret(df, k).values
        valid = sig_idx[sig_idx + k < len(df)]
        dmap = dict(zip(sig_idx, sig_dir))
        dirs = np.array([dmap[i] for i in valid])
        dr = dirs * r[valid]  # 方向对齐收益
        dr = dr[~np.isnan(dr)]
        valid = valid[:len(dr)]
        dirs = dirs[:len(dr)]
        if len(dr) == 0:
            continue
        # 基线: 同入场约定的无条件方向概率 (多=P(r>0), 空=P(r<0), 按信号方向加权)
        rv = r[~np.isnan(r)]
        base_long = (rv > 0).mean()
        base = base_long if (dirs == 1).all() else (1 - base_long) if (dirs == -1).all() else \
            ((dirs == 1) * base_long + (dirs == -1) * (1 - base_long)).mean()
        win = (dr > 0).mean()
        ev = dr.mean() - FEE_RT
        # 时间切分一致性: 两段胜率相对各自基线的方向 + 平均收益符号
        segs = {}
        for name, m in [("train", valid < cut_pos), ("test", valid >= cut_pos)]:
            if m.sum() >= 30:
                segs[name] = ((dr[m] > 0).mean(), dr[m].mean(), int(m.sum()))
        consistent = "n/a"
        if len(segs) == 2:
            w_ok = (segs["train"][0] > base) == (segs["test"][0] > base)
            r_ok = (segs["train"][1] > 0) == (segs["test"][1] > 0)
            consistent = "一致" if (w_ok and r_ok) else "不一致"
        n_flag = "" if len(dr) >= MIN_N else "  [样本不足]"
        ts = tail_share(dr)
        print(f"  {hz:>3} (持有{k:>2}根): n={len(dr):5d}{n_flag} | 胜率 {win*100:5.1f}% vs 基线 {base*100:4.1f}% "
              f"({(win-base)*100:+4.1f}pp) | 均 {dr.mean()*100:+6.3f}% 中位 {np.median(dr)*100:+6.3f}% | "
              f"右尾依赖 {ts:5.2f} | 扣费EV {ev*100:+6.3f}% {'正' if ev > 0 else '负'} | train/test {consistent}")
        for name, (w, m_, nn) in segs.items():
            print(f"        {name}: n={nn:5d} 胜率 {w*100:5.1f}% 均 {m_*100:+6.3f}%")
        out[hz] = dict(n=len(dr), win=win, base=base, mean=dr.mean(), ev=ev, consistent=consistent)
    return out
