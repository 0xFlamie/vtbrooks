"""公共回测统计工具：fwd 收益、基线、train/test 切分、扣费 EV。

约定：
- 信号在 bar t 收线确认，入场价 = open[t+1]，持有 h 根后以 close[t+h] 离场。
- fwd_ret[t, h] = close[t+h] / open[t+1] - 1（做多口径；做空取负）。
- 基线 = 全样本同口径无条件 P(fwd_ret > 0)。
- 扣费 EV = 平均收益 - FEE_RT（双边 taker 各 5bp，共 10bp）。
"""
import numpy as np
import pandas as pd

FEE_RT = 0.0010  # 双边 taker 共 10bp


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["open", "high", "low", "close", "vol"]].astype(float)


def fwd_returns(df, horizons):
    """返回 {h: Series}，index 与 df 对齐，值为 open[t+1] -> close[t+h] 的收益。"""
    o = df["open"].shift(-1)
    out = {}
    for h in horizons:
        out[h] = df["close"].shift(-h) / o - 1.0
    return out


def evaluate(name, sig_mask, fwd, df, horizons, side=1, train_frac=2 / 3):
    """对一组信号做全套统计。side=1 做多，side=-1 做空（收益取负，胜率=胜率口径反向）。"""
    rows = []
    sig_idx = np.flatnonzero(sig_mask.values)
    split_pos = int(len(df) * train_frac)
    for h in horizons:
        r = fwd[h]
        sig_r = r.iloc[sig_idx].dropna() * side
        base = r.dropna() * 1.0
        base_p = float((base > 0).mean()) if side == 1 else float((base * -1 > 0).mean())
        n = len(sig_r)
        if n == 0:
            rows.append(dict(signal=name, horizon=h, n=0))
            continue
        win = float((sig_r > 0).mean())
        mean = float(sig_r.mean())
        med = float(sig_r.median())
        skew = float(sig_r.skew())
        pos = sig_r[sig_r > 0]
        top10_share = (
            float(pos.nlargest(max(1, len(pos) // 10)).sum() / pos.sum()) if len(pos) and pos.sum() != 0 else np.nan
        )
        # train/test
        tr = sig_r[sig_r.index < df.index[split_pos]]
        te = sig_r[sig_r.index >= df.index[split_pos]]
        tr_win = float((tr > 0).mean()) if len(tr) else np.nan
        te_win = float((te > 0).mean()) if len(te) else np.nan
        tr_mean = float(tr.mean()) if len(tr) else np.nan
        te_mean = float(te.mean()) if len(te) else np.nan
        ev = mean - FEE_RT
        rows.append(dict(
            signal=name, horizon=h, n=n,
            win=win, base=base_p, dwin=win - base_p,
            mean=mean, med=med, skew=skew, top10=top10_share,
            n_tr=len(tr), tr_win=tr_win, tr_mean=tr_mean,
            n_te=len(te), te_win=te_win, te_mean=te_mean,
            consistent=(not np.isnan(tr_win)) and (not np.isnan(te_win))
            and ((tr_win - base_p) * (te_win - base_p) > 0),
            ev=ev, ev_pos=ev > 0,
        ))
    return pd.DataFrame(rows)


def fmt(res, hlabels):
    out = []
    for _, r in res.iterrows():
        if r.get("n", 0) == 0:
            out.append(f"{r['signal']} @{hlabels[r['horizon']]}: n=0")
            continue
        out.append(
            f"{r['signal']} @{hlabels[r['horizon']]}: n={r['n']} "
            f"win={r['win']:.1%} (base {r['base']:.1%}, {r['dwin']:+.1%}) "
            f"mean={r['mean']*1e4:+.1f}bp med={r['med']*1e4:+.1f}bp skew={r['skew']:.2f} "
            f"top10%贡献={r['top10']:.0%} | "
            f"train n={r['n_tr']} win={r['tr_win']:.1%}/{r['tr_mean']*1e4:+.1f}bp "
            f"test n={r['n_te']} win={r['te_win']:.1%}/{r['te_mean']*1e4:+.1f}bp "
            f"一致={r['consistent']} | 扣费EV={r['ev']*1e4:+.1f}bp {'正' if r['ev_pos'] else '负'}"
        )
    return "\n".join(out)
