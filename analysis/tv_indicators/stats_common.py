"""回测公共框架：方向条件概率 / 平均收益 / 右尾依赖 / train-test 切分 / 扣费 EV。

约定（项目既定标准）：
- 信号在收线确认，入场用下一根 open，无未来函数
- 双边 taker 各 5bp，每笔共扣 10bp
- 前 2/3 train，后 1/3 test
- 基线 = 同段内全样本无条件方向概率（同样用 open[i+1] -> close[i+h] 度量，与信号收益口径一致）
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FEE_ROUNDTRIP_PCT = 0.10  # 双边 taker 各 5bp

# 每个周期对应的未来窗口（bar 数）
HORIZONS = {"4h": {"4h": 1, "12h": 3, "24h": 6},
            "1h": {"4h": 4, "12h": 12, "24h": 24}}


def load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"okx_{tf}.csv", index_col=0, parse_dates=True)
    return df.sort_index()


def _segment_stats(r_pct: pd.Series, base_win: float) -> dict:
    n = len(r_pct)
    if n == 0:
        return {"n": 0, "win": np.nan, "base": base_win, "mean": np.nan,
                "med": np.nan, "tail": np.nan, "ev": np.nan}
    win = (r_pct > 0).mean()
    total = r_pct.sum()
    k = max(1, n // 10)
    top = r_pct.nlargest(k).sum()
    tail = top / total if total > 0 else np.nan  # 总收益中来自最好10%交易的占比
    return {"n": n, "win": win, "base": base_win, "mean": r_pct.mean(),
            "med": r_pct.median(), "tail": tail, "ev": r_pct.mean() - FEE_ROUNDTRIP_PCT}


def forward_returns(df: pd.DataFrame, h: int, direction: int) -> pd.Series:
    """信号收线于 i，入场 open[i+1]，出场 close[i+h]，单位 %。direction: +1 多 / -1 空。"""
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-h)
    r = exit_ / entry - 1.0
    return (r * direction * 100.0).dropna()


def evaluate(df: pd.DataFrame, name: str, mask: pd.Series, direction: int,
             tf: str, rows: list) -> None:
    """对单个信号 × 全部 horizon × full/train/test 计算统计，追加到 rows。"""
    split = len(df) * 2 // 3
    for hor_label, h in HORIZONS[tf].items():
        r_all = forward_returns(df, h, direction)
        # 基线：各段内全样本无条件胜率（direction 方向收益 > 0）
        for seg in ("full", "train", "test"):
            if seg == "train":
                seg_idx = df.index[:split]
            elif seg == "test":
                seg_idx = df.index[split:]
            else:
                seg_idx = df.index
            seg_r = r_all.loc[r_all.index.intersection(seg_idx)]
            base_win = (seg_r > 0).mean() if len(seg_r) else np.nan
            seg_sig_idx = mask.index[mask.fillna(False) & mask.index.isin(seg_idx)]
            r_sig = r_all.loc[r_all.index.intersection(seg_sig_idx)]
            s = _segment_stats(r_sig, base_win)
            rows.append({"signal": name, "dir": "long" if direction > 0 else "short",
                         "tf": tf, "hor": hor_label, "seg": seg, **s})


def print_report(title: str, rows: list) -> pd.DataFrame:
    rep = pd.DataFrame(rows)
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    for tf in rep["tf"].unique():
        sub = rep[rep["tf"] == tf]
        print(f"\n--- {tf} ---")
        hdr = f"{'signal':38s} {'dir':5s} {'hor':4s} {'seg':5s} {'n':>5s} {'win%':>6s} {'base%':>6s} {'Δpp':>6s} {'mean%':>7s} {'med%':>7s} {'tailDep':>7s} {'EV%':>7s}"
        print(hdr)
        for _, r in sub.iterrows():
            delta = (r["win"] - r["base"]) * 100 if r["n"] else np.nan
            flag = " 样本不足" if 0 < r["n"] < 100 else ""
            print(f"{r['signal']:38s} {r['dir']:5s} {r['hor']:4s} {r['seg']:5s} "
                  f"{r['n']:5d} {r['win']*100:6.1f} {r['base']*100:6.1f} {delta:6.1f} "
                  f"{r['mean']:7.3f} {r['med']:7.3f} "
                  f"{r['tail'] if pd.notna(r['tail']) else float('nan'):7.2f} {r['ev']:7.3f}{flag}")
    # 紧凑汇总：full 段 + train/test 一致性
    print(f"\n--- 紧凑汇总 (n=全样本; 一致性=train/test Δpp 是否同号) ---")
    print(f"{'tf':3s} {'signal':38s} {'dir':5s} {'hor':4s} {'n':>5s} {'win/base%':>10s} {'Δpp':>6s} {'mean%':>7s} {'EV%':>7s} {'trainΔ':>6s} {'testΔ':>6s} {'一致':>4s}")
    for (tf, sig, d, hor), g in rep.groupby(["tf", "signal", "dir", "hor"], sort=False):
        f = g[g["seg"] == "full"].iloc[0]
        tr = g[g["seg"] == "train"].iloc[0]
        te = g[g["seg"] == "test"].iloc[0]
        d_tr = (tr["win"] - tr["base"]) * 100 if tr["n"] else np.nan
        d_te = (te["win"] - te["base"]) * 100 if te["n"] else np.nan
        ok = "Y" if pd.notna(d_tr) and pd.notna(d_te) and d_tr * d_te > 0 else "N"
        wb = f"{f['win']*100:.1f}/{f['base']*100:.1f}" if f["n"] else "-"
        delta = (f["win"] - f["base"]) * 100 if f["n"] else np.nan
        print(f"{tf:3s} {sig:38s} {d:5s} {hor:4s} {f['n']:5d} {wb:>10s} "
              f"{delta if pd.notna(delta) else float('nan'):6.1f} {f['mean']:7.3f} {f['ev']:7.3f} "
              f"{d_tr if pd.notna(d_tr) else float('nan'):6.1f} {d_te if pd.notna(d_te) else float('nan'):6.1f} {ok:>4s}")
    return rep
