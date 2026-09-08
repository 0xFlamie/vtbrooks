"""按 Brooks 形态训练 TP-before-SL 概率模型；未达门槛时不输出生产模型。"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, ".")
from opportunity_model import FEATURES, build_dataset


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["ts"]).rename(columns={"vol": "volume"}).set_index("ts")
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def report_buckets(probability, labels):
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        mask = probability >= threshold
        wins = labels[mask]
        print(f"p>={threshold:.2f}: n={len(wins):4d} 覆盖={mask.mean():5.1%} 胜率={wins.mean():5.1%}" if len(wins) else f"p>={threshold:.2f}: n=0")


def report_setups(frame, probability, labels, cost_pct=.10):
    for setup in sorted(frame["setup"].unique()):
        mask = frame["setup"].to_numpy() == setup
        if mask.sum() < 30:
            continue
        outcomes = np.where(labels[mask] == 1, 1, -1)
        returns = outcomes * frame.loc[mask, "atr_pct"].to_numpy() - cost_pct
        print(f"  {setup:20s} n={mask.sum():4d} 胜率={labels[mask].mean():5.1%} 净EV={returns.mean():+.3f}%")


def report_conditions(frame, labels, cost_pct=.10):
    location = frame["context_location"].to_numpy()
    direction = frame["direction"].to_numpy()
    conditions = {
        "4h同向": frame["context_alignment"].to_numpy() == 1,
        "4h震荡": frame["context_alignment"].to_numpy() == 0,
        "4h逆向": frame["context_alignment"].to_numpy() == -1,
        "4h结构边缘": ((direction == 1) & (location <= .25)) | ((direction == -1) & (location >= .75)),
    }
    for name, mask in conditions.items():
        if mask.sum() < 100:
            continue
        outcomes = np.where(labels[mask] == 1, 1, -1)
        returns = outcomes * frame.loc[mask, "atr_pct"].to_numpy() - cost_pct
        print(f"  条件 {name:10s} n={mask.sum():4d} 胜率={labels[mask].mean():5.1%} 净EV={returns.mean():+.3f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="analysis/tv_indicators/data/okx_4h.csv")
    parser.add_argument("--timeframe", choices=("4h", "15m"), default="4h")
    args = parser.parse_args()
    horizon = 3 if args.timeframe == "4h" else 16
    samples = build_dataset(load_csv(args.csv), horizon)
    decided = samples[samples["label"] != 0].copy()
    print(f"候选={len(samples)} 决胜={len(decided)} 无行情/歧义={len(samples)-len(decided)}")
    if len(decided) < 300:
        raise SystemExit("样本不足300，不训练")
    split1, split2 = int(len(decided) * .6), int(len(decided) * .8)
    columns = ["setup", "direction"] + FEATURES
    x, y = decided[columns], (decided["label"] == 1).astype(int)
    prep = ColumnTransformer([("setup", OneHotEncoder(handle_unknown="ignore"), ["setup"]),
                              ("numeric", "passthrough", ["direction"] + FEATURES)])
    model = make_pipeline(prep, HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=8,
                                                               min_samples_leaf=40, learning_rate=.05, random_state=42))
    model.fit(x.iloc[:split1], y.iloc[:split1])
    for name, sl in (("valid", slice(split1, split2)), ("test", slice(split2, None))):
        probability = model.predict_proba(x.iloc[sl])[:, 1]
        labels = y.iloc[sl].to_numpy()
        print(f"\n{name}: AUC={roc_auc_score(labels, probability):.3f} Brier={brier_score_loss(labels, probability):.4f}")
        report_buckets(probability, labels)
        report_setups(decided.iloc[sl], probability, labels)
        report_conditions(decided.iloc[sl], labels)


if __name__ == "__main__":
    main()
