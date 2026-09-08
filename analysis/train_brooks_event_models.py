"""按 Brooks 事件分别训练 4H TP-before-SL 模型，防止反转与顺势形态互相污染。"""
import argparse
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, ".")
from analysis.train_opportunity_model import load_csv
from opportunity_model import FEATURES, build_dataset


def new_model(sample_count):
    leaf_size = max(15, min(40, sample_count // 20))
    return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=6, min_samples_leaf=leaf_size,
                                          learning_rate=.05, l2_regularization=1.0, random_state=42)


def report(name, frame, model, cost_pct=.10):
    labels = (frame["label"] == 1).astype(int).to_numpy()
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    auc = roc_auc_score(labels, probability) if len(set(labels)) == 2 else float("nan")
    print(f"    {name}: n={len(frame):4d} 基准胜率={labels.mean():5.1%} "
          f"AUC={auc:.3f} Brier={brier_score_loss(labels, probability):.3f}")
    for threshold in (.50, .55, .60, .65):
        selected = probability >= threshold
        if not selected.any():
            continue
        chosen = frame.loc[selected]
        outcome = chosen["label"].to_numpy()
        direction = chosen["direction"].to_numpy()
        timeout = chosen["forward_return_pct"].to_numpy() * direction
        returns = np.where(outcome == 1, chosen["atr_pct"],
                           np.where(outcome == -1, -chosen["atr_pct"], timeout)) - cost_pct
        print(f"      p>={threshold:.2f}: n={selected.sum():3d} 覆盖={selected.mean():5.1%} "
              f"胜率={(outcome == 1).mean():5.1%} 净EV={returns.mean():+.3f}%")


def evaluate_setup(frame, setup):
    samples = frame[frame["setup"] == setup].copy()
    if len(samples) < 300:
        print(f"\n{setup}: 样本={len(samples)}，不足300，不训练")
        return
    first, second = int(len(samples) * .6), int(len(samples) * .8)
    train, valid, test = samples.iloc[:first], samples.iloc[first:second], samples.iloc[second:]
    model = new_model(len(train))
    model.fit(train[FEATURES], (train["label"] == 1).astype(int))
    print(f"\n{setup}: train={len(train)} valid={len(valid)} test={len(test)}")
    report("valid", valid, model)
    report("test ", test, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="analysis/tv_indicators/data/okx_4h.csv")
    args = parser.parse_args()
    samples = build_dataset(load_csv(args.csv), horizon=3)
    print(f"4H事件样本={len(samples)}，超时/歧义={(samples['label'] == 0).mean():.1%}")
    for setup in sorted(samples["setup"].unique()):
        evaluate_setup(samples, setup)
    print("\n生产判定：仅当同一事件 valid/test 均有>=100个高置信样本、胜率>=58%、净EV>0才允许保存。")


if __name__ == "__main__":
    main()
