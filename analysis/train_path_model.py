"""训练 4H→未来12小时路径模型；达不到样本外门槛时禁止输出生产模型。"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, ".")
from market_path_model import FEATURES, build_path_dataset


def load_csv(path):
    frame = pd.read_csv(path, parse_dates=["ts"]).rename(columns={"vol": "volume"}).set_index("ts")
    return frame[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def multiclass_brier(labels, probability, classes):
    truth = np.column_stack([labels == value for value in classes])
    return np.mean(np.sum((probability - truth) ** 2, axis=1))


def calibration_error(labels, prediction, confidence, bins=10):
    correct = prediction == labels
    error = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        selected = (confidence >= lower) & (confidence < lower + 1 / bins)
        if selected.any():
            error += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return error


def report_thresholds(frame, labels, prediction, confidence, cost_pct=0.10):
    for threshold in (0.45, 0.50, 0.55, 0.60):
        selected = confidence >= threshold
        directional = selected & (prediction != 0)
        if directional.sum() == 0:
            print(f"  p>={threshold:.2f}: 方向样本=0")
            continue
        hit = prediction[directional] == labels[directional]
        actual = labels[directional]
        direction = prediction[directional]
        timeout_return = frame.loc[directional, "forward_return_pct"].to_numpy() * direction
        atr_return = frame.loc[directional, "atr_pct"].to_numpy()
        returns = np.where(actual == direction, atr_return,
                           np.where(actual == -direction, -atr_return, timeout_return)) - cost_pct
        print(f"  p>={threshold:.2f}: n={directional.sum():4d} 覆盖={directional.mean():5.1%} "
              f"路径命中={hit.mean():5.1%} 净EV={returns.mean():+.3f}%")


def evaluate(name, frame, model):
    labels = frame["label"].astype(int).to_numpy()
    probability = model.predict_proba(frame[FEATURES])
    prediction = model.classes_[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    majority = pd.Series(labels).value_counts(normalize=True).max()
    print(f"\n{name}: n={len(frame)} Accuracy={accuracy_score(labels, prediction):.3f} "
          f"基线={majority:.3f} LogLoss={log_loss(labels, probability, labels=model.classes_):.3f} "
          f"Brier={multiclass_brier(labels, probability, model.classes_):.3f} "
          f"ECE={calibration_error(labels, prediction, confidence):.3f}")
    report_thresholds(frame, labels, prediction, confidence)


def new_model():
    return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=8, min_samples_leaf=80,
                                          learning_rate=.05, l2_regularization=1.0, random_state=42)


def walk_forward(samples):
    size = len(samples)
    for fold, train_end in enumerate((.55, .70, .85), start=1):
        first = int(size * train_end)
        last = min(int(size * (train_end + .15)), size)
        model = new_model()
        model.fit(samples.iloc[:first][FEATURES], samples.iloc[:first]["label"].astype(int))
        evaluate(f"walk-forward-{fold}", samples.iloc[first:last], model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="analysis/tv_indicators/data/okx_4h.csv")
    args = parser.parse_args()
    samples = build_path_dataset(load_csv(args.csv))
    split1, split2 = int(len(samples) * .6), int(len(samples) * .8)
    train, valid, test = samples.iloc[:split1], samples.iloc[split1:split2], samples.iloc[split2:]
    print(f"样本={len(samples)} train={len(train)} valid={len(valid)} test={len(test)} "
          f"标签={samples['label'].astype(int).value_counts(normalize=True).sort_index().to_dict()}")
    model = new_model()
    model.fit(train[FEATURES], train["label"].astype(int))
    evaluate("valid", valid, model)
    evaluate("test", test, model)
    walk_forward(samples)
    print("\n生产判定：本脚本只评估，不保存模型；需验证集与测试集同门槛、方向样本均>=100、命中>=58%、净EV>0。")


if __name__ == "__main__":
    main()
