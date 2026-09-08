"""固定事件规则、时间隔离和独立校准的滚动研究；不写生产模型。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brooks_event_research import (
    MODEL_FEATURES, RULE_VERSION, SETUPS, build_event_dataset, non_overlapping, purged_window,
)

THRESHOLD = .60


def ev_interval(trades):
    """按月份整块重采样，避免将同一行情的多笔交易当独立样本。"""
    if trades.empty:
        return None
    work = trades.assign(month=pd.to_datetime(trades.signal_time).dt.to_period("M"))
    blocks = work.groupby("month").net_return_pct.agg(["sum", "count"])
    if len(blocks) < 6:
        return None
    draw = np.random.default_rng(42).integers(0, len(blocks), size=(2000, len(blocks)))
    means = blocks["sum"].to_numpy()[draw].sum(axis=1) / blocks["count"].to_numpy()[draw].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def trade_summary(events):
    trades = non_overlapping(events)
    if trades.empty:
        return {"n": 0}
    return {"n": len(trades), "target_rate": float(trades.target_hit.mean()),
            "net_win_rate": float((trades.net_return_pct > 0).mean()),
            "net_ev_pct": float(trades.net_return_pct.mean()), "ev_95pct": ev_interval(trades),
            "outcomes": trades.outcome.value_counts().to_dict()}


def logit(probability):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def train_calibrated(train, calibration):
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=4, min_samples_leaf=20,
                                           learning_rate=.05, l2_regularization=1.0,
                                           early_stopping=False, random_state=42)
    model.fit(train[MODEL_FEATURES], train.target_hit)
    raw = model.predict_proba(calibration[MODEL_FEATURES])[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibrator.fit(logit(raw), calibration.target_hit)
    return model, calibrator


def evaluate_fold(events, setup, boundaries, number):
    train_end, calibration_end, test_end = boundaries
    group = events[events.setup == setup]
    train = purged_window(group, 0, train_end)
    calibration = purged_window(group, train_end, calibration_end)
    test = purged_window(group, calibration_end, test_end)
    result = {"fold": number, "train_n": len(train), "calibration_n": len(calibration),
              "test_n": len(test), "baseline": trade_summary(test)}
    if (len(train) < 100 or len(calibration) < 40 or len(test) < 30 or
            train.target_hit.nunique() < 2 or calibration.target_hit.nunique() < 2):
        result["status"] = "insufficient_samples"
        return result, test.iloc[:0]
    model, calibrator = train_calibrated(train, calibration)
    raw = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    probability = calibrator.predict_proba(logit(raw))[:, 1]
    prior = float(calibration.target_hit.mean())
    chosen = test.loc[probability >= THRESHOLD].copy()
    chosen["probability"] = probability[probability >= THRESHOLD]
    result.update({"status": "evaluated", "auc": float(roc_auc_score(test.target_hit, probability))
                   if test.target_hit.nunique() == 2 else None,
                   "brier_raw": float(brier_score_loss(test.target_hit, raw)),
                   "brier_calibrated": float(brier_score_loss(test.target_hit, probability)),
                   "brier_prior": float(brier_score_loss(test.target_hit, np.full(len(test), prior))),
                   "candidate_coverage": float((probability >= THRESHOLD).mean()),
                   "probability_max": float(probability.max()), "selected": trade_summary(chosen)})
    return result, chosen


def run(frame, cost_pct=.10, slippage_pct=.04):
    events = build_event_dataset(frame, cost_pct, slippage_pct)
    if events.empty:
        raise ValueError("没有满足固定规则的事件")
    folds = [tuple(len(frame) * part // 100 for part in ends)
             for ends in ((40, 50, 65), (55, 65, 80), (70, 80, 100))]
    result = {"rules": RULE_VERSION, "rows": len(frame), "from": str(frame.index[0]),
              "to": str(frame.index[-1]), "event_count": len(events),
              "state_counts": events.state.value_counts().to_dict(),
              "cost_pct": cost_pct, "slippage_pct": slippage_pct,
              "threshold_fixed_before_run": THRESHOLD, "fold_bar_boundaries": folds,
              "research_only": True, "untouched_holdout": False, "setups": {}}
    for setup in SETUPS:
        summaries, selected = [], []
        for number, boundaries in enumerate(folds, start=1):
            summary, chosen = evaluate_fold(events, setup, boundaries, number)
            summaries.append(summary)
            selected.append(chosen)
        pooled = pd.concat(selected)
        result["setups"][setup] = {"candidate_n": int((events.setup == setup).sum()),
                                    "folds": summaries, "selected_oos": trade_summary(pooled)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="analysis/tv_indicators/data/okx_4h.csv")
    parser.add_argument("--cost-pct", type=float, default=.10)
    parser.add_argument("--slippage-pct", type=float, default=.04)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, parse_dates=["ts"]).rename(columns={"vol": "volume"}).set_index("ts")
    result = run(frame, args.cost_pct, args.slippage_pct)
    result["source_sha256"] = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
