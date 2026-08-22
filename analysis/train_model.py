"""LightGBM 组合特征预测 4h 方向: 严格时间切分样本外验证(独立分析, 不改生产)

问题: 单因子 IC 全≈0 (feature_explore.py 结论), 这里验证非线性组合有无 edge。
方法: 9特征 → LightGBM → train/validation/test 时间切分 → 样本外 AUC/分桶胜率/策略模拟
基线: 无脑做多(约50%)。诚实标准: 样本外 AUC > 0.53 或分桶胜率单调且显著才值得继续。
运行: python3 analysis/train_model.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

FEATURES = ["up_ratio", "up_sum", "dn_sum", "vol_up_ratio", "range_pct",
            "last_dir", "max_streak", "mom_4h", "vol_ratio_4h"]


def main():
    df = pd.read_csv("analysis/features.csv", index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURES + ["fwd_4h", "ret_4h"])
    print(f"样本 {len(df)} ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})")

    X = df[FEATURES].astype(float)
    y = df["fwd_4h"].astype(int)
    # 严格时间切分: train 60% / validation 15% / test 25%
    n = len(df)
    i_val, i_test = int(n * 0.6), int(n * 0.75)
    X_tr, X_va, X_te = X.iloc[:i_val], X.iloc[i_val:i_test], X.iloc[i_test:]
    y_tr, y_va, y_te = y.iloc[:i_val], y.iloc[i_val:i_test], y.iloc[i_test:]
    print(f"train {len(X_tr)} | valid {len(X_va)} | test {len(X_te)} (test起 {df.index[i_test]:%Y-%m-%d})")

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=8,   # 小树防过拟合
        max_depth=4, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    p_te = model.predict_proba(X_te)[:, 1]
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    base_brier = brier_score_loss(y_te, np.full(len(y_te), y_te.mean()))
    print(f"\n样本外 AUC: {auc:.3f} (0.5=随机) | Brier: {brier:.4f} vs 基线 {base_brier:.4f}")

    # 特征重要性
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("特征重要性:", ", ".join(f"{k}={v}" for k, v in imp[:6]))

    # 概率分桶校准: 每桶实际上涨比例
    print("\n=== 概率分桶校准(test期) ===")
    for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        m = (p_te >= lo) & (p_te < hi)
        if m.sum() < 30:
            print(f"  [{lo:.1f},{hi:.1f}) n={int(m.sum()):3d}  样本不足")
            continue
        print(f"  [{lo:.1f},{hi:.1f}) n={int(m.sum()):3d}  实际上涨 {y_te[m].mean()*100:5.1f}%")

    # 策略模拟: 概率>0.55做多, <0.45做空, 含0.1%成本
    print("\n=== 策略模拟(test期, 阈值0.55/0.45, 成本0.1%) ===")
    rets = df["ret_4h"].iloc[i_test:].values
    for name, lo, hi in [("做多 p>0.55", 0.55, 2.0), ("做空 p<0.45", -2.0, 0.45), ("全仓做多(基线)", -2.0, 2.0)]:
        m = (p_te >= lo) & (p_te < hi) if lo > 0 else (p_te < hi)
        if m.sum() < 30:
            print(f"  {name:16s} 样本不足")
            continue
        side = 1 if lo > 0 else -1
        r = rets[m] * side - 0.1
        cum = (1 + r / 100).prod() - 1
        print(f"  {name:16s} n={int(m.sum()):3d} 胜率{(r > 0).mean()*100:5.1f}% 每笔{r.mean():+.3f}% 累计{cum*100:+.1f}%")


if __name__ == "__main__":
    main()
