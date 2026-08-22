"""第五轮: 弱信号合成强度分 / 新特征LightGBM / 振幅回归预测
前三轮发现方向弱信号(收盘位置/距极值/BTC联动/突破)各自 EV≈0——这轮看合成能否变可用
运行: python3 analysis/fifth_explore.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from feature_explore import fetch_okx_4h

DIR_FEATS = ["pos_close", "near_hi", "near_lo", "brk_up", "brk_dn", "mom_1d", "btc_ret"]


def main():
    print("拉取 ETH/BTC 4h 5年...", flush=True)
    h4 = fetch_okx_4h(days=1825)
    btc = fetch_okx_4h("BTC-USDT", days=1825)
    h4["ret"] = h4["close"].pct_change() * 100
    h4["fwd_up"] = (h4["close"].pct_change(-1) > 0).astype(int)
    h4["fwd_ret"] = h4["close"].pct_change(-1) * 100
    h4["fwd_amp12"] = (h4["high"].shift(-1).rolling(3).max().shift(-2) - h4["low"].shift(-1).rolling(3).min().shift(-2)) / h4["close"] * 100
    h4["btc_ret"] = btc["close"].pct_change().reindex(h4.index) * 100
    hi20 = h4["high"].rolling(20).max().shift(1)
    lo20 = h4["low"].rolling(20).min().shift(1)

    # 弱信号特征(全部从已确认弱信号而来)
    pos = (h4["close"] - h4["low"]) / (h4["high"] - h4["low"])
    h4["pos_close"] = pos
    h4["near_hi"] = (hi20 / h4["close"] - 1) * 100          # 距20期高点%
    h4["near_lo"] = (h4["close"] / lo20 - 1) * 100          # 距20期低点%
    h4["brk_up"] = (h4["close"] > hi20).astype(int)
    h4["brk_dn"] = (h4["close"] < lo20).astype(int)
    h4["mom_1d"] = (h4["close"] / h4["close"].shift(6) - 1) * 100
    h4 = h4.dropna(subset=DIR_FEATS + ["fwd_up", "fwd_ret", "fwd_amp12"]).copy()

    base = h4["fwd_up"].mean() * 100
    print(f"基线: 未来4h涨 {base:.1f}% | 样本 {len(h4)}")

    # A. 强度分合成(规则加权): 收盘位置±/距极值±/BTC联动±/突破±
    print("\n=== A. 弱信号合成强度分 → 未来4h方向 ===")
    h4["score"] = 0
    h4.loc[h4["pos_close"] > 0.6, "score"] += 1
    h4.loc[h4["pos_close"] < 0.4, "score"] -= 1
    h4.loc[h4["near_hi"] < 2, "score"] += 1
    h4.loc[h4["near_lo"] < 2, "score"] -= 1
    h4.loc[h4["btc_ret"] > 0.5, "score"] += 1
    h4.loc[h4["btc_ret"] < -0.5, "score"] -= 1
    h4["score"] += h4["brk_up"] - h4["brk_dn"]
    for s in range(-4, 5):
        m = h4["score"] == s
        if m.sum() < 200:
            continue
        up = h4["fwd_up"][m].mean() * 100
        ev = h4["fwd_ret"][m].mean() - 0.1  # 含0.1%成本
        print(f"  强度分 {s:>+2d}: n={int(m.sum()):4d} 未来涨 {up:.1f}% | EV(含成本) {ev:+.3f}%")

    # B. LightGBM 新特征集(方向弱信号) → 样本外AUC
    print("\n=== B. LightGBM 新特征集(方向弱信号) ===")
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        X = h4[DIR_FEATS].astype(float)
        y = h4["fwd_up"].astype(int)
        n = len(X)
        i_val, i_test = int(n * 0.6), int(n * 0.75)
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=8, max_depth=4,
                                   min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                                   reg_lambda=1.0, random_state=42, verbose=-1)
        model.fit(X.iloc[:i_val], y.iloc[:i_val],
                  eval_set=[(X.iloc[i_val:i_test], y.iloc[i_val:i_test])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        p = model.predict_proba(X.iloc[i_test:])[:, 1]
        auc = roc_auc_score(y.iloc[i_test:], p)
        print(f"  样本外 AUC: {auc:.3f} (0.5=随机, 用弱信号特征集)")
    except Exception as e:
        print(f"  LightGBM 失败: {e}")

    # C. 振幅回归: 前4h振幅/ATR/波动 → 未来12h振幅
    print("\n=== C. 振幅预测: 特征 → 未来12h振幅 ===")
    h4["atr_pct"] = (h4["high"] - h4["low"]).rolling(14).mean() / h4["close"] * 100
    h4["range_pct"] = (h4["high"] - h4["low"]) / h4["open"] * 100
    h4["abs_ret"] = h4["ret"].abs()
    g = h4["fwd_amp12"].mean()
    print(f"  全局平均 12h振幅 {g:.2f}%")
    # 简单基准: 前4h振幅
    for lo, hi in [(0, 1), (1, 2), (2, 3.5), (3.5, 99)]:
        m = (h4["range_pct"] >= lo) & (h4["range_pct"] < hi)
        if m.sum() > 100:
            lift = (h4["fwd_amp12"][m].mean() / g - 1) * 100
            print(f"    前4h振幅[{lo:>3},{hi:>3})% n={int(m.sum()):4d} 未来12h振幅 {h4['fwd_amp12'][m].mean():.2f}% (lift {lift:+.0f}%)")
    # 组合: LightGBM 回归
    try:
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error
        Xa = h4[["range_pct", "atr_pct", "abs_ret"]].astype(float)
        ya = h4["fwd_amp12"]
        n = len(Xa)
        i_test = int(n * 0.75)
        m1 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=100,
                               reg_lambda=1.0, random_state=42, verbose=-1)
        m1.fit(Xa.iloc[:i_test], ya.iloc[:i_test])
        pred = m1.predict(Xa.iloc[i_test:])
        actual = ya.iloc[i_test:].values
        base_err = mean_squared_error(actual, np.full(len(actual), actual.mean())) ** 0.5
        model_err = mean_squared_error(actual, pred) ** 0.5
        print(f"  振幅预测 RMSE: 模型 {model_err:.3f}% vs 简单均值基准 {base_err:.3f}%")
        # 分组验证: 预测高分位 vs 实际振幅
        q = pd.qcut(pred, 4, labels=False)
        for i in range(4):
            m = q == i
            print(f"    预测分位{i+1}: n={int(m.sum()):4d} 实际平均振幅 {actual[m].mean():.2f}%")
    except Exception as e:
        print(f"  振幅回归失败: {e}")


if __name__ == "__main__":
    main()
