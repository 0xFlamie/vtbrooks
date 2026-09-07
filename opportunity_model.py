"""Brooks 候选机会、可解释特征与三重障碍标签。纯函数，供离线训练和线上推理共用。"""
import numpy as np
import pandas as pd

import vt_vote_bot as bot

FEATURES = ["body_ratio", "upper_wick", "lower_wick", "close_pos", "overlap",
            "pullback_atr", "range20_atr", "distance_edge_atr", "atr_pct", "vol_ratio"]


def triple_barrier(df, index, direction, atr, horizon, target_atr=1.0, stop_atr=1.0):
    """返回 1=先止盈、-1=先止损、0=到期/同根同时触及；只观察 index 后的K线。"""
    entry = float(df["close"].iloc[index])
    upper = entry + target_atr * atr
    lower = entry - stop_atr * atr
    for j in range(index + 1, min(index + horizon + 1, len(df))):
        hit_upper = float(df["high"].iloc[j]) >= upper
        hit_lower = float(df["low"].iloc[j]) <= lower
        if hit_upper and hit_lower:
            return 0
        if direction == 1 and hit_upper or direction == -1 and hit_lower:
            return 1
        if direction == 1 and hit_lower or direction == -1 and hit_upper:
            return -1
    return 0


def _breakout_retest(df, index, atr):
    if index < 22:
        return []
    high20 = float(df["high"].iloc[index - 21:index - 1].max())
    low20 = float(df["low"].iloc[index - 21:index - 1].min())
    prev_close = float(df["close"].iloc[index - 1])
    row = df.iloc[index]
    if prev_close > high20 and row["low"] <= high20 + 0.15 * atr and row["close"] > high20:
        return [("breakout_retest", 1)]
    if prev_close < low20 and row["high"] >= low20 - 0.15 * atr and row["close"] < low20:
        return [("breakout_retest", -1)]
    return []


def opportunities(df, index, lookback=120):
    """只在已收线数据上识别候选；同一根可出现多个不同 setup。"""
    if index < lookback - 1:
        return []
    window = df.iloc[index - lookback + 1:index + 1]
    atr = float((window["high"] - window["low"]).tail(14).mean())
    if not np.isfinite(atr) or atr <= 0:
        return []
    brooks = bot.brooks_analyze(window)
    found = []
    mapping = {"BROOKS_双腿回调H2": "second_entry", "BROOKS_双腿回调L2": "second_entry"}
    for name, direction in brooks["votes"]:
        if name in mapping:
            found.append((mapping[name], int(direction)))
        if name == "BROOKS_信号K线" and brooks["state"] != "range":
            found.append(("trend_continuation", int(direction)))
    high20 = float(window["high"].iloc[-21:-1].max())
    low20 = float(window["low"].iloc[-21:-1].min())
    current = window.iloc[-1]
    if current["high"] > high20 and current["close"] < high20:
        found.append(("failed_breakout", -1))
    if current["low"] < low20 and current["close"] > low20:
        found.append(("failed_breakout", 1))
    found.extend(_breakout_retest(df, index, atr))
    return list(dict.fromkeys(found))


def feature_row(df, index, setup, direction):
    window = df.iloc[index - 119:index + 1]
    row = window.iloc[-1]
    ranges = (window["high"] - window["low"]).replace(0, np.nan)
    atr = float(ranges.tail(14).mean())
    body = float(row["close"] - row["open"])
    rng = max(float(row["high"] - row["low"]), 1e-9)
    high20, low20 = float(window["high"].tail(20).max()), float(window["low"].tail(20).min())
    deep = bot.brooks_deep_analyze(window)
    pullback = ((high20 - row["close"]) if direction == 1 else (row["close"] - low20)) / atr
    edge = min(abs(row["close"] - high20), abs(row["close"] - low20)) / atr
    volume_mean = float(window["volume"].iloc[-21:-1].mean())
    return {"time": str(df.index[index]), "setup": setup, "direction": direction, "entry": float(row["close"]),
            "atr": atr, "body_ratio": body / rng, "upper_wick": (row["high"] - max(row["open"], row["close"])) / rng,
            "lower_wick": (min(row["open"], row["close"]) - row["low"]) / rng,
            "close_pos": (row["close"] - row["low"]) / rng, "overlap": deep["overlap"],
            "pullback_atr": pullback, "range20_atr": (high20 - low20) / atr,
            "distance_edge_atr": edge, "atr_pct": atr / row["close"] * 100,
            "vol_ratio": float(row["volume"]) / max(volume_mean, 1e-9)}


def build_dataset(df, horizon):
    rows = []
    last_seen = {}
    for index in range(119, len(df) - horizon):
        for setup, direction in opportunities(df, index):
            key = (setup, direction)
            if index - last_seen.get(key, -100) < 5:
                continue
            last_seen[key] = index
            row = feature_row(df, index, setup, direction)
            row["label"] = triple_barrier(df, index, direction, row["atr"], horizon)
            rows.append(row)
    return pd.DataFrame(rows)
