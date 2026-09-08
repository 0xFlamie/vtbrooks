"""4H 市场路径标签与原始价格行为特征。纯函数，供离线训练和线上推理共用。"""
import numpy as np
import pandas as pd


FEATURES = [
    "return_1", "return_3", "return_6", "body_ratio", "upper_wick", "lower_wick",
    "close_pos", "range_atr", "atr_pct", "volume_ratio", "overlap_5",
    "range_location", "distance_high_atr", "distance_low_atr", "ema_distance_atr",
    "ema_slope_atr", "breakout_high_atr", "breakout_low_atr",
]


def path_label(df, index, atr, horizon=3, barrier_atr=1.0):
    """返回 1=先触及上障碍、-1=先触及下障碍、0=超时或同根歧义。"""
    entry = float(df["close"].iloc[index])
    upper = entry + atr * barrier_atr
    lower = entry - atr * barrier_atr
    for offset in range(index + 1, min(index + horizon + 1, len(df))):
        hit_upper = float(df["high"].iloc[offset]) >= upper
        hit_lower = float(df["low"].iloc[offset]) <= lower
        if hit_upper and hit_lower:
            return 0
        if hit_upper:
            return 1
        if hit_lower:
            return -1
    return 0


def _true_range(frame):
    previous = frame["close"].shift()
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous).abs(),
        (frame["low"] - previous).abs(),
    ], axis=1).max(axis=1)


def feature_frame(df):
    """仅使用每行及其历史数据生成特征，输出索引与输入一致。"""
    frame = df.astype(float)
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    atr = _true_range(frame).rolling(14).mean()
    high20 = frame["high"].rolling(20).max()
    low20 = frame["low"].rolling(20).min()
    previous_high20 = frame["high"].shift().rolling(20).max()
    previous_low20 = frame["low"].shift().rolling(20).min()
    ema20 = frame["close"].ewm(span=20, adjust=False).mean()
    previous_high = frame["high"].shift()
    previous_low = frame["low"].shift()
    overlap = (np.minimum(frame["high"], previous_high) - np.maximum(frame["low"], previous_low)).clip(lower=0)
    result = pd.DataFrame(index=frame.index)
    result["return_1"] = frame["close"].pct_change() * 100
    result["return_3"] = frame["close"].pct_change(3) * 100
    result["return_6"] = frame["close"].pct_change(6) * 100
    result["body_ratio"] = (frame["close"] - frame["open"]) / candle_range
    result["upper_wick"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / candle_range
    result["lower_wick"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / candle_range
    result["close_pos"] = (frame["close"] - frame["low"]) / candle_range
    result["range_atr"] = candle_range / atr
    result["atr_pct"] = atr / frame["close"] * 100
    result["volume_ratio"] = frame["volume"] / frame["volume"].shift().rolling(20).mean()
    result["overlap_5"] = (overlap / candle_range).rolling(5).mean()
    result["range_location"] = (frame["close"] - low20) / (high20 - low20).replace(0, np.nan)
    result["distance_high_atr"] = (high20 - frame["close"]) / atr
    result["distance_low_atr"] = (frame["close"] - low20) / atr
    result["ema_distance_atr"] = (frame["close"] - ema20) / atr
    result["ema_slope_atr"] = (ema20 - ema20.shift(3)) / atr
    result["breakout_high_atr"] = (frame["close"] - previous_high20) / atr
    result["breakout_low_atr"] = (previous_low20 - frame["close"]) / atr
    return result


def build_path_dataset(df, horizon=3, barrier_atr=1.0):
    features = feature_frame(df)
    atr = features["atr_pct"] * df["close"] / 100
    rows = features.copy()
    rows["label"] = [
        path_label(df, index, float(atr.iloc[index]), horizon, barrier_atr)
        if np.isfinite(atr.iloc[index]) and index + horizon < len(df) else np.nan
        for index in range(len(df))
    ]
    rows["forward_return_pct"] = df["close"].shift(-horizon) / df["close"] * 100 - 100
    return rows.dropna(subset=FEATURES + ["label", "forward_return_pct"])
