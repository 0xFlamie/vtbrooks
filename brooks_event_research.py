"""Brooks 启发的可检验事件规则 v2；研究近似，非完整 Brooks 理论实现。"""
import numpy as np
import pandas as pd

from market_path_model import FEATURES, feature_frame

RULE_VERSION = "brooks-events-v2"
LOOKBACK = 120
SETUPS = ("trend_pullback", "breakout_retest", "range_failed_breakout")
MODEL_FEATURES = FEATURES + ["efficiency", "trend_direction", "direction"]


def validate_bars(frame, timeframe="4h"):
    if timeframe not in ("4h", "15m"):
        raise ValueError("研究周期只支持4h/15m")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("K线索引必须为 DatetimeIndex")
    if len(frame) < LOOKBACK + 3 or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("K线不足、重复或未按时间排序")
    values = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any():
        raise ValueError("K线存在无效价格或非有限值")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1)) | (frame.volume < 0)).any():
        raise ValueError("OHLC/量能关系无效")
    step = pd.Timedelta(hours=4) if timeframe == "4h" else pd.Timedelta(minutes=15)
    if not frame.index.to_series().diff().iloc[1:].eq(step).all():
        raise ValueError(f"研究输入必须为连续{timeframe.upper()} K线；缺口须先分段")


def event_context(frame):
    """状态取信号前一根收线，避免把信号K本身当成趋势环境。"""
    facts = feature_frame(frame).replace([np.inf, -np.inf], np.nan)
    travel = frame.close.diff().abs().rolling(10).sum().replace(0, np.nan)
    facts["efficiency"] = (frame.close - frame.close.shift(10)).abs() / travel
    previous = facts.shift()
    up = (previous.ema_slope_atr > .1) & (previous.ema_distance_atr > 0)
    down = (previous.ema_slope_atr < -.1) & (previous.ema_distance_atr < 0)
    directional = (previous.efficiency >= .3) & (previous.overlap_5 < .6)
    facts["trend_direction"] = np.select([directional & up, directional & down], [1, -1], default=0)
    facts["state"] = np.select(
        [facts.trend_direction != 0, (previous.efficiency < .3) & (previous.overlap_5 >= .4)],
        ["trend", "range"], default="transition")
    return facts


def event_masks(frame, facts):
    """固定三种假设：趋势回调后确认、突破次根回踩、区间边缘失败突破。"""
    high = frame.high.shift().rolling(20).max()
    low = frame.low.shift().rolling(20).min()
    old_high, old_low = high.shift(), low.shift()
    atr = facts.atr_pct * frame.close / 100
    previous = frame.shift()
    long_bar = (facts.body_ratio >= .5) & (facts.close_pos >= .75)
    short_bar = (facts.body_ratio <= -.5) & (facts.close_pos <= .25)
    long_pullback = (previous.close < previous.open) & (frame.close > previous.high)
    short_pullback = (previous.close > previous.open) & (frame.close < previous.low)
    return {
        ("trend_pullback", 1): (facts.trend_direction == 1) & long_pullback & long_bar,
        ("trend_pullback", -1): (facts.trend_direction == -1) & short_pullback & short_bar,
        ("breakout_retest", 1): (previous.close > old_high) & (frame.low <= old_high + .15 * atr)
        & (frame.low >= old_high - .5 * atr) & (frame.close > old_high) & long_bar,
        ("breakout_retest", -1): (previous.close < old_low) & (frame.high >= old_low - .15 * atr)
        & (frame.high <= old_low + .5 * atr) & (frame.close < old_low) & short_bar,
        ("range_failed_breakout", 1): (facts.state == "range") & (frame.low < low)
        & (frame.close > low) & (facts.lower_wick >= .4) & (facts.close_pos >= .6),
        ("range_failed_breakout", -1): (facts.state == "range") & (frame.high > high)
        & (frame.close < high) & (facts.upper_wick >= .4) & (facts.close_pos <= .4),
    }


def candidates(frame):
    facts = event_context(frame)
    masks = event_masks(frame, facts)
    rows = []
    for (setup, direction), mask in masks.items():
        previous_index = -LOOKBACK
        for index in np.flatnonzero(mask.to_numpy()):
            if index < LOOKBACK - 1 or index - previous_index < 5:
                continue
            row = facts.iloc[index]
            if not np.isfinite(row[FEATURES + ["efficiency"]].astype(float)).all():
                continue
            previous_index = index
            rows.append({**row.to_dict(), "signal_index": int(index), "signal_time": frame.index[index],
                         "setup": setup, "direction": direction,
                         "atr": float(row.atr_pct * frame.close.iloc[index] / 100)})
    columns = list(facts.columns) + ["signal_index", "signal_time", "setup", "direction", "atr"]
    return pd.DataFrame(rows, columns=columns).sort_values(["signal_index", "setup", "direction"]).reset_index(drop=True)


def trade_outcome(frame, index, direction, atr, horizon=3, cost_pct=.10, slippage_pct=.04):
    """次根open入场；同根双触记歧义并按止损下界计损，超时单独标注。"""
    if direction not in (-1, 1) or atr <= 0 or not np.isfinite(atr):
        raise ValueError("方向或ATR无效")
    if (index < 0 or horizon < 1 or index + horizon >= len(frame) or
            not np.isfinite([cost_pct, slippage_pct]).all() or min(cost_pct, slippage_pct) < 0):
        raise ValueError("窗口不完整或成本无效")
    entry = float(frame.open.iloc[index + 1])
    target, stop = entry + direction * atr, entry - direction * atr
    for offset in range(index + 1, index + horizon + 1):
        bar = frame.iloc[offset]
        status, exit_price = _bar_exit(bar, direction, target, stop)
        if status is not None:
            break
    else:
        status, exit_price = "timeout", float(frame.close.iloc[index + horizon])
    gross = direction * (exit_price / entry - 1) * 100
    return {"outcome": status, "target_hit": int(status == "target"), "entry": entry,
            "exit_index": offset, "label_end_index": index + horizon,
            "gross_return_pct": gross, "net_return_pct": gross - cost_pct - slippage_pct}


def _bar_exit(bar, direction, target, stop):
    if direction * (bar.open - stop) <= 0:
        return "stop_gap", float(bar.open)
    if direction * (bar.open - target) >= 0:
        return "target", target
    hit_target = bar.high >= target if direction == 1 else bar.low <= target
    hit_stop = bar.low <= stop if direction == 1 else bar.high >= stop
    if hit_target and hit_stop:
        return "ambiguous_stop", stop
    if hit_stop:
        return "stop", stop
    if hit_target:
        return "target", target
    return None, None


def build_event_dataset(frame, cost_pct=.10, slippage_pct=.04, timeframe="4h", horizon=3):
    validate_bars(frame, timeframe)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("持有窗口必须为正整数")
    events = candidates(frame)
    events = events[events.signal_index + horizon < len(frame)].copy()
    records = [trade_outcome(frame, int(row.signal_index), int(row.direction), float(row.atr),
                             horizon=horizon, cost_pct=cost_pct, slippage_pct=slippage_pct) for row in events.itertuples()]
    outcomes = pd.DataFrame(records, index=events.index)
    return pd.concat([events, outcomes], axis=1)


def purged_window(events, start, end):
    """按原始K线边界切分，不允许任何完整标签窗口跨越下一段。"""
    return events[(events.signal_index >= start) & (events.label_end_index < end)].copy()


def non_overlapping(events):
    """按已知信号时间顺序入场，持仓期间忽略新信号；不挑选未来赢家。"""
    kept, available = [], -1
    for index, row in events.sort_values(["signal_index", "setup", "direction"]).iterrows():
        if row.signal_index >= available:
            kept.append(index)
            available = row.exit_index
    return events.loc[kept]
