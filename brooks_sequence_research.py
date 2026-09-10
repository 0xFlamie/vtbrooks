"""趋势回调内的尝试序列研究；不是完整Brooks实现，不产生交易指令。"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from brooks_event_research import event_context

VERSION = "brooks-sequence-v1"
LOOKBACK = 120
ANCHOR_WINDOW = 20
MAX_EPISODE_BARS = 20
STEPS = {"4h": pd.Timedelta(hours=4), "15m": pd.Timedelta(minutes=15)}


@dataclass(frozen=True)
class Episode:
    start: int
    direction: int
    anchor: float
    floor: float
    count: int = 0
    armed: bool = True
    first: int | None = None
    rearm: int | None = None


def record(state, index, kind, reference=None):
    return {"kind": kind, "index": index, "episode_start": state.start, "direction": state.direction,
            "count": state.count, "first_attempt": state.first, "second_pullback": state.rearm,
            "anchor": state.direction * state.anchor, "invalidation": state.direction * state.floor,
            "signal_index": index - 1 if kind == "attempt" else None,
            "trigger_reference": state.direction * reference if reference is not None else None}


def advance(state, index, bar, previous, context):
    """价格已按方向镜像成多头坐标；结束条件优先于尝试计数。"""
    high, low, close = bar
    previous_high, previous_low, _ = previous
    reason = next((name for condition, name in (
        (close < state.floor, "invalidated"), (context == -state.direction, "context_reversed"),
        (index - state.start >= MAX_EPISODE_BARS, "expired"),
        (high > previous_high and low < previous_low, "ambiguous_outside"),
        (close > state.anchor, "trend_resumed")) if condition), None)
    if reason:
        return None, [record(state, index, reason)]
    if state.count >= 2:
        return state, []
    if state.count == 1 and not state.armed and high < previous_high:
        updated = replace(state, armed=True, rearm=index)
        return updated, [record(updated, index, "second_pullback")]
    if state.armed and high > previous_high:
        updated = replace(state, count=state.count + 1, armed=False,
                          first=index if state.count == 0 else state.first)
        return updated, [record(updated, index, "attempt", previous_high)]
    return state, []


def validate(frame, timeframe):
    if timeframe not in STEPS:
        raise ValueError("只支持4h/15m")
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < LOOKBACK:
        raise ValueError("需要至少120根带时间的K线")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("K线时间无效、重复或未排序")
    step = STEPS[timeframe]
    if not frame.index.equals(frame.index.floor(step)) or frame.index[-1].tzinfo is None:
        raise ValueError("K线须为带时区的周期开盘时间网格")
    if frame.index[-1] + step > pd.Timestamp.now(tz="UTC"):
        raise ValueError("含未收线K线")
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("价格或成交量无效")
    if ((frame.high < frame[["open", "low", "close"]].max(axis=1)) |
            (frame.low > frame[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("OHLC关系无效")


def context_and_quality(frame, timeframe):
    gaps = frame.index.to_series().diff().ne(STEPS[timeframe]).to_numpy()
    starts = np.flatnonzero(gaps)
    context, clean = np.zeros(len(frame), dtype=int), np.zeros(len(frame), dtype=bool)
    for start, end in zip(starts, np.r_[starts[1:], len(frame)]):
        segment = frame.iloc[start:end]
        context[start:end] = event_context(segment).trend_direction.to_numpy(int)
        good = (segment.high > segment.low) & (segment.volume > 0)
        clean[start:end] = good.rolling(LOOKBACK).sum().eq(LOOKBACK).to_numpy()
    return context, clean, gaps


def normalize(bar, direction):
    high, low, close = bar
    return (high, low, close) if direction == 1 else (-low, -high, -close)


def start_episode(bars, index, direction):
    high, low, _ = normalize(bars[index], direction)
    previous_high, previous_low, _ = normalize(bars[index - 1], direction)
    if low >= previous_low or high > previous_high:
        return None
    history = bars[index - ANCHOR_WINDOW:index]
    anchor = float(history[:, 0].max()) if direction == 1 else -float(history[:, 1].min())
    floor = float(history[:, 1].min()) if direction == 1 else -float(history[:, 0].max())
    if direction * bars[index, 2] < floor:
        return None
    return Episode(index, direction, anchor, floor)


def scan_sequences(frame, timeframe):
    validate(frame, timeframe)
    context, clean, gaps = context_and_quality(frame, timeframe)
    bars = frame[["high", "low", "close"]].to_numpy(float)
    state, events = None, []
    for index in range(1, len(frame)):
        if not clean[index] or gaps[index]:
            if state:
                events.append(record(state, index, "gap_reset" if gaps[index] else "quality_reset"))
            state = None
            continue
        if state:
            side = state.direction
            state, emitted = advance(state, index, normalize(bars[index], side),
                                     normalize(bars[index - 1], side), int(context[index]))
            events.extend(emitted)
        elif context[index]:
            state = start_episode(bars, index, int(context[index]))
            if state:
                events.append(record(state, index, "pullback_started"))
    for event in events:
        event["episode_id"] = f"{timeframe}:{frame.index[event['episode_start']].isoformat()}:{event['direction']}"
        event["bar_time"] = frame.index[event["index"]].isoformat()
        event["available_at"] = (frame.index[event["index"]] + STEPS[timeframe]).isoformat()
        event["timeframe"] = timeframe
        event["version"] = VERSION
    return events
