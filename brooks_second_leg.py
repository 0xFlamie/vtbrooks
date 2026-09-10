"""强突破后首次短回调的盘中顺势恢复；单一研究近似，不接生产。"""
import math

import pandas as pd

from brooks_failed_range import validate_minutes
from brooks_intrabar import aggregate_context

POLICY = {"version": "strong-breakout-second-leg-v1", "prior_boundary_bars": 20,
          "breakout_range_atr": 1., "body_fraction": .6, "close_end_fraction": .25,
          "first_pullback_within_bars": 5, "resume_within_bars": 5,
          "breakout_is_closed_15m": True, "pullback_signal_bar_is_closed_15m": True,
          "entry_trigger": "completed_1m_close_beyond_previous_closed_15m_extreme",
          "invalidation": "minute_touches_breakout_opposite_extreme_before_or_with_signal",
          "ambiguous_current_outside": "cancel", "end_bar_restart": False,
          "atr": "14_TR_before_live_15m", "primary": "0.25/3", "horizon_minutes": 60,
          "future_claim": "sustained_raw_space_not_profit", "production": False}


def seed_breakout(bars, index, atr):
    if index < POLICY["prior_boundary_bars"] or not math.isfinite(atr) or atr <= 0:
        return None
    bar = bars.iloc[index]
    span = float(bar.high - bar.low)
    if span < atr * POLICY["breakout_range_atr"] or span <= 0:
        return None
    side = 1 if bar.close > bar.open else -1
    if side * (bar.close - bar.open) < POLICY["body_fraction"] * span:
        return None
    end_distance = bar.high - bar.close if side == 1 else bar.close - bar.low
    if end_distance > POLICY["close_end_fraction"] * span:
        return None
    history = bars.iloc[index - POLICY["prior_boundary_bars"]:index]
    boundary = float(history.high.max() if side == 1 else history.low.min())
    if side * (bar.close - boundary) <= 0:
        return None
    return {"breakout_index": index, "breakout_at": bars.index[index].isoformat(), "direction": side,
            "breakout_level": boundary, "invalidation": float(bar.low if side == 1 else bar.high),
            "pullback_index": None, "pullback_at": None}


def finish_bar(state, bars, index, atr):
    side, bar = state["direction"], bars.iloc[index]
    opposite = seed_breakout(bars, index, atr)
    if opposite and opposite["direction"] == -side:
        return None, "opposite_breakout"
    if state["pullback_index"] is None:
        previous = bars.iloc[index - 1]
        pulled = bar.low < previous.low if side == 1 else bar.high > previous.high
        if pulled:
            return {**state, "pullback_index": index, "pullback_at": bars.index[index].isoformat()}, "pullback_known"
        return (None, "no_pullback_expired") if index - state["breakout_index"] >= POLICY["first_pullback_within_bars"] else (state, "impulse")
    if index - state["pullback_index"] >= POLICY["resume_within_bars"]:
        return None, "resume_expired"
    return state, "pullback_waiting"


def observe(state, group, stamp, previous, atr):
    partial = None
    for offset, minute in enumerate(group.itertuples()):
        if minute.Index != stamp + pd.Timedelta(minutes=offset) or minute.volume <= 0:
            return None, "quality_reset"
        partial = ({"open": float(minute.open), "high": float(minute.high), "low": float(minute.low),
                    "close": float(minute.close), "volume": float(minute.volume)} if partial is None else
                   {**partial, "high": max(partial["high"], float(minute.high)), "low": min(partial["low"], float(minute.low)),
                    "close": float(minute.close), "volume": partial["volume"] + float(minute.volume)})
        side = state["direction"]
        worst = partial["low"] if side == 1 else partial["high"]
        if side * (worst - state["invalidation"]) <= 0:
            return None, "invalidated"
        if state["pullback_index"] is None:
            continue
        reference = float(previous.high if side == 1 else previous.low)
        if side * (partial["close"] - reference) <= 0:
            continue
        if partial["high"] > previous.high and partial["low"] < previous.low:
            return None, "ambiguous_outside"
        available = minute.Index + pd.Timedelta(minutes=1)
        return {**state, "family": "second_leg", "episode_id": f"second_leg:{state['breakout_at']}:{side}",
                "bar_start": stamp.isoformat(), "available_at": available.isoformat(), "lead_minutes": 14 - offset,
                "trigger_reference": reference, "observed_price": partial["close"], "partial_bar": partial,
                "structure_reference": state["invalidation"], "atr": float(atr)}, "signal"
    if len(group) < 15:
        return None, "pending"
    return None, "waiting"


def scan_second_leg(minutes, start, end):
    validate_minutes(minutes)
    if minutes.empty:
        return [], []
    bars, groups, _, clean, gaps, atr = aggregate_context(minutes)
    state, signals, events = None, [], []
    for index, (stamp, group) in enumerate(groups):
        if index == 0:
            continue
        was_active = state is not None
        if not clean[index - 1] or gaps[index]:
            state = None
        if state is not None:
            signal, reason = observe(state, group, stamp, bars.iloc[index - 1], float(atr.iloc[index]))
            if signal:
                if start <= pd.Timestamp(signal["available_at"]) < end:
                    signals.append(signal)
                state = None
            elif reason not in ("waiting", "pending"):
                state = None
            if start <= stamp < end:
                events.append({"bar_start": stamp.isoformat(), "reason": reason})
        complete = len(group) == 15 and group.index[-1] == stamp + pd.Timedelta(minutes=14)
        if not complete or not clean[index]:
            state = None
            continue
        if state is not None:
            state, reason = finish_bar(state, bars, index, float(atr.iloc[index]))
            if start <= stamp < end:
                events.append({"bar_start": stamp.isoformat(), "reason": reason})
        elif not was_active:
            state = seed_breakout(bars, index, float(atr.iloc[index]))
            if state is not None and start <= stamp < end:
                events.append({"bar_start": stamp.isoformat(), "reason": "breakout_started"})
    return signals, events
