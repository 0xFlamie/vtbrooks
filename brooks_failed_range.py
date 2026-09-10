"""区间越界→回收→向内推进的盘中过程研究；不是完整Brooks实现。"""
from collections import Counter

import numpy as np
import pandas as pd

from brooks_event_research import event_context
from brooks_intrabar import aggregate_context, compare_blocks, describe, nonoverlap
from brooks_intrabar_controls import backgrounds, choose_control, path_classes

VERSION = "failed-range-process-v1"
POLICY = {"version": VERSION, "background": "previous_closed_15m_event_context_range",
          "range": "previous_20_closed_15m_high_low", "atr": "previous_14_closed_15m_TR_mean",
          "observation": "completed_1m_close", "stages": ["outside", "inside", "inward"],
          "distinct_observations": True, "inward": "close_beyond_reclaim_minute_inward_extreme",
          "expiry": "same_15m_bar_end", "one_attempt_per_bar": True,
          "cancellations": ["return_to_or_outside_edge", "reach_or_cross_midpoint", "missing_or_zero_minute"],
          "partial_prefix": "pending_not_expired", "entry": "next_1m_open",
          "all_alerts_retained": True, "secondary_spacing_minutes": 60,
          "primary": "0.25ATR_3_complete_minutes_within_60m_no_adverse_veto",
          "control": "existing_background_match_plus_same_range_state",
          "development_only": True, "manual_win_rate": False, "production_changes": False}


def validate_minutes(minutes):
    index = minutes.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans or
            index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.floor("min"))):
        raise ValueError("分钟时间须带时区、唯一且按分钟网格递增")
    values = minutes[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("价格/量能必须有限，价格为正，量能非负")
    if ((minutes.high < minutes[["open", "low", "close"]].max(axis=1)) |
            (minutes.low > minutes[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("OHLC关系无效")


def range_background(minutes):
    bars, groups, _, clean, gaps, atr = aggregate_context(minutes)
    states = event_context(bars).state
    return pd.DataFrame({"range_high": bars.high.rolling(20).max().shift(),
                         "range_low": bars.low.rolling(20).min().shift(), "atr": atr,
                         "is_range": states.eq("range"),
                         "eligible": np.r_[False, clean[:-1]] & ~gaps}, index=bars.index), dict(groups)


def advance_attempt(state, minute, time, low, high):
    price, mid = float(minute.close), (low + high) / 2
    if state is None:
        if low <= price <= high:
            return None, "watching"
        side = -1 if price > high else 1
        return {"direction": side, "stage": "outside", "breakout_at": time.isoformat(),
                "edge": high if side == -1 else low}, "waiting"
    side, edge = state["direction"], state["edge"]
    if side * (price - mid) >= 0:
        return state, "midpoint_consumed"
    if state["stage"] == "outside":
        if side * (price - edge) <= 0:
            return state, "waiting"
        return {**state, "stage": "inside", "reclaim_at": time.isoformat(),
                "trigger_reference": float(minute.high if side == 1 else minute.low)}, "waiting"
    if side * (price - edge) <= 0:
        return state, "returned_outside"
    if side * (price - state["trigger_reference"]) > 0:
        return {**state, "confirmation_at": time.isoformat()}, "signal"
    return state, "waiting"


def make_signal(state, group, stamp, time, low, high, atr):
    partial = {"open": float(group.open.iloc[0]), "high": float(group.high.max()),
               "low": float(group.low.min()), "close": float(group.close.iloc[-1]),
               "volume": float(group.volume.sum())}
    side = state["direction"]
    return {**state, "family": "failed_range", "episode_id": f"failed_range:{stamp.isoformat()}",
            "bar_start": stamp.isoformat(), "available_at": time.isoformat(), "atr": float(atr),
            "range_low": float(low), "range_high": float(high), "range_known_at": stamp.isoformat(),
            "observed_price": partial["close"], "partial_bar": partial,
            "structure_reference": partial["low"] if side == 1 else partial["high"],
            "lead_minutes": int((stamp + pd.Timedelta(minutes=15) - time) / pd.Timedelta(minutes=1))}


def observe_range_bar(group, stamp, low, high, atr):
    state, reason = None, "watching"
    for offset, minute in enumerate(group.itertuples()):
        if minute.Index != stamp + pd.Timedelta(minutes=offset) or minute.volume <= 0:
            return None, {"bar_start": stamp.isoformat(), "state": state, "reason": "quality_reset"}
        time = minute.Index + pd.Timedelta(minutes=1)
        state, reason = advance_attempt(state, minute, time, low, high)
        if reason == "signal":
            signal = make_signal(state, group.iloc[:offset + 1], stamp, time, low, high, atr)
            return signal, {"bar_start": stamp.isoformat(), "state": state, "reason": reason}
        if reason in ("midpoint_consumed", "returned_outside"):
            return None, {"bar_start": stamp.isoformat(), "state": state, "reason": reason}
    complete = len(group) == 15
    reason = ("expired" if state else "no_breakout") if complete else "pending"
    return None, {"bar_start": stamp.isoformat(), "state": state, "reason": reason}


def scan_failed_range(minutes, start, end):
    validate_minutes(minutes)
    if minutes.empty:
        return [], []
    frame, groups = range_background(minutes)
    signals, attempts = [], []
    for stamp, facts in frame.iterrows():
        if stamp < start or stamp >= end or not facts.eligible or not facts.is_range:
            continue
        group = groups[stamp]
        group = group.loc[group.index + pd.Timedelta(minutes=1) < end]
        signal, attempt = observe_range_bar(group, stamp, facts.range_low, facts.range_high, facts.atr)
        attempts.append(attempt)
        if signal:
            signals.append(signal)
    return signals, attempts


def match_range_controls(minutes, signals, start, end):
    frame, groups = backgrounds(minutes)
    range_frame, _ = range_background(minutes)
    frame = frame.assign(eligible=frame.eligible & range_frame.is_range)
    alerts = [pd.Timestamp(s["available_at"]) for s in signals]
    used, matches = [], []
    for signal in signals:
        control = choose_control(frame, groups, signal, start, end, alerts, used)
        if control:
            control = {**control, "family": "failed_range_control", "match_range_state": True}
            used.append(pd.Timestamp(control["available_at"]))
        matches.append(control)
    return matches


def summarize_process(rows, attempts, block_ids):
    keys = [f"{t}/{d}" for t in (.25, .5, 1.) for d in (1, 3, 5)]
    return {"n": len(rows), "bar_funnel": dict(Counter(a["reason"] for a in attempts)),
            "attempts_started": sum(a["state"] is not None for a in attempts),
            "intrabar_n": sum(r["lead_minutes"] > 0 for r in rows), "paths": path_classes(rows),
            "grid": {key: {s: describe(rows, key, s) for s in ("forward", "reverse")} for key in keys},
            "nonoverlap": {s: describe(nonoverlap(rows), "0.25/3", s) for s in ("forward", "reverse")},
            "forward_minus_reverse": compare_blocks(rows, block_ids)}
