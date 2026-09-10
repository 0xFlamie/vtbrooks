"""冻结突破价位后的首次回触收线研究；非完整Brooks实现或交易指令。"""
from collections import Counter

import numpy as np

from brooks_sequence_research import (ANCHOR_WINDOW, LOOKBACK, MAX_EPISODE_BARS,
                                      STEPS, context_and_quality, validate)
from market_path_model import feature_frame

VERSION = "first-breakout-retest-v1"


def start_breakout(frame, index, side):
    if side not in (-1, 1) or index < LOOKBACK - 1:
        return None
    history, bar = frame.iloc[index - ANCHOR_WINDOW:index], frame.iloc[index]
    level = float(history.high.max() if side == 1 else history.low.min())
    if side * (bar.close - level) <= 0:
        return None
    return {"breakout_index": index, "direction": side, "level": level,
            "breakout_close": float(bar.close),
            "breakout_stop_reference": float(bar.low if side == 1 else bar.high),
            "structure_invalidation": float(history.low.min() if side == 1 else history.high.max()),
            "episode_id": f"retest:15m:{frame.index[index].isoformat()}:{side}"}


def advance_retest(state, index, bar, context, clean=True, gap=False):
    if index <= state["breakout_index"]:
        raise ValueError("突破K本身不能算事后回触")
    side = state["direction"]
    reason = next((name for condition, name in (
        (not clean or gap, "quality_or_gap"),
        (side * (bar.close - state["structure_invalidation"]) < 0, "structure_invalidated"),
        (context == -side, "context_reversed"),
        (index - state["breakout_index"] >= MAX_EPISODE_BARS, "expired")) if condition), None)
    if reason:
        return reason
    touched = bar.low <= state["level"] if side == 1 else bar.high >= state["level"]
    if not touched:
        return "waiting"
    return "confirmed" if side * (bar.close - state["level"]) > 0 else "first_touch_failed"


def make_plan(frame, state, index):
    bar, side = frame.iloc[index], state["direction"]
    atr = float(feature_frame(frame.iloc[index - LOOKBACK + 1:index + 1]).atr_pct.iloc[-1] * bar.close / 100)
    stop = float(bar.low if side == 1 else bar.high)
    if not np.isfinite(atr) or atr <= 0 or side * (bar.close - stop) <= 0:
        raise ValueError("ATR或回触K失效参考无效")
    seed_risk = side * (state["breakout_close"] - state["breakout_stop_reference"])
    return {**state, "index": index, "kind": "first_retest_confirmation", "timeframe": "15m", "atr": atr,
            "bar_time": frame.index[index].isoformat(), "available_at": (frame.index[index] + STEPS["15m"]).isoformat(),
            "version": VERSION, "stop_reference": stop, "confirmation_close": float(bar.close),
            "wait_bars": index - state["breakout_index"],
            "close_price_improvement": float(side * (state["breakout_close"] - bar.close)),
            "close_risk_distance": float(side * (bar.close - stop)),
            "breakout_close_risk_distance": float(seed_risk),
            "close_risk_improvement": float(seed_risk - side * (bar.close - stop))}


def scan_retests(frame):
    validate(frame, "15m")
    context, clean, gaps = context_and_quality(frame, "15m")
    state, episodes, plans = None, [], []
    for index in range(LOOKBACK - 1, len(frame)):
        if state is not None:
            status = advance_retest(state, index, frame.iloc[index], int(context[index]), clean[index], gaps[index])
            if status != "waiting":
                episodes.append({**state, "end_index": index, "status": status})
                if status == "confirmed":
                    plans.append(make_plan(frame, state, index))
                state = None
            continue
        if clean[index] and not gaps[index]:
            state = start_breakout(frame, index, int(context[index]))
    if state is not None:
        episodes.append({**state, "end_index": len(frame) - 1, "status": "pending"})
    if len({p["episode_id"] for p in plans}) != len(plans):
        raise ValueError("同一突破重复确认")
    return {"episodes": episodes, "plans": plans, "breakout_n": len(episodes), "confirmed_n": len(plans),
            "statuses": dict(Counter(e["status"] for e in episodes))}


def geometry(plans):
    n = len(plans)
    if not n:
        return {"n": 0}
    return {"n": n, "price_better_n": sum(p["close_price_improvement"] > 0 for p in plans),
            "risk_closer_n": sum(p["close_risk_improvement"] > 0 for p in plans),
            "both_better_n": sum(p["close_price_improvement"] > 0 and p["close_risk_improvement"] > 0 for p in plans),
            "median_wait_bars": float(np.median([p["wait_bars"] for p in plans])),
            "median_close_risk_atr": float(np.median([p["close_risk_distance"] / p["atr"] for p in plans])),
            "median_price_improvement_atr": float(np.median([p["close_price_improvement"] / p["atr"] for p in plans]))}


def audit_case(frame, plan):
    """独立手算边界、首次触及与TR；不读取确认收线之后的数据。"""
    start, end, side = plan["breakout_index"], plan["index"], plan["direction"]
    visible = frame.iloc[:end + 1]
    history, span = visible.iloc[start - ANCHOR_WINDOW:start], visible.iloc[start + 1:end]
    level = float(history.high.max() if side == 1 else history.low.min())
    if level != plan["level"] or side * (visible.close.iloc[start] - level) <= 0:
        raise ValueError("突破价位手算不一致")
    earlier_touch = (span.low <= level).any() if side == 1 else (span.high >= level).any()
    if earlier_touch:
        raise ValueError("不是第一次回触")
    bar = visible.iloc[-1]
    if (bar.low > level if side == 1 else bar.high < level) or side * (bar.close - level) <= 0:
        raise ValueError("回触收线手算不一致")
    tail = visible.iloc[-15:]
    tr = [max(float(tail.high.iloc[i] - tail.low.iloc[i]), abs(float(tail.high.iloc[i] - tail.close.iloc[i - 1])),
              abs(float(tail.low.iloc[i] - tail.close.iloc[i - 1]))) for i in range(1, len(tail))]
    if abs(float(np.mean(tr)) - plan["atr"]) > 1e-8:
        raise ValueError("ATR手算不一致")
    return {"breakout_index": start, "confirmation_index": end, "direction": side, "level": level,
            "breakout_close": float(visible.close.iloc[start]), "confirmation_close": float(bar.close),
            "stop_reference": plan["stop_reference"], "atr_handchecked": float(np.mean(tr)),
            "first_touch_checked": True, "future_returns_used": False}
