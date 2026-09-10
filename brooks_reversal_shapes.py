"""已确认历史拐点→盘中双重测试/三推反转；固定研究近似，不接生产。"""
import pandas as pd

from brooks_failed_range import validate_minutes
from brooks_intrabar import aggregate_context

POLICY = {"version": "reversal-shapes-intrabar-v1", "families": ["double_test", "wedge"],
          "pivot_left_right": 2, "history_bars": 40, "double_band_atr": .25,
          "double_bridge_atr": .5, "double_trigger": "later_1m_close_beyond_first_test_minute_extreme",
          "double_same_15m_expiry": True, "double_one_attempt_per_anchor": True,
          "wedge": "five_alternating_confirmed_pivots_three_progressive_pushes_two_progressive_retracements",
          "wedge_min_span_atr": 1., "wedge_entry_within_bars_of_confirmation": 5,
          "wedge_stop_buffer_atr": .25, "wedge_trigger": "1m_close_beyond_previous_15m_extreme",
          "wedge_outside_ambiguity": "cancel", "same_minute_invalidation_priority": True,
          "atr": "14_TR_before_live_15m", "primary": "0.25/3", "horizon_minutes": 60,
          "no_future_pivot_backdating": True, "tracking_warmup_bars": 40,
          "emission_quality_bars": 120, "production": False}


def confirmed_swings(bars, live_index):
    result = []
    highs, lows = bars.high.to_numpy(), bars.low.to_numpy()
    for i in range(max(2, live_index - POLICY["history_bars"]), live_index - 2):
        neighbors = [i - 2, i - 1, i + 1, i + 2]
        low, high = lows[i] < lows[neighbors].min(), highs[i] > highs[neighbors].max()
        if low and high:
            result = []
            continue
        if not low and not high:
            continue
        kind, price = ("low", float(bars.low.iloc[i])) if low else ("high", float(bars.high.iloc[i]))
        point = {"kind": kind, "index": i, "price": price, "confirmed_index": i + 2,
                 "at": bars.index[i].isoformat(), "known_at": (bars.index[i + 2] + pd.Timedelta(minutes=15)).isoformat()}
        if result and result[-1]["kind"] == kind:
            more_extreme = price < result[-1]["price"] if low else price > result[-1]["price"]
            if more_extreme:
                result[-1] = point
        else:
            result.append(point)
    return result


def double_candidates(bars, live_index, atr, swings):
    result = []
    for side, kind in ((1, "low"), (-1, "high")):
        point = next((p for p in reversed(swings) if p["kind"] == kind), None)
        if point is None:
            continue
        span = bars.iloc[point["index"] + 1:live_index]
        bridge = float(span.high.max() - point["price"]) if side == 1 else float(point["price"] - span.low.min())
        if bridge < POLICY["double_bridge_atr"] * atr:
            continue
        # 已破坏的旧拐点不能被当前一次回收复活为同一个双底/双顶。
        worst = float(span.low.min()) if side == 1 else float(span.high.max())
        if side * (worst - point["price"]) < -POLICY["double_band_atr"] * atr:
            continue
        result.append({"family": "double_test", "direction": side, "anchor": point["price"],
                       "pivots": [point], "episode_id": f"double_test:{point['at']}:{side}",
                       "invalidation": point["price"] - side * POLICY["double_band_atr"] * atr, "atr": float(atr)})
    return result


def wedge_candidate(bars, live_index, atr, swings):
    if len(swings) < 5:
        return None
    points = swings[-5:]
    side = 1 if points[-1]["kind"] == "low" else -1
    kinds = ["low", "high", "low", "high", "low"] if side == 1 else ["high", "low", "high", "low", "high"]
    if [p["kind"] for p in points] != kinds:
        return None
    a, b, c, d, e = [side * p["price"] for p in points]
    if not (a > c > e and b > d and b - e >= POLICY["wedge_min_span_atr"] * atr):
        return None
    if live_index - points[-1]["confirmed_index"] > POLICY["wedge_entry_within_bars_of_confirmation"]:
        return None
    invalidation = points[-1]["price"] - side * POLICY["wedge_stop_buffer_atr"] * atr
    span = bars.iloc[points[-1]["index"] + 1:live_index]
    worst = float(span.low.min()) if side == 1 else float(span.high.max())
    if side * (worst - invalidation) <= 0:
        return None
    identity = ":".join(p["at"] for p in points)
    return {"family": "wedge", "direction": side, "pivots": points, "anchor": points[-1]["price"],
            "episode_id": f"wedge:{identity}:{side}", "invalidation": invalidation, "atr": float(atr)}


def make_signal(candidate, partial, stamp, available, reference):
    return {**candidate, "bar_start": stamp.isoformat(), "available_at": available.isoformat(),
            "lead_minutes": int((stamp + pd.Timedelta(minutes=15) - available).total_seconds() // 60),
            "observed_price": partial["close"], "trigger_reference": reference,
            "structure_reference": candidate["invalidation"], "partial_bar": dict(partial)}


def observe_shapes(group, stamp, previous, candidates, used):
    partial, states, emitted = None, {}, []
    for offset, minute in enumerate(group.itertuples()):
        if minute.Index != stamp + pd.Timedelta(minutes=offset) or minute.volume <= 0:
            break
        partial = ({"open": float(minute.open), "high": float(minute.high), "low": float(minute.low),
                    "close": float(minute.close), "volume": float(minute.volume)} if partial is None else
                   {**partial, "high": max(partial["high"], float(minute.high)), "low": min(partial["low"], float(minute.low)),
                    "close": float(minute.close), "volume": partial["volume"] + float(minute.volume)})
        for c in candidates:
            key, side = c["episode_id"], c["direction"]
            if key in used and key not in states:
                continue
            worst = partial["low"] if side == 1 else partial["high"]
            if side * (worst - c["invalidation"]) <= 0:
                used.add(key)
                states.pop(key, None)
                continue
            if c["family"] == "double_test":
                touched = side * (worst - c["anchor"]) <= POLICY["double_band_atr"] * c["atr"]
                if key not in states:
                    if touched:
                        states[key] = float(minute.high if side == 1 else minute.low)
                        used.add(key)
                    continue
                reference = states[key]
            else:
                reference = float(previous.high if side == 1 else previous.low)
            if side * (partial["close"] - reference) <= 0:
                continue
            if c["family"] == "wedge" and partial["high"] > previous.high and partial["low"] < previous.low:
                used.add(key)
                continue
            emitted.append(make_signal(c, partial, stamp, minute.Index + pd.Timedelta(minutes=1), reference))
            used.add(key)
            states.pop(key, None)
    return emitted


def scan_reversal_shapes(minutes, start, end):
    validate_minutes(minutes)
    if minutes.empty:
        return []
    bars, groups, _, clean, gaps, atr = aggregate_context(minutes)
    valid = pd.Series([len(g) == 15 and g.volume.min() > 0 and bars.high.iloc[i] > bars.low.iloc[i]
                       for i, (_, g) in enumerate(groups)])
    history = POLICY["tracking_warmup_bars"]
    tracking = valid.rolling(history).sum().shift().eq(history)
    used, signals = set(), []
    for index, (stamp, group) in enumerate(groups):
        if index < history or not tracking.iloc[index] or gaps[index] or stamp - bars.index[index - history] != pd.Timedelta(minutes=15 * history):
            continue
        swings = confirmed_swings(bars, index)
        candidates = double_candidates(bars, index, float(atr.iloc[index]), swings)
        wedge = wedge_candidate(bars, index, float(atr.iloc[index]), swings)
        if wedge:
            candidates.append(wedge)
        emitted = observe_shapes(group, stamp, bars.iloc[index - 1], candidates, used)
        if clean[index - 1]:
            signals.extend(s for s in emitted if start <= pd.Timestamp(s["available_at"]) < end)
    return sorted(signals, key=lambda s: (s["available_at"], s["family"]))
