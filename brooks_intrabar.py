"""已闭合15M背景与逐分钟可见的未完成15M形态；研究告警，不执行交易。"""
from collections import Counter

import numpy as np
import pandas as pd

from brooks_first_retest import advance_retest, start_breakout
from brooks_sequence_research import MAX_EPISODE_BARS, advance, context_and_quality, normalize, start_episode

VERSION = "brooks-intrabar-v1"
FAMILIES = ("h2", "outside", "retest")
TARGETS, DURATIONS = (.25, .5, 1.), (1, 3, 5)
POLICY = {"version": VERSION, "signal_timeframe": "15m", "observation_resolution": "completed_1m_snapshot",
          "background": "closed_15m_only", "atr": "14_true_ranges_before_live_15m_bar",
          "trigger": "current_observed_price_cross_not_future_high", "entry": "next_1m_open_after_observation",
          "families": list(FAMILIES), "one_alert_per_family_episode": True,
          "target_atr": list(TARGETS), "continuous_complete_minutes": list(DURATIONS),
          "horizon_minutes": 60, "primary_display": "0.25/3", "max_adverse_veto": None,
          "small_pullback_report_atr": [.25, .5, 1.], "cost_is_diagnostic_only_pct": .14,
          "failed_at_close_alerts_retained": True, "future_selected_windows": False,
          "tick_accuracy_claimed": False, "manual_win_rate_estimated": False, "production_changes": False}


def aggregate_context(minutes):
    grouped = minutes.groupby(minutes.index.floor("15min"))
    bars = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    complete = grouped.size().eq(15) & grouped.volume.min().gt(0)
    quality = bars.copy()
    quality.loc[~complete, "volume"] = 0.
    context, clean, gaps = context_and_quality(quality, "15m")
    previous = bars.close.shift()
    tr = pd.concat([bars.high - bars.low, (bars.high - previous).abs(), (bars.low - previous).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift()
    return bars, list(grouped), context, clean, gaps, atr


def spec(family, start, side, reference, invalidation, bars, atr):
    return {"family": family, "direction": side, "trigger_reference": float(reference),
            "invalidation": float(invalidation), "atr": float(atr),
            "episode_id": f"{family}:{bars.index[start].isoformat()}:{side}"}


def bar_specs(state, index, bars, context, atr):
    result, old = [], state["h2"]
    previous = bars.iloc[index - 1]
    if old and old.count == 1 and old.armed and index - old.start < MAX_EPISODE_BARS and context != -old.direction:
        side = old.direction
        result.append(spec("h2", old.start, side, previous.high if side == 1 else previous.low,
                           side * old.floor, bars, atr))
    old = state["outside"]
    if old and old["index"] + 1 == index and context != -old["direction"] and index - old["episode_start"] < MAX_EPISODE_BARS:
        side = old["direction"]
        result.append(spec("outside", old["episode_start"], side, previous.high if side == 1 else previous.low,
                           old["invalidation"], bars, atr))
    old = state["retest"]
    if old and index - old["breakout_index"] < MAX_EPISODE_BARS and context != -old["direction"]:
        result.append(spec("retest", old["breakout_index"], old["direction"], old["level"],
                           old["structure_invalidation"], bars, atr))
    return result


def visible_trigger(candidate, partial, previous):
    side, reference = candidate["direction"], candidate["trigger_reference"]
    if side * (partial["close"] - reference) <= 0:
        return False
    if candidate["family"] == "h2":
        return not (partial["high"] > previous.high and partial["low"] < previous.low)
    if candidate["family"] == "retest":
        return partial["low"] <= reference if side == 1 else partial["high"] >= reference
    return True


def observe_bar(group, stamp, specs, previous, unavailable):
    emitted, partial = [], None
    for offset, minute in enumerate(group.itertuples()):
        if minute.Index != stamp + pd.Timedelta(minutes=offset) or minute.volume <= 0:
            break
        partial = ({"open": float(minute.open), "high": float(minute.high), "low": float(minute.low),
                    "close": float(minute.close), "volume": float(minute.volume)} if partial is None else
                   {**partial, "high": max(partial["high"], float(minute.high)), "low": min(partial["low"], float(minute.low)),
                    "close": float(minute.close), "volume": partial["volume"] + float(minute.volume)})
        for candidate in specs:
            key, side = candidate["episode_id"], candidate["direction"]
            if key in unavailable:
                continue
            if side * (partial["close"] - candidate["invalidation"]) < 0:
                unavailable.add(key)
                continue
            if visible_trigger(candidate, partial, previous):
                available = minute.Index + pd.Timedelta(minutes=1)
                reference = (partial["low"] if side == 1 else partial["high"]) if candidate["family"] == "retest" else candidate["invalidation"]
                emitted.append({**candidate, "bar_start": stamp.isoformat(), "available_at": available.isoformat(),
                                "observed_price": partial["close"], "partial_bar": dict(partial),
                                "structure_reference": reference, "lead_minutes": 14 - offset})
                unavailable.add(key)
    return emitted


def close_bar(state, index, bars, context, clean):
    if not clean:
        return {"h2": None, "outside": None, "retest": None}
    bar, old = bars.iloc[index], state["h2"]
    pending, new = None, None
    if old:
        new, ended = advance(old, index, normalize(bar[["high", "low", "close"]].to_numpy(), old.direction),
                             normalize(bars.iloc[index - 1][["high", "low", "close"]].to_numpy(), old.direction), context)
        pending = next((e for e in ended if e["kind"] == "ambiguous_outside" and e["count"] < 2), None)
    elif context:
        new = start_episode(bars[["high", "low", "close"]].to_numpy(), index, context)
    retest = state["retest"]
    if retest:
        if advance_retest(retest, index, bar, context) != "waiting":
            retest = None
    else:
        retest = start_breakout(bars, index, context)
    return {"h2": new, "outside": pending, "retest": retest}


def scan_intrabar(minutes, start, end):
    if minutes.index.has_duplicates or not minutes.index.is_monotonic_increasing or minutes.index.tz is None:
        raise ValueError("分钟时间须带时区、唯一且递增")
    if not minutes.index.equals(minutes.index.floor("min")):
        raise ValueError("分钟数据不在分钟网格")
    if len(minutes) == 0:
        return []
    bars, grouped, context, clean, gaps, atr = aggregate_context(minutes)
    state, unavailable, signals = {"h2": None, "outside": None, "retest": None}, set(), []
    for index, (stamp, group) in enumerate(grouped):
        if index == 0:
            continue
        if not clean[index - 1] or gaps[index]:
            state = {"h2": None, "outside": None, "retest": None}
        else:
            specs = bar_specs(state, index, bars, int(context[index]), float(atr.iloc[index]))
            emitted = observe_bar(group, stamp, specs, bars.iloc[index - 1], unavailable)
            signals.extend(s for s in emitted if start <= pd.Timestamp(s["available_at"]) < end)
        # 未完成的本根不能用于推进已闭合15M状态；已发出的告警不撤销。
        if len(group) == 15 and group.index[-1] == stamp + pd.Timedelta(minutes=14):
            state = close_bar(state, index, bars, int(context[index]), bool(clean[index]))
    return sorted(signals, key=lambda s: (s["available_at"], s["family"]))


def space_probe(future, entry, atr, side, target, duration):
    worst = future.low.to_numpy(float) if side == 1 else future.high.to_numpy(float)
    best = future.high.to_numpy(float) if side == 1 else future.low.to_numpy(float)
    safe = side * (worst - entry) >= target * atr
    runs = np.convolve(safe.astype(int), np.ones(duration, dtype=int), mode="valid")
    starts = np.flatnonzero(runs == duration)
    begin = int(starts[0]) if len(starts) else None
    finish = begin + duration if begin is not None else len(future)
    adverse = np.maximum(0., -side * (worst[:finish] - entry))
    mae = float(adverse.max() / atr)
    closes = side * (future.close.to_numpy(float)[:finish] - entry)
    dips = np.flatnonzero(closes < 0)
    recovery = np.flatnonzero(closes[dips[0] + 1:] >= 0) if len(dips) else []
    longest, streak = 0, 0
    for held in safe:
        streak = streak + 1 if held else 0
        longest = max(longest, streak)
    return {"held": int(begin is not None), "touched": int((side * (best - entry) >= target * atr).any()),
            "first_run_start_minutes": begin, "first_run_proven_minutes": finish if begin is not None else None,
            "longest_complete_minutes": longest, "pre_run_or_timeout_mae_atr": mae, "mae_pct": mae * atr / entry * 100,
            "shallow_held": int(begin is not None and mae <= .25), "mild_held": int(begin is not None and mae <= .5),
            "one_atr_held": int(begin is not None and mae <= 1.),
            "recovered_after_negative_close": bool(begin is not None and len(dips)),
            "recovery_minutes": int(recovery[0] + 1) if begin is not None and len(recovery) else None,
            "minimum_space_net_pct": target * atr / entry * 100 - .14}


def score_signal(minutes, signal):
    available = pd.Timestamp(signal["available_at"])
    future = minutes.loc[(minutes.index >= available) & (minutes.index < available + pd.Timedelta(hours=1))]
    expected = pd.date_range(available, periods=60, freq="min")
    closing = pd.Timestamp(signal["bar_start"]) + pd.Timedelta(minutes=14)
    at_close = float(minutes.close.loc[closing]) if closing in minutes.index else None
    kept = signal["direction"] * (at_close - signal["trigger_reference"]) > 0 if at_close is not None else None
    base = {**signal, "above_trigger_at_15m_close": kept}
    if not future.index.equals(expected) or (future.volume <= 0).any():
        return {**base, "status": "future_unverified", "cases": {}}
    entry, atr, side = float(future.open.iloc[0]), signal["atr"], signal["direction"]
    if entry <= 0 or not np.isfinite(atr) or atr <= 0:
        raise ValueError("入场或冻结ATR无效")
    cases = {f"{target}/{duration}": {name: space_probe(future, entry, atr, direction, target, duration)
              for name, direction in (("forward", side), ("reverse", -side))} for target in TARGETS for duration in DURATIONS}
    return {**base, "status": "scored", "entry": entry,
            "entry_move_bps": side * (entry / signal["observed_price"] - 1) * 10000, "cases": cases}


def describe(rows, case, side):
    probes = [r["cases"][case][side] for r in rows if r["status"] == "scored"]
    n, unknown = len(rows), len(rows) - len(probes)
    held = sum(p["held"] for p in probes)
    successful = [p for p in probes if p["held"]]
    quantile = lambda items, q: float(np.quantile(items, q)) if items else None
    return {"n": n, "unknown_n": unknown, "held_count_bounds": [held, held + unknown],
            "held_rate_bounds": [held / n, (held + unknown) / n] if n else None,
            "touched_n": sum(p["touched"] for p in probes), "shallow_held_n": sum(p["shallow_held"] for p in probes),
            "mild_held_n": sum(p["mild_held"] for p in probes), "one_atr_held_n": sum(p["one_atr_held"] for p in probes),
            "recovered_after_negative_close_n": sum(p["recovered_after_negative_close"] for p in probes),
            "success_wait_median_minutes": quantile([p["first_run_start_minutes"] for p in successful], .5),
            "success_mae_atr_p90": quantile([p["pre_run_or_timeout_mae_atr"] for p in successful], .9),
            "all_known_mae_pct_p90": quantile([p["mae_pct"] for p in probes], .9)}


def nonoverlap(rows):
    kept, next_time = [], None
    for row in sorted(rows, key=lambda r: r["available_at"]):
        current = pd.Timestamp(row["available_at"])
        if next_time is None or current >= next_time:
            kept.append(row)
            next_time = current + pd.Timedelta(hours=1)
    return kept


def summarize(rows):
    results = {}
    for family in FAMILIES:
        alerts = [r for r in rows if r["family"] == family]
        kept = nonoverlap(alerts)
        results[family] = {"alert_n": len(alerts), "nonoverlap_n": len(kept),
                           "intrabar_n": sum(r["lead_minutes"] > 0 for r in alerts),
                           "not_above_trigger_at_close_n": sum(r["above_trigger_at_15m_close"] is False for r in alerts),
                           "status_counts": dict(Counter(r["status"] for r in alerts)),
                           "all_alerts": {f"{t}/{d}": {s: describe(alerts, f"{t}/{d}", s) for s in ("forward", "reverse")}
                                          for t in TARGETS for d in DURATIONS},
                           "nonoverlap_primary": {s: describe(kept, "0.25/3", s) for s in ("forward", "reverse")}}
    return results


def compare_blocks(rows, block_ids, deduplicate=False):
    counts = np.zeros((len(block_ids), 6))
    positions = {block: i for i, block in enumerate(block_ids)}
    for row in nonoverlap(rows) if deduplicate else rows:
        bounds = {s: [row["cases"]["0.25/3"][s]["held"]] * 2 if row["status"] == "scored" else [0, 1]
                  for s in ("forward", "reverse")}
        counts[positions[row["block_id"]]] += [1, *bounds["forward"], 1, *bounds["reverse"]]
    totals = counts.sum(axis=0)
    if not totals[0]:
        return None
    delta = lambda c: np.stack([c[..., 1] / c[..., 0] - c[..., 5] / c[..., 3],
                                c[..., 2] / c[..., 0] - c[..., 4] / c[..., 3]], axis=-1)
    draws = np.random.default_rng(42).integers(0, len(block_ids), size=(2000, len(block_ids)))
    sums = counts[draws].sum(axis=1)
    valid = sums[sums[:, 0] > 0]
    differences = delta(valid)
    return {"count": int(totals[0]), "forward_minus_reverse": delta(totals).tolist(),
            "block_bootstrap_95pct": [float(np.quantile(differences[:, 0], .025)), float(np.quantile(differences[:, 1], .975))],
            "calendar_blocks_n": len(block_ids), "valid_draws_n": len(valid)}
