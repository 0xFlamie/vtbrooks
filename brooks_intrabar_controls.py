"""盘中研究的事前背景匹配与路径分层；不改信号、不作交易裁判。"""
from collections import Counter

import numpy as np
import pandas as pd

from brooks_intrabar import aggregate_context, compare_blocks, score_signal

MATCH_POLICY = {"same_segment": True, "same_context": True, "same_utc_4h_slot": True,
                "same_intrabar_minute": True, "same_prior_20bar_position_third": True,
                "atr_pct_ratio": [.8, 1.25], "rank": "abs_log_atr_ratio_then_time_distance_then_time",
                "past_alert_exclusion_minutes": 60, "control_spacing_minutes": 60,
                "future_alerts_excluded": False, "future_quality_used_for_selection": False,
                "replacement": False, "causal_effect_claimed": False}


def backgrounds(minutes):
    bars, groups, context, clean, gaps, atr = aggregate_context(minutes)
    high, low = bars.high.rolling(20).max().shift(), bars.low.rolling(20).min().shift()
    position = (bars.close.shift() - low) / (high - low).replace(0, np.nan)
    frame = pd.DataFrame({"context": context, "atr": atr, "atr_pct": atr / bars.close.shift() * 100,
                          "position_third": np.minimum(2, np.floor(position.clip(0, 1) * 3)),
                          "eligible": np.r_[False, clean[:-1]] & ~gaps}, index=bars.index)
    return frame, dict(groups)


def visible_prefix(group, stamp, phase):
    part = group.loc[group.index < stamp + pd.Timedelta(minutes=phase)]
    if not part.index.equals(pd.date_range(stamp, periods=phase, freq="min")) or (part.volume <= 0).any():
        return None
    return {"open": float(part.open.iloc[0]), "high": float(part.high.max()), "low": float(part.low.min()),
            "close": float(part.close.iloc[-1]), "volume": float(part.volume.sum())}


def ranked_candidates(frame, signal, start, end):
    stamp, time = pd.Timestamp(signal["bar_start"]), pd.Timestamp(signal["available_at"])
    reference = frame.loc[stamp]
    phase = int((time - stamp) / pd.Timedelta(minutes=1))
    times = frame.index + pd.Timedelta(minutes=phase)
    ratio = frame.atr_pct / reference.atr_pct
    eligible = (frame.eligible & frame.context.eq(reference.context) & frame.position_third.eq(reference.position_third)
                & ratio.between(*MATCH_POLICY["atr_pct_ratio"]) & (times.hour // 4 == time.hour // 4)
                & (times >= start) & (times < end))
    choices = frame.loc[eligible].copy()
    choices["distance"] = np.abs(np.log(ratio.loc[eligible]))
    choices["time_distance"] = abs((choices.index + pd.Timedelta(minutes=phase) - time).total_seconds())
    choices["stamp"] = choices.index
    return choices.sort_values(["distance", "time_distance", "stamp"]), phase


def choose_control(frame, groups, signal, start, end, alert_times, used):
    choices, phase = ranked_candidates(frame, signal, start, end)
    for stamp, facts in choices.iterrows():
        time = stamp + pd.Timedelta(minutes=phase)
        if any(pd.Timedelta(0) <= time - a < pd.Timedelta(hours=1) for a in alert_times):
            continue
        if any(abs(time - a) < pd.Timedelta(hours=1) for a in used):
            continue
        partial = visible_prefix(groups[stamp], stamp, phase)
        if partial is None:
            continue
        return {"family": "outside", "direction": signal["direction"], "atr": float(facts.atr),
                "bar_start": stamp.isoformat(), "available_at": time.isoformat(), "lead_minutes": 15 - phase,
                "trigger_reference": partial["close"], "observed_price": partial["close"], "partial_bar": partial,
                "episode_id": f"control:{signal['episode_id']}", "matched_signal_time": signal["available_at"],
                "match_atr_pct": float(facts.atr_pct), "match_context": int(facts.context),
                "match_position_third": int(facts.position_third), "research_control_not_alert": True}
    return None


def match_controls(minutes, signals, start, end):
    frame, groups = backgrounds(minutes)
    alerts = [pd.Timestamp(s["available_at"]) for s in signals]
    used, matches = [], []
    for signal in signals:
        control = choose_control(frame, groups, signal, start, end, alerts, used)
        if control:
            used.append(pd.Timestamp(control["available_at"]))
        matches.append(control)
    return matches


def path_classes(rows):
    def category(row):
        if row["status"] != "scored":
            return "unknown"
        probe = row["cases"]["0.25/3"]["forward"]
        if not probe["held"]:
            return "no_sustained_space"
        if probe["pre_run_or_timeout_mae_atr"] == 0:
            return "no_adverse_price"
        return "small_pullback_recovery" if probe["mild_held"] else "deep_recovery"
    counts = Counter(category(row) for row in rows)
    return {name: counts[name] for name in ("no_adverse_price", "small_pullback_recovery", "deep_recovery",
                                           "no_sustained_space", "unknown")}


def paired_control_comparison(rows, controls, block_ids):
    pairs = []
    for row, control in zip(rows, controls, strict=True):
        if control is None:
            continue
        known = row["status"] == control["status"] == "scored"
        pairs.append({"block_id": row["block_id"], "available_at": row["available_at"],
                      "status": "scored" if known else "future_unverified",
                      "cases": {"0.25/3": {"forward": row["cases"]["0.25/3"]["forward"],
                                             "reverse": control["cases"]["0.25/3"]["forward"]}} if known else {}})
    return {"matched_n": len(pairs), "unmatched_n": len(rows) - len(pairs),
            "paired_known_n": sum(p["status"] == "scored" for p in pairs),
            "signal_minus_control": compare_blocks(pairs, block_ids)}


def score_controls(minutes, matches, records):
    return [{**score_signal(minutes, control), "block_id": row["block_id"],
             "segment_id": row["segment_id"]} if control is not None else None
            for control, row in zip(matches, records, strict=True)]
