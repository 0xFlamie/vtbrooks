"""逐根识图审计：事实、旧规则输出和未验证候选分别记录。"""
import numpy as np
import pandas as pd

import vt_vote_bot as bot
from brooks_event_research import event_context, event_masks, trade_outcome

WINDOW = 120
HORIZONS = {"4h": 3, "15m": 16}
SETUP_NAMES = {"trend_pullback": "趋势回调确认", "breakout_retest": "突破回踩",
               "range_failed_breakout": "区间假突破"}


def confirmed_swings(window):
    high, low = bot._swing_points(window.high.to_numpy(), 2)[0], bot._swing_points(window.low.to_numpy(), 2)[1]
    result = []
    for kind, mask, prices in (("high", high, window.high), ("low", low, window.low)):
        for index in np.flatnonzero(mask)[-3:]:
            result.append({"kind": kind, "time": window.index[index].isoformat(), "price": float(prices.iloc[index]),
                           "confirmed_at": window.index[index + 2].isoformat()})
    return result


def structure(swings):
    highs = [x["price"] for x in swings if x["kind"] == "high"]
    lows = [x["price"] for x in swings if x["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return 0, "摆动结构尚未确认", None
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return 1, "高点与低点都在抬高", lows[-1]
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return -1, "高点与低点都在降低", highs[-1]
    return 0, "高低点方向不一致", None


def inspect_bar(frame, index, timeframe):
    """所有识图入口只接收截止当前的120根K线，未来数据留给独立结算。"""
    if timeframe not in HORIZONS or index < WINDOW - 1 or index >= len(frame):
        raise ValueError("周期不支持或历史窗口不完整")
    window = frame.iloc[index - WINDOW + 1:index + 1].copy()
    swings = confirmed_swings(window)
    direction, description, invalid = structure(swings)
    facts = event_context(window)
    current = facts.iloc[-1]
    matches = [{"name": name, "title": SETUP_NAMES[name], "direction": int(side)}
               for (name, side), mask in event_masks(window, facts).items() if bool(mask.iloc[-1])]
    old = bot.brooks_analyze(window)
    last = window.iloc[-1]
    high, low = float(window.high.iloc[-21:-1].max()), float(window.low.iloc[-21:-1].min())
    return {"time": frame.index[index].isoformat(), "timeframe": timeframe, "price": float(last.close),
            "structure": {"direction": direction, "description": description, "invalid": invalid},
            "swings": swings, "support": low, "resistance": high,
            "atr": float(current.atr_pct * last.close / 100),
            "body_ratio": float(current.body_ratio) if np.isfinite(current.body_ratio) else None,
            "overlap": float(current.overlap_5) if np.isfinite(current.overlap_5) else None,
            "flat_bars_20": int((window.high.tail(20) == window.low.tail(20)).sum()),
            "bar_read": old["deep"]["bar_read"], "breakouts": old["deep"]["setups"],
            "candidates": matches, "legacy": {"state": old["state"], "setups": old["setups"]}}


def followup(frame, index, snapshot):
    """只供点击揭晓后展示；不是识图输入或候选筛选条件。"""
    horizon = HORIZONS[snapshot["timeframe"]]
    if index + horizon >= len(frame):
        return {"available": False}
    entry = float(frame.open.iloc[index + 1])
    terminal = float(frame.close.iloc[index + horizon])
    outcomes = [{**candidate, **trade_outcome(frame, index, candidate["direction"], snapshot["atr"], horizon)}
                for candidate in snapshot["candidates"]]
    return {"available": True, "horizon": horizon, "entry": entry, "terminal": terminal,
            "return_pct": (terminal / entry - 1) * 100, "outcomes": outcomes}


def replay_clip(frame, start, steps, timeframe):
    horizon = HORIZONS[timeframe]
    if start < WINDOW - 1 or steps < 1 or start + steps + horizon > len(frame):
        raise ValueError("回放窗口越界")
    begin, end = start - WINDOW + 1, start + steps + horizon
    bars = [{"time": time.isoformat(), "open": float(row.open), "high": float(row.high),
             "low": float(row.low), "close": float(row.close)} for time, row in frame.iloc[begin:end].iterrows()]
    snapshots = [inspect_bar(frame, index, timeframe) for index in range(start, start + steps)]
    answers = [followup(frame, index, snap) for index, snap in zip(range(start, start + steps), snapshots)]
    return {"title": frame.index[start].strftime("%Y-%m-%d"), "bars": bars,
            "snapshots": snapshots, "answers": answers, "offset": WINDOW - 1}
