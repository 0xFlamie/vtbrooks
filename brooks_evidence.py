"""可审计的Brooks结构事实与数据质量；不产生方向票或胜率。"""
import numpy as np
import pandas as pd

QUALITY_WINDOW = 20
STRUCTURE_WINDOW = 120
SWING_CONFIRM_BARS = 2


def quality_check(frame):
    result = {"status": "invalid", "blocked": True, "message": "历史K线不足20根",
              "bars": 0, "flat_bars": 0, "zero_volume_bars": 0, "gaps": 0}
    if frame is None or len(frame) < QUALITY_WINDOW:
        return result
    tail = frame.tail(QUALITY_WINDOW)
    result["bars"] = len(tail)
    prices = tail[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any() or
            (tail.high < tail[["open", "close", "low"]].max(axis=1)).any() or
            (tail.low > tail[["open", "close", "high"]].min(axis=1)).any()):
        return {**result, "message": "OHLC数据无效，暂停入场确认"}
    flat = tail.high == tail.low
    volume = tail["volume"] if "volume" in tail else pd.Series(np.nan, index=tail.index)
    invalid_volume = not np.isfinite(volume.to_numpy(dtype=float)).all() or (volume < 0).any()
    zero = volume == 0
    gaps = 0
    if isinstance(frame.index, pd.DatetimeIndex):
        steps = frame.index.to_series().diff().iloc[1:]
        expected = steps.median()
        gaps = int((steps.tail(QUALITY_WINDOW - 1) != expected).sum())
        if expected <= pd.Timedelta(0) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            gaps = max(1, gaps)
    blocked = bool(flat.iloc[-1] or zero.iloc[-1] or invalid_volume or gaps)
    warnings = [text for condition, text in (
        (bool(flat.any()), f"无振幅{int(flat.sum())}根"), (bool(zero.any()), f"零成交{int(zero.sum())}根"),
        (bool(gaps), f"时间间隔异常{gaps}处"), (invalid_volume, "成交量缺失或无效")) if condition]
    message = "近20根K线连续、成交与振幅可用" if not warnings else "近20根：" + "，".join(warnings)
    if blocked:
        message += "；本根不能作为可靠入场确认"
    return {"status": "degraded" if warnings else "ok", "blocked": blocked, "message": message,
            "bars": len(tail), "flat_bars": int(flat.sum()), "zero_volume_bars": int(zero.sum()), "gaps": gaps}


def confirmed_swings(frame):
    result = []
    width = SWING_CONFIRM_BARS
    for kind, column in (("high", "high"), ("low", "low")):
        values = frame[column].to_numpy(dtype=float)
        points = []
        for index in range(width, len(frame) - width):
            segment = values[index - width:index + width + 1]
            match = (values[index] == segment.max() and values[index] > segment[0] and values[index] > segment[-1]) if kind == "high" else (
                values[index] == segment.min() and values[index] < segment[0] and values[index] < segment[-1])
            if match:
                times = [value.isoformat() if hasattr(value, "isoformat") else str(value)
                         for value in (frame.index[index], frame.index[index + width])]
                points.append({"kind": kind, "time": times[0], "price": float(values[index]),
                               "confirmed_at": times[1], "confirmed_index": index + width})
        result.extend(points[-3:])
    return result


def swing_structure(points):
    high = [p["price"] for p in points if p["kind"] == "high"]
    low = [p["price"] for p in points if p["kind"] == "low"]
    if min(len(high), len(low)) < 2:
        return {"direction": 0, "description": "摆动结构尚未确认", "invalid": None}
    if high[-1] > high[-2] and low[-1] > low[-2]:
        return {"direction": 1, "description": "高点与低点都在抬高", "invalid": low[-1]}
    if high[-1] < high[-2] and low[-1] < low[-2]:
        return {"direction": -1, "description": "高点与低点都在降低", "invalid": high[-1]}
    return {"direction": 0, "description": "高低点方向不一致", "invalid": None}


def inspect_evidence(frame):
    quality = quality_check(frame)
    points = [] if quality["status"] == "invalid" else confirmed_swings(frame.tail(STRUCTURE_WINDOW))
    structure = swing_structure(points)
    direction, invalid = structure["direction"], structure["invalid"]
    anchor = max((p["confirmed_index"] for p in points), default=0)
    closes = frame.tail(STRUCTURE_WINDOW).close.iloc[anchor:] if direction else []
    if direction and (direction * (closes - invalid) < 0).any():
        structure = {"direction": 0, "previous_direction": direction, "broken": True, "invalid": invalid,
                     "description": "原上升摆动低点已被收破，等待重新确认" if direction == 1
                     else "原下降摆动高点已被收破，等待重新确认"}
    return {"quality": quality, "swings": points, "structure": structure}


def evidence_brief(evidence):
    if not evidence:
        return ""
    quality, structure = evidence["quality"], evidence["structure"]
    invalid = structure["invalid"]
    level = f"；结构重新检查位${invalid:.2f}" if invalid is not None else ""
    boundary = "；暂停使用本根形态作入场确认，仍可解释已有结构" if quality["blocked"] else ""
    return f"数据质量: {quality['message']}。已确认摆动: {structure['description']}{level}{boundary}。结构描述不是经回测的方向概率。"
