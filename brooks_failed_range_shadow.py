"""隔离前瞻账本与逐笔分钟聚合；不下单、不向用户推送。"""
from contextlib import closing
import hashlib
import json
import math
import sqlite3

import pandas as pd

from brooks_failed_range import scan_failed_range, validate_minutes
from brooks_intrabar import DURATIONS, TARGETS, space_probe

POLICY = {"version": "failed-range-prospective-shadow-v1", "venue": "Coinbase", "symbol": "ETH-USD",
          "live_transport": "websocket_matches_and_heartbeat", "warmup": "REST_context_only_35h",
          "start": "first_15m_boundary_after_live_start", "max_lag_seconds": 120,
          "entry": "generated_at_floor_minute_plus_2_minutes", "deadline": "available_at_plus_60_minutes",
          "minimum_run_days_for_review": 30, "early_success_stop": False,
          "primary": "0.25/3", "reference": "same_signal_reverse", "ordinary_control": "not_implemented",
          "on_trade_gap_or_clock_error": "stop_capture_keep_ledger", "fill_empty_minutes": False,
          "delivery_to_user_measured": False, "history_live_ohlcv_parity_verified": False,
          "orders": False, "production_changes": False}


def now():
    return pd.Timestamp.now(tz="UTC")


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def append(database, key, kind, payload):
    existing = database.execute("SELECT payload FROM events WHERE event_key=?", (key,)).fetchone()
    if existing:
        previous_value = json.loads(existing[0])
        if previous_value["kind"] != kind or previous_value["payload"] != payload:
            raise ValueError("同一事件编号内容改变，不得静默覆盖")
        return False
    last = database.execute("SELECT sha256,payload FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    current = now()
    if last and current < pd.Timestamp(json.loads(last[1])["recorded_at"]):
        raise ValueError("账本写入时钟回退")
    deadline = pd.Timestamp(payload["entry_not_before"]) if kind == "signal" and payload.get("prospective_eligible") else None
    if deadline is not None and current >= deadline:
        raise ValueError("实际写入已晚于预定入场，不得记录为前瞻")
    value = {"kind": kind, "key": key, "recorded_at": current.isoformat(), "previous_sha256": last[0] if last else None,
             "payload": payload}
    database.execute("INSERT INTO events(event_key,payload,sha256) VALUES (?,?,?)", (key, encode(value), digest(value)))
    if deadline is not None and now() >= deadline:
        raise ValueError("事务提交前已过预定入场，回滚候选")
    return True


def initialize(path, header):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, event_key TEXT UNIQUE, payload TEXT, sha256 TEXT)")
        first = db.execute("SELECT payload FROM events ORDER BY seq LIMIT 1").fetchone()
        if first and json.loads(first[0])["payload"] != header:
            raise ValueError("账本协议不同，不得覆盖或换规则")
        append(db, "protocol", "protocol", header)


def iter_audit(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT event_key,payload,sha256 FROM events ORDER BY seq")
        previous, clock = None, None
        for key, payload, checksum in rows:
            value = json.loads(payload)
            time = pd.Timestamp(value["recorded_at"])
            if (value["key"] != key or digest(value) != checksum or value["previous_sha256"] != previous or
                    time.tz is None or (clock is not None and time < clock)):
                raise ValueError("账本哈希链或时间顺序不符")
            previous, clock = checksum, time
            yield value


def audit(path):
    return list(iter_audit(path))


def write_event(path, key, kind, payload):
    audit(path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        return append(db, key, kind, payload)


class MinuteStream:
    def __init__(self, started):
        self.start = started.ceil("min")
        self.last_id = None
        self.last_time = None
        self.last_received = started
        self.trades = {}
        self.sealed = self.start - pd.Timedelta(minutes=1)

    def ingest(self, message, received):
        if received < self.last_received:
            raise ValueError("本机时钟回退")
        self.last_received = received
        if message.get("type") not in ("match", "last_match", "heartbeat"):
            return []
        if message.get("product_id") != POLICY["symbol"]:
            raise ValueError("交易对不符")
        time = pd.Timestamp(message["time"])
        if time.tz is None or not 0 <= (received - time).total_seconds() <= POLICY["max_lag_seconds"]:
            raise ValueError("行情时钟超前或迟到")
        kind = message["type"]
        if kind == "last_match":
            if self.last_id is not None:
                raise ValueError("重复初始化成交编号")
            self.last_id = int(message["trade_id"])
            return []
        if kind == "match":
            self.add_trade(message, time, received)
            return []
        if self.last_id is None or int(message["last_trade_id"]) != self.last_id:
            raise ValueError("心跳成交编号与接收数据不符")
        return self.close_minutes(time.floor("min"), received)

    def add_trade(self, message, time, received):
        identity = int(message["trade_id"])
        if self.last_id is None or identity != self.last_id + 1:
            raise ValueError("成交编号断档或重复")
        if self.last_time is not None and time < self.last_time:
            raise ValueError("成交时间倒序")
        price, size = float(message["price"]), float(message["size"])
        if not math.isfinite(price) or not math.isfinite(size) or min(price, size) <= 0:
            raise ValueError("成交价格或量能无效")
        self.last_id, self.last_time = identity, time
        stamp = time.floor("min")
        if stamp < self.start:
            return
        if stamp <= self.sealed:
            raise ValueError("已封存分钟收到迟到成交")
        if stamp >= self.start:
            self.trades.setdefault(stamp, []).append({"trade_id": identity, "time": time.isoformat(),
                "received_at": received.isoformat(), "price": price, "size": size})

    def close_minutes(self, boundary, received):
        results = []
        for stamp in sorted(t for t in self.trades if t < boundary):
            trades = self.trades.pop(stamp)
            prices = [t["price"] for t in trades]
            results.append({"stamp": stamp.isoformat(), "open": prices[0], "high": max(prices), "low": min(prices),
                            "close": prices[-1], "volume": math.fsum(t["size"] for t in trades),
                            "received_at": received.isoformat(), "trades": trades, "live": True})
        self.sealed = max(self.sealed, boundary - pd.Timedelta(minutes=1))
        return results


def frame_from_bars(bars):
    frame = pd.DataFrame(bars)
    frame.index = pd.to_datetime(frame.pop("stamp"), utc=True)
    result = frame[["open", "high", "low", "close", "volume"]].sort_index()
    validate_minutes(result)
    return result


def prospective_signal(signal, bars, started, generated):
    available = pd.Timestamp(signal["available_at"])
    prefix = [b for b in bars if pd.Timestamp(signal["bar_start"]) <= pd.Timestamp(b["stamp"]) < available]
    timely = all(b["live"] and 0 <= (pd.Timestamp(b["received_at"]) - pd.Timestamp(b["stamp"]) -
                                    pd.Timedelta(minutes=1)).total_seconds() <= POLICY["max_lag_seconds"] for b in prefix)
    expected = pd.date_range(pd.Timestamp(signal["bar_start"]), available, freq="min", inclusive="left")
    complete = [pd.Timestamp(b["stamp"]) for b in prefix] == list(expected)
    eligible = (bool(prefix) and complete and timely and pd.Timestamp(signal["bar_start"]) >= started.ceil("15min") and
                0 <= (generated - available).total_seconds() <= POLICY["max_lag_seconds"])
    return {"signal": signal, "generated_at": generated.isoformat(), "prospective_eligible": eligible,
            "status": "candidate" if eligible else "late_or_context_only",
            "entry_not_before": (generated.floor("min") + pd.Timedelta(minutes=2)).isoformat() if eligible else None,
            "deadline": (available + pd.Timedelta(hours=1)).isoformat(),
            "user_delivered_at": None, "entry_price": None, "manual_profit_estimated": False}


def record_candidates(path, bars, started, existing=None, writer=None):
    frame, generated = frame_from_bars(bars), now()
    start = max(started.ceil("15min"), frame.index[-1].floor("15min") - pd.Timedelta(minutes=15))
    signals, _ = scan_failed_range(frame, start, frame.index[-1] + pd.Timedelta(minutes=1, nanoseconds=1))
    existing = {v["key"] for v in audit(path)} if existing is None else existing
    writer = (lambda key, kind, payload: write_event(path, key, kind, payload)) if writer is None else writer
    written = []
    for signal in signals:
        key = "signal:" + signal["episode_id"]
        if key in existing:
            continue
        snapshot = prospective_signal(signal, bars, started, generated)
        if snapshot["entry_not_before"] and now() >= pd.Timestamp(snapshot["entry_not_before"]):
            raise ValueError("落盘前已过预定入场，不得写成前瞻")
        writer(key, "signal", snapshot)
        written.append(snapshot)
    return written


def settle(snapshot, bars, as_of):
    if not snapshot["prospective_eligible"]:
        return {"status": "not_eligible"}
    end, entry = pd.Timestamp(snapshot["deadline"]), pd.Timestamp(snapshot["entry_not_before"])
    if as_of < end + pd.Timedelta(seconds=POLICY["max_lag_seconds"]):
        return {"status": "pending"}
    chosen = [b for b in bars if entry <= pd.Timestamp(b["stamp"]) < end]
    expected = pd.date_range(entry, end, freq="min", inclusive="left")
    if (len(chosen) != len(expected) or any(not b["live"] or not 0 <= (pd.Timestamp(b["received_at"]) -
            pd.Timestamp(b["stamp"]) - pd.Timedelta(minutes=1)).total_seconds() <= POLICY["max_lag_seconds"] for b in chosen)):
        return {"status": "unverified", "reason": "missing_or_late_live_evidence"}
    frame = frame_from_bars(chosen)
    if not frame.index.equals(expected) or (frame.volume <= 0).any():
        return {"status": "unverified", "reason": "minute_grid_or_volume"}
    price, signal = float(frame.open.iloc[0]), snapshot["signal"]
    cases = {f"{t}/{d}": {n: space_probe(frame, price, signal["atr"], side, t, d)
             for n, side in (("forward", signal["direction"]), ("reverse", -signal["direction"]))} for t in TARGETS for d in DURATIONS}
    return {"status": "evaluated", "entry": price, "cases": cases, "manual_profit_estimated": False}
