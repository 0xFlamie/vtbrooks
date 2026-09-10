"""可恢复的隔离影子采集；只读取公开行情，不加载生产配置或账户。"""
import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import pandas as pd
import requests
from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import connect

from brooks_failed_range_shadow import (POLICY, MinuteStream, append, encode, frame_from_bars,
                                        initialize, iter_audit, now, record_candidates, settle)

ROOT = Path(__file__).resolve().parent
SOURCES = ("brooks_shadow_runtime.py", "brooks_failed_range_shadow.py", "brooks_failed_range.py",
           "brooks_event_research.py", "brooks_first_retest.py", "brooks_intrabar.py",
           "brooks_intrabar_controls.py", "brooks_sequence_research.py", "market_path_model.py")
RUNTIME_POLICY = {**POLICY, "version": "failed-range-prospective-shadow-v2",
                  "recovery": "new_session_new_full_minute_preserve_live_evidence_no_live_backfill",
                  "context": "35h_reload_REST_live_minutes_take_precedence",
                  "integrity": "exclusive_writer_startup_full_audit_transaction_tail_check",
                  "receive_time": "application_recv_processing_time_not_wire_arrival",
                  "coverage": "timely_positive_volume_minutes_over_elapsed_complete_minutes",
                  "automatic_retry": "network_errors_only_fixed_30_seconds",
                  "session_limit_seconds": 3600}
REST_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
WS_URL = "wss://ws-feed.exchange.coinbase.com"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(frozen_path):
    frozen = json.loads(frozen_path.read_text())
    for name, checksum in frozen["implementation_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py" or file_hash(path) != checksum:
            raise ValueError("原研究实现已变化，不能生成影子清单")
    return {"policy": RUNTIME_POLICY, "research_protocol_sha256": file_hash(frozen_path),
            "research_implementation_sha256": frozen["implementation_sha256"],
            "implementation_sha256": {name: file_hash(ROOT / name) for name in SOURCES}}


def verify_header(header):
    if header["policy"] != RUNTIME_POLICY or set(header["implementation_sha256"]) != set(SOURCES):
        raise ValueError("影子协议或实现清单不符；旧账本不能换规则续跑")
    if any(file_hash(ROOT / name) != checksum for name, checksum in header["implementation_sha256"].items()):
        raise ValueError("登记实现变化，须重新登记")


def register(path, manifest):
    verify_header(manifest)
    created = now()
    header = {**manifest, "registered_at": created.isoformat(),
              "review_not_before": (created + pd.Timedelta(days=30)).isoformat()}
    # 排他创建，避免并发注册覆盖已经存在的审计证据。
    with path.open("xb"):
        pass
    initialize(path, header)
    return header


@contextmanager
def exclusive(path):
    with Path(str(path) + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("账本已有采集进程；禁止重复启动") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Journal:
    """调用方持独占锁；启动全审计，写入时检查尾部，避免每分钟重读全部逐笔。"""
    def __init__(self, path):
        self.path, self.header = path, None
        for value in iter_audit(path):
            if self.header is None:
                if value["kind"] != "protocol":
                    raise ValueError("账本首项不是协议")
                self.header = value["payload"]
        if self.header is None:
            raise ValueError("账本为空")
        self.db = sqlite3.connect(path)
        self.tail = self._tail()

    def close(self):
        self.db.close()

    def _tail(self):
        return self.db.execute("SELECT seq,payload,sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()

    def __contains__(self, key):
        return self.db.execute("SELECT 1 FROM events WHERE event_key=?", (key,)).fetchone() is not None

    def write(self, key, kind, payload):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self._tail() != self.tail:
                raise ValueError("账本在独占进程外改变")
            added = append(self.db, key, kind, payload)
        self.tail = self._tail()
        return added

    def minutes(self, start, end):
        rows = self.db.execute("SELECT payload FROM events WHERE event_key>=? AND event_key<? ORDER BY event_key",
                               ("minute:" + start.isoformat(), "minute:" + end.isoformat()))
        return [{k: v for k, v in json.loads(row[0])["payload"].items() if k != "trades"} for row in rows]

    def pending(self):
        rows = self.db.execute("SELECT e.event_key,e.payload FROM events e LEFT JOIN events s "
                               "ON s.event_key='settle:'||e.event_key WHERE e.event_key>='signal:' "
                               "AND e.event_key<'signal;' AND s.seq IS NULL")
        return [(key, json.loads(payload)["payload"]) for key, payload in rows]

    def finalize(self, at):
        for key, snapshot in self.pending():
            due = pd.Timestamp(snapshot["deadline"]) + pd.Timedelta(seconds=POLICY["max_lag_seconds"])
            if snapshot["prospective_eligible"] and at < due:
                continue
            bars = self.minutes(pd.Timestamp(snapshot["entry_not_before"]), pd.Timestamp(snapshot["deadline"])) if snapshot["prospective_eligible"] else []
            result = settle(snapshot, bars, at)
            if result["status"] != "pending":
                self.write("settle:" + key, "settlement", result)

    def recover(self, at):
        rows = self.db.execute("SELECT e.event_key FROM events e LEFT JOIN events s "
                               "ON s.event_key='stop:'||substr(e.event_key,7) LEFT JOIN events r "
                               "ON r.event_key='recovery:'||e.event_key WHERE e.event_key>='start:' AND e.event_key<'start;' "
                               "AND s.seq IS NULL AND r.seq IS NULL").fetchall()
        for (key,) in rows:
            self.write("recovery:" + key, "recovery_detected", {"previous_start": key,
                       "detected_at": at.isoformat(), "actual_disconnect_at": None, "gap_filled_as_live": False})


def parse_context(payload, start, end):
    if not isinstance(payload, list) or any(not isinstance(row, list) or len(row) != 6 for row in payload):
        raise ValueError("历史K线结构无效")
    if not payload:
        return []
    frame = pd.DataFrame(payload, columns=["epoch", "low", "high", "open", "close", "volume"]).astype(float)
    frame["stamp"] = pd.to_datetime(frame.pop("epoch"), unit="s", utc=True)
    frame = frame_from_bars(frame.to_dict("records"))
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    return [{"stamp": t.isoformat(), **row.to_dict(), "live": False} for t, row in frame.iterrows()]


def history(journal, start, end):
    bars = []
    while start < end:
        finish = min(end, start + pd.Timedelta(minutes=240))
        params = {"granularity": 60, "start": start.isoformat(), "end": finish.isoformat()}
        response = requests.get(REST_URL, params=params, headers={"User-Agent": "vtbrooks/1.0"}, timeout=(5, 15))
        response.raise_for_status()
        payload, received = response.json(), now()
        parsed = parse_context(payload, start, finish)
        journal.write("context:" + uuid.uuid4().hex, "context", {"endpoint": REST_URL, "params": params,
                      "payload": payload, "received_at": received.isoformat(), "live": False})
        bars.extend({**bar, "received_at": received.isoformat()} for bar in parsed)
        start = finish
    return bars


def merge_context(context, live, start):
    merged = {bar["stamp"]: bar for bar in context}
    merged.update({bar["stamp"]: bar for bar in live})
    return [merged[key] for key in sorted(merged) if pd.Timestamp(key) >= start]


def process_message(journal, stream, message, bars, session):
    received = now()
    if message.get("type") == "error":
        raise ValueError("WebSocket订阅被拒绝")
    closed = stream.ingest(message, received)
    for bar in closed:
        journal.write("minute:" + bar["stamp"], "minute", bar)
        bars.append({k: v for k, v in bar.items() if k != "trades"})
        record_candidates(journal.path, bars, stream.start, existing=journal, writer=journal.write)
    if message.get("type") == "heartbeat":
        stamp = pd.Timestamp(message["time"]).floor("min").isoformat()
        key = f"heartbeat:{session}:{stamp}"
        if key not in journal:
            journal.write(key, "heartbeat", {"exchange_at": message["time"], "received_at": received.isoformat(),
                          "last_trade_id": stream.last_id, "sealed_through": stream.sealed.isoformat()})
            journal.finalize(received)


async def capture_session(journal, seconds):
    session, start = uuid.uuid4().hex, now()
    journal.write("start:" + session, "capture_start", {"at": start.isoformat(), "session": session})
    end = start.floor("min")
    context = history(journal, end - pd.Timedelta(hours=35), end)
    bars = merge_context(context, journal.minutes(end - pd.Timedelta(hours=35), end), end - pd.Timedelta(hours=35))
    async with connect(WS_URL, open_timeout=10, close_timeout=2, ping_interval=20, max_queue=1024) as ws:
        stream = MinuteStream(now())
        await ws.send(encode({"type": "subscribe", "product_ids": [POLICY["symbol"]], "channels": ["matches", "heartbeat"]}))
        journal.write("connected:" + session, "connected", {"at": now().isoformat(), "session": session,
                      "first_full_minute": stream.start.isoformat()})
        gap_loaded = False
        stop = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < stop:
            remaining = stop - asyncio.get_running_loop().time()
            try:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=min(10, remaining)))
            except asyncio.TimeoutError:
                if asyncio.get_running_loop().time() >= stop:
                    break
                raise
            if (not gap_loaded and message.get("type") == "heartbeat" and
                    pd.Timestamp(message["time"]).floor("min") >= stream.start):
                bars = merge_context(bars + history(journal, end, stream.start), [], end - pd.Timedelta(hours=35))
                gap_loaded = True
            process_message(journal, stream, message, bars, session)
    journal.write("stop:" + session, "capture_stopped", {"at": now().isoformat(), "session": session,
                  "reason": "bounded_session", "last_complete_minute": stream.sealed.isoformat()})


def coverage(path, at):
    counts, first, last_heartbeat, eligible, expected, timely, registered = {}, None, None, 0, 0, 0, None
    for value in iter_audit(path):
        kind, payload = value["kind"], value["payload"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "protocol":
            registered = payload["registered_at"]
        if kind == "connected" and first is None:
            first = pd.Timestamp(payload["first_full_minute"])
        if kind == "heartbeat":
            last_heartbeat = payload["received_at"]
        if kind == "signal":
            eligible += bool(payload["prospective_eligible"])
        if kind == "minute":
            lag = (pd.Timestamp(payload["received_at"]) - pd.Timestamp(payload["stamp"]) - pd.Timedelta(minutes=1)).total_seconds()
            timely += bool(payload["live"] and payload["volume"] > 0 and 0 <= lag <= POLICY["max_lag_seconds"])
    if first is not None:
        expected = max(0, int((at.floor("min") - first) / pd.Timedelta(minutes=1)))
    return {"counts": counts, "registered_at": registered, "first_full_minute": first.isoformat() if first is not None else None,
            "last_heartbeat_at": last_heartbeat, "expected_elapsed_minutes": expected, "timely_live_minutes": timely,
            "missing_or_unverified_minutes": max(0, expected - timely), "eligible_signals": eligible,
            "coverage_ratio": timely / expected if expected else None, "manual_win_rate": None,
            "running": "not_inferred_from_ledger"}


async def run(journal, seconds, continuous):
    journal.recover(now())
    journal.finalize(now())
    while True:
        journal.recover(now())
        journal.finalize(now())
        try:
            await capture_session(journal, seconds)
        except (ConnectionClosed, OSError, asyncio.TimeoutError, requests.RequestException) as exc:
            journal.write("error:" + uuid.uuid4().hex, "capture_failed",
                          {"at": now().isoformat(), "error_type": type(exc).__name__, "retryable": True})
            if not continuous:
                raise
            await asyncio.sleep(30)
        except Exception as exc:
            journal.write("error:" + uuid.uuid4().hex, "capture_failed",
                          {"at": now().isoformat(), "error_type": type(exc).__name__, "retryable": False})
            raise
        if not continuous:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "register", "capture", "audit"), required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= RUNTIME_POLICY["session_limit_seconds"]:
        parser.error("会话长度必须为1–3600秒")
    if args.mode == "freeze" and (args.frozen is None or args.manifest is None):
        parser.error("freeze需要--frozen与--manifest")
    if args.mode != "freeze" and args.ledger is None:
        parser.error("该模式需要--ledger")
    if args.mode == "register" and args.manifest is None:
        parser.error("register需要--manifest")
    if args.mode == "freeze":
        manifest = freeze_manifest(args.frozen)
        with args.manifest.open("x") as target:
            target.write(encode(manifest) + "\n")
        print(encode({"manifest_sha256": file_hash(args.manifest)}))
    elif args.mode == "register":
        print(encode(register(args.ledger, json.loads(args.manifest.read_text()))))
    elif args.mode == "audit":
        print(encode(coverage(args.ledger, now())))
    else:
        with exclusive(args.ledger):
            if not args.ledger.exists() and args.manifest is not None:
                register(args.ledger, json.loads(args.manifest.read_text()))
            journal = Journal(args.ledger)
            try:
                verify_header(journal.header)
                asyncio.run(run(journal, args.seconds, args.continuous))
            finally:
                journal.close()


if __name__ == "__main__":
    main()
