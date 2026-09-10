"""将冻结的影子账本投影为网页研究卡片；不写账本、不改变信号规则。"""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

LEDGER = Path("/var/lib/vtbrooks-shadow/ledger-v2.sqlite3")
STRATEGY_ID = "BRK-FR-15M-v1"
STRATEGY_NAME = "Brooks 区间假突破 · 15M"
RULE_SHA256 = "ed3377f183285d8272ac09aca4f2bfb52c4fc78c0e3b064e32f01a86b88585a0"
MAX_HEARTBEAT_AGE = 180
RECENT_LIMIT = 20


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("时间缺少时区")
    return result.astimezone(timezone.utc)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("点位无效")
    return value


def checked(row):
    seq, key, payload, checksum, predecessor = row
    value = json.loads(payload)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if (hashlib.sha256(encoded).hexdigest() != checksum or value["key"] != key or
            value["previous_sha256"] != predecessor):
        raise ValueError("展示记录校验失败")
    timestamp(value["recorded_at"])
    return {**value, "seq": seq}


def rows(db, prefix, limit=1):
    query = ("SELECT e.seq,e.event_key,e.payload,e.sha256,p.sha256 FROM events e "
             "LEFT JOIN events p ON p.seq=e.seq-1 WHERE e.event_key>=? AND e.event_key<? "
             "ORDER BY e.seq DESC LIMIT ?")
    return [checked(row) for row in db.execute(query, (prefix, prefix[:-1] + ";", limit))]


def exact(db, key):
    row = db.execute("SELECT e.seq,e.event_key,e.payload,e.sha256,p.sha256 FROM events e "
                     "LEFT JOIN events p ON p.seq=e.seq-1 WHERE e.event_key=?", (key,)).fetchone()
    return checked(row) if row else None


def collector_status(db, at):
    latest = [value for prefix in ("start:", "connected:", "heartbeat:", "error:", "stop:")
              for value in rows(db, prefix)]
    if not latest:
        return {"status": "waiting", "label": "等待采集启动", "last_heartbeat_at": None}
    event = max(latest, key=lambda value: value["seq"])
    heartbeats = [value for value in latest if value["kind"] == "heartbeat"]
    heartbeat = heartbeats[0]["payload"]["received_at"] if heartbeats else None
    fresh = heartbeat is not None and 0 <= (at - timestamp(heartbeat)).total_seconds() <= MAX_HEARTBEAT_AGE
    if event["kind"] == "heartbeat" and fresh:
        status, label = "live", "采集中"
    elif event["kind"] in ("capture_start", "connected") and (at - timestamp(event["recorded_at"])).total_seconds() <= MAX_HEARTBEAT_AGE:
        status, label = "warming", "预热 / 重连中"
    else:
        status, label = "stale", "采集中断或数据过期"
    return {"status": status, "label": label, "last_heartbeat_at": heartbeat}


def result_label(db, key):
    event = exact(db, "settle:" + key)
    if event is None:
        return "后续路径尚未封存"
    if event["kind"] != "settlement":
        raise ValueError("结算类型不符")
    result = event["payload"]
    if result["status"] == "unverified":
        return "后续数据不足，未判定"
    if result["status"] != "evaluated":
        return "未形成有效前瞻结果"
    held = result["cases"]["0.25/3"]["forward"]["held"]
    if held not in (0, 1):
        raise ValueError("路径结果无效")
    return "曾有持续顺向空间（非盈利判定）" if held else "未达研究设定的持续顺向空间"


def project_signal(event, at, live, result):
    snapshot, signal = event["payload"], event["payload"]["signal"]
    if (event["kind"] != "signal" or signal["family"] != "failed_range" or
            type(signal["direction"]) is not int or signal["direction"] not in (-1, 1)):
        raise ValueError("信号身份不符")
    if event["key"] != "signal:" + signal["episode_id"]:
        raise ValueError("事件编号不符")
    generated, available = timestamp(snapshot["generated_at"]), timestamp(signal["available_at"])
    entry, deadline = timestamp(snapshot["entry_not_before"]), timestamp(snapshot["deadline"])
    if (not available <= generated <= at or (generated - available).total_seconds() > 120 or
            not generated <= timestamp(event["recorded_at"]) < entry or
            deadline != available + timedelta(hours=1) or entry != generated.replace(second=0, microsecond=0) + timedelta(minutes=2)):
        raise ValueError("前瞻时序不符")
    status = "ended" if at >= deadline else "new" if live and at < entry else "observing" if live else "unverified"
    side = signal["direction"]
    low, high, observed = (number(signal[key]) for key in ("range_low", "range_high", "observed_price"))
    if not low < observed < high:
        raise ValueError("信号已不在原区间内")
    reason = ("先跌出区间，再收回区间内，随后向上突破回收分钟的高点" if side == 1 else
              "先涨出区间，再回到区间内，随后向下跌破回收分钟的低点")
    return {"id": STRATEGY_ID + ":" + signal["episode_id"], "event_id": signal["episode_id"],
            "direction": "LONG" if side == 1 else "SHORT", "status": status,
            "generated_at": generated.isoformat(), "available_at": available.isoformat(),
            "entry_not_before": entry.isoformat(), "deadline": deadline.isoformat(), "reason": reason,
            "observed_price": number(signal["observed_price"]), "atr": number(signal["atr"]),
            "range_low": number(signal["range_low"]), "range_high": number(signal["range_high"]),
            "structure_reference": number(signal["structure_reference"]), "result": result}


def read_snapshot(path=LEDGER, at=None):
    at = datetime.now(timezone.utc) if at is None else at
    base = {"strategy_id": STRATEGY_ID, "name": STRATEGY_NAME, "source": "Coinbase ETH-USD",
            "timeframe": "15M", "research_only": True, "updated_at": at.isoformat(), "signals": [],
            "history_note": "历史301研究日 / 394次 · 持续顺向空间率68.3%–68.5%，不是盈利胜率",
            "rule_note": "15M区间背景，盘中按完整1M确认：越界 → 回收 → 向区间内推进",
            "source_note": "点位属于Coinbase ETH-USD，不混用主站OKX ETH-USDC报价；不提供自动下单"}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=.25)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            event = exact(db, "protocol")
            header = event["payload"]
            policy = header["policy"]
            if (event["seq"] != 1 or event["kind"] != "protocol" or
                    (policy["version"], policy["venue"], policy["symbol"]) !=
                    ("failed-range-prospective-shadow-v2", "Coinbase", "ETH-USD") or
                    header["implementation_sha256"]["brooks_failed_range.py"] != RULE_SHA256):
                raise ValueError("不是已登记的研究策略")
            collector = collector_status(db, at)
            signals = [project_signal(row, at, collector["status"] == "live", result_label(db, row["key"]))
                       for row in rows(db, "signal:", RECENT_LIMIT) if row["payload"].get("prospective_eligible") is True]
            return {**base, "collector": collector, "signals": signals}
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return {**base, "collector": {"status": "unavailable", "label": "研究数据暂不可用", "last_heartbeat_at": None}}
