"""本地独立判决归档；保留当时材料，避免缓存扫描被当成新AI预测。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


def stamp_decision(decision, timeframe, brief):
    if timeframe not in ("4h", "15m"):
        raise ValueError("不支持的判决周期")
    result = dict(decision)
    result["decision_meta"] = {"id": uuid.uuid4().hex, "timeframe": timeframe,
                               "available_at": datetime.now(timezone.utc).isoformat(),
                               "brief_sha256": hashlib.sha256(brief.encode()).hexdigest()}
    return result


def archive_decision(path, symbol, decision, brief):
    metadata = decision["decision_meta"]
    payload = json.dumps({"symbol": symbol, "decision": decision, "brief": brief,
                          "scope": "technical_decision_before_macro_veto"}, ensure_ascii=False, allow_nan=False)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, available_at TEXT NOT NULL, payload TEXT NOT NULL)")
        database.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?)",
                         (metadata["id"], metadata["available_at"], payload))


def evaluation_entries(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        rows = database.execute("SELECT payload FROM decisions ORDER BY available_at").fetchall()
    entries = []
    for (payload,) in rows:
        stored = json.loads(payload)
        decision = stored["decision"]
        metadata = decision["decision_meta"]
        timeframe = metadata["timeframe"]
        entries.append({"symbol": stored["symbol"], "time": metadata["available_at"],
                        "dir4h" if timeframe == "4h" else "dir15m": decision.get("direction"),
                        "decision_meta": {timeframe: metadata}, "evaluation_scope": stored["scope"]})
    return entries
