#!/usr/bin/env python3
"""vtbrooks 只读观察页。浏览器只读快照，不持有交易凭证，也没有下单接口。"""
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vt_vote_bot as bot  # noqa: E402

HOST = "127.0.0.1"
PORT = 8423
STATIC = Path(__file__).resolve().parent / "static"


def read_json(name, fallback):
    try:
        with (ROOT / name).open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return fallback


def clean(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def snapshot():
    symbol = "ETHUSDC"
    ctx4 = bot.compute_4h_context(symbol) or {}
    df15 = bot.fetch_klines(symbol, "15m", 120)
    ba15 = bot.brooks_analyze(df15) if not df15.empty else {}
    levels = bot.compute_levels(symbol, "LONG") or {}
    price = bot.fetch_fast_price(symbol)
    if price is None and not df15.empty:
        price = float(df15["close"].iloc[-1])
    journal = read_json("judge_journal.json", {}).get("entries", [])
    news = read_json("news_memory.json", {})
    return clean({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "price": price,
        "4h": {"context": ctx4, "brooks": ctx4.get("brooks", {}),
               "levels": (ctx4.get("wyckoff") or {})},
        "15m": {"brooks": ba15, "levels": levels},
        "journal": journal[-8:],
        "news": (news.get("world_view") or {}).get("card") or "暂无盘面综述",
        "health": {"klines_15m": len(df15), "context_4h": bool(ctx4)},
    })


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/snapshot":
            self.send_bytes(json.dumps(snapshot(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return
        relative = "index.html" if self.path == "/" else self.path.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content_type = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                        ".js": "text/javascript; charset=utf-8"}.get(target.suffix, "application/octet-stream")
        self.send_bytes(target.read_bytes(), content_type)


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
