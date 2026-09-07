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
from analysis.tv_indicators.adx_di import adx_di  # noqa: E402

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


def adx_value(df):
    try:
        if df is None or len(df) < 30:
            return None
        return float(adx_di(df)[0].iloc[-1])
    except (IndexError, TypeError, ValueError):
        return None


def snapshot():
    symbol = "ETHUSDC"
    ctx4 = bot.compute_4h_context(symbol) or {}
    df15 = bot.fetch_klines(symbol, "15m", 120)
    ba15 = bot.brooks_analyze(df15) if not df15.empty else {}
    adx4 = adx_value(bot.fetch_klines(symbol, "4h", 100)) if ctx4 else None
    adx15 = adx_value(df15)
    levels = bot.compute_levels(symbol, "LONG") or {}
    price = bot.fetch_fast_price(symbol)
    if price is None and not df15.empty:
        price = float(df15["close"].iloc[-1])
    journal = read_json("judge_journal.json", {}).get("entries", [])
    latest = next((x for x in reversed(journal) if x.get("symbol") == symbol), {})
    stats = (read_json("market_stats.json", {}).get(symbol, {}) or {}).get("stats", {})
    rsi4 = float(ctx4.get("rsi4h", 0))
    rsi15 = float((bot.compute_levels(symbol, "LONG") or {}).get("rsi14", 0))

    def rsi_band(value):
        return "<=30" if value <= 30 else ">=70" if value >= 70 else None

    def rsi_history(value, table):
        band = rsi_band(value)
        row = (table or {}).get(band) if band else None
        if not row:
            return {"text": "该 RSI 区间暂无统计", "sample": 0}
        return {"text": f"样本 {row.get('n', 0)} · 反弹 {row.get('bounce', '—')}% · 回落 {row.get('drop', '—')}%",
                "sample": row.get("n", 0)}

    levels15 = bot.compute_levels(symbol, "LONG") or {}
    direction4 = latest.get("dir4h")
    direction15 = latest.get("dir15m")
    news = read_json("news_memory.json", {})
    return clean({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "price": price,
        "4h": {"direction": direction4, "confidence": latest.get("conf4h"),
               "latest_reasons": latest.get("reasons", [])[:2],
               "context": ctx4, "brooks": ctx4.get("brooks", {}),
               "levels": (ctx4.get("wyckoff") or {}),
               "indicators": {"RSI": rsi4, "RSI历史": rsi_history(rsi4, stats.get("rsi")),
                              "EMA排列": "多头" if ctx4.get("trend_up") else "空头" if ctx4.get("trend_dn") else "纠缠",
                              "ADX": adx4,
                              "挤压分位": ctx4.get("squeeze_pct"), "ATR": ctx4.get("atr4h_pct"),
                              "量比": ctx4.get("vol_ratio4h"), "Brooks": ctx4.get("brooks", {}).get("deep", {})}},
        "15m": {"direction": direction15, "confidence": latest.get("conf15m"),
                "latest_reasons": latest.get("reasons", [])[-2:],
                "brooks": ba15, "levels": levels15,
                "indicators": {"RSI": rsi15, "RSI历史": rsi_history(rsi15, stats.get("m15", {}).get("rsi")),
                               "EMA20": levels15.get("ema20"), "EMA50": levels15.get("ema50"),
                               "ADX": adx15,
                               "量比": levels15.get("vol_ratio"), "VWAP": bot.compute_vwap(symbol),
                               "Brooks": ba15.get("deep", {})}},
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
