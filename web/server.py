#!/usr/bin/env python3
"""vtbrooks 只读观察页。浏览器只读快照，不持有交易凭证，也没有下单接口。"""
import json
import base64
import hashlib
import struct
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vt_vote_bot as bot  # noqa: E402
from analysis.tv_indicators.adx_di import adx_di  # noqa: E402
from web.seven_signals import read_snapshot as research_snapshot  # noqa: E402
from macro_expectations import snapshot as macro_snapshot  # noqa: E402

HOST = "127.0.0.1"
PORT = 8423
STATIC = Path(__file__).resolve().parent / "static"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


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


def websocket_price(symbol):
    snap = read_json("ethusdc_spot_snapshot.json", {})
    bars = snap.get(f"{symbol}:15m") or snap.get(f"{symbol}:4h")
    try:
        return float(bars[-1][4]) if bars else None
    except (IndexError, TypeError, ValueError):
        return None


def price_action_plan(df, direction):
    if df is None or df.empty or direction not in ("LONG", "SHORT"):
        return None
    try:
        px = float(df["close"].iloc[-1])
        atr = float((df["high"] - df["low"]).tail(14).mean())
        hi, lo = float(df["high"].tail(20).max()), float(df["low"].tail(20).min())
        if not px or not atr:
            return None
        risk = max(atr * 0.8, px * 0.002)
        if direction == "LONG":
            sl = min(lo - atr * 0.15, px - risk)
            tp1 = hi if hi > px else px + risk
            tp2 = px + max(hi - lo, risk * 2)
            rule = "回调守住结构低点，或放量突破后回踩不破再做多"
        else:
            sl = max(hi + atr * 0.15, px + risk)
            tp1 = lo if lo < px else px - risk
            tp2 = px - max(hi - lo, risk * 2)
            rule = "反抽压住结构高点，或跌破后反抽不收回再做空"
        return {"entry": round(px, 2), "sl": round(sl, 2), "tp1": round(tp1, 2),
                "tp2": round(tp2, 2), "rule": rule}
    except (IndexError, TypeError, ValueError):
        return None


def plan_conflict(brooks, direction):
    """Brooks 当前结构明确反向时，不输出会误导用户的具体交易计划。"""
    quality = ((brooks or {}).get("evidence") or {}).get("quality") or {}
    if quality.get("blocked"):
        return quality.get("message") or "K线质量不足，暂停入场确认"
    if direction not in ("LONG", "SHORT"):
        return None
    expected = 1 if direction == "LONG" else -1
    deep = (brooks or {}).get("deep") or {}
    if deep.get("quality") == -expected:
        return "AI方向与当前强信号K冲突，等待同向信号K确认"
    if (brooks or {}).get("always_in") == -expected:
        return "AI方向与当前Brooks主导方向冲突，等待结构翻转确认"
    return None


def snapshot():
    symbol = "ETHUSDC"
    ctx4 = bot.compute_4h_context(symbol) or {}
    df15 = bot.fetch_klines(symbol, "15m", 120)
    ba15 = bot.brooks_analyze(df15) if not df15.empty else {}
    df4 = bot.fetch_klines(symbol, "4h", 100)
    adx4 = adx_value(bot.fetch_klines(symbol, "4h", 100)) if ctx4 else None
    adx15 = adx_value(df15)
    levels = bot.compute_levels(symbol, "LONG") or {}
    price = websocket_price(symbol)
    if price is None:
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
    conflict4 = plan_conflict(ctx4.get("brooks", {}), direction4)
    conflict15 = plan_conflict(ba15, direction15)
    plan4 = None if conflict4 else price_action_plan(df4, direction4)
    plan15 = None if conflict15 else price_action_plan(df15, direction15)
    news = read_json("news_memory.json", {})
    return clean({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "price": price,
        "macro_events": macro_snapshot(),
        "4h": {"direction": direction4, "confidence": latest.get("conf4h"),
               "latest_reasons": latest.get("reasons", [])[:2], "summary": latest.get("summary4h", ""),
               "context": ctx4, "brooks": ctx4.get("brooks", {}),
               "plan": plan4, "plan_note": conflict4,
               "levels": (ctx4.get("wyckoff") or {}),
               "indicators": {"RSI": rsi4, "RSI历史": rsi_history(rsi4, stats.get("rsi")),
                              "EMA排列": "多头" if ctx4.get("trend_up") else "空头" if ctx4.get("trend_dn") else "纠缠",
                              "ADX": adx4,
                              "挤压分位": ctx4.get("squeeze_pct"), "ATR": ctx4.get("atr4h_pct"),
                              "量比": ctx4.get("vol_ratio4h"), "Brooks": ctx4.get("brooks", {}).get("deep", {})}},
        "15m": {"direction": direction15, "confidence": latest.get("conf15m"),
                "latest_reasons": latest.get("reasons", [])[-2:], "summary": latest.get("summary15m", ""),
                "brooks": ba15, "levels": levels15,
                "plan": plan15, "plan_note": conflict15,
                "indicators": {"RSI": rsi15, "RSI历史": rsi_history(rsi15, stats.get("m15", {}).get("rsi")),
                               "EMA20": levels15.get("ema20"), "EMA50": levels15.get("ema50"),
                               "ADX": adx15,
                               "量比": levels15.get("vol_ratio"), "VWAP": bot.compute_vwap(symbol),
                               "Brooks": ba15.get("deep", {})}},
        "journal": journal[-8:],
        "research_signals": research_snapshot(),
        "news": (news.get("world_view") or {}).get("card") or "暂无盘面综述",
        "health": {"klines_15m": len(df15), "context_4h": bool(ctx4)},
    })


def websocket_frame(payload):
    body = json.dumps(payload, ensure_ascii=False).encode()
    size = len(body)
    if size < 126:
        header = bytes((0x81, size))
    elif size < 65536:
        header = bytes((0x81, 126)) + struct.pack("!H", size)
    else:
        header = bytes((0x81, 127)) + struct.pack("!Q", size)
    return header + body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        if self.path == "/ws":
            self.serve_websocket()
            return
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

    def serve_websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
            self.send_error(400)
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        try:
            while True:
                self.wfile.write(websocket_frame(snapshot()))
                self.wfile.flush()
                time.sleep(2)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
