"""ws_kline.py — 多交易所 K 线 WebSocket 实时缓冲（KKtrading / vtbrooks 共用）。

用法: python ws_kline.py --exchange binance --symbol ETHUSDT --intervals 5m,15m,4h
落盘: <--snapshot 路径>/kline_snapshot.json（每次 bar 更新原子写）

交易所适配（按服务器区域选择）:
- binance:   wss://fstream.binance.com/ws/{sym}@kline_{iv}   （全球/日本/海外）
- binanceus: wss://stream.binance.us:9443/ws/{sym}@kline_{iv} （美国，binance.us 协议）
- coinbase:  wss://ws-feed.exchange.coinbase.com               （美国，candles channel）
- okx:       wss://ws.okx.com:8443/ws/v5/public                （美国可用）

历史数据由各交易所 REST 拉取，启动时填充缓冲；ws 消息增量更新。
"""

import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
import urllib.request

import websockets

INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# ── 各交易所 REST 历史 + WS 订阅配置 ──
CONFIGS = {
    # 日本/海外服务器：Binance 全球现货 ws（fstream 合约流实测不推消息，用现货流）
    "binance": {
        "rest": "https://api.binance.com/api/v3/klines?symbol={sym}&interval={iv}&limit=300",
        "ws": "wss://stream.binance.com:9443/ws/{sym}@kline_{iv}",
        "parse": "binance",
    },
    # 美国服务器：binance.us（美区可用）
    "binanceus": {
        "rest": "https://api.binance.us/api/v3/klines?symbol={sym}&interval={iv}&limit=300",
        "ws": "wss://stream.binance.us:9443/ws/{sym}@kline_{iv}",
        "parse": "binance",
    },
    # 美国服务器：Hyperliquid ws（实测可用；binance.us wss 不通、coinbase candles channel 无效）
    "hyperliquid": {
        "rest": "https://api.hyperliquid.xyz/info",
        "ws": "wss://api.hyperliquid.xyz/ws",
        "parse": "hl",
    },
    # 备选（channel 格式待适配，当前未启用）
    "coinbase": {
        "rest": "https://api.exchange.coinbase.com/products/{sym}/candles?granularity={sec}&limit=300",
        "ws": "wss://ws-feed.exchange.coinbase.com",
        "parse": "coinbase",
    },
    "okx": {
        "rest": "https://www.okx.com/api/v5/market/candles?instId={sym}&bar={iv}&limit=300",
        "ws": "wss://ws.okx.com:8443/ws/v5/public",
        "parse": "okx",
    },
}

# 各交易所 symbol 格式差异
SYMBOL_MAP = {
    "binance": {"ETHUSDT": "ethusdt", "BTCUSDT": "btcusdt"},
    "binanceus": {"ETHUSDT": "ethusdt", "BTCUSDT": "btcusdt"},
    "coinbase": {"ETHUSDT": "ETH-USD", "BTCUSDT": "BTC-USD"},
    "okx": {"ETHUSDT": "ETH-USDT", "BTCUSDT": "BTC-USDT",
            "ETHUSDC": "ETH-USDC", "BTCUSDC": "BTC-USDC"},
    "hyperliquid": {"ETHUSDT": "ETH", "BTCUSDT": "BTC",
                    "ETHUSDC": "ETH", "BTCUSDC": "BTC"},
}


class KlineBuffer:
    """每 symbol×interval 一个 K 线缓冲：{key: [ [open_ms,o,h,l,c,v,taker_buy], ... ]}"""

    def __init__(self, snapshot_path: str):
        self.snapshot_path = snapshot_path
        self.bars: dict[str, list] = {}
        self._last_saved = 0.0

    def set_initial(self, key: str, bars: list) -> None:
        self.bars[key] = bars

    def upsert(self, key: str, open_ms: int, o, h, l, c, v, taker_buy, closed: bool) -> None:
        buf = self.bars.setdefault(key, [])
        if buf and buf[-1][0] == open_ms:
            buf[-1] = [open_ms, o, h, l, c, v, taker_buy]
        else:
            buf.append([open_ms, o, h, l, c, v, taker_buy])
            if len(buf) > 400:
                del buf[:50]
        # 网页实时流读取该快照；每秒落盘兼顾实时性与磁盘写入压力。
        if closed or time.time() - self._last_saved >= 1:
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        lock_path = self.snapshot_path + ".lock"
        tmp = f"{self.snapshot_path}.{os.getpid()}.tmp"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = {}
            try:
                with open(self.snapshot_path) as current:
                    merged = json.load(current)
            except (OSError, ValueError):
                pass
            merged.update(self.bars)
            with open(tmp, "w") as f:
                json.dump(merged, f)
            os.replace(tmp, self.snapshot_path)
        self._last_saved = time.time()


def _http_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _fetch_history(cfg: dict, sym: str, iv: str) -> list:
    """REST 拉历史 K 线 → 标准 [open_ms,o,h,l,c,v,taker_buy] 列表。"""
    ms = INTERVAL_MS[iv]
    if cfg["parse"] == "hl":
        # HL candleSnapshot 为 POST JSON
        req = urllib.request.Request(
            cfg["rest"],
            data=json.dumps({"type": "candleSnapshot", "req": {
                "coin": SYMBOL_MAP["hyperliquid"].get(sym, sym),
                "interval": iv, "startTime": int(time.time() * 1000) - ms * 300,
            }}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        out = [[int(row["t"]), float(row["o"]), float(row["h"]), float(row["l"]),
                float(row["c"]), float(row["v"]), 0.0] for row in data]
        out.sort(key=lambda r: r[0])
        return out
    url = cfg["rest"].format(sym=sym, iv=iv, sec=ms // 1000)
    data = _http_json(url)
    out = []
    for row in data:
        if cfg["parse"] == "binance":
            out.append([int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]), float(row[9])])
        elif cfg["parse"] == "coinbase":
            # coinbase 返回 [[time, low, high, open, close, volume], ...] 倒序
            out.append([int(row[0]) * 1000, float(row[3]), float(row[2]), float(row[1]),
                        float(row[4]), float(row[5]), 0.0])
        elif cfg["parse"] == "okx":
            # [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
            out.append([int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]), 0.0])
    out.sort(key=lambda r: r[0])
    return out


def _parse_ws(exchange: str, msg: dict) -> list | None:
    """ws 消息 → 标准行 [open_ms,o,h,l,c,v,taker_buy,closed] 或 None。"""
    try:
        if exchange in ("binance", "binanceus"):
            k = msg["k"]
            return [int(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]),
                    float(k["c"]), float(k["v"]), float(k["q"]), bool(k["x"])]
        if exchange == "coinbase":
            for candle in msg.get("candles", []):
                # [time, low, high, open, close, volume]
                return [int(candle[0]) * 1000, float(candle[3]), float(candle[2]),
                        float(candle[1]), float(candle[4]), float(candle[5]), 0.0, True]
            return None
        if exchange == "okx":
            for row in msg.get("data", []):
                return [int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]), 0.0, bool(row[8])]
            return None
        if exchange == "hyperliquid":
            d = msg.get("data", {})
            if isinstance(d, dict) and "c" in d:
                return [int(d["t"]), float(d["o"]), float(d["h"]), float(d["l"]),
                        float(d["c"]), float(d["v"]), 0.0, bool(d.get("closed", True))]
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return None


async def _run_symbol(sym: str, intervals: list[str], cfg: dict, exchange: str,
                      buf: KlineBuffer, stop: asyncio.Event) -> None:
    """单个 symbol：拉历史 → 订阅 ws → 增量更新。断线重连（指数退避）。"""
    for iv in intervals:
        key = f"{sym}:{iv}"  # 快照 key：{symbol}:{interval}，供 fetch_klines 直接读
        for attempt in range(3):
            try:
                buf.set_initial(key, _fetch_history(cfg, sym, iv))
                break
            except Exception as e:
                print(f"[ws] {sym} {iv} history fetch fail ({attempt+1}): {e}")
                await asyncio.sleep(2 * (attempt + 1))

    # 订阅构造：binance/HL 每周期一连接；coinbase/okx 单连接多周期
    ws_sym = SYMBOL_MAP[exchange].get(sym, sym)  # ws 用交易所格式（binance 小写、HL 用币名）
    if exchange in ("coinbase", "okx"):
        uris = [cfg["ws"]]
        if exchange == "coinbase":
            subs = [{"type": "subscribe", "channels": [
                {"name": "candles", "product_ids": [ws_sym]}]}]
        else:
            subs = [{"op": "subscribe", "args": [
                {"channel": f"candle{iv}", "instId": ws_sym} for iv in intervals]}]
    elif exchange == "hyperliquid":
        # HL subscription 必须是单个 dict，每周期一连接
        uris = [cfg["ws"]] * len(intervals)
        subs = [{"method": "subscribe", "subscription": {
            "type": "candle", "coin": ws_sym, "interval": iv}} for iv in intervals]
    else:
        uris = [cfg["ws"].format(sym=ws_sym, iv=iv) for iv in intervals]
        subs = [None] * len(uris)

    backoff = 2
    while not stop.is_set():
        for uri, sub in zip(uris, subs):
            try:
                async with websockets.connect(uri, open_timeout=15, ping_interval=20) as ws:
                    backoff = 2
                    print(f"[ws] connected {exchange} {sym} {uri}")
                    if sub is not None:
                        await ws.send(json.dumps(sub))
                    async for raw in ws:
                        msg = json.loads(raw)
                        if exchange in ("binance", "binanceus"):
                            k = msg.get("k")
                            if not k:
                                continue
                            iv = k.get("i", "")
                            row = _parse_ws(exchange, msg)
                            if row:
                                buf.upsert(f"{sym}:{iv}", row[0], *row[1:])
                        else:
                            # HL/OKX/Coinbase 消息自带周期，按实际周期 upsert
                            if exchange == "hyperliquid":
                                iv = (msg.get("data") or {}).get("i", "")
                            elif exchange == "okx":
                                iv = (msg.get("arg") or {}).get("channel", "").replace("candle", "")
                            else:
                                iv = intervals[0]
                            row = _parse_ws(exchange, msg)
                            if row and iv in intervals:
                                buf.upsert(f"{sym}:{iv}", row[0], *row[1:])
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[ws] {exchange} {sym} error: {e}, reconnect in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


async def main(exchange: str, symbols: list[str], intervals: list[str],
               snapshot: str) -> None:
    cfg = CONFIGS[exchange]
    buf = KlineBuffer(snapshot)
    stop = asyncio.Event()
    tasks = [_run_symbol(sym, intervals, cfg, exchange, buf, stop) for sym in symbols]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        stop.set()
        buf.save()
        print("[ws] snapshot saved, exit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", choices=list(CONFIGS), default="binance")
    ap.add_argument("--symbols", default="ETHUSDT")
    ap.add_argument("--intervals", default="5m,15m,4h")
    ap.add_argument("--snapshot", default="data/kline_snapshot.json")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.exchange, args.symbols.split(","),
                         args.intervals.split(","), args.snapshot))
    except KeyboardInterrupt:
        sys.exit(0)
