"""分页下载 OKX 现货历史K线，供机会模型离线训练。"""
import argparse
import time

import pandas as pd
import requests


def fetch(symbol, timeframe, years):
    start = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * years)).timestamp() * 1000)
    cursor = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = []
    while cursor > start:
        response = requests.get("https://www.okx.com/api/v5/market/history-candles",
                                params={"instId": symbol, "bar": timeframe, "after": cursor, "limit": 300},
                                headers={"User-Agent": "vtbrooks/1.0"}, timeout=(5, 20))
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        oldest = int(data[-1][0])
        if oldest >= cursor:
            break
        cursor = oldest - 1
        if len(rows) % 9000 == 0:
            print(f"已下载 {len(rows)} 根，至 {pd.Timestamp(oldest, unit='ms')}", flush=True)
        time.sleep(.08)
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "vol_ccy", "quote", "confirm"])
    frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms")
    frame = frame[frame["ts"] >= pd.Timestamp(start, unit="ms")]
    for column in ("open", "high", "low", "close", "vol"):
        frame[column] = frame[column].astype(float)
    return frame[["ts", "open", "high", "low", "close", "vol"]].drop_duplicates("ts").sort_values("ts")


def fetch_coinbase(symbol, timeframe, years):
    seconds = {"5m": 300, "15m": 900, "1h": 3600}[timeframe]
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    start = end - pd.Timedelta(days=365 * years)
    rows = []
    cursor = end
    session = requests.Session()
    while cursor > start:
        begin = max(start, cursor - pd.Timedelta(days=3))
        response = None
        for attempt in range(5):
            try:
                response = session.get(f"https://api.exchange.coinbase.com/products/{symbol}/candles",
                                       params={"granularity": seconds, "start": begin.isoformat(), "end": cursor.isoformat()},
                                       headers={"User-Agent": "vtbrooks/1.0"}, timeout=(5, 20))
                break
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        if response is None:
            raise RuntimeError(f"Coinbase连续失败: {begin} ~ {cursor}")
        if response.status_code == 429:
            time.sleep(2)
            continue
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        rows.extend(data)
        cursor = pd.Timestamp(min(row[0] for row in data), unit="s", tz="UTC")
        if len(rows) % 9000 < 300:
            print(f"已下载 {len(rows)} 根，至 {cursor}", flush=True)
        time.sleep(.06)
    frame = pd.DataFrame(rows, columns=["epoch", "low", "high", "open", "close", "vol"])
    frame["ts"] = pd.to_datetime(frame["epoch"], unit="s")
    return frame[["ts", "open", "high", "low", "close", "vol"]].drop_duplicates("ts").sort_values("ts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH-USDC")
    parser.add_argument("--source", choices=("okx", "coinbase"), default="okx")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (fetch_coinbase if args.source == "coinbase" else fetch)(args.symbol, args.timeframe, args.years)
    result.to_csv(args.output, index=False)
    print(f"完成 {len(result)} 根：{result.ts.min()} ~ {result.ts.max()}")
