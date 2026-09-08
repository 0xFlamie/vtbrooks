"""从同一份15分钟缓存生成双周期离线识图回放页，不调用网络或AI。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from price_action_replay import HORIZONS, WINDOW, replay_clip


def load_pair(path):
    frame = pd.read_csv(path, parse_dates=["ts"]).rename(columns={"vol": "volume"}).set_index("ts")
    frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
    frame.index = pd.to_datetime(frame.index, utc=True)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("原始K线必须已排序且无重复")
    if not np.isfinite(frame.to_numpy()).all() or (frame.iloc[:, :4] <= 0).any().any():
        raise ValueError("无效行情数据")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1)) | (frame.volume < 0)).any():
        raise ValueError("OHLC或成交量无效")
    if not frame.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("15分钟数据存在缺口，先补齐后再回放")
    frame = frame[frame.index + pd.Timedelta(minutes=15) <= pd.Timestamp.now(tz="UTC")]
    four = frame.resample("4h", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), count=("close", "count"))
    four = four.loc[four["count"] == 16].drop(columns="count")
    return {"4h": four, "15m": frame}


def build_payload(path, source, symbol, clips=6, steps=32):
    if clips < 1 or steps < 1:
        raise ValueError("片段数和步数必须为正数")
    pair = load_pair(path)
    result = {"version": "price-action-review-v1", "source": source, "symbol": symbol,
              "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "selection": "按时间等距抽取，未按未来收益或候选命中选择", "timeframes": {}}
    for timeframe, frame in pair.items():
        maximum = len(frame) - steps - HORIZONS[timeframe]
        if maximum < WINDOW - 1 or maximum - WINDOW + 2 < clips:
            raise ValueError(f"{timeframe}数据不足以生成片段")
        starts = np.linspace(WINDOW - 1, maximum, clips, dtype=int)
        result["timeframes"][timeframe] = {"rows": len(frame), "from": frame.index[0].isoformat(),
                                            "to": frame.index[-1].isoformat(),
                                            "flat_bars": int((frame.high == frame.low).sum()),
                                            "zero_volume_bars": int((frame.volume == 0).sum()),
                                            "clips": [replay_clip(frame, int(start), steps, timeframe) for start in starts]}
    return result


def render_html(payload):
    template = (ROOT / "web" / "static" / "replay_template.html").read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return template.replace("__REPLAY_DATA__", encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_payload(args.csv, args.source, args.symbol)
    rendered = render_html(payload)
    with Path(args.output).open("x") as stream:
        stream.write(rendered)
    count = sum(len(clip["snapshots"]) for group in payload["timeframes"].values() for clip in group["clips"])
    print(f"已生成 {args.output}；双周期共{count}个逐根审计点；不调用AI、不接入生产")


if __name__ == "__main__":
    main()
