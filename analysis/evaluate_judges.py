"""评估生产 AI 判决的方向准确率与 EV。

用法:
  python3 analysis/evaluate_judges.py --journal judge_journal.json \
      --bars4h analysis/tv_indicators/data/okx_4h.csv \
      --bars15m path/to/15m.csv

判决时刻之后第一根 K 线的 open 入场，按未来收盘计算；不修改生产数据。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


HORIZONS = {"4h": ("bars4h", (1, 3)), "15m": ("bars15m", (4, 16))}
COST_PCT = 0.10


def load_bars(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    return df


def direction_return(bars, timestamp, direction, horizon):
    if not direction or bars.empty:
        return None
    pos = bars.index.searchsorted(pd.Timestamp(timestamp), side="right")
    if pos + horizon >= len(bars):
        return None
    entry = float(bars.iloc[pos]["open"])
    exit_price = float(bars.iloc[pos + horizon]["close"])
    if entry <= 0:
        return None
    side = 1 if direction == "LONG" else -1
    return (exit_price / entry - 1) * side * 100


def factor_alignment(entry, direction):
    """返回与 AI 方向同向的已记录因子名，便于做消融比较。"""
    if not direction:
        return []
    wanted = "🟢" if direction == "LONG" else "🔴"
    return [x.get("name") for x in entry.get("factor_evidence", [])
            if x.get("direction") == wanted and x.get("name")]


def evaluate(entries, bars4h, bars15m):
    bars = {"4h": bars4h, "15m": bars15m}
    rows = []
    for entry in entries:
        for layer, (key, horizons) in HORIZONS.items():
            direction = entry.get("dir4h" if layer == "4h" else "dir15m")
            if direction not in ("LONG", "SHORT"):
                continue
            for horizon in horizons:
                ret = direction_return(bars[layer], entry.get("time"), direction, horizon)
                if ret is not None:
                    rows.append({"layer": layer, "horizon": horizon, "direction": direction,
                                 "ret": ret, "correct": ret > 0,
                                 "factors": factor_alignment(entry, direction)})
    return rows


def report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["horizon"])].append(row)
    for key, group in sorted(groups.items()):
        rets = [x["ret"] for x in group]
        print(f"{key[0]} future_{key[1]}bars: n={len(group)} accuracy={sum(x['correct'] for x in group)/len(group):.1%} "
              f"mean={sum(rets)/len(rets):+.3f}% EV={sum(rets)/len(rets)-COST_PCT:+.3f}%")
    factors = defaultdict(list)
    for row in rows:
        for factor in row["factors"]:
            factors[(row["layer"], factor)].append(row)
    print("\n因子同向分组:")
    for (layer, factor), group in sorted(factors.items()):
        acc = sum(x["correct"] for x in group) / len(group)
        ev = sum(x["ret"] for x in group) / len(group) - COST_PCT
        print(f"{layer} {factor}: n={len(group)} accuracy={acc:.1%} EV={ev:+.3f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--bars4h", required=True)
    parser.add_argument("--bars15m", required=True)
    args = parser.parse_args()
    entries = json.loads(Path(args.journal).read_text()).get("entries", [])
    report(evaluate(entries, load_bars(args.bars4h), load_bars(args.bars15m)))


if __name__ == "__main__":
    main()
