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
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_audit import evaluation_entries


HORIZONS = {"4h": ("bars4h", (1, 3)), "15m": ("bars15m", (4, 16))}
COST_PCT = 0.10


def load_bars(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    return df


def direction_return(bars, timestamp, direction, horizon):
    if direction not in ("LONG", "SHORT") or bars.empty or horizon < 1:
        return None
    index = pd.to_datetime(bars.index, utc=True)
    at = pd.to_datetime(timestamp, utc=True)
    if pd.isna(at) or len(index) < 2 or index.has_duplicates or not index.is_monotonic_increasing:
        return None
    step = index.to_series().diff().median()
    pos = index.searchsorted(at, side="left")
    end = pos + horizon - 1
    if end >= len(bars) or step <= pd.Timedelta(0):
        return None
    if index[pos] - at > step or index[end] + step > pd.Timestamp.now(tz="UTC"):
        return None
    if not index[pos:end + 1].to_series().diff().iloc[1:].eq(step).all():
        return None
    entry = float(bars.iloc[pos]["open"])
    exit_price = float(bars.iloc[end]["close"])
    if not np.isfinite([entry, exit_price]).all() or min(entry, exit_price) <= 0:
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


def evaluate(entries, bars4h, bars15m, include_legacy=False):
    bars = {"4h": bars4h, "15m": bars15m}
    rows = []
    seen = set()
    for entry in entries:
        for layer, (key, horizons) in HORIZONS.items():
            metadata = (entry.get("decision_meta") or {}).get(layer) or {}
            decision_id = metadata.get("id")
            if not decision_id and not include_legacy:
                continue
            identity = (entry.get("symbol"), layer, decision_id)
            if decision_id and identity in seen:
                continue
            if decision_id:
                seen.add(identity)
            direction = entry.get("dir4h" if layer == "4h" else "dir15m")
            if direction not in ("LONG", "SHORT"):
                continue
            for horizon in horizons:
                ret = direction_return(bars[layer], metadata.get("available_at") or entry.get("time"), direction, horizon)
                if ret is not None:
                    rows.append({"layer": layer, "horizon": horizon, "direction": direction,
                                 "ret": ret, "correct": ret > 0,
                                 "decision_id": decision_id, "legacy_unverified": not bool(decision_id),
                                 "factors": factor_alignment(entry, direction)})
    return rows


def report(rows):
    print(f"已结算窗口={len(rows)}；旧日志窗口={sum(row['legacy_unverified'] for row in rows)}（不可当独立AI样本）")
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--journal")
    source.add_argument("--audit-db", help="独立技术判决归档（宏观否决前），不含缓存重复")
    parser.add_argument("--bars4h", required=True)
    parser.add_argument("--bars15m", required=True)
    parser.add_argument("--include-legacy", action="store_true", help="仅粗略诊断旧日志，不能视为独立判决")
    args = parser.parse_args()
    entries = evaluation_entries(args.audit_db) if args.audit_db else json.loads(Path(args.journal).read_text()).get("entries", [])
    print("范围：宏观否决前独立技术判断" if args.audit_db else "范围：展示日志；缺独立ID的旧记录默认排除")
    report(evaluate(entries, load_bars(args.bars4h), load_bars(args.bars15m), args.include_legacy))


if __name__ == "__main__":
    main()
