"""审计冻结回放中的结构分歧；分类不读取未来收益，不冒充人工形态真值。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.build_price_action_replay import load_pair
from price_action_replay import inspect_bar

BASELINE_REF = "196fe32"
DIR = {"trend_up": 1, "trend_down": -1, "range": 0}


def classify(before, after, clean_input):
    """每种原因是证据标签，只有逻辑矛盾才称为缺陷。"""
    disagreement = DIR[before["legacy"]["state"]] != before["structure"]["direction"]
    quality = after["quality"]
    if quality["blocked"]:
        category = "current_input_unreliable"
    elif not clean_input:
        category = "sparse_history"
    elif after["structure"].get("broken"):
        category = "broken_structure"
    elif disagreement:
        category = "definition_difference"
    else:
        category = "aligned"
    return {"baseline_disagreement": disagreement, "category": category,
            "current_input_blocked": quality["blocked"], "clean_input_120": bool(clean_input),
            "broken_structure": bool(after["structure"].get("broken")),
            "human_label": None, "review_status": "needs_semantic_review" if disagreement else "not_adjudicated"}


def readiness(frame):
    bad = (frame.high <= frame.low) | (frame.volume <= 0)
    usable = bad.rolling(120, min_periods=120).sum().eq(0)
    return usable, {"bars": len(frame), "flat_bars": int((frame.high == frame.low).sum()),
                    "zero_volume_bars": int((frame.volume == 0).sum()),
                    "clean_input_120_count": int(usable.sum()),
                    "rule": "仅历史120根连续且均有振幅/成交量；不读取未来，不等于有交易优势",
                    "monthly": {str(month): {"bars": len(group), "clean_input_120": int(usable.loc[group.index].sum())}
                                for month, group in frame.groupby(frame.index.strftime("%Y-%m"))}}


def audit(frame_by_tf, baseline):
    output = {"baseline_ref": BASELINE_REF, "method": "deterministic_audit_not_expert_labels",
              "source_sha256": baseline["source_sha256"], "timeframes": {}, "cases": []}
    for tf, group in baseline["timeframes"].items():
        frame = frame_by_tf[tf]
        usable, metrics = readiness(frame)
        for clip_id, clip in enumerate(group["clips"]):
            for step, before in enumerate(clip["snapshots"]):
                timestamp = pd.Timestamp(before["time"])
                index = frame.index.get_loc(timestamp)
                after = inspect_bar(frame, index, tf)
                category = classify(before, after, usable.iloc[index])
                errors = []
                if after["price"] != float(frame.close.iloc[index]):
                    errors.append("price_mismatch")
                if any(pd.Timestamp(p["confirmed_at"]) > timestamp for p in after["swings"]):
                    errors.append("unconfirmed_swing")
                if after["quality"]["blocked"] and after["candidates"]:
                    errors.append("unreliable_candidate")
                output["cases"].append({"id": f"{tf}:{timestamp.isoformat()}", "timeframe": tf,
                                        "clip": clip_id, "step": step, "time": timestamp.isoformat(), **category,
                                        "invariant_errors": errors, "before_structure": before["structure"],
                                        "legacy_state": before["legacy"]["state"], "after_structure": after["structure"],
                                        "quality": after["quality"], "swings": after["swings"]})
        selected = [row for row in output["cases"] if row["timeframe"] == tf]
        metrics["audit_points"] = len(selected)
        metrics["baseline_disagreements"] = sum(r["baseline_disagreement"] for r in selected)
        metrics["disagreement_categories"] = dict(Counter(r["category"] for r in selected if r["baseline_disagreement"]))
        metrics["invariant_errors"] = sum(bool(r["invariant_errors"]) for r in selected)
        output["timeframes"][tf] = metrics
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    html = subprocess.run(["git", "show", f"{BASELINE_REF}:web/static/replay.html"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout
    baseline = json.loads(html.split('id="replay-data" type="application/json">', 1)[1].split("</script>", 1)[0])
    if hashlib.sha256(Path(args.csv).read_bytes()).hexdigest() != baseline["source_sha256"]:
        raise ValueError("缓存哈希与冻结回放不匹配，禁止混用数据")
    result = audit(load_pair(args.csv), baseline)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["timeframes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
