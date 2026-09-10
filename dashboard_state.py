"""展示用最新判决；不改变历史判例去重或技术决策归档口径。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

CURRENT_FILE = Path(__file__).with_name("dashboard_decisions.json")


def decision_layer(judge, previous, shown_at):
    reasons = judge.get("reasons") or []
    reasons = [str(x) for x in reasons[:4]] if isinstance(reasons, list) else []
    meta = judge.get("decision_meta") or {}
    result = {"direction": judge.get("direction"), "confidence": judge.get("confidence"),
              "summary": judge.get("summary", ""), "latest_reasons": reasons,
              "decision_at": meta.get("available_at"), "decision_id": meta.get("id"),
              "shown_at": shown_at, "source": "latest_display_after_macro_veto"}
    keys = ("direction", "summary", "latest_reasons", "decision_id")
    changed = previous and any(result[k] != previous.get(k) for k in keys)
    if changed:
        same = result["direction"] == previous.get("direction")
        result.update(change="方向维持，依据已更新" if same else "方向已变化，请核对新依据",
                      previous_summary=previous.get("summary", ""))
    else:
        result.update(change=previous.get("change", "首次展示，暂无前次对照"),
                      previous_summary=previous.get("previous_summary", ""))
    return result


def save_current(symbol, judge4, judge15, path=CURRENT_FILE):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {"version": 1, "symbols": {}}
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("symbols"), dict):
        raise ValueError("展示快照格式不符")
    previous = data["symbols"].get(symbol, {})
    if not isinstance(previous, dict) or any(not isinstance(previous.get(tf, {}), dict) for tf in ("4h", "15m")):
        raise ValueError("展示周期数据格式不符")
    shown_at = datetime.now(timezone.utc).isoformat()
    data["symbols"][symbol] = {tf: decision_layer(judge, previous.get(tf, {}), shown_at)
                               for tf, judge in (("4h", judge4), ("15m", judge15))}
    # 原子替换避免网页刚好读取到半份JSON；文件权限沿用仅进程用户可读写。
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".dashboard-", delete=False) as out:
            name = out.name
            json.dump(data, out, ensure_ascii=False, allow_nan=False)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def display_layers(current, latest, symbol):
    result = {}
    symbols = current.get("symbols", {}) if isinstance(current, dict) and current.get("version") == 1 else {}
    saved = symbols.get(symbol, {}) if isinstance(symbols, dict) else {}
    saved = saved if isinstance(saved, dict) else {}
    for tf, suffix in (("4h", "4h"), ("15m", "15m")):
        layer = saved.get(tf)
        if isinstance(layer, dict) and layer.get("source") == "latest_display_after_macro_veto":
            result[tf] = layer
            continue
        # 旧版最多保存4h两条与15m一条；不足三条无法反推周期归属。
        combined = latest.get("reasons") or []
        reasons = latest.get("reasons" + suffix)
        if not isinstance(reasons, list):
            reasons = (combined[:2] if tf == "4h" else combined[2:3]) if len(combined) == 3 else []
        meta = (latest.get("decision_meta") or {}).get(tf) or {}
        result[tf] = {"direction": latest.get("dir" + suffix), "confidence": latest.get("conf" + suffix),
                      "summary": latest.get("summary" + suffix, ""), "latest_reasons": reasons,
                      "decision_at": meta.get("available_at"), "shown_at": None,
                      "source": "legacy_journal", "change": "历史记录回退，非最新逐次判决", "previous_summary": ""}
    return result
