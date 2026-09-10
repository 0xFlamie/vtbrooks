"""事前市场预期：低频公开日历、版本留档、只读投影；不是实时实际值源。"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from functools import lru_cache

import requests
import macro_events as macro

SOURCE = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
SOURCE_LABEL = "Forex Factory周历参考预期"
DATABASE = Path(__file__).with_name("macro_expectations.sqlite3")
REFRESH_SECONDS = 1800
STALE_SECONDS = 7200
METRICS = {
    "CPI m/m": ("cpi", "总体环比"), "CPI y/y": ("cpi", "总体同比"),
    "Core CPI m/m": ("cpi", "核心环比"), "Core CPI y/y": ("cpi", "核心同比"),
    "PPI m/m": ("ppi", "总体环比"), "PPI y/y": ("ppi", "总体同比"),
    "Core PPI m/m": ("ppi", "核心环比"), "Core PPI y/y": ("ppi", "核心同比"),
    "PCE Price Index m/m": ("pce", "总体环比"), "PCE Price Index y/y": ("pce", "总体同比"),
    "Core PCE Price Index m/m": ("pce", "核心环比"), "Core PCE Price Index y/y": ("pce", "核心同比"),
    "Personal Spending m/m": ("pce", "个人支出环比"), "Personal Income m/m": ("pce", "个人收入环比"),
    "Non-Farm Employment Change": ("employment", "非农新增就业"),
    "Unemployment Rate": ("employment", "失业率"), "Average Hourly Earnings m/m": ("employment", "时薪环比"),
    "Unemployment Claims": ("claims", "初请"), "Continuing Jobless Claims": ("claims", "续请"),
    "ADP Non-Farm Employment Change": ("adp", "私人就业"),
    "JOLTS Job Openings": ("jolts", "职位空缺"), "Employment Cost Index q/q": ("eci", "就业成本季环比"),
    "ISM Manufacturing PMI": ("ism_manufacturing", "制造业PMI"),
    "ISM Manufacturing Prices": ("ism_manufacturing", "制造业价格"),
    "ISM Services PMI": ("ism_services", "服务业PMI"),
    "Retail Sales m/m": ("retail", "零售环比"), "Core Retail Sales m/m": ("retail", "剔除汽车环比"),
    "Advance GDP q/q": ("gdp", "GDP初值季环比年率"), "Prelim GDP q/q": ("gdp", "GDP修正值季环比年率"),
    "Final GDP q/q": ("gdp", "GDP终值季环比年率"),
    "Prelim UoM Consumer Sentiment": ("michigan", "信心初值"),
    "Revised UoM Consumer Sentiment": ("michigan", "信心终值"),
    "Prelim UoM Inflation Expectations": ("michigan", "一年通胀预期初值"),
    "Revised UoM Inflation Expectations": ("michigan", "一年通胀预期终值"),
    "Federal Funds Rate": ("fomc", "联邦基金利率"),
}


def utc_now():
    return datetime.now(timezone.utc)


def stamp(value):
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("预期记录必须带时区")
    return parsed.astimezone(timezone.utc)


def value_text(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:%|K|M|B|T)?", value):
        raise ValueError("不支持的预期数字格式")
    return value


def normalise(payload, received_at):
    received = stamp(received_at)
    if not isinstance(payload, list) or len(payload) > 5000:
        raise ValueError("周历不是合理长度的数组")
    result, rejected = {}, 0
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("周历含非对象条目")
        if raw.get("country") != "USD" or raw.get("title") not in METRICS:
            continue
        try:
            kind, label = METRICS[raw["title"]]
            release = stamp(raw["date"])
            if release <= received:
                continue  # 公布后取得的共识不允许伪装成事前记录。
            event = next((e for e in macro.EVENTS if e["kind"] == kind and e["release_at"] == release), None)
            if event is None:
                raise ValueError("与官方日历时间不一致")
            key = (event["id"], raw["title"])
            quote = {"event_id": event["id"], "metric": raw["title"], "label": label,
                     "forecast": value_text(raw.get("forecast")), "previous": value_text(raw.get("previous"))}
            if key in result and result[key] != quote:
                raise ValueError("同批次预期冲突")
            result[key] = quote
        except (ValueError, TypeError, KeyError):
            rejected += 1
    if rejected:
        raise ValueError(f"周历有{rejected}条格式/时间/重复冲突，本批次不更新")
    return list(result.values())


def connect_writer(path):
    db = sqlite3.connect(path, timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS expectation_batches (id INTEGER PRIMARY KEY, received_at TEXT NOT NULL, "
               "status TEXT NOT NULL, detail TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL)")
    return db


def capture(path, payload, received_at):
    received = stamp(received_at).isoformat()
    quotes = normalise(payload, received)
    original = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    stored = json.dumps({"quotes": quotes, "source": SOURCE, "raw": payload}, ensure_ascii=False, allow_nan=False)
    with connect_writer(path) as db:
        db.execute("INSERT INTO expectation_batches(received_at,status,detail,payload,sha256) VALUES (?,?,?,?,?)",
                   (received, "ok", "", stored, hashlib.sha256(original.encode()).hexdigest()))
    return len(quotes)


def refresh(path=DATABASE, http=requests, clock=utc_now):
    now = stamp(clock())
    with connect_writer(path) as db:
        latest = db.execute("SELECT received_at FROM expectation_batches ORDER BY id DESC LIMIT 1").fetchone()
    if latest and (now - stamp(latest[0])).total_seconds() < REFRESH_SECONDS:
        return False
    try:
        response = http.get(SOURCE, timeout=(5, 10))
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("周历响应过大")
        payload = response.json()
        capture(path, payload, clock())
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        # 记录失败并节流；网络异常消息不透传，避免未来客户端配置泄漏。
        with connect_writer(path) as db:
            db.execute("INSERT INTO expectation_batches(received_at,status,detail,payload,sha256) VALUES (?,?,?,?,?)",
                       (stamp(clock()).isoformat(), "error", type(error).__name__, "{}", ""))
        return False
    return True


@lru_cache(maxsize=2)
def _read_cached(path, modified, size):
    # 多个网页客户端共享只读结果，避免每帧重复解析历史原文；文件变化即失效。
    with sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        rows = db.execute("SELECT received_at,status,payload FROM expectation_batches "
                          "ORDER BY received_at DESC,id DESC LIMIT 128").fetchall()
    return [(stamp(t), status, {"quotes": json.loads(data).get("quotes", [])})
            for t, status, data in reversed(rows)]


def read_batches(path, now):
    if not Path(path).exists():
        return [], "not_collected"
    try:
        stat = Path(path).stat()
        rows = _read_cached(str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
        batches = [b for b in rows if now - timedelta(days=10) <= b[0] <= now]
        return batches, "ok" if batches else "not_collected"
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return [], "unavailable"


def delta_text(forecast, previous):
    if forecast is None or previous is None:
        return "缺少预期或前值，不能比较"
    f = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(%|K|M|B|T)?", forecast)
    p = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(%|K|M|B|T)?", previous)
    if not f or not p or f[2] != p[2]:
        return "口径不同，不能直接比较"
    difference = float(f[1]) - float(p[1])
    unit = "个百分点" if f[2] == "%" else (f[2] or "")
    return "预期与前值持平" if abs(difference) < 1e-10 else f"预期较前值{difference:+.4g}{unit}"


def scenario_notes(kind):
    if kind in ("cpi", "ppi", "pce", "eci"):
        return ["高于预期：先看核心分项是否同步偏热，以及美债和ETH是否确认承压。",
                "符合预期：不当成新增利空/利多，转看修正值及价格结构。",
                "低于预期：观察利率压力是否缓解、ETH能否收复关键位；不自动做多。"]
    if kind in ("fomc", "projections", "press_conference", "minutes", "fed_speech"):
        return ["偏鹰/符合/偏鸽分别准备；声明、点阵图和讲话分开核实。",
                "没有可比较的数值不强填预期，不能用降息赔率冒充利率决定。"]
    return ["高于/符合/低于预期分别评估；强数据可能利好增长也可能推高利率，不机械对应涨跌。",
            "检查分项与前值修正，再用ETH结构确认入场；方向倾向不等于现在追单。"]


def check_prior(event, quote, now):
    checks = macro.CALENDAR.get("prior_checks", [])
    prior = next((p for p in checks if p["kind"] == event["kind"] and p["metric"] == quote["metric"]
                  and stamp(p["release_at"]) == event["release_at"] and stamp(p["verified_at"]) <= now), None)
    conflict = prior and quote["previous"] is not None and prior["previous"] != quote["previous"]
    quote["prior_status"] = "conflict" if conflict else "checked" if prior else "unverified"
    quote["comparison"] = delta_text(quote["forecast"], quote["previous"])
    if conflict:
        quote["comparison"] = f"前值口径待核对：该源{quote['previous']}，官方{prior['period']}为{prior['previous']}；暂不比较。"
        quote["prior_check_source"] = prior["source"]


def event_projection(event, batches, now):
    cutoff = min(now, stamp(event["release_at"]))
    eligible = [b for b in batches if b[0] <= now and b[0] < event["release_at"] and b[1] == "ok"]
    chosen = eligible[-1] if eligible else None
    quotes = [dict(q) for q in chosen[2].get("quotes", []) if q["event_id"] == event["id"]] if chosen else []
    before = now < event["release_at"]
    for quote in quotes:
        check_prior(event, quote, now)
    data_at = chosen[0] if chosen and quotes else None
    fresh = data_at is not None and (cutoff - data_at).total_seconds() <= STALE_SECONDS
    return {"id": event["id"], "name": event["name"], "kind": event["kind"], "priority": event["priority"],
            "release_at": event["release_at"].isoformat(), "phase": "upcoming" if before else "awaiting_actual",
            "seconds_to_release": int((event["release_at"] - now).total_seconds()), "metrics": quotes,
            "captured_at": data_at.isoformat() if data_at else None,
            "expectation_status": "conflict" if any(q["prior_status"] == "conflict" for q in quotes) else
            "fresh" if fresh and before else "frozen" if fresh else "stale" if quotes else "missing",
            "calendar_source": event["source"], "scenarios": scenario_notes(event["kind"])}


def snapshot(path=DATABASE, now=None, limit=4):
    now = stamp(now or utc_now())
    batches, storage = read_batches(path, now)
    events = [e for e in macro.EVENTS if e["priority"] != "C"
              and now - timedelta(hours=2) <= e["release_at"] <= now + timedelta(days=7)]
    views = [event_projection(e, batches, now) for e in events[:limit]]
    phase = "error" if batches and batches[-1][1] == "error" else storage
    return {"updated_at": now.isoformat(), "events": views, "total_events": len(events),
            "source": SOURCE_LABEL, "source_url": SOURCE, "collector_status": phase,
            "last_attempt_at": batches[-1][0].isoformat() if batches else None,
            "calendar_verified_at": macro.CALENDAR["verified_at"], "calendar_end": macro.CALENDAR["coverage_end"],
            "calendar_stale": now - stamp(macro.CALENDAR["verified_at"]) > timedelta(days=7),
            "note": "公开周历参考预期，非官方预期；前值按该源口径，未独立复核。每30分钟尝试更新，非实时实际值通道。"}


def preparation_brief(path=DATABASE, now=None):
    view = snapshot(path, now, limit=20)
    lines = [view["note"], f"事前预期采集状态：{view['collector_status']}；失败或过期的记录只能作旧参考。"]
    for event in view["events"]:
        label = "事前准备" if event["phase"] == "upcoming" else "公布前留档，等待核实实际值"
        lines.append(f"{event['name']} {event['release_at']}：{label}；预期状态{event['expectation_status']}。")
        for q in event["metrics"]:
            previous = "待核对" if q["prior_status"] == "conflict" else q['previous'] or '未获取'
            lines.append(f"{q['label']}：预期{q['forecast'] or '未获取'}，前值{previous}；{q['comparison']}。")
        if not event["metrics"]:
            lines.append("没有公布前抓到的预期，不允许用后来新闻倒填。")
    lines.append("上述只说明预期相对前值的变化，不证明市场已经充分定价；缺美债/资金反应时明确未知。")
    return "\n".join(lines)


def preparation_version(path=DATABASE, now=None):
    view = snapshot(path, now, limit=20)
    material = [(e["id"], e["expectation_status"], e["metrics"]) for e in view["events"] if e["phase"] == "upcoming"]
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
