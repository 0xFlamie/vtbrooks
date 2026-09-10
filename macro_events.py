"""宏观事件目录和有来源的日历；不把计划公布时间当作实际数据到达。"""
from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
BEIJING = ZoneInfo("Asia/Shanghai")
CATALOG_ROWS = (
    ("cpi", "CPI消费者通胀", "A", "总体环比/同比、核心环比/同比、季调口径", "https://www.bls.gov/cpi/"),
    ("ppi", "PPI生产者通胀", "A", "总体环比/同比、核心环比/同比、能源及服务分项", "https://www.bls.gov/ppi/"),
    ("pce", "PCE通胀与个人收支", "A", "总体及核心环比/同比、实际消费、收入、前值修正", "https://www.bea.gov/news/schedule/full"),
    ("employment", "非农就业报告", "A", "新增非农、失业率、时薪、参与率、前两月修正", "https://www.bls.gov/ces/"),
    ("fomc", "FOMC利率决议", "A", "目标利率区间、声明变化、投票分歧、缩表政策", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
    ("projections", "联储经济预测与点阵图", "A", "年末利率中位数、通胀/增长/失业预测、预测修订", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
    ("press_conference", "联储主席发布会", "A", "实际讲话、问答、相对声明的新信息", "https://www.federalreserve.gov/newsevents/calendar.htm"),
    ("minutes", "FOMC会议纪要", "B", "会议日期、政策分歧、不能把旧会议观点当新决定", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
    ("claims", "初请与续请失业金", "B", "初请、续请、四周均值、对应周次及前值修正", "https://www.dol.gov/ui/data.pdf"),
    ("jolts", "JOLTS职位空缺", "B", "空缺、招聘、主动离职、前值修正", "https://www.bls.gov/jlt/"),
    ("adp", "ADP私人就业", "B", "私人就业、薪资；不能代替非农实际值", "https://adpemploymentreport.com/"),
    ("ism_manufacturing", "ISM制造业PMI", "B", "总指数、新订单、就业、价格分项", "https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/"),
    ("ism_services", "ISM服务业PMI", "B", "总指数、新订单、就业、价格分项", "https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/"),
    ("retail", "零售销售", "B", "总体、剔除汽车、控制组、前值修正", "https://www.census.gov/retail/release_schedule.html"),
    ("gdp", "GDP及价格分项", "B", "季度、初值/修正/终值、实际增长、消费、价格；区分月度PCE", "https://www.bea.gov/news/schedule/full"),
    ("michigan", "密歇根消费者调查", "B", "初值/终值、信心、一年及长期通胀预期", "https://data.sca.isr.umich.edu/survey-info.php"),
    ("eci", "ECI就业成本", "B", "季调季度增幅、工资及福利、统计季度", "https://www.bls.gov/eci/"),
    ("sp_pmi", "S&P Global美国PMI", "B", "制造业/服务业/综合、初值/终值、价格分项", "https://pmi.spglobal.com/Public/Release/ReleaseDates?language=en&os=0"),
    ("fed_speech", "联储重要讲话/杰克逊霍尔", "B", "发言人、发言时间、完整上下文、与最近数据的差异", "https://www.federalreserve.gov/newsevents/calendar.htm"),
    ("treasury_refunding", "财政部季度再融资", "B", "融资规模、期限结构、相对预告的变化", "https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding"),
    ("treasury_auction", "美债关键期限拍卖", "C", "2/10/30年、投标倍数、尾差、间接投标；不单独定方向", "https://www.treasurydirect.gov/auctions/announcements-data-results/"),
    ("liquidity", "联储H.4.1资产负债表", "C", "总资产、准备金、统计日期；周度背景不是即时流动性", "https://www.federalreserve.gov/releases/h41/"),
    ("tga_rrp", "TGA与隔夜逆回购", "C", "余额、变动、数据日期；不同更新时间不直接混算", "https://fiscaldata.treasury.gov/"),
)
CATALOG = {key: {"name": name, "priority": priority, "components": components, "source": source}
           for key, name, priority, components, source in CATALOG_ROWS}


def load_calendar(path=Path(__file__).with_name("macro_calendar_2026.json")):
    payload = json.loads(Path(path).read_text())
    if payload["version"] != 1:
        raise ValueError("不支持的宏观日历版本")
    result, seen = [], set()
    for group in payload["groups"]:
        kind = group["kind"]
        spec = CATALOG[kind]
        source = payload["sources"][group["source"]]
        if not source.startswith("https://"):
            raise ValueError("日历来源必须为HTTPS")
        for day in group["dates"]:
            if not payload["coverage_start"] <= day <= payload["coverage_end"]:
                raise ValueError("事件超出声明覆盖范围")
            release = datetime.fromisoformat(f"{day}T{group['time_et']}").replace(tzinfo=NEW_YORK)
            key = f"US:{kind}:{release.isoformat()}"
            if key in seen:
                raise ValueError("重复宏观事件")
            seen.add(key)
            result.append({**spec, "name": spec["name"] + group.get("edition", ""),
                           "kind": kind, "id": key, "release_at": release, "source": source})
    return payload, tuple(sorted(result, key=lambda row: (row["release_at"], row["id"])))


CALENDAR, EVENTS = load_calendar()


def local_time(now=None):
    now = now if now is not None else datetime.now(NEW_YORK)
    if now.tzinfo is None:
        raise ValueError("宏观时间必须带时区")
    return now.astimezone(NEW_YORK)


def events_on(day):
    return [dict(e) for e in EVENTS if e["release_at"].date().isoformat() == day]


def day_label(day):
    return " / ".join(e["name"] for e in events_on(day))


def pending(now=None):
    now = local_time(now)
    return [e for e in events_on(now.date().isoformat())
            if e["release_at"] > now and e["priority"] != "C"]


def pending_ids(now=None):
    return [e["id"] for e in pending(now)]


def preparation_key(now=None):
    now = local_time(now)
    rows = pending(now)
    # 临近公布时复核关键位；普通时段保持已有预案，避免每轮重复请求AI。
    slot = int(now.timestamp() // 1800) if rows and rows[0]["release_at"] - now <= timedelta(hours=1) else 0
    return {"event_ids": [e["id"] for e in rows], "slot": slot}


def preview_is_current(preview, now=None):
    now = local_time(now)
    ids = pending_ids(now)
    return bool(ids) and preview.get("date") == now.date().isoformat() and preview.get("event_ids") == ids


def in_window(now=None):
    now = local_time(now)
    return any(e["priority"] != "C" and timedelta(hours=-1) <= now - e["release_at"] <= timedelta(hours=2)
               for e in events_on(now.date().isoformat()))


def coverage(now=None):
    now = local_time(now)
    scheduled = {e["kind"] for e in EVENTS}
    future = {e["kind"] for e in EVENTS if e["release_at"] > now}
    return {"catalog_count": len(CATALOG), "event_count": len(EVENTS),
            "unscheduled": sorted(set(CATALOG) - scheduled),
            "without_future_date": sorted(set(CATALOG) - future),
            "actual_feed": "not_connected", "consensus_feed": "not_connected"}


def brief(now=None):
    now = local_time(now)
    day = now.date().isoformat()
    if not CALENDAR["coverage_start"] <= day <= CALENDAR["coverage_end"]:
        return "宏观日历覆盖期外：不能判断今天无事件；需更新官方日历。"
    rows = events_on(day)
    lines = [f"宏观时间基准：北京{now.astimezone(BEIJING):%Y-%m-%d %H:%M:%S}。"]
    if now - datetime.fromisoformat(CALENDAR["verified_at"]) > timedelta(days=7):
        lines.append("日历快照超过7天，需重新核验，不能保证官方未调整时间。")
    for e in rows:
        state = "未到公布时刻" if now < e["release_at"] else "已到计划公布时间，不代表已收到实际值"
        release = e["release_at"].astimezone(BEIJING)
        lines.append(f"[{e['priority']}] {e['name']} 北京{release:%m-%d %H:%M}：{state}；核对{e['components']}。")
    if not rows:
        lines.append("已核验快照当日未列事件，不代表没有事件；讲话/拍卖等仍可能缺失。")
    missing = coverage(now)["without_future_date"]
    if missing:
        lines.append("后续日程待核验：" + "、".join(CATALOG[k]["name"] for k in missing) + "。")
    lines.extend(("本日历不提供实际值、前值或共识；市场预期以单独事前记录为准，未提供时不得编造或凭标题宣称超预期。",
                  "公布前只做高于/符合/低于预期的条件预案；公布后不得再说等待公布，只能说明等待核实数据。",
                  "优先级是关注顺序，不是多空投票或胜率；不能把方向偏空等同于现在追空。"))
    return "\n".join(lines)
