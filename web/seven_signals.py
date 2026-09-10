"""七类研究提醒：只读现有WS账本，不改采集器、原策略或交易裁判。"""
from contextlib import closing
from datetime import datetime, timezone
import sqlite3
from threading import Lock

import pandas as pd

from brooks_failed_range_shadow import frame_from_bars
from brooks_intrabar import scan_intrabar
from brooks_reversal_shapes import scan_reversal_shapes
from brooks_second_leg import scan_second_leg
from web.research_signals import LEDGER, checked, number, read_snapshot as range_snapshot, rows

FAMILIES = (
    ("h2", "H2/L2二次入场", "BRK-H2-15M-v1", "54.4%–54.5%", 1072),
    ("retest", "突破首次回触", "BRK-RT-15M-v1", "59.2%", 1178),
    ("outside", "外包K后突破", "BRK-OUT-15M-v1", "54.0%", 483),
    ("failed_range", "区间假突破", "BRK-FR-15M-v1", "65.7%–65.9%", 1443),
    ("second_leg", "强突破第二腿", "BRK-SL-15M-v1", "54.6%", 1501),
    ("double_test", "双顶 / 双底", "BRK-DT-15M-v1", "61.0%–61.1%", 6119),
    ("wedge", "楔形三推", "BRK-WG-15M-v1", "56.3%–56.5%", 2087),
)
IDS = {row[0]: row[2] for row in FAMILIES}
REASONS = {"h2": "二次入场过程出现盘中推进", "retest": "突破后首次回触关键位并重新推进",
           "outside": "外包K后越过触发边界", "second_leg": "强突破首次回调后恢复推进",
           "double_test": "已确认旧拐点再次测试后，越过测试分钟的反向极值",
           "wedge": "已确认三推结构后，越过前一15M反向极值"}
CONTEXT_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"


def minute_events(db, begin, end):
    query = ("SELECT e.seq,e.event_key,e.payload,e.sha256,p.sha256 FROM events e "
             "LEFT JOIN events p ON p.seq=e.seq-1 WHERE e.event_key>=? AND e.event_key<? ORDER BY e.event_key")
    result = []
    for raw in db.execute(query, ("minute:" + begin.isoformat(), "minute:" + end.isoformat())):
        event = checked(raw)
        bar = {k: v for k, v in event["payload"].items() if k != "trades"}
        if event["kind"] != "minute" or event["key"] != "minute:" + bar["stamp"] or bar["live"] is not True:
            raise ValueError("不是原始WS分钟")
        result.append(bar)
    return result


def context_bars(events, begin, end):
    merged = {}
    for event in reversed(events):
        value = event["payload"]
        if event["kind"] != "context" or value["endpoint"] != CONTEXT_URL or value["live"] is not False:
            raise ValueError("背景来源不符")
        left, right = pd.Timestamp(value["params"]["start"]), pd.Timestamp(value["params"]["end"])
        if value["params"]["granularity"] != 60 or pd.Timestamp(value["received_at"]) > end:
            raise ValueError("背景时间或粒度不符")
        for epoch, low, high, opened, close, volume in value["payload"]:
            stamp = pd.Timestamp(epoch, unit="s", tz="UTC")
            if max(left, begin) <= stamp < min(right, end):
                merged[stamp.isoformat()] = {"stamp": stamp.isoformat(), "open": opened, "high": high,
                    "low": low, "close": close, "volume": volume, "live": False}
    return merged


def scan_six(frame, start, end):
    result = scan_intrabar(frame, start, end) + scan_second_leg(frame, start, end)[0]
    return result + scan_reversal_shapes(frame, start, end)


def timely_prefix(signal, bars, at):
    available, start = pd.Timestamp(signal["available_at"]), pd.Timestamp(signal["bar_start"])
    expected = pd.date_range(start, available, freq="min", inclusive="left")
    if not len(expected) or not 0 <= (at - available).total_seconds() < 120:
        return False
    for stamp in expected:
        bar = bars.get(stamp.isoformat())
        if not bar or bar.get("live") is not True:
            return False
        received = pd.Timestamp(bar["received_at"])
        if not stamp + pd.Timedelta(minutes=1) <= received <= at or received - stamp > pd.Timedelta(minutes=3):
            return False
    return True


def project(signal, generated, eligible, at, live):
    family, side = signal["family"], signal["direction"]
    if family not in REASONS or type(side) is not int or side not in (-1, 1):
        raise ValueError("六类信号身份不符")
    available = pd.Timestamp(signal["available_at"])
    deadline, fresh_until = available + pd.Timedelta(hours=1), available + pd.Timedelta(minutes=2)
    status = ("ended" if at >= deadline else "unverified" if not live else "history" if not eligible else
              "new" if at < fresh_until else "observing")
    return {"id": IDS[family] + ":" + signal["episode_id"], "event_id": signal["episode_id"],
            "family": family, "direction": "LONG" if side == 1 else "SHORT", "status": status,
            "available_at": available.isoformat(), "generated_at": generated.isoformat(),
            "entry_not_before": fresh_until.isoformat(), "deadline": deadline.isoformat(),
            "observed_price": number(signal["observed_price"]), "atr": number(signal["atr"]),
            "structure_reference": number(signal["structure_reference"]), "reason": REASONS[family],
            "result": "网页只读识别；未记录实际成交，不作盈利判定"}


class SevenReader:
    def __init__(self):
        self.lock, self.version, self.events = Lock(), None, {}

    def refresh(self, path, at):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=.25)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            latest = rows(db, "minute:")
            context = rows(db, "context:", 32)
            if not latest or not pd.Timedelta(0) <= at - (pd.Timestamp(latest[0]["payload"]["stamp"]) + pd.Timedelta(minutes=1)) < pd.Timedelta(minutes=2):
                raise ValueError("WS分钟未到达或已过期")
            version = (str(path.resolve()), latest[0]["key"], latest[0]["seq"], context[0]["key"] if context else None)
            if self.version == version:
                return
            begin = at.floor("15min") - pd.Timedelta(hours=35)
            bars = context_bars(context, begin, at)
            bars.update({b["stamp"]: b for b in minute_events(db, begin, at.floor("min"))})
        if not latest or not bars:
            raise ValueError("尚无WS分钟或背景")
        frame = frame_from_bars(list(bars.values()))
        if at - (frame.index[-1] + pd.Timedelta(minutes=1)) > pd.Timedelta(minutes=2):
            raise ValueError("分钟流已过期")
        candidates = scan_six(frame, at - pd.Timedelta(hours=1), at + pd.Timedelta(nanoseconds=1))
        events = {} if self.version and self.version[0] != version[0] else dict(self.events)
        for signal in candidates:
            key = signal["episode_id"]
            if key not in events:
                events[key] = (signal, at, timely_prefix(signal, bars, at))
        self.events = {k: v for k, v in events.items() if at - pd.Timestamp(v[0]["available_at"]) < pd.Timedelta(hours=2)}
        self.version = version

    def snapshot(self, path=LEDGER, at=None):
        at = pd.Timestamp(datetime.now(timezone.utc) if at is None else at)
        base = range_snapshot(path, at.to_pydatetime())
        with self.lock:
            ready = False
            if base["collector"]["status"] == "live":
                try:
                    self.refresh(path, at)
                    ready = True
                except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, IndexError, OverflowError):
                    self.version = None
            try:
                signals = [project(s, generated, eligible, at, ready) for s, generated, eligible in self.events.values()]
            except (ValueError, KeyError, TypeError, OverflowError):
                ready, signals = False, []
        signals += [{**s, "family": "failed_range"} for s in base["signals"]]
        families = []
        for family, name, identity, rate, sample in FAMILIES:
            health = base["collector"] if family == "failed_range" or ready else {
                "status": "unavailable", "label": "分钟数据未就绪 / 已中断"}
            selected = sorted((s for s in signals if s["family"] == family), key=lambda s: s["available_at"], reverse=True)[:20]
            families.append({"family": family, "name": name, "strategy_id": identity,
                             "rate": rate, "sample": sample, "collector": health, "signals": selected})
        return {**base, "families": families, "signals": [s for f in families for s in f["signals"]],
                "rule_note": "15M形态 · 完整1M盘中触发 · 七类独立提醒，不投票、不自动下单",
                "history_note": "完整三年研究 · 各卡为持续顺向空间率，不是盈利胜率；七类均未达到68%"}


READER = SevenReader()
read_snapshot = READER.snapshot
