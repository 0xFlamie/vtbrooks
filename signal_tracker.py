"""信号成绩单与裁判健康监控(2026-10-13 用户决策)

- 记录每次新判决(主裁判 / 对照裁判), 1h、4h 后用 Coinbase 1分钟K线结算涨跌, 全量保存不截断
- 每周一 09:00(北京时间) 推送成绩单, 与同时段"恒多/恒空"基线对比
- 裁判连续失败、输出被截断、模型被替换、余额不足 → Telegram 告警
"""
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(ROOT, "signal_scores.sqlite3")
HORIZONS = (("ret_1h", 3600), ("ret_4h", 4 * 3600))
CB_CANDLES = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
BJ = timezone(timedelta(hours=8))
FAIL_ALERT_AT = 3        # 连续失败几次后告警
LOW_BALANCE_CNY = 5.0    # 余额低于此值告警
_lock = threading.Lock()
_send = print            # 推送函数, main() 里注入 send_telegram


def set_sender(fn):
    global _send
    _send = fn


def _alert(text):
    try:
        _send(f"🚨 系统告警\n{text}")
    except Exception as e:
        print(f"WARN: 告警推送失败 {e}")


def _db():
    con = sqlite3.connect(DB_FILE, timeout=10)
    con.execute("""create table if not exists judgments(
        id integer primary key, ts real, variant text, layer text, symbol text,
        direction text, confidence integer, summary text, px0 real, ret_1h real, ret_4h real)""")
    con.execute("create table if not exists meta(k text primary key, v text)")
    return con


@contextmanager
def _conn():
    """加锁 + 自动提交 + 用完关闭"""
    with _lock:
        con = _db()
        try:
            with con:
                yield con
        finally:
            con.close()


def _meta_get(con, key):
    row = con.execute("select v from meta where k=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(con, key, value):
    con.execute("insert or replace into meta(k, v) values(?, ?)", (key, str(value)))


# ── 判决记录与结算 ──

def record(variant, layer, symbol, judge, ts=None):
    """记录一次新判决; 裁判失败(置信<0)不算判决, 不记录"""
    if not judge or int(judge.get("confidence", -1)) < 0:
        return
    with _conn() as con:
        con.execute("insert into judgments(ts, variant, layer, symbol, direction, confidence, summary) "
                    "values(?,?,?,?,?,?,?)",
                    (ts or time.time(), variant, layer, symbol, judge.get("direction") or "",
                     int(judge.get("confidence", -1)), str(judge.get("summary") or "")[:200]))


def coinbase_close(ts):
    """ts 所在那一分钟的 Coinbase ETH-USD 收盘价; 取不到返回 None"""
    minute = int(ts // 60 * 60)
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
    r = requests.get(CB_CANDLES, params={"granularity": 60, "start": iso(minute), "end": iso(minute + 60)},
                     timeout=10)
    if r.status_code != 200:
        return None
    for t, _low, _high, _open, close, _vol in r.json():
        if int(t) == minute:
            return float(close)
    return None


def settle(price_at=coinbase_close, limit=12, now=None):
    """给到期的判决补上 1h/4h 涨跌(%), 每次最多处理 limit 条, 只看最近3天"""
    now = now or time.time()
    with _conn() as con:
        rows = con.execute("select id, ts, px0, ret_1h, ret_4h from judgments "
                           "where ts > ? and ts < ? and (px0 is null or ret_1h is null or ret_4h is null) "
                           "order by ts limit ?", (now - 3 * 86400, now - 3600 - 120, limit)).fetchall()
    for rid, ts, px0, r1, r4 in rows:
        try:
            if px0 is None:
                px0 = price_at(ts)
                if px0 is None:
                    continue
            vals = {"px0": px0}
            for col, sec, old in (("ret_1h", 3600, r1), ("ret_4h", 4 * 3600, r4)):
                if old is None and ts + sec < now - 120:
                    px = price_at(ts + sec)
                    if px is not None:
                        vals[col] = round((px - px0) / px0 * 100, 4)
            with _conn() as con:
                con.execute(f"update judgments set {', '.join(f'{k}=?' for k in vals)} where id=?",
                            (*vals.values(), rid))
        except Exception as e:
            print(f"WARN: 成绩结算失败 {type(e).__name__}: {e}")


# ── 成绩单 ──

def _pct(a, b):
    return f"{a / b * 100:.0f}%" if b else "—"


def _layer_stats(rows):
    """rows: [(direction, confidence, ret_1h, ret_4h)] → 成绩字典"""
    out = {"n": len(rows), "n_dir": 0}
    for col, idx in (("1h", 2), ("4h", 3)):
        done = [r for r in rows if r[idx] is not None]
        sided = [r for r in done if r[0] in ("LONG", "SHORT")]
        hit = sum(1 for r in sided if (r[idx] > 0) == (r[0] == "LONG") and r[idx] != 0)
        signed = [r[idx] if r[0] == "LONG" else -r[idx] for r in sided]
        hi = [r for r in sided if r[1] >= 60]
        hi_hit = sum(1 for r in hi if (r[idx] > 0) == (r[0] == "LONG") and r[idx] != 0)
        up = sum(1 for r in done if r[idx] > 0)
        out[col] = {"done": len(done), "sided": len(sided), "hit": hit,
                    "avg": sum(signed) / len(signed) if signed else None,
                    "hi": len(hi), "hi_hit": hi_hit, "up": up}
        out["n_dir"] = max(out["n_dir"], len(sided))
    return out


def build_report(days=7, now=None):
    """近 days 天成绩单文本; 命中=方向与之后涨跌一致, 基线=同一批时点无脑做多/做空"""
    now = now or time.time()
    with _conn() as con:
        rows = con.execute("select variant, layer, direction, confidence, ret_1h, ret_4h from judgments "
                           "where ts >= ? order by ts", (now - days * 86400,)).fetchall()
    start = datetime.fromtimestamp(now - days * 86400, BJ).strftime("%m-%d")
    end = datetime.fromtimestamp(now, BJ).strftime("%m-%d")
    L = [f"📊 信号成绩单 {start} ~ {end}", "命中=判断方向与之后涨跌一致; 基线=同一批时点无脑做多/做空"]
    if not rows:
        L.append("本期没有判决记录")
        return "\n".join(L)
    groups = {}
    for v, layer, d, c, r1, r4 in rows:
        groups.setdefault((layer, v), []).append((d, c, r1, r4))
    for (layer, v) in sorted(groups, key=lambda k: (k[0] != "15m", k[1] != "主裁判", k)):
        s = _layer_stats(groups[(layer, v)])
        L.append("")
        L.append(f"【{layer} · {v}】判决{s['n']}次, 有方向{s['n_dir']}次")
        for col in ("1h", "4h"):
            x = s[col]
            if not x["done"]:
                L.append(f"  {col}后: 尚未到期")
                continue
            avg = f"{x['avg']:+.2f}%" if x["avg"] is not None else "—"
            L.append(f"  {col}后: 命中{_pct(x['hit'], x['sided'])}({x['hit']}/{x['sided']}) 平均{avg} | "
                     f"置信≥60 命中{_pct(x['hi_hit'], x['hi'])}({x['hi']}次) | "
                     f"基线 恒多{_pct(x['up'], x['done'])} 恒空{_pct(x['done'] - x['up'], x['done'])}")
    L.append("")
    L.append("注: 平均=按判断方向计的涨跌幅, 未扣手续费; 样本少时结论不可靠")
    return "\n".join(L)


def maybe_weekly_report(now=None):
    """每周一 09:00(北京) 后推送一次上周成绩单"""
    now = now or time.time()
    bj = datetime.fromtimestamp(now, BJ)
    if bj.weekday() != 0 or bj.hour < 9:
        return False
    week = bj.strftime("%G-W%V")
    with _conn() as con:
        if _meta_get(con, "last_weekly") == week:
            return False
        _meta_set(con, "last_weekly", week)
        has_rows = con.execute("select 1 from judgments where ts >= ? limit 1", (now - 7 * 86400,)).fetchone()
    if not has_rows:  # 刚上线还没有记录时不发空成绩单
        return False
    _send(build_report(7, now))
    return True


# ── 裁判健康监控 ──

_health = {"fails": 0, "alerted": False, "last_reason": ""}


def judge_ok(name, model=None):
    """一次成功的裁判调用: 清零失败计数; 模型名变化时告警"""
    recovered = _health["alerted"]
    _health.update(fails=0, alerted=False)
    if recovered:
        _alert(f"AI 裁判已恢复正常({name})")
    if model:
        with _conn() as con:
            old = _meta_get(con, f"model_{name}")
            if old != model:
                _meta_set(con, f"model_{name}", model)
        if old and old != model:
            _alert(f"AI 模型被服务商替换: {old} → {model}\n新模型输出风格可能不同, 留意判断质量")


def judge_fail(name, reason):
    """一次失败的裁判调用; 连续 FAIL_ALERT_AT 次只告警一次, 恢复后再提示"""
    _health["fails"] += 1
    _health["last_reason"] = reason
    print(f"WARN: 裁判{name}失败({_health['fails']}连): {reason}")
    if _health["fails"] >= FAIL_ALERT_AT and not _health["alerted"]:
        _health["alerted"] = True
        _alert(f"AI 裁判连续失败 {_health['fails']} 次({name})\n原因: {reason}\n"
               "失败期间所有判断都显示为观望, 不是真实判断")


def check_balance(api_key, now=None, every=3600):
    """每小时查一次 DeepSeek 余额, 低于阈值时每 12 小时最多告警一次"""
    now = now or time.time()
    with _conn() as con:
        last_check = _meta_get(con, "balance_checked")
        if last_check is not None and now - float(last_check) < every:
            return None
        _meta_set(con, "balance_checked", now)
    try:
        r = requests.get("https://api.deepseek.com/user/balance",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        info = r.json()
        bal = sum(float(x.get("total_balance", 0)) for x in info.get("balance_infos", []))
    except Exception as e:
        print(f"WARN: 余额查询失败 {type(e).__name__}: {e}")
        return None
    if bal < LOW_BALANCE_CNY or not info.get("is_available", True):
        with _conn() as con:
            last_alert = _meta_get(con, "balance_alerted")
            if last_alert is None or now - float(last_alert) >= 12 * 3600:
                _meta_set(con, "balance_alerted", now)
                _alert(f"DeepSeek 余额不足: 剩余 ¥{bal:.2f}\n欠费后 AI 判断会全部失败, 请尽快充值")
    return bal
