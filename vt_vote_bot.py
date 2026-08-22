"""
Vibe-Trading 因子投票信号机器人 v2.1
- VT因子8 + NOFX趋势4 + Brooks价格行为6 = 18票
- 8+/18 触发推送, 12+/18 强信号
- Brooks: 市场状态(spike/channel/range) / Always In / H2·L2双腿回调 /
  三推楔形·双顶底反转 / 假突破陷阱 / 信号K线质量
- 推送: K线图(图片信号卡) + 结构化文字详解, 全部大白话表达
"""
import warnings, sys, os, re, json, time, argparse, importlib, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests

VT_PKG = None
for p in sys.path:
    if p.endswith("site-packages"):
        VT_PKG = p
        break
if not VT_PKG:
    VT_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../vt-venv/lib/python3.11/site-packages")
sys.path.insert(0, VT_PKG)

# ── 配置 (仅环境变量, 密钥不入库) ──
TELEGRAM_TOKEN = os.environ.get("VT_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("VT_TELEGRAM_CHAT", "")
DS_API_KEY = os.environ.get("VT_DS_API_KEY", "")
DS_API_URL = "https://api.deepseek.com/v1/chat/completions"
DS_MODEL = "deepseek-chat"
MS_API_KEY = os.environ.get("VT_MS_API_KEY", "")
MS_API_URL = "https://api.moonshot.cn/v1/chat/completions"
MS_MODEL = "kimi-k2.5"  # 2026-08-10实测: k2.5/k2.6/k3全是思考模型(温度锁1+reasoning占额度), k2.5最快(18s/4000tok)
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vt_predictions.json")
JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_journal.json")
LESSONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_lessons.json")
OI_SNAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oi_snapshots.json")

# ── 因子配置: (name, module, ic_sign, display_name)
# ic_sign = +1 → 因子值高 = 看多, 值低 = 看空
# ic_sign = -1 → 因子值高 = 看空, 值低 = 看多
BTC_FACTORS = [
    ("ma30",     "src.factors.zoo.qlib158.ma30",       +1, "均线偏离30"),
    ("qtlu20",   "src.factors.zoo.qlib158.qtlu20",     +1, "高位阻力80分位"),
    ("gtja046",  "src.factors.zoo.gtja191.alpha_046",  +1, "量价加权GT46"),
    ("gtja134",  "src.factors.zoo.gtja191.alpha_134",  -1, "短周动量GT134"),
    ("gtja029",  "src.factors.zoo.gtja191.alpha_029",  -1, "日内价差GT29"),
    ("gtja178",  "src.factors.zoo.gtja191.alpha_178",  -1, "量价动量GT178"),
    ("resi30",   "src.factors.zoo.qlib158.resi30",     -1, "趋势偏离30日"),
    ("resi20",   "src.factors.zoo.qlib158.resi20",     -1, "趋势偏离20日"),
]

ETH_FACTORS = [
    ("a101084",  "src.factors.zoo.alpha101.alpha_084", +1, "动量结构A84"),
    ("qtlu20",   "src.factors.zoo.qlib158.qtlu20",     +1, "高位阻力80分位"),
    ("gtja150",  "src.factors.zoo.gtja191.alpha_150",  +1, "成交量异动GT150"),
    ("ma30",     "src.factors.zoo.qlib158.ma30",       +1, "均线偏离30"),
    ("gtja188",  "src.factors.zoo.gtja191.alpha_188",  +1, "波动率结构GT188"),
    ("a101047",  "src.factors.zoo.alpha101.alpha_047", +1, "量价动量A47"),
    ("gtja134",  "src.factors.zoo.gtja191.alpha_134",  -1, "短周动量GT134"),
    ("gtja029",  "src.factors.zoo.gtja191.alpha_029",  -1, "日内价差GT29"),
]

# ── 因子大白话: 让人看懂每个因子在衡量什么 ──
FACTOR_DESC = {
    "均线偏离30":     "价格偏离30均线的程度，看趋势方向和是否过热",
    "高位阻力80分位": "价格在历史高位区的位置，越靠近高位上方压力越大",
    "量价加权GT46":   "成交量加权的量价配合强度",
    "短周动量GT134":  "一两周级别的涨跌动量",
    "日内价差GT29":   "日内多空力量的价差对比",
    "量价动量GT178":  "成交量配合下的价格动量",
    "趋势偏离30日":   "价格偏离30日趋势的程度，偏离过大容易回归",
    "趋势偏离20日":   "价格偏离20日趋势的程度，偏离过大容易回归",
    "动量结构A84":    "价格在近期高低点结构中的位置，衡量动量强弱",
    "成交量异动GT150": "成交量相对平常的异常程度，放大说明有资金动作",
    "波动率结构GT188": "波动率的结构变化，波动放大往往伴随行情启动",
    "量价动量A47":    "量价配合的中期动量",
}

MIN_VOTES = 9   # 至少 N 票触发 (VT8 + NOFX4 + Brooks6 = 18总票)
STRONG = 12     # N+ 票 = 强信号
LOOKBACK = 200  # 计算因子用的历史 bar 数


def z_word(z):
    """z分数 → 大白话强弱描述, 避免推送无意义的原始数值"""
    if z >= 2: return "显著高于"
    if z >= 1: return "明显高于"
    if z > 0.3: return "略高于"
    if z >= -0.3: return "基本持平于"
    if z >= -1: return "略低于"
    if z >= -2: return "明显低于"
    return "显著低于"


# ── HTTP: 统一走 requests, 替代 subprocess curl ──
_http = requests.Session()

def http_get_json(url, timeout=8, headers=None):
    """GET JSON, 非200/解析失败抛异常"""
    r = _http.get(url, timeout=timeout, headers=headers or {})
    r.raise_for_status()
    return r.json()


INTERVAL_MS = {"5m": 5 * 60_000, "15m": 15 * 60_000, "1h": 3_600_000,
               "4h": 4 * 3_600_000, "1d": 86_400_000}

# K线缓存: (symbol, interval, limit) → (timestamp, df), TTL 60s, 同一轮内复用
_kline_cache = {}

# 近期新闻(供两层裁判简报读取): [(ts, "来源: 中文标题")], 保留12h/10条, 主循环更新
RECENT_NEWS = []

# 宏观日历(美东日期): 事件日两层简报注入, AI 提高新闻/宏观权重(2026-08-20 财政部划线漏报事故)
MACRO_EVENTS = {
    "2026-08-19": "FOMC会议纪要",  # 补录: 2026-08-20 发现遗漏(当天收益率划线行情)
    "2026-09-15": "FOMC议息首日", "2026-09-16": "FOMC议息决议+鲍威尔发布会",
    "2026-10-07": "FOMC会议纪要", "2026-10-27": "FOMC议息首日", "2026-10-28": "FOMC议息决议",
    "2026-11-18": "FOMC会议纪要", "2026-12-08": "FOMC议息首日", "2026-12-09": "FOMC议息决议",
    "2026-12-30": "FOMC会议纪要",
    "2026-09-11": "美国8月CPI", "2026-10-13": "美国9月CPI",
    "2026-11-10": "美国10月CPI", "2026-12-10": "美国11月CPI",
}


def _macro_line():
    """今天是宏观事件日则返回提示行, 否则 None"""
    try:
        d = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        ev = MACRO_EVENTS.get(d)
        if ev:
            return f"⚠️ 今日宏观事件: {ev}(美东)。事件日波动放大概率高, 新闻/宏观面的权重上调, 别只看技术面。"
    except Exception:
        pass
    return None


def _event_release_et(ev):
    """事件落地时刻(美东): CPI=08:30, 其余(FOMC等)=14:00"""
    hh, mm = (8, 30) if "CPI" in (ev or "") else (14, 0)
    return pd.Timestamp.now(tz="America/New_York").replace(hour=hh, minute=mm, second=0, microsecond=0)


def preview_visible(ep, today_et):
    """事件预判只在落地前~落地后2h显示, 之后就是过时信息(2026-08-20 用户被旧预判误导)"""
    if ep.get("date") != today_et:
        return False
    ev = MACRO_EVENTS.get(today_et)
    if not ev:
        return False
    return pd.Timestamp.now(tz="America/New_York") < _event_release_et(ev) + pd.Timedelta(hours=2)


def in_event_window():
    """事件落地窗口(落地前1h~后2h, 美东): CPI=08:30, FOMC决议/纪要=14:00。
    窗口内综述每轮强制重写, 把RSS空窗压到物理极限(2026-08-20 纪要空窗问题)"""
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        ev = MACRO_EVENTS.get(now.strftime("%Y-%m-%d"))
        if not ev:
            return False
        hh, mm = (8, 30) if "CPI" in ev else (14, 0)
        release = now.replace(hour=hh, minute=mm, second=0)
        return pd.Timedelta(hours=-1) <= now - release <= pd.Timedelta(hours=2)
    except Exception:
        return False


NEWS_MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_memory.json")


def load_news_memory():
    """事件线记忆: {thread: {summary, dir, horizon, last, hits}}"""
    try:
        if os.path.exists(NEWS_MEMORY_FILE):
            return json.load(open(NEWS_MEMORY_FILE))
    except Exception:
        pass
    return {"threads": {}}


def save_news_memory(mem):
    try:
        json.dump(mem, open(NEWS_MEMORY_FILE, "w"), ensure_ascii=False)
    except Exception:
        pass


def _threads_line(mem, limit=6):
    """进行中事件线 → 简报注入文本; 超5天无更新的主题线视为已冷却, 不再注入(2026-08-20)"""
    th = mem.get("threads") or {}
    if not th:
        return None
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=5)).isoformat()
    live = {k: v for k, v in th.items() if v.get("last", "") > cutoff}
    items = sorted(live.items(), key=lambda kv: kv[1].get("last", ""), reverse=True)[:limit]
    if not items:
        return None
    return "进行中事件: " + " | ".join(f"{k}({v.get('dir','?')}/{v.get('horizon','?')}): {v.get('summary','')}"
                                     for k, v in items)


def news_editor(items, mem):
    """DS 编辑席(2026-08-20 用户要求): 批量读标题, 判价值/分类/时效/方向/事件线, 更新记忆。
    items=[(来源,标题)]; 返回 [{relevant, category, horizon, dir, thread, note, title, src}]"""
    if not items:
        return []
    mem_txt = "; ".join(f"{k}: {v.get('summary','')}" for k, v in list(mem.get("threads", {}).items())[-8:]) or "无"
    sys_p = ("你是只交易ETH的加密交易员兼任新闻编辑。逐条判断新闻标题, 只输出JSON数组, 与输入顺序一致:\n"
             "[{\"i\": 0, \"relevant\": true, \"category\": \"...\", \"horizon\": \"即时\", \"dir\": \"利多\", "
             "\"thread\": \"...\", \"note\": \"...\"}]\n"
             "规则: relevant=是否影响ETH/BTC/加密/整体风险偏好(币圈喊单/价格预测/广告/水文=false; "
             "美股个股/行业新闻一律=false——英伟达这种也不留(2026-08-20 用户决策), 除非直接关乎流动性/系统性风险; "
             "拿不准但有潜在影响的=true, 宁可宽进); "
             "category∈{货币政策,宏观数据,地缘,监管法案,能源大宗,债券利率,汇率美元,黄金避险,行业公司,AI科技,加密行业,其他}; "
             "horizon∈{即时(<24h),中期(数天),长期(数周+)}; "
             "dir=对风险资产方向∈{利多,利空,中性}(允许产业链推导: 如OpenAI千亿建数据中心→利多芯片股, "
             "油价飙→通胀预期→利空风险资产, 美元走强→利空加密); "
             "thread=事件主题2-6字(如伊朗局势/联储路径/AI资本开支/油价冲击), 属于已有主题线就用同名, 新事件起新名; "
             "note=一句话≤25字说清市场影响。\n已有主题线: " + mem_txt)
    user = "\n".join(f"{i}. [{src}] {t}" for i, (src, t) in enumerate(items))
    try:
        r = _http.post(DS_API_URL, json={"model": DS_MODEL, "messages": [
            {"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            "max_tokens": 900, "temperature": 0.1},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=25)
        if r.status_code != 200:
            return []
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", text, re.S)
        arr = json.loads(m.group(0)) if m else []
        out = []
        for it in arr:
            i = it.get("i")
            if not isinstance(i, int) or i >= len(items):
                continue
            it["src"], it["title"] = items[i]
            out.append(it)
            th = it.get("thread")
            if it.get("relevant") and th:
                t = mem["threads"].setdefault(th, {"hits": 0})
                t.update({"summary": it.get("note", ""), "dir": it.get("dir", "中性"),
                          "horizon": it.get("horizon", "中期"), "last": pd.Timestamp.now().isoformat()})
                t["hits"] = t.get("hits", 0) + 1
        # 记忆上限30条主题线, 最旧的淘汰
        if len(mem["threads"]) > 30:
            for k in sorted(mem["threads"], key=lambda k: mem["threads"][k].get("last", ""))[:len(mem["threads"]) - 30]:
                del mem["threads"][k]
        save_news_memory(mem)
        return out
    except Exception as e:
        print(f"WARN: 新闻编辑异常 {type(e).__name__}: {e}")
        return []


def event_preview(sym):
    """宏观事件日预判(2026-08-20 用户要求: FOMC这种要有提前情景推演):
    事件日当天生成一次(美东日期), 市场预期+三情景ETH路径+关键位, 落盘供卡片/简报/推送"""
    try:
        today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        ev = MACRO_EVENTS.get(today)
        if not ev:
            return None
        mem = load_news_memory()
        ep = mem.get("event_preview") or {}
        if ep.get("date") == today and ep.get("text"):
            return ep["text"]
        wv = (mem.get("world_view") or {}).get("text", "无")
        lv = compute_levels(sym, "LONG")
        lv_txt = f"摆动高${lv['swing_high']:.2f} 摆动低${lv['swing_low']:.2f}" if lv else "无"
        pm = next((x for x in polymarket_odds() if x["topic"] == "联储降息"), None)
        pm_txt = f"预测市场联储降息概率{pm['prob']}%" if pm else "无赔率数据"
        px = fetch_fast_price(sym)
        sys_p = ("你是只交易ETH的加密交易员(BTC为联动参照)。今天有宏观事件, 写事件预判, 3-4条, 每条≤40字: "
                 "1)市场当前预期是什么 2)鸽派情景ETH怎么走(到哪个位) 3)鹰派情景ETH怎么走(到哪个位) "
                 "4)现在该做什么准备(挂单价/不动/减仓)。大白话, 落到给定关键位上, 不喊口号。")
        user = f"事件: {ev}\n盘面综述: {wv}\n现价: ${px}\n关键位: {lv_txt}\n{pm_txt}"
        r = _http.post(DS_API_URL, json={"model": DS_MODEL, "messages": [
            {"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            "max_tokens": 400, "temperature": 0.3},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=25)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text:
                mem["event_preview"] = {"date": today, "text": text}
                save_news_memory(mem)
                return text
    except Exception as e:
        print(f"WARN: 事件预判失败 {type(e).__name__}: {e}")
    return None


def update_world_view(sym, force=False):
    """盘面综述(2026-08-20 用户要求: 别呆板罗列, 要像交易员一样形成叙事):
    事件线+近期新闻+预测市场赔率+价格动作+宏观日历 → 一段4-6句的当前市场叙事, 30分钟刷新, 注入两层简报"""
    mem = load_news_memory()
    wv = mem.get("world_view") or {}
    if not force and time.time() - wv.get("ts", 0) < 1800 and wv.get("text"):
        return wv["text"]
    try:
        threads = _threads_line(mem) or "无"
        now = time.time()
        recent = " | ".join(t for ts, t in RECENT_NEWS if now - ts < 12 * 3600) or "无"
        pm = polymarket_odds()
        pm_txt = " | ".join(f"{x['topic']}{x['prob']}%" for x in pm) or "无"
        c4 = fetch_klines(sym, "4h", 50)["close"]
        chg24 = float((c4.iloc[-1] / c4.iloc[-7] - 1) * 100)
        chg7d = float((c4.iloc[-1] / c4.iloc[-43] - 1) * 100) if len(c4) >= 43 else 0.0
        macro = _macro_line() or "今日无宏观事件日"
        px = fetch_fast_price(sym)
        lv = compute_levels(sym, "LONG")
        lv_txt = f"摆动高${lv['swing_high']:.2f} 摆动低${lv['swing_low']:.2f} EMA20 ${lv['ema20']:.2f}" if lv else "无"
        sys_p = ("你是只交易ETH的加密交易员。根据给定材料写当前盘面综述(交易笔记), 分两部分:\n"
                 "【完整版】4-6句大白话: 市场在交易什么主线, 对ETH的传导路径, 风险偏好, 哪些事件在发酵(走向), "
                 "未来24-48小时最该盯什么。一切围绕ETH走势说话, 美股/大宗/地缘只提到它们影响ETH的那一环; "
                 "旧闻不引用; 价格数字只能用材料里给的(现价/摆动位/EMA), 不许编点位。\n"
                 "【卡片版】最后一行以\"卡片版:\"开头, 3-4句≤140字: 主线+对ETH的传导+一个关键位+最该盯的点, 别砍成口号。\n"
                 "像写给同行的晨会笔记, 不是新闻播报。")
        user = (f"今天是{pd.Timestamp.now(tz='Asia/Shanghai'):%Y-%m-%d}(北京) | "
                f"现价: ${px} | 关键位: {lv_txt}\n"
                f"涨跌: {sym} 24h{chg24:+.1f}% 7日{chg7d:+.1f}%\n{macro}\n"
                f"{threads}\n近期新闻: {recent}\n预测市场: {pm_txt}")
        r = _http.post(DS_API_URL, json={"model": DS_MODEL, "messages": [
            {"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            "max_tokens": 400, "temperature": 0.3},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=25)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text:
                # 拆卡片版/完整版
                card = None
                m = re.search(r"卡片版[:：]\s*(.+)$", text, re.S)
                if m:
                    card = m.group(1).strip()
                    text = text[:m.start()].replace("【完整版】", "").strip()
                mem["world_view"] = {"text": text, "card": card or text[:80], "ts": time.time()}
                save_news_memory(mem)
                return text
    except Exception as e:
        print(f"WARN: 盘面综述生成失败 {type(e).__name__}: {e}")
    return wv.get("text")


def _recent_news_lines(hours=12, limit=5):
    """简报注入: 盘面综述(叙事) + 近12h即时新闻(中文)。综述优先, 没有才退化到事件线罗列"""
    lines = []
    try:
        mem0 = load_news_memory()
        ep0 = (mem0.get("event_preview") or {})
        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        if ep0.get("text") and preview_visible(ep0, today_et):
            lines.append(f"事件预判({MACRO_EVENTS.get(today_et)}): {ep0['text']}")
        wv = (mem0.get("world_view") or {}).get("text")
        if wv:
            lines.append(f"盘面综述: {wv}")
        else:
            tl = _threads_line(mem0)
            if tl:
                lines.append(tl)
    except Exception:
        pass
    try:
        now = time.time()
        items = [t for ts, t in RECENT_NEWS if now - ts < hours * 3600][-limit:]
        if items:
            lines.append("近期新闻: " + " | ".join(items))
    except Exception:
        pass
    return lines


def _drop_unclosed(df, interval):
    """丢弃最后一根未收盘K线: open_time + 周期 > 当前时间 (按实际请求的 interval 算毫秒)"""
    if df.empty:
        return df
    ms = INTERVAL_MS.get(interval)
    if not ms:
        return df
    last_open = df.index[-1]
    if getattr(last_open, "tzinfo", None) is not None:
        last_open = last_open.tz_localize(None)
    if last_open.value // 10**6 + ms > int(time.time() * 1000):
        return df.iloc[:-1]
    return df


def _is_stale(df, interval):
    """数据停滞判定: 最新 bar 距今超过 2×周期, 或最近3根成交量全为0(低流动性交易对价格定格, 如 Binance.US 的 ETHUSDC)"""
    ms = INTERVAL_MS.get(interval)
    if not ms or df.empty:
        return False
    # 最近3根成交额合计 < $300 视为停滞 (纯零量或零星 dust 交易, 如 Binance.US 的 ETHUSDC)
    if "volume" in df.columns and len(df) >= 3:
        last3 = df.tail(3)
        if float((last3["volume"] * last3["close"]).sum()) < 300.0:
            return True
    last_open = df.index[-1]
    if getattr(last_open, "tzinfo", None) is not None:
        last_open = last_open.tz_localize(None)
    return last_open.value // 10**6 < int(time.time() * 1000) - 2 * ms


def _fetch_coinbase(symbol, interval, limit, start_ms=None, drop_incomplete=True):
    """Coinbase Exchange candles(美国可访问, 免key); 实测 ETH-USDC/BTC-USDC 已下架, 统一用 USD 对"""
    cb_sym = {"ETHUSDT": "ETH-USD", "ETHUSDC": "ETH-USD",
              "BTCUSDT": "BTC-USD", "BTCUSDC": "BTC-USD"}.get(symbol)
    # Coinbase granularity 仅支持 60/300/900/3600/21600/86400, 无 4h(14400 会 400); 4h 由 Hyperliquid 源提供
    gran = {"5m": 300, "15m": 900, "1h": 3600, "1d": 86400}.get(interval)
    if not cb_sym or not gran:
        return None
    url = (f"https://api.exchange.coinbase.com/products/{cb_sym}/candles"
           f"?granularity={gran}&limit={limit}")
    if start_ms:  # 注意 API 单次上限 300 根
        t0 = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        t1 = pd.Timestamp(start_ms + limit * gran * 1000, unit="ms", tz="UTC")
        url += f"&start={t0.isoformat()}&end={t1.isoformat()}"
    data = http_get_json(url)
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        return None
    # [time秒, low, high, open, close, volume], 最新在前
    df = pd.DataFrame(data, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["amount"] = df["volume"] * df["close"]
    df["taker_base"] = np.nan  # Coinbase 无主动买卖字段
    df = df[["date", "open", "high", "low", "close", "volume", "amount", "taker_base"]].set_index("date").sort_index()
    df = df.tail(limit)
    return _drop_unclosed(df, interval) if drop_incomplete else df


def _fetch_hyperliquid_klines(symbol, interval, limit, start_ms=None, drop_incomplete=True):
    """Hyperliquid candleSnapshot(美国可访问, 免key): 支持 4h, 补 Coinbase 无 4h 的缺口"""
    coin = {"ETHUSDT": "ETH", "ETHUSDC": "ETH", "BTCUSDT": "BTC", "BTCUSDC": "BTC"}.get(symbol)
    ms = INTERVAL_MS.get(interval)
    if not coin or not ms or interval not in ("5m", "15m", "1h", "4h", "1d"):
        return None
    t1 = start_ms + limit * ms if start_ms else int(time.time() * 1000)
    t0 = t1 - limit * ms
    d = _http.post("https://api.hyperliquid.xyz/info", timeout=10, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": t0, "endTime": t1}}).json()
    if not isinstance(d, list) or not d:
        return None
    df = pd.DataFrame([{
        "date": pd.to_datetime(int(c["t"]), unit="ms"),
        "open": float(c["o"]), "high": float(c["h"]), "low": float(c["l"]),
        "close": float(c["c"]), "volume": float(c["v"])} for c in d])
    df["amount"] = df["volume"] * df["close"]
    df["taker_base"] = np.nan  # HL 无主动买卖字段
    df = df.set_index("date").sort_index().tail(limit)
    return _drop_unclosed(df, interval) if drop_incomplete else df


def fetch_klines(symbol, interval="15m", limit=200, start_ms=None, drop_incomplete=True):
    """USDC对: Coinbase → Hyperliquid → fapi; USDT对: binance.us → Coinbase → fapi; 最终兜底 yfinance (fapi 在美国被 geo-block);
    drop_incomplete=False 保留未收盘K线(历史结算用); 数据停滞的源自动跳过"""
    if drop_incomplete:
        hit = _kline_cache.get((symbol, interval, limit))
        if hit and time.time() - hit[0] < 60:
            return hit[1]
    extra = f"&startTime={start_ms}" if start_ms else ""
    df = None
    stale_df = None
    # Binance.US 的 USDC 对子是死盘(ETHUSDC 24h 成交仅数个 ETH, 最新价常偏离真实价数美元),
    # USDC 对子跳过 binance.us, 首选 Coinbase USD 对(深度足, USD/USDC 价差可忽略)
    if symbol.endswith("USDC"):
        sources = [
            ("coinbase", None),
            ("hyperliquid", None),
            ("binance", f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}{extra}"),
        ]
    else:
        sources = [
            ("binance", f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}{extra}"),
            ("coinbase", None),
            ("binance", f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}{extra}"),
        ]
    for kind, url in sources:
        for _ in range(2):  # 每个源失败重试1次再切下一个源
            try:
                if kind == "coinbase":
                    cand = _fetch_coinbase(symbol, interval, limit, start_ms, drop_incomplete)
                elif kind == "hyperliquid":
                    cand = _fetch_hyperliquid_klines(symbol, interval, limit, start_ms, drop_incomplete)
                else:
                    data = http_get_json(url)
                    cand = _parse_binance(data, interval, drop_incomplete) if (
                        isinstance(data, list) and len(data) > 0 and isinstance(data[0], list)) else None
                if cand is None or cand.empty:
                    continue
                # staleness 检查在丢弃未收盘 bar 之后对最新已收盘 bar 做;
                # start_ms 拉历史(结算用)不查停滞
                if not start_ms and _is_stale(cand, interval):
                    stale_df = cand
                    break
                df = cand
                break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        cand = _fetch_yfinance(symbol, interval, limit, drop_incomplete)
        if cand.empty:
            pass
        elif not start_ms and _is_stale(cand, interval):
            stale_df = cand
        else:
            df = cand
    if df is None:
        if stale_df is not None:
            print(f"WARN: {symbol} {interval} 所有数据源停滞, 使用最后可用数据")
            df = stale_df
        else:
            df = pd.DataFrame()
    if drop_incomplete:
        _kline_cache[(symbol, interval, limit)] = (time.time(), df)
    return df
def _parse_binance(data, interval, drop_incomplete=True):
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","amount","trades","taker_base","taker_quote","ignore"
    ])
    for c in ["open","high","low","close","volume","amount","taker_base"]:
        df[c] = df[c].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["date","open","high","low","close","volume","amount","taker_base"]].set_index("date").sort_index()
    return _drop_unclosed(df, interval) if drop_incomplete else df


def _fetch_yfinance(symbol, interval, limit, drop_incomplete=True):
    """US-friendly data via Yahoo Finance"""
    try:
        import yfinance as yf
        yf_sym = symbol.replace("USDT", "-USD").replace("USDC", "-USD")  # ETHUSDT/ETHUSDC → ETH-USD
        tf_map = {"5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d"}
        period_map = {"5m": "5d", "15m": "7d", "1h": "30d", "4h": "60d", "1d": "730d"}
        tf = tf_map.get(interval, "15m")
        period = period_map.get(interval, "7d")
        df = yf.download(yf_sym, interval=tf, period=period, progress=False)
        if df.empty:
            raise ValueError("no data")
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = df["close"]
        df["amount"] = df["volume"] * df["close"]
        df["taker_base"] = np.nan  # yfinance 无主动买盘字段
        df = df[["open", "high", "low", "close", "volume", "amount", "taker_base"]]
        df = df.tail(limit).sort_index()
        return _drop_unclosed(df, interval) if drop_incomplete else df
    except Exception:
        return pd.DataFrame()


def build_panel(df):
    idx = df.index
    syms = ["SELF","DUMMY"]
    panel = {}
    for f in ["open","high","low","close","volume","amount"]:
        pdf = pd.DataFrame(index=idx, columns=syms, dtype=float)
        pdf["SELF"] = df[f].values
        pdf["DUMMY"] = df[f].values
        panel[f] = pdf
    panel["vwap"] = (panel["high"]+panel["low"]+panel["close"])/3.0
    return panel


def compute_factor_value(mod_name, panel):
    try:
        mod = importlib.import_module(mod_name)
        fv = mod.compute(panel)
        if isinstance(fv, pd.DataFrame):
            return fv.iloc[:, 0].values
        return np.array(fv).flatten()
    except Exception as e:
        print(f"  因子计算失败: {mod_name}: {e}")
        return None


def preload_factors():
    """Pre-import all factor modules"""
    all_factors = set(m for _, m, _, _ in BTC_FACTORS + ETH_FACTORS)
    for mod_name in sorted(all_factors):
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass


# ═══════════ Al Brooks 价格行为 (6票) ═══════════
# 精髓: 先判状态(趋势/区间)再定打法 —— 趋势里等双腿回调顺势进场(H2/L2),
# 极值处的三推楔形/双顶底做反转, 突破失败=被套交易者的反向信号,
# 判断不了就当震荡区间(市场80%时间在区间里)

def _swing_points(arr, n=2):
    """Fractal 摆动点: ±n 根内的局部极值"""
    a = np.asarray(arr, float)
    hi = np.zeros(len(a), bool)
    lo = np.zeros(len(a), bool)
    for i in range(n, len(a) - n):
        seg = a[i - n:i + n + 1]
        hi[i] = a[i] == seg.max() and a[i] > seg[0] and a[i] > seg[-1]
        lo[i] = a[i] == seg.min() and a[i] < seg[0] and a[i] < seg[-1]
    return hi, lo


def brooks_analyze(df):
    """返回 {state, spike, always_in, votes=[(name,±1)], setups=[str], sl={long,short}}"""
    res = {"state": "range", "spike": 0, "always_in": 0, "votes": [],
           "setups": [], "sl": {"long": None, "short": None}}
    c = df["close"].values; h = df["high"].values
    l = df["low"].values; o = df["open"].values
    n = len(c)
    if n < 45:
        return res
    atr = float(np.mean(h[-14:] - l[-14:]))
    if atr <= 0:
        return res
    rng = np.maximum(h - l, 1e-9)
    body = c - o
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values

    # ── spike: 连续≥3根强趋势K线(实体>60%振幅 且 振幅>0.8ATR) = 突破段 ──
    strong = np.where(body > 0.6 * rng, 1, np.where(-body > 0.6 * rng, -1, 0))
    strong = strong * (rng > 0.8 * atr)
    spike = 0; cur = 0; prev = 0
    for i in range(n - 30, n):
        d = strong[i]
        cur = cur + 1 if d != 0 and d == prev else (1 if d != 0 else 0)
        if d != 0:
            prev = d
        if cur >= 3:
            spike = d
    res["spike"] = int(spike)

    # ── 市场状态: HH/HL 结构 或 spike 同向延续 = 趋势; 否则区间 ──
    seg = c[-20:]
    hh_hl = seg[10:].max() > seg[:10].max() and seg[10:].min() > seg[:10].min()
    ll_lh = seg[10:].max() < seg[:10].max() and seg[10:].min() < seg[:10].min()
    if hh_hl or (spike == 1 and c[-1] > ema20[-1]):
        res["state"] = "trend_up"
    elif ll_lh or (spike == -1 and c[-1] < ema20[-1]):
        res["state"] = "trend_down"

    # ── Always In: 最近25根内 强K线收盘突破前20根极值的方向; 无则价格vs EMA20 ──
    ai = 0
    for i in range(n - 1, max(n - 25, 21), -1):
        if strong[i] == 1 and c[i] > h[i - 20:i].max():
            ai = 1; break
        if strong[i] == -1 and c[i] < l[i - 20:i].min():
            ai = -1; break
    if ai == 0:
        ai = 1 if c[-1] >= ema20[-1] else -1
    res["always_in"] = ai

    sw_hi = np.where(_swing_points(h)[0])[0]
    sw_lo = np.where(_swing_points(l)[1])[0]

    # ── H2/L2 双腿回调: 趋势中最经典入场。第二腿刚形成且价格转回趋势方向 ──
    def two_leg(direction):
        pts = sw_lo if direction == 1 else sw_hi
        opp = sw_hi if direction == 1 else sw_lo
        if len(pts) < 2:
            return None
        P2, P1 = pts[-1], pts[-2]
        if n - 1 - P2 > 5 or P2 - P1 < 3:
            return None
        if not len(opp[(opp > P1) & (opp < P2)]):  # 两腿之间必须有反弹腿
            return None
        prior = opp[opp < P1]
        if not len(prior) or n - prior[-1] > 30:
            return None
        arr_p = l if direction == 1 else h
        # 第二腿破第一腿太多 = 结构破坏, 可能反转而非回调
        if direction == 1 and arr_p[P2] < arr_p[P1] - 0.5 * atr:
            return None
        if direction == -1 and arr_p[P2] > arr_p[P1] + 0.5 * atr:
            return None
        if direction == 1 and c[-1] <= c[-2]:  # 需转强确认
            return None
        if direction == -1 and c[-1] >= c[-2]:
            return None
        return float(arr_p[P2] - 0.2 * atr) if direction == 1 else float(arr_p[P2] + 0.2 * atr)

    # ── 三推楔形: 三个递推极值但每腿变短 = 动能衰竭, 反转 ──
    def wedge(direction):
        pts = sw_hi if direction == -1 else sw_lo  # -1=楔形顶看空
        if len(pts) < 3:
            return False
        P1, P2, P3 = pts[-3:]
        arr_p = h if direction == -1 else l
        if n - 1 - P3 > 6:
            return False
        if direction == -1:
            if not (arr_p[P1] < arr_p[P2] < arr_p[P3]):
                return False
            if arr_p[P3] - arr_p[P2] > arr_p[P2] - arr_p[P1]:
                return False
            return c[-1] < c[-2]
        if not (arr_p[P1] > arr_p[P2] > arr_p[P3]):
            return False
        if arr_p[P1] - arr_p[P2] > arr_p[P2] - arr_p[P3]:
            return False
        return c[-1] > c[-2]

    # ── 双顶/双底: 二次测试同一极值不破, 反向确认K线触发 ──
    def double_tb(direction):
        pts = sw_hi if direction == -1 else sw_lo
        if len(pts) < 2:
            return None
        P1, P2 = pts[-2], pts[-1]
        arr_p = h if direction == -1 else l
        if P2 - P1 < 5 or n - 1 - P2 > 6:
            return None
        if abs(arr_p[P2] - arr_p[P1]) > 0.5 * atr:  # 两顶/底必须接近
            return None
        if direction == -1 and c[-1] >= l[-2]:  # 跌破前K低点确认
            return None
        if direction == 1 and c[-1] <= h[-2]:
            return None
        return float(arr_p[P2] + 0.2 * atr) if direction == -1 else float(arr_p[P2] - 0.2 * atr)

    # ── 假突破: 盘中破前20根极值但收回 → 突破者被套, 80%概率反向 ──
    def failed_breakout():
        for i in range(max(n - 5, 22), n):
            phi = h[i - 20:i].max(); plo = l[i - 20:i].min()
            if h[i] > phi and c[i] < phi and c[-1] < phi:
                return -1
            if l[i] < plo and c[i] > plo and c[-1] > plo:
                return 1
        return 0

    # ── 信号K线质量: 实体>50%振幅 且 收盘在有利1/3, 且与 Always In 同向 ──
    i = n - 1
    qbar = 0
    if rng[i] > 0.5 * atr:
        if body[i] > 0.5 * rng[i] and c[i] > l[i] + 0.66 * rng[i]:
            qbar = 1
        elif -body[i] > 0.5 * rng[i] and c[i] < l[i] + 0.33 * rng[i]:
            qbar = -1

    # ══ 汇总6票 ══
    v = res["votes"]
    if res["state"] == "trend_up":
        v.append(("BROOKS_市场状态", 1))
    elif res["state"] == "trend_down":
        v.append(("BROOKS_市场状态", -1))
    v.append(("BROOKS_AlwaysIn", ai))

    sl = two_leg(1)
    if sl and res["state"] != "trend_down" and ai == 1:
        v.append(("BROOKS_双腿回调H2", 1))
        res["setups"].append("H2双腿回调做多")
        res["sl"]["long"] = sl
    sl = two_leg(-1)
    if sl and res["state"] != "trend_up" and ai == -1:
        v.append(("BROOKS_双腿回调L2", -1))
        res["setups"].append("L2双腿回调做空")
        res["sl"]["short"] = sl

    # 反转形态合计1票, 避免楔形+双顶同时出现时的重复计票
    rev = []
    if wedge(-1):
        rev.append(-1); res["setups"].append("三推楔形顶")
    if wedge(1):
        rev.append(1); res["setups"].append("三推楔形底")
    dt = double_tb(-1)
    if dt:
        rev.append(-1); res["setups"].append("双顶"); res["sl"]["short"] = dt
    db = double_tb(1)
    if db:
        rev.append(1); res["setups"].append("双底"); res["sl"]["long"] = db
    if rev:
        v.append(("BROOKS_反转形态", max(set(rev), key=rev.count)))

    fb = failed_breakout()
    if fb:
        v.append(("BROOKS_假突破", fb))
        res["setups"].append("假突破陷阱→反向")

    if qbar and qbar == ai:
        v.append(("BROOKS_信号K线", qbar))
    return res


# Brooks 各票的大白话解释
BROOKS_TXT = {
    "BROOKS_市场状态":   {1: "市场处于上升趋势，是顺势做多的环境", -1: "市场处于下降趋势，是顺势做空的环境"},
    "BROOKS_AlwaysIn":  {1: "最近一次强势突破向上，场内方向偏多", -1: "最近一次强势突破向下，场内方向偏空"},
    "BROOKS_双腿回调H2": {1: "出现H2双腿回调形态，是趋势里胜率最高的做多入场"},
    "BROOKS_双腿回调L2": {-1: "出现L2双腿回调形态，是趋势里胜率最高的做空入场"},
    "BROOKS_反转形态":   {1: "底部出现反转形态(楔形底/双底)，跌不动了", -1: "顶部出现反转形态(楔形顶/双顶)，涨不动了"},
    "BROOKS_假突破":     {1: "向下假突破，追空的人被套，反弹概率大", -1: "向上假突破，追多的人被套，回落概率大"},
    "BROOKS_信号K线":    {1: "最新K线收得很强，支持做多", -1: "最新K线收得很弱，支持做空"},
}


def vote(symbol, config):
    """Factor voting for a single symbol"""
    df = fetch_klines(symbol, "15m", LOOKBACK)
    panel = build_panel(df)

    results = []
    bullish = 0
    bearish = 0

    for name, mod_name, ic_sign, display in config:
        values = compute_factor_value(mod_name, panel)
        if values is None or len(values) < 30:
            print(f"  因子失效跳过: {display} ({mod_name})")
            continue

        current = values[-1]
        window = values[-21:-1]
        mean20 = np.nanmean(window)
        std20 = np.nanstd(window)
        if np.isnan(mean20) or mean20 == 0:
            continue
        # 原始值量级差异巨大, 用z分数衡量相对强弱, 推送才有意义
        z = (current - mean20) / std20 if std20 > 0 else 0.0

        if ic_sign == +1:
            direction = "🟢" if current > mean20 else "🔴"
            if current > mean20: bullish += 1
            else: bearish += 1
        else:
            direction = "🔴" if current > mean20 else "🟢"
            if current > mean20: bearish += 1
            else: bullish += 1

        results.append({
            "name": display,
            "z": round(float(z), 2),
            "txt": f"{z_word(z)}近20期均值",
            "direction": direction,
        })

    # ═══ NOFX 风格趋势确认 (4票) ═══
    try:
        df4 = fetch_klines(symbol, "4h", 80)
        df15 = fetch_klines(symbol, "15m", 80)
        df5 = fetch_klines(symbol, "5m", 80)

        c4 = df4["close"]
        ema20_4 = c4.ewm(span=20, adjust=False).mean()
        ema50_4 = c4.ewm(span=50, adjust=False).mean()
        if ema20_4.iloc[-1] > ema50_4.iloc[-1]:
            bullish += 1
            results.append({"name": "NOFX_4H趋势", "txt": "4小时均线多头排列，大级别趋势向上", "direction": "🟢"})
        else:
            bearish += 1
            results.append({"name": "NOFX_4H趋势", "txt": "4小时均线空头排列，大级别趋势向下", "direction": "🔴"})

        c15 = df15["close"]
        ema20_15 = c15.ewm(span=20, adjust=False).mean()
        if c15.iloc[-1] > ema20_15.iloc[-1]:
            bullish += 1
            results.append({"name": "NOFX_15M方向", "txt": "15分钟价格站上EMA20，短线偏多", "direction": "🟢"})
        else:
            bearish += 1
            results.append({"name": "NOFX_15M方向", "txt": "15分钟价格跌破EMA20，短线偏空", "direction": "🔴"})

        c5 = df5["close"]
        ema20_5 = c5.ewm(span=20, adjust=False).mean()
        if c5.iloc[-1] > ema20_5.iloc[-1]:
            bullish += 1
            results.append({"name": "NOFX_5M确认", "txt": "5分钟价格站上EMA20，入场时机配合", "direction": "🟢"})
        else:
            bearish += 1
            results.append({"name": "NOFX_5M确认", "txt": "5分钟价格跌破EMA20，入场时机配合空头", "direction": "🔴"})

        macd_15 = c15.ewm(span=12, adjust=False).mean() - c15.ewm(span=26, adjust=False).mean()
        # 近3根均值 vs 前3根均值, 过滤单根噪声
        if len(macd_15) >= 6 and macd_15.iloc[-3:].mean() > macd_15.iloc[-6:-3].mean():
            bullish += 1
            results.append({"name": "NOFX_MACD动量", "txt": "15分钟上涨动能在增强", "direction": "🟢"})
        else:
            bearish += 1
            results.append({"name": "NOFX_MACD动量", "txt": "15分钟上涨动能在减弱", "direction": "🔴"})
    except Exception as e:
        print(f"  NOFX趋势确认失败: {e}")

    # ═══ Al Brooks 价格行为 (6票) ═══
    ba = None
    try:
        ba = brooks_analyze(fetch_klines(symbol, "15m", 120))
        for name, d in ba["votes"]:
            txt = BROOKS_TXT.get(name, {}).get(d, "")
            if d > 0:
                bullish += 1
                results.append({"name": name, "txt": txt, "direction": "🟢"})
            else:
                bearish += 1
                results.append({"name": name, "txt": txt, "direction": "🔴"})
    except Exception as e:
        print(f"  Brooks分析失败: {e}")

    total = bullish + bearish

    if bullish > bearish:
        signal = "LONG"
        votes = bullish
    elif bearish > bullish:
        signal = "SHORT"
        votes = bearish
    else:
        signal = "NEUTRAL"
        votes = 0  # 无方向时票数无意义

    strength = "⚡强" if votes >= STRONG else "📊" if votes >= MIN_VOTES else "❌不够"

    return {
        "symbol": symbol,
        "signal": signal,
        "votes": f"{votes}/{total}",
        "strength": strength,
        "bullish": bullish,
        "bearish": bearish,
        "price": fetch_fast_price(symbol) or float(df["close"].iloc[-1]),  # 实时ticker优先; K线已收盘bar最多滞后15分钟(2026-08-08 价格不准的根因)
        "details": results,
        "brooks": ba,
    }


def _escape_html(text):
    """保留 <b></b>, 其余 & < > 转义 (占位符替换法)"""
    text = text.replace("&", "&amp;")
    text = text.replace("<b>", "\x01").replace("</b>", "\x02")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("\x01", "<b>").replace("\x02", "</b>")


def send_telegram(text):
    """同步发送, 10s 超时; 整卡包 <pre> 等宽块(2026-08-08): Telegram 比例字体下全角空格宽度不一, 列对齐全废"""
    try:
        r = _http.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                       data={"chat_id": TELEGRAM_CHAT_ID, "text": f"<pre>{_escape_html(text)}</pre>",
                             "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def send_photo(path, caption):
    """Send chart image with compact signal card as caption"""
    try:
        with open(path, "rb") as f:
            r = _http.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                           data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                           files={"photo": f}, timeout=25)
        return r.status_code == 200
    except Exception:
        return False


def load_history():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN: 状态文件损坏({e}), 备份为 .bak 并重建")
            try:
                os.replace(STATE_FILE, STATE_FILE + ".bak")
            except OSError:
                pass
    return {"signals": [], "win_rate": {"total": 0, "wins": 0, "recent20": []}}


def save_history(h):
    # 原子写入: 先写 .tmp 再 replace, 避免中途崩溃写坏状态文件
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(h, f, indent=2)
    os.replace(tmp, STATE_FILE)


def load_journal():
    """裁判判例: 同 load_history 的损坏容错模式"""
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN: 判例文件损坏({e}), 备份为 .bak 并重建")
            try:
                os.replace(JOURNAL_FILE, JOURNAL_FILE + ".bak")
            except OSError:
                pass
    return {"entries": []}


def save_journal(j):
    tmp = JOURNAL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(j, f, indent=2)
    os.replace(tmp, JOURNAL_FILE)


def load_lessons():
    """AI 错题本: 无文件/损坏返回 None"""
    if os.path.exists(LESSONS_FILE):
        try:
            with open(LESSONS_FILE, "r") as f:
                d = json.load(f)
            if isinstance(d.get("lessons"), list) and d["lessons"]:
                return d
        except Exception as e:
            print(f"WARN: 错题本损坏({e}), 备份为 .bak")
            try:
                os.replace(LESSONS_FILE, LESSONS_FILE + ".bak")
            except OSError:
                pass
    return None


def save_lessons(d):
    tmp = LESSONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LESSONS_FILE)


def save_prediction(sym, sig, px, plan=None):
    h = load_history()
    entry = {"symbol": sym, "direction": sig, "entry": px,
             "time": pd.Timestamp.now().isoformat(), "verified": False,
             "correct": None, "exit_px": None, "pnl_pct": None}
    if plan:
        entry["sl"] = plan["sl"]
        entry["tp1"] = plan["tp1"]
        entry["tp2"] = plan["tp2"]
    h["signals"].append(entry)
    # Keep last 100
    if len(h["signals"]) > 100:
        h["signals"] = h["signals"][-100:]
    save_history(h)


def verify_predictions():
    """Path-based verification: check if TP1/TP2 or SL hit first using 15m bars"""
    h = load_history()
    messages = []
    now = pd.Timestamp.now(tz="UTC")

    for pred in h["signals"]:
        if pred["verified"]:
            continue
        pred_time = pd.Timestamp(pred["time"])
        if pred_time.tz is None:
            pred_time = pred_time.tz_localize("UTC")
        if (now - pred_time).total_seconds() < 1800:
            continue

        # 从信号时间起取完整K线(含未收盘), 给足走完 TP/SL 的时间
        try:
            df = fetch_klines(pred["symbol"], "15m", 1000,
                              start_ms=int(pred_time.timestamp() * 1000),
                              drop_incomplete=False)
            if df.empty:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            bars = df[df.index >= pred_time]
            if len(bars) < 2:
                continue
        except Exception:
            continue

        direction = pred["direction"]
        entry = pred["entry"]
        if pred.get("sl") and pred.get("tp2"):
            # 新数据: 用信号时的计划价位结算, 与推送建议一致
            sl, tp2 = float(pred["sl"]), float(pred["tp2"])
        else:
            # 旧数据 fallback: ±0.6% 价格波动 (50x杠杆 ≈ 30%盈亏, 1:1)
            if direction == "LONG":
                sl, tp2 = entry * 0.994, entry * 1.006
            else:
                sl, tp2 = entry * 1.006, entry * 0.994

        # Walk bars: which hit first?
        result = "timeout"
        try:
            exit_px = float(bars["close"].iloc[-1])
        except Exception:
            exit_px = entry  # fallback
        for _, bar in bars.iterrows():
            try:
                hi = float(bar["high"]) if "high" in bar.index else float(bar.iloc[1]) if len(bar) > 1 else entry
                lo = float(bar["low"]) if "low" in bar.index else float(bar.iloc[2]) if len(bar) > 2 else entry
            except Exception:
                continue
            if direction == "LONG":
                if lo <= sl:
                    result = "sl"; exit_px = sl; break
                if hi >= tp2:
                    result = "tp2"; exit_px = tp2; break
            else:
                if hi >= sl:
                    result = "sl"; exit_px = sl; break
                if lo <= tp2:
                    result = "tp2"; exit_px = tp2; break

        pnl = (exit_px - entry) / entry * 100
        if direction == "SHORT":
            pnl = -pnl

        # 未满24小时且未触发: 不结算, 下一轮继续跟踪
        if result == "timeout" and (now - pred_time).total_seconds() < 86400:
            continue

        pred["verified"] = True
        pred["result"] = result
        pred["correct"] = result == "tp2"
        pred["exit_px"] = exit_px
        pred["pnl_pct"] = round(pnl, 2)

        # 超时不计入胜率, 只统计明确触发 TP2/SL 的信号
        if result in ("tp2", "sl"):
            h["win_rate"]["total"] += 1
            if pred["correct"]:
                h["win_rate"]["wins"] += 1
            h["win_rate"]["recent20"].append(pred["correct"])
            if len(h["win_rate"]["recent20"]) > 20:
                h["win_rate"]["recent20"] = h["win_rate"]["recent20"][-20:]

        # Follow-up message
        result_emoji = {"tp2": "🎯", "sl": "❌", "timeout": "⏰"}.get(result, "⏰")
        # 显示实际结算百分比, 不再写死 ±0.6%
        tp_pct = (tp2 - entry) / entry * 100
        sl_pct = (sl - entry) / entry * 100
        if direction == "SHORT":
            tp_pct, sl_pct = -tp_pct, -sl_pct
        result_cn = {"tp2": f"止盈({tp_pct:+.2f}%)", "sl": f"止损({sl_pct:+.2f}%)",
                     "timeout": "超时未触发"}.get(result, "超时")
        pnl_str = f"+{pnl:.2f}%" if pnl > 0 else f"{pnl:.2f}%"
        messages.append(
            f"{result_emoji} 信号结算 | {pred['symbol']} {'做多' if direction=='LONG' else '做空'}\n"
            f"入场: ${entry:.2f} → 出场: ${exit_px:.2f}\n"
            f"结果: {result_cn} | 盈亏: {pnl_str}"
        )

    save_history(h)
    return messages


def record_judge(result, judge4, judge15):
    """双层判决落盘: 4h层(dir4h/conf4h/mag_tier)与15m层(dir15m/conf15m)分开记;
    双层方向都不变且1h内有记录则去重; 上限500条"""
    ba = result.get("brooks") or {}
    lv = compute_levels(result["symbol"], result["signal"])
    d4, d15 = judge4.get("direction"), judge15.get("direction")
    j = load_journal()
    if j["entries"]:
        last = j["entries"][-1]
        if (last["symbol"] == result["symbol"] and last.get("dir4h") == d4 and last.get("dir15m") == d15
                and (pd.Timestamp.now() - pd.Timestamp(last["time"])).total_seconds() < 3600):
            return
    verdict = "执行" if (d4 or d15) else "观望"
    sig0 = result["signal"] if result["signal"] != "NEUTRAL" else (
        "LONG" if result["bullish"] >= result["bearish"] else "SHORT")
    # 交易计划只按 AI 实际主张的方向生成; 观望无计划, entry 仅作观望质量基准价
    plan = calc_trade_plan(result["symbol"], d4 or d15 or sig0, result["price"], ba) if verdict == "执行" else None
    j["entries"].append({
        "time": pd.Timestamp.now().isoformat(),
        "symbol": result["symbol"], "direction": d4 or d15,  # 观望=None, 结算不再用投票方向顶替(2026-08-20 修)
        "entry_px": plan["entry"] if plan else result["price"],
        "sl": plan["sl"] if plan else None, "tp2": plan["tp2"] if plan else None,
        "atr": round(lv["atr14"], 2) if lv else None,  # 方向分归一化基准
        "verdict": verdict,
        "confidence": max(judge4.get("confidence", -1), judge15.get("confidence", -1)),
        "dir4h": d4, "conf4h": judge4.get("confidence"),
        "dir15m": d15, "conf15m": judge15.get("confidence"),
        "mag_tier": judge4.get("mag_tier"),  # 幅度档只有4h层给
        "ai_direction": d4,  # 幅度档结算以4h层方向为准
        "reasons": (judge4.get("reasons") or [])[:2] + (judge15.get("reasons") or [])[:1],
        "rsi": round(lv["rsi14"]) if lv else None,
        "state": ba.get("state"), "votes": result["votes"],
        # 结算字段已删(2026-08-23 用户决策): 机械判对错对15m信号是假精确, 复盘走AI叙事(maybe_update_lessons)
    })
    if len(j["entries"]) > 500:
        j["entries"] = j["entries"][-500:]
    save_journal(j)


def maybe_update_lessons():
    """AI 叙事复盘(2026-08-23 取代机械结算, 用户决策: 15m平仓时间随机+扛单, 机械判对错是假精确):
    每积累20条新判决, AI 读自己最近的判决+4h后实际走势(原始涨跌, 不打分), 写3-5条人话教训"""
    try:
        entries = load_journal()["entries"]
        old = load_lessons()
        n = len(entries)
        if n < 20 or n - (old or {}).get("based_on_count", 0) < 20:
            return
        c15 = fetch_klines("ETHUSDC", "15m", 96 * 7)["close"]
        lines = []
        for e in entries[-20:]:
            et = pd.Timestamp(e["time"])
            if et.tz:
                et = et.tz_localize(None)
            later = c15[c15.index >= et + pd.Timedelta(hours=4)]
            if not len(later):
                continue
            chg = (float(later.iloc[0]) - float(e["entry_px"])) / float(e["entry_px"]) * 100
            d = {"LONG": "多", "SHORT": "空", None: "观望"}.get(e.get("direction"))
            rsn = (e.get("reasons") or [""])[0][:40]
            lines.append(f"{e['time'][5:16]} 判{d}@${float(e['entry_px']):.0f} → 4h后{chg:+.1f}% | 当时理由: {rsn}")
        if len(lines) < 10:
            return
        prompt = ("你是只交易ETH的专业交易员, 正在复盘自己最近的判决。逐条看: 判的方向、4小时后实际走了多少、当时的理由。"
                  "写3-5条人话教训: 什么局面下我的判断有效/失效、哪种理由其实是借口、下次怎么改。"
                  "不许用假精确的胜率词, 只说模式和局面。只输出JSON: {\"lessons\": [...]}\n\n" + "\n".join(lines))
        r = _http.post(DS_API_URL, json={
            "model": DS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900, "temperature": 0.2},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=25)
        if r.status_code != 200:
            print(f"WARN: 叙事复盘失败(HTTP {r.status_code}), 保留旧版")
            return
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        lessons = json.loads(m.group(0)).get("lessons") if m else None
        if not isinstance(lessons, list) or not lessons:
            print("WARN: 叙事复盘解析失败, 保留旧版")
            return
        save_lessons({"lessons": [str(x).strip() for x in lessons[:5]],
                      "updated_at": pd.Timestamp.now().isoformat(), "based_on_count": n})
        print(f"  叙事复盘已更新: {len(lessons)}条教训 (基于{n}条判决)")
    except Exception as e:
        print(f"WARN: 叙事复盘异常({e}), 保留旧版")


def get_signal_number():
    h = load_history()
    h["signal_counter"] = h.get("signal_counter", 0) + 1
    save_history(h)
    return h["signal_counter"]


def get_win_rate_stats():
    h = load_history()
    recent20 = h["win_rate"].get("recent20", [])
    recent10 = recent20[-10:] if len(recent20) >= 10 else recent20[-5:] if recent20 else []

    overall_wins = h["win_rate"]["wins"]
    overall_total = h["win_rate"]["total"]
    overall_rate = overall_wins / overall_total * 100 if overall_total > 0 else 0
    recent20_rate = sum(recent20) / len(recent20) * 100 if recent20 else 0
    recent10_rate = sum(recent10) / len(recent10) * 100 if recent10 else 0

    return {
        "overall": f"{overall_wins}/{overall_total}" if overall_total > 0 else "0/0",
        "overall_rate": round(overall_rate),
        "recent20_rate": round(recent20_rate),
        "recent10_rate": round(recent10_rate),
    }


def rsi_word(rsi):
    """RSI数值 → 大白话, 让人看懂是过热还是超卖"""
    if rsi >= 70: return f"RSI={rsi:.0f}(超买区，涨多了有回调风险)"
    if rsi >= 55: return f"RSI={rsi:.0f}(偏强)"
    if rsi >= 45: return f"RSI={rsi:.0f}(中性)"
    if rsi >= 30: return f"RSI={rsi:.0f}(偏弱)"
    return f"RSI={rsi:.0f}(超卖区，跌多了有反弹需求)"


def compute_direction_probs(symbol):
    """Compute direction probability for each timeframe using local momentum + volatility"""
    lines = []
    try:
        configs = [
            ("5分钟",  "5m",  0.015, 3,  "短线噪声大，参考权重低"),
            ("15分钟", "15m", 0.030, 6,  "主参考级别"),
            ("4小时",  "4h",  0.045, 20, "长周期趋势，权重大"),
        ]
        bars = []

        for tf_label, tf, ic, lookback, note in configs:
            df = fetch_klines(symbol, tf, 50)
            c = df["close"]
            v = df["volume"]
            h, l = df["high"], df["low"]

            # 1. Momentum: N-bar return
            if len(c) >= lookback + 1:
                mom = c.iloc[-1] / c.iloc[-lookback-1] - 1
                mom_z = np.sign(mom) * min(abs(mom) / 0.01, 3)  # cap at 3σ
            else:
                mom_z = 0

            # 2. Price position: RSV
            if len(c) >= 20:
                hh, ll = h.iloc[-20:].max(), l.iloc[-20:].min()
                rsv = (c.iloc[-1] - ll) / max(hh - ll, 1e-6)
                rsv_z = (rsv - 0.5) * 2
            else:
                rsv_z = 0

            # 3. Volume anomaly
            if len(v) >= 10:
                vol_ratio = v.iloc[-3:].mean() / max(v.iloc[-10:].mean(), 1e-6)
                vol_z = np.sign(vol_ratio - 1) * min(abs(vol_ratio - 1) * 3, 2)
            else:
                vol_z = 0

            # Composite: momentum(50%) + position(30%) + volume(20%)
            composite = mom_z * 0.5 + rsv_z * 0.3 + vol_z * 0.2

            # IC-based probability
            prob_up = 50 + ic * composite * 150
            prob_up = np.clip(prob_up, 30, 70)
            prob_down = 100 - prob_up

            bars.append(f"├ {tf_label}: 🟢涨{prob_up:.0f}% / 🔴跌{prob_down:.0f}% ({note})")

        lines.extend(bars)

    except Exception:
        lines.append("├ (数据暂不可用)")

    return lines


def calc_trade_plan(sym, sig, px, ba):
    """Brooks 形态止损优先(第二腿/双顶底极值外侧), 否则 ATR×0.6; 止损距离下限 0.8×ATR; 止盈 = 1:1 scalp + 测量目标"""
    atr_val = px * 0.003
    last_leg = atr_val * 3
    try:
        df = fetch_klines(sym, "15m", 80)
        atr_val = float((df["high"] - df["low"]).tail(14).mean())
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = px * 0.003
        recent = df["close"].iloc[-30:]
        last_leg = abs(float(recent.max() - recent.min()))
        if last_leg < atr_val:
            last_leg = atr_val * 3
    except Exception:
        pass

    sl_map = (ba or {}).get("sl") or {}
    sl_struct = sl_map.get("long" if sig == "LONG" else "short")
    min_risk = atr_val * 0.8  # 止损下限: 结构止损贴脸时用波动率兜底, 防止毛刺扫损
    if sig == "LONG":
        sl = sl_struct if (sl_struct and sl_struct < px) else px - atr_val * 0.6
        sl = min(sl, px - min_risk)
        tp1 = px + (px - sl)
        tp2 = px + last_leg
    else:
        sl = sl_struct if (sl_struct and sl_struct > px) else px + atr_val * 0.6
        sl = max(sl, px + min_risk)
        tp1 = px - (sl - px)
        tp2 = px - last_leg
    risk = abs(px - sl)
    rr = abs(tp2 - px) / max(risk, 1e-9)
    return {"entry": px, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr,
            "sl_label": "结构止损(形态极值外侧)" if sl_struct else "ATR止损(按近期波动幅度)"}


def format_caption(result, plan):
    """图片信号卡标题 — 极简, 详情在文字消息"""
    sig = result["signal"]; sym = result["symbol"]
    ba = result.get("brooks") or {}
    emoji = "🟢" if sig == "LONG" else "🔴"
    setups = f" | {'; '.join(ba['setups'])}" if ba.get("setups") else ""
    return f"{emoji} {sym} {'做多' if sig == 'LONG' else '做空'} {result['votes']}票{setups} | 入场 ${plan['entry']:.2f}"


def make_chart(result, plan):
    """15m K线图: EMA20 + 摆动点 + 入场/止损/止盈线, 全中文标注"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # macOS 系统中文字体, 按优先级选第一个可用的
        available = {f.name for f in font_manager.fontManager.ttflist}
        for fname in ["PingFang SC", "Hiragino Sans GB", "WenQuanYi Zen Hei",
                       "Noto Sans CJK SC", "Arial Unicode MS", "Heiti SC", "STHeiti", "DejaVu Sans"]:
            if fname in available:
                plt.rcParams["font.sans-serif"] = [fname]
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        return None
    path = os.path.join(tempfile.gettempdir(), f"vt_{result['symbol']}.png")
    try:
        sym = result["symbol"]
        df = fetch_klines(sym, "15m", 60)
        o = df["open"].values; h = df["high"].values
        l = df["low"].values; c = df["close"].values
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
        for i in x:
            col = "#26a69a" if c[i] >= o[i] else "#ef5350"
            ax.vlines(i, l[i], h[i], color=col, lw=0.7)
            ax.vlines(i, min(o[i], c[i]), max(o[i], c[i]), color=col, lw=3.2)
        ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
        ax.plot(x, ema20, color="#ffa726", lw=1.3, label="EMA20均线")
        sw_hi = _swing_points(h)[0]; sw_lo = _swing_points(l)[1]
        ax.plot(x[sw_hi], h[sw_hi], "v", color="#ab47bc", ms=5, label="摆动高/低点")
        ax.plot(x[sw_lo], l[sw_lo], "^", color="#ab47bc", ms=5)
        for price, label, col in [(plan["entry"], "入场", "#1e88e5"),
                                  (plan["sl"], "止损", "#e53935"),
                                  (plan["tp1"], "止盈1", "#8e24aa"),
                                  (plan["tp2"], "止盈2", "#43a047")]:
            ax.axhline(price, color=col, lw=1, ls="--", alpha=0.8)
            ax.text(len(df) - 0.5, price, f" {label} {price:.1f}",
                    color=col, fontsize=9, va="bottom", ha="right")
        dir_txt = "做多" if result["signal"] == "LONG" else "做空"
        ax.set_title(f"{sym} 15分钟  {dir_txt} {result['votes']}票  |  {pd.Timestamp.now():%m-%d %H:%M}")
        tick_idx = x[::10]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([df.index[i].strftime("%H:%M") for i in tick_idx], fontsize=7)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


def format_message(result, plan, is_emergency=False, sig_num=0, reverse_from="", judge=None, ai_decision=False):
    if result["signal"] == "NEUTRAL":
        return None
    sym = result["symbol"]; sig = result["signal"]
    px = result["price"]
    votes_n, votes_total = map(int, result["votes"].split("/"))
    pct = int(votes_n / votes_total * 100)
    bar = "▓" * (pct // 10) + "░" * (10 - pct // 10)
    emoji = "🟢" if sig == "LONG" else "🔴"
    dir_cn = "做多" if sig == "LONG" else "做空"
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    spike_cn = {1: "(刚出现强势向上突破)", -1: "(刚出现强势向下跌破)"}.get(ba.get("spike", 0), "")
    ai_cn = {1: "多", -1: "空"}.get(ba.get("always_in", 0), "-")

    drivers = [d for d in result["details"] if (d["direction"] == "🟢") == (sig == "LONG")]
    oppose = [d for d in result["details"] if (d["direction"] == "🟢") != (sig == "LONG")]

    L = []
    if is_emergency:
        L.append(f"🔄 <b>紧急翻转 #{sig_num}</b> — 从 #{sig_num-1} 反转")
        L.append("")
    reverse_tag = f" (反转#{sig_num-1}的{reverse_from})" if reverse_from else ""
    L.append(f"{emoji} <b>{sym} 建议{dir_cn} #{sig_num}</b> | {'⚡强信号' if votes_n >= STRONG else '📡信号'} {votes_n}/{votes_total}票{reverse_tag}")
    if ai_decision:
        L.append("🤖 AI主判方向, 票数仅作参考")
    L.append(f"强度 {pct}% {bar}")
    # Clean market context
    market_line = f"市场: {state_cn}"
    if spike_cn:
        market_line += spike_cn
    market_line += f" | Always In: {ai_cn}"
    if ba.get("setups"):
        market_line += f" | 形态: {'; '.join(ba['setups'])}"
    L.append(market_line)
    L.append("")

    L.append("投票分布")
    groups = {"VT因子": [], "NOFX": [], "Brooks": []}
    for d in result["details"]:
        name = d["name"]
        if name.startswith("NOFX_"):
            groups["NOFX"].append(d)
        elif name.startswith("BROOKS_"):
            groups["Brooks"].append(d)
        else:
            groups["VT因子"].append(d)
    gtotal = {"VT因子": 8, "NOFX": 4, "Brooks": 6}
    gkeys = ["VT因子", "NOFX", "Brooks"]
    for gi, key in enumerate(gkeys):
        ds = groups[key]
        agree = sum(1 for d in ds if (d["direction"] == "🟢") == (sig == "LONG"))
        total = gtotal[key]
        prefix = "└" if gi == len(gkeys) - 1 else "├"
        # 🟢=看涨 🔴=看跌 ⭕=未触发
        em = "".join(d["direction"] for d in ds)
        em += "⭕" * (total - len(ds))
        L.append(f"{prefix} {key} {agree}/{total}票  {em}")
    L.append("")

    L.append("交易计划")
    L.append(f"├ 入场: ${plan['entry']:.2f}")
    L.append(f"├ 止损: ${plan['sl']:.2f} ({plan['sl_label']})")
    L.append(f"├ 止盈1: ${plan['tp1']:.2f} (赚1倍风险先落袋一半)")
    L.append(f"└ 止盈2: ${plan['tp2']:.2f} (测量目标，盈亏比1:{plan['rr']:.1f})")

    # 50x 杠杆短线建议 — 分段止盈
    risk_pct = 0.005  # 0.5% 单次风险
    sl_50x = px - risk_pct * px if sig == "LONG" else px + risk_pct * px
    tp1_50x = px + risk_pct * px if sig == "LONG" else px - risk_pct * px      # 1:1 先保本
    tp2_50x = px + risk_pct * 2 * px if sig == "LONG" else px - risk_pct * 2 * px  # 1:2
    tp3_50x = px + risk_pct * 3 * px if sig == "LONG" else px - risk_pct * 3 * px  # 1:3
    L.append("")
    L.append(f"50x分段止盈 (风险自控, 每段{risk_pct*100:.1f}%)")
    L.append(f"├ 不追现价: 等5分钟回踩再入场")
    L.append(f"├ 止损: ${sl_50x:.2f} ({'下方' if sig=='LONG' else '上方'}{risk_pct*100:.1f}%)")
    L.append(f"├ TP1: ${tp1_50x:.2f} (1:1先保本, 到这个价位减半仓)")
    L.append(f"├ TP2: ${tp2_50x:.2f} (1:2, 剩半仓推止损到成本)")
    L.append(f"└ TP3: ${tp3_50x:.2f} (1:3, 最后仓位博超额)")

    stats = get_win_rate_stats()
    if stats["overall"] != "0/0":
        L.append("")
        L.append(f"📊 历史胜率 总{stats['overall']}({stats['overall_rate']}%) 近10次:{stats['recent10_rate']}% 近20次:{stats['recent20_rate']}%")
    L.append(f"🕐 {pd.Timestamp.now():%m/%d %H:%M}")
    L.append("")
    L.append("三周期方向预测")
    L.extend(compute_direction_probs(sym))
    L.append("")
    L.append("AI裁判")
    judge = judge or {}
    reasons = judge.get("reasons") or ["-"]
    if judge.get("confidence", -1) >= 0:
        L.append(f"├ 结论: {judge['verdict']} (置信度 {judge['confidence']})")
    else:
        L.append("├ 结论: 裁判不可用")
    st = judge.get("stats") or {}
    if st.get("total", 0) >= 5:
        L.append(f"├ 近期判对率: {round(st['wins'] / st['total'] * 100)}% ({st['wins']}/{st['total']})")
    for i, rsn in enumerate(reasons):
        prefix = "└" if i == len(reasons) - 1 else "├"
        L.append(f"{prefix} 理由{i+1}: {rsn}")
    return "\n".join(L)


def format_update(result, judge, plan=None):
    """15分钟快报: AI结论 → 双周期 → 关键位 → 数据, 分段留白"""
    sym = result["symbol"]
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    sig = result["signal"]
    vn = int(result["votes"].split("/")[0])
    reached = sig != "NEUTRAL" and vn >= MIN_VOTES

    L = []
    L.append("=" * 36)
    L.append(f"📡 {sym} 快报 | {pd.Timestamp.now():%m-%d %H:%M} | 💰现价 ${result['price']:.2f}")
    L.append("")

    judge = judge or {}
    if judge.get("confidence", -1) >= 0:
        j_emoji = "✅" if judge["verdict"] == "执行" else "🛑"
        tier = judge.get("mag_tier")
        tier_label = ["⚪极小幅(<1%)", "🔵轻仓档(1-2%)", "🟣标准档(2-3%)", "🟠主攻档(3%+)"][tier] if tier is not None else None
        L.append(f"🤖 AI: {j_emoji}{judge['verdict']} (置信{judge['confidence']})" + (f" | {tier_label}" if tier_label else ""))
        for i, rsn in enumerate((judge.get("reasons") or [])[:2]):
            L.append(f"📝 理由{i+1}: {rsn}")
    else:
        L.append("🤖 AI: 裁判不可用")
    L.append("")

    ctx4 = compute_4h_context(sym)
    if ctx4:
        b4s = {"trend_up": "上升", "trend_down": "下降", "range": "震荡"}.get(ctx4["brooks"].get("state"), "-")
        L.append(f"🌏 4h: {trend4_label(ctx4)} | Brooks4h {b4s} | 挤压{ctx4['squeeze_pct']:.0f}%分位 | RSI{rsi_tag(ctx4['rsi4h'])}")
    L.append(trigger_line(result))
    L.append("")

    if ctx4 and ctx4.get("wyckoff"):
        wy = ctx4["wyckoff"]
        L.append(f"📦 威科夫TR: ${wy['support']} — ${wy['resistance']} (宽{wy['width_pct']}%)")
    lv = compute_levels(sym, sig if sig != "NEUTRAL" else ("LONG" if result["bullish"] >= result["bearish"] else "SHORT"))
    if lv:
        L.append(f"🗝️ {han_pad('支撑:', 8)} {lv['support']}")
        L.append(f"🗝️ {han_pad('压力:', 8)} {lv['resistance']}")
    L.append("")

    parts = [f"看涨{result['bullish']}/看跌{result['bearish']}", state_cn]
    if lv:
        vol_short = "放量" if lv["vol_ratio"] > 1.5 else "正常" if lv["vol_ratio"] > 0.8 else "缩量"
        parts.append(f"RSI{rsi_tag(lv['rsi14'])}")
        parts.append(f"{vol_short}{lv['vol_ratio']:.1f}倍")
    st = fetch_sentiment(sym) or {}
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        parts.append(f"费率{fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    if "taker_buy_sell_ratio" in st:
        parts.append(f"买卖比{st['taker_buy_sell_ratio']:.2f}")
    vw = compute_vwap(sym)
    if vw:
        parts.append(f"VWAP{'上' if vw[1] >= 0 else '下'}{vw[1]:+.1f}%")
    L.append("📊 " + " | ".join(parts))
    L.append(f"🚦 信号线: {'✅已达' + str(MIN_VOTES) + '票' if reached else '⏸未触发'}")
    return "\n".join(L)


def rsi_tag(r):
    """RSI 超买超卖状态灯: ≥70超买 / ≤30超卖"""
    if r >= 70:
        return f"{r:.0f}超买⚠️"
    if r <= 30:
        return f"{r:.0f}超卖⚠️"
    return f"{r:.0f}"


def trend4_label(ctx4):
    """4h 均线排列标签: 全顺多排/全逆空排/否则纠缠(口径 EMA7/25/99, 与币安盘面一致)"""
    if ctx4["trend_up"]:
        return "多排"
    if ctx4.get("trend_dn"):
        return "空排"
    return "纠缠"


NEWS_FEEDS = [
    ("Axios", "https://api.axios.com/feed/"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("GNews加密", "https://news.google.com/rss/search?q=bitcoin+OR+ethereum+OR+crypto&hl=zh-CN&gl=CN&ceid=CN:zh"),
    ("GNews加密EN", "https://news.google.com/rss/search?q=crypto+OR+bitcoin+OR+ethereum+OR+%22white+house%22&hl=en-US&gl=US&ceid=US:en"),  # 英文加密: 白宫会议这类英文源首发(2026-08-20漏报事故)
    ("GNews宏观", "https://news.google.com/rss/search?q=federal+reserve+OR+treasury+OR+FOMC&hl=en-US&gl=US&ceid=US:en"),
    ("GNews大宗", "https://news.google.com/rss/search?q=oil+OR+gold+OR+dollar+OR+treasury+yield&hl=en-US&gl=US&ceid=US:en"),
    ("GNews美股", "https://news.google.com/rss/search?q=stock+market+OR+nasdaq+OR+nvidia+OR+AI&hl=en-US&gl=US&ceid=US:en"),
    ("GNews监管", "https://news.google.com/rss/search?q=%E7%9B%91%E7%AE%A1+OR+%E6%B3%95%E6%A1%88+OR+%E6%94%BF%E7%AD%96+%E5%8A%A0%E5%AF%86&hl=zh-CN&gl=CN&ceid=CN:zh"),
]
# 新闻关键词白名单: 只推可能影响行情的地缘/宏观/币圈原生新闻
NEWS_KEYWORDS = ("iran", "israel", "fed ", "fomc", "rate cut", "rate hike", "cpi", "inflation",
                 "sec ", "etf", "bitcoin", "btc", "ethereum", "eth ", "crypto", "tariff",
                 "war", "missile", "strike", "attack", "sanction", "trump", "powell", "ceasefire",
                 "treasury", "buyback", "yield", "财政部", "联储", "降息", "加息", "通胀",
                 "比特币", "以太坊", "美联储", "纪要", "国债", "收益率", "回购")


def _norm_title(t):
    return re.sub(r"[^a-z0-9一-鿿]", "", t.lower())


def _title_sim(a, b):
    """标题相似度(3-gram Jaccard): 跨媒体同新闻/后续解读稿去重用"""
    if len(a) < 6 or len(b) < 6:
        return 0.0
    sa = {a[i:i + 3] for i in range(len(a) - 2)}
    sb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(sa & sb) / max(len(sa | sb), 1)


def fetch_news(seen):
    """RSS 拉新(不做关键词过滤, 2026-08-20起由DS编辑席判价值): link去重, 只收24h内发布(防旧闻污染综述), 每源最多8条, 共≤15条"""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    out = []
    now = time.time()
    for name, url in NEWS_FEEDS:
        per = 0
        try:
            r = _http.get(url, timeout=8)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or not link or link in seen:
                    continue
                pub = item.findtext("pubDate") or ""
                try:
                    if pub and now - parsedate_to_datetime(pub).timestamp() > 86400:
                        seen.add(link)
                        continue  # 超24h旧闻: 标记已读但不进系统(2026-08-20 CLARITY旧闻事故)
                except Exception:
                    pass
                seen.add(link)
                out.append((name, title))
                per += 1
                if per >= 8:
                    break
        except Exception:
            continue
    return out[:15]


def translate_title(title):
    """英文新闻标题→中文(DS快翻, 2026-08-10 用户要求推送中文显示); 失败回原文"""
    if not any(ord(c) < 0x2E80 for c in title):  # 已是中文不翻
        return title
    try:
        r = _http.post(DS_API_URL, json={"model": DS_MODEL, "messages": [
            {"role": "system", "content": "把英文新闻标题翻译成中文, 一句话不超过30字, 只输出译文, 不加引号"},
            {"role": "user", "content": title}], "max_tokens": 100, "temperature": 0.2},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=15)
        if r.status_code == 200:
            t = r.json()["choices"][0]["message"]["content"].strip()
            if t:
                return t
    except Exception:
        pass
    return title


def fetch_fast_price(sym):
    """实时报价三源取中位数, 防单源偏离(2026-08-09 用户反馈价格不准); 全失败返回 None
    USDC 对用 USDC 计价的所(OKX/Gate, Binance国际美区451), 不用 USDT 对冒充(用户反馈)"""
    coin = sym.replace("USDC", "").replace("USDT", "")
    if sym.endswith("USDC"):
        sources = [
            (f"https://www.okx.com/api/v5/market/ticker?instId={coin}-USDC",
             lambda d: d["data"][0].get("last") if d.get("data") else None),
            (f"https://api.gateio.ws/api/v4/spot/tickers?currency_pair={coin}_USDC",
             lambda d: d[0].get("last") if isinstance(d, list) and d else None),
            (f"https://api.exchange.coinbase.com/products/{coin}-USD/ticker", lambda d: d.get("price")),
        ]
    else:
        sources = [
            (f"https://api.exchange.coinbase.com/products/{coin}-USD/ticker", lambda d: d.get("price")),
            (f"https://api.binance.us/api/v3/ticker/price?symbol={coin}USDT", lambda d: d.get("price")),
            (f"https://www.okx.com/api/v5/market/ticker?instId={coin}-USDT",
             lambda d: d["data"][0].get("last") if d.get("data") else None),
        ]
    vals = []
    for url, pick in sources:
        try:
            d = http_get_json(url, timeout=5)
            v = pick(d) if d else None
            if v:
                vals.append(float(v))
        except Exception:
            continue
    vals.sort()
    return vals[len(vals) // 2] if vals else None


def detect_events(sym, result, prev):
    """3分钟扫描异动检测(边沿触发, 不带统计): RSI(15m/4h)穿越70/30, 量比突破2, 逼近压力/支撑(<0.3%),
    挤压进出<20%区, 3分钟急涨急跌(阈值随15m ATR自适应, 夹0.2-0.6%), VWAP穿越, 摆动极值突破, 波动骤增(30m振幅≥2.5×ATR)。
    prev 为上次指标快照; 返回 (事件列表, 当前快照)"""
    lv = compute_levels(sym, result["signal"])
    ctx4 = compute_4h_context(sym)
    px = result["price"]
    vw = compute_vwap(sym)
    cur = {
        "px": px,
        "rsi": lv["rsi14"] if lv else None,
        "rsi4h": ctx4["rsi4h"] if ctx4 else None,
        "vol": lv["vol_ratio"] if lv else None,
        "sq": ctx4["squeeze_pct"] if ctx4 else None,
        "near_hi": bool(lv and 0 < (lv["swing_high"] - px) / px < 0.003),
        "near_lo": bool(lv and 0 < (px - lv["swing_low"]) / px < 0.003),
        "vwap_up": bool(vw and vw[1] >= 0),
        "broke_hi": bool(lv and px > lv["swing_high"]),
        "broke_lo": bool(lv and px < lv["swing_low"]),
        # 波动骤增: 近30分钟振幅 ≥2.5倍ATR 且 ≥0.3%(2026-08-09: 横盘突然放大, AI必须开口)
        "vol_burst": bool(lv and lv["rng_30m"] >= max(2.5 * lv["atr14"], 0.003 * px)),
        "wy_ev": ((ctx4.get("wyckoff") or {}).get("event") if ctx4 else None),
        "ema50_up": bool(lv and px >= lv["ema50"]),
    }
    # 布林带(15m, 20/2): 触轨检测用, 与 compute_levels 共享K线缓存
    boll_up = boll_lo = None
    try:
        cb = fetch_klines(sym, "15m", 60)["close"]
        mid = cb.rolling(20).mean().iloc[-1]
        sd = cb.rolling(20).std().iloc[-1]
        boll_up, boll_lo = float(mid + 2 * sd), float(mid - 2 * sd)
    except Exception:
        pass
    cur["boll_hi"] = bool(boll_up and px >= boll_up)
    cur["boll_lo"] = bool(boll_lo and px <= boll_lo)
    # 费率极端: |8h费率|>0.05% = 一方过热
    fr_now = None
    try:
        fr_now = (fetch_sentiment(sym) or {}).get("funding_rate")
    except Exception:
        pass
    cur["fr_val"] = fr_now
    cur["fr_ext"] = bool(fr_now is not None and abs(fr_now) * 100 > 0.05)
    # 统计形态触发(2026-08-10 五年训练结论): 高费率+趋势位置组合
    ret7d = None
    try:
        c4 = fetch_klines(sym, "4h", 50)["close"]
        if len(c4) >= 43:
            ret7d = float((c4.iloc[-1] / c4.iloc[-43] - 1) * 100)
    except Exception:
        pass
    fr8pct = fr_now * 100 if fr_now is not None else None
    cur["ret7d"] = ret7d
    cur["sig_relief"] = bool(fr8pct is not None and fr8pct >= 0.03 and ret7d is not None and ret7d < -5)
    cur["sig_top"] = bool(fr8pct is not None and fr8pct >= 0.05 and ret7d is not None and ret7d > 5)
    cur["sig_sqtrend"] = bool(cur.get("sq") is not None and cur["sq"] < 20 and ret7d is not None and ret7d > 5)
    ev = []
    if prev:
        p0 = prev.get("px")
        if p0 and lv:
            # 阈值 = 0.5×15m ATR(换算成%), 夹 0.2-0.6%: 横盘收紧, 趋势放宽(2026-08-09)
            atr_pct = lv["atr14"] / px * 100
            thr = min(max(0.5 * atr_pct, 0.2), 0.6)
            chg = (px - p0) / p0 * 100
            if abs(chg) >= thr:
                ev.append(f"3分钟{'急涨' if chg > 0 else '急跌'}{abs(chg):.1f}%(${p0:.0f}→${px:.0f}, 阈值{thr:.1f}%)")
        if prev.get("vwap_up") is not None and cur["vwap_up"] != prev["vwap_up"]:
            ev.append(f"价格{'上穿' if cur['vwap_up'] else '下穿'}VWAP(${vw[0]:.2f})")
        if lv:
            if not prev.get("broke_hi") and cur["broke_hi"]:
                ev.append(f"突破近20周期高点${lv['swing_high']:.2f}")
            if not prev.get("broke_lo") and cur["broke_lo"]:
                ev.append(f"跌破近20周期低点${lv['swing_low']:.2f}")
            if not prev.get("vol_burst") and cur["vol_burst"]:
                ev.append(f"波动骤增(近30分钟振幅{lv['rng_30m'] / px * 100:.2f}%, 是常态{lv['rng_30m'] / lv['atr14']:.1f}倍)")
        r0, r1 = prev.get("rsi"), cur["rsi"]
        if r0 is not None and r1 is not None:
            if r0 < 70 <= r1:
                ev.append(f"RSI进入超买({r1:.0f})")
            elif r0 > 70 >= r1:
                ev.append(f"RSI退出超买({r1:.0f})")
            if r0 > 30 >= r1:
                ev.append(f"RSI进入超卖({r1:.0f})")
            elif r0 < 30 <= r1:
                ev.append(f"RSI退出超卖({r1:.0f})")
        r0, r1 = prev.get("rsi4h"), cur["rsi4h"]
        if r0 is not None and r1 is not None:
            if r0 < 70 <= r1:
                ev.append(f"4hRSI进入超买({r1:.0f})")
            elif r0 > 70 >= r1:
                ev.append(f"4hRSI退出超买({r1:.0f})")
            if r0 > 30 >= r1:
                ev.append(f"4hRSI进入超卖({r1:.0f})")
            elif r0 < 30 <= r1:
                ev.append(f"4hRSI退出超卖({r1:.0f})")
        v0, v1 = prev.get("vol"), cur["vol"]
        if v0 is not None and v1 is not None and v0 <= 2 < v1:
            ev.append(f"突然放量({v1:.1f}倍均量)")
        # 威科夫新事件: 假突破/假跌破/真突破出现即推(2026-08-10 用户要求)
        if cur.get("wy_ev") and cur["wy_ev"] != prev.get("wy_ev"):
            names = {"spring": "假跌破收回(Spring)", "upthrust": "假突破回落(Upthrust)",
                     "joc_up": "真突破TR上沿(JOC)", "joc_down": "真跌破TR下沿(JOC)"}
            ev.append(f"威科夫新事件: {names.get(cur['wy_ev'], cur['wy_ev'])}")
        # 价格穿越 EMA50(15m多空分水岭)
        if lv and prev.get("ema50_up") is not None and cur["ema50_up"] != prev["ema50_up"]:
            ev.append(f"价格{'上穿' if cur['ema50_up'] else '下穿'}EMA50(${lv['ema50']:.2f})")
        # 布林触轨(15m)
        if boll_up and not prev.get("boll_hi") and cur["boll_hi"]:
            ev.append(f"触及布林上轨(${boll_up:.1f})")
        if boll_lo and not prev.get("boll_lo") and cur["boll_lo"]:
            ev.append(f"触及布林下轨(${boll_lo:.1f})")
        # 费率进入极端区(只报进入, 退出不报)
        if cur["fr_ext"] and not prev.get("fr_ext"):
            side = "多付空, 多头过热" if cur["fr_val"] > 0 else "空付多, 空头过热"
            ev.append(f"费率进入极端区({cur['fr_val'] * 100:.3f}%, {side})")
        # 统计形态(五年训练 2026-08-10): 边沿触发
        if cur.get("sig_relief") and not prev.get("sig_relief"):
            ev.append(f"统计形态: 费率{cur['fr_val'] * 100:.3f}%+7日跌{abs(cur['ret7d']):.0f}% → 后24h跌≥0.5%历史占90%(下跌中继, 偏空)")
        if cur.get("sig_top") and not prev.get("sig_top"):
            ev.append(f"统计形态: 费率{cur['fr_val'] * 100:.3f}%+7日涨{cur['ret7d']:.0f}% → 后24h回落≥0.5%历史占87%(亢奋见顶, 偏空)")
        if cur.get("sig_sqtrend") and not prev.get("sig_sqtrend"):
            ev.append(f"统计形态: 7日涨{cur['ret7d']:.0f}%+挤压{cur['sq']:.0f}%分位 → 后24h涨≥0.5%历史占87%(趋势压缩突破, 偏多)")
        for key, label in (("near_hi", f"逼近压力位${lv['swing_high']:.2f}"), ("near_lo", f"逼近支撑位${lv['swing_low']:.2f}")):
            if not prev.get(key) and cur[key]:
                ev.append(label)
        s0, s1 = prev.get("sq"), cur["sq"]
        if s0 is not None and s1 is not None:
            if s0 >= 20 > s1:
                # 规则解读直接贴事件里(2026-08-10 用户要求): 不等AI, 挤压含义确定
                ev.append(f"挤压进入极值区({s1:.0f}%分位): 波动率异常安静, 回测显示未来12h振幅反而偏低(-15%), 别指望立即爆发, 方向看趋势/结构维度")
            elif s0 < 20 <= s1:
                ev.append(f"挤压离开极值区({s1:.0f}%分位): 波动开始释放, 行情已在路上")
    return ev, cur


def plain_signals(result, lv, ctx4):
    """白话信号行: 阈值触发+引用多年统计库; 概率只引统计数字, 不编。
    统计口径都是4h, 因此引统计的句子也用4h指标(RSI4h/量比4h/挤压)"""
    px = result["price"]
    mst = market_stats(result["symbol"]) or {}
    out = []
    if ctx4:
        vr4 = ctx4.get("vol_ratio4h", 0)
        if vr4 >= 1.5:
            vb = "放量上涨" if ctx4.get("up4") else "放量下跌"
            s = f"量能放大到{vr4:.1f}倍(4h口径), 有资金进场"
            t = mst.get("volume", {}).get(vb)
            if t:
                s += f"; 近3年同类{vb}后12h延续占{t['cont']}%"
            out.append(s)
        r4 = ctx4["rsi4h"]
        if r4 >= 70:
            t = mst.get("rsi", {}).get(">=70")
            s = f"4hRSI {r4:.0f}已超买"
            if t:
                s += f"(历史同类后12h回落≥1%占{t['drop']}%)"
            out.append(s + ", 追多需防回抽")
        elif r4 <= 30:
            t = mst.get("rsi", {}).get("<=30")
            s = f"4hRSI {r4:.0f}已超卖"
            if t:
                s += f"(历史同类后12h反弹≥1%占{t['bounce']}%)"
            out.append(s + ", 追空需防反弹")
        sq = ctx4["squeeze_pct"]
        if sq < 40:
            b = "0-20" if sq < 20 else "20-40"
            s = f"波动率压到近200根4h的{sq:.0f}%位置, 属低位压缩"
            t = mst.get("squeeze", {}).get(b)
            if t:
                s += f"; 历史此档后12h振幅≥2%占{t['p2']}%"
            out.append(s)
    if lv:
        dh = lv["swing_high"] - px
        dl = px - lv["swing_low"]
        if 0 < dh / px < 0.01:
            out.append(f"现价距压力位${lv['swing_high']:.2f}仅${dh:.1f}, 临界区: 突破打开上方空间, 受阻则回落")
        elif 0 < dl / px < 0.01:
            out.append(f"现价距支撑位${lv['swing_low']:.2f}仅${dl:.1f}, 临界区: 跌破打开下方空间, 守住则反弹")
    return out[:4]


def han_pad(s, width):
    """补齐到指定视觉宽度(汉字/全角=2, ASCII=1): 优先全角空格, 余1用ASCII空格微调, 保证任意宽度可达"""
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    gap = max(width - w, 0)
    return s + "　" * (gap // 2) + (" " if gap % 2 else "")


def entry_window(result, judge4, ctx4, lv, vwap):
    """入场窗口(纯规则, 不经AI): 4h层有方向为前提。
    A回调=价格贴近结构位(≤0.5×ATR15m)+RSI重置/形态扳机, RR最优; B突破=收破摆动极值+放量(>1.5倍), 单边市用。
    入场好坏的本质是失效位距离, 因此输出失效位和距离%"""
    d4 = judge4.get("direction")
    if not d4 or not ctx4 or not lv:
        return None
    px = result["price"]
    atr = lv["atr14"]
    if d4 == "LONG":
        structures = [("EMA20", lv["ema20"]), ("摆动低点", lv["swing_low"])]
    else:
        structures = [("EMA20", lv["ema20"]), ("摆动高点", lv["swing_high"])]
    if vwap:
        structures.append(("VWAP", vwap[0]))

    # A 回调: 贴近结构位 + RSI重置区(40-62) 或 Brooks 形态扳机
    name, lvl = min(structures, key=lambda s: abs(px - s[1]))
    if abs(px - lvl) <= 0.5 * atr:
        rsi = lv["rsi14"]
        has_setup = bool((result.get("brooks") or {}).get("setups"))
        if 40 <= rsi <= 62 or has_setup:
            inv = lvl - 0.8 * atr if d4 == "LONG" else lvl + 0.8 * atr
            return {"type": "A回调", "dir": d4,
                    "zone": f"{name} ${lvl:.2f} 附近", "invalid": inv,
                    "dist": abs(px - inv) / px * 100,
                    "basis": f"贴近{name} + RSI{rsi:.0f}" + (" + 形态扳机" if has_setup else "")}

    # B 突破: 收破摆动极值 + 放量
    if d4 == "LONG" and px >= lv["swing_high"] * 0.9995 and lv["vol_ratio"] > 1.5:
        inv = lv["swing_high"] - 1.2 * atr
        return {"type": "B突破", "dir": d4,
                "zone": f"${lv['swing_high']:.2f} 上方", "invalid": inv,
                "dist": (px - inv) / px * 100,
                "basis": f"收破摆动高点${lv['swing_high']:.2f} + 量比{lv['vol_ratio']:.1f}"}
    if d4 == "SHORT" and px <= lv["swing_low"] * 1.0005 and lv["vol_ratio"] > 1.5:
        inv = lv["swing_low"] + 1.2 * atr
        return {"type": "B突破", "dir": d4,
                "zone": f"${lv['swing_low']:.2f} 下方", "invalid": inv,
                "dist": (inv - px) / px * 100,
                "basis": f"收破摆动低点${lv['swing_low']:.2f} + 量比{lv['vol_ratio']:.1f}"}
    return None


def _dir_emoji(d, conf):
    """方向醒目标注: 🟢多/🔴空/⚪观望/❔不可用"""
    if conf == -1:
        return "❔不可用"
    return {"LONG": "🟢多", "SHORT": "🔴空"}.get(d, "⚪观望")


# Polymarket 热点观察: (主题, 搜索词); 预测市场赔率=市场预期的后续判断(2026-08-20 用户要求)
PM_WATCH = [
    ("联储降息", "Fed rate cut September 2026", ("no fed", "no cut", "hike", "unchanged")),
    ("伊朗停火", "Iran Israel ceasefire", ()),
    ("美国衰退", "US recession", ()),
    ("BTC新高", "Bitcoin all time high", ()),
]
_pm_cache = {"ts": 0, "data": None}


def polymarket_odds():
    """Polymarket 热点赔率: [{topic, prob, q}], 只取未结束且概率在1-99之间的活跃市场; 15分钟缓存
    排除反向表述市场(no cut/hike等), 防止赔率方向显示反(2026-08-20)"""
    if time.time() - _pm_cache["ts"] < 900 and _pm_cache["data"] is not None:
        return _pm_cache["data"]
    out = []
    now_ms = time.time() * 1000
    for topic, q, excl in PM_WATCH:
        try:
            d = http_get_json("https://gamma-api.polymarket.com/public-search?q="
                              + requests.utils.quote(q) + "&limit_per_type=2", timeout=8)
            best = None
            for ev in (d or {}).get("events") or []:
                for m in ev.get("markets") or []:
                    if m.get("closed"):
                        continue
                    ques = (m.get("question") or "").lower()
                    if any(bad in ques for bad in excl):
                        continue
                    end = m.get("endDate") or m.get("end_date_iso")
                    if end and pd.Timestamp(end).timestamp() * 1000 < now_ms:
                        continue
                    p = m.get("outcomePrices")
                    if not p:
                        continue
                    yes = float(json.loads(p)[0])
                    if not (0.01 < yes < 0.99):
                        continue
                    vol = float(m.get("volumeNum") or 0)
                    if best is None or vol > best[1]:
                        best = (yes, vol, (m.get("question") or "")[:40])
            if best:
                out.append({"topic": topic, "prob": round(best[0] * 100), "q": best[2]})
        except Exception:
            continue
    _pm_cache["ts"] = time.time()
    _pm_cache["data"] = out
    return out


_liq_cache = {"ts": 0, "data": None}


def fetch_liquidations(coin="ETH"):
    """OKX 爆仓统计(2026-08-20 交易员桌面补全): 近1h/4h 多空双边爆仓USD; 5分钟缓存
    long=多头被爆(做空者视角的空头燃料耗尽), short=空头被爆(轧空燃料)"""
    if time.time() - _liq_cache["ts"] < 300 and _liq_cache["data"] is not None:
        return _liq_cache["data"]
    try:
        d = http_get_json("https://www.okx.com/api/v5/public/liquidation-orders"
                          f"?instType=SWAP&uly={coin}-USDT&state=filled", timeout=8)
        now = time.time() * 1000
        out = {"1h": {"long": 0.0, "short": 0.0}, "4h": {"long": 0.0, "short": 0.0}}
        for blk in (d or {}).get("data") or []:
            for x in blk.get("details", []):
                usd = float(x["sz"]) * float(x["bkPx"])
                age = now - int(x["ts"])
                side = "long" if x["posSide"] == "long" else "short"
                if age <= 3600e3:
                    out["1h"][side] += usd
                if age <= 4 * 3600e3:
                    out["4h"][side] += usd
        _liq_cache.update(ts=time.time(), data=out)
        return out
    except Exception:
        return None


def _m_usd(v):
    return f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}K"


def btc_link_line(sym):
    """BTC联动行: BTC 24h涨跌 + ETH相对强弱(2026-08-20: BTC是ETH的beta锚, 用户要求关注)"""
    try:
        b = fetch_klines("BTCUSDT", "4h", 10)["close"]
        e = fetch_klines(sym, "4h", 10)["close"]
        if len(b) < 7 or len(e) < 7:
            return None
        b24 = float((b.iloc[-1] / b.iloc[-7] - 1) * 100)
        e24 = float((e.iloc[-1] / e.iloc[-7] - 1) * 100)
        diff = e24 - b24
        return f"联动: BTC 24h {b24:+.1f}% | ETH{'强' if diff >= 0 else '弱'}于BTC {abs(diff):.1f}点"
    except Exception:
        return None


def market_mood(sym):
    """整体市场情绪(2026-08-20 用户要求): 恐贪指数+24h涨跌+费率+多空比 → 一行综合判断。
    返回 (卡片行, 简报行) 或 None"""
    parts = []
    fng = None
    try:
        d = http_get_json("https://api.alternative.me/fng/?limit=1", timeout=8)
        fng = int(d["data"][0]["value"])
        cn = {"Extreme Fear": "极度恐慌", "Fear": "恐慌", "Neutral": "中性",
              "Greed": "贪婪", "Extreme Greed": "极度贪婪"}.get(d["data"][0]["value_classification"], "")
        parts.append(f"恐贪{fng}({cn})")
    except Exception:
        pass
    chg24 = None
    try:
        c4 = fetch_klines(sym, "4h", 10)["close"]
        if len(c4) >= 7:
            chg24 = float((c4.iloc[-1] / c4.iloc[-7] - 1) * 100)
            parts.append(f"24h{chg24:+.1f}%")
    except Exception:
        pass
    st = fetch_sentiment(sym) or {}
    fr = st.get("funding_rate")
    if fr is not None:
        parts.append(f"费率{fr * 100:.3f}%")
    lsr = st.get("long_short_ratio") or st.get("ls_ratio")
    if lsr:
        parts.append(f"多空比{lsr:.2f}")
    if not parts:
        return None
    # 综合判断: 恐贪极端+费率过热=危险; 恐慌+费率负=机会区
    if fng is not None and fng >= 75 and fr is not None and fr * 100 >= 0.03:
        verdict = "贪婪过热, 追多危险"
    elif fng is not None and fng >= 75:
        verdict = "贪婪但费率未过热, 趋势可能延续"
    elif fng is not None and fng <= 25:
        verdict = "恐慌区, 留意超卖反弹"
    elif chg24 is not None and chg24 >= 5:
        verdict = "强势上涨日, 顺势不逆势"
    elif chg24 is not None and chg24 <= -5:
        verdict = "弱势下跌日, 别接飞刀"
    else:
        verdict = "情绪平稳"
    line = "情绪: " + " | ".join(parts) + f" → {verdict}"
    return line, line


def confluence(result, d4, lv, ctx4, board_rows):
    """共振计分(0-6, 规则计算): 高确信信号=多条件同向的客观度量(2026-08-20 用户要求: 大把握要主动喊)
    返回 (分数, [未满足项])"""
    if not d4 or not lv:
        return 0, []
    px = result["price"]
    long = d4 == "LONG"
    sym = result["symbol"]
    checks = []
    checks.append(("价在EMA20" + ("上方" if long else "下方"), (px >= lv["ema20"]) == long))
    checks.append(("放量(量比>1.2)", lv["vol_ratio"] > 1.2))
    wy = (ctx4 or {}).get("wyckoff") if ctx4 else None
    wy_align = bool(wy and wy.get("event") and
                    ((long and wy["event"] in ("spring", "joc_up")) or
                     (not long and wy["event"] in ("upthrust", "joc_down"))))
    checks.append(("威科夫事件同向", wy_align))
    checks.append(("挤压低位(<40%分位)", bool(ctx4 and ctx4["squeeze_pct"] < 40)))
    checks.append(("统计底牌有优势", bool(board_rows)))
    try:
        fr = (fetch_sentiment(sym) or {}).get("funding_rate")
        adverse = fr is not None and ((long and fr * 100 > 0.05) or (not long and fr * 100 < -0.05))
        checks.append(("费率不逆风", not adverse))
    except Exception:
        checks.append(("费率不逆风", True))
    score = sum(1 for _, ok in checks if ok)
    return score, [name for name, ok in checks if not ok]


def self_review_block():
    """AI自我复盘材料(2026-08-23 取代机械打分, 用户决策): 近5条判决+4h后实际涨跌(原始%)+错题本,
    注入两层简报让裁判校准。走势对照只报事实不给分, 判读权在AI"""
    try:
        entries = [e for e in load_journal()["entries"] if e.get("direction")]
        if len(entries) < 10:
            return None
        c15 = fetch_klines("ETHUSDC", "15m", 96 * 3)["close"]
        lines = []
        for e in entries[-5:]:
            et = pd.Timestamp(e["time"])
            if et.tz:
                et = et.tz_localize(None)
            later = c15[c15.index >= et + pd.Timedelta(hours=4)]
            if not len(later):
                continue
            chg = (float(later.iloc[0]) - float(e["entry_px"])) / float(e["entry_px"]) * 100
            d = {"LONG": "多", "SHORT": "空"}.get(e["direction"], "观望")
            lines.append(f"{e['time'][5:16]}判{d}@${float(e['entry_px']):.0f}→4h后{chg:+.1f}%")
        try:
            lessons = load_lessons()
            if isinstance(lessons, dict):
                lessons = lessons.get("lessons") or []
            if isinstance(lessons, list) and lessons:
                lines.append("错题本: " + "；".join(str(x)[:60] for x in lessons[-3:]))
        except Exception:
            pass
        return "你的近期判决复盘: " + " ; ".join(lines) if lines else None
    except Exception:
        return None


def trader_handbook(sym):
    """训练手册(2026-08-23 用户要求: 各维度训练结果常驻简报): 统计库里有区分度(≥60%/≤40%)的边缘,
    压成一页纸给裁判当背景知识。纯规则, 数字只来自统计库"""
    mst = market_stats(sym) or {}
    rows = []
    for b, t in mst.get("squeeze", {}).items():
        if t["p2"] >= 60 or t["p2"] <= 40:
            rows.append(f"4h挤压{b}%分位→12h振幅≥2%占{t['p2']}%")
    for k, t in mst.get("rsi", {}).items():
        if t["bounce"] >= 60 or t["bounce"] <= 40:
            rows.append(f"4hRSI{k}→12h反弹{t['bounce']}%")
        if t["drop"] >= 60 or t["drop"] <= 40:
            rows.append(f"4hRSI{k}→12h回落{t['drop']}%")
    for k, t in mst.get("volume", {}).items():
        if t["cont"] >= 60 or t["cont"] <= 40:
            rows.append(f"4h{k}→12h同向延续{t['cont']}%")
    for k, t in mst.get("trend_align", {}).items():
        rows.append(f"4h{k}→12h涨{t['up']}%/跌{t['dn']}%")
    for k, t in (mst.get("m15") or {}).items():
        if k.startswith("rsi"):
            if t["bounce"] >= 60:
                rows.append(f"15m {k[3:]}→4h反弹{t['bounce']}%")
            if t["drop"] >= 60:
                rows.append(f"15m {k[3:]}→4h回落{t['drop']}%")
        elif k.startswith("量比") and (t["cont"] >= 60 or t["cont"] <= 40):
            rows.append(f"15m{k}→4h延续{t['cont']}%")
        elif k == "贴近20根高点":
            rows.append(f"15m贴20根高点→4h上破{t['brk']}%/回落{t['fail']}%")
    # 训练形态(五年): 固定三条, 独立于当前状态
    rows.append("形态: 费率≥0.03+7日跌>5%→24h跌90% | 费率≥0.05+7日涨>5%→24h回落87% | 7日涨>5%+挤压<20%→24h涨87%")
    return "训练手册(近5年回测): " + " | ".join(rows) if rows else None


def build_data_board(sym, result, lv, ctx4):
    """数据看板(2026-08-10 用户要求): 只列当前状态命中的、有统计优势(≥65%或≤35%)的历史概率;
    全是五五开就明说无优势。纯规则生成, 数字只来自统计库, 不经AI"""
    mst = market_stats(sym) or {}
    cand = []  # (label, suffix, p, n)

    def add(label, rec, key, suffix):
        if rec and key in rec and (rec[key] >= 65 or rec[key] <= 35):
            cand.append((label, suffix, rec[key], rec["n"]))

    if ctx4:
        sq = ctx4["squeeze_pct"]
        b = "0-20" if sq < 20 else "20-40" if sq < 40 else "40-70" if sq < 70 else "70-100"
        add(f"4h挤压{b}%分位", mst.get("squeeze", {}).get(b), "p2", "后12h振幅≥2%")
        cur_aln = "多头排列" if ctx4.get("trend_up") else "空头排列" if ctx4.get("trend_dn") else None
        t = mst.get("trend_align", {}).get(cur_aln) if cur_aln else None
        add(f"4h{cur_aln}", t, "up", "后12h涨≥1%")
        add(f"4h{cur_aln}", t, "dn", "后12h跌≥1%")
        r4 = ctx4["rsi4h"]
        rb = "<=30" if r4 <= 30 else ">=70" if r4 >= 70 else "30-45" if r4 <= 45 else "45-55" if r4 <= 55 else "55-70"
        t = mst.get("rsi", {}).get(rb)
        add(f"4hRSI{rb}", t, "bounce", "后12h反弹≥1%")
        add(f"4hRSI{rb}", t, "drop", "后12h回落≥1%")
    if lv:
        m15 = mst.get("m15", {})
        r15 = lv["rsi14"]
        rb = "<=30" if r15 <= 30 else ">=70" if r15 >= 70 else "30-45" if r15 <= 45 else "45-55" if r15 <= 55 else "55-70"
        t = m15.get(f"rsi{rb}")
        add(f"15m RSI{rb}", t, "bounce", "后4h反弹≥0.5%")
        add(f"15m RSI{rb}", t, "drop", "后4h回落≥0.5%")
        vr_ = lv["vol_ratio"]
        vb = "<0.6" if vr_ < 0.6 else "0.6-1.2" if vr_ < 1.2 else "1.2-2" if vr_ < 2 else ">2"
        add(f"15m量比{vb}{'阳' if lv.get('up15') else '阴'}", m15.get(f"量比{vb}{'阳' if lv.get('up15') else '阴'}"),
            "cont", "后4h同向延续≥0.5%")
    # 训练形态(五年数据): 当前激活才亮
    try:
        fr = (fetch_sentiment(sym) or {}).get("funding_rate")
        c4 = fetch_klines(sym, "4h", 50)["close"]
        ret7d = float((c4.iloc[-1] / c4.iloc[-43] - 1) * 100) if len(c4) >= 43 else None
        fr8 = fr * 100 if fr is not None else None
        if fr8 is not None and ret7d is not None:
            if fr8 >= 0.03 and ret7d < -5:
                cand.append((f"费率{fr8:.3f}%+7日跌{abs(ret7d):.0f}%", "后24h跌≥0.5%", 90, 40))
            if fr8 >= 0.05 and ret7d > 5:
                cand.append((f"费率{fr8:.3f}%+7日涨{ret7d:.0f}%", "后24h回落≥0.5%", 87, 79))
            if ret7d > 5 and ctx4 and ctx4["squeeze_pct"] < 20:
                cand.append((f"7日涨{ret7d:.0f}%+挤压{ctx4['squeeze_pct']:.0f}%分位", "后24h涨≥0.5%", 87, 46))
    except Exception:
        pass
    # 白话翻译表: 标签 → 人话(2026-08-10 用户要求)
    def _plain_label(label):
        s = (label.replace("<=30", "超卖").replace(">=70", "超买")
             .replace("30-45", "偏弱").replace("45-55", "中性").replace("55-70", "偏强")
             .replace("RSI", "RSI").replace("量比<0.6", "严重缩量").replace("量比0.6-1.2", "平量")
             .replace("量比1.2-2", "温和放量").replace("量比>2", "爆量"))
        return s

    def _plain_suffix(suffix):
        return (suffix.replace("后12h振幅≥2%", "之后12小时波动超2%")
                .replace("后12h涨≥1%", "之后12小时涨超1%").replace("后12h跌≥1%", "之后12小时跌超1%")
                .replace("后12h反弹≥1%", "之后12小时反弹超1%").replace("后12h回落≥1%", "之后12小时回落超1%")
                .replace("后4h反弹≥0.5%", "之后4小时反弹超0.5%").replace("后4h回落≥0.5%", "之后4小时回落超0.5%")
                .replace("后4h同向延续≥0.5%", "之后4小时同向延续超0.5%")
                .replace("后24h跌≥0.5%", "之后24小时跌超0.5%").replace("后24h回落≥0.5%", "之后24小时回落超0.5%")
                .replace("后24h涨≥0.5%", "之后24小时涨超0.5%"))

    # 同标签两个方向都≥65% = 波动放大而非方向优势, 合并成一行避免自相矛盾
    rows = []
    used = set()
    for i, (label, suffix, p, n) in enumerate(cand):
        if i in used:
            continue
        same = [j for j, (l2, _, p2, _) in enumerate(cand) if j > i and l2 == label and j not in used]
        if same and p >= 65 and all(cand[j][2] >= 65 for j in same):
            parts = "、".join([f"{_plain_suffix(suffix)}占{p}%"] + [f"{_plain_suffix(cand[j][1])}占{cand[j][2]}%" for j in same])
            rows.append(f"　{_plain_label(label)}: 近2年{n}次里, {parts}——大涨大跌都常见, 方向没准, 别追单")
            used.update(same)
        else:
            rows.append(f"　{_plain_label(label)}: 近2年{n}次里, {_plain_suffix(suffix)}占{p}%")
    if not rows:
        return []  # 无优势底牌时整块不显示(2026-08-12 用户决策: 没信息量的行不占版面)
    return rows


def strength_score(sym, df4=None):
    """方向强度分(2026-08-22 5年回测, 弱信号合成): 收盘位置/距20期极值/BTC联动/突破, 合成 -4..+4。
    回测: 强度分+2 → 未来4h涨54.7%, -2 → 41.2% (含成本EV仍负, 只作参考不作交易规则)"""
    if df4 is None:
        try:
            df4 = fetch_klines(sym, "4h", 60)
        except Exception:
            return None
    if df4 is None or df4.empty or len(df4) < 30:
        return None
    last = df4.iloc[-1]
    hi, lo = df4["high"], df4["low"]
    score = 0
    pos = (last["close"] - last["low"]) / max(last["high"] - last["low"], 1e-9)
    if pos > 0.6:
        score += 1
    elif pos < 0.4:
        score -= 1
    hi20 = hi.iloc[-21:-1].max()
    lo20 = lo.iloc[-21:-1].min()
    dist_hi = (hi20 / last["close"] - 1) * 100
    dist_lo = (last["close"] / lo20 - 1) * 100
    if dist_hi < 2:
        score += 1
    if dist_lo < 2:
        score -= 1
    if last["close"] > hi20:
        score += 1
    if last["close"] < lo20:
        score -= 1
    try:
        btc4 = fetch_klines("BTCUSDT", "4h", 3)
        if not btc4.empty and len(btc4) >= 2:
            bret = (btc4["close"].iloc[-1] / btc4["close"].iloc[-2] - 1) * 100
            if bret > 0.5:
                score += 1
            elif bret < -0.5:
                score -= 1
    except Exception:
        pass
    return score


def position_size(d4, d15, tier, conf, atr_pct=None, squeeze_pct=None, event_day=False):
    """仓位建议%: 幅度档基础 + 共振/置信修正, 再叠加波动率/事件风险修正, 封顶 50%。
    波动率修正(2026-08-22 回测): 仓位∝1/ATR(恒定风险); 挤压分位≥70(波动已释放)减仓、<20可加仓;
    事件日(CPI/FOMC)波动放大 → 七折"""
    if not (d4 or d15):
        return 0
    base = {0: 5, 1: 15, 2: 25, 3: 35}.get(tier, 10) if tier is not None else 10
    if d4 and d4 == d15:
        base += 5  # 双层共振加仓
    if conf >= 80:
        base += 5
    elif conf < 60:
        base -= 5
    if atr_pct and atr_pct > 0:
        base *= 3.0 / atr_pct  # 基准ATR 3%, 高波动减仓低波动加仓
    if squeeze_pct is not None:
        if squeeze_pct >= 70:
            base *= 0.7   # 波动已释放, 未来12h振幅大(回测+23%)
        elif squeeze_pct < 20:
            base *= 1.1   # 压缩中, 未来12h反而平静(回测-15%)
    if event_day:
        base *= 0.7  # 事件日波动放大(CPI日+24%), 降杠杆
    return max(5, min(50, round(base)))


def build_analyst_block(result, judge4, judge15, ew, lv, plan, prev4=None, squeeze_pct=None, event_day=False):
    """交易员观点段(2026-08-20): 观点→计划→仓位→失效→上次观点, 全部用现成素材, 不引入AI编造"""
    d4, d15 = judge4.get("direction"), judge15.get("direction")
    c4 = judge4.get("confidence", -1)
    d_adv = d4 if (d4 and d4 == d15) else (d15 or d4)
    diverged = bool(d4 and d15 and d4 != d15)  # 双层背离: 无主张, 不给计划
    tier = judge4.get("mag_tier")
    px = result["price"]
    L = []
    # 观点行: 方向+置信+幅度档, 观望/背离单独表达
    if diverged:
        L.append(f"🎯 观点: 观望——4h{'多' if d4 == 'LONG' else '空'}/15m{'多' if d15 == 'LONG' else '空'}背离, 不动手")
    elif not d_adv:
        L.append("🎯 观点: 观望——双层无方向, 没有值得下注的行情")
    else:
        cn = "做多" if d_adv == "LONG" else "做空"
        tier_txt = {0: "小幅<1%", 1: "轻仓1-2%", 2: "标准2-3%", 3: "主攻3%+"}.get(tier, "")
        src = "双层共振" if (d4 and d4 == d15) else ("仅15m" if d15 else "仅4h")
        L.append(f"🎯 观点: {cn}({c4 if c4 >= 0 else 60}%), {tier_txt}, {src}")
    # 计划行: 仓位+入场+止损+目标+RR+单笔风险(有主张且不背离才给)
    if d_adv and not diverged and plan:
        pos = position_size(d4, d15, tier, max(c4, judge15.get("confidence", -1)),
                            atr_pct=(lv.get("atr14") / px * 100) if lv and lv.get("atr14") else None,
                            squeeze_pct=squeeze_pct, event_day=event_day)
        if ew and ew.get("dir") == d_adv:
            entry_txt = f"入场{ew['zone']}"
        else:
            entry_txt = "入场等回调贴结构位或放量突破"
        sl_pct = abs(px - plan["sl"]) / px * 100
        risk_pct = pos * sl_pct / 100  # 单笔风险(1倍杠杆全仓口径), 用户按自己杠杆映射
        L.append(f"📌 计划: 仓位{pos}% · {entry_txt} · 止损${plan['sl']:.2f}(-{sl_pct:.1f}%) · 目标${plan['tp2']:.2f} · RR 1:{plan['rr']:.1f} · 风险{risk_pct:.1f}%")
        if event_day:
            L.append("⚠️ 今日宏观事件日, 波动放大已计入仓位, 别加杠杆")
    # 失效行: 方向侧结构位/入场窗口失效位
    if d_adv and not diverged:
        if ew and ew.get("dir") == d_adv and ew.get("invalid"):
            inv, dist = ew["invalid"], ew.get("dist")
            L.append(f"🚫 失效: 收破${inv:.2f}(距{dist:.2f}%)判断作废, 认错翻边")
        elif lv:
            key = "swing_low" if d_adv == "LONG" else "swing_high"
            if lv.get(key):
                L.append(f"🚫 失效: 4h收破${lv[key]:.2f}判断作废, 认错翻边")
    # 上次观点: 连续性(专业感来源), 翻转时写清新证据
    if prev4 and prev4.get("direction") and d_adv and not diverged:
        pd_cn = {"LONG": "多", "SHORT": "空"}.get(prev4["direction"], "观望")
        if prev4["direction"] == d_adv:
            L.append(f"🔁 上次: 维持{'' if d_adv == 'LONG' else ''}{pd_cn}头观点, 证据未变")
        else:
            ev = (judge4.get("reasons") or ["新证据"])[0][:24]
            L.append(f"🔁 上次判{pd_cn} → 翻{('多' if d_adv == 'LONG' else '空')}, 新证据: {ev}…")
    return L


def format_layers(result, judge4, judge15, is_reversal=False, events=None, ew=None, ptype="定时", plan=None, prev4=None):
    """双层信号卡: 4h层+15m层各自判决; events=异动列表, ew=入场窗口, ptype=推送类型(标题显眼位)"""
    sym = result["symbol"]
    d4, d15 = judge4.get("direction"), judge15.get("direction")
    c4, c15 = judge4.get("confidence", -1), judge15.get("confidence", -1)

    if d4 and d4 == d15:
        head = "🟢" if d4 == "LONG" else "🔴"
    elif d4 and d15:
        head = "⚠️"
    elif d4 or d15:
        head = "🟢" if (d4 or d15) == "LONG" else "🔴"
    else:
        head = "⚪"

    # 距4h收线倒计时: 4h K线按 UTC 0/4/8/12/16/20 点收线, 与本地时区无关
    now_utc = pd.Timestamp.now(tz="UTC")
    remain = now_utc.floor("4h") + pd.Timedelta("4h") - now_utc
    rm, rs = divmod(int(remain.total_seconds()), 60)
    rh, rm = divmod(rm, 60)
    cd = f"距4h收线{rh}h{rm:02d}m" if rh else f"距4h收线{rm}m{rs:02d}s"
    L = []
    now_cn = pd.Timestamp.now(tz="UTC").tz_convert("Asia/Shanghai")  # 卡片时间戳用北京时间(用户在国内), 倒计时仍按UTC收线算
    L.append(f"{head} 【{ptype}】{sym} | {now_cn:%m-%d %H:%M} | {cd}")
    L.append(f"💰 现价 ${result['price']:.2f}")
    ctx4 = compute_4h_context(sym)
    lv = compute_levels(sym, d4 or d15 or result["signal"])
    # 交易员观点段(2026-08-20): 结论优先放最顶部, 观点→计划→仓位→失效→上次观点
    try:
        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        ev_day = bool(MACRO_EVENTS.get(today_et))
    except Exception:
        ev_day = False
    ab = build_analyst_block(result, judge4, judge15, ew, lv, plan, prev4,
                             squeeze_pct=(ctx4 or {}).get("squeeze_pct"),
                             event_day=ev_day)
    if ab:
        L.extend(ab)
    # 波动预测行(2026-08-22 回测5年): 振幅自相关优先(前4h振幅→未来12h, +53%/-34%最强单特征),
    # 失败 fallback 挤压映射。LightGBM 组合仅再降 8% RMSE, 不值得引入模型依赖
    try:
        df4r = fetch_klines(sym, "4h", 1)
        last4 = df4r.iloc[-1]
        rng = (last4["high"] - last4["low"]) / last4["open"] * 100
        amp_est = 2.4 if rng < 1 else 3.5 if rng < 2 else 4.2 if rng < 3.5 else 5.6
        L.append(f"🌊 波动: 前4h振幅{rng:.1f}%, 未来12h预计振幅{amp_est:.1f}%")
    except Exception:
        sq = (ctx4 or {}).get("squeeze_pct")
        if sq is not None:
            amp_est = 3.1 if sq < 20 else 3.3 if sq < 40 else 3.5 if sq < 70 else 4.4
            L.append(f"🌊 波动: 挤压{sq:.0f}%分位, 未来12h预计振幅{amp_est:.1f}%")
    # 方向强度分(5年回测弱信号合成, 仅参考; 只在定时卡算, 省请求)
    if ptype == "定时":
        try:
            sc = strength_score(sym)
            if sc is not None:
                L.append(f"🧭 强度分: {sc:+d} ({'偏强' if sc > 0 else '偏弱' if sc < 0 else '中性'}, 回测胜率±5pp)")
        except Exception:
            pass
    L.append("")
    # 交易笔记卡片版(2026-08-20 用户要求: 细节系统知道就行, 卡片只留主线+盯点); 事件预判并入笔记块
    others = [e for e in (events or []) if not e.startswith("📰")]
    hc = next((e for e in others if e.startswith("🔥")), None)
    others = [e for e in others if not e.startswith("🔥")]
    try:
        wv_mem = load_news_memory()
        wvo = wv_mem.get("world_view") or {}
        wv_card = wvo.get("card") or wvo.get("text")
        if wv_card:
            L.append(f"📝 笔记: {wv_card}")
        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        ep = (wv_mem.get("event_preview") or {})
        if ep.get("text") and preview_visible(ep, today_et):
            L.append(f"🗓 {MACRO_EVENTS.get(today_et)}预判: {ep['text']}")
    except Exception:
        pass
    if hc:
        L.append(hc + " —— 系统主动喊你, 仔细看这单")
    L.append("")

    tier = judge4.get("mag_tier")
    tier_label = ["⚪极小幅(<1%)", "🔵轻仓档(1-2%)", "🟣标准档(2-3%)", "🟠主攻档(3%+)"][tier] if tier is not None else None
    L.append(f"🌏 4h层: {_dir_emoji(d4, c4)}" + (f" 置信{c4}" if c4 >= 0 else "") + (f" | {tier_label}" if tier_label else ""))
    if judge4.get("summary"):
        L.append(f"💡 {judge4['summary']}")
    NUM_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

    def _render_reasons(judge):
        """⚠️复核反对行不编号, 其余按1️⃣-4️⃣编号"""
        n = 0
        for rsn in (judge.get("reasons") or []):
            if rsn.startswith("⚠️"):
                L.append(rsn)
            elif n < 4:
                L.append(f"{NUM_EMOJI[n]} {rsn}")
                n += 1

    _render_reasons(judge4)
    wy = (ctx4 or {}).get("wyckoff")
    px = result["price"]
    L.append("")  # 双层之间空行分隔(用户要求 2026-08-10)
    bull_pct = round(result["bullish"] / max(result["bullish"] + result["bearish"], 1) * 100)
    # 因子占比跟方向对齐: 判空显示看跌, 判多显示看涨, 观望显示双侧
    if d15 == "SHORT":
        fac = f"因子看跌 {100 - bull_pct}%"
    elif d15 == "LONG":
        fac = f"因子看涨 {bull_pct}%"
    else:
        fac = f"因子看涨{bull_pct}%/看跌{100 - bull_pct}%"
    L.append(f"🌏 15m层: {_dir_emoji(d15, c15)}" + (f" 置信{c15}" if c15 >= 0 else "") + f" | {fac}")
    if judge15.get("summary"):
        L.append(f"💡 {judge15['summary']}")
    _render_reasons(judge15)
    # KK反对的落实: 给反弹/回落的一二位目标(规则计算), 并提示到位后可顺主裁方向(2026-08-10 用户要求)
    kk_line = next((r for r in (judge15.get("reasons") or []) if r.startswith("⚠️ KK反对")), None)
    if kk_line and lv:
        vw_now = compute_vwap(sym)
        cands = [("EMA20", lv["ema20"]), ("EMA50", lv["ema50"]),
                 ("摆动高", lv["swing_high"]), ("摆动低", lv["swing_low"])]
        if vw_now:
            cands.append(("VWAP", vw_now[0]))
        if "判多" in kk_line:
            tg = sorted([c for c in cands if c[1] > px], key=lambda x: x[1])[:2]
            if tg:
                line = f"　↳ 若反弹: 一位${tg[0][1]:.2f}({tg[0][0]})"
                if len(tg) > 1:
                    line += f" 二位${tg[1][1]:.2f}({tg[1][0]}), 到位滞涨可继续顺主裁方向"
                L.append(line)
        elif "判空" in kk_line:
            tg = sorted([c for c in cands if c[1] < px], key=lambda x: -x[1])[:2]
            if tg:
                line = f"　↳ 若回落: 一位${tg[0][1]:.2f}({tg[0][0]})"
                if len(tg) > 1:
                    line += f" 二位${tg[1][1]:.2f}({tg[1][0]}), 到位企稳可继续顺主裁方向"
                L.append(line)
    L.append("")
    if others:
        L.append("🌊 动量: " + "; ".join(others))
        L.append("")

    # 开单建议(2026-08-10 用户要求): 方向+入场/失效+50x双档止损+双档止盈+突破目标+依据, 合并原入场窗口
    L.append("🧾 开单建议")
    d_adv = d4 if (d4 and d4 == d15) else (d15 or d4)
    if d4 and d15 and d4 != d15:
        L.append(f"　　方向: 观望 (4h{'多' if d4 == 'LONG' else '空'}/15m{'多' if d15 == 'LONG' else '空'}背离, 不动手)")
    elif not d_adv:
        L.append("　　方向: 观望 (双层无方向)")
    else:
        src = "双层共振" if d4 == d15 else ("仅15m层, 轻仓" if d15 else "仅4h层, 轻仓")
        L.append(f"　　方向: {'做多' if d_adv == 'LONG' else '做空'} ({src})")
        if ew and ew["dir"] == d_adv:
            L.append(f"　　入场: {ew['zone']} ({ew['type']})")
            L.append(f"　　失效: ${ew['invalid']:.2f} (距现价{ew['dist']:.2f}%)")
            L.append(f"　　依据: {ew['basis']}")
        else:
            L.append("　　入场: 暂无窗口, 等回调贴近结构位或放量突破")
        if lv:
            hi, lo, atr = lv["swing_high"], lv["swing_low"], lv["atr14"]
            if d_adv == "SHORT":
                L.append(f"　　止损(50x): 一档${px + 0.5 * atr:.2f} 二档${hi:.2f}(摆动高)")
                tps = [f"一撑${lo:.2f}(摆动低)"]
                if wy and wy["support"] < px:  # 已过位的目标不显示(2026-08-20 旧TR顶出现在现价下方事故)
                    tps.append(f"二撑${wy['support']}(TR底); 跌破看${2 * wy['support'] - wy['resistance']:.0f}")
                L.append(f"　　止盈: {' '.join(tps)}")
            else:
                L.append(f"　　止损(50x): 一档${px - 0.5 * atr:.2f} 二档${lo:.2f}(摆动低)")
                tps = []
                if hi > px:
                    tps.append(f"一压${hi:.2f}(摆动高)")
                if wy and wy["resistance"] > px:
                    tps.append(f"二压${wy['resistance']}(TR顶); 突破看${2 * wy['resistance'] - wy['support']:.0f}")
                if tps:
                    L.append(f"　　止盈: {' '.join(tps)}")
    # 数据看板: 只列有统计优势的底牌(≥65%/≤35%), 无优势整块不显示(2026-08-12)
    board = build_data_board(sym, result, lv, ctx4)
    if board:
        L.append("")
        L.append("📋 数据看板")
        L.extend(board)
    # 持仓导航: 压力/支撑+突破跌破后的去向(2026-08-20 用户指定精简版); TR目标只在现价未过位时给
    if d15 and lv:
        hi, lo = lv["swing_high"], lv["swing_low"]
        wy_nav = (ctx4 or {}).get("wyckoff")
        L.append("")
        if d15 == "SHORT":
            L.append("🧭 持仓导航 (若持空)")
            L.append(f"　　压力: ${hi:.2f}(摆动高), 站稳上方 = 空头失效该撤")
            nxt = f", 跌破加速 → 下看${wy_nav['support']}(TR底)" if (wy_nav and wy_nav["support"] < px) else ", 跌破 = 加速段"
            L.append(f"　　支撑: ${lo:.2f}(摆动低){nxt}")
        else:
            L.append("🧭 持仓导航 (若持多)")
            L.append(f"　　支撑: ${lo:.2f}(摆动低), 跌破下方 = 多头失效该撤")
            nxt = f", 突破加速 → 上看${wy_nav['resistance']}(TR顶)" if (wy_nav and wy_nav["resistance"] > px) else ", 突破 = 加速段"
            L.append(f"　　压力: ${hi:.2f}(摆动高){nxt}")
    # 💬 白话信号段已移除(用户决策 2026-08-10): 内容与维度行重复(同属结构维度), 规则文案不如双模解读
    L.append("")

    if ctx4 and ctx4.get("wyckoff"):
        wy = ctx4["wyckoff"]
        px = result["price"]
        # 区间常驻: 用户决策 2026-08-08, 不再要求"有事件或贴近边界"才显示
        pos = (px - wy["support"]) / (wy["resistance"] - wy["support"]) * 100
    # 常驻指标区: 压力/支撑 → Wyckoff区间 → RSI → VWAP → 费率, 无emoji无填充空格(用户指定版式 2026-08-08)
    if lv:
        L.append(f"压力: {lv['resistance']}")
        L.append(f"支撑: {lv['support']}")
        L.append(f"EMA20: ${lv['ema20']:.2f}")
        L.append(f"EMA50: ${lv['ema50']:.2f}")
    L.append("")
    # 威科夫事件(假突破/假跌破/真突破): 有则显示; 超3根4h(12h)或价格已远离该位(行情走完)不显示(2026-08-20 过时事件事故)
    if ctx4 and ctx4.get("wyckoff") and wy.get("event") and (wy.get("event_age") or 0) <= 3:
        sup, res = wy["support"], wy["resistance"]
        if (wy["event"] in ("joc_up", "upthrust") and px > res * 1.02) or \
           (wy["event"] in ("joc_down", "spring") and px < sup * 0.98):
            wy = dict(wy, event=None)  # 价格已走出2%以上, 事件失效
    if ctx4 and ctx4.get("wyckoff") and wy.get("event"):
        sup, res = wy["support"], wy["resistance"]
        age_h = wy["event_age"] * 4
        when = f"约{age_h}小时前" if age_h > 0 else "刚刚收线"
        vw = wy["event_vol"]
        dist_res = (res - px) / px * 100
        dist_sup = (px - sup) / px * 100
        if wy["event"] == "upthrust":
            L.append(f"🔥 假突破回落({when},{vw}): 一根{vw}K线冲上区间顶${res}但没站住, 被打回区间内")
            L.append(f"   → 解读: 区间顶有主力拉高出货, ${res}一带卖压重")
            L.append(f"   → 应对: 现价距顶{dist_res:.1f}%, 反弹到${res}附近做空有优势; 4h实体收破${res}此信号作废")
        elif wy["event"] == "spring":
            L.append(f"🔥 假跌破收回({when},{vw}): 一根{vw}K线跌破区间底${sup}但没留住, 很快收回区间内")
            L.append(f"   → 解读: 区间底有主力接盘吸筹, ${sup}一带买盘重")
            L.append(f"   → 应对: 现价距底{dist_sup:.1f}%, 回踩${sup}附近做多有优势; 4h实体收破${sup}此信号作废")
        elif wy["event"] == "joc_up":
            L.append(f"🚀 真突破({when},{vw}): 4h实体收上区间顶${res}, 不是假突破")
            L.append(f"   → 解读: 上升行情启动确认, 原区间顶${res}变成支撑")
            L.append(f"   → 应对: 顺势做多, 回踩${res}附近是入场区; 跌回区间内此信号作废")
        elif wy["event"] == "joc_down":
            L.append(f"🚀 真跌破({when},{vw}): 4h实体收破区间底${sup}, 不是假跌破")
            L.append(f"   → 解读: 下降行情启动确认, 原区间底${sup}变成压力")
            L.append(f"   → 应对: 顺势做空, 反弹${sup}附近是入场区; 涨回区间内此信号作废")
    # 静态Wyckoff区间行已撤(2026-08-20 用户决策: 变化慢不读; 事件三行保留, TR顶/底在导航目标里)
    # RSI 常驻: 15m + 4h 双周期都给, lv 缺失时用 ctx4 兜底仍显示4h RSI
    rsi_parts = []
    if lv:
        rsi_parts.append(f"(15m){rsi_tag(lv['rsi14'])}")
    if ctx4:
        rsi_parts.append(f"(4h){rsi_tag(ctx4['rsi4h'])}")
    if rsi_parts:
        L.append("RSI: " + " | ".join(rsi_parts))
    vw = compute_vwap(sym)
    if vw:
        L.append(f"VWAP: {vw[1]:+.1f}%")
    st = fetch_sentiment(sym) or {}
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        L.append(f"费率: {fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    mood = market_mood(sym)
    if mood:
        L.append(mood[0])
    liq = fetch_liquidations()
    if liq:
        L.append(f"爆仓: 1h 多{_m_usd(liq['1h']['long'])}/空{_m_usd(liq['1h']['short'])} | 4h 多{_m_usd(liq['4h']['long'])}/空{_m_usd(liq['4h']['short'])}")
    lk = btc_link_line(sym)
    if lk:
        L.append(lk)
    pm = polymarket_odds()
    if pm:
        L.append("🎲 预测市场: " + " | ".join(f"{x['topic']}{x['prob']}%" for x in pm))
    # 全行左对齐 + 板块标题前强制空行(2026-08-20 用户要求)
    heads = ("🧾", "📋", "🧭", "🔥", "🚀", "🌊", "🌏")
    out = []
    for line in (ln.lstrip("　 ") for ln in L):
        if line.startswith(heads) and out and out[-1] != "":
            out.append("")
        out.append(line)
    return "\n".join(out)


def trigger_line(result):
    """15m 短周期扳机行: 投票侧方向(非AI方向)+票数+15m Brooks形态/Spike。
    方向必须按看涨/看跌票数算——完整信号卡里 result['signal'] 已被 AI 方向覆盖, 直接用会自相矛盾"""
    ba = result.get("brooks") or {}
    bull, bear = result["bullish"], result["bearish"]
    t_arrow = "🔺多" if bull > bear else "🔻空" if bear > bull else "⏸中性"
    parts = [f"{t_arrow} (看涨{bull}/看跌{bear})"]
    if ba.get("setups"):
        parts.append("形态: " + "; ".join(ba["setups"]))
    spike = ba.get("spike", 0)
    if spike:
        parts.append("Spike: " + ("强势向上突破" if spike == 1 else "强势向下跌破"))
    return "⚡ 15m扳机: " + " | ".join(parts)


def format_signal(result, plan, judge, sig_num, ai_decision=True, is_emergency=False):
    """AI 主判完整信号卡片: AI结论 → 双周期 → 关键位 → 数据, 分段留白一目了然"""
    sym = result["symbol"]; sig = result["signal"]
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    dir_cn = "做多" if sig == "LONG" else "做空"
    emoji = "🟢" if sig == "LONG" else "🔴"

    L = []
    L.append("=" * 36)
    L.append(f"{emoji} {sym} AI信号 {dir_cn} #{sig_num} | {pd.Timestamp.now():%m-%d %H:%M}")
    L.append(f"💰 现价 ${result['price']:.2f}")
    L.append("")

    judge = judge or {}
    if judge.get("confidence", -1) >= 0:
        tier = judge.get("mag_tier")
        tier_label = ["⚪极小幅(<1%)", "🔵轻仓档(1-2%)", "🟣标准档(2-3%)", "🟠主攻档(3%+)"][tier] if tier is not None else None
        head = f"🤖 AI: ✅{dir_cn} (置信{judge['confidence']})"
        L.append(head + (f" | {tier_label}" if tier_label else ""))
        for i, rsn in enumerate((judge.get("reasons") or [])[:2]):
            L.append(f"📝 理由{i+1}: {rsn}")
    else:
        L.append("🤖 AI: 裁判不可用")
    L.append("")

    ctx4 = compute_4h_context(sym)
    if ctx4:
        b4s = {"trend_up": "上升", "trend_down": "下降", "range": "震荡"}.get(ctx4["brooks"].get("state"), "-")
        L.append(f"🌏 4h: {trend4_label(ctx4)} | Brooks4h {b4s} | 挤压{ctx4['squeeze_pct']:.0f}%分位 | RSI{rsi_tag(ctx4['rsi4h'])}")
    L.append(trigger_line(result))
    L.append("")

    # 关键位: 威科夫TR边界 + 15m支撑压力 (替代原0.6%止盈止损计划)
    if ctx4 and ctx4.get("wyckoff"):
        wy = ctx4["wyckoff"]
        L.append(f"📦 威科夫TR: ${wy['support']} — ${wy['resistance']} (宽{wy['width_pct']}%)")
    lv = compute_levels(sym, sig)
    if lv:
        L.append(f"🗝️ {han_pad('支撑:', 8)} {lv['support']}")
        L.append(f"🗝️ {han_pad('压力:', 8)} {lv['resistance']}")
    L.append("")

    parts = [f"看涨{result['bullish']}/看跌{result['bearish']}", state_cn]
    if lv:
        parts.append(f"RSI{rsi_tag(lv['rsi14'])}")
    st = fetch_sentiment(sym) or {}
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        parts.append(f"费率{fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    if "taker_buy_sell_ratio" in st:
        parts.append(f"买卖比{st['taker_buy_sell_ratio']:.2f}")
    vw = compute_vwap(sym)
    if vw:
        parts.append(f"VWAP{'上' if vw[1] >= 0 else '下'}{vw[1]:+.1f}%")
    L.append("📊 " + " | ".join(parts))

    stt = judge.get("stats") or {}
    if stt.get("total", 0) >= 5:
        L.append(f"📈 判对率: {round(stt['wins'] / stt['total'] * 100)}% ({stt['wins']}/{stt['total']})")
    return "\n".join(L)


def compute_vwap(sym):
    """日内 VWAP(UTC 0点锚定): 价格在VWAP上=买方控盘, 下=卖方控盘; 返回 (vwap, 偏离%)"""
    try:
        df = fetch_klines(sym, "15m", 100)
        if df.empty:
            return None
        day_start = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        d = df[df.index >= day_start]
        if len(d) < 4:  # 刚跨日数据太少, 用近24h滚动
            d = df.tail(96)
        tp = (d["high"] + d["low"] + d["close"]) / 3
        vsum = float(d["volume"].sum())
        if vsum <= 0:
            return None
        vwap = float((tp * d["volume"]).sum()) / vsum
        px = float(d["close"].iloc[-1])
        return vwap, (px - vwap) / vwap * 100
    except Exception:
        return None


def compute_levels(sym, sig):
    """15m 关键价位/指标状态 (近20高低点/EMA/ATR/RSI/量比), 供 AI 裁判简报共用"""
    try:
        df15 = fetch_klines(sym, "15m", 80)
        c15, h15, l15 = df15["close"], df15["high"], df15["low"]
        swing_high = h15.iloc[-20:].max()
        swing_low = l15.iloc[-20:].min()
        recent_high = h15.iloc[-5:].max()
        ema20 = c15.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = c15.ewm(span=50, adjust=False).mean().iloc[-1] if len(c15) >= 50 else ema20
        atr14 = float((h15 - l15).tail(14).mean())
        rsi14 = float((c15.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] /
                        (c15.diff().abs().ewm(alpha=1/14, adjust=False).mean().iloc[-1] + 1e-10) * 100))
        vol_ratio = df15["volume"].iloc[-1] / df15["volume"].iloc[-20:].mean()
        vol_word = "明显放量，有资金进场" if vol_ratio > 1.5 else "量能正常" if vol_ratio > 0.8 else "明显缩量，观望情绪重"
        if sig == "LONG":
            resistance = f"${swing_high:.2f}(近20周期高点)"
            support = f"${swing_low:.2f}(近20周期低点)"
        else:
            resistance = f"${recent_high:.2f}(近5周期高点)、${swing_high:.2f}(近20周期高点)"
            support = f"${swing_low:.2f}(近20周期低点)"
        # EMA 不再并入压力/支撑行, 卡片单独成行(用户指定版式 2026-08-10)
        return {"support": support, "resistance": resistance, "rsi14": rsi14,
                "atr14": atr14, "ema20": ema20, "ema50": ema50,
                "vol_ratio": vol_ratio, "vol_word": vol_word,
                "rng_30m": float(h15.iloc[-2:].max() - l15.iloc[-2:].min()),  # 近30分钟振幅, 波动骤增检测用
                "swing_high": float(swing_high), "swing_low": float(swing_low),  # 数值版, 异动检测用
                "up15": bool(c15.iloc[-1] >= c15.iloc[-2]),  # 最新15m bar阴阳, 统计匹配用
                "recent_high": float(recent_high)}
    except Exception:
        return None


STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_stats.json")


def _4h_history(sym, days=1400):
    """多年 4h K线: 优先 Hyperliquid 分页(每页可达数千根, 2023年中起, 完整无缺失);
    Coinbase 并发分页兜底(美站可能被限流出空洞)"""
    coin = sym.replace("USDT", "").replace("USDC", "")
    t1 = int(time.time() * 1000)
    t0 = t1 - days * 86400 * 1000
    rows = []
    try:
        while t0 < t1:
            d = _http.post("https://api.hyperliquid.xyz/info", timeout=20, json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": "4h", "startTime": t0, "endTime": t1}}).json()
            if not isinstance(d, list) or not d:
                break
            rows.extend(d)
            last = int(d[-1]["t"])
            if last <= t0:
                break
            t0 = last + 1
            time.sleep(0.2)
        if len(rows) > 2000:
            df = pd.DataFrame([{"date": pd.to_datetime(int(x["t"]), unit="ms"),
                                "open": float(x["o"]), "high": float(x["h"]), "low": float(x["l"]),
                                "close": float(x["c"]), "volume": float(x["v"])} for x in rows])
            return df.drop_duplicates("date").set_index("date").sort_index()
    except Exception:
        pass
    return _cb_4h_history(sym, days)


def _cb_4h_history(sym, days=1400):
    """Coinbase 1h 分页拉多年数据再重采样 4h; 分页上限300根/次, 8路并发"""
    from concurrent.futures import ThreadPoolExecutor
    cb_sym = {"ETHUSDT": "ETH-USD", "ETHUSDC": "ETH-USD",
              "BTCUSDT": "BTC-USD", "BTCUSDC": "BTC-USD"}.get(sym)
    if not cb_sym:
        return None
    t_end = int(time.time())
    t_start = t_end - days * 86400
    pages = []
    t = t_end
    while t > t_start:
        pages.append((max(t_start, t - 300 * 3600), t))
        t -= 300 * 3600

    def _get(pg):
        p0 = pd.Timestamp(pg[0], unit="s", tz="UTC").isoformat()
        p1 = pd.Timestamp(pg[1], unit="s", tz="UTC").isoformat()
        try:
            d = http_get_json(f"https://api.exchange.coinbase.com/products/{cb_sym}/candles"
                              f"?granularity=3600&start={p0}&end={p1}", timeout=20)
            return d if isinstance(d, list) else []
        except Exception:
            return []

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for d in ex.map(_get, pages):
            rows.extend(d)
    if len(rows) < 2000:
        return None
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df4 = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()
    return df4 if len(df4) > 500 else None


def _cb_15m_history(sym, days=200):
    """Coinbase 15m 分页(300根/次, 8路并发): HL 15m 只给最近~5000根, 深历史走这里(2026-08-10实测)"""
    from concurrent.futures import ThreadPoolExecutor
    cb_sym = {"ETHUSDT": "ETH-USD", "ETHUSDC": "ETH-USD",
              "BTCUSDT": "BTC-USD", "BTCUSDC": "BTC-USD"}.get(sym)
    if not cb_sym:
        return None
    t_end = int(time.time())
    t_start = t_end - days * 86400
    pages = []
    t = t_end
    while t > t_start:
        pages.append((max(t_start, t - 300 * 900), t))
        t -= 300 * 900

    def _get(pg):
        p0 = pd.Timestamp(pg[0], unit="s", tz="UTC").isoformat()
        p1 = pd.Timestamp(pg[1], unit="s", tz="UTC").isoformat()
        url = (f"https://api.exchange.coinbase.com/products/{cb_sym}/candles"
               f"?granularity=900&start={p0}&end={p1}")
        for _ in range(3):  # 限流重试(2026-08-10: 8路并发64页约2/3被429)
            try:
                d = http_get_json(url, timeout=20)
                if isinstance(d, list) and d:
                    return d
            except Exception:
                pass
            time.sleep(1.0)
        return []

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for d in ex.map(_get, pages):
            rows.extend(d)
    if len(rows) < 5000:
        return None
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    for c_ in ["open", "high", "low", "close", "volume"]:
        df[c_] = df[c_].astype(float)
    return df


def _bn_4h_history(sym, days=1400):
    """Binance.US 4h 分页(1000根/页): 统计库主用价源(2026-08-10 发现 HL 分页有洞, 收益max+111%属坏数据)"""
    coin = sym.replace("USDC", "").replace("USDT", "")
    t1 = int(time.time() * 1000)
    t0 = t1 - days * 86400 * 1000
    rows = []
    try:
        while t0 < t1:
            d = http_get_json(f"https://api.binance.us/api/v3/klines?symbol={coin}USDT"
                              f"&interval=4h&limit=1000&startTime={t0}&endTime={t1}", timeout=20)
            if not isinstance(d, list) or not d:
                break
            rows.extend(d)
            last = int(d[-1][0])
            if last <= t0:
                break
            t0 = last + 1
            time.sleep(0.15)
        if len(rows) > 2000:
            df = pd.DataFrame([{"date": pd.to_datetime(int(x[0]), unit="ms"),
                                "open": float(x[1]), "high": float(x[2]), "low": float(x[3]),
                                "close": float(x[4]), "volume": float(x[5])} for x in rows])
            return df.drop_duplicates("date").set_index("date").sort_index()
    except Exception:
        pass
    return _4h_history(sym, days)  # BN 失败才回退 HL(有洞但比没有强)


def _bn_15m_history(sym, days=2000):
    """Binance.US 15m 分页(1000根/页): 2000天≈192页, 约2分钟; 统计用途深度优先(2026-08-10 用户要求)"""
    coin = sym.replace("USDC", "").replace("USDT", "")
    t1 = int(time.time() * 1000)
    t0 = t1 - days * 86400 * 1000
    rows = []
    try:
        while t0 < t1:
            d = http_get_json(f"https://api.binance.us/api/v3/klines?symbol={coin}USDT"
                              f"&interval=15m&limit=1000&startTime={t0}&endTime={t1}", timeout=20)
            if not isinstance(d, list) or not d:
                break
            rows.extend(d)
            last = int(d[-1][0])
            if last <= t0:
                break
            t0 = last + 1
            time.sleep(0.15)
        if len(rows) > 10000:
            df = pd.DataFrame([{"date": pd.to_datetime(int(x[0]), unit="ms"),
                                "open": float(x[1]), "high": float(x[2]), "low": float(x[3]),
                                "close": float(x[4]), "volume": float(x[5])} for x in rows])
            return df.drop_duplicates("date").set_index("date").sort_index()
    except Exception:
        pass
    return None


def _15m_history(sym, days=2000):
    """15m 深历史(统计库用): Binance.US 2000天主选; HL(~52天)/Coinbase(200天)兜底"""
    df = _bn_15m_history(sym, days)
    if df is not None:
        return df
    coin = sym.replace("USDT", "").replace("USDC", "")
    t1 = int(time.time() * 1000)
    t0 = t1 - days * 86400 * 1000
    rows = []
    try:
        while t0 < t1:
            d = _http.post("https://api.hyperliquid.xyz/info", timeout=20, json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": "15m", "startTime": t0, "endTime": t1}}).json()
            if not isinstance(d, list) or not d:
                break
            rows.extend(d)
            last = max(int(x["t"]) for x in d)  # 15m 响应是倒序, 不能取 d[-1](2026-08-10 实测)
            if last <= t0:
                break
            t0 = last + 1
            time.sleep(0.2)
        if len(rows) > 10000:  # HL 15m 实际只回最近~5000根, 不够就落到 Coinbase 兜底
            df = pd.DataFrame([{"date": pd.to_datetime(int(x["t"]), unit="ms"),
                                "open": float(x["o"]), "high": float(x["h"]), "low": float(x["l"]),
                                "close": float(x["c"]), "volume": float(x["v"])} for x in rows])
            return df.drop_duplicates("date").set_index("date").sort_index()
    except Exception:
        pass
    return _cb_15m_history(sym, 200)  # HL 15m 只有近~52天, 不够就走 Coinbase 200天


def market_stats(sym, refresh=False):
    """多年(约4年, 跨牛熊)历史条件统计: 挤压/RSI极值/连阳连阴/量比/费率极值 → 12h概率。
    全量计算约需数分钟(Coinbase分页+HL费率), 因此只读缓存; 无缓存才现算, 每日刷新走 --refresh-stats(cron)"""
    try:
        cache = json.load(open(STATS_FILE)) if os.path.exists(STATS_FILE) else {}
        if not refresh and cache.get(sym, {}).get("stats"):
            return cache[sym]["stats"]
    except Exception:
        cache = {}
    try:
        df = _bn_4h_history(sym)
        if df is None:
            return None
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        n = len(df)
        # 前向12h(3根4h)最大顺向/逆向偏移%
        up = pd.Series(np.nan, index=df.index)
        dn = pd.Series(np.nan, index=df.index)
        hi3 = h.shift(-1).rolling(3).max().shift(-2)
        lo3 = l.shift(-1).rolling(3).min().shift(-2)
        up = (hi3 - c) / c * 100
        dn = (c - lo3) / c * 100

        stats = {}
        # ── 挤压分位 → 12h 振幅概率 ──
        ma = c.rolling(20).mean()
        bbw = 4 * c.rolling(20).std() / ma
        # 挤压分位向量化: 每根对过去200根的分位(sliding window), rolling.apply 全量要分钟级
        from numpy.lib.stride_tricks import sliding_window_view
        arr = bbw.values
        pct = pd.Series(np.nan, index=df.index)
        if n >= 200:
            w = sliding_window_view(arr, 200)
            valid = ~np.isnan(w).any(axis=1)
            frac = np.where(valid, (w < w[:, -1:]).mean(axis=1) * 100, np.nan)
            pct.iloc[199:] = frac
        amp = pd.concat([up, dn], axis=1).max(axis=1)
        sq_stats = {}
        for lo_, hi_ in ((0, 20), (20, 40), (40, 70), (70, 100)):
            m = (pct >= lo_) & (pct < hi_)
            if int(m.sum()) >= 30:
                sq_stats[f"{lo_}-{hi_}"] = {"n": int(m.sum()),
                                            "p1": round(float((amp[m] >= 1).mean()) * 100),
                                            "p2": round(float((amp[m] >= 2).mean()) * 100),
                                            "p3": round(float((amp[m] >= 3).mean()) * 100)}
        stats["squeeze"] = sq_stats

        # ── 4h RSI 极值 → 12h 反弹/回落≥1%概率 ──
        rsi = c.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (
            c.diff().abs().ewm(alpha=1/14, adjust=False).mean() + 1e-10) * 100
        rsi_stats = {}
        for label, m in (("<=30", rsi <= 30), (">=70", rsi >= 70)):
            if int(m.sum()) >= 20:
                rsi_stats[label] = {"n": int(m.sum()),
                                    "bounce": round(float((up[m] >= 1).mean()) * 100),
                                    "drop": round(float((dn[m] >= 1).mean()) * 100)}
        stats["rsi"] = rsi_stats

        # ── 4h 连阳/连阴≥3 → 次根同向概率 ──
        sign = np.sign(c.diff())
        streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
        streak = streak * sign
        stk_stats = {}
        for label, m in (("连阳>=3", streak >= 3), ("连阴>=3", streak <= -3)):
            if int(m.sum()) >= 20:
                nxt = sign.shift(-1)
                stk_stats[label] = {"n": int(m.sum()),
                                    "cont": round(float((nxt[m] == (1 if "阳" in label else -1)).mean()) * 100)}
        stats["streak"] = stk_stats

        # ── 4h 量比>2 → 12h 同向延续≥1%概率 ──
        vr = v / v.rolling(20).mean()
        vol_stats = {}
        m_up = (vr > 2) & (c.diff() > 0)
        m_dn = (vr > 2) & (c.diff() < 0)
        if int(m_up.sum()) >= 20:
            vol_stats["放量上涨"] = {"n": int(m_up.sum()), "cont": round(float((up[m_up] >= 1).mean()) * 100)}
        if int(m_dn.sum()) >= 20:
            vol_stats["放量下跌"] = {"n": int(m_dn.sum()), "cont": round(float((dn[m_dn] >= 1).mean()) * 100)}
        stats["volume"] = vol_stats

        # ── 均线排列 → 12h 涨/跌≥1%概率(2026-08-10 维度统计扩充) ──
        e7 = c.ewm(span=7, adjust=False).mean()
        e25 = c.ewm(span=25, adjust=False).mean()
        e99 = c.ewm(span=99, adjust=False).mean()
        ta_stats = {}
        for label, m in (("多头排列", (e7 > e25) & (e25 > e99)), ("空头排列", (e7 < e25) & (e25 < e99))):
            if int(m.sum()) >= 50:
                ta_stats[label] = {"n": int(m.sum()),
                                   "up": round(float((up[m] >= 1).mean()) * 100),
                                   "dn": round(float((dn[m] >= 1).mean()) * 100)}
        stats["trend_align"] = ta_stats

        # RSI 中段分档补全(原有 <=30/>=70 极值)
        for label, m in (("30-45", (rsi > 30) & (rsi <= 45)), ("45-55", (rsi > 45) & (rsi <= 55)),
                         ("55-70", (rsi > 55) & (rsi < 70))):
            if int(m.sum()) >= 30:
                rsi_stats[label] = {"n": int(m.sum()),
                                    "bounce": round(float((up[m] >= 1).mean()) * 100),
                                    "drop": round(float((dn[m] >= 1).mean()) * 100)}

        # 量比全分档补全(原有仅>2)
        dlt = c.diff()
        for label, lo_, hi_ in (("缩量<0.6", 0, 0.6), ("平量0.6-1.2", 0.6, 1.2), ("放量1.2-2", 1.2, 2.0)):
            for dname, dm, fwd in (("阳线", dlt > 0, up), ("阴线", dlt < 0, dn)):
                m = (vr >= lo_) & (vr < hi_) & dm
                if int(m.sum()) >= 30:
                    vol_stats[f"{label}{dname}"] = {"n": int(m.sum()),
                                                    "cont": round(float((fwd[m] >= 1).mean()) * 100)}

        # ── 15m 级统计(近400天): RSI分档/量比分档/贴近20根高点 → 后4h 概率 ──
        try:
            df15h = _15m_history(sym)
            if df15h is not None and len(df15h) > 10000:
                c15, h15, l15, v15 = df15h["close"], df15h["high"], df15h["low"], df15h["volume"]
                hi16 = h15.shift(-1).rolling(16).max().shift(-15)
                lo16 = l15.shift(-1).rolling(16).min().shift(-15)
                up4 = (hi16 - c15) / c15 * 100  # 后4h最大上浮%
                dn4 = (c15 - lo16) / c15 * 100
                dlt15 = c15.diff()
                rsi15 = c15.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (
                    c15.diff().abs().ewm(alpha=1/14, adjust=False).mean() + 1e-10) * 100
                vr15 = v15 / v15.rolling(20).mean()
                m15 = {}
                for label, m in (("<=30", rsi15 <= 30), ("30-45", (rsi15 > 30) & (rsi15 <= 45)),
                                 ("45-55", (rsi15 > 45) & (rsi15 <= 55)), ("55-70", (rsi15 > 55) & (rsi15 < 70)),
                                 (">=70", rsi15 >= 70)):
                    if int(m.sum()) >= 50:
                        m15[f"rsi{label}"] = {"n": int(m.sum()),
                                              "bounce": round(float((up4[m] >= 0.5).mean()) * 100),
                                              "drop": round(float((dn4[m] >= 0.5).mean()) * 100)}
                for label, lo_, hi_ in (("<0.6", 0, 0.6), ("0.6-1.2", 0.6, 1.2), ("1.2-2", 1.2, 2.0), (">2", 2.0, 99)):
                    for dname, dm, fwd in (("阳", dlt15 > 0, up4), ("阴", dlt15 < 0, dn4)):
                        m = (vr15 >= lo_) & (vr15 < hi_) & dm
                        if int(m.sum()) >= 50:
                            m15[f"量比{label}{dname}"] = {"n": int(m.sum()),
                                                          "cont": round(float((fwd[m] >= 0.5).mean()) * 100)}
                near_hi = ((h15.rolling(20).max() - c15) / c15 * 100 <= 0.3)
                if int(near_hi.sum()) >= 50:
                    m15["贴近20根高点"] = {"n": int(near_hi.sum()),
                                           "brk": round(float((up4[near_hi] >= 0.5).mean()) * 100),
                                           "fail": round(float((dn4[near_hi] >= 0.5).mean()) * 100)}
                stats["m15"] = m15
        except Exception as e:
            print(f"WARN: 15m统计失败 {type(e).__name__}: {e}")

        # ── 资金费率极值(HL, 2023起): |8h费率|>0.05% → 12h 反向≥1%概率 ──
        try:
            coin = sym.replace("USDT", "").replace("USDC", "")
            t0 = int((time.time() - 1100 * 86400) * 1000)
            frs = []
            while True:
                d = _http.post("https://api.hyperliquid.xyz/info", timeout=15, json={
                    "type": "fundingHistory", "coin": coin, "startTime": t0}).json()
                if not isinstance(d, list) or not d:
                    break
                frs.extend(d)
                if len(d) < 500:
                    break
                t0 = int(d[-1]["time"]) + 1
                time.sleep(0.15)
            if len(frs) > 500:
                fr = pd.Series({pd.Timestamp(int(x["time"]), unit="ms"): float(x["fundingRate"]) * 8 * 100 for x in frs})
                fr8 = fr.resample("8h").sum().dropna()  # 8h费率%
                # 向量化: 价格序列转 numpy, 用 searchsorted 定位窗口, 避免逐条布尔切片
                c1h = c.resample("1h").last().ffill()
                idx = c1h.index.values.astype("datetime64[ns]").astype("int64")
                vals = c1h.values
                fstats = {}
                for label, cond in (("多付空>0.05%", fr8 > 0.05), ("空付多>0.05%", fr8 < -0.05)):
                    ts = fr8[cond].index
                    revs = []
                    for t in ts:
                        p = t.to_datetime64().astype("int64")
                        i0 = np.searchsorted(idx, p, side="right")
                        i1 = np.searchsorted(idx, p + 12 * 3600 * 10**9, side="right")
                        if i0 == 0 or i1 - i0 < 8:
                            continue
                        p0_ = vals[i0 - 1]
                        win = vals[i0:i1]
                        rev = ((p0_ - win.min()) if "多付" in label else (win.max() - p0_)) / p0_ * 100
                        revs.append(rev >= 1)
                    if len(revs) >= 20:
                        fstats[label] = {"n": len(revs), "rev": round(sum(revs) / len(revs) * 100)}
                stats["funding"] = fstats
        except Exception as e:
            print(f"WARN: 费率统计失败 {type(e).__name__}: {e}")

        cache[sym] = {"date": pd.Timestamp.now().strftime("%Y-%m-%d"), "stats": stats}
        try:
            json.dump(cache, open(STATS_FILE, "w"))
        except Exception:
            pass
        return stats or None
    except Exception:
        return None


def compute_4h_context(sym):
    """4h 大周期研判: 趋势排列/Brooks4h/波动率挤压分位/4h ATR%; 大行情(3%+)几乎只从低挤压分位+趋势共振里长出来"""
    try:
        df = fetch_klines(sym, "4h", 200)
        if df.empty or len(df) < 60:
            return None
        c, h, l = df["close"], df["high"], df["low"]
        # 趋势排列口径与币安盘面一致: EMA7/25/99; 全顺=多排/空排, 否则纠缠
        ema7 = float(c.ewm(span=7, adjust=False).mean().iloc[-1])
        ema25 = float(c.ewm(span=25, adjust=False).mean().iloc[-1])
        ema99 = float(c.ewm(span=99, adjust=False).mean().iloc[-1])
        trend_up = ema7 > ema25 > ema99
        trend_dn = ema7 < ema25 < ema99
        # 波动率挤压: 当前布林带宽(20,2)在近200根中的分位; <20% = 极度压缩, 大行情临近
        ma = c.rolling(20).mean()
        bbw = 4 * c.rolling(20).std() / ma
        squeeze_pct = float(bbw.iloc[:-1].rank(pct=True).iloc[-1] * 100)
        atr4h_pct = float((h - l).tail(14).mean()) / float(c.iloc[-1]) * 100
        rsi4h = float((c.diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1] /
                       (c.diff().abs().ewm(alpha=1/14, adjust=False).mean().iloc[-1] + 1e-10) * 100))
        # 连阳/连阴计数 & 量比(供条件统计显示)
        sgn = np.sign(c.diff())
        streak4 = int((sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1).iloc[-1] * sgn.iloc[-1])
        vol_ratio4h = float(df["volume"].iloc[-1] / df["volume"].iloc[-20:].mean())
        up4 = bool(c.diff().iloc[-1] > 0)
        ba4 = brooks_analyze(df)
        return {"trend_up": trend_up, "trend_dn": trend_dn, "above_ema20": float(c.iloc[-1]) > ema25,
                "ema7": ema7, "ema25": ema25, "ema99": ema99,
                "squeeze_pct": squeeze_pct, "atr4h_pct": atr4h_pct, "rsi4h": rsi4h,
                "streak4": streak4, "vol_ratio4h": vol_ratio4h, "up4": up4, "brooks": ba4,
                "wyckoff": _wyckoff(df, atr4h_pct)}
    except Exception:
        return None


def _wyckoff(df, atr4h_pct):
    """4h 威科夫: 近60根(约10天)交易区间(TR)识别 + 近6根 Spring/Upthrust/JOC 事件检测。
    Spring=假跌破TR下沿收回(吸筹末段, 做多信号); Upthrust=假突破TR上沿回落(派发末段, 做空信号);
    JOC=实体收过TR边界(趋势启动确认)"""
    try:
        c, h, l = df["close"].values, df["high"].values, df["low"].values
        v = df["volume"].values
        n = len(c)
        if n < 70:
            return None
        win = slice(n - 64, n - 4)
        sup, res = float(l[win].min()), float(h[win].max())
        width_pct = (res - sup) / ((res + sup) / 2) * 100
        # TR 宽度上限随波动率自适应: 约8倍4h ATR (ETH 4h 10天区间常见5-10%), 下限0.3%防死盘
        if width_pct > max(7.0, 8 * atr4h_pct) or width_pct < 0.3:  # 地板7.0: 低波动期8×ATR只有4-5%, 6.0会误拒有效TR(2026-08-10)
            return None
        vavg = float(np.mean(v[-20:])) or 1.0
        out = {"tr": True, "support": round(sup, 2), "resistance": round(res, 2),
               "width_pct": round(width_pct, 1), "event": None, "event_age": None, "event_vol": None}
        for i in range(n - 6, n):
            age = n - 1 - i
            vw = "放量" if v[i] > 1.5 * vavg else "缩量" if v[i] < 0.8 * vavg else "平量"
            if l[i] < sup * 0.999 and c[i] > sup:  # 假跌破收回
                out.update(event="spring", event_age=age, event_vol=vw)
            elif h[i] > res * 1.001 and c[i] < res:  # 假突破回落
                out.update(event="upthrust", event_age=age, event_vol=vw)
        if out["event"] is None:  # 无假动作才看 JOC 实体突破
            atr_abs = atr4h_pct / 100 * c[-1]
            if c[-1] > res and c[-1] - df["open"].values[-1] > atr_abs:
                out.update(event="joc_up", event_age=0, event_vol="放量" if v[-1] > 1.5 * vavg else "平量")
            elif c[-1] < sup and df["open"].values[-1] - c[-1] > atr_abs:
                out.update(event="joc_down", event_age=0, event_vol="放量" if v[-1] > 1.5 * vavg else "平量")
        return out
    except Exception:
        return None


def _sentiment_binance(symbol):
    """Binance USD-M 合约情绪(免key): 资金费率/持仓量1h变化/多空账户比/主动买卖比; 单接口失败跳过, 全失败返回 None"""
    base = "https://fapi.binance.com"
    s = {}

    try:  # 资金费率
        d = http_get_json(f"{base}/fapi/v1/premiumIndex?symbol={symbol}")
        s["funding_rate"] = float(d["lastFundingRate"])
    except Exception:
        pass

    try:  # 持仓量当前值
        d = http_get_json(f"{base}/fapi/v1/openInterest?symbol={symbol}")
        s["open_interest"] = float(d["openInterest"])
    except Exception:
        pass

    try:  # 持仓量1h变化: 5m×13根, 首尾对比
        d = http_get_json(f"{base}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=13")
        if isinstance(d, list) and len(d) >= 2:
            oi_first = float(d[0]["sumOpenInterest"])
            oi_last = float(d[-1]["sumOpenInterest"])
            if oi_first > 0:
                s["oi_change_1h_pct"] = (oi_last - oi_first) / oi_first * 100
    except Exception:
        pass

    try:  # 多空账户比
        d = http_get_json(f"{base}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1")
        if isinstance(d, list) and d:
            s["long_short_ratio"] = float(d[0]["longShortRatio"])
    except Exception:
        pass

    try:  # 主动买卖量比
        d = http_get_json(f"{base}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1")
        if isinstance(d, list) and d:
            s["taker_buy_sell_ratio"] = float(d[0]["buySellRatio"])
    except Exception:
        pass

    return s or None


def _sentiment_kraken(symbol):
    """Kraken Futures 回退源(美国可访问, 免key): 只有资金费率+持仓量, 无 OI 历史/多空比接口"""
    k_sym = {"ETHUSDT": "PI_ETHUSD", "ETHUSDC": "PI_ETHUSD",
             "BTCUSDT": "PI_XBTUSD", "BTCUSDC": "PI_XBTUSD"}.get(symbol)
    if not k_sym:
        return None
    try:
        d = http_get_json("https://futures.kraken.com/derivatives/api/v3/tickers")
        for t in d.get("tickers", []):
            if t.get("symbol") != k_sym:
                continue
            s = {}
            # Kraken fundingRate 是按秒计的连续资金费率(实测约 1e-10~1e-9 量级),
            # ×28800(8h秒数) 换算成 Binance lastFundingRate 同口径的 8h 费率
            if "fundingRate" in t:
                s["funding_rate"] = float(t["fundingRate"]) * 28800
            if "openInterest" in t:
                s["open_interest"] = float(t["openInterest"])
            return s or None
    except Exception:
        pass
    return None


def _oi_change_from_snap(coin, oi_usd):
    """持仓量1h变化: 本地快照首尾对比(Hyperliquid 无 OI 历史接口); 样本不足1h返回 None"""
    now = time.time()
    try:
        snaps = json.load(open(OI_SNAP_FILE)) if os.path.exists(OI_SNAP_FILE) else {}
    except Exception:
        snaps = {}
    hist = [x for x in snaps.get(coin, []) if now - x[0] < 24 * 3600]
    hist.append([now, oi_usd])
    snaps[coin] = hist
    try:
        json.dump(snaps, open(OI_SNAP_FILE, "w"))
    except Exception:
        pass
    old = [x for x in hist if now - x[0] >= 55 * 60]
    if not old or old[-1][1] <= 0:
        return None
    return (oi_usd - old[-1][1]) / old[-1][1] * 100


def _sentiment_hyperliquid(symbol):
    """Hyperliquid 回退源(免key): 资金费率+持仓量; funding 是小时费率, ×8 换算成 Binance 8h 口径; 无多空账户比接口"""
    coin = symbol.replace("USDT", "").replace("USDC", "")
    try:
        d = _http.post("https://api.hyperliquid.xyz/info",
                       json={"type": "metaAndAssetCtxs"}, timeout=10).json()
        names = [u["name"] for u in d[0]["universe"]]
        ctx = d[1][names.index(coin)]
        s = {"funding_rate": float(ctx["funding"]) * 8,
             "open_interest": float(ctx["openInterest"])}
        chg = _oi_change_from_snap(coin, float(ctx["openInterest"]) * float(ctx["markPx"]))
        if chg is not None:
            s["oi_change_1h_pct"] = chg
        return s
    except Exception:
        return None


def _sentiment_okx(symbol):
    """OKX 公共接口(免key, 美区可用): 资金费率(8h口径)/持仓量/持仓1h变化/多空账户比; 单接口失败跳过, 全失败返回 None"""
    coin = symbol.replace("USDT", "").replace("USDC", "")
    inst = f"{coin}-USDT-SWAP"
    s = {}

    try:  # 资金费率(8h, 与 Binance lastFundingRate 同口径)
        d = http_get_json(f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}")
        s["funding_rate"] = float(d["data"][0]["fundingRate"])
    except Exception:
        pass

    try:  # 持仓量当前值(张数→币数, 用 oiCcy 币本位口径与 Binance openInterest 一致)
        d = http_get_json(f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={inst}")
        s["open_interest"] = float(d["data"][0]["oiCcy"])
    except Exception:
        pass

    try:  # 持仓量1h变化: 5m 历史(全市场聚合, 美元口径), 最新 vs 12根前
        d = http_get_json(f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={coin}&period=5m")
        rows = d.get("data") or []
        if len(rows) >= 13:
            now_oi, old_oi = float(rows[0][1]), float(rows[12][1])
            if old_oi > 0:
                s["oi_change_1h_pct"] = (now_oi - old_oi) / old_oi * 100
    except Exception:
        pass

    try:  # 多空账户比(与 Binance longShortRatio 同口径)
        d = http_get_json(f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={coin}&period=5m")
        rows = d.get("data") or []
        if rows:
            s["long_short_ratio"] = float(rows[0][1])
    except Exception:
        pass

    return s or None


def fetch_sentiment(symbol):
    """合约情绪: Binance 优先 → OKX 补缺(美区可用) → Kraken/Hyperliquid 兜底; 主动买卖比用K线 taker 量算; 全空返回 None"""
    s = _sentiment_binance(symbol) or {}
    if len(s) < 5:  # 有字段缺失才打 OKX, 省一次请求
        for key, v in (_sentiment_okx(symbol) or {}).items():
            s.setdefault(key, v)
    if len(s) < 5:
        for key, v in (_sentiment_kraken(symbol) or {}).items():
            s.setdefault(key, v)
    if "funding_rate" not in s or "oi_change_1h_pct" not in s:  # 仍缺则由 HL 兜底(持仓变化靠本地快照)
        for key, v in (_sentiment_hyperliquid(symbol) or {}).items():
            s.setdefault(key, v)

    # 主动买卖比: 近20根15m, taker买量 / (总量-taker买量); yfinance 源全 NaN 则跳过
    if "taker_buy_sell_ratio" not in s:
        try:
            df = fetch_klines(symbol, "15m", 20)
            t = df["taker_base"].dropna()
            if len(t) > 0:
                v_sum = df.loc[t.index, "volume"].sum()
                s["taker_buy_sell_ratio"] = t.sum() / max(v_sum - t.sum(), 1e-9)
        except Exception:
            pass

    return s or None


def _sentiment_text(sym):
    """合约情绪行(两层简报共用): 缺失字段直接省略"""
    st = fetch_sentiment(sym)
    if not st:
        return None
    parts = []
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        parts.append(f"资金费率 {fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    if "oi_change_1h_pct" in st:
        parts.append(f"持仓量1h {st['oi_change_1h_pct']:+.1f}%")
    if "long_short_ratio" in st:
        parts.append(f"多空账户比 {st['long_short_ratio']:.2f}")
    if "taker_buy_sell_ratio" in st:
        parts.append(f"主动买卖比 {st['taker_buy_sell_ratio']:.2f}")
    return "合约情绪: " + " | ".join(parts) if parts else None


def build_brief_4h(result):
    """4h 层简报: 只含大周期数据(趋势/Brooks4h/挤压/威科夫/合约情绪), 判方向和幅度档"""
    sym = result["symbol"]; px = result["price"]
    L = [f"币种: {sym} | 现价: ${px:.2f}"]
    # 方向强度分(2026-08-22 5年回测弱信号合成, 仅参考不作规则)
    try:
        sc = strength_score(sym)
        if sc is not None:
            L.append(f"方向强度分: {sc:+d} (5年回测: +2时未来4h涨54.7%, -2时41.2%)")
    except Exception:
        pass
    ml = _macro_line()
    if ml:
        L.append(ml)
    L.extend(_recent_news_lines())
    sr = self_review_block()
    if sr:
        L.append(sr)
    hb = trader_handbook(sym)
    if hb:
        L.append(hb)
    ctx4 = compute_4h_context(sym)
    if ctx4:
        b4 = ctx4["brooks"]
        state4_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(b4.get("state"), "未知")
        sq = ctx4["squeeze_pct"]
        sq_word = "极度压缩(未来12h振幅偏低,回测-15%)" if sq < 20 else "压缩中" if sq < 40 else "正常波动" if sq < 70 else "高波动(已释放,未来12h振幅偏大+23%)"
        ai4_cn = {1: "多", -1: "空"}.get(b4.get("always_in", 0), "-")
        L.append(f"4h研判: 均线{trend4_label(ctx4)}(7/25/99), 价在4hEMA25{'上' if ctx4['above_ema20'] else '下'} | "
                 f"Brooks4h: {state4_cn}, Always In: {ai4_cn}")
        L.append(f"波动率挤压: 近200根4h的{sq:.0f}%分位 ({sq_word}) | 4h ATR: {ctx4['atr4h_pct']:.2f}%(约${ctx4['atr4h_pct'] / 100 * px:.1f}) | 4hRSI: {rsi_tag(ctx4['rsi4h'])}")
        vr4 = ctx4.get("vol_ratio4h", 0)
        # 量比常驻简报(2026-08-09 用户要求: 量能是4h方向关键证据, 不再只在>2倍时才提)
        vr_word = "明显放量" if vr4 > 1.5 else "明显缩量" if vr4 < 0.8 else "平量"
        L.append(f"4h量比: {vr4:.2f} ({vr_word}, 相对近20根均量, {'阳线' if ctx4.get('up4') else '阴线'})")
        mst = market_stats(sym)
        if mst:
            yrs = "近2年"
            ta = mst.get("trend_align", {})
            cur_aln = "多头排列" if ctx4.get("trend_up") else "空头排列" if ctx4.get("trend_dn") else None
            if cur_aln and cur_aln in ta:
                t = ta[cur_aln]
                L.append(f"统计({yrs}): 4h{cur_aln}共{t['n']}次, 后12h涨≥1%占{t['up']}%, 跌≥1%占{t['dn']}%")
            b = "0-20" if sq < 20 else "20-40" if sq < 40 else "40-70" if sq < 70 else "70-100"
            if b in mst.get("squeeze", {}):
                t = mst["squeeze"][b]
                L.append(f"统计({yrs}): 挤压{b}%档{t['n']}次, 后12h振幅≥1%占{t['p1']}%, ≥2%占{t['p2']}%, ≥3%占{t['p3']}%")
            r4 = ctx4["rsi4h"]
            rb = ("<=30" if r4 <= 30 else ">=70" if r4 >= 70 else
                  "30-45" if r4 <= 45 else "45-55" if r4 <= 55 else "55-70")
            if rb and rb in mst.get("rsi", {}):
                t = mst["rsi"][rb]
                L.append(f"统计: 4hRSI{rb}共{t['n']}次, 后12h反弹≥1%占{t['bounce']}%, 回落≥1%占{t['drop']}%")
            sk = ctx4.get("streak4", 0)
            sb = "连阳>=3" if sk >= 3 else "连阴>=3" if sk <= -3 else None
            if sb and sb in mst.get("streak", {}):
                t = mst["streak"][sb]
                L.append(f"统计: 4h{sb}共{t['n']}次, 次根延续占{t['cont']}%")
            vr4 = ctx4.get("vol_ratio4h", 0)
            if vr4 > 2:
                vb = "放量上涨" if ctx4.get("up4") else "放量下跌"
                if vb in mst.get("volume", {}):
                    t = mst["volume"][vb]
                    L.append(f"统计: 4h{vb}共{t['n']}次, 后12h同向延续≥1%占{t['cont']}%")
            else:  # 全分档常引(2026-08-10): 量比不给>2也带概率
                vb = "缩量<0.6" if vr4 < 0.6 else "平量0.6-1.2" if vr4 < 1.2 else "放量1.2-2"
                key = f"{vb}{'阳线' if ctx4.get('up4') else '阴线'}"
                if key in mst.get("volume", {}):
                    t = mst["volume"][key]
                    L.append(f"统计: 4h{key}共{t['n']}次, 后12h同向延续≥1%占{t['cont']}%")
            st_now = fetch_sentiment(sym) or {}
            fr_now = st_now.get("funding_rate")
            if fr_now is not None and abs(fr_now) * 100 > 0.05:
                fb = "多付空>0.05%" if fr_now > 0 else "空付多>0.05%"
                if fb in mst.get("funding", {}):
                    t = mst["funding"][fb]
                    L.append(f"统计: 8h费率{fb}共{t['n']}次, 后12h反向≥1%占{t['rev']}%")
        wy = ctx4.get("wyckoff")
        if wy:
            ev_map = {"spring": f"Spring向下假跌破TR下沿${wy['support']}后收回({wy['event_vol']},{wy['event_age']}根前)——吸筹末段信号,偏多",
                      "upthrust": f"Upthrust向上假突破TR上沿${wy['resistance']}后回落({wy['event_vol']},{wy['event_age']}根前)——派发末段信号,偏空",
                      "joc_up": f"实体收破TR上沿${wy['resistance']}(JOC)——上升启动确认,偏多",
                      "joc_down": f"实体收破TR下沿${wy['support']}(JOC)——下降启动确认,偏空"}
            ev = ev_map.get(wy["event"], "区间内,无Spring/Upthrust事件") if wy["event"] else "区间内,无Spring/Upthrust事件"
            L.append(f"威科夫4h: TR区间 ${wy['support']}-${wy['resistance']} (宽{wy['width_pct']}%) | {ev}")
    st_line = _sentiment_text(sym)
    if st_line:
        L.append(st_line)
    liq = fetch_liquidations()
    if liq:
        L.append(f"爆仓: 近1h多{_m_usd(liq['1h']['long'])}/空{_m_usd(liq['1h']['short'])} | 近4h多{_m_usd(liq['4h']['long'])}/空{_m_usd(liq['4h']['short'])}")
    lk = btc_link_line(sym)
    if lk:
        L.append(lk)
    pm = polymarket_odds()
    if pm:
        L.append("预测市场(Polymarket赔率=市场预期的后续走势): " + " | ".join(f"{x['topic']}{x['prob']}%" for x in pm))
    return "\n".join(L)


def build_market_brief(result, plan=None, events=None, prev=None):
    """15m 层简报: 只含短周期数据(18因子投票/15m形态/关键位/RSI/量比/VWAP/合约情绪), 判入场方向
    events=本轮扫描触发的异动(规则检测原文), prev=上一次15m判决(迟滞/连续性)"""
    sym = result["symbol"]; sig = result["signal"]
    px = result["price"]
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    ai_cn = {1: "多", -1: "空"}.get(ba.get("always_in", 0), "-")
    spike_cn = {1: "强势向上突破", -1: "强势向下跌破"}.get(ba.get("spike", 0), "无")

    # 投票分布: 三组各几票看涨(🟢)
    groups = {"VT因子": [0, 8], "NOFX": [0, 4], "Brooks": [0, 6]}
    for d in result["details"]:
        name = d["name"]
        key = "NOFX" if name.startswith("NOFX_") else "Brooks" if name.startswith("BROOKS_") else "VT因子"
        if d["direction"] == "🟢":
            groups[key][0] += 1

    L = []
    L.append(f"币种: {sym} | 现价: ${px:.2f}")
    ml = _macro_line()
    if ml:
        L.append(ml)
    L.extend(_recent_news_lines())
    sr = self_review_block()
    if sr:
        L.append(sr)
    hb = trader_handbook(sym)
    if hb:
        L.append(hb)
    L.append(f"市场状态(15m): {state_cn} | Always In: {ai_cn} | Spike: {spike_cn}")
    L.append(f"Brooks形态: {'; '.join(ba['setups']) if ba.get('setups') else '无'}")
    L.append(f"投票分布: 看涨{result['bullish']}票 / 看跌{result['bearish']}票 (" +
             " | ".join(f"{k} 看涨{v[0]}/{v[1]}" for k, v in groups.items()) + ")")

    lv = compute_levels(sym, sig)
    if lv:
        L.append(f"关键支撑位: {lv['support']}")
        L.append(f"关键压力位: {lv['resistance']}")
        L.append(f"EMA20: ${lv['ema20']:.2f} | EMA50: ${lv['ema50']:.2f} | ATR14: ${lv['atr14']:.2f}")
        L.append(f"市场情绪: {rsi_word(lv['rsi14'])}")
        L.append(f"成交量: {lv['vol_word']} (量比 {lv['vol_ratio']:.2f})")
        # 15m 级历史统计注入(2026-08-10 用户要求: 各维度都要带历史概率)
        m15 = (market_stats(sym) or {}).get("m15", {})
        if m15:
            r15 = lv["rsi14"]
            rb = ("<=30" if r15 <= 30 else ">=70" if r15 >= 70 else
                  "30-45" if r15 <= 45 else "45-55" if r15 <= 55 else "55-70")
            t = m15.get(f"rsi{rb}")
            if t:
                L.append(f"统计(近2年15m): RSI{rb}共{t['n']}次, 后4h反弹≥0.5%占{t['bounce']}%, 回落≥0.5%占{t['drop']}%")
            vr_ = lv["vol_ratio"]
            vb = "<0.6" if vr_ < 0.6 else "0.6-1.2" if vr_ < 1.2 else "1.2-2" if vr_ < 2 else ">2"
            t = m15.get(f"量比{vb}{'阳' if lv.get('up15') else '阴'}")
            if t:
                L.append(f"统计: 15m量比{vb}共{t['n']}次, 后4h同向延续≥0.5%占{t['cont']}%")
            if (lv["swing_high"] - px) / px * 100 <= 0.3 and "贴近20根高点" in m15:
                t = m15["贴近20根高点"]
                L.append(f"统计: 贴近20根高点共{t['n']}次, 后4h上破≥0.5%占{t['brk']}%, 回落≥0.5%占{t['fail']}%")
        vw = compute_vwap(sym)
        if vw:
            L.append(f"VWAP(日内): ${vw[0]:.2f} | 价在VWAP{'上' if vw[1] >= 0 else '下'} ({vw[1]:+.2f}%)")

    # 布林带位置(15m+1h, 20/2): 用户做空常用依据(缩量反弹打上轨/1h中轨), 2026-08-09 加入简报
    for tf in ("15m", "1h"):
        try:
            dfb = fetch_klines(sym, tf, 60)
            cb = dfb["close"]
            mid = float(cb.rolling(20).mean().iloc[-1])
            sd = float(cb.rolling(20).std().iloc[-1])
            up, lo = mid + 2 * sd, mid - 2 * sd
            if px >= up:
                pos = "上轨外(透支)"
            elif px >= (mid + up) / 2:
                pos = "贴近上轨(压力区)"
            elif px >= mid:
                pos = "中上轨之间"
            elif px >= (mid + lo) / 2:
                pos = "中下轨之间"
            elif px >= lo:
                pos = "贴近下轨(支撑区)"
            else:
                pos = "下轨外(超跌)"
            L.append(f"布林{tf}: 上${up:.1f} 中${mid:.1f} 下${lo:.1f} | 价在{pos}")
        except Exception:
            pass

    st_line = _sentiment_text(sym)
    if st_line:
        L.append(st_line)
    liq = fetch_liquidations()
    if liq:
        L.append(f"爆仓: 近1h多{_m_usd(liq['1h']['long'])}/空{_m_usd(liq['1h']['short'])} | 近4h多{_m_usd(liq['4h']['long'])}/空{_m_usd(liq['4h']['short'])}")
    lk = btc_link_line(sym)
    if lk:
        L.append(lk)
    pm = polymarket_odds()
    if pm:
        L.append("预测市场(Polymarket赔率=市场预期的后续走势): " + " | ".join(f"{x['topic']}{x['prob']}%" for x in pm))
    if events:
        L.append("本轮扫描触发异动: " + "; ".join(events))
    else:
        L.append("本轮扫描无异动")
    if prev and prev.get("direction"):
        d_cn = {"LONG": "做多", "SHORT": "做空"}.get(prev["direction"], "观望")
        L.append(f"上一次15m判决: {d_cn} 置信{prev.get('confidence')}。证据没有明显变化就维持这个方向。")
    return "\n".join(L)


# 裁判不可用时的保守兜底: 观望(不发信号), 避免风控失效时裸奔
JUDGE_UNAVAILABLE = {"verdict": "观望", "direction": None, "confidence": -1, "mag_tier": None, "reasons": ["裁判不可用，保守观望"]}

# 预期幅度档: 未来4-12h最大有利偏移(MFE)档位
MAG_TIERS = ["<1%", "1-2%", "2-3%", "3%+"]
MAG_TIER_FLOOR = [0.0, 1.0, 2.0, 3.0]  # 各档下限(%), 结算时用


def _judge_call(system, brief, name="DS", url=None, key=None, model=None,
                temp=0.2, max_tok=400, req_timeout=35):
    """单裁判调用: system+简报 → {direction, confidence, mag_tier, reasons}; 失败重试1次再回 JUDGE_UNAVAILABLE"""
    url = url or DS_API_URL
    key = key if key is not None else DS_API_KEY
    model = model or DS_MODEL
    for attempt in (1, 2):
        try:
            r = _http.post(url, json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": brief}],
                "max_tokens": max_tok, "temperature": temp},
                headers={"Authorization": f"Bearer {key}"}, timeout=req_timeout)
            if r.status_code != 200:
                print(f"WARN: 裁判{name} {r.status_code}: {r.text[:200]}")
                if attempt == 1:
                    time.sleep(2)
                    continue
                return dict(JUDGE_UNAVAILABLE)
            text = r.json()["choices"][0]["message"]["content"]
            # 容忍 ```json 围栏和前后多余文字; 贪婪匹配首尾大括号(dims 嵌套对象, 非贪婪会截断)
            m = re.search(r"\{.*\}", text, re.S)
            try:
                data = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                m2 = re.search(r"\{[^{}]*\}", text, re.S)
                data = json.loads(m2.group(0)) if m2 else {}
            # 方向制: 含"多"→LONG, 含"空"→SHORT, 否则观望
            d_str = str(data.get("direction", ""))
            direction = "LONG" if "多" in d_str else "SHORT" if "空" in d_str else None
            confidence = int(data.get("confidence", -1))
            # 幅度档: 解析 "3%+" / "2-3%" / "1-2%" / "<1%" → tier 0-3, 无法解析→None
            mag_str = str(data.get("magnitude", ""))
            mag_tier = None
            for t, label in ((3, "3%+"), (2, "2-3"), (1, "1-2"), (0, "<1")):
                if label in mag_str:
                    mag_tier = t
                    break
            dims = data.get("dims")
            dims = {k: str(v).strip() for k, v in dims.items()} if isinstance(dims, dict) else None
            reasons = data.get("reasons")
            if not isinstance(reasons, list) or not reasons:
                reasons = list(dims.values()) if dims else [str(data.get("reason", "")).strip() or "无理由"]
            reasons = [str(x).strip() for x in reasons[:4]]
            return {"verdict": "执行" if direction else "观望", "direction": direction,
                    "confidence": confidence, "mag_tier": mag_tier, "reasons": reasons,
                    "summary": str(data.get("summary", "")).strip(), "dims": dims}
        except Exception as e:
            print(f"WARN: 裁判{name}异常 {type(e).__name__}: {e}")
            if attempt == 1:
                time.sleep(2)
                continue
            return dict(JUDGE_UNAVAILABLE)


def _dual_judge(system, brief):
    """双模裁判(2026-08-10 用户要求): DeepSeek + Kimi 并行判决。
    同向→采用(置信取均值, 理由各取2条带来源标注); 分歧→观望; 单边故障→用另一边"""
    if not MS_API_KEY:
        return _judge_call(system, brief)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        f_ds = ex.submit(_judge_call, system, brief)
        f_km = ex.submit(_judge_call, system, brief, "Kimi", MS_API_URL, MS_API_KEY, MS_MODEL,
                         1.0, 4000, 60)  # 思考模型: 温度锁1, reasoning占~1000tok, 18s级
        ds, km = f_ds.result(), f_km.result()
    ds_ok, km_ok = ds["confidence"] >= 0, km["confidence"] >= 0
    if ds_ok and not km_ok:
        ds["_src"] = "ds"
        return ds
    if km_ok and not ds_ok:
        km["reasons"] = [f"KK: {r}" for r in km["reasons"]]
        km["_src"] = "kk"  # 顶班标记: 4h缓存到KK判决时下轮重判, DS康复即拿回话筒
        return km
    if not (ds_ok or km_ok):
        return ds
    # 主裁复核制(2026-08-10 用户决策): DS 主裁方向/置信/维度, KK 只在反对时出声
    if ds.get("dims"):
        ds["reasons"] = [f"{dim} | {txt}" for dim, txt in ds["dims"].items()]
    if ds["direction"] == km["direction"]:
        return ds
    d_cn = {"LONG": "多", "SHORT": "空", None: "观望"}
    km_say = (km.get("dims") or {}).get("结构") or (km["reasons"][0] if km["reasons"] else "")
    ds["confidence"] = max(ds["confidence"] - 10, 0)
    ds["reasons"] = ds["reasons"][:4] + [f"⚠️ KK反对(判{d_cn[km['direction']]}): {km_say}"]
    return ds


SYS_4H = ("你是只做ETH的专业加密交易员(BTC为联动参照), 看大周期: 根据4h简报判断未来4-12小时的方向和最大涨跌幅。"
          "你的交易风格: 方向不明就观望, 这是专业不是怂; 进场只在有结构/统计/宏观至少两边同向时; "
          "每笔判断都要能回答\"我这单错了会怎样\"——所以每次必须给失效条件。"
          "你的信息优势: 简报里有盘面临场综述(叙事)、你自己的判决复盘(错例)、历史统计(只引有区分度的, 概率≥60%或≤40%, 五五开底噪别引)、宏观事件日提示。"
          "财政部/联储/地缘这类宏观消息出现时, 必须编进推理链(如\"财政部压收益率=流动性宽松=偏多\"), 不许只看技术面。"
          "大行情(3%+)通常只出现在波动率挤压低分位(<40%)或一方仓位拥挤时; 威科夫Spring/JOC向上偏多, Upthrust/JOC向下偏空。"
          "简报末尾可能附带上一次判决: 趋势有惯性, 证据没有明显变化就维持, 要翻转必须写清新证据是什么。"
          "置信度校准(必须用满量程): 50-60=证据混合五五开; 60-75=单方占优; 75-85=多条件同向; 85-95=全共振的罕见机会, 该给就给, 别永远停在60-72舒适区。"
          "只输出 JSON: {\"direction\": \"做多\" 或 \"做空\" 或 \"观望\", "
          "\"magnitude\": \"<1%\" 或 \"1-2%\" 或 \"2-3%\" 或 \"3%+\", "
          "\"confidence\": 0-100 整数, \"summary\": \"一句话结论\", \"reasons\": [2-4条理由]}。"
          "summary不超过35字: 方向+概率+走势形态(反弹/滞涨/趋势延续/反转), 必须带具体点位(从简报关键位里挑), 如\"大概率(65%)先回踩1903再反弹\"。"
          "理由用交易员推演链写: 每条=盘面现象→这暴露了谁在被套/谁在出货/谁在接盘→所以最可能往哪走→如果…就…的预案。"
          "例: \"1918两次冲不上去还带放量, 说明有人在这位置持续出货, 多单别在1916上方追, 等回踩1905再看\"。"
          "只挑最有信息量的2-4条把话说透, 没话说别硬凑。用大白话, 保留关键数字, 不用术语缩写, 每条不超过40字。"
          "magnitude 与方向无关也要给。")

SYS_15M = ("你是只做ETH的专业加密交易员(BTC为联动参照), 看短线: 根据15分钟简报判断未来1-2小时的方向(入场时机)。"
           "你的交易风格: 方向不明就观望; 不被投票分布锚定(投票只是参考); 每次判断都带失效条件。"
           "你的信息优势: 简报里有盘面临场综述(叙事)、本轮异动(规则检测)、你自己的判决复盘、历史统计(只引有区分度的≥60%/≤40%)。"
           "异动要结合走势解读, 不复述事件本身: 同样的RSI超买, 强势趋势里是动能、区间顶部缩量时是回落前兆; "
           "放量+逼近压力, 突破失败和真突破含义完全相反。有异动时第一条必须先写异动解读。"
           "布林带只是位置参考之一, 最多提一次。证据没有明显变化就维持上次方向, 别因噪音翻转。"
           "置信度校准(必须用满量程): 50-60=证据混合五五开; 60-75=单方占优; 75-85=多条件同向; 85-95=全共振的罕见机会, 该给就给。"
           "只输出 JSON: {\"direction\": \"做多\" 或 \"做空\" 或 \"观望\", "
           "\"confidence\": 0-100 整数, \"summary\": \"一句话结论\", \"reasons\": [2-4条理由]}。"
           "summary不超过35字: 方向+概率+走势形态, 必须带具体点位, 如\"短线偏空(60%), 反弹到1918附近滞涨再跌\"。"
           "理由用交易员推演链写: 现象→谁被套/谁在出货→往哪走→如果…就…预案。"
           "例: \"缩量反弹到布林上轨, 没人愿意在高位接, 这种反弹是用来出的不是追的, 站稳上轨再反手\"。"
           "只挑最有信息量的2-4条把话说透, 没话说别硬凑。用大白话, 保留关键数字但不用术语缩写, 每条不超过40字。")


def _judge(system, brief):
    """裁判调度: VT_JUDGE=kimi 用月之暗面(k2.5思考模型, 贵但推理强), 默认 DS(2026-08-20 用户试用Kimi单模)。
    Kimi 连续失败时自动回退 DS 兜底, 不再观望摆烂(2026-08-20 kimi超时夜)"""
    if os.environ.get("VT_JUDGE", "ds").lower() == "kimi" and MS_API_KEY:
        j = _judge_call(system, brief, "Kimi", MS_API_URL, MS_API_KEY, MS_MODEL, 1.0, 4000, 60)
        if j["confidence"] >= 0:
            return j
        print("WARN: Kimi连续失败, 本轮回退DS裁判")
    return _judge_call(system, brief)


def ai_judge_4h(result, prev=None):
    """4h 层裁判: 大周期方向+幅度档; 历史胜率/判例不注入 prompt(用户决策 2026-08-05)
    prev: 上一根4h收线的判决, 注入简报让 AI 自己扛迟滞(2026-08-08 临界区翻转抖动)"""
    brief = build_brief_4h(result)
    if prev and prev.get("direction"):
        d_cn = {"LONG": "做多", "SHORT": "做空"}.get(prev["direction"], "观望")
        tier = {0: "<1%", 1: "1-2%", 2: "2-3%", 3: "3%+"}.get(prev.get("mag_tier"), "未知")
        brief += (f"\n上一次4h判决(上根收线): {d_cn} 置信{prev.get('confidence')} 幅度档{tier}。"
                  "证据没有明显变化就维持这个方向，别因一两根K线的噪音翻转。")
    return _judge(SYS_4H, brief)


def ai_judge_15m(result, prev=None, events=None):
    """15m 层裁判: 短周期入场方向; prev=上次判决(迟滞), events=本轮异动(规则检测, AI解读)"""
    return _judge(SYS_15M, build_market_brief(result, events=events, prev=prev))


def ai_judge(result, plan=None):
    """兼容入口(旧调用/judge-test): 15m 层裁判"""
    return ai_judge_15m(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=15, help="循环间隔(分钟), 0=单次")
    parser.add_argument("--test", action="store_true", help="仅打印, 不推送")
    parser.add_argument("--symbols", default="ETHUSDC", help="逗号分隔, 如 ETHUSDC,BTCUSDC")
    parser.add_argument("--judge-test", action="store_true", help="裁判调试: 打印简报+裁判JSON后退出")
    parser.add_argument("--refresh-stats", action="store_true", help="重算多年历史统计缓存后退出(cron每日调用)")
    args = parser.parse_args()
    if args.refresh_stats:
        for s in ("ETHUSDC",):
            t0 = time.time()
            st = market_stats(s, refresh=True)
            print(f"{s} stats refreshed in {time.time()-t0:.0f}s:", json.dumps(st, ensure_ascii=False)[:200])
        return

    factor_map = {"BTCUSDT": BTC_FACTORS, "BTCUSDC": BTC_FACTORS, "ETHUSDT": ETH_FACTORS, "ETHUSDC": ETH_FACTORS}
    watch = []
    for s in args.symbols.split(","):
        s = s.strip()
        if s in factor_map:
            watch.append((s, factor_map[s]))
        elif s:
            print(f"WARN: {s} 无因子配置, 跳过")
    if not watch:
        print("FAIL: 没有可监控的币种")
        return

    if args.test:
        global send_telegram
        send_telegram = lambda x: print(f"\n[Telegram]\n{x}\n")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARN: VT_TELEGRAM_TOKEN / VT_TELEGRAM_CHAT 未设置, Telegram 推送将失败")

    print(f"VT投票信号机器人 v4.0 | 双层AI(4h方向+幅度档 / 15m入场) | 15分钟定时信号+翻向加推 | 监控: {args.symbols}")
    print(f"{'='*60}")

    print("预加载因子...", end=" ", flush=True)
    preload_factors()
    print("OK")

    # 裁判调试: 跳过票数门槛和推送, 原样打印简报和裁判结果
    if args.judge_test:
        result = vote("ETHUSDT", ETH_FACTORS)
        plan = calc_trade_plan("ETHUSDT", result["signal"], result["price"], result.get("brooks") or {})
        print("\n── 市场简报 ──")
        print(build_market_brief(result, plan))
        print("\n── 裁判结果 ──")
        print(json.dumps(ai_judge(result, plan), ensure_ascii=False))
        print("\n── 错题本 ──")
        lessons = load_lessons()
        print(json.dumps(lessons, ensure_ascii=False, indent=2) if lessons else "无错题本")
        return

    last_dirs = {}     # sym → {"4h": dir, "15m": dir}, 上次扫描双层方向, 翻向检测用
    judge4_cache = {}  # sym → (bar_key, judge4), 4h 层每根4h收线重判一次(根内输入不变, 高频重判只会抖动)
    judge15_prev = {}  # sym → 上次15m判决, 注入简报保持连续性(迟滞)
    prev_metrics = {}  # sym → 上次扫描指标快照, 异动边沿检测用
    fast_ref = {}      # sym → 快检基准(价/阈值/摆动极值/VWAP), 3分钟休眠期10秒快检用
    news_seen = set()  # 已处理新闻link, RSS去重用
    fast_seen = set()  # 快检新闻探针去重(独立于news_seen: fetch_news会写seen, 探针不能污染编辑席去重)
    last_news_poll = 0.0  # 快检新闻轮询计时(60s), 任意新新闻触发立即全扫
    # 状态持久化(2026-08-09 用户要求): 重启不失忆, 上次判决/快照/快检基准全部恢复
    STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_state.json")

    def save_state():
        try:
            json.dump({"judge4_cache": judge4_cache, "judge15_prev": judge15_prev,
                       "last_dirs": last_dirs, "prev_metrics": prev_metrics, "fast_ref": fast_ref,
                       "news_seen": list(news_seen)[-500:], "recent_news": RECENT_NEWS[-10:]},
                      open(STATE_FILE, "w"))
        except Exception as e:
            print(f"WARN: 状态保存失败 {e}")

    if os.path.exists(STATE_FILE):
        try:
            st = json.load(open(STATE_FILE))
            judge4_cache.update(st.get("judge4_cache", {}))
            judge15_prev.update(st.get("judge15_prev", {}))
            last_dirs.update(st.get("last_dirs", {}))
            prev_metrics.update(st.get("prev_metrics", {}))
            fast_ref.update(st.get("fast_ref", {}))
            news_seen.update(st.get("news_seen", []))
            fast_seen = set(news_seen)
            RECENT_NEWS.extend([tuple(x) for x in st.get("recent_news", [])])
            print(f"状态恢复: 4h缓存{len(judge4_cache)} 判决{len(judge15_prev)} 快照{len(prev_metrics)} 新闻{len(news_seen)}")
        except Exception as e:
            print(f"WARN: 状态恢复失败 {e}, 冷启动")

    last_timed_slot = None  # 最后推过的15分钟槽, 防止慢扫描吞掉定时推送(2026-08-10)
    pending_fast = None     # 快检触发的事件, 注入下轮扫描必推+AI解读
    last_fast = {}          # (sym,触发类型) → 上次触发时间, 快检同向冷却用
    while True:
        now = pd.Timestamp.now()
        minute = now.minute
        slot = now.hour * 4 + minute // 15
        # 定时: 边界扫描, 或慢扫描跨过槽位后3分钟内补推
        is_15m = (minute % 15 == 0) or (slot != last_timed_slot and minute % 15 <= 3)

        print(f"\n[{now.strftime('%H:%M:%S')}] {'15分钟定时' if is_15m else '3分钟'}扫描...")

        # ── AI 叙事复盘(取代机械结算 2026-08-23) ──
        try:
            maybe_update_lessons()
        except Exception as e:
            print(f"  叙事复盘失败: {e}")


        for sym, config in watch:
            try:
                print(f"  {sym}...", end=" ", flush=True)
                result = vote(sym, config)

                # ── 4h 层: 每根4h收线重判一次, 中间用缓存 ──
                df4 = fetch_klines(sym, "4h", 3, drop_incomplete=False)
                bar_key = str(df4.index[-1]) if not df4.empty else ""
                ck = judge4_cache.get(sym)
                # 缓存不可用判决(置信-1)或KK顶班判决都重判, 不让API故障/顶班污染整根4h(2026-08-10)
                if not ck or ck[0] != bar_key or ck[1].get("confidence", -1) < 0 or ck[1].get("_src") == "kk":
                    judge4 = ai_judge_4h(result, prev=ck[1] if ck else None)
                    judge4_cache[sym] = (bar_key, judge4)
                    print(f"\n  {sym} 4h层重判: {judge4['direction'] or '观望'} 置信{judge4['confidence']} 档{judge4.get('mag_tier')}")
                else:
                    judge4 = ck[1]

                # 异动: RSI(15m/4h)穿越/放量/逼位/挤压进出(边沿触发) — 先于15m判决, 事件要喂给裁判解读
                events, cur_metrics = detect_events(sym, result, prev_metrics.get(sym))
                prev_metrics[sym] = cur_metrics
                # 快检触发的事件直接注入: 必推+AI解读(2026-08-10 修复快检触发但事件阈值不达导致白扫不推)
                if pending_fast:
                    events.append(f"⚡快检: {pending_fast}")
                    pending_fast = None

                # 突发新闻: RSS拉新 → 标题相似度去重 → DS编辑席判价值/时效/方向/事件线(2026-08-20)
                try:
                    raw = fetch_news(news_seen)
                    fresh = []
                    for src_name, title in raw:
                        norm = _norm_title(title)
                        if any(_title_sim(norm, _norm_title(t)) > 0.55 for _, t in RECENT_NEWS):
                            continue  # 同新闻已定价, 跳过
                        fresh.append((src_name, title))
                    if fresh:
                        got_hard_news = False
                        for it in news_editor(fresh, load_news_memory()):
                            if not it.get("relevant"):
                                continue
                            cn = translate_title(it["title"])
                            tag = f"{it.get('horizon','')}·{it.get('dir','')}"
                            events.append(f"📰 [{tag}] {it['src']}: {cn} — {it.get('note','')}")
                            RECENT_NEWS.append((time.time(), f"{it['src']}: {cn}"))
                            if it.get("horizon") == "即时":
                                got_hard_news = True
                        if got_hard_news:
                            update_world_view(sym, force=True)  # 即时新闻落地后立刻重写盘面综述
                    now_ts = time.time()
                    RECENT_NEWS[:] = [x for x in RECENT_NEWS if now_ts - x[0] < 12 * 3600][-10:]
                except Exception as e:
                    print(f"  新闻拉取失败: {e}")
                update_world_view(sym, force=in_event_window())  # 事件落地窗口每轮重写, 平时30分钟节流
                # 宏观事件日: 首次生成预判时触发一条推送(2026-08-20)
                try:
                    mem_nv = load_news_memory()
                    if event_preview(sym) and not (mem_nv.get("event_preview") or {}).get("pushed"):
                        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
                        events.append(f"🗓 今日{MACRO_EVENTS.get(today_et)}, 事件预判已出(见卡片)")
                        mem_nv["event_preview"]["pushed"] = True
                        save_news_memory(mem_nv)
                except Exception:
                    pass

                # 大额爆仓: 近1h合计≥$3M(边沿触发, 2026-08-20)
                try:
                    liq = fetch_liquidations()
                    tot1h = (liq["1h"]["long"] + liq["1h"]["short"]) if liq else 0
                    if liq and tot1h >= 3e6 and not prev_metrics[sym].get("_liq_big"):
                        events.append(f"💥 大额爆仓: 近1h多{_m_usd(liq['1h']['long'])}/空{_m_usd(liq['1h']['short'])}")
                    prev_metrics[sym]["_liq_big"] = tot1h >= 3e6
                except Exception:
                    pass

                # 预测市场重定价: 主题赔率变动≥6pp → 市场预期变了, 推送+喂裁判(2026-08-20)
                try:
                    pm_now = {x["topic"]: x["prob"] for x in polymarket_odds()}
                    pm_prev = prev_metrics[sym].get("_pm", {})
                    for topic, prob in pm_now.items():
                        if topic in pm_prev and abs(prob - pm_prev[topic]) >= 6:
                            events.append(f"🎲 预测市场重定价: {topic} {pm_prev[topic]}%→{prob}%")
                    prev_metrics[sym]["_pm"] = pm_now
                except Exception:
                    pass

                # ── 15m 层: 每次扫描都判, 携带上次判决(迟滞)+本轮异动(AI解读) ──
                judge15 = ai_judge_15m(result, prev=judge15_prev.get(sym), events=events or None)
                judge15_prev[sym] = judge15
                record_judge(result, judge4, judge15)

                dirs = {"4h": judge4.get("direction"), "15m": judge15.get("direction")}
                prev = last_dirs.get(sym)
                # 翻向: 任一层新方向非空且与上次不同 → 加推反转信号
                reversal = bool(prev) and any(dirs[k] and dirs[k] != prev.get(k) for k in dirs)
                last_dirs[sym] = dirs

                # 入场窗口: 4h有方向时, 15m贴近结构位回调/放量突破(纯规则)
                lv_main = compute_levels(sym, judge4.get("direction") or judge15.get("direction") or result["signal"])
                vw_main = compute_vwap(sym)
                ctx4_main = compute_4h_context(sym)
                ew = entry_window(result, judge4, ctx4_main, lv_main, vw_main)
                ew_key = (ew["type"], ew["dir"]) if ew else None
                if ew_key and prev_metrics[sym].get("_ew") != ew_key:  # 修复: 之前读顶层dict永远None, 窗口开启事件每轮重复触发
                    events = events + [f"入场窗口开启({ew['type']})"]
                prev_metrics[sym]["_ew"] = ew_key

                # 高确信检测(2026-08-20): 共振≥4项 且 双层同向 → 🔥立即推送, 不看时间表
                conf_score, conf_missing = confluence(result, judge4.get("direction"), lv_main, ctx4_main,
                                                      build_data_board(sym, result, lv_main, ctx4_main))
                prev_metrics[sym]["_conf"] = conf_score
                if (conf_score >= 4 and judge4.get("direction")
                        and judge4["direction"] == judge15.get("direction")
                        and prev_metrics[sym].get("_hc") != (conf_score, judge4["direction"])):
                    events = events + [f"🔥高确信: 共振{conf_score}/6项同向, 双层{('做多' if judge4['direction'] == 'LONG' else '做空')}"]
                    prev_metrics[sym]["_hc"] = (conf_score, judge4["direction"])

                if is_15m or reversal or events:
                    d_adv = judge4.get("direction") if (judge4.get("direction") and judge4.get("direction") == judge15.get("direction")) else (judge15.get("direction") or judge4.get("direction"))
                    plan_adv = calc_trade_plan(sym, d_adv or result["signal"], result["price"], result.get("brooks") or {}) if d_adv else None
                    msg = format_layers(result, judge4, judge15, is_reversal=reversal and not is_15m,
                                        events=events or None, ew=ew,
                                        ptype="反转" if reversal and not is_15m else "定时" if is_15m else "异动",
                                        plan=plan_adv, prev4=ck[1] if ck else None)
                    ok = send_telegram(msg)
                    if is_15m:
                        last_timed_slot = slot
                    tag = "定时信号" if is_15m else "🔄反转加推" if reversal else "⚡异动加推"
                    print(f"  {sym} {tag} (4h:{dirs['4h'] or '观望'} 15m:{dirs['15m'] or '观望'}, 投票{result['votes']}) {'已推送' if ok else '推送FAIL(Telegram拒收)'}")
                else:
                    print(f"4h:{dirs['4h'] or '观望'} 15m:{dirs['15m'] or '观望'} (投票{result['votes']})")

                # 快检基准: 3分钟休眠期间供10秒快检使用(2026-08-09 实时化: 检测秒级, AI判决仍触发式)
                if lv_main:
                    atr_pct = lv_main["atr14"] / result["price"] * 100
                    fast_ref[sym] = {"px": result["price"],
                                     "thr": min(max(0.5 * atr_pct, 0.2), 0.6),
                                     "hi": lv_main["swing_high"], "lo": lv_main["swing_low"],
                                     "vwap": vw_main[0] if vw_main else None}
                else:
                    fast_ref.pop(sym, None)
            except Exception as e:
                print(f"错误: {e}")

        save_state()  # 每轮全扫后落盘, 重启/部署不丢判决连续性
        if args.loop == 0:
            break
        # 睡到下一个3分钟整点, 期间每10秒快检报价: 急动/破摆动极值/穿VWAP → 立即全扫
        while True:
            now = time.localtime()
            seconds_to_next = 180 - ((now.tm_min % 3) * 60 + now.tm_sec)
            if seconds_to_next <= 10:  # 必须 >=步长粒度, 否则10秒步长永远跳不进窗口(2026-08-08 死循环事故)
                time.sleep(max(seconds_to_next, 0))
                break
            time.sleep(10)
            now_ts = time.time()
            trig = None
            tkey = None
            for sym, _ in watch:
                ref = fast_ref.get(sym)
                if not ref:
                    continue
                px = fetch_fast_price(sym)
                if px is None:
                    continue
                tkey = None
                chg = (px - ref["px"]) / ref["px"] * 100
                if abs(chg) >= ref["thr"]:
                    trig = f"{'急涨' if chg > 0 else '急跌'}{abs(chg):.1f}%(阈值{ref['thr']:.1f}%)"
                    tkey = "up" if chg > 0 else "dn"
                elif ref["hi"] and ref["px"] <= ref["hi"] < px:  # 边沿: 基准价还在下方才算突破(2026-08-10 刷屏事故)
                    trig = f"突破摆动高点${ref['hi']:.2f}"
                    tkey = "hi"
                elif ref["lo"] and px < ref["lo"] <= ref["px"]:
                    trig = f"跌破摆动低点${ref['lo']:.2f}"
                    tkey = "lo"
                elif ref["vwap"] and (px - ref["vwap"]) * (ref["px"] - ref["vwap"]) < 0:
                    trig = f"穿越VWAP(${ref['vwap']:.2f})"
                    tkey = "vwap"
                # 同向冷却15分钟: 单边行情每段新腿都触发会刷屏(2026-08-19 暴涨连推6次事故)
                if trig and tkey:
                    lk = (sym, tkey)
                    if time.time() - last_fast.get(lk, 0) < 900:
                        trig = None
                        tkey = None
                        continue
                    last_fast[lk] = time.time()
            # 新闻硬信号(2026-08-20 因果回溯): 60s轮询 RSS, 发现新新闻立即全扫(编辑席判断/推送由全扫完成)
            # 用 fast_seen 独立去重——fetch_news 会写 seen, 若直接用 news_seen 会把未解读新闻标已读, 架空编辑席
            if trig is None and now_ts - last_news_poll >= 60:
                last_news_poll = now_ts
                try:
                    raw = fetch_news(fast_seen)
                    fresh = [t for _, t in raw
                             if not any(_title_sim(_norm_title(t), _norm_title(ot)) > 0.55 for _, ot in RECENT_NEWS)]
                    if fresh:
                        lk = ("_news", "poll")
                        if time.time() - last_fast.get(lk, 0) < 120:  # 新闻触发冷却2分钟
                            trig = None
                        else:
                            last_fast[lk] = time.time()
                            trig = f"新新闻{len(fresh)}条待解读"
                            tkey = "news"
                except Exception:
                    pass
            if trig:
                print(f"\n  ⚡ {'NEWS' if tkey == 'news' else sym} 快检{trig}, 立即全扫")
                pending_fast = f"{'NEWS' if tkey == 'news' else sym} {trig}"
                break


if __name__ == "__main__":
    main()
