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
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vt_predictions.json")
JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_journal.json")

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
    gran = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(interval)
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


def fetch_klines(symbol, interval="15m", limit=200, start_ms=None, drop_incomplete=True):
    """binance.us → Coinbase → fapi → yfinance (fapi 在美国被 geo-block, Coinbase 兜底);
    drop_incomplete=False 保留未收盘K线(历史结算用); 数据停滞的源自动跳过"""
    if drop_incomplete:
        hit = _kline_cache.get((symbol, interval, limit))
        if hit and time.time() - hit[0] < 60:
            return hit[1]
    extra = f"&startTime={start_ms}" if start_ms else ""
    df = None
    stale_df = None
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
        "price": float(df["close"].iloc[-1]),
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
    """同步发送, 10s 超时"""
    try:
        r = _http.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                       data={"chat_id": TELEGRAM_CHAT_ID, "text": _escape_html(text),
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


def record_judge(result, plan, judge):
    """裁判决策落盘(放行/否决都记), 供事后虚拟结算和判例记忆; 上限200条"""
    ba = result.get("brooks") or {}
    lv = compute_levels(result["symbol"], result["signal"])
    j = load_journal()
    j["entries"].append({
        "time": pd.Timestamp.now().isoformat(),
        "symbol": result["symbol"], "direction": result["signal"],
        "entry_px": plan["entry"], "sl": plan["sl"], "tp2": plan["tp2"],
        "verdict": judge["verdict"], "confidence": judge["confidence"],
        "ai_direction": judge.get("direction"),  # AI 实际选择(LONG/SHORT/None), direction 字段是投票假定方向
        "reasons": judge.get("reasons", []),
        "rsi": round(lv["rsi14"]) if lv else None,
        "state": ba.get("state"), "votes": result["votes"],
        "outcome": None, "judgment": None,
    })
    if len(j["entries"]) > 200:
        j["entries"] = j["entries"][-200:]
    save_journal(j)


def verify_journal():
    """判例虚拟结算: 按 entry 的 sl/tp2 走K线, 判裁判对错; <24h 未触发不结算"""
    j = load_journal()
    now = pd.Timestamp.now(tz="UTC")
    changed = False
    for e in j["entries"]:
        if e.get("outcome") is not None:
            continue
        et = pd.Timestamp(e["time"])
        if et.tz is None:
            et = et.tz_localize("UTC")
        age = (now - et).total_seconds()
        if age < 1800:
            continue
        try:
            df = fetch_klines(e["symbol"], "15m", 1000,
                              start_ms=int(et.timestamp() * 1000), drop_incomplete=False)
            if df.empty:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            bars = df[df.index >= et]
            if len(bars) < 2:
                continue
        except Exception:
            continue

        sl, tp2 = float(e["sl"]), float(e["tp2"])
        outcome = "timeout"
        for _, bar in bars.iterrows():
            try:
                hi, lo = float(bar["high"]), float(bar["low"])
            except Exception:
                continue
            if e["direction"] == "LONG":
                if lo <= sl:
                    outcome = "loss"; break
                if hi >= tp2:
                    outcome = "win"; break
            else:
                if hi >= sl:
                    outcome = "loss"; break
                if lo <= tp2:
                    outcome = "win"; break
        if outcome == "timeout" and age < 86400:
            continue
        e["outcome"] = outcome
        # 执行+win=对, 执行+loss=错, 观望+loss=对(躲过), 观望+win=错(错过); timeout 不判定
        e["judgment"] = None if outcome == "timeout" else (e["verdict"] == "执行") == (outcome == "win")
        changed = True
    if changed:
        save_journal(j)


def judge_stats():
    """判对率 (wins, total): 忽略 timeout(judgment 为 null)"""
    judged = [e for e in load_journal()["entries"] if e.get("judgment") is not None]
    return sum(1 for e in judged if e["judgment"]), len(judged)


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
    """15分钟紧凑快报(无门槛): AI判断打头, 带止盈止损, 数据一行汇总"""
    sym = result["symbol"]
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    sig = result["signal"]
    vn = int(result["votes"].split("/")[0])
    total = result["votes"].split("/")[1]
    reached = sig != "NEUTRAL" and vn >= MIN_VOTES

    L = []
    L.append("=" * 36)
    L.append(f"📡 {sym} 快报 | {pd.Timestamp.now():%m-%d %H:%M} | 💰现价 ${result['price']:.2f}")

    # AI 判断是主角, 放最前
    judge = judge or {}
    if judge.get("confidence", -1) >= 0:
        j_emoji = "✅" if judge["verdict"] == "执行" else "🛑"
        L.append(f"🤖 AI判断: {j_emoji}{judge['verdict']} (置信度{judge['confidence']})")
        for i, rsn in enumerate((judge.get("reasons") or [])[:2]):
            L.append(f"📝 理由{i+1}: {rsn}")
    else:
        L.append("🤖 AI判断: 裁判不可用")

    # 交易计划(止盈止损) — NEUTRAL 时按票多一方的假定方向
    if plan:
        dir_cn = {"LONG": "做多", "SHORT": "做空"}.get(sig, "多空打平")
        d_emoji = "🔺" if sig == "LONG" else "🔻" if sig == "SHORT" else "⏸"
        L.append(f"🧭 方向: {d_emoji}{dir_cn}")
        L.append(f"🎯 入场 ${plan['entry']:.2f} | 🚫 止损 ${plan['sl']:.2f}")
        L.append(f"🥇 止盈1 ${plan['tp1']:.2f} | 🏆 止盈2 ${plan['tp2']:.2f} (盈亏比1:{plan['rr']:.1f})")

    # 数据汇总一行
    parts = [f"看涨{result['bullish']}票/看跌{result['bearish']}票", state_cn]
    lv = compute_levels(sym, sig if sig != "NEUTRAL" else ("LONG" if result["bullish"] >= result["bearish"] else "SHORT"))
    if lv:
        vol_short = "放量" if lv["vol_ratio"] > 1.5 else "正常" if lv["vol_ratio"] > 0.8 else "缩量"
        parts.append(f"RSI{lv['rsi14']:.0f}")
        parts.append(f"{vol_short}{lv['vol_ratio']:.1f}倍")
    st = fetch_sentiment(sym) or {}
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        parts.append(f"费率{fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    if "taker_buy_sell_ratio" in st:
        parts.append(f"买卖比{st['taker_buy_sell_ratio']:.2f}")
    L.append("📊 数据: " + " | ".join(parts))
    L.append(f"🚦 信号线: {'✅已达' + str(MIN_VOTES) + '票' if reached else '⏸未触发'}")
    return "\n".join(L)


def format_signal(result, plan, judge, sig_num, ai_decision=True, is_emergency=False):
    """AI 主判完整信号紧凑卡片(与快报同风格)"""
    sym = result["symbol"]; sig = result["signal"]
    ba = result.get("brooks") or {}
    state_cn = {"trend_up": "上升趋势", "trend_down": "下降趋势", "range": "震荡区间"}.get(ba.get("state"), "未知")
    dir_cn = "做多" if sig == "LONG" else "做空"
    emoji = "🟢" if sig == "LONG" else "🔴"

    L = []
    L.append("=" * 36)
    tag = "🔄紧急翻转 " if is_emergency else ""
    L.append(f"{emoji} {sym} {tag}AI信号 {dir_cn} #{sig_num} | {pd.Timestamp.now():%m-%d %H:%M}")
    L.append(f"💰 现价 ${result['price']:.2f}")

    judge = judge or {}
    if judge.get("confidence", -1) >= 0:
        L.append(f"🤖 AI判断: ✅{judge['verdict']} (置信度{judge['confidence']})")
        for i, rsn in enumerate((judge.get("reasons") or [])[:2]):
            L.append(f"📝 理由{i+1}: {rsn}")
    else:
        L.append("🤖 AI判断: 裁判不可用")

    arrow = "🔺" if sig == "LONG" else "🔻"
    L.append(f"🧭 方向: {arrow}{dir_cn}")
    sl_label = plan.get("sl_label", "").split("(")[0] or "止损"
    L.append(f"🎯 入场 ${plan['entry']:.2f} | 🚫 止损 ${plan['sl']:.2f} ({sl_label})")
    L.append(f"🥇 止盈1 ${plan['tp1']:.2f} | 🏆 止盈2 ${plan['tp2']:.2f} (盈亏比1:{plan['rr']:.1f})")

    # 数据汇总一行 (同 format_update)
    parts = [f"看涨{result['bullish']}票/看跌{result['bearish']}票", state_cn]
    lv = compute_levels(sym, sig)
    if lv:
        parts.append(f"RSI{lv['rsi14']:.0f}")
    st = fetch_sentiment(sym) or {}
    if "funding_rate" in st:
        fr = st["funding_rate"] * 100
        parts.append(f"费率{fr:.3f}%({'多付空' if fr >= 0 else '空付多'})")
    if "taker_buy_sell_ratio" in st:
        parts.append(f"买卖比{st['taker_buy_sell_ratio']:.2f}")
    L.append("📊 数据: " + " | ".join(parts))

    stt = judge.get("stats") or {}
    if stt.get("total", 0) >= 5:
        L.append(f"📈 判对率: {round(stt['wins'] / stt['total'] * 100)}% ({stt['wins']}/{stt['total']})")
    return "\n".join(L)


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
            support = f"${swing_low:.2f}(近20周期低点)、${ema50:.2f}(EMA50均线)"
        else:
            resistance = f"${recent_high:.2f}(近5周期高点)、${swing_high:.2f}(近20周期高点)"
            support = f"${swing_low:.2f}(近20周期低点)"
        return {"support": support, "resistance": resistance, "rsi14": rsi14,
                "atr14": atr14, "ema20": ema20, "ema50": ema50,
                "vol_ratio": vol_ratio, "vol_word": vol_word}
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


def fetch_sentiment(symbol):
    """合约情绪: Binance 优先 → Kraken 补缺; 主动买卖比用K线 taker 量算(现货源也有, 服务器可用); 全空返回 None"""
    s = _sentiment_binance(symbol) or {}
    if len(s) < 5:  # 有字段缺失才打 Kraken, 省一次请求
        for key, v in (_sentiment_kraken(symbol) or {}).items():
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


def build_market_brief(result, plan):
    """结构化市场简报(纯文本): 客观陈列数据, 不预设方向, 由 AI 独立判断"""
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
    L.append(f"市场状态: {state_cn} | Always In: {ai_cn} | Spike: {spike_cn}")
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

    # 合约情绪: 缺失字段直接省略
    st = fetch_sentiment(sym)
    if st:
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
        if parts:
            L.append("合约情绪: " + " | ".join(parts))

    L.append(f"参考交易计划(按投票方向试算): 入场 ${plan['entry']:.2f} | 止损 ${plan['sl']:.2f} | "
             f"TP1 ${plan['tp1']:.2f} | TP2 ${plan['tp2']:.2f} | 盈亏比 1:{plan['rr']:.1f}")
    return "\n".join(L)


JUDGE_UNAVAILABLE = {"verdict": "执行", "direction": None, "confidence": -1, "reasons": ["裁判不可用，按规则放行"]}


def ai_judge(result, plan):
    """LLM 风控裁判: 对候选信号放行/否决; 任何异常都放行, 不阻塞信号流"""
    brief = build_market_brief(result, plan)
    wins, total = judge_stats()

    # 判例记忆: 最近12条已判定案例附在简报后, 让裁判总结自己的教训
    if total:
        state_map = {"trend_up": "趋势上升", "trend_down": "趋势下降", "range": "震荡"}
        cases = [e for e in load_journal()["entries"] if e.get("judgment") is not None][-12:]
        lines = [f"你的近期判例(判对率 {wins}/{total}={round(wins / total * 100)}%):"]
        for e in cases:
            et = pd.Timestamp(e["time"]).strftime("%m-%d %H:%M")
            oc = "止盈" if e["outcome"] == "win" else "止损"
            rsi = e.get("rsi") if e.get("rsi") is not None else "-"
            ai_dir = {"LONG": "做多", "SHORT": "做空"}.get(e.get("ai_direction"), e["verdict"])
            lines.append(f"- {et} {'做多' if e['direction'] == 'LONG' else '做空'}@{e['entry_px']:.1f} "
                         f"RSI={rsi} {state_map.get(e.get('state'), '未知')} {e.get('votes', '-')}票 → "
                         f"你判{ai_dir}, 实际{oc}, {'判对' if e['judgment'] else '判错'}")
        brief += "\n\n" + "\n".join(lines)

    system = ("你是交易员，根据市场数据独立判断方向，遵循 Al Brooks 价格行为学原则。"
              "可以做多、做空或观望，不要被投票分布锚定，投票只是参考数据之一。"
              "只输出 JSON: {\"direction\": \"做多\" 或 \"做空\" 或 \"观望\", \"confidence\": 0-100 整数, "
              "\"reasons\": [2-3条理由]}。"
              "每条理由必须引用简报里的具体数字(价格/RSI/量比/票数/ATR)，只陈述数据和事实关系，"
              "禁止比喻，禁止\"可能/随时/容易/大概率/感觉\"等主观推测词，每条不超过30字。"
              "简报含合约情绪数据(资金费率/持仓量变化/多空账户比/主动买卖比)，评估时必须考虑拥挤度和资金动向。"
              "附带的判例是你自己的历史决策及结果，判错的案例要总结教训，避免重蹈覆辙。")
    try:
        r = _http.post(DS_API_URL, json={
            "model": DS_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": brief}],
            "max_tokens": 150, "temperature": 0.2},
            headers={"Authorization": f"Bearer {DS_API_KEY}"}, timeout=20)
        if r.status_code != 200:
            out = dict(JUDGE_UNAVAILABLE)
            out["stats"] = {"wins": wins, "total": total}
            return out
        text = r.json()["choices"][0]["message"]["content"]
        # 容忍 ```json 围栏和前后多余文字, 提取第一个 {...} 块
        m = re.search(r"\{[^{}]*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
        # 方向制: 含"多"→LONG, 含"空"→SHORT, 否则观望
        d_str = str(data.get("direction", ""))
        direction = "LONG" if "多" in d_str else "SHORT" if "空" in d_str else None
        confidence = int(data.get("confidence", -1))
        reasons = data.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            reasons = [str(data.get("reason", "")).strip() or "无理由"]
        reasons = [str(x).strip() for x in reasons[:3]]
        return {"verdict": "执行" if direction else "观望", "direction": direction,
                "confidence": confidence, "reasons": reasons,
                "stats": {"wins": wins, "total": total}}
    except Exception:
        out = dict(JUDGE_UNAVAILABLE)
        out["stats"] = {"wins": wins, "total": total}
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=15, help="循环间隔(分钟), 0=单次")
    parser.add_argument("--test", action="store_true", help="仅打印, 不推送")
    parser.add_argument("--symbols", default="ETHUSDC", help="逗号分隔, 如 ETHUSDC,BTCUSDC")
    parser.add_argument("--judge-test", action="store_true", help="裁判调试: 打印简报+裁判JSON后退出")
    args = parser.parse_args()

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

    print(f"VT投票信号机器人 v2.1 | VT8+NOFX4+Brooks6=18票 | ≥{MIN_VOTES}票触发 ≥{STRONG}强 | 每{args.loop}分钟")
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
        return

    last_15m_signal = {}  # sym → (dir, time, px)
    last_emergency = {}   # sym → timestamp
    reversal_count = {}   # sym → {"dir": str, "count": int}

    while True:
        now = pd.Timestamp.now()
        minute = now.minute
        is_15m = minute % 15 == 0  # :00 :15 :30 :45

        if is_15m:
            print(f"\n[{now.strftime('%H:%M:%S')}] 15分钟定时扫描...")
        else:
            print(f"\n[{now.strftime('%H:%M:%S')}] 5分钟监控...")

        # ── 验证上次预测 (合并为一条推送, 避免刷屏) ──
        try:
            followups = verify_predictions()
            if followups:
                wins = sum(1 for m in followups if "止盈" in m)
                body = "\n".join(followups)
                if len(body) > 3500:
                    body = body[:3500] + "\n...(过长截断)"
                ok = send_telegram(f"📋 本批结算 {len(followups)}条: {wins}胜{len(followups)-wins}负\n{body}")
                print(f"  验证: 合并推送{len(followups)}条 {'OK' if ok else 'FAIL(Telegram拒收)'}")
        except Exception as e:
            print(f"  验证失败: {e}")

        # ── 裁判判例虚拟结算 ──
        try:
            verify_journal()
        except Exception as e:
            print(f"  判例结算失败: {e}")

        for sym, config in watch:
            try:
                print(f"  {sym}...", end=" ", flush=True)
                result = vote(sym, config)
                vn = int(result["votes"].split("/")[0])
                qualified = result["signal"] != "NEUTRAL" and vn >= MIN_VOTES

                # ── 15分钟定时: AI 主决策, 每次出判例; AI 定向且高置信才发完整信号 ──
                if is_15m:
                    # NEUTRAL 时用票多一方作假定方向算参考 plan/shadow 判例
                    sig0 = result["signal"] if result["signal"] != "NEUTRAL" else (
                        "LONG" if result["bullish"] >= result["bearish"] else "SHORT")
                    r2 = result if result["signal"] != "NEUTRAL" else dict(result, signal=sig0)
                    plan = calc_trade_plan(sym, sig0, result["price"], result.get("brooks") or {})
                    judge = ai_judge(r2, plan)
                    record_judge(r2, plan, judge)  # 每次15m决策都记(观望也记, shadow按投票假定方向结算)
                    ai_dir = judge.get("direction")
                    if ai_dir and judge["confidence"] >= 60:
                        # AI 定向: 方向与投票假定不同则按 AI 方向重算 plan
                        if ai_dir != sig0:
                            plan = calc_trade_plan(sym, ai_dir, result["price"], result.get("brooks") or {})
                        r3 = dict(result, signal=ai_dir)
                        sig_num = get_signal_number()
                        msg = format_signal(r3, plan, judge, sig_num)
                        # 先发文字, 再发图
                        ok = send_telegram(msg)
                        if not args.test:
                            img = make_chart(r3, plan)
                            if img:
                                send_photo(img, format_caption(r3, plan))
                        save_prediction(sym, ai_dir, result["price"], plan)
                        last_15m_signal[sym] = (ai_dir, time.time(), result["price"])
                        print(f"  {sym} → AI决策 {ai_dir} (置信{judge['confidence']}, 投票{result['votes']}) {'已推送' if ok else '推送FAIL(Telegram拒收)'}")
                        continue
                    # AI 观望/低置信/不可用 → 紧凑快报 (不算信号, 不更新 last_15m_signal)
                    send_telegram(format_update(result, judge, plan))
                    print(f"  {sym} 快报已推送 ({result['signal']} {result['votes']}票)")
                    continue

                if qualified:
                    # 反转确认: 非15分钟扫描, 需要连续3次同向才触发紧急推送
                    prev = last_15m_signal.get(sym)
                    if not prev or prev[0] == result["signal"]:
                        reversal_count.pop(sym, None)
                        print(f"方向未变跳过")
                        continue
                    # 新方向, 累积确认计数
                    rc = reversal_count.get(sym, {"dir": result["signal"], "count": 0})
                    if rc["dir"] != result["signal"]:
                        rc = {"dir": result["signal"], "count": 0}
                    rc["count"] += 1
                    reversal_count[sym] = rc
                    if rc["count"] < 3:
                        print(f"反转确认 {rc['count']}/3 跳过")
                        continue
                    reversal_count[sym] = {"dir": result["signal"], "count": 0}
                    now_ts = time.time()
                    if sym in last_emergency and now_ts - last_emergency[sym] < 900:
                        print(f"紧急冷却跳过")
                        continue
                    last_emergency[sym] = now_ts

                    plan = calc_trade_plan(sym, result["signal"], result["price"], result.get("brooks") or {})
                    # AI 裁判: 放行才推送, 价格决策仍由 Brooks 规则定
                    judge = ai_judge(result, plan)
                    record_judge(result, plan, judge)  # 放行/否决都落盘, 事后结算判对错
                    if judge["verdict"] == "观望" or (judge["confidence"] != -1 and judge["confidence"] < 60):
                        print(f"AI裁判否决: {'; '.join(judge['reasons'])}")
                        continue
                    sig_num = get_signal_number()
                    msg = format_signal(result, plan, judge, sig_num, is_emergency=True)
                    if msg:
                        # 先发文字, 再发图
                        ok = send_telegram(msg)
                        if not args.test:
                            img = make_chart(result, plan)
                            if img:
                                send_photo(img, format_caption(result, plan))
                        save_prediction(sym, result["signal"], result["price"], plan)
                        print(f"  {sym} → {result['signal']} {result['votes']}票 🔄翻转{'已推送' if ok else '推送FAIL(Telegram拒收)'}")
                    else:
                        print(f"  {sym} {result['signal']} {result['votes']}票 (无消息)")
                else:
                    print(f"  {sym} {'NEUTRAL' if result['signal'] == 'NEUTRAL' else '票数不足'} {result['votes']}")
            except Exception as e:
                print(f"错误: {e}")

        if args.loop == 0:
            break
        # 睡到下一个3分钟整点 (每15分钟扫5次)
        now = time.localtime()
        seconds_to_next = 180 - ((now.tm_min % 3) * 60 + now.tm_sec)
        if seconds_to_next < 3:
            seconds_to_next += 180
        time.sleep(seconds_to_next)


if __name__ == "__main__":
    main()
