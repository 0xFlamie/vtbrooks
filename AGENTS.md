# vtbrooks — VT 投票 + AI 主决策信号机器人

## 架构（v4.0, 2026-08-07）

单文件 `vt_vote_bot.py`，systemd 服务 `vtbrooks`（服务器 ccvps, `/opt/vtbrooks`）。

定位：**信号是用户主观交易的趋同性暗示，不是自动开单指令**（用户自管仓位和止损）。

双层 AI 架构（两层数据隔离，各自判决，互不参考）：

| 层 | 数据 | AI 输出 | 回答 |
|---|---|---|---|
| 4h 层 | 趋势排列(EMA7/25/99)/Brooks4h/挤压分位/威科夫 TR/合约情绪 | 方向 + 幅度档(<1%/1-2%/2-3%/3%+) + 置信 | 有没有大行情，往哪边，多大 |
| 15m 层 | 18因子投票/15m形态/关键位/RSI/量比/VWAP/合约情绪 | 方向 + 置信 | 现在是不是入场点 |

判决频率：**4h 层每根 4h 收线重判一次（缓存，`judge4_cache`），15m 层每 3 分钟判**——4h 输入在根内不变，高频重判只会制造抖动（2026-08-06 反手抖动事故的根因）。

4h 判决迟滞（2026-08-08）：重判时把上一根判决（方向/置信/幅度档）注入简报末尾，SYS_4H 要求"证据没有明显变化就维持原方向，翻转必须写清新证据"——临界区证据接近 50/50 时，AI 曾因一根新 K 线 180° 改向且置信度不变。

推送（无快报，全部是信号卡 `format_layers`）：

- **定时信号**：:00/:15/:30/:45 各推一条
- **反转加推**：3 分钟扫描中任一层方向与上次不同 → 立即加推
- **异动加推**：RSI(15m+4h) 穿越 70/30、量比突破 2、逼近压力/支撑（<0.3%）、挤压进出 <20% 区、入场窗口开启（边沿触发）
- **异动由 AI 解读（2026-08-08）**：异动检测保持规则化（`detect_events`，确定性），但不再只配死文案——检测结果注入 15m 层简报，SYS_15M 要求结合走势解读（同样的 RSI 超买，趋势里=动能、区间顶缩量=回落前兆），解读体现在 15m 层 📝 理由；卡片 ⚡ 行仍列规则原文。15m 层也注入上次判决做迟滞（`judge15_prev`）
- **入场窗口（`entry_window`，纯规则不经 AI）**：4h 有方向为前提，A 回调=贴近结构位（≤0.5×ATR15m）+RSI 重置/形态扳机，B 突破=收破摆动极值+放量>1.5 倍；输出入场区/失效位/距离%
- 卡片含 💬 白话段（规则生成，概率只引 `market_stats` 多年统计库）、双层 emoji 方向、✅共振/⚠️背离标注
- 卡片含双层 emoji 方向、幅度档、因子看涨百分比、✅层级共振/⚠️层级背离标注
- 历史胜率/判例/错题本不注入 prompt（用户决策：历史胜率不能作为推方向的依据）

判例本（`record_judge`，双层分开记 dir4h/conf4h/dir15m/conf15m，双层同向 1h 去重，上限 500）。

结算（`verify_journal`）三指标分离：

- **方向分**（评方向判断）：`(+2h/+4h收盘 - 入场) / ATR14 × 方向符号`；≥+0.5 ATR 判方向正确
- **双向 MFE(+12h)**（评4h层幅度档）：AI 方向侧 MFE ≥ 档位下限 → `tier_correct`
- **outcome**（评交易计划）：先碰 SL/TP2，不影响 AI 判对率

错题本 `judge_lessons.json` 继续每 50 单生成，仅作复盘档案，不喂回 AI。

## 数据源（服务器为美区 IP，注意 geo-block）

| 数据 | 主源 | 兜底 |
|---|---|---|
| K线(USDC对) | Coinbase ETH-USD | fapi → yfinance |
| K线(USDT对) | api.binance.us | Coinbase → fapi → yfinance |
| 合约情绪 | Binance fapi（服务器 451 被封） | **OKX**（免 key 美区可用：费率/持仓/持仓1h/多空比）→ Kraken → Hyperliquid |
| 裁判 | DeepSeek API（`VT_DS_API_KEY`） | 失败→观望(fail-closed)+WARN 日志 |

已知坑：

- **Binance.US 的 USDC 对是死盘**（ETHUSDC 24h 成交个位数 ETH，最新价偏离数美元），USDC 对必须跳过，见 `fetch_klines`
- fapi.binance.com 美区 451；Bybit 美区 CloudFront 封；OKX/Hyperliquid/Kraken/Gate 美区可用
- CoinAnk 有付费 API（多空比/清算地图等 80 接口），未接入，免费方案已够用

## 运行环境

- 环境变量：`.env`（systemd EnvironmentFile），含 `VT_DS_API_KEY`、Telegram token/chat —— 永不入库
- 运行时文件（未跟踪）：`judge_journal.json` / `judge_lessons.json` / `vt_predictions.json` / `oi_snapshots.json`（HL 持仓快照兜底用）

## 部署链（强制）

本机验证（py_compile + 单测）→ commit → push → 服务器 `git pull --ff-only` → `systemctl restart vtbrooks`。
禁止 scp 直推、禁止服务器上直接改代码。回滚：`git reset --hard <sha>` + restart。

## 监控

```bash
ssh ccvps 'journalctl -u vtbrooks -f'                 # 实时日志
ssh ccvps 'journalctl -u vtbrooks --since today | grep WARN'  # 裁判失败记录
```

复盘统计：`judge_journal.json` 的 verdict/outcome/judgment 字段；`vt_predictions.json` 的 win_rate。
