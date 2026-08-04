# vtbrooks — VT 投票 + AI 主决策信号机器人

## 架构（v3.0, 2026-08-04）

单文件 `vt_vote_bot.py`，systemd 服务 `vtbrooks`（服务器 ccvps, `/opt/vtbrooks`）。

决策流（每 3 分钟一轮）：

1. `vote()` 计算 18 票因子（VT8 + NOFX4 + Brooks6）+ Brooks 价格行为形态
2. `build_market_brief()` 把价格/RSI/量比/支撑压力/投票分布/合约情绪写成文本简报
3. `ai_judge()` 喂给 DeepSeek（`deepseek-chat`），返回 多/空/观望 + 置信度 + 理由
4. AI 定向 + 置信 ≥60 + 方向有变化 → 推完整信号（Telegram 文字 + 图表）；否则仅 15 分钟整点发紧凑快报
5. 每次判决都落盘 `judge_journal.json`（小本本），`verify_journal()` 事后按 SL/TP 结算判对错；每 50 单 `maybe_update_lessons()` 归纳错题本 `judge_lessons.json`，喂回给 AI

**投票只是简报里的参考数据，开单方向完全由 AI 判。**

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
