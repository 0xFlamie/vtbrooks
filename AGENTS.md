# vtbrooks — VT 投票 + AI 主决策信号机器人

## 架构（v3.2, 2026-08-05）

单文件 `vt_vote_bot.py`，systemd 服务 `vtbrooks`（服务器 ccvps, `/opt/vtbrooks`）。

定位：**信号是用户主观交易的趋同性暗示，不是自动开单指令**（用户自管仓位和止损）。

决策流（每 3 分钟一轮）：

1. `vote()` 计算 18 票 15m 因子（VT8 + NOFX4 + Brooks6）+ 15m Brooks 形态
2. `build_market_brief()` = 4h 研判（趋势排列/Brooks4h/波动率挤压分位/4h ATR/**威科夫 TR+Spring/Upthrust/JOC**）+ 15m 数据 + 合约情绪
3. `ai_judge()` 喂给 DeepSeek，返回 方向 + **预期幅度档（<1%/1-2%/2-3%/3%+）** + 置信度 + 理由。**只依据当前简报，历史胜率/判例/错题本不注入 prompt（用户决策：历史胜率不能作为推方向的依据）**
4. 推送门槛：AI 定向 + 置信 ≥60 + 方向有变化 + 幅度档非 <1% + rr ≥1.5；否则仅 15 分钟整点发紧凑快报。推送卡含 4h 研判行和幅度档标签（🔵轻仓1-2% / 🟣标准2-3% / 🟠主攻3%+）
5. 每次判决落盘 `judge_journal.json`（同向 1 小时去重，上限 500 条）

结算（`verify_journal`）三指标分离：

- **方向分**（评 AI 方向）：`(+2h/+4h收盘 - 入场) / ATR14 × 方向符号`；≥+0.5 ATR 判方向正确
- **双向 MFE(+12h)**（评幅度档）：`mfe_long_12h/mfe_short_12h`，AI 方向侧 MFE ≥ 档位下限 → `tier_correct`
- **outcome**（评交易计划）：先碰 SL/TP2，衡量止损止盈摆放质量，不影响 AI 判对率

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
