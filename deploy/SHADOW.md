# 隔离影子观察部署交接

状态：配置已准备，尚未commit/push/安装或启动。必须先确认下列范围，再按Git链部署。不是主站信号接管。

## 最小提交范围（16文件）

- 运行源码8项：`brooks_event_research.py`、`brooks_failed_range.py`、`brooks_first_retest.py`、`brooks_intrabar.py`、`brooks_intrabar_controls.py`、`brooks_sequence_research.py`、`brooks_failed_range_shadow.py`、`brooks_shadow_runtime.py`。
- 测试3项：`tests/test_failed_range.py`、`tests/test_failed_range_shadow.py`、`tests/test_brooks_shadow_runtime.py`。
- 报告1项：`analysis/brooks_shadow_recovery_report.md`。
- 部署4项：本文件、`deploy/shadow-requirements.txt`、`deploy/shadow-runtime-manifest.json`、`deploy/vtbrooks-shadow.service`。

`market_path_model.py`已经在HEAD，不需重复修改；连同上述源码构成9文件运行闭包。`brooks_event_research.py`包含此前未提交的15M/窗口参数化改动，必须保留冻结哈希；不是本轮新调参。其余研究文件、`analysis/evaluate_brooks_events_v2.py`和混有大量历史未提交记录的`AGENTS.md`不纳入这次提交，留在本机。

## 环境与部署边界

- 主站基线`6c9fcb66530a780fa50e9ea73c90548f46a721bb`，不在`/opt/vtbrooks`拉取或重启任何主站服务。
- 隔离Git检出目录`/opt/vtbrooks-shadow`；独立Python虚拟环境`/opt/vtbrooks-shadow-venv`，使用已确认的服务器基础Python创建，不修改主站venv。只安装本文件夹锁定的依赖；安装/测试失败不得启动。
- 账本由systemd的DynamicUser与StateDirectory管理，位于`/var/lib/vtbrooks-shadow/ledger-v2.sqlite3`（实际可映射到private状态目录）。不使用root运行Python，不加载EnvironmentFile或用户账户密钥。
- 服务有硬校验错误时退出；仅内部网络错误重试，不使用systemd无限重启掩盖数据错误。首次运行才登记新账本，已有账本必须匹配冻结实现。
- 代码只读，状态目录可写；保留原始逐笔，不自动清理磁盘。无HTTP监听、外部通知、下单或主站信号写入。

## 正式链路

1. 本地全量unittest、编译/空白及依赖闭包核验；确认16文件范围后，只暂存这些文件并检查diff/凭证规则，再commit。不得git add全部。
2. 用户确认push后推远端；记录确切SHA。服务器从同一远端新建独立检出，核对SHA，不传未提交代码。检出必须是干净工作区。
3. 独立环境安装锁版本；核对版本、9源码哈希，运行以下三个测试。服务器上用`systemd-analyze verify`验证已经从Git取得的unit；还未执行此Linux验证，不把配置就绪当验证通过。
4. 从已提交的检出安装独立unit并启动；不重启`vtbrooks`、web或原WS服务。确认真实状态/日志、至少连续数个实时分钟和哈希账本；缺根/异常如实记录。服务运行不代表信号出现。
5. 输出SHA、北京时间、独立服务状态、账本首次实时分钟与实收数量；旧主站SHA/状态保持一致。至少30日只是最早复核时间，还要看实际覆盖及样本量。

```sh
/opt/vtbrooks-shadow-venv/bin/python -m unittest discover -s tests -p test_failed_range.py -q
/opt/vtbrooks-shadow-venv/bin/python -m unittest discover -s tests -p test_failed_range_shadow.py -q
/opt/vtbrooks-shadow-venv/bin/python -m unittest discover -s tests -p test_brooks_shadow_runtime.py -q
/opt/vtbrooks-shadow-venv/bin/python -B brooks_shadow_runtime.py --mode audit --ledger /var/lib/vtbrooks-shadow/ledger-v2.sqlite3
```

## 回滚与风险

首次部署回滚是停止并禁用`vtbrooks-shadow`，保留账本、独立代码和环境用于审计；主站原服务/SHA不变。以后若换源码必须新账本/新协议，不在旧账本下回滚换策略。没有实际部署前，不存在可宣称的影子部署SHA或启动时间。

当前每小时换会话会有采集空档；统计包含这些缺失。原始逐笔磁盘增长需监控，未做自动轮转/删除。WS与REST一致性、普通实时对照及用户到达/成交尚未验证，不能把历史约68%解释为上线后的盈利胜率。
