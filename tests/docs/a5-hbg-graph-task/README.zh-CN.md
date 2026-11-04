# A5 HBG Graph task 验证与性能采集

架构、协议和实施细节见[中文设计](../../../docs/zh-cn/design/a5-hbg-graph-task.md)。
当前实现使用 task 粒度调度，默认 T/H8；独立 block 原型仅保留为
[后续优化候选](../../../docs/zh-cn/design/a5-hbg-graph-task/block-ready-scheduling.md)。

## 当前验证结果

PR #2864 已重放主线 `313499d61`，并修复 PARAM 视图绑定的完整范围检查。
下述完整性能测量来自此前基于 `93f1fc98a` 的 `dbf39563c`，不能作为新主线和审查修复后的实测结果。

- 审查修复后 473 项 Python 单测、8 个 C++ 目标和 pre-commit 通过。
  新增回归先证明旧代码接受越界视图，再验证精确边界、偏移越界、非法 dtype 和 extent 溢出。
- 此前重放 `93f1fc98a` 后，1263 项 Python 单测、8 个 C++ 目标、pre-commit 与 MkDocs 构建通过。
- 此前 Native Graph caller-stream 专项：54 项 CPU、2 项 A5 上板通过。
- 审查修复后的 A5 定向 Graph 测试 4 个目标、caller-stream 3 项、native Graph 故障隔离 2 项通过；
  Qwen 完整 out/K/V 通过，Device 中位数 38.744 ms、p95 38.985 ms，见[最新 Qwen 复测](../../../docs/zh-cn/design/a5-hbg-graph-task/qwen-performance.md#param-范围检查修复后的复测)。
- 测试修正提交 `328674ac0` 的 [CI](https://github.com/hw-native-sys/simpler/actions/runs/41688242174) 成功。
  该运行记录对应当时提交；后续文档和归档清理的检查以 PR 当前 head 为准。
- `dbf39563c` 完整性能集为 19 项无 Graph 加 A5 Qwen，共 20 项，所有 golden 通过。
  无 Graph 每项先验证功能，再采集三个独立进程、每进程 100 稳态轮；
  Qwen 验证完整 out/K/V 后，独立采集 5 轮预热和 100 稳态轮。
- `dbf39563c` 的 H8 Qwen Device 中位数 **38.433 ms**、p95 **38.671 ms**；同卡隔离 T/G 为 **47.195 ms**。
  详细条件和选型边界见[Qwen 记录](../../../docs/zh-cn/design/a5-hbg-graph-task/qwen-performance.md)。
- 性能门禁尚未通过：[完整无 Graph 对照](../../../docs/zh-cn/design/a5-hbg-graph-task/no-graph-performance.md)
  中有 7 项 Device 中位数比历史主线高出 5% 以上；Qwen 比历史主线 29.478 ms 慢 30.38%。
  历史主线为 `8f42cef9b3f0ebbda800e239aab0949e95552f25`，设备与依赖版本差异已在报告说明。

## 可复用的采集输入

| 文件 | 用途 |
| ---- | ---- |
| [collect.py](../../../tools/benchmarks/a5_hbg_graph_task/collect.py) | 枚举场景、采集 golden 与计时、校验来源并汇总 |
| [fixed-cases.json](../../../tools/benchmarks/a5_hbg_graph_task/fixed-cases.json) | `benchmark_rounds.sh` 中 7 项无 Graph 用例 |
| [selected-st-cases.json](../../../tools/benchmarks/a5_hbg_graph_task/selected-st-cases.json) | 11 项具有调度性能价值的 ST 用例 |
| [csa-cases.json](../../../tools/benchmarks/a5_hbg_graph_task/csa-cases.json) | CSA 补充用例 |

清单仅包含场景选择、参数和筛选理由，不包含实测耗时。
Qwen 单独测量；`spmd_paged_attention` 按本次范围约定排除。
A2A3 的 Qwen 和 DeepSeek kernel 不适配 A5，不能作为本轮 A5 验证负载。

`collect.py inventory` 使用 pytest 的真实收集流程枚举 SceneTestCase 与 case；
`run` 只运行 manifest 中 `measure=true` 的场景，先验证 golden，再采集稳态轮。
每个 pytest 子进程在执行前校验实际选中的唯一 case、params 和 config 与清单一致；
清单过期时拒绝执行，避免把旧参数写入新负载的结果。输出目录可在仓库之外。
原始 STRACE 由仓库解析器读取，轮数不符或 Host/Device 缺失的记录无效。
统计排除显式 warmup，保留中位数、10% 截尾均值、p95 和 MAD。
`summarize` 检查 case 完整性、参数、golden、三批有效记录、每批至少 100 个稳态轮，以及运行库和设备一致性。
运行前后均验证运行库哈希，不能用外层 Git HEAD 代替冻结运行库身份。

上板遵循[架构预检](../../../.claude/skills/onboard-arch-precheck/SKILL.md)和设备锁规则，
整个 sweep 由一个 `task-submit` allocation 持锁。使用项目本地 venv，并设置本次独立的 `ASCEND_PROCESS_LOG_PATH`。
不得跳过 golden、伪造版本身份，或把历史候选的部分覆盖拼成当前版本的全量通过记录。

## 本地证据与提交边界

逐轮结果、初筛 inventory、实验候选数据、设备日志、源码和二进制 manifest 存放在
`outputs/a5-hbg-graph-task/`，不纳入 Git。文档目录仅保留 Markdown；可复用脚本和输入清单位于 `tools/benchmarks/a5_hbg_graph_task/`。

- 当前完整性能报告：`outputs/a5-hbg-graph-task/pr2864-full-perf-20261104/report.zh-CN.md`。
- 当前逐轮数据及比较：同目录 `comparison.json`。
- 审查修复后的 Qwen 复测：`outputs/a5-hbg-graph-task/pr2864-review-fixes/qwen/comparison.json`。
- 历史测量摘要：`outputs/a5-hbg-graph-task/evidence-summary/`；设计文档中保留的历史 JSON 文件名均指该本地归档。
- CI 修复与日志：`outputs/a5-hbg-graph-task/pr2864-ci/`。
- 长任务检查点：`outputs/a5-hbg-graph-task/checkpoint.md`，恢复规则见[实施约定](../../../docs/zh-cn/design/a5-hbg-graph-task/implementation.md#长任务的检查点与恢复)。

这些本地产物不随仓库克隆分发；缺少冻结产物时不能宣称重现了历史阶段。
原始结果保留采集时的 revision、设备和哈希，不改写成合并后的提交号。
开发阶段按候选名称组织，不要求主线保留各阶段独立 commit。
