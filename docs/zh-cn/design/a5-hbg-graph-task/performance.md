# A5 Graph task 性能与自主迭代流程

本页是用户指定的实施策略，属于[实施门禁](implementation.md)，不是可选的后续优化建议。
**方案确定后自主推进、分析和修复，直到功能和性能验收完成；不能把一次性能不佳的结果作为任务终点。**
已完成的测量、失败与覆盖缺口见[实施记录](../../../../tests/docs/a5-hbg-graph-task/README.zh-CN.md)。
以下阈值是工程门禁，不能解释为已取得的收益。

## 迭代阶段的测量成本

按用户 2026-11-02 的补充要求，性能明显落后时不做交叉测试，也不默认重采全量主线。
复用冻结基线及其已有数据，候选先通过功能验证，再单独采集其性能，尽快反馈数值并进入下一轮优化。
只有性能接近或超过基线，或者波动明显、足以影响结论时，才对相应用例增加同卡交叉对照。
该规则同时适用于 Qwen 和无 Graph 用例；不把每个候选都自动扩展为完整主线→候选→主线。

引用历史结果时注明 commit、设备、callable、配置与采集批次，不把不同轮次拼成当轮配对。
显著差距可以直接指导迭代；接近门限的收益或退化不能仅凭历史数据定论。
最终完整 19 项及模型验收仍须完成，但选择性复测的成本应与待判断的差异相称。

## 修改前主线基线

第一处 runtime 修改之前，在当前工作区的精确主线 revision 构建并保留基线。
本次设计源码基线为 `8f42cef9b3f0ebbda800e239aab0949e95552f25`；若实施前主线更新，重新核对代码并记录实际基线 SHA。
先记录 `.gitignore`/文档差异，证明 runtime 源码与该 SHA 一致；安装项目本地 venv 中的实际构建，禁止误用旧 wheel。
不新建 worktree，不为切换基线丢弃未提交工作；预先保存基线安装产物、工具链和可重复构建信息。

用真实 A5、同一 device/die、相同频率/工具链/输入/内存配置和单进程独占锁采集。
记录 Python、CANN、PTO-ISA/PTOAS、runtime 二进制来源、DFX 设置、设备状态以及任务规模。
先完成一次不跳 golden 的功能运行，再进行预热和计时；脚本中的 `--skip-golden` 仅用于已验证配置的计时轮。

B0 包含 `tools/benchmark_rounds.sh` 的 A5 HBG 固定集合，以及
`tests/st/a5/host_build_graph` 和 `examples/a5/host_build_graph` 的逐 case 清点。
必须覆盖现有 `deepseek_v4_pro_csa/B4S2Seed0`；它是 A5 无 Graph 调度负载，属于 P1，
与停止迁移的 A2A3 DeepSeek 模型无关。首轮漏采的 CSA 必须用原冻结主线产物补采，
明确标记补采时间，不把它写成第一处 runtime 修改前已取得的结果。
先用 10 个有效稳态轮做筛选，再对最终入选项至少收集 100 个有效稳态轮、3 批独立运行。
冷启动/编译/recording 首轮单独保留，不混入稳态平均数；每轮成功与否都保留，失败不计为零耗时。

运行命令沿用仓库 [benchmark](../../../../.claude/skills/benchmark/SKILL.md)、
[perf-runtime-device](../../../../.claude/skills/perf-runtime-device/SKILL.md) 和
[onboard-arch-precheck](../../../../.claude/skills/onboard-arch-precheck/SKILL.md) 的检测、锁和指标提取约定。
其中 worktree 默认流程服从本任务“不新建 worktree”的约束。这里不给出未经设备检测即可裸跑的 onboard 命令。
执行脚本必须让 `task-submit` 持有整个 sweep 的锁，不能多个 benchmark 同时竞争同一设备。

保存可机器读取的 manifest：path、case、平台、形状、任务数、Graph 数、模式、基线 revision、
命令、是否选入性能集及理由、warmup/round、原始 marker 路径、Host/Device 时延、有效性。
逐轮结果、初筛 inventory、候选实验数据和日志存放在 `outputs/a5-hbg-graph-task/`，不纳入 Git。
可复用采集脚本和用例输入清单位于 `tools/benchmarks/a5_hbg_graph_task/`，
`tests/docs/a5-hbg-graph-task/` 仅保留 Markdown 验证摘要；
设计文档记录采样条件、结果和局限。原始证据须持久保存在本地，不能只依赖 `/tmp` 或聊天记录。

## 无 Graph 用例的筛选与门禁

采用脚本的 A5 HBG 清单，逐项核对实际 orchestration 与 scheduler mode，不能从名字推断有无 Graph：

| 主线脚本 case | 验收用途 |
| ------------- | -------- |
| `alternating_matmul_add=Case1` | 无 Graph 基本流水与跨 lane 依赖 |
| `benchmark_bgemm=Case0` | 无 Graph 的大规模 kernel/dispatch |
| `paged_attention_unroll=Case1,Case2` | 无 Graph 长链与不同问题规模 |
| `paged_attention_unroll_manual_scope=Case1,Case2` | 无 Graph scope 与依赖负载 |
| `batch_paged_attention=Case1` | 无 Graph 多任务 attention |
| `qwen3_14b_decode=GraphExecutionBatch16Seq3500` | 实际使用 `rt_submit_graph`，归入 Graph 模型验收，不计入无 Graph 统计 |

脚本之外另有必测项 `deepseek_v4_pro_csa=B4S2Seed0`，保留原 54 个 kernel、79 个执行任务、
2308 条物理执行与全部七项输出验证。冻结数据缺失、版本不匹配或 golden 被跳过都不能算通过。
最终 P1 全量清单为脚本固定 7 项、CSA 1 项、实测筛选的 11 项 ST；
重点回归子集仅用于优化迭代，不能替代这 19 项在同一候选上的全量验收。
每次总览逐项列出基线、候选、版本、Host/Device/bind 和测量状态；未复测写明待测，不静默省略。

用户明确排除 `spmd_paged_attention`（用户称 `spmd_page_attention`）作为调度性能门禁，
因该负载不提供本次需要的调度覆盖；其应有的功能回归仍保留。
未来脚本若加入其他项，同样按实际 Graph/调度路径分类，不静默增删对照集合。

ST 最终入选集必须由 B0 真实结果决定，以下仅为待测候选：

| 候选目录 | 有价值的调度维度 | 排除或降级为功能用例的条件 |
| -------- | ---------------- | -------------------------- |
| `single_core_dag`、`multi_core_dag`、`single_root` | 短 kernel、依赖推进、跨核 ready | 任务过少使单轮主要测启动，且现有 case 无更大规模 |
| `fanout_imbalance` | 非均匀 fanout、completion/wake 代价 | 由人工等待而非真实调度代价支配 |
| `automatic_admission`、`early_enqueue` | ordinary/cooperative 交错、提前准备 | 专为断言注入延迟，不能形成稳定吞吐 |
| `spmd_multiblock_aic`、`spmd_mixed_waves`、`spmd_multi_fanin` | 分波次、fanin、block 分配 | kernel 过重淹没调度；只测一次 stamp |
| `spmd_cohort_with_ordinary`、`spmd_cohort_saturation` | 普通与 cohort 争用、饱和公平性 | correctness rendezvous/timeout 决定时间 |
| `single_block_mix`、`mix_coresidency` | MIX lane 协作 | 仅验证共驻且耗时由忙等决定 |

故障注入、empty_lifecycle、capacity 拒绝、失败启动清退、DFX dump、predicate correctness 等不自动列为性能门禁。
逐 case 审查 kernel 和 timing 分布；保留能稳定测量短任务、fanout、cooperative、MIX 等差异的最小代表集。
仅功能用例也记录基线运行结果与不入选原因，不能因为“慢”就排除。

比较时先看每个 case，再给总体统计；不能用大 kernel 的加权平均掩盖短任务回退。
初始门禁为：主要 Device 中位数和截尾均值均不出现超过 3% 的可复现增长，p95 不出现超过 5% 的可复现增长。
若噪声覆盖门限，增加批次、同设备 A/B/A 或交错 A/B 复测，结论应为“待确认”，不能判通过。
所有超过稳定基线噪声的系统性小回退也需分析；3% 不是忽略问题的额度。
Host bind、首任务延迟和实际峰值内存单独对照；新的失败、OOM、长尾或退出异常直接阻断，不被平均数抵消。
这些门槛由执行者落实和细化，常规调参不再询问用户，不允许为使结果通过而放宽门槛。

## A5 Qwen Graph 模型验收

按用户补充要求，采集主线基线时也必须测试现有 A5 Qwen，不能等无 Graph 门禁通过后才补基线。
主线原编译阻塞修复后，主线与候选复用同一份 A5 ChipCallable；记录 kernel/编排源码与序列化产物哈希，
先验证完整 40 层输出和 K/V cache，再采集无插桩 Host/Device/bind。基线补采与候选最终验收分别记录状态。

按用户 2026-11-02 的范围修正，本轮模型验收只使用
`examples/a5/host_build_graph/qwen3_14b_decode`。
A2A3 的 `qwen3_14b_decode` 与 `deepseek_v4_flash_decode` kernel 不适配 A5，不能作为本轮 A5 验证负载。
DeepSeek 的迁移候选、预编译或单组件试验不构成本轮验收证据，也不继续作为本轮模型移植目标。

用户于 2026-11-02 明确要求优先提供 Qwen 实测数据，无 Graph 性能不达标不阻塞这项测量。
在已有必要功能证据的候选上验证 native Graph，及时反馈主线/候选耗时、差值与采样范围；
每个独立计时批次完成后即可给出标明样本量的初步数据，完整三批完成后再形成正式汇总。
先用完整 40 层 Torch golden
检查输出、K/V cache、跨层依赖及重复运行，再测试冷启动、稳态和现有 driver 支持的 sequence 形状。
不报告未经实际运行的 batch/layer 覆盖，也不能以退出码为零代替数值检查。
Graph Definition 复用数、child 数、cooperative 数、scheduler mode 和 prepare 拒绝原因进入报告，
确认测到 AICore native Graph。

当前主对照是**同一 A5 设备、冻结主线 resident 与候选 resident**，使用相同 A5 Qwen kernel、输入与编译选项。
核实实际加载库、revision 和是否 native Graph，不能用 A2A3 数据计算本次 scheduler 加速比。
早期 legacy/native 对比属于[历史性能记录](qwen-performance-history.md)，不要求当前 runtime 保留该执行器。
B0 的 `.rodata` 构建失败及当时缺失的延迟数据仍按原始证据记录，不能补写为成功对照。

无 Graph 门禁和 A5 Qwen 的完整功能、性能验证均完成后才能验收。
候选比冻结主线慢、首 child 延迟异常或 tail 明显偏大时，进入下一节优化循环。
若计算已主导、调度改进低于噪声，则提供分解证据和多批次结果。

## 性能归因与自主优化循环

每轮按“固定复现 → 分层采集 → 单一瓶颈假设 → 修改 → 功能/协议复测 → 关闭重型诊断后采集候选”执行。
是否追加交叉对照遵循本页的测量成本规则，不能将 A/B 作为每轮默认步骤。
每次保留 revision、假设、修改范围、正确性证据、收益/代价和下一步，失败优化也留下结论并撤回。

| 观察 | 优先检查 | 可考虑的改动及约束 |
| ---- | -------- | ------------------ |
| bind 增长 | Definition validation、N-node 初始化、H2D、重复复制 | unique-definition 计划复用、减少重复扫描；保持完整拓扑验证 |
| 首 child 晚 | bootstrap、ENTER 排队、首次参数绑定、cooperative admission | 优化只读访问与控制预算；不去掉 bootstrap 屏障 |
| 短 child 间隙大 | fanin 扫描、ready steal、参数 bytes/flush | accessor 快路、按逻辑任务一次绑定；可测有界提前准备，先补所有权证明 |
| MIX/SPMD 吞吐低 | claim/commit、membership、release、wave、order head blocking | 缩短锁内工作、提高服务公平性；不拆分全局 admission 域 |
| EXIT/tail 长 | 全 sink join、control backlog、completion flush、LocalDone | 调整有界服务与真实完成批量发布；不提前释放或少等 sink |
| 内存或上传异常 | 预留 execution tail、当前 task-control stride、block×lane trace | 按真实 allocation 核算；诊断按需分配先审计所有 reader |
| kernel 本体慢 | 相同 args 与 lane/block 的单任务时间 | 用 core swimlane 区分 kernel 适配和 scheduler 问题，避免归错因 |

使用 `hbg-bind-phases`、`dfx-analyze`、chip swimlane/dep_gen；kernel 内部问题再用 core-swimlane。
先检查现有工具能否回答问题，再考虑增加诊断；不得在 AICPU 调度热路径无条件打印或引入未授权行为开关。
诊断数据和无诊断稳态数据分开采集，不能用开启重型 trace 的时间宣称性能收益。

## 最终交付证据

最终报告至少包含：基线与最终 SHA/二进制来源、完整新旧功能矩阵、无 Graph 固定集和入选 ST 的逐 case 对照、
A5 Qwen 冻结主线/候选 resident 数据、Host bind/Device/尾延迟/内存、性能迭代记录以及所有未完成或不支持项。
HBG 标准 marker 通常只有 Host/Device；缺失 Orch/Sched/Effective 时写缺失，不能当零，也不能移用 TMR 的口径。
额外阶段数据使用其工具明确的指标名，保留 raw marker 让结果可重算。

所有门禁通过后复审实现与本设计是否一致，并更新本文状态；不把“方案已写好”改写成“功能或性能已验证”。
若资源暂不可用，先完成不依赖该资源的工作、保留排队/失败证据和恢复点；不重复向用户询问常规决策，
但也不能将未执行的硬件验收标记为通过。
