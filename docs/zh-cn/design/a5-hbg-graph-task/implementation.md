# A5 Graph task 实施与功能验收

本页定义[设计总览](../a5-hbg-graph-task.md)的实施顺序。
性能基线、用例筛选及模型优化另见[性能与自主迭代流程](performance.md)。
已开始按本页实施，当前阶段与实测证据记录在[实施记录](../../../../tests/docs/a5-hbg-graph-task/README.zh-CN.md)。
下列门禁以实际代码和测试结果为准，不能将方案描述视为已实现的能力。
[独立 block 调度](block-ready-scheduling.md)现为已撤回的后续候选；
其中无 `sync_start` 的 MIX SPMD 每个 block 独立协作的思路保留供以后评估。

## 执行约定与阶段门禁

**历史决策（2026-11-03）：独立 block 原型因编译成本撤回，当时恢复到测得 38.454 ms 的 `queue-ordinary-credit-v1`。该数值只属于当时的版本。**

当前使用全任务cooperative admission、按128B Sector隔离的共享热点、默认Host CSR/remaining、同步completion、ordinary小grid的循环成员集合及resident编译单元拆分。Scalar DCache仍按64B行处理。当前代码合同见[数据布局](data-layout.md)、[依赖计数](dependency-counters.md)与[调度协议](protocol.md)；历史阶段的验证不自动覆盖后续改动。

**用户约定：方案确定后按阶段自主实现，遇到问题自行分析、修复、验证和迭代，不因常规实现选择反复询问用户。**
每阶段保留代码差异、验证结果、未解决问题和下一步，不把上下文切换当作停止任务或重新确认的理由。
不得通过缩减功能覆盖、掩盖 native fallback、删掉慢用例或放宽数值正确性来完成门禁。

执行顺序：修改前主线基线 → 分阶段实现与局部回归 → 新旧功能验证 → 无 Graph 与现有 A5 Qwen3 的性能测量和优化 → 最终复审。
用户于 2026-11-02 调整优先级：无 Graph 性能尚未达标时，也要优先测出并反馈 Qwen 的主线/候选数据。
因此 P2 数据采集不再等待 P1 门禁；已有退化和待验证项继续明确记录，不能用执行状态代替数值反馈。
某一关失败，先定位并修复，再重跑直接受影响的验证；只有变化影响整体协议或公共层时才扩大回归范围。
最终仍须在同一代码版本汇总完整功能与性能结果，不能拼接不同版本的通过记录。

沿用用户工作区约束：不新建 worktree，保留 `.gitignore` 的 `.venv-csa/` 排除，不再清理该环境。
主线基线必须在第一处 runtime 修改前采集，并记录精确 SHA；文档修改不影响这份二进制基线。
不因工具工作流默认建议 worktree 而覆盖此约束。
发布、推送和创建 PR 不属于本设计的交付前提。

开发过程中按阶段保存代码、验证结果和局部提交；合入准备将本分支最终实现及文档压为一个提交。
阶段编号描述设计与验证过程，不再对应独立、可 checkout 的 Git commit。
实施记录使用稳定的阶段名、candidate 名称、证据文件和源码/二进制哈希建立对应关系。
历史测量中的 SHA 保留原值，只用于说明测量当时的来源，不作为压缩后提交历史的导航入口。
阶段验证不能拼成最终全量验收；压缩提交也不改变已记录的失败、性能限制或未完成项目。
本地提交不等于推送、发布 PR 或合入主线。

2026-11-02 用户进一步明确：当前跨分片实现先形成阶段提交，允许保留已记录的局部性能退化，
工作重心转向 AIC/AIV/MIX 全局任务队列重构。不能继续以修复 AIV chain 回退为提交或队列重构的前置条件。
重构要消除 ready 任务对完成者 cluster 和 shard 的依赖，覆盖普通任务与 Graph child；
保留依赖、唯一领取、cooperative admission、错误传播和退出证明。
阶段提交须说明其确切验证范围，最终性能验收要求仍保留，已有基线继续复用。
用户要求同时实现全局唯一队列与分级队列进行比较，重点评估 A5 同地址原子争用。
两版共享正确性契约与负载，分别冻结源码和二进制；比较两版候选，不重采主线。
分级方案必须具有全局工作可达性，不能重新引入 shard 之间不可领取的问题。
业界调研、共同协议与 G/H1/H2 候选见[全局与分级 ready 队列设计](ready-queue-design.md)。

2026-11-03 用户再次明确当前顺序：先优化队列与 A5 Qwen 的 Graph 性能，
性能仍明显落后主线时，不提前启动完整 19 项无 Graph 性能采样。
候选迭代保留必要的正确性、协议与故障验证，只对直接受影响的少量无 Graph 场景做定位测量。
Qwen 性能接近目标后再执行完整性能回归；已有局部退化如实保留，不能用补齐报表替代优化。

### 长任务的检查点与恢复

设计、执行状态和原始证据分别保存：本目录定义契约与决策理由，实施记录保存阶段结论与验证索引，
`outputs/a5-hbg-graph-task/checkpoint.md` 保存当前操作状态，大日志和二进制保留在其相邻目录。
检查点保持简短，列出当前 revision、已应用修改、未应用补丁、活动任务 ID、最后一项验证结果、
尚未解决的问题以及下一条具体动作；阶段完成、重要失败或操作状态变化后及时刷新。

上下文压缩后，先核对检查点、实际 diff 和仍运行的任务，再继续工作。
开始新的性能候选前，完整阅读[已否定的性能尝试](rejected-optimizations.md)，并核对 investigation 索引；
不能只凭关键词搜索或恢复摘要判断该方向从未尝试。重新打开已否定方向时，先写明新增证据和重试边界。
不能仅凭摘要判定测试通过、补丁已应用或设备任务已结束；必须回查对应产物。
发布顺序、参数所有权、缓存一致性、exactly-once、退出证明等关键结论，修改前重读相关契约和代码。
被否定的方案保留原因与证据，防止恢复后再次采用。
长日志只读取相关段落，避免反复全量扫描消耗当前阶段所需上下文。

每阶段仍须形成实现、验证和复审的闭环，不能以压缩或上下文切换作为阶段完成的依据。
用户无需重复授权或提醒；若以后更换会话，交接必须包含上述文件入口与未完成动作。

## 代码改动入口

路径均相对仓库根目录；标注“新增”的名称为建议落点，避免继续把所有逻辑堆入大头文件。

| 模块 | 修改职责 |
| ---- | -------- |
| `src/common/host_build_graph/graph_execution.h` | 拆出纯 Definition wire；保持 A2A3 ABI 和校验语义 |
| `src/a5/runtime/host_build_graph/host/runtime_maker.cpp` | 分离 pack/carrier，preflight mode，创建 plan，commit 后实例化设备地址，N/T 分离 |
| A5 `host/native_graph_plan.h/.cpp`，新增 | 唯一 Definition 校验、CSR/root/sink、dense map、参数 offset、内存预算与能力结果 |
| A5 `host/runtime_maker.cpp` 的 HostPreparePlan | 自有 plan、park/resume、retry、pipeline slot publication/lease 生命周期 |
| A5 `runtime/scheduler/scheduler_graph.h` | versioned view、统一 kind/fanin/args/identity 访问器，旧路径仍直接寻址 |
| A5 `runtime/scheduler/scheduler_graph_binding.h`，新增 | AICore POD Tensor/scalar/predicate 重定位及精确范围发布 |
| A5 `runtime/scheduler/scheduler_graph_control.h`，新增 | 无资源 control inbox、完整同步后继传播、DONE 与汇总 |
| A5 `runtime/scheduler/scheduler_ready.h` | bootstrap/route/resolve 的 fanin 读取统一，控制与 kernel 分流，direct refill 保持 kernel 专用 |
| A5 `runtime/scheduler/scheduler_admission.h` | Ready唯一owner保留本地pending，提交时发布ADMITTED；成员首次观察bootstrap已发布的参数 |
| A5 `runtime/scheduler/scheduler_types.h`、layout 创建处 | node/instance/template/header wire、counts、inbox、CHECK 进度与 checked size |
| A5 `aicore/aicore_executor.cpp`、`resident_scheduler.h` | 启动握手、变体选择与窄入口声明 |
| A5 `aicore/resident_scheduler_impl.h` 及四个 `resident_*.cpp` | bootstrap、每轮调度/控制服务、完成 flush 与退出；普通/native × profile 开关显式实例化 |
| A5 `aicpu/aicpu_executor.cpp` | N-node supervision、mapped error identity、保持原 run fence |
| A5 `host/scheduler_protocol_export.h` 及 DFX reader | 双身份、控制事件、独立 expected graph、schema version |

必须搜全并迁移 `scheduler_graph_payload/descriptor/fanin_id`、`SCHEDULER_GRAPH_FANIN_COUNT_OFFSET`、
`graph_task_count`、`TaskId` 转换和 `trace_index` 使用点，不能只改 steady-state route。
至少覆盖 bootstrap、completion 单 fanin 快路、predicate、MIX payload、error dump、protocol freeze/export。
Graph 模式的 raw storage accessor 应拒绝 synthetic/child ref，防止漏改调用点读到另一个原始 SM slot。

## 分阶段实现

| 阶段 | 依赖与产物 | 离开本阶段的证据 |
| ---- | ---------- | ---------------- |
| B0 基线 | 冻结主线 SHA、工具链与工作负载 manifest；采集 ST 并筛选性能用例 | 原始日志、机器信息、用例选择理由、可比较的 timing marker 齐全 |
| I1 统一视图 | B0 后实现 view/accessor，先保持 Graph 仍走 legacy | 无 Graph 新旧结果一致；所有访问器和版本/边界 UT 通过 |
| I2 最小 native Graph | I1 后接 plan/GM、ENTER/EXIT/control inbox、普通 child 参数与 completion | 单 Graph、多个实例、交错普通任务端到端 native；进入模式有证据 |
| I3 完整调度语义 | I2 后接 SPMD、MIX、sync_start、predicate、失败与 pipeline lifecycle | 下面的并发矩阵通过，A2A3 公共层无回归 |
| I4 DFX 与完整功能 | I3 后补 identity/schema/oracle，审计 trace allocation 与所有诊断读者 | 新用例与原 A5 HBG corpus 通过；结果来自同一 revision |
| P1 无 Graph 性能 | I4 的功能合格版本与 B0 比较 | 固定 benchmark 集和实测筛出的 ST 均无可确认的劣化 |
| P2 A5 Qwen Graph 模型 | 对已有必要功能证据的候选，优先验证完整输出/KV 状态并 benchmark；可先于 P1 达标 | native 路径正确，主线/候选数值及时反馈，性能问题完成归因和迭代验证 |
| R 最终复审 | 协议、代码、功能、性能、内存/退出证据汇总 | 无未解决正确性缺陷；剩余限制和实测数据明确 |

I2/I3 的中间版本可按已完成 capability 选择整 run legacy，不能用一个总开关掩盖未实现的 shape。
最终支持集合必须达到总览范围，不能把中间 fallback 视为需求完成。
每轮优化只改变有证据支持的瓶颈；性能变化后复跑相关正确性、协议和生命周期用例。

## 功能与故障矩阵

以下是新增/扩展用例要求，优先复用 `tests/ut/cpp/a5/runtime/host_build_graph/` 的 scheduler test support，
以及 `tests/st/a5/host_build_graph/graph_execution*`、`graph_predicated_dispatch`、SPMD/MIX 场景。
不是新增几个 happy-path scene 就替代原有用例。

| 类别 | 必须覆盖的触发条件 | 断言 |
| ---- | ------------------ | ---- |
| 拓扑 | 链、菱形、多 root/sink、不连通 DAG、DUMMY-only、1024 个独立 child | 所有 sinks 加入 EXIT；每节点最多执行/resolve 一次 |
| 外部依赖 | Graph 前后普通任务、两个 Graph 串联、普通/Graph 共用 producer、RAW/WAR/WAW | 原依赖无丢失；消费者只在 EXIT 后执行 |
| fanin | 1、128、129、1023，EXIT 1024 sinks；重复/前向/自环边 | 合法图完整路由，非法图在发布前失败，无截断 |
| 多实例 | 相同 Definition、多 key、不同 boundary 地址/origin/scalar、并发 recording | 元数据共享，参数和控制互不覆盖；不跨 run 复用旧地址 |
| 参数 | 非连续 view、非零 start_offset、heap output、零 scalar、最大参数数 | 与公共 AICPU binder 的合法输入结果一致；checked arithmetic 拒绝溢出 |
| 控制服务 | 所有 slots 满、admission 暂存、长 DUMMY 链、大 fanout | 处理下一个 control 节点前完整传播当前节点后继；每次服务有界处理完整节点，无丢失/重复或递归栈增长 |
| SPMD/MIX | 单/多 block、波次、满宽、inactive lane、slow last block/lane、sync_start | identity/membership/order 正确，最后真实完成前不释放依赖 |
| predicate | true/false、PARAM/内部 operand、多个实例取不同值 | 跳过恰好一次；无多成员部分跳过；非法组合先选择明确结果 |
| 缓存 | 两 writer 相邻 scalar、两个 worker 消费同一参数、同地址新 epoch | 参数完整、无旧 cache line 覆盖，无每 block 重复绑定 |
| 完成 | EXIT 先于某 producer 的 resolve 尾部、空 control queue、有未 flush 计数 | 计数 N-H 与完整 drain 同时满足才成功 |
| 错误 | 部分绑定、非法 wire、重复 DONE、queue 发布失败、late error、退出超时 | fatal 保留身份；计数不能掩盖错误；未安全退出不得复用 backing |
| Host 生命周期 | allocation retry、park/resume、partial publication、连续及流水运行 | plan 自有字节、重新 rebase、slot 隔离、资源保留到 fence |
| 兼容 | 无 Graph 全量、已有 Graph legacy、合法不支持 shape、A2A3 公共路径 | 原语义保留，native/fallback 选择可观测且符合 capability |

执行前应用仓库 testing/onboard-arch-precheck 流程，真实检测 A5 silicon，通过 `task-submit` 持有设备锁。
运行新增 UT、受影响原 UT、A5 HBG 全部原有与新增 scene 的 sim/onboard 验证；手工标记和故障类按各自隔离要求运行。
改公共 wire/binder 校验时补 A2A3 HBG 的对应 sim 与可用硬件回归；没有对应硬件不能声称 onboard 已通过。
基线自身失败单独复现并记录，不能当成本次成功或静默跳过；本次引入的问题自行修复。

## DFX 与独立 oracle

kernel execution 的唯一身份为 `(run_epoch, node_ref, block_idx, subtask_idx)`；
用户可见身份另外携带 `(original_shell_id, local_child_id, kind)`。
ENTER/EXIT 没有 kernel trace，用控制节点事件表达；同一 shell 的两种控制事件靠 kind 区分。
不把内部 ref 写成公共 TaskId，也不让不同实例的 local_child_id 覆盖对方采样记录。

native protocol JSON 的 schema 与明确的 node map 共同定义映射语义。
`simpler_setup/tools/hbg/native_protocol.py`、export UT 和 Python oracle UT 与生产者使用同一当前格式。
当前实现导出 `hbg-native-protocol-v11`，保留领取结束证明、独立关闭标记和按资源选取的 MIX 成员，
以及按资源类型累计 block 前缀的 ordinary 有界循环成员规则，
并包含两部分原生 Graph 来源：

- `node_map`：每个 internal ref 的 kind、public TaskId、original task/shell id 与 local child id；
  ENTER/EXIT 使用相同 shell id，由 kind 区分，kernel identity 仍用公共 TaskId 与 block/subtask 信息关联。
- `graph_source`：来自未展开的 outer task payload fanin 和原始 Definition image 的 child/fanin/shape；
  不读取 native plan 的 sinks、completion map 或扩展边。源数据由本 run 的 Host owner 持有至诊断发布完成。

`native_graph.py` 从该源重建 ENTER、body、全部 sink→EXIT 及 outer completion 映射，
再与运行视图导出的 `tasks`/`node_map` 比较；`native_causality.py` 继续验证发布、成员、block 和完成顺序。
control 节点使用 protocol TaskDone actor，不伪造 kernel execution 或 legacy GraphPrepare。
无 Graph capture 输出 v10，native Graph 输出 v11；读端仅接受这两个当前格式，生产者与工具必须配套。
不维护历史或内部实验格式的兼容分支。execution identity 使用严格 v1，多 block 记录不得缺失 order_key。
执行身份是 run/task/block/subtask；slot 复用由本核事件序列与退休先于下一次 publication/permit 的关系审计，不导出 slot generation。
当前 native audit 从冻结任务形状独立重建 member_rank_offset 并核对成员集合，不以导出的偏移值作为 oracle。
所有受支持任务均遵守 cooperative admission。

expected graph 必须由原始外部 dep_gen 与原始 Definition 独立展开得到；
不能把被测 NativeGraphHostPlan 自己输出的边再当 oracle，否则同一种漏边会同时通过两边。

oracle 至少检查 ENTER gate、所有 sink→EXIT、外部消费者映射、exactly-once、runtime order、
membership、block ownership、release/TaskDone 顺序以及 abort 后的结果。
oracle 时间判定遵守原有跨核时钟约束，不能只凭不同核裸 timestamp 的大小断言因果。
chip swimlane、dep_gen、error dump 均应恢复公共身份；未采集的指标报告为缺失，不填零。
A5 HBG 的 AICore 调度路径在真实 task 边界采集参数，native Graph 子 task 复用相同协议并保留公共父/子身份。
多 block/Mix 的 AFTER 采样等待全部参与者写入可见后完成，再发布既有 task completion；详见[参数采集合同](../../../dfx/args-dump.md#311-task-time-boundaries-on-a5-hbg)。

设备真实性能诊断使用仓库 DFX 工具；最终性能数字关闭重型采集。
每次数据都保留 collector/schema 版本、run epoch、revision、case、round、scheduler mode 和有效性标记。
