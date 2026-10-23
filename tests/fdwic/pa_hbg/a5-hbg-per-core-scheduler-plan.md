# a5 HBG：每核一个 Scheduler

> 历史方案对照：本文件保留原有的 SSBUF 和 AIC 协调 Mix 方案。
> 本次 SPMD 的约束以 [SPMD 设计实现参考](a5-hbg-spmd-design.md)为准：
> Host 构图、完成驱动的依赖推进、每核自主执行；Mix 保留 AIC 有限协调，调度层不用 SSBUF。
> 运行时统一确定参与者顺序，非参与核不受约束，不设置全任务完成屏障。
> 以下条款不作为本次实现的默认选择。

## 目标

把 a5 HBG resident 调度从「每 cluster 一个 Scheduler 管 3 条 lane」改成
**每个 AICore 都是独立 Scheduler，只调度自己**。Mix 任务由 cluster 内的
**AIC 核**独占协调，它向同 cluster 的两个 AIV 投递 subtask 并聚合完成。

`scheduler_count`：36 → 108。Scheduler 与 Executor 的角色区分消失。

## 落地顺序

**看护先行，且每一阶段单独合入。**

| 阶段 | 内容 | 出口条件 |
| ---- | ---- | -------- |
| **A** | 建立功能看护：跑通现有用例建基线，补齐覆盖缺口 | 所有用例（含新补的）**在未改动的代码上全绿** |
| **0** | `task_id` → `task_ref` 并收敛访问器 | 看护全绿，无行为变化 |
| **1+** | per-core 重构（D1~D5） | 看护全绿；调度模式断言显示未回退 legacy；**per-core 不变式成立**（`dispatch_scheduler_worker_id == worker_id`，mix 除外） |

阶段 A 的硬约束：**新补的用例必须先在未改动的代码上跑通**。否则它第一次运行就在
改动之后，失败时无法区分"用例写错了"和"重构坏了"，守不住任何东西。
详见下文「阶段 A：功能看护」。

## 阶段 A：功能看护

**在动任何代码之前完成。** `tests/st/a5/host_build_graph/` 下的现成用例构成回归
基线，每条设计决策都要有用例守着，否则改坏了也看不出来。

### 现成用例 → 守哪条决策

| 看护目标 | 现成用例 |
| -------- | -------- |
| 跨 type 解依赖（D1 的根本动因） | `alternating_matmul_add`——AIC↔AIV 交替，正是"AIC 解出 AIV 下游"的典型 |
| 两级投递 + steal 覆盖 | `multi_core_dag`：`mixed_chain_{1,2,63,65,1024}`、`mixed_fanin32_1024`、`mixed_random_1024`、`mixed_multi_root_4096`。4096 多根图提供压快通道所需的并行度 |
| Mix 四种形状（D3） | `single_block_mix`：`mask ∈ {3,5,6,7}` × `{chain, burst}`，**已覆盖**文档里要求的 `0b011/0b101/0b110/0b111` |
| Mix 不含 AIC、AIC 纯协调 | `single_block_mix` 的 `mask=6`（AIV0+AIV1） |
| Mix 与普通任务共存（D4 执行流仲裁） | `single_block_mix` 的 `ordinary_mix_interleaved`（mask 逐任务变化） |
| Mix 并发压力 | `TestMixKernelRendezvous`：`burst256_mask{6,7}`，256 任务 |
| 调度延迟不退化 | `early_enqueue`——有**硬时序断言**（后继 `aicore_start` 对前驱的 overlap 窗口，回归时断言失败而非超时）。per-core 改变调度时序，**这个用例最可能先红，是预警点而非噪声** |
| legacy 回退路径不回归 | `graph_execution`、`graph_sync_start_early_dispatch`、`spmd_paged_attention`——本次它们仍整图回退 legacy，必须保持绿 |
| 真实规模的 AIC/AIV 交替流水 | **`paged_attention_unroll`**——4 段交替（func_id 0 aic → 1 aiv → 2 aic → 3 aiv），`batch=256 / num_heads=16 / context_len=8192`，最接近生产负载的跨 type 解依赖压力。**注意 `platforms: ["a5"]`，sim 跑不到，必须单独 onboard 验证**；`Case2` 是 `manual: True`，需显式触发 |
| 其他真实负载 | `paged_attention`、`batch_paged_attention`、`alternating_matmul_add`、`examples/a5/host_build_graph/` |
| DFX 产物 | `dfx/chip_swimlane`、`dfx/dep_gen`——swimlane 按 108 个 scheduler 归并后产物结构会变，需同步 |

### 三个缺口

**缺口 1（最关键）：无法断言跑的是 resident 还是 legacy。**
回退只写一行 `LOG_INFO`（`runtime_maker.cpp:1104,1257`），**测试侧没有任何钩子**，
一个意外回退 legacy 的改动会让上表每一条看护都假绿——结果正确，因为 legacy 本来
就是对的。对一个把 `scheduler_count` 从 36 改到 108 的重构，这个缺口是致命的。

补法：把 `runtime_mode`（`runtime->dev.scheduler_bootstrap.runtime_mode`，host 侧
`select_legacy_scheduler` 就是它的写入点）暴露给 SceneTestCase，加一个
`EXPECT_RESIDENT` 类字段由框架校验。**这是看护基础设施而非重构本身，属于阶段 A**，
它先落地，后续每个阶段才有可信的验证基线。

**缺口 2：mix + predicate 组合未覆盖。** `predicated_dispatch` 的四个 incore
全是 `core_type: aic`，而 D3 规定「predicate 由 AIC 求值一次、以 1 bit 随投递
下发，保证三个 subtask 一致」——这条恰好没有用例守。需要加一个
mix 形状 + predicate 的用例。

**缺口 3：负载不均未覆盖。** 现有 DAG 都较均匀，无法暴露 D2 快通道回落规则失效。
需要构造「一个 cluster 积压、其余空闲」的图，确认任务能被其他 cluster 偷走而不是
困在 SSBUF 里（见风险表"快通道藏任务"）。

### 缺口的用例设计

三个都必须**先在未改动的代码上跑通**，所以每项都给出"旧代码上的预期结果"——那是
它自身正确的证据。

#### 1. 调度模式断言钩子（框架改动，非用例）

值域已有：`SCHEDULER_RUNTIME_MODE_{RESIDENT_PENDING, RESIDENT_READY,
LEGACY_GRAPH, LEGACY_UNSUPPORTED_SHAPE}`（`scheduler_layout.h:16-22`），
host 侧 `select_legacy_scheduler` 是唯一写入点。

- **暴露路径**：host 侧留一个只读字段，经 binding 暴露给 Python。
  **不要走 STRACE span 属性**——那条通路（`early_enqueue` 在用）需要 DFX 标志才有
  数据，缺标志时断言会静默跳过，正是本缺口要消灭的失败模式
- **框架字段**：`SceneTestCase` 增加 `EXPECT_SCHEDULER = "resident" | "legacy"`，
  每个 case 跑完由框架无条件校验；未标注的类暂不校验，便于增量铺开
- **标注现状**：`graph_execution`、`graph_sync_start_early_dispatch`、
  `spmd_paged_attention` → `legacy`；其余 a5 HBG 用例 → `resident`
- **旧代码上的预期**：按上面的标注全绿。**如果某个标 `resident` 的用例报出
  legacy，说明现状就有意料之外的回退**——那本身是个发现，要先查清再往下走

#### 2. `mix_predicated_dispatch`（守 D3 的 predicate 一致性）

拼合两个现成用例的机制：`single_block_mix` 的三 incore 结构
（matmul/aic + add/aiv + mul/aiv）+ `predicated_dispatch` 的 gate 机制
（orchestration 侧 `args.set_predicate(gate[0] > 0)`，
`predicated_dispatch_orch.cpp:101`）。

- **图结构**：`gate_producer`(aic，写 `gate[0]`) → `mix_task`(带
  `set_predicate(gate[0] > 0)`，依赖 gate_producer)
- **CASES**：`mask ∈ {3,5,6,7}` × `gate ∈ {0,1}` = 8 例
- **golden**：`gate=0` → mix 的**全部** subtask 跳过，三个输出保持初值；
  `gate=1` → 全部执行
- **核心断言靠 golden 本身**：每个 subtask 写各自的 sentinel，`RTOL=ATOL=0`。
  若三个 subtask 的 predicate 求值不一致导致**部分执行**，产生的混合态既不等于
  "全跳过"也不等于"全执行"，比对直接失败。这正是 D3 把求值收归 AIC 要守的性质
- **旧代码上的预期**：全绿。现状由单个 Scheduler 串行对每个 placement slot 各求
  一次值（`scheduler_prepare_dispatch_slot:1315`），窗口小到不会分叉

#### 3. `fanout_imbalance`（守 D2 的快通道回落）

- **图结构**：单根 + 大扇出。一个 root 任务完成后解出 N 个互相独立的下游
  （N = 1024），全部在同一瞬间变 ready。执行 root 的那个核会把它们**全部**投向
  本 cluster 的同 type 核——这正是快通道容量被打穿的场景
- **CASES**：`N ∈ {64, 1024}`，`a5sim` + `a5`

**观测通路：DFX trace，不是 kernel 写回。** kernel 侧拿不到核标识——
`GlobalContext` 只有 `sub_block_id`（AIV lane 0/1，不区分 cluster，对 AIC 无意义，
`intrinsic.h:113-118`），`LocalContext` 只有 `block_idx`/`block_num`
（`intrinsic.h:148-155`）。为测试去改 `DispatchPayload` 这个 host↔device wire
结构不划算。而 `SchedulerTaskTrace` 本来就记了所需的一切：

| 字段 | 含义 |
| ---- | ---- |
| `worker_id` | 该 subtask 在哪个核执行 |
| `dispatch_scheduler_worker_id` | 哪个 scheduler 调度了它 |
| `complete_scheduler_worker_id` | 哪个 scheduler 处理了它的完成 |
| `refill_scheduler_worker_id` | 哪个 scheduler 回填了它的 slot |

**断言分两类，不能混：**

| 类别 | 内容 | 何时加 |
| ---- | ---- | ------ |
| 回归看护 | 功能正确、不超时（回落失效会退化为近似串行，大 N 直接撞 watchdog）；`worker_id` 的 distinct 数量达到阈值（按实际 `aic_count`/`aiv_count` 推导，不写死） | **阶段 A**，旧代码上即绿——现状 1024 下游由 36 个 Scheduler 分摊本就散得开，这一步建立的是"分布应有多广"的基线 |
| per-core 不变式 | 非 mix：`dispatch_scheduler_worker_id == worker_id`（自己调度自己）；mix subtask：`dispatch_scheduler_worker_id == 本 cluster AIC` | **重构阶段**。旧代码上此式**不成立**（现状调度者是 cluster 的那个 AIV），所以它不属于阶段 A |

后者一旦成立，D1/D3 的所有权模型就被钉死，比任何间接的时间/分布断言都硬，值得在
重构阶段作为主验收信号。

**缺数据必须 fail 而非 skip**：这条通路依赖 chip_swimlane 的 DFX 标志。按
`CLAUDE.md` 记录的已知坑，缺标志时断言会静默跳过而测试仍 pass。用例要先断言
trace 数据存在且条数符合预期，再做上述比对。

### 执行顺序（阶段 A 的出口条件）

**硬约束：补充的用例必须先在未改动的代码上跑通。** 否则它第一次运行就在改动之后，
失败时无法区分"用例写错了"还是"重构坏了"。

1. 环境：`source .venv/bin/activate && pip install --no-build-isolation -e .`
2. **在未改动的代码上**跑现成用例，确认全绿并留存结果作为基线：

   ```bash
   pytest examples tests/st --platform a5sim --runtime host_build_graph
   pytest examples tests/st --platform a2a3sim --runtime host_build_graph
   ```

3. **onboard 基线**（sim 覆盖不到 `paged_attention_unroll`，必须单独跑）：

   ```bash
   .claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
   task-submit --device auto --device-num 1 \
     --run "python -m pytest tests/st/a5/host_build_graph --platform a5 --device \$TASK_DEVICE"
   ```

4. 补齐上述三个缺口的用例与钩子，**再次在未改动的代码上跑通**——缺口 1 的钩子此时
   应如实报告现状的 resident/legacy 分布（graph/SPMD 用例为 legacy，其余为
   resident），这是钩子本身正确的证据
5. 阶段 A 单独合入。此后每个阶段结束都重跑 2~3，并对比基线

### 性能基线

同一组用例在阶段 A 时点留一份延迟数据，供后续阶段对照。预期收益来自调度并行度
36→108、deferred/跨 lane 逻辑消失、本 cluster 投递走 SSBUF；需确认跨 cluster 的
CAS push 没有吃掉收益。`paged_attention_unroll` 本身就是 benchmark 性质，是主要
的性能对照项。

## 阶段 0：`task_id` → `task_ref` 并收敛访问器

**在阶段 A 之后、per-core 重构之前，单独合入。**

### 为什么

resident scheduler 里叫 `task_id` 的变量**不是** `TaskId`。
`TaskId`（`src/common/host_build_graph/task_id.h:36`）是 64 位 handle，
`(space << 62) | (parent << 32) | local_id`。而 resident 侧持有的是 GLOBAL 任务的
**table index**：

- 拿到后直接与 `graph.task_count` 比较判越界（`scheduler_ready.h:1085`）
- 当数组下标用：`scheduler_task_control_at` / `scheduler_task_metadata_at`
- 已有 `task_id <= INT32_MAX` 的假设（`scheduler_ready.h:80`）

同名不同物，是后来人最容易踩的坑。

### 名字：`task_ref`

`TaskId` 的三个 space（`task_id.h:94`）里，**`SUB_TASK` 是 graph task body 内的
任务**——它不在 task table 里，靠 `parent_id()` 定位，这正是高 32 位 parent 字段
存在的理由。

叫 `task_index` 等于把「标识就是个 table 下标」固化进名字。`TaskKind::GRAPH`
今天整图回退 legacy，但 resident 迟早要支持它，届时 SUB_TASK 根本没有 table 槽位，
名字又得再改一轮。**限制写注释，不写名字。**

同理 `SCHEDULER_TASK_ID_INVALID` → `SCHEDULER_TASK_REF_INVALID`。

### 范围：改名 + 收敛访问器

阶段 0 同时把**真正依赖 index 语义的点收敛到少数访问器里**：

- `>= graph.task_count` 的越界判
- `scheduler_task_control_at` / `scheduler_task_metadata_at` 的下标运算
- `-1` 哨兵的产生与判定

收敛后，「将来支持 graph task 要改哪里」从散落 180 处变成一份可枚举的清单。
这比改名本身更有价值。

集中点统一注释：

```text
当前编码：GLOBAL 任务的 table index。SUB_TASK（graph task body 内的任务）尚未
进入 resident 路径——TaskKind::GRAPH 仍整图回退 legacy。支持它时需扩展此编码以
承载 parent，解析点集中在这里而不是散落在下标运算中。
```

### 边界：正好落在 AICore / AICPU 的编译划分上

| 侧 | 文件 | 实际含义 | 操作 |
| -- | ---- | -------- | ---- |
| resident（AICore） | `runtime/scheduler/*.h`、`aicore/aicore_executor.cpp` | `int64_t` table index | **改** |
| legacy（AICPU） | `runtime/scheduler/*.cpp`、`scheduler.h`、`scheduler_context.h`、`aicore/aicore_legacy_executor.cpp` | 真 `TaskId` handle（见 `scheduler_context.h:211` 的 `TaskId task_id_snapshots[]`） | **不改** |

与 `build_config.py:42-55` 的 `.h`→AICore / `.cpp`→AICPU 划分完全重合。

**a2a3 不受影响**：它没有 resident scheduler（`src/a2a3/.../runtime/scheduler/`
下只有 legacy 的 `.cpp` + `scheduler.h` + `scheduler_context.h` + `scheduler_types.h`），
所以 `codestyle.md` 规则 10 的「兄弟树同 commit 跟进」不适用。

规模约 180 处：`scheduler_ready.h` 94、`scheduler_graph.h` 24、
`scheduler_completion.h` 23、`scheduler_mix.h` 13、`scheduler_dispatch.h` 9、
`scheduler_types.h` 9、`scheduler_ssbuf.h` 1、`aicore_executor.cpp` 8。
另有 `host/runtime_maker.cpp` 52 处混杂两条路径，需逐处分辨归属。
访问器收敛另计十余处真实改动。

### 为什么单独合入

- 后续阶段 diff 已经很大，再混 180 处机械改名，review 无法进行
- 改名部分有编译器兜底（改漏/改错直接编不过）
- 独立有价值：即便后续方案再调整，这个名字该改还是该改
- 符合 `codestyle.md` 规则 10「一个标识符、所有出现、一个 commit」

验证：两个 sim 变体全绿（无行为变化）。

## 现状（改之前）

`src/a5/runtime/host_build_graph/`

每 cluster 3 lane（lane0=AIC，lane1/2=AIV），AICPU 在每 cluster 选**物理号较小
的那个 AIV** 当 Scheduler（`aicpu/aicore_lifecycle.cpp:252-271`），
`scheduler_index = cluster_index`，`scheduler_count = aic_count = 36`。

`aicore/aicore_executor.cpp:270-532` 一个循环用 `scheduler_worker` 分叉：

| 角色 | 行为 |
| ---- | ---- |
| Scheduler（1 个 AIV） | 拉 ready → 填 **3 lane × 2 slot** → 轮询 3 条 lane 的完成 → 解依赖 → 顺带执行自己 slot 的任务 |
| Executor（AIC + 另一 AIV） | 盲扫 SSBUF `dispatch[slot].publication` 等 generation 递增，执行后回写 `completion.publication` |

**现状的复杂度几乎全部源于「Scheduler 自己也是一条执行 lane」**：

- `SchedulerDeferredAivQueue`（`runtime/scheduler/scheduler_dispatch.h:23`）：
  给自己填任务会阻塞调度，于是先预约 slot（本地 FILLING）、延后发布
- `scheduler_fill_cluster_normal_slots` 里的 AIV worker 排序
  （`scheduler_normal_aiv_worker_precedes`：peer 优先于 self）
- `scheduler_publish_refill_candidates`（`scheduler_mix.h:134`）跨 lane 找空 slot
- Executor 的 `next_dispatch_sequence` 严格不可跳号——因为它盲扫，无法分辨
  "这个 generation 是不是给我的"

## 目标模型

```text
每核（108 个）= 一个完整 Scheduler：
  自己的 ready inbox（按 core type）
  自己的 2 个 dispatch slot，自己填、自己执行、自己观测完成、自己解依赖

AIC 核额外承担 cluster 内 Mix 协调：
  独占 MIX ready queue（AIV 核完全不碰）
  为 AIV subtask 准备 payload → 投递 SSBUF mix request
  聚合两个 AIV 的 done + 自己的 AIC subtask，凑齐 active_mask 才 DONE
```

**Mix 不需要同步启动**：`scheduler_plan_mix_placement`
（`scheduler_ready.h:1033`）显式排除 `scheduler_task_is_gang`，三个 subtask 是
同一 task 的不同部分，只在完成时聚合。所以 AIC 投递后各 AIV 自行择机消费即可，
不需要 rendezvous。这是整个 Mix 协议能做得这么轻的前提。

**Mix 不一定含 AIC**：`active_mask` 可能是 `0b110`（AIV0+AIV1）。AIC 核此时是
纯协调者，不是参与者。

## 设计决策

### D1. Ready 归属：跨核 CAS push（这是「按 core type 分区」的硬要求）

**前提澄清：任务的 type 与「谁解出它」无关。** 依赖图是异构的——
`matmul(AIC) → activation(AIV)` 是最常见的形态：A 跑在 AIC 核上，A 完成时由
**AIC 核**遍历 wake list，发现下游 B（一个 AIV 任务）fanin 清零。B 该进 AIV 的
inbox，但发现它 ready 的是 AIC 核。

现状之所以没这个问题：一个 Scheduler 管整个 cluster，三种核都有，解出任何 type
都能自己消费，所以 `scheduler_resolve_completion` 把 waiter 按 type 分到
`batches[0/1/2]` 后全推进**自己的** 3 个 inbox（`scheduler_ready.h:1588-1596`）。

现状 `scheduler_ready_batch_push`（`scheduler_ready.h:960`）**只往自己的 inbox
推**，靠「owner 是唯一发布者、且只在 head 为空时 store」这个不变式避免与 thief
的 CAS pop 冲突；非空时挂到本地 `owner_pending_endpoints`。

per-core 后这个不变式站不住：**解依赖的核和能执行该任务的核 type 不同**。
AIV 核解出一个 AIC 任务，必须让某个 AIC 核拿到它。

**inbox 改 CAS push，直接投给正确 type 的 owner。** 若改为"仍推给自己、让消费者
按 type 全局偷"，36 个 AIC 核会散落在各个 shard 里，每轮要扫遍全部 shard 才找得到
同类任务——分片本来就是为了避免这个。CAS push 顺带**删掉整套 `owner_pending_*`**
（`scheduler_ready.h:838-900` 五个函数 + `refresh_ready_inbox_type` 的一半）。

**投递目标选取**：优先本 cluster 的同 type 核——AIV 核解出 AIC 任务投本 cluster
AIC；AIC 核解出 AIV 任务按 `task_ref & 1` 分给 AIV0/AIV1。目标固定、无需查表。

本 cluster 的投递**不走这里的 GM inbox，而是走 D2 的 SSBUF 快通道**；只有快通道
不可用（队列满或目标核无空闲 slot）时才回落到 GM inbox。因此 D1 的 CAS push 实际
只承担**跨 cluster** 投递，热点比"全部走 GM"低得多。

**ABA 从根上消除。** ABA 的唯一来源是
`scheduler_publish_direct_mix_candidate`（`scheduler_mix.h:97-118`）的
「pop 出来发现 placement 失败、再 push 回去」——同一 task 二次入链才构成
Treiber ABA。改成 **先探测、能放下才 pop**，同一 task 一生只入链一次，ABA 不成立。

探测之所以可靠，依据在 D3：mix 的落点是 **AIC 专属 slot**，唯一占用者就是探测者
自己，所以"有空位"是确定事实而非 racy 的 hint。head 因此保持为裸的 64 位标识，
编码可随 graph task 支持而扩展，不被任何位域绑死。

另一条疑似路径已排除：`scheduler_publish_refill_candidates` 中
`claim.slot_index == UINT32_MAX` 时的 `batch_append` 回队，那个 task 是刚解依赖
产生、从未入过链的，不是 re-push。

**Directory bit 的竞态**：push 方 `fetch_or` 设别人的 bit，owner 发现 inbox 空时
clear 自己的 bit。最坏交错是 owner 读到空→别人 push 并 set→owner clear 覆盖，
此时 bit 丢失但 **inbox 非空且 owner 自己下一轮 refresh 会重新 set**，所以是
延迟一轮可偷，不是丢任务。可接受，但要在代码里写成注释（present-tense 不变式）。

### D1b. Inbox 按 type 编址，用已有的 `type_rank`

既然每个 queue 只有一种 core type 的核会消费它，inbox 就不必每核各留三份：

| queue | 持有者 | 槽数 | directory |
| ----- | ------ | ---- | --------- |
| AIC | 仅 36 个 AIC 核 | 36 | 36 bit → 6 shard |
| MIX | 仅 36 个 AIC 核 | 36 | 36 bit → 6 shard |
| AIV | 仅 72 个 AIV 核 | 72 | 72 bit → 11 shard |

`3 × 108 = 324` → `144` 个 inbox。索引用**已经存在的 `type_rank`**
（`aicore_lifecycle.cpp:193-195` 现在就在算 `aic_count++` / `aiv_count++`），
不需要新编号。

按 `type_rank` 编址后，一个 shard 内的 7 个 owner 全是同 type 的核，现有的
「只偷同 shard」语义直接有效，不必扩大扫描范围。

**AIC inbox 与 MIX inbox 不能合并成一条链**：MIX 的认领要先探测本 cluster 容量
（D3），不满足就留在链里；混在一条链上，队头卡住一个暂时放不下的 mix 会把后面的
纯 AIC 任务一起堵死（head-of-line blocking）。两条链分开，AIC 核轮询两个 inbox。

### D2. 两级投递：cluster 内走 SSBUF 快通道，跨 cluster 走 GM inbox

SSBUF 是 cluster 私有的片上 buffer，比 GM 快得多。**本 cluster 的投递绕开 GM
inbox，直接走 SSBUF。** 在 per-core 模型下这条通道命中率很高：AIC 解出 AIV 下游
（matmul→activation）是最常见的形态，目标往往就在本 cluster。

它同时压掉了 GM inbox 的 CAS 热点——只有跨 cluster 投递才参与 D1 的 CAS push。

**必须是 SPSC。** SSBUF 没有原子 RMW（`scheduler_ssbuf.h:125`：
"These are SPSC publication tokens, not atomic read-modify-write state"），所以
不能做 MPSC 入队。每个 (生产者, 消费者) 对一条独立队列：3 个消费者 × 2 个本
cluster 生产者 = **6 条队列**，各深度 2。

统一消息格式，mix 与普通任务共用一条通道（消费侧本来就要轮询它，不增加轮询点）：

```cpp
struct SchedulerClusterPost {   // 16B
    int64_t task_ref;
    uint8_t kind;            // NORMAL | MIX_SUBTASK
    uint8_t subtask_slot;    // MIX_SUBTASK 才有意义
    uint8_t predicate_pass;  // 同上，AIC 求值一次的结果（D3）
    uint8_t reserved;
    int32_t timing_slot;
};
```

#### 回落规则（防止负载均衡退化）

投进 SSBUF 的任务**其他 cluster 偷不走**。所以快通道只在目标核真的能很快消费时才用：

- 目标核的接收队列有空位，**且**它已发布有空闲 dispatch slot → 走快通道
- 否则 → 回落 GM inbox，让空闲 cluster 能偷走

为此每个核把自己的空闲 slot 数发布到 SSBUF header（3 核各一个字，投递方一次读
全部）。这个忙闲视图同时服务 D4 的 mix 投递决策。

#### 1 KiB 布局重划

| 区 | 大小 | 用途 |
| -- | ---- | ---- |
| header | 64B | 原有字段 + 3 核忙闲发布（占用现有 `reserved[52]`） |
| 投递队列 `[3 消费者][2 生产者]` | 6 × 64B = 384B | 令牌 8B + 2 × 16B 消息，余量填充 |
| mix 回传 `[2 AIV lane]` | 2 × 128B = 256B | executor trace ×2 + done 令牌（AIC lane 不需要） |
| 余量 | 320B | |

mix 回传只有 AIV lane 需要：普通快通道任务由接收方自己完成、自己解依赖、自己写
GM trace，投递方不关心；只有 mix subtask 要把 trace 和 done 回报给 AIC 聚合。

### D3. Mix 用 AIC 专属 slot；subtask 由 AIV 自己 prepare

#### 槽位所有权

每个 AIV 核持有两组 dispatch slot：

| 槽位 | 常量 | 唯一占用者 | 完成令牌 |
| ---- | ---- | ---------- | -------- |
| 自有 slot ×2 | `SCHEDULER_PENDING_SLOT_COUNT = 2`（不变） | AIV 自己 | 现有 2×32 打包 |
| mix 专属 slot ×2 | `SCHEDULER_MIX_SLOT_COUNT = 2`（新增） | **本 cluster 的 AIC** | 独立 done 令牌字 |

**专属所有权让 AIC 的探测精确，这是整个 mix 协议不需要回压的根据**：AIC 看这些
slot，看到的是自己投递与回收的历史，没有第二个写者。于是

- pop 之前即可确定三个 subtask 都有落点 → **pop 之后必然成功** → 无需回压
- 无回压 → 无 push-back → **D1 的 ABA 自然不成立**
- mix 不需要目标核发布忙闲（那只对 D2 的普通快通道才需要）

AIC 的判定：两个 AIV 的 mix 专属 slot 有空 **且** 自己有空闲自有 slot（当
`active_mask` 含 AIC 时）→ 认领。

**不要把 mix slot 并进 `SCHEDULER_PENDING_SLOT_COUNT`**：该常量有 49 处使用，其中
5 处是"一个 u64 装 2 个 32 位 generation"的打包（`scheduler_ssbuf.h:103`、
`scheduler_ready.h:305,323`、`scheduler_completion.h:315`、
`scheduler_dispatch.h:323`），改成 4 会全部溢出。两个独立常量则现有打包一行不动。

mix 专属 slot 需要专属 payload 落点，所以独立 payload 数组：
72 个 AIV 核 × 2 × 512B ≈ **73 KiB**。

#### prepare 在 AIV 侧

AIC 投递的是**任务引用**（`task_ref` + `subtask_slot` + predicate 结果 +
timing_slot），不是 payload。AIV 收到后走与普通任务相同的
`scheduler_prepare_dispatch_slot` 路径，只是落点是 mix 专属 slot 及其 payload。

- **省掉 payload 的 cache 往返**：AIC 写 512B → writeback → AIV invalidate → 读，
  整趟消失
- **prepare 并行化**：从一个核串行 2~3 份变成各核并行 1 份
- **复用现有路径**，不需要"为远端核 prepare"的变体

集中解析再广播（`scheduler_materialize_mix_arguments`，`scheduler_graph.h:248-261`）
省不下什么：`scheduler_parse_task_payload_arguments`（`scheduler_graph.h:189-217`）
只读 4 个 int32 再做边界检查，而写 args 的总量和各核各写一份完全相同。

**唯一留在 AIC 的是 predicate 求值。** 它决定 `function_bin_addr` 是否置 0
（跳过执行），同一 mix task 的三个 subtask 必须一致地执行或跳过。各核自己求值会把
窗口从同核串行的几十周期拉大到跨核几百周期。AIC 求值一次、以 1 bit 随投递
下发，成本几乎为零，且比现状（`scheduler_prepare_dispatch_slot:1315` 对每个
placement slot 各求一次值）更严格。

**`sub_block_id` 不用 AIC 传**：mix 时它按 `cluster_lane`（`lane == 2 ? 1 : 0`）
而非 `subtask_slot` 计算——single_aiv 的 mix 里 subtask1 可能落在 lane2，两者分叉。
AIV 自己 prepare 时最清楚自己是哪条 lane，这个值天然正确。

### D4. AIV 的执行顺序仲裁：mix 优先 + 有界让行

mix 专属 slot 不与自有 slot 争抢**落点**（D3），但两者争抢**执行流**——AIV 是单
执行流，同一时刻只跑一个 kernel。所以仍需要顺序仲裁。

- **mix 优先**：mix 完成才能释放 AIC 侧的 `mix_tracker` 与专属 slot，让后续 mix
  得以认领；mix 通常在关键路径上（AIC 在等着聚合）
- 但 AIC 回收 slot 后可以立刻再投，**稳态 mix 流入会让自有任务长期饿着**

取 **mix 优先 + 有界让行**：连续执行 `K = 2` 个 mix subtask 后，若自有 slot 有
READY 就强制放行一个。纯选择逻辑，dispatch path 上不引入任何等待
（符合 `codestyle.md` 规则 5）。自有任务的饥饿上界是 K 个 mix 的执行时间。

### D5. `next_dispatch_sequence` 严格串行约束取消

那是 Executor 盲扫 SSBUF 的产物。per-core 后每个核知道自己填了什么，按
`local_ready_mask` 取即可，不需要"generation 必须恰好等于 next"的自旋等待。
`aicore_executor.cpp:399-408` 的 `EXECUTOR_PREFERRED_SLOT_INVALID` 校验随之改写。

## 数据结构变更

| 结构 | 变更 |
| ---- | ---- |
| `SCHEDULER_CAPACITY` | `SCHEDULER_CLUSTER_CAPACITY`(36) → `SCHEDULER_WORKER_CAPACITY`(108)；inbox/directory 另按 type 编址（D1b） |
| `SchedulerReadyDirectory` | 按 type 分三段：AIC 6 shard / MIX 6 shard / AIV 11 shard，各自独立编址 |
| inbox 数组 | `READY_QUEUE_COUNT × WORKER_CAPACITY`(324) → `36 + 36 + 72`(144)，索引改 `type_rank` |
| `SchedulerReadyInbox.head` | `int64_t` 裸 task_id → **tagged**（低 32 task_id / 高 32 tag） |
| `SchedulerLocalState` | 删 `mix_trackers` 之外的跨 lane 字段；`slots[3][2]` → `slots[2]`；删 `owner_pending_endpoints`、`consumed_completion_generations[3]`→标量 |
| `SchedulerLocalSlotState` | `mix_tracker`(1B) 改名 `mix_link`：AIC 核=tracker 索引，AIV 核=mix 专属 slot 索引。仍 16B，断言不变 |
| mix 专属 slot | 新增 `SCHEDULER_MIX_SLOT_COUNT = 2`，AIV 核各 2 个，**AIC 独占写**（D3）。不并进 `SCHEDULER_PENDING_SLOT_COUNT`，否则 5 处 2×32 generation 打包溢出 |
| mix payload 数组 | 新增，72 AIV 核 × 2 × 512B ≈ 73 KiB |
| `SchedulerLocalConfig` | 删 `scheduler_lane`；`worker_ids[3]` 仅 AIC 核需要（投 mix），AIV 核只需知道本 cluster AIC |
| `AicoreSchedulerLayout` | 无新增字段（mix 不需要独立 payload，见 D3） |
| `SchedulerActivityBuffer` 数组 | `CLUSTER_CAPACITY` → `WORKER_CAPACITY`（按 `scheduler_index` 索引）。**内存从 864 KiB 涨到 2.6 MiB**，仅 `enable_activity_profiling` 时分配 |
| gang 结构 | `gang_participants` / `gang_commands` 仍按 `CLUSTER_CAPACITY` 索引。本次 SPMD/sync_start 继续整图回退 legacy，**但必须确认没有任何路径用 108 的 `scheduler_index` 索引它们**，否则越界 |

## 删除清单（重构的主要收益）

- `SchedulerDeferredAivQueue` 全套：`scheduler_dispatch.h:18-44, 246-388`
  （`deferred_aiv_pop_front` / `release_deferred_aiv_reservation` /
  `deferred_aiv_peer_lane` / `drain_deferred_aiv_to_peer` /
  `publish_deferred_aiv_local`）
- `scheduler_normal_aiv_worker_precedes` + `AivWorkerSlots` 选择循环
  （`scheduler_dispatch.h:38-44, 138-241`）
- `scheduler_publish_refill_candidates` 的跨 lane 找槽（`scheduler_mix.h:134-194`）
- `owner_pending_*` 五函数（`scheduler_ready.h:838-900`）
- `aicore_executor.cpp` 的 Executor 分支：盲扫 SSBUF、
  `scheduler_observe_dispatch_payload_*`、SSBUF completion 回写
- `SchedulerLocalState::is_scheduler()` 及其所有调用点（恒为 true）

## 执行循环形态（改后）

```text
AIV 核：
  1. 填空闲 slot：
       自有 slot ← 自己的 GM inbox，或本 cluster 同伴经 SSBUF 快通道投来的普通任务
       mix 专属 slot ← AIC 投来的 mix subtask（自己 prepare）
  2. refresh 自己的 GM inbox（含 directory bit 维护）+ 发布自己的空闲 slot 数
  3. 执行：mix 优先 + 有界让行（D4）
  4. 观测自己的完成：
       自有任务 → DONE → 解依赖 → 两级投递（本 cluster 快通道 / 跨 cluster GM）
       mix subtask → 回 SSBUF done 令牌 + trace，不碰 DONE（由 AIC 聚合）

AIC 核：
  1. 填自己的空闲自有 slot（来源同上，外加自己承担的 mix AIC-subtask）
  2. 轮询两个 inbox（AIC 链与 MIX 链分开，见 D1b）。MIX 链认领前先探测本 cluster
     容量——两个 AIV 的 mix 专属 slot 有空且自己有空闲 slot；专属所有权使这个判断
     精确，认领后必然放得下（D3），不需要回压
     求值 predicate 一次 → 投递任务引用给目标 AIV lane
  3. 执行自己 slot 中最老的 READY
  4. 轮询两个 AIV lane 的 done 令牌 + 自己的完成 → 更新 mix_tracker
     → 凑齐 active_mask 才 DONE + 解依赖
```

## 改动文件

| 文件 | 改动 |
| ---- | ---- |
| `runtime/scheduler/scheduler_types.h` | 容量常量、tagged inbox head、layout 字段 |
| `runtime/scheduler/scheduler_layout.h` | 无改动（mix 不需要独立 payload） |
| `runtime/scheduler/scheduler_ssbuf.h` | lane 字段语义重映射（大小不变） |
| `runtime/scheduler/scheduler_ready.h` | 本地状态瘦身、CAS push、删 pending、per-core 填槽 |
| `runtime/scheduler/scheduler_dispatch.h` | 删 deferred 全套，`fill_cluster_normal_slots` → `fill_own_slots` |
| `runtime/scheduler/scheduler_mix.h` | 重写为 AIC 侧投递 + 聚合协议 |
| `runtime/scheduler/scheduler_completion.h` | 完成观测改本地；mix done 聚合 |
| `aicore/aicore_executor.cpp` | 删 Executor 分支，单一 per-core 循环 |
| `aicpu/aicore_lifecycle.cpp` | 每核 `is_scheduler=1`、`scheduler_index=worker_id`、`scheduler_count=108` |
| `host/runtime_maker.cpp` | inbox/directory 按 type 编址；swimlane 按 108 个 scheduler 归并 |

`runtime/scheduler/` 下 **`.h` 编进 AICore target、`.cpp` 编进 AICPU target**
（`build_config.py:42-55`）。本次只动 `.h` 侧 + `aicore_executor.cpp`；
`scheduler_dispatch.cpp` / `scheduler_completion.cpp` 是 AICPU legacy 路径，不要误改。

AICPU/AICore 侧禁 STL 容器；新增 header 用 `#pragma once`。

## 不在本次范围

- SPMD（`logical_block_num > 1`）与 sync_start：继续整图回退 legacy。
  参见 `/data/poursoul/docs/a5-aicore-scheduler-plan.md`，那份 SPMD 方案建立在
  cluster-scheduler 模型上，**本次改完后需要重写**（共享槽位数组的"按空闲 lane 数
  认领"在 per-core 下变成"每核认领 1 个 block"，反而更简单）。
- `TaskKind::GRAPH`
- a2a3 的 HBG resident 路径

## 风险

| 风险 | 说明 |
| ---- | ---- |
| CAS push 竞争 | 本 cluster 投递走 SSBUF 快通道后，GM CAS 只承担跨 cluster 投递，热点已大幅下降。若实测仍劣化，退路是按 `task_ref` 散列到多个同 type owner |
| 快通道藏任务 | 投进 SSBUF 的任务其他 cluster 偷不走。回落规则（目标核队列有空位 **且** 已发布有空闲 slot）必须生效，否则负载不均时会比全走 GM 更差。验证时要专门构造"一个 cluster 积压、其余空闲"的用例 |
| SSBUF 布局重划 | 1 KiB 硬上限，6 条队列 + mix 回传后余量 320B。新增任何 per-lane 字段前先重算，`static_assert(sizeof(region) == 1024)` 要保留 |
| 改名与重构混淆 | 阶段 0 必须单独合入；若与后续阶段混在一个 PR，180 处机械改名会淹没真正的逻辑改动 |
| graph task 的编码扩展 | `TaskKind::GRAPH` 仍回退 legacy，但 resident 迟早要支持 `SUB_TASK`（不在 task table 里，需 parent 定位）。阶段 0 的访问器收敛就是为这一步留的接口；**任何新代码不得再假设"标识即下标"** |
| activity buffer 内存 | profiling 开启时 2.6 MiB。若紧张则把容量降到 512 条/核 |
| gang 结构索引越界 | `scheduler_index` 从 ≤36 变 ≤108，任何用它索引 gang 数组的残留路径都会越界。改前先 grep 全部 `gang_participants` / `gang_commands` 使用点 |
