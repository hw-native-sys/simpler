# A5 HBG AICore 调度：按执行顺序走读

本文对应当前采用两个共享业务 ready 队列与 GM MIX 的 A5 HBG resident 实现；own slot 在执行后直接退休，按 ready_head 保持本核 FIFO。GraphView 为 v3，ready pool 为主线 v11，native graph image 为 v2，native protocol 为 v10/v11，execution identity 为 v1。
范围：从 device 已拿到本次运行的 GM 数据开始，覆盖初始化、依赖、取任务、准入、kernel 调用、完成传播和退出。Host 预建的 CSR/remaining 作为输入合同介绍，不展开 Host 构图实现；legacy 只标入口。
本文的源码链接使用相对于本 Markdown 文件的路径；进入文件后可按链接名称或上下文中的符号定位。VS Code Markdown 会将 `/` 开头的链接按工作区根目录解析，因此此处不使用磁盘绝对路径。

## 阅读顺序

阅读字段前可先看 [数据结构的作用范围与用途](00-data-structure-scopes.md)：按整轮、每任务、每worker、每slot、每cluster及每次异步执行说明存储、读写者和生命周期。

1. [初始化与启动数据](01-initialization.md)：谁初始化什么，Handshake、WorkerContext、RunControl 的字段。
2. [依赖、ready 队列与执行](02-ready-execution.md)：任务如何就绪、谁领取它、payload 如何构造、kernel 怎么调用。
3. [MIX/SPMD、完成与退出](03-cooperative-completion.md)：多核准入、MIX mailbox、完成传播、最终退出。
4. [native graph 与异步完成](04-native-async.md)：主链中的条件分支及其数据结构。

首次按上述顺序读；遇到结构表时对照定义。普通单核任务先贯穿主链，再读 MIX/SPMD；native graph 和 async 仍属于 resident，按实际输入条件进入。

## 整条调用链

| 顺序 | 执行者 | 代码位置 | 本步完成的事情 |
| ---- | ------ | -------- | -------------- |
| 1 | AICPU | [pre_handshake_init](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | 清 discovery 状态与退出 gate，记录核数和寄存器表 |
| 2，与 1 并行 | 每个 AICore | [平台 kernel 入口](../../../src/a5/platform/onboard/aicore/kernel.cpp) → [aicore_execute](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp) | 建立本核编号，报告物理核和类型 |
| 3 | AICPU 多线程 | [handshake_partition](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | 接受本轮报告，按物理核查寄存器地址 |
| 4 | AICPU leader | [post_handshake_init](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | 校验 1 AIC + 2 AIV 拓扑，填写每核 context |
| 5 | AICPU 多线程 | [publish_context_partition](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | 先写 context 地址，再发布 GM READY |
| 6 | 每个 AICore | [aicore_execute 的 READY 接收](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp) → [run_resident_scheduler](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 重新观察 context，初始化本核私有调度状态 |
| 7 | 所有 AICore | [bootstrap_ready_graph](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 分片扫描图，按 remaining 或兼容 wake-list 路由，发布初始 ready、准备首批 slot |
| 8 | AICPU leader / 各线程 | [wait_bootstrap_complete](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) → [release_partition](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | 等 bootstrap 完成，写 DMB=IDLE，开放 kernel 执行 |
| 9 | 每个 AICore | [run_ready_dispatch_loop](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 在本核循环做调度和执行 |
| 10 | 每个 AICore | [控制节点与异步完成服务](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 推进 native 控制节点与 async FINISH |
| 11 | 每个 AICore | [fill_own_slots](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_dispatch.h) | 领取 ready，必要时走有序准入，再准备自己的 payload |
| 12 | 每个 AICore | [local_ready_peek 与最终准入](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 查看 ready_head，许可通过才消费 ready 位并置为 EXECUTING |
| 13 | 每个 AICore | [execute_task](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) | 间接调用 `function_bin_addr(args)` |
| 14 | 每个 AICore | [scheduler_complete_own_slot](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h) | seal async、完成可选参数抓取和返回门禁后直接退休当前 slot，推进 join/依赖，再回到步骤 10 |
| 15 | AICPU supervisor | [run](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp) | 等整个图的 resolved 计数满足完成条件 |
| 16 | 双方 | [AICore 收尾](../../../src/a5/runtime/host_build_graph/aicore/resident_scheduler_impl.h) / [AICPU shutdown](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp) | EXIT → 清理调度义务 → nonce ACK → 窗口收尾 → GM release → 返回 |

步骤 1 与 2 是并行启动，不能理解为 AICPU 先完整执行完所有初始化，AICore 才开始运行。

```mermaid
sequenceDiagram
    participant CPU as AICPU
    participant CORE as 每个 AICore
    participant GM as GM 调度状态
    CPU->>GM: 清 teardown gates
    CORE->>GM: 写核身份和本轮报告
    CPU->>GM: 读取报告，补齐拓扑 context
    CPU->>GM: task=context；发布 READY
    CORE->>GM: 获取 context，建立本地状态
    CORE->>GM: 扫图、检查依赖计数、发布 ready、准备首批 slot
    GM-->>CPU: bootstrap_complete
    CPU->>CORE: DMB=IDLE
    loop 本核调度循环
        CORE->>GM: 维护 ready、服务控制节点、领取任务
        CORE->>CORE: 准入与构造 payload
        CORE->>CORE: READY → EXECUTING，调用 kernel
        CORE->>CORE: 返回收尾、EXIT/错误门禁、直接退休
        CORE->>GM: 发布 lane/block 完成并传播依赖
    end
    CORE->>GM: 批量汇总 resolved_task_count
    CPU->>GM: 检查完成计数
    CPU->>CORE: DMB=EXIT
    CORE->>CORE: drain 剩余调度义务
    CPU->>GM: 发布 exit_request
    CORE->>CPU: COND 回显 nonce
    CPU->>GM: 窗口收尾后发布 post_close_release
    GM-->>CORE: 允许返回
```

## resident 的编译边界

启动和模板选择保留在 `aicore_executor.cpp`；常驻循环与辅助函数位于 `resident_scheduler_impl.h`。
`resident_ordinary.cpp`、`resident_native.cpp` 及对应 `_profiled.cpp` 分别实例化普通/native 与采集开/关四种组合。
入口模板的 `CoreType` 参数区分 AIC 与 AIV 符号，内部实现保持 TU 局部链接；每个实例仍执行同一套协议。

[completion resolver](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion_resolver.h) 是显式 `static noinline` 边界，
局部参数类型由 `alignof <= 8` 断言约束。函数边界不改变 DONE、依赖推进、预算续行或 ready 发布合同。
构建方式与 device 调度行为应分别理解，不能仅由外联推断执行耗时下降。

## 编号与存储先分清

| 名称 | 含义 |
| ---- | ---- |
| `worker_index` / 入口 `block_idx` | runtime 的启动编号；用于 workers/context 数组索引 |
| `physical_core_id` | 硬件核编号；用于寄存器寻址，不保证连续 |
| `type_rank` | 在同类型核中的紧凑编号；业务 ready 队列按 AIC/AIV 共享，不按此编号分片 |
| `cluster_index` | 1 AIC + 2 AIV 所在组的编号 |
| `cluster_lane / self_lane` | 0=AIC，1=第一个 AIV，2=第二个 AIV |
| `task_ref` | resident 调度节点索引；不是物理核，也不是公开 TaskId 的完整编码 |
| `LocalContext.block_idx` | 当前逻辑任务的第几个 block；和入口 block_idx 不同 |
| `subtask_slot` | 当前任务的 AIC/AIV0/AIV1 子任务身份 |
| `ready_head` | 两个 own slot 中先发布的 READY 槽下标；许可未通过时保持不变 |
| `group_seq` | 一个 cluster 的 MIX group 序列，用于跨核到场与释放确认 |

`scheduler_state_base + xxx_offset` 是调度状态内地址；`xxx_address` 通常已经是完整 GM 地址。TaskPayload 中的 SelfRelativePtr 又不同：目标=字段自身地址+delta。

数据分为三处：GM 中的跨核状态、每个 AICore 私有的本地状态、用于生命周期的 DMB/COND 寄存器。普通 resident 任务的领取和完成不经过 AICPU 的逐任务寄存器投递。

业务 ready 只有两个队列：AIC 队列接收纯 AIC 和所有 MIX 任务，AIV 队列接收纯 AIV 任务。Graph 控制节点使用独立 control inbox。领取 ready 决定任务由谁接手；admission、SPMD block 领取和 MIX mailbox 决定哪些核参与执行。每个参与核都用自己的两个 slot 准备、执行和回收 payload。

## legacy 只标入口

- [AICPU 入口判断](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp)：`!aicore_scheduler_runtime_enabled(runtime)` 进入 `legacy_aicpu_execute`。
- [AICore 入口判断](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp)：mode 不是 `RESIDENT_PENDING` 或 `RESIDENT_READY` 时进入 `legacy_aicore_execute`。
- [mode 常量](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_layout.h)：resident 为 1/2，明确 legacy 为 3/4。0 表示未发布选择。
- 以下只走读 resident 路径。

导航时读 `scheduler_dispatch.h`、`scheduler_completion.h`；它们的同名 `.cpp` 是独立的 AICPU 调度实现，不在此次主线。
一些共享文件的开头注释仍以 AICPU 投递模型叙述；判断 resident 行为应沿上述实际调用点。

## 三个不能混同的完成点

1. `slot READY`：payload 准备完成；仍可能受最终 admission 检查约束。
2. kernel 返回：某个 block 的某个 subtask 返回；MIX、SPMD、异步条件可能仍未完成。
3. `resolved_task_count` 增加：逻辑节点的完成传播已走完；这是 supervisor 收敛判断使用的计数。

空 ready 队列只说明本轮没有可领取工作，不能作为整个图已经结束的证明。
