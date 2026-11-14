# 1. 初始化与启动数据

[总索引](README.md) · [结构总览](00-data-structure-scopes.md) · [下一段：发现与执行](02-ready-execution.md)

## 设备入口的输入

Host 已完成本轮 orchestration，持有独立的任务镜像、Graph readonly、业务 heap、Call/Binding、
调度控制及按请求分配的诊断 backing。launch descriptor 和 `Runtime.dev.scheduler_bootstrap`
给出 mode、worker context base、参与核数和本轮报告身份。
这些地址只证明输入的位置；READY 与 DMB 分别证明配置可读和 kernel 可以开始。

平台 kernel 进入 `aicore_execute`，先清本核 Scalar binding，再观察 bootstrap 与本核 Handshake。
AIC/AIV 报告自己的物理核号、类型和本轮 epoch。stamped run 在 epoch 标记前清理旧 READY/反向握手词，
以 barrier 将本轮报告和前一轮留下的值分开。并行的 AICPU 不执行 orchestration 或分派 kernel。

```text
aicore_execute(runtime, worker_index, core_type):
    observe bootstrap and own handshake
    stage physical identity and current report epoch
    publish report after barrier
    validate mode, context-base alignment and checked extent
    wait for this run's READY or failure-retirement signal
    acquire READY and observe current context
    enter resident scheduler
```

## AICPU 拓扑发现与配置

AICPU 分区接受报告，并从已验证物理身份找到寄存器窗口。每个窗口只有一个清退 owner。
leader 要求完整 cluster：一个 AIC 和两个 AIV，核 ID、类型、cluster membership 都不能重复或缺失。
它还检查实际 AIC/AIV 数能否满足图的资源需求；不满足时阻止 READY/执行。

每个 WorkerContext 包含 worker_index/runtime_worker_count、core_type/type_rank、三个 cluster worker IDs，
以及 graph、prepared、metadata、trace/PMU/args dump 的地址或 offset。
所有核均自行调度，bootstrap 按同一 worker 身份分区，不再发布独立 scheduler 角色和编号。
AICPU 发布实际 active_worker_count 与 watchdog budget；PMU 仅在本轮确已初始化时进入对应结算。

`publish_context_partition` 只回复已接受本轮报告且配置成功的核：
先写 Handshake.task=context_address，store barrier 后写 READY。
不能把一个未报告核当作可运行参与者，也不能从缺失报告推断它已经退休。

## READY 后的本核 bootstrap

本核重新观察 READY 的四行配置，校验 graph view v6、epoch、可信 extent、prepared Call 区、目录和 cluster peers。
WorkerContext 最后一行归本核诊断；其字段不会与配置写者并发覆盖。
普通/native Graph 使用相同 bootstrap barrier：

1. 用 worker_index/runtime_worker_count 划分 expanded node 区间。
2. 初始化必须的 admission 记录；cluster coordinator 清本轮 signal。
3. 原子增加 bootstrap_scan_arrived_count，最后一个核发布 bootstrap_scan_complete。
4. 屏障后初始化两个 FREE slot，运行有界 control scan 并尝试准备首批工作。
5. 原子增加 bootstrap_arrived_count，最后一个核发布 bootstrap_complete。

Host 已生成所有 child 参数和 Call；bootstrap 不建立动态 wake-list、不填后继 ready 队列，也不在设备绑定参数。
每个 worker 是否已准备到 kernel 不影响其必须到达 barrier 的责任。

AICPU 等待 bootstrap_complete 或首错，随后按分区写唯一 DMB launch gate。
等待期间仍服务 bounded args dump 请求。核在 gate 打开前可观察 Call、依赖和控制状态，但不能调用业务 kernel。

## 无效启动与清退

descriptor 的 mode、地址、对齐、加法/乘法 extent 或 epoch 无效时，
AICore 保留本轮报告身份，进入 EXIT/nonce/teardown/post-close release，不能读取未发布 backing。
配置或分区初始化失败时，AICPU 先 join 已启动的 phase owner；已发过 READY 或放行的分区仍需完成清退。

首错和清退结果分别保留。缺失 ACK 不证明寄存器窗口已停，
失败 terminal 不能归还仍有借用者的 backing。启动错误没有另一个执行器可回退。
