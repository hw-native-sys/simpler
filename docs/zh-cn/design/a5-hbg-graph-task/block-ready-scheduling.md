# 独立 block 调度与 MIX lease：后续优化候选

本文历史测量所引用的 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`，不随仓库分发；下文保留文件名用于追溯。

**状态：2026-11-03 用户决定撤回原型代码，恢复 `queue-ordinary-credit-v1` 的 T/H8 普通领取额度版本。**
该版本与 `queue-ordinary-credit-v1` 的 725 个冻结源码文件逐一一致，Qwen 历史 Device 中位数为 38.454 ms。
本页保留独立 block 调度思路，作为后续待选方案之一；不属于当前实现，也不自动进入下一轮实施。
当前执行协议仍是[原有 cooperative admission](protocol.md)，六种队列选型继续保留。

## 决策与实测依据

原型尝试将无 `sync_start` 的任务按实际执行资源独立调度，减少统一 admission 对普通任务与 SPMD 重叠的限制。
最终原型 `queue-work-pubdrain-v1` 的完整 Qwen out/K/V 校验通过，但有限的模型收益不足以接受构建成本：

| 指标 | 恢复版本 | 撤回原型 |
| ---- | -------- | -------- |
| Qwen Device 中位数 | 38.454 ms | 37.855 ms |
| Qwen Device p95 | 38.707 ms | 38.108 ms |
| A5 HBG AICore 冷构建 | 744.257 s（12 分 24 秒） | 2295.011 s（38 分 15 秒） |
| AIC+AIV `.text` | 6,896,152 B | 12,691,224 B |

模型时间仅比历史同卡旧版低约 1.56%，未做交叉采样；冷构建耗时为旧版的 3.08 倍。
两版使用同一 CCEC 构建和 `-O3 -g` 参数，只构建设备运行时；重新编译的旧版 `.text` 与旧冻结产物逐字节一致。
原型定向真机回归组还触发 600 秒 pytest 会话超时，完整组验收和完整 work L4 因果审计均未完成。
数据见编译对照（本地 `work-compile-comparison.json`）和
[原型性能记录](qwen-block-work.md)。上述历史数据不等于回退后重新测量。

代码膨胀已定位到三个实现问题：

1. 新 work 路径通过运行时判断叠入公共函数，旧 ordered/admission 路径仍参与代码生成。
   初始化时清空 admission 指针没有形成编译期裁剪。
2. lease、cohort、发布与完成处理大量内联进 bootstrap、主循环和 resident 三组函数。
   三组函数的 24 份架构/模板实例几乎包含全部 `.text` 增量；定义函数总数仍为 44。
3. 无 profiling 的 native 循环仍生成 work DFX 记录代码，运行时才检查 offset 并跳过。

这些是二进制、地址到源码映射及源码核对的结果；没有采集 compiler pass 耗时，不能断言某个 pass 占主要成本。
之前单纯给 lease/cohort 推进函数加 `noinline` 的试验未通过 MIX 真机验证，
不应直接重用，见[撤回记录](rejected-optimizations.md#work-protocol-outline)。

## 保留的调度思路

| 逻辑任务 | 候选调度单元 | 协作范围 |
| -------- | ------------ | -------- |
| 普通 AIC/AIV | 一个 block work | 一个匹配的物理执行位置 |
| 无 `sync_start` 的纯 AIC/AIV SPMD | 每个 block 独立 work | 每个 block 独立执行 |
| 无 `sync_start` 的 MIX / MIX SPMD | 每个 block 一个完整 MIX work | 对应 cluster 的完整成员组 |
| 带 `sync_start` 的任务 | 一个 cohort work | 全部规定执行位置同步启动 |
| ENTER、EXIT、DUMMY、inline 服务 | 不生成 kernel work | 保留依赖和控制节点服务 |

普通任务和 Graph child 使用同一契约。无 `sync_start` 的 MIX SPMD 中，每个 block 都是独立 MIX 任务；
其 AIC/AIV 子任务不能分别竞争资源后再互等。改变调度单元不改变 Graph 依赖、kernel ABI 或真实完成语义。

**优先领取 SPMD，不等待所有 SPMD 完成。** 原型在 AIC/AIV/MIX 各类型内设置 SPMD、普通两个优先级分区；
先尝试已发布且资源可满足的 SPMD，再尝试普通工作。同优先级保持有界轮转和跨组可达性。
全部 SPMD block 已被领取后，空闲位置可以运行普通任务。
已有 MIX lease、同步请求和错误处理优先于新领取；不能为新 SPMD 阻塞已经建立的成员协作。
该策略不承诺尚未发布的 work 的全局严格顺序，也不能反复撤销已准备的普通 payload。

## 原型的数据、发布和完成契约

Host 为 work 分配稳定身份，与逻辑 `task_ref` 分离：
`work_id -> {task_ref, block_idx, kind}`，每个逻辑节点对应一个 work range。
原型 unit 和 range 各 8 bytes，wire 数据只含索引/offset，按实际 work 总数 W 分配容量并检查溢出。
每个类型有两个优先级分区；T/G、T/H4、T/H8、L/G、C/W1、C/W4 共享语义。
原型使用 ready pool version=4；当前恢复版本的格式不能与它混用。

这种显式展开并非未来实现的唯一选择。逐 block 入队增加存储、队列操作及 MIX 握手成本，
重新评估时应比较批量发布/领取、range/chunk 表示和资源预约的摊销方式，不能只比较队列拓扑。
普通任务跨逻辑边界的合批在本原型中没有实现；发布预算上限 8 不代表实际每批均有 8 项。

发布必须维持以下不变量：

- 一个逻辑任务只有一个 READY 发布者，已发布前缀与私有后缀互斥且完整。
- 发布按有限预算服务，原型每次最多 8 个 work 或 inline 服务，预算跨逻辑任务使用。
- linked/chunk 的 RETRY 保留同一私有批次；保留批次装不下剩余预算时，下一轮优先给它完整预算。
- chunk 的 descriptor 数不等于 work 数。原型准备的 ID 连续，实际数量为 `next - pending.first`，
  发布预算、成功扣减和 DFX 区间必须统一使用该数量。
- 有私有发布后缀时可以暂缓新的 discovery；已有 lease/cohort、完成和错误处理仍须继续。
  不得设置“只要 pending 就禁止所有 kernel”的入口门禁，否则 MIX 成员可能互相等待。
- ticket 预留与发布在一次服务内完成；不得持有未完成的共享预留去执行 kernel。

领取授予唯一 work 所有权；执行代际用于拒绝重复 completion。
普通 block 完成后贡献一次计数，MIX 由协调者在全部成员释放后贡献一次；
仅最后一个实际 block 完成才将逻辑任务置 DONE 并解除后继依赖。
同步 cohort 由唯一 owner 在全体释放后完成逻辑任务，不能把领取完或队列空当作执行完成。

## MIX lease、同步 cohort 与退出

原型每个 cluster 一个活动 MIX lease，由 AIC 协调；其 descriptor 包含 run epoch、work、task、block 和序号。
协议按 `IDLE → RESERVED → ARRIVED → RUNNING → RELEASED → IDLE` 推进。
完整 cluster 的三个 lane 均参与到达/释放，inactive lane 不执行 kernel，但也不能漏掉握手。
不同 cluster 的独立 block 可并行推进，无需进入统一全任务 admission 顺序。

已准备的普通 slot 可以暂存；只有 lease 和 cohort 都允许普通 entry 时才恢复。
普通 kernel 的入口须重新检查资源许可，不能复用领取时的旧观察。
若有限 kernel 的许可先于 lease 发布，协调者必须等其返回并对新序号到达；旧确认不能代替新到达。
下一 descriptor 不得覆盖仍可能被旧成员读取的内容，协调者只在全部释放后退休 lease。

`sync_start` 独立使用 cohort 请求和成员确认：

1. 原型所有 worker 先排空规定资源，已建立的 MIX lease 优先完成。
2. cluster 的 AIC 先封住新的 MIX 接纳，AIV 观察对应 seal 后再确认排空。
3. 每个规定位置准备自己的 payload，全部准备就绪后才允许 kernel entry。
4. 全部执行/必要非执行成员释放后，唯一 owner 完成任务并退休 cohort。
5. 迟到的旧 cohort 退休观察不能让普通任务越过新 MIX lease；此交错必须有回归保护。

原型 lease mailbox 为 640 bytes，cohort control 为 384 bytes、每 worker ack 为 128 bytes。
信号采用隔离 cache line 上的单写者发布；同一信号行不混用 atomic RMW、cached store 和 dirty writeback。
A5 的实际发布/观察规则必须逐字段证明，不能仅依赖 CPU acquire/release 语义。

错误退出保留原始错误身份，不回滚已经执行的前缀，不伪造完成计数。
所有成员完成退出 fence 之后，才可复用 queue、mailbox、cohort 或同址新 run 的 backing。
work identity、发布/领取、lease/cohort、最后贡献者和退出状态都需要独立 DFX 证据；
局部 sequence 完整不能代替完整因果审计。

## 重新选择该候选的条件与资料

未来重新选择这一方向时，应先解决编译结构，再扩大功能接入：

1. 给 work 与旧 admission 建立明确的编译边界，避免两个完整执行体系同时留在 resident 热路径。
2. 将 profiling 编译期选择贯通到采集入口，让关闭诊断的实例真正裁掉记录逻辑。
3. 控制重型完成解析和退出服务的内联扩张；函数边界须验证 AIC/AIV 链接、调用 ABI 和实际真机行为。
4. 将冷构建、头文件修改后的增量构建、函数体积和运行性能同时纳入比较，避免再次出现三倍构建成本。
5. 重新验证 MIX 各合法 mask、SPMD/普通重叠、同步排空、迟退交错、错误与同址复用，
   完成原型未完成的回归与 L4 审计，再比较 Qwen 和代表性负载。

原型的阶段记录按功能和候选标识保留，不再要求 Git 历史保留分阶段提交：

| 历史阶段 | 稳定的证据入口 |
| -------- | -------------- |
| work 身份、Host 映射、work pool、发布/领取、MIX lease | `work-unit-primitives.json` 的对应验证区段 |
| 执行/cohort 接入 | `queue-block-work-v1`，见 `qwen-block-work-screen.json` |
| 身份采集与诊断 | `queue-block-work-dfx-v1`，见 `qwen-block-work-diagnosis.json` |
| 跨逻辑任务发布预算 | `queue-work-publish8-v3`，见 `qwen-work-publish8.json` |
| 发布优先与 chunk 重试修复 | `queue-work-pubdrain-v1`，见 `qwen-work-pubdrain.json` |

上述 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`。源码、二进制与输入身份由各证据中的 manifest 和哈希核对。
其中原始 commit 字段仅记录实验时的来源，不应据此要求合并后的分支还能 checkout 每个阶段。
完整旧开发历史另有本地归档；复现实验需要相应冻结产物，当前源码不代表已撤回原型。

原型入口包括 `host/ready_work_plan.h`、`host/ready_work_pool_plan.h`，以及
`runtime/scheduler/` 下的 `scheduler_work_unit.h`、`scheduler_work_ready.h`、
`scheduler_block_lease.h`、`scheduler_work_cohort.h`、`scheduler_work_dispatch.h`、`scheduler_protocol_work.h`。
这些文件属于历史原型，已经从当前代码中撤回。
尚未提交的 v9/v10 审计实现保存在本地 `outputs/a5-hbg-graph-task/rollback-to-ordinary-credit/`，不作为已验收功能。
本页保留原型的关键设计契约；详细测量和阶段验证见下列证据。

原型分阶段验证见测试记录（本地 `work-unit-primitives.json`），
性能全过程见[历史测量](qwen-block-work.md)。本次仅恢复旧实现和归档候选，不恢复自主优化循环。
