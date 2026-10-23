# A5 HBG SPMD 设计实现参考

本方案在 A5 HBG 每核 scheduler 上增加 SPMD，保留完成驱动的依赖推进。
核心合同是：**参与者同序，逐核推进；本核完成 A 的全部参与义务后，可进入独立的 B，
不等待 A 全局完成。不参与 A 且没有相关资源义务的核，不受 A 约束。**

共同顺序在运行时确定，不等同于 Host 提交顺序：A 尚未 ready、独立的 B 已 ready，
可以先统一发布 B，再发布 A。发布之后，任何参与者都不能自行颠倒这两个任务。
这里的“顺序门槛”不是全芯片执行锁，也不要求所有参与者同时开始。

代码基线为 f5cc679bb，约定更新于 2026-10-23。新状态及伪代码描述目标协议，
不表示当前 runtime 已实现同名接口。

阅读顺序：[运行时有序准入](spmd/ordered-execution-protocol.md)说明依赖、共同顺序与参与集合；
[block 与 Mix 执行](spmd/block-and-mix-execution.md)说明领取、局部结束和完成发布；
[边界条件和验证](spmd/corner-cases-and-validation.md)给出反例、测试矩阵和验收要求。
准备编码时继续阅读 [字段、状态机与接线契约](spmd/implementation-contract.md)
及 [可执行测试和当前红绿状态](spmd/test-plan-and-status.md)。

## 已确认约束和实施边界

### 已经确定

- Host 负责 orchestration、DAG 和内存规划；Scalar 不构图，不做运行期 heap 分配。
- 复用现有 ready/wake-list，由前驱完成推进依赖；不改成每核轮询整份 fanin。
- AIC/AIV 各自执行调度循环，不使用一个 AIV 常驻代管 cluster。
- Mix 允许 AIC 做有限协调；各执行 lane 自己准备 context 和调用 kernel。
- 调度层先使用 GM，不使用 SSBUF；原 kernel 的 TPipe 等通信方式不因此改变。
- ready 的独立任务可以先获准执行；共同执行顺序以一次统一发布的结果为准。
- 顺序只约束实际参与核及必要资源参与者，不限制无义务的其他核。
- 本核关闭 A 后可进入 B；真实 A→B 数据依赖仍须等待 A 的逻辑完成。
- 先原样跑通 CSA 的普通多 block／Mix 分波，再完成 sync_start。

### 普通多 block 与 sync_start 为什么分开

这一区分针对执行位置的分配方式，不是两套相互矛盾的任务顺序。

| 类型 | 示例 | 分配必须满足的条件 |
| ---- | ---- | ------------------ |
| 普通多 block | 8 个核执行 32 个独立 block | 可分波；同一核返回后继续领下一 block |
| sync_start | 8 个核执行 8 个相互等待的 block | 必须最终落到 8 个不同执行位置，不能同核串行排两个 |
| 超额共同驻留 | 8 个核执行 9 个相互等待的 block | 启动前拒绝，两个 pending slot 不会增加并发容量 |

建议普通任务先采用动态领取：空闲核领取下一个 block，快核可以多做几个。
sync_start 也可以动态领取，但每个执行位置至多领一个相互等待的 block，
且必须保证剩余 block 仍有足够的未领取参与者；固定单射映射是另一种实现选择。
两者都不必由 Host 写死归属，也不要求等所有核空闲后才开始。
**具体领取粒度与成员选择算法仍是实现建议，不是已确认的静态分配要求。**

### 首阶段的范围

普通纯 AIC、纯 AIV、多 block Mix 必须支持原逻辑 N，包括超过物理容量的分波。
Mix 的 mask 3、5、6、7 保留各自语义；mask 6 可以由 AIC 协调，但不能伪造 AIC kernel。

sync_start 在普通多 block／CSA 验收之后开放。Graph body、异步 kernel、
运行期扩图和多 block predicate 不自动获得支持；继续明确拒绝或沿用整图 legacy 回退。
要求 resident 的验收必须把任何回退判为失败，不能用正确输出代替调度来源检查。
已有严格“全体到齐后才执行”的 sync_start 调用合同仍须逐项核对，不能静默放宽。

## 当前代码与需要补齐的部分

### 可以复用的基础

f5cc679bb 已把 scheduler 数量改为实际 worker 数量，各核持有自己的普通 dispatch slots。
完成路径关闭 wake-list、推进剩余依赖、按 AIC/AIV/Mix 类型发布 ready；
同类型后继还可以直接填入本核空出的 slot，不必先回到队列。

ready 队列是按类型和 owner 分布的 CAS FIFO，支持分片内领取／偷取。
它不是一个所有核共享的全局顺序源：不同队列、direct refill 和 Mix 通道之间
没有 SPMD 所需的统一准入顺序。可复用这些工作发现机制，不能把 FIFO 当作顺序证明。

Mix 当前由 cluster 的 AIC 选择 placement、准备各 lane payload、聚合完成，
AIV 通过 SSBUF 接收任务。目标保留 AIC 的组级协调，把远端参数准备交回各 lane，
并将调度控制迁移到 GM。普通核的自调度能力不需要重新实现。

### 不能只放开形状门禁

- scheduler_resident_v0_task_shape_supported 仍要求 N 等于 1 且没有 sync_start。
- runtime_maker 以逻辑 block 数估计 worker demand，lifecycle 再与实际核数比较；
  普通分波不能沿用“全部 block 都需同时占核”的需求检查。
- 普通槽 LocalContext 仍预填 block_idx 为 0、block_num 为 1；独立 Mix payload 区还缺少
  context 后缀初始化，新用例已在 N=1 复现。两类槽都要完整初始化，再按 block 设置。
- Mix tracker、完成路径和 trace 索引仍不能直接表达一个 task 的多个 block。
- 多 block 的最后一次完成才可发布 TaskDone 和推进 wake-list，不能每个 block 都释放后继。
- 新增的 scheduler_runtime_mode 与 scheduler_mode_publish_count 可复用作回退门禁，
  不必要求所有功能测试先打开完整泳道。
- 旧 resident gang 脚手架及 scheduler_task_is_gang 已移除；不能把旧字段当作可接续实现。

## 目标合同和参考接线

### 身份与完成

| 身份 | 定义 | 关键区别 |
| ---- | ---- | -------- |
| task_ref | Host DAG 中的逻辑任务引用 | 当前 GLOBAL 表下标，不是运行顺序 |
| order_key | ready 任务运行时获准执行的统一序号 | 不由各核独立生成，不等于 task_ref |
| block_idx | 任务内逻辑 block，范围为 0 到 N 减一 | 分波时不重编号 |
| subtask_slot | AIC、AIV0、AIV1 的逻辑 kernel 槽 | 不一定等于最终物理 AIV lane |

核执行身份为 run_epoch、task_ref、block_idx、subtask_slot；通信槽另校验 generation。
LocalDone(w, A) 表示 w 不再有 A 的未执行工作、待发布完成或资源义务，
且之后不会再向它分配 A。TaskDone(A) 表示全部必要执行及输出发布已经完成。
两者不能混用；普通动态领取的具体 LocalDone 条件见 block 协议。

对于参与集合相交的任务，各核按已发布 order_key 履行自己的义务。
集合不相交的任务可独立推进；执行者还需满足真实数据依赖和局部资源合同。
任务参与集合一经发布便不再扩张，不能在某核越过 A 后又要求它回来参加 A。

### 分层流程

~~~text
前驱完成 → 现有 wake-list / ready 路由
                         ↓
              一次性发布任务准入记录
              共同顺序 + 冻结参与集合
                         ↓
          各核处理属于自己的最早未关闭义务
                         ↓
       领取 block / Mix 组配对 → 自备参数 → 执行
                         ↓
       发布本核／本 block 完成 → 可以进入后序任务
                         ↓
       最后一个 block 完成 → TaskDone → 推进后继
~~~

统一发布不等于全员投票：一个合法发布者提交完整记录即可，各核随后观察同一结果。
不设置全核到达确认，不用最慢核的前沿限制快核，也不等待 A 的 TaskDone 才发布 B。

### 接入清单

下列短路径位于 src/a5/runtime/host_build_graph/；普通主循环及快速路径必须一起审计。

| 位置 | 保留 | 补齐 |
| ---- | ---- | ---- |
| host/runtime_maker.cpp | 图、静态参数、元数据上传 | 预分配准入记录、block 控制、独立 context 与完整 trace |
| aicpu/aicore_lifecycle.cpp | 拓扑发现、每核 scheduler、退出 | 分波需求与共同驻留需求分开；校验有效成员 |
| runtime/scheduler/scheduler_ready.h | ready/wake 路由、CAS 队列、分片领取 | ready 不直接等价于多核执行许可；统一准入、身份及去重 |
| runtime/scheduler/scheduler_completion.h | 本核退役、依赖推进、direct refill | block 完成与逻辑 task 完成分层；refill 不能越过本核义务 |
| runtime/scheduler/scheduler_mix.h | AIC 的组级 placement 和完成跟踪 | 同 task 多 block、GM 控制、各 lane 自备参数 |
| aicore/aicore_executor.cpp | 每核调用 kernel 和生命周期 | 普通／Mix 双入口统一检查；不能普通 slot 永久压住必需 Mix |
| runtime/scheduler/scheduler_types.h 与 scheduler_layout.h | POD 布局、代次 | 准入序号、成员、领取／完成计数、溢出校验 |
| runtime/scheduler/scheduler_graph.h | 引用和形状校验 | 后端及测试齐备后分阶段开放形状 |
| runtime/scheduler/scheduler_memory.h | 当前设备原语及发布约束 | 每个新增共享字段明确 writer、reader、缓存和顺序 |
| host/runtime_maker.cpp 的泳道导出 | 调度来源、时间和 trace 导出 | 记录 block、order_key、组身份，区分等待原因 |

所有大小、乘积和字段窄化在 Host 校验；地址来自既有预分配存储。
运行时改变调度状态、领取 block 不是构图或 heap 分配，不改变 HBG 的职责划分。

## 实施顺序和验收门槛

| 顺序 | 交付 | 出口条件 |
| ---- | ---- | -------- |
| 1 | 保留当前每核调度基线，增加共同准入和纯 C/V 普通多 block | 分波不受 N 大于核数阻挡，执行身份不重不漏 |
| 2 | AIC 有限协调的 GM Mix 分波 | 同 block 配对，mask 3/5/6/7、局部资源和完成聚合正确 |
| 3 | 原样 CSA 的 resident 验收 | 原 kernel、原 block 数、七项 golden、完整执行身份和来源证明 |
| 4 | sync_start 的共同驻留与进度保证 | 容量合法；早到可开始；快核可进入下一 task；无循环等待 |
| 5 | 性能验证和后续扩展 | 先比较当前基线，再评估领取批量、成员选择和控制发布成本 |

完成、错误收敛、代次和必要的无 kernel 服务节点随相关步骤交付，不推迟到功能验收之后。
多 block predicate 等扩展单独开放；普通单 block 的已有 predicate 行为必须回归。

固定 CSA 样本包含 79 个有 kernel 的逻辑任务和 2 个无 kernel 依赖节点，
60 个多 block 任务，最大 N 为 512，预期核执行身份 2308 个。
这些数仅作固定样本核对，实际验收从本轮图推导。
运行入口见 [CSA 独立用例](../../../examples/a5/host_build_graph/deepseek_v4_pro_csa/README.zh-CN.md)。

普通多 block 阶段就必须证明跨 task 重叠，不等到 sync_start 才检查是否误加全局屏障。
两核普通 A 的 block 均已领取，一个核完成、另一个慢返回时，前者可进入独立 B。
之后的 sync_start 再验证相同性质，同时增加任务内等待参与者的条件。

## 代码依据与参考

- [每核拓扑和容量检查](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp)
- [Host 形状、需求和参数初始化](../../../src/a5/runtime/host_build_graph/host/runtime_maker.cpp)
- [ready/wake 路由和本核状态](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready.h)
- [完成与 direct refill](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_completion.h)
- [Mix 协调](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_mix.h)
- [设备主循环的普通与 Mix 入口](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp)
- [设备内存原语](../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h)
- [scheduler mode 验收 fixture](../../st/a5/host_build_graph/conftest.py)
- [LaunchSpec](../../../src/common/host_build_graph/submit_types.h)与 [legacy 范围领取](../../../src/common/host_build_graph/runtime_types.h)
- [sync_start 重试 ABA 调查](../../../docs/investigations/2026-07-sync-start-drain-retry-aba.md)
- [原 per-core 方案](a5-hbg-per-core-scheduler-plan.md)：保留历史对照；本次采用 GM 和 AIC 有限协调，不沿用远端 payload 代备。
