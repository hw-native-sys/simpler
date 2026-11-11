# A5 HBG 两个 ready 队列设计与开发参考

**状态：实现及本轮 A5 验证完成，未测得显著 L1 加速。** 核对基线为 `619051ea2`，日期为 2026-11-11。
18 组模型功能通过；14 组 L1 合计变化 −0.56%，按 ±5% 标准视为无变化。
完整三臂 L1/L3、边界项复测及验证范围见[实测报告](../../../../tests/docs/a5-hbg-graph-task/two-ready-queues/README.zh-CN.md)。
目标是让 A5 HBG AICore resident scheduler 只保留两条业务 ready 队列：
纯 AIC 与 MIX 共用一条，纯 AIV 使用另一条；不设置分片。
每个消费者只探测自己对应的队列，保留现有 SPMD、MIX、异步完成和退出语义。

本方案减少队列发现与路由的复杂度，但将同类型生产者、消费者集中到共享队头和队尾。
是否改善性能取决于扫描节省与共享竞争的实际差值；它不构成 SPMD 切换从 50 µs 降到 5 µs 的保证。
当前实现及历史对照见[ready 队列设计](ready-queue-design.md)。

## 1 目标范围与数据布局

### 队列类别与执行资源分开

| ready 队列 | 接收任务 | 合法消费者 | 领取后的执行方式 |
| ---------- | -------- | ---------- | ---------------- |
| AIC | 纯 AIC、AIC SPMD、所有 MIX，包括纯双 AIV MIX | 合法 AIC coordinator | 根据原 metadata 进入 AIC 或 MIX 路径 |
| AIV | 纯 AIV、AIV SPMD | AIV | 根据原 metadata 进入 AIV 路径 |

生产者不受自身核类型限制：AIV 完成后可以向 AIC 队列发布后继，反之亦然。
MIX 路由到 AIC 队列，只表示由 AIC 负责领取与组织，不改变 active/resource mask。
每个逻辑 SPMD/MIX 任务只入队一次，block 和 lane 不分别入队。

建议引入独立的 ready 类别，名称仅为接口设计示意：

```cpp
enum class SchedulerReadyClass : uint32_t { AIC = 0, AIV = 1 };

// 调用前已验证任务 executable，且 active mask 与任务 flags 合法。
return scheduler_task_is_mix(flags) || active_mask.has_aic()
           ? SchedulerReadyClass::AIC
           : SchedulerReadyClass::AIV;
```

不能将所有原有 `SCHEDULER_MIX_QUEUE` 引用机械替换成 AIC：部分位置还承担
MIX 专用 direct candidate、放置或执行语义，必须依据任务 flags/mask 保留这些判断。
Graph 控制节点的 publication kind 与业务 ready 类别使用独立标签，避免编号碰撞。

“两个队列”只约束业务 executable ready pool。Graph ENTER/EXIT/DUMMY 控制服务、
MIX mailbox、async CHECK/FINISH 目录、本地 slot/candidate、错误与退休状态继续存在。
原有有界私有 candidate 是已领取任务的所有权状态，不新增本地 ready 队列、隐藏分片或 stealing 层。
用户前端表达、计算 kernel、kernel ABI、原始任务身份不改变。

### Host 布局

核对基线完整 32 AIC + 64 AIV 拓扑有 4/8/4 共 16 个分片。实现布局在所有合法拓扑下都只有
2 个 descriptor 和 2 个 queue control，容量允许为 0：

```text
capacity[AIC] = 可能入队的纯 AIC 逻辑任务数 + MIX 逻辑任务数
capacity[AIV] = 可能入队的纯 AIV 逻辑任务数
```

普通任务与展开后的 Graph child 使用同一分类器。容量覆盖所有可能入队节点；
只有能证明不会入队的节点才可排除，不用实际 block 执行数计算容量。
每个节点一轮最多占一个 entry；已领取 entry 不在 run 内覆盖、回收或重新入队。

| 布局部分 | 目标合同 |
| -------- | -------- |
| Header | 保留版本、任务数、两队列寻址、依赖区和 allocation 边界；删除分片语义字段 |
| Descriptor | 固定两份，各自保存 entries offset、capacity；偏移使用 allocation 相对 offset |
| Control | 每队列独立 head、reserved_tail；两者分开到 64B cacheline，并与其他队列隔离 |
| Entry | 发布 task_ref；初始为 INVALID，发布后本轮不变；发布协议见下一节 |
| Run 生命周期 | 初始化在 READY 前完成；所有 worker 退出和记录发布完成后才能复用 backing |

删除 `group_counts/queue_begins/group_width/ref_chunk` 与 ref 哈希分组。
队列数量固定为 2，不保留可配置 group_count，也不通过每核实例变相恢复分片。
Header 的实际大小以 `sizeof/offsetof` 检查为准，数据布局变化同步更新 wire 版本、
Host writer、device reader、导出器与测试；错误版本明确拒绝，不新增历史格式兼容分支。

## 2 发布领取与进展协议

### 发布与唯一领取

推荐沿用 HBG 有限任务集的 append-only ticket 结构，避免增加环形覆盖与运行中回收协议。
队列操作必须有界，不持有共享预留位置去等待 peer、执行 kernel 或等待 MIX 资源。

1. resolver/bootstrap/控制服务确认逻辑任务 ready，并取得唯一发布责任。
2. 按 ready 类别预留位置；同一批只包含一个类别。批量上限沿用现有有界预算，不等待凑批。
3. 在不返回通用调度循环的短路径中完成预留范围的发布；若用可续行状态，必须持久保存同一范围和发布游标，不能带着共享空洞去执行 kernel。
4. CAS 竞争失败重试不得重新发布同一任务、重新预留已拥有的范围，或丢弃未发布后缀。
5. 消费者只对已发布的 entry 尝试推进 head；成功 CAS 才授予唯一私有 claim。
6. 领取后根据 metadata 走普通、SPMD 或 MIX 路径；领取不等于 kernel 获得执行许可。

队列位置的顺序不等于 kernel 的全局执行顺序。依赖、admission、资源许可及 async DONE 仍是权威条件。
重复 task 发布由原有唯一完成/发布所有权协议防止，不能仅依靠 entry 不覆盖来证明。

### A5 内存协议与现存风险

硬件依据为工作区 `simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`。
atomic 交权、数据发布屏障和 cacheline 所有权需要分别证明；DSB 不是跨核 barrier，
也不能认为 C++ acquire/release 标签自动给出 A5 数据可见性。

核对基线的 `scheduler_ticket_publish` 直接使用 `st_dev` 发布 ref。
手册已记录 repeated-store 终值回退并将该写路径列为待迁移路径，不能把现有代码或少量通过的用例当作正确性证明。
两个队列的最终协议不得以“单写者 + st_dev + DSB”作为 entry 交权依据。

安全发布参考如下，具体 helper 需以同构 A5 探针确认：

- head、reserved_tail 和 entry 发布字使用 atomic 访问完成交权，不与 ordinary store/DCCI 数据混行。
- entry 可将 ref 本身作为 atomic 发布字，保留 INVALID 哨兵；先满足数据发布顺序，再原子发布 ref。
- 初版保守布局让每个 entry 发布字独占 64B。将多个 entry 紧凑到同一行需另行证明整行 atomic-only
  且生命周期一致；不能以缩小队列为由让不同 writer 对同一行做 cached 写回。
  Host 和 device 按相同 entry 类型/stride 寻址，仅对齐整个数组的基址不构成逐 entry 隔离。
- `ld_dev` 仅作可过期的预探测。正提示不能授权执行，正式领取前对 entry 作权威 atomic 确认，
  再 CAS head；取得 claim 后继续原有 payload/输出可见性协议。
- 所有数据先发布、控制后交权；不对 atomic-only 行做普通 DCCI 写回。

按保守 entry 布局，存储约为 `64 * N_ready + 2 * sizeof(QueueControl)` 加 Header/Descriptor 和对齐，
不能宣称减少队列数必然减少总内存。此项会增加原子开销，必须与拓扑收益分开测量。
若先修正 entry 发布 primitive，应作为独立前置改动，并让分片基线与两队列候选使用相同 primitive、
entry stride 和屏障。另保留未经修正的原始基线作总效果参照，分别报告两次差值。

### 每次探测只访问一条业务队列

```text
may_have_work(queue):
    capacity == 0                         → false
    读取 head 提示
    head > capacity                       → true，交正式检查处理异常
    head == capacity                      → false
    head 位置的 entry 提示不是 INVALID     → true
    否则                                  → false

try_pop(queue):
    在现有有界尝试次数内读取 head 和 entry 提示
    未发布                                → NOT_READY
    权威确认 ref，检查范围和 ready 类别
    CAS(head, observed, observed + 1) 成功 → SUCCESS(ref)
    尝试次数耗尽                          → RETRY
    权威检查发现非法状态                  → ERROR
```

正式读取和确认必须遵守上一节内存合同；伪代码没有省略必要 acquire/barrier 的授权含义。
NOT_READY/RETRY 不等于图已完成；探测负结果只在本次有限调度机会内用于跳过 discovery。
预探测可以过期，不能缓存为持续空队列证明。实现可在同一轮复用负提示，正提示不缓存成所有权。
“一条队列”不承诺一次 GM load：head、entry、权威确认和 CAS 是不同访问。

### 队列为空仍须推进协作和异步任务

每个 worker 的顺序保持为：本地完成/控制服务 → 检查本类 ready → 已提交的
admission/MIX 进展 → 本地可执行 slot → async 有界服务 → 必要的空闲退避。
具体相对顺序沿用执行器的合法边界，不在本方案中重排业务完成和错误处理。

尤其保留以下反例对应的路径：

- AIV 队列为空，但 AIC 已发布 MIX assignment：AIV 必须观察 mailbox 并参与 arrive/release。
- SPMD task 已被某核取走、队列已空：其他成员仍须观察 admission 并领取其 block。
- 所有 kernel 已返回，但通知未到：CHECK/FINISH 仍须服务，不能发布 task DONE 或结束 run。
- 本核已持有 candidate 或有待提交 admission：继续推进所有权，不能因共享队列为空遗失任务。

纯 AIC 与 MIX 合并后，不在共享队头等待 MIX 可安置。当前 ordered 路径的
`placement=nullptr` 行为可作为入口：先取得逻辑任务，再有界推进 admission/MIX。
仍使用前置 placement 的调用方必须逐个适配，不能把非 MIX ref 传给 MIX 专用规划器。
原有两个候选的额度、ordinary 持有约束和已建立协作的优先权继续生效，不增加批量私有囤积。

## 3 代码改动与开发边界

除明确标注仓库根目录的路径外，下表路径相对 `src/a5/runtime/host_build_graph/`；
函数名对应核对基线，实施时重新核对调用图。

| 位置 | 需要的改动 |
| ---- | ---------- |
| `runtime/scheduler/scheduler_types.h` | 分开 ready 类别和执行类型；检查三类常量、数组、control 标签及 MIX 专用判断 |
| `runtime/scheduler/scheduler_ready_pool.h` | 固定两队列寻址与验证；删除组宽、哈希分组和组起点，更新 wire 合同 |
| `host/ready_pool_plan.h` | 按合并类别统计容量并初始化；零容量和非完整合法拓扑也只有两份 control |
| `runtime/scheduler/scheduler_ticket_queue.h` | 保留有界 reserve/pop 和一次性 entry 生命周期；独立落实安全发布 primitive |
| `runtime/scheduler/scheduler_ready_publication.h` | 按两类合批与发布；保留控制节点入口、唯一发布和可续行状态 |
| `runtime/scheduler/scheduler_ready.h` | 单队列 probe/claim；去掉 own group、remote-first、victim cursor 和 pool_poll_counts 等无用状态 |
| `runtime/scheduler/scheduler_dispatch.h` | 删除 AIC/MIX 两次 ready 搜索；保留候选、admission tail、MIX mailbox 和执行许可检查 |
| `runtime/scheduler/scheduler_completion_resolver.h` | 审核所有发布/direct handoff 入口，按原资源类型判断可直接接续，不能只看合并后的 ready 类别 |
| `aicore/resident_scheduler_impl.h`、bootstrap 与 Host 布局调用方 | 同步删除失效 ready 游标与初始化；不改变 async 服务、执行入口和退出证明 |
| `host/scheduler_protocol_export.h`、仓库根目录的 `simpler_setup/tools/hbg/` 及相关 reader | 区分 queue class 与实际 MIX/AIC 执行身份；同步更新布局、事件、验证器和文档 |

不能只把 `group_counts` 改成 1。当前 `scheduler_ready_pool_claim` 使用
`own = self_type_rank() / 8`，再判断 `own >= groups`；直接改数量会让多数核拒绝领取。
也不能将 MIX queue ID 简单别名成 AIC 后保留原来的双路循环，否则会重复探测同一队列，
并可能破坏 direct candidate、统计数组或 control publication 的含义。

代码审计至少覆盖 `SCHEDULER_READY_QUEUE_COUNT`、`SCHEDULER_MIX_QUEUE`、
`scheduler_task_ready_queue`、`scheduler_ready_queue_holder`、`group_counts`、
`queue_begins`、`ref_group`、`victim_cursors` 和 `pool_poll_counts` 的全部读写者。
本轮不改变 SPMD block 分配、完成汇合、MIX 协议、sync_start 或模型 kernel，
避免把队列拓扑与其他调度算法的效果混在一起。

不在运行时保留旧拓扑开关或历史 reader。基线以冻结源码/构建保存，与候选分开运行。
纯 AIC/AIV、MIX、native Graph 和 async 是仍需支持的功能，不作为历史兼容逻辑删除。

## 4 性能风险与已有证据

| 风险或收益 | 影响范围 | 开发与验证要求 |
| ---------- | -------- | -------------- |
| 少扫描空分片 | 稀疏 ready、短依赖链 | 检查 probe 次数、ld_dev 数和 exclusive 耗时，不按分片数线性预测收益 |
| head/tail 同址争用 | 多任务同时 ready、多核同时领取 | 观察 CAS 失败、reserve/pop 延迟和有效领取率；返回值不是 kernel 启动许可 |
| 预留后未发布的空洞 | 整个 AIC/MIX 或 AIV 队列 | 暂停早期 publisher、完成后续发布，证明不会越过空洞、重复领取或停止其他服务 |
| AIC 与 MIX 发现次序耦合 | MIX 启动延迟、AIV 利用率 | 混排长 AIC 与 MIX，检查 ready 到 admission/执行的延迟，不强加新的优先级队列 |
| 较快核连续抢到任务 | 私有持有与资源分布 | 保留候选额度；不承诺每核均分，验证没有任务遗失、协作停滞或无限私有囤积 |
| 单队列提示误作全局完成 | async、MIX、SPMD 与错误退出 | 用显式状态交错验证队列空时仍有进展；最终完成仍沿原计数和退休协议 |
| 简化布局但增加发布成本 | atomic entry、cacheline 隔离 | 同 primitive 同布局比较拓扑，披露额外原子次数、内存及完整 Device 耗时 |

历史 T/G 为 AIC/AIV/MIX 各一队列，共三队列；它不是本方案的两队列。
旧 Qwen 同设备、同 AICore 二进制对照中，T/G 的 Device 中位数为 49.496 ms，
T/H8 的 4/8/4 分片为 40.642 ms。它说明取消分片存在已测回退风险，
不能外推为当前两个队列必然失败或必然回退相同比例。
参见[历史对照与边界](ready-queue-design.md)、
[已撤回优化](rejected-optimizations.md)和[SPMD 策略取舍](../../investigations/2026-11-a5-hbg-dsv-scheduler-review.md)。

atomic_probe 当前硬件 SyncAll 后的同址读测量中，64 个活动核的
`ld_dev` 整波中位数为 1.298 µs，`atomicAdd(0)` 为 10.975 µs。
这些是混合拓扑、无并发 writer 的探针结果，且使用 -O3，不是 64 AIV 的实际 pop 延迟，
也不能作为真实队列耗时预测。它仅支持“同址集中访问有成本”的风险判断。

不把一次现有 L3 图中的全部空泡归于队列，不将其他核同时轮询的区间相加。
若后续引入减少搜索者、批量领取或直接接续，应作为独立候选验证；不暗中增加分片，
也不将不同算法组合的效果称为“两队列本身”的收益。

## 5 验证矩阵与实施顺序

### 功能与协议验证

| 验证项 | 必须证明的结果 |
| ------ | -------------- |
| 路由与容量 | 普通/Graph child、双 AIV MIX、零任务/单类型任务、合法非完整拓扑分类正确，entries 总容量按逻辑任务计算 |
| 全核可达性 | 包括旧 rank/8 大于 0 的 AIC/AIV 都能领取；AIV 不能领取 MIX，不能因合并类别丢失 MIX mask |
| 并发发布和领取 | 同队列多 producer/consumer 恰好一次，预留范围不重叠，最后一个 entry 竞争不会越界 |
| 空洞与暂停 | 前序预留暂停时后序不可被越过；恢复后可推进；不以定时 sleep 或“足够慢”构造正确性交错 |
| 资源未就绪 | MIX 被领取后 AIV 暂忙，既有候选/slot 满时仍有界服务，不在共享队头等待资源 |
| 队列为空仍有责任 | admission、MIX assignment/ACK、CHECK/FINISH、待提交候选继续得到服务 |
| SPMD | 小 grid、大于核数、多波、非整除 grid 与 sync_start 保持 block 身份、执行数量和真实完成 |
| 异步与控制节点 | 计算关闭与最后通知并发、延迟通知、Graph ENTER/EXIT、predicate skip 和多实例不提前 DONE |
| 错误与复用 | 错误版本/offset/ref/capacity 拒绝；错误不伪造完成；同址新 run 不读旧 entry，原图和 trace 安全退休 |

优先扩展现有 `tests/ut/cpp/a5/runtime/host_build_graph/` 下的
`test_hbg_ready_pool.cpp`、`test_hbg_ticket_queue.cpp`、`test_hbg_scheduler_dispatch.cpp`、
`test_hbg_admission.cpp`、`test_hbg_ordered_execution.cpp` 及现有 async/Graph 协议测试。
使用 latch、gate、可观测原子状态安排交错，超时只用于终止失去进展的测试。
CPU 检查不能代替真实 A5 的 cache、atomic、MIX 和通知验证；真实模型 kernel 不替换为模拟忙等。

### 性能与 DFX 验收

- 只构建 A5 onboard，使用已优化编译时间的正常 `-O3 -g` 配置，不再使用 DEV；保持原 `CMAKE_BUILD_TYPE`，不额外添加 `Release/-DNDEBUG`。基线与候选的依赖、Host runner、
  kernel、输入、设备和 affinity 保持一致，保存 commit、dirty patch 与二进制哈希。
- 使用[生产调度器的 dsv41_scheduler](../../../../tests/fdwic/dsv41_scheduler/README.md)。
  18 组保留功能覆盖；性能决策采用 6 个 attention 场景加 Engram、两种实现共 14 组单卡场景。
  MoE 与含跨卡 MoE 的 19–21 层共 4 组也采集 L1/L3、报告波动范围，但仅作参考，不纳入优化收益考核。
- Attention/Engram 只用一张卡；两卡只用于原本需要通信的功能用例。
  每版每级预热 1 次、正式 10 次，报告全部样本和中位数；必要的 A/B/A 控制每臂也不超过 10 次正式计时。
- 主指标是 L1 捕获的完整 AICPU 设备端入口至退出时间，明确实际时间戳边界。
  不用 Host 时间、首末 kernel 窗口或跨核 phase 累加值替代完整 Device 耗时。
- L3 用于解释差异，检查 probe、claim、发布、CAS 重试、admission 和 MIX 进展。
  继续逐次记录 atomic、ld_dev、DCCI、DSB，不靠采样、丢记录或改变计时边界制造收益。
  记录 raw 未压缩字节数及 L3/L1 膨胀，目标增幅不超过 50%，文件大小优先级低于耗时。
- 若诊断数不足，只添加必要的分核计数/已有 DFX 导出，禁止 AICPU/执行器热路径打印；
  基线与候选使用相同观测配置，完整记录零丢失、零截断。
- ±5% 以内视为无明显变化。分别披露 PyPTO-lib、DSL、14 项合计和单项回退，
  合计是独立用例中位数之和，不等于完整 forward。综合有明确收益才作为性能优化保留。
  仅简化代码的价值单独说明，不能把它记作 Device 加速。

### 开发交付顺序

1. 冻结基线与模型输入，核对所有三类/分片消费者，完成路由和并发反例测试设计。
2. 若安全发布 primitive 尚未完成，先独立实现并用同构 A5 用例验证；生成相同 primitive 的拓扑对照基线。
3. 一次性迁移 Host 布局、生产者、消费者和 DFX reader 到两队列，删除失效分片状态，保持其他调度策略不变。
4. 通过协议测试和真实 A5 功能后，按上述 14 组、10 次、相同正常优化构建口径采集 L1/L3 对照。
5. 每个有明确收益的实现独立本地 commit，提交中记录功能覆盖、测量边界和回退项；未经用户允许不 push。

完成定义是：实际运行只有两条业务 ready 队列、所有核可达、原语义不变、证据可复核。
本轮已完成上述范围的实现和验证；保留价值是结构简化，性能按持平记录，不宣称通用加速。
