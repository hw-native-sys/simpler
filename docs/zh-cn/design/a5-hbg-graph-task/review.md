# A5 Graph task 设计复审

本文历史测量所引用的 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`，不随仓库分发；下文保留文件名用于追溯。

本页记录[方案](../a5-hbg-graph-task.md)在 `main@8f42cef9b3f0ebbda800e239aab0949e95552f25`
上的源码核对、反例检查与设计修正。初始核对日期：2026-10-31；ready 队列增量复审：2026-11-03。
“设计已闭合”表示契约中已有明确处理，不表示代码已实现或硬件验证已经通过。

## 复审结论

采用静态拓扑展开、共用 resident 调度和按逻辑任务一次绑定，可以避免新增 AICPU Graph 物化调度器，
也避免引入第二个 cooperative 顺序域。该方向与主线 scheduler 的所有权和完成模型兼容。
但它不是仅“把 Graph 放开”即可完成的局部修改：Host 载体、所有 ref reader、控制节点资源独立性、
参数跨核可见性和退出证明必须同时落实。

当前设计层已处理下面的阻断反例；仍需通过实现、并发测试和真实硬件证明其实现正确性。
性能优势尚未证明；bootstrap 并行参数准备候选把复制移出 ready→kernel 临界路径，但增加启动工作量，
仍为 O(N) 扫描，
旧 execution tail 仍然预留，不能把预期写成实测收益。

全局/分组 ready 池复审另外核对了真实 metadata 容量、offset 解引用前的可信边界、
head/tail 与 entry 的缓存行隔离、未发布 reservation 的进度限制、唯一领取及 MIX placement 顺序。
发现并修复本组成功领取重置远端游标的问题；新增多组持续非空用例在旧代码上失败、修复后通过。
这不改变 dependency、admission 或 run fence 的完成条件，队列暂时无可领取项也不能证明 run 已完成。
具体协议、业界方案边界和版本化实测见[队列设计](ready-queue-design.md)及
候选证据（本地 `ready-pool-candidates.json`）。

算法层复审进一步区分 ticket 拓扑比较与不同并发协议，增加以下接入约束；这些不是已经通过的 runtime 验收：

- 链式 FIFO 的挂链 CAS 成功即移交，tail 更新失败不能重复发布；未成功挂链的 RETRY 不能沿用 bool false 的 abort 路径。
- 私有 pending 必须跨调用持久保存，允许只在仍独占的链尾合并新任务；进入长 kernel 前需要有界服务与 backlog 策略。
- 消费允许越过滞后 tail 的简化依赖全 run 不回收、不复用节点；新增复用会撤销无 ABA 的依据。
- chunk 的 head 前移不授予 task 所有权；只有块内 cursor CAS 成功才领取对应前缀。旧窗口仍须走同一 CAS，不能因 head 已移过而直接执行。
- chunk 的首 ref 唯一性来自任务只发布一次；容量按 N 个 descriptor 预留，不能按每批最多 8 项反推只分配 N/8。
- FAA 消费作废空 slot 后，入队尝试数可超过任务数；现有精确 N 容量证明不能沿用到 FAAArrayQueue/LPRQ。
- 聚合代理降低共享原子流量的同时引入等待；CPU 上的 blocking 代理不能直接进入 AICore 协作调度循环。

L/G 已接入持久 pending 与 bootstrap/dispatch 服务。复审核对了正常完成仍依赖逻辑计数、
错误轮询在 pending 分支之前、挂链不等待 slot/peer、links 与 next_waiter 分离，以及 pool offset 验证。
生产池 CPU 测试包含 full slots 下 pending 合并、三个类型同时 pending、abort 与静默期重新初始化。
这些测试不等价于实际设备 bootstrap/协作/退出交错，runtime 和模型验证仍开放。
具体状态机与未完成验证见[算法比较](ready-queue-algorithms.md)。

## 阻断反例与修正

严重性表示若按不完整方案实现会产生的后果，不是宣称主线已有相应 bug。

| 编号 | 严重性与触发反例 | 设计修正与验收条件 |
| ---- | ---------------- | ------------------ |
| R1 | 正确性：child ref 直接寻址原始 320-byte SM，读到别的任务或越界 | N/T 与 TaskId 分离；统一 accessor，所有映射 UT；raw accessor 拒绝 mapped child |
| R2 | 正确性：只等待最后记录 child，另一不连通分支仍在写输出 | EXIT 等待所有 sink；验证完整 DAG，测试多 sink、慢 sink、DUMMY sink |
| R3 | 死锁：ENTER/EXIT 借 kernel slot，slots 全被待协作任务占用 | 独立 control inbox，每轮有界服务，slot 全满/暂存 admission 压力用例 |
| R4 | 丢唤醒：bootstrap 尚未链接 next 就完成 ENTER | scan_complete 前禁止 control pop；稳态使用 CLOSED-aware CAS，分别验证两个阶段 |
| R5 | 数据破坏：多 block 同时绑定参数，或不同 child 写回同一 scalar line | 唯一 bootstrap 分区 owner；每 child 对齐 scalar/predicate；跨核同地址压力验证 |
| R6 | 提前执行：参数写回未完成就打开 gate，reader 看到半成品 | 参数 writeback+barrier 先于 scan arrival；gate 后普通 owner 和成员首次 invalidate |
| R7 | 死锁/吞吐：持有全局 commit lock 绑定大量参数，或每 Graph 独立 admission 域 | bootstrap 在无 kernel 资源占用时准备参数；共用 runtime order 与 membership |
| R8 | 提前释放：EXIT DONE 时某 producer 仍在 resolve 其他 waiter，或 MIX lane 未 release | 节点计数位于 resolve 尾部，另外保留 LocalDone/ordered-exit/run fence |
| R9 | 假完成：错误清理推进计数，被 supervisor 判成功 | count equality、error==0、完整退出证明同时满足；late error 回归 |
| R10 | 悬空地址：plan 引用局部 GraphHostState；retry 沿用上次 GM base | 自有无地址 plan、commit 时实例化；park/resume/pipeline 隔离测试 |
| R11 | 不可访问：把 AICPU LaunchEnvelope 地址给 AICore | native Definition 独立 GM backing；A5 不从 AICPU LaunchEnvelope 读取 Definition |
| R12 | 容量：直接沿用顶层 128 fanin 上限，EXIT 有 1024 sinks | Graph accessor 用宽 count/cursor；完整 checked capacity，不截断、不静默跳边 |
| R13 | 内存误报：不写 320N storage 就声称释放其空间，或重复分配 args | 复用已预留 tail，报告真实 reserved_heap；未来 allocator 优化单独证明 |
| R14 | 进度误判：大 Graph shell 未完成或 worker 持续忙，AICPU 看不到内部进度 | 按真实 node resolved 监督，定期 flush 非空局部计数；不发送假 heartbeat |
| R15 | 漏测：oracle 直接读取被测 planner 的错误边；DFX 混淆 shell/ref | 独立外部图+原始 Definition oracle，新 schema 携带两种身份 |
| R16 | 假收益：实际执行器或产物身份不符、无 Graph 集混入 Graph，或仅测试 CSA 子算子 | 每 case 保留 Graph/mode 证据；现有 A5 Qwen 完整模型与同卡主线对照，CSA 按完整负载验证 |
| R17 | 越界：先解引用损坏的 header offset，再用 header 自报长度验证自身 | view 携带 allocation layout 的实际 bytes；先验 header 范围，再核对 image_bytes；坏 offset/长度 UT |
| R18 | 缓存：local config 校验 native header 时，bootstrap 的整 cache observe 尚未发生 | native 首次读取前完成本 run 的 bootstrap observe；保持无 Graph 时序；同地址新 epoch 回归 |

R1–R18 已纳入数据布局、协议与验收契约。实现与分阶段证据见
[实施记录](../../../../tests/docs/a5-hbg-graph-task/README.zh-CN.md)；当前已有拓扑、绑定、协议与局部设备回归，
完整功能和性能门禁尚未完成，不能将这些风险统一标记为已关闭。
不把某个反例的逻辑分析当作对应 device regression 的复现证据。

## 推理与模型检查边界

### 全 sink join 的充分必要性

有限 DAG 中每个节点沿后继都能到达至少一个 sink。若所有 sink 已完成，且依赖只在 producer
逻辑完成后释放，那么每条通向 sink 的祖先链都已完成，覆盖全部节点。
反方向显然成立。因此 EXIT 对所有 sink 的 join 可以替代每 child 一次的 Graph remaining 原子。
前提包括：所有依赖保留、无环、所有实际输出工作包含在 TaskDone 之前。

最小反例是 body 有 `0→1` 和孤立 `2`：记录顺序最后的 `2` 完成时，`1` 仍可未完成。
所以“最后提交 child”“最后一个 output producer”都不能替代完整 sink 集。
root→ENTER 的反向写法也不正确：应为 ENTER→root，root 的外部 gate 不由参数是否准备决定。

设计检查枚举了 1..6 个拓扑编号节点的全部 33,867 个有向无环边集，以及 462,033 个依赖闭合的完成集合，
核验“所有 sink 完成 ⇔ 全部完成”、非空完成集包含 root、加入外部 producer/ENTER/EXIT/consumer 后边仍严格递增。
复现算法是枚举 `i<j` 的边子集，再枚举 done bitmask，排除存在未完成 producer 的集合。
它不模拟 wake-list 原子、cache、admission 或设备执行，不能替代协议 UT/onboard。

### 所有权与退出

MPMC pop 成功后只产生一个 owner；wake-list detach 的剩余链归 continuation 私有。
每个 waiter 在迁移前读取旧 next，之后只转交一次到 producer 或 ready queue，维持单链所有权。
这需要沿用原有原子 next 读取和尾链发布，不能用普通缓存读取替换。

EXIT 的逻辑完成可能早于某个 producer 的 resolve 尾部，所以还要计数尾部完成。
即使全部尾部完成，worker 也可能仍在有序资源释放中，所以继续保留原退出 fence。
这三个层次不能合并成一个 Graph DONE bit。

### 开销判断

bootstrap 并行准备并未减少参数总写入量，而是改变其并行度和关键路径位置；
完整 control 初始化、bootstrap 扫描和每实例参数存储仍存在。
每个实例仍有 n+2 个 metadata/control，两个控制队列往返也有成本；小 Graph 可能得不偿失。
一次参数绑定的额外临界路径应以首 child 和短任务间隙验证，不能只比较总量级大的模型 Device 平均值。

## 并行改动核对

以下是 2026-10-31 的 open/draft PR 快照，不是已合入基线的事实。
实施前重新读取最新主线与这些改动的状态，遇到重叠自行解决，不直接依赖未合入接口。

| PR | 核对 head | 与本设计的关系 |
| -- | --------- | -------------- |
| [#2533](https://github.com/hw-native-sys/simpler/pull/2533) | `62a7be43a2f08177940452e2d70801c25081d167` | 公共 Graph roots/external-ready 双事件握手；native 静态 gate 避免复制它，该历史握手用于核对语义，A5 当前执行无需该物化器 |
| [#2538](https://github.com/hw-native-sys/simpler/pull/2538) | `00ea17ce3f1382008a5ab49ae0f5f1e789dbc213` | Graph config key collision；采用当前主线 key 契约，不能预先声称新规则可用 |
| [#2646](https://github.com/hw-native-sys/simpler/pull/2646) | `6cb08c739b4ffe211f4c2a007d66b323fd7608fd` | Graph packet/restore 与 Host bind 布局重叠 |
| [#2647](https://github.com/hw-native-sys/simpler/pull/2647) | `2dd4cbec9a97f96f5bb7a66e9543e9c65ab1376d` | Kernel Mode 与 Graph 资源生命周期重叠 |
| [#2719](https://github.com/hw-native-sys/simpler/pull/2719) | `dea341f83107f9be9a3c05fbd2af827ac8e81b93` | launch attrs/dep_gen/基准工具重叠，影响 DFX 与 oracle 输入 |
| [#2750](https://github.com/hw-native-sys/simpler/pull/2750) | `f0f6b314bb3645ec5ecbb279621d5d6170cd0c5d` | resident task-time args dump；child 参数和身份接入点重叠 |
| [#2759](https://github.com/hw-native-sys/simpler/pull/2759) | `f5ce0e4bf0bd714455a9799bcbd360af40f1fec9` | 历史 AICPU spare lane 路径；不属于当前 native Graph 执行合同 |

PR #2533 描述的是 portable C++ memory model 下的可允许执行：两个独立 release/acquire 发布者可能互相读到旧值，
因而都没有推进 root；它不是本次发现的硬件故障复现。
其修复用同一原子对象上的 RMW 将事件串接。该历史问题不改变当前 native 静态 gate 的协议；其他活跃运行时仍按各自合同验证。

## 尚需实证的风险与完成条件

| 风险 | 要获得的证据 | 未通过时的动作 |
| ---- | ------------ | -------------- |
| CCEC 编译、POD/地址空间访问、局部栈大小 | Host/CCEC/sim ABI 检查与目标编译 | 调整内部结构和 helper，保持 wire/发布语义 |
| GM 原子、cache writeback/invalidate | 多核参数与 wake/admission 压力、same-address new epoch | 自主定位可见性/所有权错误后复测 |
| control queue 公平性与切片预算 | slot 全满、长链、高 fanout、真实模型 tail | 调整有界服务并复跑普通/cooperative 用例 |
| 唯一 Definition 计划的复杂度和容量 | 最大 fanin、实例数、host bind/H2D/峰值存储 | 消除重复扫描和浪费；合法容量不足保持发布前明确结果 |
| 无 Graph 热路径额外开销 | 修改前固定集与筛选 ST 的逐 case A/B | 分离 mapped 快路、减少无 Graph 分支/额外状态后复测 |
| 参数准备与模型性能 | 现有 A5 Qwen 冻结主线/候选 resident、bootstrap、首 child 和间隙分解 | 按性能流程继续迭代，不以功能通过代替性能交付 |
| 新增诊断 offset 为零 | 所有 reader/error/export 审计与 DFX 功能回归 | 审计未完成则保留真实分配，不能报告虚构内存节省 |

实现前没有无法表达的必要协议环节；仍不能因此保证不存在未知漏洞。
最终关闭条件是实现与契约一致、新旧功能验证完成、无 Graph 门禁通过、现有 A5 Qwen 的完整功能与性能闭环，
且退出/内存安全证据齐全。执行按用户约定自主迭代，详见[性能流程](performance.md)。
