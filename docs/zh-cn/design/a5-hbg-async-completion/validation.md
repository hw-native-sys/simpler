# A5 HBG 异步完成验证与性能验收

[返回总览](../a5-hbg-async-completion.md)。本文定义测试方法和通过条件。抽象模型、CPU helper 测试、目标编译、仿真、真实设备和性能资格分别报告，不能互相替代。

## 1. 协议与接口验证

模型应允许在原子操作返回、局部责任产生、记录写入、状态发布、目录修改、计数贡献及 FINISH 激活之间暂停。唯一完成者必须由原子返回值确定，不能在模型中预先用全局布尔锁强制唯一。

| 验证项 | 必须观察的结果 |
| ------ | -------------- |
| 既有登记接口、容量 0/64/65 | 无新注册声明也能登记；合法零登记完成；条件溢出锁存首错且不提前发布完成 |
| 记录身份与格式 | 旧 run、错误 group/task/block/lane、未知 kind、记录越界或未对齐均明确失败；不将内部身份检查等同于外部资源持有 |
| 本次执行绑定复用 | 返回复用同次准入的映射；错误 base/bytes/run/group/offset、task/block/lane 或 slot generation 均不得公开 CHECK/FINISH；开放预留及重复封口仍拒绝 |
| 既有 owner | heap、输入引用和 domain 保留到实际访问结束；scalar context 沿用既有使用登记；同地址复用必须等待旧使用结束 |
| 消费者读取 | 同步消费者、alias、多 fanin、Graph PARAM/SUB_TASK 均读取完整新 payload；signal 满足不能替代数据验证 |
| 必要获取 | 验证现有读取路径；仅在确需缓存操作时解析范围并检查 cache line 共享，不能用未经验证的空操作掩盖缺口 |
| 登记预留 | 副作用前预留容量，按实际公共完成记录 commit；开放预留不能被 count 提前公布；失败不生成成功条件 |
| lease 交接 | 旧版本 CAS 失败；同一记录同时至多一个 owner；交权前进度可见 |
| 活动目录 | bit 0/63、重复置/清、同字并发不改邻位；合法 126 次邻位变化后仍可完成 |
| 目录超界 | 超出一次性生命周期合同明确失败，不能跳过未完成发布后继续业务 |
| P/R/closed 汇合 | 晚到登记、最后计算释放/最后 CHECK 任意次序；所有非零比例、零登记、skip、MIX 均不提前完成且只授予一次完成权 |
| 计算释放顺序 | SPMD 每 block 的 trace 发布与 slot 释放先于 completed_blocks；MIX 收齐 release 后贡献，不能提前关闭计算 |
| 初始化前缀与预留后缀 | 保持 64 条容量；脏值填充后缀、只拷贝前缀、跨 run 复用仍正确；零登记不写后缀，首次 CHECK/FINISH 初始化完整，重复封口不能覆盖 OWNED |
| 分离 header/token | 连续 header、token 对齐/范围/不重叠；旧合并长度被拒绝；真实 AsyncCtx 指针指向 token 区，登记和观察一致 |
| run 活动门控 | 零登记跳过空扫描；异核首次发布在最多 16 个冷空闲服务机会内可见，业务返回和本核错误绕过延迟，实际消费者获取不抽样；未激活时仍传播错误，跨 run 重置 |
| 长 fanin/fanout | 每一步受预算约束；长 fanin 扫描让出前已就绪前缀可被消费；跨核续接不丢失、不重复 waiter 或 ready 发布 |
| 共用完成主体 | 普通完成直接处理到底并保留 direct refill/MIX 前缀；FINISH/Graph 以小预算续接；并发消费者恰好收到每个后继一次，错误前已发布前缀与未公开后缀边界明确 |
| 责任移交 | waiter 挂到下一 producer 后立即被其改写，旧 FINISH 仍沿自己保存的旧链继续 |
| resolve/退休 | TaskDone 可以早于 drain，run 完成及 backing 回收不能早于最后尾部 |
| 服务点 | publication 持续 RETRY、cohort 汇合未完成、admission 失败、持续业务 ready 时各类服务仍可达 |

CPU 测试必须调用生产算法或生产 helper。可控原子适配器用于构造最坏合法交错，真实线程测试用于发现并发访问问题；不能只测试与实现同构的另一套计数器。

至少保留以下有效负对照：尚有零登记成员未释放时提前关闭计算、让普通观察者读零后再次激活、resolve 前计入完成、挂链后重新读取旧 waiter 链接。每个负对照都必须能使相应测试失败。

[漏登记负对照](../../../../tests/st/a5/host_build_graph/async_completion/test_oracle_negative_control.py) 保留 waiter 的业务记录和正常返回，仅故意漏掉条件登记。在没有 publisher 的图中，条件持续不满足，运行时仍可按实际零登记正常完成；独立业务检查必须准确拒绝提前进入的消费者。运行时超时、设备错误或缺少执行记录均不能算负对照成功。

消费者可见性的负对照按实际读取路径设计：已有旧 cache 时绕过必要获取、Graph EXIT 后读取错误实例、MIX 或 sync-start 只有部分成员读到新数据，均不能获得通过。若采用额外获取游标或 ack，还须验证其不可跨核、slot 或数据版本复用；没有这种状态的实现不要求为了测试而增加它。真实 A5 可见性须用旧 cache 和完整 payload 建立独立证据。

### 固定执行资源饱和

基础场景为 [defer_wait_progress](../../../../tests/st/a5/host_build_graph/defer_wait_progress/test_defer_wait_progress.py)。核数由当前 run 的 `rt_available_aiv_count()` 获取，不固定设备或核数；waiter、普通 probe 和 publisher 使用同一 AIV 资源池。

令 C 为实际参与测试的同类执行核数。依次测试 C−1、C、C+1 个普通 waiter，先提交 waiter，再提交使用相同核资源的 publisher。publisher 不能依赖 waiter；消费者依赖 waiter 并校验实际 payload。C 个阻塞 waiter 已足以形成资源饥饿，无须超过 C。

必须观测 waiter 登记返回、publisher 获得执行、条件迟后满足、消费者最终运行；未满足条件不能放行消费者。CHECK 每次有界并交回调度权，不能把阻塞循环包装成新 task。另测持续 ready 业务下条件仍能被检查、长 PENDING 不虚报进展。

持续 ready 的真机资格针对不可抢占的 AICore；仿真宿主可能暂停持有 CHECK/FINISH lease 的线程并耗尽有限业务集合，因此 CPU 层用生产 helper 的确定性服务步验证额度与进展。真机用例在 bootstrap 时一次公开全部无依赖业务任务，不用定时循环建立压力。业务核直接观测 waiter 的 `P=0/R>0/closed`，以及其唯一消费者已发布的 ticket；随后在稳定 head 上读到尚未领取的业务 ticket，证明 CHECK、FINISH 推进时原有 ready 集合仍未排空。只观察“任务未完成数”或只在最终退出核对计数不构成这个证明。每个业务调用有界返回，最终同时校验消费者和全部业务输出。

## 2. 真实异步与任务形状

### SDMA 基本闭环

任务 A 发起真实搬运并登记条件，B 依赖 A 消费输出，C 独立。用可观察的记录证明以下全部成立：

1. A 的 kernel 已返回且实际完成记录仍为 PENDING。
2. C 在这个窗口内推进，A 不再占用执行资源。
3. B 在条件满足之前不进入业务 kernel。
4. 条件满足后 B 执行，搬运结果和 B 的计算结果均正确。

未观察到真实 PENDING 窗口只能证明普通正确性，不能获得异步重叠资格。通过手工提前改标志或跳过条件产生的结果不算真实 SDMA 证据。

[默认队列复用用例](../../../../tests/st/a5/host_build_graph/sdma_queue_reuse/test_sdma_queue_reuse.py) 使用满 AIV 同步启动任务，让同一 cluster 的两个 AIV 各提交 33 次，在每条默认队列上累计 66 次；保留通信域后再各提交一次。必须确认每条实际完成记录的地址保持不变、postId 连续、两个 AIV 共享同一队列，并验证不同 epoch 的完整 payload、消费结果和保护区。覆盖两/四 rank、诊断关闭/开启；不能将累计跨 64 槽解释为 64 次同时在途或已经证明 SDK 背压。

### 通知与缓存

使用真实 `remote_store → notify → waiter → consumer`。先让 waiter 达到实际 AIV 饱和压力，再提交 publisher，确认释放执行核后 publisher 能推进。保持原 workload 的依赖图，不通过给 waiter 增加不必要的 publisher 依赖来回避阻塞问题。

为每种受支持条件验证 AIC、AIV、自核和异核观察组合；消费核要覆盖与提交核、观察核不同的情况。预热消费者旧 cache，并构造目标旧 dirty cache，验证完整 payload 与依赖输出，不能只校验标志。

在实际生产读取路径验证以下矩阵，不预设额外的 Host 计划或 Device 准备状态机：

| 消费场景 | 验收判据 |
| -------- | -------- |
| 不登记条件的同步消费者 | 仍满足必要的本核获取；已有旧 cache 时读到完整新 payload |
| 多 fanin、重叠 view/alias | 完整数据正确；确需显式获取时覆盖全部相关范围，不同数据版本不被误合并 |
| `异步写 X → Graph EXIT → 同步读 X` | 获取落实到业务消费者；多 Graph invocation 保持各自 实例地址及运行身份 |
| 消费核与提交/观察核均不同 | 诊断确认实际核身份，旧 cache 确实在消费核建立；不能依靠偶然同核通过 |
| slot 复用及更换执行核 | 在实际新消费核仍读到新数据；若存在额外获取状态，不复用旧核或旧数据版本的结果 |
| 普通候选暂未获准入 | 本核执行其他业务后再消费仍正确，不阻塞更早的 cooperative obligation |
| MIX 3/5/6/7 | 各 active lane 读取不同范围，全部结果正确；若确需分段获取，额外覆盖成员错峰完成 |
| MIX mask 6 的 inactive coordinator | AIC 无业务 kernel 仍能按既有协议协调 AIV 启动与释放；不把逻辑异步完成作为启动条件 |
| sync-start / MIX sync-start | 保持原 cohort 启动合同，全部成员返回后即释放资源，等待条件不扣住下一 cohort |
| predicate 读取异步 payload | 覆盖 predicate-only Tensor 不在 body 参数中的普通/Graph 任务；获取先于求值，合法 skip 不发 DMA，并按原合同结算 |
| 既有读取路径 | 给出具体指令/backend/核资格，旧 cache、跨核、多轮结果通过；不把 CHECK 观察完成当作获取资格 |
| 已就绪外部输入 | 全同步 run 沿用既有 owner、输入就绪与读取保证；无真实 producer 时不生成伪异步依赖 |

等待 cohort 或必要的分段获取期间，独立 CHECK/FINISH 和错误服务必须推进；取得最终许可后不再插入异步服务。预热/marker 本身不得增加会改变被测 DAG 的依赖，旧 dirty line 必须按发布侧合同在外部写入前处理；不能把事后 flush 旧数据当成有效获取。

| 任务形状 | 重点 |
| -------- | ---- |
| 普通单 lane | 非零、零登记、已经满足、实际延迟、1/64 条条件、多类型/多资源混合 |
| SPMD | 超过并发容量的分波、不同 block 不同条件数、零登记与非零混合 |
| MIX 3/5/6/7 | 同步 lane、零登记 lane 和 pending lane 混合；coordinator 每 block 只贡献一次计算释放 |
| sync-start | 容量边界，取得许可后不插入检查；释放后的等待不扣住下一 cohort |
| 单 lane predicate | skip 不发 DMA，预分配 group 仍结算；非法组合拒绝 |
| Graph | 多实例、内外 fanin、多个 sink、不同实例相同 local id、普通任务交错 |
| PyPTO waiter | 沿用现有 waiter 登记接口，多 peer、多窗口，保留禁止组合的负例 |

[defer_wait_shapes](../../../../tests/st/a5/host_build_graph/defer_wait_shapes/test_defer_wait_shapes.py) 覆盖 COUNTER 的普通 AIC/AIV、超过执行容量的 SPMD、MIX 3/5/6/7、满宽 sync-start、两个 Graph 实例及单 lane predicate true/skip。各物理执行混合 0/1/64 条条件，消费者仅依赖 waiter 并检查全部必要条件和返回标记；该结果不能替代 SDMA 与旧 payload cache 的验证。

[defer_wait_observer](../../../../tests/st/a5/host_build_graph/defer_wait_observer/test_defer_wait_observer.py) 在 AIC 和 AIV 上直接执行生产条件读取器，检查 COUNTER 的 `INT32_MIN/-1/0/1/INT32_MAX` 与非负阈值，以及 SDMA post-id 跨 `2³²`、`2⁶³` 和 `UINT64_MAX` 的无符号比较。CPU、模拟、设备编译和真机运行分别记录资格；静态读取边界不能替代真实 DMA 发布链或消费者缓存验证。负信号随后变为满足值的调度进展由 `defer_wait_progress` 的负初值 C+1 场景覆盖。

[defer_wait_visibility](../../../../tests/st/a5/host_build_graph/defer_wait_visibility/test_defer_wait_visibility.py) 验证 AIC/AIV 的普通与 Graph predicate-only 输入：预热整个消费者执行池，publisher 将独立 predicate Tensor 更新为与旧值相反的值，业务 kernel 只接收两个重叠 payload view 和输出。分别校验执行、skip、两个 Graph 实例的绑定，以及 Graph EXIT 后同步消费者读取新 payload。publisher 不依赖 waiter 完成；Graph 通过既有输入 Tensor 依赖连接预热，不要求新的边界接口。

SDMA 还须覆盖同核连续提交、多个公共完成 record、flag/SQ 压力和零字节请求。核对实际 AIV 与 SDK channel group 的映射；若多个执行者共享 channel，必须验证并落实 session/postId/SQ 状态的独占提交，不用取模分配或偶然串行通过替代并发正确性。已知非零请求返回异常零 handle 时必须失败，不能被归类为合法零登记。

### 信号与多个 payload 的接入验证

构造 `dispatch_wait → dispatch_gather`：waiter 只接收 `data_arrived` 并登记 epoch 条件；gather 接收 `recv_x`、`recv_scale`、`recv_weights`、`recv_routes`，通过 waiter 的 TaskId 建立依赖。不为 waiter 补传 payload 来掩盖缺失的持有链，也不增加 publisher→waiter 的图依赖改变被测等待顺序。

- 核对四个 payload 和 signal 分别由哪个现有 run/domain/参数引用持有；在调用方 scope 结束、任务返回及 pending close 时仍覆盖实际最后访问。
- 两卡、四卡覆盖全部 peer、多个 epoch 和缓冲区复用；消费者分别校验四份数据与未写范围，不只检查通知。
- 在实际 AIC/AIV 消费核预热旧缓存。脏缓存用例须先完成旧写入向外部写者的合法交接，再验证获取不会把旧数据覆盖到新 payload；干净缓存通过不能替代该验证。
- C++ 等价场景与真实 PyPTO 接入分别验收。PyPTO 须核对生成的参数、显式依赖、domain 使用登记和实际读取指令；仅生成异步能力或容量元数据不能获得数据安全资格。

[signal_payload_defer](../../../../tests/st/a5/host_build_graph/signal_payload_defer/test_signal_payload_defer.py) 覆盖两/四 rank、AIC/AIV 标量读取及 AIV `TLOAD/TSTORE`、干净与已合法交接的旧脏缓存、scope 结束与保留 domain 的三个 epoch。waiter 只接收信号，消费者独立接收四个 payload；全宽预热和消费覆盖实际执行池，输出逐项校验全部 peer 数据及未写区域。该 C++ 场景不替代真实 PyPTO 生成代码的验收。

### 多卡用例复用与补齐

本 runtime 的 [sdma_defer_wait](../../../../tests/st/a5/host_build_graph/sdma_defer_wait/test_sdma_defer_wait.py) 从既有 A5 SDMA peer-fetch 示例适配，分别使用两 rank 和四 rank，覆盖 64 KiB、4 MiB 搬运及零工作。每个 rank 获取下一 rank 的不同输入，消费者执行现有 TLOAD/计算/TSTORE 路径；scalar context 通过既有 `device_context()` 登记 domain 使用。

[sdma_shared_channel](../../../../tests/st/a5/host_build_graph/sdma_shared_channel/test_sdma_shared_channel.py) 使用同 cluster 的两个 AIV，各连续提交 16 次请求；校验 32 个 postId 不重复、每个执行的 16 条登记均保留，以及完整搬运与消费者结果。[notify_defer_wait](../../../../tests/st/a5/host_build_graph/notify_defer_wait/test_notify_defer_wait.py) 复用共享 A2/A3 场景的 remote-store、发布、notify 顺序，覆盖两/四 rank、全部其他 peer 的独立数据与 counter、C−1/C/C+1 个 waiter 后提交独立 publisher。

[mixed_async_completion](../../../../tests/st/a5/host_build_graph/mixed_async_completion/test_mixed_async_completion.py) 在两/四 rank 上交错登记每个 peer 的真实 SDMA 记录与 COUNTER，覆盖普通 AIV、两 block SPMD/sync-start 和 MIX 3/5/6/7；MIX 的 AIC 成员只登记 COUNTER。延迟通知先等待全部登记和 peer 就绪握手，再发布独立的通知 payload。另以已满足的 COUNTER 配合 4 MiB SDMA 验证普通与 Graph AIV 路径。两个 Graph 实例复用同一 Definition、绑定不同 view；publisher 位于 Graph 外，消费者仅依赖 producer 或 Graph EXIT。诊断关闭/开启分别运行，校验全部搬运结果、通知数据、登记数、保护区及 domain 回读；真实 native 记录须确认实例身份与完成、EXIT 依赖顺序。

优先复用 a2a3 已有多卡异步通信场景的 DAG、通信顺序和 golden，再适配 A5 HBG resident。以下入口提供可复用的业务语义，其原有 runtime 或平台测试结果不构成新 resident 路径资格：

| 现有入口 | 可复用内容 | A5 HBG 适配要求 |
| -------- | ---------- | --------------- |
| [a2a3 SDMA deferred demo](../../../../examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py) | 两卡 peer fetch，搬运输出及依赖消费者输出的 golden | 接入 A5 的 workspace、完成记录和全核观察；证明返回后真实 PENDING 窗口 |
| [共享 deferred_notify](../../../../tests/st/worker/comm_domain/deferred_notify/test_deferred_notify.py) | 两卡远端 payload、signal waiter 与消费者 | 保留原依赖，验证 A5 缓存可见性和 resident 完成；该入口本身不列 A5 onboard |
| [共享 async_notify](../../../../tests/st/worker/comm_domain/async_notify/test_async_notify.py) | a2a3/A5 两卡通知 counter 和依赖结果 | 从 TMR 接到 HBG resident，增加饱和 waiter 与异核检查 |
| [A5 SDMA deferred demo](../../../../examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py) | A5 通信域 SDMA 资源配置与两卡 golden | 复用 backend 配置，验证HBG resident 调度和既有生命周期 |
| [HBG async boundary](../../../../tests/fdwic/pa_hbg/spmd/test_async_boundary.py) | 未提供合法异步上下文时的拒绝边界 | 保留适用负例；不能用不发起异步工作的边界测试替代真实通信 |

没有对应正例或故障判据的组合必须构造新增用例；不能只修改平台列表和 runtime 名称。新场景须验证实际运行的是 A5 HBG resident 及目标产物，不能由 legacy、TMR 或 kernel 内阻塞 fallback 获得通过。

| 多卡覆盖 | 必须覆盖的情形 |
| -------- | -------------- |
| 基础通信 | 两卡双向 SDMA、remote_store/notify、真实消费者结果与独立任务进展 |
| 多 peer | 至少四卡的多个 peer 等待；通知乱序、迟到 peer、每 peer 数据身份可区分 |
| 多条件/多资源 | 同组多 peer、多窗口及 SDMA/COUNTER 混合，必须等待全部必要条件 |
| 饱和与任务形状 | waiter 达到真实 AIV 饱和压力；SPMD、MIX、sync-start、Graph 与通信交叉，包含同步消费者与各成员不同的完成时刻 |
| 重复与代次 | 多轮 run、保留域复用、合法独立槽、旧 token/旧窗口、peer 提前覆写下一 epoch |
| 失败与关闭 | peer 未发布、提交/登记失败、长 PENDING、pending close、核故障和未知外部访问 |

多卡 fixture 保留逐 rank golden、延迟条件及 peer 身份，不以各 rank 结果相同掩盖串数据。故障用例的预期结果可以是明确失败和资源隔离，但不能是提前成功或未知资源被释放。四卡或故障环境不可用时对应项保持未验证，不用两卡正常场景代替。

## 3. 错误、复用与关闭

[跨 run 映射刷新用例](../../../../tests/st/a5/host_build_graph/async_completion/test_run_reuse.py) 在同一 Worker、callable 和保留域上交替使用不同任务数、block 数、AIC/AIV/MIX 掩码、0/1/4/64 条条件及合法的单 lane、单 block predicate skip，连续两轮重复这些形状并更换 payload epoch。每次校验完整执行记录、输出和 signal 邻接保护区，验证 run 入口获取足以刷新复用存储中的只读映射，不能只用相同图形状的重复运行证明这一点。

[跨卡通知复用用例](../../../../tests/st/a5/host_build_graph/async_peer_counter/test_peer_counter.py) 在同一两卡通信域上执行通知值 `1 → 2 → 重置为 1`，使用独立的 payload 标签 `101/102/103`；重置只发生在前一轮全部消费者结束后。另测实际 AIV 容量 `C+1` 个 waiter。接收端完成全部条件登记后才向发送端确认就绪，publisher 再发布数据与通知；独立 AIC 上的有界登记 gate 不占用被测 AIV 资源池。诊断关闭和开启时均检查完整 payload、保护区、执行次数和握手，开启时再校验真实 native 完成与依赖记录。

| 注入位置或输入 | 验收判据 |
| -------------- | -------- |
| 提交前容量或输入验证失败 | 该次请求没有设备副作用，未使用预留被归还 |
| 已尝试提交后失败或登记异常 | 首错保留，相关 owner/backing 仍持有，没有成功 TaskDone 或自动重放 |
| peer 提前写下一 epoch | 验证业务通信协议通过消费者完成确认或独立槽阻止提前覆盖；COUNTER 比较不提供 payload 版本检查，allocation 存活不能被当作正确性证明 |
| 旧 token/旧窗口 | 旧操作未结束时不得复用资源，不能只靠 generation 检查 |
| 长时间 PENDING | watchdog 不被纯轮询刷新，错误有具体 task/group/条件身份 |
| owner 暂停在交权/清位/贡献边界 | 正常路径不提前退休；恢复后没有重复贡献 |
| owner 硬故障 | 不强制接管半写记录；未知外部访问阻止复用和释放 |
| pending 时 close | 不接收新调用，正常 drain 或明确失败并保留未知使用 |
| 返回/ACK 已到而 DMA/peer 仍访问 | 实际资源账本拒绝提前释放 |
| 清理步骤失败 | 原始错误不被覆盖，尚未释放的 owner 可追踪，不能报告清理成功 |

检查实际资源 owner、引用及后续 admission 行为，不能只检查 scheduler 的 FAILED 位。诊断截断应保留“证据不足”状态，不补造未观察到的完成事实。

[多卡 pending close 用例](../../../../tests/st/a5/host_build_graph/signal_payload_defer/test_pending_close.py) 在两/四 rank 提交后立即关闭 Worker。publisher 延后 200 ms，消费者使用 AIV `TLOAD/TSTORE` 读取四个 payload；调用方在 Worker 初始化前持有共享结果缓冲区，关闭后校验全部 peer 数据、完成状态及 domain 已释放并回收。这验证正常关闭对已接受跨卡访问的排空，不代表故障后的未知 DMA 已停止。

[SDMA 错误保留用例](../../../../tests/st/a5/host_build_graph/sdma_defer_wait/test_error_close.py) 在两/四 rank 先校验 4 MiB peer TGET，再于新 run 的成功提交后触发登记容量溢出。关闭诊断采集时，逐卡读取独立目录中的设备错误日志，确认原始 `-102`、逻辑完成数为零、通信域回收被拒绝且关闭后不能创建新资源。该结果证明错误隔离与保留，不将它等同于引擎已停止或强制复用已安全。

## 4. 性能与选型实验

功能正确性优先：先修复问题并完成新增功能、单卡/多卡及受影响原有用例的回归，再检查性能与基线是否接近。不为小幅性能差异延后功能修复，也不以性能改善抵消错误结果。性能标准采用 [PR #2864](https://github.com/hw-native-sys/simpler/pull/2864) 的用例集合，复用当前仓库中的清单及采集器；历史耗时只作背景，必须在同机同卡重新采集冻结基线与候选。

| 基准输入 | 范围 |
| -------- | ---- |
| [fixed-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/fixed-cases.json) | 7 项固定无 Graph benchmark |
| [selected-st-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/selected-st-cases.json) | 11 项具有调度性能价值的 ST |
| [csa-cases.json](../../../../tools/benchmarks/a5_hbg_graph_task/csa-cases.json) | 1 项 CSA 补充场景 |
| A5 Qwen3 decode | 40 层、B16、seq_len=3500，验证完整 out/K/V，再单独采集 |

共 19 项无 Graph 用例加 Qwen。使用 [collect.py](../../../../tools/benchmarks/a5_hbg_graph_task/collect.py) 核对实际 pytest case 的参数/config、golden 和产物身份，清单与当前代码不一致时先修正原因。先完成一轮基线/候选对照，每个用例预热 5 轮后采集 100 稳态轮；出现明显差异或异常长尾时才追加独立重复。`spmd_paged_attention` 不加入该参考性能集，但 SPMD 功能验证仍是必需项。

新增异步和多卡性能对照作为补充，不能替换上述原有性能集。完整 sweep 在同一设备 allocation 内串行安排同设备的基线/候选，保留所有原始数据，禁止选择历史不同产物的最佳结果拼成当前资格。

[SPMD/MIX 补充清单](../../../../tools/benchmarks/a5_hbg_graph_task/spmd-mix-cases.json)逐项覆盖单 block 的 chain/burst 与 mask 3/5/6/7、普通任务交错、多 block AIC/AIV、超核数分波、最后 block/lane 释放、SPMD 间依赖、高 fanin、混合波次及 sync-start。按实际 case、参数和代码版本分别报告，不能用一个混合波次用例的百分比代表全部 SPMD，或用一个 MIX 链代表全部 MIX。预期拒绝、故障注入和强制开启诊断的用例保留功能/协议结论；不得将其失败等待时间或诊断耗时混作 L1 吞吐。Qwen 单列完整模型结果，未采集时明确标为未测。

依赖输出验证来检查本次调度模式的场景，在性能清单中声明 `timing_requires_golden: true`，计时轮继续执行原有校验。两臂使用相同策略，仍从 STRACE 的调用区间提取 Host/Device 时间；不以关闭断言来适配统一的 `--skip-golden`。

性能归因须先列出同步执行路径上相对基线新增的操作，注明每 run、每 task、每 block/lane 或每调度循环的执行次数，及其内存读写、缓存维护、原子操作和上传成本，再用分段数据验证。Host 总耗时包含等待 Device；同时报告 bind、publish_image 和 runner_run，不能将 Device 等待误算为 Host CPU 开销，也不能因公共 Host 文件未改而忽略 A5 HBG 私有 Host 的准备成本。

性能检查使用 L1；L3/L4 用于诊断归因，结果不混入 L1。固定机器、设备、输入、计算 kernel 和工作量；明显回退采用 A/A 噪声和交换顺序的 B/A/A/B 对照复核。三组成对测量不是基本验收的前置条件；未发现明显回退时不继续追踪微小波动。保存所有样本及异常，不只报告最优值。

同步回归至少包含 CSA、SWA、Qwen 和短任务调度负载；分别测整个 run 零登记、少量异步 task 混排、各执行分支零登记的情形。异步对照使用等工作量的 kernel 内等待与 deferred；阻塞饱和负例单独报告正确性，不能将其超时当作正常性能样本。

| 场景 | 主要观测量 |
| ---- | ---------- |
| 非零 group 比例从低到高 | 静态 G/动态 P 的真实更新成本、Device 与 Host 开销 |
| 大 E、小在途数及稀疏尾部 | 目录扫描数、满足到发现延迟、初始化字节 |
| 单热点、多核竞争与同字集中结束 | 有效观察、lease 碰撞、目录更新及最后 CHECK 清位并发布 FINISH 的固定尾部 |
| 每组 1/4/64 条条件 | 每条有效观察的移交和屏障成本 |
| 短/长业务 kernel 混合 | 业务吞吐、检查响应、FINISH 尾延迟 |
| 长 fanin/fanout、control 混排 | resolver slice 成本、ready 发布和业务公平性 |
| 不同 payload 范围及 cohort 完成错峰 | 既有读取及必要维护成本、实际新增状态字节、满足到消费者执行延迟 |
| 连续 SDMA 提交 | SQ/flag 背压、kernel 返回时间、实际重叠窗口 |

基本对照逐例报告 Device、Host bind/端到端、均值及 P95。出现稳定、超出 A/A 噪声的明显退化时定位并解决，不能用局部异步收益抵消或隐去。kernel window、分配/初始化/H2D 字节、峰值在途 group 和总 group 数用于解释具体回退，不要求为未出现问题的路径逐项追加优化实验。

目录与尾部实验分别报告有效条件观察、目录更新的次数/重试和耗时、lease 交权及屏障、条件满足到首次发现、到消费者进入 kernel 的延迟。覆盖同字集中结束、单热点多核争用、短业务 kernel 和稀疏尾部；单独列出最后 CHECK 两次目录更新的 256 原子操作预算及额外 lease/join/屏障，不能把它隐入 8 个工作单元。用 L3/L4 诊断解释组成，在相同负载的 L1 对照中判断整图影响。

候选仅在满足相同功能、错误和生命周期合同后比较。选择依据见[选型文档](choices.md)，不将单次 CAS 或局部微基准的收益直接外推为整图收益。

## 5. 执行环境与交付检查

硬件测试先识别实际 A5 silicon，通过仓库 `onboard-arch-precheck`，并按 `task-submit` 规则独占所需设备、隔离每次运行的日志。构建与运行使用项目独立 venv；源码、编译产物、PTO-ISA、CANN 和实际加载路径必须一致。架构检查通过本身不构成功能资格。

交付必须提供支持矩阵、正常与故障结果、sim/CPU/A5 分层结果、同步性能回归、异步真实窗口及数值结果、原始 L1 样本，以及明确的未支持组合。关闭诊断后的正确性也必须验证；诊断字段不能成为运行时所需的隐藏身份来源。

基础实现与选定优化均应具备针对生产路径的回归测试。仅有设计模型、单个原语测试或目标编译，不能声明整项异步功能交付完成。
