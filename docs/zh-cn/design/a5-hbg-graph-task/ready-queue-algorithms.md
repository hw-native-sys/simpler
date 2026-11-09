# Ready 队列算法：发布、领取与竞争控制

本文历史测量所引用的 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`，不随仓库分发；下文保留文件名用于追溯。

本页是 [ready 队列设计](ready-queue-design.md)的算法层补充，研究日期为 2026-11-03。
G/H4/H8 首轮只比较了同一种 ticket 协议的拓扑，不能据此完成算法选型。
以下区分原论文事实、针对 A5 的推断、原型验证与生产接入；CPU 吞吐不作为 A5 性能证据。

当前代码只保留 task 粒度的 **T/H8**。T/G、T/H4、L/G、C/W1/W4 的实现与配置入口已删除，因为它们没有完成当前异步实现的验证；将来需要基于现有调度协议重新实现，不能直接恢复旧候选。
下文的候选状态、结构尺寸、API 名称与性能均记录历史原型，不能视为当前源码提供的能力。当前池 header 为 version 4，固定组宽 8；不含 linked/chunk 字段或本核 pending 发布列表。重新开发与验收要求见[异步选型约定](../a5-hbg-async-completion/choices.md#ready-queue-implementation-scope)。
历史 `queue-ordinary-credit-v1` 的 Qwen 中位数为 38.454 ms。
独立 block 原型的 37.855 ms 结果伴随 3.08 倍冷构建成本，按用户要求撤回；
其发布预算、MIX lease、同步 cohort 和审计改动不在当前代码中。
完整思路与再次评估条件见[后续优化候选](block-ready-scheduling.md)，历史数据见[原型记录](qwen-block-work.md)。
队列算法选型并未结束，但本次回退不继续启动优化。

## 比较轴与候选状态

必须分别记录四个轴，不能将增加组数记作新算法：

| 比较轴 | 取值示例 | 解决的问题 |
| ------ | -------- | ---------- |
| 存储和发布协议 | ticket 数组、常驻链式 FIFO、sealed chunk、可复用环 | 发布停顿、空间和可见性 |
| 消费协议 | 单项 CAS、credit 限制的批量 CAS、FAA 加 slot 仲裁、块内领取 | 原子次数与任务所有权 |
| 拓扑和路由 | 每类型唯一队列、分组共享、producer 子队列、本地 deque 加 injector | 热点地址分布与发现成本 |
| 同步聚合 | 同一 producer 合批、多个消费者合并请求、代理批量分配 | 同地址原子流量与代理等待 |

历史实测配置包括 `T/G/claim1`、`T/H4/claim1`、`T/H8/claim1`、`L/G/claim1`、`C/W1/claim1` 与 `C/W4/claim1`。
其中 T 为 append-only ticket。H4/H8 是组宽，G 是每类型唯一物理队列。

| 候选 | 实质变化 | 历史证据 |
| ---- | -------- | -------- |
| T/G、T/H4、T/H8 | 同一 ticket 算法，改变拓扑 | 已接生产调度，有功能和 Qwen 数据 |
| T/claim2 | 仅改变领取粒度，一次 CAS 最多取两个 | 底层原语已有 CPU 测试，未接 dispatch |
| L/G | 完整批次先准备，link CAS 挂链，tail helping | 已接生产；65 仿真、Graph 真机审计及 Qwen golden 通过，Qwen 51.179 ms，慢于 T/G |
| L/H | L 协议结合分组 publisher | 未接入；需先解决按组保存 pending 的空间和服务预算 |
| C/G 的 W1、W4 | 发布 sealed chunk，共享块内 cursor，改变可见窗口 | 已接生产；各 65 仿真、Graph 真机审计及 Qwen golden 通过；W1 48.186 ms、W4 44.915 ms，仍慢于同卡 T/H8 |
| C/H | chunk 协议结合分组 publisher | 未接入；按组 pending 的空间和公平服务仍需证明 |
| FAAArrayQueue、LPRQ、SCQ | FAA 领取结合 slot 状态/作废/复用 | 原论文与实现调研，未移植 |
| Flat Combining、Aggregating Funnels | 合并多个调用者的同步请求 | 调研；直接阻塞式移植不满足当前调度约束 |
| 本地 deque、producer 子池、SALSA 所有权转移 | 改变单写者关系、发现和迁移协议 | 见主设计调研，未实现 |

## 全局 FIFO 的不同实现

### T：预留后发布的 ticket 数组

一次 producer batch 执行一个 tail FAA，再逐项发布 immutable ref。消费者先观察已发布连续前缀，
再 CAS 推进 head；空探测不预领未来任务。单 run 不复用槽位，Host 精确分配容量。

优势是连续存储和简单生命周期；缺点是预留空洞挡住同队列后续发布，以及共享 head 的每任务竞争。
它没有完整 SCQ/LCRQ 的 slot 作废与重新定位协议，不继承这些算法的进度证明。
当前批次最多 8 项，消费仍逐项；producer 原有 `next_waiter` 链构造与读回成本也没有消失。

### L：批次挂链的常驻节点 FIFO

Michael–Scott 的原设计通过 next CAS 发布节点，并允许其他线程帮助推进 tail。
本项目借鉴该机制，但增加批量发布与批量领取，并利用 run 内不回收的条件。
下面是本项目适配协议，不能用原论文代替对适配的证明。
来源：[Michael–Scott §3](https://www.cs.rochester.edu/u/scott/papers/1996_PODC_queues.pdf)。

**布局与生命周期：**每个逻辑 task 一个独立 64-bit queue link，每个物理队列一个 sentinel。
head/tail 各独立 128-byte 原子行。全池 link 空间为 `8 × (N + Q)`，控制块为 `256 × Q`，
不含对齐、descriptor 和原有 scheduler state；N 为 task metadata 数，Q 为物理队列数。
link 不复用 `next_waiter`，避免依赖解析或批次重建改写已经公开的链。
所有 link 行只使用原子写，禁止 cached store、整行脏回写及混用 bypass 写。

**发布：**调用者拥有尚未发布的所有 task，先构造最多 8 个 ref 的私有链，完成屏障后，
CAS 某个已发布尾节点的 `next: INVALID → batch.first`。成功即整个批次转入共享池；
随后的 tail CAS 只是提示更新，失败不能再次发布。挂链前暂停不占共享位置；挂链后暂停，
其他消费者仍能领取，其他生产者能够沿链帮助 tail 前进。

**领取：**head 指向 sentinel 或上次成功领取的最后一个 task。观察 head 后最多两个已发布节点，
一次 CAS 将 head 推进到该前缀末尾，成功后执行读取屏障，才拥有这些 task。
link 一旦公开只从 INVALID 变到唯一后继，节点直到 run fence 都不复用，因此旧 head 值不能出现 ABA。
批量领取可能越过滞后的 tail；因为不回收旧节点，tail 仍指向有效链前缀，生产者可以继续帮助它前进。
这与需要及时回收节点的通用 MS 实现不同，不能日后添加运行中回收而保留该简化。

**A5 读取：**`ld_dev` 对 atomic-published word 可能暂时读旧值。旧 head/tail 仍在常驻链上；
旧 INVALID next 只能导致空探测或失败的 link CAS；唯一所有权由成功的 head/link CAS 确认。
合法非空 next 不再变化，不能混入可复用节点或修改已发布的中间 link。
NOT_READY 只表示本次没取到，不能用作 run 完成或全局无任务证明。

**有界重试：**一次调用最多尝试有限次 link CAS/help，耗尽后返回 RETRY。
未挂链批次仍归调用者，已挂链批次只能返回 SUCCESS。`append_private` 已提供私有链合并原语：
仅操作同一 publisher 独占、尚未共享的链尾，转移 incoming 所有权并清空其本地描述符。
每次 prepare 最多 8 项，多个私有批次可以合并为最多 N 项的待发布链；共享领取仍受 1/2 个 credit 限制。
必须在 scheduler 的持久状态里保存未发布工作，下一服务轮继续，不能沿用旧 bool push 的“false 就 abort”。
历史 L/G 原型曾接入 Host plan、bootstrap 和 dispatch；每个类型一个持久 pending 描述符，RETRY 返回成功接收但保留发布责任。
它已通过仿真、Graph 真机协议与 Qwen 数值校验，但同卡性能慢于 T/G；对应历史回归记录不证明当前异步资格。

### L 的待发布状态与 scheduler 接入契约

第一条 L/G 候选按类型各保存一个本核 publisher 描述符；L/H 需要按目标组保存，不能把不同组的私有链合并。
存储只包含 ref/count，任务本体仍在 run-resident GM，空间不能随未发布任务数增加本地数组。
历史 `SchedulerLinkedBatch` 是 24 bytes；L/G 三类型 pending 描述符占 72 bytes/core。控制解析的独立续接游标占 184 bytes，`SchedulerLocalState` 共 808 bytes（含可选异步服务状态的 8-byte 引用）；控制解析和可移交 FINISH 的待发布 batch 由各自游标持有，不借用上述本核 pending 列表。
ready pool header 保持 128 bytes，version 升至 2，新增 kind 与 links_offset；Host 初始化独立 links 和每队列 sentinel。
首个接入候选只允许 LINKED + group_width=0，拒绝未实现的 L/H。当时的 T/G/H4/H8 可用同一设备 binary 进行比较；当前已不提供这些切换入口。
L/H 若每核保存全部 Q 个组则为 `24 × Q bytes/core`，必须与 GM publisher descriptors 等替代方案比较。

| 状态 | 唯一 owner | 允许操作 | 退出条件 |
| ---- | ---------- | -------- | -------- |
| EMPTY | 本核 | 接收新私有批次 | 转为 PENDING |
| PENDING | 本核持久描述符 | 合并后来的私有链；有限次 try_push | RETRY 保持原链；成功转 EMPTY |
| SHARED | 对应物理队列 | 消费者观察、CAS 领取；其他 producer 帮助 tail | 所有已发布 link 不再由原 owner 改写 |
| ABORTED | run 清理协议 | 保留 backing 等待退出 fence；丢弃本地未发布描述符 | fence 后才可回收或下一 run 初始化 |

服务次序必须明确接入 `bootstrap_ready_graph` 和 `run_ready_dispatch_loop`：

1. bootstrap 的本地批次转入持久 publisher，不能返回后让栈上 batch 消失。扫描 barrier 的声明必须区分
   “metadata 与参数已准备”和“所有 ready 已发布”；不能无审计地沿用旧注释的含义。
2. steady loop 在 completion/control resolve 后、进入 kernel 前服务 publisher；每轮每类型一次有限预算。
   内部争用 RETRY 保留状态，不报告图损坏、不推进任务完成计数。
3. pending 非空时继续处理 completion、control、admission、退出信号和 watchdog，不能在 push helper 内等到成功。
   在 backlog 归零前暂缓该核开始新的长 kernel，避免私有待发布工作被 kernel 执行时间压住。
4. 暂缓 kernel 不应形成资源循环：该队列不需要消费者释放容量，挂链/help 不等待 kernel 或 peer。
   当前 run 任务总数有限，每次成功挂链转移至少一个新任务；该论证不允许日后悄悄加入有界可复用 ring。
   仍需测试已授予协作许可、远端成员已就绪及 pending 同时存在的交错，不能只测独立 AIC。
5. abort 不再发布新的私有链；已经共享的批次不回滚。AICPU 必须等现有退出 fence 后回收 backing。
   正常完成不能从 publisher/queue 为空推导，仍使用逻辑完成计数与既有退出证明。

必须新增的接入测试是：持续 RETRY 后有新任务到达、不同类型同时 pending、bootstrap 带 pending 进入主循环、
full slots/MIX admission 下 pending 服务、错误退出与同 backing 新 epoch。
现有原语测试只能证明私有合并、竞争和静默期复用，不能替代这些 runtime 测试。

### C：sealed chunk 与块内共享领取

一种全局队列仍可由不可变小块组成。producer 先填写 2/4/8 项的 chunk，再将 descriptor 挂入全局链。
所有同类型消费者可访问 chunk；领取一个 task 不意味着独占整个 chunk。

候选 C 的控制面分成两部分：全局链只维护块间发现，每个块有独立 cursor。
单项 CAS 发生在块内 cursor，块耗尽后才推进全局块头。好处是减少块间同步次数，
并允许有界窗口内的多个块同时承担领取；只允许访问唯一首块则仍有每任务同地址争用，不能声称已经消除热点。
窗口会放松全局 FIFO，必须保持轮转和旧块的服务机会。

只有实际 candidate credit 对应的前缀可以成为私有任务，其余条目保持共享。
传统 MIX 派发必须先完成本地 placement 再领取；native ordered 路径则沿用有界 candidate 与统一 admission。
不能将一个含多个 MIX 的 chunk 从全局链摘走后等待资源，也不能绕过既有 cooperative 顺序域。
初版优先 sealed 后不可变的 payload，descriptor 的 next/cursor 与 payload 分行；
不同时引入边追加边消费、运行中回收和跨核所有权迁移。

需要证明的交错包括：发布者 seal 前暂停、挂链后暂停、最后两个消费者竞争耗尽、
旧 head 的消费者落后于全局头推进、MIX placement 失败、abort 时被领取与未领取后缀的生命周期。
Host 对 chunk 容量不能直接用 `ceil(N / 8)`：短链和稀疏唤醒可能每批只有一个 task，
必须允许最多 N 个 descriptor，或提供有证明的合并/复用方案并报告空间。

### C/G 的首个实现边界

以下契约曾落实到现已删除的 `scheduler_chunk_queue.h`、ready pool、Host plan 和领取路径。
`C/W1` 与 `C/W4` 是同一种 chunk 算法的窗口配置，
不能再计作两种底层算法。它与 L/G 的实质区别是：任务领取 CAS 转移到每块 cursor，
全局 head 仅在跳过耗尽块时推进；块中未被 credit 领取的任务始终共享。

- 每个非空批次最多 8 个 ref，以该批第一个 task ref 作为唯一 chunk ID。任务不重复发布，
  因此不需额外全局 descriptor 分配计数器；最多 N 个 descriptor 足够，不能按 N/8 分配。
- 初版每个 descriptor 为 256 bytes：128-byte atomic-only cursor 行，128-byte immutable payload 行
  （count、8 个 ref 与 padding）。next 存放在独立 atomic-only links，沿用持久链的挂链与 tail helping。
  额外空间为 `256N + 8(N+Q) + 256Q`，另计 header/descriptor/对齐；必须同时报告 Host 初始化/H2D 成本。
- Host 初始化 cursor=0、count=0、ref=INVALID。producer 独占未发布 descriptor，bypass 写 payload 后执行屏障，
  再发布 link；公开后不改 payload。一个 chunk 必须同类型，发布后不再向块内追加 task。
- head 初始为本类型 sentinel，之后指向最早尚未跳过的块。cursor 耗尽后才允许帮助 head 前移；
  最后一个空块保留到后继发布。移过 head 的块仍可被旧读者访问，直到 run fence 才回收。
- 一次探测最多检查 W 个已挂链块，不预留未发布位置。消费者轮转优先块，并定期先服务最老块；
  已耗尽前缀的跳过也有有限预算。NOT_READY/RETRY 都不是全局完成证明。
- 先读取候选前缀并检查类型/placement/credit，再 CAS 对应块 cursor。成功只转移该前缀的所有权，
  失败不保留任何 task；旧读者的 CAS 不可因 head 前移而跳过。MIX 不批量领取，沿用其现有 admission 契约。
- 仍用每类型持久私有链处理挂链 RETRY，但 descriptor.count 统计块数，task 计数保持独立。
  待发布 descriptor 可连接，已公开块的 payload 和私有 task 所有权不能回退。

验收检查项包括：seal 前暂停、挂链后暂停、cursor 最后两项竞争、过期 head/peek、窗口内跨块领取、
老块在后继持续可取时的服务机会、全槽占用、MIX placement 失败、abort 与静默期再初始化。
W4 允许块间非 FIFO，须核对依赖与 cooperative 顺序仍由既有协议决定，不能把队列顺序当成执行许可。
生产实现每次最多跳过 8 个耗尽块，每四次探测先服务最老块，其他轮次按 worker rank 与 poll 轮转。
该候选采用 pool header version=3，新增 `chunks_offset` 与 `chunk_window`；仅 C 接受窗口 1/4，其他协议要求窗口和块偏移为零。
已撤回的 [block work 原型](block-ready-scheduling.md) 使用 version=4；当前恢复的 task pool 和上述冻结结果使用 version=3。
生产路径仍为 claim1，原语的 credit2 能力不表示 dispatch 已启用批量领取；本核状态仍为 632 bytes。

11 个 chunk 原语用例和 23 个生产池用例分别重复 20 次，通过相关 97 个 C++ 目标；
三候选各 65 仿真、6 个 Graph 真机专项及独立协议审计通过。
同卡完整 Qwen 的 W1/W4/T/H8 为 48.186/44.915/40.948 ms：扩大窗口降低 6.79%，但 W4 仍慢于 T/H8 9.69%。
不把这个差值全部归因为原子指令；空间、窗口遍历、发布与 Host 初始化都需要计入代价。
常规回归和故障验证的独立状态见chunk 候选证据（本地 `chunk-ready-pool-candidates.json`）。

### FAAArrayQueue、LCRQ 与 LPRQ

FAAArrayQueue 的原实现以数组节点组成链，FAA 分配入队/出队位置，slot 通过 CAS 或 exchange 仲裁。
消费者可以将尚未填入的 slot 标为 taken，生产者发现后重新选择位置；数组用完再链接新节点。
这与 T 的“只领取已发布前缀、不作废槽位”不同。
来源：[作者实现](https://github.com/pramalhe/ConcurrencyFreaks/blob/master/CPP/queues/array/FAAArrayQueue.hpp)。

LCRQ 将类似机制组织为可复用的 concurrent ring，并处理 slot 代际及关闭。
LPRQ 给出了不使用 CAS2 的可移植变体，因此不能以 A5 没有现成双字 CAS wrapper 为由排除整个算法族。
来源：[LCRQ](https://www.cs.tau.ac.il/~mad/publications/ppopp2013-x86queues.pdf)、
[LPRQ](https://nikitakoval.org/publications/ppopp23-lprq.pdf)。

**A5 推断：**FAA 消费能减少 head CAS 失败重试，但共享 FAA 本身仍然串行，slot 仲裁也要付原子成本。
取消空槽还破坏 T 的“恰好 N 个槽位足够”依据：一次任务可能因被消费者抢先作废而尝试多个 slot。
不能只把 T 的 head CAS 换成 FAA，或者直接给 FAAArrayQueue 固定 N 个一次性槽位。
需要完整处理空轮询、容量、循环代际、溢出与回收后再比较。

## 原子聚合不是免费批量化

Flat Combining 通过 publication records 收集请求，由取得锁的 combiner 批量处理，其他线程等结果。
它与当前 producer 在同一次 resolve 内合批不同：前者减少多个调用者的共享访问，后者没有跨核等待。
来源：[Flat Combining §2](https://www.cs.bgu.ac.il/~hendlerd/papers/flat-combining.pdf)。

Aggregating Funnels 将 FAA 分散到 aggregator，再由代理向主计数器一次申请总量，各参与者计算自己的返回值。
论文证明其线性化性质，同时明确它是 blocking：参与者等待代理返回。
来源：[Aggregating Funnels §1、§3](https://arxiv.org/html/2411.14420v2)。

**A5 推断：**这直接针对用户指出的同地址原子热点，值得独立评估；但不能把论文 CPU 吞吐映射为 A5 收益。
AICore 同时承担 kernel 执行和调度服务，持代理角色的核执行长 kernel 或等待协作成员，可能拖住全组请求。
直接同步调用不进入当前实现。若采用异步 request/result 形式，需要持久 descriptor、epoch、可继续服务、
代理退出协议及 MIX 不囤积任务的证明；不能仅在 wrapper 内等待其他核响应。
当前先比较不增加跨核等待的 producer batching、credit 限制的 consumer batching、分组以及 sealed chunk。

## 操作成本、接入顺序与验收

下面是无竞争成功路径的结构性下界，不是实测延迟；还必须加上失败重试、读取、屏障和 payload 准备。
B 为实际发布批量，C 为实际领取数，不能用上限 8/2 代替测得的分布。

| 候选 | 发布侧 | 消费侧 | 不能遗漏的成本 |
| ---- | ------ | ------ | -------------- |
| T | 每批 1 次 tail FAA；每项 bypass store | 每 C 项 1 次 head CAS | 预留洞、发布屏障、旧 batch 链构建 |
| L | 每批 1 次成功 link CAS、通常 1 次 tail CAS | 每 C 项 1 次 head CAS | 每项 link 原子准备、help、链读取、pending 服务 |
| C | 每块一次共享挂链 | 每 C 项块内 cursor CAS；耗尽时推进块头 | seal、descriptor、空块处理、窗口公平性 |
| FAAArrayQueue 类 | FAA 加 slot 仲裁，偶尔扩链 | FAA 加 slot 仲裁，可能跳过作废槽 | 空轮询、取消槽和容量管理 |
| 聚合代理类 | 每次合并有少量主地址原子 | 按请求合并返回 | 本地请求行、代理串行、等待和接管 |

后续重新实现候选时，采用以下顺序：

1. 从当前 T/H8 和异步完成协议建立基线，重新设计候选的发布、领取、续接与资源参与方式；历史证据仅供定位。
2. 先证明正常、RETRY、abort、close 的所有权和服务顺序，不以私有待发布工作阻塞业务或 CHECK/FINISH。
3. 全局与分组候选使用相同的领取额度和任务形状；MIX 保持先准入/placement 后转移所有权，不能以囤积任务伪装算法收益。
4. 完成 CPU、SIM、单卡、多卡和 native protocol 验证后，再采集同机同卡、相同计算产物的完整模型与固定场景性能。旧基线和历史最优值不能拼成当前对照。
5. 只有功能相同且差异有重复证据时才替换 T/H8；保留结论和证据路径，未验证的候选不保留为生产分支。

先采队列行为分布和调度间隙，判断性能变化是否与假设一致；不得把 DFX 的全部 scheduling 时间归因于原子争用。
评估至少覆盖单 producer 宽 fanout、多 producer 汇聚、短链、间歇 ready、MIX/普通混合以及完整 A5 Qwen。
队列微测试只提供正确性和原子争用机制证据，模型性能决定最终选择。

### 历史原语证据

已删除的 `scheduler_linked_queue.h` 曾实现单项/批量领取和有界发布原语，AIC/AIV CCEC 编译探针通过。
对应 CPU 用例验证未挂链 producer 暂停、已挂链 tail 落后、消费越过 tail、耗尽重试保留批次、
空/非空转换、过期 head、非法输入、最后任务唯一领取和并发 payload 可见性。
CPU 原语测试与并发重复结果见原语证据（本地 `linked-queue-primitives.json`）。
L/G 接入后新增生产池布局、跨原 shard 领取、MIX placement 失败、滞后 tail 引发 RETRY 后合并新工作、
full slots 下发布、abort 保留私有链和非法 link 防护测试；原有并发池测试扩展到 L/G。
三候选同卡 Qwen 的 L/G、T/G、T/H8 分别为 51.179、49.532、40.960 ms，完整 out/K/V 均通过。
当时保留 T/H8 为默认；L/G 仅作为独立算法对照，不作为性能胜者。常规硬件回归与故障矩阵状态见
生产接入证据（本地 `linked-ready-pool-candidates.json`），与原语证据分开记录。
