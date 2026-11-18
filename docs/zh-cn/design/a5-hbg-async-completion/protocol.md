# A5 HBG 异步调度与完成协议

**共同前提：** 本页的 A5 调度设计遵守 [A5 硬件与调度统一原则](../a5-hbg-graph-task/build-execute-separation.md#a5-hardware-principles)。与本原则冲突的旧设计按此修订；本页描述当前 CntNotify 计数器路径及 GM 回退，未来 STARS 后端不属于当前合同。

[返回总览](../a5-hbg-async-completion.md)。记录身份与布局见[接口文档](interfaces-and-layout.md)，外部条件观察和 payload 可见性见[后端文档](backends-and-lifetime.md)。计数器分配及阈值见[构建与执行分离设计](../a5-hbg-graph-task/build-execute-separation.md)。

## 1. group 与 lease 状态机

CHECK 使用 64-bit lease；首次发布采用唯一写者初始化，发布后的转换使用原子操作。64-bit 是值宽，独立竞争单元的目标布局为 128B L2 侧隔离，不是 64B DCache 隔离。低 3 位为 phase，高 61 位为 lease generation；phase 编码为 DORMANT=0、RUNNABLE=1、OWNED=2、TERMINAL=3、FAILED=4，其余编码非法。lease 在首次发布前不具有可读取的初值，目录零位使其对扫描器不可见。唯一首次封口者先发布进度，以 bypass store 初始化 RUNNABLE(1)，完成 DSB 后发布目录位；不先读取或 CAS 未初始化的 lease。不得重新初始化已发布或 OWNED 的记录。交权时 generation 递增，溢出报协议错误，不能回绕。run identity 保存在不可变身份区，不能用 lease generation 替代。

| 转换 | 操作者 | 必须先完成的动作 |
| ---- | ------ | ---------------- |
| 未发布 → RUNNABLE(1) | 唯一首次封口者 | 发布封口数据及初始进度；单写者初始化 lease 并完成 DSB，之后设置目录位 |
| RUNNABLE(g) → OWNED(g) | 任意参与核 | 完整版本 CAS 成功；成功后才允许读取可变进度 |
| OWNED(g) → RUNNABLE(g+1) | 当前 owner | 发布进度及屏障；目录位保持一 |
| OWNED → TERMINAL | 当前 owner | 本记录负责的检查与最后进度写入结束；清目录位后结算完成责任 |
| 首次封口失败 / OWNED → FAILED | 尚未发布的执行写者或当前 owner | 首次封口失败可初始化 FAILED；重复封口只能报错，不能覆盖已发布 lease。锁存首错并保留 backing，不贡献正常完成 |

失败监督者不能覆盖仍由某核持有的 OWNED，也不通过超时抢占 lease。owner 故障按整个 run 的错误关闭处理，不恢复半写的短事务。

发现候选时先检查索引范围及记录类别。成功领取并按发布合同观察不可变身份后，核对 run、task、block/subtask 和 group 映射，再读取可变进度；非法索引不能形成地址访问，身份不符不能通过回滚 lease 继续运行。

kernel 返回后的执行尾部先核对登记错误、开放预留、count 与容量，再封口。非零条件组在 GM 路径先增加 task 的 pending 责任；计数器路径的条件 credit 已由 Host 规划，不更新 GM join。两条路径都初始化未满足 mask 和游标，发布数据、RUNNABLE 及目录位后才释放本次执行的局部义务。零登记/skip 不创建 CHECK 责任，也不初始化或发布 lease。任何提交副作用不明或登记错误都禁止正常 TaskDone。

封口发布只写回 header 的可变前缀及独立 token 区的 `[0, count)`，然后完成一次发布屏障。header 的只读尾部已由 Host 发布，预留但未登记的条目没有读取者。零登记组只发布可变 header，不更新 join、lease 或 CHECK 进度，也不触及预留后缀。计数器路径仍必须结算该 lane 已预留的条件 credit：普通执行可与物理 credit 合并，MIX 则在空封口尾部结算条件 credit。分配容量仍是登记上限，不表示每次执行都需要发布整个预留区。

CHECK 每次取得一个 group，按条件游标观察尚未满足的条件。一次 slice 不重复观察同一未满足条件；超过预算的条件在后续 slice 继续。READY 位不再重读，前提是本次等待期间条件单调且不会提前重置。COUNTER32 与 SDMA post-done 通过相应宽度的读取判断完成；CHECK 不代替消费者的数据获取。后端若确有额外发布动作，只在证实其必要性后接入有界步骤。

## 2. 活动目录与内存顺序

当前仅 CHECK 拥有平坦位图，每个字包含 64 条记录。GM 的 FINISH 是 join RMW 授予的内联完成权，不分配独立记录、lease 或目录。每位在一个 run 中只经历一次 `0→1→0`，重试不修改目录。所有写者只使用专用目录原语；运行中禁止 cached store、整行 writeback、初始化或复用。

### 有界目录更新

初始 compare 值可由可靠 ld_dev 获得，最终更新由 strong CAS 裁定；不为纯读取增加 atomic-add-zero。strong CAS 只改变目标位。置位前必须为零，清位前必须为一；状态不符直接返回协议错误，不改邻位。CAS 失败返回的当前值作为下一次 compare 值。

对合法调用，其余 63 位各最多改变两次。当前实现保守保留最多128次 CAS，加一次初始 ld_dev，覆盖初次 compare 失配、其余位变化和最后成功；后续 compare 复用上次 CAS 返回值。超过此界限是协议损坏，不是可忽略的 RETRY；调用者锁存 run 错误、禁止进入新业务并保留资源。此上界不适用于会反复重用的普通 ready 位图，也不表示固定硬件耗时。

该原语不能独立识别一位完整清除后的非法再次发布，调用者必须用 group 生命周期和 generation 维持一次性合同。禁止用算术补偿恢复重复更新。

### 发布与观察顺序

| 共享数据 | 写入与发布 | 读取 |
| -------- | ---------- | ---- |
| task→group 映射、task 形状、header 数组基址与 token 边界 | Host 初始化并随 run 镜像发布，run 内保持只读 | run 入口完成全 DCache 获取后普通 cached 读取，不逐项重复失效 |
| 封口身份、token | 执行核完成对应 cached 写回及实际屏障，再发布 RUNNABLE 和目录位 | 成功领取后按实际 cache 合同失效/观察，再消费数据 |
| 已发布的 lease、join、目录、task 完成状态 | 唯一领取、交权和多写者合并使用 atomic；无 cached 整行写回，目标按 128B 隔离，相关依赖边界显式 DSB | 可靠 ld_dev 用于条件观察或 CAS compare 值；不能用读取授予独占权，也不追加 FAA(0)确认。最终状态转换仍由 RMW 裁定 |
| run 异步活动标记 | 现有布局为独占 64 B DCache 行，并非 128B 争用隔离的证明；Host 在 run 前置零，执行核在首次发布 CHECK 前以 bypass store 写一并完成 DDR 屏障 | bypass load；本核见一后可以一直缓存 true，run 内不清零 |
| owner 可变进度 | 独占期间 cached 写入，对应行写回及屏障完成后交权 | 下一 owner CAS 成功后失效并观察进度行，完成读取屏障后读取 |
| payload、外部完成记录 | 遵守相应 backend 的可见性合同 | 不能将运行时 atomic-add-zero 原语用于 DMA/peer 写入的外部地址 |

AICore helper 的 `acquire/release` 参数不自动产生全部所需屏障；实现必须分别落实缓存维护、设备屏障及原子线性化。CPU/sim 的 C++ 原子语义不能替代这项设备实现。

首次发布的尾部必须完成目录置位后才允许进入业务 kernel。终态按“最后写入 → TERMINAL → 清目录位 → 完成贡献”的顺序执行。每条实际时序依赖以 DSB 建立，包括连续 atomic 之间；源码顺序和 memory_order 名称不构成 A5 SoC 保序证明。目录暂空、陈旧活动位、CAS 失败或 OWNED 均不能证明 run 完成。

## 3. 按实际登记汇合与任务形状

GM 回退路径中，每个可登记的逻辑 task 使用一个初值为零的 atomic-only join word：低 16 位为未完成非零 group 数 P，接着 16 位为累计登记 group 数 R，第 32 位为 computation-closed，其余位为零。R 保留“曾经登记”信息，即使所有条件提前满足，也不会把该 task 误认为同步任务。P ≤ R ≤ 32767，Host 继续检查物理执行展开上限。这里 P 表示未完成条件组数，区别于计数器方案中的物理 credit 数；计数器路径使用预规划的物理与条件 credit，不走此 join。

| 转换 | 唯一操作者 | 顺序及返回结果 |
| ---- | ---------- | -------------- |
| P、R 各加一 | 非零组封口者 | 必须先于 CHECK 发布及本次执行资源释放；closed 后禁止新增 |
| P 减一 | 最终 CHECK owner | 条件全部满足、group 终态及目录清位后；旧值 P=1 且 closed 时获得内联 FINISH 完成权 |
| 设置 closed | 真实计算汇合的最后贡献者 | 全部物理子任务已返回并封口登记；trace 和 MIX 资源按各自收尾顺序完成；旧值 P=0、R>0 时获得内联 FINISH 完成权 |

GM 路径的 canonical completion gate 仅额外保存一个 `has_async` 门闩，不重复累计 R。非空组先执行 join.register；旧值 R=0 的唯一首注册者用原有校验 CAS 将 canonical debt 加一，其他组直接继续发布。首注册者建立门闩之前不会提交自己的物理完成，因此即使其他组先返回或 CHECK 完成，remaining 仍至少为一，不会提前 DONE。R>0 而门闩尚未建立是允许的短暂状态，其他组无需等待它。真实异步计数仍由 join 的 P/R 表达。

每次转换只使用一次 unsigned GM fetch-add，并根据它返回的旧值判断结果。错误状态使用既有 run 首错 latch；非法位、溢出、下溢或重复 close 不作算术补偿，也不能获得完成权。唯一执行及 lease 身份防止重复贡献，计数校验不能替代身份验证。

GM 计算汇合使用统一的物理子任务计数，总数为 block 数乘 active lane 数。普通/SPMD 执行在返回并封口后贡献一次 `completed_subtasks`；MIX coordinator 收齐全部资源 release 后，按 active lane 数批量贡献，计算 lane 不重复贡献。最后贡献者依据同次 RMW 中的单个异步门闩决定是否关闭 computation。普通执行的贡献可以先于执行 trace 发布；MIX 仍遵守各 lane 的 trace → release → coordinator 汇聚顺序。释放 slot 的业务绑定前保存稳定身份，不能从已清空的 slot 反查。

P 暂时为零不证明计算结束，其他尚未返回的成员仍可能登记。仅在 GM 路径，设置 closed 或递减最后 P 的原子返回值授予 `FINISH_OWNER`。获权者立即调用 `scheduler_async_complete_task`，在同一尾部完成 canonical DONE 发布，不创建 FINISH lease、目录、resolver 游标或可让出的服务记录。普通读取不能授予此完成权。

所有成员均零登记时，GM 路径不创建 CHECK 工作，最后物理贡献沿用同步完成。门闩为一时依据 join 的 close/settle 结果内联完成或继续等待；不能根据先返回成员的零登记或暂时 P=0 推断整个任务同步。计数器路径即使零登记，也必须结算预留的条件 credit。全局逻辑完成计数可以早于最终 trace 写出；MIX 的物理汇聚仍须等待完整资源 release。存储回收必须等待全部 trace、资源义务和 EXIT 握手完成。

### 不同任务形状

- **普通/SPMD：** GM 非零条件组登记并结算自己的 pending 责任；计数器路径分别结算物理和条件 credit。每个实际执行参与计算汇合；分波执行不要求同时驻留。
- **MIX：** lane 的条件可以早于或晚于其他 lane 返回。条件 group 完成不代替物理子任务完成或 coordinator 的资源退役，也不长期占用 MIX mailbox 或 cohort。inactive AIC 仍须发布资源 release；正常 EXIT drain 可补齐已到达的 inactive coordinator，不能执行尚未返回的业务 kernel。
- **sync-start：** 启动许可继续约束原计算 cohort。条件未满足不会长期占用已返回执行的资源；已取得执行许可的核不得夹入 CHECK 服务；GM 内联 FINISH 只发生在合法完成尾部。若特定读取路径确需额外本核获取，按[消费者可见性](consumer-preparation.md)在最终许可前完成，不将其作为所有任务的固定阶段。
- **predicate skip：** 合法 skip 走 runtime 封口和执行资源释放尾部；业务 kernel 未调用时不创建 CHECK pending 责任；计数器路径仍结算规划内的物理及条件 credit。
- **Graph：** 每个实例的 child 使用独立 group/join。child 的 TaskDone 通过原 body→EXIT 依赖推进，ENTER/EXIT 本身不分配执行 group。未激活 child 仍保留责任；不能把未执行当成取消。

## 4. GM 内联 FINISH 与计数器完成

GM join 的 close 或最终 settle RMW 选出唯一 `FINISH_OWNER`。调用者立即执行 `scheduler_async_complete_task`，以不可变任务几何和 join 结果构造预期 gate，再通过 CAS 将 `(N, 0, 1)` 转为 DONE。完成记录、任务 DONE 信号和本核 `pending_completed` 在这一固定尾部发布；不能把完成权入队、让出后重新领取或交给另一个 owner。

计数器路径由 Host 规划每个 task 的阈值及真实祖先之间的链复用。普通执行返回贡献物理 credit；MIX coordinator 收齐全部 release 后贡献物理 credits。可延迟完成的 lane 额外预留一个条件 credit：非空 group 在最后 CHECK 满足后贡献，空组在封口尾部贡献。普通空组允许在返回尾部连续提交两次单位 INC，MIX 则保留物理与条件贡献各自的边界。

后继通过计数器阈值直接判断 readiness，后台唯一记账分区随后发布 canonical GM DONE 并结算 `pending_completed`。阈值就绪不依赖该后台记账核立即获得调度，也不代表 run 资源已经可以回收。没有可用计数器映射的任务继续使用 GM 完成门。

当前后继由 token/control 调度逻辑检查自己的 fanin；完成尾部不再遍历 CSR、摘取 wake-list、续接 resolver 或向 ready 队列推送后继。CHECK 保留可续接的条件观察；FINISH 不再是一类独立控制记录。

若启用受支持的 ArgsDump AFTER，输出采集及其封口必须发生在 TaskDone 之前。尚未接通这一边界的组合在 bind 前拒绝，不能先公开 DONE 再等待输出采集。

ready 只授予 admission 的资格，不证明实际消费者核已经获取 payload。ENTER/EXIT 和内联 FINISH 不执行消费者的本核获取；同步消费者同样遵守[消费者数据可见性](consumer-preparation.md)。依赖与 admission 判断继续由原调度规则执行，完成贡献不能直接强制消费者进入 kernel。

CHECK 不计入业务 task 数。`pending_completed` 沿用持久核本地批量记账和 flush 机制，不能将局部记账等同于已发布的全局完成计数。逻辑 DONE、最终 trace 发布、MIX 资源退役和 run 存储回收分别满足各自的边界。

## 5. 服务点、预算与进展

正常 resident 循环按以下结构组织：

```text
退出/错误检查
  → 完成回收、Graph control、已有 ready 发布
  → 异步安全服务门
  → 推进既有候选与 cohort 汇合
  → 按既有规则尝试最终业务 admission
  → 已获许可则直接执行 kernel；否则返回未获许可的服务循环
```

所有正常 pending-publication、admission/cohort 暂不可进入的分支都经过安全服务门。前置服务必须能返回或具备独立上界；仅保证门在代码上可达不够。退出/错误路径可以直接收敛退出。

本核尚未观察到异步活动、没有业务返回且没有本核错误时，服务门首次探测，此后每 16 个空闲服务机会探测一次；中间继续普通调度，不执行空异步服务的 GM 读取。每轮至多计算一次此资格，跳过不能计为进展。有业务返回的安全服务点、本核错误及已激活工作不受此预算延迟；既有主循环错误检查保持原规则。实际探测到活动标记为零时，不扫描 CHECK 目录。首次非零登记在发布可发现工作之前置一。该标记不归零，也不用于判断 run 完成；各核只有观察到一后才永久跳过后续标记读取。消费者执行入口仍直接读取活动标记，不使用空闲探测预算。它与原子发布的目录不同，整行禁止 device cached 写入和 RMW，保证 bypass 发布/观察配对。

活动开始后的初始调度参数为：有可合法执行业务时最多累计 4 次业务返回，下一次 admission 前执行至多一个 slice；无可执行业务时每轮至多一个 slice。CHECK 从第一次活跃服务开始，每两个活跃服务机会执行一次扫描，并保持自己的 word/bit 游标。失败探测、遇到 OWNED 和失败 CAS 也推进游标并消耗额度；一次成功领取后不再领取第二条记录。

| 普通 slice 额度 | 初始上限 |
| --------------- | -------- |
| 目录字观察 | 4 |
| 候选 lease 探测 | 2 |
| 领取 CAS | 2 |
| 条件观察 | 8 |

上述额度不包含可省略的“免费尾部”。领取前必须为本次可能发生的进度发布、交权、CHECK 目录终态更新，以及 GM join/内联 FINISH 或计数器条件贡献保留独立最坏预算；目录更新按每次最多 128 个原子操作计价。达到工作预算后不再开始新工作，只完成必要尾部。不同类型操作的实际耗时由[性能验证](validation.md)校准。

最后一个 CHECK 清除自己的目录位后，GM 路径可能通过 join settle 获得内联 FINISH 完成权，计数器路径则贡献条件 credit。这一尾部只有 CHECK 清位的一次目录更新，最多 128 次 CAS 加一次初始 ld_dev；不存在第二次 FINISH 目录发布。lease、join、DONE/计数器贡献、数据发布和屏障另行计价。普通条件观察额度为 8 不限制已预留的固定尾部；该数值是协议上界，不是常态耗时或已测性能。

若后续因具体读取缺口增加分段获取，须纳入同一工作预算和公平性证明，不能各追加一份隐藏 slice。等待 cohort 的核继续经过安全服务门，最终许可后不再插入服务。cohort start 及既有已预留 ready 批次按实际操作数保留固定尾部预算。

每核使用 CHECK 游标；有扫描资格的循环最多领取一条记录，普通条件观察额度为 8。有业务可执行时，累计最多 4 次执行返回后在最终 admission 之前服务；没有可执行 slot 或仍有 ready 发布积压时，每轮经过一次服务门。最后一个不满 64 位的目录字扫描结束后，游标回到 bit 0；不能对总记录数取模后跳过前面的低位。

返回核封口时保留自己的 dirty 发布区，直接检查并写回 count/token；失效该区会丢弃刚登记的条件。其他核取得 lease 后再观察已发布的区和进度。普通/SPMD 的物理完成贡献晚于本次执行封口登记，不要求其他 lane 的资源 release 或最终 trace 已发布。MIX 的物理完成由 coordinator 在收齐全部资源 release 后聚合贡献；每核的 trace 发布仍先于自己的 slot 复用、MIX release 和 EXIT ACK，coordinator 随后退役资源。

CAS 失败立即返回正常服务，不等待某个 owner，不 sleep，也不给某记录设置固定重试时间。持续 PENDING、扫描与碰撞不刷新进展 watchdog；真实登记完成、条件推进、执行释放和后继发布才属于协议进展。

进展依赖参与核持续返回服务点、原子操作能完成及合法依赖图。所有核运行长 kernel 时不承诺固定微秒发现延迟；本功能不抢占业务 kernel。服务规则必须保证 CHECK、计数器记账、ready 和控制工作都不会永久饥饿。

无 cooperative admission 的普通执行，每个核只认领下一项待执行任务；已有本地 READY 执行时，不再把其他普通 ready 任务预取到第二个槽。否则 publisher 运行期间可能把尚未执行的 waiter 扣在同核，其他空闲核也无法领取。cooperative slot 和参与者协议保持各自约束；该普通任务认领策略的调度成本纳入同步性能回归。
