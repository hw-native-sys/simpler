# A5 HBG 异步调度与完成协议

[返回总览](../a5-hbg-async-completion.md)。记录身份与布局见[接口文档](interfaces-and-layout.md)，外部条件观察和 payload 可见性见[后端文档](backends-and-lifetime.md)。本协议定义基础实现，选型候选不能隐式改变它。

## 1. group 与 lease 状态机

CHECK 和 FINISH 都使用一个 atomic-only 64-bit lease word。低 3 位为 phase，高 61 位为 lease generation；phase 编码为 DORMANT=0、RUNNABLE=1、OWNED=2、TERMINAL=3、FAILED=4，其余编码非法。lease 在首次发布前不具有可读取的初值，目录零位使其对扫描器不可见。唯一首次发布者先初始化进度及 DORMANT(1)，再发布 RUNNABLE 和目录位；不得重新初始化已发布或 OWNED 的记录。交权时 generation 递增，溢出报协议错误，不能回绕。run identity 保存在不可变身份区，不能用 lease generation 替代。

| 转换 | 操作者 | 必须先完成的动作 |
| ---- | ------ | ---------------- |
| DORMANT → RUNNABLE | 首次发布者 | 发布封口数据及初始进度；之后在固定尾部设置目录位 |
| RUNNABLE(g) → OWNED(g) | 任意参与核 | 完整版本 CAS 成功；成功后才允许读取可变进度 |
| OWNED(g) → RUNNABLE(g+1) | 当前 owner | 发布进度及屏障；目录位保持一 |
| OWNED → TERMINAL | 当前 owner | 本记录负责的检查与最后进度写入结束；清目录位后结算完成责任 |
| 首次封口失败 / OWNED → FAILED | 尚未发布的执行写者或当前 owner | 首次封口失败可初始化 FAILED；重复封口只能报错，不能覆盖已发布 lease。锁存首错并保留 backing，不贡献正常完成 |

失败监督者不能覆盖仍由某核持有的 OWNED，也不通过超时抢占 lease。owner 故障按整个 run 的错误关闭处理，不恢复半写的短事务。

发现候选时先检查索引范围及记录类别。成功领取并按发布合同观察不可变身份后，核对 run、task、block/subtask 和 group 映射，再读取可变进度；非法索引不能形成地址访问，身份不符不能通过回滚 lease 继续运行。

kernel 返回后的执行尾部先核对登记错误、开放预留、count 与容量，再封口。非零条件组先增加 task 的 pending 责任，再初始化未满足 mask 和游标，发布数据、RUNNABLE 及目录位后才释放本次执行的局部义务。零登记/skip 不创建 CHECK 责任，lease 保持 DORMANT。任何提交副作用不明或登记错误都禁止正常 TaskDone。

封口发布只写回 header 的可变前缀及独立 token 区的 `[0, count)`，然后完成一次发布屏障。header 的只读尾部已由 Host 发布，预留但未登记的条目没有读取者。零登记组只发布可变 header，不更新 join、lease 或 CHECK 进度，也不触及预留后缀。分配容量仍是登记上限，不表示每次执行都需要发布整个预留区。

CHECK 每次取得一个 group，按条件游标观察尚未满足的条件。一次 slice 不重复观察同一未满足条件；超过预算的条件在后续 slice 继续。READY 位不再重读，前提是本次等待期间条件单调且不会提前重置。COUNTER32 与 SDMA post-done 通过相应宽度的读取判断完成；CHECK 不代替消费者的数据获取。后端若确有额外发布动作，只在证实其必要性后接入有界步骤。

## 2. 活动目录与内存顺序

CHECK 与 FINISH 分别拥有平坦位图，每个字包含 64 条记录。每位在一个 run 中只经历一次 `0→1→0`，重试不修改目录。所有写者只使用专用目录原语；运行中禁止 cached store、整行 writeback、初始化或复用。

### 有界目录更新

初值由权威原子读取获得，strong CAS 只改变目标位。置位前必须为零，清位前必须为一；状态不符直接返回协议错误，不改邻位。CAS 失败返回的当前值作为下一次 compare 值。

对合法调用，其余 63 位各最多改变两次，每次 strong-CAS 失败必对应至少一次值改变，因此最多 127 次 CAS，加一次初始原子读。超过此界限是协议损坏，不是可忽略的 RETRY；调用者锁存 run 错误、禁止进入新业务并保留资源。此上界不适用于会反复重用的普通 ready 位图，也不表示固定硬件耗时。

该原语不能独立识别一位完整清除后的非法再次发布，调用者必须用 group 生命周期和 generation 维持一次性合同。禁止用算术补偿恢复重复更新。

### 发布与观察顺序

| 共享数据 | 写入与发布 | 读取 |
| -------- | ---------- | ---- |
| task→group 映射、task 形状、header 数组基址与 token 边界 | Host 初始化并随 run 镜像发布，run 内保持只读 | run 入口完成全 DCache 获取后普通 cached 读取，不逐项重复失效 |
| 封口身份、token | 执行核完成对应 cached 写回及实际屏障，再发布 RUNNABLE 和目录位 | 成功领取后按实际 cache 合同失效/观察，再消费数据 |
| lease、join、目录、task 完成状态 | 专用原子操作，控制行无 cached 写入 | 权威判断使用原子观察；bypass 只能作允许过期的提示 |
| run 异步活动标记 | 独占 64 B 行；Host 在 run 前置零，执行核在首次发布 CHECK/FINISH 前以 bypass store 写一并完成 DDR 屏障 | bypass load；本核见一后可以一直缓存 true，run 内不清零 |
| owner 可变进度 | 独占期间 cached 写入，对应行写回及屏障完成后交权 | 下一 owner CAS 成功后失效并观察进度行，完成读取屏障后读取 |
| payload、外部完成记录 | 遵守相应 backend 的可见性合同 | 不能将运行时 atomic-add-zero 原语用于 DMA/peer 写入的外部地址 |

AICore helper 的 `acquire/release` 参数不自动产生全部所需屏障；实现必须分别落实缓存维护、设备屏障及原子线性化。CPU/sim 的 C++ 原子语义不能替代这项设备实现。

首次发布的尾部必须完成目录置位后才允许进入业务 kernel。终态按“最后写入 → TERMINAL → 清目录位 → 完成贡献”的顺序执行。目录暂空、陈旧活动位、CAS 失败或 OWNED 均不能证明 run 完成。

## 3. 按实际登记汇合与任务形状

每个可登记的逻辑 task 有一个初值为零的 atomic-only join word：低 16 位为未完成非零 group 数 P，接着 16 位为累计登记 group 数 R，第 32 位为 computation-closed，其余位为零。R 保留“曾经登记”信息，即使所有条件提前满足，也不会把该 task 误认为同步任务。P ≤ R ≤ 32767，Host 继续检查物理执行展开上限。

| 转换 | 唯一操作者 | 顺序及返回结果 |
| ---- | ---------- | -------------- |
| P、R 各加一 | 非零组封口者 | 必须先于 CHECK 发布及本次执行资源释放；closed 后禁止新增 |
| P 减一 | 最终 CHECK owner | 条件全部满足、group 终态及目录清位后；旧值 P=1 且 closed 时获得 FINISH 激活权 |
| 设置 closed | 真实计算汇合的最后贡献者 | 全部物理子任务已返回并封口登记；trace 和 MIX 资源按各自收尾顺序完成；旧值 P=0、R>0 时获得 FINISH 激活权 |

每次转换只使用一次 unsigned GM fetch-add，并根据它返回的旧值判断结果。错误状态使用既有 run 首错 latch；非法位、溢出、下溢或重复 close 不作算术补偿，也不能获得完成权。唯一执行及 lease 身份防止重复贡献，计数校验不能替代身份验证。

计算汇合使用统一的物理子任务计数：每个实际执行 lane 在返回并封口登记后贡献一次 `completed_subtasks`，总数为 block 数乘 active lane 数。最后贡献者依据同次 RMW 中累计的保守异步票决定是否关闭 computation。贡献和后继解析可以先于执行 trace 发布；MIX coordinator 收齐资源 release 后只负责资源退役，不重复贡献逻辑完成。释放 slot 的业务绑定前保存稳定身份，不能从已清空的 slot 反查。

P 暂时为零不证明计算结束，其他尚未返回的成员仍可能登记。只有设置 closed 或递减最后 P 的原子返回值授予 FINISH 激活权；普通读取只可验证已有 FINISH 的就绪条件，不能再激活。获权者先初始化完整 resolver 游标和 lease，再发布预分配 FINISH 的 RUNNABLE 和目录位后返回，FINISH 继续受预算约束。

所有成员均零登记的普通、SPMD、MIX 或合法 skip task 沿用同步完成及本核后继路径，不创建 CHECK/FINISH 工作。最后物理贡献的保守异步票为零时可跳过 join；票非零时检查实际 P/R 并按 close 结果完成或激活 FINISH。不得根据先返回成员的零登记或暂时 P=0 推断整个任务同步；曾登记但条件已经满足的 task 仍进入 FINISH。全局逻辑完成计数可以早于最终 trace 和 MIX 资源收尾，但存储回收必须等待全部 trace、资源义务和 EXIT 握手完成。

### 不同任务形状

- **普通/SPMD：** 非零物理执行登记并结算自己的 P，每个实际执行通过统一计数参与计算汇合；分波执行不要求同时驻留。
- **MIX：** lane 的条件可以早于或晚于其他 lane 返回。条件 group 完成不代替物理子任务完成或 coordinator 的资源退役，也不长期占用 MIX mailbox 或 cohort。inactive AIC 仍须发布资源 release；正常 EXIT drain 可补齐已到达的 inactive coordinator，不能执行尚未返回的业务 kernel。
- **sync-start：** 启动许可继续约束原计算 cohort。条件未满足不会长期占用已返回执行的资源；已取得执行许可的核不得夹入 CHECK/FINISH。若特定读取路径确需额外本核获取，按[消费者可见性](consumer-preparation.md)在最终许可前完成，不将其作为所有任务的固定阶段。
- **predicate skip：** 合法 skip 走 runtime 封口和执行资源释放尾部；业务 kernel 未调用时不创建 pending 责任。
- **Graph：** 每个实例的 child 使用独立 group/join。child 的 TaskDone 通过原 body→EXIT 依赖推进，ENTER/EXIT 本身不分配执行 group。未激活 child 仍保留责任；不能把未执行当成取消。

## 4. FINISH 与可续接后继发布

FINISH 是无业务资源需求的控制记录，由一个 lease owner 推进。基础路径始终先发布到 FINISH 目录，再由服务循环领取。它和 CHECK 使用相同的交权规则。

若启用受支持的 ArgsDump AFTER，输出采集及其封口必须作为 TaskDone 之前的可续接服务完成，再进入下表的 DONE/关闭解析权短事务；不能先公开 DONE 再等待输出采集。未接通这一边界的组合在 bind 前拒绝。

`resolve_step(graph, cursor, budget)` 的步骤结果区分 WAITING、READY_TO_PUBLISH、YIELD、DRAINED 和 ERROR，并说明当前节点责任是否已转移。

| 阶段 | 操作与续接规则 |
| ---- | -------------- |
| 初始化 | 核对 join 与 run 错误，唯一发布 TaskDone 并关闭 wake head；CSR 路径要求旧 head 为空，兼容路径保存摘下的链；整个短事务不让出 |
| CSR 通知 | 保存遍历位置，每步对一个后继 remaining 原子减一；旧值 1 取得唯一 ready 发布权，旧值 0 报错 |
| 兼容 waiter | 无依赖 header 时先保存旧 `next_waiter`，再推进该节点的 fanin 状态 |
| 兼容 fanin/挂链 | 每步检查一个 producer 或尝试一次 CAS；失败计入预算并保存状态，挂链成功立即转交 waiter 责任 |
| ready 发布 | 保存总容量为 8 的唯一待发布批次、目标普通/control 队列及发布阶段 |
| 预留后发布 | ticket 一旦预留成功，在固定尾部完成该批次；禁止让出、再次预留或重新路由已发布节点 |
| 完成 | 所有后继、批次和诊断公开后，FINISH terminal、清目录位，最后贡献一次 `pending_completed` |

CSR cursor 随每条已消费的边前进，lease 交权后的 owner 不得重放 decrement。
无依赖 header 的兼容路径挂链成功后，原 FINISH 不再读写 waiter，因为下一 producer 可立即摘链并改写它。后续只读取自己保存的旧链引用。预算耗尽或队列暂时不可开始发布时，保存唯一责任后交权；不能重新扫描原 wake-list 重建进度。

ready 只授予 admission 的资格，不证明实际消费者核已经获取 payload。ENTER/EXIT 和 FINISH 不执行消费者的本核获取；同步消费者同样遵守[消费者数据可见性](consumer-preparation.md)。优先复用已有读取路径，不固定新增准备状态机。

普通 ready 与 Graph control 的入队都必须提供可识别的“尚未预留、已预留待发布、已公开”状态。内部仍有无界 fanin/CAS 循环，或把未完成发布藏进观察核私有描述符的 helper，不满足此接口。依赖与 admission 判断继续由原调度规则执行，FINISH 不能直接强制消费者进入 kernel。

TaskDone 可以早于 resolve 尾部；run 完成数必须晚于尾部。CHECK/FINISH 不计入业务 task 数。`pending_completed` 沿用已有持久核本地批量记账和 flush 机制，不把该局部值误认为已发布的全局完成计数。

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

本核尚未观察到异步活动、没有业务返回且没有本核错误时，服务门首次探测，此后每 16 个空闲服务机会探测一次；中间继续普通调度，不执行空异步服务的 GM 读取。每轮至多计算一次此资格，跳过不能计为进展。有业务返回的安全服务点、本核错误及已激活工作不受此预算延迟；既有主循环错误检查保持原规则。实际探测到活动标记为零时，不扫描 CHECK/FINISH 目录。首次非零登记在发布可发现工作之前置一。该标记不归零，也不用于判断 run 完成；各核只有观察到一后才永久跳过后续标记读取。消费者执行入口仍直接读取活动标记，不使用空闲探测预算。它与原子发布的目录不同，整行禁止 device cached 写入和 RMW，保证 bypass 发布/观察配对。

活动开始后的初始调度参数为：有可合法执行业务时最多累计 4 次业务返回，下一次 admission 前执行至多一个 slice；无可执行业务时每轮至多一个 slice。CHECK/FINISH 轮换优先级，每类独立保持 word/bit 游标。失败探测、遇到 OWNED 和失败 CAS 也推进游标并消耗额度；一次成功领取后不再领取第二条记录。

| 普通 slice 额度 | 初始上限 |
| --------------- | -------- |
| 目录字观察 | 4 |
| 候选 lease 探测 | 2 |
| 领取 CAS | 2 |
| 条件观察或依赖路由步骤 | 合计 8 |

上述额度不包含可省略的“免费尾部”。领取前必须为本次可能发生的进度发布、交权、目录终态更新、join 贡献、FINISH 激活和已预留批次发布保留独立最坏预算；目录更新按每次最多 128 个原子操作计价。达到工作预算后不再开始新工作，只完成必要尾部。不同类型操作的实际耗时由[性能验证](validation.md)校准。

最后一个 CHECK 可能先清 CHECK 位，再因 G 递减成为最后贡献者而发布 FINISH 位。这一尾部仅两次目录更新就最多需要 `2 × (1 次原子读 + 127 次 CAS) = 256` 个原子操作；lease、join、数据发布和屏障另行计价。普通工作额度为 8 不限制这个已预留的固定尾部，也不允许省掉其中一次发布。该数值是协议上界，不是常态耗时或已测性能。

若后续因具体读取缺口增加分段获取，须纳入同一工作预算和公平性证明，不能各追加一份隐藏 slice。等待 cohort 的核继续经过安全服务门，最终许可后不再插入服务。cohort start 及既有已预留 ready 批次按实际操作数保留固定尾部预算。

每核用独立 CHECK/FINISH 游标轮换选择一种服务；一个循环最多领取一条记录，普通工作总额度为 8。有业务可执行时，累计最多 4 次执行返回后在最终 admission 之前服务；没有可执行 slot 或仍有 ready 发布积压时，每轮经过一次服务门。最后一个不满 64 位的目录字扫描结束后，游标回到 bit 0；不能对总记录数取模后跳过前面的低位。

返回核封口时保留自己的 dirty 发布区，直接检查并写回 count/token；失效该区会丢弃刚登记的条件。其他核取得 lease 后再观察已发布的区和进度。物理完成贡献晚于本次执行封口登记，不要求其他 lane 的资源 release 或最终 trace 已发布。每核的 trace 发布仍先于自己的 slot 复用、MIX release 和 EXIT ACK；coordinator 收齐 release 后退役资源。

CAS 失败立即返回正常服务，不等待某个 owner，不 sleep，也不给某记录设置固定重试时间。持续 PENDING、扫描与碰撞不刷新进展 watchdog；真实登记完成、条件推进、执行释放和后继发布才属于协议进展。

进展依赖参与核持续返回服务点、原子操作能完成及合法依赖图。所有核运行长 kernel 时不承诺固定微秒发现延迟；本功能不抢占业务 kernel。服务规则必须保证 CHECK、FINISH、ready 和控制工作都不会永久饥饿。

无 cooperative admission 的普通执行，每个核只认领下一项待执行任务；已有本地 READY 执行时，不再把其他普通 ready 任务预取到第二个槽。否则 publisher 运行期间可能把尚未执行的 waiter 扣在同核，其他空闲核也无法领取。cooperative slot 和参与者协议保持各自约束；该普通任务认领策略的调度成本纳入同步性能回归。
