# A5 HBG SPMD 的 block 与 Mix 执行

[返回设计总览](../a5-hbg-spmd-design.md)；前置协议见 [运行时有序准入](ordered-execution-protocol.md)。
普通任务可分波，sync_start 需要共同驻留；两者共享参与者同序的合同，
但不能直接共用一个不区分容量的领取循环。

## 普通多 block 的动态领取建议

### 唯一领取与分波

任务共同准入时冻结成员集合，初始化 next_block 为 0，保留原始 N。
纯 AIC/AIV 由成员核领取；Mix 由候选 cluster 的 AIC 以整组为单位领取。
各 lane 不得分别从全局 cursor 获取自己的 block_idx。

首版建议一次领取一个 block，使用有界 CAS 将 next_block 从 b 改为 b 加一；
只有成功者拥有 b，b 不小于 N 时不再递增。查询可以预筛选，但不能代替 CAS 所有权。
一次成功领取不可撤销，直接绑定领取核／cluster，不向已经关闭任务的其他核转移。

普通任务的同一核可执行多个 block。8 个核执行 32 个 block 时，一个核返回便可继续领取，
不要求每波 8 个都做完才开始下一波。纯 C/V 的慢 block 不要求其他 block 等它返回；
Mix 的组内协调责任见后文。
若 kernel 本身包含跨 block 全员等待，就不是这种普通分波合同。

领取后准备本核 LocalContext：block_idx 为领取值，block_num 始终为原始 N。
不重复给 tensor 指针加 block 偏移，不把逻辑子槽直接当物理 lane，
保留已有纯 AIV 归一化、kernel 自身切片及共享 workspace／reduction 的语义。

### 什么时候可以离开 A

对上述不可撤销、无转移的 cursor 协议，一个充分的 LocalDone 条件是：

1. 本核以有所有权语义的检查确认领取范围已耗尽，不会再获得 A 的新 block。
2. 本核已经领取的 A 工作全部执行结束，没有预取、参数准备或 kernel 在途。
3. 本核的完成贡献已经发布，通信与局部共享资源义务已经关闭。
4. 本核随后只向前推进，不再领取 A。

~~~text
两个核，A 有两个 block，已分别领取：

核 0：A.block0 完成 → 发布贡献 → 关闭 A → 进入独立 B
核 1：A.block1 仍执行 ─────────────────→ 完成 A
~~~

领取耗尽不是执行结束。其他核拥有的 block 可以仍未返回，甚至仍在准备参数；
只要它们的所有权已经不可逆地绑定，本核不用等它们完成或等它们发布 LocalDone。

不能脱离上述条件，只看 next_block 达到 N 就离开。例如 cursor 预留后还要交给另一个核、
允许退回／转移或存在未完成组绑定时，数值到 N 不代表分配已经封闭。
本参考将成功 CAS 的执行者直接视为所有者；若实现拆成多个领取步骤，必须补充事务状态。

只要 A 仍有未领取工作，本核就不能仅因完成一个 block 而进入 B。
反过来，即使 N 大于成员数，满足上述四个条件后也可离开，不必等其他核的尾部。
N 小于成员数时，没有抢到 block 的成员完成耗尽检查便可关闭；真正不在成员集合的核直接跳过。
只有零工作时的临时队列空、CAS 失败或一次过时观察，都不是上述耗尽证明。

### 暂不引入的优化

批量领取、动态缩放成员、已领取工作的转移及跨前沿 work stealing 不作为首版前提。
后续批量领取要把整个已领取范围计入本核义务；不能只完成范围头就离开。
核发生错误时整轮失败，不把其领取自动退回 cursor，让另一个核重复执行有副作用的 atomic。

## sync_start 的后续分配策略

### 保证执行位置，不等待同时起跑

sync_start 可以仍在运行时选择参与者，但每个相互等待的 block 必须绑定独立执行位置。
编码契约选用冻结候选集合内的动态领取、每个执行位置至多一次：纯 AIC 对 AIC，
纯 AIV 对 AIV，Mix 对符合 mask／ABI 的不同 cluster 组。
在共同准入时固定单射映射只是后续可比较的替代策略，不是当前实现的要求。

这些位置可以还在执行更早的任务，不要求准入时全都空闲。
公共顺序保证它们最终会结束前序义务并进入当前任务，不能被后序相交任务截走。
早到核可立即开始；不增加全 block 到齐后才准许执行的启动 barrier。

同一核完成自己的当前 block 和局部资源收尾后可以进入后序任务，
其他核尚在当前 kernel 的尾部是合法的。不等待整个 task 的完成计数达到 N。

### 与普通领取的区别

普通任务允许一个快核先后领取多个 block；需要共同驻留的任务不允许同一位置
占有两个相互等待的 block，即使它有两个 pending slots。
动态领取本身不是问题，**没有每核一次的绑定与容量保证的普通 cursor 才是问题**。

超过共同驻留容量的 task 在 kernel 启动前拒绝，不能用分波或较长 timeout 伪装支持。
普通 512-block CSA 则不能被这个限制误伤；普通任务按可分波容量规划，
仅 sync_start 按其全部同时驻留需求检查。类型容量和 Mix 配对容量分别计算。

也可以在固定的足够大成员集合内逐个动态认领：每个执行位置至多领一个 block，
尚未领取的成员保留参与义务，直到领到自己的 block 或确认全部 block 已有归属。
剩余 block 数不超过剩余可领取位置数，且这些位置不能越序进入后序协同 task。
放行早到核前要建立这个可达性保证，不需要等待实际领取全部完成，更不需要全员到齐。

## AIC 有限协调的 GM Mix 协议

### 职责边界

AIC 可以负责本 cluster 的整 block 领取、组身份发布、成员完成聚合和组资源复用。
它不替 AIV 准备执行 payload，不代其运行 scheduler，也不持有整个 task 的执行锁。
所有 kernel 调用前都释放准入／发布临界区，避免对方需要该控制路径才能到达。

mask 6 只有两个 AIV kernel；AIC 仅做控制服务，不计入核执行总数。
协调者和必要资源 lane 都列入成员集合，不能把它们误当非参与者后放去执行后序阻塞工作。

### 一组一份身份

Host 为每个候选 cluster 预分配一个可复用 GM mailbox，
另为每个 Mix task／cluster 预分配保留到 run 结束的 close seal。
AIC 领取 b 后发布 task_ref、order_key、block_idx、mask、物理组、run_epoch 和组序号。
跨运行仍须 reset 和校验代次；普通两个参数 slot 的复用另按生命周期保护。

每个 cluster 首版任意时刻只保留一个执行中的 Mix 组，包括不同 task 之间。
AIV 读取 AIC 发布的同一个组身份，自行装配自己的 LocalContext 与参数；
不能因为 task_ref 相同就默认配对，也不能独立领取恰好不同的 block。
全局 block 编号不要求在每个 cluster 连续，局部组序号则必须连续。

记录分离 AIC 发布字段与各 lane 的到达／完成字段；每个 lane 只写自己的 token。
多核不能用普通 store 合写一个 mask，需聚合时使用明确 owner 或已验证的原子。
调度代码中的 SSBUF 指针检查、初始化、发布和完成读取一起迁移，不只替换一处通知。

### 本地资源的进入和退出

TPipe 等机制可能使用同组缓冲和事件；执行 mask 不等于完整资源依赖。
首版可保守地把 Mix 资源域限定为所属 cluster，将相关非执行 lane 纳入资源义务；
没有证据不能扩大到整个芯片，也不能给纯 AIC 任务附加无关 AIV 义务。

~~~text
AIC 发布本 block 组身份
  → 各相关 lane 关闭前序局部资源义务，发布 ARRIVED
  → 本地资源参与者均 ARRIVED
  → 各执行 lane 自备参数并进入同一 block 的 kernel
  → 计算、通信收尾完成，各自发布 EXEC_DONE
  → AIC 汇总本 block 完成；满足局部资源复用条件
  → 领取本 task 的下一个 block，或发布本 cluster 不再有该 task 的工作
~~~

这是本地组的资源确认，不等待其他 cluster 的 A。
cluster 0 关闭自己的 A 后可以进入 B，cluster 1 仍在 A 是合法状态。
后续 sync_start 仍允许某些 cluster 提前进入，仅保留必要的 cluster 内资源确认。

AIC 的“本 cluster 不再有 A”必须携带最后组序号，并晚于其领取和所有组发布；
AIV 消费完该序号之前的组、完成本地收尾后才可关闭 A。
不能因为自己暂时没看见新 dispatch 就退出 A，也不能在 AIC 还有未发布组时放行 B。

不得把 ARRIVED、写过请求、kernel 返回、通信已结束四件事混为一个标志。
旧 block 全部必要 lane 发布 released_seq、保证不再访问旧 descriptor/payload 后，
AIC 才可复用 mailbox 及硬件资源。close seal 按 task 保留到整轮安全结束，
不会被后续 task 的 mailbox 复用覆盖；慢读者核对 epoch／group_seq，不把新组当成旧组确认。

## 完成聚合和设备可见性

### 建议按 block 汇总逻辑完成

当前 Mix tracker 已聚合各 lane 完成。建议在其基础上区分两层：

- 纯 AIC/AIV：一个 kernel 完成对应一个 block 完成。
- Mix：协调 AIC 收齐本 block 的 active mask 后，贡献一次 block 完成。
- task：N 个 block 均完成，最后有效贡献者唯一发布 TaskDone，并推进一次 wake-list。

核执行身份总数仍为 N 乘 active_subtasks，用于 trace 和不重不漏检查，
不能因为逻辑 counter 改为 N 就少检查 AIV 或把协调 AIC 算作 kernel。
现有接口的计数合法范围仍按其原类型校验，内部 counter 扩宽不自动放宽公共合同。

block counter 本身不能排除“一次重复加一次遗漏”。唯一领取、每 lane 代次消费、
Mix completed mask 及本地退役状态一起保证每个执行和每个 block 恰好贡献一次。
本核关闭 A 之前要完成自己的贡献；AIC 不能带着未聚合的 A 组进入后序阻塞 kernel。
全图逻辑 task 完成数只加一次，不能每退役一个 block 就增加一次 pending_completed。

### 发布链

~~~text
kernel 输出和通信收尾
  → lane 完成发布
  → Mix 协调者观察全部必要 lane 完成（纯核略过）
  → block 完成贡献
  → 最后贡献者发布 TaskDone
  → 既有 wake-list 推进后继，后继按合同消费数据
~~~

每一条箭头都需要明确设备 ordering 和可见性，不因用了 atomic 就自动成立。
各执行核写自己的 LocalContext；共享 task 参数保持只读，
不能让多个 block 共写一个 context 或对共享输出做相互覆盖的整 tensor 缓存写回。

f5cc679bb 的 scheduler_gm_store 在 AICore 上使用 atomicExch；
scheduler_gm_query 是 ld_dev 观察／预筛选，关键链接的 scheduler_gm_load 使用 atomicAdd(0)。
因此不能为新协议随意退回普通 st_dev 发布，也不能把所有热轮询都升级为同地址原子读取。
逐字段复用或验证实际原语，必要的 writeback、invalidate 和 barrier 单独列清；
Host/sim 的 memory_order 参数不等于 AICore 已自动实现相同语义。

至少审计：准入尾部与记录、成员集合、cursor 所有权、组身份、lane 到达／完成、
cluster 关闭、TaskDone 及退出标志。不同 writer 的字段不能因缓存行写回而互相覆盖。
参数消费的 acquire 还必须覆盖 payload，不只是观察到了一个控制字。

## 错误和验收边界

任何 block 失败都不补写成功计数；局部组失败也不填满 EXEC_DONE 让后继运行。
错误要能定位 run、task、block、lane、order_key 和最近发布代次。
某核已卡进 kernel 时，不允许释放其 GM、参数或复用记录来“解除”死锁。

重复执行同图以及改变 N、shape 时重新初始化全部可读控制字段，
确认上一轮所有设备访问安全结束后再复用。generation／序号溢出不能静默回绕。

普通分波先验收不同速度、领取耗尽但仍有在途 block、零工作成员和局部退出；
Mix 增加配对错位、mask 6 控制者、资源 lane、关闭标志早到和慢读者；
sync_start 后续再验收共同驻留及任务内 barrier。
完整项目见 [边界条件和验证](corner-cases-and-validation.md)。
