# A5 HBG SPMD 编码契约与接线设计

基线：`f5cc679bbf3d`，2026-10-23。[设计总览](../a5-hbg-spmd-design.md)
定义已确认语义；本页把它收敛为可开始编码的正确性基线。
共同准入、Host 预分配、纯 C/V 的 own/direct 入口、cursor 和 LocalDone 已接入生产路径。
GM Mix、完整协议 trace、原样 CSA resident 和 sync_start 仍未完成。
本页保留全部目标合同；已实现字段和实测范围以 [实现过程](implementation-progress.md) 为准，
不以存在字段替代执行验收。
初始 TDD 覆盖和历史红绿状态见 [测试入口和验收状态](test-plan-and-status.md)。

## 一、存储、身份与能力边界

### 沿用什么，新增什么

Host 构图、TensorMap、kernel 注册、内存规划、原始 tensor/view/标量 ABI 全部沿用。
device 沿用完成驱动的 wake-list、ready inbox、直接后继候选和每核执行循环。
不在 Scalar 建图、分配 heap，不复制一份每核 DAG，不把同一 intrusive ready 节点反复入队。

建议的最小数据分解如下；设备镜像内使用 offset/index，不放 Host 指针。

| 存储 | 字段／大小依据 | 写入者与生命周期 |
| ---- | -------------- | ---------------- |
| 现有 TaskMetadata | 原 N、mask、函数、参数、trace base | Host 初始化，本轮只读 |
| 现有 TaskControl 的原子区 | 新增 next_block、completed_blocks、admission_state | 领取者／完成者；只能原子访问 |
| AdmissionRecord | epoch、task_ref、order_key、成员位图、形状／策略、Mix seal offset | 唯一准入者准备；发布后只读 |
| AdmissionLog | 长度为可协同准入 task 数的 task_ref 数组；commit_tail、短提交锁 | 任意合法发布者；一次提交一个引用 |
| 本核本地状态 | scan_cursor、current_obligation、两个候选身份、当前 lease、sync_claimed | 仅本核修改，不增每核全图表 |
| 每 cluster 一个 GM GroupMailbox | 当前组 descriptor、published_seq、3 个 lane ack | AIC 写组身份，各 lane 写自己的 ack |
| MixCloseSeal | 每 Mix task、每候选 cluster 一个 64 位结束 token | 本 cluster AIC 单次发布，直到本轮退出不复用 |

TaskControl 的第一条 64B line 原有 state/wake_list_head/next_waiter 三个 64 位原子字段，
现已使用原 40B padding 中的 24B 承载 next_block/completed_blocks/admission_state，仍为 128B。
新增字段不得与第二条普通写回 line 混用；补 sizeof、offsetof、alignment、POD 静态断言。
AdmissionRecord 的位图按紧凑 worker index 编码，按实际 worker 数规划 word 数，
不用物理 core ID 直接寻址。独立热点控制对象以 128B 对齐／步长隔离，
这只是地址布局基线，不声称能够消除同一 cursor 的原子竞争。

当前布局为 64B record 头加实际 bitmap word 数，整条 record stride 向 128B 对齐，
避免不同准备者对相邻 task 位图整行写回时相互覆盖。
TaskMetadata 使用 int16 timing_slot 与 uint16 原 N，共享原 4B 尾部，维持 16B；
GraphSummary 的现有 padding 承载协议 offset／容量／Host run_epoch，维持 128B。
AdmissionControl 的 lock/tail 各占 128B；GroupMailbox 为 descriptor 128B、
published_seq 独占 128B、三个 lane ack 各 128B，共 640B。

纯 C/V 执行接线后，`SchedulerAdmissionLocalState` 为 184B，含两个各 48B 的候选；
`SchedulerLocalSlotState` 为 24B，含 block 身份和 predicate 缓存，
`SchedulerLocalState` 为 504B。候选只保留 task 所有权与重建所需状态，
不会增加物理并发容量，也不保存每核全图或全部 block 历史。

AdmissionLog 只存引用，记录按 task_ref 定位；不按 block 数扩大日志。
紧凑 log 单元和 close seal 都只能通过原子读写，不能普通写单元再 dcci 整行；
短提交锁不会自动消除不同发布者缓存中的旧行副本。
MixCloseSeal 为 8 × Mix task 数 × 候选 cluster 数字节，避免复用 mailbox 后丢失前任务关闭通知。
它不是第二套依赖图；所有大小都由 Host 按实际图分配，并做乘积／对齐溢出检查。
参考模型为了审计保存所有 block 历史，生产路径不必为每 block 常驻完整模型对象。

### 身份与数值边界

- 执行身份：`(run_epoch, task_ref, block_idx, subtask_slot)`。
- 排序身份：`(run_epoch, order_key)`；不使用 task_ref、各通道 slot generation 排序。
- 本地槽身份：仍须校验槽编号及该槽 generation；不能跨 own/Mix 通道直接比较原始 generation。
- Mix 组身份：epoch、task、block、order、cluster、group_seq、active mask、placement。
- Host 最窄边界是 `N > 0 && N * popcount(mask) <= INT16_MAX`，即 32767。
  resident maker 已改用该上界；uint16 的存储类型不代表允许到 65535。
- 位移、block/subtask 乘积、trace 前缀先用足够宽类型计算，再检查后窄化。
  原子计数和 seq 不回绕；临近上限在新领取／发布前失败。
- 无 kernel 的服务节点独立走既有逻辑完成，不假造一个 kernel block。

### 阶段能力判定

普通多 block 只要求有相应类型的执行位置，不要求 N 个位置同时空闲。
sync_start 的 N 必须不超过本轮冻结成员中不同执行位置数：C 按 AIC，V 按 AIV，
Mix 按完整 cluster；两个 pending slots 不是两个执行位置。
先开放普通 C/V、GM Mix、原样 CSA，再开放 sync_start。
多 block predicate、Graph body、异步返回先保留现有拒绝／整图回退，不顺手删除形状门禁。
普通单 block predicate 的现有语义必须保留。

当前 `scheduler_resident_task_shape_supported` 允许普通纯 C/V 的 N≥1；
`scheduler_resident_graph_shape_supported` 对“纯多 block + 任意 Mix”整图回退，
避免已接线的 own/direct 协议与尚未接线的 SSBUF Mix 通道混跑。
全 N=1 图中的既有 Mix 路径继续保留；无 Mix 的纯 C/V 调度不再要求 SSBUF。
这只是分步开放边界，不是最终允许削减多 block Mix／原样 CSA 的范围。

## 二、完成驱动准入及普通旁路的线性化

### 准入函数与状态转换

`scheduler_admission.h` 已提供 `scheduler_try_claim_admission`、
`scheduler_prepare_admission_record`、`scheduler_try_commit_admission`、
`scheduler_refresh_local_obligation` 和 `scheduler_can_enter_ordinary` 基础 helper，
参数复用现有 graph/state/local context。调用方先唯一取得 ready 候选，
仅成功 claim 才将其所有权转移给持久 pending；RETRY 不能让调用方丢弃候选。
本地协议状态必须成功初始化；epoch 0 是合法身份，不能拿它兼作“未初始化”哨兵。
纯 C/V 的 `scheduler_fill_ordered_slots` 发现／发布候选，executor 的
`scheduler_ordered_can_enter_slot` 在实际执行前校验许可；完成路径的 direct 候选
进入同一个持久候选池。GM Mix 入口尚未接入，不能声称三入口已经全部统一。

| 转换 | 条件与唯一动作 | 暂时不能推进时 |
| ---- | -------------- | -------------- |
| DATA_READY → CLAIMED | 既有 pop/refill 唯一取得逻辑 task；再校验一次状态 | 返回无工作，不丢候选 |
| CLAIMED → PREPARED | 冻结成员与角色、校验形状、初始化 cursor/计数 | 错误终止本轮；不假造完成 |
| PREPARED → ADMITTED | 短锁内填 order、发布记录与日志元素、最后推进连续 tail | 竞争失败保留候选，先做可推进控制工作 |
| ADMITTED → TASK_DONE | 所有 block 成功完成且可见 | 继续服务，不以领取耗尽代替完成 |

ready pop 只做一次；task 离开 ready 后由准入记录维护 block cursor。
**不采用“长时间 peek 队头，最后一块再 generic pop”的主方案。**
现有 generic pop 接收旧 endpoints 后会重试并可能取走后一个 task，
不是“仅删除我先前看见的 A”。若将来采用延迟摘头，必须另有 expected-task 条件摘除原语；
新增 C++ 回归已固定这一边界，不能误用旧接口。

准备成员在锁外进行。锁内只完成有限的记录提交：
检查 epoch/error/capacity → 取当前 tail → 填完整记录及引用 → 发布 → tail 加一 → 解锁。
不可先公开 fetch_add 的预留洞；不可持锁等待其他核、ready、kernel、组到达或 TaskDone。
持锁者异常走 run failure，不在其他核还可能使用记录时擅自“抢锁恢复”。

共同准入适用于普通多 block、所有多 lane Mix、sync_start。
纯 C/V 普通单 block 和无 kernel 服务节点不必逐个进入全局序列。
模型为简化会把有 kernel task 全部表示为准入节点；旁路须按下面合同另测，不能声称模型已覆盖。

### 成员冻结与每核前沿

纯 C 或 V 默认选择当前可用拓扑中的对应类型候选核；不是只选“此刻空闲”的核。
Mix 基线选择完整候选 cluster；C 有协调义务，非执行 lane 可能有共享资源义务。
后续缩小成员集合属于独立性能优化，必须在发布前完成，不能在某核跳过 A 后再加人。

worker 扫描已发布日志的本核投影，跳过非成员，停在最早未 LocalDone 的成员义务。
已知 B 可以先于尚未 ready 的 A 提交；B 发布后不因 A 后来 ready 而重新排序。
允许 B 在 A 整体结束前发布和执行，但相交成员必须先关闭自己的 A。

### 普通 slot 与 direct refill 不能有隐藏旁路

目标要求普通、远端 Mix、direct refill 三条 kernel 入口服从同一前沿。
启用共同准入的纯 C/V 图，其 own/direct 已统一调用 `scheduler_ordered_can_enter_slot`；
GM Mix 接线时还须消除“普通槽先执行、无普通槽才看 Mix”的固定优先级。
已经预填的后序候选不代表已经获得执行许可。

普通单 block 的基线线性化点是**执行前对 commit_tail 的关键原子读**：
本核先处理到可见前缀，再以 atomic-current read 确认 tail 未推进，且没有未关闭义务。
该次读授予仅本次 kernel 的 entry permit；之后提交的协同 task 可以包含该核，
但必须容忍这一个已获准的有限普通 kernel 收尾。读前已经提交的义务不可忽略。
每次续接都重新验证；两个预填槽不能共用一个过期 permit。

若检查发现较早义务：保存后序 candidate 的 task_ref/所有权，释放可重建 payload 的临时占用，
让最早义务使用执行资源。禁止重复入队、覆盖候选或用伪完成“腾槽”。
短提交锁和尚未发布的组责任都不得跨入一个可能阻塞的后序 kernel。

当前候选池与两个 payload 槽分开：普通候选实际进入 kernel 后才释放候选额度，
较早协同义务到来时，只停放尚未进入 kernel 的 READY 普通槽，缓存已求值 predicate，
释放 payload 后重建，不重复求值、不重复入队。direct 候选池满时沿既有队列首次入队。
来源与发布方式分开：保留到候选池后再填槽的后继仍为 DIRECT_RESOLVE 来源，
实际 publication_mode 为 DISPATCH；只有在 completion 内直接填槽才是 REFILL。
两个普通 direct 链可能持续占满候选池，使队列中的协同任务推迟到某条有限链结束才被发现；
有限合法 DAG 的推进不等于固定轮数的发现公平性，当前没有这样的公平上界保证。

## 三、block 领取、GM Mix 与局部关闭

### 纯 C/V 的不可撤销领取

当前纯 C/V 使用 `scheduler_claim_ordered_block`：
先确认本核最早义务、类型与候选容量，再用有界 CAS 将 next_block 从 b 改为 b+1。
b 等于 N 返回 EXHAUSTED，不继续 fetch_add 造成无限越界；成功即绑定唯一 owner，
不回滚 cursor、不偷取已领取工作。参数准备失败必须失败收敛，不能重新发出同一 block。

LocalContext 始终设置原 N 与本次 b；参数 view 不自动再加 block 偏移。
普通模式本核最多两个已领取未退役候选，实际 kernel 同时至多一个。
进入下一任务的条件是 cursor 已封闭且本核已领取 work 全部发布、执行、完成。
快核可以在远端最后一个 block 仍运行时满足该条件；零工作成员也可以合法关闭。

后续 sync_start 可复用动态领取，但本核对该 task 至多成功一次，**早返回也不能再领一次**。
未领取成员只在 cursor 耗尽后关闭；已领取者完成自己义务后可关闭。
N 不超过冻结成员容量，加上不可跳过前序义务、公平控制推进，保证剩余 block 最终有不同位置。
不加全员 ARRIVED 屏障，也不等全部核空闲才让首个 block 开始。

### GM Mix 状态机

基线每 cluster 同时只保留一组，先不做两组交错流水。
AIC 只负责取 block、选择物理 placement、发布身份及聚合完成；各执行 lane 自备自己的 payload。

~~~text
EMPTY → CLAIMED → DESCRIPTOR_PUBLISHED → LOCAL_RESOURCES_READY
                                            ↓
                         各执行 lane 自备参数并调用 kernel
                                            ↓
                     返回／输出可见／通信收尾／逐 lane 释放
                                            ↓
                     本组完成，计入 task，才可复用 mailbox
~~~

1. AIC 领取前检查其最早义务；领取后立即负有发布责任，不能再进入后序阻塞 kernel。
2. descriptor 是独立只读发布区；包含完整组身份与 slot→物理 lane 映射。
   写回 descriptor 后发布 group_seq，各 lane 观察序号后失效并读取 descriptor。
3. 每 lane 用自己的原子 ack 区发布 arrived_seq、released_seq，比较必须带当前 group_seq。
   不用重复清零的裸 done bit 区分代次；迟到旧 ack 不能满足新组。
4. 需要本地共享资源的 lane 全部交出旧义务并到达后，才调用该组 kernel；
   这是局部资源检查，不是全 task 或全芯片到齐。
5. inactive lane 只履行资源义务；mask 6 的 AIC 只协调，不增加 kernel 或 subtask 完成数。
6. 各 lane 发布 released_seq 前完成输出、通信、trace 收尾，之后不再读取旧 descriptor/payload。
   AIC 观察全部必要 release 后，唯一地完成该 block，再允许写下一组 descriptor。

mask 3/5 的单 AIV 可沿用合法物理 placement；逻辑 slot 不能因此改号。
mask 6/7 的两 AIV 必须按正确物理 lane 配对，GlobalContext.sub_block_id 从 placement 得到。
调度控制不使用 SSBUF；kernel 内既有 TPipe/L1/硬件事件另按原合同收尾。

### 为什么 cursor=N 不能让 AIV 直接离开

AIC 可能已领走最后一个 b，却停在 descriptor 发布之前。
因此 AIV 关闭 A 必须观察本 cluster 的 `MixCloseSeal[A, cluster]`，不能只看 cursor。
AIC 仅在“不会再领 A”且本 cluster 的已领组全部退役后发布 seal；
内容为该 task 最后发布的 group_seq 加一，零工作 cluster 用 1，0 表示尚未关闭。
AIV 核对自己已服务该 task 的最后序号并结束资源义务，才 LocalDone。

seal 按 task 留存到 run 结束，后续 B 可以复用 mailbox 而不覆盖 A 的关闭证据。
这不等待其他 cluster，也不要求 TaskDone(A)；快 cluster 可继续执行独立 B。
cursor、组发布、组完成、cluster seal、LocalDone 是五件不同的事，不能共用一个标志。

## 四、完成、发布和失败收敛

### 两级完成，唯一释放后继

纯任务的本核退役相当于 block 完成；Mix 由本组 AIC 聚合 active lanes 和资源释放后完成。
纯 C/V 的 `scheduler_finish_ordered_block` 对 completed_blocks 加一；
只有读到旧值 N−1 的完成者进入现有逻辑完成路径，
关闭 wake-list、推进后继、更新逻辑 task 计数一次。旧值大于等于 N 属于错误。
重复 lane/slot 完成先通过 epoch、slot/group generation 和本地消费状态拦截，
不能寄希望于“总数最终等于 N”。

TaskDone 负责依赖，LocalDone 负责成员顺序；不能互相代用。
服务节点沿用一次性完成协议，Host inline 已完成节点不能在设备再完成一次。
predicate 跳过不伪造执行身份；不支持的多 block 组合仍由能力检查处理。

纯 C/V 的 trace 索引包含 `block_idx * popcount(mask)`；Host 导出使用该完整索引作为
`reg_task_id`，保留逻辑 task_ref 作为 task token。转换器把 reg_task_id 纳入执行身份，
避免同一核连续执行同一 task 的不同 block 时合并记录或连错 flow。
这只补齐逐 block 存储／连接键，不提供尚未采集的 order、LocalDone 和 GM 组事件。

### 每条发布链必须有设备侧证明

| 状态 | 写方动作 | 读方动作 |
| ---- | -------- | -------- |
| AdmissionRecord/log element | 写只读区，按涉及 cache line 写回并 barrier，最后原子发布 tail | query 可作预筛；关键读确认 tail，失效记录，再使用 |
| TaskControl 新字段 | 始终使用现有原子包装 | 控制转移用关键原子操作，不用过时快照 |
| Mix descriptor | C 写回完整 descriptor，barrier，原子发布 seq | lane 原子确认 seq、失效 descriptor、barrier、核对身份 |
| lane ack／cluster seal | 完成对应先行动作，再原子发布带代次值 | 关键读核对 epoch/seq，不把旧值当新值 |
| 输出 → block 完成 → TaskDone | 按 kernel 及运行时合同完成输出发布，再递交各层完成 | 消费者遵守已有 payload/输出可见性边界，不仅检查总数 |

现有 scheduler_gm_query 是 ld_dev 预筛，不是所有权或 payload acquire。
AICore 上包装函数的 C++ memory_order 参数本身不建立设备 fence；
复用 scheduler_memory.h 中明确的原子、写回、失效、barrier，逐字段审计。
多写者原子 line 不能被普通 dcci 写回旧副本。CPU 原子模型不证明这些硬件性质。

### 失败和退出

错误记录首先保存 epoch/task/block/worker/seq/site，停止新 kernel 准入，不补成功计数。
已返回工作允许必要清理；尚在 kernel 内的核必须按已有设备错误与停机流程安全退出。
队列空、日志暂空、cursor 耗尽、TaskDone 数达标中的任何一个单独成立，都不代表可回收。

正常退出需图终态完整、所有局部义务和控制事务收尾、trace 完整；
异常重跑需确认旧访问静止后再清零并更换 epoch。保持 run 级存储直到生命周期握手完成。
超过 counter/seq 上限、缺成员、trace 截断都明确失败，不静默恢复成较弱验收。

## 五、可分步提交的编码清单

下表短路径均相对 `src/a5/runtime/host_build_graph/`；每步只开放已经通过门槛的能力。

| 顺序 | 修改位置与具体工作 | 通过门槛 | 当前范围 |
| ---- | ------------------ | -------- | -------- |
| 前置 | 公共 payload/context 初始化及 bootstrap | 六种 mask 的 N=1 重复运行，不通过 kernel 伪造 context | 已验证；详见实现过程 |
| 1 | scheduler_types.h/layout.h 与 Host 实际大小预分配 | 布局／初始化、二次运行、溢出及边界拒绝 | 已实现 |
| 2 | ready 唯一领取、准入发布及 executor/completion 统一 entry gate | B 先于未 ready A；三入口不可越序；旁路发布竞态 | own/direct 已接线；GM Mix 入口待实现 |
| 3 | 普通 C/V cursor、原 N、block 退役与分波 demand | N=1、容量两侧、512；晚 block 不早放后继；快核跨 task 重叠 | 原生纯 C/V 已有通过证据 |
| 4 | GM descriptor/ack/seal，各 lane 自备 payload，无 SSBUF | mask 3/5/6/7、配对、暂停最后发布者、局部关闭和槽复用 | 未实现 |
| 5 | block/order/placement/来源导出及严格审计器 | 完整因果链、不重不漏、无截断；原样 CSA 七项 golden 与 resident 来源 | 逐 block 索引／连接键已接入；完整协议与 CSA 待验收 |
| 6 | sync_start 一次领取／不同位置容量 | 满核独立任务、交叠成员、超容量、早返回不二次领取 | 未开放 |
| 7 | 相同 kernel/输入/拓扑的控制开销对比 | 区分无完整插桩性能与诊断运行，不能由 CPU 模型推算收益 | 未完成 |

纯 C/V 已有生产 helper 和原生执行证据；GM Mix 的实际可见性、三入口协同、
完整 native trace adapter 仍需随生产实现补齐，参考模型不是这些实现的替代品。
不要先全局放开 shape predicate，再用 legacy 的正确输出宣布 SPMD 完成。
