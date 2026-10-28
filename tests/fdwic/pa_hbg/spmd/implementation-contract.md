# A5 HBG SPMD 编码契约与接线设计

基线：`f5cc679bbf3d`，2026-10-23。[设计总览](../a5-hbg-spmd-design.md)
定义已确认语义；本页把它收敛为可开始编码的正确性基线。
共同准入、Host 预分配、纯 C/V 的 own/direct 入口、cursor 和 LocalDone 已接入生产路径。
GM Mix 也已接入，原样 CSA 的数值与严格 resident L4 门禁已通过；
首错误身份、退出冷日志及显式 block/slot 身份已接线，原样 CSA 身份审计通过；
L4 原生协议采集、因果审计和原样 CSA 严格门禁已通过；N>1 的 sync_start 已接线，
仿真／A5 各 47 项及原生审计通过；N=1 公共请求已确认保留归一化，详见下文两层合同。
当前先做边界验收，性能优化按用户要求留待后续，不以性能完成代替正确性验收。
初次同卡无完整插桩对照中，固定 CSA 的 Device 中位数约为 AICPU 设计基线的 11.41 倍。
后续 GM service 按本核阶段跳过无进展调用，相对同批 resident 对照降低 Device 中位数 6.58%，
关键错误检查保留；整体回退尚未消除，不能将功能通过写成已经达到设计基线性能。
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
| 现有 TaskControl 的原子区 | next_block、claim_tickets、completed_blocks、admission_claim_lock | 领取者／完成者；只能原子访问 |
| TaskControl 的阶段信号区 | admission_state，独占 128B | 唯一准入者发布；禁止 AICore RMW 或 cached writeback |
| AdmissionRecord | epoch、task_ref、order_key、成员位图、形状／策略、Mix seal offset | 唯一准入者准备；发布后只读 |
| AdmissionLog | 长度为可协同准入 task 数的 task_ref 数组；commit_tail、短提交锁 | 任意合法发布者；一次提交一个引用 |
| 本核本地状态 | scan_cursor、current_obligation、两个候选身份、当前 lease、sync_claimed | 仅本核修改，不增每核全图表 |
| 每 cluster 一个 GM GroupMailbox | 当前组 descriptor、published_seq、3 个 lane ack | AIC 写组身份，各 lane 写自己的 ack |
| MixCloseSeal | 每 Mix task、每候选 cluster 一个 64 位结束 token | 本 cluster AIC 单次发布，直到本轮退出不复用 |

TaskControl 为 512B：state/wake_list_head/next_waiter 仍在第一条 64B line，admission_claim_lock 的 offset 为 40；
offset 24 的 claim_tickets 复用原 padding，offset 32–39 为零 padding；第二条普通写回 line 仍从 offset 64 开始。
next_block、completed_blocks 分别独占 offset 128、256 起的 128B 区域，均只能原子访问。
admission_state 独占 offset384 起的 128B 区域，由 st_dev＋DSB 发布、ld_dev 读取，
整条信号 line 不与原子字、cached store 或 dirty writeback 混用。
保持 sizeof、offsetof、alignment、POD 静态断言，Host 初始化／上传与所有访问步长从 sizeof 推导。
AdmissionRecord 的位图按紧凑 worker index 编码，按实际 worker 数规划 word 数，
不用物理 core ID 直接寻址。独立热点控制对象以 128B 对齐／步长隔离，
这只是地址布局基线，不声称能够消除同一 cursor 的原子竞争。

当前布局为 64B record 头加实际 bitmap word 数，整条 record stride 向 128B 对齐，
避免不同准备者对相邻 task 位图整行写回时相互覆盖。
TaskMetadata 使用 int16 timing_slot 与 uint16 原 N，共享原 4B 尾部，维持 16B；
GraphSummary 的现有 padding 承载协议 offset／容量／Host run_epoch，维持 128B。
AdmissionControl 的 lock/tail 各占 128B；GroupMailbox 为 descriptor 128B、
published_seq 独占 128B、三个 lane ack 各 128B，共 640B。

`SchedulerAdmissionLocalState` 为 256B，含两个各 48B 的候选、40B 组状态、sync_claimed，
以及本轮不变拓扑的 AIC 位图与有效 worker 数缓存；缓存随每轮初始化重置。
`SchedulerLocalSlotState` 为 24B，含 block 身份和 predicate 缓存，
`SchedulerLocalState` 为 504B。候选只保留 task 所有权与重建所需状态，
不会增加物理并发容量，也不保存每核全图或全部 block 历史。

AdmissionLog 只存引用，记录按 task_ref 定位；不按 block 数扩大日志。
紧凑 log 单元在提交锁保护下由唯一发布者用 st_dev＋DSB 写入，发布后不可变，由 ld_dev 读取；
commit_tail 同样使用隔离的 bypass 信号区。close seal 仍使用原子读写。
这些紧凑数组都不能普通写单元再 dcci 整行；短锁不会自动消除不同发布者缓存中的旧行副本。
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
- 公共 HBG `LaunchSpec::set_block_num` 以 int64_t 接收请求，检查 int16_t 范围后再窄化。
  超出范围保存为非法 0，既有 Host submit 以 INVALID_ARGS 拒绝且不发布任务存储。
  字段布局维持 4B／align 2；可表示值原样保存，0 和负值仍由 submit 拒绝。
  旧编排必须重新编译；调用者显式提前转为 int16_t 后的信息丢失不在 setter 内补造。
- 位移、block/subtask 乘积、trace 前缀先用足够宽类型计算，再检查后窄化。
  原子计数和 seq 不回绕；临近上限在新领取／发布前失败。
- 无 kernel 的服务节点独立走既有逻辑完成，不假造一个 kernel block。

### 阶段能力判定

普通多 block 只要求有相应类型的执行位置，不要求 N 个位置同时空闲。
sync_start 的 N 必须不超过本轮冻结成员中不同执行位置数：C 按 AIC，V 按 AIV，
Mix 按完整 cluster；两个 pending slots 不是两个执行位置。
先开放普通 C/V、GM Mix、原样 CSA，再开放 sync_start。
多 block／Mix predicate 保留 Host 拒绝，Graph body 保留整图回退，不顺手删除形状门禁。
异步返回未开放：resident payload 的 task token 无效、completion 三指针为空、capacity=0，
公开 register_completion_condition 返回 false。Host 没有任意异步 kernel 的自动识别／回退机制。
save_expected_notification_counter 不传播注册 bool，也不自动等待；不能用它假定同步降级。
SDMA adapter 的 event.Wait 分支是独立的同步等待合同，不等于本版支持 deferred 返回。
普通单 block predicate 的现有语义必须保留。

当前 `scheduler_resident_task_shape_supported` 允许普通 C/V/Mix 的 N≥1、sync_start 的 N>1，
仍检查 N×active_subtasks≤32767；纯多 block 与 Mix 可以同图，不再设置组合隔离门禁。
所有 Mix 包括 N=1 均走 GM，生产 resident 调度不使用 SSBUF。
`scheduler_resident_worker_demand_checked` 对 sync_start 检查实际完整拓扑和不同位置容量；
Mix 按每 block 一 AIC、两 AIV 的资源需求计数，不把 inactive lane 当作可让出的资源。
准入准备再次核验冻结拓扑与容量。多 block／Mix predicate 仍沿用 Host 明确拒绝。

N=1 的边界分为两层，用户已确认保留现有公共归一化行为：

- 公共 LaunchSpec 接受 N=1、require_sync_start=true，但生成的 TaskAttrs 不设置 SYNC_START。
  请求对象、原 N、kernel 身份、参数和依赖不因此改变；独立的纯 AIV1→AIV0 归一化仍保留。
- Mix 仍是单逻辑 block 的 Mix，继续共同准入、GM 组配对、三条资源 lane 握手和完成聚合。
  不得把“不带多 block 同步属性”解释成“一个执行核”或“取消组内协调”。
- 绕过公共提交而直接制造 N=1 且显式带 SYNC 标志的设备 metadata，仍属于畸形输入并拒绝。
  CPU 模型 Task.sync_start 表示这一存储层标志，不是公共 LaunchSpec 的原始请求。

N>1 的合法请求继续设置 SYNC_START，超过对应类型执行位置容量的请求仍在 Host 拒绝。

## 二、完成驱动准入及普通旁路的线性化

### 准入函数与状态转换

`scheduler_admission.h` 已提供 `scheduler_try_claim_admission`、
`scheduler_prepare_admission_record`、`scheduler_try_commit_admission`、
`scheduler_refresh_local_obligation` 和 `scheduler_can_enter_ordinary` 基础 helper，
参数复用现有 graph/state/local context。调用方先唯一取得 ready 候选，
仅成功 claim 才将其所有权转移给持久 pending；RETRY 不能让调用方丢弃候选。
本地协议状态必须成功初始化；epoch 0 是合法身份，不能拿它兼作“未初始化”哨兵。
唯一领取使用 task 自己的 admission_claim_lock：CAS 0→1 后再次确认阶段为 UNCLAIMED，
发布 CLAIMED 后释放锁，再将候选转入 pending。迟到的重复领取必须因阶段非零而拒绝；
取得锁后遇错须释放自己的锁，没有取得锁则不能释放别人的锁。
此锁不跨准备、kernel 或等待持有，也不是保持到 TaskDone 的全局锁。
PREPARED／ADMITTED 由同一准入 owner 发布；扫描者仍核验阶段、身份、epoch、成员和顺序。
纯 C/V 的 `scheduler_fill_ordered_slots` 发现／发布候选，executor 的
`scheduler_ordered_can_enter_slot` 在实际执行前校验许可；完成路径的 direct 候选
进入同一个持久候选池。GM Mix 使用本核 own slot，入口再核对组、block、slot 和 generation。

满足三 lane release 后，同一次服务可退役当前组并领取／发布下一组；
本地 close 后可继续观察下一义务，每次 fill 最多连续处理两个已关闭义务。
等待伙伴时立即返回，不以连续推进替代错误检查、执行许可或全员 release 确认。
完成服务可能在本轮 discovery 之后产生 direct 候选；其中 inline 服务和多 block／Mix 候选
必须再经过 discovery 的服务／准入分类，不能落入普通单 block 的物理槽准备路径。

Mix扇出有两个已就绪后继且等待链仍有剩余时，Resolve最多提前发布一次Ready前缀，
剩余后继继续按原依赖逐个验证，最后批量入同一coordinator inbox；已发布链不再改写。
前缀之后不重新建立direct Mix旁路，避免后缀越过前缀；纯AIV的按首节点奇偶分流规则不改。
整次Resolve不是事务性提交：发现错误之前合法前缀可能已执行，错误发布后原准入和entry门禁
必须拒绝新工作，不回滚已发布前缀，不继续发布未验证后缀。

| 转换 | 条件与唯一动作 | 暂时不能推进时 |
| ---- | -------------- | -------------- |
| DATA_READY → CLAIMED | 既有 pop/refill 唯一取得逻辑 task；短领取锁内重查阶段并发布 CLAIMED | pending 已占用则 RETRY；重复所有权则 ERROR，不丢候选 |
| CLAIMED → PREPARED | 冻结成员与角色、校验形状、初始化 cursor/计数 | 错误终止本轮；不假造完成 |
| PREPARED → ADMITTED | 短锁内填 order、发布记录与日志元素、最后推进连续 tail | 竞争失败保留候选，先做可推进控制工作 |
| ADMITTED → TASK_DONE | 所有 block 成功完成且可见 | 继续服务，不以领取耗尽代替完成 |

ready pop 只做一次；task 离开 ready 后由准入记录维护 block cursor。
**不采用“长时间 peek 队头，最后一块再 generic pop”的主方案。**
现有 generic pop 接收旧 endpoints 后会重试并可能取走后一个 task，
不是“仅删除我先前看见的 A”。若将来采用延迟摘头，必须另有 expected-task 条件摘除原语；
新增 C++ 回归已固定这一边界，不能误用旧接口。

准备成员不持锁。独立的全局 commit_lock 内只完成有限的记录提交：
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

普通、GM Mix、direct refill 三条 kernel 路径服从同一前沿，
实际入口统一调用 `scheduler_ordered_can_enter_slot`；
Mix 作为当前义务推进，不采用“普通槽先执行、无普通槽才看 Mix”的固定优先级。
已经预填的后序候选不代表已经获得执行许可。

普通单 block 的线性化点是**执行前对 commit_tail 的关键 bypass 读**：
本核先处理到可见前缀，再以 ld_dev 确认 tail 未推进，且没有未关闭义务。
tail 已整体切换到隔离信号发布域：持有提交锁的唯一活动写者用 st_dev＋DSB_DDR 发布，
该 line 不混用 AICore 原子、cached store 或 dirty writeback；不是只替换旧原子 writer 的 reader。
该次读授予仅本次 kernel 的 entry permit；之后提交的协同 task 可以包含该核，
但必须容忍这一个已获准的有限普通 kernel 收尾。读前已经提交的义务不可忽略。
每次续接都重新验证；两个预填槽不能共用一个过期 permit。

普通 payload 的准备判断只在同时存在未准备候选和空槽时执行，
同一次 fill 可以复用该判断准备两槽，但不跨 fill 保存。
准备判断不是 kernel entry permit，实际入口仍遵守上述逐次关键读。

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
先确认本核最早义务、类型与候选容量。N1 和 sync_start 用有界 CAS 将 next_block 从 b 改为 b+1，
b 等于 N 返回 EXHAUSTED。普通 N>1 时用 claim_tickets 的 FAA 返回值作为实际 block 票号，
P 为纯 C／Mix 的 cluster 数或纯 V 的两倍 cluster 数，不按 task ID 或模型名称特判。
有效票 t<N 再对 next_block 加一；这条路径的 next_block 是有效领取确认数，不是当前票号。
每个位置至多一次无效尾票，本地 ticket_exhausted 随当前义务保持并在关闭／新 run 初始化时清除，
故原始票数最多 N+P，而有效领取数仍不超过 N。无效票不产生 block owner、payload 或完成贡献。
成功票号即不可撤销的单块所有权，不回滚、不偷取；参数准备失败必须失败收敛，不能重发该 block。
只持有本次 helper 调用的一张票，不扩大两槽或 Mix 的既有在飞额度。
记录准备及正式准入提交均要求 claim_tickets 为零，不能重新使用带旧票号的 task。

LocalContext 始终设置原 N 与本次 b；参数 view 不自动再加 block 偏移。
普通模式本核最多两个已领取未退役候选，实际 kernel 同时至多一个。
进入下一任务的条件是 cursor 已封闭且本核已领取 work 全部发布、执行、完成。
取号分路也必须强确认 next_block==N；仅看到票号耗尽不能越过尚未发布有效领取确认的最后一张票。
快核可以在远端最后一个 block 仍运行时满足该条件；零工作成员也可以合法关闭。

sync_start 复用动态领取，本核对该 task 至多成功一次，**早返回也不能再领一次**。
成功 CAS 后 sync_claimed 跨 kernel 返回保留，当前义务关闭后清除；发布未来任务不清除此状态。
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

本版 Mix 的逻辑 slot 与物理 lane 固定对应：mask 3→AIV0、mask 5→AIV1，
mask 6/7→各自两条 AIV。CSA 的 NO_SPLIT 搬运使用固定 SingleModeVec0，
不能只修改 payload.sub_block_id 就把执行迁到另一 lane；descriptor 接收方也验证该对应关系。
GlobalContext.sub_block_id 仍由实际 placement 得到。
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
服务节点由 ready 唯一 owner 执行 DONE→既有 resolve_completion→pending_completed，
后继首次入队，不递归准备槽；ordered 每轮受两个候选额度约束。
不增加 slot generation，不生成 payload、kernel trace 或虚假的 direct refill。
Host 的逻辑任务数仍计入服务节点，物理 subtask/trace 数为零；三条 ready transition 路径
均不写它的 trace 前缀，避免覆盖后一个 kernel。Host inline 已完成节点不能在设备再完成一次。
predicate 跳过不伪造执行身份；不支持的多 block 组合仍由能力检查处理。

纯 C/V 的 trace 索引包含 `block_idx * popcount(mask)`；Host 导出使用该完整索引作为
`reg_task_id`，保留逻辑 task_ref 作为 task token。转换器把 reg_task_id 纳入执行身份，
避免同一核连续执行同一 task 的不同 block 时合并记录或连错 flow。
入口身份另以 `aicore_execution_identities` 对象导出，schema 为 `hbg-execution-identity-v1`，
producer 为 aicore，records 用 epoch/core/reg_task_id 与执行行一一连接。
字段包括 task_id、block_idx、block_num、subtask_slot、pending_slot、dispatch_generation、
order_key、group_seq、cluster_index、cluster_lane、executed。未参与共同准入或 GM 组时相应字段为 null；
predicate 跳过保留 executed=false，不冒充 kernel 执行。N/block 来自调用前的 payload 快照，
不从行序、trace 索引或完成数推导。Trace 保持 384B，最后一条 cache line 发布身份后才调用 kernel，
completion 的既有 valid 最后发布；L0 不写身份。读端拒绝缺字段、错绑、重复执行及槽代次。
CSA 严格门禁按冻结图核对完整 task/block/slot 集合；历史无身份 capture 保持未验证。
执行身份对象本身不提供 LocalDone、共同准入发布和 GM 组释放事件。
L4 另以 `aicore_protocol` 对象导出这些真实记录，当前 schema 为 `hbg-native-protocol-v5`，
同时保留 Host 冻结图和设备终态。统一读端在 `simpler_setup/tools/hbg/native_protocol.py`；
它校验 schema 与逐核序号完整性，`native_causality.py`、`native_mix_audit.py` 与
`native_order.py` 进一步检查冻结图／成员、两槽生命周期、实际完成票号、组握手与关闭、
TaskDone 和依赖，并与独立入口身份逐字段连接。
v2–v5 要求每个 task control 的 claim_tickets 字段，终态取号任务必须在 [N,N+P]，其他任务必须为零；
v2 的普通任务阈值严格为 N>2P，v3 为 N≥2P，v4/v5 为 N>1，N1／sync_start 均不取号。
v5 另要求 admission_claim_lock 原始字段，成功终态必须为零；失败 wire 记录保留实际值用于诊断。
next_block==completed_blocks==N 的有效 block 条件保持原样。读端继续严格支持 v1–v4 历史记录，
不为缺失字段补默认值，也不接受 v1 混入票数字段或在同一 schema 下改变阈值语义。
失败记录保留原始票数，不能仅靠 wire 解析宣称终态正确。
既有泳道 `core_id` 使用紧凑 worker 编号；协议拓扑 `workers[].core_id` 保存硬件核号，
二者不能混作连接键。

审计按本核 sequence 建立顺序链，发布线性化点限制在实际 before／after 之间，
跨核边来自原子票号、已读 tail／cursor／ack／seal 和冻结依赖，最后检查约束图有无矛盾环。
sync_start 另核验每位置至多一次；提前关闭必须属于已领取并退役的位置。
关闭时的 cursor 关键读位于本核前一事件与 closed 之间，实际值约束前后领取线性化点，
不要求远端最后一块先被领取，也不把记录时刻冒充读发生的时刻。
不按跨核 cycle 全局排序；普通 permit 的关键读可以早于新的共同准入，
消费者观察也可以早于发布者记录 after。虚拟线性化点不是新增设备事件，不占用 sequence。
`protocol_verified=true` 仅在完整审计通过时产生；历史无协议对象仍可读，但不能通过严格门禁。
CSA 还对照独立 dep-gen 核验冻结任务、kernel、原 N 和全部 fanin；
原生路径的依赖证明为该因果检查；CSA 原有 kernel 依赖时序检查作为额外保守检查继续保留，
不能用时间戳检查代替原生证明，也不能在有原生证明时静默跳过旧检查。

### 每条发布链必须有设备侧证明

| 状态 | 写方动作 | 读方动作 |
| ---- | -------- | -------- |
| AdmissionRecord/log element/tail | 写回完整 record 并 barrier；st_dev＋DSB 发布 log cell；原子置 ADMITTED 并 barrier；最后 st_dev＋DSB 发布 tail | ld_dev 关键读取 tail/log，失效 record/bitmap，原子确认 ADMITTED 并核验完整记录，再使用 |
| TaskControl 新字段 | 始终使用现有原子包装 | 控制转移用关键原子操作，不用过时快照 |
| Mix descriptor | C 写回完整 descriptor，barrier，原子发布 seq | lane 原子确认 seq、失效 descriptor、barrier、核对身份 |
| lane ack／cluster seal | 完成对应先行动作，再原子发布带代次值 | 关键读核对 epoch/seq，不把旧值当新值 |
| 输出 → block 完成 → TaskDone | 按 kernel 及运行时合同完成输出发布，再递交各层完成 | 消费者遵守已有 payload/输出可见性边界，不仅检查总数 |

对原子发布的协议字，scheduler_gm_query 是 ld_dev 预筛，不是所有权或 payload acquire。
AICore 上包装函数的 C++ memory_order 参数本身不建立设备 fence；
复用 scheduler_memory.h 中明确的原子、写回、失效、barrier，逐字段审计。
多写者原子 line 不能被普通 dcci 写回旧副本。CPU 原子模型不证明这些硬件性质。

准入 tail 独占 128B；commit_lock 保持 CAS／原子释放，轮换写者必须完成发布后才能交接。
紧凑 log 中每个 cell 每 run 只写一次，整片 log 统一使用 bypass 发布／读取，不能让相邻 cell
混入原子更新或 cached store／dirty writeback。Host 初始化在启动前完成，导出发生在静止之后。
error status 使用相同的受限 64 位发布 helper，但身份字段与 ADMITTED 等其他原子字不因此改协议。
29 个原生发布场景分别在真机／仿真验证轮换发布者、首次读取、相邻追加和满核共享前缀，
再由原样 CSA 的严格 L4 核验实际入口、身份与依赖；不能仅以 CPU acquire/release 推断硬件可见性。

### 失败和退出

错误记录首先保存 epoch/task/block/worker/seq/site，停止新 kernel 准入，不补成功计数。
RunControl 为 512B，error_claimed/site 的既有 offset 为 256/328；
offset 336–383 的 48B 承载 epoch/order/block/group_seq/worker 五个 64 位字段、lane/identity_fields 两个 32 位字段。
错误状态位于独占 128B 区域的 offset 384，旧 offset 264 为零填充，不与原子身份字段共 cache line。
identity_fields 的 bit 0–5 分别表示 epoch/order/block/group_seq/worker/lane 已采集，
不表示数值已通过协议合法性检查。未采集值为全 1；合法 epoch 0/MAX 依靠位图区分未知。

通用 writer 位于 scheduler_error.h，取得 first-error CAS 后写全身份，barrier 后发布错误状态；
状态字的 AICore 写入使用 st_dev，完成 DSB_DDR 后返回；状态读取统一走 scheduler_read_error 的 ld_dev。
这一单写者状态 line 不允许任何 AICore atomic，包括 fetch_add(0)，也不允许 cached store／旧行写回。
AICPU 首写者先写回完整身份区，再 release store 状态并写回独立状态区；CPU／仿真 reader 使用 acquire load。
bootstrap、execution、terminal 与冷日志的缓存失效范围均覆盖新状态位置，不能仍停在旧 384B 边界。
状态改走同一非原子发布／读取路径不改变 claim、发布、执行入口、完成、关闭或退出处的错误门禁。
其他 cursor／seq／ack／seal 继续使用原来的原子协议，不能套用此读取规则。
OK 不占 claim，后写者不修改首现场。ready 从已有本核状态提取身份，
不匹配当前义务时不挪用它的 order/group；当前 task 已退役组可保留最近身份，
task_last_seq 清零后不再拿前 task 的组解释新 task。准备和 completion 使用确切 slot，
不在两个预填槽之间猜测；无共同准入上下文时 epoch/order 保持未知。
AICPU 首写者明确记录未知 task/core/扩展身份，bootstrap/execution/terminal 的统一冷日志
在 acquire 发布状态后重新失效完整错误区，覆盖跨两条 64B line 的身份读取。

已返回工作允许必要清理；尚在 kernel 内的核必须按已有设备错误与停机流程安全退出。
队列空、日志暂空、cursor 耗尽、TaskDone 数达标中的任何一个单独成立，都不代表可回收。

公开 Worker 当前对任何 finalize 失败保守毒化 run lane，并保留 caller buffer 借用；
不能把有界 kernel 最终返回解释成同 Worker 可以再次提交或提前 free。
有界 timeout 验收分别检查真实 kernel returned／visible、所属 AICPU 分区退出时序、
未释放依赖、失败收尾、拒绝重提／free，以及 close 后新 Worker 的正常运行。
失败 capture 不满足成功协议的全图终态，不能要求 protocol_verified=true 来证明故障；
workers.finished=2 只表示 ACK 前的失败记录，必须结合 fence、退出与关闭证据。

正常退出需图终态完整、所有局部义务和控制事务收尾、trace 完整；
异常重跑需确认旧访问静止后再清零并更换 epoch。保持 run 级存储直到生命周期握手完成。
超过 counter/seq 上限、缺成员、trace 截断都明确失败，不静默恢复成较弱验收。

## 五、可分步提交的编码清单

下表短路径均相对 `src/a5/runtime/host_build_graph/`；每步只开放已经通过门槛的能力。

| 顺序 | 修改位置与具体工作 | 通过门槛 | 当前范围 |
| ---- | ------------------ | -------- | -------- |
| 前置 | 公共 payload/context 初始化及 bootstrap | 六种 mask 的 N=1 重复运行，不通过 kernel 伪造 context | 已验证；详见实现过程 |
| 1 | scheduler_types.h/layout.h 与 Host 实际大小预分配 | 布局／初始化、二次运行、溢出及边界拒绝 | 已实现 |
| 2 | ready 唯一领取、准入发布及 executor/completion 统一 entry gate | B 先于未 ready A；三入口不可越序；旁路发布竞态 | own/direct 与 GM Mix 已接线，原生协议审计通过 |
| 3 | 普通 C/V cursor、原 N、block 退役与分波 demand | N=1、容量两侧、512；晚 block 不早放后继；快核跨 task 重叠 | 原生纯 C/V 已有通过证据 |
| 4 | GM descriptor/ack/seal，各 lane 自备 payload，无 SSBUF | mask 3/5/6/7、配对、暂停最后发布者、局部关闭和槽复用 | C++、仿真／A5 矩阵和重叠通过；首错误身份与冷日志已接线 |
| 5 | block/order/placement/来源导出及严格审计器 | 完整因果链、不重不漏、无截断；原样 CSA 七项 golden 与 resident 来源 | CSA 七项 golden／2308 身份／来源、原生因果审计和独立图一致性门禁通过 |
| 6 | sync_start 一次领取／不同位置容量 | 满核独立任务、交叠成员、超容量、早返回不二次领取 | N>1 的 C++／Python、仿真／A5 与原生审计通过；N=1 保留公共归一化，验收见实现过程 |
| 7 | 相同 kernel/输入/拓扑的控制开销对比 | 区分无完整插桩性能与诊断运行，不能由 CPU 模型推算收益 | 初次同卡对照为设计基线 11.41 倍；GM service 阶段裁剪相对同批 resident 对照降低 6.58%，整体回退未消除 |

纯 C/V、GM Mix 已有生产 helper 和原生执行证据；
N>1 sync_start 与性能对照已有实测证据，N=1 公共契约已确认，最终逐项验收仍待收口；
参考模型不是实现的替代品，性能回退也未因完成测量而得到解决。
私有错误区原生探针验证生产 writer 和状态发布／读取，不触发真实 runtime 错误，
不能据此声称任意物理核异常的静止／回收均已获得设备验证。
不要先全局放开 shape predicate，再用 legacy 的正确输出宣布 SPMD 完成。
