# A5 Graph task 调度与发布协议

本文是[设计总览](../a5-hbg-graph-task.md)的设备侧实施契约。
节点编号和 GM 区域见[数据布局](data-layout.md)。
当前代码恢复到全任务 cooperative admission。曾划分独立 block、MIX lease 和同步 cohort 的
[重构原型](block-ready-scheduling.md)已经撤回，只保留为后续待选方案。
两个固定业务 ready 队列已实现，完整硬件、故障及性能验收仍在进行中；下文历史测量不作为本次验收结果。

## 两个业务 ready 队列 {#two-business-ready-queues}

普通任务与 native Graph child 使用相同分类：纯 AIC 和所有 MIX 进入 AIC 队列，纯 AIV 进入 AIV 队列。
包含双 AIV、没有 AIC kernel 的 MIX 仍由 AIC coordinator 领取；原 flags、active/resource mask、SPMD block 数和执行许可不变。
任意核类型都可向任一业务队列发布后继。每个逻辑任务在一个 run 内最多占一个 entry，block/lane 不分别入队。

| 结构 | 当前 wire 合同 |
| ---- | -------------- |
| `SchedulerReadyPoolHeader` | version 11，64 B；寻址两个队列及依赖区，校验 allocation 边界 |
| `SchedulerReadyPoolDescriptor` | 每份 16 B，固定两份；容量按逻辑 executable metadata 精确计数，允许为零 |
| `SchedulerTicketQueue` | 每份 128 B，固定两份；head 与 reserved_tail 分别独占 64 B 行 |
| `SchedulerTicketEntry` | 每项 64 B；原子发布 task_ref，初始 INVALID，run 内不覆盖或回收 |

Host writer 与 device reader 分别见 [ready_pool_plan.h](../../../../src/a5/runtime/host_build_graph/host/ready_pool_plan.h)、
[scheduler_ready_pool.h](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ready_pool.h)；
entry 交权见 [scheduler_ticket_queue.h](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_ticket_queue.h)。
head、tail 与 entry 发布字所在行不接受 cached 整行写回。数据发布先于原子交权，DSB 不替代跨核所有权协议。

同一批最多发布 8 个同类 ref；预留后在有限路径内完成发布，不带着共享空洞去执行 kernel 或等待 MIX 资源。
消费者只检查自身 AIC/AIV 队列，以 `ld_dev` 作可过期提示，再原子确认 entry、校验 ref 和 ready 类别，成功 CAS head 后取得唯一 claim。
零容量直接返回空；前序预留尚未发布时不越过空洞，下轮重新探测。空提示只对本轮 discovery 有效，不证明图已完成。
领取后的 SPMD/MIX 继续进入 admission/GM MIX，不能在共享队头等待 AIV 空闲；普通 direct refill 显式排除 MIX。
Graph control inbox 不属于这两个业务队列，仍使用独立的 `CONTROL` publication 标签和控制节点协议。

## Bootstrap 与控制节点

Host 初始化完整 N-node control 表，只有确认已在 Host 完成的原始节点设为 HOST_DONE。
其余节点初态沿用 `SchedulerTaskState::BLOCKED`；ready、私有 claim 是所有权位置，不新增 GM task state。
Graph body 的 DUMMY 也必须实际经过依赖路由，不能复制 recording 阶段的 early-resolve 状态。

native header 的第一次读取必须在本 run 的缓存刷新之后。native 路径在取得 READY 后的 context
并检查基础地址/长度后、local config/admission 初始化前，先执行一次 bootstrap cache observe，
再读取 header/template/binding；后面的同类 observe 不重复执行。无 Graph 路径保留原时序。
这仅是每 run 的启动操作，不允许扩展为每 child 的整 cache 刷新。
同地址连续 run 使用不同 Definition、boundary 和 scalar 的回归必须能发现旧 epoch 数据残留。

所有 worker 按现有区间划分扫描 N 个节点：先读 kind，再决定跳过 HOST_DONE、路由控制节点或路由 kernel。
Host 为有边的图预建 fanout CSR 与 remaining；bootstrap 读取初始 remaining，零值节点 ready，非零节点等待通知。
控制节点进入 control inbox，kernel 按 AIC（含 MIX）/AIV 分类进入 ready pool。跨 Graph 的 producer 已由统一访问器映射。
没有依赖 header 的兼容路径才扫描 fanin，并把节点注册到第一个未完成 producer；详见[依赖计数](dependency-counters.md)。
**全部扫描结束前，不 pop control inbox，也不完成任何 ENTER/DUMMY。**
`bootstrap_scan_complete` 后才可以处理控制节点和准备首批 slot；AICPU 仍等待现有
`bootstrap_complete` 后通过唯一 DMB gate 放行执行。

此屏障同时保证 remaining 初始读取不与完成通知竞争，并保留兼容路径
`scheduler_bootstrap_route_task` 的 exchange-before-link 前提。兼容路径稳态使用带 CLOSED 检查的 CAS 路由，
不能把 bootstrap 的无并发完成假设带到稳态。

### 独立 control inbox

现有 `scheduler_complete_inline_task` 由 slot fill 路径调用，而且验证了 kernel 风格的 metadata。
原样复用会让 ENTER/EXIT 等待空闲 payload slot。控制节点使用独立入口：

```cpp
bool scheduler_control_enqueue(/* base, local, node_ref */);
bool scheduler_service_controls(/* graph, base, local, run, budgets */);
```

每 worker 一个 control inbox，复用 `SchedulerReadyInbox` 的 64-bit head/tail 原子协议和
`next_waiter` 链，不使用 AIC/AIV/MIX 的资源过滤、type rank 或 directory mask。
控制 queue push/pop 接收 inbox 地址，保持原有 CAS、尾链发布和有限重试语义。
运行中 ready 的控制节点先推到 resolver 所在 worker 的 inbox；bootstrap 由各扫描分区的 owner 发布。
所有 worker 都可消费控制节点，先查本地、再按轮转游标探查一个 peer，最终覆盖所有 worker。
业务 ready pool 的空提示不能屏蔽这次控制队列探查。

一个节点经 remaining 归零或动态 wake-list 路由成为 ready 后，先按 kind 分流：控制节点只进 control inbox；kernel 才能 direct refill、
进入对应业务队列或 admission。禁止把 ENTER/EXIT 当作 AIV kernel，禁止递归调用其后继完成函数。
MPMC pop 的成功 CAS 是唯一 owner 的获取点，不再增加 Graph 级 owner lock。

### 控制完成切片

kernel 返回收尾与 EXIT/错误门禁之后直接处理当前 slot completion，再运行控制切片、ordered obligations、slot fill 和下一次 kernel。
即使所有 payload slot 占满或 admission candidate 被暂存，也必须运行控制切片。
已有 kernel completion 的控制后继只入队；下一切片处理，避免 completion→control 的递归。

每轮最多处理8个控制节点，作为内部常量。单个节点的completion同步完成：

1. pop获得唯一节点owner，校验kind、BLOCKED和无kernel资源，以原子操作发布DONE。
2. 动态wake-list原子关闭head并取得链；CSR候选则直接读取后继行，不领取wake-list。
3. 依次处理后继：CSR减remaining，动态挂链保存旧next后扫描剩余fanin、就绪或转挂到未完成前驱。
4. 挂链成功后不得再访问waiter；Ready批次全部发布后才能累计该producer的pending_completed。

没有core-local resolver续行。Ready业务ticket在同次调用中完成预留与发布；control inbox成功CAS后立即完成旧tail挂链与屏障。
普通completion保留direct refill；ordered admission存在时也可直接持有MIX后继的准入责任。
async FINISH在join满足后获得唯一lease，同步执行TaskDone与后继传播，再终结lease与清理directory；FINISH不保存192B进度区。异步条件CHECK的预算扫描保留。
本地状态避免新增过度对齐的大栈对象；GM wire 的对齐要求不直接搬到 CCEC 局部变量上。

## 参数绑定与资源准入

Host 只规划参数地址；AICore 使用 Definition record 和本次实例 boundary 生成 kernel ABI 参数。
绑定过程不执行 kernel、不等待 producer 输出值、不发现依赖。它必须是有限的纯重定位/复制操作。
bootstrap 按既有 node 分区并行绑定全部 CHILD_KERNEL，在 scan barrier 打开前完成；
未取得 ready 的节点也可以准备描述符，但仍不能读取 predicate 值或执行 kernel。
例如 PARAM Tensor 继承本次输入的地址和 origin，SUB_TASK Tensor 绑定本实例 heap；
inherited scalar 取本次 boundary scalar，固定 scalar 取只读 Definition。

| 路径 | 唯一参数写者 | 绑定位置 | 消费者 |
| ---- | ------------ | -------- | ------ |
| 单 lane、单 block | bootstrap node 分区 owner | scan arrival 前 | ready owner 首次观察参数后派发 |
| SPMD、MIX、sync_start | bootstrap node 分区 owner | scan arrival 前 | 固定 membership 中需要尝试领取或执行 block 的 worker |
| predicated 单 lane、单 block | bootstrap node 分区 owner | 只提前绑定地址；取值仍在 ready 后 | 跳过路径或本 worker |
| ENTER/EXIT/DUMMY | 无 | 无参数绑定 | 无 |

每个 bootstrap 分区互不相交，child 参数的唯一写者由 node_ref 确定。每次绑定执行完整边界检查、
精确范围 writeback 和 DDR barrier；这些操作完成后才增加 scan arrival，最后到达者打开 scan gate。
现有所有 ready 消费路径均在该 gate 后进入，因此无须逐 child 增加 ARG_PREPARING 锁或发布位。
绑定失败锁存错误，该分区不报告成功到达；其他 worker 在等待 gate 时沿现有 error/watchdog 退出。

普通 ready owner 可能不是参数写者，首次准备 payload 前必须 invalidate 参数范围并执行 barrier。
私有 pending candidate 用 `args_observed` 保存已观察状态，随 task_ref park/restore，最终释放或 run 失败才清除。
cooperative 的 claim owner 只准备 admission record；成员首次 adopt 后、尝试领取 block 前观察参数，后续 block 复用。
普通非 MIX、非 sync_start 的小 grid 使用从 `member_rank_offset % eligible_count` 开始、长度为 block_num 的循环成员区间。
Host 按 task_ref 顺序分别累计 AIC/AIV 普通 SPMD 的 block 数；MIX、sync_start、单 block 和另一类型任务不推进该前缀；
其余 ordinary、MIX 和 sync_start 保留完整可用成员集合。冻结成员位图只决定参与资格，不直接分配 block。
普通纯 SPMD 的迟到成员若不持有该任务的执行 slot，且通过原有强 atomic 路径确认 claim seal 已覆盖全部 block，
可以记录 observed/closed 并直接推进本地 obligation，无须读取参数或暂存普通 payload。
该路径仍校验 epoch、全局顺序、任务形状和 membership，每轮最多扫描八条记录；
扫描游标推进也计为调度进展，即使本轮未保留 current task，也不会触发空闲退避。
不增加 completed_subtasks、不标记逻辑任务完成，MIX 和 sync_start 仍走原有准入与关闭路径。
全局ADMITTED发布、membership消费、短commit lock和kernel资源准入均不提前；Ready到pending只转移本核私有所有权。

`SchedulerLocalSlotState` 用三态 predicate（未求值、通过、不通过）保存一次求值的结果，
park/restore 与 `args_observed` 一起迁移，不因重试而读取更新后的 predicate operand。
`SchedulerWorkerContext` 为 320 bytes，`SchedulerLocalConfig` 为 104 bytes，
slot 保持 24 bytes，`SchedulerLocalState` 为 208 bytes（含可选异步服务状态的 8-byte 引用），均有 64-bit 构建的静态断言。
最初 I4 测试快照使用两个 predicate 布尔值，slot 为 32 bytes、总状态为 616 bytes；
后续性能证据必须明确区分该快照与三态布局。
这些局部状态也用于无 Graph 路径，P1 必须实测其开销，不能仅凭 native 分支未执行就断言零影响。

Definition predicate 先重定位 operand 地址，在依赖满足后按当前 resident predicate 语义取值。
false 分支完成一次逻辑任务并唤醒后继，不执行 kernel，也不创建 cooperative record。
predicate 的非法 type、越界或不支持的 shape 必须在 Host preflight 拒绝。
不能先分配 membership，再用 predicate 跳过其中部分成员。

绑定失败必须携带 node_ref、原始 shell/local identity 和失败字段，锁存 run error。
该候选不得发布payload或ADMITTED；参数区可以保留部分内容，退出前不再有合法reader。
绑定失败不能包装为一次正常 child completion。

### Cooperative 与 MIX 保持同一顺序域

所有顶层和 Graph child 的 cooperative 任务共用一个 admission log，顺序仍由运行时 admission 决定。
Graph 的拓扑编号不决定 runtime order key。不能为每个实例设置私有 log 或整图 reservation。
记录以 node_ref 寻址，membership 冻结后不可扩展；同一个 child 的 args 地址对全体成员相同。

SPMD 的逻辑完成仍为全部 blocks 完成；`sync_start` 仍要求单个 child 的规定成员共同到达。
MIX 的资源成员仍为 active mask 加 AIC coordinator；双 AIV MIX 的 AIC 即使不执行 kernel 也参与协调，
未被资源 mask 选中的 AIV 不额外参与握手。
每个实际执行 lane 返回时贡献一次物理子任务完成；只有最后一个贡献者且异步条件满足，才允许完成 child 并解析后继。
AIC 观察全部资源 lane 的 release 后退役 MIX 组，这属于资源收尾，不再聚合一次逻辑完成。
逻辑完成可以先于协调核收尾，但 EXIT 仍须等待资源义务清空。block owner、lane release、seal 和 ordered-exit 记录不会因 Graph shell 完成而清除。

## GM 可见性与所有权

基于 [scheduler_memory.h](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h)
的硬件 helper 实现；表中的 release/acquire 是协议要求，不能只用 C++ `volatile` 代替。

| 数据 | 写者与存续期 | 发布与观察 |
| ---- | ------------ | ---------- |
| Definition、template、binding、实例地址 | Host；单 run 只读 | H2D/publication 完成后启动；AICore 使用既有启动观察屏障 |
| child Tensor/scalar/predicate ABI | 唯一 bootstrap 分区 owner；run 内不覆盖 | 精确范围 writeback + DDR barrier → scan arrival → scan complete → reader invalidate + barrier |
| admission immutable record/member bits | bootstrap唯一writer；commit owner只赋order_key | scan gate发布immutable字段；commit按record、log、ADMITTED、tail排序，成员按epoch/order校验后acquire |
| child 参数 reader | 普通 ready owner 或固定成员 | 普通首次 payload 准备、成员首次 adopt 后 invalidate 参数范围；不在每个 block 重复 invalidate |
| task state、wake head、queue endpoint、next link | 多 worker 的原子协议 | 保持 GM 原子读写；缓存 load/`ld_dev` 只可作为 hint，不能确认原子发布完成 |
| remaining | 多 producer 的原子通知；Host 初始化 | 独立原子区域，无 cached store/整行写回；1→0 的通知者发布 ready |
| fanout CSR/header | Host；单 run 只读 | READY 后校验；通知范围与引用均需合法 |
| fanin cursor、诊断字段 | 当前路由 owner | 与原子行分离，所有权交接前按现有规则 writeback/observe |
| run error、完成汇总、退出状态 | 沿既有多核协议 | error payload 先发布，错误标记后发布；完成计数和退出证明分别检查 |

不同 child 的 scalar/predicate 独占 64-byte cache line，不能沿用 packed scalar pool 后让多核并行整行写回。
TaskControl 为 640 bytes：state/链指针/claim tickets 共用首个原子行；cached resolver metadata、
next_block、completion_state、bypass admission signal 和 claim seal 各占一行。禁止 cached 写回触及原子行。
参数存放与定义存放不能混用一条可写 cache line。无参数时不对零地址或零长度范围做 cache 操作。
运行中不使用全 cache flush 作为每 Graph/每 child 的发布方法，也不假设 AICPU LaunchEnvelope 对 AICore 可寻址。

参数 ABI 的发布不包含 Tensor 数据本身的 scalar DCache 刷新。kernel 通过标量 load 读取其他任务更新的 GM 时，
须在 producer 完成后对所读缓存行执行 observe，或采用已验证的 bypass 读取；MTE 读取与 scalar load 不可混同。
例如 A5 Qwen 各层复用同一 64-byte `inv_rms` scratch，RoPE 消费前需观察这条缓存行。
legacy 每任务整 cache 刷新曾掩盖该输入约束；native 不以恢复全 cache 操作替代 kernel 的精确输入观察。

普通 owner 同核消费也必须满足现有 payload 发布要求；不能用“目前同核”省去未来 helper 依赖的可见性。
每次 run 的 args_backing 不同，复用物理 allocation 时启动和首次 adopt 都要按新 epoch 重新观察。
模型测试只验证逻辑 interleaving；真实硬件的 dcci/barrier 顺序仍须 onboard 验证。

bootstrap 并行绑定 `p2-bootstrap-bind` 已通过 93 个 C++ 目标、65 项 A5sim、6 个 Graph 真机目标及独立协议审计；
完整 Qwen out/K/V 通过，单批 Device 为 56.663 ms，仍明显慢于历史主线 29.478 ms。
它把 O(N) 参数准备移入启动阶段，L3 的单 worker bootstrap 中位数为 570.739 µs；
详见[候选测量与限制](qwen-performance.md)。最终全量功能、故障和性能验收仍须完成。

## ready 为空时的持续进展

普通图和 native Graph 都使用上述两队列发现路径，所有合法 AIC/AIV rank 可领取对应业务队列。
ordered poll 在本类队列为空时继续观察 admission tail、已有候选、待提交 admission 和当前 GM MIX 状态。
因此 SPMD 被取走后其他成员仍可领取 block，AIV 队列为空时仍可参与 AIC 发布的 MIX assignment/ACK。
控制切片和 async CHECK/FINISH 继续独立获得有界服务，通知未到或责任未退休时不得提前 DONE 或结束 run。

cooperative claim/commit、固定成员、全局顺序和 block permit 仍是执行资格的依据。
负提示只对本轮 discovery 有效，不消耗任务、不授予执行权，也不消除已持有候选的责任。

历史 `p2-global-ready` 跨分片候选曾通过 93 个 C++ 目标、65 项 A5sim、6 个 Graph 真机专项和独立协议审计；
完整 Qwen out/K/V 通过，单批 Device 为 43.676 ms，仍慢于当时主线。该数据及其目录扫描代价仅对应历史版本，
见[测量口径与限制](qwen-performance.md)，不代表当前两队列实现的验收结果。

当前提交还传递了 ordinary/native 编译期发现能力，A5/A5sim 构建、93 个 C++ 目标与 65 项 A5sim 通过；
该编译增量没有单独上板测量。用户要求保留已知退化形成阶段提交，随后重构全局唯一队列和分级队列，
对比全局工作可达性、同地址原子争用和端到端性能。跨分片实现作为可追溯的中间版本保留。

## 完成计数与安全退出

`N = T - G + Σ(n_g + 2)`；H 表示真正 HOST_DONE 的原始节点数。
`run_control.expected_task_count = N`，`inline_completed_count = H`，监督目标为 `resolved_task_count == N-H`。
ENTER/EXIT/body DUMMY 每个都在 resolve 尾部贡献一次，kernel child 每个逻辑任务贡献一次，不能按 block/lane 计数。
运行中完成的控制节点不计入 Host 的 inline_completed_count。

公共 host_total_tasks、原始 SM task count 仍为 T；AICore graph view 的 task_count 为 N。
AICPU supervisor、错误报告、DFX export 必须显式区分两种数量。
Graph shell 的公共完成由 EXIT 映射得出；成功结束后可以归约顶层完成状态，不新增每 child 一次的 shell RMW。

pending_completed 只累计已完成逻辑解析和后继发布的节点，按现有原子加法批量汇总。
native 模式增加每轮调度检查的定期 flush：达到 32 个完成，或非空计数距上次 flush 达 1 ms，就发布一次；
idle 和退出前仍 flush。时间使用平台频率转换，32/1 ms 是待基准调整的内部初值。
它让 AICPU 看到真实进度，不能发送不对应节点完成的假 heartbeat。
单个 kernel 长于既有 watchdog 或 op timeout 的支持边界保持不变。

以下条件必须全部成立才有成功结果：bootstrap 完成、内部计数相等、首错为空、既有 worker 退出与资源释放证明通过。
EXIT 只保证数据依赖意义上的完成，不保证所有 predecessor 的 wake resolve 尾部已退出。
例如慢 producer 的剩余 waiter 仍在 continuation 中时，EXIT 可能已 DONE；该 producer 尚未进入 resolved 汇总，
因此 N-H 检查会阻止提前退出。随后仍需 `scheduler_drain_ordered_exit`、LocalDone 和平台 run completion fence。

输出、trace、Definition、参数、admission、close seals 均保持到上述退出证明完成后再回收。
无环图内每个 child 都可到达某个 sink，所有 sink DONE 才能使 EXIT DONE；
这要求 kernel 返回对应真实完成，不能支持未登记的异步输出写入。

## 错误与进度边界

Host 必须在发布前完成形状、参数布局和容量校验；不满足当前合同的输入直接失败。
选定 native 后，地址错误、publication 失败、CAS 状态非法、参数错误、超时统一进入现有 fatal/abort 路径。
能够观察到 fatal 的 worker 禁止继续发布新成功工作；并发中已发布或在途的工作仍要按退出协议处理。
错误锁存与全部 worker 同时停发不是一个原子操作，不能承诺零条在途 publication。

正常调度路径读取本worker独占cacheline的错误signal；全局错误状态用于AICPU与终止诊断。
首个错误发布者先发布错误身份，再发布全部worker signal，最后发布全局状态。
观察到全局非零状态意味着各worker signal均已发布；后续错误发布者不覆盖任何一份状态。
每个worker仍在原有许可检查点读取错误signal，不将较早一次读取作为新的执行许可。

现有失败清理可能推进完成统计，以便收敛；因此数值达到 N-H 不能覆盖非零 error。

启动描述无效时，AICore 仍报告本轮身份，但不解引用 scheduler backing，
只参与 EXIT、nonce ACK 与 teardown gate。AICPU 负责发现本轮已经报告的核，
发送 EXIT 并完成寄存器窗口清理、退出 gate 放行与退休确认。
失败 terminal 必须保持错误状态；未证明退休时不得复用该 run 的存储。
AICPU 退出循环后仍检查 error，Host 仍检查最终 run status 和 completion fence。
未完成退出证明的 backing 进入 poison/quarantine，不能归还池后被下一 run 改写。

进度论证依赖 worker 持续获得执行机会、kernel 有限返回、queue CAS 无永久饥饿以及全局 admission 的既有进度性质。
控制队列不持有 kernel 资源；控制切片每轮获得机会；同步resolver一次处理全部后继，CSR credit不重复，动态wake-list不丢失唯一owner。
因此控制节点不会新增“等待空闲槽才能释放空闲槽”的环。
这不是对硬件公平性或既有 cooperative 协议的全新形式化证明；压力测试须专门覆盖 slot 全满与长控制链。

## 依赖读取的验证边界

一次路由或 bootstrap 扫描先构建 `SchedulerFaninView`，检查 view 版本、行范围与 native 元数据；
随后在该不可变行内遍历，避免每条边、每次重复边检查都重新解析整个 Graph view。
普通任务读取 int32 原始 refs；native child/EXIT 读取 uint16 局部 refs 并加 child 基址；
顶层依赖在校验原 task ID 后查 completion map；无内部 fanin 的 child 使用合成 ENTER 边。
每次读取仍验证源索引范围和最终 producer 先于 consumer，不能凭已校验行就省略映射结果检查。
该局部 view 只在当前 READY publication 内有效，不跨 run 或 epoch 缓存。

### 只读元数据的 worker 局部校验状态

`SchedulerGraphView` 为 version 3、56-byte wire ABI。设备侧 `SchedulerGraphAccess` 复制该 view，
另外保存已校验的 header 指针和最近一个 Definition template 指针，64-bit 环境共 72 bytes。
构造只复制 view，不访问 GM；第一次 native 读取仍必须位于本 run 的启动 cache observe 之后。
复制访问对象也重建校验状态，赋值被禁止，避免把已校验指针隐式带入另一个访问范围。
仅在原有完整校验成功后保存指针，template 以检查过的表范围内的精确地址匹配。
Definition 切换时重新校验新条目，不分配按最大 Graph 数量扩展的局部数组。

executor 为每个 run 的 bootstrap 和 dispatch 分别构造新的访问对象；同地址复用、epoch 改变或 Definition 改变
都不能继承前一对象的校验状态。其有效期依赖 READY 内 header/template 不可变，以及退出证明完成前 backing 存活。
不在 wire 的 reserved 位设置信任标记；Host 和测试从原始 wire view 进入时建立新的校验范围。
kind/source/index、instance、completion map、child plan 和参数范围仍逐节点检查；
task control、queue、predicate 取值和 Tensor 数据不进入该状态，参数 publish/acquire 屏障也不改变。

该候选已通过 93 个 C++ 目标、65 项 A5sim、Qwen 完整 golden 和独立 L4 协议审计，
测量与范围限制见 [Qwen 记录](qwen-performance.md)。额外局部状态与代码生成可能影响无 Graph 路径；
五项诊断子集支持继续迭代，最终接受仍须通过完整 19 项验收。

### Graph 内部依赖与性能分析

普通 `deps.json` 的外层 Graph task 不能替代内部 kernel DAG。使用
`simpler_setup.tools.hbg.native_graph.export_kernel_dependencies(section, executed_task_ids)`
从已通过协议审计的当前原生 Graph v11 来源构造实际执行 kernel 的依赖；
该函数独立复审原始来源与 node map，
穿透 ENTER/EXIT/DUMMY、Host service 和未执行的 predicate，并保留所有 root/sink 的前驱关系。
未知 execution identity、缺失的非 predicate kernel 和不一致的原始边必须拒绝。
该投影对应一次实际 predicate 结果；复用到另一 capture 前必须核对 callable、输入以及逐 task/block 执行集合。
无插桩性能、单独 L3 性能采集和 L4 协议/拓扑采集分别报告。

### 执行子任务完成与资源收尾

每个实际 dispatch 子任务只贡献一次 `completed_subtasks`，总数为
`logical_block_num × active_mask.count()`。普通任务、SPMD 和 MIX 使用同一计数路径；
MIX 中不执行 kernel 的协调 lane 不贡献计数。predicate 跳过的 dispatch 仍退休一次，
inline/control 节点使用各自的控制完成路径。

`completion_state` 的低 16 位记录子任务完成数，16–31 位保留 Host 初始化的总子任务数，
高 32 位累计需要检查 async join 的保守票数。核内从同一次 RMW 的返回值比较完成数与总数，
无需再次读取任务形状；总数不超过 INT16_MAX，合法贡献不会修改总数字段。
每个子任务在同一次 atomic add 中增加完成数，并按需增加一票；最大子任务数不会使低位进位。
本核执行在登记异步条件后必然激活本地 async 状态，因此最后一次 RMW 返回的累计零票
可证明所有已返回的执行均无异步登记，不需要另一轮 async-close 查表、屏障及 RMW。
高位非零只表示需要检查，不等价于本任务一定有异步条件；外部 completion token 在有 async
上下文时保守投票。JSON 的 `completed_subtasks` 和 completion ticket 始终导出低位实际完成数。

需要检查时，最后一个贡献者关闭 async computation；存在待处理 `defer_wait` 时仍由 FINISH 完成逻辑任务，
否则发布 TaskDone 并解析后继。MIX 的 lane release、协调核退役和 close seal 只管理资源，
不重复贡献计数。逻辑 TaskDone 可以早于协调核收尾，正常退出 drain 会处理 RELEASED 组，
所有本地资源收尾与平台 run fence 完成后才允许复用存储。

执行核在 kernel 返回并封闭异步条件登记后直接处理完成槽，不经下一轮自身完成槽扫描。
resident 返回入口使用执行前已验证的本核槽身份，不再重复外部 completion token 的
索引、任务形状和 tracker 检查；外部 token 入口仍执行完整检查。
resident 保留执行前映射的私有 AsyncWriter；Host 构造的 group 地址、身份和 layout 在本次运行内不变，
返回时直接读取注册状态完成 seal，不重复校验这些不变字段。外部 writer 入口仍校验绑定。
seal 使用一次读取的 count/error/reservation 状态；
已封闭 group 的重复 seal 仍失败，首次错误、未结束的 reservation 和越界计数仍阻止发布。
没有异步条件的 resident group 不进入 CHECK 队列；它的 header 写回由紧随其后的
完成计数 RMW 前 DSB 一并发布。存在条件时，seal 仍在发布 join、lease 和目录前完成自身屏障；
外部 standalone seal 入口也保留自身发布屏障。
完成计数及后继 Ready 发布先于该执行的 trace 整理与写回；trace 不参与业务依赖。
启用 ordered admission 时，完成核可将一个已就绪 MIX 后继直接留在自己的候选槽，
包括由 AIV 完成前驱的情况。该槽只授予准入发布权，不授予 MIX 执行权；
MIX 的 AIC 协调、成员选择、到达同步及执行许可仍由原有协议处理。
候选槽已满或没有 ordered admission 时，MIX 后继仍通过共享 Ready 队列发布。
trace 必须在本核复用槽、释放 MIX lane、关闭成员义务及确认 EXIT 前发布完毕。
Host 仅在所有核退出握手完成后读取 trace，不能用 TaskDone 作为 trace 可读的依据。

协议沿用 v10（普通图）/v11（原生 Graph）标签，只支持当前字段布局；在每条执行记录中保留贡献前后时间、ticket 和 worker；
审计要求 ticket 覆盖全部实际子任务且无重复。非 deferred TaskDone 必须由最后贡献者发布。
MIX block 记录的 completion 区间表示资源退役，ticket 为 `UINT64_MAX`。
