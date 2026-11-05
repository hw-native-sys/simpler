# A5 Graph task 调度与发布协议

本文是[设计总览](../a5-hbg-graph-task.md)的设备侧实施契约，接口均为拟新增或拟调整。
节点编号和 GM 区域见[数据布局](data-layout.md)。
当前代码恢复到全任务 cooperative admission。曾划分独立 block、MIX lease 和同步 cohort 的
[重构原型](block-ready-scheduling.md)已经撤回，只保留为后续待选方案。

## Bootstrap 与控制节点

Host 初始化完整 N-node control 表，只有确认已在 Host 完成的原始节点设为 HOST_DONE。
其余节点初态沿用 `SchedulerTaskState::BLOCKED`；ready、私有 claim 是所有权位置，不新增 GM task state。
Graph body 的 DUMMY 也必须实际经过依赖路由，不能复制 recording 阶段的 early-resolve 状态。

native header 的第一次读取必须在本 run 的缓存刷新之后。当前主线在 local config/admission 初始化后
才执行整 data cache 的 bootstrap observe；如果把 native header 校验放进 local config，这个顺序会太晚。
native 路径应在取得 READY 后的 context 并检查基础地址/长度后，先执行一次 bootstrap cache observe，
再读取 header/template/binding；后面的同类 observe 不重复执行。无 Graph 路径保留原时序。
这仅是每 run 的启动操作，不允许扩展为每 child 的整 cache 刷新。
同地址连续 run 使用不同 Definition、boundary 和 scalar 的回归必须能发现旧 epoch 数据残留。

所有 worker 按现有区间划分扫描 N 个节点：先读 kind，再决定跳过 HOST_DONE、路由控制节点或路由 kernel。
无 fanin 的控制节点进入 control inbox，kernel 进入原有按 AIC/AIV/MIX 分类的 ready inbox。
有 fanin 的节点注册到第一个未完成 producer；跨 Graph 的 producer 已经由 Host 映射。
**全部扫描结束前，不 pop control inbox，也不完成任何 ENTER/DUMMY。**
`bootstrap_scan_complete` 后才可以处理控制节点和准备首批 slot；AICPU 仍等待现有
`bootstrap_complete` 后通过唯一 DMB gate 放行执行。

这保留了 `scheduler_bootstrap_route_task` 的 exchange-before-link 前提：构造 wake-list 时没有 resolver。
稳态必须继续使用带 CLOSED 检查的 CAS 路由，不能把 bootstrap 的无并发完成假设带到稳态。

### 独立 control inbox

现有 `scheduler_complete_inline_task` 由 slot fill 路径调用，而且验证了 kernel 风格的 metadata。
原样复用会让 ENTER/EXIT 等待空闲 payload slot。新增以下控制入口：

```cpp
bool scheduler_control_enqueue(/* base, local, node_ref */);
bool scheduler_service_controls(/* graph, base, local, run, budgets */);
```

每 worker 一个 control inbox，复用 `SchedulerReadyInbox` 的 64-bit head/tail 原子协议和
`next_waiter` 链，不使用 AIC/AIV/MIX 的资源过滤、type rank 或 directory mask。
将现有 queue push/pop 的机械操作提取为接受 inbox 地址的共用 helper，保持原有 CAS、尾链发布和有限重试语义。
运行中 ready 的控制节点先推到 resolver 所在 worker 的 inbox；bootstrap 按扫描 owner 分片。
所有 worker 都可消费控制节点，先查本地、再按轮转游标探查一个 peer，最终覆盖所有 worker。
kernel inbox 的空目录提示不能屏蔽这次控制队列探查。

一个节点从 wake-list detach 后先按 kind 分流：控制节点只进 control inbox；kernel 才能 direct refill、
进入 type inbox 或 admission。禁止把 ENTER/EXIT 当作 AIV kernel，禁止递归调用其后继完成函数。
MPMC pop 的成功 CAS 是唯一 owner 的获取点，不再增加 Graph 级 owner lock。

### 控制完成切片

每轮先处理已有 kernel completion，再运行一个控制切片，然后继续 ordered obligations、slot fill 和 kernel。
即使所有 payload slot 占满或 admission candidate 被暂存，也必须运行控制切片。
已有 kernel completion 的控制后继只入队；下一切片处理，避免 completion→control 的递归。

建议初值为每轮最多新完成 8 个控制节点、路由 64 个 detached waiter，作为内部常量。
每 worker 只有一个私有 `ControlResolveContinuation { producer_ref, next_waiter }`：

1. pop 一个 ready 控制节点，校验 kind、BLOCKED 和无 kernel 资源；以原子写发布 DONE。
2. 原子 exchange 将其 wake-list 置为 CLOSED，取得完整旧链；重复 CLOSED 是协议错误。
3. 先原子读取当前 waiter 的 next，再调用通用 fanin 访问器路由此 waiter。
4. 路由结果只能是重新挂到某个 producer、进入某个 ready inbox，或错误；不能同时存在两个位置。
5. 达到 waiter 预算就保存剩余链头；返回前发布本切片所有 ready batch。下轮先继续这条链。
6. 链耗尽且 batch 发布完成后才增加本 worker 的 `pending_completed`，清空 continuation。

continuation 私有拥有整条剩余链；不把它重新入队，不再次关闭 producer，也不再次发布 DONE。
每个 waiter 的一次路由不可中途悬停，其 fanin 扫描最多为该行长度，当前上限为 1024。
因此 64 是 waiter 数预算，**不是 64 次 GM 访问或硬实时周期上限**。
kernel completion 原有路径仍可能一次处理较长链；此处只约束新增控制工作，后续据测量决定是否统一切片。
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
| SPMD、MIX、sync_start | bootstrap node 分区 owner | scan arrival 前 | 固定 membership 中首次 adopt 的 worker |
| predicated 单 lane、单 block | bootstrap node 分区 owner | 只提前绑定地址；取值仍在 ready 后 | 跳过路径或本 worker |
| ENTER/EXIT/DUMMY | 无 | 无参数绑定 | 无 |

每个 bootstrap 分区互不相交，child 参数的唯一写者由 node_ref 确定。每次绑定执行完整边界检查、
精确范围 writeback 和 DDR barrier；这些操作完成后才增加 scan arrival，最后到达者打开 scan gate。
现有所有 ready 消费路径均在该 gate 后进入，因此无须逐 child 增加 ARG_PREPARING 锁或发布位。
绑定失败锁存错误，该分区不报告成功到达；其他 worker 在等待 gate 时沿现有 error/watchdog 退出。

普通 ready owner 可能不是参数写者，首次准备 payload 前必须 invalidate 参数范围并执行 barrier。
私有 pending candidate 用 `args_observed` 保存已观察状态，随 task_ref park/restore，最终释放或 run 失败才清除。
cooperative 的 claim owner 只准备 admission record；所有成员首次 adopt 时仍观察参数，后续 block 复用。
现有 PREPARED/ADMITTED、membership、短 commit lock 和 kernel 资源准入均不提前。

`SchedulerLocalSlotState` 用三态 predicate（未求值、通过、不通过）保存一次求值的结果，
park/restore 与 `args_observed` 一起迁移，不因重试而读取更新后的 predicate operand。
slot 保持 24 bytes，`SchedulerLocalState` 为 632 bytes，均有 64-bit 构建的静态断言。
最初 I4 测试快照使用两个 predicate 布尔值，slot 为 32 bytes、总状态为 616 bytes；
后续性能证据必须明确区分该快照与三态布局。
这些局部状态也用于无 Graph 路径，P1 必须实测其开销，不能仅凭 native 分支未执行就断言零影响。

Definition predicate 先重定位 operand 地址，在依赖满足后按当前 resident predicate 语义取值。
false 分支完成一次逻辑任务并唤醒后继，不执行 kernel，也不创建 cooperative record。
predicate 的非法 type、越界或 native 不支持的 shape 必须在 Host preflight 区分为输入错误或 legacy 能力回退。
不能先分配 membership，再用 predicate 跳过其中部分成员。

绑定失败必须携带 node_ref、原始 shell/local identity 和失败字段，锁存 run error。
该候选不得发布 payload/PREPARED；参数区可以保留部分内容，退出前不再有合法 reader。
运行中不能改走 legacy，也不能把绑定失败包装为一次正常 child completion。

### Cooperative 与 MIX 保持同一顺序域

所有顶层和 Graph child 的 cooperative 任务共用一个 admission log，顺序仍由运行时 admission 决定。
Graph 的拓扑编号不决定 runtime order key。不能为每个实例设置私有 log 或整图 reservation。
记录以 node_ref 寻址，membership 冻结后不可扩展；同一个 child 的 args 地址对全体成员相同。

SPMD 的逻辑完成仍为全部 blocks 完成；`sync_start` 仍要求单个 child 的规定成员共同到达。
MIX 保留 AIC coordinator 和必要的三资源 lane 参与，包括 inactive lane 的资源握手，
AIC 观察 release 后才聚合 block 完成。AIV 单 lane 返回不能提前完成 child 或释放 EXIT。
复用现有 block owner、LocalDone、seal 和 ordered-exit；不因 Graph shell 完成而清除这些记录。

## GM 可见性与所有权

基于 [scheduler_memory.h](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_memory.h)
的硬件 helper 实现；表中的 release/acquire 是协议要求，不能只用 C++ `volatile` 代替。

| 数据 | 写者与存续期 | 发布与观察 |
| ---- | ------------ | ---------- |
| Definition、template、binding、实例地址 | Host；单 run 只读 | H2D/publication 完成后启动；AICore 使用既有启动观察屏障 |
| child Tensor/scalar/predicate ABI | 唯一 bootstrap 分区 owner；run 内不覆盖 | 精确范围 writeback + DDR barrier → scan arrival → scan complete → reader invalidate + barrier |
| admission immutable record/member bits | claim owner，现有协议 | 沿 PREPARED/ADMITTED 和 log tail 发布，成员按原有 epoch/order 校验后 acquire |
| child 参数 reader | 普通 ready owner 或固定成员 | 普通首次 payload 准备、成员首次 adopt 后 invalidate 参数范围；不在每个 block 重复 invalidate |
| task state、wake head、queue endpoint、next link | 多 worker 的原子协议 | 保持 GM 原子读写；缓存 load/`ld_dev` 只可作为 hint，不能确认原子发布完成 |
| fanin cursor、诊断字段 | 当前路由 owner | 与原子行分离，所有权交接前按现有规则 writeback/observe |
| run error、完成汇总、退出状态 | 沿既有多核协议 | error payload 先发布，错误标记后发布；完成计数和退出证明分别检查 |

不同 child 的 scalar/predicate 独占 64-byte cache line，不能沿用 packed scalar pool 后让多核并行整行写回。
已有 task control 的 128-byte 原子域隔离不缩小，不能把 cached 字段塞入原子行再整行写回。
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

## native ready 的跨分片发现

native Graph 的 worker 先取自身 inbox，再搜索本地同类型 ready 分片；仍未取得任务时，
按循环顺序搜索其他同类型分片。扫描上限由实际 owner_count 决定，最后一个分片屏蔽未启用 owner。
普通图仍使用本地分片策略；bootstrap/dispatch 的既有图类型模板继续传入 slot fill、ordered poll 和 ready claim，
在普通图实例中编译期移除 remote discovery，不依赖大函数内联后再消除 native 分支。
远程分片不屏蔽与自身 rank 同余的 bit，也不能对负的 rank 差做位移。

所有取任务操作复用既有 inbox CAS；跨分片只改变发现范围，不改变依赖、入队唯一性或发布屏障。
cooperative 任务被取出后仍走相同 claim/commit、固定成员、全局顺序和 block permit，
不会因为被远程 worker 发现就直接取得执行资格。控制节点 inbox 的协议保持独立。
ordered idle hint、ordered fill 和普通 slot fill 的 steal hint 必须采用同一发现范围；
negative hint 只对本轮有效，不消耗任务、不授权执行。

该候选标识为 `p2-global-ready`：93 个 C++ 目标、65 项 A5sim、6 个 Graph 真机专项和独立协议审计通过；
完整 Qwen out/K/V 通过，单批 Device 为 43.676 ms，仍慢于历史主线。详见[测量口径与限制](qwen-performance.md)。
代价是空闲时额外读取目录分片，以及跨分片竞争同一 inbox；须以 Qwen 核分布、稳态性能和无 Graph 回归验证。

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

Host 必须在发布前完成能力判断：非法输入直接失败；合法但 native 不支持则整 run legacy。
选定 native 后，地址错误、publication 失败、CAS 状态非法、参数错误、超时统一进入现有 fatal/abort 路径。
能够观察到 fatal 的 worker 禁止继续发布新成功工作；并发中已发布或在途的工作仍要按退出协议处理。
错误锁存与全部 worker 同时停发不是一个原子操作，不能承诺零条在途 publication。

现有失败清理可能推进完成统计，以便收敛；因此数值达到 N-H 不能覆盖非零 error。
AICPU 退出循环后仍检查 error，Host 仍检查最终 run status 和 completion fence。
未完成退出证明的 backing 进入 poison/quarantine，不能归还池后被下一 run 改写。

进度论证依赖 worker 持续获得执行机会、kernel 有限返回、queue CAS 无永久饥饿以及全局 admission 的既有进度性质。
控制队列不持有 kernel 资源；控制切片每轮获得机会；路由游标只前进，wake-list 不会丢失唯一 owner。
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

`SchedulerGraphView` 的 64-byte wire ABI 保持不变。设备侧 `SchedulerGraphAccess` 复制该 view，
另外保存已校验的 header 指针和最近一个 Definition template 指针，64-bit 环境共 80 bytes。
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
从已通过协议审计的原生 Graph v9 或历史 Graph v6 来源构造实际执行 kernel 的依赖；
该函数独立复审原始来源与 node map，
穿透 ENTER/EXIT/DUMMY、Host service 和未执行的 predicate，并保留所有 root/sink 的前驱关系。
未知 execution identity、缺失的非 predicate kernel 和不一致的原始边必须拒绝。
该投影对应一次实际 predicate 结果；复用到另一 capture 前必须核对 callable、输入以及逐 task/block 执行集合。
无插桩性能、单独 L3 性能采集和 L4 协议/拓扑采集分别报告。
