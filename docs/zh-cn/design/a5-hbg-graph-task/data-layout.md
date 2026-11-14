# A5 原生 Graph 的数据布局与 Host 绑定

本文是[总设计](../a5-hbg-graph-task.md)的数据契约。类型名为内部接口。
使用 C++17 固定宽度 POD；所有 wire 类型必须通过 standard-layout、trivially-copyable、
`sizeof` 和关键 `offsetof` 检查，并在 Host、AICPU、AICore 编译目标上核对。

## 身份与拓扑映射

设原始顶层任务数为 T，Graph 实例数为 G，第 g 个实例有 n[g] 个 child：

```text
N = T - G + sum(n[g] + 2)
completion_ref[top_level_id] = 普通任务的 node_ref 或 Graph 的 EXIT ref
child_ref(g, i) = instances[g].child_begin + i
```

按原始 task ID 升序遍历：普通任务分配一个 ref；Graph 连续分配 ENTER、所有 children、EXIT。
Host 已完成的普通节点仍保留一个 ref，初始状态 DONE，列入 host-done 数而不再入队。
Graph recording 中的 DUMMY 在每次 invocation 中仍是控制节点，不因记录时有 early-resolve hint 就提前完成。

例：原始 `A(0) -> G(1) -> B(2)`，G 内部为 `c0 -> c1`：

| node_ref | 节点 | 对外身份 |
| -------- | ---- | -------- |
| 0 | A | `GLOBAL(0)` |
| 1 | ENTER G | `GLOBAL(1)`，另带 ENTER kind |
| 2 | c0 | `SUB_TASK(parent=1, local=0)` |
| 3 | c1 | `SUB_TASK(parent=1, local=1)` |
| 4 | EXIT G | `GLOBAL(1)`，另带 EXIT kind |
| 5 | B | `GLOBAL(2)` |

此时 `completion_ref = [0, 4, 5]`。任何使用 `task_ref` 直接寻址原始 SM table 的代码都必须改走访问器。
ENTER 和 EXIT 可共享 shell 身份，但没有 kernel execution identity，不能伪造两次 shell kernel 执行。

`SchedulerNodeKind`：`TOP_KERNEL`、`CHILD_KERNEL`、`CONTROL_DUMMY`、`GRAPH_ENTER`、
`GRAPH_EXIT`、`HOST_DONE`。kind 独立于现有 task flags；不占用 admission 校验中保留的 flag bit。

12-byte `SchedulerNodeBinding`：

| 字段 | 类型 | 含义 |
| ---- | ---- | ---- |
| source_index | `uint32_t` | 顶层原始 ID，或实例内 child index |
| instance_index | `uint32_t` | 实例下标；非 Graph 节点为 `UINT32_MAX` |
| kind | `uint8_t` | 上述 enum 的 wire 值 |

CONTROL_DUMMY 的来源由 instance_index 判别。kernel 身份由 kind、source_index 与实例的 original_shell_id 导出。
不存在 Graph 时 `native_header_offset == 0`，不分配 node bindings，保持当前普通任务 ref 与 SM slot 一一对应的访问路径。
存在 Graph 时所有节点均有 binding，禁止部分调用点假设普通任务仍使用原始 ID。

`SchedulerGraphView` 为 56 bytes，version 3。字段为 storage_address、task_count、
task_window_last_index、scheduler_state_address、native_header_offset、scheduler_state_bytes
六个 `uint64_t`，original_task_count 一个 `uint32_t`，view_version/view_bytes 两个 `uint16_t`。
原始任务数受公共 TaskId 的 20-bit parent 容量约束，写入 `uint32_t` 前显式检查。
无 Graph 时 header offset 为零；Host/AICPU/AICore 必须配套重编译，不承诺旧二进制读取新 view。

以下 scheduler 结构按 64-byte cache line 排布，不以命名 reserved/padding 字段占位；
独立写者通过 `alignas(64)` 隔离，结构体尾部的自然对齐空隙不承载协议含义。

| 类型 | 大小 | cacheline 所有权 |
| ---- | ---- | ---------------- |
| SchedulerTaskControl | 640 B | state/claim tickets、block cursor、completion、admission signal、claim seal 分别隔离128B Sector |
| SchedulerWorkerContext | 320 B | 4 行 READY 发布的配置，1 行本核写回诊断 |
| SchedulerTaskTrace | 320 B | publication header；kernel timing；dispatch/completion；refill/startup；execution identity，各一行 |
| SchedulerRunControl | 7296 B | 384 B配置、生命周期原子及错误身份前缀，加108个独占cacheline的worker错误signal |
| SchedulerTicketQueue | 256 B | head 和 reserved_tail 各一个128B区域；reserved_tail 是有效预留游标 |
| SchedulerTicketEntry | 64 B | 每个 entry 的 task_ref 独占原子行，发布后本轮不覆盖 |
| SchedulerGroupMailbox | 384 B | descriptor 128 B、publication 64 B、三个 lane ack 各 64 B |
| SchedulerMixCloseSeal / SchedulerReadyInbox | 64 B | 每项独立原子行 |

scheduler allocation 的 Host/设备基址均满足128B对齐。TaskControl、TicketQueue 和 AdmissionControl 的选定原子热点按128B隔离；TicketEntry保留64B步长；标量DCache维护仍为64B，局部状态保持自然对齐。

常量分别表达不同合同：`SCHEDULER_SCALAR_DCACHE_LINE_BYTES=64`用于标量缓存维护，`SCHEDULER_GM_ATOMIC_SECTOR_BYTES=128`用于选定GM热点隔离，`SCHEDULER_STATE_BASE_ALIGNMENT`由共享GM类型的对齐要求推导。

依据与适用范围：CANN master预览版[SIMT原子操作简介](https://asc.gitcode.com/api/SIMT-API/atomic_operations/atomic_operations_intro.html)给出512B L2 Cache Line及每行4个128B Sector；[优化章节](https://asc.gitcode.com/guide/operator_practice/simt_operator_optimization/instruction_optimization/atomic_instruction_optimization.html)以不同Sector分布减少争用。[标量AtomicCas](https://asc.gitcode.com/api/SIMD-API/basic_api/atomic_operations/AtomicCas.html)说明绕过DCache，但没有承诺争用粒度；[DCache维护](https://asc.gitcode.com/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.html)明确64B行。当前布局的128B选择还由AIC/AIV标量atomicAdd与ld_dev上板对照支持，不构成所有标量接口的排序或无争用保证。

TaskControl按Sector的布局为：0起为state/链指针/claim tickets等协议字；128起为next_block；256起为completion_state，320起的独立DCache行放resolver timing；384起为admission_state；512起为claim_seal。timing只在计算贡献完成后由唯一resolver写入，复用已静止counter所在Sector，避免把每个冷字段都扩大为128B。总大小640B，相对384B每任务增加256B；ticket entry仍64B，两个业务queue及commit控制对象合计固定增加384B，另有allocation对齐padding。

TaskControl 的 state、wake_list_head、next_waiter、claim_tickets、wake_cursor 共用首个原子Sector；wake_cursor位于offset32。
next_block、completion_state 与 claim_seal 分别占用独立的 128 B 原子区域，隔离并发领取、退休与领取结束观察。
completion_state 的低 16 位是实际子任务完成数，16–31 位是 Host 初始化的总子任务数，
高 32 位是需要检查异步完成的保守票数。每次子任务完成通过同一次 RMW 更新完成数和票数，
并从返回值取得总数；总子任务数不超过 INT16_MAX，合法贡献不会进位到总数字段。
导出的 completed_subtasks 只取低 16 位。
所有原子行禁止 cached 整行写回；resolver metadata 与 admission bypass signal 保持各自独立的写入区域。
wake_cursor 原子打包 next ref 与下一 fanin 位置；resolver timing 在offset320的64B行，与offset256的已静止completion计数共用一个Sector；admission_state 单独 bypass 发布。
等待链、MIX admission、sync_start 和 defer_wait 的状态均保留；completion 同步完成后继传播，不保存 resolver 续行状态。

AdmissionRecord 为 64B，offset 40 的 `uint64_t member_rank_offset` 保存同类型普通非同步 SPMD 的 block 前缀，
与 block 数/成员数同处 header 行。Host 按 task_ref 分别累计 AIC/AIV，并检查加法溢出；
设备对 eligible_count 取模后构造冻结成员集合，不增加共享计数器或逐任务前缀扫描。

WorkerContext 的首行保存 core_type/physical_core_id/type_rank/is_scheduler，以及 run_control_offset、
task_controls_offset、scheduler_state_base_address、dispatch_payload_offset、worker_index、worker_contexts_offset。
启动 watchdog 与 peer route 校验只需观察这一行。第二行保存 graph view 与 graph_summary_offset；
第三行从 offset 128 开始，保存 metadata、kernel-entry table、worker 数量，以及 offset 160 开始的 scheduler/cluster 计数和索引；
第四行从 offset 192 开始，保存 cluster_worker_ids、AICPU lifecycle offset、offset 224 开始的 trace/activity/PMU offsets 和 offset 248 的 args_dump_address。
前四行是配置；offset 256 开始的第五行是本核诊断行，保存 bootstrap_task_count、
bootstrap_start_cycles、target_bootstrap_end_cycles 和四个 trace 启动时间戳。
未被读取的聚合统计不写入 GM；本地 completion ID 计数和退出时间戳仍参与调度控制。

native header 为 128-byte、64-byte 对齐 POD，version 2。magic/version 是两个 `uint32_t`，
随后为 image_bytes/node_count/original_task_count/definition_count/instance_count、
bindings_offset/completion_refs_offset/templates_offset/instances_offset/control_inboxes_offset、
control_inbox_count/readonly_end_offset/run_epoch 共十三个 `uint64_t`，最后为 `uint32_t header_bytes`。
image_bytes 覆盖整个 scheduler backing；
保留现有 offset 0 的 `SchedulerRunControl` 固定前缀；其后的 native 只读区域按 header、binding、
completion_ref、template、instance、blob、child plan、sink 的顺序排列。
每个 region 按其 ABI 对齐；除固定 run control 前缀以外，其余 mutable control/queue 等区域放在 readonly_end 后。
无 Graph 时不插入这段 native 只读区域，也不预留 control inbox。
普通图与 native Graph 都使用[ready pool 的独立追加区域](ready-queue-design.md)，不再承诺完整 allocation layout 与旧队列实现相同。
每个 offset/count 对、区域不重叠、epoch、原始/展开数量一致性都在 Host 和设备 bind helper 中校验。
`SchedulerGraphView.scheduler_state_bytes` 由实际 allocation layout 提供；读取 header 之前先用它
检查 header offset/size 和 base 加法不溢出，再要求 header.image_bytes 与该可信长度相等。
不能先解引用任意 header offset，或仅使用待校验 header 自己声明的 image_bytes 作为边界。
completion_ref 使用 `int64_t[T]`，值必须位于 `[0,N)`，Graph shell 必须映射到其 EXIT。

## Definition 与实例 wire

将公共 `graph_execution.h` 的纯 Definition wire 拆入拟新增 `graph_definition.h`，保持公共布局不变。
包含 AICPU 指针、`std::atomic` 和对象构造的 `GraphExecution` 留在原实现头中。
AICore 读端使用 address-space-aware 的 POD 访问辅助函数，不复制 C++ AICPU execution 对象。
同时将 `TensorData` 的字段及原有基础复制方法拆入轻量 `host_build_graph/tensor_data.h`，
将共用的 `PTOBufferHandle` 拆入 `task_interface/buffer_handle.h`，避免 Definition 读端引入
`tensor.h` 的字符串、流及完整 Tensor 操作。拆分保持 96/128-byte ABI；包括 `address_space`
在内的全部字段必须复制，Tensor 的记录 owner identity 保持公共 binder 的既有语义。

每个唯一 Definition 生成一份 128-byte A5 template：`definition_offset/definition_bytes/child_plans_offset/`
`fanin_offsets_offset/fanin_indices_offset/sinks_offset/task_count/edge_count/sink_count/args_bytes/`
`execution_count/required_heap/reserved_execution_storage_bytes/graph_key` 十四个 `uint64_t`。
fanin offset/indices 指向 blob 内既有 CSR，不另复制一份 CSR；sink 为额外的 `uint16_t[]`。
所有偏移以 native scheduler image 起点为基准，使用 `uint64_t`；数量在转换为 unsigned wire 前先拒绝负值。
Definition 内原始 32-bit section offsets 保持原布局，解析时做 checked widening/addition。

64-byte `SchedulerGraphChildPlan`：

| 字段 | 大小 | 含义 |
| ---- | ---- | ---- |
| metadata | 16 B | 现有 `SchedulerTaskMetadata` 的 shape；template 中 trace base 为零 |
| tensors_offset / scalars_offset / predicate_offset | 3 × 8 B | 相对实例 args_base；缺失 predicate 用 `UINT64_MAX` |
| tensor_count / scalar_count | 2 × 4 B | 实际数量，受原 kernel ABI 容量限制 |
| execution_prefix | 8 B | Definition 内此前 kernel 的 block×active-lane 总数 |

64-byte `SchedulerGraphInstance`：

| 字段 | 类型 |
| ---- | ---- |
| definition_index / original_shell_id | `uint32_t` 各一个 |
| enter_ref / child_begin / child_count / exit_ref | `uint32_t` 各一个 |
| boundary_tensor_count / boundary_scalar_count | `uint32_t` 各一个 |
| boundary_tensors_address / boundary_scalars_address | `uint64_t` 各一个 |
| heap_base / args_base | `uint64_t` 各一个 |

四个 address/base 是本次 run 的外部 GM allocation 地址，必须在实际 device backing 确定后填入。
同一 native image 内部引用使用 offset；不把 Host pointer 或虚拟 heap 地址写进这些字段。
boundary 地址取 compacted device SM 的 payload regions；不能取尚未 rebase 的 host_sm_handle。

每个 node 仍保留独立的 16-byte metadata 与 640-byte control，方便原调度热路径直接寻址。
child metadata 由 template 复制 shape，并设置实例自己的 trace base；不重新执行 shape 推导或依赖发现。
ENTER/EXIT/CONTROL_DUMMY 无 active lane，trace index 为无效值。读端必须先按 kind 分流，
不得将空 lane 解释成非法 kernel，也不得用 `is_executable(flags)` 单独跳过控制节点。

## 依赖与参数访问器

`SchedulerGraphAccess` 统一普通/native 的只读访问，`SchedulerFaninView` 保存已校验的 fanin 行。
Host 依赖计划与设备兼容路由使用同一套访问器；kind、参数与公共身份也从该视图解析。
非法 kind、越界、溢出不得伪装成零 fanin 或空参数。实际接口见
[scheduler_graph.h](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h)。

| 节点 | fanin 来源 |
| ---- | ---------- |
| TOP_KERNEL、顶层 CONTROL_DUMMY、ENTER | 原始 SM fanin 的每个 producer，经 completion_ref 转换 |
| 非 root child，包括 body DUMMY | Definition CSR local index，加 child_begin |
| root child，包括 body DUMMY | 唯一 producer 为 ENTER |
| EXIT | template 的所有 sink local index，加 child_begin |
| HOST_DONE | 无须注册；消费者观察其 DONE |

默认Host由这些fanin行生成统一的fanout CSR与remaining，去重并过滤HostDone边；
其 header/数组为 ready pool 的独立追加区，见[前向依赖计数](dependency-counters.md)。
未附CSR依赖区的有效图使用同步wake-list，保留fanin位置单调前进和CLOSED sentinel语义。
Graph 的内部 row 可达 1023 个 producer，EXIT 可有 1024 个 sink；使用至少 32-bit count/cursor，
不套用普通顶层 `CHIP_MAX_FANIN=128` 的准入限制。

参数地址可在 Host 完整规划，不读取生产者的输出数据：PARAM record 绑定本次 boundary 的 buffer 与 origin；
SUB_TASK record 绑定 graph heap；固定 scalar 来自 blob；inherited scalar 来自实例 boundary。
predicate operand 用同样的重定位规则，加元素偏移并检查边界，在真正派发时读取值。
自身输出 predicate、缺失 boundary 来源和溢出在计划验证时拒绝。

参数区按 child 索引规划：

```text
for child i:
    tensors_offset[i] = cursor; cursor += 128 * tensor_count[i]
    scalars_offset[i] = cursor; cursor += align_up(8 * scalar_count[i], 64)
    predicate_offset[i] = cursor if predicated else UINT64_MAX
    cursor += 64 if predicated else 0
```

cursor 初始 64-byte 对齐。Tensor 使用现有 128-byte实体 ABI；96-byte Definition record 不能直接传给 kernel。
每个 child 拥有完整的 scalar/predicate cache line，零长度区间不读写。
所有加法、乘法和 align-up 都检查溢出，再验证总大小。
Host 在独立镜像中写入每实例 child 参数，随 launch metadata 发布；
Device 消费前观察参数范围。完整准备与消费合同见[统一开发参考](build-execute-separation.md)。
每 run 重建本次实例地址与 scalar，不跨 epoch 复用参数内容；Tensor 数据和 predicate 值不在准备阶段读取。

首版 args_base 指向 `heap_base + definition.required_heap`，复用当前 shell 已预留的 execution tail。
检查 args_base 的 64-byte 对齐，以及 `native_args_bytes <= definition.execution_storage_bytes`。
这是发布前的硬校验；不等式失败就拒绝 prepare，不改变执行器或在运行中换地址。
每个 child 的 tensor/scalar offset 必须等于对应池的连续游标，最终游标必须覆盖完整池。
参数区间重叠、留洞和未使用的池尾部都不属于当前 Definition 合同。
该区域只写 native 参数，不同时 placement-new `GraphExecution` 或 child `ChipTaskStorage`。
旧 tail 的预留大小仍计入内存报告，不能宣称已节省其 320×child_count 字节。

## Host 准备与生命周期

`bind_graph_definitions` pack 并校验 Definition，`create_scheduler_state` 在 commit 阶段创建 resident image。
它们按下面顺序执行：

1. 等待所有 recording job 完成，保持现有 Graph shell 提交、heap 预留与 boundary 校验。
2. pack 并验证所有唯一 Definition；从 fanin 推导 roots/sinks，检查扇出 CSR 与其一致。
3. 汇总整个 run 的形状与容量，拒绝非法图或不满足当前 resident 合同的输入。
4. 生成不含本次设备地址的 `NativeGraphHostPlan`：blob 自有字节、template、node map、prefix、计数和静态元数据。
5. 通过当前静态 arena/heap 流程取得实际 device backing，compact/rebase 原始 SM。
6. 根据实际 GM base 填实例地址，生成 native scheduler image，初始化每个 control 和 admission record。
7. 将所有待上传内容交给本 run slot 的 publication batch；完成既有 publication 后才能启动。

native 模式的 Definition 位于同一份 GM scheduler backing 的只读区域，每个 Definition 每 run 复制一次。
Definition 不附到每个 AICPU launch envelope；Host 另外保留原始 Definition section，供独立 protocol oracle 使用。
总定义字节影响一次合并上传的大小，实例数不增加 Definition 复制次数；并不承诺整个 run 只有一次 H2D。

`GraphHostState` 是 orchestration 函数的局部对象，返回后销毁。
`NativeGraphHostPlan` 必须在它销毁前取得自有 blob 和索引，保存在 A5 的 HostPreparePlan 中，
不能保存 recording vector、unordered_map element 或它们的地址。
可通过 A5 自有的 plan 包装现有 `RetainedHostPreparePlan`；不把 A5 类型塞入公共 A2A3 plan。

| 事件 | 所有权契约 |
| ---- | ---------- |
| prepare 尚未 commit，申请资源失败 | 保留无地址的 plan；retry 根据新的 GM backing 重新实例化所有地址 |
| park/resume | plan、publication source、tensor leases 都归同一 pipeline slot，不能借用临时 Runtime 的地址 |
| run N+1 prepare 与 run N execute 重叠 | 两者的 Definition、参数区、控制区、epoch 及 pinned source 均独立 |
| publication 任一步失败 | 不启动；若已有异步提交，沿原有 Partial/drain/quarantine 路径保留源和目标 |
| 正常结束 | 通过现有 worker/resource 退出证明后释放或归还池 |
| 退出证明失败 | poison 并保留借用存储，不给后续 run 复用 |

所有 mutable state 每 run 初始化一次，任何 early reader 之前已完成初始化。
运行中不复用 child ref、参数区或 close seal，因此无需在 Graph 内引入 epoch 回绕或槽回收协议。

## 容量和预算

同时验证：T 的公共 TaskId parent 容量、每 Definition 的 1..1024-child 范围、当前 16-Definition 上限、
N 的 `INT32_MAX` ready endpoint 上限、所有偏移/字节乘积、block×lane 的 `INT16_MAX` 契约、
cooperative record/log 容量以及 trace prefix 的 32-bit 地址范围。
原始 `ring_task_window` 限制 T；不能把 N 写回该配置而无声改变公共任务容量语义。

用一次 O(n+e) 验证构造每个 Definition 的计划；重复边可用 n 大小的标记数组检查，
不要对每个实例执行普通分类器中的二重 duplicate scan。
Host 展开 binding/metadata、初始化 control 和设备 bootstrap 仍是 O(N)；CSR 存储为唯一 Definition 的 O(e)。
设备 bind 验证 header/region 容量和版本，bootstrap 分片验证各自 node 的映射；不让每个 worker 重新遍历全图验证。
完整语义校验在 Host 按唯一 Definition 做一次，设备按需读取仍检查索引/地址；不能以省启动时间为由取消边界检查。

```text
native_scheduler_bytes = header + Definition blobs + templates + child plans + sinks
                       + 8*T completion refs + 64*G instances
                       + 12*N node bindings + 16*N metadata + 640*N controls
                       + control inboxes + actual cooperative records/log/seals
                       + ready pool descriptors/controls + actual kernel ready entries
                       + existing worker/run/ready regions + alignment padding
                       + actual enabled diagnostics
reserved_graph_heap = sum(required_heap + reserved_execution_storage_bytes)
```

native_scheduler_bytes 是 scheduler backing 总量，不是相对于旧 scheduler 的增量；公共原始 SM 和其他 runtime backing 另计。
公式中的 reserved_graph_heap 已包含 native 参数 backing，禁止再把同一参数区算成额外 allocation。
cooperative metadata 只计合法 kernel，ENTER/EXIT/DUMMY 不增加 admission log capacity。
首版可保留现有按 N 索引的 admission records；必须如实计算这种稀疏分配成本。

当前 task trace 为 320 bytes，layout 不因采集关闭自动省略它；执行数按 block×lane 累加。
实施时显式传入诊断需求：无执行/采样采集时 trace offset 可为 0，PMU/protocol 各按真实需求分配。
必须先审计所有 trace 读写者，包括 host export 和错误路径，再允许零 offset；有采样时仍按其需要保留索引。
如果暂未完成这项审计，验收必须按保留 trace 的实际成本报告，不能用条件分配的理想数字代替。
