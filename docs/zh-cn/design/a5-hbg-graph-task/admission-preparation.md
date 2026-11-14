# A5 HBG admission 静态计划与一次验证：开发参考

**状态：完整目标设计，尚未实现或验证收益。** 更新日期：2026-11-12。
当前核对基线：`main@a013b1710aee9d9fe7117ba249a709c3a64a3e43`，已包含 slot 直接退休和 Scalar profiling 更新。
旧局部实现已不在当前工作树中；历史实验不能代替本基线的功能和性能验证。
**采用一次完整交付：Host 静态计划、bootstrap 分核验证、静态/动态布局分离、热路径简化和全部形状覆盖同时落地。**
性能以 **L0 完整 device 耗时**为准，没有固定无变化区间，1% 的收益也计入。

## 1. 目标方案与通用边界

Host 每次 bind 生成 admission 静态计划。设备在既有 bootstrap 分区扫描中，对每条计划完整验证一次；
原有 scan gate 放行后，prepare、commit 重试和消费者只处理动态状态，不再重算完整成员集合。
这种验证转移以本 run 内静态计划无写者、每核读取新数据及现有启动门禁为依据。

以下五项缺一不可，不能把“只做 Host 预填”或“只优化纯核普通 SPMD”作为交付结果：

1. 所有 cooperative 任务使用同一套静态计划和成员算法。
2. order_key 与静态字段分属不同 cache line，计划发布后直到 run 退休不再写入。
3. 每条计划的完整设备验证只发生在 bootstrap，失败时不能进入首波 slot fill。
4. 热路径和 RETRY 路径都删除静态重算；动态所有权、错误、序号、计数及完成检查保留。
5. 布局、Host/device writer/reader、DFX 导出、测试和文档配套更新，按完整改动做 L0 对照。

### 覆盖全部已有合法形状

| 场景 | 静态成员规则 | 保留的动态语义 |
| ---- | ------------ | -------------- |
| 纯 AIC/AIV、非同步 SPMD | 同类型循环区间，成员数 min(N, 可用核数) | ready、顺序 admission、原 block 领取与完成 |
| 纯 AIC/AIV、sync_start | 完整同类型候选集合 | 不同执行位置各至多领取一次、共驻与关闭约束 |
| MIX：单 block、SPMD、sync_start | 完整 resource mask，包含 AIC coordinator | 每组领取、lane 会合、执行、release、退休 |
| native Graph cooperative child | 按每实例展开后的 node_ref 应用上述规则 | 参数绑定、ENTER/依赖、EXIT、实例独立性 |

适用范围由 `scheduler_task_requires_admission(metadata)` 和合法形状决定，不使用 sync/MIX/Graph 排除条件。
普通单 block、inline 和控制节点不新增 admission；零 cooperative task 保持零协议分配。
`sync_start + MIX + Graph` 组合和恢复 `defer_wait` 的场景均在功能覆盖内。
既有非法/不支持形状沿用现有拒绝或调度选择合同，不扩大前端能力。

### 避免过度定制

只依赖 metadata、已规划/已验证拓扑、布局和 run 身份。
不按模型名、task ID、层号、64/96 核样本或特定 block 数选择路径；文中的 t29 仅作解释。
不增加模型专属队列、逐任务 validated 原子位、每核整图副本、跨 run 计划缓存或新行为开关。
保留两个 ready 队列、kernel 和用户表达；不同时改 block 分配、kernel/payload 算法或完成传播。
代码可以按依赖顺序开发和检查，但只接受这套完整机制，不保留可切换的中间实现或旧 wire 兼容路径。

## 2. 静态计划、布局与 Host 准备

### 一份计划，一份动态发布记录

将现有 header+bitmap 的两行重新组织，采用独立类型和只读访问接口：

| 区域 | 拟定字段与所有权 | 当前 A5 大小 |
| ---- | ---------------- | ------------ |
| `SchedulerAdmissionPlan` | epoch、task_ref、MIX seal 行 offset、rank_offset、成员 words、block 数、member_count、active mask、flags；Host 唯一写者 | 64 B，对齐 64 B |
| `SchedulerAdmissionPublication` | order_key；Host 初始化 UINT64_MAX，commit owner 唯一运行时写者 | 64 B，对齐 64 B |

plan 中的前四项为四个 64-bit 字段，成员位图当前为两个 64-bit word；
block 数/member_count 为两个 32-bit 字段，mask/flags 为两个单字节字段，合计 58 B 后自然对齐到 64 B。
容量按 `ceil(SCHEDULER_WORKER_CAPACITY / 64)` 推导，不在算法中硬编码两个 word；
现有 capacity=108，因此每任务总 stride 仍可为 **128 B**。编译期检查实际 size、alignment 和 offset。
未来容量变化按类型大小重新规划，不能静默截断；寻址使用类型/offsetof，不复制魔法字节数。

bitmap 内联在只读 plan，删除可写的 `member_words_offset`，按布局计算其位置。
保留原按 node_ref 寻址方式和 cooperative log 容量；不压缩成新字典，也不保留原 record 再复制一份 plan。
publication 行仍是单写者普通数据行，沿用写回/观察协议；不得与 atomic-only 的 TaskControl 行混放。
admission_state、claim seal、计数器、mailbox、lane ACK 留在原有动态结构中。

### 共用完整成员计算

在轻量公共头中提供 Host 和 bootstrap 共用的纯计算，不把整个设备 admission 实现 include 到 Host。
输入为合法 metadata、worker/cluster 数、按 worker_id 编码的 AIC/AIV0 拓扑位图及 member_rank_offset；
AIV1 由有效位补集得到。输出为 bitmap、member_count 和容量合法性。
函数本身不读 GM、不执行 cache/atomic、不分配；完整计算退出所有调度热路径。
GENERAL 的现有测试/可规划调用者显式提供映射，不把 GENERAL 枚举当成“跳过预填/验证”的开关。

Host 按最终 node_ref 顺序分别累计纯 AIC/AIV 非同步 SPMD 的 block 前缀并检查溢出。
非同步 N 小于候选数时，从 `rank_offset % candidates` 起循环选择 N 个成员；否则全选。
MIX、sync_start、普通单 block 和另一资源类型不推进该前缀；不能改成固定低编号核。

sync_start 的 rank_offset=0，使用完整候选集合；N 不得超过不同执行位置的容量：
纯 AIC 为 C、纯 AIV 为 2C、MIX 为 C 个 cluster。pending slot 数不增加容量。
MIX 使用 `scheduler_mix_resource_mask(active_mask)`，不按 N 裁剪。
active mask、resource mask、成员数和 block 容量分别决定执行者、会合参与者、参与资格和共驻容量，不能混用。
例如 C=32 时 1C1V 有 64 个成员、1C2V 有 96 个；双 AIV MIX 也包含非执行 AIC coordinator。

### 原始任务和 Graph 使用同一个准备入口

`create_scheduler_state` 完成 native 展开、`task_metadata = expanded` 后，统一生成计划。
校验最终 metadata 数量与 layout.task_count、cooperative/MIX 计数一致，再准备对应行；
检查全部乘加溢出、对齐、bitmap 高位清零、seal 行边界及上传 byte extent。
无记录/错误记录应被拒绝，设备不现构造一份来掩盖缺失。

例如 `ENTER0, child1, EXIT2, ordinary3, ENTER4, child5, EXIT6`，
若三个 kernel 都是 cooperative，计划按 ref 1、3、5 寻址。
同一 Definition 的实例可共享 Definition，但每实例的 plan/publication、计数及 seal 行独立。
node_ref 决定地址，ready 决定实际 commit order；不能用原顶层 ID 或 Definition 内 child index 代替。

Host 沿 `add_pending_metadata → publish_run_image_impl → stage_launch_metadata → launch_metadata_.publish` 发布。
现有 staging 保存值快照，复制到受 slot 生命周期保护的 pinned 区，再在 descriptor/entry 之前排队 H2D。
所有计划必须在这份快照和上传范围内；不能只填 Host 镜像，或在上传后原地修改。

## 3. 一次验证与运行时交权

### 把静态证明放到已有并行扫描中

`post_handshake_init` 逐 worker 核对实际核类型、compact identity、cluster 和计划一致，之后发布 context READY。
Host 不提前写 `COMPACT_LAUNCH` 证明。每个 AICore 随后使用已有入口 cache 维护读取本次新数据。

在 `bootstrap_ready_graph` 已有互不重叠的 node_ref 分区中加入静态验证：
每个 cooperative plan 只由该 node 的分区 owner 验证一次，且在其路由/ready 发布之前完成。
检查 plan 与 metadata/节点身份、epoch、shape、成员集合、同步容量、MIX seal 行和布局边界；
非 cooperative/control 节点验证其分类与空计划合同，避免遗漏或错把控制节点当 kernel。
公共布局边界在访问行之前已经成立；不能先解引用非法 offset 再验证。

owner 只有验证完本分区所有节点、并完成原有 Graph 参数绑定和路由工作，才贡献现有 scan arrival。
失败设置原有错误，不贡献成功 arrival；其他核在原有 error/watchdog 路径退出。
不新增全任务 AICPU 扫描、第二次 node 遍历、逐任务 ACK 或新的 barrier。

```mermaid
flowchart LR
    H["Host 生成只读计划"] --> T["H2D 与实际拓扑 READY"]
    T --> V["既有 bootstrap 分区：每计划验证一次"]
    V --> G["既有 bootstrap_scan_complete"]
    G --> F["首波 slot fill / 后续调度"]
    F --> C["动态校验及 commit"]
    C --> M["动态序号校验 + 本核成员位"]
    M --> K["原 block / MIX / kernel 协议"]
```

关键边界是 **bootstrap_scan_complete**，不是最后的 bootstrap_complete：
当前首波 `scheduler_fill_own_slots` 已经在 scan gate 后、最终启动放行前调用，必须同样使用验证后的计划。
用现有控制流和一个绑定本次入口的本地只读视图表达证明；不为每任务重复读取共享 validated 标志。
配置 initialized 不等于计划已验证；裸 base 指针或可跨 run 保留的 bool 不能绕过门禁。

### 热路径只保留动态检查

| 路径 | 目标行为 |
| ---- | -------- |
| ready claim | task_ref 范围、唯一所有权、当前 phase、错误状态；从已验证计划取得形状 |
| prepare | 检查 pending/CLAIMED、claim seal、三个初始计数及初始 publication.order；不重算 shape/bitmap，不写 plan |
| 首次 commit | 本次动态校验通过后拿短锁，动态分配 order，按原序发布 publication、log、ADMITTED、tail |
| commit RETRY | 静态证明在同一 run 内继续成立；动态 phase、计数、order、错误和容量必须重新检查 |
| 消费者 refresh | 校验 log ref、动态 order/ADMITTED、当前执行身份；读取本核 bitmap 位，不进行完整成员重算 |
| 完成/异步关闭 | 原 block、sync_claimed、MIX lane 和 defer_wait 协议不变 |

`prepared_now` 表示本次完成动态校验并推进 PREPARED；不能把它改成 Host 曾准备过的永久执行许可。
完整静态验证函数仅供 Host/启动使用，prepare、refresh 和 retry 的生成代码中都不得保留其重算逻辑。
消费者对非成员也验证动态序号，不能用成员位绕过损坏的 log；已有有界扫描和前沿推进规则保留。
动态计数/错误/phase 的强观察属于本方案的协议边界，不能以“一次验证”为由删除。

### 静态只读和缓存新鲜性必须闭合

所有运行时 accessor 返回只读 plan；审计 prepare、commit、MIX、async、DFX 和退出路径，确保没有合法写者。
order_key 移到 publication 行后，commit 的整行写回不再触及 plan。
采用现有发布原语维护动态行，保留其 cache observe、写回、DSB 和控制交权；本次不改为新的 st_dev 或原子协议。

现有普通/native 入口在 bootstrap 前均调用 `scheduler_observe_data_cache`。
保留并验证每核在 H2D/READY 后、首次读取 plan 前执行该维护；其他核验证成功不会刷新本核缓存。
此前提与只读生命周期成立后，稳态不再对 plan 执行逐次 DCCI，也不重新扫描 metadata 做全量比较。
publication 的动态观察仍保留。编译产物不得把 plan 读取提前到入口维护之前。
只读允许缓存复用，不保证缓存命中：现有 kernel 边界维护和 cache 逐出仍可能使后续读变冷，不增加广播预取或新 ENTIRE 维护。

新 bind 重新生成并验证计划；同地址复用先取得原有 reader/DMA 停止和资源退休证明。
epoch=0/UINT64_MAX 是合法身份，数值相同或不同都不能替代本次入口验证，也不能触发跨 run 缓存命中。
prepare 重试仍属于同一 run；框架 capture/replay 与 native Graph 分开处理，保持既有支持范围、完整初态恢复和失败隔离。

### 保留业务语义，明确静态损坏检测边界

sync 保留每个不同执行位置至多一次领取，不新增所有 block 同时到达的全局 barrier。
MIX 保留 AIC coordinator、按资源 lane 的 arrive/release 和 seal 关闭，非执行 coordinator 不贡献 kernel 完成。
Graph 保留参数绑定、scan gate、ENTER/依赖和 EXIT；异步通知仍需真实满足，计划存在不代表数据已就绪。

本设计将静态错误检查集中到启动边界：违反身份、形状、成员或边界合同的 plan 必须拒绝进入首波 fill。
验证后由只读所有权保证合法运行不会改写计划；不承诺检测冻结后任意非法写入/硬件损坏的每一次后续影响。
这与原测试中“commit 后人为改 bitmap，refresh 必须立即报错”的防损坏能力不同，必须在测试/文档中明示。
静态损坏回归移至新验证边界，并增加门禁和无合法写者证明；动态 order、phase、计数、重试等故障仍在原时点拒绝。
不能直接删除失败测试后声称完全等价，也不另加昂贵 hash、影子副本或行为开关维持两套合同。

## 4. 预期收益、来源与估算边界

### 能确定的工作量变化

设 W 为实际扫描某记录的核数，R 为 commit 重试中完整复核的次数。
正常路径中该记录的完整成员计算约为 `1 + W + R` 次；目标为设备 bootstrap **一次**，另有 Host 生成一次。
96 核全扫描且无重试的例子是设备约 97→1；并行工作量减少不能解读为 97 倍延迟加速。

| 收益来源 | 基线开销 | 完整目标方案 |
| -------- | -------- | ------------ |
| owner 的静态工作 | metadata/shape/成员计算、静态 GM 写入 | 热路径删除，启动分区统一验证 |
| 每个日志消费者 | 全量 static_record/bitmap 校验，含非成员 | 动态校验 + 本核一个成员位 |
| prepare 的 cache 操作 | header/bitmap 两次 observe、两次 OUT | 只观察动态 publication；不写/刷 plan |
| refresh 的 cache 操作 | header/bitmap 两行观察，再读 metadata | 只观察 publication；静态数据正常只读加载 |
| commit 重试 | 每次重做完整静态校验 | 只重做动态校验 |
| 新增成本 | Host 原静态字段准备与原 bootstrap | Host 生成完整计划；每任务一次并行设备验证 |

commit 锁、动态 publication 写回、三个初始计数 atomic、MIX 会合、真实依赖等待和 kernel 时间仍存在。
不把这些算入可消除项；缓存/总线压力改善若已包含在热路径实测中，不再次相加。

### 条件估算：低个位数百分比，中性预算约 3%

当前没有本目标方案的 L0 A/B，以下是**工程情景假设，不是实测、置信区间或保证上下限**。
设 F 为基线 L0 中落在关键路径上的目标静态开销占比，E 为这部分实际消除比例，
B 为并行验证增加的最长 bootstrap 路径占 L0 的比例，则 `收益率 ≈ F × E − B`。
B 取最长分区及门禁等待的增量，不能简单用总验证时间除以核数；Host 增量另外披露。
F/E 描述相关区间的耗时，不是功能覆盖比例；完整静态重算必须全部退出热路径，但必要的计划读取仍有成本。

| 情景 | 假设 F | 假设 E | 假设 B | L0 净收益估算 | 假设基线 1 ms 时 |
| ---- | ------ | ------ | ------ | ------------- | ---------------- |
| 保守 | 2% | 50% | 1% | 0% | 约 0 µs |
| 中性 | 5% | 70% | 0.5% | 3% | 约 30 µs |
| 偏乐观 | 8% | 80% | 0.5% | 5.9% | 约 59 µs |

这些 F/E/B 尚未测得，用于量级判断，不能作为 18 个 case 的预测结果。
以 E=70%、B=0.5% 为例，获得 1% 收益需要 F 至少约 2.14%；若目标开销很少或启动分区不均衡，也可能回退。
小任务多、admission 密集的流程更可能受益；长 kernel、少任务或真实等待主导的流程可能低于 1%。
实现不按这些场景选算法或阈值；只用完整 18 组 A/B 决定是否保留，1% 及更小的可信收益都如实统计。

### 历史 L3 只定位作用区间

SWA/lib 历史 t22→t29 的 kernel 间隔为 61.578 µs，其中 prepare=5.355 µs，
首个消费者 refresh=3.771 µs；后者所在 adopt+首次 claim 区间总计 7.766 µs，不能全部当成静态校验。
若只计这两个 scope 且极端地全部消失，直接减少也仅 9.126 µs；目标仍保留其中动态工作，
该计算只说明局部作用范围，不是整例 L0 收益或包含间接并发影响的严格上界。

证据：`01_swa__pypto_lib__candidate-two-queues__L3__rank0.json`，device 4、run 10、epoch 11；
分析位于工作区 `fwic/artifacts/swa-lib-t22-t29-20261111/`。
它早于当前 slot/profiling 更新；未归因 Scalar 时间不能全算可消除，原语发出时长也不是全部完成成本。

硬件约束参考工作区 `pypto-a5-dev/simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`：
保持 64 B 行所有权和 atomic/普通数据隔离；不把 DSB 当跨核会合，不从定向实验推导任意屏障可删。
`cache_preload_usage_guide.md` 的冷/热读差距只支持只读缓存的可能性，不可直接套算本模型收益。
保留[循环成员规则](fixed-prefix-members.md)，避免重复[已撤回的 admission 尝试](../../investigations/2026-11-a5-hbg-admission-shortcuts.md)。

## 5. 一体化实施与验收

### 改动位置及依赖顺序

下列顺序是实现依赖，不是可单独交付的缩减版本。所有项完成后才做完整候选的性能取舍。

| 文件 / 函数 | 必须完成的改动 |
| ----------- | -------------- |
| `scheduler_types.h`、布局规划和 `scheduler_init_spmd_task_records` | plan/publication 拆分、完整生成、size/offset/边界/初态检查 |
| `host/runtime_maker.cpp::create_scheduler_state` | native 展开后统一准备，正确 snapshot/upload 与每 run 生命周期 |
| 共用成员计算、`bootstrap_ready_graph` | 完整验证只在唯一分区 owner 执行，复用 scan gate，覆盖首波 fill |
| `scheduler_prepare_admission_record`、`scheduler_ordered_publish_pending/try_commit_admission` | 删除静态重算和写回，明确动态校验及 RETRY 证明范围 |
| `scheduler_refresh_local_obligation` | 使用只读计划和独立 publication，移除完整成员复核，保留动态身份与本核成员判断 |
| `scheduler_gm_mix.h`、dispatch、async 及所有 record accessor | 静态字段只读；动态 order 使用 publication，不改变业务协议 |
| `aicpu/aicore_lifecycle.cpp`、入口 cache 维护 | 核对拓扑/每核可见性证明先于计划验证和消费，不新增全任务初始化 |
| `host/scheduler_protocol_export.h`、DFX reader、夹具和协议文档 | 同步实际布局；JSON 若需 bitmap 地址从 offsetof 推导，不保留多余 wire 字段 |

冻结当前 main 的源码、构建、绑定、模型和参数，禁止把旧半成品作基线。
完成布局与全 writer/reader 迁移，再接通启动证明、移除热路径重算；静态审计和定向功能检查可随实现执行。
Host/AICPU/AICore 必须配套构建，更新实际受影响的协议/解析合同，不借机修改无关 ready 队列格式。
最终没有原布局兼容分支、场景排除开关或未接入的备用算法；已有未变化的动态协议继续复用。

### 最小但充分的功能与结构证据

复用现有 admission、ordered execution、GM MIX、startup、native Graph 和 async 用例，只补实际缺口：

| 必须证明 | 验收证据 |
| -------- | -------- |
| 每计划设备完整验证一次 | 各 node 分区不重叠；定向测试/DFX 检查调用位置和覆盖；不能只确认结果数值 |
| 门禁覆盖所有进入路径 | 后一分区验证失败时，已入 ready 的前一分区任务也不能进入首波 fill；使用明确握手，不靠睡眠 |
| 形状和实例正确 | 独立 bitmap 期望、全部 MIX mask、循环跨 word、同步容量边界、Graph 内 MIX+sync、多实例交错 |
| 稳态没有静态重算 | AIC/AIV 普通/native 生成代码中无对应 metadata 比较/成员构造循环；不只搜索被内联的函数名 |
| 重试及动态损坏仍被拒绝 | commit RETRY 后错误 phase/order/计数、非法 log ref、错误当前执行身份不能发布或执行 |
| 无合法静态写者及复用安全 | 所有 plan accessor/存储范围审计，同地址新 bind、epoch 边界、既有 replay 支持入口、失败退休 |
| 数值和关闭行为不变 | 实际 block 次数、sync 每位置一次、MIX release、defer_wait、Graph EXIT 和最终数值 |

静态篡改测试按第 3 节的新验证边界更新，动态故障保留原时点；不能隐藏检测能力的变化。
CPU 测试验证算法/状态机，真实 A5 验证可见性、共驻和数值；不以仿真代替实测。
L3 应能区分 bootstrap 静态验证与热路径动态校验，且仍显示相关 DCCI/DSB/atomic；不向 L0 热路径加入诊断计数。

### L0 性能口径与完整交付

- 使用当前构建配置，仅比较 HBG + AICore resident；两侧 kernel、输入、设备及运行配置一致。
  L0 设置 `enable_chip_swimlane=0`、`enable_dep_gen=false`，读取 `chip.run.runner_run.device_wall` 的设备起止周期。
  主指标为完整 AICPU entry→exit，包含新增 bootstrap 成本；不用 Host 墙钟、首末 kernel 间隔或 L1/L3 替代。
- 使用 `tests/fdwic/dsv41_scheduler` 全部 18 组，每侧 **1 次预热 + 10 次计时**。
  attention/Engram 单卡，原有 EP2 case 两卡；按 case 同卡成对比较，记录先后顺序和环境。
- L0 无泳道 raw 时，按 Host invocation 身份区分 reference、prefix、warmup 与被测 case，核对 level=0、resident mode 及数值。
  缺完整 device_wall 或样本不能唯一匹配时计时无效，不能沿用依赖 L1 raw epoch 的采集逻辑。
- 每项保留 10 个原始值、中位数、最小/最大值、绝对差和 `(baseline−candidate)/baseline` 收益率。
  没有固定死区；1% 收益和回退都展示。方向不稳定时标为待确认，针对相关 case 反向顺序复核，不无理由扩成 100 次。
- 综合考核为 14 组单卡中位数之和的前后变化，不是平均百分比，也不代表一次真实 forward。
  4 组 MoE/第 19–21 层保留数据与数值结果，通信波动不计综合收益；逐 rank 展示，并辅助报告逐轮 max-rank 的中位数。
- L1/L3 只诊断，必要时采 SWA/lib 的 bootstrap、prepare、commit、refresh 与 kernel 间隔。
  Host 准备、内存/H2D 字节和新增 bootstrap 成本单列；不能只报告中间计算阶段而隐去启动回退。
- 功能与结构目标全部满足且综合 L0 有可信收益才保留；逐项回退及原因仍披露。
  18 组不覆盖的 native/sync 形状用已有功能用例补齐，不新增模型专属性能考核。
  有收益的完整修改单独本地 commit；本轮不包含 push。交付版本/配置 hash、完整 diff、18 组原始 L0、诊断图及复现命令。

本方案的验收对象始终是完整目标机制。只有布局/Host 预填完成，或 L3 局部变短而 L0 无改善，都不算达到目标。
