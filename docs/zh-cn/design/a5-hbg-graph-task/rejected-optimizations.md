# 已否定的性能尝试

本文历史测量所引用的 JSON 仅保存在本地 `outputs/a5-hbg-graph-task/evidence-summary/`，不随仓库分发；下文保留文件名用于追溯。

独立 block 调度原型也已整体撤回：用户不接受 3.08 倍冷构建成本，当前代码恢复 `queue-ordinary-credit-v1`。
该思路没有被永久排除，保留为[后续待选方案](block-ready-scheduling.md)，含重启条件和历史证据。

本页记录 A5 HBG native Graph 实施中已经测量并撤回的候选，避免恢复上下文后重复采用。
新增的[固定前缀 SPMD 成员试验](fixed-prefix-members.md)单独记录工作集中导致的退化。
当前验收状态以[实施记录](../../../../tests/docs/a5-hbg-graph-task/README.zh-CN.md)为准。

## 将 native 元数据检查移出内联

`scheduler_native_node_checked` 是 native 节点读取前的边界校验入口。
Graph 支持使普通任务也经过包含 native 分支的 helper；曾尝试仅给该函数加 `noinline`，
将较大的校验主体移出调用方，降低调度代码体积。该候选标识为 `p1-outline`。

功能验证没有发现行为变化：170 个 C++ 目标、A5 sim 的 37 个 L2 与 28 个 standalone 目标通过，
真机 Graph 六个 pytest 目标通过，独立协议审计覆盖 15 份 capture、2,881 次 execution。
这些结果不能替代性能验证。

设备 7 的每项测量先校验 golden，再采集三批各 100 个稳态轮。
三个用例共 12 个功能/计时进程均核验 fresh resident 模式。
以下为 Device 中位数相对此前相同设备 `p1-opt` 测量的变化；两次测量不是交错 A/B/A，保留此限制。

| 用例 | Device 中位数变化 | Device p95 变化 |
| ---- | ----------------- | --------------- |
| batch paged attention / Case1 | −0.35% | +5.79% |
| fanout 1024 | +11.90% | +11.72% |
| AIV chain 64 | +13.31% | +12.88% |

候选对最近主线的 Device 中位数仍分别增加约 2.91%、16.41%、43.90%。
因此撤回该 `noinline` 改动，保留 `p1-outline/compare-opt.json`、`compare-baseline.json`
及冻结运行库作为失败证据。局部代码变小不等于延迟改善，也不能据此推断硬件 I-cache 容量。

只有新的调用边界或代码生成证据能说明开销发生了变化，才值得重新考虑该方向；
重试仍须使用完整功能与无诊断性能对照，不能用开启 profiling 的结果代替普通运行。

## 三个校验函数外提与完整 Qwen 复测

后续 `p2-native-outline` 对 header、template、node 三个 checked helper 同时加 `noinline`，
在已完成热循环图类型特化和 canonical ordinary view 的版本上复测，并补充此前没有的完整 Qwen 对照。
此次恢复时漏读了上节的单函数否定记录；不能将这个方向记为首次尝试。

任务 `task_20261102_125537_261804917999` 在设备 3 独占完成，exit=0。
Qwen 及五项无 Graph 均按旧 native → 候选 → 旧 native 执行，每臂三批各 100 稳态轮，
先验证完整 golden，并逐进程核验 fresh resident 模式。五项无 Graph 共 60 个功能/计时进程。
完整数值见 Qwen 配对（本地 `qwen-outline-performance.json`）和
无 Graph 配对（本地 `native-outline-focus.json`）。
这里的参照是此前候选 `p1-ordinary-view`，不是主线，五项定位集也不替代完整 19 项验收。

Qwen Device 中位数为 65.237 → 63.221 → 65.155 ms，候选对后臂降低 2.97%；
Host 降低 2.70%，bind 增加 5.67%。完整 out/K/V 通过。
无 Graph 对照的结果如下；参照 Device 前后漂移为 +0.10%～+1.01%，不能解释主要回退。

| 用例 | 后臂旧版（ms） | 候选（ms） | Device 变化 |
| ---- | -------------- | ---------- | ----------- |
| batch paged attention / Case1 | 3.124 | 3.177 | +1.69% |
| fanout 1024 | 1.953 | 2.190 | +12.13% |
| AIV chain 64 | 0.473 | 0.530 | +12.14% |
| mixed chain 1024 | 6.039 | 6.613 | +9.50% |
| mixed fanin32 1024 | 6.937 | 7.636 | +10.07% |

native 主循环符号从 580464/591348 bytes 降至 373076/392900 bytes，
普通路径却从 272164/282972 bytes 增至 374752/393856 bytes。
该证据说明编译器的内联决策影响了其他路径，不证明具体的 I-cache miss 或硬件容量。
93 项 C++、28 项独立 A5sim 和 37 项场景仿真通过；Qwen L4 原始完整 golden 通过，
独立协议审计覆盖 237088 个事件、11165 个设备节点、19208 次执行和 641 个 cooperative task，
见 验证摘要（本地 `qwen-outline-validation.json`）。

结论仍为撤回，三个 `noinline` 标注均已移除；不能用 Qwen 约 3% 的收益交换短任务退化。
冻结候选、源码差异、符号表、原始日志及所有计时样本保留在 `outputs/a5-hbg-graph-task/p2-native-outline/`。
后续候选须先读取本页；以不同调用边界重新试验时，明确区分新假设与这个已否定组合。

## 普通任务让行与重复扫描 {#ordinary-service-and-pool-probe}

任务 `task_20261103_122234_11015103109` 在设备 0 顺序验证两个独立候选，均保留 T/H8 队列：

- `queue-ordinary-service-v1`：本核无 pending/current obligation 且持有普通 executable candidate 时，
  让未提交的 cooperative candidate 延后一轮 discovery；下一轮必须尝试 admission。
  不持有 commit lock/log position，不改变普通任务 entry gate 或已提交 obligation 的优先级。
- `queue-direct-pool-probe-v1`：在 ordered fill 与普通 slot fill 中，省去全局 pool 领取前只用于生成
  `allow_steal` 的目录扫描；pool claim 本身不使用该参数，仍独立执行轮转、检查与领取 CAS。
  此候选不包含上述 admission 改动。

两者沿用固定 37-kernel callable、40 层、B16、seq_len=3500、seed=1234；各先做三轮完整 out/K/V，
再独立采 5 轮预热与 100 轮稳态，108 次执行均验证 fresh native mode=1。

| 候选 | Device 中位数 / p95（ms） | Host 中位数（ms） | 对此前同设备 T/H8 的 Device 变化 |
| ---- | ------------------------- | ----------------- | -------------------------------- |
| 普通任务让行 | 41.094 / 41.375 | 46.105 | +0.355% |
| 省去重复扫描 | 40.800 / 41.067 | 45.945 | −0.362% |

参照为此前设备 0 的 T/H8 40.948 ms，未在本任务重测；历史主线设备 3 的 29.478 ms 也未重测。
两个候选均未建立有用的收益，不能把约 0.36% 的变化解释成确定加速或回退；撤回两者并保留冻结源、运行库与逐轮数据。
原来的重复 payload publication 现象仍值得研究，但这次结果不支持仅用本地一轮让行解决主要等待。

各 65 项 A5 仿真、6 个 Graph 真机目标、15 份协议 capture（各 2,881 次执行）通过。
让行候选的六项新增 CPU 测试在修改前失败、修改后通过；最终 ordered target 共 122 项通过。
97 个相关 C++ 目标是在最后一次排除 inline candidate 的小修正之前通过，不能记作最终源码的全套验证；
重复扫描候选未单独跑 C++ 全套。两者均未运行完整故障矩阵或完整 19 项性能集。
完整身份、数据和验证日志哈希见候选证据（本地 `qwen-scheduler-candidates.json`）。

## 控制节点提前发布 ready 前缀 {#control-ready-prefix}

`queue-control-prefix8-v1` 基于 `queue-ordinary-credit-v1` 的普通任务领取额度策略，保留 T/H8。
控制节点服务原本在本轮路由结束后发布 executable batch；候选在同类 ready 积累到 8 项时提前发布，
尝试让其他核心执行前缀，同时当前核心继续路由。8 个控制节点、64 个 waiter 的服务预算，
continuation、完成计数的发布顺序及 kernel entry permit 均不改变。
L/C 后端仍可进入私有待发布链，不能把调用 batch push 等同于其他核心已经可见。

任务 `task_20261103_151326_41791814623` 在设备 0 完成固定 37-kernel callable 的完整 Qwen：
40 层、B16、seq_len=3500、seed=1234；三轮完整 out/K/V 通过，另采 5 轮预热、100 轮稳态。
108 次执行均核验 fresh native mode=1。Device 中位数/p95 为 **38.408/38.644 ms**，
Host 中位数 **43.625 ms**，bind 中位数 **1.481 ms**。
相对历史普通任务领取额度版本的 38.454 ms 仅低 **0.120%**；历史参照在设备 6，未在本任务重测。
该幅度不足以建立有用收益，因此撤回提前发布改动，保持 `queue-ordinary-credit-v1` 的运行行为；主线没有重新采集。

提前前缀回归在旧代码失败，跨三次服务的 130 waiter 链测试在两版均通过；最终 97 个相关 C++ 目标、
65 项 A5 仿真、6 个 Graph 真机目标、15 份协议 capture（2,881 次执行）全部通过。
未运行完整故障矩阵或完整 19 项性能集。冻结源码、运行库、撤回补丁和逐轮样本保留，
见本轮证据（本地 `qwen-control-prefix.json`）。

## Work 协议推进函数的内联拆分 {#work-protocol-outline}

`queue-work-outline-v1` 保留空闲检查，将复杂 MIX lease/cohort 推进标为 `noinline`。
生产函数 17 项连续 2,000 轮、AIC/AIV 编译探针、15 项模拟场景通过。
设备 6 的任务 `task_20261103_192056_54555513936` 中，MIX 用例出现非法 GM 地址访问，
随后多个用例报告运行通道已失败；Qwen 对照未执行，不能记录性能收益。

任务 `task_20261103_192542_72923325067` 使用独立进程复验，
`burst256_mask6` 的 L3 捕获仍失败，纯 vector 的 L3/L0 均通过。
ELF 检查确认两个推进函数在 AIC/AIV 对象中为同名弱符号，最终 ELF 只保留了 AIC 的两份实现。
这是明确的链接缺陷：这类架构相关函数需要内部链接，不能跨 AIC/AIV 合并。

为检验归因，另行复制失败产物，仅按架构重命名、局部化两个函数并重复链接，
得到 `queue-work-outline-local-v1`，最终 ELF 保留四个 LOCAL 函数。
任务 `task_20261103_193450_10324693829` 的独立 MIX 复验仍失败，vector L3/L0 通过。
因此弱符号合并还不是经验证的完整解释；重链接也不等同于重新编译 `static` 源码。
`static` 源码通过了 CPU 压力与编译探针，但完整构建在本轮撤回时终止，未取得真机结论。

代码体积也需区分包装函数和真正的调度循环：DFX 版本的 native resident 包装函数
约 200/221 KB，拆分后约 79/82 KB；实际 `run_ready_dispatch_loop` 仅缩小约 5–11%，
整个 `.text` 从 12,570,392 B 增至 13,450,520 B。不能由包装函数变小推断取指收益。

本轮撤回全部内联拆分改动，保留失败源码、对象、转换步骤和日志，
只将有独立回归证据的跨逻辑任务发布预算修正进入下一候选。
该内联试验既未通过真机功能验收，也没有 Qwen 性能数据。

## 缩短 idle 退避 {#idle-backoff}

基于前向依赖计数版 `d9d0cf3e8`，只把 resident loop 的最大 idle NOP 次数从 1024 改为 64，
检验更频繁的 ready 发现是否降低调度延迟。任务 `task_20261109_222758_383387026056`
在设备 0 的同一次锁内按候选、冻结对照顺序执行；各 5 轮预热、50 轮稳态，输入与 37 个 kernel 相同。

| 指标（ms） | 1024-NOP 对照 | 64-NOP 候选 |
| ---------- | ------------- | ----------- |
| Device 中位数 | 36.007848 | 36.180286 |
| Device 均值 | 36.016596 | 36.186801 |
| Device 最小值–最大值 | 35.821683–36.289257 | 36.000535–36.449640 |

候选完整 out/K/V 校验通过。独立 L1 采样的首 kernel 到末 kernel 区间为 37.254962 ms，
逐执行关键路径 kernel 为 16.314651 ms、gap 为 20.940311 ms；这些不是无 DFX 稳态时间。
对照本轮只复测性能，完整数值验证来自相同冻结二进制的前一轮。

该测量未建立收益，因此恢复 1024。不能由此断言 idle 退避没有成本，
也不能把更多轮询必然带来更短延迟作为后续设计前提。源码差异、输入和二进制身份、逐轮日志
保存在本地 `outputs/a5-hbg-schedule-8ms/{idle64,counters-control}/`。

## ordered fill 的编译边界 {#ordered-fill-outline}

2026-11-10 基于 `92c42a64f`，为 `scheduler_fill_ordered_slots` 加 `static noinline`，
尝试避免重型逻辑同时内联进 bootstrap 和稳态循环。正常 AICore 构建从 167.590 秒降至
73.819 秒；完整原版构建为 168.115 秒。并发上限每目标 32，慢的是单元自身而非并行槽不足。

仅 `resident_native_profiled.cpp` 的 AIC 单元进行编译器专项采样，wall 为 142.064→76.873 秒。
Frontend 均约 1.446 秒，Backend 为 140.518→75.338 秒，其中 Optimizer 为 76.070→36.057 秒，
CodeGenPasses 为 64.410→39.258 秒。LICM、GVN 和 InstCombine 是主要耗时之一。
单元 `.text` 从 745904 降至 501016 bytes，主循环从 389620 降至 153412 bytes，
bootstrap 从 288916 降至 84528 bytes，新增独立 fill 函数为 195708 bytes。
父 pass 与子 pass 的计时重叠，不能直接相加；也没有据此证明设备 I-cache 收益。

该候选未通过硬件功能验证。设备 0 的一般 L2 场景通过，但带诊断的同步双 AIV 场景
`test_cohort_blocks_land_on_distinct_clusters` 报 507018。任务
`task_20261110_084841_360515418731` 因此没有执行 Qwen 性能阶段。
随后同锁隔离复验中，冻结 range 版本通过，outline 版本再次失败。CANN 报 AI Core timeout/trap，
尚无足够证据把它判为特定调度死锁或编译器缺陷。

上述初验和隔离复验均使用设备 0。该设备随后在未修改代码的 PR #2975 复现中出现
可用 core 超时和 `rtSetDevice(0)` 初始化失败，因此这些历史结果存在设备状态混杂。
原版通过、候选失败仍是已观察到的差异，但尚无健康设备复验，不能据此确认候选代码的故障根因；
也不能仅凭设备 0 异常就认定候选正确。后续若重启此路线，须在同一健康设备上重新对照。

八个 fill 实例均为 LOCAL；对应 AIC/AIV 对象没有共享外部函数名，未发现历史弱符号合并问题。
另将 SSBUF 参数改为 uint64 传递后再恢复指针，构建为 73.232 秒，同一场景仍失败。
两臂 `.text` 和 `.rodata` 逐字节相同，因此这个源层参数修改没有形成新的机器码试验，
不能用它排除或确认调用 ABI 根因。后续候选应先核对代码身份，再决定是否需要重复上板。

用户同意停止这条构建提速路线；正式代码保留验证通过的 inline 版本。
只有新的安全编译边界或明确故障证据才值得重试，不得以编译提速交换 MIX/sync/DFX 功能。
原始命令、编译器 trace、阶段统计和失败日志保存在本地
`outputs/a5-hbg-schedule-8ms/{compile-analysis,outline-fill,outline-fill-u64}/`，不随仓库分发。

<!-- markdownlint-disable-next-line MD033 -->
<a id="scalar-observer-inlining"></a>

## 强制内联 Scalar 采集包装

2026-11-11，将 `scheduler_scalar_memory_issue` 和 `scheduler_scalar_memory_result`
从 `noinline` 改成 `always_inline`，保留所有 primitive、时间戳及记录，尝试减少包装调用开销。
参照为已包含 completion 直接回收及先解依赖后写 trace 的 `release-first` 版本。

相同 `-O3 -g`、DFX 和 16 并发下，完整 AICore 冷编译链接从 257.749 秒增至 1031.377 秒。
普通 profiled AIV 对象中的 resolver 符号从 43652 增至 108792 bytes；静态代码体积不能证明 I-cache 开销。
SWA 三轮 L3 测量的 RoPE completion 中位数从 9.728 变为 9.784 µs，
kernel 结束到 completion 返回的中位数从 11.471 变为 11.390 µs。
参照使用设备 1，候选使用设备 5，未形成严格同卡 A/B，未建立明确性能收益。

用户要求撤回该实验，当时两个包装恢复 `noinline`，此前的 completion 修改保留。
当前主线的编译期 Scalar policy 使用 `always_inline` 的薄入口；启用记录时，入口调用
`noinline` 的 `scheduler_scalar_memory_issue_runtime` / `scheduler_scalar_memory_result_runtime`。
实际采集仍保留独立编译边界，本节试验并不等同于当前薄入口的内联。
后续优先减少 completion/依赖解析实际路径的冗余判断、重复读取和状态转换，
以实际执行指令和 L3 耗时验证；不得将包装函数内联本身视为指令数减少。
原始结果和产物身份保存在本地 `outputs/swa-current-20261111/rebased-full-l3/`，
对照入口为 `completion-comparison.json`，不随仓库分发。

<!-- markdownlint-disable-next-line MD033 -->
<a id="scalar-primitive-constants"></a>

## 内存原语记录的常量参数特化

2026-11-11，将内存原语的 kind、固定 task_ref=-1 和 count=1 改为编译期参数，
保留既有 noinline 边界、全部 L3 时间戳、记录和内存屏障。
普通 profiled AIV 生产对象中，四类代表性包装各减少 6 条静态指令，
每种原语的记录提交函数为 70 条，原通用提交函数为 78 条。
新增模板实例使该对象总静态指令数从 295424 增至 296232。

相同编译选项、14 个对象和 16 并发的完整冷构建为 337.925→402.551 秒。
该对照未控制共享机器负载，不能把全部编译时间差归因于新增模板。
两版均在设备 3 预热一次、测量三次，全部 L3 记录完整：

- 最后必要前驱 kernel 结束→resolver 返回：原版 4.747 / 4.670 / 4.705 µs；候选 5.414 / 6.596 / 5.157 µs。
- 最后必要前驱 kernel 结束→completion 返回：原版 7.046 / 6.878 / 6.660 µs；候选 7.557 / 8.706 / 7.220 µs。
- RoPE 结束→attention 开始：原版 93.413 / 94.160 / 98.510 µs；候选 97.272 / 119.809 / 107.127 µs。

未建立性能收益，候选已撤回。静态局部指令减少不等于实际关键路径缩短；
这些数据也不能单独证明具体的 I-cache 或原子争用根因。
原始证据位于本地 `outputs/swa-current-20261111/primitive-constant-metrics.json`、
`primitive-constant-production-instructions.json` 和 `runtime-completion-primitive-constant/`。

<!-- markdownlint-disable-next-line MD033 -->
<a id="direct-executor-record"></a>

## 直接写执行记录与提前保留候选组合

2026-11-11 的 `direct-record` 候选将临时 executor trace 改为直接写本核槽位，
并在同步 CSR resolver 中提前保留直接后继，跳过其 Ready 队列地址计算。
功能和记录检查通过，但没有建立性能收益。设备 5 上三轮 L3 的最后必要前驱
kernel 结束到 resolver 返回为 7.255、6.766、6.994 µs；保留的 `seal-fence`
版本在设备 6 上为 6.103、5.829、6.607 µs。两组设备不同，不能据此确定
代码回退的因果来源，也不能把删减源码操作当作性能改善。

按用户要求撤回该组合，运行时源码逐文件恢复为冻结的 `seal-fence` 版本。
后续分析暂缓；原始数据、二进制及回退哈希核对保存在本地
`outputs/swa-current-20261111/`，不随仓库分发。

## Disabled Scalar observer branches

2026-11-13，针对 L1 没有启用 Scalar 记录、却仍进入 `noinline` 记录包装函数的问题，
候选在 `scheduler_scalar_memory_issue/result()` 和 `scheduler_scalar_result()`
外层判断 trace/buffer；关闭时直接执行 operation，开启时沿用原记录函数。
没有修改业务原子、DCCI/DSB、SSBUF 或 kernel。该候选已撤回，未提交生产修改。

同一 A5 自动分配中的 SWA PyPTO-lib 对照，L0/L1 各预热 1 次、测量 10 次：

| 指标 | 修改前 | 候选 | 结果 |
| ---- | -----: | ---: | ---- |
| 完整 device L0 均值 / ms | 0.7986280 | 0.7961422 | 不归为业务收益：四个非 profiling 执行器的代码、只读数据与重定位相同 |
| L1 device 均值 / ms | 1.0048495 | 1.0325257 | 增加 2.75% |
| L3 单次 device / ms | 1.317421 | 1.373917 | 单次增加 4.29%，不用于稳定收益判断 |
| A5 HBG 完整构建 / s | 284.43 | 742.66 | 增加至 2.61 倍 |

18 项既有记录器检查、14 个 A5 IR 编译单元检查和上述实机数值检查通过。
四个不带 profiling 的 AIC/AIV ordinary/native 对象，其所有分配节及按符号解析的
重定位逐项一致；IR 仅有调试位置元数据差异。原始提交记录的所有 task timing slot
均为 -1，确认本用例 L0 选择非 profiling 入口。这不保证不同链接布局或测量噪声为零，
但没有可归因于本候选的 L0 调度指令收益。

例如 ordinary-profiled AIC 的优化 IR，静态指令行数由 177,027 增至 238,766，
基本块标签由 30,887 增至 54,560。运行时分支同时保留原操作与记录路径，确实扩大了
编译器处理的控制流；不能据此宣称已经定位 CCEC 编译耗时或运行延迟的完整根因。
本轮使用 CCEC clang 15.0.5（2026-09-22 构建），按当前 A5 onboard 参数编译。

撤回理由是没有降低测量扰动且显著增加编译时间。后续应在编译单元层面区分
任务计时与 Scalar 细节采集，避免每个访存原语都加入一条动态选择分支；该替代方案
仍须独立验证，不能沿用本候选的通过结果。

可复现证据位于 `artifacts/five-us-swa-20261113/`：
`observer-fast/rejected-wrapper-bypass.patch`、`build.log`、`ir/`、
`plain-object-comparison.json`、`plain-ir-comparison.json`、`profiled-ir-size.json`，
以及 `perf-observer-fast/results.json` 和 `comparison-summary.json`。
两臂复用冻结的 callable、输入和 runner，实际分配为 device 4。

## Oldest-ready local pending order

2026-11-14，以 `c1d0f3f8668` 为基线，尝试让普通 pending 按本核持有先后扫描，
跳过未就绪项，优先执行已就绪的旧项。四个 token 的物理索引存于一个 32-bit
排列，替换原轮转 cursor；fresh/PARKED 接手排末尾，READY slot 回收排优先位置。
没有模型名、task 编号、形状或 kernel 时长特判，没有新增共享原子或改动业务 kernel。
候选已撤回，生产代码仍保留原选序。

动机来自 C2A reuse/lib 的 L3：AIV34 早于 t20 取得 t8；t8 的共同计算前驱完成后，
该核仍先进入约 469 µs 的 t20，再执行 t8。最小回归确认原轮转会作出这种选择，
候选能改变它。它是合法执行之间的性能策略，并非数值错误；单个尾部被改善不保证整图更快。

完整 18 组使用相同 callable、输入、HBG AICore 运行方式，分别预热 1 次、测量 10 次 L0。
14 个非通信用例在同一自动分配的单卡上按 case 交替两臂顺序；4 个通信 case 使用
同一自动分配的双卡，每轮取 rank 最大值，仅报告、不参与性能门槛。

| 非通信 L0 均值总和 / ms | 基线 | 候选 | 变化 |
| ----------------------- | ---: | ---: | ---: |
| PyPTO-lib，7 组 | 9.7079507 | 9.7698664 | +0.638% |
| DSL，7 组 | 11.0015004 | 10.9602499 | -0.375% |
| 全部，14 组 | 20.7094511 | 20.7301163 | +0.100% |

各 case 比值的几何均值变化为 -0.014%，也未给出实质改善证据。
SWA 两种实现分别改善约 1.51%/1.04%，但 C1A full/lib、C1A full/DSL
分别回退约 2.80%/3.13%；不能依据 SWA 选择性保留。所有 18 组数值校验通过。
另外 4 个 C++ 套件和 17 项 A5 Graph/MIX/sync_start/defer_wait/prepared-call
实机回归通过，2 项只支持模拟器的检查跳过。

独立 C2A reuse/lib 对照的 L0 为 1281.711 → 1285.558 µs，L3 单次却为
1547.288 → 1410.629 µs。L3 改善不能替代 L0 收益；候选改变分支的执行重叠，
并不能解决下一个任务 ready 时全部相关核已经执行长 kernel 的等待。

另采 L1，将下一项所有可追溯计算前驱均已在前一 kernel 返回前结束的同核相邻
执行单独列出：基线 55 个窗口的最小值/中位数为 5.770/7.715 µs；候选 52 个窗口
为 5.798/7.882 µs。这不证明已经预持有 token 或满足全部资源/异步许可，不能作为
完整快入口验收；也没有证实小于 5 µs 的目标。后续应继续审查已有 pending 的
固定完成、转交和实际入口开销，而非只更改合法候选的优先顺序。

本地可复现材料位于 `artifacts/execution-token-implementation/`：
`pending-order/experiment.patch`、`pending-order-snapshot/`、
`pending-order-18/{configuration,matrix,comparison}.json` 和 `report.md`，
以及 `c2a-reuse-mix-priority/`、`c2a-reuse-pending-order/`、`c2a-reuse-handoff-l1/`。
配置保存 bundle、runner 和 runtime SHA256，运行日志保存实际 `TASK_DEVICE`。
A5 构建耗时 290.87 秒；IR 扫描 15/16 个 TU 零 findings，未解码的 vendor SIMT
wrapper 已逐项核对分配段与 canonical relocations，不能称为完整 clean scan。

## Completed catalog prefix retirement

2026-11-14，以 `c1d0f3f8668` 为基线，尝试让普通、MIX、cooperative 发现共用
每 worker 私有的已完成目录前缀，跳过终态任务的新票确认和 PARKED 探测。
只有 canonical completion gate 的计算剩余量和异步债务均归零且格式有效，
并经原子确认后才推进前缀；取票耗尽不是退休依据。目录在运行中不可变，
前缀每次运行重新初始化，不替代 token、SSBUF、admission 或 async 退休。
这是通用状态策略，没有模型、task 编号或形状特判；候选仍因收益证据不足撤回。

完整 18 组保持原 kernel、输入、HBG AICore 方式，各臂 1 次预热、10 次 L0 计时。
非通信两臂使用同一自动分配的单卡，按 case 交替先后；通信使用同一自动分配的
双卡，每轮取 rank 最大值，只报告、不参与保留门槛。全部数值校验通过。

| 非通信 L0 均值总和 / ms | 基线 | 候选 | 变化 |
| ----------------------- | ---: | ---: | ---: |
| PyPTO-lib，7 组 | 9.7442054 | 9.6351154 | -1.120% |
| DSL，7 组 | 11.0266336 | 11.1173801 | +0.823% |
| 全部，14 组 | 20.7708390 | 20.7524955 | -0.088% |

各 case 比值的几何均值变化为 -0.071%。保留这些带符号结果，不使用 ±5%
忽略区间，但本轮尚不能确认综合收益。对现有十轮样本按组、按臂独立重采样，
总和比值变化的 95% bootstrap 区间为 [-1.239%, +1.081%]；这还未覆盖运行顺序
漂移或跨轮相关性。C1A full/lib 改善 9.00%，同时 C1A reindex/lib 回退 4.07%，
C1A reuse/DSL 回退 2.64%，不能只选择受益项留下实现。

独立 C2A reuse/lib 的 L0 为 1317.305 → 1269.680 µs，L3 却为
1423.400 → 1564.787 µs。L3 的全核 atomicAdd 次数 22,778 → 11,505，
refill 的累计 inclusive worker-time 下降 26.77%，但 ld_dev 次数增加，
新的完成确认仍形成同址查询热点。例如一次 28.231 µs 的 probe 中，
ld_dev 占 23.947 µs、atomicAdd 占 3.500 µs；不能把父子 scope 相加成墙钟时间。
这一现象与 `atomic_probe/ATOMIC_USAGE_GUIDE.md` §4.6 的同址竞争风险相符，
但不是将其受控微基准延迟直接套入业务的证明。

L1 中筛选两侧均为普通任务、下一任务的计算前驱已在前一 kernel 返回前结束的
跨任务窗口：基线 47 个，最小/中位 6.147/7.306 µs；候选 45 个，6.033/7.385 µs。
这些样本不证明预持有 token 或全部异步/资源许可，不能作为完整快入口验收，
也没有小于 5 µs 的交接证据。重开此方向需要减少完成确认热点或固定交接成本的
新机制，以及完整矩阵收益；只减少扫描次数不足以推断成功。

4 个 C++ 调度套件、61 项 Python DFX 检查及 17 项选定 A5 特性检查通过；
2 项 sim-only 检查跳过。构建 328.90 秒；IR 扫描 15/16 TU 零 findings，
未解码 wrapper 的分配段和重定位与前版相同，不是完整 clean scan。
本地证据位于 `artifacts/execution-token-implementation/` 下的
`catalog-retirement/`、`catalog-retirement-snapshot/`、`catalog-retirement-18/`
与 `c2a-reuse-catalog-retirement/`。保存了补丁、构建来源、逐轮数据、18 组表格、
不确定性计算脚本、L1/L3 及恢复证明；生产源码恢复为基线。

## 普通续领前移到快速 token 推进 {#fast-ordinary-continuation}

2026-11-14，以 `fc53cc96be4` 为基线，把已有普通 SPMD fresh block 续领从
完整 fill 路径移入 `scheduler_progress_execution_tokens`。已有 ready token
优先、三次续领后访问目录、resident 每四轮服务边界和正式 MIX 责任均保留；
不按模型、task ID 或 shape 区分。

全部 18 组两臂数值通过，14 组非通信 L0 均值之和为
**20.4091776 → 20.4196695 ms，+0.0514%**；lib −0.6299%，DSL +0.6475%。
每臂预热一次、计时十次，各 case 同卡对照、交替先后顺序，自动分配设备。
通信按每轮较慢 rank 再求均值，四组结果均记录但不参与保留门槛。
已有十轮数据的组内 bootstrap 区间为 [−1.072%, +1.196%]，不包含顺序漂移。
没有使用百分比免考核区间；未建立综合收益，因此撤回。

Engram/lib 的同 task 普通 block 间隙 L3 中位数 24.974 → 19.390 µs，
500 block 执行窗口 207.390 → 175.133 µs，参与核数均为 64、每核最多 8 次。
现有 task-timing 低插桩采样中，间隙中位数 14.013 → 12.473 µs；
这不是无插桩间隙，也未达到 5 µs。局部收益不能替代完整 L0 的验收。

四组 C++ 套件、17 项 A5 特性用例通过，另有两项 sim-only 跳过。
候选构建 322.13 秒；IR 15/16 个 TU 零 findings，未解码 vendor wrapper
按分配段及重定位与已验证版本对照，不称为完整 clean scan。
首次单例候选在设备初始化报 507033，未执行调度；原始失败记录保留，
后续通过不表示原初始化问题已归因。

本地证据：`artifacts/execution-token-implementation/fast-continuation-18/`
的 `report.md`、`comparison.json`、逐轮日志，`fast-continuation/` 的撤回补丁，
`fast-continuation-snapshot/` 的冻结运行库，以及 `engram-fast-continuation/`
和 `handoff-audit/fast-host-timing-export/engram/` 的诊断数据。
只在新的服务边界或代码生成证据解释了整体成本后重开，不通过单例反复试参保留。

## 不可变 Call 接收缓存复用 {#immutable-call-cache-reuse}

2026-11-14，以 `fc53cc96be4` 为基线，候选仅删除
`scheduler_receive_execution_token` 对 512B Call 的八次定向 DCCI。
可变 Binding 的两次 DCCI、尾 DSB、compiler barrier、Claim、run/单元身份
校验、最终入口许可和 async 业务数据观察全部保留，没有模型或形状特判。

当前 Call 在 Host 发布后才启动 resident；每次启动的 DCI 先于 Call 读取，
run 内没有 Device Build 或地址复用。旧 `aabab11ec` 的失败发生在启动后
Device builder 写 Call 的不同发布顺序，不能直接套用，也不因此撤销旧结论。
另见[静态 Binding 实验的合同说明](../../investigations/2026-11-a5-prepared-binding-publication.md)。

全部 18 组配对数值通过。每臂一次预热、十次 L0 计时，按 case 交替顺序；
单卡 lib/DSL 分别自动分配设备 4/5，通信自动分配设备 0、3，按每轮较慢 rank
再求均值。14 组非通信均值之和 **20.4800550 → 20.5771805 ms，+0.4742%**；
lib −0.0588%，DSL +0.9413%，几何平均时间比 +0.4885%。
组内 bootstrap 区间 [−0.826%, +1.803%] 不包含顺序及系统状态漂移。
本次未建立综合收益，生产代码与统一设计恢复原定向接收合同。

四组 C++ 套件与 17 项 A5 Graph/MIX/sync_start/defer_wait/prepared-call
检查通过，另两项 sim-only 跳过；构建 354.76 秒，IR 15/16 个 TU 零 findings，
vendor wrapper 保留已审查的解码例外。不能由 CPU 或 IR 结果替代缓存可见性实测。

C2A full 与 Engram 同卡另采 L1/L2/L3。L3 中被记录的 Call 接收区间缩短，
但候选大量区间已落到 1 µs 过滤阈值以下，不能把剩余样本的中位数当作全体。
L2 的 Engram 同 task 间隙均值为 15.298 → 15.369 µs，其中完成记录到下一项
发布仍约 11.2 µs；少量 cache 指令并未消除这段补给链路。L2 本身有记录开销，
不是无插桩延迟；不同运行的所有权不能与这些时间拼成同一次执行的证据。

本地证据位于 `artifacts/execution-token-implementation/`：
`immutable-call-cache-18/` 的完整配对数据，`immutable-call-cache/` 的缓存
合同审查、撤回补丁、CPU/IR/特性结果及 L2 分析，`immutable-call-cache-snapshot/`
的冻结运行库，以及 `c2a-full-immutable-call-cache/`、
`engram-immutable-call-cache/` 的原始记录和转换泳道图。
