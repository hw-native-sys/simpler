# A5 HBG SPMD 实现过程记录

## 目标、依据和验收边界

开始日期：2026-10-25；起点：`6c367c69f`；工作分支：`feat/fully_distributed_within_core`。
按用户要求逐步实现、阶段性详细中文 commit，**不 push**。
用户随后单独授权的推送已完成至 `98215088b`；后续实现继续遵守不自动 push 的原目标。
本文件集中记录实现、失败定位、测试证据和剩余工作，不能用阶段完成替代整个目标完成。

依据：

- [SPMD 总体设计](../a5-hbg-spmd-design.md)。
- [per-core 历史方案](../a5-hbg-per-core-scheduler-plan.md)：仅复用已实现基础，不采用其中 SSBUF 调度和 AIV 代管方案。
- [编码契约](implementation-contract.md)：接线及边界的主要依据。

保持 Host 构图与预分配、完成驱动依赖推进、AIC/AIV 每核自主调度。
Mix 保留 AIC 有限协调，各执行 lane 自备参数，控制通信使用 GM。
顺序只约束参与者：本核关闭 A 后可进入独立 B，不等待 A 的全局完成；非参与者可以旁路。
普通多 block 允许分波；sync_start 保证不同执行位置和最终参与，不添加全员同时开始屏障。
Graph、异步返回、多 block predicate 不自动开放；既有单 block predicate 必须回归。
2026-10-26 用户确认先做边界验收，性能优化后续自行处理；保留公共 N=1 的归一化行为。
下文历史“待确认”与早期反向预期记录保留原貌，以本节和最新验收记录为准。

## 分步交付与状态

| 顺序 | 交付 | 出口证据 | 当前状态 |
| ---- | ---- | -------- | -------- |
| 前置 | ordinary／Mix payload context 初始化与重绑定 | 原生 N=1 红灯转绿；真实初始化函数的 C++ 测试 | 已通过 C++、仿真与 A5 六种 mask 各十轮 |
| 1 | 有界存储、准入记录、block 控制、GM 组状态 | POD／布局／溢出／重跑初始化测试 | 已落地；纯 C/V 和 GM Mix 已使用 |
| 2 | 一次 ready 领取、共同准入、三入口统一顺序门禁 | 发布竞态、B 可先准入、非成员旁路、无重复入队 | own/direct、持久候选与 GM Mix 均已接线；完整原生审计通过；B 先于仍等前驱的 A 已有仿真／A5 各 84 项原生证据 |
| 3 | 纯 C/V 多 block、完成聚合、分波容量 | N=1／容量两侧／512；身份不重不漏；独立任务可重叠 | A5／仿真功能和 L1/L4 逐 block 检查通过 |
| 4 | GM Mix descriptor／ack／seal、各 lane 自备 payload | mask 3/5/6/7；最后发布者暂停；代次／槽复用 | A5／仿真矩阵、重叠及 TPipe PA 已通过；首错误身份与冷日志已接线 |
| 5 | 完整 native 身份 trace 和原样 CSA resident | 七项 golden；2308 执行身份；来源严格门禁 | CSA 数值／2308 显式身份／来源、全协议因果与独立图门禁通过 |
| 6 | sync_start 不同位置领取和局部跨任务推进 | 两独立满核任务、部分交叠、超容量、早返回不重领 | N>1 已实现；仿真／A5 各 47 项基础、112 项异构交叠／依赖／分波切换及原生审计通过；N=1 公共归一化已确认，新增验收见末节 |
| 7 | 性能对比 | 同 kernel／输入／拓扑，无完整插桩运行；诊断运行分开 | 初次 Device 为设计基线 11.41 倍；共享错误 RMW 已有显著归因差分，GM service 阶段裁剪相对同批 resident 对照降低 6.58%，整体回退未消除 |

退出、错误收敛、epoch／generation、Dummy 服务节点随功能交付，不延后处理。
CPU 模型、原生探针、真实 CSA 各有独立证明范围；legacy 的正确结果不能证明 resident 功能。

## 2026-10-25：起点核对与 payload 前置修复

### 当前代码事实

- 分支与起点核对无误；未跟踪的 atomic_probe、TMR、原始 CLA 文件保持原样。
- `host/runtime_maker.cpp::create_scheduler_state` 只遍历普通 dispatch slots，
  填写两个 context 后缀；`layout.mix_payloads_offset` 对应的独立区没有相同初始化。
- `scheduler_init_data_from_layout` 已负责 retained state 的逐轮清零，适合作为统一初始化入口；
  要使用设备基址生成指针，不能把 Host 镜像地址上传给 AICore。
- 形状门禁仍只接收 N=1 且非 sync_start；本修复不开放多 block。
- 本机 CANN 查询确认 Ascend950PR_958b、32 AIC、64 AIV、2 DIE。
  `npu-smi`、`task-submit` 不在 PATH；按用户明确要求直接使用本机设备，运行没有队列锁隔离。

### 验证记录

- 本地 `.venv-csa`，Torch 2.12，设置 `CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8`。
  后续核对发现 runtime 编译器显式传入 `--parallel min(cpu_count, 32)`，
  且多 runtime 并发构建；因此环境变量并不代表全流程实际总并发上限为 8。
  基线与修改后 editable 构建均通过：`outputs/spmd-build-baseline-20261025.log`、
  `outputs/spmd-payload-build-20261025.log`。
- 红灯：`mask3_n1` a5sim 报 `SPMD LocalContext/GlobalContext missing pointer`，
  两个 task 的保护记录各计数 1；日志 `outputs/spmd-mix-context-sim-red-20261025.log`。
  公共初始化函数的两条新增 C++ 测试也先失败（该函数原来只清零，普通槽另在 maker 初始化），
  日志 `outputs/spmd-payload-cpp-red-20261025.log`。
- 生产修改：`scheduler_init_data_from_layout` 显式接收设备基址，
  为全部 ordinary／Mix payload 初始化两项后缀、N=1、无效 async token；
  保留原有全状态清零，拒绝零地址、未对齐地址及地址范围溢出。
  maker 删除重复的 ordinary 初始化并传入实际设备基址；不改 kernel ABI 和 shape gate。
- 新增 5 个 C++ 用例：两类全槽初始化、Host/Device 地址分离、污染后重绑定、无效地址拒绝。
  加既有 106 个 C++ case，共 111 个通过；日志 `outputs/spmd-payload-cpp-green-20261025.log`。
- 六种 mask 的 N=1 原生仿真各三轮，共 18 次通过；
  保留逐轮 golden、保护区与新鲜 resident mode 检查，未改原 kernel。
  命令为 `python tests/fdwic/pa_hbg/spmd/test_spmd_contract.py -p a5sim`
  加六个 `--case mask{1,2,3,5,6,7}_n1` 参数及 `--rounds 3`；
  日志 `outputs/spmd-payload-sim-green-20261025.log`。
- 既有 SPMD／CLA／CSA Python 检查 1021 项通过，
  日志 `outputs/spmd-payload-python-20261025.log`。
- 修改前真机对照 `single_aic_root` 失败，日志与设备日志均保留于
  `outputs/spmd-onboard-baseline-20261025/`：site=101 及 bootstrap timeout，
  runtime 随后确认设备恢复。不是只在新增 Mix 探针出现，不归因于 context 修复。
- HBG 仿真回归：`pytest examples tests/st --platform a5sim --runtime host_build_graph`
  加 `--device 0-7 --max-parallel 8 --pto-session-timeout 600 -q`，
  42 个 pytest 测试通过、379 项因平台／runtime 选择未运行；耗时 147.85 秒。
  这是默认非 manual 范围，不冒充所有参数场景或真机全范围验收。
  日志 `outputs/spmd-payload-a5sim-regression-20261025.log`。
- 修改后真机 `mask3_n1 --rounds 3` 在首轮仍以 site=101 失败，后两轮未执行，
  设备恢复已由 runtime 确认。日志 `outputs/spmd-payload-onboard-20261025/`。
  目前只证明 context 缺口修复，不证明真机 resident 基线或 SPMD 已实现。
- 提交检查首次发现本地缺少 clang-tidy；安装默认 23 后触发旧代码大量
  signed-bitwise 新规则。按 `.github/workflows/_pre-commit.yml` 的 CI 工具选择
  对齐为项目 venv 中的 clang-tidy 18.1.8，全部适用钩子通过，未增加 suppress 或跳过检查。
  最终日志 `outputs/spmd-payload-precommit-r3-20261025.log`。

### 后续接线重点

TaskControl 原子区可复用 40B padding 中的 24B；禁止普通写回覆盖多写者原子字段。
不能以 generic pop 的旧 endpoints 作为“仅摘除预期 task”的保证。
`TaskDone` 与 `LocalDone` 分离，Mix cursor 耗尽与 cluster seal 分离。
前置 context 修复独立提交。下一项先补 bootstrap 超时的精确状态证据并修复真机基线，
同时按上述顺序推进实际存储与执行路径，不只改模型或放宽验收。

## 2026-10-25：resident bootstrap 真机基线调查与修复

前置修复已提交为 `0a6460982`，未 push。
当前完整目标仍未达到：共同准入、多 block、GM Mix、原样 CSA resident、sync_start 和性能验收均未完成。

已有设备日志仅能证明 AICPU 等 bootstrap 结束超时，不能区分卡在扫描注册还是首波 dispatch。
在 `aicore_lifecycle.cpp::wait_bootstrap_complete` 的错误／超时冷路径补充
worker 数、scheduler 数、scan arrived/complete、dispatch arrived/complete 的一次性快照。
不改变调度路径，不加环境开关，不在热循环输出日志；据真机快照确定下一步。
诊断构建日志：`outputs/spmd-bootstrap-diagnostic-build-20261025.log`。

首次状态快照来自 `outputs/spmd-bootstrap-state-20261025/`：
32 AIC／64 AIV 已完成发现，失败时 `workers=96 schedulers=96 scan_arrived=96`
且 `scan_complete=1 dispatch_arrived=96 bootstrap_complete=1`。
AICPU 非 leader 线程在约 20 秒超时，leader 约 37.4 秒后才打印快照并退出。
这不能直接证明“某个核从未到达”，也不能仅凭最终值证明它何时可见。
下一次诊断仅增加 bootstrap poll 的调用次数、单次 cache observe 最大耗时与总耗时，
区分观察阻塞与真正晚到；诊断计时不作为最终性能实现保留。

第二次加入诊断计时、仍保留原 cache observe 的单根用例偶然通过，
但设备执行约 13.625 秒（`outputs/spmd-bootstrap-observe-20261025/`），
并非稳定基线。A/B 把 AICPU poll 改成 acquire load 后仍失败：
`polls=34 max_observe_cycles=21012475385 elapsed_cycles=21110745578`，
最终四个 bootstrap 状态仍为 96/1/96/1。
证据在 `outputs/spmd-bootstrap-acquire-20261025/`；
因此撤回此尝试，不以“删除 civac”作为已证实的修复。

继续核对发现 AICore 的 register-release／exit wait 每次循环都查询同一个 GM timeout 字段，
而该配置由 AICPU 在 READY 前发布后不再写入。
下一候选只在 READY 后缓存该常量（bootstrap scan wait 同样缓存），
保持 AICPU 原来的 cache observe 路径，检验集中 GM 轮询与异常长观察区间的关系。
构建／执行日志：`outputs/spmd-bootstrap-local-timeout-build-20261025.log`、
`outputs/spmd-bootstrap-local-timeout-20261025/`；未验证前不声明根因或性能收益。

该候选的同一单根真机用例连续三轮通过；AICPU bootstrap 等待分别为
50306／48838／54377 cycles（本机 1 GHz，即约 50.306／48.838／54.377 微秒）。
观察区间最大值约 10.0／14.5／10.1 微秒，未再出现前述秒级阻塞。
这是启动路径的对照证据，不是最终 SPMD 或 CSA 的性能报告。
当前可复现改善边界是减少不变 timeout 的共享 GM 轮询；
尚未证明具体 cache／HA／bank 仲裁的微架构原因，不以此推导新的硬件规则。

最终候选移除临时逐 poll 计时和成功日志，只保留出错冷路径的 bootstrap 状态快照。
不改 watchdog 的阈值、比较方式或错误处理；timeout 的写入者仅有 Host 初始化
和 READY 前的 AICPU 配置，READY 后本核缓存具有明确的不变性依据。

无临时计时版本的验证：

- A5 真机：六种 mask 的 N=1 各十轮，共 60 次，全部通过。
  同一 Worker 运行，run_epoch 连续 1–60；保留原探针／golden／保护区／来源断言，
  不放宽 timeout，不关闭多 block 的红灯门禁。
  命令为 `python tests/fdwic/pa_hbg/spmd/test_spmd_contract.py -p a5 -d 0`
  加六个 `--case mask{1,2,3,5,6,7}_n1` 参数与 `--rounds 10`。
  日志 `outputs/spmd-bootstrap-fix-native-20261025/run.log`。
- C++：原有六个目标加 lifecycle retirement 目标，共 7 个目标、118 个 case 全部通过。
  日志 `outputs/spmd-bootstrap-fix-cpp-20261025.log`。
- 构建和首轮提交钩子通过：`outputs/spmd-bootstrap-fix-build-20261025.log`、
  `outputs/spmd-bootstrap-fix-precommit-20261025.log`。
- HBG 默认仿真回归：42 passed、379 deselected，日志
  `outputs/spmd-bootstrap-fix-a5sim-regression-20261025.log`。
- 首次宽目录真机回归：59 passed、4 skipped、1 failed、1 error，日志
  `outputs/spmd-bootstrap-fix-a5-regression-20261025/run.log`。
  error 来自被收入的 TMR 双卡 RDMA 用例，单卡 setup 报 `need 2 devices`，未执行该业务。
  failure 来自 DFX 测试：未开启泳道时 output_prefix 为 `.`，测试 helper 递归读取了历史
  `chip_swimlane_records.json`，误把已转换的旧 clock_alignment 当作本轮提前写入。
  独立真机复跑仍复现，见 `outputs/spmd-bootstrap-fix-dfx-isolated-20261025/run.log`；
  该失败发生在后处理断言，不是 kernel 数值或 resident 初始化失败。
- 修正 DFX helper：只有本轮开启泳道时才枚举 capture，保持原 finalizer 及 after_finalize 回调。
  不删除历史输出，不放宽启用 profiling 时的严格断言。
  新增 5 个 Python 回归：关闭 profiling 不扫描目录；已有 clock_alignment 仍拒绝；
  三级缺 Host 时钟日志仍拒绝；一／二级多余日志仍拒绝。
  红灯为 1 failed、4 passed，修复后结合全部 SPMD／CLA／CSA 回归共 1026 passed。
  日志 `outputs/spmd-clock-guard-red-20261025.log`、`outputs/spmd-bootstrap-fix-python-20261025.log`。
- 最终真机改用明确范围：`pytest examples/a5/host_build_graph tests/st/a5/host_build_graph`
  加 `--platform a5 --exclude-level 4 --device 0 --max-parallel 1 --pto-session-timeout 600 -q`。
  产品的分组 runner 全部成功退出；子任务总结累计 48 passed、4 skipped，
  不把子任务的 deselected 数累加为新的测试项。
  日志 `outputs/spmd-bootstrap-fix-a5-hbg-only-20261025/run.log`。
- 单独运行 `TestSchedulerPhases::test_run --enable-chip-swimlane 3` 真机通过，
  保留 provenance、时钟日志、task timing、scheduler/lifecycle 等原断言；
  日志 `outputs/spmd-bootstrap-fix-a5-hbg-only-20261025/dfx-level3.log`。
- 最终提交检查全部适用钩子通过，包含 Python 类型、格式以及 C++ 检查；
  日志 `outputs/spmd-bootstrap-fix-precommit-r2-20261025.log`。

### 当前交付边界与下一项

普通／Mix context 初始化和当前单 block resident 基线已修复并验证。
尚未增加共同准入日志、SPMD cursor、GM Mix 组控制或 native 全身份 trace。
接下来实现编码契约第 1 项：有界 POD／offset／初始化／溢出检查及 Host 实际大小预分配，
然后接入共同准入与普通多 block；保持现有 legacy 形状门禁直到相应 native 验收通过。
完整目标继续进行，不因前置基线转绿而标为完成。

## 2026-10-25：有界协议存储与 Host 接线

本轮起点 `89b09f026`。并行审查 Host/epoch 链和缓存行布局，主线程实现布局与初始化，
独立测试任务编写 C++ 边界用例；构建、设备验证和提交由主线程统一执行。

### 具体落地

- TaskMetadata 的 timing_slot 值域来自 TaskAttrs（-1 或 0..15），改用 int16；
  腾出的 uint16 保存原 N，维持 16B，并补齐 metadata 手工复制路径，不能只改 struct。
- TaskControl 在既有原子行 byte 24/32/40 放入三个 uint64 字段，整体仍为 128B；
  第二条普通写回行的偏移仍为 64。GraphSummary 复用 padding，WorkerContext 不增长。
- 新增 AdmissionRecord、AdmissionControl、GroupDescriptor/LaneAck/Mailbox 的 POD，
  record 头 64B，后接按实际 W 计算的位图，stride 向 128B 取整；
  log 只存 cooperative_count 个引用。无协同图不分配新协议区；无 Mix 图不分配组和 seal。
- 每 cluster 一个 640B mailbox，descriptor、published_seq 和各 lane ack 分开；
  close seal 保持 MixCount×C×8 字节。位图整行写回不能跨记录；log/seal 将采用原子读写，
  不因短提交锁存在就普通写回可能过时的整行。
- Host 使用实际 worker 数和完整 cluster 数规划；lifecycle 在 READY 前核对实际发现结果。
  `api->run_epoch()` 已由 NativeRunDescriptor 传递，直接复用，不另造计数器。
  UINT64_MAX 是生成器最后一个合法 epoch；布局不擅自拒绝。记录本身仍保持未准入状态。
- 乘积检查使用原 int16 上界，trace 前缀在窄化前检查；resident 专用的 trace 容量错误
  放在原 unsupported-shape fallback 之后处理，不改变超大 legacy 图的原回退优先级。
- 每轮统一清零新控制状态，使用无效 task/order/lane 哨兵；静态记录填写先完整验证
  所有 metadata 与数量，再写入，失败不留下半初始化记录。
- 修正旧 block 文档中“每 task/block 一份且不复用”的描述，统一为编码契约的
  per-cluster mailbox 与持久 per-task/cluster seal；同步明确 sync_start 的动态一次领取策略。

### 当前验证及未开放部分

新增 `test_hbg_spmd_layout.cpp` 的 104 项检查覆盖 3/63/66/96/108 worker、
对齐/不重叠、空图/无协同/纯 SPMD/Mix、污染重初始化、设备地址重绑定、
无效拓扑/数量/形状不部分写入、epoch 0/MAX、N×slot/trace 边界、无 profiling 时 N 的复制。
另对同一大容量 retained block 做大 Mix+profiling → 小纯 SPMD → 大 Mix 的切换，
以不同 epoch/N/mask 与 fresh 镜像逐字节比较，并检查小布局不会写入未使用尾部的 canary。
最终 C++ 八个相关目标共 222 个 case 通过；日志 `outputs/spmd-layout-cpp-final-20261025.log`。
editable 全构建通过（`outputs/spmd-layout-build-20261025.log`）；
1026 个 Python 回归通过（`outputs/spmd-layout-python-20261025.log`），
HBG 默认仿真 42 passed、379 deselected（`outputs/spmd-layout-sim-20261025.log`）。

A5 device 0 六种 mask 的 N=1 各三轮、共 18 次通过；同一 Worker 的 epoch 为 1–18。
真机默认 HBG 回归 60 passed、4 skipped；本次只排除宽目录中被误收的 TMR 双卡 RDMA 目录，
未排除 HBG 用例。单独三级泳道检查通过。
日志依次为 `outputs/spmd-layout-native-20261025/{probe,hbg,dfx3}.log`。
目前只落存储与静态接线，尚无运行期准入发布或 block cursor 执行；
N>1/sync_start 仍维持原门禁，不能将本轮布局用例转绿当作原生 SPMD 通过。

入口审查另发现 AIV 完成普通任务时，直接唤醒的 Mix 后继可被留在无人消费的局部 candidate。
独立真实 helper 复现为 AIV0/AIV1×mask 3/5/6/7 共 8 项失败，
AIC direct 路径和禁用 direct sink 的 16 项对照通过；日志
`outputs/spmd-direct-mix-repro-20261025.log`。这是后续依赖推进的前置问题，
新复现文件与修复会单独提交，不混入已全绿的存储布局提交。

## 2026-10-25：AIV 完成到 Mix 后继的交接修复

本轮起点 `3c72b320a`。此前三笔提交已按用户单独的 push 指令推送到
`nalinaly/simpler:feat/fully_distributed_within_core`；本轮恢复阶段性提交、不自动 push。

### 原因和修改范围

`scheduler_service_cluster_completion_slot` 原先在 has_mix 且 candidates 非空时，
无论完成者是 AIC 还是 AIV，都向依赖推进函数提供 direct Mix sink。
但 `scheduler_fill_mix_after_completions` 只由 AIC 消费该 sink，
executor 的 candidates 又是单轮局部变量。因此 AIV 解出唯一 Mix 后继时，
前驱已成功 DONE、wake-list 已关闭，而后继既不在 ready inbox，也不在下一轮状态里。
这不是前驱未完成或 Mix kernel 自身等待，而是调度候选所有权交接丢失。

唯一生产调用点增加 `is_mix_coordinator()` 条件：AIC 保留原 direct fast path，
AIV 不接收自己无法消费的 Mix candidate，沿原完成驱动路径首次发布到 Mix inbox。
不重复入队，不增加全局 ready 表，不改变 kernel、shape gate、任务完成语义或 SSBUF 协议。

### 验证记录

- 修复前重跑 24 项真实 helper 回归，16 passed、8 failed；失败正好是两条物理 AIV
  × mask 3/5/6/7。日志 `outputs/spmd-direct-mix-red-20261025.log`。
- caller 条件修复后，同一 24 项全部通过，日志 `outputs/spmd-direct-mix-green-20261025.log`。
  fixture 覆盖真实 route、wake-list、dispatch、completion 及后续 AIC placement，
  不使用参考模型替代生产 helper。
- 扩展为 60 项交接测试，新增已有 Mix 队头、同轮两个 Mix 后继、placement 满后恢复，
  并检查 AIC completion 的中间态确实保留 direct candidate，防止“全部关闭快路”假绿。
  结合原 222 项 C++ 回归，共九个 target、282 个 case 通过；
  日志 `outputs/spmd-direct-mix-cpp-final-20261025.log`。
- 新增 `test_aiv_mix_handoff.py` 和专用 Host orchestration，复用全部四个既有 kernel，
  不复制设备实现。显式 TaskId 依赖形成 `AIV_i → Mix_i → AIV_(i+1)`，
  四种 Mix mask 各取一对／17 对串链，保证每个普通 AIV 的唯一后继是 Mix。
  校验 root 和 Mix 两套记录的 exactly-once、原 N、身份、保护区、padding，
  保留每次调用的新鲜 resident mode；不额外声称验证了前驱数据可见性的完整时序。
- editable 构建通过：`outputs/spmd-direct-mix-build-20261025.log`。
  首轮 Python 命令遗漏 CSA 已有的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，
  Torch 2.12 加载系统不匹配的 torch_npu，在导入阶段失败，未进入设备执行。
  补回原有环境设置后重跑；不降低 Torch 版本，不增加 runtime 开关。
- 新交接探针 a5sim 八场景各三轮，共 24 次通过；A5 device 0 八场景各十轮，
  共 80 次通过，同一 Worker 的 run_epoch 为 1–80。日志分别为
  `outputs/spmd-direct-mix-native-sim-r2-20261025.log`、
  `outputs/spmd-direct-mix-native-20261025/probe-r2.log`。
  本机仍无 npu-smi/task-submit；CANN 查询再次确认 A5，按用户授权直接用卡、无队列锁隔离。
- 追加检查发现 pytest 的 importlib 模式不能使用 standalone 的裸同目录导入；
  按 `__package__` 选择相对或同目录导入，不吞 ImportError、不复制基类。
  pytest 入口八场景通过（一个 pytest item），日志 `outputs/spmd-direct-mix-pytest-fixture-20261025.log`。
- Python 合同／CLA／CSA／clock guard 本轮选择范围共 942 项通过，
  日志 `outputs/spmd-direct-mix-python-r2-20261025.log`；不将它与前轮不同选择的计数相加。
- HBG 默认仿真 42 passed、379 deselected；真机 60 passed、4 skipped、356 deselected。
  真机仅显式排除宽目录误收的双卡 TMR RDMA 目录，未排除 HBG；
  超时参数按当前 CI 分别为 600/1200 秒。日志分别为
  `outputs/spmd-direct-mix-sim-r2-20261025.log`、`outputs/spmd-direct-mix-native-20261025/hbg.log`。
- A5 `TestSchedulerPhases::test_run --enable-chip-swimlane 3` 通过，
  保留原 provenance、task timing、scheduler/lifecycle 与时钟检查；
  日志 `outputs/spmd-direct-mix-native-20261025/dfx3.log`。
- 首轮全部适用提交钩子通过，日志 `outputs/spmd-direct-mix-precommit-20261025.log`；
  最终暂存后检查也全部通过，日志 `outputs/spmd-direct-mix-precommit-final-20261025.log`，
  未使用 SKIP 或 no-verify。

该修复只建立后续共同准入所依赖的可靠候选交接；并未开放多 block 或 GM Mix。

## 2026-10-25：共同准入基础层与入口接线审查

起点 `d78e3fbad`，该修复提交留在本地，未 push。
本轮先实现可由真实 ready／completion 接入的准入基础层并补 C++ 交错测试；
完整 entry gate、slot 让位和 block 执行仍是后续接线，不用 helper 通过代替 native 验收。

### 接线时不能遗漏的边界

- pending admission 必须跨 executor 循环保留。普通候选采用总量为两个的持久池，
  pop／direct resolve 时取得一个容量名额，预填 slot 仅绑定该候选。
  前序义务到来时可释放未执行 payload、保留候选身份，不能另加两个 backup 又让原池继续取任务。
  释放预填 slot 不发布完成、不回退 dispatch generation；重新物化用新 generation。
- 当前 prepare 会计算 predicate；暂存再恢复候选时不能无意重复求值。
  接线需缓存已求值结果，或把 ordered 模式的求值移至真正 entry permit 后，只算一次。
- 纯 C/V 多 block 首次开放时，只对不含 Mix 的整图启用新协议；
  任意 Mix（包括 N=1）与纯多 block 同图仍须整图回退，直到 GM Mix 一起接入共同顺序。
  全 N=1 图继续保留当前 resident 能力，不能倒退成 legacy。
- 普通分波的 demand 是一个位置／一个组的最低需求，不是原 N 个同时驻留位置；
  lifecycle 的完整 cluster、实际 W/C 和类型校验保持，不以 N 大而放宽拓扑。
- 每次 prepare 都须写本次 block_idx 和原 block_num，包括多 block 后复用为 N=1 的场景。
  dispatch、completion、ready-transition 和 sampled timing 都要按 block 处理 trace。
- Host 目前以 task_ref 作 reg_task_id，同核多波会重号；导出需给每次执行唯一 join 身份。
  converter 的 `(epoch, task_id, core_id)` 索引也会覆盖同核的不同 block，
  需要随真实多 block trace 接线补回归，不能仅改 Host 七／五列记录就宣布全身份审计完成。
- 当前 Host 没有识别任意异步 kernel 的 has_async 属性；resident 以无效 async token
  使 deferred 注册失败。保留这条实际能力边界，不声称 Host 会自动识别并回退所有异步 kernel。

### 本轮基础层范围

准备记录与提交分开：成员依据紧凑 worker index 和已验证的真实核类型冻结，
准备阶段不推进可见 tail；短锁只提交完整 record/log，最后发布连续 tail。
每核扫描停在最早成员义务；不属于该任务的核跳过。
ordinary permit 不隐式复用旧判断：先处理可见前缀，最后关键原子读取 tail，
只有与本核 scan_cursor 一致且无 pending／current 义务时才允许一次 kernel。
测试将用“先观察空前缀、另一核提交、再申请许可”的确定性交错检查这一点。

本轮尚不实现 cursor 领取／LocalDone 关闭，不用手工清 current 的测试伪造这些性质。
主循环、context、完成聚合、分波容量与完整 trace 齐备之前，不开放 N>1 的 native shape gate。

### 已实现基础层与检查证据

- `scheduler_admission.h` 实现初始化、claim、prepare、commit、refresh 和 ordinary permit，
  使用已有 layout、ready claim、GM 原子／cache 包装及 first-error 通道；
  新增 site 110–116，不改变旧诊断编号。
- 每核协议状态为 88B，未增大既有 SchedulerLocalState，也未放入每核全图表。
  prepare 先 acquire Host 初始化记录，按真实 compact worker index 冻结成员；
  纯 C/V 只选对应类型，Mix 包含完整 cluster 的协调／资源成员。
  普通 N=512 不要求成员数达到 N；sync 的准备容量检查按不同位置计，尚无 sync 执行。
- record 和 bitmap 分行发布，packed log 仅原子单元访问；
  短锁提交 order/ref/ADMITTED 后才推进 tail，锁竞争保留 pending。
  refresh 每次最多扫描八条；缓存当前义务后不重复扫描 topology，不用 TaskDone 清除当前义务。
- 首批 70 项真实 helper 测试通过。另补 epoch=0、未初始化本地状态的反例，
  初版 ordinary permit 错误返回 true；真实红灯日志
  `outputs/spmd-admission-uninitialized-red-20261025.log`。
  在已有尾 padding 加 initialized 标志，只有初始化全部成功才置位，保持 88B；
  71 项全部转绿，日志 `outputs/spmd-admission-green-20261025.log`。
- 12 个发布线程提交 48 项任务时，增加一个真正并发的非成员 AIV reader。
  写者首项提交后必须让读者观察非空未完结前缀，再继续后续项；
  读者真实调用 refresh，检查单调前缀及成员义务，最后再核对完整唯一日志。
  该测试重复 200 次通过，日志 `outputs/spmd-admission-concurrency-20261025.log`。
  ready pop 在线程启动前完成，因此不把它声称为并发 ready FIFO 测试。
- 首次 A5 编译发现 GM metadata 引用不能传给要求 Local Memory 引用的判断函数，
  而 CPU 编译对此不报错；日志 `outputs/spmd-admission-build-20261025.log`。
  两处改为复用 `scheduler_load_dispatch_metadata` 取得本地值，不新增另一套接口。
- 修正后 editable 全构建通过，日志 `outputs/spmd-admission-build-r2-20261025.log`。
  补充旧布局零 W/C summary、无协议区域的兼容测试后，准入专项共 72 项，
  十个相关 C++ target 共 354 项通过，日志 `outputs/spmd-admission-cpp-verified-20261025.log`；
  最终版本的并发专项又重复 200 次通过，日志 `outputs/spmd-admission-concurrency-r2-20261025.log`。
- Python 相关回归 942 项通过（`outputs/spmd-admission-python-20261025.log`）；
  HBG 默认仿真 42 passed、379 deselected（`outputs/spmd-admission-sim-20261025.log`）；
  A5 默认 HBG 60 passed、4 skipped、356 deselected
  （`outputs/spmd-admission-native-20261025/hbg.log`）。真机保持原有单卡 HBG 范围，
  仅排除被宽目录误收的 TMR 双卡目录；CANN 再次确认 A5，无 task-submit 锁隔离。
- AIV→Mix 交接仿真八场景各三轮共 24 次通过，
  日志 `outputs/spmd-admission-handoff-sim-20261025.log`。
  A5 同样各三轮共 24 次通过，三级 `TestSchedulerPhases` 也通过；
  日志 `outputs/spmd-admission-native-20261025/{handoff,dfx3}.log`。
- 首轮 pre-commit 仅自动调整两处 C++ 测试格式；重新暂存后全部适用钩子通过，
  日志 `outputs/spmd-admission-precommit-r2-20261025.log`，未跳过适用检查。

当前初始化的范围检查依赖合法 Host 镜像：检查地址算术、对齐和区域相对顺序，
没有 allocation total_size 参数，不能宣称防住任意伪造的越界 offset。
A5 构建及旧用例回归只验证新头的设备编译和已有行为；
新发布协议的 GM 可见性仍须随 executor 接线进行真机验证。

## 2026-10-25：重复运行暴露的 A5 模拟器 TLS 生命周期问题

接入纯 C/V 执行后，原生纯任务矩阵在 A5 和仿真各 90 次通过；
新增边界用例的 A5 全部 120 次也通过。仿真执行到同一进程第 101 次时，
在新轮握手阶段超时，尚未进入准入循环；此前 30 个场景（含重叠和真实依赖）均通过。
当时正在执行 512×3 → 1×3 → 65×3 → 1×3 → 512×3 的复用链，
第 100 次已经返回并通过校验，第 101 次卡在 N=1 的第二轮，不能归因为 block 领取。
日志 `outputs/spmd-ordered-contract-sim-20261025.log` 保留失败位置。

检查发现模拟 AICore SO 每轮卸载，而生产 `kernel.cpp` 的首次 wrapper 调用
通过 pthread_once 创建十个进程级 TLS key，原代码既不回收也不检查创建失败。
本机 key 上限为 1024。独立 C++ 回归将实际生产 wrapper 编成可装卸 SO，
用轻量 executor 只检查身份，不复制 TLS 实现；真实执行／卸载后，
剩余 key 从 1021 降为 1011，确认每轮漏十个。
旧代码在只剩五个 key 的部分创建场景还发生 SIGSEGV；这不是单纯的超时推测。
原挂起仿真进程的调试器 attach 不可用；保存日志和线程列表后，仅终止本次失败进程，
退出状态为 143，不计作成功，也没有停止其他会话或重置设备。

修复保持 pthread TLS，不改回已知有 aarch64 TLSDESC 问题的 thread_local：

- 逐项记录已创建 key，部分失败立即回滚，只删除实际拥有的 key。
- SO 的生命周期对象在所有 core 线程已 join 后、模块卸载时释放全部 key。
- 新增模拟包装内部初始化入口，与 wrapper 共享同一个 pthread_once；
  loader 在线程启动前调用，失败关闭 SO、清空 handle/function pointer，
  沿原内部错误返回，不 throw／abort 用户进程，也不让任务进入握手等待。
- HBG/TMR 两种 Runtime ABI 各 23 项、共 46 项独立测试通过，
  覆盖 128 次真实 wrapper 重载、剩余 0–9 个 key、回滚、其他 key 不被覆盖和重载恢复。

这是模拟平台的独立修复，与纯 C/V 调度接线分开提交；不改变 A5 真机包装层。
完整 editable 重建已通过（`outputs/spmd-ordered-build-r2-20261025.log`）；
主构建树中 HBG/TMR 两个 TLS target 的 46 项也全部通过，
日志 `outputs/spmd-ordered-cpp-final-20261025.log`。
原失败场景不改用例、保持每轮数值和新鲜 resident 来源断言，
在同一进程内完成全部 32 场景、120 次调用，退出状态为 0；
日志 `outputs/spmd-ordered-contract-sim-r2-20261025.log` 明确包含第 101–120 次。
同期最终版本 A5 120 次也通过（`outputs/spmd-ordered-native-20261025/contracts-r2.log`），
这不表示 TLS 修复修改了真机路径，而是同一工作区纯 C/V 实现的验收证据。

平台修复按最宽范围运行 `pytest examples tests/st --platform a5sim`，
使用八个虚拟设备、并发 8、session timeout 600；默认非 manual 范围不额外筛 runtime。
资源阶段 59 项、L2 HBG 18 项、L2 TMR 30 项，共 107 项通过，314 项按平台等条件未选中，
总体退出状态为 0，耗时 241.59 秒；日志 `outputs/spmd-ordered-all-sim-20261025.log`。
这里的 pytest 项目数不等于各 scene 内参数场景或每轮 kernel 调用数。

首次提交检查扫描到已不存在的 `fully_distributed_within_core` runtime 的旧编译数据库，
因缺少当前 TaskId include 路径产生 clang-tidy 解析错误，并误报非空构造函数可 default。
核对当前两架构仅有 HBG/TMR build_config 后，将六份旧 runtime 顶层数据库原地改名为
`compile_commands.json.obsolete-runtime-20261025` 留存；未删除旧构建产物或参考源码，
也未修改有效 runtime 的编译选项来回避检查。
使用当前 HBG/TMR 数据库的全部适用钩子通过，
日志 `outputs/spmd-tls-precommit-r2-20261025.log`。

TLS 修复独立提交为 `030297fc7`，未 push；下面的调度接线不包含在该提交中。

## 2026-10-25：纯 C/V 有序分波、局部关闭与完成聚合

### 实际接线和能力边界

- 在无 Mix 的图中初始化本核准入状态，存在协同任务时才启用有序执行路径。
  纯 C/V 普通 N>1 开放；“纯多 block + 任意 Mix”暂时整图回退，
  不让已接线的 own/direct 与旧 SSBUF Mix 绕过共同顺序。
  全 N=1 Mix 原路径保留，GM Mix／sync_start 仍未实现。
- 两个候选额度独立于两个 payload 槽。ordinary 取得实际 entry permit 后才释放候选额度；
  早序义务出现时停放尚未进入 kernel 的普通 READY 槽，保存已求值 predicate，
  释放可重建 payload，不重复入队、不重复求值，也不伪造完成。
- pure block 以最多八次 CAS 尝试不可撤销领取，cursor 达 N 时用关键原子读确认，
  不继续累加溢出。领取成功立即绑定 task/block/本核 slot；每次 prepare 都填原 N 和实际 block。
- 每个 block 返回后经既有 slot generation 防重复消费，再递交 completed_blocks。
  只有旧值为 N−1 的完成者写逻辑 DONE、推进 wake-list、增加逻辑完成数。
  LocalDone 只要求 cursor=N 且本核的该 task 槽全部退役，不等待远端 TaskDone。
- Host demand 按相应类型的最少执行位置计，不要求原 N 同时驻留；
  mask 6 的 AIC 协调职责不因没有 AIC kernel 而丢失。
  普通 N=2 和 sync N=2 在同一既有数值用例中分别断言 resident 与 legacy 来源。
- 无 Mix 的调度不查找、不初始化、不等待 SSBUF；prepare/completion/bootstrap 接受空 SSBUF。
  本地 slot 为 24B、SchedulerLocalState 为 504B、准入状态含两个 48B 候选共 184B，
  没有增加每核全图表或运行期 heap 分配。
- trace 地址包含 block 索引，Host 的 reg_task_id 使用唯一 trace_index，
  converter 的执行连接键增加 reg_task_id，避免同 task 同核的不同 block 共用事件锚点。
  这还不是含 order／LocalDone／GM 组身份的完整协议 adapter。

两个普通 direct 链可持续占满候选池，导致协同候选直到某条有限链结束才被发现。
合法有限 DAG 的推进不等于固定轮数的发现公平性；本实现没有后者的保证。
当前 probe 有额外原子事件记录，所有时间仅用于诊断，不能据此报告调度性能收益。

### TDD 与功能验收

- 实际 ordered helper 的 C++ 40 项覆盖 24 项容量形状、8 项领取／完成／LocalDone／
  predicate 停放行为，以及 8 项无 SSBUF 和 probe 时间边界。
  最后 8 项初次运行时，2 项无 SSBUF 直接暴露真实空指针门禁问题；
  其中计时断言还遇到通用 CPP fixture 的 scheduler_cycles 恒零，不能把它算成生产计时红灯。
  target 改用既有 CPU_SIM 后，暂撤生产计时修复，得到 36 通过／4 个 ordinary probe 失败
  （`outputs/spmd-ordered-clock-red-20261025.log`）；恢复实际 claim 前后计时后 40 项全绿
  （`outputs/spmd-ordered-no-ssbuf-green-20261025.log`）。cooperative 不伪造队列 probe。
- converter 新增 15 项先得到 13 失败／2 通过，再修正完整执行连接键后全部通过。
  新原生夹具 96 项覆盖坏记录、反向独立准入、真实依赖及每一轮来源门禁。
- 13 个 C++ target 共 443 项通过，其中 HBG 397、TLS 46；
  日志 `outputs/spmd-ordered-cpp-final-20261025.log`。
- Python 当前相关回归 884 项全部通过，无失败／跳过：SPMD 合同 649、
  converter/归属/捕获/时钟/开销 223、四种 exporter 身份 12。
  exporter 首轮因缺少测试二进制跳过 8 项，补编四个既有 collector targets 后 12 项全部运行；
  日志 `outputs/spmd-python-regression-counts-final-20261025.log`。
- 原纯任务矩阵 A5／仿真各 90 次通过；新增 ordered 32 场景每轮 golden 与来源门禁均启用。
  30 个普通场景各三轮，加两个 512→1→65→1→512 retained 序列各三轮，
  A5 和修复后仿真分别完成 120 次；包含慢块位于头／中／尾、真实跨类型后继、
  无依赖的 C→C／V→V／C→V／V→C 实际重叠。
  日志 `outputs/spmd-ordered-native-20261025/contracts-r2.log` 与
  `outputs/spmd-ordered-contract-sim-r2-20261025.log`。
- 默认 A5 HBG 回归 60 passed、4 skipped、356 deselected，耗时 198 秒，
  日志 `outputs/spmd-ordered-native-20261025/hbg.log`。
  沿用已记录的单卡排除项：被宽目录 runtime 过滤误收的 TMR 双卡 RDMA 目录；
  本机无 task-submit，运行没有队列锁隔离。仿真最宽范围 107 项结果见 TLS 一节。

### L1/L4 实际产物检查暴露的记录缺口

在 A5 对 pure_c_n65、pure_v_n65、alternate_n1_spmd_mask1 各采集 L1/L4，
每场景一轮，保留 golden 和新鲜来源检查；六次调用全部成功。
相应执行条数为 130、130、583，raw 的唯一连接键、规范化结果及两个视图的执行数量均核对。
不用 retained 五轮场景做多轮泳道证明：它目前复用同一 output_prefix，固定文件仅保留最后一轮。

L4 交替图虽有完整 583 条 kernel／完成记录，却只导出 580 条发布 phase：
三个普通 direct 后继先进入持久候选池，再由 ordered fill 发布，仍携带旧 REFILL 标记；
旧 exporter 认为这类发布应由前驱完成路径的 refill phase 表达，因此抑制自身 dispatch。
新路径没有在 completion 内填槽，不能保留这一标记。
新增 C/V 两项测试真实完成 N=2 的生产任务并取得 N=1 direct 后继，
确认 completion 未填槽、候选未丢失，再调用 ordered fill；两项仅在 publication_mode
实际 REFILL、预期 DISPATCH 处失败，其余身份、计时、依赖和完成断言通过。
日志 `outputs/spmd-ordered-deferred-publication-red-20261025.log`。
实际修复只在 ordered 普通候选填槽处使用既有 DISPATCH 标记，保留 DIRECT_RESOLVE 来源，
不修改旧 inactive 路径。修复后 42 项全部通过，
日志 `outputs/spmd-ordered-deferred-publication-green-20261025.log`。

最终 editable 重建 `outputs/spmd-ordered-build-r4-20261025.log` 成功；
13 个 C++ target 重新构建运行，共 445 项（HBG 399、TLS 46）通过，
日志 `outputs/spmd-ordered-cpp-final-r2-20261025.log`。
所有适用 pre-commit 钩子通过，日志 `outputs/spmd-ordered-precommit-final-20261025.log`；
先前一次只由 clang-format 自动修整已有 AICPU 文件和测试格式，未跳过检查，随后重建复验。

A5 同样六个 L1/L4 场景复采成功；逐项审计结果为：

- 每份 raw、规范化行、Worker bars 均为 130／130／583；L4 Scheduler timing 和 bars 一一匹配。
  每 task 的 reg_task_id 完整覆盖各自 trace 前缀，没有重复键，且实际覆盖同核执行多个 block。
- L4 的每 task 发布 phase 数与执行数相等。交替图 dispatch 582 + worksteal 1 = 583；
  task 2／4／6 各恢复一条 dispatch，不再缺失。
- 交替图仅 task 0 有一次真正队列 state_probe，1577 个 cycle；两个纯 N65 图无伪造 probe。
  这只是诊断事实，不将其外推为性能基准。
- 每份 L4 的 96 个 scheduler streams 均为 aicore，dropped=0、truncated=0；
  complete 按 block、resolve 按逻辑 task 的条数也匹配。
- 审计前后 raw/merged 的 SHA-256 不变。日志
  `outputs/spmd-ordered-trace-artifact-audit-final-20261025.log` 记录六个准确产物目录。

JSON 尚未导出 ready_source 数值、显式 block/N/order/generation 全套字段，
没有准入／可见性／LocalDone／GM 组的完整因果事件链。
逐 block 泳道通过不能代替第五项的完整协议审计；这部分继续保持未完成。

最终版本 ordered 场景在 A5／仿真再次各 120 次通过，
日志 `outputs/spmd-ordered-native-20261025/contracts-final.log` 与
`outputs/spmd-ordered-contract-sim-final-20261025.log`。
原有三级 `TestSchedulerPhases::resolve_dummy` 也通过；
日志 `outputs/spmd-ordered-native-20261025/dfx3-final.log`。
最终 HBG 默认仿真 42 passed、379 deselected，耗时 58.61 秒，
日志 `outputs/spmd-ordered-hbg-sim-final-20261025.log`。
最终 A5 HBG 默认回归 60 passed、4 skipped、356 deselected，耗时 156.98 秒，
日志 `outputs/spmd-ordered-native-20261025/hbg-final.log`；完整命令链退出状态为 0。

同步设计总览、编码契约、测试入口和 CSA README 的实际门禁名称／能力说明。
保留原有历史红灯与原 CSA kernel/数据；未将已开放纯 C/V 误写成 CSA resident 已可用。
下一项仍是所有 resident Mix（含 N=1）共同准入、GM descriptor/ack/seal 和各 lane 自备参数，
之后才有资格开放纯多 block 与 Mix 同图，并执行原样 CSA resident 的严格验收。

## 2026-10-25：GM Mix 分波与各 lane 自备参数接线

纯 C/V 已提交为 `c8374e37b`，TLS 修复为 `030297fc7`；本轮未 push。
提交后版本标记重建成功，日志 `outputs/spmd-ordered-postcommit-build-20261025.log`。
改动前用原 `mask3_n2` 仿真入口复验，实际运行后被新鲜来源门禁以 mode=4 拒绝，
退出状态为 1；这是能力红灯，不是编译失败，日志 `outputs/spmd-gm-mix-native-red-20261025.log`。

采用现有 640B GroupMailbox 和持久 MixCloseSeal，不增加运行期分配。
本核增加 40B GM 组状态（准入本地状态共 224B），区分全 run 的 group_seq、
本 task 的 task_last_seq，以及 CLAIMED/PUBLISHED/ARRIVED/RUNNING/RELEASED。
每 cluster 同时只保留一组；resource_mask 为本 cluster 的三条 lane，不扩大成全芯片屏障。
单 AIV 的 mask 3/5 按 block 奇偶选物理 lane 1/2，逻辑 subtask 不改；
mask 6/7 固定双 AIV 对应关系。inactive lane 只到达／释放，不制造 kernel 或完成贡献。

接线分工：低层组协议放在 completion 可见的 `scheduler_gm_mix.h`；
ordered 填槽移至 dispatch，避免 completion/admission/Mix 互相包含。
各 lane 共用 own-slot prepare/entry/trace/退役，只有组协调 AIC 收齐 release 后贡献一次 block。
完成驱动的后继仍沿既有 ready/wake 路径；当前 ordered Mix 后继首次入队，
不放进旧 SSBUF 临时候选后遗失，也不共享普通候选的可变 sink。

成功 EXIT 还须关闭未消费的 seal 和本核义务，才能向 AICPU ACK；
该 drain 不领取新 block、不准备／执行 kernel。错误 EXIT 不补 released 或成功计数。
全图 task 完成计数不被用来取代每 cluster 的最后发布／释放／关闭证据。

Host contracts 新期望先得到 23 项中 3 个真实失败：普通多 block Mix、
纯多 block 与 Mix 的图级限制及提交顺序组合；日志 `outputs/spmd-gm-mix-host-red-20261025.log`。
最初一次编译撞到并发编辑的未完成结构字段，已明确排除，不把该编译错误算作行为红灯。
GM 协议、真实 prepare/completion、跨 cluster 和退出场景的 C++ 测试正在补齐；
截至此记录，尚未宣布新协议的仿真或 A5 执行已通过。

首批 GM Mix 生产 helper 测试 57 项全部通过，旧 11 个相关 C++ target 也全部通过；
日志 `outputs/spmd-gm-mix-cpp-first-20261025.log` 与 `outputs/spmd-gm-mix-existing-cpp-20261025.log`。
完整构建先在实际 executor 编译单元暴露 EXIT 路径误用仅接受 int64 的 scheduler_gm_load；
改用既有 uint64 的 fetch_add(0) 关键读后重建成功，日志 `outputs/spmd-gm-mix-build-r2-20261025.log`。
这一编译错误不在 helper 单元测试的编译范围内，不把 C++ 单测通过外推为完整构建成功。

原生仿真先运行 mask3_n1、mask3_n2、mask5_n2，各三轮共九次通过。
第十次 mask6_n2 在退出阶段约十秒后报 core 20/21 not released，未返回测试结果，
日志 `outputs/spmd-gm-mix-smoke-sim-20261025.log`。
线程检查仅剩 sim-aiv-20/21 两个执行线程自旋；调试器 attach 不可用，没有拿到现场 seal 值。
保存线程／调试器结果后仅终止本次挂起仿真进程，退出状态 143，不计作通过；未上卡或重置设备。

静态核对发现一个确定的控制流遗漏：startup 直接看到 EXIT 的分支绕过主循环，
也绕过新加入的成功退出 drain。零工作 cluster 的 AIC 可以在 bootstrap 只关闭第一条义务后，
错过短暂 IDLE 而直接 ACK，留下后续 Mix seal 未发布；已进入主循环的 AIV 则等待该 seal。
这是与现场线程分布吻合的解释，但不能声称已经从原挂起进程读到该状态。
sim 的 retirement 等待十秒，drain watchdog 为二十秒，晚 ACK 后不会补发 return gate，
因此后续持续自旋也可能已在 post_close_release 等待，不能把线程存活直接当成仍在 drain。
下一步使用实际 executor 的 startup EXIT 路径造确定性红灯，
将成功 drain 汇合到所有退出分支的实际 EXIT 确认之后、ACK 之前，而不是增加 timeout。

实际 executor 的确定性回归已建立：`test_hbg_gm_mix_startup.cpp` 在同一编译单元
包含生产 executor，先通过真实准入／组准备／测试 kernel／完成路径在 cluster 0 完成两个 Mix，
然后让零工作 cluster 1 的 AIC 直接看到 EXIT。三个成功场景 `(3,6)/(6,6)/(6,7)`
均准确失败于第二项 seal 仍为 0；第一项 seal、EXITED、没有新增 kernel、没有 SSBUF 读取均成立。
错误轮的控制项通过，未伪造 seal；日志 `outputs/spmd-gm-mix-startup-exit-red-20261025.log`。
这证明 startup 分支确实存在提前 ACK 的遗漏，不需要把原挂起现场中未读取的值当作证据。
修复把唯一 drain 放到共同尾部的真实 EXIT 确认后、最终统计与 ACK 前，
删除主循环内的另一份 drain，保持错误退出不补成功状态。

### 原样 CSA 的单 AIV 通信 ABI 校正

首版按 block 奇偶轮换 single-AIV 的假设不能沿用到冻结 CSA。
实际 managed PTO-ISA pin 为 `418de18dde8fa33773dc5c384fdfe5506463b628`：
`include/pto/npu/a5/TPush.hpp` 的 NO_SPLIT Acc→Vec 使用 SingleModeVec0，
`TMov.hpp` 将其转为固定 subBlockId=false 并传给硬件搬运；目标 UB 不由 runtime payload 改写。
CSA 的 `qr_head_aiv.cpp` 使用这一模式，其编排为 AIC/AIV0；所有十一个 MixedKernels 声明
均为 AIC/AIV0/INVALID。既有 legacy 也按逻辑 bit 固定 AIV0/AIV1。
旧 resident helper 可以任选 lane，不足以证明这种业务通信 ABI 可以任选；
CCE builtin 与 payload sub_block_id 也不是同一个证据，不能靠它们的名字推断兼容。

据此新增四项实际红灯：mask3/5 在奇偶 block 都须保留逻辑槽对应的物理 lane，
观察者须拒绝偷偷挪到另一条 AIV 的 descriptor；旧映射四项均失败，
日志 `outputs/spmd-gm-mix-single-vector-abi-red-20261025.log`。
发布和校验统一改为 lane==subtask：mask3→AIV0、mask5→AIV1、mask6/7→各自两条 AIV。
原测试里“必须轮换”的断言同步纠正为固定 ABI，不删身份／原 N／物理 sub_block 校验。
不修改冻结 kernel，不把该源码证据冒充已经上卡测过错误放置。

另增 44 个 GM 原生事件场景，每轮实际 60 次调用、三轮 180 次，
覆盖四种 mask、N1/2/65、512→1→65→1→512 重用、慢头／中／尾、真实依赖和独立任务重叠。
共享 ordered golden/comparator 按 mask 枚举全部 active slot，旧 32 场景不变；
旧 96 项加新 184 项 CPU 夹具检查全部通过，日志 `outputs/spmd-gm-mix-native-fixture-cpu-20261025.log`。
多 slot golden 的原错误先由 mask3_n1 的实际 CPU 夹具红灯固定，
没有把旧“非纯 C 一律只检查 slot1”的比较逻辑照搬成 Mix 假绿。

共尾退出与固定 ABI 修复后，61 项 GM helper 和 4 项真实 executor 启动退出测试全部通过，
日志 `outputs/spmd-gm-mix-exit-abi-green-20261025.log`；完整构建 r3 也通过。
原五场景仿真／A5 冒烟各三轮、15 次均通过，包含此前卡住的 mask6_n2。
随后原始六种 mask × N 矩阵及槽复用共 90 场景各三轮，
仿真和 A5 分别完成 270 次，全部保留逐轮输出和新鲜 resident 门禁。
日志 `outputs/spmd-gm-mix-matrix-sim-20261025.log` 与
`outputs/spmd-gm-mix-native-20261025/matrix.log`。

新 GM 原生事件场景在仿真／A5 各完成 180 次，覆盖 mask6 无 AIC 事件、
纯任务与 Mix 的真实前驱关系、快 cluster 跨 task 重叠、全 active slot 及大小图重用。
日志 `outputs/spmd-gm-mix-events-sim-20261025.log` 与
`outputs/spmd-gm-mix-native-20261025/events.log`。
默认 HBG 仿真 42 passed、379 deselected（54.87 秒），A5 60 passed、4 skipped、
356 deselected（158.71 秒），日志分别为 `outputs/spmd-gm-mix-hbg-sim-20261025.log`、
`outputs/spmd-gm-mix-native-20261025/hbg.log`。
真机仍直接使用已由 CANN 确认的设备 0，未使用不存在的 task-submit，也未获得队列锁隔离。

### 无 kernel 服务节点的真实完成路径

真实 TPipe PA SmallCase1 的 A5 L4 已通过：72 条执行和 72 条 scheduler 记录，
producer=aicore、96 个 scheduler stream 全为 aicore，无 dropped/truncated。
日志 `outputs/spmd-gm-mix-native-20261025/tpipe-small-l4.log`，
产物 `outputs/TestSpmdPagedAttentionHbgA5_SmallCase1_20261025_094510/`。
它证明原 kernel 通信可运行，但不替代完整 CSA 或 sync_start 验收。

随后原样 CSA 的 deps/timing 两轮均通过七项 golden，严格 L4 却拒绝了 2310 条执行：
期望是 2308，新增两条来自 task 14／27 两个 kernel_ids 全为 -1 的 Dummy 服务节点。
`outputs/csa-hbg-l4-ojdh7gr2/result.json` 为 passed=false；
原始执行与 scheduler 都为 2310、producer=aicore，无丢弃和截断。
旧路径不是单纯导出问题：Host 为 pending inline 节点合成 AIV0 形状并分配一个 trace，
设备 prepare 空 payload、发布 READY、走一次空 kernel 调用及完成记录。
不得以过滤导出或放宽 2308 门禁代替修复执行语义。

`test_hbg_inline_services.cpp` 先建立七项真实 RED：普通／ordered 路径占用假执行槽，
无关 Mix 等待时服务节点被滞留，bootstrap／ordinary／direct 三处 ready transition
覆盖后续 kernel 的零长度 trace 前缀，completion 误报 refill，以及 N2→Dummy→N1 依赖链。
最初测试编译曾使用不存在的 queue 常量；改用实际 config.self_core_type 后才执行，
七项全部实际失败，日志 `outputs/spmd-inline-services-red-20261025.log`。

修复由 ready 唯一 owner 完成服务节点：DONE→既有 resolve_completion→pending_completed，
后继首次入队而不递归 dispatch，不创建 payload、不增 generation、不生成 kernel trace。
ordered discovery 收到服务候选后即释放候选额度，不受无关参与义务挡住；
普通 fill 的临时 FILLING 槽恢复 FREE，direct completion 只有真正 READY 才报告 refill。
Host 保留逻辑任务计数及 AIV ready 路由，但物理 subtask/trace 数为零，timing_slot=-1；
三处 ready transition 均跳过 inline，避免写到下一 kernel 的 trace cell。

补充长服务链每轮至多两项、Host 已完成节点不重做、重复完成和 resolve 失败不补成功计数，
以及 sync/SPMD/Mix/predicate/kernel/N0/N2/非 executable 八种畸形 metadata 拒绝。
累计 18 项全部通过，日志 `outputs/spmd-inline-services-green-20261025.log`。
完整 runtime 重编译与原样 CSA 的后续验收单独记录，不以 CPU helper 通过替代设备证明。

首轮重编成功、13 个相关 C++ targets 共 477 项通过；原样 CSA 在 Host prepare 返回 -1000，
尚未进入设备，也没有生成 deps。产物 `outputs/csa-hbg-l4-xaxog_8m/result.json` 记录失败。
核对发现 `scheduler_trace_count_checked` 有意拒绝 added=0，maker 在把 Dummy 的物理数改零后
仍调用它，误设置 trace_overflow_task_ref。该检查的真实 kernel 合同不放宽；
maker 对服务节点直接保持 trace 前缀，只有真实 kernel 调用非零计数与容量检查。
这条 Host 缺口由实际业务入口暴露，不能用 helper 测试通过掩盖。

修正后重新编译并执行同一条严格命令：

```bash
python examples/a5/host_build_graph/deepseek_v4_pro_csa/run.py \
  l4 -d 0 --require-aicore-scheduler
```

`outputs/csa-hbg-l4-fdakpyh4/result.json` 为 passed=true：deps/timing 两次各七项 golden 全过，
79 个有 kernel 任务、2308 条物理记录（AIC 864／AIV 1444）、单次正数 epoch、133 条依赖检查；
producer=aicore、distributed_scheduler_verified=true，Host capture 107/107，dropped=0。
严格门禁同时检查 scheduler 记录关联、stream 的 committed/dropped/truncated，均通过。
未改变 54 个 kernel、52 项 ABI、数据、golden、block 数或验证器标准。
`block_identity_verified=false` 仍如实保留：这份既有 schema 尚无显式 block_idx/subtask_slot，
也没有完整 order/claim/LocalDone/group 事件，不能把数量完整升级成全身份协议证明。

相关 CPU 回归 1427 passed（38.10 秒），包含协议、原生夹具、CSA 业务与真实快照、转换器，
日志 `outputs/spmd-gm-mix-python-final-20261025.log`；服务节点修复后的 HBG 仿真
42 passed、379 deselected（52.87 秒），日志 `outputs/spmd-inline-services-hbg-sim-20261025.log`。

随后完整业务入口的 12 个业务变体加 3 个严格 resident 变体，各三轮，共 45 次全部通过，
逐轮七项 golden（315 项）与新鲜调度发布检查保留，全部 scheduler_mode=1；
日志 `outputs/spmd-gm-mix-csa-business-resident-20261025.log`。
这包括页重排、scale 初值扰动、原 Worker/callable 反复运行；不改变冻结 workload。

### GM Mix 阶段提交前复验与剩余工作

所有适用 pre-commit 检查通过；首次 clang-format 只修正格式，重新暂存后复跑全绿，
未使用 SKIP 或 no-verify。格式化后完整 runtime 重编及上述 13 个 C++ targets 再次通过，
日志 `outputs/spmd-gm-mix-final-build-20261025.log`、`outputs/spmd-gm-mix-final-cpp-20261025.log`。
最终 GM 事件仿真再完成 44 场景／180 次调用，
日志 `outputs/spmd-gm-mix-final-events-sim-20261025.log`。
最终 A5 HBG 回归 60 passed、4 skipped、356 deselected（215.89 秒），
日志 `outputs/spmd-gm-mix-final-hbg-a5-20261025.log`。
仍显式排除已知会被 runtime 过滤误收集的双卡 TMR RDMA 目录，不把单卡 HBG 验证冒充多卡回归。

最终源码重编后的严格 CSA L4 再次通过：
`outputs/csa-hbg-l4-96e7ioam/result.json`、`outputs/spmd-gm-mix-final-csa-l4-20261025.log`。
独立核对原始产物为 2308 条执行、2308 条 scheduler 记录、96 个 aicore stream，
dropped 合计 0、truncated 全 false；两轮七项输出通过，完整身份缺口仍如实保留。
最终 A5 GM 事件复跑同样完成 44 场景／180 次调用、epoch 1–180，
日志 `outputs/spmd-gm-mix-final-events-a5-20261025.log`。

此阶段支持普通 C/V/Mix 原 N 分波和原样 CSA resident，但不是整个目标完成：

- 错误记录还须补 epoch/order/block/group_seq/紧凑 worker/lane，覆盖准备失败、首写者竞争和退出错误。
- 现有 L4 尚无完整原生协议事件；必须补显式身份、共同准入、claim、LocalDone、组发布／释放及严格 adapter。
- sync_start 尚未开放；后续须实现不同位置容量与每位置一次领取，早返回不能重复领块，不加全员起跑屏障。
- 同 kernel／输入／拓扑的无完整插桩性能比较尚未执行，不以 L4 或 CPU 模型推算收益。

后续继续沿真实 RED→生产修复→CPU／仿真／A5 验证推进，过程只更新本文件，阶段提交不自动 push。

## 2026-10-25：补齐首错误身份与退出诊断

起点 `6f2b470d4` 已完成阶段提交、未 push；提交后重编成功，CSA 入口确认
Torch 2.12、冻结源码一致、native binding 与提交一致，三个 runtime 二进制 SHA256
与最终严格 L4 相同。工作树只有原有四项用户未跟踪内容，无遗留后台测试。

原错误区只记录 task/core/site，48B 预留区清零；局部 admission 信息没有传入底层记录。
`scheduler_admission_fail` 进一步退化成 shared context，EXIT watchdog 使用无效 task，
AICPU 终态收敛虽然能返回错误，却不打印晚到错误的完整现场。
另一个确定缺陷是 AICore writer 对 OK 状态也占用 first-error claim，导致后续真正错误不能发布。

新增真实 C++ 测试 `test_hbg_error_identity.cpp`，最初直接读取现有预留区的 wire 字节，
没有先添加未实现接口来制造编译红灯。17 项中 16 项实际失败，布局控制项通过；
包括初始化／retained reset、epoch 0、各组阶段与 mask6 协调者、旧 task 残留、
实际 Mix prepare 失败、AICore/AICPU 双向首写者和 32 轮并发一致性读取。
日志 `outputs/spmd-error-identity-red-20261025.log`；初步补齐后 17 项全绿。

追加三项实际 RED 揭示：UINT64_MAX 也可为合法 epoch，需有效性位；
只有 shared context 时仍应保留已知 worker/lane；纯任务两个预填槽需要精确归属失败槽。
日志 `outputs/spmd-error-identity-boundary-red-20261025.log` 为 17 绿／3 红。
诊断仍复用 48B：epoch/order/block/group_seq/worker 各 64 位，lane 和有效性位各 32 位，
RunControl 仍为 384B，error_claimed/site 的原 offset 不变，新增字段 offset 静态断言。
未知字段保留全 1，并以独立 valid_fields 区分合法极值；epoch 0 不作未初始化哨兵。

共同准入和组的本地结构声明移到 ready 的局部状态区，布局不变，不新增运行期缓存或分配。
错误冷路径直接读取已经持有的状态；当前 task 不匹配时不套用另一义务的 order/group。
IDLE 只有 task_last_seq 明确归属当前 task 时保留最近已退役组；切到下一 task 后不使用旧组字段。
参数准备按实际 FILLING 槽记录 block，generation 和 block 完成错误也传递具体槽；
仅无共同准入上下文的普通 N1 不声称知道 epoch/order，已知 block 0 仍记录并标有效。

首错误 CAS 的唯一获胜者写全部身份，经 barrier 最后发布 scheduler_error；
AICPU 的首写者显式发布未知身份，后写者不修改已发布现场。OK 不占 claim。
AICPU 冷日志在观察发布状态后再次失效完整错误区，避免第二条 64B line 的旧预取数据；
bootstrap、execution、terminal 均可输出相同身份格式，正常路径不打印错误日志。

两项实际调用生产路径的 RED 也已建立：零工作 AIV 在 EXIT drain 等不到 AIC seal，
应定位当前 task/epoch/order/worker；真实 terminal finalizer 应打印晚到错误的全部已知字段。
`outputs/spmd-error-exit-red-20261025.log` 两个目标分别新增一项失败，既有控制项仍绿。
接线后 20 项身份检查、5 项真实 executor startup/EXIT、6 项真实 AICPU finalizer
共 31 项通过，日志 `outputs/spmd-error-exit-green-20261025.log`。
正在补普通 completion 和空上下文边界、完整回归与设备原语验证；尚未完成该阶段验收或提交。

补充检查发现 AICPU 首写者还会保留已污染的旧 task/core/type，
26 项中该项实际失败；统一清为未知后完整 15 个 C++ targets 共 510 项通过。
日志 `outputs/spmd-error-identity-aicpu-red-20261025.log` 与 `outputs/spmd-error-identity-cpp-20261025.log`。
完整 runtime 首轮重编也通过，日志 `outputs/spmd-error-identity-runtime-build-20261025.log`。

新增设备侧私有 GM 探针及 254 项 CPU oracle 检查，检查首写者、完整／部分身份、
epoch 0/MAX、保留区、每块观测和本轮 resident 来源。探针不写真实 scheduler 的错误区，
也不把构造的 worker 身份当作真实物理 placement；生产调用点的状态提取由 C++ 测试单独覆盖。
最初探针引入整个 ready 头，因用户 kernel include 路径没有私有 inner_kernel 时钟声明而编译失败，
两次失败分别为未声明 get_sys_cnt_aicore、找不到 inner_kernel.h，未进入执行。
没有增加假的时钟实现或修改公共 kernel 编译路径。

通用 first-error 原子发布提取到 scheduler_error.h，ready 保留本核状态提取与调用点封装，
设备探针直接复用这一个发布实现；不复制 writer，也不需要执行器的时钟／寄存器依赖。
读端等待采用有限次原子查询，观察发布后核对完整身份；后写者不得覆盖首写者。
仿真最小原生用例通过，日志 `outputs/spmd-error-identity-smoke-sim-r3-20261025.log`。
CPU 探针 oracle 254 passed，日志 `outputs/spmd-error-identity-native-fixture-20261025.log`。
设备 precheck 仍因缺少 npu-smi 退出，CANN 查询已确认设备 0 为 Ascend950PR_958b，
按用户要求直接使用本机卡、无 task-submit 锁，查询留存 `outputs/spmd-error-identity-device-20261025.log`。

通用发布入口提取后完整 runtime 重编通过，15 个相关 C++ targets 共 510 项再次通过，
另补跑实际退休接线 7 项通过，共 517 项；日志为 `outputs/spmd-error-identity-final-cpp-20261025.log`
和 `outputs/spmd-error-identity-retirement-20261025.log`。所有适用静态／格式检查通过。
首轮 44 个设备探针场景在仿真／A5 各三轮、132 次通过；
日志 `outputs/spmd-error-identity-matrix-sim-20261025.log`、`outputs/spmd-error-identity-native-20261025/matrix.log`。
HBG 仿真 42 passed、379 deselected（102.17 秒），A5 60 passed、4 skipped、356 deselected（157.83 秒）。
单卡 A5 仍显式排除已知误收集的双卡 TMR RDMA 目录；未以此声称多卡回归完成。
Python 合约／CSA 快照／转换器等首轮 1681 项通过，日志 `outputs/spmd-error-identity-python-20261025.log`。

原样 CSA 严格 L4 保持通过，`outputs/csa-hbg-l4-buvxzzb6/result.json`：
两轮各七项 golden、2308 条执行、133 条依赖时序、107/107 Host 构图记录，producer=aicore；
block_identity_verified=false 仍保留，既有 L4 不因此获得完整协议身份能力。

首轮共享地址场景不强制多个物理执行位置同时到达，故不能单凭它声称实际竞争一定发生。
追加两个 N=1、mask3 的 GM Mix 场景：同组 C/V 进入后在探针自有词上有界会合，
两方到达后才对同一私有错误区发布；检查到达数 2、每 lane 一次执行、赢家及其 lane 一致。
它不跨组等待，不打开 sync_start，不写真实调度器错误区。总计 46 场景，CPU oracle 增至 262 项，
仿真三轮 138 次通过，日志 `outputs/spmd-error-identity-matrix-sim-final-20261025.log`；
对应 A5 最终矩阵仍在复验，终态结果另记。

### 追加设备验证暴露只提取 text 的跳转表遗漏

46 场景的 A5 复跑实际失败在第一个原有单 AIC 场景（scenario0），没有到达新增 Mix 场景。
首次运行报告 AICore 异常和 507018，之后同一 Worker 已 poisoned，后续条目都是连带拒绝，
不能计成 46 次独立硬件失败，更不能把原先 132 次成功算作新版本通过。
日志 `outputs/spmd-error-identity-native-20261025/matrix-final.log`；现场 PID 1976630 的
device/plog 保存在该输出目录的 ascend/debug 下。框架自动 force reset 设备 0，
明确报告 probe confirmed clean 和旧 generation 已退休；未手工扩大复位范围，源码与 Host 数据保留。

保存并仅离线重编生成物后发现确定缺陷：
`outputs/spmd-error-identity-elf-20261025/error_identity_aic.cpp.incore_wn8losko.o`
的 .rela.text 在 offset 0x440 引用 JTI0_0；链接 ELF 的 .rodata 在 0x1000、24B，
.text 在 0x2000、1448B。六路条件被 CCEC 降为跳转表；当前提取器只带走 .text，
已链接的映像又没有剩余 .rela.text，于是遗漏必要 .rodata 仍被接受。
设备故障 PC 超出实际 callable code 范围，与读到错误跳转表后的非法跳转相符；
本机 llvm-objdump 对该指令集只输出 not available，未据此伪造指令级反汇编结论。

新增 Host 门禁 RED：非空 allocated .rodata/.rodata.* 必须拒绝，空／非 allocated 控制项允许；
硬件缓存键须随提取合同变化，不能继续复用已经丢弃跳转表的旧裸代码。
探针的 expected_fields 改由 Host 标量提供，保留独立 Host oracle，不改 frozen CSA 或生产调度语义。
该修复仍在验证；本阶段未提交、未 push，不以仿真通过替代新的 A5 复验。

### text 门禁修复后的重新验收

Host 门禁先获得 8 项实际 RED（其余 47 项控制测试通过），再实现 allocated
`.rodata`／`.rodata.*` 拒绝；空或未分配的只读段仍接受。硬件 incore 缓存键携带
text 提取策略版本，外层 callable 键随之变化；仿真缓存键不受影响。
这不是把完整 ELF 装载能力加入 runtime，也没有改 linker、kernel ABI 或冻结 CSA。
日志 `outputs/spmd-error-rodata-red-20261025.log` 与 `outputs/spmd-error-rodata-green-20261025.log`。

离线复核原失败 ELF 已被门禁拒绝；传入 expected_fields 标量后的 AIC/AIV
分别提取出完整的 1360B／1504B text。原样 CSA 的 54 个 kernel 在新门禁下预编译通过，
日志 `outputs/spmd-error-rodata-csa-precompile-20261025.log`。
恢复测试的单 AIC 场景三轮通过；随后使用 `--case lane2_*` 被 CLI 拒绝，
此入口只接受精确场景名，该次没有设备执行。改用两个完整 Mix 场景名后各三轮通过，
日志为 `recovered-mixed-zero.log`、`recovered-mixed-max.log`，均在上述 native 输出目录。

修正后的完整 46 场景在仿真／A5 各三轮、138 次调用全部通过：
`outputs/spmd-error-identity-matrix-sim-safe-20261025.log`、
`outputs/spmd-error-identity-native-20261025/matrix-safe.log`。
本轮 Python 合约、CSA 快照、转换器、ELF、编译缓存及 kernel compiler 回归 1768 项通过，
日志 `outputs/spmd-error-identity-python-safe-20261025.log`。
16 个相关 C++ targets 再跑全部通过，包含 517 项具体检查，
日志 `outputs/spmd-error-identity-cpp-safe-20261025.log`。

新 text 门禁下原样 CSA 严格 L4 再次通过：`outputs/csa-hbg-l4-vtupzkkq/result.json`。
两轮各七项 golden，2308 条执行（AIC 864、AIV 1444）、133 条依赖时序，
107/107 条 Host 构图记录，producer=aicore；冻结 case 与基线一致。
`block_identity_verified=false` 没有放宽，完整原生协议事件仍是下一项实现工作。

用户随后明确要求推送已有提交，`6f2b470d4` 已核对远端 SHA 并推送到
`nalinaly/simpler` 的 `feat/fully_distributed_within_core`；本节未提交改动没有随之推送。
后续恢复阶段性中文 commit、不自动 push 的原目标，最终 HBG 回归及提交检查结果续记于此。

最终 HBG 仿真 42 passed、378 deselected（140.49 秒），A5 60 passed、4 skipped、
356 deselected（288.94 秒）。仿真也显式排除相同双卡 TMR 目录，故比前一轮少一个 deselected，
不是丢失 HBG 测试。日志 `outputs/spmd-error-identity-hbg-sim-safe-20261025.log` 和
`outputs/spmd-error-identity-native-20261025/hbg-safe.log`。本机依旧直接用卡，无 task-submit 锁。

补核对四份保存 ELF 时，最初误把原 AIV 也列为应拒绝，检查脚本因此失败；
readelf 确认原 AIV 无 allocated .rodata，1640B text 可接受。修正的是离线核对预期，
不是放宽门禁：原失败 AIC 拒绝，原 AIV 及修正后 AIC/AIV 接受。
C/V 后端对同一条件表达式的生成物不相同，不能把 AIC 的跳转表结论直接套给 AIV。

最终提交检查全部适用项通过，日志 `outputs/spmd-error-identity-precommit-final-r2-20261025.log`。
首轮 ruff 仅重排探针 CASES 表达式，已比较格式化前后 AST 相同；
格式化后探针 oracle、ELF、缓存、compiler 共 341 项再次通过，
日志 `outputs/spmd-error-identity-postformat-20261025.log`。未跳过检查，未改生产 C++ 或 kernel。
本次阶段提交涵盖首错误身份、确切槽定位、AICPU 冷日志、原生探针和 text 门禁及其缓存失效；
不含用户未跟踪的原始泳道、TMR CSA、atomic_probe，不改 frozen kernel／golden。
下一项仍是显式执行身份与完整原生协议 trace，之后实现 sync_start 和同配置性能比较。

## 2026-10-25：接通显式执行身份

起点 `ae6474adf` 已完成详细中文提交、未 push。提交后重编成功，binding 版本与 HEAD 一致，
三份 runtime 二进制 SHA256 与 `csa-hbg-l4-vtupzkkq` 的已验收运行相同，冻结源码仍一致。
本阶段继续实现完整 trace 的执行身份部分；准入／claim／LocalDone／组释放等原生事件仍需随后接线，
不能用身份完整替代全协议审计。

沿用 DFX 的 runtime extension 机制，保留原七列执行行及五列 scheduler 行，
增加可选 `aicore_execution_identities` 对象，以 epoch/core/reg_task_id 精确连接。
新测试对当前真实转换器获得 48 红／1 绿：原实现忽略该对象，缺字段、错绑、重复身份均可漏过。
日志 `outputs/spmd-execution-identity-red-20261025.log`；严格读取后 49 项通过。

设备在真实 kernel 入口、调用前采集 payload 的原 N 与 block，连同 logical slot、
pending slot、generation、epoch、order、group、cluster/lane 和是否实际调用 kernel。
使用 SchedulerTaskTrace 最后一条 64B line 的预留与尾部对齐空间，大小保持 384B；
不增加本核 DAG、槽数量、运行期分配或新的功能开关。L0 不采集，predicate 跳过显式标记 executed=false。
记录先发布到 GM，既有 completion 再发布 trace.valid；导出不从行号推导 block。

首次编译因新 helper 放在 scheduler_trace_index_at 定义之前失败；移动到实际依赖之后，
没有补假的接口。第二轮 runtime 与相关 C++ 编译通过，日志分别为
`outputs/spmd-execution-identity-runtime-build-r2-20261025.log` 与
`outputs/spmd-execution-identity-cpp-build-r2-20261025.log`。

新增 12 个真正调用 executor 循环的 CPU 参数用例：AIC/AIV0/AIV1 × L0/1/2/4，
测试 kernel 主动改写 payload 为 block=77、N=99，仍要求 trace 保留入口 0/1，防止事后采样假绿。
第一次执行在 AIV 用例 SIGSEGV；gdb 定位到测试 kernel，原因是夹具未填写 cluster_worker_ids，
其测试指针使用了无效 lane 映射。修正夹具并用 20 秒进程超时复跑，17 项（旧 5＋新 12）全部通过，
日志 `outputs/spmd-execution-identity-entry-gdb-20261025.log`、`outputs/spmd-execution-identity-entry-r2-20261025.log`。
没有因此改变生产调度路径。

CSA 图审计新增 6 项实际 RED，旧实现即便有显式身份仍一概返回 false，且不拒绝同数量的重复 block、
错 slot、错 N、部分缺字段或未执行行。严格 resident 入口另有 2 项 RED，确认仅 producer=aicore
不足以满足新的身份验收；已有 --require-aicore-scheduler 现要求身份验证为 true，不新增 CLI 开关。
历史不带身份的 capture 仍可读取，但保持 block_identity_verified=false。
相关 parser／CSA／converter 456 项通过，日志 `outputs/spmd-execution-identity-reader-regression-20261025.log`。

最小 mask3 N1 仿真和 A5 都通过，并逐项核对各 5 条真实导出身份；输出分别为
`outputs/TestHbgSpmdContract_mask3_n1_20261025_134301`、`outputs/TestHbgSpmdContract_mask3_n1_20261025_134305`。
原样 CSA 严格 L4 通过：`outputs/csa-hbg-l4-y2ihg7u5/result.json`，两轮各七项 golden，
2308 条执行的 task/block/slot 集合与冻结图完全一致，block_identity_verified=true，producer=aicore。
这仍不是完整原生准入与组协议事件的证明；本阶段尚未完成全部回归或提交。

读端补充 Mix 四种 mask、epoch 0/MAX、跨 epoch 重用连接键、显式 null 与历史缺字段的区别，
共 63 项检查通过。Perfetto 的 Worker/Scheduler 视图起初未带新字段，两个测试实际 RED；
保留执行身份到两个视图的 args 后转绿，未更改时间、flow 连接键或普通七列／五列格式。
共享采集器另验证新 extension 的对象根、单次写入、封存后独立导出、下一 run 不继承。
7 个共享 targets 共 125 项通过；16 个调度 targets 共 529 项通过，相关 Python 首轮 1754 项通过。
日志 `outputs/spmd-execution-identity-common-green-20261025.log`、
`outputs/spmd-execution-identity-final-cpp-20261025.log`、`outputs/spmd-execution-identity-final-python-20261025.log`。

90 场景两轮功能矩阵在仿真和 A5 各 180 次通过，但 runner 明确记录
`chip swimlane disabled: --rounds > 1`，该结果只能作为无泳道功能回归。
离线核对预期产物时发现没有文件，随后改用单轮采集，不修改 runner 的现有规则。
单轮 L4 仿真 90 场景通过，逐文件读取共 26303 条显式身份，
日志 `outputs/spmd-execution-identity-l4-matrix-sim-20261025.log`；对应 A5 单轮矩阵仍待运行。

对 `csa-hbg-l4-y2ihg7u5` 重新转换为 `L4-identities.json`：Scheduler View 有 2308 条，
Worker View 共 2334 条中的额外 26 条是 Host alloc 标记，不是 kernel。
初次离线脚本把所有 Worker event 都当成 kernel，计数断言失败；核对名称、phase 与
kernel-duration-us 后，两视图均有 2308 个唯一 kernel 身份，非 kernel 标记不补造 block。
最终 HBG 仿真 42 passed、378 deselected（145.79 秒），A5 60 passed、4 skipped、356 deselected
（294.01 秒）；两者均排除已知误收集的双卡 TMR 目录，未声称已验证多卡。

格式化与静态检查发现两个新校验函数的分支数超限，已按字段检查／连接一致性、
执行身份／依赖时序拆开，不用 noqa 绕过。格式化后 runtime 重编、5 个受影响 C++ targets
及 Python 1761 项通过；该轮之后又增加 5 项身份配对反例，下述最终回归另计。
单轮 A5 L4 的 90 场景通过，`outputs/spmd-execution-identity-native-20261025/l4-matrix.log`。
仿真／A5 各 90 份文件、26303 条身份，目录不重合；按原测试计划精确核对 task/block/slot 集合、
原 N、普通与协同任务的 order/group 空值边界、Mix 同 block 同 cluster/group，以及各 scheduler
stream 无 dropped/truncated。不是只检查身份行数等于普通执行行数。

最终原样 CSA L4：`outputs/csa-hbg-l4-k7xbr0k2/result.json`，两轮各七项 golden、2308 个
唯一身份、producer=aicore、block_identity_verified=true；本次直接生成的 L4.json 在两个
执行视图均保留 2308 个唯一身份。另外，CLA 原图形状的 7 个探针场景在仿真／A5 单轮 L4 通过，
各 2746 个身份与其计划逐项匹配；这是 CLA 调度形状探针，不是原始 CLA 数学 kernel。
日志 `outputs/spmd-execution-identity-cla-sim-20261025.log` 和上述 native 目录的 `cla.log`。

审计器补充 5 项实际 RED：同一 Mix block 的 cluster/group_seq 不一致、缺一条／全部 group_seq，
以及多 block 缺 order_key。补齐跨 lane 同组检查和必需 order 后，68 项读端检查通过，
与 CSA、转换器合计 475 项通过；日志 `outputs/spmd-execution-identity-pairing-red-20261025.log`、
`outputs/spmd-execution-identity-pairing-green-20261025.log`。加严后的读端重新接受上述所有真实采集，
并重新通过 CSA 严格入口。完整协议事件仍未采集，身份配对验证不冒充准入／LocalDone／释放审计。

适用提交检查全部通过，日志 `outputs/spmd-execution-identity-precommit-final-20261025.log`。
本阶段只增加 DFX 身份采集与验收，不修改冻结 CSA、调度顺序、block 领取或通信协议；
后续仍须实现完整原生协议事件、sync_start 和同配置性能比较，不自动 push。

最后一轮合约／CSA／转换器／显式身份 Python 回归 1766 项通过，
日志 `outputs/spmd-execution-identity-final-r2-python-20261025.log`。本次阶段提交包含入口快照、
共享 DFX extension 的透明传递、严格连接与配对检查、两视图展示、CSA 身份门禁和相应 TDD；
不包含用户未跟踪的 atomic_probe、原始泳道或 TMR CSA。历史结果不改写为新的通过证据。

## 2026-10-25：完整协议采集的存储与准入／局部关闭接线

起点为 `fb413ad12`，本阶段不 push。完整协议采集仍在实现，不能把本节的局部事件
或 CPU helper 测试当作原样 CSA 的全协议审计通过。

Host 布局增加可选的协议记录区，使用已有 trace 索引空间以及实际 task、worker、
协同任务数量规划：task 的准入／完成两条独立 64B line、逐成员观察／关闭记录、
block owner／三 lane 记录、kernel 执行记录和逐 worker 序号／终态。
没有全局 trace 原子计数器，没有每核一份全 block 历史，没有 Scalar 动态分配。
GraphSummary 复用 padding 存放协议区 offset，仍为 128B；本核常驻状态仍为 504B，
仅可选 profiling 状态从 240B 增至 248B。Host 生产入口尚未开启该区，
后续须接完事件和导出审计后复用既有 L4 配置，不增加新的用户功能开关。

新增布局测试覆盖 3／63／66／96／108 个 worker、精确大小与互不重叠、epoch 0/MAX、
旧缓冲区重用清零、缩小运行不暴露旧尾部、乘积／总长溢出和关闭诊断时保持原布局。
首轮 5 项实际 RED 暴露 Summary 的 protocol offset 要在布局初始化时发布，而不能等到
后续任务记录初始化；修复后该测试目标共 114 项通过。
日志 `outputs/spmd-protocol-layout-red-20261025.log`、`outputs/spmd-protocol-layout-green-20261025.log`。

真实准入 helper 在短锁内记录成功发布的前／后区间；锁竞争 RETRY 不生成虚假事件。
成员首次观察记录实际关键读得到的 commit_tail，重复 poll 不重复记事件。
普通 LocalDone 记录实际已耗尽 cursor 和本核未退役槽位；GM Mix 的 AIC 记录其实际 cursor
与写出的 seal，AIV 记录读到的 seal，cursor 用 UINT64_MAX 表示未观察，不从 N 伪造观察值。
这些记录均发生在清除本核义务之前；零工作成员无需等 TaskDone。
发布者的结束时间可能晚于消费者观察时间，因此保留区间与每核局部序号，
后续审计不能简单按全局时间排序推断因果。

新增准入测试先得到 4 项 RED，接线后转绿；GM 零工作 cluster 关闭测试另有 1 项 RED 后转绿。
还覆盖了不同发布者共同定序、epoch 0/MAX、worker 65 与物理核 ID 区分、重复记录不覆盖、
序号不回绕、坏布局／残留 worker 状态拒绝、关闭采集不写协议区等。
诊断错误用单写者 sticky flags 保存，不伪造成功记录，也不因采集失败改写正常调度完成数。
真实 executor 的公共退出尾部刷新 worker 终态，成功与失败分开；测试实际调用该 executor，
确认晚启动 C/V 零工作成员先完成全部本核关闭，再 ACK；已有错误或 seal 超时保留不完整状态。

16 个相关 C++ targets 共 571 项通过，日志 `outputs/spmd-protocol-all-cpp-20261025.log`。
首轮 runtime 重编未通过：AICore 链接报告 `__multi3` 未定义，不能据此运行旧二进制作新代码验收。
最小 CCEC 探针 `outputs/spmd-protocol-arithmetic-probe-20261025.cpp` 的 ELF 重定位显示：
唯一 `__multi3` 调用位于动态 `tasks > UINT64_MAX / workers` 溢出检查函数；
固定 640B 除法以及先验证各操作数上界的函数没有该调用。
修正为检查 task 数不超过 INT32_MAX、worker 数不超过既有容量，再计算乘积；
Host 的通用 checked arithmetic 保留。修正后的 runtime 重编及受影响测试正在执行，结果续记。
格式化使用仓库 pre-commit 固定的 clang-format 21；系统版本 18 的额外换行已由规范版本纠正。

尚待接线：block claim／完成票号、组发布／到达／释放、kernel entry／return／输出可见／退役、
逻辑 TaskDone 和实际 ordinary entry permit、Host 导出、完整原生因果审计及严格 CSA 门禁。
本节没有完成 sync_start，也没有形成关闭完整 trace 后的性能结论。

修正后的 runtime 重编通过，日志 `outputs/spmd-protocol-events-runtime-build-r2-20261025.log`；
AIC 合并对象不再含未定义 `__multi3`。重新构建的 16 个相关 C++ targets 共 571 项通过，
Python 合约／CSA／转换器／身份检查共 1766 项通过，日志分别为
`outputs/spmd-protocol-all-cpp-r2-20261025.log`、`outputs/spmd-protocol-python-20261025.log`。
本阶段相对上一提交增加 42 项 C++ 检查，不将完整测试目标的数量当作全是新增用例。

HBG 仿真 42 passed、378 deselected（52.76 秒），A5 60 passed、4 skipped、356 deselected
（150.29 秒）。日志 `outputs/spmd-protocol-hbg-sim-20261025.log` 和
`outputs/spmd-protocol-native-20261025/hbg.log`。按当前 CI 配置，仿真 session timeout 为 600 秒，
onboard 为 1200 秒；两者显式排除已知误收集的双卡 TMR RDMA 目录，不宣称多卡覆盖。
架构预检查脚本依旧因本机没有 npu-smi 退出；CANN query 确认 Ascend950PR_958b、32 C／64 V，
按用户明确要求直接在设备 0 运行；本机没有 task-submit，不宣称有该队列的隔离保障。

原样 CSA 严格 L4 回归：`outputs/csa-hbg-l4-oktn_ne5/result.json`，两轮各七项 golden、
2308 条执行身份、133 条依赖边、107/107 Host 记录、producer=aicore、
block_identity_verified=true，冻结 case 与基线一致。
这里使用的是新编译 runtime，但 Host 尚未分配协议区；因此证明现有 CSA 路径未回归，
不证明上述新增事件的设备采集或完整协议审计。后者必须在剩余接线和导出完成后另行验收。
本次阶段提交只固化已通过检查的记录基础和接线，不包含输出产物或用户未跟踪文件。

适用 pre-commit 检查全部通过，日志 `outputs/spmd-protocol-precommit-20261025.log`；
未跳过 clang-format、clang-tidy、cpplint 或文档检查。此后只补充本段验收记录，未改变已测源码。

## 2026-10-25：block 与 GM 组协议事件接线

起点 `60e0e37e7` 已详细中文提交、未 push；提交后重编成功，binding commit 与 HEAD 一致，
三个 runtime SHA256 均与 `csa-hbg-l4-oktn_ne5` 的通过产物相同。本节继续完整协议采集，
Host 生产入口仍未启用协议区，既有执行身份不能替代这里尚未完成的设备事件验收。

真实 cursor CAS 成功后记录领取前／后时间和唯一 owner；失败重试及 EXHAUSTED 不生成事件。
时间在实际 CAS 两侧读取，序号只在成功领取后分配，不为失败尝试制造序号洞。
block 完成记录真实 completed_blocks 原子的返回票号，不能用 block_idx 充当完成顺序。
纯 C/V 使用确切 slot 的 block，GM 使用 AIC 当前组的 block；记录区越界仅留下诊断错误，
不越界写入，不篡改正常调度计数。

GM 发布前后包住 descriptor 写回和 published_seq 原子发布。
AIC 记录自己发布后的 descriptor 就绪，AIV 在实际观察、校验 descriptor 后记录就绪；
各 lane 保存实际 descriptor 的 epoch/task/order/group/block/cluster/N/mask/placement，
不是从最终 kernel 数量还原。arrive/release 各记录实际原子发布的前后区间。
全体 ack 检查保存实际三次关键读的数值以及观察事件；旧值或未来值导致等待／错误时，
不生成“收齐”证据。相同有效 ack 被重复检查不重复生成事件。

为了保留独立的 descriptor／ack 观察点和三条原始 ack，lane 诊断记录由 128B 增为 256B，
整个 block 记录由 640B 增为 1024B；owner 复用原预留区，仍为 192B。
各写者依然分开 cache line，block 记录仍按实际 trace 索引空间一次性预分配，
没有每核全图副本、全局 trace 原子计数器或 Scalar 内存分配。
记录区尚未对外导出，因此没有把旧捕获按新布局强行解析；Host 和设备须一起重编。

最初 6 项新增检查实际 RED（纯 C/V 两项、四种 Mix mask 各一项），接线后转绿。
后续补充旧 ack／跨组复用、不同 cluster 相同 group_seq、UINT64_MAX−1 组序号、
重复发布保护、越界 trace prefix、并发 cursor 的唯一 owner 和连续局部序号。
并发测试只验证真实 cursor 原语，不把直接批量调用 cursor 当作两个 payload 槽的完整执行循环。
已有未来 arrive/release ack 两项测试也开启采集并断言没有伪造成功观察点。

相关 16 个 C++ targets 首轮通过 582 项；追加并发检查后 ordered target 为 46 项通过。
日志 `outputs/spmd-protocol-block-red-20261025.log`、`outputs/spmd-protocol-block-all-cpp-20261025.log`、
`outputs/spmd-protocol-block-concurrent-20261025.log`。
runtime 重编通过，日志 `outputs/spmd-protocol-block-runtime-build-20261025.log`。
下一步仍是 kernel 入口／返回／输出可见／退役、TaskDone 和普通 entry permit，
然后完成 Host 导出与严格因果审计；不以本节的结构和 CPU 测试宣布完整协议验收。

追加并发 cursor 检查后，最终 16 个 C++ targets 共 583 项通过，相对 `60e0e37e7` 新增 12 项，
日志 `outputs/spmd-protocol-block-final-cpp-20261025.log`；Python 1766 项通过，
日志 `outputs/spmd-protocol-block-python-20261025.log`。
HBG 仿真 42 passed、378 deselected（54.05 秒），A5 60 passed、4 skipped、356 deselected
（153.27 秒），日志 `outputs/spmd-protocol-block-hbg-sim-20261025.log` 和
`outputs/spmd-protocol-block-native-20261025/hbg.log`。仍显式排除双卡 TMR 误收集项。
原样 CSA 严格 L4 `outputs/csa-hbg-l4-_x6yab27/result.json` 通过两轮各七项 golden、
2308 条身份、133 条依赖边，producer=aicore、block_identity_verified=true，冻结 case 未改。
架构仍由 CANN query 确认 Ascend950PR_958b；依用户要求直接使用本机设备 0。
这些 native 结果是新增协议区未启用时的回归证据，新增事件值目前由真实 C++ helper 检查，
不能据此宣布 GM 协议事件已经完成 A5 采集验收。

后续接入 completion 事件时还须区分实际代码中的 block 计数贡献、槽清理、trace 发布：
纯 C/V 当前先贡献 completed_blocks，再清理本核槽和发布 trace，最后推进逻辑完成。
不能把这些不同位置伪装成同一个时间点；native 审计必须利用实际票号、发布区间和可见性链，
不能为了满足参考模型的单一全局时间顺序而补造事件，或引入等待所有核 trace 收尾的全局屏障。

适用 pre-commit 检查全部通过，日志 `outputs/spmd-protocol-block-precommit-20261025.log`。
本次阶段提交包含上述 block／GM 接线及 12 项新增检查，不包含输出产物或用户未跟踪内容，
不启用 Host 协议区，不新增用户配置开关，不 push。后续 kernel／TaskDone 接线仍须完成。

## 2026-10-25：执行许可、kernel 阶段与逻辑完成接线

起点 `9d5abf944`，继续完整协议采集，Host 协议区仍未启用。
TaskDone 覆盖普通无准入上下文、协同任务、无 kernel 服务节点三条真实完成路径，
区间包住 DONE 的实际发布；普通／服务节点的 order 为未知哨兵，不挪用无关当前义务。
服务节点不产生执行身份、payload 或 kernel 事件。新增／加严测试取得 8 项实际 RED 后转绿，
日志 `outputs/spmd-protocol-taskdone-red-20261025.log`、`outputs/spmd-protocol-taskdone-green-20261025.log`。

为让 ready 内联服务节点和 completion 共用 recorder，低依赖记录 helper 通过模板接受现有
本核状态；不移动调度状态、不新增构图或分配职责、不引入头文件循环依赖。
既有 scheduler_cycles 原样下移到该公共记录层，时钟来源不变。
执行记录扩至 256B，保存最终 payload 发布快照、entry permit、kernel 入口／返回／输出可见、
完成代次消费、既有 trace.valid 发布，以及对应身份、来源和 publication mode。
GraphSummary／RunControl／SchedulerTaskTrace 及本核调度状态大小不因此改变。

普通许可保存实际 commit_tail 关键读的值；入口不重新读 tail 来改写已经授予的许可。
无准入日志的运行明确标记 NO_LOG，不伪造原子读；协同许可保存实际当前 order／group。
kernel 入口在调用前保存 payload 的原 N 和 block，返回点与既有输出屏障后的可见点分开采样。
predicate 跳过只记录 dispatch／许可／消费，不伪造 kernel 返回或输出可见事件。
完成代次的消费记录位于真实 mark_completion_consumed 之后；direct refill 覆盖槽后，
旧 trace 发布仍使用先保存的 task/block/subtask/generation 定位，不挪用后继的身份。

真实 executor 的三个 L4 lane 检查先实际 RED：kernel 会把 payload 改成 block=77、N=99，
记录必须保留调用前 0/1，并区分上述阶段。日志 `outputs/spmd-protocol-execution-entry-red-20261025.log`。
纯记录 helper 与实际调用 kernel 的包装分开；包装使用平台原有 OUT_OF_ORDER_STORE_BARRIER，
A5 仍是 dsb(DSB_DDR)。首次扩展到无 inner_kernel 的纯 CPU 测试时确实遇到屏障宏未声明，
随后把设备执行包装放入独立头文件，不用 host 空屏障替代设备语义。
两个调度夹具在开启采集时调用同一生产包装执行真实 no-op 函数，
不是通过手填 entry／returned 事件让完整链看起来存在。

payload 发布快照的首版被合法“预填→让槽→重建”反例击穿：
`outputs/spmd-protocol-execution-cpp-r4-20261025.log` 中一项失败。
普通预填尚未获得 entry permit，因此可被重建，不能把两次准备判作两次执行。
修正后只保留最终将进入 kernel 的实际发布区间、slot/generation、来源和发布次数；
这类可撤销快照不占用不可撤销事件序号，避免覆盖快照造成序号洞或伪造时序。
获准后的快照禁止覆盖；已领取的协同 block 禁止借此重复发布。
后一个限制另取得实际 RED，见 `outputs/spmd-protocol-execution-cooperative-republish-red-20261025.log`。
predicate 缓存仍被保留，重建不重新求值；两个预填槽没有共用一次旧许可。

额外检查覆盖：许可之后新准入不能改变已读 tail、重复返回保留首现场、
无效 pending slot 不访问越界、退出后不得写快照，以及真实 executor 中 direct refill
的旧完成代次与新 kernel 代次分离。准备两个任务的测试图曾使用容量 1 的夹具类型，
在运行前核对定义后改为容量 2，没有放宽生产图容量。
最终首轮 16 个 C++ targets 共 591 项通过，日志 `outputs/spmd-protocol-execution-all-cpp-20261025.log`。

设备首轮重编失败于 TaskDone 的 GM 引用传给要求 Scalar 本地引用的形状 helper，
日志 `outputs/spmd-protocol-execution-runtime-build-20261025.log`。
修正为先读取 16B SchedulerTaskMetadata 本地快照再调用原 helper，不改判定规则；
修正后的 runtime 与 C++ 重编正在执行，结果续记。本节尚未完成新增事件的 A5 采集验收。

第二轮 CCEC 仍拒绝从 GM 隐式复制构造整个 POD；CPU 16 个目标仍全部通过，
说明普通 C++ 编译并不能验证 CCEC 地址空间规则。
按已有 scheduler_load_dispatch_metadata 的逐字段读取方式，
只把判定需要的 flags 和 logical_block_num 读入本地形状快照，再复用原判定函数。
失败日志 `outputs/spmd-protocol-execution-runtime-build-r2-20261025.log` 保留，不用旧二进制补报通过。

第三轮 runtime 重编通过，日志 `outputs/spmd-protocol-execution-runtime-build-r3-20261025.log`。
修正后重新构建的 16 个 C++ targets 共 591 项通过；相对 `9d5abf944` 新增 8 项，
另外加严了原有 executor、Mix 和预填让槽用例，未把修改过的原有检查重复计作新增。
Python 合约／CSA／转换器／身份共 1766 项通过，日志分别为
`outputs/spmd-protocol-execution-final-cpp-20261025.log`、`outputs/spmd-protocol-execution-python-20261025.log`。

HBG 仿真 42 passed、378 deselected（50.91 秒），A5 60 passed、4 skipped、356 deselected
（153.18 秒），日志 `outputs/spmd-protocol-execution-hbg-sim-20261025.log` 和
`outputs/spmd-protocol-execution-native-20261025/hbg.log`。仍排除已知双卡 TMR 误收集项，
设备由 CANN query 确认 Ascend950PR_958b；本机无 npu-smi／task-submit，按用户要求直接用卡。
原样 CSA 严格 L4 `outputs/csa-hbg-l4-fhlpnykf/result.json`：两轮各七项 golden、
2308 条身份、133 条依赖边、107/107 Host 记录，producer=aicore、block_identity_verified=true，
冻结 case 与基线一致。两个新头文件在这次验收前加入索引，因此 provenance 的
git diff HEAD 源码指纹包含它们；没有通过提交／索引状态掩盖采集代码。

这些 native 结果仍是 Host 协议区未启用时的回归；新增执行事件目前由实际 CPU executor
和生产包装检查，不能声称已完成它们在 A5 上的整链采集。
后续直接接通既有 L4 下的 Host 预分配、原生导出和严格因果审计，然后完成 sync_start 与性能对比。
本阶段不 push，不改变既有完成驱动依赖推进，也不增加全核完成屏障。

适用 pre-commit 检查全部通过，日志 `outputs/spmd-protocol-execution-precommit-final-20261025.log`。
阶段提交只包含本节接线、测试与统一过程记录；不包含输出产物或用户未跟踪内容。

## 2026-10-25：Host 冻结图与原生协议导出

起点 `6416d5c08`，继续完整协议采集。新增 `AicoreProtocol` 对象扩展，沿用既有
ChipSwimlaneCollector 的 run-owned 快照和 JSON 写出，不另建采集器。
导出用例先在 writer 未接线时失败：sealed run 丢失 `aicore_protocol`；补齐后
HBG／TMR 两种导出目标通过，下一轮不继承旧对象，旧枚举值保持不变。
日志 `outputs/spmd-protocol-export-common-red-20261025.log`、
`outputs/spmd-protocol-export-common-green-20261025.log`。

Host 在既有 L4 下启用协议预分配，同时保存按实际任务数和依赖边数分配的冻结图，
内容为原 TaskMetadata 和原 fanin，不保留会被下一轮改写的 orchestration mirror 指针。
L0–L3 不分配新增协议区。冷路径下载后先校验本轮 epoch、完整长度、对齐、协议头与
Host 计划及 GraphSummary 的一致性，再从受检区域导出；成员 bitmap 必须属于自己的
record，seal 必须落在计划的 Mix 区内，不能跟随坏 offset 任意读取。

`hbg-native-protocol-v1` 保存冻结图、实际拓扑、完整准入日志、成员位图与 seal、
TaskControl 终态、逐 worker 序号／错误／退出状态，以及各实际记录点。
Stamp 使用 `[cycles, sequence]`，保留 cycle 0、epoch 0/MAX 和未知字段原始哨兵；
未写入的全零事件对象可省略，部分写入、发布次数非零或错误状态不能被过滤成成功。
输出没有 `protocol_verified` 之类的完成声明，序列化不替代后续因果审计。

新增 Host 导出测试首轮因缺少实现头文件编译 RED；实现后的行为测试又发现 4 项失败：
通用区域检查误将合法 `RunControl` 起始 offset 0 当作无效地址。
修正为检查范围和类型对齐，协议是否启用另由 layout 字段判断，12 项全部通过。
覆盖冻结图复用、Host 已完成节点不伪造事件、epoch 边界、损坏 header／summary、
截断和错位缓冲、丢失任务、错绑 bitmap、越界 seal，以及保留原始异常记录／实际拓扑。
日志 `outputs/spmd-protocol-export-red-20261025.log`、
`outputs/spmd-protocol-export-cpp-r1-20261025.log`、
`outputs/spmd-protocol-export-cpp-r2-20261025.log`。

包含两架构／两 runtime collector、两种 run export 和新 Host exporter 的 7 个 C++ targets
全部通过，日志 `outputs/spmd-protocol-export-common-cpp-20261025.log`。
生产 runtime 正在重编；本节此时还没有启用协议后的 A5 捕获证据，后续结果续记。

首轮重编成功。A5 `mask7_n2` L4 golden 通过，真实协议对象含 96 个正常退出 worker、
192 条成员记录、4 条 block、13 条执行，scheduler／worker 错误均为零；产物
`outputs/TestHbgSpmdContract_mask7_n2_20261025_180150/chip_swimlane_records.json`。
90 项仿真矩阵也完成，逐份确认存在协议对象，共 26303 条执行、全部 worker finished=1
且 error_flags=0；日志 `outputs/spmd-protocol-export-sim-matrix-20261025.log`。
这些只是实际采集和终态完整性的观察，不是完整因果审计结论。

CSA 首轮失败发生在 L4 的 Host prepare，dep-gen 那一轮七项 golden 已通过；
`outputs/csa-hbg-l4-2igazvbf/result.json` 如实记录 passed=false。
检查发现冻结图把“无 kernel 但使用 AIV 路由的 INLINE 服务节点”误当成普通 kernel，
其 active_mask 为 2 并不意味着有 AIV 执行。补充真实图／metadata 形状反例，
实际 RED 抛出 `invalid frozen protocol graph`；允许已标记 INLINE 的服务节点，
保留其原 flags／路由／依赖且不伪造 kernel，13 项 Host 导出测试全部通过。
日志 `outputs/spmd-protocol-export-inline-red-20261025.log`、
`outputs/spmd-protocol-export-inline-green-20261025.log`。修正后的 runtime 正在重编。

修正后 runtime 重编成功，原样 CSA 严格 L4 通过，产物
`outputs/csa-hbg-l4-am5bizn1/result.json`：两轮各七项 golden、2308 条执行身份、133 条依赖边、
107/107 Host 记录、producer=aicore，冻结 case 与基线一致。
这一轮实际启用了协议区，不再是关闭新增采集的回归。
原始数据在 `outputs/TestDeepSeekV4ProCSAHBG_B4S2Seed0_20261025_180836/chip_swimlane_records.json`。

新增统一读端 `simpler_setup/tools/hbg/native_protocol.py`，严格核对对象字段、整数类型、
表长度、稀疏索引和 Stamp 格式，保留失败／部分记录供诊断。
独立的 `audit_protocol_sequences` 检查 epoch、worker 身份、错误和退出状态，
按每核 sequence 检查唯一、连续、与 next_sequence 一致及本核时钟单调；
不按全局时间排序，不为可重建 payload 的发布快照补 sequence。
其结果只标记 `sequence_integrity_verified`，不代表完整协议因果或执行集合已获证明。

读端测试先因缺少模块实际 RED，补实现后 58 项通过，覆盖所有根字段缺失、未知 schema、
布尔冒充整数、越界／重复行、坏 Stamp、序号洞／重复／截断、跨 epoch、错误和未结束 worker，
并验证 cycle 0、epoch 0/MAX 及诊断读取不改变输入。
日志 `outputs/spmd-protocol-reader-red-20261025.log`、`outputs/spmd-protocol-reader-green-20261025.log`。
读取真实产物后：90 份仿真捕获共 427122 个事件，90 份 A5 捕获共 459234 个事件，
每组各 26303 条执行；CSA 另有 46733 个事件／2308 条执行。全部通过逐核序号检查，
没有以执行结果填补缺事件。A5 证据 `outputs/spmd-protocol-export-native-20261025/sequence-audit.log`。

17 个相关 C++ targets 共 604 项通过（此前 591 项加本次 13 项），
Python 读端／合约／CSA／转换器共 1824 项通过。
日志 `outputs/spmd-protocol-export-all-cpp-20261025.log`、
`outputs/spmd-protocol-export-python-r2-20261025.log`。
更宽的 A5sim 全 runtime 命令退出 0，覆盖其资源调度子进程；最终 pytest 的 deselected 数
不是通过数，未把它计入 90 份协议矩阵或虚构为 scene pass 数。
日志 `outputs/spmd-protocol-export-all-sim-20261025.log`。
A5 全 runtime 命令因单卡池无法容纳双卡 `TestUrmaDeferredCompletionDemo` 在资源检查阶段退出，
日志 `outputs/spmd-protocol-export-native-20261025/all-runtime.log`；没有把它算作通过，
另行执行相关 HBG 回归，不宣称双卡硬件覆盖。

最终 A5 HBG 回归为 60 passed、4 skipped、356 deselected（148.87 秒），
日志 `outputs/spmd-protocol-export-native-20261025/hbg.log`。
本机设备经 CANN query 确认为 Ascend950PR_958b、32 C／64 V；npu-smi／task-submit 缺失，
仍按用户明确要求直接用设备 0，未宣称队列隔离。
读端 lint 报告的长行已整理，嵌套事件 schema 检查抽为独立 helper；没有屏蔽任何规则。
整理后 Python 再次 1824 项通过，日志 `outputs/spmd-protocol-export-python-final-20261025.log`；
全部适用 pre-commit 检查通过，日志 `outputs/spmd-protocol-export-precommit-final-20261025.log`。

本次只提交已验收的 Host 采集／导出、统一 schema／序号检查、测试及中文状态记录，不 push。
尚待完成：把冻结成员、准入／领取／许可／完成／seal／依赖连成严格原生因果审计，
接入 CSA 严格门禁，再实现 sync_start 和无完整插桩的性能比较。
当前不能用 sequence_integrity_verified 代替 protocol_verified，也不能以启用 L4 后的耗时
推断实际调度性能收益。

## 2026-10-25：原生协议因果审计与严格门禁

起点 `9c545163e` 已本地中文提交、未 push。提交后重编成功，binding 与该 HEAD 一致，
HBG 三个运行时二进制 SHA256 与 `csa-hbg-l4-am5bizn1` 的通过产物完全一致。
本节继续原生审计与 CSA 门禁，不改变设备调度路径。

新增审计器核对冻结图、真实拓扑、准入日志／成员投影、所有 block 的唯一 owner／完成票号、
最终 payload／许可／kernel／消费／trace 的完整链、两槽代次、GM descriptor／arrive／release、
保留 seal、LocalDone、TaskDone 及依赖。真实入口身份另作逐字段连接，不用 trace 序号猜 block。
首批测试含“逐核序号与计数仍然正确、但协议内容错误”的变异，不能仅复用完整性检查使它们通过。

真实 CSA 捕获先暴露审计侧两项错误假设，均保留失败日志。
一是已有泳道 `core_id` 实为紧凑 worker 编号，不能与拓扑的硬件核号直接连接；
生产 `trace->worker_id = slot_claim.worker_id` 和 `SchedulerLocalState::worker_id()` 已核对。
补充二者不同的反例取得 RED，再按原接口连接；日志
`outputs/spmd-native-causality-worker-join-red-20261025.log`。
二是按跨核 cycle 区间重建原子完成票号顺序会误拒绝真实记录：CSA task 22 的票号 4/5
在不同 worker 上有相反的数值时间区间。这一现象不足以单独认定调度错误或时钟偏移原因。
日志 `outputs/spmd-native-causality-captures-r2-20261025.log` 保留该失败。

因果检查使用逐 worker 的事件序号链，以及发布前／后之间的虚拟线性化点。
这些虚拟点是审计约束，不伪装成设备采集事件，不占实际 sequence。
跨核边来自实际原子票号、已读 tail／ack／seal 与冻结依赖；不把跨核时间戳当全局时钟。
只在本核使用 cycle 单调性和 payload 快照的时间边界，最终检查整个约束图是否存在矛盾环。
普通 permit 的关键读位于前一条本核事件与 permit 记录之间，允许新准入在读之后、
记录／kernel 入口之前发布；不能因采样开销而给它补一道全局完成屏障。

合法测试明确覆盖：B 可以先于较小 task_ref 的 A 准入、本核关闭 A 后在远端 TaskDone(A)
之前进入 B、不参与 A 的核旁路、独立核时钟偏移、消费者观察早于发布者 after stamp、
普通 payload 重建和 predicate 跳过。另一个真实 RED 指出仅检查 payload 发布时的两槽复用
仍可能漏掉三次提前不可撤销领取；已把每个纯任务 claim 也绑定到前次槽消费之后。
日志 `outputs/spmd-native-causality-capacity-red-20261025.log`。

初版已通过 CSA 加仿真／A5 矩阵共 182 份原生捕获，日志
`outputs/spmd-native-causality-captures-r4-20261025.log`；加严两槽领取后的矩阵还需复验。
转换器与 CSA 门禁的 7 项实际 RED 保存在 `outputs/spmd-native-causality-gates-red-20261025.log`。
接线后，存在协议对象就必须通过统一审计；历史无对象仍可读取但不获得原生证明。
严格 CSA 还把本轮冻结图与独立 dep-gen 的任务、kernel、原 N 和全部 fanin 逐项核对。
原生路径引入因果证明；原有 kernel 依赖时序检查仍需作为额外保守检查保留，见本节后续收口。
本节正在进行边界扩展、格式检查及完整复验，尚未提交，也未宣称 sync_start 或性能工作完成。

加严后通过标准转换器重新读取全部 182 份捕获，原始协议与入口身份完整连接，
仿真 90 份共 427122 个事件／26303 条执行，A5 92 份（90 份矩阵加小型 Mix 和 CSA）
共 506539 个事件／28624 条执行，全部得到 protocol_verified=true。
日志 `outputs/spmd-native-causality-converter-matrix-20261025.log`。
新增测试还明确允许两个待退役领取、direct refill 先准备新 payload 再清理旧 trace、
epoch 0/MAX 和任意 JSON 行序；测试数据的类型标注和长行已整理，没有屏蔽 lint 规则。

本阶段新增 63 项原生因果／转换器测试、2 项严格门禁拒绝测试、5 项独立图一致性测试，
与已有读端／SPMD／CSA／快照／转换器合并共 1894 项通过（38.26 秒）。
日志 `outputs/spmd-native-causality-python-20261025.log`。
代码改动均在 Host 分析／用例侧，设备二进制和调度热路径未改变。

随后重新上 A5 设备 0 运行原样 CSA 严格 L4，
`outputs/csa-hbg-l4-10n3f_mf/result.json` 通过两轮各七项 golden、2308 条显式身份、
133 条图内依赖、107/107 Host 记录及完整原生审计。
实际原生证据为 46733 个事件、96 个 worker、81 个设备逻辑节点（79 kernel + 2 服务节点）、
61 个协同 task、4640 条成员义务、1457 个协同 block；
block_identity_verified／protocol_verified／distributed_scheduler_verified 均为 true，
dependency_order_source=native_protocol。原 kernel、构图和固定数据与基线一致。
生成的 L4.json metadata 同样保留 hbg_protocol_audit，不靠最终输出数值替代协议检查。
设备由 CANN query 确认 Ascend950PR_958b；本机缺少 npu-smi／task-submit，按用户要求直接用卡。

SPMD 设计／编码契约和 CSA 中文入口说明已更新为当前严格门禁，历史结果不追认新证明。
剩余目标仍是 sync_start 的不同位置容量／至多一次领取／最终可推进，以及关闭完整插桩后的性能比较；
本节没有扩大支持范围，没有新增运行配置开关，也没有 push。

收口时增加反例：即使原生协议对象合法，独立 kernel 行的前驱结束／后继开始发生明显冲突，
也不能跳过 CSA 原有时序检查。首版 native 分支确实漏过该反例，实际 RED 记录于
`outputs/spmd-native-causality-timing-guard-red-20261025.log`。
修正为原有 timing guard 始终执行，原生因果审计和独立图校验作为额外必需条件；
报告保留 dependency_order_source=native_protocol，并显式标记 dependency_timing_guard=true。
这不把未校准的跨核 cycle 用作原生协议全序，也不因新证明而放宽既有业务门禁。

最终增加空图、纯 Host 完成、纯服务节点三项检查，以及保留 timing guard 的一项反例。
本阶段新增总计 74 项（66 项因果／转换器、2 项严格门禁、6 项图一致性／时序守护），
完整 Python 选择集 1898 项通过（37.61 秒），日志
`outputs/spmd-native-causality-python-final-r2-20261025.log`。
保留 timing guard 后再次新跑原样 A5 CSA，`outputs/csa-hbg-l4-8gsn4ngo/result.json`
同时通过原数值、来源、显式身份、原生协议、独立图和原时序检查；三个 verified 标记及
dependency_timing_guard 均为 true，原始 case 仍与基线一致。
全部适用 pre-commit 检查通过，日志 `outputs/spmd-native-causality-precommit-final-r3-20261025.log`。
本次阶段提交包含审计、转换器／CSA 接线、TDD 和中文状态记录，不含输出产物或用户未跟踪内容，
仅本地提交，不 push。后续继续 sync_start 和无完整插桩的性能验证。

## 2026-10-25：sync_start 的不同位置领取与局部推进

起点 `67251d83c`。上轮仅报告完成度，没有代码推进；本节继续目标中的 sync_start，
保留完成驱动依赖、每核自主执行、GM Mix 和原样 CSA，不增加全员启动／完成屏障。

复用现有 C++ 生产调度夹具，增加纯 C/V 的两槽至多一次领取、早返回后不再次领取、
发布未来任务不清除当前绑定、快核先进入独立满核 B、普通后继恢复双槽分波，
以及零工作成员只等领取耗尽、不等 TaskDone。Mix 四种 mask 检查本 cluster 早关闭、
远端未领取时进入 B、mailbox 被 B 复用后迟到 AIV 仍能消费 A 的保留 seal。
首轮测试编译暴露测试代码误用仅接收 int64 的 scheduler_gm_load 读取 uint64 计数；
改用原有 fetch_add(0)，未修改原语。随后 11 个新增行为测试实际失败，见
`outputs/spmd-sync-start-cpp-red-20261025.log`。

实现仅在本核准入状态增加 sync_claimed，成功 CAS 后置位，跨 kernel 返回保留，
只在当前义务关闭／新义务装入时清除，不在未来任务准入时清除。
本地状态由 224B 增至 232B，主 SchedulerLocalState 和共享 POD 布局不变。
纯 C/V 完成自身槽退役后可按实际 cursor 提前关闭；未领取成员仍等 cursor=N。
Mix 每 cluster 至多一组，本组全部 release、聚合完成后 AIC 才能提前发 seal，
AIV 仍逐一核验保留 seal，不用 cursor 替代资源收尾。11 个用例转绿；随后扩展夹具为
实际大小的三 cluster，补四种 mask 的零工作 cluster／最后组尚未发布反例，共新增 15 项，
两个目标通过，日志 `outputs/spmd-sync-start-cpp-green-r2-20261025.log`。

原生审计新增 33 项 sync_start 检查，先实际 RED，见
`outputs/spmd-sync-start-audit-red-20261025.log`。审计依据冻结拓扑检查不同位置容量，
拒绝同核／同 cluster 重复领取；允许已领取成员退役后提前关闭，未领取成员不能提前关闭。
关闭 cursor 的关键读限制在本核前一事件与 closed 之间，并与前后原子领取线性化点连接；
不把读到的 cursor 强行视为 closed stamp 时刻的全局值。Mix seal 约束改接实际读值，
保留全部 release 和组退役约束。新增测试及原有测试共 99 项通过，日志
`outputs/spmd-sync-start-audit-green-20261025.log`。

Host 入口正在接通独立容量检查与 demand：普通任务不按 N 占核，sync_start 纯 C/V
按 N 个对应类型核、Mix 按 N 个完整 cluster（包括 inactive 资源 lane）计算。
N=1 sync_start、multi-block predicate、Graph／async 的原有边界不扩大。
容量 helper 的编译测试发现头文件包含顺序依赖，待当前构建结束后修正并复验；
此处尚无新 runtime 或 A5 验收通过的声明。

原生用例复用现有记录器，加入实际跨 block 等待和 block 0 早返回两类 kernel 行为，
等待超过 5 秒写错误并返回，不把超时当作成功。覆盖两份独立任务、三种规模、
纯 V 满 64 block、Mix 满 32 block、17 次任务复用；仍强制本轮 resident 来源与精确 golden。
上板前按 onboard-arch-precheck 工作流检查；脚本因 npu-smi 不在 PATH 无法检测，
CANN query 实际确认 Ascend950PR_958b、32 C／64 V。task-submit 不存在，
按用户明确要求直接使用设备 0，不声称队列隔离。构建、上板和回归结果续记。

容量 helper 的包含顺序问题已通过显式使用平台容量常量和头文件解决；
首轮 runtime 构建也确实报同一错误，没有用旧二进制报通过。
修正及格式整理后的重编成功，日志 `outputs/spmd-sync-start-runtime-build-r3-20261025.log`。
17 个 C++ targets 共 621 项通过，Python 相关选择集 1945 项通过；日志
`outputs/spmd-sync-start-all-cpp-r2-20261025.log`、`outputs/spmd-sync-start-python-r2-20261025.log`。
相对上一提交新增 C++ 17 项、Python 47 项，不把修改的既有 shape 测试重复计作新增。

仿真探针首跑因用户 kernel 默认 include 路径不含 inner_kernel.h 编译失败。
测试夹具改为按已选 platform 附加原有 sim/onboard AICore 头文件目录，
复用现成编译器／缓存与 get_sys_cnt_aicore，不增加 feature macro 或新的时钟实现。
修正后 C/V/Mix 三项仿真小规模同步通过；A5 七种 mask 的 N=2 同步均通过。
日志 `outputs/spmd-sync-start-sim-smoke-r2-20261025.log` 与
`outputs/spmd-sync-start-native-20261025/smoke.log`。

首轮完整矩阵暴露两类问题，失败记录保留。
仿真的实际拓扑为 8 C／16 V（原生协议 worker_count=24，代码 SIM_AUTO_BLOCKDIM=8），
32／64 block 的 sync_start 被 Host 正确拒绝，不能把普通分波矩阵直接用作共同驻留矩阵。
现按平台分别覆盖满核：仿真 C/Mix=8、V=16，A5 C/Mix=32、V=64；小规模 2／7 共同覆盖。
两个平台各有 47 个成功场景，没有用 skip 或放宽容量来掩盖失败。

A5 首轮满核部分场景在探针内等待 5 秒后报错，见
`outputs/spmd-sync-start-native-20261025/matrix.log`。
读取失败捕获确认所有 block 位置唯一、协议因果审计通过，但部分 kernel 持续约 5e9 cycles。
检查探针发现其只在 ld_dev query 看见目标计数后才进行关键原子读；
scheduler_memory.h 已明确 query 可能看不到尚未从原子单元排出的写入，不能把它作为唯一进度条件。
探针改为直接原子读取 arrival 计数，保持原 5 秒 timeout 和完整 golden，不修改生产调度原语。
修正后的新一轮仿真／A5 各 47 项全部通过，日志
`outputs/spmd-sync-start-sim-matrix-r2-20261025.log` 与
`outputs/spmd-sync-start-native-20261025/matrix-r2.log`。两组输出目录无重合；失败捕获未计入通过数。

另增加独立私有 worker 的真实 Host 拒绝测试，防止故意 bind 失败污染正常 scene worker。
首轮使用旧用例中的 -1000 异常码不符当前 API，实际为 prepare_native_run 的 -7。
继续核对日志发现超容量已由既有 Host orchestration 的 REQUIRE_SYNC_START_INVALID 提前拒绝，
不会到达新增 resident maker demand 检查。测试应核验这个真实拒绝原因；
新增 maker 检查和完整 cluster demand 则由 C++ helper 覆盖，不能把二者混称同一路径。

N=1 的公共接口行为还发现一处契约不一致：common Host submit_task 仅在 N>1 时设置
SYNC 标志，N=1 的 LaunchSpec 请求被归一化为普通单 block；因此公共接口不会触发
resident maker 中“已经带 SYNC 标志但 N=1”的拒绝。这是既有行为，不是本次放开的结果。
当前参考文档和 CPU 模型把 N=1 写成拒绝，尚未区分公共请求与设备 metadata。
已向用户询问保留归一化还是改变共用 HBG 接口，未擅自修改 common Host。
N=1 公共接口的七项 RED 保留，不能声称全部边界验收完成；
超容量七项单独复验，原样 CSA 与已有 HBG 回归继续执行。

通过标准转换器逐份复验修正后的 94 份捕获：仿真 47 份共 20402 个事件／1093 条执行，
A5 47 份共 54914 个事件／2629 条执行，均获得 protocol_verified=true。
仿真有 28 条、A5 有 14 条本核关闭记录的实际 cursor 仍小于 N，
证明生产路径确实使用了提前局部关闭，而非只在模型中允许；不能据此单独断言下一任务
一定早于远端返回，后一个合同另由生产 helper 的可控交错覆盖。
日志 `outputs/spmd-sync-start-native-audit-20261025.log`。

超容量七项在仿真与 A5 的实际资源子进程中逐项通过；外层汇总为 7 deselected，
不能把外层数字当作通过数。日志 `outputs/spmd-sync-start-capacity-sim-overflow-20261025.log`
与 `outputs/spmd-sync-start-native-20261025/capacity-overflow.log`。
这次没有重跑 N=1 的七项未决断言，未将它们算成通过。

已有 HBG 仿真回归首轮 41 passed、1 failed：
`TestSpmdShapeSelection` 的 sync_start N=2 仍要求 mode=4，实际输出正确且 mode=1。
这是新能力开放后需要加严到 resident 的旧期望，不应删除来源检查或允许两种模式；
等正在运行的 A5 回归结束后再修改该文件并重跑。日志
`outputs/spmd-sync-start-hbg-sim-20261025.log`。

将已有 shape-selection 的 sync_start N=2 来源断言加严到 resident 后，完整 HBG 回归
仿真 42 passed、A5 60 passed／4 skipped。日志
`outputs/spmd-sync-start-hbg-sim-r2-20261025.log`、
`outputs/spmd-sync-start-native-20261025/hbg-r2.log`。
仅排除已知误收集的双卡 TMR 项，以及尚有 N=1 契约争议的新私有 worker 测试文件；
后者的七项超容量已有独立通过证据，不把七项 N=1 RED 混入全绿声明。

原样 CSA 新一轮严格 L4 `outputs/csa-hbg-l4-jrabrt2o/result.json` 通过，
case_sources_match_baseline、block_identity_verified、protocol_verified、
distributed_scheduler_verified、dependency_timing_guard 均为 true，producer=aicore；
仍为两轮各七项 golden、2308 条执行身份及 133 条依赖边，未修改 CSA kernel 或构图。

本次阶段提交限定已验证的 N>1 sync_start、对应审计／探针、既有来源断言与状态文档。
新私有 worker 文件 `tests/st/a5/host_build_graph/sync_start_capacity/test_sync_start_capacity.py`
保留在本地未提交：其中七项超容量已通过，七项 N=1 预期仍待用户确认，不能进入本次绿灯提交。
N=1 的现有公共归一化未改变；完整目标仍须收口该契约，并完成无完整插桩的性能比较。

最终 Python 选择集再次 1945 项通过，日志 `outputs/spmd-sync-start-python-final-20261025.log`；
全部适用 pre-commit 检查通过，日志 `outputs/spmd-sync-start-precommit-final-20261025.log`。
既有普通 90 场景也在新探针参数布局下重新执行，结果续记；这些普通场景不启用 sync_start。

新参数布局下的普通 C/V／Mix 90 场景在仿真和 A5 均全部通过，含 N=512 分波和 17 task 复用，
日志 `outputs/spmd-sync-start-ordinary-sim-20261025.log`、
`outputs/spmd-sync-start-native-20261025/ordinary.log`。没有让不同平台同时写入同名 case 输出目录。

普通矩阵另通过标准转换器复验 180 份捕获：仿真 427122 个事件、A5 459234 个事件，
各 26303 条执行全部通过原生审计，日志 `outputs/spmd-sync-start-ordinary-audit-20261025.log`。
本次合计重新审计普通／sync_start 矩阵 274 份捕获，CSA 独立验收不混入该数量。
阶段提交仅本地保存，不 push；提交后按新 HEAD 重编绑定与 runtime，避免后续运行命中旧提交缓存。

## 2026-10-25：同一 CSA 的无完整插桩性能对照

起点 `a4132b6df`，上一轮属于实际代码／验证／提交进展，不是仅重述状态。
沿用原构图、kernel、输入与七项 golden。N=1 公共接口的选择尚未收到用户确认，
保持归一化不动；本节推进不依赖该选择的性能验证，不把目标缩成只验证 N>1。

提交后构建 `outputs/spmd-sync-start-postcommit-build-20261025.log` 完成；
实际 binding 的 **build_commit** 与 a4132b6df 一致，三个 HBG runtime 二进制 SHA256
与已通过的 `csa-hbg-l4-jrabrt2o` 完全相同，原 CSA source manifest 仍匹配。

使用 benchmark 工作流的隔离 worktree／独立 venv／同卡串行方法。
上游 main 更新后的 merge-base 是 7e19db4df，但本任务的设计基线明确为 f5cc679bb；
因此固定该版本做对照，不将持续变化的 main 或十月十四日历史耗时作为本次基线。
临时 worktree 为 `tmp/spmd-perf-baseline-20261025`，分离 HEAD，不占用／移动现有开发分支。
原工作区、simpler-scalar_scheduler 和既有 review worktree 均保持原样。

已用 git diff 核对 f5cc679bb 到 a4132b6df 的 CSA kernel、编排、manifest、golden、
验证器、测试入口与 analyze_perf 均未改变；PTO-ISA pin 均为
`418de18dde8fa33773dc5c384fdfe5506463b628`。
基线环境单独安装 Torch 2.12.0+cpu、numpy 2.5.2、nanobind 2.15.0、
scikit-build-core 1.1.1、cmake 4.4.4、pytest 9.1.1，与当前主要版本一致。
首次构建因独立环境缺 pathspec 在 metadata 阶段失败，补齐原有依赖后重编，
保留 `outputs/spmd-perf-baseline-build-20261025.log` 与 r2 日志，不混用主工作区绑定。
400MB 固定数据使用原 fixture.import_snapshot 导入并校验 SHA256，不重新生成 golden。

计划复用现有 `run.py bench`：每臂先三轮七项正确性检查，再五轮预热和一百轮采样；
完整 chip swimlane 为 0，仅保留既有 STRACE 时钟与固定阶段记录。
Host 与 Device 分开报告；Device 为既有 AICPU RunWall 全设备生命周期，包含初始化／退出，
不冒称纯 kernel 时间或 HBG 不存在的 Effective/Sched 窗口。
调度来源另用 L4 核对，不将 L4 诊断耗时与无完整插桩基准混算。
本机无 task-submit，沿用用户明确授权的设备 0 顺序执行；不声称独占锁隔离或跨设备可比性。
当前仅在构建基线，尚未产生本轮性能结论。

基线编译完成，binding 对应 f5cc679bb，源代码 diff 为空。L4 首次在导入 Worker 时
发现缺少 cloudpickle，尚未运行 kernel；按当前环境补齐 cloudpickle 3.1.2 及其余主要
Torch 依赖版本后再运行，未改基线源码。失败结果 `csa-hbg-l4-u6meeup3` 保留。
新基线 L4 `tmp/spmd-perf-baseline-20261025/outputs/csa-hbg-l4-ka862xev/result.json`
两轮各七项 golden 通过，scheduler producer=aicpu。它只作为 AICPU 对照，
不冒充新 resident 的协议验收。

同设备 0 按 A1→B1→B2→A2 顺序采样，A=f5cc679bb，B=a4132b6df。
每组都有独立三轮正确性检查和五轮预热＋一百轮测量，完整泳道关闭；
四组均 passed=true，源码／二进制身份在每组前后未变化，fixture manifest／构图摘要相同。
标准解析器重新读取四份 bench.log，与各自 result.json 的统计逐项一致。
汇总为 `outputs/spmd-perf-comparison-20261025.json`，保留全部样本与阶段时长。

| 组别 | 调度来源 | Device 中位数（μs） | 结果目录 |
| ---- | -------- | ------------------- | -------- |
| A1 | 设计基线 AICPU | 2161.126 | 临时 worktree 的 outputs/csa-hbg-bench-zmz8ikow |
| B1 | 当前 AICore resident | 24509.032 | outputs/csa-hbg-bench-pgmp0zkn |
| B2 | 当前 AICore resident | 24606.033 | outputs/csa-hbg-bench-k2s_6ngr |
| A2 | 设计基线 AICPU | 2137.535 | 临时 worktree 的 outputs/csa-hbg-bench-4uu673jh |

合并每臂两百个测量样本，**当前 CSA 明显回退，不能宣称性能优化成功**：

| 窗口 | A 中位数（μs） | B 中位数（μs） | B/A | 变化 |
| ---- | -------------- | -------------- | --- | ---- |
| Host | 77485.754 | 100344.804 | 1.295 | +29.50% |
| Device | 2151.344 | 24554.408 | 11.414 | +1041.35% |

Device 的 A 臂 P10/P90 为 2093.639／2204.554 μs，B 臂为 23838.803／26182.207 μs。
反向复跑的 A 臂中位数只变化约 1.1%，B 臂约 0.4%；这次量级差异不能解释成普通采样漂移。
没有 task-submit 锁隔离，仍不作“环境完全无干扰”的承诺，也不外推到其他 workload。
各组 Host bind 中位数约 69.1–69.4 ms、validate 约 5.45–5.49 ms，
而 runner_run 从约 2.42 ms 增为 24.8 ms；主要可见增量在设备窗口，
不能把 Host 的全部 100 ms 都称为 Scalar 调度耗时。

L4 仅作为独立诊断：当前已验证捕获有 61 次准入、1457 个 block owner、
833 次 Mix 组发布、4640 条成员义务；2290 条执行来自 COOPERATIVE，
DIRECT_RESOLVE 有 6 条，所有 2308 条 payload publication 都为 DISPATCH。
当前按多核汇总的 idle 区间约 1.356 秒，不能当成设备串行等待时间。
同核相邻 kernel 间隙中位 364.190 μs、P90 1978.991 μs，包含依赖与控制等待和采样开销。
成功 block claim 的 L4 区间中位 0.325 μs，不含失败重试／未记录的轮询链；
不能用它乘次数或用 L4 区间相减直接解释 L0 的 24.554 ms。
诊断输出 `outputs/spmd-perf-current-l4-control-20261025.log`。

源码中 scheduler_admission_context_valid 对同一 run.scheduler_error 做关键原子读，
且多个热路径 helper 会重复调用，是待定量排查的共享热点候选；此处不是已证明根因。
原 CSA 没有 sync_start 请求，所以本对照是整个普通多 block／GM Mix resident 方案
相对 AICPU 基线的差异，不能归咎于新增 sync_start 一次领取分支。
继续用同一临时 worktree 的 detached HEAD 对照 `6f2b470d4`（已有普通 resident，
尚未加入完整协议采集），检查关闭 L4 时后续采集接线的影响。正式 A/B 基线仍为上述两版；
已有四组结果保留，当前开发分支不切换，不修改调度代码来迎合基准。

前期 resident 6f2b470d4 重编后先通过 L4 的七项 golden、执行数量与 aicore producer 检查，
结果为临时 worktree 的 `csa-hbg-l4-3wooaho3`；该版本尚无当前的完整原生协议证明，
不能把旧报告的 distributed_scheduler_verified 字段追认为新门禁通过。
随后三轮正确性和五预热＋一百采样通过，`csa-hbg-bench-jkd494k5` 的
Host 中位数 100604.256 μs、Device 中位数 24726.374 μs。
当前合并 Device 中位数比此前期版本约低 0.70%，属于本次小幅波动范围；
大幅回退已在普通多 block／GM Mix resident 接入时存在，不能说从完整协议采集才开始。
这也不等于采集分支零成本：这里只测整体净差异，未逐项隔离后续提交。

静态代码体积核对：AIC 的 .text 由 236840B 增至 527056B，AIV 由 240384B 增至 535080B。
这里读取的是 ELF section，不是含调试信息的整个 .o 大小；体积增大本身不是已证明的
性能根因，也不能据此把 L0 回退倍数归给 instruction cache。
编译器身份、选项及 PTO pin 保存在 `outputs/spmd-perf-toolchain-20261025.json`，
CANN CCEC/LLD 为所装 SDK 的 15.0.5，Host 为 GCC 13.3.0。

临时 worktree 的完整 outputs 已复制到
`outputs/spmd-perf-worktree-evidence-20261025/`，diff -qr 核对一致。
最终比较文件为 `outputs/spmd-perf-comparison-final-20261025.json`，A1/A2 与前期 resident
的结果路径指向该归档；各原始 result.json 保留原执行路径作为历史 provenance，不修改原记录。
临时 worktree、独立 venv 和复制的固定数据按 benchmark 工作流清理；源码可由上述提交重建，
固定数据仍在原 CSA 目录，所有独有实测日志／捕获已归档。未删除主工作仓或用户已有数据。

本节仅完成性能验证和缩小归因范围，不宣称取得性能收益、不静默重构调度协议。
下一步仍须确认 N=1 公共契约并完成最终逐项验收；若进入性能优化，需要用上述相同口径
逐项验证共享轮询／拓扑检查等候选热点，同时保留所有执行与错误收敛合同。

最终通过统一读端核对三版 L4：metadata.num_cores 均为 96、kernel 行均为 2308，
实际执行行覆盖的紧凑 core_id 数也均为 96。
当前冻结任务 metadata 没有 SYNC 标志，进一步限定了本次性能结论的适用范围。
收口校验脚本曾错误地把 AICPU 的 core_to_thread 当作 raw 顶层公共字段，
又把原始紧凑 kernel 数组当成字典，均在读取阶段报错；改为共同的 metadata.num_cores
及标准 read_perf_data 解码后才完成上述核对。性能样本、统计与结论未改变。
本次主仓改动仅为过程记录与三份状态文档，不改变生产代码；适用 pre-commit 检查全部通过，
日志 `outputs/spmd-perf-precommit-r3-20261025.log`。表格对齐检查的前两次失败已修正，未绕过钩子。
阶段提交继续使用详细中文日志，不 push；N=1 公共契约的未决测试仍留本地，未混入本次提交。

## 2026-10-25：异构 sync_start 的原生重叠与依赖验收

起点 `892fe6cac`。上一轮只核对完成度，没有代码推进；本节补最终验收中的直接证据，
不改公共 N=1 行为，不调整生产调度协议，也不把功能用例的等待时间用于性能结论。
此前 47 项同步矩阵覆盖同类任务连续执行，原生提前 LocalDone 记录并不能单独证明
后任务已进入 kernel；新增检查直接观察两个任务的 kernel 原子事件序号。

复用 CLA 的 32 列 plan、显式 fanin、128B 执行记录、原子 entry/return 序号和
既有 resident 新鲜来源门禁。Host 提交循环提取到共用 cla_submit.h，原三参数入口不变；
新入口以 Host-only sync_flags 指定每项请求，另用一个 Scalar 指定是否构造可控交错，
合计四个 Tensor 和一个 Scalar；不给普通任务加入同步语义。
任务内会合复用同一条有界原子 arrival 检查，5 秒超时写哨兵并失败；
不依赖跨核 cycle，不改变生产发布原语。真实依赖场景不等待后继；
独立场景只让尾块有界等待另一任务的首个 entry，其他执行位置先正常返回，
从而直接检验“局部关闭后进入后任务”而不是依赖有限延迟恰巧产生交叠。

新增 test_sync_ordered_execution.py：每个平台覆盖 mask 1–7 的全部 49 种有序配对，
各有独立重叠和真实依赖两个场景；再加入 7 种 mask 的同步→普通／普通→同步切换，
合计每平台 112 项。同步按实际满容量：仿真 C/Mix=8、V=16，A5 C/Mix=32、V=64；
普通一侧使用容量加一，检查能恢复分波。独立任务不强制 Host 提交顺序，
但必须观察到执行区间交叠；真实依赖则要求后继进入晚于前驱所有 block 返回，
并保留逐输出的可见性检查。两类场景都不接受 legacy 回退或重复／缺失身份。

新增验收器的第一轮测试因模块尚未创建而报 FileNotFoundError，
`outputs/spmd-sync-ordered-oracle-red-20261025.log` 仅是缺入口记录，不是生产行为 RED。
接通后新旧有序探针 CPU 检查 336 项通过；首轮仿真／A5 各 112 项通过。
继续审查发现“最终 arrival 数正确”不能证明任务内会合真正等待了所有入口：
增加保持事件全排列、完整 arrival 与正确输出的早返回反例，原验收器实际 DID NOT RAISE，
见 `outputs/spmd-sync-ordered-barrier-oracle-red-20261025.log`。
加严为每个同步任务 max(entry) < min(return) 后，新旧有序探针 337 项通过，
日志 `outputs/spmd-sync-ordered-oracle-green-r2-20261025.log`；对应两平台矩阵正在重跑。

上板预检脚本缺少 npu-smi，CANN query 实际确认 device 0 为 Ascend950PR_958b、32C/64V；
沿用用户明确要求直接用卡，不宣称 task-submit 锁隔离。上一提交后的构建已完成，
binding build_commit 为 892fe6cac，三个 HBG runtime 二进制 SHA256 与已验证 CSA 捕获
csa-hbg-l4-jrabrt2o 相同。当前只修改测试／探针和过程记录，无生产 runtime 源码变更。
加严后的原生协议审计、共用探针回归与提交检查结果续记。

加严验收器后的第二轮：A5 112 项通过，仿真 111 项通过、mask1→4 的独立场景
未产生交叠而失败。失败捕获的原生协议仍通过，仿真统一时钟下两项 kernel 区间为
63017.687–86108.066 μs 和 87049.650–146433.276 μs，确实没有交叠；
原样单独复跑通过，但不能据此抹掉批次失败或声称固定忙循环可稳定制造交错。
因此独立场景改为上述有界同伴 entry 等待，保留所有输出、事件顺序与重叠断言。

可控交错首跑的 Host 接线误用了 scalar(4)：ChipTaskArgs 的 Scalar 下标独立于 Tensor，
实际应读取 scalar<int64_t>(0)。未初始化槽值导致仿真普通侧没有 entry 计数、
A5 部分真实依赖场景错误启用同伴等待，均被验收器捕获；这是探针错误，不是生产调度回退。
仿真批次自然退出；检查 Worker 的 finally/close 路径后，仅向本次 A5 runner 发送 SIGINT，
其退出码 130，未强杀设备进程、未重置设备；失败日志保留为 matrix-r3。
修正后还增加逐尾块 active lane 的等待分支次数，防止未启用等待的探针误报可控交错成功。

第四轮仿真／A5 各 112 项全部通过，分别为 63 项独立重叠和 49 项真实依赖。
日志 `outputs/spmd-sync-ordered-sim-matrix-r4-20261025.log`、
`outputs/spmd-sync-ordered-native-20261025/matrix-r4.log` 保存逐场景的实际原子事件区间。
244 项新增 CPU 验收器检查覆盖每一项 plan、拒绝串行伪重叠、允许两种 ready 顺序、
早返回／超时／缺失到达／未执行等待分支、错误身份和 inactive 槽；
与旧有序探针合计 340 项通过，完整相关 Python 选择集合计 2189 项通过。
日志 `outputs/spmd-sync-ordered-oracle-green-r4-20261025.log`、
`outputs/spmd-sync-ordered-python-r4-20261025.log`。

标准转换器逐份复验 224 份第四轮捕获，全部原生因果审计通过；另对照请求逐项检查
原 N、kernel IDs、SYNC 标志、完整拓扑与真实 fanin。纯 AIV1 请求由既有 Host
归一化为 mask2／kernel_id=2，审计按这个实际 ABI 核对，不误写成 Mix 的 slot2 placement。
日志 `outputs/spmd-sync-ordered-native-audit-r4-20261025.log`。

第四轮原生审计：仿真 62192 个事件／3608 条执行，A5 244976 个事件／14360 条执行。
63 项独立场景中，分别有 28／37 项是第二个提交的任务先进入，验收没有把 Host 顺序
误当共同 ready 顺序。A5 纯 C→Mix7 场景的原子事件为：C 首入 1、末入 32、
首返 33、末返 65；Mix 首入 64、末入 160、首返 161、末返 256。
64 < 65 是后任务进入早于前任务最后返回的直接证据，不需要比较跨核时钟。

共用探针首次仿真回归为 5 passed／2 failed，失败的 CLA 和首错误身份探针均无法找到
inner_kernel.h。record.h 在 sync_start 接线时引入计时头文件，而这两项只需要记录布局，
没有使用添加平台 AICore include 路径的有序探针编译入口；不能把它们遗漏在回归之外。
将未改变的 128B ExecutionRecord 与 tensor_data 提取到 execution_record.h，
CLA／错误身份探针直接引用布局，有序／同步探针仍引用原计时和同步 helper，
不复制编译器、不改全局 include 路径、不扩展生产 API。
调整后仿真和 A5 的 7 个共用探针测试类全部通过，包括普通分波、原 47 项同步、
有序执行、GM Mix、CLA 图、AIV→Mix 交接与错误身份。
日志 `outputs/spmd-sync-ordered-probe-regression-sim-r2-20261025.log`、
`outputs/spmd-sync-ordered-native-20261025/probe-regression-r2.log`。

布局拆分后的第五轮仿真／A5 新矩阵再次各 112 项通过；224 份捕获再次经完整原生审计，
事件／执行总数与第四轮相同。这轮第二个提交任务先进入的独立场景分别为 27／28 项，
共同顺序取决于本次 ready 竞争，不人为固定。相关 Python 选择集再次 2189 项通过。
最终矩阵、审计和 Python 日志为 `outputs/spmd-sync-ordered-sim-matrix-r5-20261025.log`、
`outputs/spmd-sync-ordered-native-20261025/matrix-r5.log`、
`outputs/spmd-sync-ordered-native-audit-r5-20261025.log`、
`outputs/spmd-sync-ordered-python-r5-20261025.log`。
提交检查发现 Ruff E501 的诊断打印行过长，以及两个新 Python 文件的 docstring
不符合源码语言检查；分别拆分 witness 打印并调整源码说明，中文过程文档保持不变。
这些修正不改变验收语义，未绕过钩子。

最终六个代表场景（C/V 不相交、C/Mix、mask6 协调、真实依赖及两向普通分波切换）
分别在仿真／A5 连续运行十轮，每平台 60 次均通过；逐轮重新核验 resident 来源和全部哨兵，
没有用一次成功替代后续轮次。多轮按框架规则关闭完整泳道，不混入上述 224 份 L4 审计数量。
日志 `outputs/spmd-sync-ordered-repeat-a5sim-20261025.log`、
`outputs/spmd-sync-ordered-repeat-a5-20261025.log`；最终新旧 oracle 再次 340 项通过。
适用 pre-commit 检查通过，日志 `outputs/spmd-sync-ordered-precommit-r3-20261025.log`。
本次阶段提交只包含测试／探针／中文过程记录，不包含产物或用户未跟踪文件，不 push。
公共 N=1 契约仍待确认；性能回退根因仍未定位，本节不能被当作性能优化完成。

## 2026-10-25：较晚提交的 B 先于仍等前驱的 A 执行

起点 `215ee611e`；上一轮已完成测试实现、验证和本地提交，不是只报告状态。
其提交后构建已完成，binding build_commit 与 HEAD 一致，三个 HBG runtime 二进制
SHA256 与已验证的 csa-hbg-l4-jrabrt2o 相同。本节没有改生产 runtime。

继续核对边界文档的 C39：“A 等 P、B 已 ready 时，B 先发布并执行，
A 此时尚未 ready”。原有 HbgAdmission.ReadyTaskBMayCommitBeforeTaskAWhoseDependencyIsNotReady
检查生产 helper 的路由和准入，但上一轮两个 ready 根任务的任意顺序不能替代该原生证据。

新增 test_ready_order_execution.py，以 Host 顺序提交 P、A、B，只有 P→A 一条真实依赖。
P 是 B 不参与的核类型上的普通单 block，有限等待 B 的首个 entry 才返回；
A/B 的尾块继续使用已有可控交错协议，B 尾块等 A 首入，其他位置正常返回。
因此 B 可以先进入而不会被 P 占据必需位置，A 仍必须等 P 真正完成；
随后 A 必须能早于 B 全局完成进入，不能用全局完成屏障替代成员局部顺序。

复用同一组三个 kernel、128B 记录、原子事件序号和逐轮 resident 来源校验。
CLA Host helper 的等待目标改为逐 task 的 Host-only 引用数组，检查边界、自等待和 dummy，
原双任务入口继续映射为 1/0 或不等待，保持其 Tensor／Scalar ABI；
新入口是五个 Tensor，不读混合总下标作为 Scalar 下标。
同步 guard 的 golden 与检查提取为共用函数，不复制相同校验。

新矩阵每个平台 84 项：A 的七种 mask、B 的纯 C／AIV0／AIV1 三种形状，
各选四种普通／sync_start 组合。同步用本机／仿真的类型满容量，普通用容量加一；
P 的 N=1 不带 sync_start，不涉及尚未确认的公共接口契约。

新验收器先复用完整身份、真实依赖和同步 guard 检查，三个特定反例实际 RED：
Host 顺序全串行、B 虽先于 A 但 P 已完成、A 被迫等 B 全局完成。
日志 `outputs/spmd-ready-order-oracle-red-20261025.log` 为 3 failed／178 passed；
加入 B.entry < P.return 及 A.entry < B.last_return 的强断言后，
新旧有序／同步 oracle 共 521 项通过，日志 `outputs/spmd-ready-order-oracle-green-20261025.log`。
它们都是同一原子序列中的事件，不将跨核 cycle 数值当作顺序证明。

三个仿真和 A5 冒烟场景已通过，完整矩阵、原生共同顺序审计和共用入口回归继续执行。
上板预检缺少 npu-smi，CANN query 实际确认 Ascend950PR_958b、32C/64V；
按用户要求直接使用设备 0，不宣称 task-submit 锁隔离。

新矩阵仿真／A5 各 84 项全部通过，完整相关 Python 选择集 2370 项通过；
日志 `outputs/spmd-ready-order-sim-matrix-20261025.log`、
`outputs/spmd-ready-order-native-20261025/matrix.log`、
`outputs/spmd-ready-order-python-20261025.log`。
168 份原生捕获逐份审计通过：共同日志严格为 [2, 1]，即 B 的 order_key=0、A=1，
P 没有共同准入记录，冻结图只有 P→A。另核对原 N、SYNC、kernel IDs 和实际拓扑，
且每份原子事件均满足 B.entry < P.return < A.entry < B.last_return。
仿真为 39156 事件／2662 执行，A5 为 148020 事件／10054 执行；
日志 `outputs/spmd-ready-order-native-audit-20261025.log`。
A5 的 A=Mix7、B=纯 C 同步场景中，B 首入为 1、P 返回为 6、A 首入为 66、B 末返为 67，
直接证明准入不受未 ready 的早提交任务阻塞，也不添加 B 的全局完成屏障。

共用入口混合回归在两平台均为 3 passed／1 failed：旧同步双任务类取得了
sig_count=3、child_count=2 的父类 callable，而自身需要四个 Tensor 和三个 kernel。
失败发生在 Host 参数核验阶段，没有执行错误的设备 kernel；不能把它解释成 AICore 调度失败。
定位到 SceneTestCase 使用 getattr(type(self), "_st_l2_handle")，从父类继承了注册句柄，
且未将缓存绑定到 Worker 或 callable。新增九项 CPU 测试先取得 7 failed／2 passed，
日志 `outputs/spmd-ready-order-registration-red-20261025.log`。

缓存改为测试实例持有的 (Worker, callable, handle)，只有两个对象身份都匹配才复用；
注册成功后才发布新缓存，失败不覆盖原有效记录。保留同实例跨 case／round 的一次注册，
不修改 Worker、Callable 或生产调度接口；清除两个旧测试中已无意义的类属性重置。
修正后连同新 ready-order 类的五类混合回归在仿真和 A5 均通过，
日志 `outputs/spmd-ready-order-probe-regression-sim-r2-20261025.log`、
`outputs/spmd-ready-order-native-20261025/probe-regression-r2.log`。

扩大框架回归时另有两项 CLI 测试因裸 import conftest 取到 tests/ut/py/conftest.py
而缺少根插件函数；单独复跑同样失败，导入路径复核也确认不是句柄缓存造成。
这两项改为从当前 pytest pluginmanager 按根文件路径取已加载插件，保留原断言，
不改根 conftest 行为；完整框架回归与最终提交检查继续执行。

调整 CLI 测试导入时曾误删文件内其他测试仍使用的 importlib，出现九项 NameError；
补回该导入后，包含九项新注册缓存回归的框架选择集 216 项全部通过，
日志 `outputs/spmd-ready-order-framework-green-r3-20261025.log`。未跳过失败测试。
SPMD／CSA 相关 Python 选择集再次 2370 项通过，
日志 `outputs/spmd-ready-order-python-r2-20261025.log`；
完整 HBG 仿真回归 42 passed，日志 `outputs/spmd-ready-order-hbg-sim-20261025.log`。

完整 HBG A5 回归 60 passed／4 skipped，日志 `outputs/spmd-ready-order-native-20261025/hbg.log`。
继续仅排除已知误收集的双卡 TMR 目录和仍有 N=1 契约争议的本地容量测试文件，
没有把未决断言计作通过；也不据此声称双卡回归已完成。
共用场景框架的 A2/A3 HBG 仿真回归为 30 passed／7 skipped，
日志 `outputs/spmd-ready-order-hbg-a2a3sim-20261025.log`，没有在 A5 卡上运行 A2/A3 二进制。

句柄缓存修复后的原样 CSA 严格 L4 验收通过，结果 `outputs/csa-hbg-l4-lii5bvds/result.json`；
producer=aicore，原样源文件、2308 执行身份、133 条依赖、原生因果审计与 timing guard 均通过，
两轮各七项数值 golden 保持通过。没有修改业务 kernel、构图或输入来适配新测试。

六个代表 ready-order 场景在两平台各重复十轮，分别 60 次通过，逐轮仍验证新鲜 resident
来源、全部输出、同步 guard 和 B/P/A 的原子顺序；日志
`outputs/spmd-ready-order-repeat-a5sim-20261025.log`、`outputs/spmd-ready-order-repeat-a5-20261025.log`。
多轮不启用完整泳道，不计入 168 份 L4 证据。
本构造明确限定先 ready 的 B 为纯 C/V，A 覆盖全部七种 mask；
不把它冒称为 B 为 Mix 的相同原生构造，也不将 CPU 模型当作任意硬件故障退出的证明。
N=1 公共契约仍待用户选择，本轮没有改动；已有性能回退也未因这些功能验证得到解决。

适用 pre-commit 检查全部通过，日志 `outputs/spmd-ready-order-precommit-20261025.log`。
本次阶段提交限定 ready-order 验收、必要的共用探针复用、实际暴露的场景句柄缓存修复、
相关测试和中文过程记录，不包含生成产物或用户未跟踪文件，仅本地 commit，不 push。

## 2026-10-25：隔离测量拓扑重复缓存失效的成本

起点 `da8f676d7`。上一轮实现与提交已完成；提交后构建成功，binding build_commit 与 HEAD 一致，
三个 HBG runtime 二进制 SHA256 与已验证 CSA 版本相同。本轮转向 11.41 倍耗时回退的归因，
不把共享原子读或代码体积直接当作已证明根因。

源码有两个不同候选：scheduler_admission_context_valid 在多层 helper 中重复原子读取
共享 scheduler_error；scheduler_admission_expected_members 每次重建成员集合时遍历全部
worker，对每个 context 的 core_type 和 worker_index 所在两条 line 分别 invalidate＋barrier。
先只隔离后者，暂不改变任何错误检查、block 领取、准入或组协议。

发布链核对：Host 初始化 worker_index，AICPU post_handshake_init 完成实际 core_type／拓扑，
cache_flush_range＋wmb 后通过 hs_config_done 的 release/acquire 才允许各线程发布 READY。
run_resident_scheduler 在 bootstrap 和第一次准入扫描前调用整段 scheduler_observe_data_cache；
本轮运行没有 writer 再改上述 context 前缀。这与动态准入 record／member bitmap 不同，
后者的发布和观察操作仍全部保留。

候选仅在独立 detached worktree tmp/spmd-topology-cache-20261025 的
scheduler_admission_expected_members 中删除两条重复 observe 调用，并注释既有获取边界。
仍逐 worker 校验索引、类型、数量，逐任务校验完整 member bitmap，未新增运行期开关或本地缓存字段。
主工作仓生产代码保持不变；基准固定当前 da8f676d7，而不是改用持续变化的 main。

沿用 benchmark 工作流的独立 venv，Torch 2.12.0+cpu、NumPy 2.5.2、nanobind 2.15.0、
scikit-build-core 1.1.1、CMake 4.4.4 与 PTO pin
418de18dde8fa33773dc5c384fdfe5506463b628 保持一致。58 个冻结数据文件独立导入并校验，
不生成新的 golden。第一次 fixture 导入已复制数据，但随后的身份检查因尚无 _task_interface
而退出，导致其后的 build 没有启动；调整为先 build 再核验，未使用旧绑定。
构建及 fixture 核验日志为 outputs/spmd-topology-cache-build-20261025.log、
outputs/spmd-topology-cache-fixture-20261025.log。

候选的 test_a5_hbg_admission 实际运行 100 项全部通过，包括错误拓扑和非连续 worker 映射；
ctest 汇总的 1 个目标不冒称只有 1 项或新增 100 项。候选严格 CSA L4 验收继续执行。
最初主仓观测 csa-hbg-bench-pf66hu81 的 Device 中位数为 24660.832 μs，
Host 为 106717.269 μs；它处于环境准备阶段，保留但不纳入正式对照。
正式 A/B/B/A 将在构建结束后于同一设备 0 顺序执行，每批三轮正确性、五轮预热和一百轮测量；
完整泳道关闭，Device 仍为含初始化／退出的 AICPU RunWall，不当作纯 kernel 时间。
本机没有 npu-smi／task-submit，CANN query 确认 A5 32C/64V，按用户要求直接用卡，
不宣称设备独占隔离，也不在测量期间查询设备或并行编译。

候选构建完成后，严格 CSA L4 已通过：两轮各七项 golden、2308 条入口身份、133 条依赖、
46733 个协议事件和原有 timing guard 均通过，producer=aicore。Host 的 107 条采集完整，
无丢失记录。候选结果已按字节一致归档到
outputs/spmd-topology-cache-worktree-evidence-20261025/csa-hbg-l4-37scv3ro。
编译参数另核对 Host 49 项、AICPU 32 项以及 AICore 的实际 CCEC 命令：
归一化工作树／venv 根路径后相同；Ninja depfile 哈希差异不当作编译参数差异。
记录见 outputs/spmd-topology-cache-compile-comparison-20261025.log。

正式同卡 A/B/B/A 全部结束，每批三轮正确性、五轮预热和一百轮测量；
A 为原样 da8f676d7，B 仅删除上述两条 observe。
复用 CSA 的 analyze_perf.summarize 重新解析每份 bench.log，与 result.json 完全一致；
冻结输入 manifest、原构图摘要、Torch 和代码提交一致，同一实验臂的源码／二进制身份不变。

| 批次 | Host 中位数（μs） | Device 中位数（μs） | 结果目录后缀 |
| ---- | ----------------- | ------------------- | ------------ |
| A1 | 102094.267 | 24617.169 | csa-hbg-bench-ok9v4d1f |
| B1 | 123086.929 | 24539.536 | csa-hbg-bench-v29doil1 |
| B2 | 101738.210 | 24611.080 | csa-hbg-bench-rq56g0mr |
| A2 | 100980.506 | 24527.915 | csa-hbg-bench-kpzlnx1_ |

两臂各合并 200 个测量样本：Device 中位数 A=24564.380 μs、B=24565.080 μs，
变化 +0.00285%；P10/P90 分别为 23976.630/26143.776 和 24150.215/25270.439 μs。
Host 中位数 A=101571.050 μs、B=102797.530 μs，变化 +1.208%；
B1 的 Host 明显较慢，不能用 Device 相近宣称 Host 同样稳定。
没有观测到可重复性能收益，不把静态调用次数当作耗时归因，候选不合入生产。
两个 B 结果与 L4 目录均经 diff -qr 逐文件核对后归档到上述 worktree-evidence 目录，
候选源码另存该目录的 scheduler_admission.h；完整对照与样本见
outputs/spmd-topology-cache-comparison-20261025.json。A 结果仍在主仓 outputs 中。

## 2026-10-25：隔离 admission 错误状态共享原子读

上一轮完成度答复只核对状态，没有推进实现。本轮先复核并归档上述负结果，
再继续定位性能，不改变未确认的 N=1 语义。

scheduler_admission_context_valid 在 prepare、commit、刷新本核义务、领取、关闭、
GM close/retire/observe/arrive/prepare_lane 等路径内读取同一个 scheduler_error，
且读取使用 fetch_add(0)。外层循环已有低频 query，不能因此推断这些嵌套关键读没有成本。
源码调用关系只是候选证据，还没有证明它导致 11.41 倍回退。

复用独立 detached worktree，先恢复拓扑的两条 observe，再仅把 context_valid 的
scheduler_error 读取从 fetch_add(0) 改为既有 scheduler_gm_query。
这是诊断性替换，不是等价生产修复：ld_dev 对原子写可暂时看到旧值，
即使正常 CSA 的 golden／协议通过，也不能证明错误后停止新 kernel 的契约。
其他原子读取、错误 writer、退出尾部的关键错误读取均不改动；不添加环境开关，
不在主仓弱化错误检查。本实验只判断共享 RMW 读取的性能影响，不能据此合入该替换。

上板 gate 因缺少 npu-smi 无法识别；CANN query 实际确认 device 0 为
Ascend950PR_958b、32C/64V，按用户已明确要求直接用卡，无 task-submit 锁隔离。
构建采用独立 venv 的 Torch 2.12 与既有 PTO pin，日志
outputs/spmd-error-query-build-20261025.log；构建完成前不启动测量或修改其源码。

构建完成后诊断版严格 L4 通过，结果位于临时工作树
outputs/csa-hbg-l4-98xcvvfl/result.json：两轮各七项 golden、2308 条执行身份、
133 条依赖、完整原生协议及原有 timing guard 均通过，producer=aicore。
该运行只证明没有触发错误的 CSA 路径，不证明 query 能替代错误状态的关键原子读。
同卡 A/B/B/A 使用新的四个批次，构建和 C++ 检查全部结束后才开始；
不把上一项实验的 A 样本混入本次正式对照。

为后续优化保留实际错误行为门禁，复用现有 AdmissionFixture／GmMixFixture 新增六项 C++ 回归：
先通过生产 scheduler_record_error 从另一个 worker 发布错误，再验证
ready 所有权不转移、已准备准入不提交、Mix 不继续领取／发布、已准备的 active lane
不能取得 kernel permit、已释放组不增加成功完成计数、局部完成的 sync_start 不补成功 seal。
四项 Mix 检查分别遍历 mask 3/5/6/7；不把循环里的形状数冒称为独立 GTest 用例数。
现有错误首现场和候选／组身份保持不变；没有修改生产代码或增加测试开关。

原样生产实现下两个目标共 184 项通过（admission 102、GM Mix 82），
日志 outputs/spmd-error-stop-cpp-20261025.log。
另在 tmp/spmd-error-stop-mutation 中复制调度头文件，仅删除 context_valid 的错误检查，
使用实际 CMake 编译／链接参数编译独立测试二进制；六项新增回归全部失败，
JUnit 分别核对 2/2 和 4/4 失败，日志 outputs/spmd-error-stop-mutation-20261025.log。
这是检查测试敏感性的故障变体，不是对生产行为的 RED 声明；主仓源码和二进制没有被替换。
CPU 的 query 与 fetch_add 都是宿主原子操作，这组测试不能区分它们在 A5 上的可见性，
因此不会把诊断替换通过这些测试当作生产可合入的依据。

四批正式测量均完成并通过三轮七项 golden；现有 CSA 解析器重算 bench.log 与 result.json
全部一致。完整泳道关闭，同卡顺序执行期间没有并行编译或设备查询。

| 批次 | Host 中位数（μs） | Device 中位数（μs） | 结果目录后缀 |
| ---- | ----------------- | ------------------- | ------------ |
| A1 | 103052.185 | 24824.301 | csa-hbg-bench-dqcpfgq9 |
| B1 | 90570.544 | 11672.296 | csa-hbg-bench-6sv78jz7 |
| B2 | 97394.226 | 11637.792 | csa-hbg-bench-6pxr85zc |
| A2 | 112716.177 | 24610.656 | csa-hbg-bench-smi3zsyj |

每臂 200 个样本合并，Device 中位数为 A=24762.414 μs、B=11667.579 μs，
变化 −52.882%，差值 13094.835 μs；P10/P90 分别为
24186.687/25722.204 与 11233.502/12333.263 μs。
Host 中位数 A=107625.305 μs、B=92743.799 μs，变化 −13.827%；
Host 两批各自存在漂移，单独报告，不能与 Device 合成一个收益分数。

结论是该共享 RMW 读取路径对当前 CSA 耗时有显著影响，
不是说已经完成了可合入的 52.882% 优化，也不是说全部回退都由它造成。
这项差分仍包含改指令后带来的竞争与调度交错变化，不能把 13.095 ms
直接除以调用数当作单次 atomic 延迟。诊断版约 11.668 ms 仍明显慢于设计 AICPU 基线。
后续生产改造必须保留错误后的 kernel entry、准入提交、完成计数和 Mix seal 门禁，
优先区分只观察／无进展轮询与真正控制转移，不能直接推广本实验的一行 query 替换。

两个 B 结果和 L4 已逐文件核对后归档到
outputs/spmd-error-query-worktree-evidence-20261025，含诊断源码；
汇总与全部样本位于 outputs/spmd-error-query-comparison-20261025.json。
源码提交、PTO pin、Torch、冻结输入／构图在四组一致，同一臂的源码及二进制身份一致。
主仓生产 runtime 无改动，本次阶段提交只收录六项错误回归、结果和调查索引。

最终两个 C++ 目标复验通过，共 184 项；适用 pre-commit 检查通过，日志
outputs/spmd-error-stop-cpp-final-20261025.log、outputs/spmd-error-query-precommit-r2-20261025.log。
首次检查发现新表格分隔线未对齐、调查索引不符合既有英文索引规范；
已修正分隔线，并让索引仅按原格式链接本中文过程文档，没有拆出第二份过程报告或绕过钩子。
诊断版三个 runtime 二进制已另存归档 runtime 子目录，SHA256 与捕获身份一致；
旧拓扑候选的 100 项 GTest 原始日志也已保存。
确认相关运行全部结束后删除本次独立临时工作树，原始证据仍可在主仓 outputs 归档中恢复，
不保留可误用的 query 诊断版工作仓。主仓用户未跟踪文件均保持原样，阶段提交不 push。

## 2026-10-25：按 GM 本核阶段跳过必定无进展的 service 调用

起点 9e540ea05，上一轮完成了六项错误回归、两项隔离测量和阶段提交。
共享错误 RMW 已有显著性能差分，但不能把 query 诊断版合入。
本轮只调整 scheduler_service_gm_mix 对既有 helper 的调用条件，不更改原子原语或错误 writer，
不引入错误缓存、降频开关、协议字段、SSBUF 或新的调度顺序。

目前每轮无条件调用 close，然后调用 retire 或 observe，最后无条件 arrive 和 prepare_lane。
这些 helper 在关键错误检查之后，会按本核 phase 直接返回无进展：
retire 只在 AIC 的 RELEASED 阶段推进，observe 只在 AIV 的 IDLE/RELEASED 阶段推进，
arrive 只处理 PUBLISHED，prepare_lane 只处理 ARRIVED。
本核 phase 只由本核改变，不是远端可更新的共享轮询值。

候选在 service 调用处按上述阶段选择 helper，仍每轮先调用完整 close：
它保留关键错误读取、GM context 校验、AIV 的 retained seal 观察和关闭。
真正领取、发布、到达、准备、退役、kernel permit 与完成的 helper 保留原错误检查。
每次调用后重新检查当前 phase，不能把轮首快照贯穿整轮，
否则同一轮 publish→arrive→prepare 会被错误拆开。
retire 成功仍结束本次 service，不在同一调用内顺手领取下一组。

新增 13 项 C++ 检查先在未改动的生产实现上运行：
11 项覆盖 AIC 六个合法阶段、AIV 五个合法阶段的已发布错误，检查无状态／计数推进；
另两项检查 publish→arrive 的同轮推进、inactive lane 资源释放，
以及全部 release 前不退役、退役后下一轮才领取新组。
复用真实生产 helper 进入各阶段，不直接伪造 phase。
测试不声称 CPU 能证明 A5 原子延迟；性能仍需无完整插桩的同卡对照。

固定基线 worktree 为 tmp/spmd-mix-service-baseline-20261025，detached 在 9e540ea05，
使用独立 venv、Torch 2.12、同一 PTO pin 和冻结 CSA 输入。
主仓上一提交的重建与基线构建仍在进行，运行时源码尚未修改。

主仓上一提交重建已成功，binding 为 9e540ea05，AICore runtime SHA256
仍为 76c289ea68f7f0895542e7d0468e7c472472ffaae06192ac217f1962f5ba7ced。
新 13 项检查在原实现上通过，GM Mix 共 95 项；日志
outputs/spmd-mix-service-before-cpp-20261025.log。
随后仅在 scheduler_dispatch.h 落地上述 phase 条件，关键原子 helper 无改动；
改造后 admission 102 项、GM Mix 95 项通过，日志 outputs/spmd-mix-service-cpp-20261025.log。

临时头文件副本另构造两项故障变体，不修改生产源码：
把所有 phase 判断固定为 service 轮首值，两项同轮推进／退役节奏测试全部失败；
仅在 RUNNING 跳过 close，11 项错误阶段测试中的 AIC／AIV RUNNING 两项失败，
其余九项通过。JUnit 核对用例数与失败数，见 outputs/spmd-mix-service-mutation-20261025.log。
这验证了新增条件仍必须允许本轮状态变化、且不能把“无可推进工作”当作跳过错误检查的理由。

基线独立构建已成功；CANN query 再次确认 device 0 为 A5 32C/64V。
上板预检仍因缺少 npu-smi 无法识别，按用户要求直接用卡，未使用 task-submit 锁。
运行进入 UTC 2026-10-26，沿用本次实验启动时的 20261025 日志前缀。
候选重编、扩大的 C++ 回归和基线严格 CSA L4 验证正在进行，尚无本轮性能结论。

候选重编成功；20 个相关 C++ 目标全部通过，实际 721 项 GTest，
原始 GTest 输出保存为 outputs/spmd-mix-service-cpp-regression-gtest-20261025.log。
候选严格 CSA L4 已通过，结果 outputs/csa-hbg-l4-a3kjfkz4/result.json；
基线为独立工作树 outputs/csa-hbg-l4-2lofkxj3/result.json。
两者均为 aicore 来源，两轮各七项 golden、2308 条入口身份、133 条依赖、
完整原生协议和原有 timing guard 通过，原业务 kernel／构图／输入不变。

九类专项仿真测试全部通过，日志 outputs/spmd-mix-service-probes-sim-20261025.log；
本轮显式打开 L4，以各类 CASES 的实际平台筛选逐项定位新捕获，
使用统一 read_perf_data 重新执行完整原生因果审计，要求每份 producer=aicore、
protocol_verified=true，且原始执行行未在解析时丢失。
470 份新捕获全部通过，共 972798 个协议事件、61907 条执行记录，
清单及分组计数见 outputs/spmd-mix-service-probes-sim-audit-20261025.json。
retained 多次调用仍逐次校验 golden 和新鲜来源；上述捕获数不冒称覆盖额外的所有调用快照。
完整 HBG 仿真回归另有 42 项通过，见 outputs/spmd-mix-service-hbg-sim-20261025.log。
继续排除已知误收集的双卡 TMR 目录和未决 N=1 公共容量用例，不把它们计作通过。

A5 九类专项测试同样全部通过，日志 outputs/spmd-mix-service-native-20261025/probes.log；
对应 470 份新 L4 捕获全部经相同完整因果审计，通过 1380222 个协议事件和 81587 条执行记录，
见 outputs/spmd-mix-service-probes-a5-audit-20261025.json。
两平台合计 940 份捕获；分别核对每个平台 CASES 的完整集合，不以文件数量相等替代逐项对应。
测试继续覆盖普通 N=512 分波、GM mask 3/5/6/7、原 N 与槽复用、
47 项基础 sync_start、112 项异构同步／普通交叠和依赖、84 项较晚 ready B 先行，
既有哨兵、每轮新鲜 resident 来源和真实原子事件顺序断言均保留。

基线与候选的 Host 49 项、AICPU 32 项编译命令及 AICore 8 条实际 CCEC argv
在归一化工作树／venv 根路径后完全相同，见 outputs/spmd-mix-service-compile-comparison-20261025.json。
两边 build/pto-isa 的实际 HEAD 均为 418de18dde8fa33773dc5c384fdfe5506463b628。
主仓 Host／AICPU runtime SHA256 与改造前相同，仅 AICore runtime 变化；
独立工作树的路径相关二进制差异不当作算法差异。
完整 HBG A5 回归仍在进行，正式性能测量将在全部回归和审计进程结束后开始。

完整 HBG A5 回归已通过：60 passed、4 skipped，见
outputs/spmd-mix-service-native-20261025/hbg.log。没有改动或跳过新失败项。
正式同卡 A/B/B/A 已在相关构建、审计和测试全部结束后启动；
A 为独立 worktree 的原样 9e540ea05，B 为主仓仅一处 service 调用条件的候选。
每批三轮七项 golden、五轮预热和一百轮测量，完整泳道关闭。

正式四批全部通过，原样重算 bench.log 与 result.json 的统计一致；
两边同一臂的源码／二进制身份前后不变，冻结输入、构图、PTO、Torch 和实际编译参数一致。
结果如下，Device 仍为包含初始化／退出的 AICPU RunWall，不是纯 kernel 时间。

| 批次 | Host 中位数（μs） | Device 中位数（μs） | 结果目录后缀 |
| ---- | ----------------- | ------------------- | ------------ |
| A1 | 105755.094 | 24808.418 | csa-hbg-bench-fjj_tveq |
| B1 | 101458.451 | 23222.263 | csa-hbg-bench-akfmq10w |
| B2 | 100511.835 | 23064.148 | csa-hbg-bench-kl6o3ngt |
| A2 | 102958.212 | 24691.261 | csa-hbg-bench-gp8624xf |

每臂合并 200 个测量样本：

- Device：A=24763.732 μs，B=23135.453 μs，变化 −6.575%，减少 1628.280 μs。
  P10/P90 分别为 24144.372/26067.692 与 22612.178/24075.239 μs。
- Host：A=104099.092 μs，B=100788.646 μs，变化 −3.180%。
  P10/P90 分别为 101580.604/120434.830 与 98824.743/105320.947 μs。

这是保留全部关键错误读取和状态转换检查的实际改造结果，与上一轮仅供归因的 query 替换不同。
候选通过功能／错误回归、完整协议审计，并在两批测量和反向对照中均有收益，保留该改动。
不能把两次不同实验的百分比相加，也不能把此结果称为已恢复设计 AICPU 基线性能：
约 23.135 ms 仍明显慢于初次对照的约 2.151 ms，剩余回退还需继续定位。

完整汇总与样本见 outputs/spmd-mix-service-comparison-20261025.json。
基线两个 benchmark 和 L4 目录经 diff -qr 核对后归档到
outputs/spmd-mix-service-baseline-evidence-20261025；其中 runtime 子目录保留基线三个二进制。
候选二进制保存在 outputs/spmd-mix-service-candidate-runtime-20261025，
候选原始 L4 和两个 benchmark 目录仍在主仓 outputs；归档 SHA256 与捕获身份对应。

适用 pre-commit 检查全部通过，日志 outputs/spmd-mix-service-precommit-final-20261025.log。
确认没有相关活跃构建／运行后已删除独立基线工作树，基线证据可从上述归档恢复；
主仓生产候选保留。两份设计入口只同步当前能力／性能摘要，所有过程仍集中在本文件。
阶段提交限定 service 条件、13 项回归及三份已有中文文档，不提交输出产物、
N=1 未决用例或用户未跟踪文件，不 push。

## 2026-10-26：只在确有普通 payload 可准备时检查准备许可

起点 bb0d976e4；上一轮已提交 GM service 阶段裁剪，实测 Device 降低 6.58%。
提交后重建已成功，binding 与 HEAD 一致，AICore 二进制仍为
d978ea3a80b0f5cbde27495bce6eac788921142d43967e04169f2cfebb248342。

本轮检查 scheduler_fill_ordered_slots：刷新本核义务后，没有协同义务时会先调用
scheduler_can_enter_ordinary，再扫描候选和空槽。因此没有普通候选、候选已经预填，
或没有空槽的轮次仍执行共享错误 RMW 和 commit_tail 关键读，却不准备任何 payload。
这是源码上的无效调用候选，不直接把调用次数换算成性能收益。

候选仅把准备检查移动到首次找到“未准备候选＋FREE 槽”之后、写入槽状态之前。
may_prepare 仍是本次 fill 调用的栈局部值，同轮准备可复用，跨轮不保留；
它不赋予 kernel 执行许可，scheduler_ordered_can_enter_slot 仍逐次关键读取最新 tail。
fill 入口的错误检查、本核义务刷新、协同领取／完成／关闭和真实执行入口均不改变。
无可准备工作时跳过的是准备判断，不是执行判断，也不跳过发现新的协同义务。

新增三项 C++ 检查先在原实现上通过：空轮询仍停止于已发布错误；
空轮询不能授权后来出现的普通任务越过新协同准入；
两个预填 ordinary 槽中的第一个已进入后，第二个仍须重新检查后来发布的协同任务。
最后一项核对第一个 permit_tail=0、第二个在协同义务关闭后才得到 permit_tail=1，
并保留每个任务只执行一次的断言。原实现的 ordered execution 共 63 项通过，
日志 outputs/spmd-ordinary-prepare-before-cpp-20261026.log 和对应 before-gtest 原始记录。

主仓改动目前只涉及 scheduler_dispatch.h 的准备判断位置；
独立基线 worktree 固定 bb0d976e4，路径 tmp/spmd-ordinary-prepare-baseline-20261026，
使用独立 venv、Torch 2.12、相同 PTO pin 与冻结输入。
候选和基线均在构建，尚无本轮设备验证或性能结论。

候选的 ordered execution 63 项、GM Mix 95 项、admission 102 项均通过，
日志 outputs/spmd-ordinary-prepare-cpp-20261026.log。
临时头文件副本分别去掉真实普通 kernel entry 的最新 tail 检查、去掉关键错误读取，
对应新增测试各有一项实际失败，JUnit 均核对 1/1 failed；
日志 outputs/spmd-ordinary-prepare-mutation-20261026.log。
这些故障副本未用于设备运行，生产 helper 的关键错误与实际入口读取都保持原样。

候选／基线构建均成功；20 个相关 C++ 目标全部通过，共 724 项 GTest，
日志 outputs/spmd-ordinary-prepare-cpp-regression-20261026.log 与对应 gtest 原始记录。
归一化路径后，Host 49 项、AICPU 32 项及 AICore 8 条实际编译命令完全相同，
见 outputs/spmd-ordinary-prepare-compile-comparison-20261026.json。
上板 gate 仍因无 npu-smi 无法识别；CANN query 确认 device 0 为 A5 32C/64V，
按用户指示直接用卡，不声称 task-submit 锁隔离；预检日志保存为
outputs/spmd-ordinary-prepare-preflight-20261026.log。
独立基线严格 CSA L4 已通过，位于其 outputs/csa-hbg-l4-ty6i9tky/result.json。
候选 L4 与专项仿真继续验证；新一轮捕获使用运行前路径清单排除历史捕获，
不能把上一轮同名场景的成功记录算进本次验收。

候选严格 CSA L4 已通过，结果 outputs/csa-hbg-l4-rzfs_upz/result.json：
两轮各七项 golden、2308 条入口身份、133 条依赖、完整原生协议与原有 timing guard 均通过，
producer=aicore。九类专项仿真测试全部通过，见
outputs/spmd-ordinary-prepare-probes-sim-20261026.log。
两边实际 PTO HEAD 均为 418de18dde8fa33773dc5c384fdfe5506463b628。
本轮仍需完成新捕获审计、A5 专项、完整 HBG 回归和无完整插桩对照；
不把已通过的功能结果先写成性能收益。

收口审查还发现 test-plan-and-status.md 的细分表保留旧计数、
“当前原生矩阵未包含 sync_start”等与其页首及实现过程矛盾的状态。
需要依据已完成证据统一测试入口和覆盖边界，同时保留真实故障静止、N=1 公共语义等未决范围，
不能只通过改文案把缺少设备证明的项目宣称完成。

A5 九类专项测试全部通过，见 outputs/spmd-ordinary-prepare-native-20261026/probes.log。
每个平台按 CASES 逐项选择本轮新捕获，排除运行前清单中的历史路径；
仿真 470 份、A5 470 份均通过统一读端的完整原生因果审计。
仿真为 972798 个事件／61907 条执行，A5 为 1380222 个事件／81587 条执行，
明细分别保存在 outputs/spmd-ordinary-prepare-probes-sim-audit-20261026.json 和
outputs/spmd-ordinary-prepare-probes-a5-audit-20261026.json。
完整 HBG 回归为仿真 42 passed、A5 60 passed／4 skipped；日志
outputs/spmd-ordinary-prepare-hbg-sim-20261026.log 和本轮 native 目录的 hbg.log。
仍明确排除双卡 TMR 误收集与未决 N=1 容量文件，不把它们算作通过。

同步 test-plan-and-status.md 的当前计数和覆盖说明：ordered execution 63、GM Mix 95、
admission 102，补齐已验证的异构同步和 ready 顺序入口，撤销“当前未包含 sync_start”的过时描述。
保留该文档第四节明确标注的初始失败记录，保留实际尚欠证明的范围；
具体实现过程和性能数据仍只在本文件维护。
适用 pre-commit 通过，见 outputs/spmd-ordinary-prepare-precommit-r2-20261026.log。

所有回归、审计和构建结束后启动同卡 A/B/B/A；A 为原样 bb0d976e4，
B 为同 HEAD 下只移动普通准备判断位置的主仓候选。
每批三轮七项 golden、五轮预热和一百轮无完整泳道测量，分别报告 Host 和 Device。

原 A/B/B/A 四批均通过正确性和计时完整性检查，现有 CSA 解析器重算 bench.log
与 result.json 完全一致。固定的四批结果保存在 outputs/spmd-ordinary-prepare-comparison-20261026.json。

| 批次 | Host 中位数（μs） | Device 中位数（μs） | 结果目录后缀 |
| ---- | ----------------- | ------------------- | ------------ |
| A1 | 168422.594 | 22987.996 | csa-hbg-bench-8iasba9u |
| B1 | 98768.795 | 22300.737 | csa-hbg-bench-xj14x17r |
| B2 | 98663.466 | 22603.330 | csa-hbg-bench-rh2poyl5 |
| A2 | 122954.141 | 23122.565 | csa-hbg-bench-w09jtbt4 |

原四批每臂 200 个样本合并：Device 中位数 A=23039.548 μs、B=22514.035 μs，
变化 −2.281%，差值 525.514 μs；P10/P90 分别为 22464.513/23853.512 与
22004.325/23136.517 μs。幅度较小，接近 benchmark 流程的 ±2% 波动区，不能夸大。
Host 合并中位数为 A=160867.916 μs、B=98689.505 μs，但两批 A 差异很大，
不把表面上的 −38.652% 作为该 AICore 改动的收益。

明确发现 Host 波动后追加 B3→A3，参数仍为三轮 golden、五轮预热和一百轮测量；
它是补充复测，不混入原 A/B/B/A 的 200 样本统计，也不删除原始不利样本。

| 批次 | Host 中位数（μs） | Device 中位数（μs） | 结果目录后缀 |
| ---- | ----------------- | ------------------- | ------------ |
| B3 | 169824.176 | 22957.606 | csa-hbg-bench-_blyo46q |
| A3 | 100094.614 | 23288.372 | csa-hbg-bench-lefnhq_c |

补充 Device 差异 −1.420%，方向相同，但落入上述波动区，不能单独视为显著收益。
Host 则反向变化 +69.664%，进一步排除“本改动带来稳定 Host 大幅收益”的说法。
现有 STRACE 显示 Host 跳变集中在 bind 和 validate：高耗时的 A1/B3 分别约
133.576/134.456 ms 与 10.576/10.862 ms，低耗时的 B1/B2/A3 分别约
69.662/69.614/70.054 ms 与 5.478/5.472/5.496 ms。
这是阶段归属证据，尚未定位其根因，不把它直接归因为页错误、频率或外部负载。
补充检查与阶段统计见 outputs/spmd-ordinary-prepare-followup-20261026.json。

保留该局部实现：它仅把原准备检查移到确有工作的位置，无新增持久状态，
关键执行许可和错误检查均保留；功能与原生审计通过，Device 多批观察方向一致。
性能结论限定为小幅改善观察，不承诺稳定 −2.281%，不与前一项优化的百分比相加，
也不宣称整体回退已解决。准备判断与真实 entry permit 的区别同步写入编码契约。

所有六批源码／二进制身份在各自实验臂一致，输入、构图、PTO、Torch 和实际编译参数一致。
基线的三个 benchmark 和 L4 目录逐文件核对后归档到
outputs/spmd-ordinary-prepare-baseline-evidence-20261026，其 runtime 子目录保存基线二进制；
候选二进制保存到 outputs/spmd-ordinary-prepare-candidate-runtime-20261026，
候选三个 benchmark 和 L4 原始目录仍保留在主仓 outputs。

最终适用 pre-commit 全部通过，见 outputs/spmd-ordinary-prepare-precommit-final-20261026.log。
确认本轮运行均已结束后删除独立基线工作树；基线结果和二进制已归档可恢复。
阶段提交限定按需准备、三个边界回归、准备／执行许可合同和测试状态纠偏，
不包含产物、N=1 未决文件或用户未跟踪内容，不 push。
下一步转向完整目标的逐项验收收口，不能用这些局部优化替代 N=1 公共契约及剩余证据审查。

## 2026-10-26：公共 N=1 同步请求的归一化边界验收

### 用户决策与范围

用户明确要求先验收边界、性能后续自行处理，并确认保留归一化行为。
公共 LaunchSpec 的 N=1、require_sync_start=true 请求被接受，生成的 TaskAttrs 不带
多 block SYNC_START；它不把 Mix 改成纯 V，不取消组内配对、资源握手或完成聚合。
内部绕过公共接口制造的 N=1+SYNC metadata 继续拒绝。

这是对现有公共行为的确认，不是修改 Host 接口来迁就测试。
原先未提交容量文件中七项“公共 N=1 必须拒绝”的预期按确认后的合同替换为完整正向验收；
没有使用 xfail、skip、legacy 回退或省略校验来消除历史红灯。
前面各节当时的未决／失败记录保留，当前决定同步到设计总览、编码契约和测试状态。

### 可执行检查

- 复用 test_hbg_submit_poison.cpp 的真实 Host orchestrator fixture，向毒化任务表提交
  七种 mask × N=1/2 × 请求 true/false，共 28 项。验证输入 LaunchSpec 未改写、
  N 和子任务数量、有效 kernel 与 mask、输出数量、标量内容、真实 fanin；
  N=1 生成标志为 false，合法 N=2 保留请求。纯 mask4 的旧槽归一化与 Mix 固定槽分开断言。
- test_shapes.py 将原单项拒绝明确命名为“显式存储标志”的拒绝，并扩展为七种 mask；
  Task 类型文档明确 sync_start 是协议 metadata，不是公共请求。
- 正式纳入 tests/st/a5/host_build_graph/sync_start_capacity/test_sync_start_capacity.py。
  复用 test_spmd_contract.py 的原编排、三个 record kernel、fanin kernel 和 golden，
  不复制一套 kernel。七个 N=1 场景每个在同一 Worker/callable 上连续执行 1→0→1 请求，
  每轮两项被测任务加真实 fanin 后继，共每平台 21 次成功调用。
- 每次核对 scheduler_mode_publish_count 新增、mode 为 resident、全部整数输出和保护区，
  并使用正式 read_perf_data 的完整协议审计。额外核对生成 metadata 不带 SPMD/SYNC 标志、
  原 kernel/slot/N 不变、exactly-once 身份集合、Mix 仍有两项准入和每组 3 条资源 lane。
  mask6 的实际执行集合仅含两个 AIV；纯 V 无虚假组。kernel 内有界到达检查验证
  N=1 Mix 的真实参与者仍能相互会合，关闭多 block 标志并不取消组内协作。
- 七个超容量场景继续走真实 Host 拒绝：纯 V 的 N=73、C/Mix 的 N=37，覆盖当前实现
  支持的最大 36C/72V 拓扑以上。每个错误绑定使用独立 Worker，要求确切 -7 返回及
  REQUIRE_SYNC_START_INVALID 日志；不是把设备超时算成容量拒绝。

### 已核对的结果

代码与已安装绑定均为 d91821298275817e27d3dd14b9c3d01bd5f00118，Torch 2.12.0+cpu。
本轮没有改动 runtime、公共 Host 实现、业务 kernel 或输入。
架构脚本因 npu-smi 缺失无法识别；按用户的本机执行授权，使用既有 cann_query 查询确认
device 0 为 Ascend950PR_958b、32C/64V。A5 单卡串行运行，无 task-submit 独占，
本轮只作正确性与协议验收，不据耗时作性能推断。

- C++ 真实 Host 目标：29 项通过，含原 1 项毒化测试与新增 28 项；
  outputs/spmd-n1-submit-cpp-final-20261026.log。
- Python 形状边界：196 项通过；完整 hbg_spmd_contracts 包：1540 项通过；
  outputs/spmd-n1-shapes-pytest-20261026.log、outputs/spmd-n1-contracts-pytest-20261026.log。
- 首轮仿真和 A5 各 14 个独立 pytest 用例全部通过，每平台包含 21 次正向调用和 7 次拒绝。
  outputs/spmd-n1-boundary-sim-20261026.log、outputs/spmd-n1-boundary-a5-20261026/pytest.log。
  主 pytest 最后显示 no tests ran 是资源调度器把 14 个普通函数分派到子进程后的汇总口径，
  逐项 PASS 和各子 pytest 的 1 passed 才是本轮用例通过证据。

首轮使用 pytest 临时目录，后续子进程会清理较早 capture；为保留可复核原始证据，
用例改为每个 mask 创建独立 outputs/sync-start-n1-* 目录，并重新执行两平台完整矩阵。
这只改变诊断产物保留位置，不改变参数、调度策略或断言。

最终文件版本在仿真和 A5 均为 14/14 通过：
outputs/spmd-n1-boundary-sim-final-20261026.log、
outputs/spmd-n1-boundary-a5-20261026/pytest-final.log。
另逐文件重新调用归一化检查及正式原生审计，确认每平台恰有 21 个完整 capture、
93 个实际 kernel 执行身份和 24 个 Mix block 组；每个 mask 的三轮 run_epoch 均递增且不重复。
两平台共 42 个原始 capture 的路径及计数见 outputs/spmd-n1-capture-audit-20261026.json。
每个 capture 还包含原始协议和执行记录，可以独立复核，不把 golden 数量冒充协议事件数量。

本轮收口的是 N=1 公共合同，不等于 C01–C56 已全部完成。
后续继续逐项核对边界，尤其保留发布者／协调者失败、阻塞 kernel 的真实静止和回收
尚缺设备证明这一事实；不能把私有错误区的 writer 探针当成实际 runtime 故障停机。
阶段提交仅包含上述测试与中文合同／过程，不包含产物和用户未跟踪文件，不 push。

## 2026-10-26：资源交接、真实超时与输入边界验收

### 审计结论和本轮范围

继续遵守“先验收边界，性能后续处理”。按 C01–C56 对照真实源码与现有证据，
不把模型、生产 helper、原生 kernel 三个层次互相替代。已有 A5 capture 的 96 个紧凑
worker 对应不连续物理 ID，C11 并非完全缺少实卡证据；本轮主要补 C19/C27 的受控资源交接、
C21 的来源守卫、C31/C32 的真实有界失败、C03/C04 的公共入口边界，以及 C34 的实际能力描述。

### 普通任务尚占用 Mix 资源：24 项 helper 反例

test_hbg_gm_mix.cpp 新增四种 mask × 三条物理 lane × 两个 own slot，共 24 场景。
普通任务通过真实 route、batch push、fill、entry permit 进入执行状态，不只是手填一个忙标志。
分别在“尚未返回”和“local completion generation 已写、尚未消费”时尝试 Mix arrive，
必须仍为 PUBLISHED、arrived_seq=0，其他伙伴不得 prepare；真实 completion 消费后才允许握手。
inactive lane 仍履行资源义务但不增加 kernel；slot0 场景同时验证另一个普通候选被停放、
在 Mix 关闭后恢复，不丢失也不重复。

完整 GM helper 目标 119/119 通过，日志 outputs/spmd-mix-ordinary-resource-final-20261026.log。
临时副本只删除 scheduler_gm_mix_arrive 的 busy-slot 检查，24/24 红灯，
首个错误即非法推进为 ARRIVED；主生产源码未改。
副本位于 tmp/spmd-mix-arrive-mutation.X37ooR，日志
outputs/spmd-mix-ordinary-resource-mutation-20261026.log。
另以合成坏事件检查正式原生审计器，它拒绝旧 ordinary kernel 内出现新的组调度事件；
未发现这一 reader 漏洞，不为此另造审计器。
这些是受控 helper／坏记录证据，不声称在 A5 上精确暂停了同一时点的普通 kernel。

### 宽 block 请求静默窄化：真实 RED 与最小修复

真实 Host fixture 补 32 项可表示范围边界：七种 mask 的 0/-1/INT16_MIN，
双 lane 的 16383/16384、三 lane 的 10922/10923、单 lane 的 32767。
拒绝要求 INVALID_ARGS、无有效 TaskId、allocator 及下一条 poison storage/state 均未改变；
合法值核对原 N、精确乘积、kernel 映射和 PENDING。与此前 29 项合计 61/61 通过，
见 outputs/spmd-host-shape-boundary-full-20261026.log。

继续追查窄化前的输入发现真实缺陷：set_block_num 的 int16_t 形参使宽值在进入函数前
丢失信息；本机 int64_t 的 65537、131073、-65535 都变成 1，真实 submit 返回有效任务。
CSA 生成编排存在直接传 int64_t 表达式的合法调用，不能只靠模型拒绝宽整数证明公共接口。

新增六个 int64_t 极值／越界值 × 七种 mask × sync false/true，共 84 项正式回归，
直接传宽值，不在测试里先转 int16_t。修复前 42 失败、42 通过；42 个失败不仅错误接受
任务，还改写 poison storage/state，日志 outputs/spmd-host-wide-count-red-20261026.log。

公共 common HBG setter 改为 int64_t 形参，在 int16_t 范围内原样保存，范围外保存现有非法 0，
交给既有 Host submit 拒绝。没有新增字段、宏或重载，LaunchSpec 的 sizeof=4、align=2
由实际编译断言保护；N=1 归一化和合法 N、可表示负值的旧行为不变。
这影响 A5/A2A3 的 HBG，不顺带修改独立的 TMR setter。
修复后完整 Host 目标 145/145 通过，见 outputs/spmd-host-wide-count-green-20261026.log。

兼容边界不能省略：精确成员函数指针签名发生变化，仓内未找到这类绑定；
旧已编译编排仍需重新编译，不能只更换 runtime；调用者显式先 cast 后丢失的信息无法恢复。
现有磁盘编译缓存递归哈希传递头，实际 CSA 的 32 个 include 闭包文件包含本 common 头，
SPMD probe 的 24 个闭包文件也包含；新 Python 进程会使旧编排缓存失效。
进程内 _compile_cache 在内容 key 前返回，热改头后仍须重启进程或显式清缓存。

存量 host_build_graph_validation 新增六种宽输入的真实编排路径，A5sim/A2A3sim 各
14/14 通过，分别见 outputs/spmd-host-validation-a5sim-20261026.log 与
outputs/spmd-host-validation-a2a3sim-20261026.log；不修改原八种拒绝场景的含义。

### predicate：来源与拒绝形状分别验收

predicated_dispatch 的真／假两场景增加每轮 publish_count 与 resident mode 检查，
放在真实 compare_outputs 路径，standalone 同样生效；拒绝 skip_golden。
mix_predicated_dispatch 复用原有效 kernel，扩展为 Mix N1/N2 和纯 C/V N2，共 22 个拒绝场景，
每个形状分别使用真假 gate。必须同时出现 prepare -1000 与生产 Host 的完整合并诊断，
不能把任意编译／绑定失败当作 predicate 拒绝，也不把合并诊断称为专用 predicate 错误码。

两平台的 22 个独立拒绝用例和一个 scene-test 类内的两个正向场景均通过，
日志 outputs/spmd-c21-predicate-sim-20261026.log、outputs/spmd-c21-predicate-a5-20261026.log。
新增 test_predicate_native_fixture.py 的 25 项 CPU 检查覆盖错误 mode、旧 latch、
重复轮次缺发布、坏 golden、skip 和真实参数矩阵；它们不是设备执行。

### 有界 kernel 阻塞触发真实 HBG watchdog

新增 resident_failure 原生入口及薄 kernel；复用 ExecutionRecord、verify_fanin、正式编译器。
七种 mask 的每个实际 kernel lane 分别阻塞，N=1/2，共 24 个场景；其中 mask6 的
AIC 仍只是协调者，实际阻塞者为 V0 或 V1。kernel 按设备 SYS_CNT 等待 2500ms 后返回，
复用已有 SIMPLER_SCHEDULER_TIMEOUT_MS=2000，不引入新生产开关，也不直接写真实错误区。

先健康运行，再在同 Worker/callable 上触发超时，要求真实 finalize -100。
同一外部分配 buffer 在错误返回后正常 D2H，检查已进入 kernel 的最后写入与返回标记、
未执行后继的零哨兵、所有 padding/保护区；错误后同 Worker 重提必须拒绝，
两个 caller buffer 的 free 必须因仍被 run 借用而拒绝。close 有界结束后，新 Worker
运行同 callable 的健康版本并通过完整 golden 与成功原生审计。

这里没有修改 Worker 的全局故障合同：当前任何 finalize 失败都会保守毒化 lane、
保留 caller borrow；kernel 自行返回不等于调用者可以复用失败 Worker。
初次草案根据 executor 内部支持而使用一个 AICPU 线程，实际 healthy prepare 即 -1000；
核对公共 sim/onboard runner 后确认都要求 0 或至少两个线程，改用合法的两个线程。
执行进度 watchdog 与 peer shutdown-barrier watchdog 可以竞争首写，合法 site 分别为 95/102，
都必须对应真实 scheduler_error=TIMEOUT(10)，不接受 bootstrap 等无关超时。

A5 的退出 ACK 预算固定为 1 秒，sim 为 10 秒；有界 kernel 在调度超时后约 0.5 秒返回，
不是用超过退出预算的长 stall 冒充正常退出。本用例不声称验证 STARS 杀死不可返回 kernel。

失败 capture 不要求成功协议的 protocol_verified：检查实际 entry/returned/visible 及其
同核 sequence、真实持续周期、后继无入口、未伪造 completed_blocks/TaskDone/resolved_count。
全部 worker 必须唯一完整且 finished=2、无协议记录错误；这只是 ACK 前的失败记录，
另外按 stalled worker 的紧凑 index 对应实际 AICPU 分区，要求该分区的 EXIT 发生在
kernel 内，exit_wait 结束不早于该 kernel 的 visible，再结合 native fence 返回、
buffer 保留、close 和新 Worker 恢复构成证据。不能用别的分区的晚结束时间顶替。

CPU 坏记录自检先抓出 missing/duplicate worker 两个漏检（2 RED），加严集合后转绿；
另增加错分区、提前退休、缺返回、假完成、重复控制记录等反例。
独立 review 再发现自洽截短 worker 清单和 EXIT start 不能代表发送结束两处弱断言；
两项先 RED，再以 metadata.num_cores 为独立数量锚点、要求所属分区的 signal_end
早于 kernel returned 且不晚于 wait_start，转绿。正式 oracle 共 34 项。
测试日志保留在 outputs/spmd-runtime-failure-oracle-red-20261026.log 及后续 green/final 日志。

仿真与 A5 的完整矩阵均 24/24 通过，每平台 72 次实际运行（48 次健康、24 次预期失败），
另有 24 次重提被毒化 lane 拒绝，未计作新的设备执行。
原始目录 outputs/resident-failure-* 每项保存三份 capture 和 result.json；
汇总见 outputs/spmd-runtime-failure-capture-audit-20261026.json。
全部 144 份捕获独立复核：96 份健康记录走正式成功审计，48 份失败记录走上述失败断言，
失败不是伪造成功记录。加严完整 worker／所属分区检查后对全部旧捕获重审通过；
wide setter 重编后又在 A5 跑纯 V1 和 2V Mix 两项，覆盖最终检查及新 Host 二进制。
日志 outputs/spmd-failure-matrix-sim-20261026.log、
outputs/spmd-runtime-failure-a5-20261026/matrix.log 与 postfix.log。

该证据覆盖有界 kernel 阻塞触发真实调度错误及失败隔离；
不代替组描述发布前协调者失败、无 kernel 的 AIC 协调者物理失败、不可返回核的全部退出证明。
未把这些剩余范围写成已经完成，也未用性能测量替代边界验收。

### 重编和原样业务回归

common 头变更后 editable 重编通过，日志 outputs/spmd-boundary-wide-count-build-20261026.log。
原样 CSA 的严格 L4 入口通过，结果 outputs/csa-hbg-l4-jt01od6w/result.json：
两轮各七项 golden、2308 个执行身份、133 条依赖边、完整原生协议、原时序守卫，
来源为 aicore，业务源码仍匹配冻结基线。记录包含本轮 tracked diff 和实际三份二进制身份，
没有修改 CSA 输入、kernel 或 block 数。

### deferred API 能力边界的原生检查

审计纠正了“async 与 Graph 一样由 Host 自动识别并回退”的笼统描述：Host 只显式识别
Graph 等现有形状，不能通过 kernel 函数识别任意异步行为。resident 不提供 deferred
上下文，register_completion_condition 返回 false；save_expected_notification_counter
丢弃返回值，不自动等待。SDMA adapter 另有 event.Wait 分支，但本轮不声称验证了
真实 SDMA 的同步降级或错误传播，也不新增异步返回支持。

新增 test_async_boundary.py，复用原 SPMD 编排、ExecutionRecord、verify_fanin 和编译 helper。
薄 kernel 在原记录的空闲 metadata 中记录有效 token、三个非空指针、非零容量的位掩码，
两次公开注册返回值及完成哨兵；正常 resident 的前三项均为零，哨兵为 0xC34。
counter 使用真正分配且已满足的零值，不发 DMA；即使错误走 legacy，注册条件也不会
人为制造无界等待，但来源／返回值／上下文检查必须失败。

七种 mask × N1/N2，共 14 场景；同 Worker/callable 每场景三轮，逐轮 fresh resident
来源、完整原身份、保护区、API 记录和真实 fanin 均验证。新增 CPU oracle 54 项，
检查错误上下文、错误返回、缺哨兵、旧来源、坏输出、跳过校验及父类不得混收。

A5 首轮通过；sim 首轮编译失败，首错误是 PTO type.hpp 使用 uint8_t 而探针此前未先
包含 cstdint。补齐探针自己的标准类型头后，两平台最终各 14 场景、42 次运行通过，
日志 outputs/spmd-c34-async-native-sim-final-20261026.log 与
outputs/spmd-c34-async-native-a5-final-20261026.log。没有修改 PTO 仓、公共 async API 或 runtime。
本用例按三轮功能口径运行，不声称三轮都采集了完整 L4。

Python 合同包此前 1651 项通过，outputs/spmd-boundary-contracts-all-final-20261026.log；
review 新增两个坏记录反例后的最终数量与日志见本节收尾。
Host submit 与 GM Mix 两个最终 C++ 目标分别 145、119 项通过，合计 264 项，
outputs/spmd-boundary-cpp-final-20261026.log；不能把两个目标通过写成全部 C++ 仓库通过。
最后按完整 worker、正确生命周期分区、唯一 task_controls 及同 Worker epoch 递增
重新审计原 144 个 capture 和重编后追加的六个 capture，结果见
outputs/spmd-runtime-failure-capture-final-audit-20261026.json。

当前仍不以这些阶段检查宣布整体完成；原目标的逐项证据审查继续，
未覆盖的不可返回／发布者物理故障等范围保留明确边界。阶段提交使用详细中文日志，不 push。

本节最终收尾：review 后 Python 合同包 1653/1653 通过，
outputs/spmd-boundary-contracts-reviewed-final-20261026.log；150 份捕获按最终 worker 数量锚点
及所属分区 signal_end 时序重审通过，outputs/spmd-runtime-failure-capture-reviewed-audit-20261026.json。
全部适用 pre-commit 通过，outputs/spmd-boundary-precommit-final-20261026.log。
用户要求本次提交完成后先停下；提交后暂停，不再启动新编译、测试或后续开发，不 push。

## 2026-10-26：恢复边界验收与证据收口

用户恢复目标并要求完成剩余工作，同时澄清此前对原始 FDWIC 文档的讨论仅作参考，
不转成本轮的新实现要求。继续验收当前 HBG／SPMD 合同，不因此改回 SPSC、改变 2V
协调者或拆除现有 Mix 资源握手；性能优化仍留待后续。起点为已推送的 `98215088b`，
本轮新增提交不 push。既有未跟踪 CLA、TMR 和 atomic_probe 文件继续保持原样。

先补 C39 的 B=Mix 原生组合。已有三节点 P→A、独立 B 用 P 的非成员核等待 B，
但 Mix 包含全部资源成员，不能直接把 B 的 mask 扩为 Mix：若 B 先准入，P 可能被
普通入口挡住，而 B 的尾部等待 A、A 又等待 P，会由测试构造自己制造循环等待。

拟复用现有 kernel／编排接口加入一个普通 V 的桥接任务 Q：P 为普通 C，Q 在 kernel
里等 P 已进入后返回；真实图依赖是 P→A、Q→B，P 等 B 首次进入，B 尾部等 A 首次进入。
Q 与 P 不争同一个执行核；B 只有在 P 已进入后才 ready，其他 cluster 可先执行 B
并让 P 返回。该控制只用于测试，不修改 runtime、参数 ABI 或添加行为开关。
将保留原三节点纯 C/V 场景，并覆盖 A/B 七种 mask 与普通／sync_start 的四种组合。
正式测试和实测结果在完成后追加，不将此计划当作通过证据。

### C39：A/B 全 mask 的 ready 顺序与局部重叠

上述四任务桥接已落地，保留原三任务纯 B 路径；复用原 sync_ordered kernel 和 plan ABI，
没有修改生产调度器。矩阵从每平台 84 项扩为 196 项，合计 392 项。
首次 RED 在 editable 重编期间误用了旧版本绑定，五项 import/setup error 不计作有效行为红灯。
重编完成后真正的夹具 RED 为五项：原矩阵缺少 Mix B，四种 Mix B 均没有桥接。
补齐后 ready fixture 429/429 通过，含 20 个 Mix B 错误时序／桥接反例。
日志为 outputs/spmd-c39-mix-b-fixture-red-20261026.log 与 green 日志。

本轮 editable 重编成功，outputs/spmd-boundary-resume-build-20261026.log；
本机无 npu-smi／task-submit，既有 CANN query 确认 Ascend950、32C/64V、A5，
按用户要求直接使用 device 0，不把本轮无独占条件下的时间作为性能基线。

仿真／A5 的完整矩阵均 196/196 通过，另有两平台各两项 smoke。
矩阵启用 L4 与独立 dep-gen：每份捕获均走 read_perf_data 的正式因果审计，
要求 sequence_integrity_verified 和 protocol_verified；再比对原输入的 N、函数／mask、
真实 P→A／Q→B、独立 deps.json 及 B→A 的共同日志。
纯 B 的准入日志为 [2,1]，Mix B 为 [3,2]，不靠 Host task_ref 顺序推测设备执行。
392 份均通过，明细及原始文件哈希在 outputs/spmd-c39-capture-audit-20261026.json。
运行日志为 outputs/spmd-c39-mix-b-a5-matrix-20261026.log 和 a5sim 对应日志。
error 日志级别没有 Host 时钟对齐记录，不影响本核 sequence／跨核 ticket 的协议证据；
不据此声称 Host 与设备精确同钟或得到性能收益。

本项完成后的合同包 1901/1901 通过，outputs/spmd-boundary-contracts-resume-20261026.log。
后续不可返回 kernel 的夹具又增加测试，最终总数另记。

### C31/C32：不可返回 kernel 的真实超时、保留与恢复

复用 resident_failure 的编译入口、ExecutionRecord、真实 fanin 和 caller buffer，
测试 kernel 的负 stall_ms 输入表示 block0 指定 lane 永久循环。其余输入仍为原有有界等待。
没有新增生产 env、宏、错误注入接口或特殊 runtime；正式 runtime 二进制没有变化。
单独的硬件测试固定 N=2、遍历七种 mask 的全部 12 条实际执行 lane。
仿真无 STARS，平台 marker 与运行时断言都禁止该入口进入 a5sim；原 24 个有界场景仍可仿真。

超时链复用现有 TMR 故障用例的配置：scheduler 2000ms、op 3000000us、stream 4000ms，
不启用 SDMA。测试仍先健康运行，在同 Worker 上失败，然后验证重提和两个 caller buffer
的 free 均被拒绝；close 后新 Worker 使用同 callable 再通过完整输出和成功原生协议。
STARS 故障后设备访问可能已被毒化，不发 D2H；失败结果明确 output_verified=false，
并且要求存在真实 Host 导出的失败协议，不将缺失 capture 静默当成功。

失败 oracle 复用已有检查并区分两类：有界 kernel 必须有真实 return/visible 和全部失败退出
worker；不可返回 kernel 必须有精确 block/slot 入口，return、visible、消费、trace 发布均为零，
唯一未确认退出的 worker 必须就是阻塞者。所属 AICPU 分区的 EXIT ACK 等待至少一秒，
其他分区较晚的时间不能替代本分区。两类均保留真实根任务／后继、完整 worker 集合、
首错误 TIMEOUT(10)、非成功 task_controls、未释放 fanin 和 epoch 检查。
全局 watchdog 不知道的首错误字段仍保持 unknown；不拿首写者冒充实际阻塞 kernel。

新增 CPU 检查先 RED，再补入两类判定后 51/51 通过。包含缺入口、假返回／可见／消费／trace、
假退出 ACK、多一个未知未退出核、错 AICPU 分区、短 ACK 等待、提前退出和伪完成等反例；
这是 oracle 自检，不算设备故障。日志 outputs/spmd-nonreturning-oracle-red-20261026.log
与 green 日志，平台隔离检查另有 fixture-red／fixture-green 日志。

A5 全部 12/12 通过，单次 run 报错 4.319–4.337s，失败 Worker close 8.680–10.147s，
这里只用来验证有界退出，不能当作正常调度性能。36 份捕获包含 24 份健康成功协议和
12 份不可返回失败协议，离线按正式 reader／失败 oracle 重审通过，三轮 epoch 严格递增。
原始 capture 与哈希、每项三轮结果见 outputs/spmd-nonreturning-capture-audit-20261026.json；
设备矩阵日志 outputs/spmd-nonreturning-a5-20261026/matrix.log。
最初单项 smoke 的 result 未挂入失败 capture，后续独立读取确认原文件存在，
随后正式矩阵改为强制验证失败记录；不拿初版宽松 smoke 替代最终 12 项结果。

本证据证明执行 kernel 永不返回时的当前超时链、caller borrow 保留、错误后拒绝与恢复。
组描述发布前／无 kernel AIC 协调者的错误门禁，仍使用现有生产 helper 的受控阶段测试，
其中四种 Mix mask 均覆盖失败后不得领取／发布／执行／贡献完成／写成功 seal。
这些证据层次已经明确；不为匹配参考方案而添加生产故障开关，
也不把“任意物理坏核、驱动失效、掉电后继续运行”追加为本次实现要求。

### 参考边界 C01–C56 的覆盖核对

下表是对当前实现的证据索引，不把参考文档中的建议、旧 FDWIC 的通道形态或未开放能力
追加为强制实现要求。H 表示真实 Host／生产 C++ helper，M 表示模型／坏记录检查，
D 表示真实 runtime／kernel 的仿真或 A5 记录。三者不能互相替代；
普通多 block、Mix 分波、sync_start 最终一致推进和原样 CSA 仍是本次功能主线。
C++ 入口在 tests/ut/cpp/a5/runtime/host_build_graph，模型在 tests/ut/py/hbg_spmd_contracts；
原生专项入口位于本目录，另有 tests/st/a5/host_build_graph 的能力及失败回归。

| 编号 | 当前实现与对应证据 | 明确的范围 |
| ---- | ------------------ | ---------- |
| C01 | H submit/layout、M shapes、D spmd_contract | 原 N、block/slot 集合与全输出；不只比较计数 |
| C02 | H ordered、D ordered/gm_mix_execution、原样 CSA N=512 | 普通分波；不同平台容量另按本轮拓扑检查 |
| C03 | H submit_poison 的宽 int64、0、负数；D validation | 窄化前拒绝，非法请求不得改 poison storage |
| C04 | H submit_poison/layout 乘积、前缀与对齐溢出 | Host 可表示范围而非 uint16 字段最大值 |
| C05 | H admission 容量、D 纯 C/V、sync_start | V 按实际 AIV 数，不按 cluster 数 |
| C06 | H gm_mix、D 四种 Mix mask 的身份／组协议 | mask6 的 C 只协调，不伪造 kernel |
| C07 | H zero-work close、M order、D 小 N 的组协议 | 零工作成员仍关闭义务，非成员可直接越过 |
| C08 | H admission/layout 坏拓扑、M 原生审计坏记录 | 唯一位置、冻结成员和重复身份均拒绝 |
| C09 | M/native fixture 对漏块和重复块的反例、D 精确集合 | 总数正确不能掩盖身份错误 |
| C10 | H admission/bootstrap 无效拓扑与成员校验 | 不在线缩减已发布成员以掩盖缺核 |
| C11 | H 紧凑位图、M 稀疏 ID、D 96 worker 的稀疏物理 ID | 物理 ID 与 worker index 分离 |
| C12 | D 原样 CSA、原生 task view、payload_context | 保留业务 ABI；不擅自给共享 tensor 叠加 block 偏移 |
| C13 | H ordered/gm_mix、D overlap 与 sync_ordered | 本地关闭可越入独立后继，不等远端 TaskDone |
| C14 | D 相同负载的 edge/fanin、独立 deps 和完整协议 | 真依赖仍等全部前驱输出发布 |
| C15 | H admission 并发提交、M 反序死锁反例、D order_key | 执行依据共同准入，不依据各核观察顺序 |
| C16 | M 部分相交集合、H 成员投影、D C/V/Mix 交叠 | 当前生产选固定类型／完整 cluster，不新增任意用户成员接口 |
| C17 | H ready/wake、M 固定 seed DAG、D CLA/CSA | 完成驱动推进链、菱形及跨类型 fanin |
| C18 | H 并发 wake-list、D fanout_imbalance | 有限 HBG 内不丢唤醒；不承诺无限生产者下的固定等待轮数 |
| C19 | H 24 项真实 ordinary→Mix 资源交接、M 坏记录 | 精确暂停点是 helper 证据，不冒充硬件逐指令暂停 |
| C20 | H partial completion、D 慢头／中／尾与 fanin | cursor=N 不等于 TaskDone |
| C21 | H predicate、D 两个正向及 22 个拒绝组合 | 单 block 原语保留；多 block／Mix predicate 未开放 |
| C22 | H inline_services、M 空图、D 带 Dummy 的 CSA | 无 kernel 服务不增加物理执行计数 |
| C23 | H sync_once、D sync_start/sync_ordered | 早到可执行，晚到成员的前序义务不被截走 |
| C24 | H 容量门禁、D 满容量和超容量拒绝 | 两个 pending slots 不算两个物理位置 |
| C25 | H gm_mix/payload_context、D 原生组协议与通信 | 每 lane 自备参数，同一组身份配对 |
| C26 | H fast cluster、D 独立 Mix 跨 task 重叠 | 不新增全 cluster 的 TaskDone 屏障 |
| C27 | H 四种 mask × 三条 lane × 两槽资源交接 | inactive lane 仍等待旧资源，但不计作 kernel |
| C28 | H generation/seal、D 多 block 连续复用与 CSA | 每组 release 收齐再复用 mailbox |
| C29 | H dispatch/gm_mix/layout、M 旧 epoch/generation 反例 | 旧消息不贡献新一轮完成 |
| C30 | H 布局／序号极值拒绝、M wrap 反例 | 不依赖实际执行到 uint64 上限才验收 |
| C31 | H 已发布错误在各阶段关闭入口，D 有界／不可返回 kernel | 非执行协调者／发布者失败的阶段门禁用 H；不宣称任意物理故障注入 |
| C32 | D 真实超时身份、borrow 保留、拒绝重提、close／恢复 | 不可返回核不伪造 ACK，也不做故障后 D2H |
| C33 | H retained reset、D 大小 N 交替及同 callable 重复 | 每轮新 epoch／新来源，未使用区不读取旧状态 |
| C34 | D Graph 回退和 deferred API false、M 来源守卫 | 不开放异步返回；不把 API false 说成自动识别任意 async kernel |
| C35 | H payload_context、D 精确 block/slot/N/generation | own slot 复用不覆盖其他 block 的上下文 |
| C36 | H 发布原语、D 原生 ticket／本核 sequence 因果链 | 输出可见与完成消费分开；不只看 counter 阈值 |
| C37 | H 不可撤销 cursor/slot 所有权、M 迟到转移反例 | 当前无动态转移接口，不增加已关闭成员 |
| C38 | H ready/admission 一次性所有权及旧 generation 拒绝 | 不把旧 intrusive 节点无条件再次入队 |
| C39 | D 新增每平台 196 项，392 份 L4＋独立 deps 审计 | A/B 全 mask，ready B 先行且 A 可早于 B 尾部进入 |
| C40 | H 真实并发准入、D 唯一共同日志／身份审计 | 同 task 只准入一次，不按 block 重复 generic pop |
| C41 | H 准备不发布、锁竞争不造洞、坏记录拒绝 | 物理发布者任意停机未逐点注入；错误使用 run 失败链，不抢锁恢复 |
| C42 | D overlap、C39 与协议里的独立后继准入 | 短锁不持有到 task 全局完成 |
| C43 | H nonmember entry、D C/V 独立重叠 | 无义务且无资源冲突的类型不被全局最慢核挡住 |
| C44 | H 两候选停放、direct refill、逐次 entry permit | 不丢所有权，不重复 predicate，不沿用旧许可 |
| C45 | H ordered/Mix service、D 有限图和长串 Mix | 已知义务优先；候选池不保证固定轮数内发现全部协同任务 |
| C46 | H 冻结位图／坏成员拒绝、M 迟到成员反例 | 不支持发布后扩张成员集合 |
| C47 | H cursor 饱和、未退役 slot 和暂停领取者 | 无转交事务；不可撤销所有权成立后才允许零工作关闭 |
| C48 | H claim 后暂停 publish、D 组发布／close 审计 | 没看见新 mailbox 不等于没有未发布组 |
| C49 | H 保留 seal 与最后组交错、D LocalDone 因果链 | seal 携带最后序号，不覆盖后就丢旧关闭通知 |
| C50 | H startup/drain、M 暂空、D 长尾与 fanin | 暂空不当 EOF；正常退出排空控制义务 |
| C51 | M fresh mode/count 反例、D 每轮来源与 aicore producer | 不凭旧 latch 或正确 golden 接受 legacy |
| C52 | H 能力门禁、D 支持／拒绝／回退分组 | 保留未支持形状，不为“全绿”盲改 marker |
| C53 | H nullptr SSBUF startup、D 全部 GM Mix 及 CSA | 实际 resident 路径不要求 SSBUF |
| C54 | H pending/prepared 禁止 kernel entry；提交代码有限短段 | 锁内无 kernel、成员／ready 等待；不改为全任务锁 |
| C55 | H sync 提前关闭、D 有界 barrier 的迟到／重叠场景 | 保证最终一致位置，不增加同时开始门槛 |
| C56 | H mask6 coordinator/close、D 2V Mix 全身份 | AIC 协调义务完成后才局部关闭，不增加 C kernel |

本轮重编并运行上述 A5 HBG 目录全部 21 个 C++ 目标，共 842 个 GTest 用例通过，
日志 outputs/spmd-boundary-a5-hbg-cpp-build-20261026.log 与 cpp-tests 日志。
不把这些目标说成整个仓库所有 C++ 单元测试已运行。
同时纠正测试入口页 ordered 目标数量的旧笔误：实际为 62，而非 63。

Python 全量首轮在收集 Qwen 用例时发现 .venv-csa 缺少已在 pyproject test extras 声明的
safetensors；只在该虚拟环境补齐 0.8.0，没有改系统 Python、Torch 2.12 或测试预期。
完整重跑 tests/ut/py 得到 5842 passed、84 skipped、1191 warnings，耗时 592.78s，
outputs/spmd-boundary-all-py-resume-final-20261026.log。首次收集失败日志保留，
不把缺依赖的运行记为通过；84 项既有跳过也不算已验证。

### 本轮回归与收口

拆出共用故障执行器、增加负 stall 输入后，原有有界故障场景在仿真／A5 再次各 24/24
通过，没有只跑新故障入口。两平台合计 144 份捕获按最终 oracle 重审，96 份健康协议、
48 份有界失败协议全部通过；outputs/spmd-bounded-refactor-capture-audit-20261026.json。
运行日志 outputs/spmd-bounded-refactor-sim-20261026.log 和 a5 对应日志。
仿真执行时明确 deselect 12 项不可返回用例，避免在无 STARS 的环境中永久阻塞。

最后原样 CSA 的严格 L4 再次通过，outputs/csa-hbg-l4-pug5_ig0/result.json：
两轮各七项 golden、2308 个执行身份、133 条依赖、96 worker、46733 个协议事件，
producer=aicore、protocol_verified/block_identity_verified 均为 true。
冻结业务源码仍一致，Torch 为 2.12.0+cpu；使用 HEAD 98215088b 的生产 runtime，
本轮修改仅为测试和说明。该结果的 tracked_diff_sha256 只覆盖 src/python/simpler_setup，
空值不代表整个工作树干净；测试／文档 diff 由本次提交保留。
CSA README 的旧“sync_start 尚未完成”文字同步纠正：CSA 自身不使用该属性，
同步能力由独立矩阵验收，不修改冻结业务代码来制造同步证据。

C39 另逐项复核 frozen flags 与输入 sync_flags，392/392 一致，
outputs/spmd-c39-flags-audit-20261026.json。最终专项 Python 合同包为 1918/1918，
outputs/spmd-boundary-contracts-resume-final-20261026.log；适用 pre-commit 全部通过，
outputs/spmd-boundary-resume-precommit-20261026.log。

当前功能和边界验收按以上支持范围收口；不以参考文档添加旧 SPSC/SSBUF 架构要求，
也不把未开放的 predicate／Graph resident／deferred／动态转移扩展列作本轮必做。
性能优化仍按用户安排留待后续，现有性能回退没有被“测试通过”掩盖。
阶段提交仅包含本轮相关测试与中文记录，不包含既有未跟踪文件，不 push。
