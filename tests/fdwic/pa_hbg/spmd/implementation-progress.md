# A5 HBG SPMD 实现过程记录

## 目标、依据和验收边界

开始日期：2026-10-25；起点：`6c367c69f`；工作分支：`feat/fully_distributed_within_core`。
按用户要求逐步实现、阶段性详细中文 commit，**不 push**。
用户随后单独授权的推送已完成至 `6f2b470d4`；后续实现继续遵守不自动 push 的原目标。
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

## 分步交付与状态

| 顺序 | 交付 | 出口证据 | 当前状态 |
| ---- | ---- | -------- | -------- |
| 前置 | ordinary／Mix payload context 初始化与重绑定 | 原生 N=1 红灯转绿；真实初始化函数的 C++ 测试 | 已通过 C++、仿真与 A5 六种 mask 各十轮 |
| 1 | 有界存储、准入记录、block 控制、GM 组状态 | POD／布局／溢出／重跑初始化测试 | 已落地；纯 C/V 和 GM Mix 已使用 |
| 2 | 一次 ready 领取、共同准入、三入口统一顺序门禁 | 发布竞态、B 可先准入、非成员旁路、无重复入队 | own/direct、持久候选与 GM Mix 均已接线；完整事件审计待补 |
| 3 | 纯 C/V 多 block、完成聚合、分波容量 | N=1／容量两侧／512；身份不重不漏；独立任务可重叠 | A5／仿真功能和 L1/L4 逐 block 检查通过 |
| 4 | GM Mix descriptor／ack／seal、各 lane 自备 payload | mask 3/5/6/7；最后发布者暂停；代次／槽复用 | A5／仿真矩阵、重叠及 TPipe PA 已通过；首错误身份与冷日志已接线 |
| 5 | 完整 native 身份 trace 和原样 CSA resident | 七项 golden；2308 执行身份；来源严格门禁 | CSA 数值／2308 显式身份／来源通过；全协议事件未完成 |
| 6 | sync_start 不同位置领取和局部跨任务推进 | 两独立满核任务、部分交叠、超容量、早返回不重领 | 未实现 |
| 7 | 性能对比 | 同 kernel／输入／拓扑，无完整插桩运行；诊断运行分开 | 未测量 |

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
