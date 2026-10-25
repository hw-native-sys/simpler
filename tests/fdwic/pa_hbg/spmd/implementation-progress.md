# A5 HBG SPMD 实现过程记录

## 目标、依据和验收边界

开始日期：2026-10-25；起点：`6c367c69f`；工作分支：`feat/fully_distributed_within_core`。
按用户要求逐步实现、阶段性详细中文 commit，**不 push**。
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
| 1 | 有界存储、准入记录、block 控制、GM 组状态 | POD／布局／溢出／重跑初始化测试 | 已落地并通过布局与 N=1 回归；尚未运行协议 |
| 2 | 一次 ready 领取、共同准入、三入口统一顺序门禁 | 发布竞态、B 可先准入、非成员旁路、无重复入队 | 未实现 |
| 3 | 纯 C/V 多 block、完成聚合、分波容量 | N=1／容量两侧／512；身份不重不漏；独立任务可重叠 | 未实现 |
| 4 | GM Mix descriptor／ack／seal、各 lane 自备 payload | mask 3/5/6/7；最后发布者暂停；代次／槽复用 | 未实现 |
| 5 | 完整 native 身份 trace 和原样 CSA resident | 七项 golden；2308 执行身份；来源严格门禁 | 未实现 |
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
