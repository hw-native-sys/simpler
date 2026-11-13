# A5 HBG legacy 调度路径移除研究与实施边界

研究日期为 2026-11-12，研究基线为 `ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0`。
状态：第一阶段已通过 #3064 合入主线（`e37461139e7e58dc57f3b8914d278494c76bddd7`）；第二阶段 #3071 已合入（`3bcaf8d4f`）；当前整块删除余留 A5 旧调度栈和 kernel ABI 残留。第一阶段 A5DT 正常用例有通过结果，故障验收仍未通过。按用户要求不运行本地 CPU 验证。
首次发布 rebase 到 `6c691bef8a1752aff34889b366204da36fe41701`，保留上游 scalar trace 所有权修复；
该基线下 HBG/percounter 生产构建与完整提交静态检查通过。
第一阶段后续 review 更新另随 upstream/main rebase；其历史上板数据早于这些更新，
不能当作第二阶段二进制的设备验收。第二阶段验证单独记录。
本文保留删除依据、当前合同和后续边界。研究基线链接固定到该 commit，不能当作当前文件位置。
A2A3 HBG、A5 TMR 和显式可构建的 percounter 是活跃运行时，不属于历史兼容分支。

## 删除依据与当前输入合同

A5 HBG 当前只使用 resident ordinary/native Graph 执行。无效输入在 prepare 阶段失败；
无效设备启动描述进入失败启动与退核，不选择另一套成功执行路径。

### 普通任务形状

研究基线中，Host 的 `create_scheduler_state` 已检查 lane 数 1–3、block 数为正、
`block_num * active_subtasks <= INT16_MAX`，并拒绝 `sync_start && block_num <= 1`。
inline 服务的 synthetic shape 为 AIV0、一个 block。
这些条件与 resident 的支持谓词一致，shape fallback 没有剩余合法输入。
当前保留校验并直接拒绝不一致，不再发布 legacy mode。
证据为[基线 Host 校验][host-shape]与[resident 支持谓词][shape-predicate]；
当前实现见 [runtime_maker.cpp](../../../src/a5/runtime/host_build_graph/host/runtime_maker.cpp)。

### Graph 参数容量与连续区间

设 child 数为 `n`、各 child tensor 数量总和为 `T`、scalar 数量总和为 `S`。
每条 runtime Tensor 为 128 B；每个 child 的 scalar 区按 64 B 对齐，至多补齐 56 B，
predicate 至多占 64 B，因此：

```text
native_args <= 128*T + 8*S + 120*n
reserved_execution_tail >= 128*T + 8*S + 320*n
```

后一式尚未计入 GraphExecution、fanin CSR 与 states。当前 recorder 累加参数总数，
并按连续游标填每个 child 的 offset，所以它的产物满足此容量推导。
证据为[recording 总数][recording-layout]、[连续 offset][recording-fill]、
[参数布局][native-plan]和[共享 execution 布局][execution-layout]。

基线 planner 只检查每个 span 在池内，允许人为重叠；此时每个 child 分别展开，
`sum(child.tensor_count)` 可以大于 header 的池大小，不能套用上界。
当前 A5 planner 要求 child offset 等于连续游标，最终游标覆盖完整 tensor/scalar 池。
重叠、留洞和未使用尾部均拒绝。保留 `native_args <= execution_storage_bytes` 硬校验，
容量不足直接使 prepare 失败，不创建额外参数 backing 或另选执行器。
当前实现见 [native_graph_plan.cpp](../../../src/a5/runtime/host_build_graph/host/native_graph_plan.cpp)。

### Definition 的两个用途

每个唯一 Definition 在 resident scheduler backing 的只读区域存放一份，供设备绑定参数。
Host 的 `original_definition_section` 另保留原始录制来源，
`scheduler_protocol_freeze_source` 用原 outer payload 与 Definition 建立独立 oracle。
它不能改成从已经展开的 bindings 反推，否则会让被测 planner 自己证明自己。
A5 不再向每个 AICPU launch package 发布旧 Graph carrier。
证据见[基线来源快照][graph-source]和当前 [Host plan](../../../src/a5/runtime/host_build_graph/host/native_graph_plan.h)。

## 失败启动与退核

删除前，非 resident 入口借 `legacy_aicpu_execute` 清理已启动 AICore，
AICore 的另一入口也承担退出握手。仅删除调用并直接返回会丢掉实际清退责任。
基线见 [AICPU 入口][aicpu-entry]与 [AICore 入口][aicore-entry]。

当前 AICore 先报告本轮物理身份与 epoch，再验证 mode、worker 数量和 context base。
mode 不是 1/2、base 缺失/未对齐或 worker-context 地址计算溢出时，
不解引用 scheduler backing，只等待 EXIT 并参与 nonce ACK、teardown gate。
AICPU 先发现本轮报告，再通过 scheduler 步骤校验启动；拒绝时锁存 `INVALID_ARGS`，
各分区执行失败退休，最终参与者发布失败 terminal。
启动阶段必须等待其配置 owner 完成，leader 统一发布不可再改变的启动结果；
已经放行的分区在启动失败时补做 EXIT/ACK。重复物理窗口只保留一个退休 owner，
缺失或重复 report 以及缺失 ACK 均不构成完整退休证明。
非法 AICPU 线程数使用 callback 0 的平台 startup-abort，不进入无效参与数的屏障。
其他初始化错误保留其原始错误，不能由通用参数错误覆盖。

```text
AICore entry:
    publish_this_run_identity()
    validate_mode_and_context_address()
    if invalid:
        wait_EXIT_without_scheduler_backing()
        acknowledge_nonce()
        wait_teardown_release()
        return
    acquire_READY_context()
    bootstrap_and_run_resident()
    acknowledge_nonce_and_wait_release()

AICPU init:
    discover_this_run_reported_cores()
    validate_scheduler_steps()
    if failed:
        latch_first_error()
        retire_each_discovered_partition()
        publish_failed_terminal_with_retirement_outcome()
```

有效启动仍走既有 READY、bootstrap barrier、DMB execution gate。
正常退出和失败退出都保留各 core 的 EXIT/ACK、寄存器窗口 close/read-back/drain、
teardown release 与退休结果。错误 terminal、core 退休和存储可复用是三个不同判断。
不能以空 ready、无 task trace、内部计数相等或 AICPU 线程已返回代替退休证明。
当前实现见 [AICore entry](../../../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp)、
[AICPU executor](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp)与
[lifecycle](../../../src/a5/runtime/host_build_graph/aicpu/aicore_lifecycle.cpp)。

## 删除范围与仍在使用的结构

### 执行路径删除

以下均为 `src/a5/runtime/host_build_graph` 下第一阶段删除的旧实现，
不是整个同名 `scheduler/` 目录：

| 删除对象 | 原职责 |
| -------- | ------ |
| `aicpu/aicpu_legacy_executor.cpp` | AICPU 逐任务 executor、旧 terminal/reset 入口 |
| `aicore/aicore_legacy_executor.cpp` | MMIO task-token dispatch、early gate 与 kernel 循环 |
| `runtime/scheduler/scheduler_dispatch.cpp` | AICPU dispatch |
| `runtime/scheduler/scheduler_completion.cpp` | AICPU FIN/slab/completion |
| `runtime/scheduler/scheduler_cold_path.cpp` | 旧初始化、退休、诊断 |
| `runtime/scheduler/scheduler.cpp` | 旧 profiling 与统计实现 |
| `runtime/scheduler/scheduler_context.h` | 旧 SchedulerContext |

保留 resident ordinary/native、profiled/unprofiled 编译单元、`aicore_lifecycle.*`、
`args_dump_service.*`，以及 resident dispatch/completion/admission/async/control/ready 的 `.h`。
同名 `.cpp` 与 `.h` 的职责不同。inline 空节点仍直接完成，native DUMMY 仍使用 control 节点。

### shared arena 与镜像

A5 不再调用公共 `runtime_reserve_layout`、`runtime_init_data_from_layout` 或
`runtime_wire_arena_pointers`。设备池中的 runtime arena 从偏移 0 开始就是紧凑任务镜像：

```text
arena 请求字节数 = sm_image_bytes = pending_publication.bytes
pending_publication.device_target = gm_sm_ptr = arena base
prebuilt_arena_base = nullptr; prebuilt_runtime_offset = 0
```

因此不分配旧 SchedulerState、early/ready/sync/dummy/graph 队列、completion mailbox
或 SharedMemoryHandle 包装，也不分配或上传旧 RuntimeContext header。
Host 编排直接使用 OrchContext/OrchestratorState，不再创建空的 RuntimeContext。
资源等待重试保留任务镜像、native plan、leases、诊断来源与 callback 结果。
commit 按原 reservation pitch attach 已建好的 Host mirror，不重新 init，保留 task count 与执行 header。
共享 prepare owner 拆为不含 arena 的状态基类，以及供 A2A3/WSE 使用的带 arena 派生类。

compact/rebase 直接写 staging 起点；ordinary task storage、native boundary tensor/scalar
和协议导出的原始 Definition 来源仍使用各自真实基址。resident scheduler backing 独立申请、
先于任务镜像发布，READY 与退核流程没有改变。A5 同时删除旧 AICPU 队列 population 容量门槛，
仅使用 resident 自身的计划容量校验。`BindImageStage` 报告 `copied=0`；不再产生
`BindArenaBuild/BindRuntimeInit`，`BindStaticArena.arena` 就是紧凑镜像大小。

A2A3/WSE 的公共完整 arena 仍有活跃消费者，不能删除其布局和生命周期 API。
percounter 编译 A5 maker，使用同一个无前缀镜像。A2A3/WSE 显式编译 `common/host_build_graph/full_arena/`，旧 mailbox 也归到该目录；A5 构建不再包含这些实现。
历史布局证据为[共享 RuntimeContext][runtime-context]、[arena 生命周期][shared-arena]和[Host 调用][host-arena]。

shared `TaskPayload` 的 early-dispatch、staged masks 与 deferred 状态仍服务 A2A3。
resident ordinary/top-kernel 仍读取 320 B ChipTaskStorage、TaskPayload offset 和参数 delta；
`TASKPAYLOAD_*` 常量不能因旧分配删除而一起移除。后续若缩减任务镜像，必须同步明确
A5 投影、reader、packing 与布局断言；本次不改变这些结构。

### kernel context 与其他活跃 runtime

当前 `DispatchPayload` 仍为 512 B，`args[48]`/`args[49]` 指向 LocalContext/GlobalContext。
block_idx/block_num、physical AIV lane 与 resident async image/epoch/group 都有实际调用者。
当前删除 `src_payload` 和 LocalContext 中 40 B 的 legacy async/slab/token，LocalContext 仅保留 8 B block geometry。
args 起点仍为 64，GlobalContext 仍为 48 B、偏移 464，64 B 对齐后的 payload 仍为 512 B。
Host、resident、SDMA 与 percounter 同步迁移；percounter 在发布 L1 record 时由 task 的 shell/local/id 重建身份。
kernel 登记仅使用 resident group，无 group 时拒绝登记且不访问 GM，保留 epoch/布局校验、首错、封口和输出抓取限制。
旧 A5 SchedulerState、mailbox、wait-list 已删除；共享 Host 用例仍测两种架构，旧 AICPU 协议用例归 A2A3，独立 TMR mailbox 用例归 TMR。
不为历史 `.o` 增加兼容读取，所有当前生产者与消费者按同一合同构建。

percounter 直接编译 HBG 的 host/aicpu/runtime，并通过 scheduler 步骤替换默认实现。
它自己的 mode、context 与设备入口是活跃实现，不因 HBG 仅接受 resident 1/2 而失效。
A5 TMR 有自己的 intrinsic、dispatch payload 和 async slab；A2A3 HBG 编译 common HBG。
不能全仓替换同名头或删除这些运行时的当前功能。
证据为 [percounter build][percounter-build]、[payload 初始化][percounter-payload]、
[percounter 步骤][percounter-steps]、[TMR build][tmr-build]与[A2A3 build][a2a3-build]。

## 验收状态与后续

### 紧凑 arena 验证（第二阶段）

验证源码基于第一阶段合并点 `e37461139`。随后同步到 `60a2ba4cf` 的主线变更仅涉及 CI、
测试环境和文档，没有改变生产运行时源码。A5DT 为 Ascend950DT_9581、CANN 9.3.0，
PTO-ISA 固定为 `90a3796597a5fabc2ce5066e73f785c2e85822f7`。

- Host/AICPU/AIC/AIV 的 HBG、percounter 生产构建通过；构建与 callable 预编译均在设备分配前完成。
- 两套运行时的 A5 IR 扫描完成 16/16 TU，零发现；使用 CANN `llvm-link -S` 读取 IR。
- `task_20261113_174600_387839227358`：设备 0，12 pytest 通过、1 模拟器专用项跳过，
  覆盖 ordinary、native Graph、AIC/AIV/MIX、predicate、形状选择和参数抓取。
- async/reuse 共 20 个不同 nodeid 通过：`task_20261113_174647_392966120227` 汇总 13 项后
  达到本次整会话 600 s 上限（exit 124）；`task_20261113_175859_49518430845` 补跑其余 7 项通过，
  exit 0。覆盖两种完成顺序、全部 lane mask、sync-start、双 Graph instance、跨 run epoch/映射复用。
  中断批次不记为整批通过；没有修改测试断言，前一批中未汇总的在途用例也计入补跑。
- 新增 empty/ordinary/Graph 的 arena 大小、镜像地址/长度和 prebuilt 字段断言，同时保留
  A2A3 完整 arena 的非空 scheduler/mailbox 契约。按用户要求未运行本地 CPU/模拟器测试。
- 保留 populated mirror header 的修复后，两套生产构建再次通过；
  `task_20261113_193544_236908424818` 重跑正常路径 12 项通过、1 项跳过；
  `task_20261113_193653_244280826261` 单独重跑两种跨 run epoch 复用，2 项通过。
  首个混合层级批次筛掉这 2 项，故单独提交，不将 deselected 项记为通过。
- 完整提交静态检查（含 clang-tidy）与严格 MkDocs 构建通过；没有性能比较结论。

### 执行路径移除验证（第一阶段历史记录）

第一阶段的测试修改与实现一并维护：

- 删除只验证旧 terminal/mode/executor 的 A5 case 和 CMake source 引用。
- nonce、report epoch、失败启动和退休覆盖转到当前 resident/lifecycle seam，保留 A2A3 用例。
- `spmd_shape_selection` 保留普通和 sync-start 多 block 的输出与新 mode 发布检查。
- Definition 测试覆盖连续区间、重叠/留洞/未覆盖尾部的拒绝和容量校验。

候选尚不能视为已通过完整设备或性能验收。用户明确要求不运行 CPU 验证；
测试代码供 CI 执行，本次报告必须分别记录编译、静态检查、模拟器、设备和性能的实际范围。

本次实际验证范围：

- `build_runtimes.py --platforms a5 --runtime host_build_graph --runtime percounter` 成功，覆盖两个运行时的 Host、AICPU、AIC/AIV 生产编译；末次 PMU 所有权与错误保留修改再由两者的 AICPU CMake 目标增量编译、链接通过。
- 两份 AICPU 链接产物中没有旧 executor、`LegacyAicpuExecutor` 或 `SchedulerContext` 执行符号；已移除 mode 和 executor 符号的源码引用检查通过。
- `git diff --check`、修改 Python 文件的静态语法检查和本地 Markdown 链接检查通过。新增或调整的 CPU 测试目标未编译、未运行，没有 CPU 或模拟器功能验证结果。
- 本地环境未能确认 A5 芯片；随后在独立 A5DT 工作目录上传候选源码快照并校验 SHA256，完成相同的 HBG/percounter 生产构建。源码基线为本文开头的 commit，快照 SHA256 为 `a5fc528a130e86b0b15169ae57280bcffd13d3e72330a1b3a7f21ec5bc18bdb1`，PTO-ISA 固定为 `90a3796597a5fabc2ce5066e73f785c2e85822f7`。
- 编译、依赖安装与工作负载预编译均在占卡前完成；所有板端用例经 `task-submit` 独占分配，在同一作业内预检为 `a5 | Ascend950DT_9581 | Ascend950`，CANN 为 9.3.0。
- A5 IR 扫描完成，12/12 个设备 TU、0 findings。系统 `llvm-dis` 无法读取 CANN 的 `block_local` 扩展，故使用 CANN 自带 `llvm-link -S` 读取单个模块；保留原始 `hiipu64-hisilicon-cce` target 与 data layout，没有更改扫描规则或生产编译选项。

2026-11-13 的板端结果按独立 pytest 子进程与主进程合计；重复复现不计入唯一通过数：

| 分组 | 实际结果 | task-submit 任务 |
| ---- | -------- | ---------------- |
| empty lifecycle、single root、shape selection | 5 passed | `task_20261113_144331_13372657398` |
| ordinary/native Graph 与 topology | 4 passed、1 skipped | `task_20261113_144500_139591226696` |
| 单 block MIX 与 sync-start AIV | 4 passed | `task_20261113_144504_139801215749` |
| ordinary/native/MIX args dump | 4 passed | `task_20261113_144725_1502297912` |
| async dependency completion、run reuse | 19 passed | `task_20261113_144721_149898324324` |
| predicates、单 block sync-start、超容量拒绝 | 18 passed、1 failed | `task_20261113_145005_162321731680` |

跳过项为 `test_repeated_aicore_so_load`，其硬件 skip 条件明确要求仅在 `a5sim` 运行。
失败项为 `test_single_block_sync_request_preserves_execution[1]`，纯 AIC、单 block、首次 run 返回 `507018`；
在设备 2 单独复现仍失败（`task_20261113_145733_197734932124`）。
设备日志先出现 AICore 执行异常，之后才有 AICPU watchdog 与退休失败，不能从 Host 错误码直接称作调度死锁。
随后在设备 0 做精确对照：未修改基线通过（`task_20261113_151136_257206913971`），
候选也通过（`task_20261113_151439_28252818265`）。因此正常用例共有 55 个唯一 case 至少通过一次，
但设备 2 的两次失败没有同设备基线证据，既不能排除回归，也不能归为硬件问题。
该设备后来被其他任务占用，排队中的自有对照任务已取消；本次没有强占、重置设备或以重跑通过覆盖原失败。

bounded/native 故障验收的五个唯一 case 均失败。四个 AIV/MIX 用例的候选任务为
`task_20261113_150513_23216247292`，纯 AIC 在上述候选设备 0 对照任务中重跑。
未修改基线 `ca27b263d` 使用独立 venv、相同 torch 2.5.1、numpy 2.4.6、CANN 与 PTO-ISA，
在同一设备 0 的任务 `task_20261113_150946_2472589528` 复现全部五项失败：

| 故障用例后缀 | 候选 fault run 返回时间 | 基线返回时间 | 共同失败断言 |
| ------------ | ----------------------- | ------------ | ------------ |
| bounded `[1-1-0]`：单 block AIC | 30.50 s | 31.37 s | `elapsed < 20` |
| bounded `[2-4-2]`：双 block AIV | 30.53 s | 40.37 s | `elapsed < 20` |
| bounded `[2-7-1]`：双 block MIX | 31.59 s | 29.97 s | `elapsed < 20` |
| native Graph `[2-1]`：AIV | 30.80 s | 30.44 s | `elapsed < 20` |
| native Graph `[7-2]`：MIX | 30.18 s | 30.79 s | `elapsed < 20` |

这五项最新对照中的健康预跑通过，注入故障后返回预期 `507018`，但在验证后续 containment/恢复前
就因返回时限失败，不能宣称完整故障合同已验收。AIV 样本设备日志中 AICPU 约 2 s 锁存 watchdog、
约 2.5 s 返回 `-100`；Host 约 30.5 s 后才得到最终错误，尚未定位这段差值的根因。
没有放宽断言或增加 timeout 以换取通过。最初误纳入手工矩阵的任务
`task_20261113_150326_22473446534` 在纯 AIC 健康预跑失败后取消，不能计为已完成的矩阵；
其健康异常后来在设备 0 重跑没有复现，但仍计入异常记录。

结论：普通执行、Graph、MIX、predicate、sync-start 形状/容量、args dump、async 与 reuse
已有上述定向板端证据；这不是全量 A5 CI、启动畸形输入的板端故障注入或性能回归验收。
故障返回时限是已在基线复现的问题，另有设备 2 的健康执行异常未归因。
本次没有修改生产源码来绕过失败，没有运行 CPU/模拟器测试。
第一阶段 #3064 随后由维护者合并；这些限制仍然存在，不能据此宣告第二阶段设备验收全绿。

现有 A5 HBG ordinary/native、AIC/AIV/MIX、sync_start、predicate、inline、async、
args dump、profiling 与重复 run/reuse 功能都需要保持，不能用删除场景换取通过。

本轮验证基于 #3071 合并点 `3bcaf8d4f` 加当前改动，A5DT/CANN/PTO-ISA 与第二阶段一致。
HBG/percounter、独立 A5 TMR 生产构建通过；A2A3 HBG/WSE 仅生产编译通过；静态 hooks、严格 MkDocs 通过，A5 IR 16/16 TU 零发现，四个 HBG/percounter Host/AICPU 库无旧 full-arena 调度符号。
五个 task-submit 任务共 43 项通过、1 模拟器专用项跳过：`task_20261113_204352_22613465517`（设备 2，普通/Graph/shape/trace 31 passed/1 skipped）；
`task_20261113_204609_240126732380`（设备 3，async 容量、最后条件、epoch 复用 7 passed）；`task_20261113_204614_240687818604`（设备 4/5，peer SDMA 3 passed）；`task_20261113_204524_23541065717`（设备 6，percounter L1 轮转与重复 run 1 passed）。
`task_20261113_211054_377107531077`（设备 0，defer-wait 数据可见性 1 passed）；六个 DSV4.1 wait adapter 的 A5 编译也通过。
编译预热在分配设备前完成，plain probe 的入口仍会调用编译缓存；未运行本地 CPU/模拟器功能验证，不代表全量 CI、故障验收或性能提升。
空图 fast path 与性能优化另行依据证据决定。源码、生成代码或保留空间变小不自动意味着更快；
应在相同设备和负载比较真实 bind/arena bytes、cold build 与 Device latency。
当前运行合同另见 [runtime design](../../../src/a5/runtime/host_build_graph/docs/RUNTIME_LOGIC.md)、
[Graph design](../design/a5-hbg-graph-task.md)和[resident 走读](../a5-hbg-aicore-walkthrough/README.md)。

[host-shape]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1417
[shape-predicate]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_graph.h#L71
[recording-layout]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/common/host_build_graph/host/graph_recording.cpp#L329
[recording-fill]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/common/host_build_graph/host/graph_recording.cpp#L499
[native-plan]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/host/native_graph_plan.cpp#L85
[execution-layout]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/common/host_build_graph/graph_execution.h#L190
[graph-source]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L1865
[aicpu-entry]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp#L858
[aicore-entry]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp#L64
[runtime-context]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/common/host_build_graph/runtime_core.h#L120
[shared-arena]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/common/host_build_graph/shared/runtime_init.cpp#L210
[host-arena]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/host_build_graph/host/runtime_maker.cpp#L2826
[percounter-build]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/percounter/build_config.py#L63
[percounter-payload]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/percounter/host/state.h#L258
[percounter-steps]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/percounter/aicpu/steps.cpp#L51
[tmr-build]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a5/runtime/tensormap_and_ringbuffer/build_config.py#L36
[a2a3-build]: https://github.com/hw-native-sys/simpler/blob/ca27b263d5b6576b9f5dbd9050fa7b7ab6c53ff0/src/a2a3/runtime/host_build_graph/build_config.py#L47
