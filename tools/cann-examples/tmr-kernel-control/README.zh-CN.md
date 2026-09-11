# TMR kernel control 真机协议探针

本探针使用生产 `KernelCoreGroup`、两架构 `aicore_kernel_mode.o`、K2 的
`KernelExecutionState` / platform ops、生产 `enqueue_kernel_launch_sequence`，
以及 `LoadAicpuOp::LaunchBuiltInOp` 的真实 RTS CPU 参数复制通路。
不使用 Torch，不修改 CANN，不以 Host 模拟通过代替 cache/MMIO 验证。

## 1. 验证范围和资源归属

Host 在 prepare 一次性申请真实 HBM：设备 Runtime 描述、空闲 payload、KernelArgs、
context descriptor、AICore envelope、control/report 和结果区；寄存器地址通过
生产 `init_aicore_register_addresses` 获得，A2/A3 FFTS 地址来自 `rtGetC2cCtrlAddr`。
只启动一个 MIX block，即 1 个 AIC 和 2 个 AIV。

这是**协议专用测试 owner**，不是 K3/provider 的实现：其 descriptor 不提供可执行的
heap/SM/arena，不可送入公开 TMR invocation 入口。测试 SO 只调用生产 report 收集、
开窗、收拢和结果发布主体，不重写调度算法。其一次 CPU 入口只运行 1 个线程，
因此原有 30 次 eager / 30 次 replay 不替代 K7 的 N/M 线程四相 gate 验收；
下述新增 gate 子探针单独给出这部分证据，不追溯扩大原有结果的覆盖范围。

每轮通过 caller 上的两项精确 memset 清 control 和 report，再用 Start event 放行
两条 hidden stream；CPU/Core Done 均回到 caller，只有 caller 完成后才读结果或复用。
对照实现中的生产 sequence，包括 AICore 先提交、CPU 入队失败后 Host cancel 的补偿。

| Case | Acceptance |
| ---- | ---------- |
| 正常开窗、空闲收拢 | 所有核确实写过 `COND=IDLE`，再完成 EXITED、close、release |
| CPU 准入拒绝 | 无开窗，所有核通过 CANCEL 返回；runtime error 与 cleanup error 分开 |
| Host CPU enqueue 拒绝 | 不提交 CPU，生产 sequence 只写 4B `host_cancel`；Core Done 可正常 join |
| 旧 DMB 非零 | 每轮启动前由 AICPU 对真实有效寄存器写入 EXIT sentinel 并 readback；新 wrapper 不把旧值当作启动依据 |
| 漏清 control | report 正常清，保留上一轮 completion；收集阶段拒绝并取消本轮，下一正常轮仍成功 |
| clear 越界 | 前后各 64B canary 不变；整 Runtime 和包含 guard 的 clear plan 在 Host 被拒绝 |
| 两图 A/B/A | 捕获正常/拒绝两个图，交替 replay 30 轮；两条 hidden stream 每轮都执行 |
| 常驻状态与复用 | 30 次 eager + 30 次 replay 之间不 reset；Runtime、KernelArgs、descriptor 字节不变；测试 allocator 数量和 bytes 不增长 |

**不是本探针声称的证据：**普通 child kernel 执行、early-dispatch doorbell 内部取消、
完整 TMR executor、公开 launch、K3/K10 provider、任意跨 stream 重叠 replay。
这些必须由对应测试补充。这里的正常路径只进入生产空闲 dispatch loop。
没有删除 report clear 后冒险在共享卡上运行；漏清 control 是可恢复的污染负例。

### 1.1 N/M 四相 gate 子探针

`device/gate_probe.cpp` 的最小 executor **直接实例化生产
`execute_kernel_round_impl`**，成员使用真实 `KernelRoundGate` 和
`KernelCoreGroup`。CoreGroup 外层只统计 `finish` / `publish_status` 调用次数，
不替代 cache、MMIO、等待或取消逻辑；探针没有另一套 gate，也没有在返回后补屏障。

在 A2/A3，prepare 复用正式 `probe_aicpu_topology` / `compute_allowed_cpus` 取得
两个实际允许的 CPU ID；`M=2`，`N` 取已有架构 launch 常量，即 `6`。
Host 使用 `LaunchBuiltInOp(..., N, control_probe_gate)` 的真实 CANN 多线程任务，
每个 worker 先走正式 `platform_aicpu_prepare_kernel_thread`，再用实际 CPU ID 入 gate。
不把 Host 线程模型当作 CANN 多线程执行，也不把未接入拓扑 prepare 的 A5 标为通过。

| Case | Acceptance |
| ---- | ---------- |
| 正常 N/M 执行 | 6 个 native worker 全部进入生产 coordinator；恰好 2 个 init/run，两个 execution role 不重复，另 4 个均不执行 |
| init verdict 拒绝 | execution role 1 在完成自己负责的开窗后返回 `-47`；两个 init 均结束，全局 verdict 阻止所有 run；3 个真实 Core 均被收拢 |
| admission 拒绝 | 不 prepare/init/run，不开窗；生产 coordinator 仍使 6 个 worker 统一读到 `-31`，并正常取消 Core |
| exactly-once | 每轮真实 CoreGroup finish、结果发布、finalize 以及最后读者 clear 各恰好一次；control/report 与这份结果一致 |
| 所有读者离开 | 每个 native worker 在生产 coordinator 返回后才写自己独占的 64B 结果槽；Host 同时核对全部 6 槽的 epoch、final status 和返回值，无测试侧额外汇合修补流程 |
| 失败后复用 | 同进程 `Open → InitReject → Open → AdmissionReject → Open` 循环 5 次；随后捕获成功/初始化拒绝两个图，执行 A/B/A 共 15 次 replay；gate 不 reset |
| 常驻与清理 | 沿用原始两区域精确 clear plan、双 hidden stream 和 canary；gate 结果区在 prepare 分配，每轮不分配设备缓冲 |

gate 子探针的最小 executor 不构造假的 TMR callable / SM / arena；它不执行具体算子，
不证明完整 TMR executor 的功能，也不证明跨 stream 重叠提交可用。
context descriptor 仍属于原来的单 block Core 协议 fixture；N/M 是测试 owner 的独立
prepare 配置，不将这个 descriptor 伪装成生产 K3 资源视图。
每轮通过相同 hidden CPU stream 和完整双分支 tail 串行；上一轮 native task 的全部
worker 返回后才读回或复用，任何意外 cleanup 失败都停止后续执行。

A5 仍保留原单 CPU 协议构建/运行能力；新 gate 子探针会明确打印 `SKIP`。
其设备侧代码可编译不等于 A5 CANN N/M 已验证；后续必须接入正式设备 occupancy
查询和 A5 launch plan，不能直接拿 A2/A3 的 CPU ID 或固定 launch 数代替。

### 1.2 独立终端 native-error 负例

A2/A3 的正常/可控拒绝/replay 全部完成、graph 全部销毁并打印上述两组结果后，
最后单独运行一次 `TerminalInitReject`。它仍在生产 coordinator 中使 execution
role 1 的 init 返回原始状态 `-47`；`cleanup_status == 0` 时得到内部分类
`KernelDispatchStatus::ExecutionFailed`（6），再复用生产 adapter 返回 CANN
`KERNEL_STATUS_INNER_ERROR`（2）。原始执行状态、cleanup、内部分类和 native 返回
分别记录；onboard 构建用 static_assert 核对 SDK 正式常量，不将内部枚举直接返回 CANN。

这是受控实测定位的修复：同一个 direct CPU entry/SO、无 gate/Core，仅改变返回值，
1 的 replay 得到 507011，6 得到 507005（EOS），2 得到 507011；本轮 HBM 记录均一致。
SDK 明确 runtime 解释固定的 0–3，2 的日志为 `execute kernel inner error`。
6→EOS 是该 CANN 9.0.0 / A2/A3 通路的实测，不伪称 SDK 的 EOS 常量等于 6。

这个 case **没有增加 N-worker 屏障或 Host 放行门**。每个 CPU 仅在自己的
coordinator 返回后写独占结果槽并 flush，再立即返回 native 状态。终端结果槽以 epoch
最后提交，避免将被中断的半写帧误当作本轮完整结果；这不是线程间同步。
生产 CoreGroup 的 ACK / 关窗 / release 已在最终状态可读之前完成，但 CANN 收到
首个非零返回后，仍可能取消尚未退场的其他 CPU。因此不能沿用前 40 轮的
`final_read_depart=all` 来声称终端错误轮也全员完成。

默认 Host 顺序执行有界 hidden CPU / Core / caller 同步，并打印每条分支的原始返回码。
默认 native 错误传播验收明确要求 hidden CPU 同步返回
`ACL_ERROR_RT_AICPU_EXCEPTION`（507018）；任何成功、超时、未分类错误或仅有
sticky driver error 都不自动算作本项通过，必须进一步查该任务的 CANN 日志。
由于异步错误可能在 Host 继续提交 Done/join 时就暴露，额外打印 native enqueue
是否成功、失败阶段和 tail 是否已提交，不作 Host cancel 或重试修补。

随后尽力读回 control、report、各 CPU 结果槽和最后读者 summary。读取失败明确输出
`readable=0` / `unverified`，不依赖错误后的 D2H 必须可用；结果槽只统计本轮 epoch，
未读到某槽不能单独证明 CANN 取消了该 CPU。`coordinator_returns=-1/N` 表示整个
结果区不可读。即使读回 N 个结果槽，也不声称 CANN 已收齐 N 个 native 返回码。
如果可读证据与期望状态冲突，进程返回失败。最终通过还要求本轮 control 的
`raw=-47 / cleanup=0`、全部 Core 的退出/release，以及至少一个已提交 CPU 结果槽的
`dispatch=6 / native=2` 均可核对；不要求全部 N 个结果槽或最后读者 summary。取证不足时输出
`UNVERIFIED` 并以非零退出，不能仅凭 507018 将其他设备错误误判为预期拒绝。
取证使用 prepare 时申请的 pinned Host buffer，复用现有 Core stream 提交
`aclrtMemcpyAsync`，每次配套 1 秒 `aclrtSynchronizeStreamWithTimeout`；不增加新的
stream，也不使用无限等待。buffer 在可能仍有 DMA 使用时一直保留到进程退出。

终端测试无论通过还是失败，都直接退出进程；保留 buffer、SO、stream 和所有 owner
直到进程退出，不调用普通 owner 回收、`aclrtResetDevice`，不再启动或复用 context。
仍须独占设备并使用 `task-submit` 进程级时限；本探针不会故意让 Core 丢失 ACK。

### 1.3 caller-first 与终端 replay 变体

可在命令末尾传入以下一个选项；两者互斥，不带选项时保留上一节的 eager CPU-first
顺序及判据。三个模式必须用三个独立进程运行，不能在同一 context 上连续运行终端
负例。显式选择变体而该架构未实现 gate topology 时直接拒绝，不以 `SKIP` 当作通过。

| CLI | 最后一个 case | 同步顺序与验收 |
| --- | ------------- | -------------- |
| 无 | eager `TerminalInitReject` | hidden CPU → Core → caller；CPU 必须为 507018，caller 仅诊断 |
| `--terminal-eager-caller-first` | 同一 eager `TerminalInitReject` | caller → hidden CPU → Core；分别判断 native 错误和 caller 错误传播 |
| `--terminal-replay` | 单独 capture 同一 `TerminalInitReject`，然后 replay 一次 | caller → 原 capture hidden CPU → Core；要求实际 replay caller 为 507011（MODEL_EXECUTE），不要求原 capture CPU stream 仍承载图的执行 |

所有模式都先完成既有 100 轮正常/可控拒绝测试并销毁其图。终端 replay **没有执行
会污染 context 的 eager warmup**：一次 `aclmdlRICaptureBegin`，经原生产 sequence
提交完整双分支，`aclmdlRICaptureEnd`，再一次 `aclmdlRIExecuteAsync`。
capture begin、capture 内 enqueue、capture end 及 replay enqueue 的原始返回码分别
打印。capture 任一步失败或未包含完整双分支均终止，不尝试修补或重新捕获；replay
enqueue 非零可继续有界取证，但不能算作本模式验收通过。终端图与引用的全部资源
保留到 `_Exit`，不调用 graph destroy、普通 owner 清理、reset 或下一次 replay。

两个新增变体仍要求上一节的当前 epoch control、全部 Core retirement，以及至少
一个完整 CPU 结果槽；SDK 错误码本身不足以判定是预期
`raw=-47 / cleanup=0 / dispatch=6 / native=2`。
eager caller-first 若 caller 为 0、CPU 为 507018 且严格证据齐全，会分别打印
`native_error=PASS`、`caller_error=NOT_PROPAGATED` 和
`overall=FAIL overall_reason=caller_error_contract`，以非零退出。
这个结果是**公开组合入口的 caller 错误传播前提未成立**，不是 K7 编排或 Core 清理
失败。caller 为超时或其他错误也不算 caller 合同通过。
replay 仅以实际 replay caller 的 507011 作为 native 错误传播条件；原 capture
hidden 流的同步返回码只是诊断，不能用其任意非零替代 caller 判据。
此 graph 包装语义由同一 direct entry 的 eager/replay 对照证实；旧版要求 replay
也必须为 507018 没有对应依据。旧失败记录保留，EOS、timeout、其他未分类错误仍拒绝。
取证不足仍明确 `UNVERIFIED` 且非零退出，不通过额外屏障强迫所有 CPU 写完结果。

## 2. 构建

先在 checkout 隔离环境构建生产 onboard 产物，必须得到所选架构的
`aicore_kernel_mode.o`，不能用 `aicore_kernel.o` 替代。
以下命令从仓库根目录执行，构建输出可另指向 reports 下的目录。

```bash
export CCACHE_DIR="$PWD/.ccache"
export CMAKE_BUILD_PARALLEL_LEVEL=4
ccache -s

cmake -S tools/cann-examples/tmr-kernel-control/device -B build/control-probe/device \
  -DARCH=a2a3 \
  -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build/control-probe/device --parallel 4

cmake -S tools/cann-examples/tmr-kernel-control/host -B build/control-probe/host \
  -DARCH=a2a3 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build/control-probe/host --parallel 4
```

A5 使用 `-DARCH=a5`，并传入 A5 生产产物；不能在 A2/A3 上运行 A5 构件。
`ASCEND_HOME_PATH` 必须指向与产品构建一致的 SDK。

## 3. 真机运行与证据

探针主动对整个设备的有效 DMB 寄存器写入 sentinel，**必须独占该设备且无并行任务**。
先做架构预检，再通过 `task-submit` 单卡运行。每次 stream 同步上限 10 秒，
进程级上限交给 `task-submit`。出现超时/cleanup 失败立即退出，不复用现场，也不在
失败时逐块释放可能仍被设备使用的资源。A2/A3 最后的 native-error 负例直接保留
资源到进程退出，不 reset；A5 未执行该负例的正常路径，在全部图销毁和资源释放后
的进程 teardown 执行一次 `aclrtResetDevice`。

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
mkdir -p build/control-probe/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/build/control-probe/ascend"
task-submit --device auto --device-num 1 --max-time 600 \
  'build/control-probe/host/tmr_control_probe "$TASK_DEVICE" \
    build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so \
    build/control-probe/device/libtmr_control_probe_device.so \
    build/lib/a2a3/onboard/tensormap_and_ringbuffer/aicore_kernel_mode.o'
```

分别以新进程运行额外变体；日志目录也应逐次隔离。以下为终端 replay 命令，
caller-first 对照仅将最后一个选项替换为 `--terminal-eager-caller-first`：

```bash
task-submit --device auto --device-num 1 --max-time 600 \
  'build/control-probe/host/tmr_control_probe "$TASK_DEVICE" \
    build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so \
    build/control-probe/device/libtmr_control_probe_device.so \
    build/lib/a2a3/onboard/tensormap_and_ringbuffer/aicore_kernel_mode.o \
    --terminal-replay'
```

提交后使用 `task-submit --status <task_id>` 和 `task-submit --log <task_id>`
读取状态；不使用会因排队等待超时取消任务的 `--run` / `--wait`。
运行前检查四个路径是否存在，特别是生产 binary 的实际构建路径。
原有单 CPU 协议成功行必须包含：

```text
control_probe PASS eager=30 replay=30 hidden_streams=2 between_round_reset=0
```

A2/A3 完整运行还必须在此前出现独立的 gate 验收行：

```text
control_probe gate PASS N=6 M=2 eager=25 replay=15 final_read_depart=all finish_publish=once
```

原有单 CPU 成功行单独出现不算 gate 通过；`gate SKIP` 只表示该平台未进行此项验收。

A2/A3 随后进入终端负例。典型输出形状如下（设备错误后的取证可能不可用）：

```text
control_probe terminal BEGIN epoch=41 raw_status=-47 dispatch_status=6 native_status=2 N=6 extra_barrier=0 reuse=0 mode=eager-cpu-first
control_probe terminal sync branch=aicpu rc=507018 name=ACL_ERROR_RT_AICPU_EXCEPTION
control_probe terminal sync branch=aicore rc=... name=...
control_probe terminal sync branch=caller rc=... name=...
control_probe terminal observations coordinator_returns=.../6 control=... core_retirement=... all_cpu_native_returns=UNVERIFIED
control_probe terminal native_error=PASS cpu_rc=507018 readable_regions=.../4 readable_evidence_consistent=1 expected_error_evidence=1 resources=retained_until_process_exit reset=0 reuse=0
control_probe terminal caller_error=... caller_rc=... core_rc=... mode=eager-cpu-first overall=PASS overall_reason=none
```

`native_error` 行确认预期 native 错误传播及上述最小取证；caller 的错误传播和全部 CPU
native 返回不由该行证明。默认模式退出码为 0 要求 native 错误符合预期、最小证据齐全，
并且任何可读证据都没有矛盾；新增变体还要求 `caller_error=PASS`，见上一节。
设备日志还会写入带 `intentional native error`
的每线程原始状态与 native 分类，便于对照 CANN 的首个异常来源。

保存命令、源码 revision/dirty patch、产品 binary hash、CANN 版本、芯片架构、
完整 Host 输出和本次 `ASCEND_PROCESS_LOG_PATH`。仅编译成功不算真机验收通过；
本 README 不预先记录任何平台已通过。
