# Kernel 模式集成测试说明

本文说明 kernel 模式集成线上的端到端测试是怎么做的：调用方侧如何准备设备和
stream，simpler 的四个入口各做了什么，一次 launch 在三条 stream 上怎样排布，
设备侧如何执行，以及图模式目前验证到哪一步。

整合了哪些 PR、每个 PR 冻结在哪个 head、跑过哪些测试，见
[集成验证记录](../kernel-integration-validation.md)。逐条冲突裁决见仓库根目录的
`INTEGRATION-LOG.md`。

## 1. 被测的是什么

kernel 模式下，simpler 是一个被调用的库：它借用调用方已经设好的设备和调用方
自己的 stream，提交一个有界的异步算子，不自己初始化 ACL，不 reset 设备，launch
路径上不做任何同步。这正是它区别于 program 模式的地方。

端到端数值用例是
`tests/ut/py/test_kernel_mode_c_api.py` 里的
`test_kernel_eager_launch_executes_fresh_tensor_and_scalar_snapshots`。它用 ctypes
直接调用 host runtime 动态库，自己扮演调用方，不经过 PyTorch，也不经过 simpler
的 Python Worker。走 Worker 的同一个算子见 §4.1。

被执行的算子是一个 AIV 向量加标量，`y[i] = x[i] + scalar`：

| 组成 | 源文件 | 编译产物 |
| ---- | ------ | -------- |
| 编排 | `tests/ut/py/kernel_eager_orchestration.cpp` | AICPU 上运行的 .so，提交一个 AIV 任务 |
| 设备算子 | `examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp` | AIV 核上的代码段 |

两者在测试开头现场编译，打包成一个 `ChipCallable`，签名依次是输入、输出、标量。
单个 tensor 是 16,384 个 FLOAT32，64 KB。

## 2. 怎么运行

该用例带 `requires_hardware`、`platforms(["a2a3"])`、`runtime("tensormap_and_ringbuffer")`
和 `device_count(1)` 标记，只能在 a2a3 真机上运行。仿真平台的 kernel init 返回
`PTO_RUNTIME_ERR_UNSUPPORTED`。

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device auto --device-num 1 \
    --run "python -m pytest tests/ut/py/test_kernel_mode_c_api.py --platform a2a3 --device \$TASK_DEVICE"
```

pytest 用例本身只是一个壳。它读取分配到的卡号，再启动一个子进程运行同一个文件，
参数为 `a2a3 tensormap_and_ringbuffer <卡号> eager_values`。放进子进程，是为了让
`aclInit` / `aclFinalize` 这类进程级状态不影响 pytest 进程。

## 3. 调用方侧：设备、stream 与显存

以下调用全部由测试完成，simpler 不代劳。ACL 符号从已加载的 `libhost_runtime.so`
句柄上解析，实际命中它依赖的 CANN 库。

```text
aclInit(NULL)
aclrtSetDevice(卡号)                 # 把这张卡设为当前线程的 current device
aclrtCreateStream(&caller_stream)    # 调用方自己的 stream
create_device_context()              # 只在 host 上构造对象，不接触设备
```

每一轮，测试用 `aclrtMalloc` 分配输入、输出两块显存，用 `aclrtMemcpy` 把输入拷到
设备，并把输出预填为 `-999`。

收尾顺序同样由测试控制：`finalize_device`、`destroy_device_context`、`aclrtFree`、
`aclrtDestroyStream`、`aclrtResetDevice`、`aclFinalize`。simpler 的 finalize 只释放
自己申请的资源。

同文件的生命周期用例用 `LD_PRELOAD` 拦截 `aclInit`、`aclrtSetDevice`、
`aclrtResetDevice`、`aclFinalize`、`rtDeviceReset` 等函数并计数，断言 kernel 模式下
simpler 没有调用过它们。

## 4. simpler 的四个入口

| 入口 | 做什么 |
| ---- | ------ |
| `simpler_kernel_mode_init` | 用 `aclrtGetDevice` 确认当前卡与传入卡号一致，只核对不设置；用 `rtStreamCreate` 创建两条私有 stream（AICPU、AICore）和 5 个 event；加载 AICPU 执行体。容量配置此时固定 |
| `simpler_kernel_mode_prepare_callable` | 上传 callable 镜像并铸造 callable_id（出参返回），把编排 .so 注册到设备；首次调用时提交 pooled arena，上传常驻 Runtime 与 KernelArgs |
| `simpler_kernel_mode_launch` | 校验上下文与 callable 驻留，把本次参数连同该 callable 的设备镜像地址与长度编码进参数包，在三条 stream 上排布一串异步操作后返回 |
| `finalize_device` | 释放上下文拥有的资源；测试断言 committed memory 归零 |

init 期间的执行体加载和 prepare 期间的 callable 注册，会同步上下文自己的 AICPU
stream。launch 路径不同步任何 stream。

## 4.1 Python 侧的 Worker 入口

上面四个是 C 入口。Python 侧对应的公开对象是 L2 `Worker`，构造时用
`execution_mode` 选定分派面，之后只读：

```python
worker = Worker(level=2, execution_mode="kernel", device_id=d,
                platform="a2a3", runtime="tensormap_and_ringbuffer")
worker.init(config=cfg)                       # -> simpler_kernel_mode_init
cid = worker.kernel_prepare_callable(chip)    # -> ..._prepare_callable，返回铸好的 id
worker.kernel_launch(cid, args, stream)       # -> ..._launch，stream 每次显式传
worker.close()                                # -> finalize_device
```

`config` 是 context 固定配置，kernel 模式必填、program 模式拒收；program 模式的
`prewarm_config` 反之。init 与 prepare 都不收 stream，只有 launch 收，且不保存。
两个分派面互斥：kernel 模式下 `register` / `submit` / `run` 被拒，program 模式下两个
`kernel_*` 入口被拒，报错里点名构造时固定的模式。

`kernel_prepare_callable` 是纯注册：同一个 callable 注册两次得到两个不同且都有效的
id，没有去重也没有 lookup，容量按注册次数计。id 只在本 context 内有效、成功过的不
复用，`close()` 之后整体失效——这三条合起来取代了 id 上的 generation 字段。

Worker 层的端到端用例是 `tests/ut/py/test_worker/test_kernel_mode_entry.py` 的
`test_worker_kernel_mode_eager_end_to_end`：与第 2 节的 C API 用例同一个向量加标量
算子，同样由测试自己持有设备、stream 和显存，区别只是经过 Worker 而不是 ctypes。

prepare 之后，测试自己同步一次 caller stream 再读 committed memory。prepare 的设备侧
注册错误由它自己的返回值报告，不依赖这次同步。

## 5. 一次 launch 的时序

launch 在 `src/common/platform/onboard/host/kernel_launch_owner.cpp` 中实现，把参数包交给
binder，在三条 stream 上排布如下操作：

```text
caller stream              AICPU 私有 stream              AICore 私有 stream
record(Start)
                           wait(Start)
                           清零握手区与 gate
                           record(AicoreStart)
                                                          wait(AicoreStart)
                                                          rtKernelLaunchWithHandleV2
                                                          record(AicoreDone)
                           rtsLaunchCpuKernel(参数包)
                           wait(AicoreDone)
                           record(AicpuDone)
wait(AicpuDone)
record(SerialTail)
```

链式串联 caller ⇄ AICPU ⇄ AICore：caller 与 AICore 之间没有直接 event 边，
ACLGraph capture 分两跳传播。launch 只负责入队，全部入队成功即返回 0，此时设备上
可能尚未开始执行。caller stream 上排有等待 `AicpuDone` 的操作，而 AICPU 侧又等
`AicoreDone`，所以调用方同步 caller stream 时会一直等到两侧都完成。内部不同步、
错误由调用方同步暴露，就是这样实现的。

`AicoreStart` 必须在 AICPU launch 之前 record：AICPU 上的调度器会自旋等待 AICore
的握手，若在其后 record，就会形成 AICPU 等握手、AICore 等 event 的环。Host 入队
顺序上仍是 AICore 先于 AICPU，这也保留了 AICPU 尚未驻留时取消等待中 AICore 的窗口。

## 6. 设备侧执行

AICPU 入口 `simpler_aicpu_kernel_exec` 收到参数包后：

1. 校验包头，以及参数包里 callable 镜像跨度的合法性（非零、按 `ChipCallable`
   对齐、不小于 `sizeof(ChipCallable)`、加上长度不回绕），不合法即拒绝。
2. 进入 TMR 执行路径，从参数包中的 binding 地址读出常驻 KernelArgs 与 Runtime。
3. 设置平台寄存器，由 leader 线程接纳本次调用，多个 AICPU 线程在 barrier 汇合。
4. 运行编排函数，编排提交的 AIV 任务经握手区派发给 AICore。
5. AICore 执行加标量，结果写入测试分配的输出显存。

## 7. 断言了什么

| 顺序 | 断言 |
| ---- | ---- |
| init 后 | `simpler_kernel_mode_supported(ctx) == 1` |
| prepare 并同步后 | context 的 committed memory 大于 0 |
| 每轮 launch | 返回 0；返回后立即清空主机侧参数对象 |
| 每轮同步后 | 输出与 `x + scalar` 逐元素精确相等，覆盖全部 16,384 个元素 |
| 每轮 | committed memory 与 prepare 后相同 |
| 第二轮后 | 第一轮输出仍然正确，未被覆盖 |
| finalize 后 | committed memory 为 0 |

两轮的输入与标量：

| 轮次 | 输入第 i 个元素 | scalar |
| ---- | --------------- | ------ |
| 1 | `i % 127` | 1.25 |
| 2 | `i % 127 + 257` | -3.5 |

launch 返回后立即清空主机侧参数，是为了证明参数快照在入队时就已取走：如果
simpler 仍引用主机侧参数，本次执行会读到被清空的数据，结果不可能正确。

## 8. 图模式验证到哪一步

第 2 节的数值用例是 eager 调用，不包含任何 `aclmdlRICapture` 调用。图模式由两套
测试覆盖，一套验证原语，一套验证公开入口。

**独立探针 `tests/st/a2a3/kernel_capture/`** 验证三流五事件原语本身：

1. 在一条 warmup stream 上 eager 执行一遍。
2. 在 caller stream 上 `aclmdlRICaptureBegin`，执行同一串操作，`aclmdlRICaptureEnd` 得到图。
3. `aclmdlRIExecuteAsync` 回放 100 次，每次更换输入数据并校验结果，证明回放时内部 kernel 确实重新执行。
4. 用链接器 `--wrap` 拦截 `rtStreamAddToModel`、`rtStreamGetCaptureInfo`、`aclmdlRICaptureGetInfo`，断言调用次数为零。

它直接构造执行状态对象，自行加载 AICPU 执行体，手写 record / wait 序列，使用
测试专用的小 kernel，不经过 TMR runtime，也不经过 launch owner 与 binder。

**场景测试 `tests/st/a2a3/tensormap_and_ringbuffer/kernel_mode_capture/`** 走的是
公开入口：24 个场景在 `aclmdlRICaptureBegin` / `End` 之间调用
`simpler_kernel_mode_launch`，取到图后改写输入显存再 `aclmdlRIExecuteAsync` 回放并
比对输出，覆盖冷图、热图、多 callable、跨 stream、重建图、长链、DAG 以及 eager 侧
的批量与拒绝路径。

这套测试用 `LD_PRELOAD` 挂一个观察层（`kernel_capture_observer.cpp` +
`prepare_gate.cpp`，编译成 `observer.so`），按名字截获 CANN 与 runtime 符号，做三件事：

| 职责 | 手段 |
| ---- | ---- |
| 断言 simpler 不做什么 | 截获四个同步入口与 `aclrtQueryEventStatus`，按作用域判定：launch 作用域内**任何**同步都被拒绝并计数（入队路径不得含等待，否则图里装不下）；注册作用域内只拒绝**调用方自己的 stream**，上下文私有 AICPU stream 的那次同步照常放行 |
| 观察它做了什么 | 截获 `rtKernelLaunchWithHandleV2`（AICore）与 `rtsLaunchCpuKernel`（AICPU），解包校验 binding 地址与常驻 `KernelArgs` 在多次调用间不变、两侧 launch 成对；另计 event wait / record 与 `aclrtMemsetAsync` 次数，给出事件拓扑 |
| 注入故障 | `capture_observer_fail_prepare` 让 AICPU 注册 launch 返回 -4333；`prepare_gate.cpp` 用 `aclrtLaunchCallback(ACL_CALLBACK_BLOCK)` 在 AICPU stream 上插一个阻塞回调，供 `blocked_same` / `stream_busy` 制造"前一次 launch 的 serial tail 未完成"的状态 |

注册作用域与 launch 作用域必须分开：注册**会**同步上下文自己的 AICPU stream（见
第 4 节），那次等待正是设备侧注册失败能成为 `prepare_callable` 自身返回值的原因。
阻塞门也随之挂在 launch 上而不是注册上——注册返回时它已经完成，没有"注册尚未完成"
的窗口可言。

## 9. 并入主线时的接口裁决

集成线并入当前主线（已含 #2064）时，按以下原则处理分歧：已合入的 K1 决定契约，
集成线保留主线骨架中没有的实现。

| 事项 | 结论 |
| ---- | ---- |
| 主机错误码编号 | 采用主线编号：`INVALID_ARGUMENT` 为 `BASE - 4`；集成线新增的 callable 与容量错误码顺延为 `BASE - 5` 至 `BASE - 8`（`CALLABLE_STALE` 随 callable generation 一并退役，`BASE - 8` 由 `CAPACITY_EXCEEDED` 占用） |
| launch 的 callable id 越界 | 返回 `INVALID_ARGUMENT`，与 prepare 一致 |
| kernel 入口符号解析 | 每个 runtime 仍导出全部四个入口；ChipWorker 只在 `supported` 非零时解析 init、prepare、launch |
| kernel 模式容量规则 | 使用 K3 的共享 static arena bank，同时覆盖 onboard 与仿真 |
| 同步语义说明 | init 与 prepare 会同步上下文自己的 AICPU stream，launch 路径不同步 |
| callable id 与版本 | prepare 铸 id 经出参返回，失败写 `-1`；纯注册不去重，id 在 context 内不复用、close 后整体失效，不带 generation |
| 设备侧如何找到 callable | 参数包直接携带该 callable 的设备镜像地址与长度，由 binder 从本 context 已提交的驻留信息填入；不再有独立的驻留描述符 |

完整错误码表：

| 名称 | 值 |
| ---- | -- |
| `PTO_RUNTIME_ERR_INTERNAL` | -1000 |
| `PTO_RUNTIME_ERR_UNSUPPORTED` | -1001 |
| `PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE` | -1002 |
| `PTO_RUNTIME_ERR_INVALID_STATE` | -1003 |
| `PTO_RUNTIME_ERR_INVALID_ARGUMENT` | -1004 |
| `PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED` | -1005 |
| `PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED` | -1006 |
| `PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT` | -1007 |
| `PTO_RUNTIME_ERR_CAPACITY_EXCEEDED` | -1008 |

## 10. 已知边界

- 公开 HBG kernel 执行尚未打通。H4 没有提交 PR，HBG 的 kernel 能力位为 0，init 返回 `UNSUPPORTED`。
- A5 只经过编译、单元测试与仿真，没有真机结果。
- 数值用例只覆盖一个固定形状的单算子，不覆盖多算子图、其他形状与并发 launch。
- 公开 launch 路径尚未在 ACLGraph capture 窗口内验证，见第 8 节。
- 同一 host runtime 动态库内限制同一设备只能有一个活动 kernel 上下文；跨动态库副本或跨进程需要调用方自行串行化。
