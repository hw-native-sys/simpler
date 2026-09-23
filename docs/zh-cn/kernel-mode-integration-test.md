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

端到端数值用例在 `tests/ut/py/test_kernel_mode_c_api.py`。TMR 入口是
`test_kernel_eager_launch_executes_fresh_tensor_and_scalar_snapshots`，HBG 入口是
`test_hbg_kernel_eager_launch_executes_graph_snapshots`。它们用 ctypes 直接调用 host
runtime 动态库，自己扮演调用方，不经过 PyTorch，也不经过 simpler 的 Python Worker。

经过 Worker 的是另外两个文件，入口都是 `Worker(level=2, execution_mode="kernel")`：

| 文件 | 硬件 | 覆盖 |
| ---- | ---- | ---- |
| `tests/ut/py/test_worker/test_worker_kernel_mode.py` | 不需要 | 用假的 ChipWorker 检查参数校验、`init(config=...)` 的路由、program 与 kernel 两种模式的接口互斥、prepare/launch/close 之间的串行闸门，以及 teardown 失败后 `close()` 可重试 |
| `tests/ut/py/test_worker/test_worker_kernel_mode_hw.py` | a2a3 真机 | 分别用 TMR 与 HBG 驱动 `init(config=...)`、`kernel_prepare_callable`、`kernel_launch(..., caller_stream=...)` 和 `close()`，再同步调用方 stream 核对结果 |

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
| `simpler_kernel_mode_init` | 核对当前设备，创建专用 AICPU stream、隐藏 AICore stream 和事件；提交固定执行资源并上传常驻 Runtime/KernelArgs。HBG 同时冻结 heap、runtime/SM、Definition、A5 scheduler 与 registry 容量 |
| `simpler_kernel_mode_prepare_callable` | 上传 callable 镜像并铸造 callable_id。TMR 在首次 launch 发布设备驻留信息；HBG 在专用 AICPU stream 注册 slot 和 callable，并等待注册完成 |
| `simpler_kernel_mode_launch` | TMR 编码固定 dispatch packet；HBG 在 Host build 后生成不可变 graph template，并复制本次 HostArgs。两者均通过 binder 在三条 stream 上入队 |
| `finalize_device` | 释放上下文拥有的资源；测试断言 committed memory 归零 |

kernel init 在核对借用的设备之后、创建 stream/event 之前，选择 CANN **进程级硬件
capture event** 模式。已有硬件设置直接复用；如果应用已显式固定为软件模式，则以
ERROR 日志等级输出警告，继续使用软件 event 完成初始化。其他查询或设置错误仍导致
初始化失败。设置失败后会再次查询，允许另一
初始化调用并发地先选中硬件模式。CANN 固定使用硬件模式且不支持模式 API 的平台，
保留其原生行为。该设置影响同进程其他框架算子，close 不会恢复；program 模式的 init
不执行这项设置。应用应在 kernel Worker 初始化之前确定进程级 event 策略。

init 期间的执行体加载会同步上下文自己的 AICPU stream，因此 init 必须在 capture 之外
完成。TMR 与 HBG prepare 都不做 stream/event/device 同步，可以在 capture 内调用。
HBG 的 slot 注册由 init 完成；prepare 只上传 context 自有镜像并保存注册元数据，
每次 launch 在同一 AICPU stream 上先执行幂等 callable 注册，再执行图。两者一起被
capture，按序 replay；新 callable 的 eager 调用不依赖某张已捕获图先执行。
H2D 上传走 `capture_memcpy_h2d`，临时切到 RELAXED 模式后同步拷贝再恢复。
TMR close 在调用方已完成执行且销毁相关 graph 之后，等待上下文自有 AICPU stream 完成
设备注销及回执复制，再释放资源；不替调用方同步 caller stream。等待有超时，注销、
等待或释放失败均保留相应资源供 close 重试。

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

prepare 之后，测试自己同步一次 caller stream 再读 committed memory。prepare 只保证镜像
已上传、注册任务已被接受下发，设备侧注册错误要等这次 warmup 同步才暴露。

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

TMR 的 `simpler_aicpu_kernel_exec` 执行固定参数包：

AICPU 入口 `simpler_aicpu_kernel_exec` 收到参数包后：

1. 校验包头，以及参数包里 callable 镜像跨度的合法性（非零、按 `ChipCallable`
   对齐、不小于 `sizeof(ChipCallable)`、加上长度不回绕），不合法即拒绝。
2. 进入 TMR 执行路径，从参数包中的 binding 地址读出常驻 KernelArgs 与 Runtime。
3. 设置平台寄存器，由 leader 线程接纳本次调用，多个 AICPU 线程在 barrier 汇合。
4. 运行编排函数，编排提交的 AIV 任务经握手区派发给 AICore。
5. AICore 执行加标量，结果写入测试分配的输出显存。

HBG 使用相同入口名，但由 HBG 目标内的强 `consume_kernel_task` 接管：

1. 按 context generation 读取 prepare 阶段注册的 callable 和 slot，校验 graph packet 的 callable、参数计数、设备与 runtime binary 身份。
2. leader 把 captured runtime/SM、Definition 和可选 scheduler 镜像恢复到 context 的固定工作区，重建 RuntimeContext 内部指针、队列与 mailbox；整个过程不申请设备内存。
3. 从已驻留 callable 重绑 AICore 子函数地址，并把外层 Runtime 指向本次恢复出的 SM/arena。
4. 发布 prelaunch `READY`；已经在隐藏 AICore stream 上启动的 kernel 越过 gate，AICPU 进入既有 `aicpu_execute` 调度路径。
5. 所有 AICPU 线程结束后 retire 本次 restore；失败会把 slot/context 置为不可继续 dispatch 的状态。

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
| 断言 simpler 不做什么 | 截获四个同步入口与 `aclrtQueryEventStatus`：注册与 launch 两个作用域内**任何**同步都被拒绝并计数（两者都可能在调用方的 capture 内执行，图里装不下等待） |
| 观察它做了什么 | 截获 `rtKernelLaunchWithHandleV2`（AICore）与 `rtsLaunchCpuKernel`（AICPU），解包校验 binding 地址与常驻 `KernelArgs` 在多次调用间不变、两侧 launch 成对；另计 event wait / record 与 `aclrtMemsetAsync` 次数，给出事件拓扑 |
| 注入故障 | `capture_observer_fail_prepare` 让 AICPU 注册 launch 返回 -4333；`prepare_gate.cpp` 用 `aclrtLaunchCallback(ACL_CALLBACK_BLOCK)` 在 AICPU stream 上插一个阻塞回调，供 `blocked_same` / `stream_busy` 制造"前一次 launch 的 serial tail 未完成"的状态 |

两个作用域仍然分开，但分的不再是同步策略：注册作用域另外武装注册故障注入，
`prepare_fail_register` 只对这一次调用生效。阻塞门挂在 launch 上而不是注册上——
注册只入队，本来就没有可阻塞的等待。`prepare_in_capture` 场景在 capture 打开的状态下
注册第二个 callable，再把它的 launch 一起录进同一张图。

**HBG 场景测试 `tests/st/a2a3/host_build_graph/kernel_mode_capture/`** 通过公开 Worker
入口验证单任务图和含内部中间 Tensor 的两任务图。每个场景 replay 100 次，覆盖更新
输入、依赖调用、反馈图异步回放、eager/replay 混用、图重建和长链。另在 global 与
thread-local capture 中测试首次 prepare，以及已有一次 launch 后 prepare 新 callable。
观察层拒绝 prepare 内任何 stream/event/device 同步；结束 capture 后，先用新 callable
执行 eager，再执行首次 replay，检查图间没有隐含的首次执行依赖。

## 9. 并入主线时的接口裁决

集成线并入当前主线（已含 #2064）时，按以下原则处理分歧：已合入的 K1 决定契约，
集成线保留主线骨架中没有的实现。

| 事项 | 结论 |
| ---- | ---- |
| 主机错误码编号 | 采用主线编号：`INVALID_ARGUMENT` 为 `BASE - 4`；集成线新增的 callable 与容量错误码顺延为 `BASE - 5` 至 `BASE - 8`（`CALLABLE_STALE` 随 callable generation 一并退役，`BASE - 8` 由 `CAPACITY_EXCEEDED` 占用） |
| launch 的 callable id 越界 | 返回 `INVALID_ARGUMENT`，与 prepare 一致 |
| kernel 入口符号解析 | 每个 runtime 仍导出全部四个入口；ChipWorker 只在 `supported` 非零时解析 init、prepare、launch |
| kernel 模式容量规则 | 使用 K3 的共享 static arena bank，同时覆盖 onboard 与仿真 |
| 同步语义说明 | init 会同步上下文自己的 AICPU stream，必须在 capture 外完成；TMR/HBG prepare 无 stream/event/device 同步，允许 capture 内首次调用；HBG 每次 launch 按 AICPU stream 顺序注册后执行，launch 不同步，HostArgs 入队失败直接返回并使 context 进入 poison |
| callable id 与版本 | prepare 铸 id 经出参返回，失败写 `-1`；纯注册不去重，id 在 context 内不复用、close 后整体失效，不带 generation |
| 设备侧如何找到 callable | TMR 参数包直接携带 callable 的设备镜像地址与长度；HBG 每次 launch 先入队幂等注册，再由图执行按 callable id 与 context generation 查表 |

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

- HBG C ABI、Worker eager 和 ACLGraph capture/replay 已在 A3 实机验证；具体用例见第 8 节。
- A5 只经过编译、单元测试与仿真，没有真机结果。
- HBG 数值用例覆盖固定形状向量、内部中间 Tensor、连续异步 launch 和多调用依赖链；尚不代表完整模型或任意形状的覆盖。
- 同一 host runtime 动态库内限制同一设备只能有一个活动 kernel 上下文；跨动态库副本或跨进程需要调用方自行串行化。
- 上述 Simpler 测试不经过 PyPTO/PyTorch 前端，不能作为该入口支持 HBG 的证据。前端仓库仍需完成以下集成项：
  - 按支持的调用契约开放 `python/pypto/torch/launch.py::describe_eager_call` 的 HBG runtime 门禁。
  - 对齐预期的 Simpler revision，并重新构建安装 `pypto._torch_npu` adapter；只重装 Simpler 不会更新 adapter，不能绕过 revision 校验。
  - 从实际 PyPTO/PyTorch 入口验证 eager、capture 内首次准备及 ACLGraph replay，并检查数值、参数寿命和资源释放。
