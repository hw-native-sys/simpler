# simpler 接口增改清单

[返回设计入口](../rdma-a3-simpler.md)

本页只定义 simpler 接入。函数前的“新增/修改/复用”区分提案与已有实现；`pld.rdma.*`、lib 包装、PyPTO 编译装配均不在这里定义。现有公共资源 C ABI 不增加函数，所需变化进入已有配置和分配/释放实现。

## PTO-ISA PR 2155 的依赖合同

依赖固定到 [PR 2155](https://gitcode.com/cann/pto-isa/pull/2155/diffs) 的 `596dc095892da42f1990622830c5209309d025cf`；2026-11-27 再次核对该 PR 当前 head 与此一致。以下接口使用该版本的真实声明，不能以 simpler 自定义同名类型替代。

该 SHA 同时由官方仓 `https://gitcode.com/cann/pto-isa.git` 的 `refs/merge-requests/2155/head` 提供；已用空 Git 仓验证按完整 SHA fetch 成功。现有依赖管理器在主镜像无法取得该提交时使用 GitCode 回退，并校验最终 HEAD，不能回退成另一个版本。

| PTO 接口 / 类型 | 接入必须遵守的约束 |
| --------------- | ------------------ |
| `domain::GetCommWorkspaceSize(config,&sizes)` | 检查 `ACL_SUCCESS`，据返回容量创建独立业务窗口、workspace 和临时 device context；不猜测 workspace 大小 |
| `domain::BuildComm(config,addresses,sizes,context)` | 返回 bool；配置 RDMA-only、External Bootstrap 和 HostExchange，外部 GM 由 simpler 持有；失败不发布 Python domain |
| `domain::TryDestroyComm(context)` | 返回 bool；确认 provider 销毁成功后才释放外部 GM 和端口，失败保留原 owner |
| `BuildAsyncSession<DmaEngine::RDMA>(scratch,workspace,my_rank,session,sync_id,qp_index)` | 检查 bool；独立 UB scratch 采用 `kRdmaScratchBytes`；IN_DIE 的 sync_id 为 0～7，QP 下标先校验再转换 |
| `TPUT_ASYNC<RDMA>(remote_dst,local_src,session,peer)` / `TGET_ASYNC<RDMA>(local_dst,remote_src,session,peer)` | peer 必须按该版本专用重载传入 `uint32_t`；先验证 64 位参数再窄化；远端地址来自已注册 MR |
| `AsyncEvent` / `CompletionRecordCount/At` | 检查 postFailed、error handle、记录数及 kind/address/expected；不得把 handle 当作 GM 指针或只保存 CQE 地址 |
| `RdmaInfo`、`RdmaMemInfo`、`in_die::Queue/Ring` | 直接包含固定版本的 Host-safe 定义；校验 magic/version/IN_DIE 和布局，不复制 HNS1825 结构 |
| AIV `Progress()` | 保持 CQ/SQ tail、Queue.error 和 CQ doorbell 的唯一回收职责；simpler AICPU observer 只读，不接管该职责 |

`register_completion_condition()` 属于 simpler 现有运行时，不宣称为 PR 2155 新增的 PTO API。AICPU 的只读 observer 也是 simpler 的适配实现；其设备可见性假设仍需要 A3 实机验收。

## 1. SDK 及参数接口

### 复用：通用编译与 callable 装配

RDMA 沿用 SDMA 的普通原生 kernel 接入方式：准备 AICPU orchestration 和 AIV 源码，复用 `KernelCompiler`、`CoreCallable.build()`、`ChipCallable.build()` 生成可注册产物。共享源码模板放 `src/a2a3/runtime/tensormap_and_ringbuffer/builtins/rdma/`，不新增 `python/simpler/rdma.py` 或 RDMA 专用公共 builder。

```text
AICPU orchestration + AIV PUT/GET 源码
  → KernelCompiler.compile_orchestration / compile_incore
  → CoreCallable.build
  → ChipCallable.build
  → Worker.register / init
```

SDMA 已有两种对应做法：示例在 `SceneTestCase.CALLABLE` 中声明 orchestration/incores，由通用 `compile_chip_callable_spec()` 装配；测试文件中另有私有 `_build_sdma_callable()`，也只是封装同一通用编译函数。这说明 SDMA 同样需要编译装配，但后端不要求专用公开工厂函数。

对照依据为 2026-11-27 核对的 upstream main `a723ae8f7201ed6db25b089fd68af6d0a4e6babc`：[SDMA 示例声明](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py#L58)、[私有 SDMA 装配 helper](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/tests/st/worker/comm_domain/test_a3_host_source_domain.py#L97)、[通用编译装配](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/simpler_setup/scene_test.py#L1777)。整体 A3 接入基线仍见设计入口的固定版本。

RDMA 示例可按同一惯例提供私有 `_build_rdma_callable(op,platform,runtime)`，集中选择 PUT/GET 共享源码、声明规格并调用通用编译装配；也可以直接使用 `SceneTestCase.CALLABLE`。它属于示例便利代码，不是后端能力或 SDK 契约。直接 SDK 用户可以直接调用既有 KernelCompiler/CoreCallable/ChipCallable，无须依赖测试框架。

PUT 和 GET 仍为两个普通入口产物，运行参数不固化在二进制中。复用既有构建缓存，并使键覆盖源码、PTO commit、平台/runtime、工具链及请求 ABI 版本。缺少 A3 provider 构建能力时在构建或加载阶段报错。编译装配只生成产物，后续仍由既有 Worker/domain API 创建执行和通信资源。

### 应用/示例 Host 编排：`rdma_host_orchestration()`

本次 A3 RDMA 使用示例的 Host 编排函数统一命名为：

```python
from functools import partial


def rdma_host_orchestration(orch, _args, config, *, request) -> None:
    ...

worker.run(partial(rdma_host_orchestration, request=request), config=call_config)
```

它由应用/示例实现，定义于 `examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/test_rdma_async_completion_demo.py`。函数负责创建或复用 domain、准备本次 TaskArgs，并用 `submit_next_level()` 提交通用 RDMA ChipCallable。PUT/GET 可以共用这段 Host 编排，通过传入的请求和 callable handle 选择操作。

这个名称是本设计的示例约定，不新增 simpler 公共 Host API；应用已有编排函数时可以直接接入这些步骤。独立 RDMA ChipCallable 包含薄层 AICPU `aicpu_orchestration_entry()` 和 AIV 内核，不包含此 Python Host 编排函数。Host 入口由 `Worker.run()` 执行，AICPU 入口由 ChipCallable 的执行路径调用。业务已有 AICPU 编排时，也可以直接在其中提交 RDMA AIV，不必再经过一个 RDMA 专用编排入口。

### 修改：已有 `RdmaConfig`

位置：`python/simpler/workspace_config.py`。main 中它已经存在，当前是无字段的 immutable dataclass。扩展为：

```python
@dataclass(frozen=True)
class RdmaConfig:
    qp_num: int = 1
    base_port: int | None = None
    roce_ips: tuple[str, ...] = ()
```

| 字段 | 功能与校验 |
| ---- | ---------- |
| `qp_num` | 每 peer 的 QP 数；首版 simpler 接受 1～8，所有 rank 一致。资源数随 `(rankNum-1)*qp_num` 增长；实际能力以 HCCP 返回为准 |
| `base_port` | `None` 由 L3 coordinator 租用；显式值须在 1～65535 且可租用，所有 rank 一致。不能把默认 60032 无条件用于所有域 |
| `roce_ips` | 空元组使用 PTO 自动发现；非空时与 domain members 顺序一一对应，保存每个设备的 RoCE IP。每项少于 64 个 ASCII 字节且不含 NUL，在集体分配前完成校验。不能使用普通 Host 网卡 IP 代替 |

规范化要拒绝 bool 冒充整数、非法 IP、重复或不完整地址及超范围参数。`roce_ips` 属于域的 immutable 配置；子进程只收到自己选中的 local IP。自动发现遇到多个候选地址时保留 PTO 的报错行为，不任选一个。未请求 RDMA 的域不探测 `libra` 或 RoCE。

`workspace_engines=("rdma",)` 继续等价于默认 `RdmaConfig()`。新增非默认字段首版仅向 A3 `IN_DIE` 接入；A5 若未完成同字段适配，显式拒绝相应配置，不能静默忽略。默认 A5 行为保持现有契约。

### 复用：已有业务入口

| 接口 | 功能 | 本次需要注意的行为 |
| ---- | ---- | ------------------ |
| `Worker(...)` / `register()` / `init()` / `run()` | 创建执行层级、注册产物、启动环境、执行一次 Host orchestration | 同一个 callable 可在多个 run 重复提交 |
| `KernelCompiler` / `CoreCallable.build()` / `ChipCallable.build()` | 编译并装配普通 AICPU/AIV 执行产物 | 直接 SDK 和示例复用；不新增 RDMA 专用公共装配 API |
| `Orchestrator.allocate_domain(...)` | 创建域、窗口和 workspace | 保持签名；A3 允许 RDMA-only DEVICE 域 |
| `get_or_allocate_domain(...)` | 精确配置复用 | 配置比较包含 qp_num、IP、端口请求；自动分配的实际端口是 owner 状态，不改变“请求 None”的复用键 |
| `CommBufferSpec` / `CommDomainHandle.buffer*` | 窗口内逻辑 buffer carve 和获取 | 复用已有对齐和长度校验 |
| `CommDomainHandle.device_context(worker, workspace_engines=("rdma",))` | 取本 rank GM context，登记本轮资源引用 | 必须在每次 Host callback 内获取，不能只在初始化时缓存裸指针 |
| `TaskArgs` / `CallConfig` | 传入参数、读写方向及依赖 | 本设计的六个 uint64 逐轮传入，64 位值不得经 float 中转 |
| `orch.submit_next_level(handle,args,config,worker=rank)` | 提交 chip callable | 本地依赖由普通 TaskId/显式 deps 表达 |
| `CommDomainHandle.release()` / `Worker.close()` | 显式释放或最终关闭 | 受本地 in-flight 与远端静默约束；未知完成不能提前清理 |

不新增 Host 的 `rdma_open/alloc/register/submit/poll/wait/close` 系列接口。直接 SDK 使用的能力由上述已有资源/提交接口和通用编译装配能力提供；示例 helper 仅减少装配代码。

## 2. 现有 Host C ABI 与内部资源函数

### 修改配置结构，保留函数签名

`src/common/platform_comm/comm.h::CommWorkspaceConfig` 现有三个字段保持用途，拟追加固定布局字段：

```cpp
uint32_t rdma_qp_num;       // 默认 1
uint32_t rdma_base_port;    // 0 表示自动；native 预留候选并全 rank 确认后才建链
char rdma_local_ip[64];     // NUL 结尾；空字符串由 PTO 发现
```

结构总长度从 12 B 变为 84 B。Python binding、control payload 编解码、native 转发、A3/A5/sim 生产者和消费者必须同步重建；不支持新旧布局混用。所有路径必须初始化默认字段，禁止把未初始化内存传到未使用该配置的后端。

| 已有接口 / 位置 | 修改内容与功能 |
| --------------- | -------------- |
| `Worker._allocate_domain/_dispatch_control_domain`，`python/simpler/worker.py` | 编码规范化配置和端口请求，发起各 rank 分配；native 以相同 allocation 身份选择候选并投票，复用已有 join/abort 事务，全部成功后才发布 |
| `_handle_ctrl_alloc_domain`，同文件 | 解码、校验长度/配置，转发到当前 chip 的 native allocator |
| `ChipWorker::comm_alloc_domain_windows`，`src/common/worker/chip_worker.cpp` 及 binding | 携带扩展结构；不能在 native 成功前发布 context |
| `comm_alloc_domain_windows(...)`，`src/a2a3/platform/onboard/host/comm_hccl.cpp` | 从原先拒绝全部非零 mask，扩展为支持 RDMA-only；分配本地 HBM、PTO 外部内存，建链并上传 simpler context |
| `DomainAllocation`，同文件 | 扩展现有 owner，保存 PTO Host context、外部 GM 分配、端口占用记录和清理状态；不增加另一套 domain owner 类 |
| `release_domain_allocation()` / `release_domain_windows()`，同文件 | 在现有公共 release/abort 后接入 RDMA 分支；复用 PTO TryDestroyComm，成功后再 free 外部 GM、释放端口，失败不 erase owner |
| `comm_release_domain_windows(...)`，同文件 | 确认所有 rank 已停止访问后销毁 PTO provider，再释放外部 GM；失败保留 owner |
| `comm_abort_domain_windows(...)`，同文件 | 初始化失败且所有参与请求已终止后，清理未发布 owner；不是已运行 DMA 的通用取消函数 |
| `comm_destroy` / `DomainAllocation` 清理路径，同文件 | 不能绕过 RDMA retain 状态、无条件清空 owner map 或让析构提前 free GM |
| `comm_alloc_host_domain_windows`，同文件 | 保留 Host-backed RDMA 不支持的明确错误，不误走 SDMA Host mapping |
| `CommContext` / `comm_workspace<RDMA>()`，`src/common/platform_comm/comm_context.h` | 保持 1104 B 布局和现有 RDMA 枚举值 2；填入 MR VA 和独立 workspace，不新增通用裸指针字段 |

### 复用现有 owner 与清理入口 {#host-resource-ownership}

资源接入直接修改现有 `src/a2a3/platform/onboard/host/comm_hccl.cpp`，不新建 `rdma_domain.{h,cpp}`。不增加 `A3RdmaDomain` 及其 Prepare/Close/AbortInitialization 方法，也不增加 `DomainRequest`、`DomainStatus`、`QuiescenceProof` 类型。配置使用扩展后的 `CommWorkspaceConfig`，owner 使用扩展后的 `DomainAllocation`，外部状态继续使用现有 C ABI 错误码。

| 职责 | 复用或修改位置 | 新接口判断 |
| ---- | -------------- | ---------- |
| 组织 RDMA 准备 | `comm_alloc_domain_windows()` 的 RDMA 分支；按已有 `domain_alloc_via_ipc/fabric` 惯例拆出文件内 `domain_alloc_via_rdma()` | 仅保留一个私有分配 helper，不建立独立生命周期 API |
| 注册 MR、建 QP、初始化 workspace | PTO `GetCommWorkspaceSize()`、`BuildComm()` 及其内部 `InitRdmaWorkspace()` / manager Init | 直接复用，不复制 NIC 初始化实现 |
| 配置与布局检查 | Python 已有配置规范化、PTO 初始化检查、RDMA 分配分支和既有 AIV checked helper | 删除独立 `validate_rdma_context()`；在各自边界校验，保留 simpler extent、PTO pin、workspace backend/version 等接入检查 |
| 生成 simpler GM context | 逐字段转换 PTO 的 `rdmaHostCtx`，复用 `comm_domain_onboard::upload_context()` | 不新增 context 上传接口，不把两种 CommContext 强转 |
| 正常关闭与初始化失败清理 | 已有 `comm_release_domain_windows()` / `comm_abort_domain_windows()` → `release_domain_allocation()` / `release_domain_windows()` → PTO `TryDestroyComm()` | 修改既有 RDMA 清理分支；复用入口并保留正常 release 与本地 initialization abort 的不同前置条件 |
| 端口占用 | 既有 `_allocate_domain()` / native 分配分支协调候选；占用记录、独占锁句柄和实际端口归 `DomainAllocation` | 不新增 `RdmaPortLease::Acquire/Release` 类接口；仍需新增私有端口预留/确认逻辑及释放代码 |

SDMA 对照依据为 upstream main `a723ae8f7201ed6db25b089fd68af6d0a4e6babc`：[`DomainAllocation`](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/platform/onboard/host/comm_hccl.cpp#L93)、[`SdmaWorkspaceOwner` / `dma_workspace_provision()`](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/platform/onboard/host/comm_hccl.cpp#L1103)、[既有 release/abort 共用路径](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/platform/onboard/host/comm_hccl.cpp#L1776)。复用 SDMA 的外部分配和 owner 模式；其 Worker 级 `dma_workspace_provision()` 只接受 SDMA mask，且不接收 domain members/业务 MR 参数，不能原样用于 RDMA。RDMA owner 必须绑定域。PTO `DestroyWorkspace()` 不返回清理成功状态，RDMA 保留所选 `BuildComm()` / `TryDestroyComm()` 路径，不机械照搬 SDMA 的销毁调用。

### 仅保留分配 helper 和必要的交换回调

| 私有函数 | 最小职责 |
| -------- | -------- |
| `domain_alloc_via_rdma(...)` | 使用现有 allocation 参数、配置和 `DomainAllocation*`，分配外部 GM、配置 PTO、调用 BuildComm、完成接入检查并上传 simpler context。端口预留状态随 owner 保存 |
| `domain_host_all_gather(void*,const void*,size_t,void*) -> bool` | 适配 PTO `HostExchange::allGather`。复用既有 allocation/run 命名空间、原子文件发布和有截止时间的等待，补上固定长度记录的按 rank 汇集；不交换另一套 MR/QP 私有协议 |

这两个函数只在 `comm_hccl.cpp` 内使用，不对 SDK 导出。端口协调中需要的局部实现不构成另一组 backend 接口；缩减接口不意味着该逻辑已由现有代码自动提供。

PTO 已有 `GetRdmaExchange()` / `HcclHostAllGather()`，但需要匹配当前 domain 的 communicator。simpler 的 base HCCL communicator 可能包含额外 rank 或不同顺序，不能直接用于任意 members 子集，否则会发生 collective 参与者不一致。本设计继续使用薄的 HostExchange callback，不新建每域 HCCL communicator，不引入 MPI。既有 `file_barrier()` 只证明到达，不传输记录；不能直接冒充 all-gather。`publish_marker()` 的 payload 是文本，PTO 的不透明二进制记录须无损编码后发布，解码时校验精确长度，不能直接传含 NUL 的原始指针。[PTO 交换选择源码](https://gitcode.com/cann/pto-isa/blob/596dc095892da42f1990622830c5209309d025cf/kernels/include/communication/host/detail/rdma_transport.hpp#L80)

当前未找到可直接复用的跨域 RDMA 端口分配器。保留原端口隔离合同，使用系统独占锁实现占用并接入现有分配/释放路径；已有 `GenerationLease` 是固定 generation 目录的共享锁，不能直接充当 device/port 独占锁。其 inode/owner 校验可参照，不能把共享锁改成独占锁而影响现有 generation 生命周期。[GenerationLease 源码](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/common/platform/include/host/comm_generation_lease.h#L32)

状态复用 `COMM_DOMAIN_WORKSPACE_UNSUPPORTED`、`COMM_INITIALIZATION_ABORT_REQUIRED` 和 `COMM_RECLAMATION_UNCONFIRMED`；错误记录保留 PTO `rdmaReport`、ACL/HCCP 原错误和阶段/rank。静默条件仍须由已有 run/domain owner 与 release 协议证明：停止新请求、本地任务完整结束、远端业务访问结束、控制操作全部 join。清理失败保留 map owner、GM 和端口，不能仅凭调用返回或析构尝试而判定成功。

## 3. AIV 请求、callable 与完成接口 {#aiv-request-interfaces}

### 新增公共字节 ABI

拟新增 `src/common/rdma/rdma_request.h`，供构建器、chip orchestration 和 AIV wrapper 共享：

```cpp
struct RdmaTransferArgsV1 {
    uint64_t comm_context;   // simpler CommContext 的 GM 地址
    uint64_t peer_rank;      // domain 内 dense rank，不是物理 device id
    uint64_t local_offset;   // 相对本 rank 窗口起点，字节
    uint64_t remote_offset;  // 相对 peer 窗口起点，字节
    uint64_t nbytes;
    uint64_t qp_index;
};
// sizeof == 48；offsetof 依次为 0,8,16,24,32,40；alignof == 8。
```

PUT/GET 由 callable 种类区分，无 opcode 扩展位。请求没有 Host 指针、QP 句柄或 rkey；这些归 domain owner 管理。此结构是新增协议，不能被描述成 main 已有接口。TaskArgs 传六个 uint64 scalar，chip 层转成同一语义的 CoreTaskArgs；不得 memcpy 含 padding 的临时类型作为网络协议。

### 复用 AICPU 编排契约，独立 callable 只提供薄层入口 {#aicpu-orchestration}

不新增 `rdma_orchestration_config()`、`rdma_orchestration_entry()` 作为 RDMA 后端必备接口。复用现有 `OrchestrationConfig`、`ChipTaskArgs`、`CoreTaskArgs`、`rt_submit_aiv_task()` 和编排 SO 的加载/调用机制；无需修改 AICPU executor 或 SDMA 编排实现。

| 使用方式 | 入口实现 | config 处理 |
| -------- | -------- | ----------- |
| 已有业务 AICPU 编排中调用 RDMA | 在业务入口中组装六个 scalar 并调用 `rt_submit_aiv_task(rdma_kernel_id,core_args)`；RDMA AIV 加入该 ChipCallable 的子内核列表 | 复用业务 config；若业务入参合同变化，同步修改其声明和校验，不强制业务入口只接收六个 scalar |
| 独立 PUT/GET ChipCallable，供 SDK 直接注册调用 | `builtins/rdma/rdma_orchestration.cpp` 提供普通 `aicpu_orchestration_entry(const ChipTaskArgs&)` 薄层实现 | 同一 SO 导出普通 `aicpu_orchestration_config()`，返回 `expected_arg_count=6` |

这两个 `aicpu_*` 名称是已有默认符号约定，不代表框架已经提供了可直接执行的函数体。当前没有找到可直接复用的“任意六个 scalar 转发到单个 AIV”的通用入口实现；独立 callable 仍需上述薄层源码。PUT/GET 可共用同一编排 SO，由各自 ChipCallable 将对应 AIV 子内核绑定为 `func_id=0`。业务自定义入口名称也可继续通过 `ChipCallable.build(func_name=...,config_name=...)` 指定。

对照 upstream main `a723ae8f7201ed6db25b089fd68af6d0a4e6babc`：

- [`AicpuExecutor::load_orch_so()`](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp#L559) 先选择指定符号或默认符号，再执行 `dlsym`。entry 缺失会注册失败；config 缺失仅记录 ERROR 并置空，当前 run 路径跳过它。因此 config 不是当前加载器的硬要求，但独立模板仍导出它，保留参数预检并避免错误日志。
- [`OrchestrationConfig`](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/orchestration_api.h#L441) 只有 `expected_arg_count`；[`run_orchestration()` 的检查](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp#L731) 是 `tensor_count + scalar_count >= expected_arg_count`。它不声明参数类型，也不校验 op/产物 ABI。独立入口必须另查 `tensor_count()==0 && scalar_count()==6`，失败用已有 `rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS,...)` 报错，不能只打印日志后返回。
- [SDMA 示例编排](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/kernels/orchestration/sdma_async_completion_orch.cpp#L20) 自己定义 config/入口，固定读取三个 tensor 和一个 scalar，再提交 producer、consumer 两个 AIV；[示例声明](https://github.com/hw-native-sys/simpler/blob/a723ae8f7201ed6db25b089fd68af6d0a4e6babc/examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py#L58) 指定其入口名称。这是模块的具体编排，不是 SDMA 后端公共 API，函数体不能原样处理 RDMA 的六个 scalar。

独立模板的入口逻辑如下，实现在 `builtins/rdma/rdma_orchestration.cpp`：

```cpp
extern "C" __attribute__((visibility("default")))
OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    if (args.tensor_count() != 0 || args.scalar_count() != 6) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "RDMA expects six uint64 scalars");
        return;
    }
    CoreTaskArgs core_args;
    for (uint32_t i = 0; i < 6; ++i) {
        core_args.add_scalar(args.scalar<uint64_t>(i));
    }
    rt_submit_aiv_task(0, core_args);
}
```

模板只做入参形状检查和任务转发；MR/QP 资源由 Host domain owner 准备，请求字段合法性与完成容量由 AIV checked helper 校验，完成事件由 AICPU completion backend 汇合。op/子内核绑定及产物 ABI 在构建装配合同中校验，不塞进 `OrchestrationConfig` 的计数字段。

### 设备与调度接口

| 函数 | 位置 | 功能 |
| ---- | ---- | ---- |
| `aicpu_orchestration_config()` | `builtins/rdma/rdma_orchestration.cpp` | 复用默认符号约定，独立模板实现最少六个参数的预检声明；不检查类型或 op/ABI |
| `aicpu_orchestration_entry(const ChipTaskArgs&)` | 同上 | 独立模板的薄层实现；严格检查零 tensor、六 scalar，转发到 `rt_submit_aiv_task(0,...)`；已有业务入口可直接包含同类提交逻辑 |
| `AicpuExecutor::load_orch_so()` / `run_orchestration()` | `aicpu/aicpu_executor.cpp:476/631` | 原样复用指定/默认符号加载、config 调用及 entry 调用，不新增 RDMA 分支 |
| `rt_submit_aiv_task(int32_t,const CoreTaskArgs&)` | `orchestration/orchestration_api.h:194` | 原样复用普通 AIV 提交；内核编号必须匹配当前 ChipCallable 的子内核绑定 |
| `kernel_entry(...)` | `builtins/rdma/rdma_put.cpp`、`rdma_get.cpp` | 共享模板生成两个普通 AIV 内核，调用 checked helper |
| `rdma_put_async_checked/rdma_get_async_checked` | `runtime/backend/rdma/rdma_completion_kernel.h` | 新增；完整校验请求及完成容量、构造 session、发 WQE，再复用现有函数登记真实 event |
| `register_completion_condition(AsyncCtx&,const CompletionToken&)` | `runtime/async_kernel_api.h:45` | 复用；向本任务 slab 追加 IN_DIE 完成 token 并更新 count，无须增加另一套登记 API |
| `pto2::detail::defer_flush(AsyncCtx&)` | `runtime/completion_context.h:51` | 复用；将记录和计数按 A3 cache/屏障协议发布给 AICPU，必须先于 FIN |
| `poll_rdma_in_die_event(handle,workspace,owner_epoch,InDieCompletionCursor &cursor)` | `runtime/backend/rdma/rdma_completion_scheduler.h` | AICPU 有界只读查询成功前缀，返回 READY/PENDING/FAILED，不回收 CQ；每个完成条件独占 cursor |
| `retire_rdma_in_die_event(InDieCompletionCursor &cursor)` | 同上 | 清空当前完成条件的观察状态；不修改 PTO tail、不 free workspace |
| `completion_backend_ops_for()` | `runtime/async_wait.h` | 注册 A3 RDMA 条件分发，校验 engine/type，接入 FIN 汇合 |

上表中的 `runtime/` 均位于 `src/a2a3/runtime/tensormap_and_ringbuffer/`。内部 owner_epoch 在每个完成条件的 cursor 中保存，不改变 PTO 事件编码，也不把悬空 scalar 指针变成可跨 domain 使用的句柄。

首版内置 AIV kernel 每次只发一个 PUT/GET，并独占当前任务的 completion slab，直到登记并返回 FIN。在检查与登记之间，PTO 调用不写 simpler AsyncCtx，因此只需在提交前检查还剩一条容量；不增加 reserve/commit/cancel 接口。现有 A3 SDMA `send_request_entry()` 已采用同一所有权和容量检查模式。不同 AIV 任务使用自己的 slab；共享 QP 不等于共享 slab。详见[完成协议](completion.md#aiv-submission)。

新增条件类型拟命名 `COMPLETION_TYPE_RDMA_IN_DIE_EVENT_HANDLE=4`；现有 `COMPLETION_ENGINE_ROCE=1` 复用。这里是 **simpler completion type**；PTO 的 `CompletionKind::RDMA_IN_DIE_CQE=4` 属于另一枚举，数值相同不表示可以直接强转。原 A5 `RDMA_EVENT_HANDLE=3` 不重新解释为 A3。

A3 的分发表显式处理未支持的 type2/3，不能直接在当前两项数组末尾 append 后误用下标。token 保持 `addr=完整 event.handle`、`backend_cookie=PTO workspace 地址`、`expected_value=0`；PTO record 仅用于交叉验证。`CompletionCondition` 保存内部 `owner_epoch=TaskId::to_uint64(task_token)+1`，由 mailbox consumer 填充，传给 observer；这不是 AIV 参数或 PTO generation 字段，也不是独立的跨 run 世代。受 run lease 保护的 workspace 在本轮内不释放，每次登记/退休都重置私有 cursor。

## 4. 构建、配置与测试接口

| 位置 | 修改或复用 | 功能 |
| ---- | ---------- | ---- |
| `simpler_setup/pto_isa.py` 和 pin 管理 | 修改依赖版本 | 固定包含 PR 2155 的 commit，校验需要的 headers 可打包 |
| `simpler_setup/kernel_compiler.py` | 修改 | A3 RDMA artifact 使用 2201 及既有 `PTO_RDMA_SUPPORTED`、`PTO_RDMA_BACKEND_IN_DIE_SUPPORTED` 定义；不夹带 A5 define |
| `simpler_setup/runtime_builder.py/runtime_compiler.py` | 修改 | 扩展既有 `SIMPLER_ENABLE_PTO_RDMA_WORKSPACE` 对 A3 的接入；构建缓存区分平台/backend/PTO pin |
| A3 `platform/onboard/host/CMakeLists.txt` | 修改 | 为现有 comm_hccl.cpp 的 RDMA 分支配置匹配 runtime 的 HCCP/verbs/driver/tsd 头文件；保留 PTO 的 `libra.so/libtsdclient.so` 动态加载 |
| A3 TMR Host/AICPU/AIV 构建与安装清单 | 修改 | 共享 ABI、只读 observer 和 builtins 源码进入 source/wheel 产物；AICPU 只包含 Host-safe 布局，不拉入 AIV intrinsic/HCCP Host 实现 |
| sim provider | 复用明确能力边界 | 普通 sim 不伪装实际 RDMA ready；单测使用显式 fake 队列/provider 注入 |
| `tests/ut/{py,cpp}` 与 A3 examples/ST | 新增 | ABI、资源回滚、CQ 绕圈/并发、FIN 汇合、真实双 rank PUT/GET；详见验收文档 |

不新增环境变量或新 backend 开关；复用现有构建开关和 PTO 已定义的 backend 宏。硬件、runtime 和 provider 的不兼容要在 capability/构建检查中明确失败，不能以换成 SDMA、Host staging 或 event.Wait 来掩盖。
