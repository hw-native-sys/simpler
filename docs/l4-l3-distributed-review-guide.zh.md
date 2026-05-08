# L4/L3 分布式实现 Review Guide

本文面向代码 review。它先说明 `/home/ntlab/zhouzhe/simpler_l4/pypto_top_level_documents/UBL128_serving.md` 要做的完整 serving 系统是什么，再说明当前代码里的 L4/L3 原型实现了其中哪些基础能力、没有实现哪些 serving 功能，最后按模块拆解代码路径和 review 风险点。

先给结论：**UBL128_serving.md 描述的是完整的 prefill/decode 解耦推理服务系统；当前 Simpler L4 代码不是完整 serving 系统，而是实现了一个可被 serving 系统复用的“分层 Worker 远程派发 + Tensor 数据面传输基座”。**

当前实现已经打通：

- **L4 -> remote L3 控制面**：L4 本地 scheduler/mailbox 到远端 L3 daemon 的 dispatch。
- **Callable 分发**：L4 注册的 Python callable 通过 catalog 推送到 L3 daemon/backend。
- **Tensor 数据面抽象**：small tensor inline，大 tensor 走 TensorPool handle。
- **Transport backend 边界**：默认 gRPC，RXE/ibverbs RDMA write MVP，HCOMM 可选适配层。
- **Output writeback**：大 `OUTPUT / OUTPUT_EXISTING` 可由 L3 通过 RXE 写回 L4 本地 output buffer，失败时 fallback 到 response/TensorPool 路径。

当前实现没有做：

- 外部 HTTP/gRPC serving frontend。
- prefill/decode 角色拆分、continuous batching、request scheduler/router。
- KV Meta Server、prefix radix tree、SSU LBA 分配、KV block 生命周期。
- NPU -> SSU 的 SO/UB Urma KV 数据面。
- 生产设计中的 SO uRPC hot-path 协议栈。
- UBL128/PC16/SSU 的配置驱动拓扑管理。

## 术语表

这张表只解释本文 review 里会反复出现的词，重点是帮助 reviewer 判断“当前代码实现的是哪一层能力”。

| 术语 | 含义 | 在当前实现里的状态 |
|------|------|--------------------|
| L4 | Simpler runtime 的 level-4 orchestrator/parent worker 层，负责向下一层 worker 派发任务。 | 已实现远程 L3 派发入口；不是完整 serving frontend。 |
| L3 | Simpler runtime 的 level-3 worker 层，接收 L4 派发的 task 并继续执行/下发。 | 远端由 `L3Daemon` + backend process 承载。 |
| Worker | Simpler 的层级执行单元。L4/L3/L2 等层级通过 mailbox 或 RPC 组织任务。 | 当前复用本地 hierarchical worker 模型，并用 remote mailbox shim 接远端 L3。 |
| L3Daemon | 远端 L3 的 gRPC server 进程，接收 dispatch、catalog、TensorPool RPC。 | 已实现；真正执行在 daemon 启动前 fork 出来的 backend process。 |
| backend process | L3 daemon 后面的执行进程，持有 `Worker(level=3)`、`TensorPool` 和 transport backend。 | 已实现；避免在已有 gRPC 线程的进程里继续 fork worker。 |
| 控制面 | 传 task 元数据、callable id、tensor handle、错误码、heartbeat 等小消息的路径。 | 当前主要是 gRPC + protobuf。 |
| 数据面 | 传 tensor bytes 的路径。 | 当前支持 inline、gRPC chunk、RXE RDMA write，HCOMM 为可选适配。 |
| Dispatch | L4 把一个 callable + args + config 发给 L3 执行的动作。 | 已实现 `DispatchReq/DispatchResp`。 |
| Callable Catalog | callable 的分发/同步表。L4 注册 Python 函数后，把序列化 payload 推给 L3。 | 已实现；使用 `cloudpickle`，只适合受信任集群。 |
| closure | Python 函数捕获的外部变量。 | 会被序列化到 L3；远端修改的是副本，不会改 L4 本地对象。 |
| TensorPool | L3 backend 内部的 tensor byte storage 和 handle 管理器。 | 已实现；当前是进程内 bytearray storage，不是 UBL128 设计里的 SSU KV 存储。 |
| TensorRef | dispatch 消息里的 tensor 参数描述，可能是 inline bytes，也可能指向 `TensorHandle`。 | 已实现。 |
| TensorHandle | TensorPool/transport 返回的远端数据句柄，包含 shape、dtype、size、transport、transport_desc 等。 | 已实现。 |
| inline tensor | 小 tensor 直接放进 protobuf 消息里传输。 | 已实现；由 inline threshold 控制。 |
| transport backend | TensorPool 注册内存和搬运 bytes 的后端抽象。 | 已实现 `grpc`、`rxe`、`hcomm` 三类入口。 |
| gRPC | 基于 HTTP/2 + protobuf 的通用 RPC。 | 当前控制面主路径；在 UBL128 设计里适合 DCN/运维/非 hot-path。 |
| RDMA | Remote Direct Memory Access，远端直接内存访问，允许一端把数据直接写入/读出另一端注册内存。 | 当前 RXE helper 用 RDMA write 做实机数据面 smoke。 |
| RoCE | RDMA over Converged Ethernet，在以太网上承载 RDMA verbs。 | 当前机器 `ibv_devices` 可见时用于 smoke；不是 UBL128 SO/UBG 生产路径。 |
| RXE | Linux Soft-RoCE provider，用软件方式把普通以太网设备暴露成 RDMA verbs 设备。 | 当前用于开发/烟测真实 ibverbs 流程；性能和生产硬件 RoCE/UB 不是一回事。 |
| ibverbs / verbs | Linux RDMA 用户态接口族，应用通过 verbs 创建 MR/QP/CQ 并发起 RDMA write/read。 | `rxe_verbs_helper.c` 直接使用。 |
| MR / QP / CQ | RDMA 里的 Memory Region、Queue Pair、Completion Queue。 | RXE helper 内部使用；文档里不要求 reviewer 深入 verbs 细节。 |
| HCOMM | 昇腾通信相关库/接口集合，这里作为可选通信后端候选。 | 当前只在 Simpler 侧做适配层，不依赖修改 `3rd/hcomm`。 |
| CANN | 昇腾软件栈基础组件。 | HCOMM runtime 可能依赖；当前 L4/L3 主路径不依赖 NPU CANN 执行。 |
| UB / UBG | UBL128 设计里的统一总线/Scale Out 物理协议族。 | 当前代码没有实现 UB/UBG。 |
| Urma | UB 上的可靠内存访问语义层，类似面向 UB 网络的 RDMA。 | UBL128 目标路径；当前未实现。 |
| uRPC | 基于 Urma 的轻量 RPC，用于 SO/SU hot-path 内部 RPC。 | UBL128 目标路径；当前未实现，当前用 gRPC 做原型。 |
| DCN | 数据中心 RoCE/TCP 网络，CPU 可见，承载外部接入、运维和非 hot-path 服务。 | 当前 gRPC 原型可类比 DCN 控制路径。 |
| SO | Scale Out 网络，跨 UBL128 的 NPU/CPU/SSU any-to-any 网络。 | UBL128 目标数据面/热控制面；当前未实现。 |
| SU | Scale Up 网络，UBL128 域内 NPU 间高带宽网络，主要承载 EP/DP。 | 当前未实现。 |
| UBL128 | 设计文档里的 128 NPU high bandwidth domain。 | 当前代码不感知该拓扑。 |
| PC16 | 设计文档里的 16 NPU 服务器单元。 | 当前代码不感知该拓扑。 |
| SSU / SSU12 | 设计文档里的 KV/prefix 持久化存储单元/机框。 | 当前未实现；不要和 TensorPool 混淆。 |
| KV cache | LLM attention 的 key/value 缓存。 | UBL128 核心设计内容；当前 L4/L3 原型未实现 KV 语义。 |
| Prefix cache | 按 token prefix 复用已计算 KV 的 cache。 | 当前未实现。 |
| ChunkRecord | UBL128 KV Meta Server 返回的 KV block 元数据，描述每层 KV 在哪个 SSU/LBA。 | 当前未实现。 |
| LBA | Logical Block Address，SSU 上无文件系统的块地址。 | 当前未实现。 |
| HBM | NPU 高带宽显存。 | 当前 RXE smoke 传的是 host memory，不是 NPU HBM。 |
| `INPUT` / `OUTPUT` / `INOUT` | Simpler tensor 参数方向 tag。 | 已支持远程参数语义；`INOUT` 尚未做完整双向 RXE fast path。 |

## 1. UBL128 Serving 总设计先做什么

`UBL128_serving.md` 的目标是一个带 prefix cache 的 prefill/decode 解耦推理服务。它不是单个 runtime API，而是一套跨 CPU、NPU、SSU 存储和多张网络的完整 serving 系统。

核心角色如下：

| 角色 | UBL128 文档中的名字 | 物理载体 | 主要职责 |
|------|--------------------|----------|----------|
| 前端/调度器 | `frontend` / F | 鲲鹏 CPU service_access | 外部 ingress、tokenize、PrefixMatch、选择 prefill/decode worker、回流 token |
| KV 元数据 | `kv-meta` / M | 鲲鹏 CPU | prefix radix tree、ChunkRecord、SSU LBA 分配、引用计数 |
| Prefill host | PC | PC16 host CPU | 代 NPU 做 metadata RPC、给 NPU doorbell/mailbox |
| Prefill NPU | PN | Ascend NPU | 计算 prompt/unmatched tail，写新 KV |
| Decode host | DC | PC16 host CPU | 管理 decode slot、代 NPU 做 metadata RPC |
| Decode NPU | DN | Ascend NPU | 读 KV、decode、continuous batching |
| KV 存储 | SSU / S | SSU12 存储框 | LBA-direct KV/prefix block 存储 |

典型请求路径：

```mermaid
flowchart LR
  U[external user] -->|HTTP/gRPC on DCN| F[frontend / scheduler]
  F -->|PrefixMatch on SO uRPC| M[KV Meta Server]
  M -->|ChunkRecord list| F
  F -->|DispatchPrefill on SO uRPC| PC[Prefill host CPU]
  PC -->|doorbell / mailbox| PN[Prefill NPU]
  PN -->|SO Urma read/write| S[SSU KV store]
  F -->|DispatchDecode on SO uRPC| DC[Decode host CPU]
  DC -->|doorbell / mailbox| DN[Decode NPU]
  DN -->|SO Urma read/write| S
  DN -->|token stream via host| F
  F -->|HTTP/gRPC on DCN| U
```

网络/协议分层是这个设计的硬约束：

| 网络 | 设计用途 | 协议选择 |
|------|----------|----------|
| DCN/RoCE | 外部 ingress、运维、健康检查、POSIX FS 等非 hot-path | gRPC 或 HTTP/JSON |
| SO/UBG | 跨 UBL128 hot-path RPC、KV bytes、prefill->decode 移交 | uRPC over UB Urma；数据面用 Urma read/write |
| SU | UBL128 域内 NPU 间 EP/DP 通信 | Urma，不承载普通 RPC |
| 服务器内 UB | host CPU 与本机 NPU doorbell/mailbox | 本地 mailbox/doorbell |

所以，完整 serving 系统需要闭合三件事：

1. 请求级控制面：frontend/scheduler/router，把请求拆成 prefill 和 decode 阶段。
2. KV/prefix 语义：KV Meta Server、prefix radix tree、ChunkRecord、SSU LBA 分配与回收。
3. 数据面：NPU/CPU/SSU 之间通过 SO/SU/UB 的真实数据搬运。

## 2. 当前代码里的 L4 是什么

当前代码里的 **L4** 指 Simpler runtime 的 level-4 orchestrator/parent worker 层，不等同于 UBL128 文档里的完整 frontend/scheduler。它现在解决的问题更底层：

> L4 进程如何把一个 Simpler task 派发到远端 L3 daemon，并让远端 L3 能拿到 callable、scalar 参数、tensor 参数，执行后把 output tensor 返回或写回。

也就是说，当前 L4/L3 原型更接近 UBL128 文档中 **F -> PC/DC 的 Dispatch 基础能力**，外加一个临时的 host-side tensor transport MVP。它还没有实现请求级 serving 编排，也没有实现 KV/prefix 的业务语义。

当前原型的大图：

```mermaid
flowchart LR
  subgraph L4Host[L4 host process]
    A[User L4 orchestrator]
    B[L4 C++ scheduler]
    C[remote PROCESS mailbox shim]
    D[RemoteWorkerProxy]
    E[local output RXE region optional]
  end

  subgraph L3Host[L3 daemon host]
    F[gRPC L3Daemon]
    G[backend process]
    H[TensorPool]
    I[Worker level=3]
    J[transport backend grpc/rxe/hcomm]
  end

  A --> B
  B --> C
  C --> D
  D -->|DispatchReq/Resp gRPC| F
  D -->|TensorPool RPC / RXE write| H
  F -->|Pipe ops| G
  G --> H
  G --> I
  H --> J
  I -->|output bytes| H
  I -->|RXE writeback optional| E
```

当前代码把远端 L3 包装成一个“看起来像本地 PROCESS child”的 mailbox shim。这样 L4 C++ scheduler 的调度模型可以先不重写；Python shim thread 负责把 mailbox task 转成远端 RPC。

## 3. 与 UBL128_serving.md 的功能映射

| UBL128 设计能力 | 当前代码状态 | 相关代码 | 说明 / 差距 |
|----------------|--------------|----------|-------------|
| 外部 HTTP/gRPC ingress | 未实现 | 无 | 当前没有服务接入层、tokenize/detokenize、用户请求协议。 |
| Frontend scheduler/router | 部分基础可复用 | `python/simpler/worker.py`, `remote_proxy.py` | 当前 L4 能向远端 L3 派发 task，但还不是请求级 scheduler，不懂 prefill/decode、slot、batch、quota。 |
| F -> PC/DC Dispatch | 已实现原型 | `Worker.add_remote_worker()`, `_remote_worker_loop()`, `RemoteWorkerProxy.dispatch()` | 现在走 gRPC 到 L3 daemon；设计文档中的 hot-path 目标是 SO uRPC。 |
| Callable/任务代码分发 | 已实现 | `catalog.py`, `l3_daemon.py` | 用 `cloudpickle` 推送 callable。closure 是远端副本，不能修改 L4 本地 Python 状态。 |
| Scalar 参数派发 | 已实现 | `serialization.py`, `dispatch.proto` | scalar-only dispatch 可复用 persistent L3 Worker。 |
| Tensor 参数派发 | 已实现原型 | `serialization.py`, `tensor_pool.py`, `transport_backend.py` | small inline，大 tensor handle；host-side tensor，不是 NPU HBM KV。 |
| L3 output 返回 | 已实现原型 | `remote_proxy.py`, `serialization.py` | 支持 inline、L3 TensorPool handle、RXE local output ACK。 |
| DCN gRPC 控制面 | 已实现原型 | `rpc.py`, `l3_daemon.py`, `dispatch.proto` | 当前 gRPC 适合 smoke/control path；与 UBL128 文档中 DCN 非 hot-path gRPC 一致。 |
| SO uRPC hot-path | 未实现 | 无 | 目前没有 uRPC over UB Urma runtime；gRPC 不能代表最终 SO hot-path。 |
| RXE/RoCE 数据面 smoke | 已实现 MVP | `transport_backend.py`, `rxe_verbs_helper.c` | 基于 ibverbs RC RDMA write；用于实机 smoke，不是 UBL128 SO/UBG 生产路径。 |
| HCOMM 适配边界 | 部分实现 | `transport_backend.py`, `hcomm_abi_shim.cc` | 只做 Simpler 外部适配，不依赖修改 `3rd/hcomm`。 |
| Prefill worker 角色 | 未实现 | 无 | 当前 L3 只是 generic `Worker(level=3)`，不区分 prefill。 |
| Decode worker 角色 | 未实现 | 无 | 当前没有 decode slot table、continuous batching、sampling loop。 |
| KV Meta Server | 未实现 | 无 | 没有 prefix radix tree、ChunkRecord、LBA free list、refcount。 |
| Prefix cache | 未实现 | 无 | 没有 token chunk hash、longest prefix match、cache hit/miss。 |
| SSU LBA-direct KV store | 未实现 | 无 | 当前 TensorPool 是进程内 bytearray storage，不是 SSU。 |
| NPU -> SSU SO Urma read/write | 未实现 | 无 | 当前 RXE 是 host memory RDMA write smoke；不涉及 NPU HBM/SSU。 |
| EP/DP on SU | 未实现 | 无 | Simpler 当前没有 UBL128 拓扑感知 EP/DP 通信调度。 |
| 配置驱动 A/B/C 三档硬件 | 未实现 | 无 | 当前 remote endpoint/transport 是局部配置，不是完整拓扑配置系统。 |

Review 时建议把当前 PR/分支按“runtime 基座”来评估，而不是按“完整 UBL128 serving 系统”来验收。更具体地说：

- 可以 review：L4 到远端 L3 的 task 派发语义、callable catalog、tensor handle 生命周期、RXE/HCOMM backend 边界、测试覆盖。
- 不应误判为已完成：prefill/decode serving、KV cache、prefix cache、SO uRPC、NPU/SSU 数据面。

## 4. 主要代码入口

后续章节按当前实现拆解。主要代码入口：

```text
python/simpler/worker.py
python/simpler/distributed/
  catalog.py
  rpc.py
  remote_proxy.py
  l3_daemon.py
  serialization.py
  tensor_pool.py
  transport_backend.py
  rxe_verbs_helper.c
  hcomm_abi_shim.cc
  proto/dispatch.proto

src/common/hierarchical/worker_manager.cpp
tests/ut/py/test_distributed/
tools/test_rxe_data_plane.sh
tools/benchmark_rxe_data_plane.py
```

## 5. 当前 L4/L3 原型总览

### 5.1 为什么需要 remote L3

原有 L4->L3 路径假设 L4 可以在本机 fork L3 child worker，并通过共享内存 mailbox 与该 child 交互。跨 host 后这个假设失效：

- 远端 L3 不能由 L4 直接 `fork()`。
- L4 本地 pointer 不能直接发给远端。
- L4 本地 callable registry 不能通过 fork copy-on-write 自动继承。
- tensor payload 需要显式数据面传输。

当前设计把远端 L3 包装成一个“看起来像本地 PROCESS child”的 mailbox shim：

```mermaid
flowchart LR
  A[L4 C++ scheduler] --> B[本地 remote mailbox]
  B --> C[Python remote shim thread]
  C --> D[RemoteWorkerProxy]
  D -->|gRPC| E[L3Daemon]
  E -->|Pipe| F[L3 backend process]
  F --> G[Worker level=3]
```

L4 C++ scheduler 不知道远端存在；它只看到一个普通 PROCESS-mode next-level mailbox。Python shim thread 负责把 mailbox 中的 task 转成远端 RPC。

### 5.2 控制面和数据面的边界

```mermaid
flowchart TB
  subgraph ControlPlane[控制面]
    A[Callable Catalog]
    B[DispatchReq / DispatchResp]
    C[Heartbeat]
    D[TensorPool Alloc/Free/Refresh]
  end

  subgraph DataPlane[数据面]
    E[inline bytes]
    F[gRPC PushTensor/PullTensor chunks]
    G[RXE ibverbs RDMA write]
    H[HCOMM optional write facade]
  end

  B --> E
  D --> F
  D --> G
  D --> H
```

控制面只传元数据、handle、callable id、错误信息和生命周期操作。数据面负责 tensor bytes 的实际搬运。

## 6. 模块拆解

### 6.1 Worker remote child 接入

代码：

- `python/simpler/worker.py`
- `src/common/hierarchical/worker_manager.cpp`

关键入口：

- `Worker.add_remote_worker(endpoint, **options)`
- `Worker._ensure_distributed_catalog()`
- `Worker._init_hierarchical()`
- `Worker._start_hierarchical()`
- `_remote_worker_loop(buf, proxy)`

职责：

1. L4 用户调用 `add_remote_worker()` 注册远端 endpoint。
2. `Worker.init()` 时创建本地 shared-memory mailbox。
3. 创建 `RemoteWorkerProxy(endpoint, catalog, **options)`。
4. `_start_hierarchical()` 时先 `proxy.handshake()`，再启动 `_remote_worker_loop` thread。
5. 把 remote mailbox 注册给 C++ `_Worker.add_next_level_process(...)`。
6. C++ scheduler 发布 `TASK_READY` 后，remote shim thread 读取 mailbox，调用 `proxy.dispatch(...)`。
7. dispatch 成功或失败后，shim 写回 mailbox error 区和 `TASK_DONE`。

流程：

```mermaid
sequenceDiagram
  participant User
  participant W4 as Worker(level=4)
  participant Cpp as C++ scheduler
  participant Shim as remote shim thread
  participant Proxy as RemoteWorkerProxy
  participant L3 as L3Daemon

  User->>W4: add_remote_worker(endpoint)
  User->>W4: init()
  W4->>Proxy: create proxy
  User->>W4: run(l4_orch)
  W4->>Cpp: submit_next_level(...)
  Cpp->>Shim: TASK_READY in mailbox
  Shim->>Proxy: dispatch(cid,args,cfg)
  Proxy->>L3: gRPC Dispatch
  L3-->>Proxy: DispatchResp
  Proxy-->>Shim: return / raise
  Shim->>Cpp: TASK_DONE + error code
```

Review 关注点：

- `add_remote_worker()` 是否应该限制只能在 `level >= 4`、`init()` 前调用。
- remote mailbox 与本地 child worker mailbox 是否共享同一状态协议。
- `_remote_worker_loop()` 是否能正确传播异常到 mailbox error 区。
- remote proxy close 是否和 `_SHUTDOWN` 时序匹配。
- `worker_manager.cpp` 的 tensor tag 扩展是否保持旧 blob 格式兼容。

### 6.2 Callable Catalog

代码：

- `python/simpler/distributed/catalog.py`
- `python/simpler/worker.py`
- `python/simpler/distributed/l3_daemon.py`

职责：

- L4 侧把注册过的 callable 用 `cloudpickle` 序列化。
- callable id 保持和 L4 registry 一致。
- 版本号用 payload hash 表示，dispatch 时带上 `callable_version`。
- L3 daemon 和 backend process 都维护一份 catalog mirror。

流程：

```mermaid
sequenceDiagram
  participant W4 as Worker.register
  participant Cat as L4 Catalog
  participant Proxy as RemoteWorkerProxy.handshake
  participant D as L3Daemon CatalogService
  participant B as backend catalog mirror

  W4->>Cat: register(fn, callable_id)
  Proxy->>D: PushCallable(payload)
  D->>D: install in daemon catalog
  D->>B: pipe ("push", cid, version, payload)
  B->>B: cloudpickle.loads(payload)
```

Review 关注点：

- `cloudpickle` 是受信任集群内机制，不能暴露给不可信客户端。
- callable closure 被序列化到 L3，远端修改的是反序列化副本，不会修改 L4 本地 Python 对象。
- version mismatch 时当前行为是 L3 lookup 失败并返回 error。

### 6.3 RPC/protobuf 控制协议

代码：

- `python/simpler/distributed/proto/dispatch.proto`
- `python/simpler/distributed/rpc.py`
- generated `dispatch_pb2*.py`

核心 service：

```protobuf
service L3Worker {
  rpc Dispatch(DispatchReq) returns (DispatchResp);
  rpc Heartbeat(Empty) returns (Health);
}

service Catalog {
  rpc PullCallable(CallableRef) returns (CallablePayload);
  rpc PushCallable(CallablePayload) returns (Empty);
}

service TensorPool {
  rpc AllocTensor(TensorAllocReq) returns (TensorHandle);
  rpc FreeTensor(TensorFreeReq) returns (Empty);
  rpc RefreshTensor(TensorRefreshReq) returns (TensorHandle);
  rpc PullTensor(TensorHandle) returns (stream TensorChunk);
  rpc PushTensor(stream TensorChunk) returns (TensorHandle);
}
```

重要消息：

- `DispatchReq.tensor_args`：旧地址式 tensor metadata，主要用于没有 tensor_refs 的兼容路径。
- `DispatchReq.tensor_refs`：当前分布式 tensor 数据面路径，小 tensor inline，大 tensor handle。
- `DispatchResp.output_tensors`：L3 output 回传，可能是 inline、L3 TensorPool handle，或 L4 local output ACK handle。
- `TensorHandle.transport`：`grpc` / `rxe` / `hcomm`。
- `TensorHandle.transport_desc`：transport 私有描述。

Review 关注点：

- protobuf 字段是否后向兼容。
- `tensor_args` 和 `tensor_refs` 同时存在时的语义是否明确。当前 L4 生成 tensor_refs 时会把 `tensor_args=[]`。
- `TensorHandle.node_id` 用于区分 L3 pool handle 和 L4 local output ACK。

### 6.4 L4 RemoteWorkerProxy

代码：

- `python/simpler/distributed/remote_proxy.py`

职责：

- 执行 heartbeat/catalog handshake。
- 把本地 `TaskArgs` staged 成 wire `TensorRef`。
- 分配/释放 L3 TensorPool handle。
- 对 input tensor 选择 gRPC/HCOMM/RXE push。
- 对 output tensor 选择 L4 local RXE writeback 或普通 PullTensor。
- 处理错误和资源释放。

关键函数：

- `handshake()`
- `dispatch()`
- `_stage_tensor_args()`
- `_stage_local_output_tensor()`
- `_push_remote_tensor_rxe()`
- `_push_remote_tensor_hcomm()`
- `_push_remote_tensor_grpc()`
- `_write_output_tensors()`
- `_free_remote_handles()`
- `_close_local_output_regions()`

#### 6.4.1 input tensor staging

```mermaid
flowchart TD
  A[TaskArgs tensor] --> B{nbytes <= inline_threshold?}
  B -->|yes| C[TensorRef.inline_data]
  B -->|no| D[TensorPool.AllocTensor on L3]
  D --> E{handle.transport}
  E -->|rxe| F[RDMA write into L3 buffer]
  E -->|hcomm| G[HCOMM write facade]
  E -->|grpc| H[PushTensor chunks]
  F --> I[TensorRef.handle]
  G --> I
  H --> I
```

#### 6.4.2 output tensor staging

当前只有大 `OUTPUT / OUTPUT_EXISTING` 且 transport 是 `rxe` 或 `auto` 时，会走 L4 local RXE region：

```mermaid
flowchart TD
  A[OUTPUT/OUTPUT_EXISTING tensor] --> B{rxe/auto and large?}
  B -->|yes| C[L4 RxeRuntime.server_start(local output ptr)]
  C --> D[TensorRef.handle node_id=l4-rxe-*]
  B -->|no| E[普通 input-style staging / inline]
```

`INOUT` 没有走 local output RXE fast path。它仍然走 input staging，因为它需要把初始值发给 L3。

#### 6.4.3 dispatch 返回处理

```mermaid
flowchart TD
  A[DispatchResp.output_tensors] --> B[zip with local output tensor indexes]
  B --> C{is local RXE ACK?}
  C -->|yes| D[skip copy, remote already wrote local buffer]
  C -->|no| E[inline or PullTensor]
  E --> F[memmove into local output ptr]
  D --> G[free L3 input handles]
  F --> G
  G --> H[close local output RXE servers]
```

Review 关注点：

- `_stage_tensor_args()` 会对所有 large INPUT 分配远端 handle，并在异常时 best-effort free。
- local output RXE server lifetime 覆盖 dispatch 整个远端执行和写回。
- `auto` 模式下 RXE/HCOMM push 失败会回退 gRPC；显式 `rxe`/`hcomm` 失败会让远端不可用。
- `OUTPUT_EXISTING` 在当前设计中不会把旧值发送给 L3。如果用户依赖旧值，应使用 `INOUT`。
- `_write_output_tensors()` 依赖 output tensor 顺序与 L3 返回顺序一致。

### 6.5 L3Daemon 与 backend process

代码：

- `python/simpler/distributed/l3_daemon.py`

职责：

- 提供 gRPC service。
- 在 gRPC server 启动前 fork backend process。
- 通过 Pipe 把 Catalog/TensorPool/Dispatch 操作转发给 backend。
- backend process 拥有真实 `TensorPool` 和 `Worker(level=3)`。

进程结构：

```mermaid
flowchart LR
  subgraph Daemon[L3Daemon process]
    A[gRPC threads]
    B[_BackendTensorPoolService facade]
    C[_BackendCatalogService facade]
    D[L3Worker.Dispatch]
  end

  subgraph Backend[backend process]
    E[Catalog mirror]
    F[TensorPool]
    G[transport backend]
    H[Worker level=3]
  end

  A --> B
  A --> C
  A --> D
  B -->|Pipe op| F
  C -->|Pipe push| E
  D -->|Pipe dispatch| H
  F --> G
```

为什么需要 backend process：

- gRPC server 有线程。
- `Worker(level=3)` 内部还可能 fork sub/chip worker。
- 在有活跃 gRPC 线程的进程里 fork 风险较高。
- backend process 在 gRPC server 启动前创建，后续 Worker fork 发生在 backend 中。

#### 6.5.1 backend op

`_backend_loop()` 当前处理：

```text
stop
push
tensor_alloc
tensor_free
tensor_refresh
tensor_pull
tensor_push
dispatch
```

TensorPool gRPC facade 不直接持有 pool，只把请求 serialize 后通过 Pipe 发送给 backend。

#### 6.5.2 dispatch 执行策略

```mermaid
flowchart TD
  A[backend dispatch] --> B{req.tensor_refs?}
  B -->|no| C[decode_task_args tensor_args]
  C --> D{persistent inner exists?}
  D -->|no| E[create Worker level=3]
  D -->|yes| F[reuse inner]
  E --> G[run_inner.run]
  F --> G

  B -->|yes| H[decode tensor_refs into mmap buffers]
  H --> I[create ephemeral Worker level=3]
  I --> J[run_inner.run]
  J --> K[encode_output_tensor_refs]
  K --> L[close ephemeral Worker]
```

重点：有 tensor_refs 的 dispatch 使用 **ephemeral inner Worker**。原因是 tensor payload 被 materialize 到 backend process 的 mmap buffer，L3 sub/chip children 必须在这些 mmap 存在后 fork，才能继承同一映射。

Review 关注点：

- scalar-only dispatch 复用 persistent inner，提高普通控制面路径效率。
- tensor dispatch 每次创建/关闭 inner Worker，语义安全但性能较重。
- Pipe 同步调用由 `_backend_lock` 串行化，目前没有并发 dispatch 并行执行。
- backend 异常会被序列化为 `(False, traceback)`，daemon handler 转为 error resp 或 gRPC abort。

### 6.6 Tensor serialization

代码：

- `python/simpler/distributed/serialization.py`

职责：

- `CallConfig` encode/decode。
- 旧 `ContinuousTensorRef` decode。
- 新 `TensorRef` materialize。
- output tensor encode/writeback。

关键函数：

- `encode_task_args()`
- `decode_task_args()`
- `decode_task_args_with_tensor_refs_and_writebacks()`
- `encode_output_tensor_refs()`

#### 6.6.1 L3 materialize TensorRef

```mermaid
flowchart TD
  A[TensorRef] --> B{local RXE output handle?}
  B -->|yes| C[allocate empty mmap buffer sized by shape*dtype]
  B -->|no| D[pool.materialize_ref inline or L3 handle]
  C --> E[TaskArgs ContinuousTensor ptr=mmap addr]
  D --> E
  C --> F[record RemoteTensorWriteback]
```

`RemoteTensorWriteback` 记录 tensor index 和 L4 output handle，供执行后写回。

#### 6.6.2 output encoding

```mermaid
flowchart TD
  A[TaskArgs output tensor] --> B{has writeback?}
  B -->|yes| C[try RxeDataPlaneClient.write_handle to L4]
  C -->|success| D[return ACK TensorRef(handle=L4 handle)]
  C -->|failure| E[fallback pool.put_bytes]
  B -->|no| E
  E --> F[inline or L3 TensorPool handle]
```

Review 关注点：

- `OUTPUT / OUTPUT_EXISTING` 可作为 remote output writeback。
- `INOUT` 当前不在 `_REMOTE_OUTPUT_TAGS`，避免丢初始输入值。
- RXE writeback 失败被吞掉并 fallback 到 pool path；这保证语义，但可能隐藏性能路径失败，后续可以加日志。
- `_shape_nbytes()` 依赖 `get_element_size(dtype)`。

### 6.7 TensorPool

代码：

- `python/simpler/distributed/tensor_pool.py`

职责：

- 管理 backend process 内的 bytearray storage。
- 提供 handle/lease/capacity/GC。
- 暴露 TensorPool gRPC servicer 形状。
- 调用 `TensorTransportBackend` 注册/注销 region。

核心对象：

```text
TensorPool._entries[handle_id] = _Entry(
  data=bytearray,
  nbytes,
  expires_at_ms,
  shape,
  dtype,
  tag,
  region=RegisteredRegion(...)
)
```

生命周期：

```mermaid
flowchart TD
  A[AllocTensor] --> B[bytearray]
  B --> C[transport_backend.register_region]
  C --> D[TensorHandle]
  D --> E[PushTensor/RDMA write]
  E --> F[RefreshTensor]
  F --> G{backend has refresh_region?}
  G -->|yes RXE| H[close old one-shot server, start new server]
  G -->|no| I[lease only]
  D --> J[FreeTensor or GC]
  J --> K[transport_backend.unregister_region]
```

Review 关注点：

- capacity 与 lease 是否满足远端执行时间。
- `refresh_region()` 目前主要服务 RXE one-shot server 重建。
- `bytearray` 地址在 entry 生命周期内稳定；但 Python 对象生命周期必须由 `_Entry.data` 持有。
- gRPC `PushTensor` fallback 仍写同一个 pool buffer。

### 6.8 Transport backend

代码：

- `python/simpler/distributed/transport_backend.py`
- `python/simpler/distributed/rxe_verbs_helper.c`
- `python/simpler/distributed/hcomm_abi_shim.cc`

抽象：

```python
class TensorTransportBackend:
    def register_region(self, data: bytearray, *, tag: str) -> RegisteredRegion: ...
    def unregister_region(self, region: RegisteredRegion) -> None: ...
```

#### 6.8.1 gRPC backend

`GrpcTensorTransport` 只是返回本地 buffer 地址，实际传输走 `PushTensor/PullTensor`。

适合作为 fallback 和默认路径。

#### 6.8.2 RXE backend

Python 层：

- `RxeTensorTransport`
- `RxeRuntime`
- `RxeDataPlaneClient`

C 层：

- `simpler_rxe_server_start`
- `simpler_rxe_server_stop`
- `simpler_rxe_write`

RXE input 流程：

```mermaid
sequenceDiagram
  participant L3 as L3 TensorPool/RxeTensorTransport
  participant S as rxe server helper
  participant L4 as RxeDataPlaneClient
  participant C as rxe client helper

  L3->>S: server_start(buffer addr, size)
  S-->>L3: ip, port, rkey, addr, size
  L3-->>L4: TensorHandle transport_desc
  L4->>C: simpler_rxe_write(local addr, desc)
  C->>S: TCP connect, exchange QP info
  C->>S: IBV_WR_RDMA_WRITE
  C-->>L4: CQ success
  L4->>L3: RefreshTensor
  L3->>S: stop old server, start new server
```

RXE desc v2：

```text
magic="SRXE"
version=2
header_size
port
gid_index
rkey
addr
size
ip[64]
device[64]
```

Review 关注点：

- C helper 是 MVP：每个 region/write 还是临时 QP/TCP 控制连接，不是连接池。
- `server_stop()` 会 shutdown listen fd 和 accepted fd，避免失败路径挂住。
- `_build_rxe_verbs_helper()` 动态编译 C helper 到 `.cache`，依赖本机 rdma-core include/lib。
- desc parser 兼容旧 JSON desc。

#### 6.8.3 HCOMM backend

HCOMM 只做 Simpler 侧适配：

- 不修改 `3rd/hcomm` 源码。
- `HcommRuntime` 加载 `libhcomm.so` 和 CANN/HCOMM 依赖。
- `hcomm_abi_shim.cc` 只补 stock HCOMM 本地 build 可能缺的 ABI 符号。
- CPU RoCE channel E2E 在当前 910B1 host 环境不是主路径。

Review 关注点：

- HCOMM explicit 模式失败应清晰报错。
- `auto` 默认不会因为 HCOMM/RXE 不可用而破坏 gRPC。
- HCOMM source tree 不应被本项目提交依赖修改。

## 7. 端到端流程详解

### 7.1 初始化流程

```mermaid
sequenceDiagram
  participant User
  participant W4
  participant Catalog
  participant Proxy
  participant L3
  participant Backend
  participant Cpp

  User->>W4: register(l3_sub/l3_orch)
  W4->>Catalog: register callable payload
  User->>W4: add_remote_worker(endpoint)
  User->>W4: init()
  W4->>Proxy: create RemoteWorkerProxy
  User->>W4: run()
  W4->>Proxy: handshake()
  Proxy->>L3: Heartbeat
  Proxy->>L3: PushCallable for every payload
  L3->>Backend: pipe push catalog
  W4->>Cpp: add_next_level_process(remote mailbox)
```

### 7.2 scalar-only dispatch

```mermaid
sequenceDiagram
  participant Cpp as L4 scheduler
  participant Shim
  participant Proxy
  participant L3
  participant Backend
  participant Inner as persistent Worker(level=3)

  Cpp->>Shim: TASK_READY callable,args,cfg
  Shim->>Proxy: dispatch()
  Proxy->>L3: DispatchReq tensor_args/scalars
  L3->>Backend: pipe dispatch
  Backend->>Inner: create if missing
  Backend->>Inner: run(orch_fn,args,cfg)
  Inner-->>Backend: done
  Backend-->>L3: DispatchResp error_code=0
  L3-->>Proxy: DispatchResp
  Proxy-->>Shim: return
  Shim->>Cpp: TASK_DONE
```

### 7.3 large input with RXE

```mermaid
sequenceDiagram
  participant Proxy as L4 RemoteWorkerProxy
  participant TP as L3 TensorPool facade
  participant Backend as L3 backend TensorPool
  participant RxeS as L3 RXE server
  participant RxeC as L4 RXE client
  participant L3 as L3 Dispatch

  Proxy->>TP: AllocTensor(nbytes,shape,dtype,tag)
  TP->>Backend: pipe tensor_alloc
  Backend->>RxeS: register_region / server_start
  Backend-->>Proxy: TensorHandle transport=rxe
  Proxy->>RxeC: write_handle(handle, local bytes)
  RxeC->>RxeS: RDMA write
  Proxy->>TP: RefreshTensor(handle)
  TP->>Backend: pipe tensor_refresh
  Backend->>RxeS: refresh_region stop/start
  Proxy->>L3: DispatchReq TensorRef(handle)
```

### 7.4 large output with RXE writeback

```mermaid
sequenceDiagram
  participant Proxy as L4 RemoteWorkerProxy
  participant RxeS as L4 RXE server on output ptr
  participant L3 as L3 backend
  participant W as Worker(level=3)
  participant RxeC as L3 RXE client

  Proxy->>RxeS: server_start(local output ptr)
  Proxy->>L3: DispatchReq TensorRef(handle=node_id l4-rxe-*)
  L3->>L3: allocate mmap temp output buffer
  L3->>W: run with temp output tensor
  W-->>L3: output bytes in temp buffer
  L3->>RxeC: write_handle(L4 output handle)
  RxeC->>RxeS: RDMA write into L4 buffer
  L3-->>Proxy: DispatchResp TensorRef(handle ACK)
  Proxy->>Proxy: ACK recognized, skip PullTensor
  Proxy->>RxeS: close local output server
```

Fallback：

```mermaid
flowchart TD
  A[L3 RXE writeback fails] --> B[catch TransportBackendError/Unavailable]
  B --> C[pool.put_bytes output in L3 TensorPool]
  C --> D[DispatchResp TensorRef inline or L3 handle]
  D --> E[L4 _read_tensor_ref]
  E --> F[memmove into local output ptr]
```

## 8. Error handling and lifecycle

### 8.1 remote unavailable

- heartbeat RPC 失败：`RemoteUnavailable`。
- dispatch RPC 失败：mark proxy unavailable，free already allocated remote handles。
- remote response `error_code != 0`：free remote handles and close local output regions, raise `RuntimeError` with remote traceback.

### 8.2 tensor handle cleanup

L4 owns two kinds of temporary resources:

```text
remote_handles:
  L3 TensorPool allocated handles for large input or fallback output.
  cleanup: TensorPool.FreeTensor best-effort.

local_output_regions:
  L4 RXE servers opened on local output buffer.
  cleanup: RxeRuntime.server_stop best-effort.
```

L3 backend owns:

```text
TensorPool entries:
  freed by FreeTensor, GC, or TensorPool.close().

ephemeral Worker for tensor dispatch:
  closed in finally.

mmap keepalive buffers:
  kept alive until run and output encoding finish, then cleared.
```

### 8.3 fallback policy

```text
L4 -> L3 input:
  explicit rxe/hcomm failure: error
  auto mode rxe/hcomm failure: fallback to gRPC PushTensor

L3 -> L4 output:
  RXE writeback failure: fallback to L3 TensorPool / gRPC output response
```

Review 关注点：

- fallback 是否应该记录 warning/log。当前 output fallback 是静默的。
- explicit `rxe` input 失败不 fallback，这是为了避免用户以为走了真实 RDMA。
- cleanup 是 best-effort，失败不会覆盖原始 dispatch 错误。

## 9. 当前测试覆盖

主要测试：

```text
tests/ut/py/test_distributed/test_l4_l3_remote.py
  - scalar remote dispatch
  - inline tensor input
  - large handle tensor input
  - output tensor writeback
  - INOUT writeback
  - sub-worker tensor path
  - heartbeat behavior

tests/ut/py/test_distributed/test_tensor_pool.py
  - TensorPool alloc/free/refresh
  - inline/handle encode/decode
  - output TensorRef encode

tests/ut/py/test_distributed/test_transport_backend.py
  - HCOMM struct layout
  - RXE desc v2 roundtrip
  - RXE legacy JSON desc compatibility

tests/ut/py/test_distributed/test_real_e2e_smoke.py
  - real L4/L3 TensorPool handle E2E
  - real RXE ibverbs smoke
  - real L4/L3 RXE transport E2E
  - HCOMM endpoint/mem export smoke

tests/ut/py/test_distributed/test_rxe_real.py
  - ibv_rc_pingpong smoke
```

脚本：

```bash
tools/test_rxe_data_plane.sh
```

当前已验证结果：

```text
38 passed, 3 skipped
1 passed
2 passed, 2 deselected
RXE data-plane tests passed.
```

Benchmark：

```bash
PYTHONPATH=python tools/benchmark_rxe_data_plane.py \
  --sizes 8192,65536,1048576 \
  --repeats 10 \
  --warmup 2 \
  --transports grpc,rxe
```

## 10. 已知局限

1. **RXE helper 不是连接池**

   当前是 TCP 控制连接 + RC QP + MR 的 MVP，每个 region/write 成本较高。`RefreshTensor` 重建 server 解决可用性，不解决性能上限。

2. **tensor dispatch 使用 ephemeral inner Worker**

   这是为了让 sub/chip child fork 后继承 mmap tensor storage。语义安全，但性能重。要复用 persistent L3 worker，需要更强的跨进程 tensor 注入机制。

3. **INOUT 未做完整双向 RXE fast path**

   `INOUT` 需要先传初始值到 L3，再把结果写回 L4。当前走 input staging + response output 路径，不走 local output RXE ACK。

4. **Pipe backend 串行化**

   `_backend_lock` 让 L3 daemon 到 backend 的操作串行执行。当前简单可靠，但限制并发。

5. **Catalog 是受信任执行模型**

   `cloudpickle` payload 是可执行代码反序列化，只能用于受信任集群。

6. **output RXE fallback 静默**

   L3->L4 output RXE writeback 失败时会 fallback 到 TensorPool response，功能正确但不容易发现性能路径失效。

7. **跨节点和多并发覆盖不足**

   当前实机验证主要是单机 RXE。跨 host RoCE、多 remote worker、多并发 dispatch、长时间压测还需要补。

8. **HCOMM 不是当前主路径**

   HCOMM adapter 可加载 endpoint/mem，CPU RoCE channel E2E 在当前 910B1 host 环境不是主验证路径。

## 11. Review Checklist

### 控制面

- `Worker.add_remote_worker()` 的 API 语义是否和 `add_worker()` 一致。
- remote mailbox shim 是否正确模拟 PROCESS child worker。
- remote worker thread 与 C++ scheduler 的状态机是否有 race。
- heartbeat 失败后 `_available` 的语义是否符合调度期望。
- Catalog version/hash 是否足以避免 stale callable。

### 数据面

- inline threshold 的默认值和配置入口是否合理。
- L4 input staging 是否会错误读取 OUTPUT-only tensor 的旧值。
- `OUTPUT_EXISTING` 不发送旧值、`INOUT` 发送旧值的语义是否清晰。
- L3 output 顺序与 L4 output tensor index zip 是否可靠。
- TensorPool lease/GC 是否可能在长任务中提前释放 handle。

### RXE

- RXE desc v2 字段是否足够支持跨 host。
- RXE helper 的 direct struct/TCP control protocol 是否可接受为 MVP。
- `server_stop()` 在 client failure、timeout、FreeTensor 时是否不会 hang。
- refresh 重建 server 是否有重复写/并发写 race。
- 显式 `rxe` 失败是否应该 fallback，还是继续 fail-fast。

### L3 backend

- 有 tensor_refs 时创建 ephemeral Worker 的成本是否可接受。
- backend Pipe 串行是否满足当前需求。
- backend exception 到 `DispatchResp.remote_traceback` 的传播是否足够。
- TensorPool facade 把所有操作转发 backend 是否有死锁风险。

### 测试

- 是否需要把 example 加入自动测试。
- 是否需要新增多 remote worker 分发测试。
- 是否需要新增 RXE output fallback 强制失败测试。
- 是否需要跨 host RoCE smoke。
- 是否需要 benchmark 阈值或只保留观测输出。

## 12. 建议 reviewer 先看的文件顺序

1. `python/simpler/worker.py`

   先看 `add_remote_worker()`、`_remote_worker_loop()`、`_init_hierarchical()`、`_start_hierarchical()`，理解 remote worker 如何伪装成本地 PROCESS child。

2. `python/simpler/distributed/proto/dispatch.proto`

   看清楚 `DispatchReq`、`DispatchResp`、`TensorHandle`、`TensorRef`。

3. `python/simpler/distributed/remote_proxy.py`

   这是 L4 侧最核心逻辑，重点看 `_stage_tensor_args()`、`_push_remote_tensor_*()`、`_write_output_tensors()`。

4. `python/simpler/distributed/l3_daemon.py`

   看 daemon/backend process 分离、backend op、`_backend_dispatch()`。

5. `python/simpler/distributed/serialization.py`

   看 TensorRef 到 mmap buffer 的 materialize，以及 output writeback/fallback。

6. `python/simpler/distributed/tensor_pool.py`

   看 handle 生命周期和 refresh hook。

7. `python/simpler/distributed/transport_backend.py`

   看 RXE/HCOMM/gRPC backend 边界和 `build_tensor_transport()` 策略。

8. `python/simpler/distributed/rxe_verbs_helper.c`

   最后看真实 ibverbs MVP 是否符合你对数据面的预期。
