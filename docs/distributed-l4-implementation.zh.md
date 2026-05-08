# L4 到 L3 跨 Host Dispatch 当前实现说明

本文记录当前 `simpler.distributed` 的实现状态。当前版本是 Python-first 的 MVP：使用 gRPC/protobuf 做跨进程或跨 Host 控制面传输，通过本地 mailbox shim 复用现有 C++ scheduler，不新增 C++ 或 nanobind 接口。

## 当前目标

原有单机 L4 到 L3 路径依赖 `Worker.add_worker(l3_worker)`、`os.fork()` 和共享内存 mailbox。跨 Host 后，L4 不能再直接 fork 出远端 L3，也不能依赖 fork 后继承的 callable registry 和共享地址空间。

当前实现把边界替换为：

```text
L4 Worker
  -> 本地 PROCESS mailbox
  -> Python shim thread
  -> gRPC Dispatch
  -> L3Daemon
  -> backend process
  -> inner Worker(level=3).run(...)
```

这使 L4 用户侧仍然通过 `orch.submit_next_level(...)` 下发任务，C++ scheduler 仍然看到一个普通的 PROCESS-mode next-level worker。

## 新增代码结构

```text
python/simpler/distributed/
  __init__.py
  rpc.py                 # RpcServer / RpcClient 薄封装
  catalog.py             # callable_id + version + cloudpickle payload
  serialization.py       # CallConfig / TaskArgs 与 protobuf 的转换
  remote_proxy.py        # L4 侧 RemoteWorkerProxy
  l3_daemon.py           # L3 侧长驻 daemon
  tensor_pool.py         # inline / handle 字节池表面
  proto/
    dispatch.proto
    dispatch_pb2.py
    dispatch_pb2_grpc.py
    _gen.sh

tests/ut/py/test_distributed/
  test_import.py
  test_rpc_roundtrip.py
  test_catalog.py
  test_l4_l3_remote.py
  test_tensor_pool.py
  test_heartbeat.py

examples/distributed/l4_l3_remote/
  l3_worker.py
  l4_master.py
  README.md
```

`pyproject.toml` 新增运行时依赖：

```toml
dependencies = ["grpcio>=1.80", "protobuf>=4.25", "cloudpickle>=2.2"]
```

`grpcio-tools>=1.80` 放在 test optional dependencies，用于重新生成 protobuf 文件。

## 协议设计

协议定义在 `python/simpler/distributed/proto/dispatch.proto`。

当前主要 service：

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

`DispatchReq` 当前承载：

- `task_id`: L4 侧生成的请求 id
- `callable_id`: L4 registry 中的 callable id
- `callable_version`: callable payload 的 blake2b 版本号
- `config_blob`: 序列化后的 `CallConfig`
- `scalar_args`: 标量参数
- `tensor_args`: `ContinuousTensor` 元数据
- `tensor_refs`: 当前数据面使用的 tensor 引用；小 tensor 直接 inline，大 tensor 使用 L3 `TensorPool` handle

`DispatchResp` 当前承载：

- `error_code`: `0` 表示成功
- `error_msg`: 远端失败摘要
- `remote_traceback`: 远端 Python traceback
- `output_tensors`: 当前用于 output 回传；L3 返回 `OUTPUT / INOUT / OUTPUT_EXISTING` tensor 的 inline bytes 或 handle

## L4 侧实现

入口是 `Worker.add_remote_worker(endpoint, **options)`。

调用时机要求和 `add_worker()` 一致：

- 只能在 `level >= 4` 的 Worker 上调用
- 必须在 `Worker.init()` 前调用
- 会复用 L4 侧已经 `register()` 的 callable registry

初始化时做的事：

1. 为每个 remote worker 分配一个本地 `SharedMemory` mailbox。
2. 创建 `RemoteWorkerProxy(endpoint, catalog, **options)`。
3. `RemoteWorkerProxy.handshake()`：
   - 先发 `Heartbeat`
   - 把本地 catalog 里的 callable payload 全部 `PushCallable` 到远端
   - 启动后台 heartbeat thread
4. 启动 `_remote_worker_loop` shim thread。
5. 把 remote mailbox 注册给 C++ `_Worker.add_next_level_process(...)`。

之后 C++ scheduler 下发任务时，只是在本地 mailbox 写入 `TASK_READY`。shim thread 读出：

- callable id
- `TaskArgs`
- `CallConfig`

然后调用：

```python
RemoteWorkerProxy.dispatch(callable_id, args, cfg)
```

如果 `TaskArgs` 里包含 `ContinuousTensor`，L4 侧不再把 `tensor.data` 裸地址发给远端。当前策略是：

1. 读取本地 tensor 指针和 `tensor.nbytes()`，拷贝出一份 bytes。`OUTPUT / OUTPUT_EXISTING` 只需要远端写入空间，当前按同样路径 staging。
2. `nbytes <= 4KB` 时直接放进 `DispatchReq.tensor_refs.inline_data`。
3. `nbytes > 4KB` 时先通过 `TensorPool.AllocTensor` 在 L3 backend pool 分配 handle。
4. L4 通过 `TensorPool.PushTensor` 分片把 bytes 写入该 handle。
5. `DispatchReq.tensor_refs` 只携带 handle、shape、dtype、tag。
6. `Dispatch` 成功返回后，L4 读取 `DispatchResp.output_tensors`，按本地 output tensor 顺序写回原始 buffer。
7. L4 best-effort 调 `TensorPool.FreeTensor` 释放 input/output handle。

为支持 remote mailbox 恢复 tensor tags，C++ PROCESS mailbox 在旧的 `[T][S][tensors][scalars]` 后追加了向后兼容的 tags 扩展：

```text
uint32 magic = "SL4T"
int32  tags[T]
```

C++ `read_blob` 仍按旧格式读取并忽略尾部；Python `_read_args_from_mailbox()` 识别该扩展并恢复 `TensorArgType`。没有这个扩展时，远端无法区分 `INPUT / OUTPUT / INOUT`。

远端返回成功后，shim thread 把 mailbox 状态写回 `TASK_DONE`。如果远端失败，则把错误写入 mailbox error 区域，后续由现有 drain/error propagation 路径抛回 L4 调用者。

## Callable Catalog

`Catalog` 解决 fork-COW registry 在跨 Host 场景不可用的问题。

注册逻辑：

```python
cid, version = catalog.register(fn, callable_id=cid)
```

版本号计算方式：

```text
version = uint64(blake2b(cloudpickle.dumps(fn), digest_size=8))
```

当前使用 `cloudpickle` 而不是标准库 `pickle`，原因是现有 L4/L3 测试和用户代码经常使用嵌套函数、lambda、closure。标准库 `pickle` 无法覆盖这些形态。

安全边界：

callable payload 是可执行 Python 代码的反序列化结果，只能用于受信任集群内部。不要把 `Catalog` service 暴露给不可信客户端。

## L3Daemon 实现

`L3Daemon` 是远端 L3 节点的常驻入口。

启动方式：

```bash
python -m simpler.distributed.l3_daemon --port 5050 --num-sub-workers 1
```

重要实现点：daemon 不是直接在 gRPC handler thread 中运行 `Worker`。它会先启动一个 backend process：

```text
L3Daemon process
  - gRPC server threads
  - Catalog service
  - L3Worker service
  - Pipe to backend

Backend process
  - Catalog mirror
  - lazy inner Worker(level=3)
  - inner Worker 的 sub/chip fork
```

这样做是为了避开 grpcio 与 fork 的冲突。`Worker(level=3)` 内部仍会 fork sub worker 或 chip worker；如果这个 fork 发生在已经启动 gRPC worker threads 的进程中，grpcio 可能触发 fork-safety 问题。当前设计让 backend process 在 gRPC server 启动前 fork 出来，后续所有 inner Worker 逻辑都在 backend process 内完成。

Catalog push 时：

1. gRPC `Catalog.PushCallable` 先安装到 daemon 进程的 catalog。
2. 同步转发 `("push", cid, version, payload)` 到 backend process。
3. backend process 安装到自己的 catalog mirror。

Dispatch 时：

1. gRPC handler 收到 `DispatchReq`。
2. 将 protobuf bytes 通过 pipe 发给 backend。
3. 查找 `req.callable_id / req.callable_version` 对应的 orch fn。
4. 反序列化 `TaskArgs` 和 `CallConfig`。
5. 无 tensor 的 scalar dispatch 使用持久 inner `Worker(level=3)`。
6. 带 `tensor_refs` 的 dispatch 使用临时 inner `Worker(level=3)`：先把 tensor materialize 到 shared mmap，再 init/fork L3 sub/chip 子进程，保证子进程继承同一段映射。
7. backend 把 catalog 中所有 callable 安装进 inner worker 的 `_callable_registry`。
8. 调用：

```python
inner.run(orch_fn, args, cfg)
```

## Heartbeat 与错误传播

`RemoteWorkerProxy` 在 handshake 后启动 heartbeat thread。

默认参数：

- `heartbeat_interval=5.0`
- `heartbeat_timeout=1.0`
- `heartbeat_failures=3`

连续失败达到阈值后，proxy 标记为 unavailable。后续 dispatch 会 fast-fail，不再进入 RPC 热路径。

错误传播路径：

```text
backend exception
  -> DispatchResp(error_code=1, error_msg, remote_traceback)
  -> RemoteWorkerProxy.dispatch raises RuntimeError
  -> _remote_worker_loop writes mailbox error
  -> existing Worker.run drain path raises to caller
```

## Tensor 数据面当前状态

当前已经有 `tensor_pool.py` 和 proto 中的 `TensorRef / TensorHandle / TensorChunk`。

已实现：

- 小 tensor inline，默认阈值 4KB
- L3 backend process 内的 Python `TensorPool`
- `TensorPool` 的 transport backend 抽象，默认后端是 `grpc`
- `AllocTensor / FreeTensor / RefreshTensor`
- handle lease、TTL、过期 GC、pool 容量限制
- `PullTensor` streaming
- `PushTensor` streaming
- L4 `RemoteWorkerProxy` 的 tensor staging：本地指针读取 bytes，小 tensor inline，大 tensor remote alloc + push + handle
- 可选 HCOMM backend：L3 pool 注册 byte buffer，handle 中带 `transport="hcomm"` 和 `transport_desc`
- L4 收到 HCOMM handle 时，可用 `HcommWriteWithNotifyNbi` + `HcommChannelFence` 推 input tensor
- L3 backend dispatch 时把 `TensorRef` materialize 成 shared mmap，并构造 `ContinuousTensor`
- 带 tensor 的 L3 dispatch 使用 per-dispatch 临时 inner worker，使 L3 sub/chip fork 后继承 tensor mmap
- `OUTPUT / INOUT / OUTPUT_EXISTING` output tensor 回传到 L4 原始 buffer
- scalar args 与 tensor refs 混合传输

尚未完成：

- 完整 RDMA/Urma/SHM 零拷贝 transport；当前默认数据面仍通过 gRPC streaming，HCOMM 只覆盖 input push 的第一版接线
- 持久 L3 worker 复用场景下的动态 tensor mmap 注入；当前带 tensor 的 dispatch 为保证 fork 继承映射，会使用临时 inner worker
- 与 torch tensor / NPU device memory 的完整数据面打通
- output tensor 的 HCOMM 写回协议；当前 output 仍走 gRPC `PullTensor`
- pool 中默认 `grpc` 后端的 `remote_addr/rkey` 仍是协议占位：`remote_addr` 是 Python bytearray buffer 地址，`rkey=0`

## HCOMM backend 当前边界

当前没有直接依赖 HCOMM 内部 `HostCpuRoceChannel` 类，而是通过公开/实验 C API 做一层 Python facade：

- L3 侧：`HcommMemReg` 注册 `TensorPool` 的 byte buffer，`HcommMemExport` 导出内存描述。
- proto：`TensorHandle` 新增 `transport` 和 `transport_desc`，避免 HCOMM 内存描述丢失。
- L4 侧：当 `RemoteWorkerProxy` 配置 `tensor_transport="hcomm"` 或 `auto`，且远端 handle 标记为 `hcomm` 时，优先用 `HcommMemImport` 导入 `transport_desc`，把本地源数据拷贝到 HCOMM 注册过的 host staging buffer，再调用 `HcommWriteWithNotifyNbi` 和 `HcommChannelFence`。
- endpoint：可以直接传 `SIMPLER_HCOMM_ENDPOINT_HANDLE`，也可以用 `SIMPLER_HCOMM_ENDPOINT_IP` 和可选 location 字段自动创建。
- channel：可以直接传 `SIMPLER_HCOMM_CHANNEL_HANDLE`；也可以基于最新 public `HcommChannelCreate` ABI 自动创建 CPU RoCE channel。`SIMPLER_HCOMM_CHANNEL_ROLE` 选择 `client`/`server`，`SIMPLER_HCOMM_CHANNEL_PORT` 选择 listen/connect 端口；`SIMPLER_HCOMM_SOCKET_HANDLE` 仍可选传入，但不再是 Python facade 自动建 channel 的必要条件。
- `auto` 模式：HCOMM 必要资源不存在或写入失败时自动回落到 gRPC。
- 显式 `hcomm` 模式：缺库或缺必要资源会报错，不静默假装走 RDMA。

启用方式：

```bash
SIMPLER_TENSOR_TRANSPORT=hcomm \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=192.168.0.243 \
SIMPLER_HCOMM_CHANNEL_ROLE=client \
SIMPLER_HCOMM_CHANNEL_PORT=60001 \
python -m simpler.distributed.l3_daemon --port 5050 --tensor-transport hcomm
```

或者先用 `auto` 做兼容运行：

```bash
SIMPLER_TENSOR_TRANSPORT=auto python -m simpler.distributed.l3_daemon --port 5050 --tensor-transport auto
```

注意：

- 当前 facade 只依赖 public `include/hcomm_res.h`/`include/hcomm_primitives.h` 中的 endpoint、memory、channel create/destroy 和 write/fence 接口；不会直接包含内部 `api_c_adpt/hcomm_c_adpt.h`。
- 最新 `HcommChannelCreate` 的 CPU RoCE 路径可以在 socket 为空时按 `role`/`port` 自建连接。server 侧用 `exchangeAllMems=true` 触发 listen，client 侧注册 staging mem 后连接并写远端导入内存。
- `transport_desc` 已随 handle 传输，L4 input push 已可通过 `HcommMemImport` 解析远端内存；缺 endpoint handle 时会退回使用 handle 中的 `remote_addr`。
- 如果使用外部预建 channel，调用方需要保证该 channel 的本地 MR 覆盖 L4 发送 staging buffer；自动创建 channel 时会使用 client 自己注册的 staging mem handle。
- output tensor 仍走 gRPC pull，因为当前 CPU RoCE 公开路径还缺少和 Simpler output 语义匹配的读回或远端写回协议。

真实 HCOMM smoke test 默认跳过，需要显式打开：

```bash
SIMPLER_HCOMM_REAL_TEST=1 \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=127.0.0.1 \
python -m pytest tests/ut/py/test_distributed/test_transport_backend.py -q
```

如果要跑真实 `WriteWithNotify + Fence`，还需要外部预建 channel，并设置：
`SIMPLER_HCOMM_ENDPOINT_HANDLE`、`SIMPLER_HCOMM_CHANNEL_HANDLE`、`SIMPLER_HCOMM_REMOTE_ADDR`、`SIMPLER_HCOMM_REMOTE_NBYTES`。

单机 HCOMM E2E smoke test 会创建 server/client 两个进程，通过最新 channel desc
结构自动建 CPU RoCE channel，然后 client 使用 `HcommWriteWithNotifyNbi` 写 server
注册的 host buffer。该测试默认跳过，需要显式打开，并要求已经构建好
`libhcomm.so`：

```bash
SIMPLER_HCOMM_E2E_REAL_TEST=1 \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=192.168.0.243 \
SIMPLER_HCOMM_CHANNEL_PORT=60001 \
python -m pytest tests/ut/py/test_distributed/test_hcomm_e2e_real.py -q
```

RXE/Soft-RoCE 可以作为更底层的实机 smoke test，用来确认本机 ibverbs/RoCE
数据通路能不能真的跑起来。它验证的是 `rxe*` 设备、GID、RC QP 和 verbs
读写握手，不等价于 HCOMM 端到端 channel 测试；HCOMM 仍然需要自己的 endpoint、
channel 资源和 shared library。

当前机器上 `ibv_devices` 能看到：

- `rxe0`：绑定 `192.168.0.243`，GID index 1 为 `::ffff:192.168.0.243`
- `rxe_lo`：绑定 localhost

已固化一个默认跳过的 pytest：

```bash
SIMPLER_RXE_REAL_TEST=1 \
SIMPLER_RXE_DEVICE=rxe0 \
SIMPLER_RXE_GID_INDEX=1 \
SIMPLER_RXE_SERVER_IP=192.168.0.243 \
python -m pytest tests/ut/py/test_distributed/test_rxe_real.py -q
```

在当前开发机上该测试实际启动 server/client 两个 `ibv_rc_pingpong` 进程并通过。
如果不设置 `SIMPLER_RXE_GID_INDEX` 和 `SIMPLER_RXE_SERVER_IP`，测试会尝试从
`/sys/class/infiniband/<device>/ports/1/gids` 自动找第一个 IPv4-mapped GID。

所以当前端到端 remote dispatch 测试覆盖两类路径：

- scalar `TaskArgs` 和 Python callable 执行链路
- L4 到 L3 backend/orch fn 的 tensor input/output 数据面，包括 inline 小 tensor、handle 大 tensor、INOUT 写回
- L4 到 L3 sub worker 的 tensor input/output 路径

## 使用示例

终端 1 启动 L3：

```bash
python examples/distributed/l4_l3_remote/l3_worker.py --port 5050
```

终端 2 启动 L4：

```bash
python examples/distributed/l4_l3_remote/l4_master.py --remotes 127.0.0.1:5050
```

期望输出：

```text
remote counter=7
```

## 当前测试方式

安装或构建：

```bash
python -m pip install -e .
```

运行新增分布式测试和原有 L4 递归回归：

```bash
python -m pytest tests/ut/py/test_distributed tests/ut/py/test_worker/test_l4_recursive.py -q
```

当前验证结果：

```text
44 passed, 5 skipped, 1 warning
```

额外检查：

```bash
python -m compileall -q python/simpler/distributed tests/ut/py/test_distributed examples/distributed/l4_l3_remote
git diff --check
```

当前注意事项：

- Python 3.13 下测试会出现多线程进程中 fork 的 `DeprecationWarning`。
- 现有本地 L4 recursive 测试也会触发同类 warning。
- 当前测试通过；warning 不代表断言失败。

## 当前实现边界

已完成：

- gRPC/protobuf 控制面
- callable catalog push/install
- L4 `add_remote_worker()`
- mailbox shim thread
- L3 daemon backend process
- scalar args dispatch
- input tensor data plane MVP（L4 到 L3 backend/orch fn 和 L3 sub）
- HCOMM input push 的可选后端接线和 fake 单测覆盖
- output tensor 自动回传（`OUTPUT / INOUT / OUTPUT_EXISTING` 写回 L4 原始 buffer）
- 带 tensor dispatch 的 shared mmap materialization 和 per-dispatch inner worker
- remote traceback 传播
- heartbeat fail-fast
- 示例和测试

未完成：

- 完整 RDMA/Urma/SHM 零拷贝 tensor transport
- HCOMM socket/channel 生命周期完全自动创建与跨节点握手
- HCOMM output tensor 写回路径
- 持久 L3 worker 中向已 fork 子进程动态注入新 tensor mapping
- 多 remote 负载均衡策略
- 节点发现或服务注册
- 鉴权、TLS、租户隔离
- C++ hot-path `RemoteWorkerThread`
- Urma/RDMA 数据面

## 设计取舍

当前实现优先保证低侵入：

- 不改 C++ scheduler
- 不改 nanobind binding
- 不改变用户 orch function 的写法
- 通过本地 mailbox shim 接入现有 PROCESS-mode 语义

代价是：

- 每个 remote worker 多一个 Python shim thread
- gRPC 路径不是 hot-path 最优
- tensor 数据面还没有真正接入
- callable 反序列化要求可信环境

这个版本适合作为 L4 到 L3 跨 Host dispatch 的功能 MVP，用于继续验证协议、调度语义和错误传播；后续性能优化可以再下沉到 C++ 或替换数据面。
