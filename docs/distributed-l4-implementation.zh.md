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
- `tensor_refs`: 为后续真实 tensor 数据面预留

`DispatchResp` 当前承载：

- `error_code`: `0` 表示成功
- `error_msg`: 远端失败摘要
- `remote_traceback`: 远端 Python traceback
- `output_tensors`: 为后续 output 回传预留

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
3. backend lazy 创建 inner `Worker(level=3)`。
4. backend 把 catalog 中所有 callable 安装进 inner worker 的 `_callable_registry`。
5. 查找 `req.callable_id / req.callable_version` 对应的 orch fn。
6. 反序列化 `TaskArgs` 和 `CallConfig`。
7. 调用：

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

- 小字节数据 inline
- 大字节数据注册为 handle
- `PullTensor` streaming
- `PushTensor` streaming
- `ContinuousTensor` 元数据随 `DispatchReq.tensor_args` 传输

尚未完成：

- 远端真实 tensor materialization
- output tensor 回写
- `OUTPUT_EXISTING` 的远端到本地同步
- 与 torch tensor / NPU device memory 的完整数据面打通

所以当前端到端 remote dispatch 测试主要覆盖 scalar `TaskArgs` 和 Python callable 执行链路。

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
32 passed
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
- remote traceback 传播
- heartbeat fail-fast
- 示例和测试

未完成：

- 完整 tensor 数据面
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
