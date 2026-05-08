# L4/L3 分布式控制面与 RXE 数据面实现说明

本文记录当前 Simpler L4 到 L3 分布式 dispatch 的实现状态，重点说明这次新增的真实 RXE/ibverbs 数据面、L4 控制面流程、TensorPool handle 语义、测试方式和已知局限。

当前实现遵循一个边界原则：**不修改 `3rd/hcomm` 源码**。HCOMM 只作为可选运行时能力被 Simpler 侧 shim/adapter 使用；RXE 数据面实现、构建逻辑、测试脚本都放在 `simpler/` 内。

## 这次整体修改

主要新增和修改的代码如下：

```text
python/simpler/distributed/
  transport_backend.py       # gRPC/HCOMM/RXE transport backend 抽象与运行时加载
  rxe_verbs_helper.c         # Simpler 自有 ibverbs RC RDMA write helper
  remote_proxy.py            # L4 侧 remote worker proxy，接入 RXE input/output 数据面
  l3_daemon.py               # L3 daemon，backend process 内创建 TensorPool transport
  serialization.py           # TensorRef 解码、output writeback、RXE fallback
  tensor_pool.py             # TensorPool refresh hook，支持 RXE region 重建

tests/ut/py/test_distributed/
  test_transport_backend.py  # RXE desc 编解码、backend 基础测试
  test_real_e2e_smoke.py     # 实机 L4->L3 RXE 数据面 E2E
  test_rxe_real.py           # 实机 ibv_rc_pingpong smoke

tools/
  test_rxe_data_plane.sh       # 一键测试脚本
  benchmark_rxe_data_plane.py  # gRPC vs RXE 端到端 benchmark
```

已有的 protobuf 消息没有新增字段。当前 `TensorHandle.transport` 和 `TensorHandle.transport_desc` 已足够承载不同 transport 的数据面描述。

## 总体架构

分布式路径分成控制面和数据面：

```text
L4 Worker
  -> 本地 PROCESS mailbox
  -> Python remote shim thread
  -> RemoteWorkerProxy
  -> gRPC control plane
  -> L3Daemon
  -> backend process
  -> Worker(level=3)

Tensor data plane:
  small tensor <= inline threshold
    -> DispatchReq.tensor_refs.inline_data

  large input tensor
    -> L3 TensorPool.AllocTensor
    -> L3 TensorPool returns TensorHandle(transport=rxe/grpc/hcomm)
    -> L4 writes payload into that handle
    -> DispatchReq only carries TensorRef(handle=...)

  large output tensor
    -> L4 registers local output buffer as RXE region
    -> DispatchReq carries TensorRef(handle=local-rxe-output)
    -> L3 runs task into temporary local buffer
    -> L3 writes output bytes back to L4 RXE handle
    -> DispatchResp returns ACK-style TensorRef(handle=local-rxe-output)
```

控制面负责调度、catalog、租约、错误传播和 handle 生命周期；数据面负责 tensor payload 的实际搬运。

## L4 控制面实现

L4 用户仍然通过原有接口使用远端 L3：

```python
w4 = Worker(level=4, num_sub_workers=0)
w4.add_remote_worker("127.0.0.1:5050", tensor_transport="rxe")
w4.init()
```

`Worker.add_remote_worker()` 会在 `Worker.init()` 时创建一个本地 mailbox 和一个 `RemoteWorkerProxy`。C++ scheduler 仍然看到的是一个普通 PROCESS-mode next-level worker，Python shim thread 负责把 mailbox 中的任务转换成 gRPC dispatch。

L4 dispatch 的关键步骤：

1. shim thread 从 mailbox 读出 callable id、`TaskArgs` 和 `CallConfig`。
2. `RemoteWorkerProxy` 把 callable catalog 预先推送到 L3 daemon。
3. 对每个 tensor 参数判断 inline、remote input handle 或 local output handle。
4. 通过 `L3Worker.Dispatch` 发出 `DispatchReq`。
5. 收到 `DispatchResp` 后，把 output 写回本地用户 buffer，释放临时 handle。

控制面的核心文件：

- `remote_proxy.py`
- `l3_daemon.py`
- `catalog.py`
- `rpc.py`
- `serialization.py`
- `proto/dispatch.proto`

## L3 控制面实现

`L3Daemon` 是远端 L3 节点入口。它不会直接在 gRPC handler 线程里运行 `Worker(level=3)`，而是启动一个 backend process：

```text
L3Daemon process
  - gRPC server
  - Catalog service
  - TensorPool control service facade
  - Pipe to backend process

Backend process
  - Catalog mirror
  - TensorPool
  - transport backend: grpc / rxe / hcomm / auto
  - lazy Worker(level=3)
```

这样做是为了避免 grpcio 线程和 `Worker(level=3)` 内部 fork sub/chip worker 发生冲突。TensorPool 的真实对象也在 backend process 内，因此它注册的 buffer 地址和实际执行任务的地址空间一致。

启动示例：

```bash
python -m simpler.distributed.l3_daemon --port 5050 --tensor-transport rxe
```

## 数据面抽象

数据面抽象定义在 `transport_backend.py`：

```python
class TensorTransportBackend:
    name = "grpc"
    def register_region(self, data: bytearray, *, tag: str) -> RegisteredRegion: ...
    def unregister_region(self, region: RegisteredRegion) -> None: ...
```

TensorPool 分配大 tensor 时，会创建 `bytearray` 并调用 backend 的 `register_region()`。返回的 `RegisteredRegion` 被编码到 `TensorHandle`：

```protobuf
message TensorHandle {
  string node_id = 1;
  uint64 handle_id = 2;
  uint64 remote_addr = 3;
  uint32 rkey = 4;
  uint64 nbytes = 5;
  uint64 lease_deadline_unix_ms = 6;
  string transport = 7;
  bytes transport_desc = 8;
}
```

当前 backend：

- `GrpcTensorTransport`：默认路径，payload 仍通过 `TensorPool.PushTensor/PullTensor` 的 gRPC chunk 传输。
- `HcommTensorTransport`：可选 HCOMM C API 适配层，只在 Simpler 内做 ABI shim/loader，不修改 HCOMM。
- `RxeTensorTransport`：真实 RXE/ibverbs 数据面。

`build_tensor_transport()` 支持：

```text
grpc
rxe
hcomm
auto
```

`auto` 默认保守，不自动启用 RXE。需要显式 `SIMPLER_RXE_AUTO=1` 才会在 auto 模式优先尝试 RXE。

## RXE 数据面实现

RXE 数据面由两层组成：

```text
Python:
  RxeTensorTransport
  RxeDataPlaneClient
  RxeRuntime

C helper:
  rxe_verbs_helper.c
```

### L3 侧注册 region

L3 TensorPool 分配大 input handle 时：

1. 创建 `bytearray(nbytes)`。
2. `RxeTensorTransport.register_region()` 获取 buffer 地址。
3. `RxeRuntime.server_start()` 调 C helper：
   - 打开 RXE device
   - 创建 PD/CQ/QP
   - 注册 MR
   - 启动 TCP 控制 server
   - 等待 L4 建立 RC QP 后接收一次 RDMA write
4. 返回 `TensorHandle(transport="rxe", transport_desc=...)`。

### L4 侧写 input

L4 收到 L3 分配的 RXE handle 后：

1. `RemoteWorkerProxy._push_remote_tensor_rxe()` 创建本地 source buffer。
2. `RxeDataPlaneClient.write_handle()` 解析 `transport_desc`。
3. `simpler_rxe_write()` 建立 TCP 控制连接，交换 QP 信息。
4. L4 发起 `IBV_WR_RDMA_WRITE`。
5. CQ completion 成功后，L4 调 `TensorPool.RefreshTensor`。

`RefreshTensor` 不只是续租。对 RXE backend，它还会调用 `refresh_region()`，关闭旧的一次性 server 并在同一 buffer 上重建新 server。因此同一个 TensorPool handle 后续仍可再次写入。

### L3 到 L4 output 写回

这次新增了 output 方向的真实数据面：

1. L4 遇到大 `OUTPUT / OUTPUT_EXISTING` tensor 时，不再先把旧内容推到 L3。
2. L4 直接把本地 output buffer 注册成 RXE region，生成本地 `TensorHandle(node_id="l4-rxe-...", transport="rxe")`。
3. `DispatchReq.tensor_refs` 携带这个 local output handle。
4. L3 `decode_task_args_with_tensor_refs_and_writebacks()` 识别该 handle：
   - 在 L3 backend process 内分配临时 mmap buffer 给 Worker 执行。
   - 记录 `RemoteTensorWriteback`。
5. L3 task 执行结束后，`encode_output_tensor_refs()` 用 `RxeDataPlaneClient` 把临时 output buffer RDMA write 回 L4 handle。
6. `DispatchResp.output_tensors` 返回同一个 handle 作为 ACK。
7. L4 看到 ACK 属于本地 output handle，就不再 PullTensor。

当前 output RXE writeback 覆盖：

- `TensorArgType.OUTPUT`
- `TensorArgType.OUTPUT_EXISTING`

`INOUT` 暂时仍走 input staging 路径，因为它同时需要把初始值送到 L3，再把结果写回 L4。这个双向语义还没有在单个 handle 上完全优化。

### RXE transport desc v2

旧版本 `transport_desc` 是 JSON。当前版本改为二进制头，减少解析歧义并为后续扩展留空间：

```text
magic       = "SRXE"
version     = 2
header_size
port
gid_index
rkey
addr
size
ip[64]
device[64]
```

解析逻辑仍兼容旧 JSON desc，便于已有测试和临时 handle 过渡。

## HCOMM 现状

HCOMM 相关改动只保留在 Simpler 侧：

- `hcomm_abi_shim.cc`
- `HcommRuntime`
- `HcommTensorTransport`
- `HcommDataPlaneClient`

当前机器上 stock HCOMM CPU RoCE channel 对 910B1 host 场景不满足能力要求，因此没有把 HCOMM channel E2E 作为主路径。RXE backend 是当前真实数据面 smoke/E2E 的主验证路径。

## 测试与验证

一键测试脚本：

```bash
cd /mnt/data/ntlab/zhouzhe/simpler_l4/simpler
tools/test_rxe_data_plane.sh
```

脚本执行：

1. Python 编译检查。
2. distributed 常规 UT。
3. RXE/ibverbs `ibv_rc_pingpong` smoke。
4. L4/L3 RXE tensor 数据面 E2E。

已验证结果：

```text
38 passed, 3 skipped
1 passed
2 passed, 2 deselected
RXE data-plane tests passed.
```

可选 benchmark：

```bash
SIMPLER_RUN_RXE_BENCHMARK=1 tools/test_rxe_data_plane.sh
```

也可以单独运行：

```bash
PYTHONPATH=python tools/benchmark_rxe_data_plane.py \
  --sizes 8192,65536,1048576 \
  --repeats 10 \
  --warmup 2 \
  --transports grpc,rxe
```

输出 CSV：

```text
transport,size_bytes,repeats,mean_ms,p50_ms,p95_ms,min_ms,max_ms
```

## 环境变量

常用配置：

```bash
export SIMPLER_TENSOR_TRANSPORT=rxe
export SIMPLER_RXE_DEVICE=rxe0
export SIMPLER_RXE_GID_INDEX=1
export SIMPLER_RXE_SERVER_IP=192.168.0.243
```

rdma-core 构建路径默认使用本机已验证路径：

```bash
export SIMPLER_RXE_INCLUDE_DIR=/home/ntlab/rdma-build/rdma-core-50.0/build/include
export SIMPLER_RXE_LIB_DIR=/home/ntlab/rdma-build/rdma-core-50.0/build/lib
```

如果不设置 `SIMPLER_RXE_DEVICE / SIMPLER_RXE_GID_INDEX / SIMPLER_RXE_SERVER_IP`，`RxeRuntime` 会尝试从 `/sys/class/infiniband/rxe*` 和 IPv4-mapped GID 自动推断。

## 当前局限性

1. RXE helper 仍是 MVP

   当前 C helper 是 RC QP + TCP 控制面的一次 write server。虽然 TensorPool refresh 会重建 server，使同一 handle 后续可以继续写，但它还不是长期连接池，也没有 QP 复用。

2. 性能不是最终形态

   当前 RXE 路径每个 region/write 都包含 TCP 控制连接、QP 创建、MR 注册等成本。小 tensor 或短迭代下可能比 gRPC 更慢。benchmark 用来观察趋势，不代表最终设计性能上限。

3. `INOUT` 还没有完整双向 RXE 优化

   `OUTPUT / OUTPUT_EXISTING` 大 tensor 已支持 L3->L4 RXE writeback。`INOUT` 仍按 input staging 处理，后续需要同时支持初始值 L4->L3 和结果 L3->L4。

4. transport desc 还不是 protobuf

   当前 v2 是 Simpler 自定义二进制头，旧 JSON 兼容。后续如果 desc 需要跨语言稳定演进，可以把 desc 单独 protobuf 化或纳入统一 metadata schema。

5. 错误恢复偏保守

   L4->L3 input 的显式 `rxe` 模式失败会报错；`auto` 模式才回退 gRPC。L3->L4 output 写回失败会退回 TensorPool/gRPC response 路径，以保证语义正确。

6. 当前主要验证是单机 RXE

   实机测试覆盖本机 RXE device 和 ibverbs RC pingpong。跨节点 RoCE、多 rank、多并发 worker、长时间压测还需要补测试。

7. 安全边界仍是受信任集群

   Catalog 使用 cloudpickle 传 callable payload，本质是可执行代码反序列化。Catalog/gRPC 服务不应暴露给不可信客户端。

## 后续建议

优先级较高的后续工作：

1. 把 RXE C helper 改成长连接或连接池，复用 PD/CQ/QP/MR。
2. 完成 `INOUT` 双向 RXE 数据面。
3. 增加多并发 dispatch 压测，覆盖 server refresh 和 FreeTensor 时序。
4. 把 `transport_desc` 迁移到稳定 schema。
5. 加入性能指标基线，持续对比 gRPC chunk 与 RXE write 的延迟和吞吐。
