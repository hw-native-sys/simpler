# L3/L2 Multi-Channel MPMC Send/Receive 设计

本文描述在 simpler 的 L3 host CPU 与 L2 NPU runtime 之间增加 `send/recv`
语义的设计。目标是对上提供 MPMC（multi-producer, multi-consumer）消息接口，
内部用多条 SPSC lane 组合实现，避免在 CPU/NPU 共享边界上直接依赖跨端 CAS
MPMC 队列。

## 背景

当前 L3 到 L2 的主路径是同步 task dispatch：

```text
L3 Worker / Scheduler
  -> ChipWorker.run(callable, TaskArgs, CallConfig)
  -> host runtime 自动 H2D
  -> L2 AICPU/AICore 执行
  -> host runtime 自动 D2H
```

这个模型适合一次性 kernel 调度，但不适合下列场景：

- L3 CPU 在 L2 kernel 运行期间持续发送小消息、token、控制事件。
- L2 AICPU/AICore 持续向 L3 CPU 返回完成事件、采样结果、调试记录。
- 多个 CPU 线程和多个 L2 producer/consumer 同时使用同一逻辑通道。

已有机制可以复用：

- `ContinuousTensor.child_memory` 支持把 L3 预先 bootstrap 出来的 device pointer
  直接传给 L2，绕过自动 H2D/D2H。
- `tensor-dump` 在 a2a3 上已有 host/device 共享内存、SPSC free/ready queue、
  background polling thread 的实现经验。
- `tensormap_and_ringbuffer` runtime 内部已有 Vyukov MPMC ready queue，但它运行在
  device runtime 内部，不直接解决 CPU/NPU 共享原子可见性问题。

## 目标

对用户和上层 runtime 暴露：

```cpp
Status send(ChannelHandle ch, const MessageView &msg);
Status recv(ChannelHandle ch, MessageView *out, RecvOptions opts);
```

语义上支持：

- 多个 CPU producer 同时向 L2 发送。
- 多个 L2 producer 同时向 CPU 发送。
- 多个 CPU consumer 或 L2 consumer 从逻辑 channel 接收。
- 有界队列和反压，不允许无界堆积。
- 支持小消息 inline 和大 payload handle。
- 支持 sim、a2a3、a5 三类 backend，允许不同性能等级。

非目标：

- 第一版不实现跨 CPU/NPU 的真 lock-free MPMC CAS ring。
- 第一版不承诺跨 host 通信；跨 host 仍走 L4/L3 distributed transport。
- 第一版不替代现有 `ChipWorker.run()` task dispatch，只提供并行数据/消息通道。

## 核心决策

### 用 multi-lane SPSC 实现 MPMC 语义

跨 CPU/NPU 共享内存边界直接实现 MPMC，需要两端对 CAS、acquire/release、
cache flush/invalidate 有完全一致的语义。这个风险较高，尤其在真实硬件上容易
表现为偶发乱序、重复消费或死锁。

因此设计采用：

```text
CPU -> L2 方向

CPU producer 0 ── SPSC tx lane 0 ─┐
CPU producer 1 ── SPSC tx lane 1 ─┼── L2 broker poll/merge ──► L2 dispatch queue
CPU producer N ── SPSC tx lane N ─┘

L2 -> CPU 方向

L2 producer 0 ── SPSC rx lane 0 ─┐
L2 producer 1 ── SPSC rx lane 1 ─┼── CPU broker poll/merge ──► CPU consumer queue
L2 producer M ── SPSC rx lane M ─┘
```

每条 lane 保持单 producer、单 consumer：

- `head` 只由 consumer 写。
- `tail` 只由 producer 写。
- descriptor slot 只在 producer 发布后由 consumer 读取。
- 不需要跨 CPU/NPU CAS。

上层看到的是 MPMC channel；内部通过 producer 到 lane 的绑定、broker 合并、
consumer demux 实现并发语义。

## 总体架构

```text
L3 host process
  ├─ Channel API
  ├─ CPU producer threads
  ├─ CPU broker thread
  │    - poll L2->CPU lanes
  │    - demux to local consumers / callbacks
  └─ ChipWorker / DeviceContext
       │
       │ open_channel()
       ▼
Device memory region
  ├─ ChannelHeader
  ├─ CPU_TO_L2 LaneGroup
  │    ├─ lane[0] descriptors
  │    ├─ lane[1] descriptors
  │    └─ ...
  ├─ L2_TO_CPU LaneGroup
  │    ├─ lane[0] descriptors
  │    ├─ lane[1] descriptors
  │    └─ ...
  └─ PayloadPool
       - fixed-size slabs or ring arena

L2 runtime
  ├─ AICPU channel broker
  │    - poll CPU_TO_L2 lanes
  │    - route messages to L2 consumers
  │    - collect L2 producer output
  └─ AICPU/AICore tasks
```

## Channel 内存布局

所有跨端可见结构使用固定大小 POD，按 cacheline 对齐。建议第一版把 control
region 与 payload region 分开，便于后续使用不同映射策略。

```cpp
constexpr uint32_t HDCH_MAGIC = 0x48444348; // "HDCH"
constexpr uint32_t HDCH_VERSION = 1;

struct alignas(64) HostDeviceChannelHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t lane_count_cpu_to_l2;
    uint32_t lane_count_l2_to_cpu;
    uint32_t lane_depth;
    uint64_t control_bytes;
    uint64_t payload_base;
    uint64_t payload_bytes;
    uint64_t fatal_status;
    uint8_t reserved[64 - 48];
};

struct alignas(64) HostDeviceLane {
    volatile uint32_t head;      // consumer writes
    volatile uint32_t tail;      // producer writes
    uint32_t depth_mask;         // depth must be power-of-two
    uint32_t flags;
    uint64_t dropped_count;
    uint64_t blocked_count;
    uint8_t reserved[64 - 32];
    HostDeviceDesc slots[];
};

struct alignas(64) HostDeviceDesc {
    uint32_t opcode;
    uint32_t flags;
    uint64_t seq;
    uint64_t correlation_id;
    uint64_t payload_addr;
    uint32_t payload_bytes;
    uint16_t inline_bytes;
    uint16_t route;
    uint8_t inline_data[32];
};
```

`HostDeviceDesc` 的 slot 大小固定为 64B，保证 producer 可以先写完整 descriptor，
最后发布 `tail`。大 payload 放在 `PayloadPool`，descriptor 只携带 device address
和长度。

## API 草案

### C runtime ABI

在 `src/common/worker/pto_runtime_c_api.h` 后续增加一组可选符号。为了不破坏现有
runtime，第一阶段可以用 `dlsym` optional probing；所有平台实现后再改为强制符号。

```c
typedef void *HostDeviceChannelHandle;

typedef struct {
    uint32_t lane_count_cpu_to_l2;
    uint32_t lane_count_l2_to_cpu;
    uint32_t lane_depth;
    uint64_t payload_bytes;
    uint32_t flags;
} HostDeviceChannelConfig;

HostDeviceChannelHandle open_host_device_channel_ctx(
    DeviceContextHandle ctx,
    const HostDeviceChannelConfig *cfg);

int close_host_device_channel_ctx(
    DeviceContextHandle ctx,
    HostDeviceChannelHandle ch);

int host_device_send_ctx(
    DeviceContextHandle ctx,
    HostDeviceChannelHandle ch,
    uint32_t route,
    const void *data,
    size_t nbytes,
    uint64_t correlation_id);

int host_device_recv_ctx(
    DeviceContextHandle ctx,
    HostDeviceChannelHandle ch,
    void *dst,
    size_t dst_capacity,
    size_t *out_nbytes,
    uint64_t *out_correlation_id,
    uint32_t timeout_us);
```

### Python API

```python
ch = worker.open_channel(
    cpu_to_l2_lanes=8,
    l2_to_cpu_lanes=8,
    lane_depth=1024,
    payload_bytes=64 << 20,
)

ch.send(route=0, data=b"...", correlation_id=req_id)
msg = ch.recv(timeout_us=1000)
ch.close()
```

### Device API

Device-side API 先只放在 AICPU runtime。AICore 直接参与 channel 需要额外处理
AICore 对 GM 写入、completion ordering 和 cache，可作为第二阶段。

```cpp
bool hdch_try_recv(HostDeviceChannel *ch, uint32_t route, HostDeviceDesc *out);
bool hdch_try_send(HostDeviceChannel *ch, uint32_t route, const void *data, size_t nbytes);
void hdch_poll_once(HostDeviceChannel *ch);
```

## Lane 分配策略

### CPU producer 到 lane

默认使用 thread-local lane 绑定：

```text
lane_id = hash(thread_id) % lane_count_cpu_to_l2
```

如果该 lane 满，策略由 channel 配置决定：

- `STRICT`: 阻塞或返回 `WOULD_BLOCK`，保持单 producer lane 语义。
- `STEAL`: 尝试占用备用 lane，但必须通过 CPU 侧 mutex/lease 确保该备用 lane
  在同一时刻仍是单 producer。
- `BROKERED`: 所有 CPU producer 先写本地 MPSC 队列，由 CPU broker 单线程写
  CPU_TO_L2 lanes。这个模式延迟多一次 hop，但最容易保证正确性。

第一版建议实现 `STRICT` 和 `BROKERED`，暂缓 `STEAL`。

### L2 producer 到 lane

L2 producer 可以按 AICPU thread、core group 或 route 绑定 lane：

```text
lane_id = producer_id % lane_count_l2_to_cpu
```

如果多个 L2 producer 可能写同一 lane，必须经过 L2 broker，由 broker 单线程写
L2_TO_CPU lane。不能让多个 AICPU/AICore 同时直接写同一 SPSC lane。

### Consumer demux

descriptor 的 `route` 字段用于 demux：

- `route=0`: 默认控制消息。
- `route=1..N`: 上层可注册 callback 或 consumer queue。
- `correlation_id`: 用于 request/response 匹配。

CPU broker 负责从所有 L2_TO_CPU lanes round-robin poll，然后投递到本地 consumer
队列。L2 broker 对 CPU_TO_L2 lanes 做同样处理。

## 消息和 payload

小消息 inline：

```text
nbytes <= 32
  -> desc.inline_data
  -> desc.inline_bytes = nbytes
  -> desc.payload_addr = 0
```

大消息走 payload pool：

```text
nbytes > 32
  -> producer 从本端可写 payload pool 分配 slab
  -> 写 payload
  -> desc.payload_addr = device addr
  -> desc.payload_bytes = nbytes
  -> consumer 处理后释放或回 ACK
```

Payload pool 第一版建议用固定大小 class：

- 256B
- 4KB
- 64KB
- large external handle

这样可以避免跨端可见的复杂变长 allocator。large payload 可以复用
`ContinuousTensor.child_memory`，只在 descriptor 中传 device pointer 和长度。

## 内存一致性

### SPSC 发布顺序

Producer 写入一条消息：

```text
1. 等待 slot 可用：next_tail != observed_head
2. 写 payload
3. payload flush / DMA completion fence
4. 写 descriptor slot
5. descriptor flush / release fence
6. 写 tail
7. tail flush / doorbell optional
```

Consumer 读取一条消息：

```text
1. invalidate / acquire load tail
2. 如果 head == tail，队列空
3. invalidate descriptor slot
4. 读取 descriptor
5. invalidate/read payload
6. 处理消息
7. release/free payload
8. 写 head
9. head flush
```

### 平台差异

sim：

- 共享内存就是 host memory。
- 使用 C++ atomics 或 volatile + thread fence 即可。

a2a3：

- 优先参考 `tensor-dump` 的 host-register/shared-memory 路径。
- AICPU 写 host 可见区域后，需要 `cache_flush_range()`。
- AICPU 读 host 更新区域前，需要 `cache_invalidate_range()`。
- CPU 写 device memory 后，需要保证 `rtMemcpy` 或 host mapping write 已完成，再发布 `tail`。

a5：

- 当前 a5 缺少 a2a3 这种 host-register 路径时，第一版可退化为 memcpy batch/poll。
- 语义可先跑通，性能不作为目标。

## 反压与公平性

每条 lane 是有界 ring：

- 满时 `send()` 返回 `WOULD_BLOCK` 或按配置阻塞。
- 空时 `recv()` 返回 `WOULD_BLOCK` 或等待到 timeout。
- 维护 `blocked_count`、`dropped_count`、`max_occupancy` 等统计。

Broker poll 策略：

- 默认 round-robin，避免单 lane 饥饿。
- 可选 weighted round-robin，用于控制消息优先于数据消息。
- 每次 poll 最多处理 `budget` 条，避免 broker 长时间占用 AICPU。

## 错误处理

错误码建议：

| 错误 | 含义 |
| --- | --- |
| `OK` | 成功 |
| `WOULD_BLOCK` | lane 满或空，非阻塞调用未完成 |
| `TIMEOUT` | 等待超时 |
| `MSG_TOO_LARGE` | payload 超过 channel 配置 |
| `BAD_ROUTE` | route 未注册 |
| `CHANNEL_CLOSED` | 对端已关闭 |
| `FATAL_REMOTE` | 对端写入 fatal 状态 |
| `CORRUPT_DESC` | magic/version/seq 校验失败 |

Channel close 使用双向 `CLOSE` descriptor：

```text
CPU close:
  CPU -> L2: CLOSE
  L2 drain pending messages
  L2 -> CPU: ACK_CLOSE
  host releases channel memory
```

如果 L2 fatal，写 `header.fatal_status`，CPU broker 观察后唤醒所有等待者。

## 与现有 simpler 的接入点

### ChipWorker

`ChipWorker` 当前已经封装：

- `device_malloc_ctx`
- `device_free_ctx`
- `copy_to_device_ctx`
- `copy_from_device_ctx`
- `run_runtime`

HostDeviceChannel 应作为 `ChipWorker` 的长期资源，生命周期与 `DeviceContextHandle`
绑定，而不是与单次 `run_runtime()` 绑定。

### bootstrap_context

L3 多 chip 场景已经有 per-chip bootstrap：

```text
Worker(level=3)
  -> chip child
  -> ChipWorker.bootstrap_context()
  -> parent 获得 ChipContext(buffer_ptrs)
```

后续可以扩展 `ChipBootstrapConfig`：

```python
ChipBootstrapConfig(
    ...,
    host_device_channels=[
        HostDeviceChannelSpec(name="control", cpu_to_l2_lanes=8, ...)
    ],
)
```

bootstrap 成功后，`ChipContext` 暴露：

```python
ctx.channels["control"]
```

L3 orch function 可以把 channel handle 作为 scalar 或 child-memory tensor 传给 L2。

### Runtime

L2 runtime 需要在 AICPU 初始化阶段拿到 channel base pointer：

- `KernelArgs` 增加 channel table 指针；或
- `Runtime` 增加 host-device channel table；或
- 通过 `TaskArgs` scalar 传入某个 channel handle。

推荐第一版用 `TaskArgs` scalar 显式传入，避免修改所有 runtime 的固定 ABI。稳定后再提升为
`KernelArgs` 常驻字段。

## 为什么不直接用真 MPMC ring

真 MPMC ring 的典型实现需要：

- 全局 `enqueue_pos` 和 `dequeue_pos` CAS。
- 每 slot sequence number。
- acquire/release ordering。
- payload 与 descriptor 的严格发布顺序。

在单 CPU 进程内，这些条件容易满足；在 CPU/NPU 共享内存边界上，还需要证明：

- CPU CAS 与 AICPU/AICore atomic 对同一地址互相可见。
- 两端 cache hierarchy 对 release/acquire 的解释一致。
- host-mapped device memory 上的原子操作不会退化或被禁止。
- 所有硬件平台行为一致。

这些验证成本高于 multi-lane SPSC。除非后续 profiling 证明 broker/lane 方案成为瓶颈，
否则不建议第一阶段实现真 MPMC。

## 实现路线

### Phase 0: 文档和 POD ABI

- 新增 `host_device_channel.h`，定义 header/lane/desc POD。
- 写 sim-only 单元测试，验证 ring push/pop、wrap、full/empty。
- 静态断言结构大小和对齐。

### Phase 1: sim backend

- 在 sim `DeviceRunner` 中分配 channel region。
- CPU 和 simulated AICPU 线程通过同一 host memory 测试双向 send/recv。
- 增加 Python `ChipWorker.open_channel()` 实验接口。

### Phase 2: a2a3 memcpy MVP

- control region 和 payload region 放 device memory。
- CPU send/recv 使用 `copy_to_device_ctx` / `copy_from_device_ctx` 轮询。
- 先保证语义正确，不追求低延迟。

### Phase 3: a2a3 host-mapped fast path

- 参考 `tensor-dump` 的 host-register 方案，把 control region 映射到 host。
- CPU broker 直接 poll mapped control ring。
- AICPU 使用 `cache_flush_range()` / `cache_invalidate_range()` 维护可见性。

### Phase 4: L2 broker 集成

- AICPU runtime 增加 `hdch_poll_once()`。
- 支持 route 注册和 callback。
- 支持 request/response correlation。

### Phase 5: 性能和扩展

- 增加 batch send/recv。
- 增加 priority lane。
- 大 payload 接入 child-memory tensor handle。
- 评估是否需要真 MPMC ring。

## 测试计划

CPU/sim UT：

- 单 lane SPSC：空、满、wrap、seq monotonic。
- multi-lane MPMC：4 producer、4 consumer，无重复、无丢失。
- close/fatal/timeout。

sim ST：

- CPU producer 多线程发送消息，AICPU broker 回 ACK。
- L2 producer 多线程模拟发送消息，CPU broker demux。
- route/correlation_id 正确匹配。

a2a3 ST：

- memcpy MVP：小消息 ping-pong。
- 大 payload：64KB、1MB payload 校验 checksum。
- backpressure：lane_depth 很小，验证 `WOULD_BLOCK` 和阻塞 send。
- 长时间压测：百万级消息，无重复、无丢失、无死锁。

一致性测试：

- producer 写 payload 后延迟发布 descriptor，consumer 不应读到半包。
- descriptor 发布后 payload checksum 必须正确。
- close 期间 pending message drain 顺序正确。

## 开放问题

1. a2a3 host-register 对 control region 的映射是否支持 CPU 低延迟写 `tail`，还是必须通过
   `rtMemcpy`/doorbell 触发可见性。
2. AICore 是否需要直接 send/recv，还是第一版只允许 AICPU broker 代表 AICore 发送。
3. `PayloadPool` 是否应与现有 GM heap ring 合并，还是独立预留，避免影响 task output heap。
4. route/correlation_id 是否需要纳入更通用的 runtime event schema。
5. a5 是否有计划支持 host-mapped device memory；没有的话 a5 只能长期走 memcpy fallback。

