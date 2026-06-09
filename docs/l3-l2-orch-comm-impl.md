# L3-L2 Orchestrator Communication Implementation Guide

This document turns
[l3-l2-orch-comm-design-support-inflight.md](l3-l2-orch-comm-design-support-inflight.md)
into an implementation guide.

The design document is the contract. This file is the construction map: which
files to touch, what shape the ABI should take, where the service should live,
how the first implementation should be staged, and what tests should close each
stage.

## Reader Assumptions

The reader already knows the current hierarchy model:

- L3 `Worker` owns one or more NEXT_LEVEL chip children.
- Each chip child is a forked Python process running `_chip_process_loop`.
- Parent `WorkerThread` and child process communicate through a 4096-byte
  task mailbox.
- The chip child runs a `ChipWorker`, which loads `libhost_runtime.so` and
  delegates device work through the C ABI in `pto_runtime_c_api.*`.
- `tensormap_and_ringbuffer` runs orchestration on L2 AICPU, so the chip child
  main loop is blocked in `ChipWorker.run(...)` while the L2 orchestration
  task is active.

The new feature adds a second, independent control path for in-flight
communication between the L3 orchestration function and the running L2 AICPU
orchestration task.

## Core Implementation Rule

Do not send in-flight L3-L2 payload or signal commands through the existing
task mailbox.

The existing mailbox is only safe for bootstrap before the L2 task starts. Once
`ChipWorker.run(...)` begins, `_run_chip_main_loop` is not polling
`CONTROL_REQUEST`; it will not service `CTRL_COPY_TO`, `CTRL_COPY_FROM`, or any
new task-mailbox command until the L2 run returns.

The allowed split is:

```text
Existing task mailbox:
  bootstrap-only, before first dependent L2 run

New L3-L2 control service:
  ALLOC_REGION
  FREE_REGION
  PAYLOAD_WRITE
  PAYLOAD_READ
  SIGNAL_NOTIFY
  SIGNAL_WAIT
```

The control service must live in the chip child runtime side, not in the L3
Parent process and not in the child Python polling loop.

## Target User Flow

L3 orchestration:

```python
def l3_orch(orch, args, cfg):
    region = orch.create_l3_l2_region(worker_id=0, payload_bytes=payload_bytes)

    l2_args = TaskArgs()
    for value in region.descriptor_scalars():
        l2_args.add_scalar(value)
    # Add wrapper-level schema: offsets, dtype, shape, nbytes.
    l2_args.add_scalar(header_offset)
    l2_args.add_scalar(input_offset)
    l2_args.add_scalar(output_offset)
    l2_args.add_scalar(nbytes)

    orch.submit_next_level(l2_handle, l2_args, cfg, worker=0)

    for seq in range(1, rounds + 1):
        region.payload_write(input_offset, host_input)
        region.payload_write(header_offset, host_header)
        region.notify(seq)

        region.wait(seq, timeout=timeout_s)
        region.payload_read(output_offset, host_output)
        check(host_output)

    region.payload_write(header_offset, stop_header)
    region.notify(rounds + 1)
```

L2 orchestration:

```cpp
L3L2OrchEndpoint ep(desc);

while (true) {
    if (!ep.wait(seq, timeout)) {
        return endpoint_error(...);
    }

    PayloadView header{};
    if (!ep.payload_read(header_offset, sizeof(ChannelHeader), &header)) {
        return endpoint_error(...);
    }
    ChannelHeader hdr = wrapper_read_header(header.gm_addr);
    if (hdr.opcode == STOP) break;

    PayloadView input{};
    PayloadView output{};
    if (!ep.payload_read(input_offset, nbytes, &input)) {
        return endpoint_error(...);
    }
    if (!ep.payload_read(output_offset, nbytes, &output)) {
        return endpoint_error(...);
    }

    ContinuousTensor in = make_child_memory_tensor(input.gm_addr, shape, dtype);
    ContinuousTensor out = make_child_memory_tensor(output.gm_addr, shape, dtype);
    launch_aicore(in, out);
    wait_aicore_done();

    ep.notify(seq);
    ++seq;
}
```

The bottom layer does not know `DATA`, `STOP`, tensor shape, dtype, stride, or
stream layout. Those are wrapper/example protocol fields.

## Implementation Layers

Implement five layers and keep their responsibilities separate.

```text
Layer 1: Shared ABI
  command enum
  descriptor POD
  request/response POD
  signal-slot constants
  validation helpers

Layer 2: Runtime control service
  per-chip singleton under DeviceRunner
  service thread
  shared request/response memory
  GM region table
  DMA and signal execution

Layer 3: Parent bridge
  bootstrap command via WorkerThread mailbox
  C++ Worker/Orchestrator forwarding
  nanobind bindings

Layer 4: Python facade
  Orchestrator.create_l3_l2_region
  L3L2OrchRegion handle
  host tensor pointer/contiguity validation
  lifetime, poison, deferred free

Layer 5: L2 endpoint and example
  descriptor decode from TaskArgs scalars
  payload GM views
  L2 wait/notify
  closed-loop DATA/STOP example
```

## Existing Code Anchors

Use these files as the current implementation anchors.

Parent hierarchical runtime:

```text
src/common/hierarchical/worker_manager.h
src/common/hierarchical/worker_manager.cpp
src/common/hierarchical/orchestrator.h
src/common/hierarchical/orchestrator.cpp
python/bindings/worker_bind.h
python/simpler/orchestrator.py
python/simpler/worker.py
```

Chip worker and host runtime ABI:

```text
src/common/worker/chip_worker.h
src/common/worker/chip_worker.cpp
src/common/worker/pto_runtime_c_api.h
src/common/platform/onboard/host/c_api_shared.cpp
src/common/platform/sim/host/c_api_shared.cpp
```

DeviceRunner shared behavior:

```text
src/common/platform/onboard/host/device_runner_base.h
src/common/platform/onboard/host/device_runner_base.cpp
src/common/platform/sim/host/device_runner_base.h
src/common/platform/sim/host/device_runner_base.cpp
```

Per-arch platform exports and stubs:

```text
src/a2a3/platform/onboard/host/pto_runtime_c_api.cpp
src/a2a3/platform/sim/host/pto_runtime_c_api.cpp
src/a5/platform/onboard/host/pto_runtime_c_api.cpp
src/a5/platform/sim/host/pto_runtime_c_api.cpp
```

L2 runtime/orchestration include areas:

```text
src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/
src/a2a3/runtime/tensormap_and_ringbuffer/runtime/
src/a5/runtime/tensormap_and_ringbuffer/orchestration/
src/a5/runtime/tensormap_and_ringbuffer/runtime/
```

## Files To Add

Recommended additions:

```text
src/common/hierarchical/l3_l2_orch_comm.h
src/common/platform/include/common/l3_l2_orch_comm.h
src/common/platform/include/aicpu/l3_l2_orch_endpoint.h
src/common/platform/include/host/l3_l2_orch_comm_service.h
src/common/platform/shared/host/l3_l2_orch_comm_service.cpp
python/simpler/l3_l2_orch_comm.py
tests/ut/py/test_worker/test_l3_l2_orch_comm.py
tests/st/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm/
tests/st/a5/tensormap_and_ringbuffer/l3_l2_orch_comm/
examples/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm_stream/
```

The split between `src/common/hierarchical/l3_l2_orch_comm.h` and
`src/common/platform/include/common/l3_l2_orch_comm.h` is optional. Use two
headers only if the parent/child wire structs need different include paths from
the AICPU endpoint descriptor. If one shared header can compile cleanly in both
host and AICPU contexts, prefer one header.

## Files To Modify

Parent/hierarchical side:

```text
src/common/hierarchical/worker_manager.h
src/common/hierarchical/worker_manager.cpp
src/common/hierarchical/orchestrator.h
src/common/hierarchical/orchestrator.cpp
python/bindings/worker_bind.h
python/simpler/orchestrator.py
python/simpler/worker.py
```

Chip/runtime ABI side:

```text
src/common/worker/pto_runtime_c_api.h
src/common/worker/chip_worker.h
src/common/worker/chip_worker.cpp
src/common/platform/onboard/host/c_api_shared.cpp
src/common/platform/sim/host/c_api_shared.cpp
src/common/platform/onboard/host/device_runner_base.h
src/common/platform/onboard/host/device_runner_base.cpp
src/common/platform/sim/host/device_runner_base.h
src/common/platform/sim/host/device_runner_base.cpp
```

Build files:

```text
src/a2a3/platform/*/host/CMakeLists.txt
src/a5/platform/*/host/CMakeLists.txt
src/a2a3/platform/*/aicpu/CMakeLists.txt
src/a5/platform/*/aicpu/CMakeLists.txt
python/bindings/CMakeLists.txt
```

Add arch-specific code only when the backend really differs. Most command
validation, region table, poisoning, and request serialization should be common
between a2a3/a5 and sim/onboard.

## Shared Descriptor ABI

Descriptor encoded into `TaskArgs` scalars:

```cpp
struct L3L2OrchRegionDesc {
    uint64_t magic_version;
    uint64_t region_id;
    uint64_t payload_base;
    uint64_t payload_bytes;
    uint64_t l3_to_l2_signal_base;
    uint64_t l2_to_l3_signal_base;
};
```

Scalar layout:

```text
scalar[i + 0] = magic_version
scalar[i + 1] = region_id
scalar[i + 2] = payload_base
scalar[i + 3] = payload_bytes
scalar[i + 4] = l3_to_l2_signal_base
scalar[i + 5] = l2_to_l3_signal_base
```

Recommended `magic_version` packing:

```text
bits 63..32: magic, e.g. 'L3L2'
bits 31..16: major ABI version
bits 15..0:  minor ABI version
```

The first version should reject unknown major versions. Minor version can be
accepted only if all fields used by the endpoint are present and compatible.

Descriptor validation:

- `magic_version` matches expected ABI.
- `region_id != 0`.
- `payload_base != 0`.
- `payload_bytes != 0`.
- `payload_base + payload_bytes` does not overflow.
- both signal bases are nonzero.
- both signal bases are 64-byte aligned.
- signal bases are not inside the payload range unless the implementation
  intentionally carves one allocation and still keeps signal slots outside
  payload bounds.

The descriptor does not contain:

- dtype;
- shape;
- stride;
- tensor rank;
- stream header layout;
- ring layout;
- tile layout;
- child runtime lookup pointer.

## Command Namespace

Use a dedicated command namespace named `L3_L2_ORCH_COMM`. Operation enum:

```cpp
enum class L3L2OrchCommCmd : uint32_t {
    ALLOC_REGION = 1,
    FREE_REGION = 2,
    PAYLOAD_WRITE = 3,
    PAYLOAD_READ = 4,
    SIGNAL_NOTIFY = 5,
    SIGNAL_WAIT = 6,
};
```

Request payload:

```cpp
struct L3L2OrchCommRequest {
    uint32_t cmd;
    uint32_t reserved0;
    uint64_t region_id;
    uint64_t offset;
    uint64_t host_ptr;
    uint64_t nbytes;
    uint64_t signal_slot;
    uint64_t seq;
    uint64_t timeout_ns;
};
```

Response payload:

```cpp
struct L3L2OrchCommResponse {
    int32_t status;
    uint32_t error_kind;
    uint64_t region_id;
    uint64_t observed_signal;
    L3L2OrchRegionDesc desc;
    char message[256];
};
```

This is illustrative. The exact layout can change, but preserve these
properties:

- fixed-size POD only;
- no C++ containers;
- no tensor payload bytes;
- no process-private object pointers except Host data pointers used only by
  the child runtime helper for DMA;
- response carries `region_id` for region-scoped errors;
- `message` is diagnostic, not the attribution mechanism.

Because the first version serializes requests per chip, request ids are not
needed. If a later version supports multiple outstanding requests, add
request ids then.

## Control Shared Memory

Recommended layout for the independent service shared memory:

```text
offset 0:
  atomic<uint32_t> state
offset 4:
  uint32_t reserved
offset 8:
  L3L2OrchCommRequest request
offset 8 + sizeof(request), aligned:
  L3L2OrchCommResponse response
```

State machine:

```text
IDLE
  parent writes request
  parent release-stores READY

READY
  service acquire-loads READY
  service release-stores RUNNING

RUNNING
  service executes command
  service writes response
  service release-stores DONE

DONE
  parent acquire-loads DONE
  parent reads response
  parent release-stores IDLE
```

The parent side should serialize access with a per-worker mutex, so only one
L3 thread writes the request slot at a time. This matches the design's
one-request-in-flight contract.

Do not reuse the existing 4096-byte task mailbox for this layout. The
independent shared memory can be created by the parent and passed by name in
the bootstrap command.

## Bootstrap Command

Add a bootstrap-only mailbox control command:

```cpp
static constexpr uint64_t CTRL_L3_L2_ORCH_COMM_INIT = 12;  // next free id
```

Update both:

```text
src/common/hierarchical/worker_manager.h
python/simpler/worker.py
```

Parent path:

```text
Orchestrator.create_l3_l2_region(worker_id, payload_bytes)
  -> Python Worker ensures hierarchical children are started
  -> if service not ready:
       if target WorkerThread is busy: raise bootstrap-ordering error
       create POSIX shm for service request/response slot
       send CTRL_L3_L2_ORCH_COMM_INIT through task mailbox
       wait CONTROL_DONE
       mark service ready for worker
  -> send ALLOC_REGION over independent service
  -> return L3L2OrchRegion
```

Child path:

```text
_run_chip_main_loop sees CTRL_L3_L2_ORCH_COMM_INIT
  -> read shm name from OFF_ARGS
  -> cw.l3_l2_orch_comm_init(shm_name)
  -> ChipWorker calls host_runtime.so C ABI
  -> DeviceRunner starts service thread
  -> child writes CONTROL_DONE
```

The task mailbox is allowed here only because bootstrap must happen before the
L2 run that needs the service.

Bootstrap failure rules:

- if `CTRL_L3_L2_ORCH_COMM_INIT` fails, no Python region handle is returned;
- if target worker is busy and service is not ready, fail immediately;
- if service is already ready, `create_l3_l2_region` must not touch the task
  mailbox and may allocate while L2 is in flight;
- service lifetime is per chip child and ends when `ChipWorker.finalize()` or
  child process exit tears it down.

## Host Runtime C ABI

Export the new ABI from each `libhost_runtime.so`. `ChipWorker.init` should
dlsym these symbols unconditionally, matching the existing comm ABI style.
Unsupported platforms export stubs.

Recommended C ABI:

```c
int l3_l2_orch_comm_init_ctx(DeviceContextHandle ctx,
                             const char *control_shm_name);
int l3_l2_orch_comm_shutdown_ctx(DeviceContextHandle ctx);

int l3_l2_orch_comm_alloc_region_ctx(
    DeviceContextHandle ctx,
    uint64_t payload_bytes,
    struct L3L2OrchRegionDesc *out_desc);

int l3_l2_orch_comm_free_region_ctx(
    DeviceContextHandle ctx,
    uint64_t region_id);

int l3_l2_orch_comm_payload_write_ctx(
    DeviceContextHandle ctx,
    uint64_t region_id,
    uint64_t offset,
    const void *host_ptr,
    uint64_t nbytes);

int l3_l2_orch_comm_payload_read_ctx(
    DeviceContextHandle ctx,
    uint64_t region_id,
    uint64_t offset,
    void *host_ptr,
    uint64_t nbytes);

int l3_l2_orch_comm_signal_notify_ctx(
    DeviceContextHandle ctx,
    uint64_t region_id,
    uint32_t signal_slot,
    uint64_t seq);

int l3_l2_orch_comm_signal_wait_ctx(
    DeviceContextHandle ctx,
    uint64_t region_id,
    uint32_t signal_slot,
    uint64_t seq,
    uint64_t timeout_ns);
```

There are two possible designs for command submission:

1. Parent sends every command through the independent shared memory and waits
   for the service thread.
2. Parent calls individual C ABI functions in the chip child, and those
   functions enqueue work into the service thread.

Given the parent process cannot call functions inside the child process, the
actual parent-to-child path still needs independent IPC. The individual C ABI
functions are useful inside the child process and for tests, but the in-flight
parent path should be the independent shared-memory service initialized by
`l3_l2_orch_comm_init_ctx`.

Practical recommendation:

- use C ABI `init_ctx` to start the service with a shared-memory slot;
- use shared-memory requests for normal L3 API commands;
- keep small C ABI helper methods for chip-child tests if useful;
- keep all device operations in `DeviceRunner`, not in Python.

## `ChipWorker` Changes

Modify:

```text
src/common/worker/chip_worker.h
src/common/worker/chip_worker.cpp
```

Add dlsym function pointer types and methods:

```cpp
using L3L2OrchCommInitFn = int (*)(void *, const char *);
using L3L2OrchCommShutdownFn = int (*)(void *);

void l3_l2_orch_comm_init(const std::string &control_shm_name);
void l3_l2_orch_comm_shutdown();
```

If you choose to expose direct child-local helper methods, add:

```cpp
L3L2OrchRegionDesc l3_l2_orch_comm_alloc_region(uint64_t payload_bytes);
void l3_l2_orch_comm_free_region(uint64_t region_id);
void l3_l2_orch_comm_payload_write(...);
void l3_l2_orch_comm_payload_read(...);
void l3_l2_orch_comm_signal_notify(...);
void l3_l2_orch_comm_signal_wait(...);
```

But do not mistake these methods for the parent in-flight path. Parent cannot
call these while the child is blocked in `ChipWorker.run(...)`; the independent
service is what makes progress.

`ChipWorker.finalize()` should best-effort shut down the service before
destroying the device context. Service shutdown must not hang if the service
was never initialized.

## DeviceRunner Service

The runtime service should be owned by `DeviceRunnerBase` or by a helper object
owned from `DeviceRunnerBase`.

Suggested state:

```cpp
class L3L2OrchCommService {
 public:
  int start(DeviceRunnerBase *runner, const char *control_shm_name);
  int stop();
  bool started() const;

 private:
  DeviceRunnerBase *runner_;
  void *control_shm_;
  size_t control_shm_size_;
  std::thread thread_;
  std::atomic<bool> stop_;
  std::mutex regions_mu_;
  std::unordered_map<uint64_t, Region> regions_;
  uint64_t next_region_id_;
};
```

Region record:

```cpp
struct Region {
    uint64_t region_id;
    void *payload_dev;
    uint64_t payload_bytes;
    void *l3_to_l2_signal_dev;
    void *l2_to_l3_signal_dev;
    bool released;
    bool poisoned;
};
```

Service loop:

```text
attach current thread to DeviceRunner device id
while !stop:
  wait/poll control state READY
  mark RUNNING
  decode request
  validate request
  execute with DeviceRunner allocation/copy/signal helpers
  write response
  mark DONE
```

For onboard, the service thread must attach to the target device before
calling CANN runtime APIs. Follow the existing `DeviceRunnerBase::create_thread`
pattern or call `attach_current_thread(device_id())` at thread entry.

For sim, bind the simulated device in the service thread, matching
`SimDeviceRunnerBase::create_thread`.

## Region Allocation Details

First version allocation can be:

```text
allocation A: payload_bytes, aligned to backend requirements
allocation B: 64 bytes, l3_to_l2 signal
allocation C: 64 bytes, l2_to_l3 signal
```

or:

```text
one allocation:
  payload range
  padding to 64B
  l3_to_l2 signal 64B
  l2_to_l3 signal 64B
```

The descriptor contract is the same either way.

On allocation:

1. validate `payload_bytes > 0`;
2. allocate GM;
3. allocate or carve signal slots;
4. initialize both signal slots to zero;
5. allocate a new `region_id`;
6. store the region record;
7. return descriptor.

On free:

1. find region;
2. mark released;
3. free physical GM only when Host cleanup says all submitted L2 work is
   drained;
4. remove region from table after physical free.

The service can physically free immediately only when the Python side already
knows there is no in-flight L2 task with the descriptor. The public
`region.free()` method does not know that by itself, so it should queue
deferred free until orchestration cleanup.

## Payload Operations

L3 `payload_write`:

```text
validate Host handle
validate host tensor contiguous and child-visible
extract host pointer and byte span
send PAYLOAD_WRITE(region_id, offset, host_ptr, nbytes)
service validates region and bounds
service copies Host DRAM -> Device GM
service synchronizes
parent returns
```

L3 `payload_read`:

```text
validate Host handle
validate destination contiguous and child-visible
extract host pointer and byte span
send PAYLOAD_READ(region_id, offset, host_ptr, nbytes)
service validates region and bounds
service copies Device GM -> Host DRAM
service synchronizes
parent returns with destination readable
```

Validation before command issue:

- region handle is live;
- region handle is not poisoned;
- region handle belongs to this L3 orchestration execution;
- `offset >= 0`;
- `nbytes > 0`;
- `offset + nbytes <= payload_bytes`;
- tensor is contiguous;
- tensor has enough bytes;
- tensor storage is child-visible shared Host memory.

The first version should not support Python `bytes`, temporary `bytearray`, or
ordinary private tensors as payload buffers. That would require staging through
child-private memory, which is explicitly not the main path.

Small wrapper headers are payload too. They must also live in shared Host
storage and be copied with `payload_write`.

## Host Buffer Visibility

The hard part is not extracting a pointer. The hard part is proving the child
runtime process can dereference the pointer.

Acceptable first-version sources:

- tensors allocated by `orch.alloc(...)` if that allocator returns MAP_SHARED
  host-visible storage suitable for L3 use;
- `torch.Tensor.share_memory_()` buffers created before chip children are
  forked, if tests prove the child can access the same storage address;
- a new runtime-provided shared Host tensor allocator owned by this feature.

Reject:

- ordinary Parent-private tensors;
- temporary Python bytes;
- post-fork memory not mapped in the child;
- non-contiguous tensors;
- tensors whose data pointer cannot be extracted through a supported binding.

Stage 4 currently implements the conservative subset: `payload_write` and
`payload_read` accept only live `ContinuousTensor` handles returned by
`orch.alloc(...)` in the current orchestration run. Other shared Host tensor
sources remain future extensions until their child-visible address contract is
tested.

If detection is incomplete, fail conservatively with a message such as:

```text
L3L2OrchRegion.payload_write requires child-visible shared Host tensor storage;
private or post-fork buffers are not supported in the first version
```

## Signal Semantics

Signal slots:

```text
l3_to_l2_signal_base: L3 notify, L2 wait
l2_to_l3_signal_base: L2 notify, L3 wait
```

Each slot stores one `uint64_t` sequence number.

Notify:

```text
store exactly seq
```

Wait:

```text
current == seq -> success
current < seq  -> keep waiting
current > seq  -> protocol error
timeout        -> timeout error
```

L3 `SIGNAL_WAIT` timeout poisons the Host region. L2 wait timeout returns an
endpoint error carrying `region_id`.

Do not use CAS, fetch-add, or any atomic read-modify-write protocol in the
first version.

AICPU spin-wait must not use `std::this_thread::yield()` or `sched_yield()`.
On AICPU, yielding to the OS scheduler is too expensive for tight protocol
waits. Use an empty loop or the local spin hint.

## Cache And Ordering

The first version can keep ordering simple:

```text
L3 payload_write completes H2D DMA
L3 notify(seq)
L2 wait(seq)
L2 reads payload / passes GM view to AICore
AICore writes output
L2 waits for AICore completion
L2 notify(seq)
L3 wait(seq)
L3 payload_read completes D2H DMA
```

Implementation responsibilities:

- H2D copy completes before L3 notify returns.
- L2 wait is the acquire point before reading Host-written payload.
- AICPU reads of Host-written payload should invalidate payload cache range
  before load when needed by the platform.
- AICore input visibility should be handled by runtime tensor/cache policy or
  kernel convention.
- AICore output must be visible in GM before L2 notify.
- D2H copy completes before L3 `payload_read` returns.

If cache maintenance details are uncertain for a platform, write a small
platform unit or ST case that has L3 write a changing pattern, L2/AICore read
it, and L3 verify output over multiple rounds.

## Python Facade

Add:

```text
python/simpler/l3_l2_orch_comm.py
```

Suggested class:

```python
class L3L2OrchRegion:
    def __init__(self, owner, worker_id: int, descriptor, payload_bytes: int):
        self._owner = owner
        self._worker_id = worker_id
        self._descriptor = descriptor
        self._payload_bytes = payload_bytes
        self._released = False
        self._poisoned = False
        self._expired = False

    def descriptor_scalars(self) -> list[int]: ...
    def payload_write(self, offset: int, host_tensor, nbytes: int | None = None) -> None: ...
    def payload_read(self, offset: int, host_tensor, nbytes: int | None = None) -> None: ...
    def notify(self, seq: int) -> None: ...
    def wait(self, seq: int, timeout: float) -> None: ...
    def free(self) -> None: ...
```

Add to `python/simpler/orchestrator.py`:

```python
def create_l3_l2_region(self, *, worker_id: int, payload_bytes: int) -> L3L2OrchRegion:
    if self._worker is None:
        raise RuntimeError("create_l3_l2_region requires an Orchestrator bound to a Worker")
    return self._worker._create_l3_l2_region(worker_id, payload_bytes)
```

Add owning state to `python/simpler/worker.py`:

```python
self._l3_l2_orch_comm_ready: set[int] = set()
self._l3_l2_orch_comm_shms: dict[int, SharedMemory] = {}
self._live_l3_l2_regions: list[L3L2OrchRegion] = []
self._pending_free_l3_l2_regions: list[L3L2OrchRegion] = []
```

Lifecycle:

- handle is valid only inside one L3 orchestration execution;
- after `region.free()`, reject all operations except idempotent `free()`;
- after poison, reject payload/signal/descriptor operations;
- after L3 orch function returns, mark handles expired;
- drain-time cleanup sends physical free for live/released/poisoned regions.

## C++ Parent Bridge

Add internal C++/binding operations for the Python facade rather than letting
Python manually poke mailboxes.

Potential `_Worker` methods:

```cpp
void l3_l2_orch_comm_bootstrap(int worker_id, const char *control_shm_name);
L3L2OrchRegionDesc l3_l2_orch_comm_alloc_region(int worker_id, uint64_t payload_bytes);
void l3_l2_orch_comm_free_region(int worker_id, uint64_t region_id);
void l3_l2_orch_comm_payload_write(int worker_id, ...);
void l3_l2_orch_comm_payload_read(int worker_id, ...);
void l3_l2_orch_comm_signal_notify(int worker_id, ...);
void l3_l2_orch_comm_signal_wait(int worker_id, ...);
```

The bootstrap method uses `WorkerThread::control_*` style mailbox access. The
normal command methods use the independent service shared memory and a
per-worker mutex.

In `WorkerThread`, avoid mixing new in-flight service commands with existing
`run_control_command`. Add a separate object owned by `Worker` or
`WorkerManager` for independent control service state:

```cpp
struct L3L2OrchCommClient {
    std::mutex mu;
    void *shm_addr;
    size_t shm_size;
    bool ready;
};
```

The C++ parent client does:

```text
lock client.mu
wait state IDLE
write request
store READY
wait DONE with timeout
read response
store IDLE
unlock
```

Timeout waiting for the service response should poison the targeted Host
region when there is a targeted region. For `ALLOC_REGION`, no region handle
exists yet, so it raises without poisoning.

## L2 Endpoint

Add endpoint header for orchestration code:

```text
src/common/platform/include/aicpu/l3_l2_orch_endpoint.h
```

Illustrative API:

```cpp
struct L3L2OrchPayloadView {
    uint64_t gm_addr;
    uint64_t nbytes;
};

enum class L3L2EndpointErrorKind : uint32_t {
    NONE = 0,
    BAD_DESCRIPTOR = 1,
    OUT_OF_BOUNDS = 2,
    SIGNAL_TIMEOUT = 3,
    SIGNAL_PROTOCOL = 4,
};

struct L3L2EndpointError {
    L3L2EndpointErrorKind kind;
    const char *op;
    uint64_t region_id;
    uint64_t seq;
    const char *message;
};

class L3L2OrchEndpoint {
public:
    explicit L3L2OrchEndpoint(const L3L2OrchRegionDesc &desc);

    const L3L2EndpointError &error() const;

    bool payload_read(uint64_t offset, uint64_t nbytes, L3L2OrchPayloadView *out);
    bool payload_write(uint64_t offset, const void *src, uint64_t nbytes);
    bool wait(uint64_t seq, uint64_t timeout);
    bool notify(uint64_t seq);
};
```

The current implementation is header-only in
`src/common/platform/include/aicpu/l3_l2_orch_endpoint.h`.
The endpoint uses a header-local monotonic counter for `wait()` timeout checks,
so orchestration `.so` files do not need an extra link dependency on
`device_time.cpp`.

`payload_read` returns a GM view and does not copy. It is used for tensor input
and output views. The streaming wrapper may also use the returned GM view for
its fixed-size header, but that header interpretation stays wrapper-local and
does not add another bottom-layer operation.

`payload_write` is narrow and intended for metadata/status only. It is not the
primary output path.

Endpoint validation:

- descriptor ABI;
- payload range;
- signal alignment;
- sequence monotonicity;
- timeout.

Endpoint errors must carry `region_id`; L3 uses it for poisoning the matching
Host handle.

## Building Tensor Views On L2

The endpoint returns only:

```text
gm_addr
nbytes
```

The orchestration wrapper combines this with schema from task scalars:

```text
input_offset
output_offset
dtype
shape
nbytes
```

Example:

```cpp
PayloadView input{};
PayloadView output{};
if (!ep.payload_read(input_offset, nbytes, &input)) {
    return endpoint_error(...);
}
if (!ep.payload_read(output_offset, nbytes, &output)) {
    return endpoint_error(...);
}

ContinuousTensor input_tensor;
input_tensor.data = input.gm_addr;
input_tensor.child_memory = true;
input_tensor.dtype = dtype;
input_tensor.shapes = shape;

ContinuousTensor output_tensor = input_tensor;
output_tensor.data = output.gm_addr;
```

Use the actual runtime tensor construction helpers already present in
`tensormap_and_ringbuffer`; the example above is only the semantic shape.

## Platform Support

Target behavior:

| Platform | Behavior |
| --- | --- |
| `a2a3sim` | Full support. |
| `a5sim` | Full support. |
| `a2a3` onboard | Full support. |
| `a5` onboard | Export symbols and return clear not-supported errors. |

Simulation must preserve:

- same descriptor ABI;
- same timeout behavior;
- same poison behavior;
- same `current > seq` protocol error;
- same bootstrap ordering rule;
- same no-payload-in-control-path rule.

The sim implementation may use host allocations and `std::memcpy`, but the
test path should still prove that task mailbox dispatch is not servicing the
in-flight commands.

## Error And Poison Rules

Poison on:

- L3 signal wait timeout;
- L2 signal wait timeout reported with a valid `region_id`;
- DMA failure after a command is issued;
- signal notify failure;
- service fatal error after region allocation;
- signal protocol error such as `current > seq`;
- service response timeout for a targeted region.

Do not poison on pre-command validation:

- malformed Python arguments;
- non-contiguous tensor;
- unsupported Host buffer;
- out-of-bounds offset detected before command issue;
- descriptor extraction after release or poison;
- `ALLOC_REGION` failure before a descriptor is returned.

Host poisoned state:

```text
payload_write -> raise
payload_read  -> raise
notify        -> raise
wait          -> raise
descriptor    -> raise
free          -> allowed/idempotent
cleanup       -> allowed
```

L2 endpoint failure should not attempt recovery. It returns an orchestration
runtime error with structured metadata.

## Cleanup And Lifetime

Region lifetime is one L3 orchestration execution.

Handle state transitions:

```text
LIVE -> RELEASED
LIVE -> POISONED
POISONED -> RELEASED
LIVE/RELEASED/POISONED -> EXPIRED after orch function returns
```

Physical free must happen after submitted L2 work that could hold the
descriptor has drained.

Practical cleanup sequence in `Worker.run`:

```text
before orch_fn:
  open L3 run region registry

inside orch_fn:
  create regions
  submit L2 work
  region.free marks released and queues deferred free

after orch_fn:
  drain submitted work
  for each region in this run:
      send FREE_REGION if service ready and region has descriptor
      mark handle expired
  clear run registry
```

If the orchestration function raises, cleanup should still run after the
runtime reaches a safe drain/cancel point. If current Worker error handling
cannot safely drain after orch exceptions, document the limitation and make
region cleanup best-effort.

## Streaming Wrapper Example

Use one region:

```text
payload offset 0..63: ChannelHeader
payload offset input_offset: input tensor slice
payload offset output_offset: output tensor slice
```

Header:

```cpp
enum ChannelOpcode : uint32_t {
    DATA = 1,
    STOP = 2,
};

struct ChannelHeader {
    uint64_t seq;
    uint32_t opcode;
    uint32_t reserved;
};
```

Round:

```text
L3 writes input slice
L3 writes header {seq, DATA}
L3 notify(seq)

L2 wait(seq)
L2 loads header
L2 runs AICore input -> output
L2 notify(seq)

L3 wait(seq)
L3 reads output slice
```

Stop:

```text
L3 writes header {seq + 1, STOP}
L3 notify(seq + 1)
L2 observes STOP and returns
Worker.run drain is the acknowledgement
```

STOP does not require a third signal slot.

The sim ST implementation lives under:

```text
tests/st/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm/
tests/st/a5/tensormap_and_ringbuffer/l3_l2_orch_comm/
```

Each test creates one region, submits one persistent L2 orchestration task, and
then drives multiple DATA rounds entirely through `payload_write`, `notify`,
`wait`, and `payload_read`. The L2 wrapper decodes the descriptor scalars,
views the input/output payload slices as fixed `FLOAT32[128 * 128]` tensors,
submits AIV work, and notifies L3 after the output producer is complete. The
STOP round is represented only by `ChannelHeader` metadata; L2 returning from
the orchestration task lets `Worker.run` drain act as the acknowledgement.

## Development Stages

### Stage 1: ABI And Validation

Implement:

- shared descriptor struct;
- command enum;
- request/response structs;
- descriptor scalar encode/decode;
- payload bounds helpers;
- signal slot validation helpers.

Tests:

- descriptor accepts valid six-scalar layout;
- descriptor rejects bad magic/version;
- descriptor rejects zero payload bytes;
- bounds helper rejects overflow;
- command structs contain no payload byte array;
- `PAYLOAD_WRITE` / `PAYLOAD_READ` names are used consistently.

Exit criteria:

- unit tests pass without any platform runtime changes;
- docs and tests agree on `payload_read` / `payload_write` naming.

### Stage 2: Sim Service Without L2 Endpoint

Implement:

- service shared-memory state machine;
- sim service thread;
- sim region allocation/free;
- sim payload write/read;
- sim signal notify/wait;
- Python handle for direct L3 calls.

Tests:

- allocate region;
- write bytes;
- notify/wait from test helper;
- read bytes;
- timeout poisons;
- free is idempotent.

Exit criteria:

- `a2a3sim` and `a5sim` service unit tests pass;
- no task mailbox is used for normal service commands.

### Stage 3: Bootstrap Integration

Implement:

- `CTRL_L3_L2_ORCH_COMM_INIT`;
- `_run_chip_main_loop` handler;
- `ChipWorker` dlsym/init wrapper;
- `Worker` per-chip ready state;
- bootstrap-ordering failure if worker busy and service not ready.

Tests:

- first `create_l3_l2_region` bootstraps service;
- second region allocation reuses ready service;
- bootstrap after persistent L2 task starts fails clearly;
- allocation after prior bootstrap works while L2 task is in flight.

Exit criteria:

- bootstrap behavior is deterministic and covered in sim.

### Stage 4: L2 Endpoint

Implement:

- AICPU endpoint descriptor decode;
- `payload_read` GM view;
- narrow `payload_write` for metadata/status;
- L2 signal wait/notify;
- structured endpoint errors.

Tests:

- endpoint validates descriptor;
- endpoint rejects out-of-bounds;
- endpoint detects `current > seq`;
- endpoint timeout reports `region_id`;
- AICPU can load a changing header over multiple rounds.

Exit criteria:

- L2 can wait on L3 signal and notify L3 without AICore work.

### Stage 5: Closed-Loop Sim Example

Implement:

- example or ST with one persistent L2 run;
- fixed contiguous tensor shape/dtype;
- one input slice and one output slice;
- AICore writes output directly into region output slice;
- L3 validates golden output every round;
- STOP returns L2 cleanly.

Tests:

- multiple DATA rounds;
- STOP round;
- output changes each round;
- no mailbox command services in-flight payload/signal path.

Exit criteria:

- `a2a3sim` and `a5sim` closed-loop tests pass.

### Stage 6: Onboard A2A3

Implement:

- onboard DeviceRunner service thread attach;
- GM allocation and free;
- H2D/D2H copy;
- signal load/store;
- stream sync or equivalent completion;
- cache maintenance required by AICPU/AICore visibility.

Tests:

- locked `task-submit` ST for closed-loop a2a3;
- timeout poison;
- repeated sequence rounds;
- changing data pattern;
- optional stress loop for stale cache detection.

Exit criteria:

- a2a3 onboard passes closed-loop tests under `task-submit`.

### Stage 7: A5 Onboard Stubs

Implement:

- export all symbols;
- return not-supported error;
- avoid partial service state;
- ensure `ChipWorker.init` succeeds if only symbol availability matters;
- region creation fails clearly on use.

Tests:

- symbols resolve;
- `create_l3_l2_region` raises not-supported;
- no leaked Python Host handle;
- repeated failure stays clean.

Exit criteria:

- a5 onboard behavior is explicit and stable.

### Stage 8: Multi-Region And Poison Isolation

Implement:

- multiple live regions per L3 run;
- structured error mapping from `region_id` to Host handle;
- per-region poison only.

Tests:

- create two regions;
- transfer different data;
- force endpoint error on one region;
- matching region poisons;
- other region remains usable.

Exit criteria:

- region attribution does not depend on parsing diagnostic strings.

## Test Matrix

Unit tests:

```bash
python -m pytest tests/ut/py/test_worker/test_l3_l2_orch_comm.py -q
```

Sim ST:

```bash
python -m pytest \
  tests/st/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm \
  --platform a2a3sim -q

python -m pytest \
  tests/st/a5/tensormap_and_ringbuffer/l3_l2_orch_comm \
  --platform a5sim -q
```

Onboard a2a3:

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
TEST_DIR=tests/st/a2a3/tensormap_and_ringbuffer/l3_l2_orch_comm
task-submit --device auto --device-num 1 \
  --run "python -m pytest ${TEST_DIR} --platform a2a3 --device \$TASK_DEVICE -q"
```

Onboard a5 stub:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
TEST_DIR=tests/st/a5/tensormap_and_ringbuffer/l3_l2_orch_comm
task-submit --device auto --device-num 1 \
  --run "python -m pytest ${TEST_DIR} --platform a5 --device \$TASK_DEVICE -q"
```

## Review Checklist

Before considering the implementation ready:

- task mailbox is used only for bootstrap;
- in-flight commands use the independent service;
- payload bytes never appear in control requests;
- Host Parent never directly touches Device GM;
- signal slots are outside payload bounds;
- signal slots initialize only on region allocation;
- waits require finite timeouts;
- `current > seq` is a protocol error;
- timeout poisons only the corresponding region;
- descriptor extraction fails after release or poison;
- physical free is delayed until after L2 drain;
- a5 onboard exports stubs, not missing symbols;
- sim and onboard share API and error semantics;
- closed-loop example uses AICore direct output into the region;
- STOP is payload metadata, not a separate signal slot.

## Open Decisions

These are implementation choices that can be made during coding without
changing the design contract.

1. Whether to keep one shared ABI header or split parent-wire and AICPU
   endpoint headers.
2. Whether the service uses one shared request slot or an internal queue fed by
   the shared request slot. First version should still expose only one
   request in flight.
3. Whether region physical storage is one allocation or three allocations.
4. Which exact Host shared tensor allocator is accepted first.
5. Whether C ABI direct command helpers are exported for tests or all normal
   commands go only through shared-memory service requests.

None of these should change the public first-version semantics.
