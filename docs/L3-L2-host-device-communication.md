# L3/L2 Host-Device Communication

This page explains Simpler's L3/L2 host-device communication primitive:
`HostDeviceMappedRegion`.

It is written for people learning the Simpler library. It covers ownership,
lifetime, the public ABI, Python usage, datacopy, signal semantics, and how
mapped regions relate to the normal `TaskArgs` tensor path. It intentionally
does not document mailbox payload layouts, backend allocation internals, or
test implementation details.

For the surrounding runtime model, see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md) and
[task-flow.md](task-flow.md). 

## Why It Exists

Simpler's normal task path is tensor-oriented:

1. User code builds `TaskArgs`.
2. The runtime stages tensor inputs to device memory.
3. A chip task runs.
4. Output tensors are copied back or validated through the task runtime.

That path is the right default for ordinary kernel inputs and outputs. It is
less suitable when host code and device code need to coordinate across several
steps, reuse one device-visible buffer, or exchange small phase signals without
turning every phase into a new task payload.

`HostDeviceMappedRegion` fills that gap. A mapped region gives host code:

- a reusable data area that is visible to device code,
- one or more signal slots for simple phase handshakes,
- explicit host-to-region and region-to-host byte copies, and
- explicit notify/wait operations.

The primitive is intentionally small. It does not define a queue, channel,
message format, tensor descriptor, or scheduler dependency policy. Higher-level
protocols can build those rules on top of the data area and signal slots.

## Runtime Ownership

The mapped region is owned by the process that owns the chip-side
`DeviceContext`.

At L2, a `Worker(level=2)` owns one `ChipWorker` and talks to one chip
directly:

```text
Python Worker(level=2)
    |
    +-- ChipWorker
            |
            +-- L2 chip runtime and kernels
```

At L3, a `Worker(level=3)` owns one chip child process per device id. The L3
parent exposes the same mapped-region methods, and `worker_id` selects which
chip child owns the region:

```text
Python Worker(level=3)
    |
    +-- chip child 0 -> ChipWorker -> L2 chip 0
    +-- chip child 1 -> ChipWorker -> L2 chip 1
    +-- sub workers
```

In L3 process mode, the chip child process owns `ChipWorker`, `DeviceRunner`,
the loaded host runtime, and the mapped-region registry. The L3 parent owns
only a Python `MappedRegion` wrapper and reaches the child through the
existing parent-child control path.

The NPU does not own the allocation lifetime. Device code participates by
reading and writing the device-visible addresses returned by
`mapped_region_info()`.

Pointer semantics follow the ownership rule:

- `device_data_ptr` and `device_signal_ptr` are public device-visible
  addresses. They can be passed to kernels through `TaskArgs` scalars or
  tensor metadata.
- `host_data_ptr` and `host_signal_ptr` are not public Python dereferenceable
  addresses. The Python API reports them as `0`.
- Host code accesses the region through datacopy methods, not by writing a
  mapped host pointer directly.

The L3 mailbox is only a host-side proxy transport. It is not the CPU-NPU
mapped-region primitive itself.

## Lifetime And Handle Ownership

Each opened region is registered in the owning `DeviceContext`. All `info`,
datacopy, notify, wait, and close operations validate that the handle belongs
to that context.

A handle from another context, a stale handle, a closed handle, or a
double-close is invalid. In Python, `MappedRegion` also records the owning
`worker_id`; passing a different `worker_id` to a later operation is rejected
before the request is sent to the chip child.

`close_mapped_region()` releases the owner-side host mapping and device
allocation. It is not a device synchronization operation. The caller must
ensure no in-flight kernel, AICPU code, or other device participant still uses
`device_data_ptr` or `device_signal_ptr` before closing the region.

Usually that means waiting for task completion or for the protocol's
completion signal before close:

```text
host writes input
host notify input_ready
device reads input and writes output
device publish output_ready
host wait output_ready
host reads output
host close mapped region
```

`Worker.close()`, `finalize_device()`, and `destroy_device_context()` clean up
remaining mapped regions as a resource fallback. That cleanup does not make it
safe to close while device code is still accessing the region.

## Public ABI

The runtime C ABI defines an opaque handle plus config and info structures in
`src/common/worker/pto_runtime_c_api.h`. Most library users do not call this
ABI directly, but it is the stable boundary that `ChipWorker` resolves from a
runtime shared object.

```cpp
typedef void *HostDeviceMappedRegionHandle;

typedef struct HostDeviceMappedRegionConfig {
    uint64_t data_bytes;
    uint32_t signal_count;
    uint32_t flags;
} HostDeviceMappedRegionConfig;

typedef struct HostDeviceMappedRegionInfo {
    uint64_t host_data_ptr;
    uint64_t device_data_ptr;
    uint64_t data_bytes;
    uint64_t host_signal_ptr;
    uint64_t device_signal_ptr;
    uint32_t signal_count;
    uint32_t reserved0;
    uint64_t total_bytes;
    uint32_t flags;
    uint32_t reserved1;
} HostDeviceMappedRegionInfo;
```

`flags` is reserved and must be `0`.

The C ABI entry points are:

```cpp
int open_host_device_mapped_region_ctx(
    DeviceContextHandle ctx,
    const HostDeviceMappedRegionConfig *cfg,
    HostDeviceMappedRegionHandle *out_region
);

int close_host_device_mapped_region_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region
);

int host_device_mapped_region_info_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    HostDeviceMappedRegionInfo *info
);

int host_device_mapped_region_datacopy_h2region_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint64_t offset,
    const void *src,
    size_t nbytes
);

int host_device_mapped_region_datacopy_region2h_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint64_t offset,
    void *dst,
    size_t nbytes
);

int host_device_mapped_region_notify_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint32_t signal_id,
    uint32_t value
);

int host_device_mapped_region_wait_ctx(
    DeviceContextHandle ctx,
    HostDeviceMappedRegionHandle region,
    uint32_t signal_id,
    uint32_t target,
    uint32_t timeout_us
);
```

The ABI uses negative errno-style return codes:

- `0`: success.
- `-EINVAL`: invalid context, handle, config, range, signal id, value, or
  pointer.
- `-EAGAIN` / `-EWOULDBLOCK`: non-blocking wait miss or bounded wait timeout.
- `-ENOMEM`: allocation or wrapper construction failure.
- `-EIO`: backend mapping, datacopy, or signal failure.
- `-ENOTSUP`: unsupported platform or unsupported backend feature.

Python maps invalid user input to `ValueError`, wait miss or timeout to
`TimeoutError`, and backend or unsupported-platform failures to `RuntimeError`.

## Python API

The user-facing API is exposed through `Worker`:

```python
region = worker.open_mapped_region(
    data_bytes,
    signal_count=2,
    flags=0,
    worker_id=0,
)

info = worker.mapped_region_info(region)

worker.mapped_region_datacopy_h2region(region, offset, payload)
payload = worker.mapped_region_datacopy_region2h(region, offset, nbytes)

worker.mapped_region_notify(region, signal_id, value)
worker.mapped_region_wait(region, signal_id, target, timeout_us)

worker.close_mapped_region(region)
```

Direct L2 calls execute in the owner process. L3 calls route to the selected
chip child while preserving the same public method names.

`MappedRegion` is a lightweight Python wrapper, not a pointer. It records:

- the opaque runtime handle,
- the owning `worker_id`,
- the requested `data_bytes`,
- the requested `signal_count`,
- the reserved `flags` value, and
- whether the region is closed.

Follow-up operations default to `region.worker_id`. Passing a different
`worker_id` is a user error. Operations on a closed region are also user
errors.

`mapped_region_info()` returns a `MappedRegionInfo` object with:

- `device_data_ptr`: device-visible base address of the data area,
- `device_signal_ptr`: device-visible base address of the signal slots,
- `data_bytes`: usable data bytes,
- `signal_count`: number of signal slots,
- `total_bytes`: backend allocation size, and
- `flags`: currently `0`.

The host pointer fields are always reported as `0` in the public Python API.

`mapped_region_datacopy_h2region()` accepts bytes-like contiguous buffers.
`str` is rejected; encode text explicitly before passing it. Non-contiguous
buffers are invalid.

`mapped_region_datacopy_region2h()` returns a new `bytes` object.

## Datacopy Semantics

The datacopy APIs move raw bytes between a caller-provided host buffer and the
mapped region data area:

```text
datacopy_h2region:
    host buffer -> region data[offset:offset+nbytes]

datacopy_region2h:
    region data[offset:offset+nbytes] -> host buffer
```

The data area is raw bytes. Simpler does not interpret offsets inside it, does
not construct tensor descriptors, and does not attach protocol meaning to a
range. The caller's protocol decides which offsets contain inputs, outputs,
headers, or message payloads.

Bounds are checked against the configured `data_bytes`. A zero-length copy at
`offset == data_bytes` is valid; a non-zero copy past the end is invalid.

Datacopy does not wait, notify, check protocol phase, update ring metadata, or
publish TensorMap dependencies. Protocols compose the primitives explicitly:

```text
producer write = datacopy_h2region + notify
consumer read  = wait + datacopy_region2h
```

Datacopy alone is not a synchronization boundary. Visibility to the other
participant is established by composing datacopy with the signal protocol
described below.

For direct L2 calls, the runtime can copy through the owner process's mapped
host view. For L3 parent calls, the same Python method is proxied to the chip
child:

```text
h2region:
    parent buffer -> child request payload -> child mapped region

region2h:
    child mapped region -> child reply payload -> parent bytes
```

The proxy path is an implementation detail. The Python contract is the same
for L2 and L3.

## Signal Semantics

Signal slots provide lightweight phase or sequence synchronization:

```text
notify(signal_id, value)
    publish value to signal[signal_id]

wait(signal_id, target, timeout_us)
    complete when observed signal[signal_id] >= target
    otherwise raise or return a timeout result
```

Signal values are `uint32_t`. A signal slot is best treated as a phase word
for bounded protocol epochs, not as a long-lived channel sequence counter.
Higher-level protocols that need long-running head, tail, or sequence values
should define their own metadata in the mapped data area.

Signal values should be monotonic within one protocol epoch. Wrap-around
handling is not part of this primitive.

`wait` has two modes:

- `timeout_us == 0`: non-blocking probe.
- `timeout_us > 0`: bounded wait.

There is no infinite wait mode.

All signal slots start at zero. Therefore a non-blocking wait for target zero
can succeed immediately.

`device_signal_ptr` points to the device-visible signal slot array. Device code
may use the documented signal layout directly. For example, a kernel can poll
signal slot 0, read input data, write output data, and then publish signal
slot 1.

### Memory Ordering

`notify` is a release publication point for writes sequenced before it. `wait`
is an acquire observation point for reads sequenced after it.

For CPU produces / NPU consumes:

```text
host datacopy_h2region(...)
host notify(signal_id, seq)
device wait or poll signal_id >= seq
device reads data
```

If device code observes `signal_id >= seq`, device reads after that
observation must see host writes completed before the matching `notify`.

For NPU produces / CPU consumes:

```text
device writes data
device publishes signal_id = seq
host wait(signal_id, seq, timeout_us)
host datacopy_region2h(...)
```

If host wait succeeds, host reads after wait must see device writes completed
before the matching device signal publication.

Device code that accesses signal slots directly must preserve the same
ordering contract: polling a signal that reaches the target is an acquire
operation, and publishing a signal after data writes is a release operation.

## Relationship To Task Tensor Payloads

Simpler's normal task tensor payload path is built around `TaskArgs` and
`ContinuousTensor`. It is task-scoped and tensor-oriented:

1. The user adds a `ContinuousTensor` to `TaskArgs`.
2. The task is dispatched to a chip child.
3. The chip runtime prepares device-side task arguments.
4. For ordinary tensors, the runtime allocates device memory and copies from
   the host pointer in `ContinuousTensor.data`.
5. The runtime replaces the tensor's data pointer with the device pointer
   before launching device orchestration and kernels.
6. During validation or copy-back, recorded tensor pairs can be copied from
   device memory back to the original host pointer.

That path is convenient for normal kernel inputs and outputs. The runtime owns
the per-task tensor staging details, and the user describes tensors rather than
explicit data movement phases.

`child_memory=True` is an opt-out from that automatic staging path. When a
`ContinuousTensor` is marked as child memory, the chip runtime treats
`ContinuousTensor.data` as an existing child-managed device pointer. It passes
the tensor through without allocating new device memory and without staging the
contents again. The caller is responsible for allocating and populating the
device buffer, commonly through `orch.malloc` plus `orch.copy_to`.

`HostDeviceMappedRegion` is different from both:

- Ordinary task tensor: `ContinuousTensor` host pointer, implicit staging and
  optional copy-back around a task, task/runtime-managed lifetime, TensorMap
  dependencies for synchronization.
- `child_memory=True` tensor: existing device pointer, caller-managed copies,
  caller/child-managed lifetime, TensorMap can still see the tensor argument.
- Mapped region: data offsets plus signal slots, explicit datacopy, explicit
  open/close on a chip-owned region, explicit notify/wait.

### Difference From `copy_to_device()`

`copy_to_device()` copies from a host buffer into ordinary device memory. It is
used by the task runtime to stage tensor payloads before execution, and it is
also exposed through worker/orchestrator copy helpers for manually managed
device buffers.

Mapped-region datacopy targets the mapped region's data area, not an arbitrary
device allocation. It is paired with `mapped_region_info()`, which exposes
device-side views of the region, and with signal slots that let a protocol
publish readiness or completion.

In short:

```text
copy_to_device:
    host buffer -> device allocation

mapped_region_datacopy_h2region:
    host buffer -> chip-owned mapped-region data area
```

The mapped-region path is not a replacement for tensor staging. It is the
primitive to use when host and device need a persistent data area plus explicit
synchronization semantics.

### Difference From `child_memory=True`

`child_memory=True` changes how the task runtime interprets a tensor argument.
It says: this `ContinuousTensor.data` value is already a valid child-side
device pointer, so the runtime should not allocate, copy, or free it as an
ordinary task tensor.

Mapped regions can provide such a pointer, but they do not by themselves make
a tensor. A caller may wrap `info.device_data_ptr` in a
`ContinuousTensor(..., child_memory=True)` when a kernel expects tensor
metadata. The mapped region still owns the backing allocation and signal
slots; `child_memory=True` only prevents the task runtime from trying to stage
that pointer again.

This composition is useful for a kernel-facing data path:

```text
host:
    region = open_mapped_region(...)
    info = mapped_region_info(region)
    mapped_region_datacopy_h2region(region, 0, input_bytes)
    mapped_region_notify(region, 0, 1)

task args:
    tensor = ContinuousTensor.make(
        info.device_data_ptr,
        shape,
        dtype,
        child_memory=True,
    )
    args.add_tensor(tensor, TensorArgType.NO_DEP)
    args.add_scalar(info.device_signal_ptr)

device:
    wait or poll signal[0]
    read or write data through device_data_ptr
    publish signal[1]

host:
    mapped_region_wait(region, 1, 1, timeout_us)
    output = mapped_region_datacopy_region2h(region, output_offset, nbytes)
```

The important boundary is that `child_memory=True` is a task-argument staging
flag, while `HostDeviceMappedRegion` is an allocation, address exposure,
datacopy, and signal primitive.

### Choosing A Data Path

Use ordinary tensors for standard task input/output payloads. Use
`child_memory=True` for manually allocated device buffers that should be passed
as tensors without automatic staging. Use `HostDeviceMappedRegion` when a
protocol needs a CPU/NPU-visible data area, persistent lifetime, explicit
byte-level datacopy, and signal slots.

These choices can be combined. A mapped region can provide the backing device
address for a `child_memory=True` tensor, while the mapped-region signal slots
provide the protocol ordering.

Mapped-region datacopy and signal operations do not publish TensorMap
dependencies and do not replace `TensorArgType` dependency tags. If a
mapped-region-backed tensor is submitted through `TaskArgs`, choose the tensor
tag deliberately. `NO_DEP` is usually the right tag when synchronization is
handled by the mapped-region signal protocol.

## L2 Example

This is the direct one-chip shape. It opens one region, reuses it across
iterations, and passes the device-visible addresses to a chip callable:

```python
worker = Worker(
    level=2,
    platform="a2a3sim",
    runtime="tensormap_and_ringbuffer",
    device_id=0,
)
worker.init()

region = worker.open_mapped_region(data_bytes * 2, signal_count=2)
info = worker.mapped_region_info(region)

for seq in range(1, 11):
    worker.mapped_region_datacopy_h2region(region, 0, make_payload(seq))
    worker.mapped_region_notify(region, 0, seq)

    args = TaskArgs()
    args.add_scalar(info.device_data_ptr)
    args.add_scalar(info.device_signal_ptr)
    args.add_scalar(seq)
    args.add_scalar(data_bytes)
    worker.run(chip_cid, args, cfg)

    worker.mapped_region_wait(region, 1, seq, 1_000_000)
    out = worker.mapped_region_datacopy_region2h(
        region,
        data_bytes,
        data_bytes,
    )

worker.close_mapped_region(region)
worker.close()
```

## L3 Example

In L3, the parent `Worker` may have multiple chip children. `worker_id`
selects the chip child that owns the mapped region:

```python
worker = Worker(
    level=3,
    device_ids=[0, 1],
    platform="a2a3sim",
    runtime="tensormap_and_ringbuffer",
)
worker.init()

region0 = worker.open_mapped_region(
    data_bytes,
    signal_count=2,
    worker_id=0,
)
info0 = worker.mapped_region_info(region0)

region1 = worker.open_mapped_region(
    data_bytes,
    signal_count=2,
    worker_id=1,
)
info1 = worker.mapped_region_info(region1)
```

Each region belongs to exactly one chip child. Do not pass a region opened for
`worker_id=0` to operations for `worker_id=1`, and do not pass its device
pointers to a task running on a different chip unless a higher-level protocol
explicitly supports that.

## Platform Support

Mapped regions are available on:

- `a2a3sim`
- `a5sim`
- `a2a3` onboard

`a5` onboard currently reports mapped regions as unsupported.

The portable contract is the public Python behavior described here: raw byte
datacopy, explicit signal notify/wait, masked host pointers, opaque handles,
and device-visible addresses suitable for task arguments.

## Example Location

The round-trip example lives at:

```text
examples/a2a3/tensormap_and_ringbuffer/host_device_mapped_region_round_trip/
```

Run it on simulation with:

```bash
cd examples/a2a3/tensormap_and_ringbuffer/host_device_mapped_region_round_trip
python main.py -p a2a3sim -d 0
```

Run it on a2a3 hardware with:

```bash
cd examples/a2a3/tensormap_and_ringbuffer/host_device_mapped_region_round_trip
python main.py -p a2a3 -d 0
```
