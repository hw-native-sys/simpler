# Python API reference

The surface you write against, hand-maintained: what `**config` keys `Worker`
accepts, `CallConfig` defaults, and the argument-order footguns a generator
cannot state. For the complete generated listing of every public symbol and
signature, see the API pages on the
[documentation site](https://hw-native-sys.github.io/simpler/user/reference/api/worker/).
Treat the source as authoritative when the two disagree, and fix this page in the
same change.

`Worker` is available from the package root; the remaining task and callable
types live in `simpler.task_interface`. Both resolve on first access, so
`import simpler` alone stays cheap and does not require the `_task_interface`
extension.

```python
from simpler import Worker, register_chip_control_extension
from simpler.task_interface import (
    ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs,
    ChipTensor, CoreCallable, DataType, TaskArgs, TaskHandle,
)
from simpler_setup import KernelCompiler, SceneTestCase, scene_test
```

## `Worker`

```python
Worker(level: int, **config)
```

`level` is the only declared parameter; everything else is a keyword collected
into `**config` and validated later. The recognized keys:

| Key | Applies to | Meaning |
| --- | ---------- | ------- |
| `platform` | all | `a2a3`, `a2a3sim`, `a5`, `a5sim` |
| `runtime` | all | `tensormap_and_ringbuffer` or `host_build_graph` |
| `device_id` | L2 | the single chip this worker drives |
| `device_ids` | L3+ | one chip child process per entry |
| `num_sub_workers` | L3+ | host-side Python callables to fork |
| `py_control_timeout_s` | L3+ | finite timeout for Python control-plane operations; defaults to 30 seconds |
| `external_transfer_max_pending` | L3 | maximum unretired external transfers across all local chips; defaults to 2 |
| `external_transfer_max_bytes` | L3 | maximum sum of in-flight transfer span lengths; defaults to 1 GiB |
| `enable_sdma` | a2a3 | provisions the SDMA workspace; defaults to `False` |
| `heap_ring_size` | all | heap ring sizing |
| `remote_heap_ring_size`, `remote_session_timeout_s` | L4 | remote-session sizing and timeout |

`level` selects the topology: `2` is one chip, `>= 3` is hierarchical. Anything
else raises. Remote-worker and remote-memory calls require `level >= 4`.

### Lifecycle

| Method | Notes |
| ------ | ----- |
| `register(target, *, workers=None) -> CallableHandle` | **Before `init()`.** Accepts a `ChipCallable` or, at L3+, a Python callable |
| `unregister(handle_or_slot)` | Releases a registration |
| `add_worker(worker) -> int` | Attaches a child worker; returns its id |
| `add_remote_worker(spec: RemoteWorkerSpec) -> int` | L4; see the remote-L3 design doc |
| `init(prewarm_config=None)` | Resolves runtime binaries, opens the device, forks children. First place setup errors appear |
| `close()` | Releases the device and reaps children. Put it in a `finally` — a skipped `close()` leaves the device held |

### Chip-control extensions

Chip-control extensions let a trusted integration run a short synchronous
control operation inside every chip-child process owned by one L3 Worker. They
are not available on L2 or L4+ Workers.

```python
def handler(chip_worker, payload: bytes, device_id: int):
    ...
    return None

register_chip_control_extension("my-extension", handler)
worker = Worker(level=3, ..., py_control_timeout_s=30.0)
worker.init()
worker.run_chip_control_extension("my-extension", b"request")
```

Registration must happen before that Worker's `init()`. The Worker snapshots
the process-wide registry when startup begins, and every chip child receives
the same snapshot. Registering another name later affects only Workers whose
`init()` has not started.

The handler receives the child-local `ChipWorker`, an immutable payload, and
the physical device id. Returning `None` or another false value succeeds;
returning a true value reports its string form as an error. Raising also fails
the operation. Handlers run synchronously while ordinary runs are excluded, so
they must return promptly or move long-running work to an asynchronous service.
`timeout_s=None` uses the Worker's finite `py_control_timeout_s`; an explicit
timeout must also be positive and finite.

The call broadcasts concurrently and returns only after every chip child has
responded. If some handlers succeed and another fails, the successful effects
are not rolled back; the caller receives one aggregated `RuntimeError` naming
the first reported child error.

Asynchronous device access must use the managed external-transfer API below.
A chip-control extension's admission fence ends when its handler returns; it
does not protect background DMA launched by that handler.

### Managed external transfers

An external transfer names registered allocations, not caller-supplied device
pointers. The Runtime validates ownership, access rights, and byte extents,
routes to the allocation's chip, binds the transfer thread to that device,
and retains the operation until the provider certifies that device access has
stopped. Only local L3 `DEVICE_MALLOC` allocations are supported; communication
windows and remote workers are rejected.

```python
from simpler import (
    ExternalBufferRange, ExternalTransferResult,
    register_external_transfer_provider,
)
from simpler.buffer import AccessMode

def storage_provider(request, cancellation):
    if cancellation.requested:
        return ExternalTransferResult(False, error="cancelled before access")
    # A trusted adapter performs and fences its device access here.
    # request.buffers contains checked address/nbytes/access spans.
    # request.payload contains adapter-specific metadata, not KV bytes.
    return ExternalTransferResult()

register_external_transfer_provider("storage", storage_provider)  # before worker.init()
# After initialization and allocation of kv_buffer:
transfer = worker.submit_external_transfer(
    "storage",
    (ExternalBufferRange(kv_buffer, offset=0, nbytes=4096, access=AccessMode.WRITE),),
    payload=b"object metadata",
)
transfer.wait(timeout=30)
```

The provider is trusted native integration code, **not a sandbox**. It receives
no `ChipWorker`, allocator, or runtime stream. It must not reset the device,
change its runtime-owned context/streams, free supplied memory, or retain
addresses/registrations beyond its result. Adapter-owned streams, memory
registration and SDK work must be drained and released before returning.
Internal SDK memory is not included in Runtime's committed-memory accounting.

| Event | Resource semantics |
| ----- | ------------------ |
| `ExternalTransferResult(succeeded=True)` | All access stopped; `wait()` returns the bytes payload |
| `ExternalTransferResult(succeeded=False, error=...)` | All access stopped; `wait()` raises `RuntimeError` |
| Unexpected exception or invalid result | Quiescence unconfirmed; `wait()` raises `ExternalTransferUnconfirmedError`, buffers stay retained |
| `wait(timeout=...)` expires | Only the wait expires; buffers stay retained |
| `request_cancel()` | Cooperative flag only; no thread interruption or DMA cancellation |
| Submission acknowledgement is lost | `ExternalTransferSubmissionError.handle` retains the operation; do not blindly resubmit |

The two exception types are available from `simpler.external_transfer`.
`done()` checks completion without blocking and retires a confirmed result;
it returns `False` for unconfirmed quiescence. `result()` aliases `wait()`.
Handles remain observable after `Worker.close()` has begun. Close drains them
within its existing cleanup budget; if access is still outstanding it leaves
the tree intact and CLOSED. A later close can retry after confirmed completion.
An unconfirmed provider exception requires external recovery; this API does
not force-reset a device or pretend a stopped Python thread proves DMA stopped.

Submission must occur outside orchestration callbacks and after compute runs
complete. While any transfer remains unretired, compute submission and device
control (including copy/free and legacy chip extensions) fail fast. Transfers
may overlap one another across different chips or nonconflicting ranges;
overlapping reads are allowed, overlapping access involving a write is refused.
Serving still owns KV page pinning and reuse. This conservative implementation
does not provide compute/transfer overlap or automatic prefix-cache scheduling.

Capacity checks reject submissions before dispatch. The byte limit counts
submitted span lengths, not whole pinned allocations or SDK staging memory.
Each bounded slot has a fork-inherited event; completion uses a small
shared-memory result and a wakeup, without mailbox polling or KV host staging.
Payloads are limited to 1 MiB, results to 4096 bytes, and requests to 65536 spans.
`timeout_s` bounds the control acknowledgement only and defaults to
`py_control_timeout_s`; it is not a backend DMA deadline. A Mooncake adapter
must translate backend failures into a result only when it can certify no
further access, and publish a cache manifest only after every shard succeeds.

### Memory

| Method | Notes |
| ------ | ----- |
| `malloc(size, worker_id=0) -> int` | Returns a device pointer as an integer |
| `free(ptr, worker_id=0)` | |
| `copy_to(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | H2D; `dst` is a device `Buffer`, `src` a host `Buffer` from `create_buffer` (at L2, also any torch tensor or writable buffer). `nbytes` defaults to the rest of the host side after `src_offset`, so `copy_to(dst, src)` transfers the whole host backing |
| `copy_from(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | D2H; `dst` is the host `Buffer` (at L2, also any writable buffer). Same defaulting, measured from `dst_offset` on the host side |
| `create_buffer(nbytes) -> Buffer` / `Buffer.close()` | Shared host backing this Worker owns; build a view over `handle.shm.buf`, name it on the wire with `handle.tensor(shapes, dtype)` |
| `remote_malloc` / `remote_free` / `remote_copy_to` / `remote_copy_from` / `remote_export` / `remote_import` / `remote_release_import` | L4 only |

A partial update names the **whole allocation plus an offset** — `copy_to(dev, src,
dst_offset=32, nbytes=16)`. A handle rebuilt at `base + 32` is not an interior view of that
allocation; it is a different canonical identity that names no allocation at all, and is refused.
Both offsets are bounded together with the length against the *registered* extent, so an offset
cannot walk a legal-looking length past the end.

### Execution

| Method | Notes |
| ------ | ----- |
| `run(callable, args=None, config=None) -> None` | Blocks until the run completes |
| `submit(callable, args=None, config=None) -> RunHandle` | Non-blocking |
| `live_domains() -> dict[str, CommDomainHandle]` | Currently allocated communication domains |

`RunHandle` exposes `done() -> bool`, `wait(timeout=None)`, and
`result(timeout=None)`.

At L2 the `callable` is a registered `ChipCallable` handle. At L3+ the top-level
callable is a **Python orchestration function** `f(orch, args, cfg)`, where
`orch` is the `Orchestrator`:

| Method | Notes |
| ------ | ----- |
| `submit_next_level(callable_handle, args, config=None, *, worker: int) -> TaskHandle` | Hands a callable to one exact NEXT_LEVEL child and returns an opaque handle for explicit task dependencies |
| `submit_next_level_group(callable_handle, args_list, config=None, *, workers: list[int]) -> TaskHandle` | Submits one group DAG node and returns its opaque handle |
| `submit_sub(callable_handle, args=None)` | Schedules a registered host-side Python callable |
| `allocate_domain(name, workers, window_size, buffers=[...])` | Context manager returning a handle indexed by domain-local rank |
| `malloc(worker_id, size) -> int` | **Argument order is `(worker_id, size)`** — the reverse of `Worker.malloc(size, worker_id=0)` |

## Callables and task args

```python
CoreCallable.build(signature=[ArgDirection...], binary=kernel_bytes)

ChipCallable.build(
    signature=[ArgDirection...],
    func_name="my_orchestration",     # the exported orchestration symbol
    binary=orch_bytes,
    children=[(func_id, core_callable), ...],
)
```

`ArgDirection` is `SCALAR`, `IN`, `OUT`, or `INOUT`. The signature list is
positional and defines the task-arg order. `func_id` must match the id the
orchestration submits. `ChipCallable` exposes `binary_size`.

For L3+ graph construction, `TaskArgs.add_dep(*handles)` adds `WAIT | RETAIN`
edges: each consumer waits for its producers and keeps their task-owned
resources alive until it completes. `TaskArgs.add_dep_wait(*handles)` adds
ordering-only `WAIT` edges. Handles must come from a NEXT_LEVEL submit in the
same orchestration run; they are opaque and cannot be constructed by the
caller.

```python
args = ChipStorageTaskArgs()
args.add_tensor(ChipTensor.make(dev_ptr, (rows, cols), DataType.FLOAT32))
```

Tensors are added **in signature order**. `ChipTensor.make(data_ptr, shape, dtype)`
takes a device pointer; `DataType` carries the element types.

## `CallConfig`

`CallConfig()` defaults are fine for an ordinary run.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `aicpu_thread_num` | `0` | AICPU threads for this run; `0` selects the architecture default |
| `enable_chip_swimlane` | `0` | `0` off; `1`–`4` select detail. L2 only |
| `enable_dump_args` | `0` | Capture per-task arguments |
| `enable_pmu` | `0` | `0` off; `>0` selects the event type |
| `enable_dep_gen` | `0` | Emit the dependency graph |
| `enable_scope_stats` | `0` | Writes `<output_prefix>/scope_stats/scope_stats.jsonl` |
| `output_prefix` | `""` | **Required whenever any diagnostic is enabled** |
| `runtime_env` | — | `ring_task_window`, `ring_heap`, `ring_dep_pool`; `tensormap_and_ringbuffer` only |

`validate()` runs at every submit/run entry point and throws if a diagnostic is
on without `output_prefix`, or if a ring override breaks the ring's constraints.

## `simpler_setup`

Compilation and test scaffolding. Exported from the package root:

| Name | Use |
| ---- | --- |
| `KernelCompiler` | `compile_incore(source_path, core_type, pto_isa_root, extra_include_dirs)`, `compile_orchestration(runtime_name, source_path)`, `get_orchestration_include_dirs(runtime)` |
| `ensure_pto_isa_root()` | Clones/updates the pinned pto-isa checkout and returns its path |
| `extract_text_section(binary)` | Required on hardware platforms before wrapping a kernel `.o` |
| `scene_test(level, runtime)` / `SceneTestCase` | The decorator and base class for declarative examples and tests |
| `Tensor`, `Scalar`, `TaskArgsBuilder`, `CallableNamespace` | Scene-test arg construction |
| `make_chip_tensor_arg`, `torch_dtype_to_datatype` | torch interop |
| `parse_platform` | Platform string parsing |
| `RuntimeBuilder` | Runtime build orchestration |

Analysis CLIs live under `simpler_setup.tools`; see
[command-line flags and tools](cli.md).

## See also

- [How-to: write and run a kernel](../how-to/write-and-run-a-kernel.md)
- [Task Flow](../../task-flow.md) — how these handles travel through the runtime
- [Communication Domains](../../comm-domain.md) — `allocate_domain` semantics
