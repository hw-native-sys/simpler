# Python API reference

The surface you write against, hand-maintained: what `**config` keys `Worker`
accepts, `CallConfig` defaults, and the argument-order footguns a generator
cannot state. For the complete generated listing of every public symbol and
signature, see the API pages on the
[documentation site](https://www.pypto.ai/simpler/user/reference/api/worker/).
Treat the source as authoritative when the two disagree, and fix this page in the
same change.

`Worker` is available from the package root; the remaining task and callable
types live in `simpler.task_interface`, and the tracing helpers in
`simpler.trace`. All of them resolve on first access, so `import simpler` alone
stays cheap and does not require the `_task_interface` extension.

```python
from simpler import Worker, trace     # or: from simpler.worker import Worker
from simpler.task_interface import (
    ArgDirection, CallConfig, ChipCallable, CoreCallable,
    DataType, TaskArgs, TaskHandle, Tensor, TensorArgType,
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
| `enable_sdma` | a2a3 | provisions the SDMA workspace; defaults to `False` |
| `workspace_budget_bytes` | onboard L2 | Optional positive byte limit on internal retained temporary storage and three pooled arenas; both owned and borrowed contexts manage these regions by default |
| `heap_ring_size` | all | heap ring sizing |
| `remote_heap_ring_size`, `remote_session_timeout_s` | L4 | remote-session sizing and timeout |

`level` selects the topology: `2` is one chip, `>= 3` is hierarchical. Anything
else raises. Remote-worker and remote-memory calls require `level >= 4`.

### Lifecycle

| Method | Notes |
| ------ | ----- |
| `register(target, *, workers=None) -> CallableHandle` | Accepts a `ChipCallable` or, at L3+, a Python callable. Pre-init registrations enter the startup snapshot; post-init registration installs the callable on eligible live targets before returning |
| `unregister(handle_or_slot)` | Releases a registration |
| `add_worker(worker) -> int` | Attaches a child worker; returns its id |
| `add_remote_worker(spec: RemoteWorkerSpec) -> int` | L4; see the remote-L3 design doc |
| `init(prewarm_config=None, *, borrowed_context_generation=None)` | Resolves binaries; owns or borrows the direct L2 device context, or starts children |
| `close()` | Retires runtime resources and children; resets only an owned device. Put it in a `finally` |
| `session_cleanup_reports() -> dict[int, SessionCleanupReport]` | What closing each remote L3 session established, keyed by remote worker id. Keeps three events apart: whether the peer read the shutdown command, whether its process exited and with what status, and what its own release returned. `UNKNOWN` means the route cannot see that far, not that nothing happened |

### Borrowed device context and caller ordering

For direct onboard L2, `worker.init(borrowed_context_generation=token)` borrows
an already-current RTS context. `token` is a nonzero uint64 supplied by the caller,
which must keep that context alive and use a new token for a new context generation.
It is not a reset detector or an allocation-ownership certificate. Simulator and
SDMA options are rejected for borrowed initialization.

Both onboard ownership policies install the same internal workspace accounting.
`workspace_budget_bytes` optionally limits the retained temporary buffer and
three pooled arenas, including retained generations. It excludes caller tensors,
registered code, diagnostics, RTS argument blocks and provider memory; it is not
a device-wide memory ceiling. An absent budget imposes no byte limit. A budget
refusal whose native preparation rolls back completely releases this invocation's
Buffer references. The same Worker can accept a smaller valid call; the Buffer
can close once all its other users retire. A rollback without complete release
evidence retains its references and stops further native admission.

`worker.submit(handle, args, caller_stream=stream)` uses a nonzero integer RTS
stream handle for this invocation; `run` accepts the same keyword. The stream is
required for borrowed execution and may also be supplied to owned L2 execution.
L3+/Remote do not accept these process-local context/stream values. Registration,
TaskArgs, Buffer/Tensor and RunHandle remain the same public interfaces.

Borrowed submission accepts DEVICE operands and explicit HOST/NONE controls; it
rejects implicit tensor H2D. Keep external allocations, context and stream alive
through retirement. Successful submit has already enqueued both kernels and the
caller exit dependency, so work appended to that caller stream is ordered after
the operator. Caller synchronization can establish device results; `RunHandle.wait`
still performs runtime retirement. Capture is rejected before run/source state is
published. Borrowed close does not reset the caller device or destroy its stream.

Frameworks with a separate host emission queue must place the submission in
that queue's callback; passing a stream integer alone does not order a Python
call against framework work still waiting for host emission. The optional
[A3 framework eager reference](../../../examples/workers/l2/framework_eager/README.md)
uses the same public Worker for TMR and HBG, preserves frame/storage owners and
uses the final framework completion boundary before runtime retirement. It
demonstrates ordering, not overlapping execution: native admission, predecessor
retirement and capacity backpressure can still wait. It
requires matching torch/torch_npu/CANN and does not enable capture submission.

### Memory

| Method | Notes |
| ------ | ----- |
| `malloc(size) -> Buffer` | L2 only; allocates on this worker's chip |
| `alloc_child_tensor(worker_id, shapes, dtype) -> Buffer` | Allocates on an L3 worker's chip child; use `Tensor(handle, shapes=shapes, dtype=dtype)` to name a task argument |
| `free(handle)` | Releases a device `Buffer` returned by either allocation method |
| `copy_to(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | H2D; `dst` is a device `Buffer`, `src` a host `Buffer` from `create_buffer` (at L2, also any torch tensor or writable buffer). `nbytes` defaults to the rest of the host side after `src_offset`, so `copy_to(dst, src)` transfers the whole host backing |
| `copy_from(dst, src, *, dst_offset=0, src_offset=0, nbytes=None)` | D2H; `dst` is the host `Buffer` (at L2, also any writable buffer). Same defaulting, measured from `dst_offset` on the host side |
| `create_buffer(nbytes) -> Buffer` / `Buffer.close()` | Shared host backing this Worker owns; build a view over `handle.shm.buf`, name it on the wire with `Tensor(handle, shapes=shapes, dtype=dtype)` |
| `remote_malloc` / `remote_free` / `remote_copy_to` / `remote_copy_from` / `remote_export` / `remote_import` / `remote_release_import` | L4 only |

A partial update names the **whole allocation plus an offset** — `copy_to(dev, src,
dst_offset=32, nbytes=16)`. A handle rebuilt at `base + 32` is not an interior view of that
allocation; it is a different canonical identity that names no allocation at all, and is refused.
Both offsets are bounded together with the length against the *registered* extent, so an offset
cannot walk a legal-looking length past the end.

### Execution

| Method | Notes |
| ------ | ----- |
| `run(callable, args=None, config=None, *, caller_stream=None) -> None` | Blocks until the run completes |
| `submit(callable, args=None, config=None, *, caller_stream=None) -> RunHandle` | Non-blocking |
| `live_domains() -> dict[str, CommDomainHandle]` | Currently allocated communication domains |

`RunHandle` exposes the `done` property, `wait(timeout=None)`, and
`result(timeout=None)`.

Direct L2 handles validate the execution context inside the native wait.
`ExecutionContextRefused`, imported from `simpler.task_interface`, is a
`RuntimeError` subclass for current-context preflight refusal. It supplies no
completion verdict: the handle, its argument snapshot and its Buffer uses remain
intact; restore the required context before retrying.

`RunPreparationRefused`, also from `simpler.task_interface`, is a terminal
`RuntimeError` subclass stating that this invocation enqueued no operator kernel and
released its native input bindings and preparation resources. It can surface
from submit, or from wait when preparation was deferred. It also covers launch
setup that explicitly returned `NotStarted`, once finalize has released all
bindings and preparation resources. The original launch error code and message
remain observable, and a usable Worker can retry with the same Buffer. A backend
exception or absent active-execution pointer alone supplies no such proof.
The Worker releases
only that invocation's source references; another live run still protects a
shared Buffer. It does not certify that the environment accepts another call,
or undo host orchestration effects and completed synchronous copies.
Unclassified native errors and partial launches retain unproven uses; error
messages do not determine the classification. Borrowed-context initialization
is available for direct onboard L2 through
`Worker.init(borrowed_context_generation=...)`. Neither refusal type proves
that an external allocation belongs to that context.

READY Worker operations validate through the shared operation lease, outside the
lifecycle lock and before their bodies run. Direct L2 operations check the native
current-context predicate; L3+ operations retain their own child/environment
contracts. This covers allocation, free, copies, registration and future leased
APIs. Only `device_location`, a source-descriptor query with no device operation,
explicitly skips current-context validation. Close checks separately before its
CLOSED admission fence and teardown-attempt publication. An unaccepted context
refusal preserves READY, callable handles and source uses; restore the context
before retrying. Once close has been accepted, a teardown failure retains CLOSED
and its cleanup debt. Re-entering close in the required context can retry that
debt; it does not reopen the Worker. Native registration and the legacy
ChipWorker registry also check before their state changes. Direct native
allocation, free, copies, device-memory queries and diagnostic flush validate
before calling the backend; cached host counters and descriptors do not perform
device operations.

At L2 the `callable` is a registered `ChipCallable` handle. At L3+ the top-level
callable is a **Python orchestration function** `f(orch, args, cfg)`, where
`orch` is the `Orchestrator`:

| Method | Notes |
| ------ | ----- |
| `submit_next_level(callable_handle, args, config=None, *, worker: int) -> TaskHandle` | Hands a callable to one exact NEXT_LEVEL child and returns an opaque handle for explicit task dependencies |
| `submit_next_level_group(callable_handle, args_list, config=None, *, workers: list[int]) -> TaskHandle` | Submits one group DAG node and returns its opaque handle |
| `submit_sub(callable_handle, args=None)` | Schedules a registered host-side Python callable |
| `allocate_domain(*, name, members, window_size, buffers=(), retain_after_run=False)` | L3/L4 creation returning `CommDomainHandle`; legacy L3 `workers=` and chip-id indexing remain supported |
| `get_or_allocate_domain(*, name, members, window_size, buffers=(), retain_after_run=False)` | Reuse the same live handle after full configuration validation, or create it |
| `get_domain(name)` / `release_domain(handle)` | Creator-local lookup / deferred owning release |
| `copy_to_domain(handle, rank, data, ...)` / `copy_from_domain(handle, rank, nbytes, ...)` | Synchronous bounded staging before task submission; L3 requires declared slices |
| `alloc_child_tensor(worker_id, shapes, dtype) -> Buffer` | Delegates allocation to the owning Worker; target chip memory is named by the returned handle |

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
orchestration submits. `ChipCallable` exposes `binary_size` and
`scalar_count`. `scalar_count` counts the signature's `SCALAR` entries,
including for callables loaded from cached bytes. The tensor count is
`sig_count - scalar_count`.

Public `Worker` calls use `TaskArgs` containing address-free `Tensor` views at
every level. The L2 leaf resolves those views into the internal
`ChipStorageTaskArgs` / `ChipTensor` representation; callers do not pass that
internal representation to `Worker.run()`.

Direct L2 `Worker` and forked chip children each own a native chip binding.
They share runtime initialization; the parent Worker owns public callable handles,
and each child resolves them to its local native slots.
The lower-level public `ChipWorker` remains a separate compatibility entry;
its handles and Python registry are not used by `Worker` internally.

The lower-level `ChipWorker.run(handle, args)` accepts the same `TaskArgs` for
scalars and contiguous HOST/H2D tensors. DEVICE Buffer arguments use
`Worker(level=2).submit`, which supplies live-source validation. Keep host backing
storage alive until the synchronous call returns; after a native-call error,
keep it alive until `finalize()` succeeds. ChipWorker retains imported mappings
and refuses further public runs until that teardown. `ChipStorageTaskArgs`
remains accepted only as a resolved-address compatibility path. See
[invocation binding](../../buffer-abi.md#direct-l2-invocation-binding).

For L3+ graph construction, `TaskArgs.add_dep(*handles)` adds `WAIT | RETAIN`
edges: each consumer waits for its producers and keeps their task-owned
resources alive until it completes. `TaskArgs.add_dep_wait(*handles)` adds
ordering-only `WAIT` edges. Handles must come from a NEXT_LEVEL submit in the
same orchestration run; they are opaque and cannot be constructed by the
caller.

```python
args = TaskArgs()
args.add_tensor(Tensor(device_buffer, shapes=(rows, cols), dtype=DataType.FLOAT32), TensorArgType.INPUT)
```

`device_buffer` is a `Buffer` returned by `malloc`, `alloc_child_tensor`, or
`Buffer.wrap` for a local direct-L2 device source.
`Tensor(buffer, *, shapes, dtype, strides=None, byte_offset=0)` validates a view
without allocation or transfer. `buffer.tensor(...)` remains a compatibility
forwarder. Views carry descriptor values; they do not own the backing allocation.
Add tensors **in signature order**, before any scalars. Use `INPUT` for an
input, `OUTPUT_EXISTING` for a caller-allocated output, and `INOUT` for an
input/output view. `DataType` carries the element types.

### Eager workspace capacity

Owned and borrowed direct-L2 calls reuse internal arena capacity and grow it
when a later invocation needs more storage. An empty first HBG call does not
require a nonempty warmup or a new Worker before submitting a larger graph.
A failed replacement allocation preserves the existing arena addresses and
capacities; public submission errors retain their existing cleanup requirements.

Each native run retains its bank through retirement. Concurrent HBG preparation
uses a different bank, while shared-bank runtimes require compatible storage
before preparing ahead of an executing run. This does not make capture/replay
supported: captured graphs require their own address-retention contract.

### Borrowed device storage

The example uses an initialized direct L2 Worker and a caller stream belonging
to its current RTS context. Supplying the stream works with owned initialization
and is required with borrowed initialization.

```python
from simpler.buffer import Buffer

source = Buffer.wrap(address=ptr, nbytes=capacity, location=worker.device_location)
args = TaskArgs()
args.add_tensor(Tensor(source, shapes=(count,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
worker.submit(callable_handle, args, caller_stream=stream).result()
source.close()
```

`worker.device_location` is available after direct L2 initialization. The token
identifies that native environment binding, including its process, incarnation
and context generation; it cannot be replaced by a device ordinal or used on
another Worker. Source checks reject a changed or successfully retired native
binding even if the Python facade still reports READY. Reading the location
performs no RTS call; current-context validation remains an execution precondition. `Buffer.wrap(..., access=...)` accepts
an existing `AccessMode` grant (default `READWRITE`) and registers caller-owned
storage without consumer attachment. Submission performs that attachment.

The caller owns the allocation and guarantees its real device, capacity and
lifetime. `source.close()` revokes an idle registration without freeing memory;
it refuses in-flight or unproven completion. `Worker.free` and Worker copy
operations do not manage this external allocation. Use its allocator for IO
and physical release. HOST and remote wrapping are outside this interface.
The provider may use owned or borrowed RTS initialization; this does not transfer
physical ownership of the wrapped allocation. See the [source lifetime contract](../../buffer-abi.md#borrowed-device-sources-at-direct-l2)
for failure and Program Worker teardown behavior.

## `CallConfig`

`CallConfig()` defaults are fine for an ordinary run.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `aicpu_thread_num` | `0` | AICPU threads for this run; `0` selects the architecture default |
| `enable_chip_swimlane` | `0` | `0` off; `1`–`4` select detail. L2 and same-host L3; a lower level simply records fewer device streams, and cross-Rank merging works at any of them |
| `enable_dump_args` | `0` | Capture per-task arguments |
| `enable_pmu` | `0` | `0` off; `>0` selects the event type |
| `enable_dep_gen` | `0` | Emit the dependency graph |
| `enable_scope_stats` | `0` | Writes `<output_prefix>/scope_stats/scope_stats.jsonl` |
| `output_prefix` | `""` | **Required whenever any diagnostic is enabled** |
| `runtime_env` | — | TRB uses `ring_task_window`, `ring_heap`, and `ring_dep_pool`. HBG uses `ring_task_window[0]` for graph task capacity; its graph heap is sized after orchestration |

`validate()` runs at every submit/run entry point and throws if a diagnostic is
on without `output_prefix`, or if a ring override breaks the ring's constraints.

## `simpler.trace`

Puts your own phases on the same host timeline the runtime's spans land on, on
the same clock and in the same log. One producer, one span, one gate:

| Name | Use |
| ---- | --- |
| `span(name, **attributes)` | A timed region, as a `with` block or a decorator. Emits even when the block raises, and does not swallow the exception. A marker is a span whose body does no work — there is no separate zero-duration call |
| `producer(name)` | A `Producer` with the same `span` / `enabled`, for a library that wants its own name instead of the default `app` |
| `enabled()` | Whether spans are being emitted right now — ask before building attributes that cost more than the span |

```python
from simpler import trace

with trace.span("my_phase", batch=16, layer=3):
    ...
```

Names are prefixed `ext.<producer>.`, so your spans can never be mistaken for
the runtime's own. See
[host trace markers](../../dfx/host-trace.md#emitting-your-own-spans-with-simplertrace)
for what the namespace guarantees in each view, the cost of one attempt, and how
to read the records back.

## `simpler_setup`

Compilation and test scaffolding. Exported from the package root:

| Name | Use |
| ---- | --- |
| `KernelCompiler` | `compile_incore(source_path, core_type, pto_isa_root, extra_include_dirs)`, `compile_orchestration(runtime_name, source_path)`, `get_orchestration_include_dirs(runtime)` |
| `ensure_pto_isa_root()` | Clones/updates the pinned pto-isa checkout and returns its path |
| `extract_text_section(binary)` | Required on hardware platforms before wrapping a kernel `.o` |
| `scene_test(level, runtime)` / `SceneTestCase` | The decorator and base class for declarative examples and tests |
| `TensorArg`, `Scalar`, `TaskArgsBuilder`, `CallableNamespace` | Scene-test arg construction; `TensorArg` is separate from the runtime `Tensor` type |
| `make_chip_tensor_arg`, `torch_dtype_to_datatype` | torch interop |
| `parse_platform` | Platform string parsing |
| `RuntimeBuilder` | Runtime build orchestration |

Analysis CLIs live under `simpler_setup.tools`; see
[command-line flags and tools](cli.md).

## See also

- [How-to: write and run a kernel](../how-to/write-and-run-a-kernel.md)
- [Task Flow](../../task-flow.md) — how these handles travel through the runtime
- [Communication Domains](../../comm-domain.md) — `allocate_domain` semantics

### Host controls in HBG orchestration

`Tensor` describes a view; transfer is chosen separately for each `TaskArgs` entry:

```python
from simpler.task_interface import TensorTransfer

args.add_tensor(control, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
args.add_tensor(control, TensorArgType.INPUT, transfer=TensorTransfer.H2D)
```

With HOST backing, the first entry stays on the host for HBG `get_tensor_data`.
The second becomes a DEVICE operand after Program copies it. Pass each entry to
the matching consumer; host access to DEVICE and device access to HOST are errors.
Host `set_tensor_data` requires an OUT/INOUT declaration and changes only the HOST
binding. It does not upload that write to the separate device binding. The caller
keeps the backing alive through completion and explicitly waits/copies any prior
device result before submitting a host consumer. TMR device orchestration rejects
HOST/NONE. Neither path creates a cross-side mapped view.
