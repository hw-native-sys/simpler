# Python API reference

The surface you write against, hand-maintained: what `**config` keys `Worker`
accepts, `CallConfig` defaults, and the argument-order footguns a generator
cannot state. For the complete generated listing of every public symbol and
signature, see the API pages on the
[documentation site](https://hw-native-sys.github.io/simpler/user/reference/api/worker/).
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
    DataType, TaskArgs, TaskHandle, TensorArgType,
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
| `init(prewarm_config=None)` | Resolves runtime binaries, opens the device, forks children. First place setup errors appear |
| `close()` | Releases the device and reaps children. Put it in a `finally` — a skipped `close()` leaves the device held |

### Memory

| Method | Notes |
| ------ | ----- |
| `malloc(size) -> Buffer` | L2 only; allocates on this worker's chip |
| `alloc_child_tensor(worker_id, shapes, dtype) -> Buffer` | Allocates on an L3 worker's chip child; use `handle.tensor(shapes, dtype)` to name a task argument |
| `free(handle)` | Releases a device `Buffer` returned by either allocation method |
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
orchestration submits. `ChipCallable` exposes `binary_size`.

Public `Worker` calls use `TaskArgs` containing address-free `Tensor` views at
every level. The L2 leaf resolves those views into the internal
`ChipStorageTaskArgs` / `ChipTensor` representation; callers do not pass that
internal representation to `Worker.run()`.

For L3+ graph construction, `TaskArgs.add_dep(*handles)` adds `WAIT | RETAIN`
edges: each consumer waits for its producers and keeps their task-owned
resources alive until it completes. `TaskArgs.add_dep_wait(*handles)` adds
ordering-only `WAIT` edges. Handles must come from a NEXT_LEVEL submit in the
same orchestration run; they are opaque and cannot be constructed by the
caller.

```python
args = TaskArgs()
args.add_tensor(device_buffer.tensor((rows, cols), DataType.FLOAT32), TensorArgType.INPUT)
```

`device_buffer` is a `Buffer` returned by `malloc` or `alloc_child_tensor`.
Add tensors **in signature order**, before any scalars. Use `INPUT` for an
input, `OUTPUT_EXISTING` for a caller-allocated output, and `INOUT` for an
input/output view. `DataType` carries the element types.

## `CallConfig`

`CallConfig()` defaults are fine for an ordinary run.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `aicpu_thread_num` | `0` | AICPU threads for this run; `0` selects the architecture default |
| `enable_chip_swimlane` | `0` | `0` off; `1`–`4` select detail. L2 and same-host L3; cross-rank merging requires detail level `4` |
| `enable_dump_args` | `0` | Capture per-task arguments |
| `enable_pmu` | `0` | `0` off; `>0` selects the event type |
| `enable_dep_gen` | `0` | Emit the dependency graph |
| `enable_scope_stats` | `0` | Writes `<output_prefix>/scope_stats/scope_stats.jsonl` |
| `capture_clock_anchors` | `False` | Anchors the Host and Device clocks so device timestamps land on an absolute Host timeline. Set by the ChipWorker child for an L3 chip-swimlane capture; not a caller knob |
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
