# Buffer / Tensor — the L3+ memory model

At L3 and above, tasks name their data with typed, self-describing **buffer
handles and views**, not raw pointers. This replaces the legacy "raw pointer +
`child_memory` bool" mechanism with an ABI that carries a canonical identity, a
backend descriptor, and a strided view — so a buffer can be resolved exactly
across the L3→L2 (and L4→L3) boundaries without a side table.

This page is the user-facing how-to. The byte layout itself is pinned by the
`static_assert`s in [`src/common/task_interface/buffer.h`](../src/common/task_interface/buffer.h)
— sizes, field offsets, and enum values all fail the build if they drift, and
[`tests/ut/cpp/types/test_buffer.cpp`](../tests/ut/cpp/types/test_buffer.cpp)
pins them again from the outside.

All three types are that header's C++ structs bound directly, so Python and C++
name one definition of the layout with no encode step between them. Everything
that is host-side by nature — the `Buffer` that owns a POSIX shm, the consumer's
`ImportRegistry` — lives in [`python/simpler/buffer.py`](../python/simpler/buffer.py)
and composes them.

**No bytes cross that binding in either direction.** You build a `Tensor` from its
fields and receive one already decoded; turning wire bytes into a `Tensor` happens
only inside C++, where `validate_tensor` runs on the way out. That is what makes
the validator a gate rather than a convention — there is no second way in to
forget it on.

## Three types

| Type | What it is | Where it lives |
| ---- | ---------- | -------------- |
| **`Buffer`** | An owned backing (POSIX shm / fork-COW / device malloc) with a canonical identity + lifecycle. Stays with the Worker that created it. | owner side (L3+) |
| **`Tensor`** | A **self-describing task argument**: the full buffer descriptor embedded + a strided view `(byte_offset, shapes, strides, dtype)`. The wire element of `TaskArgs`. Carries no materialized address. | `simpler.buffer`, re-exported from `simpler.task_interface` |
| **`ChipTensor`** | A task argument as it arrives at the chip runtime: a resolved address plus a strided view, and nothing else. Exists **only** at the L2 device-runtime boundary. | L2 leaf |
| **`simpler::{hbg,tmr}::Tensor`** | One runtime's working form — a `ChipTensor`'s geometry plus what that runtime decided about it (producing task, overlap version, dependency treatment, derived caches). `Runtime::set_orch_args` adopts each argument into it, on the host. | inside one runtime |

**You never build a `ChipTensor` for the L3+ submit path.** Name a `Tensor` over a buffer and submit it;
the address is resolved on the consuming endpoint, and the C++ orchestration on
the chip receives the resolved form.

**A kernel or orchestration translation unit sees only the last two rows.** A
`#include "tensor.h"` on a runtime's include path supplies `ChipTensor` and that
runtime's own tensor **under the unqualified name `Tensor`**, and neither it nor
`orchestration_api.h` reaches the wire `Tensor` in `task_interface/buffer.h`. Two
headers keep that true: the runtime's entry-arg storage, which needs the argument
template, sits in `entry_args.h` beside `tensor.h` rather than inside it; and
`task_args.h` carries only the template and the L2 ABI alias, while the L3+
`TaskArgs`, the blob codec and the submit checks — the parts whose element *is* the
wire `Tensor` — sit in `task_args_wire.h`.

That separation is what makes the unqualified spelling legal, so it is enforced
rather than assumed: `tests/lint/check_kernel_wire_isolation.py` fails pre-commit on
any new includer of `buffer.h` or `task_args_wire.h` outside the allowlist of places
that genuinely handle the L3+ form. Without it, one added include turns a runtime's
`using Tensor = …` into a redeclaration, and the error surfaces in whichever kernel
happens to pick the header up rather than in the file that added the edge.

> **Status.** `TaskArgs.add_tensor` takes a `Tensor`, and `simpler.task_interface`
> re-exports it: the public submit surface names the type its own submit call accepts.
> The Scope / status section at the end of this page says what is and is not connected.

## Why `Tensor` and `ChipTensor` are two types

The two sit on opposite sides of one resolution step. A `Tensor` *cannot* carry an
address: at submit time none exists — a `POSIX_SHM` backing maps to a different VA
in every process, and a `DEVICE_MALLOC` one is only valid on its owner chip. A
`ChipTensor` *must* carry one; it is what the kernel dereferences. Two ways to
collapse that into a single type were considered and dropped.

### Rejected: merge `Tensor` into `ChipTensor`

Drop `buffer.addr`, add the buffer descriptor, and have the H2D staging step
rewrite the backend tag and body (and mint a fresh identity for the device copy).
That is self-consistent, but it charges the device for host-side fields:

- **Device cost.** The payload embeds its tensors **by value** in the device task
  ring, and the AICPU scheduler strides it with `TASKPAYLOAD_TENSOR_STRIDE`, so the
  runtime's working tensor is pinned at **128 B** (2 cache lines). A merged struct is
  ~192–216 B, taking `TaskPayload` from 4864 B to 6912–7680 B — **+42% to +58% per
  task slot**.
- **Zero return for that cost.** Most of what a merge would add has no reader on
  the far side. The field-by-field split is below.

#### What each type carries, and what the other has no use for

Only the middle block means the same thing in both — the geometry of the view.
Everything above it answers *"where does this live?"*, everything below it is
*"what has the L2 runtime decided about it?"*, and neither question is open on
the other side of materialization.

Both types have a member called `buffer`, and they are not the same thing: on the
wire it is the `BufferDescriptor` that says how to find the backing, on the device
it is the resolved `addr` + `size`. That is the whole of what materialization does.

| `Tensor` (144 B, wire) | rel | `ChipTensor` (72 B, L2 argument) |
| ---------------------- | --- | -------------------------------- |
| `buffer.magic` | ⟂ | — |
| `buffer.identity` (32 B) | ⟂ | — |
| `buffer.backend_kind` | ⟂ | — |
| `buffer.body[32]` + `body_len` | ⟂ | — |
| `buffer.nbytes` | ≈ | `buffer.size` |
| `buffer.access` | ⟂ | — |
| `buffer.owner_worker_path_id` | ⟂ | — |
| — | ⟂ | `buffer.addr` |
| `byte_offset` (bytes) | ≈ | `start_offset` (elements) |
| `shapes[5]` / `strides[5]` / `ndims` / `dtype` | = | `shapes[5]` / `strides[5]` / `ndims` / `dtype` |
| `buffer.address_space` | = | `address_space` (still spelled `child_memory` until the wire flip) |

**Dead on the device** (`Tensor`-only): `magic` discriminates untrusted bytes at
a decode boundary the device does not have. `identity` / `backend_kind` / `body`
/ `body_len` are the *recipe* for finding the backing — spent once at
materialization, never consulted again; what a body may contain is fixed per
backend, see [Backends](#backends). `access` has no device enforcement point
(it is checked at submit). `owner_worker_path_id` is diagnostic.

**Meaningless before materialization** (`ChipTensor`-only): `buffer.addr` is the
resolved address, which by definition does not exist while the argument is still
crossing processes.

Neither column carries the L2 OverlapMap state — the producing task, the overlap
version, whether dependency tracking is creator-only. Those are *decided by* the
chip runtime, so neither an L3 builder nor an L2 caller has anything to put there;
they live on `simpler::{hbg,tmr}::Tensor`, which `Runtime::set_orch_args` adopts
each argument into. The same goes for `is_contiguous` and `extent_elem_cache`:
derived from `shapes`/`strides`, and cached only where a hot path reads them per
task. Contiguity follows PyTorch's rule for nonempty tensors: dimensions of
size one do not constrain their stride. For example, shape `(64, 1)` with
strides `(1, 64)` is contiguous and permits a zero-copy runtime reshape;
strides `(2, 64)` still describe gapped storage. The boundary descriptor,
runtime caches, and HBG graph validation use the same rule. The contiguity
checks do not rewrite strides or change the storage extent.

So a merged struct would carry ~70 B that the AICore never reads, in a type whose
size is pinned at two cache lines precisely because the scheduler walks it per
task.

> The one row with a plausible future device reader is `identity`: keying the L2
> OverlapMap by it rather than by `buffer.addr` would make two views of one
> backing bucket together by construction. That needs 32 B — which fits the
> existing `_pad_cl2[36]` at `sizeof == 128`, i.e. **without** merging anything
> else. If it is ever done, the H2D staging step must mint a *new* identity for
> each staged copy, because the device buffer is a distinct backing from the host
> one it was copied from.

### Rejected: keep the wire type transport-only, use `ChipTensor` in the L3 orch

This keeps one user-visible type but pays a conversion at every hop: the orch
builds a `ChipTensor`, the mailbox needs the self-describing form, and the
consumer needs a `ChipTensor` again. Each conversion has to **rewrite the
address** — and the sender cannot know the receiver's, so the receiver ends up
guessing. That is the pre-P1-B mechanism: `_rewrite_blob_host_addrs` rewrote
addresses by **numeric range** and mis-rewrote device pointers that happened to
fall inside a registered host range, patched by adding a `child_memory` skip
whose own comment records the hazard.

### Shipped instead: each named for what it is

Both are called what they are, and by the **same** name in both languages, which
is what [codestyle rule 13](../.claude/rules/codestyle.md) requires of a public
type. The L3+ wire type is `Tensor` — `simpler.buffer.Tensor` in Python (it joins
`simpler.task_interface`), the
global `Tensor` of `src/common/task_interface/buffer.h` in C++. The device
POD is `ChipTensor` in both, from `src/common/task_interface/tensor.h`.

Rule 13 is what makes the plain names available: the device POD held the global
`Tensor` name until #1681 moved it to `ChipTensor`, prefixing it for the chip
context it belongs to. Neither type needs a namespace to disambiguate it, and
a Python module that imports the wrong one now names a type that exists and
behaves differently — so the mismatch surfaces where it is used.

**So the split costs the user nothing to know.** You name a `Tensor`, submit it,
and the chip's C++ orchestration receives it resolved; the type name does not
even appear in your code — you write `buffer.tensor(shapes, dtype)` and
`args.add_tensor(t, tag)`. Resolving the address in between is the framework's
job, and keeping the two forms as separate types is what makes that boundary a
type change rather than a silently wrong address.

## Allocating buffers

```python
h  = worker.create_buffer(nbytes)                 # kind3: explicit shared host buffer (POSIX shm)
h  = worker.alloc_shared_tensor((M, N), dtype)    # kind3: shape-sized create_buffer
# inside an orch fn, for device (chip-private) memory:
d  = orch.alloc_child_tensor(worker, (M, N), dtype)  # kind4: DEVICE_MALLOC on that chip
```

`create_buffer` returns a `Buffer` backed by a POSIX shm the consumer maps
lazily on first receipt of a tensor over it (map-once, by identity);
`alloc_shared_tensor` returns a runtime-managed intermediate over a FORK_SHM ring
VA. `alloc_child_tensor` allocates device memory on a specific next-level worker
and wraps the pointer; its `.base` is the device pointer (the `orch.copy_to`
destination), and a tensor over it must be dispatched only to that worker.

## Naming a view

`buffer.tensor(...)` names a view over the backing:

```python
v = h.tensor(shapes=(M, N), dtype)                        # contiguous (row-major strides)
v = h.tensor(shapes=(N, M), dtype, strides=(1, M))        # transposed
v = h.tensor(shapes=(M, K), dtype, byte_offset=off)       # sub-region
```

`strides` are **element** strides and are strictly > 0 — broadcast (stride 0) and
negative step are unsupported, and a singleton dimension's stride is never
normalized away. `byte_offset` is a **byte** offset and must be a multiple of the
dtype size.

## Submitting a task

```python
ta = TaskArgs()
ta.add_tensor(a_h.tensor((SIZE,), DataType.FLOAT32), TensorArgType.INPUT)
ta.add_tensor(out_h.tensor((SIZE,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
orch.submit_next_level(chip_handle, ta, cfg, worker=0)
```

`TaskArgs` carries `Tensor`s. Tags drive dependency inference, which keys on
the **canonical identity** — buffer granularity, the successor of the former
buffer-address key, so every view of one backing lands on one key and no real
dependency hides behind a differing offset. Which of those candidates actually
conflict is then decided by their **view geometry**, in two stages:

1. **Bounding range.** `[byte_offset, byte_offset + extent)` on each side.
   Disjoint boxes are disjoint views, answered in O(1).
2. **Per-dimension intersection**, when the boxes do overlap. Each origin is
   decomposed into per-axis coordinates and the axes are intersected one by
   one; disjoint on any single axis makes the views disjoint.

So two disjoint slices of one buffer carry no edge and run concurrently —
`x[0]` and `x[1]` of a rank-major tensor by stage 1, and `x[:, 0:4]` versus
`x[:, 8:12]` of a matrix by stage 2, whose bounding boxes interleave because
every row of one sits between two rows of the other.

This is the host-side counterpart of the L2 OverlapMap cascade, and carries the
same limits. Stage 2 models only pairs sharing one canonical row-major layout —
same dtype and `ndims`, identical `strides` descending as exact multiples down
to 1, origins on that layout's lattice. A transposed pair, a stepped slice, or
two views of different rank over one backing fall through as *overlapping*, as
does a producer a later write only partly covers. Every fallback is in that one
direction: an extra edge is possible where the truth is subtler, and a real
dependency is never dropped.

An extra edge is normally just lost concurrency. It is not always: where two
tasks rendezvous with each other, a spurious edge holds back the dispatch that
would release the other, and the run deadlocks. That is why the disjointness
above is answered precisely rather than conservatively.

## Reading and writing data — torch only at the boundaries

The orch fn is a pure DAG builder: computing on data there would be invisible to
dependency inference. So **torch is used only outside `run()`** (fill inputs,
read outputs) or **inside a Python sub-worker** (a compute leaf):

```python
h = worker.create_buffer(n * 4)
torch.frombuffer(h.shm.buf, dtype=torch.float32, count=n).fill_(5.0)  # before run()
worker.run(my_orch, ...)                                             # orch names tensors only
result = torch.frombuffer(out_h.shm.buf, dtype=torch.float32, count=n)  # after run()
```

> **`h.shm` is transitional — wrap it.** Byte access belongs on the view, not on the
> backing object: a `POSIX_SHM` buffer has an `shm` and a device one never will, so
> code written against `h.shm.buf` forks by backend and cannot be told "this backing
> has no host mapping" — it just gets `None`. The accessor that replaces it lands with
> the view algebra; until then, put `h.shm.buf` behind one helper of your own so the
> switch is a one-line change. Note also that a `frombuffer` tensor keeps the
> memoryview alive: drop it before `close()`, or the release raises `BufferError`.

## How a tensor reaches its consumer (three-way split)

A `Tensor` on the wire is materialized differently by each consumer:

| Consumer | What it does |
| -------- | ------------ |
| **Chip leaf (L2 runtime)** | Decode the POD blob with `read_args_from_blob`, then materialize each one to a `ChipTensor` (map-once, keyed by canonical identity) via `ImportRegistry.materialize_args`, including **strided** views, and submit with `_submit_chip_run_materialized`. |
| **Python sub-worker** (compute) | Map each one into a `MappedArg`; the callable computes with `torch.frombuffer(arg.buffer, ...)`. No `ChipTensor`. |
| **Nested L4→L3 orch** (forwarding) | **Re-export** each backing to a descriptor `H'` that keeps the source's canonical identity — no pass-through, no map on the forwarding hop. |

**Re-export (no pass-through).** An upper-level tensor is forwarded on receipt as a
descriptor `H'` that keeps the source's **canonical identity unchanged** (invariant
across every edge — an L4 buffer forwarded L4→L3→L2 carries one identity at all
three layers; only role / materialized VA / view change), per-backing and without
mapping. A downstream compute leaf maps lazily, so pure forwarding carries no map
cost. Dependency inference keys on the invariant identity, so an alias /
retain-release does not split across layers.

## Canonical identity

Every backing carries a fixed-length 32-byte identity — an opaque per-incarnation
`owner_instance_id`, a `buffer_id` unique within that incarnation, and a `generation` that starts at
1 and increments whenever a `buffer_id` slot is reused (so a stale descriptor for a recycled slot is
rejected rather than silently resolving). It is **invariant across every edge**: an L4 buffer
forwarded L4→L3→L2 carries one identity at all three layers.

Identity is what dependency inference and the map-once import cache key on — never a materialized
address, which means nothing different in another process.

Nothing inside the identity bounds a read: it has no length field. Hashing and comparison are
therefore in-bounds for any bytes that arrive, structurally rather than by validation. The owning
worker's tree path is deliberately **not** part of it — a path is reused across restarts and
contributes no uniqueness, so it is interned to a diagnostic `owner_worker_path_id` whose table lives
only in the owning process. An id another process minted renders as `<path#N>`; nothing routes,
gates, or keys on it.

## Backends

| Backend | Materializes to | `body` must be | Used for |
| ------- | --------------- | -------------- | -------- |
| `POSIX_SHM` | a named shm mapped into the consumer | the shm name: 1–32 bytes of printable ASCII, no `/` | `create_buffer` / `alloc_shared_tensor` |
| `FORK_SHM` | the same VA, no map | exactly 8 bytes, a non-zero u64 LE base | a pre-fork `MAP_SHARED` host buffer (e.g. `share_memory_()`), writable from the child |
| `FORK_COW` | the same VA, no map | exactly 8 bytes, a non-zero u64 LE base | a pre-fork plain host buffer: copy-on-write, so **READ only** |
| `VMM_WINDOW` | the device VA of the window carved by `allocate_domain`, no map | exactly 8 bytes, a non-zero u64 LE base | communication-domain window / buffer |
| `DEVICE_MALLOC` | the device pointer, no map (chip-local) | exactly 8 bytes, a non-zero u64 LE base | `alloc_child_tensor` |
| `REMOTE_SIDECAR` | (P2) resolved via the remote transport | empty | an arg to a remote L3; the descriptor rides in the sidecar |

**The body rule is enforced, not advisory.** `backend_kind` is what selects how a
consumer reads `body`, so a body that does not fit that reading is not a smaller
backing — it is a different value, and a short address body reads as a truncated
pointer indistinguishable from a real one. Construction and every decode run the
same check, so a descriptor that breaks the table above cannot be built or
received. Two consequences worth knowing:

- **Bytes past `body_len` must be zero**, as must a `Tensor`'s `shapes` /
  `strides` slots past `ndims`. Both are regions a producer chose not to fill,
  and the whole struct crosses the process boundary, so whatever the owner's
  memory held there would cross with it. The `_pad` fields are *not* covered:
  they are alignment slack, and `CanonicalIdentity`'s is deliberately tolerated
  so two decodes of one backing still key alike.
- **A POSIX shm object smaller than the descriptor's `nbytes` is refused at
  import.** Every view bound check is `byte_offset + extent <= nbytes`, so an
  unverified `nbytes` makes all of them vacuous.

## What a submit checks

Naming an argument is not enough on its own — two things are verified where the
values are final, at submit:

- **`access ⊆ granted`.** A tag may only request what the backing grants: `INPUT`
  needs READ, `OUTPUT_EXISTING` needs WRITE, `INOUT` needs READWRITE. This is
  re-checked at submit rather than trusted from `add_tensor`, because a tag stays
  mutable afterwards. It is what stops a plain (copy-on-write) tensor from being
  named as an output and then silently losing every write in the child.
- **No overlapping writes within one task.** Two arguments of one task that name
  intersecting bytes of the same backing are rejected: they belong to one node,
  so there is no order between them to express, and a device-staged copy of a
  host backing does not even alias on the device for the L2 overlap map to
  notice. Disjoint slices of one buffer stay legal — that is what `byte_offset`
  is for, and this check runs the same two-stage comparison dependency
  inference does, so a tiled write of one buffer passes rather than tripping on
  bounding boxes that merely interleave.

Group members are not compared against each other: a group is one node, and
naming one buffer as every member's output is how it publishes a single
completion token for a downstream task to depend on.

## Scope / status

This page describes the memory model end to end. **The dispatch wire is
connected**: `TaskArgs` carries the `Tensor`, `buffer.tensor(...)` reaches a
consumer, and the submit-time checks above run on every submit. Still absent are
the other allocators (`alloc_shared_tensor`, `alloc_child_tensor`), and the
endpoint x `address_space` check inside `materialize` — a device backing resolved
there yields a pointer meaningful only on its owner chip, and today that is
enforced where a task is submitted rather than where it is materialized.

**There is no second blob format.** A `Tensor` travels in the TaskArgs mailbox
blob that `write_blob` / `read_blob` implement in
[`task_args_wire.h`](../src/common/task_interface/task_args_wire.h); that blob's
element is the `Tensor`, and `TaskArgsView::tensors` validates each one as it decodes
it. `ChipTensor` survives only in `ChipStorageTaskArgs`, the POD `ChipWorker`
consumes — an L2 worker materializes into it inside `run` before calling down.

Single-machine (host + device) L3→L2 and L4→L3→L2 dispatch is implemented and
verified in `a2a3sim` and onboard `a2a3`. The remote **receive**
side and the buffer lifecycle robustness (`release_buffer`, in-flight retain /
deferred-free) are later phases (P2).
