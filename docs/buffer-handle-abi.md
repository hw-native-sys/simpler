# BufferHandle / BufferRef — the L3+ memory model

At L3 and above, tasks name their data with typed, self-describing **buffer
handles and views**, not raw pointers. This replaces the legacy "raw pointer +
`child_memory` bool" mechanism with an ABI that carries a canonical identity, a
backend descriptor, and a strided view — so a buffer can be resolved exactly
across the L3→L2 (and L4→L3) boundaries without a side table.

The wire ABI is frozen in
[`.docs/L3-new/worker-memory-model/bufferhandle-abi.md`](../.docs/L3-new/worker-memory-model/bufferhandle-abi.md);
this page is the user-facing how-to.

## Three types

| Type | What it is | Where it lives |
| ---- | ---------- | -------------- |
| **`BufferHandle`** | An owned backing (POSIX shm / fork-COW / device malloc) with a canonical identity + lifecycle. Stays with the Worker that created it. | owner side (L3+) |
| **`BufferRef`** | A **self-describing view**: the full handle descriptor embedded + a strided view `(byte_offset, shapes, strides, dtype)`. The wire element of `TaskArgs`. Carries no materialized address. | on the wire (L3→L2, L4→L3) |
| **`Tensor`** | The materialized form (address + strided view). Exists **only** at the L2 device-runtime boundary. | L2 leaf |

**L3+ orch functions never touch a C++ `Tensor`.** They allocate handles, build
views, and submit refs. A `Tensor` only appears when a `BufferRef` is
materialized on the L2 chip child.

## Allocating buffers

```python
h  = worker.create_buffer(nbytes)                 # kind3: explicit shared host buffer (POSIX shm)
h  = worker.alloc_shared_tensor((M, N), dtype)    # kind3: shape-sized create_buffer
# inside an orch fn, for device (chip-private) memory:
d  = orch.alloc_child_tensor(worker, (M, N), dtype)  # kind4: DEVICE_MALLOC on that chip
```

`create_buffer` / `alloc_shared_tensor` return a `BufferHandle` whose backing is
a born-shared POSIX shm attached into every forked child. `alloc_child_tensor`
allocates device memory on a specific next-level worker (via `orch.malloc`) and
wraps the pointer; its `.base` is the device pointer (the `orch.copy_to`
destination), and its ref must be dispatched only to that worker.

## Building views (the view algebra)

`handle.ref(...)` names a view; the view algebra mirrors the C++ `Tensor`
(pure metadata, returns a new `BufferRef`):

```python
v = h.ref(shapes=(M, N), dtype)     # contiguous full view (row-major strides)
v.slice(dim, start, end, step=1)    # any (incl. strided) view
v.transpose(x, y)                   # swap two dims
v.permute((1, 0))                   # reorder dims
v.view(shapes, offsets)             # sub-region
v.reshape(new_shapes)               # contiguous only
```

`slice` / `transpose` / `permute` / `view` are unconstrained (strided views are
supported); `reshape` requires a contiguous view — exactly the `Tensor`
constraints. Strides are element strides (> 0); `byte_offset` is a byte offset.

## Submitting a task

```python
ta = TaskArgs()
ta.add_ref(a_h.ref((SIZE,), f32), ArgDirection.INPUT)
ta.add_ref(out_h.ref((SIZE,), f32), ArgDirection.OUTPUT_EXISTING)
orch.submit_next_level(chip_handle, ta, cfg, worker=0)
```

`TaskArgs` carries `BufferRef`s. Tags drive dependency inference, which keys on
the **canonical identity** (buffer granularity, the successor of the former
buffer-address key); byte-range overlap between same-buffer views is refined by
the L2 OverlapMap on the materialized tensors, not by the L3 key.

## Reading and writing data — torch only at the boundaries

The orch fn is a pure DAG builder: computing on data there would be invisible to
dependency inference. So **torch is used only outside `run()`** (fill inputs,
read outputs) or **inside a Python sub-worker** (a compute leaf):

```python
h = worker.create_buffer(n * 4)
torch.frombuffer(h.shm.buf, dtype=torch.float32, count=n).fill_(5.0)  # before run()
worker.run(my_orch, ...)                                             # orch names refs only
result = torch.frombuffer(out_h.shm.buf, dtype=torch.float32, count=n)  # after run()
```

## How a ref reaches its consumer (three-way split)

A `BufferRef` on the wire is materialized differently by each consumer:

| Consumer | What it does |
| -------- | ------------ |
| **Chip leaf (L2 runtime)** | Materialize each ref to a `Tensor` (map-once, keyed by identity), including **strided** views; hand the Tensor blob to `run_from_blob`. |
| **Python sub-worker** (compute) | Map each ref into a `MappedArg`; the callable computes with `torch.frombuffer(arg.buffer, ...)`. No `Tensor`. |
| **Nested L4→L3 orch** (forwarding) | **Re-export** each backing to a local handle `H'` under this level's identity — no pass-through, no map on the forwarding hop. The inner orch sees only its own handles. |

**Re-export (no pass-through).** Each L3+ level owns only its own handles: an
upper-level ref is relabeled to a fresh local identity (same backing) on receipt,
per-backing and without mapping. A downstream compute leaf maps lazily, so pure
forwarding carries no map cost. This is required for fork-COW correctness (the
VA chain is per fork level) and keeps dependency ownership clean per level.

## Backends

| Backend | Materializes to | Used for |
| ------- | --------------- | -------- |
| `POSIX_SHM` | a named shm mapped into the consumer | `create_buffer` / `alloc_shared_tensor` |
| `FORK_SHM` | the same VA (copy-on-write inherited), no map | a pre-fork host buffer, zero-copy |
| `DEVICE_MALLOC` | the device pointer, no map (chip-local) | `alloc_child_tensor` |
| `REMOTE_SIDECAR` | (P2) resolved via the remote transport | an arg to a remote L3; the descriptor rides in the sidecar |

## Scope / status

Single-machine (host + device) L3→L2 and L4→L3→L2 dispatch is implemented and
verified in `a2a3sim` and onboard `a2a3`. The remote **receive** side and the
buffer lifecycle robustness (`release_buffer`, in-flight retain / deferred-free)
are later phases (P2); see [`.docs/L3/P1.md`](../.docs/L3/P1.md) for the phase
breakdown.
