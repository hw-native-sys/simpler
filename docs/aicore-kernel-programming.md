# AICore Kernel Programming Guide

How to write an AICore kernel for the `tensormap_and_ringbuffer` runtime
— the SPMD execution-context contract, the supported accessors, and
the things that break silently when ported from native CANN code.

For the broader picture see
[hierarchical_level_runtime.md](hierarchical_level_runtime.md) (where
kernels sit in the L0–L6 layering),
[task-flow.md](task-flow.md) (end-to-end task data flow), and
[chip-level-arch.md](chip-level-arch.md) (Host / AICPU / AICore tiers).
The kernel-author contract for the `host_build_graph` runtime is not
covered here; this guide is `tensormap_and_ringbuffer`-specific.

---

## 1. What a kernel sees

Every AICore kernel in this runtime has the signature

```cpp
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args);
```

`args[]` is a flat array of 64-bit slots whose meaning is positional
and fixed at compile time:

```text
args layout (tensormap_and_ringbuffer):
  [0 .. tensor_count-1]                = tensor GM pointers
  [tensor_count .. +scalar_count-1]    = scalar values
  ...
  [SPMD_LOCAL_CONTEXT_INDEX  = 48]     = (uint64_t)&LocalContext   per-dispatch
  [SPMD_GLOBAL_CONTEXT_INDEX = 49]     = (uint64_t)&GlobalContext  per-core
```

The trailing two slots are written by the scheduler before each
dispatch and hold the **SPMD execution context** described below. They
exist on every dispatch — you can rely on them unconditionally.

The constants live in
[`src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/common/intrinsic.h`](../src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h);
treat them as private to the runtime and always go through the
accessor functions defined in that header.

---

## 2. SPMD execution context

Three pieces of topology data are exposed to user kernels:

| Accessor (use these) | Returns | Lifetime | Source |
| -------------------- | ------- | -------- | ------ |
| `get_block_idx(args)` | logical block index in `[0, block_num)` | per-dispatch | `LocalContext.block_idx` |
| `get_block_num(args)` | total logical blocks for this task | per-dispatch | `LocalContext.block_num` |
| `get_sub_block_id(args)` | AIV lane in cluster (0 = AIV0, 1 = AIV1) | per-core, init once | `GlobalContext.sub_block_id` |

`sub_block_id` is **only meaningful for AIV kernels in MIX tasks**.
AIC kernels and single-AIV tasks should not depend on it. AIV0 is the
"left" lane, AIV1 the "right" lane; they execute the same kernel
binary and use `sub_block_id` to pick which half of the work they own
(for example: head 0 of a `(head0, head1)` pair vs head 1).

The scheduler initialises `GlobalContext.sub_block_id` once per AIV
core at startup, based on each core's position in its cluster
(`scheduler_cold_path.cpp::SchedulerContext::init`). `LocalContext` is
rewritten by `build_payload()` before each dispatch.

### Logical vs physical block_dim

`get_block_num(args)` returns the **logical** block count baked into
this task. It is **not** the same as the physical AICore-block count
that the runtime launches:

| Symbol | Meaning |
| ------ | ------- |
| `RUNTIME_CONFIG.block_dim` (Python `CallConfig.block_dim`) | Number of physical AICore blocks the runtime launches per dispatch. |
| `get_block_num(args)` | Logical block count the kernel partitions work across. Currently always 1; multi-logical-block (`block_num > 1`) is not yet implemented. |

When you set `CallConfig.block_dim = 24` in Python and your kernel sees
`get_block_num(args) == 1`, that is by design — every physical block
runs the same kernel and the kernel partitions work however it likes
using `get_block_idx()` against whatever it expects. Don't conflate
the two.

---

## 3. Do **not** use the CCE topology intrinsics

The CCE / AscendC headers ship a parallel set of topology intrinsics:

```cpp
// from kernel_operator.h / tikcfw — DO NOT use in this runtime
get_subblockid();
get_block_idx();
get_block_num();
```

These read **AICore hardware registers** that the
`tensormap_and_ringbuffer` runtime does not program. They were
designed for the native CANN dispatch model, where the OS-level
scheduler sets the registers per launch. simpler's runtime keeps the
same data in software (the `LocalContext` / `GlobalContext`
structures in §1) and does **not** poke the registers.

The consequence is silent miscompute, not an error. Specifically:

- `get_subblockid()` returns whatever stale value the sub-block
  register holds. In simpler's MIX dispatch that is **0 for both
  AIV0 and AIV1 of every cluster**, so a kernel that partitions
  heads on `sub_block_id` parity has AIV1 redo AIV0's work and never
  writes AIV1's share of the output. This is the partial-zero
  failure mode in issue #900 / PR #899 `spmd_paged_attention_highperf`:
  the ported AIV kernel compiled clean, ran without error, and
  produced 16 correct heads + 16 zero heads out of 32. Resolved by
  switching the three intrinsics to the `(args)` accessors above.
- `get_block_idx()` / `get_block_num()` are not redirected either —
  they reflect physical block topology, not simpler's logical
  partitioning.

### Porting checklist

When moving a kernel into this runtime from
ascend-transformer-boost / AscendC / any other native-CANN code path:

```text
get_subblockid()        →  get_sub_block_id(args)
get_block_idx()         →  get_block_idx(args)
get_block_num()         →  get_block_num(args)
```

Plumb `args` (or just `block_idx`, `block_num`, `sub_block_id` as
plain `uint32_t` arguments) down through whichever templates,
class methods, or static helpers the kernel uses internally. Do not
leave a single CCE-intrinsic call in the AICore code path; otherwise
the silent-miscompute mode will resurface the next time someone
refactors the call graph.

PR #899's resolution
([commit `0964b4`](https://github.com/hw-native-sys/simpler/pull/899/commits))
is a worked example — the AIC and AIV classes grew `pto_block_idx`,
`pto_block_num`, `pto_sub_block_id` parameters threaded all the way
down from `kernel_entry` into `UnpadAttentionDecoderAic::SetArgs` and
`UnpadAttentionDecoderAiv::SetArgs`.

---

## 4. Related

- [`src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h`](../src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h)
  — declarations of the args-based accessors and the
  `LocalContext` / `GlobalContext` layout. Same file for a5 under
  `src/a5/runtime/tensormap_and_ringbuffer/common/intrinsic.h`.
- [`src/a2a3/runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md`](../src/a2a3/runtime/tensormap_and_ringbuffer/docs/SUBMIT_BY_CLUSTER.md)
  — how the orchestration side dispatches AIC + AIV0 + AIV1 as a
  single MIX task (the producer of the `sub_block_id` distinction).
- [`docs/scheduler.md`](scheduler.md) — how the scheduler turns a
  submitted task into a per-core dispatch payload (the writer of
  `LocalContext`).
- Examples worth reading as templates:
  `tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/`
  (single-AIV SPMD) and
  `tests/st/a2a3/tensormap_and_ringbuffer/spmd_multiblock_mix/`
  (MIX with both AIV lanes).
