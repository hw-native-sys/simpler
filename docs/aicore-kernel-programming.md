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

### Each block must write to its own cache line

**Two AICore blocks running on different cores must never write to the
same cache line.** This is a hardware constraint, not a software policy.

Each core holds its own copy of a cache line; on `dcci` (clean+invalidate)
it writes back the **entire 64-byte line** (16 `float`s on a2a3), including
the bytes it never touched — which in its copy are stale. When N cores each
write a different element of the *same* line and flush, the last core to
flush wins and overwrites every other core's element with a stale value.
There is no per-element flush; `dcci` granularity is one whole cache line.

So a kernel like this is **wrong** on silicon (it happens to pass on `sim`,
which models no cache):

```cpp
// BROKEN: out has block_num elements packed into one cache line; every
// block flushes the whole line -> last-writer-wins -> [0,0,0,last].
out[block_idx] = value;
dcci(&out[block_idx], SINGLE_CACHE_LINE, CACHELINE_OUT);
```

The fix is to give each block a cache-line-isolated output region — stride
each block's output by at least one cache line (`>= 16` `float`s on a2a3),
or have each block write a full cache-line-aligned tile (the usual case:
real kernels write a head / row / tile per block, which is already aligned):

```cpp
constexpr int CACHE_LINE_FLOATS = 16;            // 64 B / sizeof(float)
out[block_idx * CACHE_LINE_FLOATS] = value;      // distinct line per block
dcci(&out[block_idx * CACHE_LINE_FLOATS], SINGLE_CACHE_LINE, CACHELINE_OUT);
```

See [hardware/cache-coherency.md](hardware/cache-coherency.md) for the full
`dcci` / cache-line model.

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

### When the no-arg call is inside the pto-isa tile-pipe library

The porting checklist above assumes you own every `get_subblockid()` call
site. You do **not** when the kernel drives the pto-isa tile-pipe library
(`TPUSH` / `TPOP` / `TFREE` with `TileSplitAxis::TILE_UP_DOWN` or
`TILE_LEFT_RIGHT`): those templates compute their per-AIV FIFO offset from the
**no-arg** `get_subblockid()` internally
(`TPush.hpp::pushVec2GMFiFo` / `popVecTileFromGMFiFo` —
`subAIVOffset = get_subblockid() * rows * cols * sizeof(T)`), and you cannot
thread `args` into a third-party library template.

Under simpler's MIX dispatch that library call returns 0 for both lanes, so its
auto-split contributes nothing and **both AIV lanes read/write the same FIFO
half**. The other half is never produced; attention then reads uninitialised GM
and the output is wrong/partial (observed as `NaN` in the qwen3 case here).

**Do not** bridge it with a file-scope cache:

```cpp
// WRONG — the .o will not load. See §4.
[[block_local]] static int32_t lane;   // per-core static
#define get_subblockid() lane          // redirect the library's no-arg call
// ... lane = get_sub_block_id(args); once in kernel_entry
```

A `[[block_local]]` (or any non-const) static is read via a `.text` relocation
that the AICore loader rejects (§4), and AICore forbids ordinary global/static
data, so there is no relocation-free global to redirect into either.

The working pattern (as in `spmd_paged_attention`) leaves the library's
`get_subblockid()` at its native 0 and adds the per-lane offset **explicitly**
on the tile-pipe, computed from `get_sub_block_id(args)`:

```cpp
int32_t lane = get_sub_block_id(args);                        // 0 or 1, real lane
// TILE_UP_DOWN splits an M×N tile by rows; AIV1 starts one sub-tile in.
pipe.cons.setEntryOffset(lane * Rows * Cols * sizeof(ConsT)); // C2V pop side
pipe.prod.setEntryOffset(lane * Rows * Cols * sizeof(ProdT)); // V2C push side
```

The library adds `get_subblockid() * bytes` (= 0) **plus** your `entryOffset`,
so the explicit offset now carries the entire lane split. Set it once after the
pipe is constructed; `entryOffset` persists across `TPUSH` / `TPOP`. See
`examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/kernels/aiv/fa_fused_aiv.cpp`
for a worked example.

---

## 4. The AICore loader runs raw `.text` only

simpler loads a kernel by copying the **literal `.text` section bytes** out of
the compiled `.o` and jumping to them
([`simpler_setup/elf_parser.py`](../simpler_setup/elf_parser.py)). It does
**not**:

- apply ELF relocations (`.rela.text`), or
- merge out-of-line template instantiations (`.text._Z*` COMDAT groups).

If a kernel `.o` carries either, the loader refuses it with
"AICore loader cannot extract a runnable payload …" rather than load a binary
whose `BL`/`B` targets are left as `imm26 = 0` — those would branch to garbage
on device, producing CANN 507018 watchdog timeouts or silently-wrong partial
output (the failure modes behind issue #900, PR #830 / issue #831). The loader
is strict **by design**; it is not relaxed to let "benign" relocations through.

Two things produce a rejected `.o`:

| Cause | Why it relocates | Fix |
| ----- | ---------------- | --- |
| Out-of-line call to another function (a non-inlined `static` helper, or a template instantiation emitted to its own section) | `BL <fn>` needs an `R_AARCH64_CALL26` relocation the loader can't apply | Mark every function in the call chain `__attribute__((always_inline))` so the compiler folds them into one `.text` |
| Reading a non-const global / `static` / `[[block_local]]` variable | The address load needs a relocation against the data symbol | Don't use module-level mutable data on AICore — pass the value as a function argument down from `kernel_entry` |

Verify a kernel object before chasing a device hang:

```bash
readelf -SW kernel.o | grep -E '\.text'  # want only ".text"; ".text._Z*" or ".rela.text" = reject
readelf -r  kernel.o                      # want: no relocation entries
```

This is exactly why §3's sub-block-id fix uses an explicit `setEntryOffset`
argument rather than a cached `[[block_local]]` static: the static would
reintroduce a `.rela.text` the loader rejects.

---

## 5. Hard constraints — what AICore physically cannot do

These are not design preferences; the hardware refuses or the chip
hangs. They cap what protocol the kernel author can ask of the AICore
side. Confirmed on a3 silicon — see
[`docs/hardware/mmio-performance.md`](hardware/mmio-performance.md)
for the measurements and
[`docs/investigations/2026-06-aicore-mmio-to-spr.md`](investigations/2026-06-aicore-mmio-to-spr.md)
for the verdict trail.

- **No SPR-write to `DATA_MAIN_BASE`.** `MOV DATA_MAIN_BASE, x` is
  rejected at compile time — the CCEC backend has no destination
  encoding for that SPR. Only `MOV %0, DATA_MAIN_BASE` (read self) is
  available. Use the `=l` constraint to accept either uint32 or
  uint64; `=r` rejects uint64.
- **No load or store into the SPR MMIO window.** Issuing a
  `LDR`/`STR` from inside an AICore at `peer.reg_addr + offset` (or
  your own `reg_addr + offset`) hangs the AICore. The CCECPU monitor
  kills `aicpu-sd` 50 s later. This applies symmetrically to peer
  cores' DMB, peer cores' COND, and any other AIC_CTRL register —
  the chip only accepts SPR-window transactions from AICPU.
- **DMB is hardware-unidirectional.** Combining the two above, an
  AICore has no path to mutate any DATA_MAIN_BASE — its own or a
  peer's. If a protocol needs an AICore to publish a value into DMB,
  route through GM (write field + `dcci`, see
  [`cache-coherency.md`](hardware/cache-coherency.md)) and let an
  AICPU thread forward the value into DMB by MMIO STR.
- **COND is the only AICPU-visible per-task signal an AICore can
  emit.** `write_reg(RegId::COND, MAKE_FIN_VALUE(task_id))` (or
  `MAKE_ACK_VALUE`) is the production path — the SPR write retires
  in ~5–10 ns and lands at the COND MMIO register that AICPU's
  scheduler polls.

If a kernel needs to publish anything other than a per-task COND
update — for example, a counter, a profiling slot, or a
ring-buffer-style record — it must go through GM with `dcci`. There
is no "fast direct register" alternative on the AICore side.

### 5.1 AIC vs AIV on SPR self-access — single-run measurements

These were sampled on a3 silicon during the experiment that produced
[`mmio-performance.md`](hardware/mmio-performance.md). They are
single-run readings; treat as "the direction of the surprise is solid,
the exact magnitude needs verification before you optimise on it."

- **AIC and AIV are indistinguishable on a hot SPR self-cadence path.**
  A tight loop of `set_cond(FIN) + read_reg(DMB)` runs at ~9.7 ns /
  iter on both AIC and AIV. This **refutes** the commonly-stated
  hypothesis that AIC SPR writes trigger a pipeline flush that AIV
  avoids. If you've seen a kernel slowdown attributed to "AIC SPR
  pipeline penalty," look for the real cause elsewhere — the bare
  set_cond + read_reg cost is the same on both engines.
- **AIC tolerates rotating SPR write targets; AIV does not.** A
  tight loop of `STR` cycling across N distinct SPR target registers
  costs ~5 ns / STR on AIC (no extra cost vs same-target). On AIV the
  same loop costs ~26 ns / STR (about 5× slower than same-target).
  Direction is opposite of the obvious guess ("AIC has more cube
  state, so target switching should hurt more"). Practical
  implication: AIV kernels that publish across multiple SPRs in a
  tight loop pay a per-switch tax that AIC kernels don't; if you can
  batch writes to one SPR before moving to the next, do so on AIV.

Both findings come from one sample. Re-verify before relying on
either as a design constraint. The
[`aicore-notification-perf`](../tools/cann-examples/aicore-notification-perf/)
tool's producer kernel is the closest scaffolding — its `producer.cce`
already has a mode-aware tight-loop runs on AICore with AICPU-side
tick capture. Extending it to time the AICore-internal
`set_cond + read_reg` cadence (Phase 5) or rotating-target SPR writes
(Phase 6) is the minimum work — add a new `NotifPerfMode` value, a
matching branch in `producer.cce`, and result fields the consumer
sums into the existing `NotifPerfResult`. Build the producer once
with `-DCCE_AICORE_ARCH=dav-c220-cube` for AIC and once with
`-DCCE_AICORE_ARCH=dav-c220-vec` for AIV; both findings need the
AIC/AIV comparison to be meaningful.

## 6. Related

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
