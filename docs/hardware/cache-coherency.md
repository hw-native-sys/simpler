# Cache Coherency (GM ↔ AICore/AICPU)

This page is the authoritative reference for **when to insert a cache
operation** on Ascend onboard hardware. The coherency model differs
between chip generations supported in this repo: a2a3 requires AICPU
cache maintenance for host-DMA and SDMA-published GM, while a5 AICPU
loads are coherent with DMA/HBM and do not need those invalidates.
Cycle costs and a few code-path examples below are sampled on a2a3
unless stated otherwise. Misapplying these rules either leaks stale data
(correctness bug) or burns a `dsb sy` per record on a hot path (perf
bug — see PR #863 lineage). Read it before touching any code that
writes `dc civac`, `dc cvac`, `cache_invalidate_range`, or any `dcci`
on AICore.

## Who shares GM, and with what consistency

GM (HBM) is read and written by four parties on Ascend onboard. Their
relationship to **AICPU's data cache** is **not symmetric** on a2a3,
and a5 has a stronger DMA/HBM coherency model:

| Writer | a2a3: AICPU sees write automatically? | a2a3: AICPU must invalidate before read? | a5: AICPU must invalidate before read? |
| ------ | ------------------------------------- | ---------------------------------------- | -------------------------------------- |
| AICPU itself | Yes (own cache) | No | No |
| **AICore** (AIC / AIV / MIX) | **Yes** | **No** | **No** |
| Host DMA (`rtMemcpy` from host RAM) | No | **Yes** | **No** |
| SDMA engine (device-side DMA) | No (assumed) | **Yes** | **No** |

Likewise, **AICore's own cache is non-coherent with GM** in the other
direction: AICore-side writes stay in its data cache until explicitly
pushed out with `dcci`. AICPU writes that AICore needs to see follow a
mirror of this table (AICore must `dcci` to invalidate before reading
host- or AICPU-written GM) — except where the AICore reads with a
bypass load, for which see
[the AICPU → AICore section](#the-aicpu-to-aicore-path-for-one-access-shape-the-a2a3-teardown-gate).

The rest of this doc fills in why each row of the table is what it is,
and what code lives on each side.

## The two cache primitives

| Primitive | Side | Purpose | Cost (rough, a2a3 / DAV_3510) |
| --------- | ---- | ------- | ----------------------------- |
| `dcci` (`__attribute__((aicore))` intrinsic) | AICore | Push a cache line out to GM (clean+invalidate). Required after AICore stores that AICPU or peer AICore must read. | 1 cache line per call + a following `dsb` to commit ordering. |
| `cache_invalidate_range(addr, size)` (`src/common/platform/include/aicpu/cache_maintenance.h`) | AICPU | `dc civac` + `dsb sy` + `isb` over a byte range on onboard builds; sim builds keep it a no-op. Required on a2a3 before AICPU reads GM that **a non-coherent writer** (host DMA, SDMA) most recently published. | `dsb sy` dominates (tens to hundreds of cycles, fixed regardless of range). |

`cache_invalidate_range` is the protocol-correct primitive for the
**host-DMA → AICPU** case on a2a3. It was introduced in PR #204
specifically for `Runtime` struct hand-off where the host writes via
`rtMemcpy` and AICPU reads. **It is NOT the right primitive for
AICore → AICPU, and it is not required for a5 DMA/HBM reads.**

### `dcci` is whole-cache-line: no parallel sub-line writes

`dcci` cleans+invalidates **one whole cache line** (64 B on a2a3 = 16
`float`s); there is no sub-line granularity. A core's writeback emits its
entire line copy, including bytes it never wrote (stale in its copy).

**Consequence — two AICore cores must never write different elements of the
same cache line in parallel.** Each core flushes the whole line, so the last
flush clobbers the others' elements with stale values (classic false
sharing → last-writer-wins). This is invisible on `sim` (cache modeled as
no-op: `SINGLE_CACHE_LINE == 0` in `platform/sim/aicore/inner_kernel.h`) and
only fails on silicon. The kernel-author rule (each SPMD block writes its own
cache line) lives in
[../aicore-kernel-programming.md](../aicore-kernel-programming.md#each-block-must-write-to-its-own-cache-line).

## The "AICore → AICPU" path: AICPU does not invalidate, but DOES barrier

AICore and AICPU share a coherency domain on GM. When AICore writes a
slot, the correct handshake is:

```text
AICore                              AICPU
  store slot fields                   read COND (MMIO, Device-nGnRE)
  store task_id (last)                check FIN bit
  dcci slot, SINGLE_CACHE_LINE   →    rmb()                ← load-load barrier
  dsb (commit dcci before FIN)        read slot fields     ← Normal cacheable
  write FIN → COND                ←
```

Two separate concerns, often conflated:

- **Cache coherency** (Do we need `dc civac`?): No. AICore's `dcci`
  pushes the line to GM and the AICPU's cache is in the same coherency
  domain, so a subsequent AICPU load fetches the fresh value. `dcci`
  is load-bearing on AICore's side — AICore's data cache is not
  coherent with GM in the other direction, so without `dcci` the line
  never reaches HBM and AICPU would observe an old value.

- **Load-load ordering** (Do we need `rmb()`?): **Yes.** The COND
  register is `Device-nGnRE` memory; the slot is Normal cacheable
  memory. ARM64 does not implicitly order Device reads against
  subsequent Normal reads — they can be reordered if there is no
  data/address dependency. In this path, the slot address is computed
  from a value the caller already holds (the dispatched `task_id`),
  not from the just-read COND value, so there is no architectural
  dependency. Without `rmb()` (`dsb ld`), the CPU can speculatively
  satisfy the slot load before the COND read indicates FIN, returning
  whatever was in the AICPU's cache at speculation time (likely a
  stale value from a previous round). The AICPU side must emit
  `rmb()` between the COND check and the slot reads.

Concretely, the chip swimlane staging-slot read in
`src/common/platform/shared/aicpu/chip_swimlane_collector_aicpu.cpp` does
**not** call `cache_invalidate_range` on the slot, but it **does** call
`rmb()` before reading `slot->task_id` and the timing fields. All of
those fields are AICore writes covered by the AICore-side `dcci` in
`chip_swimlane_aicore_record_task`. The same pattern applies to the PMU
staging slot
(`src/{a2a3,a5}/platform/shared/aicpu/pmu_collector_aicpu.cpp`).

### Historical pitfall

PR #540 (2026-04-15) added `cache_invalidate_range(slot, 64)` on the
AICPU side of the chip swimlane staging slot, mirroring the
host-DMA-protocol pattern from PR #204. The two situations are
**not** the same: host DMA bypasses the AICPU cache; AICore stores
plus `dcci` do not. The cache invalidate was redundant — but the
embedded `dsb sy` inside `cache_invalidate_range` was inadvertently
providing the COND→slot load-load ordering as a side effect. Replacing
the whole call with nothing would have left the ordering implicit (and
dependent on microarchitectural quirks); replacing it with an explicit
`rmb()` keeps the ordering as part of the documented protocol while
dropping the unnecessary cache op.

If you find yourself about to write `cache_invalidate_range` on an
AICPU-side read of an AICore-published value, **stop**. The right fix
is `rmb()` (for load-load ordering against a prior COND read) plus
making sure the AICore side does `dcci` before signaling. The cache
invalidate itself is not needed on this path.

## The "AICPU to AICore" path for one access shape: the a2a3 teardown gate

On a2a3, an AICPU ordinary GM store is observable by an AICore `ld_dev` bypass load of
the same address **without an AICPU-side cache clean**. No extra maintenance is required
to expose that store to that reader.

**Basis: a project architecture premise.** This is stated by the project owner as an
architectural fact. It is **not** a vendor specification quoted here, **not** proven from
the public SDK, and **not** something this repository measured. Read it exactly as
written and no wider: it is about an a2a3 AICPU ordinary GM store, observed by an AICore
`ld_dev` load of the same address. It does not extend to the host↔device edge (a separate
premise, in the host-DMA section below), to the reverse direction, or to a reader that
reaches the same field through any other access path.

**Coherence here is not ordering, and does not retire a writer.** The premise says the
value is visible. It supplies no happens-before between the store and any other
operation, makes no multi-field publication atomic, and says nothing about whether an
earlier run's reader is still live on the line. For this one field, existing code
provides the **two intra-run ordering edges** below, under the lifetime preconditions
that code already assumes — it does not establish those preconditions:

- **Reset before window-open.** `memset` over the gates then `wmb()`
  (`scheduler_cold_path.cpp:1183-1184` TRB, `:890-891` HBG). The barrier is what orders
  the resets before `hs_setup_done_` and before any register window opens — a window is a
  plain `Device-nGnRE` store carrying no release semantics of its own. No core can read
  its gate before its window opens.
- **RELEASE after window-close.** The quiesce pass, then the CLOSE pass and its
  readbacks, then `rmb()`, then the relaxed `post_close_release` stores
  (`platform_regs.cpp:82`, `:142`, `:149`). The drain is what orders every store after
  the CLOSE it belongs to, so an open return gate is never paired with an open window.

**Lifetime across runs is a separate, open question.** Whether a reader left behind by a
failed or partially retired run can still be live when a later launch resets and
releases the same gates is **not** established — the recovery path is untraced, and
neither the premise nor the two edges above settle it. See
[the exceptional-exit limit in the descriptor visibility investigation](../investigations/2026-09-runtime-descriptor-input-visibility-chain.md#exceptional-and-partial-exits-audited-separately).

The AICore reads that word with `wait_for_post_close_release`
(`src/a2a3/platform/onboard/aicore/inner_kernel.h:52-59`), a `ld_dev` spin followed by
`dsb(DSB_DDR)`. Nothing else is published through the gate: it is permission-to-return,
one `uint32_t` in its own 64-byte line.

**This licenses no deletion.** In particular the descriptor-wide
`cache_invalidate_range` in `deinit` stays: `dc civac` also cleans, and its role toward
other readers — including the host's D2H direction, which this premise does not address —
is a separate question tracked in
[the descriptor visibility investigation](../investigations/2026-09-runtime-descriptor-input-visibility-chain.md).

## The "host DMA → AICPU" path: a2a3 MUST invalidate, a5 does not

On a2a3, when the host writes via `rtMemcpy`, the AICPU cache is not snooped.
A cached value from a previous round survives the DMA write, so the
next AICPU load returns stale data. The original PR #204 fix was:

```cpp
// src/a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp:455
cache_invalidate_range(runtime, sizeof(runtime->dev));
```

This is **necessary on a2a3** and must not be removed there. It is the
load-bearing usage of `cache_invalidate_range` in the a2a3 runtime.

Two details of the call as it stands, both verified against the source:

- The extent is the device-copied descriptor, `sizeof(runtime->dev)`, not
  `sizeof(Runtime)`. The host-only tail is never uploaded (#2309, #2315).
- The call sits in `AicpuExecutor::deinit()`, at the **end** of a run, so it
  precedes the *next* invocation's read rather than the current one's. The
  earliest descriptor read of a run — the platform AICPU kernel's affinity gate —
  has no cache operation before it.

On a5, host DMA writes to GM are coherent with AICPU reads, so the
matching runtime hand-off code does not call `cache_invalidate_range`.

**Basis: a project architecture premise.** The host↔device distinction is stated by
the project owner: **A3 host/device is not cache-coherent; A5 host/device is.** Take
that as the premise this section rests on. It is an architectural statement from the
project, **not** a vendor specification quoted here and **not** something this
repository measured — no external citation is linked from this file, and none should
be inferred.

Three limits on how far that premise reaches, because the rest of this document makes
endpoint-specific claims that do **not** follow from it:

- **It is about the host↔device edge only.** The `AICore → AICPU` and
  `SDMA → AICPU` sections below concern on-device producers. Their claims rest on
  their own basis and are not derived from this premise.
- **It names A3.** A2 is not covered by it. The `a2a3` tree serves both and remains on
  the conservative path; this documentation does not change that path. No A2 coherence
  model follows from the premise, and nothing here establishes that an additional
  clean-and-invalidate is harmless on any architecture — that would be a separate
  question about the operation's outbound role, below.
- **Coherence is not ordering, atomicity or ownership.** It says a host write becomes
  visible without AICPU-side invalidation. It does not make a multi-field publication
  atomic, does not supply happens-before between a writer and a reader, and does not
  permit overwriting a field another agent is live on. Those obligations are tracked
  separately in
  [the descriptor visibility investigation](../investigations/2026-09-runtime-descriptor-input-visibility-chain.md).

**Consequently this premise licenses no code change on its own.** In particular it does
not authorise deleting any cache operation now in the tree: a `dc civac` call also
cleans, so its outbound role toward on-device readers is a separate question from the
inbound freshness the premise settles.

**In-tree history of the a5 difference.** a5 `host_build_graph` calls
`cache_invalidate_range` over the descriptor at two sites while a5
`tensormap_and_ringbuffer` calls it at none. The statement itself entered the tree with
[`153daee6b`](https://github.com/hw-native-sys/simpler/commit/153daee6bb302554bbff85e4e91f80e66691df25)
(#1235), whose message gives the rationale as "per hardware guidance" and names no
document, version or section. The sequence:

1. [`153daee6b`](https://github.com/hw-native-sys/simpler/commit/153daee6bb302554bbff85e4e91f80e66691df25)
   (#1235) removed `cache_invalidate_range` over the descriptor from **both** the a5
   `host_build_graph` and a5 `tensormap_and_ringbuffer` deinit paths.
2. [`26b67fdb5`](https://github.com/hw-native-sys/simpler/commit/26b67fdb5c54aef29d477698440e6cb612ba5159)
   (#1661) migrated a5 `host_build_graph` onto the host-orchestrated runtime and the
   call is present again afterwards. That commit enumerates six deviations from the
   a2a3 runtime it copied; this call is not among them.
3. [`52f25af4a`](https://github.com/hw-native-sys/simpler/commit/52f25af4a81b22f39836accefd2fd48db725a06a)
   (#2090) added `aicpu_legacy_executor.cpp`, which contains the call from the point
   the file was created.

One reading of 1–3 is that the call returned as part of the migration copy rather than
by a decision specific to a5. **That is an inference from commit history, and history
proves neither a hardware model nor its absence** — it records when the call returned,
not why it was kept. It is kept here because it explains how the two a5 runtimes came
to differ, which is a maintenance fact, independent of the premise above.

## The "SDMA → AICPU" path: a2a3 is conservative, a5 is coherent

SDMA is a separate device-side DMA engine. On a2a3 it writes GM but is
not known to snoop AICPU's data cache. Current a2a3 runtime code (see
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_scheduler.h`
and the SDMA-engine async-wait completion path in
`runtime/async_wait.h` / `runtime/scheduler/scheduler.h`)
**does** invalidate before reading SDMA-written counters and records,
on the conservative assumption that SDMA is not in AICPU's coherency
domain.

On a5, SDMA writes are coherent with AICPU reads, so the matching a5
SDMA completion helpers do not invalidate or flush cache lines.

Note the basis: SDMA is an **on-device** engine, so this row is *not* covered by the
host↔device architecture premise recorded in the host-DMA section above. It rests on
whatever established it originally, which this file does not record.

## Quick decision table

When you are about to insert a cache operation, ask in order:

1. Who actually wrote the bytes I'm reading? Look at the producer
   code, not the address.
2. Is this a5? If yes → no AICPU invalidate/flush for GM reads written
   by host DMA, SDMA, AICPU, or AICore.
   The **host DMA** arm of this rule is the one the architecture premise in the
   host-DMA section supports. The other three producers are on-device and rest on
   their own basis; do not cite that premise for them.
3. On a2a3, is the producer in the AICPU coherency domain (AICPU itself
   or AICore)? If yes → no invalidate. If no (host, SDMA) → invalidate.
4. For AICore writes specifically, does the producer already `dcci`
   before signaling? If not, fix that instead of papering over it
   with an AICPU-side invalidate.
5. Did I just read a completion flag (COND / mailbox / counter) from
   a different memory type (Device-nGnRE MMIO) before this load? If
   yes, and there is no data/address dependency between that read and
   this one, insert `rmb()` between them — coherency does not imply
   load-load ordering on ARM64.
6. Am I about to add an AICPU clean so an **AICore** read sees my
   store? On a2a3, for an ordinary GM store read back by an AICore
   `ld_dev` bypass load, no clean is needed — see the AICPU → AICore
   section. What you still owe that reader is *ordering*, from a
   barrier and the surrounding protocol, not from cache maintenance.

If the answer to (1) is "I'm not sure" — find out. The cost of one
wrong `cache_invalidate_range` is silent perf rot; the cost of a
missing `rmb()` is a stale-data bug that may only fire under specific
buffer-reuse patterns or aggressive OOO speculation. Both are paid
forever once they ship.

## Related code

- `src/common/platform/include/aicpu/cache_maintenance.h` — public cache-maintenance wrappers.
- `src/common/platform/onboard/aicpu/cache_ops.cpp` — onboard implementation (`dc civac` / `dsb sy` / `isb`).
- `src/common/platform/sim/aicpu/cache_ops.cpp` — sim no-op.
- AICore-side `dcci` usage lives in the chip swimlane / PMU AICore collectors and any kernel that publishes to a GM slot AICPU reads.

## Related docs

- [PMU staging-slot ordering](../dfx/pmu-profiling.md) —
  detailed AICore-side `dcci` + barrier order for staging-slot writes.
- [chip swimlane profiling](../dfx/chip-swimlane-profiling.md) —
  the consumer of the rules above on the chip swimlane path.
