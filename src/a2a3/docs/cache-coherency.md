# Cache Coherency on a2a3 (GM ↔ AICore/AICPU)

This page is the authoritative reference for **when to insert a cache
operation** on a2a3 onboard hardware. Misapplying these rules either
leaks stale data (correctness bug) or burns a `dsb sy` per record on a
hot path (perf bug — see PR #863 lineage). Read it before touching any
code that writes `dc civac`, `dc cvac`, `cache_invalidate_range`, or any
`dcci` on AICore.

## Who shares GM, and with what consistency

GM (HBM) is read and written by four parties on a2a3 onboard. Their
relationship to **AICPU's data cache** is **not symmetric** — that
asymmetry is the entire reason this page exists.

| Writer | AICPU sees write automatically? | AICPU must invalidate before read? |
| ------ | ------------------------------- | ---------------------------------- |
| AICPU itself | Yes (own cache) | No |
| **AICore** (AIC / AIV / MIX) | **Yes** | **No** |
| Host DMA (`rtMemcpy` from host RAM) | No | **Yes** |
| SDMA engine (device-side DMA) | No (assumed) | **Yes** (until hardware confirms otherwise) |

Likewise, **AICore's own cache is non-coherent with GM** in the other
direction: AICore-side writes stay in its data cache until explicitly
pushed out with `dcci`. AICPU writes that AICore needs to see follow a
mirror of this table (AICore must `dcci` to invalidate before reading
host- or AICPU-written GM).

The rest of this doc fills in why each row of the table is what it is,
and what code lives on each side.

## The two cache primitives

| Primitive | Side | Purpose | Cost (rough, DAV_3510) |
| --------- | ---- | ------- | ---------------------- |
| `dcci` (`__attribute__((aicore))` intrinsic) | AICore | Push a cache line out to GM (clean+invalidate). Required after AICore stores that AICPU or peer AICore must read. | 1 cache line per call + a following `dsb` to commit ordering. |
| `cache_invalidate_range(addr, size)` (`src/a2a3/platform/onboard/aicpu/cache_ops.cpp`) | AICPU | `dc civac` + `dsb sy` + `isb` over a byte range. Required before AICPU reads GM that **a non-coherent writer** (host DMA, SDMA) most recently published. | `dsb sy` dominates (tens to hundreds of cycles, fixed regardless of range). |

`cache_invalidate_range` is the protocol-correct primitive for the
**host-DMA → AICPU** case. It was introduced in PR #204 specifically
for `Runtime` struct hand-off where the host writes via `rtMemcpy` and
AICPU reads. **It is NOT the right primitive for AICore → AICPU.**

## The "AICore → AICPU" path: AICPU does not invalidate, but DOES barrier

AICore and AICPU share a coherency domain on GM. When AICore writes a
slot, the correct handshake is:

```text
AICore                              AICPU
  store slot fields                   read COND (MMIO, Device-nGnRnE)
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
  register is `Device-nGnRnE` memory; the slot is Normal cacheable
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

Concretely, the L2 perf staging-slot read in
`src/a2a3/platform/src/aicpu/l2_perf_collector_aicpu.cpp` does **not**
call `cache_invalidate_range` on the slot, but it **does** call `rmb()`
before reading `slot->task_id` and the timing fields. All of those
fields are AICore writes covered by the AICore-side `dcci` in
`l2_perf_aicore_record_task`. The same pattern applies to the PMU
staging slot (`src/a2a3/platform/src/aicpu/pmu_collector_aicpu.cpp`).

### Historical pitfall

PR #540 (2026-04-15) added `cache_invalidate_range(slot, 64)` on the
AICPU side of the L2 perf staging slot, mirroring the
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

## The "host DMA → AICPU" path: AICPU MUST invalidate

When the host writes via `rtMemcpy`, the AICPU cache is not snooped.
A cached value from a previous round survives the DMA write, so the
next AICPU load returns stale data. The original PR #204 fix was:

```cpp
// src/a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp
cache_invalidate_range(runtime, sizeof(Runtime));  // before reading host-written Runtime
```

This is **necessary** and must not be removed. It is the
load-bearing usage of `cache_invalidate_range` in this repo.

Same protocol applies to anything else the host DMAs into a GM region
that AICPU subsequently reads.

## The "SDMA → AICPU" path: treat as non-coherent until proven otherwise

SDMA is a separate device-side DMA engine. It writes GM but is not
known to snoop AICPU's data cache. Current runtime code (see
`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_scheduler.h`
and the SDMA-engine async-wait completion path in
`runtime/pto_async_wait.h` / `runtime/scheduler/pto_scheduler.h`)
**does** invalidate before reading SDMA-written counters and records,
on the conservative assumption that SDMA is not in AICPU's coherency
domain.

If a hardware confirmation lifts that assumption — i.e. SDMA writes
become snooped — these invalidates can be removed using the same
reasoning as the AICore case. Until that confirmation exists, **leave
them in place**.

## Quick decision table

When you are about to insert a cache operation, ask in order:

1. Who actually wrote the bytes I'm reading? Look at the producer
   code, not the address.
2. Is the producer in the AICPU coherency domain (AICPU itself or
   AICore)? If yes → no invalidate. If no (host, SDMA) → invalidate.
3. For AICore writes specifically, does the producer already `dcci`
   before signaling? If not, fix that instead of papering over it
   with an AICPU-side invalidate.
4. Did I just read a completion flag (COND / mailbox / counter) from
   a different memory type (Device-nGnRnE MMIO) before this load? If
   yes, and there is no data/address dependency between that read and
   this one, insert `rmb()` between them — coherency does not imply
   load-load ordering on ARM64.

If the answer to (1) is "I'm not sure" — find out. The cost of one
wrong `cache_invalidate_range` is silent perf rot; the cost of a
missing `rmb()` is a stale-data bug that may only fire under specific
buffer-reuse patterns or aggressive OOO speculation. Both are paid
forever once they ship.

## Related code

- `src/a2a3/platform/onboard/aicpu/cache_ops.cpp` — `cache_invalidate_range` implementation (`dc civac` / `dsb sy` / `isb`).
- `src/a2a3/platform/sim/aicpu/cache_ops.cpp` — sim no-op.
- AICore-side `dcci` usage lives in the L2 perf / PMU AICore collectors and any kernel that publishes to a GM slot AICPU reads.

## Related docs

- [PMU staging-slot ordering](../../../docs/dfx/pmu-profiling.md) —
  detailed AICore-side `dcci` + barrier order for staging-slot writes.
- [L2 swimlane profiling](../../../docs/dfx/l2-swimlane-profiling.md) —
  the consumer of the rules above on the L2 perf path.
