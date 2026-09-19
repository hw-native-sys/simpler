# The Runtime descriptor's input-visibility chain: what the code establishes and what it does not

**Date**: 2026-09-19
**Verdict**: open questions delimited — no defect established, no cache operation changed, no new contract proposed

## Question

The per-field inventory on #2254 noted an asymmetry: three of the four onboard
variants call `cache_invalidate_range` over the whole device descriptor, and a5
`tensormap_and_ringbuffer` does not. The tempting reading is "a5 TRB is missing
an invalidate."

That reading assumes the operation is what makes the next invocation's host bytes
visible. This entry tests that assumption by tracing the whole chain — host
snapshot through to first device read — for each variant, and separating three
things the earlier note ran together: **what the code does**, **what is thereby
guaranteed**, and **what is merely consistent with working**.

The outcome is that the asymmetry is real but is *not* the interesting part, and
that no variant's chain is closed by evidence available in this repository.

## Baseline

Every anchor below was read at **`31da0560e202b7bb378773f7dfc1d0074fb28e89`**
(`main`, 2026-09-19, the merge of #2363). Nothing here was built, run, simulated
or measured; this is a reading of source at one commit. Hardware observations
from earlier entries are cited only where explicitly attributed, and no number
from a different commit is carried forward as applying to this one.

## The chain, stage by stage

### 1. Host source snapshot

The host mutates its own `Runtime` object throughout `prepare_execution`.
`KernelArgsHelper::prepare_runtime_args`
(`src/common/platform/onboard/host/device_runner_helpers.cpp:80`) then takes a
snapshot of exactly the device-read prefix into host heap storage:
`RuntimeLaunchImage::prepare` (`src/common/platform/include/host/runtime_launch_image.h:33`)
resizes a `std::vector<std::byte>` to `runtime_device_copy_size(runtime)` and
`memcpy`s from `&runtime`.

### 2. Allocation and reuse

The destination is **not** per-run. It is owned by the pipeline slot —
`SlotPersistentArgs::runtime_args`, one per slot in
`std::array<SlotPersistentArgs, PTO_PIPELINE_MAX_DEPTH>`
(`device_runner_base.h`), allocated lazily on a slot's first run
(`device_runner_helpers.cpp:99-107`) and released only in `finalize_common_impl`.
Per-slot rather than per-runner because the copy is not stream-ordered, so a
prepared successor would otherwise overwrite the image its predecessor is
executing against (`device_runner_helpers.h`).

**This is the fact that makes the whole question non-trivial:** the device
address is reused across every run on that slot, so any device-side cache line
covering it can hold bytes from a previous invocation. A fresh allocation per run
would make most of what follows moot.

### 3. Prepare/publish transfer

`RuntimeLaunchImage::publish` moves the snapshot into a function-local vector,
clears the member, and hands the local's `data()`/`size()` to a copy callback;
`KernelArgsHelper::publish_runtime_args` (`device_runner_helpers.cpp:113`) supplies
a synchronous `rtMemcpy(..., RT_MEMCPY_HOST_TO_DEVICE)`.

Two lifetime properties follow, and only the first is currently exercised:

- The moved-to local lives until `publish` returns, so a **synchronous** copy
  reads a valid source. An asynchronous copy would not: the source dies at
  return. The seam is shaped for later decoupling but is not asynchronous today.
- The snapshot is consumed exactly once; a repeated or reentrant publish is
  rejected rather than copying twice.

PR #2360 documents this host-side seam, including that the sole production caller
`init_runtime_args_with_metadata` (`device_runner_base.cpp:2782`) invokes prepare
and publish consecutively with only a return-code check between them, and that
**a non-null `args.runtime_args` does not prove publication happened** — there is
no publication-state gate in launch. This entry does not restate that; it starts
where #2360 stops, at the device side of the same transfer.

### 4. Launch ordering

Publication happens inside `prepare_execution`, before the AICPU and AICore
kernels are launched. The descriptor's device address travels to the AICPU as a
by-value launch argument (`KernelArgs::runtime_args`), so the *pointer* is fresh
per launch. The *pointee* is ordinary device memory read through the AICPU's
data cache.

### 5. First device read — the same on all four variants

The earliest reader is **not** the runtime executor. It is the platform AICPU
kernel entry, which reads the affinity-gate fields before calling
`aicpu_execute`:

| arch | anchor | reads |
| ---- | ------ | ----- |
| a2a3 | `src/a2a3/platform/onboard/aicpu/kernel.cpp:65,93` | `get_aicpu_allowed_cpu_count()`, `get_aicpu_launch_count()`, `get_aicpu_allowed_cpus()` |
| a5 | `src/a5/platform/onboard/aicpu/kernel.cpp:82,121` | same three |

**Neither file performs any cache operation.** `grep` for
`cache_invalidate_range` / `cache_flush_range` in either returns nothing. So on
every variant, the first device read of host-published descriptor bytes happens
with no cache maintenance earlier in that invocation.

This matters beyond hygiene because the gate decides *how many threads proceed*:
a stale `aicpu_launch_count` does not corrupt a value, it changes the thread
population that reaches the barrier.

On a5 `host_build_graph` there is a second early read of the same kind:
`aicpu_execute` (`src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp:627`)
branches on `aicore_scheduler_runtime_enabled(runtime)`, which reads
`runtime->dev.workers[0].aicpu_ready` (`aicpu/aicore_scheduler_state.h:27`) — a
**host-published** mode word — to choose the resident or legacy executor.

### 6. Last access and cache operations

The only descriptor-wide cache operation in the tree is
`cache_invalidate_range(runtime, sizeof(runtime->dev))`, at four sites, and
**every one of them is inside `deinit()` — the end of a run, not before a read**:

| site | variant |
| ---- | ------- |
| `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:1035` | a2a3 TRB |
| `src/a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp:455` | a2a3 HBG |
| `src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp:563` | a5 HBG resident |
| `src/a5/runtime/host_build_graph/aicpu/aicpu_legacy_executor.cpp:414` | a5 HBG legacy |

a5 TRB's `deinit` takes `Runtime * /*runtime*/` unnamed and performs none
(`src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:1029`).

What the primitive actually is (`src/common/platform/onboard/aicpu/cache_ops.cpp:20`):

- **`dc civac`** — clean *and* invalidate to Point of Coherency. Not a pure
  invalidate: a dirty line is **written back** before being invalidated.
- Applied per 64-byte line over `[addr, addr+size)` rounded **outward** to line
  boundaries, so it can touch bytes outside the requested range. This is what the
  `sizeof(DeviceRuntimeLaunchDesc) % 64 == 0` static_assert protects.
- Followed by `dsb sy` and `isb`.
- The non-aarch64 branch (`:54`) and the whole sim implementation
  (`src/common/platform/sim/aicpu/cache_ops.cpp`) are **inert**, so no simulation
  run can exercise any of this.

The AICPU does dirty descriptor lines — it writes `workers[i].task`
(`scheduler_cold_path.cpp`) and, on a2a3, zeroes and sets `teardown_gates`. So at
`deinit` the `dc civac` is performing a **write-back of AICPU-authored bytes into
the descriptor**, in addition to invalidating. That direction is the opposite of
what "invalidate so the next round's DMA is visible" suggests, and both effects
are real.

### 7. Reuse

The next run rewrites the same slot block (step 2) and the cycle repeats. In
**kernel mode** it does not: `PersistentKernelArgs::prepare_once`
(`src/common/platform/onboard/host/kernel_persistent_args.cpp:24`, called once
from `device_runner_base.cpp:871`, guarded by `prepared_`) allocates and H2Ds the
descriptor **exactly once per context**. Every subsequent run reads descriptor
bytes the host has not rewritten — so for kernel mode the end-of-run `dc civac`
invalidates a region with no new producer, and the visibility question is
different in kind rather than a variant of the program-mode one.

## Four-variant matrix

Publication and first read are identical across all four (steps 1–5). What
differs is the end-of-run operation and the device-side readers.

| variant | descriptor invalidate at `deinit` | earliest descriptor read | second early read | AICore reads descriptor |
| ------- | --------------------------------- | ------------------------ | ----------------- | ----------------------- |
| **a2a3 TRB** | yes (`:1035`) | platform gate, `kernel.cpp:93` | `dev.aicpu_thread_num` in executor init | `workers[block_idx]`, `teardown_gates[block_idx]` |
| **a2a3 HBG** | yes (`:455`) | platform gate, `kernel.cpp:93` | `dev.host_total_tasks`, `dev.sm_image_bytes` at boot | `workers[block_idx]`, `teardown_gates[block_idx]` |
| **a5 TRB** | **no** | platform gate, `kernel.cpp:121` | `dev.aicpu_thread_num` in executor init | `workers[block_idx]` |
| **a5 HBG resident** | yes (`:563`) | platform gate, `kernel.cpp:121` | **`workers[0].aicpu_ready` mode dispatch** (`:627`) | `workers[block_idx]`, incl. host-published `aicpu_ready`/`task` |
| **a5 HBG legacy** | yes (`aicpu_legacy_executor.cpp:414`) | platform gate, `kernel.cpp:121` | same mode word, then legacy boot | `workers[block_idx]` |
| **kernel mode (both arches)** | same `deinit` sites run | same platform gate | same | same |

The resident/legacy split is decided *by reading the descriptor* — so the mode
word's freshness is a precondition of picking the right executor, and neither
branch can invalidate before the branch is taken.

## Three places the documented rule and the code disagree

`docs/hardware/cache-coherency.md:130-145` states the rule, and
`docs/hardware/chip-architecture.md:218-219` restates it in the end-to-end flow.
Against the code at this SHA:

1. **Placement.** The rule is written as *"before reading host-written Runtime"*,
   and `chip-architecture.md` puts it at step 2 of the flow, i.e. at AICPU entry.
   All four real call sites are in `deinit`, at the end of the previous run. Those
   are different mechanisms: "invalidate then read" is self-contained within one
   invocation, while "invalidate at the end of the previous run" depends on
   nothing re-populating the line across a kernel boundary before the next read.
2. **Extent.** The rule's snippet is `cache_invalidate_range(runtime, sizeof(Runtime))`.
   The code narrows it to `sizeof(runtime->dev)`, correctly — the host-only tail
   was never uploaded. The doc predates #2309/#2315.
3. **a5.** The rule says *"On a5, host DMA writes to GM are coherent with AICPU
   reads, so the matching runtime hand-off code does not call
   `cache_invalidate_range`"*, and its decision table step 2 says for a5 "no AICPU
   invalidate/flush for GM reads written by host DMA". **a5 HBG calls it anyway,
   at two sites.** a5 TRB matches the rule.

So the asymmetry flagged on #2254 inverts once the documented rule is taken into
account: a5 TRB is the variant that *follows* the written rule, and a5 HBG is the
one that departs from it. This entry does not resolve which is right, because the
rule's own basis is not established — see unknowns.

## The a5 HBG handshake line has three writers, and one ordering token

`Handshake` is exactly 64 B and 64 B-aligned
(`src/common/host_build_graph/runtime.h:98-112`), so one worker's handshake is
exactly one cache line. On a5 HBG that single line is written by all three tiers:

| writer | words | operation |
| ------ | ----- | --------- |
| **host** | `aicpu_ready` = `SCHEDULER_RUNTIME_MODE_*`, `task` = worker's `SchedulerWorkerContext` address | part of the descriptor H2D (`a5/runtime/host_build_graph/host/runtime_maker.cpp:977`, `:1275-1277`) |
| **AICore** | `physical_core_id`, `core_type`, `aicore_done` | `dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT)` — whole-line write-back (`aicore/aicore_executor.cpp:545`) |
| **AICPU** | `task`, `aicpu_ready` = `RESIDENT_READY` | `cache_flush_range(&handshakes[lo], n * sizeof(Handshake))` = `dc cvac`, whole lines (`aicpu/aicore_lifecycle.cpp:322-328`) |

Both device-side operations are **whole-line** write-backs of a line each party
holds a full copy of. They cannot merge at word granularity: whichever writes
back later replaces all 64 bytes with its own cached view. The header comment
states this hazard — *"A word the AICPU must publish independently cannot live
here — a stale line writeback would overwrite it"* — and then a5 HBG has the host
and the AICPU both publishing words there.

**What keeps it correct is an ordering token, and that is the finding.** The AICPU
does not reach `publish_context_partition` (`aicore_lifecycle.cpp:311`) until it
has observed every one of its cores' `aicore_done` — spinning with a per-line
`cache_invalidate_range` before each read (`aicore_lifecycle.cpp:110-111`).
`aicore_done` is published by the very `CACHELINE_OUT` that would otherwise
clobber. And the AICore reads the host's `task` only after its own flush, from the
copy its entry `scheduler_observe_cache_line` pulled in
(`aicore_executor.cpp:533`). So the three whole-line writers are serialized:

```text
host H2D  →  AICore observe + write + CACHELINE_OUT (publishes aicore_done)
          →  AICPU sees aicore_done  →  AICPU writes task/aicpu_ready + dc cvac
          →  AICore re-observes in its RESIDENT_READY wait loop
```

This ordering is load-bearing and is documented nowhere. It is what a future
change would silently break — by moving another word into the line, by relaxing
the `aicore_done` gate, or by letting any party write the line outside its turn.
Recording it is the most immediately useful output of this entry.

## What is established, and what is not

**Established from code at this SHA:**

- The descriptor's device destination is slot-owned and reused across runs, so
  stale lines are architecturally possible rather than hypothetical.
- Publication is a synchronous H2D completed before launch.
- The earliest device read on all four variants is the platform AICPU kernel's
  affinity gate, with no cache operation earlier in that invocation.
- The only descriptor-wide cache operation is `dc civac` at end-of-run, present on
  three variants and absent on a5 TRB, and it both writes back and invalidates.
- The a5 HBG three-writer line is serialized by `aicore_done` as traced above.
- Sim exercises none of this: both primitives are no-ops there.

**Not established — and specifically not to be assumed:**

- **That an end-of-run invalidate makes the next invocation's bytes visible.** It
  completes before the next launch, but nothing in the repo prevents a line from
  being re-populated between then and the next read, and `dsb sy; isb` after
  `dc civac` orders only the maintenance itself. It creates no ordering against a
  *later* host DMA.
- **That the first invocation is covered at all.** On a slot's first run there is
  no previous `deinit`, so no invalidate has ever executed for those lines. What
  makes that read correct is unknown; "a fresh allocation cannot be cached" is an
  assumption about allocator and driver behaviour that this repo does not state,
  and `2026-09` work elsewhere found fresh device allocations are not reliably
  zero, which is a different claim but from the same family of assumptions.
- **That invalidation is equivalent to an acquire.** It is not. Even a correctly
  placed invalidate before the read establishes freshness of the line, not
  ordering with respect to the producer's other writes.
- **That the absence of a failure proves the chain closes.** Every variant passes
  CI today. That bounds how often any gap fires under the exercised shapes; it
  does not establish a guarantee, and it cannot, because the mechanism that would
  make it safe is unidentified.

**Unknown, and not answerable from this repository:**

1. **Whether a5's host-DMA→AICPU coherency claim is true, and its source.**
   `cache-coherency.md` asserts it with no citation to an SDK document, a driver
   source, or a measurement. If true, a5 needs nothing and a5 HBG's two calls are
   redundant; if false, a5 TRB has a real gap. Both readings fit the code.
2. **Whether the AICPU kernel launch performs cache maintenance, or starts cold.**
   If the driver invalidates or the op begins with a cold cache, every variant is
   safe regardless of what the runtime does, and the three end-of-run calls are
   all redundant. This is `rtsLaunchCpuKernel`/driver behaviour.
3. **Whether the descriptor's device mapping is cacheable on the AICPU at all.**
   If `rtMalloc`'d GM is mapped uncached or write-through for the AICPU, the whole
   question dissolves. The repo documents `Device-nGnRE` for the **MMIO** window
   only (`.claude/rules/ascend.md`), which says nothing about GM.
4. **Whether `dc civac`'s write-back of AICPU-dirtied descriptor lines can race a
   subsequent host H2D to the same lines.** The run boundary probably separates
   them, but "probably" is the state of the evidence.

Any one of 1–3 being true would close the chain for some or all variants. Because
none is established, the code's asymmetry cannot be read as a defect — and
equally cannot be read as safe.

## Minimal discriminating probe specification — not run

Specified so a future session does not have to re-derive it. **This entry ran
none of it.** Each item names what it can and cannot prove.

**P1 — is a5 GM coherent for host DMA→AICPU?** The claim that would retire
unknown 1. Host writes a known sentinel to a descriptor-sized slot block twice
with different values across two runs; the AICPU reads the word at the earliest
point (platform kernel entry) and records what it saw, with **no** cache
operation anywhere on the path. Run the same binary shape on a2a3 as the
**positive control** — a2a3 is documented non-coherent, so a2a3 must show a stale
read for the probe to have any power. *Can* prove non-coherence (a stale read is
decisive). *Cannot* prove coherence: a run of fresh reads is consistent with
coherence, with a cold cache, and with an uncached mapping, which P2/P3 separate.

**P2 — cold cache or coherence?** Same probe, but the AICPU deliberately dirties
the line early in run *N* (write a different sentinel, no flush) and the host
publishes its sentinel for run *N+1*. If the AICPU reads its own dirty value in
*N+1*, the cache survives the kernel boundary and launch does not clear it; if it
reads the host's, launch or coherence covers it. *Cannot* distinguish launch-time
maintenance from coherence on its own — that needs P3.

**P3 — is the mapping cacheable?** Read the page attributes for an `rtMalloc`'d
GM range from the AICPU side, or time a repeated read of one line against a known
uncached MMIO read and a known cached GM read as the two reference points. A flat
ratio against the MMIO reference indicates uncached. *Cannot* be inferred from
timing alone without both references present in the same run.

**Provenance is part of every result.** Record the CANN version, the driver
version, the exact `libhost_runtime.so` / AICPU `.so` md5, and the commit — this
repo has been bitten by an incremental build that produced byte-identical
artifacts across arms, so the md5 is the control that the two arms actually
differ. Record device occupancy at each run; this box is shared and a timing-based
arm is not interpretable without it.

**What no probe here can do:** establish an architectural guarantee. A probe can
demonstrate a violation, or fail to find one under the shapes tried. Retiring
unknowns 1–3 as *guarantees* needs an SDK or architecture statement, not a
measurement.

## Why not fix anything now

Three of the four candidate "fixes" are mutually exclusive, and which is correct
depends on unknowns 1–3:

- If a5 is coherent: delete a5 HBG's two calls as redundant.
- If a5 is not coherent: add one to a5 TRB.
- If launch clears the cache or GM is uncached: delete all four.
- If the end-of-run placement is itself wrong: move all of them to before the
  first read — which means into the platform kernel entry, ahead of the affinity
  gate, not into the runtime executors where they are today.

Shipping any of them now would encode a guess as an invariant. The asymmetry is
not evidence for any particular one of them.

## What this hands to other work

- The three-writer ordering on the a5 HBG handshake line, and `aicore_done` as
  its serializing token, is an input to any later redesign that moves words into
  or out of that line or changes worker publication. It is not a constraint this
  entry invents — it is one the current code already depends on.
- The kernel-mode single-upload path (step 7) is a different visibility problem
  from program mode and should not inherit program mode's answer.
- Three documentation corrections are identified but **not made here**, because
  each depends on an unknown: the placement and extent mismatches in
  `cache-coherency.md` / `chip-architecture.md`, and a5 HBG's departure from the
  a5 rule. Correcting them requires knowing which of the readings is right; doing
  it now would replace one unsourced claim with another.

## Decisions for the maintainer, not taken here

1. Which unknown to retire first — 1, 2 or 3 — and whether a probe is worth the
   hardware time versus obtaining an SDK statement.
2. Whether the `aicore_done` ordering should become a stated invariant (a comment
   on the line, or a doc paragraph) independently of the unknowns. It is
   documentable today because it is a property of the current code, not of the
   hardware.
3. Whether `cache-coherency.md`'s a5 claim should be marked unsourced pending
   evidence, rather than left as a rule the code is checked against.

## References

- #2254 — the per-field inventory that raised the asymmetry.
- #2360 — the host-side prepare/publish seam and its current scope; this entry
  begins at the device side of the same transfer.
- #2309, #2315 — narrowed the copy to the descriptor, which is why the documented
  `sizeof(Runtime)` extent is stale.
- `docs/hardware/cache-coherency.md`, `docs/hardware/chip-architecture.md` — the
  rule this entry checks the code against.
- `docs/aicpu-kernel-launch-mechanisms.md` — the launch methods and the
  publication seam.
- `.claude/rules/ascend.md` — hard hardware constraints, including the
  `Device-nGnRE` MMIO attribute that does **not** extend to GM.
