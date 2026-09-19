# The Runtime descriptor's input-visibility chain: what the code expresses and what remains unestablished

**Date**: 2026-09-19
**Verdict**: host↔device coherence supplied as a project premise; the remaining
ordering, atomicity, ownership and lifetime obligations stay open — no defect
established, no cache operation changed, no contract proposed or proved
**Revision**: the first version of this entry shipped in #2368. Static review found
that several of its conclusions promoted source control flow, or observations that
do not discriminate, into hardware guarantees. Those conclusions are withdrawn in
place below — each marked **Withdrawn from the first version** — rather than
deleted, so a reader arriving from #2368 can see which claim was retracted and
why. Nothing in the source changed between the two versions; the baseline SHA is
the same.

## Question

The per-field inventory on #2254 noted an asymmetry: three of the four onboard
variants call `cache_invalidate_range` over the whole device descriptor, and a5
`tensormap_and_ringbuffer` does not. The tempting reading is "a5 TRB is missing
an invalidate."

That reading assumes the operation's only role is making the next invocation's
host bytes visible. This entry traces what the code actually expresses, for each
variant and each processor, and separates three things the earlier note ran
together: **what the code does**, **what that alone establishes**, and **what it
merely expresses an intention about**.

The outcome is that the asymmetry is real and is not evidence for any particular
change. The host↔device coherence prerequisite is **supplied as a project
architectural premise** (A3 not coherent, A5 coherent — see the premise section
below) and is no longer open; the remaining obligations — ordering, publication
atomicity, writer ownership, outbound publication, run lifetime and cross-stream
dependencies — are not settled by it and stay open. Several statements in the first
version of this entry are withdrawn below and marked as such.

## Baseline

Every anchor was read at **`31da0560e202b7bb378773f7dfc1d0074fb28e89`**
(`main`, 2026-09-19). Nothing here was built, run, simulated or measured; this is
a reading of source at one commit. No number from another commit is carried
forward as applying to this one.

## The chain, stage by stage

### 1. Host source snapshot

The host mutates its own `Runtime` object throughout `prepare_execution`.
`KernelArgsHelper::prepare_runtime_args`
(`src/common/platform/onboard/host/device_runner_helpers.cpp:80`) snapshots exactly
the device-read prefix into host heap storage: `RuntimeLaunchImage::prepare`
(`src/common/platform/include/host/runtime_launch_image.h:33`) resizes a
`std::vector<std::byte>` to `runtime_device_copy_size(runtime)` and `memcpy`s from
`&runtime`.

### 2. Allocation and reuse

The destination is not per-run. It is owned by the pipeline slot —
`SlotPersistentArgs::runtime_args`, one per slot in
`std::array<SlotPersistentArgs, PTO_PIPELINE_MAX_DEPTH>`, allocated lazily on a
slot's first run (`device_runner_helpers.cpp:99-107`) and released only in
`finalize_common_impl`. Per-slot rather than per-runner because the copy is not
stream-ordered, so a prepared successor would otherwise overwrite the image its
predecessor is executing against.

So the same device address is written by successive runs on that slot. Whether any
device cache line covering it retains bytes from a previous invocation depends on
the mapping and on cache behaviour across a kernel boundary, neither of which is
established here.

> **Withdrawn from the first version:** "a fresh allocation per run would make
> most of what follows moot." A new allocation is new *ownership*; it is not proof
> of new physical backing or of a cold cache, and this entry elsewhere records
> both as unknown. The sentence assumed what it was investigating.

**What the code does establish about reuse.** Two separate predicates, which the
first version of this entry ran together:

- **Same-slot exclusion is unconditional.** `try_reserve_native_run`
  (`device_runner_base.cpp:2919`) rejects a reservation outright if any live
  reservation names the same `pipeline_slot` — or the same owner — before any
  successor logic runs (`:2931`). A live reservation on a slot therefore blocks
  that slot entirely; there is no "unless".
- **The prepared-successor rule is about the predecessor, and needs a different
  slot.** The later condition (`:2940`) additionally requires
  `active == existing->owner`, i.e. that the *existing* reservation is the
  currently-active run. It says nothing about the caller being active. Combined
  with the unconditional check above, a prepared successor necessarily occupies a
  **distinct** slot.

So same-slot exclusion does not rest on the successor-admission rule, and must not
be argued from it.

**Same-slot reuse follows completed prior device execution.** On the success path
this is a device-execution boundary, not merely host call ordering.
`drain_execution` calls `reap_run`, which calls `wait_run_fence`
(`device_runner_base.cpp:2539`) before any teardown: when the run is fenced that
waits **both** run-completion events and then additionally calls
`sync_stream_pair` on both streams; when no boundary covers the submitted work it
falls back to `sync_stream_pair` on both streams directly. A non-zero result
returns early into recovery, so **no completion is claimed for a failed
synchronization** — the concrete contract holds on the success path only.

What that establishes is that the prior run's device execution has completed, and
that its reservation is released, before the slot is prepared again. What it does
**not** establish is anything about cache retention or input freshness across the
boundary: neither an event boundary nor a stream synchronize is a statement about
which lines a device cache still holds, or about when a device-side write-back
retires. Execution completion and cache state are different properties, and this
entry's open questions are about the second. Nor does this end-of-run boundary bear
on the still-unproved startup marker / whole-line hand-off discussed later — that
is a different edge of the same run.

### 3. Prepare/publish transfer

`RuntimeLaunchImage::publish` moves the snapshot into a function-local vector,
clears the member, and hands the local's `data()`/`size()` to a copy callback;
`KernelArgsHelper::publish_runtime_args` (`device_runner_helpers.cpp:113`) supplies
a synchronous `rtMemcpy(..., RT_MEMCPY_HOST_TO_DEVICE)`.

- The moved-to local lives until `publish` returns, so a **synchronous** copy
  reads a valid source. An asynchronous copy would not: the source dies at return.
- The snapshot is consumed exactly once; a repeated or reentrant publish is
  rejected rather than copying twice.

PR #2360 documents this host-side seam, including that the sole production caller
`init_runtime_args_with_metadata` (`device_runner_base.cpp:2782`) invokes prepare
and publish consecutively, and that a non-null `args.runtime_args` does not prove
publication happened. This entry starts where that stops, at the device side.

### 4. Launch ordering — two streams, no ordering between them

The descriptor's device address reaches both processors as a by-value launch
argument, so the *pointer* is fresh per launch; the *pointee* is device memory.

Publication precedes both launches. The two kernels then go to **distinct
streams, AICore submitted first**: `launch_aicore_kernel(streams.aicore, …)`
followed by `launch_aicpu_kernel(streams.aicpu, …)`
(`src/a2a3/platform/onboard/host/device_runner.cpp:664-687`; a5 has the same
shape). Nothing orders one stream's execution against the other's.

This is the correction that reshapes step 5: **there is no basis for a single
chronological "first device read" across both processors.** First reads must be
given per processor.

### 5. First descriptor reads, per processor and mode

> **Withdrawn from the first version:** "the earliest reader is the platform AICPU
> kernel entry … on all four variants." That is the earliest traced **AICPU**
> reader. It says nothing about the AICore, which is submitted first on another
> stream.

**AICPU side, all variants.** The earliest traced AICPU read is the platform
kernel's affinity gate, before `aicpu_execute`:

| arch | anchor | reads |
| ---- | ------ | ----- |
| a2a3 | `src/a2a3/platform/onboard/aicpu/kernel.cpp:65,93` | `get_aicpu_allowed_cpu_count()`, `get_aicpu_launch_count()`, `get_aicpu_allowed_cpus()` |
| a5 | `src/a5/platform/onboard/aicpu/kernel.cpp:82,121` | same three |

Neither file performs any cache operation. Because the gate decides which threads
proceed, a stale read here changes the thread population rather than a value.

On a5 HBG a second AICPU read of the same kind follows: `aicpu_execute`
(`src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp:627`) branches on
`aicore_scheduler_runtime_enabled(runtime)`, reading the host-published
`dev.workers[0].aicpu_ready` (`aicpu/aicore_scheduler_state.h:27`) to choose the
resident or legacy executor.

**AICore side.** The first descriptor access differs by path, and on a5 HBG it is
a *read of host-published state* that precedes any AICPU permission:

| path | first AICore descriptor access | before AICPU permission? |
| ---- | ------------------------------ | ------------------------ |
| a5 HBG (mode select) | `scheduler_observe_cache_line(handshake)` then `handshake->aicpu_ready` mode checks (`aicore/aicore_executor.cpp:533-535`) | yes — this read selects resident vs legacy |
| a5 HBG resident | then host-published `handshake->task` as `SchedulerWorkerContext*`, and that pointee, before READY (`:553-561`) | yes |
| a2a3 TRB | **write** of `physical_core_id`/`core_type`/`aicore_done` to `workers[block_idx]`, then `dcci(…, CACHELINE_OUT)` (`aicore/aicore_executor.cpp:72-83`); first explicit read is `my_hank->task` after window-open (`:103-104`) | the first access is a write, not a read |
| a2a3 HBG | same shape (`aicore/aicore_executor.cpp:72-83`, read at `:103-104`) | same |
| a5 TRB | same shape (`aicore/aicore_executor.cpp` report then window-open read) | same |
| a5 HBG legacy | entered via the mode read above; thereafter the report-then-window-open shape | mode read yes |

So a5 HBG is the only path whose AICore consumes host-published descriptor
content early; on the other paths the AICore's first descriptor access is its own
report write, and its first *read* is the task pointer after window-open.

### 6. Last descriptor access and cache operations, per path

The only descriptor-**wide** cache operation is
`cache_invalidate_range(runtime, sizeof(runtime->dev))`, at four sites, all inside
`deinit()` — the end of a run, not before a read:

| site | variant |
| ---- | ------- |
| `a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:1035` | a2a3 TRB |
| `a2a3/runtime/host_build_graph/aicpu/aicpu_executor.cpp:455` | a2a3 HBG |
| `a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp:563` | a5 HBG resident |
| `a5/runtime/host_build_graph/aicpu/aicpu_legacy_executor.cpp:414` | a5 HBG legacy |

a5 TRB's `deinit` takes `Runtime * /*runtime*/` unnamed and performs none
(`a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:1029`).

**AICore-side final descriptor access differs per path and must not be pooled:**

| path | last descriptor access on the AICore |
| ---- | ------------------------------------ |
| a2a3 TRB | `dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT)` (`:276`), then a **read** — bypass-load of `dev.teardown_gates[block_idx].post_close_release` (`:280`) |
| a2a3 HBG | same shape: `dcci` (`:301`), then teardown-gate read (`:305`) |
| a5 TRB | `dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT)` (`:255`); no teardown gate exists on this variant |
| a5 HBG legacy | final `dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT)` before return, commented "Flush all dirty cache lines to HBM before kernel exit" (`aicore_legacy_executor.cpp`, end of function) |
| a5 HBG resident | **no final handshake `dcci`**; the function ends at `write_reg(RegId::COND, AICORE_EXITED_VALUE)` (`aicore_executor.cpp:695`) |

The a2a3 paths' last descriptor access is a *read* placed after their own
write-back, which is a different exit shape from a5 TRB's write-back-and-return
and from a5 HBG resident's no-final-flush. The resident path's exit must not be
described using the legacy path's flush.

**What the primitive is** (`src/common/platform/onboard/aicpu/cache_ops.cpp:20`):

- **`dc civac`** — clean *and* invalidate to Point of Coherency, not a pure
  invalidate. It has two roles: it can make externally written bytes visible to
  this observer, and it can push this observer's modified bytes outward.
- Per 64-byte line over `[addr, addr+size)` rounded **outward**, so it can touch
  bytes outside the requested range — what the
  `sizeof(DeviceRuntimeLaunchDesc) % 64 == 0` static_assert protects.
- Followed by `dsb sy` and `isb`.
- The non-aarch64 branch (`:54`) and the whole sim implementation
  (`src/common/platform/sim/aicpu/cache_ops.cpp`) are inert, so no simulation run
  exercises any of this.

The AICPU **does store** into descriptor lines — `workers[i].task`
(`scheduler_cold_path.cpp`) and, on a2a3, `teardown_gates`. Whether any given line
is *dirty* at the `dc civac` call depends on the mapping's write policy and on
prior maintenance, neither of which is established here; so the clean half of that
call has a *possible* role in publishing AICPU-authored bytes, not a demonstrated
one. Either way, the call is not reducible to "invalidate for the next DMA".

> **Withdrawn from the first version:** "the `dc civac` is performing a write-back
> of AICPU-authored bytes." Observed stores do not establish a dirty write-back
> cache, which unknown 3 says is unestablished.

### 7. Reuse, and what kernel mode does not do

Program mode rewrites the same slot block and repeats.

**Kernel mode is scaffolding at this baseline, and its device-read chain is not
delivered.** `PersistentKernelArgs::prepare_once`
(`kernel_persistent_args.cpp:24`, called once from `device_runner_base.cpp:871`)
does allocate and H2D the descriptor. But `simpler_kernel_mode_supported` returns
**0** (`c_api_shared.cpp:1561`) and `simpler_kernel_mode_launch`
(`c_api_shared.cpp:1700`) validates its arguments and then **unconditionally**
returns `PTO_RUNTIME_ERR_INVALID_STATE`; the section header says so directly —
*"Launch remains a rejecting stub, so supported() reports 0."*

| kernel-mode stage | status at this baseline |
| ----------------- | ----------------------- |
| allocation + single upload | implemented (`prepare_once`) |
| repeated launch through affinity gate / executor | **not delivered** — launch is a rejecting stub |
| repeated-run first/last device read | **not delivered**, therefore not traced |
| end-of-run maintenance over an unrewritten descriptor | **not delivered** |

> **Withdrawn from the first version:** the claim that in kernel mode "every
> subsequent run reads descriptor bytes the host has not rewritten", and the
> matrix row asserting its first/last reads. No production path repeatedly
> launches this prepared descriptor, so no repeated-run cache contract can be
> inferred from the prepare-once helper.

## Where the documented rule and the code differ

`docs/hardware/cache-coherency.md:130-145` states the rule;
`docs/hardware/chip-architecture.md:218-219` restates it in the end-to-end flow.
Two differences are plain matters of location and extent, independent of any
hardware question, and are corrected in those files by this change:

1. **Placement.** The rule reads *"before reading host-written Runtime"* and the
   flow puts it at AICPU entry. All four call sites are in `deinit()`, at the end
   of the previous run.
2. **Extent.** The rule's snippet is `sizeof(Runtime)`; the code is
   `sizeof(runtime->dev)`, narrowed by #2309/#2315.

A third difference is **not** corrected, because settling it needs evidence this
entry does not have: the rule says a5 does not call the operation for host-DMA
reads, and its decision table says so for a5 generally, yet **a5 HBG calls it at
two sites** while a5 TRB does not. Recorded as an open discrepancy.

`cache-coherency.md` now carries the verified history of that discrepancy: the call
was removed from both a5 runtimes by #1235, is present again in a5 HBG after the
migration in #1661 whose six enumerated deviations do not mention it, and was carried
into `aicpu_legacy_executor.cpp` from the point #2090 created that file. A migration
copy is *one* reading of that sequence, and the provenance note labels it an
inference: the history fixes when the call returned, not why it was kept. History
proves neither a hardware model nor its absence, so this lineage is a maintenance
fact and is independent of the coherence premise recorded above.

## The a5 HBG resident handshake line: three writers, and a dependency the code expresses

**Scope: a5 HBG resident startup.** This section is about that path only. It is
not a lifetime account of the other paths, whose exits are in step 6.

`Handshake` is exactly 64 B and 64 B-aligned (`src/common/host_build_graph/runtime.h:98-112`),
so one worker's handshake is one cache line. On a5 HBG that line is written by all
three tiers:

| writer | words | operation |
| ------ | ----- | --------- |
| host | `aicpu_ready` = `SCHEDULER_RUNTIME_MODE_*`, `task` = worker's `SchedulerWorkerContext` address | part of the descriptor H2D (`a5/runtime/host_build_graph/host/runtime_maker.cpp:977`, `:1275-1277`) |
| AICore | `physical_core_id`, `core_type`, `aicore_done` | store, then `dcci(…, CACHELINE_OUT)`, then `dsb` |
| AICPU | `task`, `aicpu_ready` = `RESIDENT_READY` | stores, then `cache_flush_range` = `dc cvac` over whole lines |

The sequence the code expresses, in order:

1. AICore: `scheduler_observe_cache_line(handshake)` (`aicore_executor.cpp:533`),
   read `aicpu_ready` to select mode (`:534-535`).
2. AICore: store `physical_core_id`, `core_type`; `OUT_OF_ORDER_STORE_BARRIER()`;
   store `aicore_done`; `dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT)`;
   `dsb` (`:541-546`).
3. AICore: read host-published `handshake->task`, then its pointee, then enter the
   READY wait, re-observing the line each iteration (`:553-592`).
4. AICPU: per worker, `cache_invalidate_range(handshake, sizeof(*handshake))` then
   read `aicore_done`, spinning until non-zero, then read the reported payload
   (`aicpu/aicore_lifecycle.cpp:110-128`).
5. AICPU: only after report collection and configuration —
   `publish_context_partition` (`aicore_lifecycle.cpp:311`) stores `task` and
   `RESIDENT_READY`, then `cache_flush_range` over those lines, then `wmb()`
   (`:322-328`).

**What this establishes:** the code expresses a dependency — the AICPU's
publication is placed after it has observed `aicore_done` from each core. That is
where the intended hand-off is written down, and it is a fact about control flow.

**What this does not establish**, and what the marker read alone cannot:

- That observing `aicore_done` implies the producer's **whole-line write-back has
  completed**. The `dsb` at step 2 orders the producer's own subsequent accesses;
  it does not make a remote load of the marker an observation that the `dcci`
  retired. Marker visibility and line-writeback completion are separate
  properties, and nothing here ties them.
- That the AICPU's **clean** at step 5 — `cache_flush_range`, `dc cvac`, outbound
  only — cannot write back an older cached copy of the line, overwriting words
  another tier published. The AICPU's inbound operation is separate: the
  `cache_invalidate_range` (`dc civac`) in its step 4 poll. Keeping the two apart
  matters, because the outbound and inbound edges of this line are what the open
  questions distinguish.
- That the AICore's `CACHELINE_OUT` at step 2 cannot write back a copy that
  predates a concurrent write by another tier.

These are precisely the publication and ownership assumptions this entry leaves
unresolved elsewhere, so the dependency cannot be promoted into a correctness
result using the same document's unknowns.

> **Withdrawn from the first version:** "What keeps it correct is an ordering
> token", "the three whole-line writers are serialized", and the characterisation
> of this as a proved invariant ready to publish independently. **No currently
> executing race is established by this finding either** — the point is that the
> code's intent is legible while its hardware preconditions are not recorded.

What is worth carrying forward is narrower and still useful: a single 64-byte line
carries words authored by three tiers, each publishing with a whole-line
operation, and the only thing sequencing them is a marker dependency whose
hardware preconditions are unwritten. Any change that adds a word to this line,
alters who writes it, or relaxes the report dependency is changing something whose
safety argument does not currently exist in written form.

## Assumptions this entry explicitly does not make

- **That an end-of-run invalidate establishes next-invocation visibility.** It
  completes before the next launch, but `dsb sy; isb` after `dc civac` orders the
  maintenance, not anything against a *later* host DMA.
- **That a slot's first run is covered on a non-coherent path.** On a slot's first
  run no previous `deinit` ran, so no descriptor-wide maintenance has ever executed
  for those lines. On A5 the premise removes the need; on A3 it does not, and what
  covers that first read is open.
- **That invalidation is an acquire.** It is not.
- **That stores imply a dirty write-back cache** (see step 6).
- **That passing CI bounds failure frequency.** It is an observation of the
  exercised cases. Without a defined experiment and sampling basis it is not a
  quantitative bound, and the first version of this entry wrongly implied one.

## Supplied premise: the host↔device coherence model

The project owner states the architecture: **A3 host/device is not cache-coherent; A5
host/device is.** This entry takes that as given and no longer lists it as an open
question. It is a project architectural premise — **not** a vendor specification quoted
here, and **not** something measured for this entry.

What it settles: the **inbound host→AICPU freshness edge** per architecture. On A5 a
host-published descriptor byte becomes visible to an AICPU read without AICPU-side
invalidation. On A3 it does not, so on that path **some** mechanism must make the host's
bytes visible before the reader's first read. Which mechanism, and whether the
descriptor call at its current end-of-previous-run placement supplies it, is not
established here — the premise states the architecture, not the role of any particular
call.

What it does **not** settle, and what the rest of this entry is about:

- **Ordering.** Coherence supplies no happens-before between a writer and a reader.
- **Atomicity of a multi-field publication.** A descriptor or a 64-byte handshake line
  updated field-by-field is not made a single observable event by coherence.
- **Writer ownership.** It grants no permission to overwrite a field another agent is
  live on, which is the whole subject of the three-writer section above.
- **The outbound direction.** `cache_invalidate_range` is `dc civac`, which also
  cleans; what that clean publishes to on-device readers is a separate obligation.
- **Anything off the host↔device edge.** On-device producers — AICore, SDMA — and the
  question of how GM is mapped are not covered. Nor is **A2**: the owner named A3, the
  `a2a3` tree serves both and takes the conservative path, and that is where this entry
  leaves it.

**It authorises no code change.** In particular it does not license deleting any cache
operation now in the tree, nor a handshake redesign.

## Remaining obligations, with the premise applied

| obligation | status with the premise |
| ---------- | ----------------------- |
| inbound host→AICPU freshness, A5 | **supplied** — coherent per the premise |
| inbound host→AICPU freshness, A3 | non-coherent per the premise, so maintenance is required there. Whether the existing **end-of-previous-run** placement discharges it for the next run's first read is a separate question this entry does not answer — nor is any defect claimed; see step 6 for where the calls sit, and note a slot's first run has no prior `deinit` |
| launch-time cache state | still open, and now only relevant where the premise leaves work to do: it bears on A3's placement question above, not on A5's inbound edge |
| outbound AICPU→AICore/host publication | **still open** — the clean half of `dc civac`, untouched by an inbound coherence premise |
| whole-line / multi-field publication atomicity | **still open** — see the three-writer section |
| ordering and happens-before between the three writers | **still open** — the marker dependency and its unproved preconditions |
| allocation and run lifetime, slot reuse | **still open** as described in steps 2 and 7 |
| cross-stream dependencies (AICore and AICPU launched independently) | **still open** as described in step 4 |
| GM mapping and write policy | **still open**, and not answerable from the premise |

## Unknowns, and what each would and would not settle

The questions below remain unanswerable from this repository. Crucially, **none of them
alone licenses removing or relocating an existing operation**, because the
descriptor-wide call has more than one possible role (step 6).

| unknown | if resolved, addresses | does **not** address |
| ------- | ---------------------- | -------------------- |
| **1. Does AICPU kernel launch leave the cache cold or perform maintenance?** | on A3, whether the end-of-previous-run placement suffices for the next run's first read | whether prior device-authored bytes needed publishing to another observer before that point — a cold next cache says nothing about the previous run's outbound obligations. Moot for A5's inbound edge, which the premise settles |
| **2. What publishes AICPU-authored descriptor bytes outward, and to whom?** The clean half of `dc civac`, plus the `dc cvac` in a5 HBG's context publication. | whether removing or moving either call would drop a publication an on-device reader depends on | inbound freshness, which the premise covers per architecture |
| **3. Is GM cacheable for the AICPU, and under what write policy?** The documented `Device-nGnRE` attribute covers the **MMIO** window only. | whether cached descriptor lines can exist at all, and whether the clean half can have anything to write back | nothing about externally written staleness **if the answer is write-through**: a write-through cache still holds readable lines, and the write policy alone does not establish that host DMA invalidates or updates them |

> **Withdrawn from the first version:** the claim that an uncached **or
> write-through** mapping "dissolves" the question. Write-through concerns writes;
> it is not a sufficient answer to input freshness. Also withdrawn: the
> "mutually exclusive candidate fixes" list, which presented each unknown as
> licensing a specific deletion.

## Obligations any future change must account for

Rather than candidate fixes, the individual obligations the existing
descriptor-wide call could be discharging. A change may only remove or relocate
the call once **every** row it currently covers is accounted for by something:

| obligation | who needs it | currently discharged by |
| ---------- | ------------ | ----------------------- |
| host-published bytes visible to the AICPU's first read | AICPU affinity gate, executor init, a5 HBG mode select | on A5, the coherence premise. On A3 the premise says maintenance is needed, and nothing in the same invocation supplies it — unknown 1 covers whether the end-of-previous-run placement discharges it |
| host-published bytes visible to the AICore's early reads | a5 HBG resident mode + context reads | the AICore's own `scheduler_observe_cache_line`. The host↔AICPU premise does not speak to an AICore reader |
| AICPU-authored descriptor bytes made visible outward | AICore readers of `workers[i].task`; host D2H diagnostics | possibly the clean half of `dc civac`; unknowns 2–3 |
| a slot's **first**-run reads, AICPU observer, A5 | AICPU affinity gate, executor init | the coherence premise — a first run is not a special case for an edge that needs no maintenance |
| a slot's **first**-run reads, AICPU observer, A3 | same | nothing in that invocation, and no prior `deinit` has run for those lines. What the non-coherent path requires there is open, as above |
| a slot's **first**-run reads, AICore observer | a5 HBG resident mode + context reads | the AICore's own `scheduler_observe_cache_line`, on every run including the first |
| line-granular exclusivity on the a5 HBG handshake line | host, AICPU, AICore | a marker dependency with unwritten preconditions |

Relocating the call to before the first read would move it into the platform
kernel entry ahead of the affinity gate, not into the runtime executors where the
four sites sit today — and would not by itself discharge the outward-publication
row.

## Open questions for future measurement — not a proven probe plan

The first version presented these as probes with decisive outcomes. They are
narrowed here to what each could observe and what would remain ambiguous.
**Nothing was run for this entry, and none of this is an actionable plan yet.**

**Q1 — on A3, does the end-of-previous-run placement discharge the inbound
obligation?** The coherence question this slot used to hold is answered by the premise;
what is left is the A3 placement question, where the premise says maintenance *is*
required. Observing it has the same difficulty the original framing did: a run must
first **establish that a stale readable line existed at the moment of the read** — a
verified initial cache state plus a read-back confirming the line's content before the
host write — because fresh reads on their own cannot show which mechanism prevented
staleness. The premise supplies the architecture, not the initial state of any
particular line. Without that, a run is a comparison rather than a control.

**Q2 — did the reader's cache survive the kernel boundary?** The
dirty-sentinel idea does not discriminate. Reading the old value is consistent
with a surviving cache *and* with the old dirty line having been written back over
the later host value and then read fresh from memory. Reading the host value is
consistent with launch maintenance, with coherence, *and* with plain eviction
before the read. Because the experiment introduces a write-back race into a
read-freshness question, it needs observations that separate those four cases —
at minimum a verified initial state and a way to distinguish "read from cache"
from "read from memory" — before any verdict is assignable.

**Q3 — the mapping's attributes: a question, not a probe.** Timing against an
MMIO reference cannot classify page attributes even with a second known-cached
reference: access path, ordering attributes and latency all differ, so a ratio
reports the sampled access behaviour rather than the mapping. The
"flat ratio indicates uncached" rule from the first version is **withdrawn**. This
question is answered by verified page attributes or a documented mapping
contract, not by measurement from inside the runtime.

**Provenance, for whenever any of this is attempted:** record the CANN and driver
version, the exact `libhost_runtime.so` / AICPU `.so` md5, the commit, and device
occupancy. This repository has been misled by an incremental build producing
byte-identical artifacts across arms; the md5 is the control that the arms differ
at all.

**What no measurement here can do:** establish an architectural guarantee. It can
demonstrate a violation. Retiring unknowns 1–3 as guarantees needs an SDK or
architecture statement — the kind the host↔device premise supplies for its own edge,
and which the remaining unknowns do not have.

## Why nothing is changed in the runtime

The coherence premise settles the inbound edge per architecture; it does not discharge
the outbound, atomicity, ordering or lifetime obligations, and the remaining unknowns
do not either. Adding an invalidate to a5 TRB, or deleting a5 HBG's two calls, or
deleting all four, would each rest on the premise answering more than it does — the
clean half of the operation stays unaccounted for in all three.

## What this hands to other work

- The a5 HBG resident handshake line carrying words from three tiers, with only a
  marker dependency sequencing them and no written hardware preconditions, is an
  input to any later redesign touching that line or worker publication. It is
  offered as an unresolved constraint, not as a proved invariant.
- Kernel mode's device-read chain is **not delivered** at this baseline and must
  be traced when launch is implemented, not inferred from `prepare_once`.
- The per-path exit shapes in step 6 differ; a lifetime account for one path may
  not be reused for another.

## Decisions for the maintainer, not taken here

1. Which unknown to pursue first, and whether an SDK/architecture statement is the
   cheaper route than any device experiment.
2. Whether to record the a5 HBG three-tier line and its marker dependency as a
   stated *constraint* — not an invariant — near the struct, given that its
   preconditions are unestablished.
3. Whether the host↔device premise should also be recorded anywhere beyond
   `cache-coherency.md` and this entry — and, separately, whether the on-device
   coherency rows in that file (AICore, SDMA) warrant their own basis being written
   down, since the premise does not reach them.

## References

- #2254 — the per-field inventory that raised the asymmetry.
- #2360 — the host-side prepare/publish seam; this entry begins at the device side.
- #2309, #2315 — narrowed the copy to the descriptor, hence the stale extent.
- `docs/hardware/cache-coherency.md`, `docs/hardware/chip-architecture.md`.
- `docs/aicpu-kernel-launch-mechanisms.md` — launch methods and the seam.
- `.claude/rules/ascend.md` — the `Device-nGnRE` MMIO attribute that does not
  extend to GM.
