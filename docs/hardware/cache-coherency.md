# Cache Coherency (GM ↔ AICore/AICPU)

This page is the authoritative reference for **when to insert a cache
operation** on Ascend onboard hardware. The coherency model differs
between chip generations supported in this repo: a2a3 requires AICPU
cache maintenance for host-DMA and SDMA-published GM, while a5 GM
accesses need no AICPU-side cache clean or invalidate. Completion and
ordering remain explicit on both architectures.
Cycle costs and a few code-path examples below are sampled on a2a3
unless stated otherwise. Misapplying these rules either leaks stale data
(correctness bug) or burns a `dsb sy` per record on a hot path (perf
bug — see PR #863 lineage). Read it before touching any code that
writes `dc civac`, `dc cvac`, `cache_invalidate_range`, or any `dcci`
on AICore.

## Who shares GM, and with what consistency

GM (HBM) is read and written by several parties on Ascend onboard. Their
relationship to **AICPU's data cache** is **not symmetric** on a2a3,
and a5 has a stronger DMA/HBM coherency model:

| Writer | a2a3: AICPU sees write automatically? | a2a3: AICPU must invalidate before read? | a5: AICPU must invalidate before read? |
| ------ | ------------------------------------- | ---------------------------------------- | -------------------------------------- |
| AICPU itself | Yes (own cache) | No | No |
| **AICore** (AIC / AIV / MIX) | **Yes** | **No** | **No** |
| Host DMA (`rtMemcpy` from host RAM) | No | **Yes** | **No** |
| SDMA engine (device-side DMA) | No (assumed) | **Yes** | **No** |
| Device writes to RDMA/URMA completion records (CQEs) | N/A (unsupported) | N/A (unsupported) | **No** |

The RDMA/URMA backends are supported only on A5. A2/A3 is outside their
supported platform scope; the corresponding cache-policy entries are not applicable.

These are AICPU-side data-cache rules, not instructions to flush an
AICore L2 cache. The AICPU cache can contain an old copy of a GM address;
a non-coherent DMA update changes HBM without invalidating that copy.
An AICPU load can then return the old value even after the DMA has finished.
The maintenance decision follows the writer and architecture, not the
name of the copy API or whether the address happens to be in HBM.

**Basis: project architecture premises, clarified in #2745.** The owner
confirmed that A5 requires no AICPU-side GM cache maintenance, including
Host DMA, SDMA and communication completion reads, and that completed
AICPU GM writes are visible to consumers without an additional clean.
The a2a3 Host DMA path and conservative SDMA invalidation remain intact.
These are project-supplied architecture facts, not vendor specifications
or new silicon measurements. They do not establish ordering, atomicity,
writer ownership or safe reuse by themselves.

Likewise, **AICore's own cache is non-coherent with GM** in the other
direction: AICore-side writes stay in its data cache until explicitly
pushed out with `dcci`. AICPU writes that AICore needs to see follow a
mirror of this table (AICore must `dcci` to invalidate before reading
host- or AICPU-written GM) — except where the AICore reads with a
bypass load, for which see
[the AICPU → AICore section](#the-aicpu-to-aicore-path-for-one-access-shape-the-a2a3-teardown-gate).

The rest of this doc fills in why each row of the table is what it is,
and what code lives on each side.

## Cache primitives and barriers

| Primitive | Side | Purpose | Cost (rough, a2a3 / DAV_3510) |
| --------- | ---- | ------- | ----------------------------- |
| `dcci` (`__attribute__((aicore))` intrinsic) | AICore | Push a cache line out to GM (clean+invalidate). Required after AICore stores that AICPU or peer AICore must read. | 1 cache line per call + a following `dsb` to commit ordering. |
| `cache_invalidate_range(addr, size)` (`src/common/platform/include/aicpu/cache_maintenance.h`) | AICPU | a2a3 onboard: `dc civac` over whole cache lines plus completion barriers. Required before reading GM published by a **non-coherent writer** (Host DMA, conservative SDMA). A5 onboard and sim implementations are empty, with no cache operations or barriers. | a2a3: line count plus the completion barrier; `dsb sy` can dominate. |
| `cache_flush_range(addr, size)` | AICPU | a2a3 onboard: `dc cvac` plus completion barriers. A5 onboard and sim implementations are empty, with no cache operations or barriers. | a2a3: line count plus the completion barrier. |

`dc civac` means **clean and invalidate to the Point of Coherency**,
not pure invalidation: it can write dirty bytes back. The existing platform
build selects separate a2a3 onboard, A5 onboard and common sim source files.
Only a2a3 performs cache-line operations and their completion sequence
`dsb sy; isb` for nonempty ranges; zero-length ranges return immediately.
Callers, including shared Region and diagnostic code, express protocol
ordering through explicit `rmb()`/`wmb()` or their existing acquire/release
operations. A cache call supplies no ordering on A5 or sim.

Removing a cache call does not automatically require a replacement barrier.
Scalar completion polling and diagnostic snapshots do not consume a payload
published by the observed counter; their existing completion/dispatch protocols
provide the hand-off. Region notifications do publish payload for a following
read, so their stores use release and observations use acquire. The run-terminal
record is read after kernel completion with its slot reserved, so it needs no
additional A5 publication barrier. An error flag followed by reads of its error
details needs read ordering at that consumption point, not on every status poll.

`cache_invalidate_range` is the protocol-correct primitive for the
**host-DMA → AICPU** case on a2a3. It was introduced in PR #204
specifically for `Runtime` struct hand-off where the host writes via
`rtMemcpy` and AICPU reads. It is not required **for cache coherency**
when reading properly published AICore writes or a5 DMA/HBM writes.
Cache visibility and protocol ordering are separate: a caller must not rely
on the a2a3 implementation's embedded barrier for a cross-platform protocol.
See the [historical pitfall](#historical-pitfall).

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

For the task-slot handoff below, AICore and AICPU share a coherency
domain on GM. When AICore writes a slot, the correct handshake is:

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

- **Reset before window-open.** AICPU resets the request/release words in each
  64-byte `AicoreTeardownControl` with a whole-control clear, then executes
  `wmb()` in `pre_handshake_init`. AICPU is its only writer; AICore only
  bypass-loads the request and release words. The barrier orders
  the resets before `hs_setup_done_` and before any register window opens — a window is a
  plain `Device-nGnRE` store carrying no release semantics of its own. No core can read
  its gate before its window opens.
- **RELEASE after window-close.** The quiesce pass, then the CLOSE pass and its
  readbacks, then `rmb()`, then the relaxed `post_close_release` stores
  in `platform_retire_aicore_group`. The drain is what orders every store after
  the CLOSE it belongs to, so an open return gate is never paired with an open window.

**Lifetime across runs is a separate, open question.** Whether a reader left behind by a
failed or partially retired run can still be live when a later launch resets and
releases the same gates is **not** established — the recovery path is untraced, and
neither the premise nor the two edges above settle it. See
[the exceptional-exit limit in the descriptor visibility investigation](../investigations/2026-09-runtime-descriptor-input-visibility-chain.md#exceptional-and-partial-exits-audited-separately).

The AICore reads the release word with `acknowledge_aicore_cond_nonce_and_wait` in
`src/a2a3/platform/onboard/aicore/inner_kernel.h`: bypass `ld_dev` loads followed
by a final `dsb(DSB_DDR)`. The release word is permission-to-return. It shares an
AICPU-owned 64-byte line with `exit_request`. AICore publishes its response in
COND, so no GM ACK field or AICore writeback shares this line.
`AicoreTeardownControl` has 64-byte size and alignment and requires matching
host, AICPU and AICore builds.

### Exit acknowledgement freshness

`COND=EXITED` can survive a launch. A current report does not prove that the
current core has observed EXIT, and failure can precede its first IDLE. Waiting
for IDLE or using one fixed exit marker does not establish current retirement.

#### Unified COND nonce

A2/A3 and A5 HBG/TMR use `platform_retire_aicore_group` for Native, legacy and
unstamped launches. Native report epochs still commit the current physical-core
report; the epoch neither selects an exit protocol nor acknowledges retirement.

| Order | AICPU | AICore |
| ----- | ----- | ------ |
| 1 | Reset request/release before READY; accept the current report; claim each target once | Publish the current physical-core report |
| 2 | Read that target's retained COND; choose `0xfffffff1` or `0xfffffff2`, different from that observation | Continue or fail before the first IDLE |
| 3 | Store each request, `wmb()`, then broadcast DMB EXIT from the retirement helper | Observe current EXIT; complete preceding writes with `dsb(DSB_DDR)` |
| 4 | Accept only COND equal to that target's request, within the shared absolute deadline | Bypass-load the request, publish the nonce once, then perform no further SPR writes |
| 5 | Close acknowledged targets, read back, drain, then publish post-close release | Bypass-poll release; final drain; return |

The two nonce values are reserved FIN encodings excluded by the scheduler's
task-ID allocator. Ordinary FIN, IDLE and fixed EXITED cannot satisfy the match.
Selection uses the actual target's old COND, rather than a global launch parity:
a physical core can skip intervening launches. Only this physical core writes
its COND, and a uniquely claimed retirement publishes one request.

A5 HBG resident shutdown can broadcast EXIT before the partition's retirement
helper publishes its request. The Core waits for the reserved request before
echoing it; this ordering does not require a second protocol or a request
update. Its pre-READY failure and local-watchdog paths wait for actual AICPU
DMB EXIT before reading the control, because AICPU resets it before that signal.
Thus a prior request or RELEASE cannot let an early failure acknowledge or return.

The Core's completion barrier precedes the request load and dependent COND
write. AICPU accepts the MMIO nonce match itself; there is no following GM ACK
load, ACK-line invalidate, or AICore ACK-line initialization/writeback.
The a2a3 descriptor retains its Host DMA maintenance below; A5 uses
completion ordering without descriptor-wide cache operations.

The protocol proves a response to this retirement request without requiring an IDLE
observation. A silent core retains EXIT, its open window and its closed return
gate when the shared deadline expires; the original host recovery owns it.
Descriptor reuse still requires the previous generation to have completed or
been recovered. The nonce match does not make concurrent reuse valid.

The gate protocol alone does not decide descriptor cache policy. On a2a3,
the descriptor-wide `cache_invalidate_range` in `deinit` stays for the
non-coherent Host DMA path. On A5, the broader GM coherence premise covers
both incoming reads and outgoing completed AICPU writes, so neither side
requires a descriptor clean/invalidate. Required completion and retirement
ordering remains separate. The earlier, narrower audit is preserved in
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

The host distinction is a project premise: **A3 Host DMA is not coherent
with AICPU reads; A5 is.** The a2a3 implementation retains the conservative
path for both A2 and A3. The broader A5 premise at the top of this page
also covers outgoing completed AICPU writes; it does not change the
ordering, atomicity or lifetime requirements of a launch.

**In-tree history of the a5 difference.** Before #2745, a5
`host_build_graph` called `cache_invalidate_range` over the descriptor at
two sites while a5 `tensormap_and_ringbuffer` called it at none. The
architecture distinction entered the tree with
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
not why it was kept. The A5 descriptor call sites are removed under the
clarified coherence contract; the a2a3 descriptor maintenance stays.

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
SDMA completion helpers do not invalidate or flush cache lines. The same
A5 premise covers device-written RDMA/URMA completion records (CQEs);
their owner/sequence checks,
acquire ordering and completion retirement remain part of the backend
protocol. This follows the broader A5 GM premise recorded above, not an
inference from the narrower Host DMA distinction.

## Quick decision table

When you are about to insert a cache operation, ask in order:

1. Who actually wrote the bytes I'm reading? Look at the producer
   code, not the address.
2. Is this a5? If yes → no AICPU cache clean/invalidate for GM, including
   Host DMA, SDMA, communication completion and AICore publications.
   Preserve the protocol's completion and ordering requirements.
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
6. Am I about to add an AICPU clean to publish my store? On A5,
   completed AICPU GM writes need no clean. On a2a3, for an ordinary GM
   store read back by an AICore
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
- `src/a2a3/platform/onboard/aicpu/cache_ops.cpp` — a2a3 cache operations and their completion barriers.
- `src/a5/platform/onboard/aicpu/cache_ops.cpp` — empty A5 implementations, with no cache operations or barriers.
- `src/common/platform/sim/aicpu/cache_ops.cpp` — sim no-op.
- AICore-side `dcci` usage lives in the chip swimlane / PMU AICore collectors and any kernel that publishes to a GM slot AICPU reads.

## Related docs

- [PMU staging-slot ordering](../dfx/pmu-profiling.md) —
  detailed AICore-side `dcci` + barrier order for staging-slot writes.
- [chip swimlane profiling](../dfx/chip-swimlane-profiling.md) —
  the consumer of the rules above on the chip swimlane path.
