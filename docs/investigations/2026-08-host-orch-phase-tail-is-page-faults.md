# 2026-08 — The host-orchestration phase tail is page faults, not the code in the phase

> **Phase and type names in this entry are the ones the tooling emitted at the
> time.** The per-in-graph-task record phase was `record_node` and is now
> `record_in_graph_task`; `GraphRecordedNode` is now `RecordedInGraphTask` and
> `recording.nodes` is `recording.tasks`. The measurements and the archived run
> directories below keep the old spellings because that is what the logs say, and
> `strace_timing.py` still accepts `record_node` so those logs remain readable.

## Question

`host_build_graph`'s host-side bind path shows two shapes on every swimlane of the
dsv4 FLASH decode workload, and neither is explained by the code the phase names:

1. `graph_begin`, `record_node` and their neighbours have a small, stable median and a
   maximum two orders of magnitude above it. `record_node` over one orchestration:
   median 0.85 µs, mean 4.28 µs, max 162.78 µs, 9% of calls carrying 79% of the total.
2. The submitting thread has gaps far longer than any work the generated orchestration
   does between two runtime calls.

Fourteen attempts at shortening the code in those phases moved the control-plane total
by less than its run-to-run spread. This entry is why.

Counts below are per **orchestration**: one `bind_callable_to_runtime`'s host
orchestration, which is what a single `bind phase=host_orch` record spans. For this
workload that is 86 `graph_begin` (8 of which start a recording, 78 reuse one), 8
Definitions, 1679 recorded nodes, 19 ordinary tasks and 24 `alloc_tensors`. A
`--rounds N` run over two ranks performs 2N of them, and the Definition cache does not
survive a bind: two consecutive `host_orch` records in one process both report
`build_definition count=8` and `record_node count=1679`, so each one records everything
again.

## Answer

> **Two of this answer's three claims are superseded — read the last amendment
> before acting on it.** What holds is the *exclusion*: a writer of `mmap_lock`
> excludes every faulting thread in the address space, which is what makes a fault
> here cost 14–33 µs against ~1.7 µs on an idle box. What does not hold is the
> framing of the tail as *count × price* — two arms settle that in opposite
> directions, one removing 86% of the faults for no time and one removing 6% for
> 29–43% — nor `mmap`/`munmap` as the writer, which names what the off-tree
> reproducer below used. **In tree that writer is `mprotect`**, from glibc opening a
> non-main arena.

**The tail is minor page faults on freshly allocated memory, and a fault here costs
14–33 µs instead of the ~1.7 µs it costs on an idle box — because the process's own
`mmap`/`munmap` traffic holds `mmap_lock` for write and excludes every faulting
thread in the address space.**

Both shapes follow from that, and so does the measurement noise that hid it:

- Which phase or segment a long call is attributed to is **where the allocation
  happened to sit**, not where the work is. A call that spends 85 µs building one
  128-byte tensor is not doing that segment's work.
- The fault *count* is a deterministic property of the allocation pattern; the *cost
  per fault* is set by concurrent address-space activity, and varies 2.4× between runs
  of the same binary. Any in-tree duration below a millisecond is therefore mostly a
  measurement of the box's state.

## Evidence

### The long calls are faults, by count

A per-segment probe inside `record_node` (seven stamps, the capacity of every container
the call can grow, plus `getrusage(RUSAGE_THREAD)` minor faults and
`CLOCK_THREAD_CPUTIME_ID` for the same window), on 3358 recorded nodes across 16
Definitions:

| calls of `record_node` | calls | share of `record_node` time |
| ---------------------- | ----- | --------------------------- |
| took ≥ 1 minor fault | 638 / 3358 (19%) | **79%** |
| above 10 µs | 449 | 76% |
| above 10 µs **and** faulted | **447 of 449** | — |
| above 10 µs, no fault, off-CPU | 1 | — |
| above 10 µs, no fault, on-CPU | 1 | — |

Median 1 fault per long call, 22.5 µs per fault. Preemption is refuted by the same
table: exactly one long call was off the CPU without a fault.

### The count is the code's; the cost is the box's

Three runs of the same binary: **1063, 1065, 1168** faults — but 13.9, 22.5 and
32.7 µs per fault. The count is a property of what one orchestration allocates; the
cost is not.

### Which memory faults

Every segment allocates from exactly one place, and a fault costs 14–33 µs where the
segments' own work is sub-microsecond, so the segment that dominates a faulting call is
where the fault landed. Over the 447 faulting calls above 10 µs (867 faults, 19.5 ms):

| Allocation | Element size | calls | faults | share of faulted time |
| ---------- | ------------ | ----- | ------ | --------------------- |
| `node.tensors` — one fresh `vector<ChipTensor>` per node | 128 B | 151 | 174 | **31.7%** |
| `recording.nodes` — `vector<GraphRecordedNode>`, doubles | 120 B | 97 | 256 | **22.1%** |
| `recording.tensor_sources`, doubles | 24 B | 70 | 174 | **16.1%** |
| `recording.internal_fanins` (8 B) and `.predicates` (192 B), double | — | 77 | 195 | **14.7%** |
| `recording.tensor_map` entry pool, initialized on write | 128 B | 36 | 43 | 4.5% |
| `recording.scalars` (8 B) / `.scalar_sources` (16 B, since renamed `.scalar_inheritance` and narrowed to 4 B), double | — | 16 | 25 | 2.7% |

165 of the 447 faulting calls reallocated no recording-owned container at all, which
leaves the per-node vector as the allocation: it is the single largest source.

**But the structure that makes the others expensive is the hazard map.**
`ChipTensorMap::init` takes four `new[]` allocations per recording — 4096×8 buckets,
16384×128 entries, 16384×8 free list, 1024×8 task heads = **2.17 MB**, of which the 2 MB
entry pool is one block — and they are freed when the recording is destroyed at the end
of the orchestration. Only ~115 KB is ever touched (the buckets and task heads `init`
clears, plus the entries actually used), which is why it takes just 4.5% of the faults —
but eight recordings' worth per orchestration, allocated and freed far above glibc's
128 KB mmap and trim thresholds, is exactly the mapping traffic the reproducer below
shows inflating everyone else's faults 10×.

So both halves — how many faults there are, and what each one costs — trace to the same
act: **handing the recording's memory back to the kernel at the end of every orchestration**,
for a workload whose allocation shape is identical every time.

### `graph_begin`'s tail is not the `graph_submit` nested inside it

Pairing the two records by containment: nested `graph_submit` is 41% of `graph_begin`,
and of the 8 entries above 5× the median — which carry 46% of all `graph_begin` — **73%
is outside the nested submit**. The serial probe names which entries those are, by the
path the entry took:

| path | what it does | calls | sum | median | max | on-CPU |
| ---- | ------------ | ----- | --- | ------ | --- | ------ |
| 1 | hit a published Definition | 61 | 658 µs | 5.72 µs | 80.9 µs | 82% |
| 2 | hit an in-flight recording | 95 | 985 µs | 4.80 µs | 107.2 µs | 94% |
| **3** | **starts a recording** | **16** | **1103 µs** | **46.92 µs** | **220.3 µs** | **70%** |

9% of the calls carry 40% of `graph_begin`. Splitting path 3 into its four steps:

| step | what it does | share | median |
| ---- | ------------ | ----- | ------ |
| **`graph_recording_init_tensor_map`** | the 2.17 MB of `new[]` + the 40 KB `init` clears | **66.0%** | 19.52 µs |
| nested `graph_submit` | the outer shell | 12.7% | — |
| boundary deep copy | 25 `ChipTensor` + tags | 11.4% | 2.54 µs |
| the `GraphRecording` object | 2520 B, value-initialized | 5.9% | 1.38 µs |
| in-flight entry + map insert | | 0.8% | 0.41 µs |

The worst entry spends 160 of its 220 µs there. So the same allocation that inflates
every other thread's faults is also, two thirds of the time, what makes `graph_begin`
long — and unlike `record_node` this sits on the submitting thread, i.e. on the
orchestration's critical path: eight recording-starts at ~47 µs is ~240 µs of a ~1.3 ms
`host_orch`.

`init` already clears only the buckets and task heads rather than the whole pool, so what
costs here is acquiring the memory, not preparing it.

**Measurement base.** Every number here was taken at `b24092b9`, where the same 2.17 MB
came from one `DeviceArena::commit` (a single `std::malloc`) rather than four `new[]`.
PR #1962 changed the shape and left the sizes, the init-on-write entry pool and the
per-recording lifetime alone, so the finding carries over; anything re-measured should be
re-measured on top of it.

### Reproduced off-tree, including the inflation

`.docs/bench_recording_alloc.cpp` replicates only the allocation pattern — one fresh
vector per node plus six doubling containers, freed at the end, one orchestration per
thread concurrently. It reproduces both in-tree constants: **median 0.85 µs** and **72
faults per thread** (against 66 per Definition in the tree). It prices a fault at
**1.7 µs**.

Its third argument adds the one thing the pattern alone lacks — a thread mapping and
unmapping a region while the workers run:

| 8 threads, same pattern | per fault | worst call | on-CPU |
| ----------------------- | --------- | ---------- | ------ |
| alone | 1.7–1.9 µs | 20–32 µs | 100% |
| + one thread mmap/munmapping 64 MiB | 4.1–24.6 µs | 309–401 µs | 34–52% |

**Three** unmaps during the whole run were enough. The on-CPU fraction collapsing is
the signature: the faulting threads are blocked, not working. That is the in-tree
distribution, and it identifies the mechanism as address-space exclusion rather than
the fault itself.

### Removing the return-to-kernel behaviour removes the faults

Same binary, same workload, only glibc tunables (`MALLOC_MMAP_THRESHOLD_` and
`MALLOC_TRIM_THRESHOLD_` at 1 GiB, `MALLOC_TOP_PAD_` at 256 MiB), so freed memory is
kept rather than handed back:

| glibc behaviour | faults | faulted calls | calls > 10 µs | on-CPU |
| --------------- | ------ | ------------- | ------------- | ------ |
| baseline | 1063 | 638 (79% of time) | 449 | 79% |
| memory kept | **29** | **29 (3% of time)** | **16** | **99.9%** |

## What this does *not* show

**The tunables are not a fix and their A/B does not measure what the fix is worth.**

- In the probe build every orchestrator phase halved (`host_orch` 5.39 → 2.51 ms,
  `graph_begin` 2704 → 1239 µs per orchestration, `record_node` 16447 → 5696). In the clean build
  the same tunables move `record_node` 2745 → 1993 µs per orchestration but leave `host_orch`
  unchanged (1.322 → 1.409 ms) and make `graph_begin` *worse* (550 → 1043 µs).
- Two clean runs of identical code differ by more than that: `host_orch` 2.049 vs
  1.322 ms, `record_node` 4587 vs 2745, `graph_begin` 897 vs 550. The clean A/B is
  inside the run-to-run band and resolves nothing.
- `args` regresses badly (1.67 → 2.68 s): with the mmap threshold at 1 GiB, the 42 GiB
  of staging comes off the heap top.

Two lessons, both already cost time here:

- **The probe amplified the effect it measured.** 1679 `getrusage` syscalls plus eight
  clock reads per node widen the window in which a fault and its lock wait can land, so
  the probe build's −53% is an artifact of the probe. Only counts and off-tree
  measurements from that build are usable. This is the third time in this investigation
  that a probe perturbed its own subject, after a `LOG_WARN` inside a measured window
  and a `printf` reading stack garbage.
- **Recorder-side savings are not whole-orchestration savings.** `record_node` runs on eight
  recorder threads in parallel with the submitting thread, so deleting its faults
  shortens the orchestration only where `recording_wait` is on the critical path.

## Refuted

| Hypothesis | How it died |
| ---------- | ----------- |
| **Transparent huge pages** (`enabled=[always]`, so a first touch can fault in 2 MB, and 22 µs is about what clearing one costs) | `PR_SET_THP_DISABLE` via an `LD_PRELOAD` constructor: the fault count is unchanged (1063 → 1168, where 2 MB pages becoming 4 KB ones would multiply it), and the cost per fault *rose* |
| **Preemption / descheduling** | 1 of 449 long calls was off-CPU without a fault; `nivcsw` = 0 in earlier per-phase counters |
| **The node's own work** (more tensors, more fanins, a bigger hazard map) | 85 µs to build one 128-byte tensor; the long segment is a different one on every long call |
| **Node shape** as an explanation of the deterministic half | The probe read fields that do not bound the work: `fanin_count` is 0 for all 1679 nodes (a Graph node's producers go into the payload's fanin region, not `internal_fanins`) and `tensors.size()` reached 306, past every per-task cap. Its r = −0.02 was meaningless, not informative |
| **Per-phase page-fault counts** as evidence either way | Correlation of `host_orch` against `minflt` over whole phases is −0.71. Faults matter *per call*; at phase granularity the load proxy (`args`, r = +0.79) dominates |

## Where a fix would go

**All four have shipped — see the amendments below before treating any of them as open
work.** Items 1-3 landed as #1981 (the recorder thread owns its recording storage), item 4
as #1988 and the retained SM mirror. What the list got right was the mechanism; what it
got wrong is that none of it reached the ~1100 minor faults the submitting thread takes
per bind, which is what is actually left. In order of the evidence behind them:

1. **Stop returning the recording's memory to the kernel between orchestrations.** The 2.17 MB
   hazard-map arena and the per-node vectors are re-acquired every orchestration for a
   workload whose shape is identical every time. A per-recorder pool that outlives one
   removes the faults at their source, with no dependence on glibc tunables and without
   `args`'s regression. This is also the only item that shortens the **critical path**:
   the arena stand-up is 66% of the recording-starts that make `graph_begin` long, on
   the submitting thread, and the arena's own alloc/free is what inflates the rest.
2. **Reserve the six recording containers once** from the previous orchestration's high-water
   mark, so 42 of 291 calls stop reallocating — that is 53% of the faulted time
   (`nodes`, `tensor_sources`, `internal_fanins`, `predicates`, `scalars`).
3. **Give `node.tensors` storage that is not a fresh allocation per node** — the largest
   single source at 31.7%. Its addresses are borrowed by the caller through
   `TaskOutputTensors`, so they must stay valid for the whole recording, which a
   per-recording bump region satisfies and a flat array with a stable base does too.
4. **Reduce the orchestration's own `mmap`/`munmap`/trim traffic**, which is what makes each
   remaining fault cost 14–33 µs instead of 1.7 µs. Items 1 and 3 do this by
   construction.

Any of these must be measured by fault *count* first, and only then by duration, on the
same rank and with `args` carried alongside as a load proxy — see
`.claude/rules/discipline.md` §4 and the entries on this file's dead ends.

**What the list missed.** #1981 removed every allocation items 1-3 name and reported that
the submitting thread's ~1100 faults per bind *did not move*, so they were never the
recording's. The retained mirror is the second independent measurement of the same thing.
Attributing those ~1100 needs `mincore()` on a buffer's pages before the write that would
fault them — the `page-faults` perf event carries no ADDR, which has been checked — not
another guess at which allocation it is.

## Amendment 2026-08-23 — the Definition image is not part of this tail

Recording a Definition image no longer allocates: the images go straight into the
retained upload staging. That removes a per-orchestration allocation the list above
does not name — the `std::vector` each recording built its image into, 8 of them per
dsv4 bind at ~126 KB each, acquired and returned every bind. Interleaved A/B on dsv4
(`base, measure, base, measure`, six rounds each, 8 Definitions / 86 Graph
submissions / 129 host tasks per bind on both arms), both arms taken at
`3069f1aff`'s recording storage, i.e. **before** the per-recorder ownership of #1981:

| per warm bind | base | measure |
| ------------- | ---- | ------- |
| `host_orch` minflt, min / median | 1100 / 1248, 1047 / 1138 | 1062 / 1107, 1082 / 1124 |
| `graph_upload` minflt, min / median / max | 2 / 38 / 47, 34 / 43 / 133 | 0 / 1 / 3, 0 / 1 / 5 |
| `graph_upload` dur min (ms) | 0.225, 0.282 | 0.141, 0.106 |
| control plane, min of sums (ms) | 1.194, 1.365 | 1.549, 1.199 |

**`host_orch`'s fault count did not move** — the two repetitions disagree in sign
on both the min and the median, which is this file's own criterion for "not
resolvable". Only `graph_upload` moved, consistently: the ~40 faults per bind it
took allocating one staging vector per Definition are gone, and so is the copy.

The reason is worth keeping: **a freed 126 KB block is reused from the heap without
re-faulting**, so an allocation this size that is acquired and returned in the same
orchestration was never a fault source, while the 2.17 MB hazard-map arena — far
above glibc's mmap and trim thresholds — is. **Size against those thresholds, not
byte count, decides what shows up in this tail**, which is why the entry's evidence
points at the recording's own storage and not at the largest thing a bind allocates.

The control-plane duration is **not** resolvable from this A/B either (+0.355 ms
then −0.166 ms), which is the expected outcome of a change worth ~0.1 ms on a box
whose load average sat between 40 and 66 throughout.

## Amendment 2026-08-25 — retaining the SM mirror, and why a retained buffer must not be zeroed

The host mirror of the runtime shared memory is now the platform runner's, one
buffer per pipeline slot held across binds until Worker finalization, instead of a
`new uint8_t[]` per bind. At dsv4's `ring_task_window` of 16384 that buffer is
**82.46 MB**, so every bind used to be one `mmap` and one guaranteed `munmap` of
that size — the mapping traffic this entry's off-tree reproduction prices at 10x
on every other fault in the address space.

Interleaved A/B on dsv4 (`base, retained, base, retained`, three rounds over two
ranks, so six binds and four warm ones per arm), `mallinfo2` and `smaps_rollup`
sampled at four points per bind. A **third** arm is included because the first
implementation of the retained buffer was a `std::vector<std::byte>`, and it is
the instructive one:

| per process | base (one block per bind) | retained, `vector::resize` | retained, uninitialized block |
| ----------- | ------------------------- | -------------------------- | ----------------------------- |
| `hblkhd` at `bind_end`, 6 of 6 binds | 435.82 MB (falls back) | 518.28 MB (holds) | 518.28 MB (holds) |
| `host_orch` minflt, the two **cold** binds | 1218 / 1022, 1164 / 1173 | **20194 / 21403, 21213 / 16107** | 1098 / 1130, 1059 / 1018 |
| `host_orch` minflt, four warm binds (median) | 1197.5, 1204.5 | 1256.0, 1273.0 | 1229.5, **950.5** |
| Rss at the last `bind_end` | 45.446 GB, 45.474 GB | 45.589 GB, 45.588 GB | 45.479 GB, 45.522 GB |

**What resolves.** Two things, both counts, both agreeing on every bind of every
run:

- `hblkhd` stops returning to its pre-bind value. That is the mirror being mapped
  and unmapped per bind, and then not.
- **A retained buffer has to be handed over uninitialized.**
  `std::vector::resize` value-initializes, so the first bind of each rank faulted
  in the *whole* capacity — 82460928 / 4096 = 20132 pages, which is exactly the
  ~20k excess above — and left all 82 MB resident for the rest of the run. The
  owning-block version faults only the pages a bind writes, so its cold binds
  match base and its Rss is within the run-to-run spread of it. A container was
  the wrong reach here precisely because the layout is init-on-write: zeroing is
  work whose result nothing reads.

**What does not resolve.** `host_orch`'s own warm-bind fault count, in either
direction. Base sits in [1160, 1256] across its eight warm binds; the retained
block spans [181, 1268], with two binds well below anything base reached and a
median that moves +32 on one repetition and −254 on the other. The mirror is
~6 THP faults of a ~1200-fault bind (see the decomposition above), so this
was never a signal this instrument could carry. Control-plane duration likewise:
minimum-of-sums 1.724 → 1.795 ms then 1.532 → 1.010 ms, opposite signs.

**One reading retracted.** An earlier pass over these logs attributed a
sign-consistent +3-6% warm-minflt rise in the `vector` arm to glibc's dynamic
`mmap`/`trim` thresholds no longer being raised by the freed 82 MB block, on the
strength of `fordblks` at `bind_begin` reading 13.6 MB on base against 1.4 MB
retained. That comparison is invalid: it pairs the *first* bind of each arm, whose
heap state predates the mirror in both, and by the last bind both arms sit at
~14.1 MB. The threshold mechanism is real in glibc, but nothing here measures it,
and the retained-block arm reverses the sign it was invented to explain.

**What is left, and what it is not.** Not the hazard-map arena: #1981 made the
recorder thread own it, so it is stood up once per thread and `reset()` after
that. Every item of "Where a fix would go" above is now implemented, and the
~1100 faults the submitting thread takes per bind survived all of them — #1981
reported them unmoved when the recording's allocations went away, and this arm
says the same about the mirror's. Two things follow. The decomposition in this
entry attributed those faults to allocations that no longer happen, so it no
longer explains the steady state; and node/tensor storage is retained only at
each thread's own high-water mark.

## Amendment 2026-08-25 (later) — the high-water mark is not the steady state

The paragraph above guessed that pre-sizing the recorder's node and tensor storage
"would only move each thread's *first* recording off the growth path, not touch a
warm bind". **That is wrong, and a counter proves it.** #1981's retention is
per-thread, and which body a thread records is decided by the one FIFO all eight
pool workers wait on, so a thread whose array is shorter than the body it is handed
extends it — on whatever bind that happens to be. Counting slot creations per bind
on dsv4, whose eight Definitions differ in size:

| bind | 1 | 2 | 3 | 4 | 5 | 6 |
| ---- | - | - | - | - | - | - |
| node slots created | 365 | 1666 | **1336** | **401** | **219** | **56** |

`standups=0` on binds 3-6 confirms the threads and their storage did survive; the
slots are new all the same. So the growth is recurring, not amortized, and it is
non-deterministic in which bind pays it — the shape this entry describes as a small
median with a maximum two orders of magnitude above it.

**And the obvious fix is a trap.** Reserving each node's own buffer to
`CORE_MAX_TENSOR_ARGS` makes it 32 x 128 B = **exactly one page**, so a 1679-node
body touches 1679 pages to hold ~210 KB of tensors. Measured against `a2ca70cff`,
that took `host_orch`'s minflt from ~1070 to ~2540 and `record_node` from 4.4 ms to
12-14 ms per bind, sign-consistent across two interleaved repetitions. Packing all
of a body's tensors into one bump region — item 3's "flat array with a stable base",
4 MB per thread, allocated once and never grown — is what actually helps: the same
body touches ~53 pages, `record_node`'s warm minimum goes 1702/3423 -> 1239/1563 µs
and the control plane's minimum of per-bind sums 837/1303 -> 719/943 µs.

Two rules fall out, and both cost a wrong PR to learn:

- **Retention is per thread, so a per-thread high-water mark is not the workload's.**
  Where work is handed out by a shared queue, size the storage by the *contract*, not
  by what this thread has seen.
- **A reservation that is not packed can cost more than no reservation.** Per-object
  buffers rounded up to a cap turn into a page each; the fault count follows the
  number of *pages touched*, not the bytes reserved.

## Amendment 2026-08-25 (last) — the count is not the lever; the price is

Two arms measured after every item of "Where a fix would go" had shipped point in
opposite directions, and together they refute the *count × price* framing above.

| arm | faults | control-plane duration |
| --- | ------ | ---------------------- |
| glibc keeps freed memory (`MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_` at 1 GiB, `MALLOC_TOP_PAD_` at 256 MiB; glibc 2.36) | 1019 → 140, **−86%** | **unchanged** |
| #2015, one flat tensor region per recorder thread | median 1078 → 1010, **−6%** | median 1.157/1.454 → 0.824 ms, **−29…−43%** |

An arm that removes 86% of the faults buys no time; an arm that removes 6% buys a
third of the phase. **The fault count is not a lever. Only the price is** — which is
the half of the Answer that survives.

The tunable arm is not new evidence; it is the one already recorded under "Removing
the return-to-kernel behaviour removes the faults". What was new was reading its flat
duration as *"the tunables are not a fix"* instead of *"the count does not buy time"*.
The datum sat here through three rounds of work with the wrong conclusion attached.

### The in-tree `mmap_lock` writer is `mprotect`, and it was never traced

glibc reserves a non-main arena with `mmap(PROT_NONE)` and opens it up with
**`mprotect`** (`grow_heap`), so a recorder thread whose arrays grow issues one — and
`mprotect` takes `mmap_lock` for write, excluding every faulting thread in the address
space exactly as `munmap` does.

Every strace in this investigation traced `madvise`, `mmap`, `munmap` and `brk`.
**None traced `mprotect`.** One `strace -ff -e trace=mprotect,madvise,brk` over three
rounds, in the non-main-arena band `0xfff0…0xfff4`:

| syscall, in that band | over 6 binds |
| --------------------- | ------------ |
| `mprotect(PROT_READ\|PROT_WRITE)` | **157 calls — 26 per bind**, 85.2 MB |
| `madvise(MADV_DONTNEED)` | **0** (3300 calls / 26 GB elsewhere, none of it arena) |

Those arenas only grow; nothing shrinks them back. That is why #2015 — which sizes
every recorder array from its contract at thread stand-up and never grows one again —
removes the writer, and why it is the first change here to buy time.

### The residual faults are a warm-up cost, and they end

Every figure above this line calls its per-bind counts steady-state. **They are not.**
Measured at `f40cacf30`, with #1988, #2013, #2015, #2019 and #2022 all in, `host_orch`'s
`minflt` per bind in arrival order, on three runs at different round counts:

| run | cold binds | then, bind by bind |
| --- | ---------- | ------------------ |
| `--rounds 2` (4 binds) | 992, 949 | 165, 130 |
| `--rounds 5` (10 binds) | 989, 983 | 114, 173, 54, **3, 13, 13, 11, 8** |
| `--rounds 8` (16 binds) | 931, 837 | 112, 216, 164, 10, 2, 1, 1, 8, 57, 10, 8, 11, **0, 0** |

`args` follows the same curve (699310 on the cold bind, then 0–2) and so does
`graph_upload` (246 cold, 0 after). So the tail **decays over roughly six binds and then
reaches zero**; the ~1100 faults this entry chased for three rounds are a warm-up cost,
not a per-bind one.

**Which is why the per-bind counts here are wrong, all of them.** They came from runs of
three to six rounds, divided by the bind count — so each one averages two cold binds and
three or four still-decaying ones into a figure presented as steady state. The correction
is not a smaller number; it is that the quantity being divided was never per-bind. See
[`docs/dfx/hbg-bind-phases.md`](../dfx/hbg-bind-phases.md)'s trap on warm not meaning
steady-state, which this measurement is what added.

What still holds from the arms above: the tunable arm and #2015 do point in opposite
directions, so within the warm-up window the count is not a lever and the price is. What
does not hold is that anything "survived" the four changes — in the steady state there is
nothing left to survive, and **the mechanism that went undetermined for three rounds
turned out not to need determining.** Userspace tools were exhausted on it — `mincore`
reports presence, not writability; pagemap's bit 56 is `mapcount == 1`, also not
writability; no interface exposes a PTE's write bit — and none of that mattered.

Where the control plane stands at `f40cacf30`, `--rounds 5`, the 8 warm binds: total
0.529 ms min / 0.570 median, of which `host_orch` 0.341/0.364, `graph_upload`
0.129/0.133, `arena_h2d` 0.056/0.058. Two of those 8 are still decaying (0.443 and 0.427
against 0.341–0.397 for the rest), so even this is not a steady-state figure — it is the
best available one, and it needs `--rounds 12` to become clean.

### What the per-site attribution produced, and its shelf life

`perf record -e page-faults -c 1 -k mono --call-graph fp`, filtered to the worker
processes and to the `host_orch` windows, names every site. Per warm bind at
`87deeab42`, before #2019:

| site | per bind |
| ---- | -------- |
| `graph_end` → `memset(image, 0, total_bytes)` — **deleted by #2019** | 245 |
| `_M_fill_assign` — the per-node tensor buffers, **replaced by #2015** | 199 |
| `graph_record_submit_node` | 133 |
| the two orchestration `.so`s | 79 |
| flat arrays (`_M_default_append`) | 58 |
| `ChipTensorMap::reset`, `operator new`, others | ~140 |
| the SM mirror (`prepare_task` + `submit_task_common`, ~6 THP faults) | 6 |

Two of the top three are gone, which is the point: **a per-site fault table dates
quickly, and none of the entries it drove were worth what they measured.** Keep the
method, not the numbers.

Also refuted while attributing them, each with a direct measurement: `MADV_DONTNEED`
(0 calls in that band), fork-COW (every `clone` carries `CLONE_VM`), KSM (`run=0`),
AutoNUMA hinting (one rank's `total_numa_faults` never moves while its minflt stays
~1000), THP (`PR_SET_THP_DISABLE` leaves the memset's faults at 0.97/page, unchanged),
writing bytes never written before (per-slot high-water counter: 0 such assigns in the
steady state), and the shape of the allocation (moving the Definition staging from a
heap vector to its own anonymous mapping: 1960 → 1969 faults). The measurement itself
was controlled with an empty `getrusage` window, which never counts a fault.

Three traps in the tooling, each of which produced a wrong conclusion first:

- **`perf record` without `-k mono`** stamps samples with a clock that is not the
  `CLOCK_MONOTONIC` a phase window is expressed in. The offset is small enough that
  "is `start_ns` between the first and last sample" still passes, and a
  millisecond-wide window still lands on the wrong stretch: the first attempt put
  99.9% of `host_orch`'s faults in `libtorch_cpu.so`, on a thread belonging to the
  *parent* process, whose faults never enter a worker's `getrusage`.
- **`--no-buildid-cache`** leaves the data depending on the `.so` at its recorded
  path, so rebuilding it turns every symbol in an already-recorded run into `[unknown]`.
- **Neither `mincore` nor pagemap reports write permission**, so neither can establish
  that a page "was handed back". Both were used to argue exactly that.

### What is closed

"Where a fix would go" is fully implemented: items 1–3 by #1981, item 4 by #1988 plus
 #2013, and the flat-region form of item 3 by #2015. No untried item remains, and the
one that bought time did so by removing a **syscall**, not by removing allocations.

## References

- Probe commits on the local `probe-on-main` branch (measurement only, unpushed):
  per-segment `record_node` stamps, then per-node fault count and CPU time.
- Off-tree: `.docs/bench_recording_alloc.cpp` (pattern + churn), `.docs/no_thp.c`
  (`PR_SET_THP_DISABLE` preload), `.docs/rnprobe_classify.py`.
- Archived runs under `outputs/commits/24_record_node_segments` … `30_clean_no_trim`.
