# Two Kinds of Dependency: WAIT/RETAIN Split + Transitive Reduction

Design and status doc for the `tensormap_and_ringbuffer` dependency-flag
work, reimplementing issue #1375 on current `main` (a2a3 first, a5 after
human confirmation). Supersedes the `feat/two-kinds-of-dep` prototype
(commit `69ea5c66`), which must not be cherry-picked: it predates the
early-dispatch wiring flow and carries a payload-init ordering bug.

## 1. Problem

One physical dependency edge today carries two independent semantics:

- **WAIT / readiness**: the consumer cannot become ready until the
  producer completes (`fanin_count` / `fanin_refcount`, producer fanout
  notification).
- **RETAIN / lifetime**: the producer's slot and packed output buffer
  must stay alive until the consumer releases (`fanout_count` /
  `fanout_refcount`, reclaimed at `CONSUMED`).

Coupling them blocks safe transitive reduction: in `A -> B -> C` plus
direct `A -> C`, the direct WAIT is redundant (ordering is transitive)
but RETAIN is not (C may read A's packed output; `A -> B -> C` does not
keep A's output allocated). The target end state per #1375:

- creator (`owner_task_id`) edge: `WAIT | RETAIN`
- tensormap modifier edge: `WAIT`
- explicit edge: caller-selectable, default `WAIT | RETAIN`
- duplicate discoveries of the same `(producer, consumer)` pair: flags
  OR-merged, never first-wins
- reduction over the WAIT subgraph only: `WAIT|RETAIN -> RETAIN-only`
  when another WAIT path connects the pair; edge removed only when no
  flags remain

## 2. Why reimplement (prototype retrospective)

Kept ideas (validated on the prototype: no perf regression, race fixed):

- classify creator vs tensormap deps at discovery
- compact per-edge kind storage
- WAIT-only edges drop their lifetime pin early (after wiring)

Discarded:

- 1-bit `DepKind` (RESOURCE/EXECUTION) — cannot express RETAIN-only, so
  the target reduction is unrepresentable
- `fanin_inline_dep_kind_mask` side field — written before
  `payload.init()`, zeroed by it (known bug); a mask field in the
  payload is avoidable altogether (see storage below)
- the wiring base — main has since landed early-dispatch
  (#1304/#1319/#1326/#1328/#1329/#1331/#1336/#1340) and predicated
  dispatch (#1314), which re-derived the wiring/fast-path invariants

## 3. Design on current main

### 3.1 Storage: tag the producer pointer, 2 bits

Edges are bare `PTO2TaskSlotState*` values in two stores:

- inline fanin array in `PTO2TaskPayload` (64 entries)
- `PTO2FaninSpillEntry` in the per-ring fanin pool (one pointer each)

`PTO2TaskSlotState` is `alignas(64)`, so pointer bits [1:0] are zero.
The tagged value is wrapped in a dedicated trivially-copyable type
`PTO2TaggedSlotPtr` (`make()` tags, `slot()` untags, `flags()` reads),
so dereferencing a still-tagged pointer is a compile error rather than
silent corruption. Both stores keep their exact current size/layout
(all `static_assert`s unchanged); no new payload field means nothing
for `payload.init()` to clobber — the prototype's bug class is
impossible by construction. The fanout `PTO2DepListEntry::slot_state`
(consumer pointer) uses the same encoding so the completion walk can
recognize WAIT-only edges.

Alignment is guarded at three levels: the `alignas(64)` type contract,
a `static_assert(alignof(PTO2TaskSlotState) >= 4)`, and a cold-path
`always_assert` on the suballocated `slot_states` base in both
`init_data_from_layout` and `reset_for_reuse` (the SM layout walks
64-byte-aligned segments today, but only the assert keeps that loud).

All fanin reads go through `for_each_fanin_storage`
(`pto_ring_buffer.h`), which untags and hands `(slot_state, flags)` to
the callback.

```cpp
enum PTO2DepFlags : uint8_t {  // bitmask, plain enum per codestyle
    PTO2_DEP_NONE   = 0,
    PTO2_DEP_WAIT   = 1 << 0,
    PTO2_DEP_RETAIN = 1 << 1,
};
```

### 3.2 Discovery: provenance -> flags, OR-merge dedup

`compute_task_fanin`'s emit callback gains a flags argument
(`emit(producer_task_id, flags)`); the dep_gen replay oracle lambda
takes and ignores it, the annot mirror already classifies
creator/tensormap/explicit and stays bit-equivalent.

| source                             | flags            |
| ---------------------------------- | ---------------- |
| Step A creator (`owner_task_id`)   | `WAIT \| RETAIN` |
| Step B tensormap modifier          | `WAIT`           |
| explicit `set_dependencies`        | `WAIT \| RETAIN` |
| explicit `set_execution_deps` (new)| `WAIT`           |

Dedup in `append_fanin_or_fail` currently collapses duplicates via an
epoch-stamped seen array, first-wins. First-wins is wrong under flags:
a tensormap hit (WAIT) followed by a creator hit on the same producer
must still gain RETAIN. The seen array gains a parallel index array
(`fanin_seen_index[ring][slot]`, valid iff epoch matches — no reset
cost) so a dedup hit can OR the new flags into the existing entry.
Claim accounting is unchanged: one `fanout_count++` per unique pair,
under the producer's `fanout_lock`, atomic with the generation/CONSUMED
check.

### 3.3 Pin protocol (lifetime accounting per edge kind)

The submit-time `fanout_count++` claim pins the producer slot's
generation until the edge's final disposition. What happens next
depends on the flags:

| edge flags     | fanout entry wired | pin released at              |
| -------------- | ------------------ | ---------------------------- |
| `WAIT\|RETAIN` | yes (if live)      | consumer `on_task_release`   |
| `WAIT` only    | yes (if live)      | producer completion fanout walk |
| `RETAIN` only  | no                 | consumer `on_task_release`   |

Release points that must each handle WAIT-only pins:

1. Producer completion fanout walk (live-fanin path): the fanout entry
   carries the edge flags (same pointer-tag encoding as fanin storage);
   the walk releases WAIT-only pins on the scheduler thread and runs one
   batched CONSUMED check per producer. This keeps per-edge atomics off
   the orchestrator submit path.
2. Wiring, for WAIT-only edges on already-completed producers: no
   fanout node exists for them, so the pin is released in
   `wire_fanin_task` directly.
3. all-producers-completed fast path: no wiring happens, so WAIT-only
   pins are released inline there.
4. zero-WAIT fast path: no WAIT edges exist; RETAIN pins persist and
   are released by `on_task_release` as usual.

`on_task_release` releases only RETAIN-flagged entries — WAIT-only
edges are already gone from the refcount by then. RETAIN-only edges
never create a fanout node, so they cost no completion-time traversal.

Invariants preserved: `CONSUMED iff fanout_refcount == fanout_count`;
the scope bit stays orthogonal; the heap/task deadlock detector's
`rc == (fc & ~SCOPE_BIT) && COMPLETED` condition keeps its meaning
("only scope_end can release the head") — WAIT-only early release just
reaches that state sooner, which is the same state.

### 3.4 Readiness and early-dispatch accounting

- `fanin_count = wait_edge_count + 1` (was: total edge count + 1).
- Fast-path selection keys on `wait_edge_count`, not total edges:
  `wait == 0` -> immediate-ready path; `all WAIT producers completed`
  -> seed path; else wire live.
- `fanin_actual_count` keeps meaning *storage length* (all edges,
  incl. RETAIN-only) — the spill/inline walks and profiling depend on
  it.
- Early-dispatch candidacy compares `dispatch_fanin` against the WAIT
  producer count. Today that is `fanin_actual_count`; with RETAIN-only
  edges present it must be `fanin_count - 1`. RETAIN-only producers
  never propagate (no fanout entry) and are excluded from the
  wiring-time seed and the `all_claimed_fanin_allow_early_resolve`
  gate, so the counter and its target stay consistent.
- `all_claimed_fanin_completed` becomes "all WAIT producers completed"
  (RETAIN-only producers are irrelevant to readiness).

### 3.5 Transitive reduction (deferred — not in this branch)

This branch intentionally ships only the attribute split; no WAIT
deduplication is performed. The follow-up reduction work (tracked
separately) can build on these flags: run reduction over the WAIT
subgraph only, downgrade `WAIT|RETAIN -> RETAIN-only` when another
WAIT path connects the pair, and keep the edge as RETAIN-only while
lifetime requires it.

A depth-1 online design was prototyped and measured on this branch
(commit `367fd6e9`, removed in the flags-only cleanup): at submit
time, for each `WAIT|RETAIN` candidate A->C, scan each WAIT sibling
B's stored fanin for A (same ring, A.id < B.id, pointer-compare
only). It fired 1024 edges/op on batch_paged_attention but cost
~60-90us of orchestrator time there (see section 7), with no
measurable sched-side win on that graph — so it was pulled from this
branch. The soundness arguments (claim-pinned payloads, id-order
anti-aliasing) remain valid for the future implementation.

When RETAIN-only edges do arrive, revisit the retain-first fanin
ordering (3.8): wire's walk then also has a skippable kind, and the
two readers' subsets overlap (wire wants WAIT|RETAIN + WAIT-only,
release wants WAIT|RETAIN + RETAIN-only), so a single linear order no
longer serves both optimally — consider a three-segment layout or two
ranges. This constraint is called out in comments at the ordering
site and the release walk.

### 3.6 Explicit deps: WAIT|RETAIN only (user-side constraint)

User-supplied (manual) dependencies can express exactly one edge kind:
`WAIT|RETAIN`. Both entry points — the primitive
`L0TaskArgs::set_dependencies` and the convenience-layer `add_dep()` —
feed a single STEP 1 list that `submit_task` emits with
`PTO2_DEP_WAIT_RETAIN` (pto_orchestrator.cpp). There is deliberately no
kind/flags parameter anywhere in the public API.

Rationale for not exposing flags (or a kind enum) to users:

- The flags value domain and the set of caller-sound states disagree.
  `RETAIN`-only is only meaningful as a transitive-reduction artifact —
  its soundness premise ("another WAIT path exists") is global graph
  knowledge the caller does not have — and `NONE` is meaningless. A
  flags-typed API would carry 4 values of which 2 are traps.
- WAIT-only declares "the producer's output may die before I finish";
  getting it wrong is use-after-reclaim. The short name stays the safe
  tier, the long name the expert tier.
- Caller-local flag arithmetic ("someone else retains this, I can drop
  RETAIN") is unsound in general: retention correctness depends on
  global graph state (what reduction will downgrade, who else holds the
  tensor).

Planned extension (deferred, rides with the reduction work): a second
named entry point — `set_execution_dependencies(deps, count)` at the
primitive layer, `add_execution_dep(...)` at the convenience layer —
backed by a second explicit list, emitted with `PTO2_DEP_WAIT` in
STEP 1. Two named tiers, never a flags parameter; RETAIN-only remains
unreachable from user code. Intended for codegen in manual scopes that
knows an explicit dep protects ordering only. Requires extending the
dep_gen record format to carry the second list.

Duplicate explicit deps and explicit-vs-auto duplicates are already
handled: every edge funnels through `append_fanin_or_fail`, whose
(ring, slot)-keyed seen array dedups and OR-merges flags
(never first-wins), so an explicit `WAIT|RETAIN` merged with a
tensormap `WAIT` stays `WAIT|RETAIN` regardless of discovery order.

### 3.7 DFX / replay

dep_gen replay already records edge provenance
(`explicit|creator|tensormap` in deps.json), from which construction
flags derive deterministically; the emit-signature change keeps the
oracle/annot differential intact. Recording the runtime's reduction
decisions (which edges were downgraded to RETAIN-only) is a follow-up;
the runtime currently records no reduction (deferred, see 3.5).

### 3.8 Retain-first fanin ordering + select-driven walks

Fanin storage (payload inline array, then the spill range) is written
with RETAIN-bearing edges first, WAIT-only edges after, and
`PTO2TaskPayload::fanin_retain_count` records the prefix length. The
field occupies the existing slack between the early-dispatch block and
the predicate (offset 564, after `early_sync_drain_state`), so payload
layout and all offset static asserts are otherwise unchanged. Tasks whose fanin spills keep discovery order
and set `retain_count == actual_count`, falling back to a full walk.

`for_each_fanin_storage` internalizes the traversal strategy via a
`PTO2DepFlags select` parameter and two callback forms picked at
compile time:

- `fn(slot, flags)` with `select == PTO2_DEP_NONE`: unfiltered,
  branch-free, every edge reported with its flags — for callers that
  dispatch on the kind per edge (`wire_fanin_task` handles both kinds
  in one pass: RETAIN-only skip, fanout wiring, WAIT-only pin
  release).
- `fn(slot)` with a flag select: exact-kind match, the callback fires
  only for matching edges and never sees flags. Sorted storage
  resolves `WAIT_RETAIN` to the `[0, retain_count)` prefix
  (`on_task_release`) and `WAIT` to the `[retain_count, actual_count)`
  suffix (all-completed fast-path release) with no per-entry check.
  `NONE` also serves has-WAIT filters today (`all_claimed_*`), since
  every constructed edge carries WAIT. Unsorted storage (the
  submit-time builder) and the spilled fallback use a per-entry
  exact-match test inside the walker, never in callbacks.

The ordering is an optimization hint only — the flag checks inside the
walker remain the authority. With only the two current kinds
(WAIT|RETAIN, WAIT-only) the prefix is exactly the W|R set. When
RETAIN-only edges arrive (3.5), the prefix still releases them
correctly, but the wire walk gains a skippable kind and the two
readers' subsets overlap, so the ordering should be redesigned
(three-segment layout or two ranges).

Scope decision: only inline fanins (<= 64 edges) are sorted. Spilled
fanins deliberately keep discovery order and pay a full walk with
per-entry flag checks — sorting the spill range would cost a
pool-range read-modify-write in DRAM per big-fanin submit (or a much
larger stack builder holding every edge until finalize), and fanin
beyond the inline cap is rare (the dep-degree warning fires at 16).
If a real graph makes big fanins common, revisit with the
buffered-builder variant (hold all edges in an enlarged stack
builder, write the final order to inline + pool once at finalize,
which also collapses the per-entry `ensure_space` calls into one).

## 4. Non-goals / later phases

- any WAIT deduplication / transitive reduction (deferred per scope
  decision; the measured depth-1 prototype is documented in 3.5,
  including the scope-close bitset alternative)
- a5 port (after a2a3 passes and human confirms; file set mirrors)
- deps.json schema change for runtime reduction results
- codegen adoption of `set_execution_dependencies`

## 5. Test plan

1. sim (`--platform a2a3sim`): `dummy_task`, `dep_gen`,
   `dep_gen_chain`, `spmd_basic`, `spmd_multiblock_mix`,
   `mixed_example`, `vector_example` — behavior preserved.
2. dep_gen ST: oracle/annot differential still passes after the emit
   signature change (flags round-trip through discovery).
3. onboard a3: `batch_paged_attention` (the prototype's race repro),
   `alternating`, then the tmr benchmark suite vs upstream/main —
   same pass set, device time within noise.
4. reduction validation: DFX-gated counter + golden-output correctness
   on reduction-heavy graphs (paged attention family).

## 6. Status

- [x] design doc
- [x] flags storage + provenance + pin protocol + release filtering
      (pointer-tag 2-bit flags in inline + spill fanin storage and in
      fanout dep-pool entries; OR-merge dedup via epoch|index seen
      array; WAIT-only pin released in the producer completion fanout
      walk / at wiring for pre-completed producers / in the
      all-completed fast path; `on_task_release` releases RETAIN only;
      early-dispatch candidacy target is `fanin_count - 1`)
- [x] flags-only scope: depth-1 reduction prototyped, measured, and
      removed again (section 3.5); branch ships attribute split only
- [x] replay/annot adaptation (oracle emit signature; annot unchanged)
- [x] verification: sim a2a3 tmr suite 47/47; onboard via task-submit
      9/9 core cases; benchmark 8/8 cases pass
- [ ] `set_execution_dependencies` explicit WAIT-only API — deferred:
      needs a dep_gen record format change (shared header) so replay
      stays complete; discuss before landing
- [ ] a5 port (after human confirmation)

## 7. Perf notes (batch_paged_attention)

Benchmark vs merge-base (same device, ABAB-interleaved, `-n 20`),
flags-only build as shipped on this branch: 7/8 benchmark cases within
noise; `batch_paged_attention (Case1)` Eff +2.1% / Orch +2.9%
(+~75us orch on ~2.9ms) — the residual cost of the flags machinery
itself: per-edge tag/seen-index accounting in `append_fanin_or_fail`,
flag branches in the wire/fast paths, and WAIT-only pin releases.
`paged_attention_unroll_manual_scope (Case2)` measured +2.7% here but
drifts ±3% across sessions; treated as noise.

Historical decomposition (reduction-inclusive builds, commit
`367fd6e9` era, for the follow-up reduction work): the depth-1
reduction added ~60-90us orch on this graph — cache fill ~28us
(`reduce_transitive_waits` stack caches), sibling pred-set walks
~20us (1024/op), scan loops ~90 -> ~60us (O(n·candidates); stack
flag caches helped 139 -> 111us), while firing 1024 edge downgrades
per op with no measurable sched-side win. A separate ~+100us bug
(per-submit zero-init of the builder's 64-entry tagged array) was
found and fixed via trivial default construction of
`PTO2TaggedSlotPtr`. Investigation artifacts:
`tmp/perf_investigation_20260719/` (device-log profiling blocks,
back-to-back benchmark captures).
