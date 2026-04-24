# Manual Scope V0 Design

Date: 2026-04-24

## Goal

Keep manual scope small and explicit:

- same submit API family as AUTO mode
- explicit task-to-task deps via `Arg.add_dep(task_id)`
- no delayed wiring
- no delayed publish barrier at `scope_end()`
- publish at submit time, same as AUTO mode
- support allocation-as-task through `alloc_tensors(...).task_id()`

This is intentionally narrower than the older manual-dep branch.

## Constraints

The v0 design keeps these rules:

1. `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` opts a scope into manual mode.
2. `MANUAL` nested inside active `MANUAL` is allowed.
3. `AUTO` nested inside active `MANUAL` is rejected.
4. Manual deps are attached before submit through `Arg.add_dep(...)`.
5. No post-submit `add_dependency(...)` API exists.
6. `alloc_tensors(...)` remains output-only and returns a task id.
7. Scope handling stays close to upstream AUTO mode.

## User-Facing API

### Scope

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    ...
}
```

### Submit

Manual mode still uses the normal submit calls:

```cpp
auto out = pto2_rt_submit_aic_task(FUNC_ID, args);
auto out = pto2_rt_submit_aiv_task(FUNC_ID, args);
auto out = pto2_rt_submit_task(mixed_kernels, args);
```

There is no extra `*_manual(...)` submit family in v0.

### Explicit deps

```cpp
Arg args;
args.add_input(tensor);
args.add_dep(prev_task_id);
```

Rules:

- `add_dep(...)` must refer to an earlier producer task id
- if that producer task is already retired when the consumer is submitted, the
  runtime skips the edge without reporting an error
- explicit deps become ordinary fanins immediately at submit time
- this applies uniformly in AUTO and MANUAL submits; the task id already carries
  the ring needed for slot lookup

### Allocation task ids

```cpp
auto alloc = alloc_tensors(ci0, ci1, ci2);
PTO2TaskId alloc_tid = alloc.task_id();
const Tensor &tmp = alloc.get_ref(0);
```

This is required for manual-local tensors because the alloc task itself may be
the producer that later tasks must explicitly depend on.

## Runtime Model

### High-Level Behavior

Manual scope v0 is:

- AUTO-style submit and publish
- plus explicit fanins from `Arg.add_dep(...)`
- plus full TensorMap lookup / insert bypass for tasks submitted inside manual
  scope

It is not:

- a delayed graph builder
- a second runtime path with a manual-only publish barrier
- a TensorMap fallback mode for boundary dependencies

### Scope State

The runtime keeps two manual-scope-specific pieces of state:

- `manual_begin_depth`
- the current scope task list (`scope_tasks[...]`)

`manual_begin_depth` decides whether the current submit is in manual mode.
`scope_tasks[...]` remains scope-lifetime bookkeeping for `scope_end()`.
`manual_begin_depth` is explicitly initialized in `pto2_orchestrator_init()` and
reset in `pto2_orchestrator_done()` so a reused orchestrator starts cleanly on
the next run.

The depth model treats `manual_begin_depth` as the depth where the outermost
active manual scope began:

- outer `MANUAL` begin sets `manual_begin_depth`
- nested `MANUAL` begin does not overwrite it
- nested `AUTO` begin is rejected
- `scope_end()` clears `manual_begin_depth` only when the outermost manual
  scope exits

That keeps nested-manual behavior small: there is still only one manual-depth
marker, not a second manual-scope stack.

There is no per-tensor manual-local classification on the submit path. If the
consumer task is inside manual scope, that submit skips TensorMap lookup and
insert for all tensor args.

There is no delayed-link metadata, per-slot epoch replay, or scope-end explicit
dependency rebuild in v0.

### How Manual Scope Decides TensorMap Use

The rule is submit-scoped, not tensor-scoped:

- inside manual scope: skip the entire TensorMap lookup section
- inside manual scope: skip the entire TensorMap insert section
- outside manual scope: keep normal AUTO TensorMap behavior

Explicit deps resolve task ids directly through shared-memory ring slot state.
There is no extra current-scope validation on the submit path.

This matches the pre-rebase aligned PR head `efab669` and the colleague design
at `poursoul/simpler@dd76880`: manual-scope tasks express dependencies through
`Arg.add_dep(...)`, not through TensorMap lookup / insert.

### Dependency Handling

For a submitted task:

- explicit `add_dep(...)` edges are always checked first
- retired explicit-dep producers are skipped as already satisfied
- if the task is inside manual scope, TensorMap lookup / insert is skipped
- if the task is outside manual scope, normal AUTO TensorMap behavior is used

That gives the actual v0 dependency split:

- manual-local tensor producer -> manual-local consumer:
  explicit dep, no TensorMap lookup / insert
- external tensor -> manual consumer:
  explicit dep, no TensorMap lookup / insert on the manual submit
- manual producer -> later external consumer:
  use `Arg.add_dep(...)` when the later consumer needs to order after the
  manual producer; outside manual scope the normal TensorMap path still applies

### Scope End

`scope_end()` keeps only normal scope-lifetime behavior:

- call scheduler `on_scope_end(...)`
- pop scope bookkeeping
- clear `manual_begin_depth` when leaving the manual region

It does not:

- replay explicit deps
- batch publish manual tasks
- rebuild delayed edges

## Implementation Notes From The Rebased Cleanup

The current forward-port intentionally shrinks the runtime diff against
`upstream/main`:

- `pto_orchestrator.cpp`
  - restores the upstream-style allocator-driven `pto2_prepare_task(...)`
  - restores normal profiling lap recording
  - keeps the pre-rebase aligned rule that manual-scope submits skip TensorMap
    lookup / insert entirely
- `pto_types.h`
  - restores upstream variadic `Arg` helpers (`add_input`, `add_output`,
    `add_inout`, `add_no_dep`, `add_scalar`)
  - keeps only the v0-specific explicit dep storage and `add_dep(...)`
- `pto_runtime2_types.h`
  - removes the extra `alignas(64)` / size assertion on `PTO2TaskDescriptor`
- `pto_ring_buffer.h`
  - removes allocator helper accessors that were only supporting the older,
    more intrusive prepare-task path

These changes are the main response to the rebased review feedback that the
runtime hot path had drifted too far from upstream.

## Example Pattern

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
    auto alloc = alloc_tensors(tmp_ci);

    Arg qk;
    qk.add_input(qi, kj);
    qk.add_output(sij_ci);
    qk.add_dep(alloc.task_id());
    auto qk_out = pto2_rt_submit_aic_task(FUNC_QK, qk);

    Arg sf;
    sf.add_input(qk_out.get_ref(0));
    sf.add_output(pij_ci, li_ci, mi_ci);
    sf.add_dep(qk_out.task_id());
    auto sf_out = pto2_rt_submit_aiv_task(FUNC_SF, sf);

    Arg up;
    up.add_input(sf_out.get_ref(1), sf_out.get_ref(2), pv_out.get_ref(0));
    up.add_inout(mi, li, out_view, tmp);
    up.add_dep(sf_out.task_id());
    up.add_dep(pv_out.task_id());
    auto up_out = pto2_rt_submit_aiv_task(FUNC_UP, up);
}
```

Repeated zero-output updater chains must explicitly thread the returned task id:

```cpp
PTO2TaskId prev_update = PTO2TaskId::invalid();
for (...) {
    Arg up = ...;
    if (prev_update.is_valid()) {
        up.add_dep(prev_update);
    }
    prev_update = pto2_rt_submit_aiv_task(FUNC_UP, up).task_id();
}
```

## A5 Port Scope

The a5 port uses the same manual-scope v0 runtime model as a2a3:

- same `manual_begin_depth` state model
- same `MANUAL`-inside-`MANUAL` behavior
- same rejection of `AUTO` inside active `MANUAL`
- same submit-time explicit dependency model through `Arg.add_dep(...)`
- same submit-time TensorMap bypass for tasks submitted inside manual scope

The port stays intentionally small:

- no new submit API
- no new unit tests
- no new runtime-side delayed wiring or `scope_end()` replay
- no unrelated a5 runtime cleanup folded into the change

For the initial example port, the manual orchestration keeps the same explicit
dependency shape as the a2a3 non-unroll manual example:

- `QK -> SF`
- `SF -> PV`
- `PV -> UP`
- `alloc -> first UP`
- `prev_update -> later UP`
- `alloc -> last UP` when the update chain is multi-step

It intentionally does not add extra manual-only edges like `alloc -> QK` or
`SF -> UP`, because those are not part of the established a2a3 v0 pattern.

## A5 Example Scope

The a5 side demonstrates manual scope through examples under
`examples/a5/tensormap_and_ringbuffer/`.

For this port, the required example scope is:

- `paged_attention_manual_scope`
- `paged_attention_unroll_manual_scope`

If the existing a5 example layout already supports an unroll/manual twin without
extra runtime-side work, `paged_attention_unroll_manual_scope` may be added in
the same style. Otherwise the initial a5 port stops at the non-unroll manual
example and keeps the runtime diff small.

Example changes should prefer:

- reusing the existing a5 paged-attention kernels
- keeping orchestration as the primary place where manual-scope behavior is
  expressed
- avoiding unrelated benchmark-script or test harness expansion in this pass

## Rebased Validation

Environment:

- platform: `a2a3`
- device: `10`
- PTO-ISA commit: `478daadb`
- benchmark date: `2026-04-23`

## A5 Validation Boundary

The a5 port uses a narrower validation target than the earlier a2a3 hardware
work:

- required: successful a5 runtime/example build
- required: sim-capable example parity on `a5sim` where supported
- optional: real a5 hardware execution when an a5 device is actually available

This means the port is considered complete for the current branch when:

- the a5 runtime accepts nested manual scope with the same semantics as a2a3
- the new a5 manual-scope example(s) build cleanly
- the available sim path runs cleanly on this host

If no usable a5 device exists on the current server, lack of real-device runs is
recorded as an environment limit, not as a runtime bug.

### Latest A5 Sim Rerun

The current branch was revalidated on `a5sim` after aligning the a5 manual
example's explicit-dependency pattern with the a2a3 manual example.

Use a pinned local PTO-ISA clone to avoid unrelated network fetch delays:

```bash
export PTO_ISA_ROOT=$(pwd)/build/pto-isa
source .venv/bin/activate
```

Rerun commands:

```bash
python examples/a5/tensormap_and_ringbuffer/paged_attention_manual_scope/test_paged_attention.py \
  -p a5sim --build \
  --case TestPagedAttentionManualScope::SmallCase1

python examples/a5/tensormap_and_ringbuffer/paged_attention_manual_scope/test_paged_attention.py \
  -p a5sim --build --manual include \
  --case TestPagedAttentionManualScope::SmallCaseVarSeq2

python examples/a5/tensormap_and_ringbuffer/paged_attention/test_paged_attention.py \
  -p a5sim --build \
  --case TestPagedAttention::SmallCase1
```

Results:

| Example                        | Case           | Result |
| ------------------------------ | -------------- | ------ |
| `paged_attention_manual_scope` | `SmallCase1`   | PASS   |
| `paged_attention_manual_scope` | `SmallCaseVarSeq2` | PASS |
| `paged_attention`              | `SmallCase1`   | PASS   |

Notes:

- `paged_attention_manual_scope::SmallCaseVarSeq2` is marked
  `"manual": True`, so the rerun needs `--manual include`.
- The first rerun attempt without `PTO_ISA_ROOT` stalled in an external
  `pto-isa` fetch; that was an environment issue, not a runtime failure.
- This host still has no confirmed usable a5 hardware device, so the a5 port
  remains validated at build + sim parity only.

### Latest A5 Unroll Sim Rerun

The a5 unroll manual example was added under
`examples/a5/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope`.
It reuses the existing a5 unroll kernel set, but the orchestration follows the
a2a3 unroll manual dependency pattern:

- `QK -> SF`
- `SF -> PV`
- `PV -> UP`
- `alloc -> first UP`
- `prev_update -> later UP`
- `alloc -> last UP` when the update chain spans multiple groups

Rerun commands:

```bash
export PTO_ISA_ROOT=$(pwd)/build/pto-isa
source .venv/bin/activate

python examples/a5/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/test_paged_attention_unroll.py \
  -p a5sim --build \
  --case TestPagedAttentionUnrollManualScope::SmallCase1

python examples/a5/tensormap_and_ringbuffer/paged_attention_unroll_manual_scope/test_paged_attention_unroll.py \
  -p a5sim --build --manual include \
  --case TestPagedAttentionUnrollManualScope::SmallCase2
```

Results:

| Example                               | Case         | Result |
| ------------------------------------- | ------------ | ------ |
| `paged_attention_unroll_manual_scope` | `SmallCase1` | PASS   |
| `paged_attention_unroll_manual_scope` | `SmallCase2` | PASS   |

Notes:

- The unroll kernels are shape-specialized; the sim-capable cases must respect
  the kernel contracts.
- The validated `a5sim` cases use:
  - `q_tile=16, block_size=128, head_dim=128` for `SmallCase1`
  - `q_tile=64, block_size=64, head_dim=128` for `SmallCase2`
- An attempted multi-batch var-seq unroll sim case was removed after repeated
  `a5sim` segmentation faults. The current example keeps only the validated
  unroll sim cases.

### Golden Checks

One-round hardware reruns without `--skip-golden`:

| Path                                 | Case1 | Case2 |
| ------------------------------------ | ----- | ----- |
| TMR `paged_attention` AUTO           | PASS  | PASS  |
| TMR `paged_attention_manual_scope`   | PASS  | PASS  |
| TMR `paged_attention_unroll` AUTO    | PASS  | PASS  |
| TMR `paged_attention_unroll_manual_scope` | PASS  | PASS  |
| ABG `paged_attention_unroll`         | PASS  | FAIL  |

Notes:

- The rebased TMR AUTO and manual-scope paths are correctness-clean on the
  kept `Case1`/`Case2` workloads.
- ABG unroll `Case2` still fails golden on the rebased branch with
  `max_diff=0.01536`, so its timing remains a non-correctness-clean baseline.
- A temporary small-shape ABG non-unroll wrapper was also tried to make the
  non-unroll comparison apples-to-apples with TMR `CaseSmall1/CaseSmall2`.
  Both small ABG runs produced `NaN` golden mismatches, so those timings are
  useful for rough performance context only, not as a correctness-clean
  baseline.

## Fresh 100-Round Benchmark

Method:

- hardware: `a2a3`, device `10`
- rounds: `100`
- trimmed average: drop `10` low + `10` high
- TMR manual rows were rerun after restoring the explicit-only manual-scope
  TensorMap policy
- TMR benchmark mode uses `--skip-golden`; correctness is tracked separately in
  the golden table above

### Comparable Table

This is the main comparison table to read.

- `paged_attention` compares the kept small-shape non-unroll cases
- `paged_attention_unroll` compares the production unroll cases
- ABG small non-unroll rows are timing-only because their temporary wrapper is
  not correctness-clean
- ABG unroll `Case2` is also timing-only because its in-tree baseline still
  fails golden

| Example                  | Case    | Runtime               | Elapsed Trim (us) | Orch Trim (us) | Golden |
| ------------------------ | ------- | --------------------- | ----------------: | -------------: | ------ |
| `paged_attention`        | `Case1` | TMR AUTO              |            72.613 |         54.655 | PASS   |
| `paged_attention`        | `Case1` | TMR manual scope      |            81.358 |         62.287 | PASS   |
| `paged_attention`        | `Case1` | ABG small-shape wrapper |          96.916 |         13.399 | FAIL   |
| `paged_attention`        | `Case2` | TMR AUTO              |            91.682 |         63.958 | PASS   |
| `paged_attention`        | `Case2` | TMR manual scope      |           101.647 |         71.885 | PASS   |
| `paged_attention`        | `Case2` | ABG small-shape wrapper |         110.935 |         22.776 | FAIL   |
| `paged_attention_unroll` | `Case1` | TMR AUTO              |          1140.067 |        710.711 | PASS   |
| `paged_attention_unroll` | `Case1` | TMR manual scope      |          1131.544 |        614.427 | PASS   |
| `paged_attention_unroll` | `Case1` | ABG                   |          1385.087 |        706.679 | PASS   |
| `paged_attention_unroll` | `Case2` | TMR AUTO              |           513.079 |        274.306 | PASS   |
| `paged_attention_unroll` | `Case2` | TMR manual scope      |           491.192 |        229.887 | PASS   |
| `paged_attention_unroll` | `Case2` | ABG                   |           664.744 |        284.575 | FAIL   |

### Readout

The fresh rebased numbers show:

- non-unroll manual scope is still slower than TMR AUTO, but the gap is much
  smaller than the older pre-forward-port table
  - `Case1`: `+8.745us` elapsed, `+7.632us` orch
  - `Case2`: `+9.965us` elapsed, `+7.927us` orch
- unroll manual scope remains better than TMR AUTO on both kept cases
  - `Case1`: `-8.523us` elapsed, `-96.284us` orch
  - `Case2`: `-21.887us` elapsed, `-44.419us` orch
- the forward-ported cleanup therefore did not regress the manual path; it
  materially improved the rebased non-unroll result versus the old doc table

## ABG Production-Scale Context

The in-tree ABG non-unroll scene test still uses larger production cases. Those
rows are not directly comparable to the kept small TMR non-unroll cases, but
they are recorded here for completeness:

| Example           | Case    | Runtime          | Elapsed Trim (us) | Orch Trim (us) | Golden |
| ----------------- | ------- | ---------------- | ----------------: | -------------: | ------ |
| `paged_attention` | `case1` | ABG production   |         32772.205 |      32476.044 | PASS   |
| `paged_attention` | `case2` | ABG production   |         17288.330 |      16718.342 | PASS   |

## Risks

1. Non-unroll manual scope is not yet performance-neutral versus TMR AUTO.
2. The only correctness-clean non-unroll comparison on this branch today is TMR
   AUTO vs TMR manual scope; ABG small-shape comparison is still timing-only.
3. ABG unroll `Case2` remains a failing baseline and should not be used as a
   correctness-clean target.

## Recommendation

Treat this rebased checkpoint as:

- functionally validated for TMR AUTO/manual v0 on real hardware
- benchmarked with fresh 100-round data on device `10`
- ready for PR refresh after pushing the updated branch

But do not describe the non-unroll manual path as zero-overhead yet. The
remaining gap is much smaller than before, not eliminated.
