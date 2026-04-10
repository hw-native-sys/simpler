# Manual Scope Non-Unroll Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce non-unroll `tensormap_and_ringbuffer_partial_manual` orchestration overhead by removing common-case manual bookkeeping while preserving AUTO mode and cross-scope correctness.

**Architecture:** Keep TensorMap semantics unchanged. Optimize only the manual runtime path by adding a common-case `scope_end()` bypass for `dep_pool_mark` repair and an O(1) task-id lookup for `pto2_add_dependency()`. Use a conservative fallback flag so retroactive manual edges still take the old repair path.

**Tech Stack:** C++, pytest, hardware example runner, benchmark_rounds.sh

---

### Task 1: Add failing/manual-risk regression coverage

**Files:**
- Modify: `tests/ut/test_manual_scope_boundary.py`
- Modify: `tests/ut/test_manual_scope_guards.py`
- Create: `tests/ut/test_manual_scope_perf_invariants.py`
- Test: `tests/ut/test_manual_scope_perf_invariants.py`

- [ ] **Step 1: Write regression tests for the two correctness-sensitive cases**

Add a new UT file with explicit coverage for:
- tail-consumer edge insertion does not require repair fallback
- retroactive edge insertion to an older consumer still requires repair fallback

Use this test skeleton:

```python
import pytest

from tests.ut.hardware_test_utils import run_example_subprocess


def test_manual_scope_tail_consumer_path_keeps_fast_publish():
    result = run_example_subprocess(
        "tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_outer_multiwrite/kernels",
        golden_py=None,
        extra_env={
            "PTO2_DEBUG_DUMP_MANUAL_SCOPE": "1",
            "PTO2_EXPECT_MANUAL_SCOPE_REPAIR": "0",
        },
    )
    assert result.returncode == 0
    assert "manual_scope_repair_needed=0" in result.stdout


def test_manual_scope_retroactive_edge_enables_repair_fallback():
    result = run_example_subprocess(
        "tests/st/a2a3/tensormap_and_ringbuffer/manual_scope_outer_multiwrite/kernels",
        golden_py=None,
        extra_env={
            "PTO2_DEBUG_FORCE_RETROACTIVE_MANUAL_EDGE": "1",
            "PTO2_DEBUG_DUMP_MANUAL_SCOPE": "1",
            "PTO2_EXPECT_MANUAL_SCOPE_REPAIR": "1",
        },
    )
    assert result.returncode == 0
    assert "manual_scope_repair_needed=1" in result.stdout
```

- [ ] **Step 2: Run the new test file and confirm it fails before implementation**

Run:

```bash
pytest tests/ut/test_manual_scope_perf_invariants.py -v
```

Expected:

```text
FAIL
```

because the runtime does not yet expose or enforce the new manual-scope fast/fallback invariant.

- [ ] **Step 3: Keep existing guard/boundary UT coverage visible in the plan**

Run:

```bash
pytest tests/ut/test_manual_scope_boundary.py tests/ut/test_manual_scope_guards.py -v
```

Expected:

```text
PASS
```

These are baseline safety tests. Do not weaken them while implementing the optimization.

- [ ] **Step 4: Commit the test scaffolding once it is stable**

```bash
git add tests/ut/test_manual_scope_perf_invariants.py tests/ut/test_manual_scope_boundary.py tests/ut/test_manual_scope_guards.py
git commit -m "Add: manual scope optimization guard tests"
```

### Task 2: Add manual-scope state for safe fast-path publish

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Test: `tests/ut/test_manual_scope_perf_invariants.py`

- [ ] **Step 1: Add explicit manual-scope state to the orchestrator**

Extend `PTO2OrchestratorState` with narrowly scoped state for the current active manual scope:

```cpp
struct PTO2OrchestratorState {
    ...
    bool manual_scope_active{false};
    bool manual_scope_needs_dep_pool_repair{false};
    PTO2TaskId *manual_scope_lookup_keys{nullptr};
    int32_t *manual_scope_lookup_values{nullptr};
    int32_t manual_scope_lookup_capacity{0};
    ...
};
```

The new state is only for the current manual scope. It is not global task tracking.

- [ ] **Step 2: Initialize and destroy the new state**

Update orchestrator init/destroy to allocate and free the lookup storage:

```cpp
orch->manual_scope_lookup_capacity = pto2_next_pow2(PTO2_TASK_WINDOW_SIZE * 2);
orch->manual_scope_lookup_keys =
    static_cast<PTO2TaskId *>(pto2_aligned_zalloc(sizeof(PTO2TaskId) * orch->manual_scope_lookup_capacity, 64));
orch->manual_scope_lookup_values = static_cast<int32_t *>(
    pto2_aligned_zalloc(sizeof(int32_t) * orch->manual_scope_lookup_capacity, 64)
);
for (int32_t i = 0; i < orch->manual_scope_lookup_capacity; i++) {
    orch->manual_scope_lookup_keys[i] = PTO2TaskId::invalid();
    orch->manual_scope_lookup_values[i] = -1;
}
```

The lookup capacity must stay power-of-two because the probe path uses
`& (capacity - 1)` masking.

Destroy path:

```cpp
free(orch->manual_scope_lookup_keys);
free(orch->manual_scope_lookup_values);
orch->manual_scope_lookup_keys = nullptr;
orch->manual_scope_lookup_values = nullptr;
```

- [ ] **Step 3: Reset the manual-scope state at manual scope boundaries**

At `pto2_scope_begin(..., PTO2ScopeMode::MANUAL)`:

```cpp
orch->manual_scope_active = true;
orch->manual_scope_needs_dep_pool_repair = false;
for (int32_t i = 0; i < orch->manual_scope_lookup_capacity; i++) {
    orch->manual_scope_lookup_keys[i] = PTO2TaskId::invalid();
    orch->manual_scope_lookup_values[i] = -1;
}
```

At manual `scope_end()` teardown:

```cpp
for (int32_t i = 0; i < orch->manual_scope_lookup_capacity; i++) {
    orch->manual_scope_lookup_keys[i] = PTO2TaskId::invalid();
    orch->manual_scope_lookup_values[i] = -1;
}
orch->manual_scope_needs_dep_pool_repair = false;
orch->manual_scope_active = false;
```

Keep this reset explicit and conservative. Stale manual lookup state is a correctness risk.

- [ ] **Step 4: Run the perf-invariant UT file**

Run:

```bash
pytest tests/ut/test_manual_scope_perf_invariants.py -v
```

Expected:

```text
still FAIL or partially FAIL
```

The state exists, but the submit/add-dependency path has not started using it yet.

- [ ] **Step 5: Commit the state addition**

```bash
git add src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp
git commit -m "Update: add manual scope repair state"
```

### Task 3: Replace linear manual task lookup with O(1) lookup

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Test: `tests/ut/test_manual_scope_perf_invariants.py`

- [ ] **Step 1: Register manual tasks in the lookup table on submit**

Right after a manual task is appended to the current scope, populate the active-scope lookup:

```cpp
int32_t scope_offset = orch->scope_tasks_size - current_manual_scope_begin(orch) - 1;
always_assert(scope_offset >= 0);
uint32_t slot = static_cast<uint32_t>(task_id.raw) & (orch->manual_scope_lookup_capacity - 1);
while (orch->manual_scope_lookup_keys[slot].is_valid()) {
    slot = (slot + 1) & (orch->manual_scope_lookup_capacity - 1);
}
orch->manual_scope_lookup_keys[slot] = task_id;
orch->manual_scope_lookup_values[slot] = scope_offset;
```

If a helper makes this clearer, add one:

```cpp
static inline void manual_scope_register_task(PTO2OrchestratorState *orch, PTO2TaskId task_id, int32_t scope_offset);
```

- [ ] **Step 2: Add a lookup helper for current manual scope tasks**

Replace the double linear search with a current-scope helper:

```cpp
static int32_t find_current_manual_scope_task_index_fast(PTO2OrchestratorState *orch, PTO2TaskId task_id) {
    uint32_t slot = static_cast<uint32_t>(task_id.raw) & (orch->manual_scope_lookup_capacity - 1);
    while (true) {
        PTO2TaskId key = orch->manual_scope_lookup_keys[slot];
        if (!key.is_valid()) {
            return -1;
        }
        if (key == task_id) {
            return orch->manual_scope_lookup_values[slot];
        }
        slot = (slot + 1) & (orch->manual_scope_lookup_capacity - 1);
    }
}
```

If the storage shape needs refinement after inspection, keep the rule:
- no scans over `scope_tasks[]`
- no scans over the whole current manual scope
- explicit validation that the task belongs to the current active manual scope

- [ ] **Step 3: Use the fast lookup in `pto2_add_dependency()`**

Change:

```cpp
int32_t producer_idx = find_current_manual_scope_task_index(orch, producer_id);
int32_t consumer_idx = find_current_manual_scope_task_index(orch, consumer_id);
```

to:

```cpp
int32_t producer_idx = find_current_manual_scope_task_index_fast(orch, producer_id);
int32_t consumer_idx = find_current_manual_scope_task_index_fast(orch, consumer_id);
```

Keep the existing invalid-args behavior unchanged when either lookup fails.

- [ ] **Step 4: Run the focused UTs**

Run:

```bash
pytest tests/ut/test_manual_scope_perf_invariants.py tests/ut/test_manual_scope_boundary.py tests/ut/test_manual_scope_guards.py -v
```

Expected:

```text
manual guard/boundary tests PASS
perf-invariant tests may still FAIL until Task 4
```

- [ ] **Step 5: Commit the lookup change**

```bash
git add src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp
git commit -m "Update: speed up manual dependency lookup"
```

### Task 4: Make manual `scope_end()` skip repair in the common case

**Files:**
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`
- Modify: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h`
- Test: `tests/ut/test_manual_scope_perf_invariants.py`

- [ ] **Step 1: Detect whether `add_dependency()` is retroactive**

In `pto2_add_dependency()`, mark the fallback only when the consumer is not the
current tail task in the active manual scope:

```cpp
int32_t current_tail_idx = orch->scope_tasks_size - current_manual_scope_begin(orch) - 1;
if (consumer_idx != current_tail_idx) {
    orch->manual_scope_needs_dep_pool_repair = true;
}
```

Keep the common case fast:

```cpp
if (!orch->manual_scope_needs_dep_pool_repair) {
    consumer_slot_state->dep_pool_mark = dep_pool.top;
}
```

Do not try to be cleverer than this in the first pass. Conservative fallback is
part of the safety design.

- [ ] **Step 2: Gate the repair loop in `pto2_scope_end()`**

Change manual `scope_end()` from unconditional repair:

```cpp
int32_t dep_pool_mark_prefix = 0;
for (int32_t task_idx = 0; task_idx < count; task_idx++) {
    ...
}
```

to:

```cpp
if (orch->manual_scope_needs_dep_pool_repair) {
    int32_t dep_pool_mark_prefix = 0;
    for (int32_t task_idx = 0; task_idx < count; task_idx++) {
        PTO2TaskSlotState *slot_state = orch->scope_tasks[begin + task_idx];
        PTO2TaskPayload *payload = slot_state->payload;
        if (payload->fanin_actual_count > PTO2_MAX_INPUTS) {
            ...
        }
        if (slot_state->dep_pool_mark < dep_pool_mark_prefix) {
            slot_state->dep_pool_mark = dep_pool_mark_prefix;
        } else {
            dep_pool_mark_prefix = slot_state->dep_pool_mark;
        }
    }
} else {
    for (int32_t task_idx = 0; task_idx < count; task_idx++) {
        PTO2TaskSlotState *slot_state = orch->scope_tasks[begin + task_idx];
        PTO2TaskPayload *payload = slot_state->payload;
        if (payload->fanin_actual_count > PTO2_MAX_INPUTS) {
            ...
        }
    }
}
orch->scheduler->publish_manual_scope_tasks_and_end_scope(&orch->scope_tasks[begin], count);
```

The overflow check stays on both paths. Only the repair scan is conditional.

- [ ] **Step 3: Add optional debug logging for the invariant tests**

Add a low-noise debug gate for the UTs:

```cpp
if (getenv("PTO2_DEBUG_DUMP_MANUAL_SCOPE") != nullptr) {
    LOG_INFO(
        "manual_scope_repair_needed=%d count=%d",
        orch->manual_scope_needs_dep_pool_repair ? 1 : 0,
        count
    );
}
```

This is test support, not a permanent verbose logging mode. Keep it behind env control.

- [ ] **Step 4: Run the focused tests until they all pass**

Run:

```bash
pytest tests/ut/test_manual_scope_perf_invariants.py tests/ut/test_manual_scope_boundary.py tests/ut/test_manual_scope_guards.py -v
```

Expected:

```text
PASS
```

- [ ] **Step 5: Commit the common-case bypass**

```bash
git add src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h tests/ut/test_manual_scope_perf_invariants.py
git commit -m "Update: bypass manual scope repair in common case"
```

### Task 5: Run device validation and benchmark the real impact

**Files:**
- Modify: `docs/manual-dep-for-tensormap-design.md`
- Test: `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual/kernels/orchestration/paged_attention_orch.cpp`
- Test: `tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual/kernels/orchestration/paged_attention_orch.cpp`

- [ ] **Step 1: Run partial-manual real-device validation**

Run:

```bash
python examples/scripts/run_example.py --build \
  -k tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual/kernels \
  -g tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual/golden.py \
  -p a2a3 -d 4 -c d96c8784 --case Case1

python examples/scripts/run_example.py --build \
  -k tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual/kernels \
  -g tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_partial_manual/golden.py \
  -p a2a3 -d 4 -c d96c8784 --case Case2
```

Expected:

```text
PASS
```

- [ ] **Step 2: Run unroll partial-manual sanity after the rebase ABI fix**

Run:

```bash
python examples/scripts/run_example.py --build \
  -k tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual/kernels \
  -g tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual/golden.py \
  -p a2a3 -d 4 -c d96c8784 --case Case1

python examples/scripts/run_example.py --build \
  -k tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual/kernels \
  -g tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll_partial_manual/golden.py \
  -p a2a3 -d 4 -c d96c8784 --case Case2
```

Expected:

```text
PASS
```

- [ ] **Step 3: Run the fresh 4-way benchmark**

Run current branch:

```bash
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r aicpu_build_graph --build
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r tensormap_and_ringbuffer --build
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r tensormap_and_ringbuffer_partial_manual --build
```

Run the unmodified baseline from its worktree:

```bash
cd /data/uvxiao/pto-runtime/.worktrees/tmr-unmodified-a71ba16
. .venv/bin/activate
export PTO_ISA_ROOT=/data/uvxiao/pto-runtime/.worktrees/manual-dep-merge-forward/examples/scripts/_deps/pto-isa
./tools/benchmark_rounds.sh -d 4 -n 5 -c d96c8784 -r tensormap_and_ringbuffer_unmodified --build
```

Expected:

```text
fresh logs with Trimmed Avg / Orch Trimmed Avg for all four runtimes
```

- [ ] **Step 4: Update the design note with gain attribution**

Add the new numbers to `docs/manual-dep-for-tensormap-design.md` and clearly tie
the measured gain to:

- common-case `scope_end()` repair bypass
- faster manual edge lookup
- any unchanged remaining boundary TensorMap cost

Use a short table and a short interpretation section.

- [ ] **Step 5: Commit verification + docs refresh**

```bash
git add docs/manual-dep-for-tensormap-design.md
git commit -m "Update: refresh manual scope optimization results"
```

### Task 6: Final verification before PR update

**Files:**
- Modify: `docs/manual-dep-for-tensormap-design.md`
- Test: current branch state

- [ ] **Step 1: Re-run focused UTs**

Run:

```bash
pytest tests/ut/test_manual_scope_perf_invariants.py tests/ut/test_manual_scope_boundary.py tests/ut/test_manual_scope_guards.py -v
```

Expected:

```text
PASS
```

- [ ] **Step 2: Re-check worktree status and commit stack**

Run:

```bash
git status --short
git log --oneline --max-count=8
```

Expected:

```text
clean worktree
new optimization commits visible on top of 9896375
```

- [ ] **Step 3: Prepare PR update summary**

Summarize:

- what runtime logic changed
- what risk fallback remains in place
- new non-unroll comparison table
- whether partial-manual moved closer to `aicpu_build_graph`

- [ ] **Step 4: Final commit if any verification/docs touch remains**

```bash
git add docs/manual-dep-for-tensormap-design.md
git commit -m "Update: record manual optimization verification"
```
