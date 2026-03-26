# Submit by Cluster - Requirements and Main-Branch-Aligned Design

## 1. Goal

Define a single, main-branch-aligned specification for PTO2 cluster submission that combines:

1. Product requirements (what must be true).
2. Runtime design (how it is implemented on current main baseline).

The target model is: one submitted graph node is one `MixedTask`, and dispatch/completion is mixed-task-granular.

## 2. Background and Motivation

Future Ascend hardware is expected to provide stronger locality within an AICore cluster (`1 AIC + 2 AIV`).
The runtime therefore needs a "submit together, run together" model for related AIC/AIV kernels.

Legacy per-task submit (`kernel_id + worker_type`) cannot express atomic co-dispatch of multiple kernels to one cluster.

## 3. Scope

### In Scope

1. New orchestration-facing submit API for cluster-aware mixed submission.
2. Runtime/backend scheduler and executor changes to treat a mixed submit as one atomic scheduling unit.
3. Dependency gating, readiness, dispatch, completion, and reclamation at mixed-task granularity.
4. AIV slot equivalence (`AIV0` and `AIV1` are equivalent execution targets).

### Out of Scope

1. User-facing cluster pinning (`allocate_cluster/free_cluster`-style APIs).
2. New worker types beyond AIC/AIV.
3. Cross-cluster user placement policies.
4. Hardware topology changes beyond `1 AIC + 2 AIV` per cluster.

## 4. Main-Branch Baseline Constraints

Design must preserve the current main runtime architecture:

1. Multi-orchestrator runtime wiring (`orchestrators[]`, `orch_count`, thread-local `pto2_current_orch_idx`).
2. Executor threading split (orchestrator threads vs scheduler threads), and post-orchestrator transition (`transition_requested_` + `reassign_cores_for_all_threads()`).
3. Shared-memory hot/cold split (`PTO2TaskDescriptor` hot + `PTO2TaskPayload` cold).

## 5. Terminology

1. `cluster`: one physical unit with `1 AIC + 2 AIV`.
2. `MixedKernels`: 3 submit slots (`AIC`, `AIV0`, `AIV1`) with `INVALID_KERNEL_ID` for inactive slots.
3. `MixedTask`: one runtime graph node created by one submit call.
4. `active_mask`: bitmask of active subtask slots.
5. `resource shape`: normalized lane demand class of a mixed task.

## 6. API Contract

```cpp
inline constexpr int32_t INVALID_KERNEL_ID = -1;

struct MixedKernels {
    int32_t aic_kernel_id{INVALID_KERNEL_ID};
    int32_t aiv0_kernel_id{INVALID_KERNEL_ID};
    int32_t aiv1_kernel_id{INVALID_KERNEL_ID};
};

static inline void pto2_rt_submit_task(PTO2Runtime* rt,
                                       const MixedKernels& mixed_kernels,
                                       PTOParam* params,
                                       int32_t num_params);

static inline void pto2_rt_submit_aic_task(PTO2Runtime* rt,
                                           int32_t kernel_id,
                                           PTOParam* params,
                                           int32_t num_params);

static inline void pto2_rt_submit_aiv_task(PTO2Runtime* rt,
                                           int32_t kernel_id,
                                           PTOParam* params,
                                           int32_t num_params);
```

Rules:

1. One submit call creates one `MixedTask`.
2. All active slots share the same `params` and `num_params`.
3. At least one slot must be active.
4. `aiv0_kernel_id` and `aiv1_kernel_id` are semantically equivalent.
5. Wrappers are orchestration sugar only (inline in orchestration API); no dedicated runtime ops entries.
6. Submit-contract types are defined once in a shared header-only submit-types surface consumed by orchestration and runtime headers.
7. Invalid submits follow existing PTO2 behavior (`always_assert`), not a new recoverable return-code API.

## 7. Data Model (Requirements + Design)

`PTO2TaskDescriptor` (hot path) carries mixed-task identity/state:

1. `task_id`
2. `active_mask`
3. `subtask_done_mask`
4. `kernel_id[3]` for `(AIC, AIV0, AIV1)`
5. dependency heads/counters and packed-buffer metadata

`PTO2TaskPayload` (cold path) carries:

1. shared params/tensors/scalars copied once per mixed submit
2. fanin mixed-task IDs
3. other cold-path submit metadata

Producer identity in TensorMap is mixed-task ID end-to-end.

## 8. Scheduling Model

### 8.1 Resource Shapes

Runtime uses shape-based ready queues (not worker-type queues):

1. `AIC_ONLY`
2. `AIV_X1`
3. `AIV_X2`
4. `AIC_AIV_X1`
5. `AIC_AIV_X2`

Queueing key is normalized resource shape (not raw slot label).

### 8.2 Atomic Cluster Dispatch

1. Dispatch decision unit is one mixed task.
2. For multi-slot mixed tasks, partial launch is forbidden.
3. A mixed task is dispatchable only when one local owned cluster can satisfy all required lanes.
4. Compatible mixed tasks may co-reside over time if they use disjoint free lanes.

### 8.3 Dependency and Completion

1. Fanin release/readiness remains dependency-correct and graph-level.
2. Two-stage completion:
   - `on_subtask_complete(task_id, subslot)`
   - `on_mixed_task_complete(task_id)` only when `subtask_done_mask == active_mask`
3. Downstream release is triggered once per mixed task completion, not once per subslot.

## 9. Executor Ownership and Numbering

### 9.1 Canonical Flattened Numbering (Unchanged)

Given `block_dim` clusters:

1. AIC IDs: `[0, block_dim)`
2. AIV IDs: `[block_dim, 3 * block_dim)`
3. Cluster `i`: `{i, block_dim + i, 2 * block_dim + i}`

This project-defined flattened numbering is kept unchanged.

### 9.2 Cluster Ownership

1. One cluster must be owned by one scheduler domain/thread at a time.
2. No split-cluster ownership in either:
   - initial `assign_cores_to_threads()`
   - post-orchestrator `reassign_cores_for_all_threads()`
3. Lane occupancy bookkeeping must remain consistent with ownership after reassignment.

## 10. Functional Requirements

### 10.1 Valid Mixed Shapes

1. AIC only
2. AIV only (1 or 2 AIV lanes)
3. AIC + 1 AIV
4. AIC + 2 AIV

### 10.2 Runtime Behavior per Submit

1. Validate submit arguments.
2. Allocate mixed-task ID and initialize descriptor/payload once.
3. Build fanin/fanout at mixed-task granularity.
4. Enqueue by shape when ready.
5. Dispatch all active lanes atomically when resources allow.
6. Aggregate completion and release downstream once.

## 11. Non-Functional Requirements

1. Correctness: no dependency violation, no partial mixed-task dispatch.
2. Determinism: dependency-correct ordering preserved; AIV lane choice may vary but remains semantically equivalent.
3. Fairness: resource-aware polling heuristic is allowed; strict starvation-free guarantee across all shapes is not required.
4. Performance: no obvious regression for non-cluster workflows.
5. Observability: lifecycle visibility for submit/ready/dispatch/block/complete.

## 12. Acceptance Criteria

Feature is accepted when:

1. Orchestration compiles and submits via `MixedKernels` API/wrappers.
2. Scheduler dispatches each mixed task as one cluster scheduling decision.
3. Dependencies gate mixed-task readiness correctly.
4. AIV execution remains cluster-local and semantically equivalent across lanes.
5. Existing non-cluster workflows continue to pass without behavior regression.
6. Cluster ownership is never split across scheduler domains before/after transition.

## 13. Verification Matrix

Recommended validation coverage:

1. Mapping correctness for cluster-to-core ID relation.
2. Atomic dispatch for multi-slot shapes.
3. Dependency gating and completion aggregation (`done_mask == active_mask`).
4. Lane-occupancy co-residency behavior for compatible shapes.
5. Multi-orchestrator and core-transition ownership stability.
6. Invalid submit handling (`always_assert` path).
7. Regression coverage for existing examples/tests.

Milestone command (device):

```bash
python examples/scripts/run_example.py \
  -k tests/st/tensormap_and_ringbuffer/batch_paged_attention/kernels \
  -g tests/st/tensormap_and_ringbuffer/batch_paged_attention/golden.py \
  -p a2a3 -d 9
```

Final validation:

```bash
./ci.sh
```

## 14. Resolved Decisions

1. Legacy orchestration-facing single-task submit is replaced by mixed submit contract.
2. Invalid mixed submits fail with existing submit-time assert behavior.
3. Per-cluster concurrent capacity is lane-occupancy-driven, not a fixed constant.
4. Submit-contract types live in one shared header-only surface.
5. Resource-aware dispatch heuristics are allowed without a strict starvation-free guarantee.

