# PR1 Plan: Require Explicit NEXT_LEVEL Placement

## Objective

Make the public NEXT_LEVEL submit contract require a concrete stable worker ID
for every task. Keep the existing affinity-aware dispatch implementation in
place so this PR changes the contract and validation without changing queue
ownership or dependency scheduling.

This PR is the compatibility boundary for the stacked series. After it lands,
new public submissions cannot request automatic worker selection.

## Scope

- Require `worker=` on `Orchestrator.submit_next_level`.
- Require `workers=` on `Orchestrator.submit_next_level_group`.
- Require one worker ID per group member.
- Reject negative IDs, duplicate group IDs, unknown workers, and targets that
  are outside the final callable/data eligibility set.
- Preserve stable NEXT_LEVEL worker-ID semantics for local and remote workers.
- Update in-repository callers to pass their intended worker IDs.
- Add focused API and validation tests.
- Update contract documentation touched by the API change.

## Explicit Non-Goals

- Do not change `submit_sub` or `submit_sub_group`.
- Do not add SUB worker IDs.
- Do not change ready-queue topology.
- Do not remove the scheduler's idle-worker fallback yet.
- Do not change dependency inference, group completion, failure poisoning,
  buffer lifetime, or callable registration.
- Do not add compatibility flags or environment-variable gates.

## Planned File Changes

### Public Python and binding API

- `python/simpler/orchestrator.py`
  - Make `worker` and `workers` required keyword-only arguments.
  - Reject negative and duplicate targets before calling C++.
  - Describe the values as exact targets rather than affinities.
- `python/bindings/worker_bind.h`
  - Remove nanobind defaults for `worker` and `workers`.
  - Update binding help text.

### C++ submit validation

- `src/common/hierarchical/orchestrator.cpp`
  - Reject missing or negative NEXT_LEVEL targets.
  - Require group target count to equal group size.
  - Reject duplicate group targets.
  - Continue validating target membership in `eligible_worker_ids`.
- `src/common/hierarchical/orchestrator.h`
  - Remove public default values for NEXT_LEVEL target arguments.
  - Update comments to state the exact-placement contract.

### Call sites and tests

- Update examples, scene tests, Python unit tests, and C++ unit tests that rely
  on the old unconstrained default.
- Use worker `0` only where the fixture creates exactly one NEXT_LEVEL worker.
- Use explicit member IDs for group fixtures.
- Preserve tests for callable/data eligibility by selecting an ID from the
  final eligible set.

### Documentation

- Update `docs/orchestrator.md`, `docs/scheduler.md`, `docs/task-flow.md`,
  `docs/hierarchical_level_runtime.md`, and remote-L3 placement text.
- Remove statements that `-1`, `None`, or an empty list means unconstrained.

## Validation

- Python tests prove missing `worker` and `workers` fail at the API boundary.
- C++ tests prove negative, unknown, duplicate, count-mismatched, and
  ineligible targets fail before slot allocation.
- Existing local and remote explicit-placement tests continue to pass.
- Relevant examples and scene tests collect with the required arguments.

## Size Budget

- Production code: at most 250 added lines.
- Tests and call-site updates: at most 350 added lines.
- Documentation: at most 250 added lines.
- Total target: at most 850 added lines.

## Completion Criteria

- Every public NEXT_LEVEL submit carries exact target worker IDs.
- No SUB API or behavior changes.
- No queue or dispatch algorithm changes.
- The PR builds and its focused Python/C++ tests pass independently.
