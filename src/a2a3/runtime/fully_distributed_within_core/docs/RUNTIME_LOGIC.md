# Runtime Logic: fully_distributed_within_core

**Target design.** Orchestration, scheduling, and execution all run on the AI
cores in SPMD fashion; the AICPU is removed from orchestration/scheduling. The
authoritative specification is:

- [`docs/fully_distributed_within_core.md`](../../../../docs/fully_distributed_within_core.md)

Core elements (see the spec):

- Task ownership via a claim race over two global cursors (`cube_cursor`,
  `vector_cursor`); `owner = builder = executor`.
- Per-core full-duplicate TensorMap for dependency discovery (pull model via a
  global `task_completed_flag` ring).
- Per-core private task ring + block-shared `block.won[N]` deposit table for
  multi-core (MIX / 2V) co-ownership (anchor push + follower async drain).
- Deterministic, per-core-replicated GM output heap with frontier-based
  reclamation.

## Current state (re-based on tensormap_and_ringbuffer)

This runtime is re-based on `tensormap_and_ringbuffer` to reuse its
`PTO2TensorMap`, `MixedKernels`/`ActiveMask`, `L0TaskArgs`, the
`pto_orchestration_api.h` submit API, and kernel-address resolution. The
distributed model is layered on incrementally:

- `runtime/` — adds global claim cursors, a global completion-flag ring, a
  deterministic GM output heap, and per-core replicated TensorMap + private task
  ring on top of the reused types.
- `aicore/` — the SPMD run-ahead orchestrate+execute loop (spec section 6).
- `aicpu/` — reduced to an init/wire/signal/wait stub (no orchestration,
  scheduling, or dispatch).
- `host/` — runtime maker / compile info (orchestration entry is invoked on the
  cores).
- `orchestration/` — the PTO2 orchestration API (unchanged surface).

The legacy AICPU orchestrator/scheduler sources inherited from
`tensormap_and_ringbuffer` (`runtime/scheduler/`, the orchestrator pipeline) are
progressively replaced or bypassed by the distributed path.
