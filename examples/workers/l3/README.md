# L3 — Host-level multi-chip examples

**L3 = HOST**: one host machine that drives multiple L2 chips plus M
SubWorkers (plain Python callables), coordinated by an Orchestrator running in
the host process. This is where you first see the *DAG* model — you submit a
task per chip, each task carries a dependency graph via `orchestrator` APIs,
and the runtime schedules them onto available devices.

See [`docs/hierarchical_level_runtime.md`](../../../docs/hierarchical_level_runtime.md)
for the full L0–L6 diagram and [`docs/task-flow.md`](../../../docs/task-flow.md)
for data-flow end to end.

## Minimum Worker lifecycle

L3 adds two steps before `init()`:

```python
from simpler.worker import Worker

worker = Worker(
    level=3,
    platform="a2a3sim",
    runtime="tensormap_and_ringbuffer",
    device_ids=[0, 1],        # two chips
    num_sub_workers=1,        # one Python post-processing callable
)

# 1. Register sub-worker callables BEFORE init (level >= 3 only).
#    Returns an opaque handle you pass to orchestrator.submit_sub(...) later.
postprocess_handle = worker.register(
    lambda args: print("post-process received", args)
)

worker.init()                 # forks chip child processes + sub children,
                              # then starts the C++ scheduler

def my_orch(orch, args, cfg):
    # orch is the Orchestrator. Submit one task per chip + any sub work.
    # orch.submit_next_level(...) schedules a ChipCallable onto a free chip.
    # orch.submit_sub(postprocess_handle, sub_args) schedules a Python callable.
    ...

try:
    worker.run(my_orch, my_args, my_config)
finally:
    worker.close()            # shuts down child processes and releases shm
```

Two things to know before reading the example:

1. **This example registers callables before `init()`**. That keeps startup
   simple and lets chip children pre-warm their callable state before the
   first DAG dispatch.
2. **The orchestration function is a *plain Python function*, not a C++
   kernel.** It runs in the host process and calls `orch.submit_*(...)` to
   hand work to chip children. The children get the submitted `ChipCallable`
   through shared-memory mailboxes.

## What each example demonstrates

| Directory | New concept |
| --------- | ----------- |
| [`multi_chip_dispatch/`](multi_chip_dispatch/) | Two chips + one SubWorker. An orchestration fn dispatches a `ChipCallable` to each chip, then submits a Python callable to collect/verify results. |
| [`child_memory/`](child_memory/) | `orch.malloc` + `ContinuousTensor(child_memory=True)` to load a weight once and reuse it across multiple kernel invocations on the same chip. |
| [`allreduce_distributed/`](allreduce_distributed/) | One communication domain allocated inside the orchestration via `orch.allocate_domain`, with PTO-ISA remote reads over the domain window. |
| [`allreduce_ring_distributed/`](allreduce_ring_distributed/) | Same domain/window setup as mesh allreduce; **chunked ring RS+AG** (`2(P−1)` rounds, one chunk per step) instead of O(P) full-vector remote reads. |
| [`allgather_distributed/`](allgather_distributed/) | One communication domain via `orch.allocate_domain`; each rank stages its slice, synchronizes across ranks, then gathers every rank's window data into a full output. |
| [`reduce_scatter_distributed/`](reduce_scatter_distributed/) | One communication domain via `orch.allocate_domain`; each rank stages all input chunks, synchronizes, then reduces the per-rank chunk across peers into a rank-local output. |
| [`broadcast_distributed/`](broadcast_distributed/) | One communication domain via `orch.allocate_domain`; root stages into the window, synchronizes, then every rank reads the root's scratch slot into its output. |
| [`all_to_all_distributed/`](all_to_all_distributed/) | One communication domain via `orch.allocate_domain`; scratch indexed by destination rank, barrier, then each rank gathers the slice peers sent to it. |
| [`ffn_tp_parallel/`](ffn_tp_parallel/) | Local compute followed by one-domain cross-rank reduction through a domain scratch window. |
| [`ep_dispatch_combine/`](ep_dispatch_combine/) | MoE-style dispatch/combine over a one-domain communication window. |
| [`domain_rank_map/`](domain_rank_map/) | Small two-domain example showing domain-local ranks, missing-domain `KeyError`, separate window slices, and real per-domain allreduce. |
| [`dual_domain_overlap/`](dual_domain_overlap/) | Two overlapping communication domains where worker 1 participates in both, each allocated inside the orchestration via `orch.allocate_domain` and indexed by domain-local rank. |

## Prerequisites

Same as L2 (see [`../l2/README.md`](../l2/README.md)): venv + `pip install .`.

Additionally, L3 runs real child processes via `fork()`. On macOS you *can*
run the L3 sim path, but fork + Python state can surface issues that don't
appear on Linux. When in doubt, run L3 examples on a Linux host.
