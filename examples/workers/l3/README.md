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
#    Returns an integer id you pass to orchestrator.submit_sub(...) later.
postprocess_id = worker.register(
    lambda args: print("post-process received", args)
)

worker.init()                 # forks chip child processes + sub children,
                              # then starts the C++ scheduler

def my_orch(orch, args, cfg):
    # orch is the Orchestrator. Submit one task per chip + any sub work.
    # orch.submit_next_level(...) schedules a ChipCallable onto a free chip.
    # orch.submit_sub(cid, sub_args) schedules a Python callable.
    ...

try:
    worker.run(my_orch, my_args, my_config)
finally:
    worker.close()            # shuts down child processes and releases shm
```

Two things to know before reading the example:

1. **`register()` and `add_worker()` MUST run before `init()`**. The Python
   callables get captured via copy-on-write when `init()` forks child
   processes, so anything registered after the fork is invisible to the
   children.
2. **The orchestration function is a *plain Python function*, not a C++
   kernel.** It runs in the host process and calls `orch.submit_*(...)` to
   hand work to chip children. The children get the submitted `ChipCallable`
   through shared-memory mailboxes.

## What each example demonstrates

| Directory | New concept |
| --------- | ----------- |
| [`multi_chip_dispatch/`](multi_chip_dispatch/) | Two chips + one SubWorker. An orchestration fn dispatches a `ChipCallable` to each chip, then submits a Python callable to collect/verify results. |

## Prerequisites

Same as L2 (see [`../l2/README.md`](../l2/README.md)): venv + `pip install .`.

Additionally, L3 runs real child processes via `fork()`. On macOS you *can*
run the L3 sim path, but fork + Python state can surface issues that don't
appear on Linux. When in doubt, run L3 examples on a Linux host.
