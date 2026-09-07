# How-to: run on multiple chips

One host driving several chips is **L3**. The single-chip API you already know
still applies; L3 adds a host-side orchestration function that submits one task
per chip, plus optional Python "sub worker" callables for post-processing.

Worked examples live in
[`examples/workers/l3/`](../../../examples/workers/l3/README.md), whose README
enumerates what each one demonstrates.

## The lifecycle

```python
from simpler.worker import Worker

worker = Worker(
    level=3,
    platform="a2a3sim",
    runtime="tensormap_and_ringbuffer",
    device_ids=[0, 1],        # one chip child process per entry
    num_sub_workers=1,        # optional host-side Python callables
)

# These registrations enter the children's startup snapshot.
chip_handle = worker.register(chip_callable)
postprocess = worker.register(lambda args: print("post-process got", args))

worker.init()                 # forks the chip children and sub children, starts the scheduler

def my_orch(orch, args, cfg):
    # Plain Python, running in the host process — not a C++ kernel.
    orch.submit_next_level(chip_handle, chip_args, worker=0)   # target a specific chip
    orch.submit_sub(postprocess, sub_args)                     # schedule host-side work

try:
    worker.run(my_orch, args, config)
finally:
    worker.close()            # shuts the children down and releases shared memory
```

Three differences from L2 that trip people up:

- **`device_ids` (plural), not `device_id`.** Each entry becomes its own forked
  child process holding its own chip.
- **The orchestration function is Python and runs on the host.** At L2 the
  orchestration is a compiled `.cpp`; at L3 the top-level orchestration is a
  Python callable that hands compiled `ChipCallable`s down to the children.
- **Register before use.** `init()` is the fork point. Earlier registrations
  enter the startup snapshot; later registrations are installed through the
  control plane on eligible live children before `register()` returns. Set up
  the worker topology before `init()`.

To direct a task at one specific child rather than any free one, see
[Directed NEXT_LEVEL Scheduling](../../directed-next-level-scheduling.md).

## Collectives across chips

Cross-chip collectives run through a **communication domain**: a symmetric
memory window that every participating rank can address. Allocate it inside the
orchestration function, as a context manager, and index it by *domain-local*
rank:

```python
def orch_fn(orch, _args, cfg):
    with orch.allocate_domain(
        name="default",
        workers=list(range(nranks)),
        window_size=window_size,
        buffers=[CommBufferSpec(name="scratch", dtype="float32",
                                count=elems, nbytes=scratch_nbytes)],
    ) as handle:
        for i in range(nranks):
            domain = handle[i]          # domain-local rank i
            ...                         # build per-rank args, submit per chip
```

A worker can belong to more than one domain, and each domain gets its own window
slice. Full semantics — allocation, lifetime, the rank map, and what happens
when you ask for a domain a worker is not in — are in
[Communication Domains](../../comm-domain.md).

The shipped collectives are hand-written AIV kernels that compute peer pointers
from the window, not HCCL collective calls; HCCL is used only to bootstrap the
domain. The algorithm corpus (allreduce in several modes, allgather,
reduce_scatter, broadcast, all_to_all) lives in
`tests/st/worker/collectives/` — read those scene tests for the real
implementations.

## Scope limits worth knowing up front

- **Collectives are single-host, multi-card.** Both backends state this
  explicitly; there is no cross-node collective path today.
- **Same-host L3 supports `--enable-chip-swimlane`.** Each chip child writes
  under `rankN/dN`; cross-rank merging requires detail level `4` (the bare
  flag's default). Automatic merging across NETWORK1/L4 is rejected because
  those captures need a node namespace. See [Chip Swimlane Profiling](../../dfx/chip-swimlane-profiling.md).
- **Multi-host (L4) remote L3 uses the shipped host_tcp data plane.** You
  can register a remote worker, the remote side builds a genuine L3 subtree,
  and remote buffers move through the host TCP session. The RoCE / HCCS / UB
  peer-memory profiles are documented contracts, not working paths. See [Remote L3 Worker Design](../../remote-l3-worker-design.md)
  and the [Capability Survey](../../capability-survey.md) before designing
  against it.

## Running

```bash
# simulation, two virtual chips
python examples/workers/l3/multi_chip_dispatch/main.py -p a2a3sim -d 0,1

# hardware
pytest tests/st/worker/collectives --platform a2a3 --device 4-7
```

On shared machines, hardware runs should hold the devices they use rather than
racing other users for them; see [Testing](../../testing.md) for how the suites
are invoked.
