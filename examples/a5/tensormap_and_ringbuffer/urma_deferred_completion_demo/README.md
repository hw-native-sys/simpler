# urma_deferred_completion_demo — the same protocol, over URMA

The a5-only twin of
[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/). Both run the
identical two-rank protocol; they differ in which transport moves the bytes and
which completion path reports it done.

```text
producer:  TGET_ASYNC the peer's input from the window into local `out`,
           register the AsyncEvent through the deferred-completion path
consumer:  depends on that producer output, writes result = out + 1
```

Checking both `out` and `result` is what makes it a test of two things at once:
`out` proves completion polling saw the transfer land, and `result` proves the
deferred-release dependency held the consumer back until it had.

## What actually differs from the SDMA demo

`kernels/aiv/kernel_consumer.cpp` is **byte-identical** between the two
directories. Only the transfer kernel and the orchestration change:

| What | URMA | SDMA |
| ---- | ---- | ---- |
| Transfer kernel | `kernel_urma_tget_async.cpp` | `kernel_sdma_tget_async.cpp` |
| Completion header | `backend/urma/urma_completion_kernel.h` | `backend/sdma/sdma_completion_kernel.h` |
| Build selection | default A5 provider | default A5 provider |

Read them side by side and the transport is the only variable — which is
exactly what you want when deciding which one a workload should use.

## Independent workspaces

SDMA and URMA providers are always compiled on A5. The host runtime can include
optional RDMA alongside them; each domain explicitly selects its resources. This
example requests `workspace_engines=("urma",)` and reads
`comm_workspace<CommWorkspaceKind::URMA>(*ctx).addr`.

`TestSdmaUrmaWorkspaceMix` requests both SDMA and URMA on one domain and runs
both callables on the same Worker. A shared result output orders URMA after
SDMA on each rank, and separate SDMA and URMA outputs validate both transfers.
This does not assert unordered simultaneous submission on shared engine queues.

`TestSdmaUrmaSameTask` uses one producer kernel to fetch the first half of its
output through SDMA and the second half through URMA. Both requests register
completion conditions in the same task. The consumer reads the entire output
and adds one, so both transfers must finish before its dependency is released.
The two ranks post the engines in opposite orders. This covers outstanding
requests on independent engines; it does not cover multiple producers posting
to one engine queue.

## Host validation and testing

These cases run on A5 TMR with two dies. Domain allocation requests the required
engines and rejects failed initialization or invalid URMA workspace descriptors.
Before passing the communication context to a task, orchestration calls
`handle.device_context(rank, workspace_engines=("urma",))`; mixed tasks require
both `("sdma", "urma")`. Missing engines and released domains raise on the host.
The cases run in standard A5 onboard CI without a URMA environment selection.

## Run

Use an A5 host and a project-local venv:

```bash
source .venv/bin/activate
pip install --no-build-isolation -e .
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --timeout 1800 --max-time 1800 \
  --run ".venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo --platform a5 --device \$TASK_DEVICE"
```

## See also

[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/) — the
default SDMA variant of the same protocol.
