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
| Build selection | `SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON` | default |

Read them side by side and the transport is the only variable — which is
exactly what you want when deciding which one a workload should use.

## Independent workspaces

URMA remains opt-in at build time through
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON`. The host runtime can include SDMA,
URMA and RDMA together; each domain explicitly selects its resources. This
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

## Build and test gates

| Gate | Effect |
| ---- | ------ |
| `CASES[*]["platforms"] = ["a5"]` | deselected on any other `--platform` |
| `CASES[*]["config"]["device_count"] = 2` | needs two dies |
| `@pytest.mark.skipif(not _urma_workspace_enabled())` | skipped unless `SIMPLER_ENABLE_PTO_URMA_WORKSPACE` is one of `1` / `ON` / `TRUE` / `YES` in the environment |
| `urma_deferred_completion_orch_fn()` raises | re-checks the env var immediately before allocating the communication domain |

Since the URMA CMake option defaults `OFF`, a stock build skips this test even on a5
hardware. **A green CI run says nothing about URMA.** Treat it as a manual
bring-up check, not as coverage.

## Run

```bash
# 1. rebuild the host runtime with the overlay on
SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON pip install --no-build-isolation -e .

# 2. run it, with the same variable visible to pytest (the skipif reads the env)
SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON \
  pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo \
  --platform a5 --device 0-1
```

The variable is needed twice for different reasons: `runtime_builder.py`
forwards it to CMake so the overlay is compiled in, and the test reads it from
the environment to decide whether to skip. Setting only one of the two gives
you either a skipped test or a a host-side unsupported-workspace error.

Wrap the hardware run in `task-submit` on a shared box.

## See also

[`../sdma_async_completion_demo/`](../sdma_async_completion_demo/) — the
default SDMA variant of the same protocol.
