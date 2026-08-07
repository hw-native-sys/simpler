# sdma_async_completion_demo — deferred completion over SDMA

Two ranks, one transfer, one dependency:

```text
producer:  TGET_ASYNC the peer rank's input from the HCCL window into local
           `out`, then register the PTO AsyncEvent via defer_pto_async_event
consumer:  depends on the producer's output, writes result = out + 1
```

Checking `out` and `result` tests two separate things: `out` proves SDMA
completion polling saw the transfer land, `result` proves the deferred-release
dependency held the consumer until it had. A consumer that ran early would
still produce a plausible `result` from a partially written `out`, which is why
both are checked.

The remote address is plain symmetric-window arithmetic — take the local
pointer's offset from `windowsIn[rankId]` and add it to `windowsIn[peer_rank]`.
Every rank's window is laid out identically, so an offset is rank-independent.

## Requirements

The a5 host runtime includes the async-SDMA workspace by default:

| Gate | Effect |
| ---- | ------ |
| `CASES[*]["platforms"] = ["a5"]` | deselected on any other `--platform` |
| `CASES[*]["config"]["device_count"] = 2` | needs two dies |
| `@pytest.mark.skipif(_urma_workspace_enabled())` | skipped when `SIMPLER_ENABLE_PTO_URMA_WORKSPACE` selects the URMA backend |

URMA replaces SDMA in a URMA build. Rebuild without
`SIMPLER_ENABLE_PTO_URMA_WORKSPACE` before running this demo.

```bash
pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo \
  --platform a5 --device 0-1
```

Wrap the hardware run in `task-submit` on a shared box.

## Compare with

- [`../urma_deferred_completion_demo/`](../urma_deferred_completion_demo/) — the same protocol over URMA. `kernel_consumer.cpp` is byte-identical; only the transfer kernel, its completion header, and the build flag differ. **The two overlays are mutually exclusive in one build**, so comparing them means rebuilding — that README has the detail.
- [`examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/`](../../../a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/) — the a2a3 port of this demo, which needs no overlay flag.
