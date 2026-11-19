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
| Domain selection | `workspace_engines=("urma",)` | `workspace_engines=("sdma",)` |
| Peer address | Peer MR base + input offset | Mapped peer window base + input offset |

Both demos use the same producer-consumer protocol. The URMA demo places its
input after a 256-byte prefix and leaves the peer window slot zero.

## URMA window addressing

The example selects `workspace_engines=("urma",)` on its two-rank Device
domain. Each rank's `windowsIn[rankId]` points to local HBM. The producer
checks that the peer slot is zero before issuing the transfer. It computes the
input offset from the local base and adds it to the peer MR base in the URMA
workspace. The reserved 256-byte prefix exercises this offset on both ranks.

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
