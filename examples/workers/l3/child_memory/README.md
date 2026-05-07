# child_memory — load weight once, run kernel many times

Demonstrates the **`child_memory=True`** opt-out from the runtime's auto
malloc/free for tensors whose lifetime you manage yourself. The idiom is
"upload weights once, fire many kernels against them" — the LLM serving
pattern in miniature.

## What this exercises

| Step | API | What's new vs `multi_chip_dispatch` |
| ---- | --- | ----------------------------------- |
| 1 | `orch.malloc(worker_id=0, nbytes)` | Control-plane device-side `malloc` forwarded to chip child via mailbox IPC. |
| 2 | `orch.copy_to(worker_id=0, dev, host, n)` | Same, for H2D byte copy. |
| 3 | `ContinuousTensor.make(dev_ptr, ..., child_memory=True)` | Flag the runtime to **skip auto-malloc and skip auto-free** for this tensor — the buffer survives task teardown. |
| 4 | `orch.submit_next_level(...)` × 2 | Both kernel invocations share `w_dev`. The second proves the buffer was **not** freed after the first. |

There is no `orch.free` — the buffer is reclaimed by `DeviceRunner::finalize`
when `worker.close()` runs. That asymmetry is the whole point of
`child_memory`: you opt out of the runtime's bookkeeping.

## Run

```bash
python examples/workers/l3/child_memory/main.py -p a2a3sim -d 0   # simulator
python examples/workers/l3/child_memory/main.py -p a2a3   -d 0   # hardware
```

Only `a2a3` / `a2a3sim` are supported — the example reuses the kernel
sources from `examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels`,
which are a2a3-only.

## Verification

The example asserts two things:

1. **Both outputs match a torch golden** — `f = (a + w + 1) * (a + w + 2) + (a + w)`.
2. **`f1 == f2` byte-for-byte** — if `child_memory` were ignored and the
   weight buffer freed after the first task, the second task would read
   garbage; this cross-task check is what catches the regression.
