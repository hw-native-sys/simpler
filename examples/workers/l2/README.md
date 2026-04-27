# L2 — Single-chip examples

**L2 = CHIP**: one NPU device, managed by one on-device AICPU scheduler plus
AICore/AIVector workers. This is the smallest self-contained unit of the
runtime; everything larger (L3 multi-chip DAG, L4 multi-host) is built on top
of L2 primitives.

For the full hierarchy model see
[`docs/hierarchical_level_runtime.md`](../../../docs/hierarchical_level_runtime.md).

## Minimum Worker lifecycle

Every L2 program looks like this:

```python
from simpler.worker import Worker

worker = Worker(
    level=2,
    platform="a2a3sim",                    # or a2a3 / a5sim / a5
    runtime="tensormap_and_ringbuffer",    # default user-facing runtime
    device_id=0,
)
worker.init()             # load host.so + aicpu.so + aicore.o, set device
try:
    # ... allocate device buffers, build ChipCallable, run ...
    worker.run(chip_callable, task_args, call_config)
finally:
    worker.close()        # release ACL resources and device
```

The `try/finally` is important — if anything between `init()` and `close()`
raises, you still want the device released. The
[L2 conftest leak issue](https://github.com/hw-native-sys/simpler/issues/604)
is a reminder that a stale `ChipWorker` on a self-hosted runner can poison
the next job.

## What each example demonstrates

| Directory | New concept | Runnable on |
| --------- | ----------- | ----------- |
| [`hello_worker/`](hello_worker/) | `Worker.init()` / `close()` contract, venv + build prereqs. No kernels. | `a2a3sim`, `a5sim` |
| [`vector_add/`](vector_add/) | Compile one AIV kernel → `ChipCallable`. Build `TaskArgs` with host→device buffer copy, run, copy back, compare against numpy. | `a2a3sim`, `a2a3` |

Both examples use the same `main.py` shape:

```python
def main() -> int:
    args = parse_args()
    worker = build_worker(args)
    worker.init()
    try:
        do_the_work(worker, args)
    finally:
        worker.close()
    return 0
```

## When to reach for scene_test instead

If your answer to *any* of these is "yes", `@scene_test` is the right tool:

- I want my example to run in CI as part of `pytest examples tests/st ...`
- I want automatic golden comparison across multiple parametrized cases
- I need the kernel compiled once and cached across cases
- I want `--rounds N` for benchmarking

`Worker` API examples here are for learning the plumbing — not for shipping
tests.
