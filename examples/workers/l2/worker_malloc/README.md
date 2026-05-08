# worker_malloc — Worker memory primitives, no kernels

The smallest example that exercises every host<->device memory primitive on
the `Worker` API in isolation:

| Step | API | What it proves |
| ---- | --- | -------------- |
| 1 | `worker.malloc(n)` | Device allocator returns a non-null pointer. |
| 2 | `worker.copy_to(dev, host, n)` | H2D byte copy. |
| 3 | `worker.copy_from(host, dev, n)` | D2H byte copy. |
| 4 | `worker.free(ptr)` | Slab returns to the allocator (we churn 8x). |

There is **no `worker.run()` call** anywhere — that's deliberate. On real
hardware the CANN device context is per-thread, so `rtMalloc` only succeeds
on a thread previously bound by `rtSetDevice`. `Worker.init(...)` is the
only thing that performs that bind for the Python caller thread; if that
path is broken, `worker.malloc()` fails with CANN error 107002 *before*
any kernel ever runs. Every example that does `init() -> run() -> ...`
accidentally masks that bug because the run path re-binds the device on the
same thread just before allocations happen. This example doesn't.

## Run

```bash
python examples/workers/l2/worker_malloc/main.py -p a2a3sim -d 0   # simulator
python examples/workers/l2/worker_malloc/main.py -p a2a3   -d 0   # hardware
```

Same for `a5sim` / `a5`.

## What you should see

```text
[worker_malloc] init on a2a3 device=0 ...
[worker_malloc] single-buffer round trips:
[worker_malloc]     4096 bytes round-trip OK (ptr=0x...)
[worker_malloc]    65536 bytes round-trip OK (ptr=0x...)
[worker_malloc]    12345 bytes round-trip OK (ptr=0x...)
[worker_malloc] concurrent live allocations:
[worker_malloc]   3 concurrent buffers, all distinct, freed cleanly
[worker_malloc] alloc/free churn:
[worker_malloc]   8x alloc/free of 4096 bytes OK
[worker_malloc] close OK.
```

If you see `rtMalloc failed: 107002` on `a2a3` / `a5` (but the same example
passes on `a2a3sim` / `a5sim`), the per-thread `rtSetDevice` is not happening
during `Worker.init()` — see `simpler_init` in
`src/{arch}/platform/onboard/host/pto_runtime_c_api.cpp` and confirm it
forwards to `DeviceRunner::attach_current_thread`.
