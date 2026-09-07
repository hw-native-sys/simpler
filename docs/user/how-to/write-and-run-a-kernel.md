# How-to: write and run a kernel

Two ways to get a kernel onto a device. Pick by what you are doing:

| You want | Use | Why |
| -------- | --- | --- |
| A test or example that CI will run | **`@scene_test`** | Compilation, arg building, golden comparison and per-platform cases are handled for you |
| To see or control every stage yourself | **The `Worker` API directly** | Nothing is hidden; you own compile, malloc, copy, run |

Both need the same three source pieces:

```text
my_example/
  kernels/orchestration/my_orch.cpp   # runs on AICPU (or host) — submits tasks
  kernels/aiv/my_kernel.cpp           # runs on an AICore vector unit
  test_my_example.py                  # or main.py for the direct-API style
```

The orchestration source is what builds the task graph; the in-core sources are
the per-task kernels it submits. See
[AICore Kernel Programming](../../aicore-kernel-programming.md) for what may go
inside a kernel body.

## Option A — `@scene_test` (recommended)

Declare the sources and the per-case matrix; the framework compiles and runs
them. Copy
[`examples/a2a3/tensormap_and_ringbuffer/vector_example/`](../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/)
as your starting point.

```python
from simpler.task_interface import ArgDirection as D
from simpler_setup import SceneTestCase, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMyExample(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/my_orch.cpp",
            "function_name": "aicpu_orchestration_entry",   # must match the exported symbol
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,                                # matches rt_submit_aiv_task(0, ...)
                "source": "kernels/aiv/my_kernel.cpp",
                "core_type": "aiv",                          # "aiv" or "aic"
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],                # sim needs no hardware
            "params": {},
        },
    ]

    def generate_args(self, params):
        ...        # build the input tensors

    def compute_golden(self, args, params):
        ...        # return the expected output; the framework compares for you
```

Two things to get right, because both fail confusingly:

- **`function_name` must match the symbol** your orchestration `.cpp` exports
  with `__attribute__((visibility("default")))`.
- **`func_id` must match the id** the orchestration passes to its submit call.
  `func_id=0` corresponds to the first `rt_submit_aiv_task(0, ...)`.

Run it:

```bash
# simulation — no device needed
pytest examples/my_example --platform a2a3sim

# hardware
pytest examples/my_example --platform a2a3 --device 4-7
```

Standalone (same file, no pytest):

```bash
python examples/my_example/test_my_example.py -p a2a3sim
```

`level` and `runtime` are decorator arguments; **platforms are declared per case
in `CASES`**, not on the decorator. Ordinary cases should omit `"config"` and
use the architecture's automatic AICPU thread count. Add `aicpu_thread_num`
only when a test specifically depends on a thread-count or scheduler-topology
behavior. `runtime_env` holds TRB ring-sizing overrides; HBG also reads
`ring_task_window[0]` to size its graph task table. See the [Python API reference](../reference/python-api.md)
for the full `CallConfig` field list.

## Option B — the `Worker` API directly

Use the runnable
[`examples/workers/l2/vector_add/main.py`](../../../examples/workers/l2/vector_add/main.py)
example to follow compilation, callable construction, registration, device
allocation, copies, execution, and golden comparison in one place:

```bash
python examples/workers/l2/vector_add/main.py -p a2a3sim -d 0
```

The same implementation is exercised by
[`test_vector_add.py`](../../../examples/workers/l2/vector_add/test_vector_add.py).
Its cases are manual; run the simulator case with:

```bash
pytest examples/workers/l2/vector_add/test_vector_add.py --platform a2a3sim --manual include
```

The example uses contiguous CPU torch tensors of shape `(128, 128)` and dtype
`torch.float32`. Device allocations return `Buffer` handles. For partial copies,
pass `nbytes`, `src_offset`, and `dst_offset` by keyword.

Lifecycle and argument rules:

- **Register before use.** Registration before `init()` enters the startup
  snapshot. Registration after `init()` installs and prewarms the callable
  before returning; at L3+ it also publishes to eligible live children. The
  worker topology must be established before `init()`.
- **`close()` belongs in a `finally`.** Skipping it leaves the device held, and
  the next job on that device hangs.
- **Task-arg order must match the callable `signature`**, positionally.

`run()` blocks and returns `None`. For a non-blocking submit use `submit()`,
which returns a handle with `done()` / `wait(timeout)` / `result(timeout)`.

## After it runs

- Timing: [How-to: profile a kernel](profile-a-kernel.md)
- A failure or a hang: [How-to: debug a failed run](debug-a-failed-run.md)
- Multiple chips: [How-to: run on multiple chips](run-on-multiple-chips.md)
