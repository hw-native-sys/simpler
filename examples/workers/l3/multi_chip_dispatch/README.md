# `multi_chip_dispatch/` — two chips, one orchestration, one SubWorker

Dispatches the same `vector_add` `ChipCallable` to two chips (each with its own
input tensors), then fires a Python `SubWorker` callable that depends on both
chip outputs. The smallest correct L3 program.

## What's new vs. L2 `vector_add`

| Concept | Where it shows up in `main.py` |
| ------- | ------------------------------ |
| Shared Buffers | `create_shared_tensor(worker, tensor)` returns a task argument and a shared Torch view |
| `TensorArgType` tags | `INPUT` / `OUTPUT_EXISTING` drive DAG dependency tracking |
| ChipCallable handle | `chip_handle = worker.register(chip_callable)` **before** `init()` |
| Python SubWorker | `sub_handle = worker.register(fn)` **before** `init()` |
| `Worker(level=3)` config | `device_ids=[0, 1]`, `num_sub_workers=1` |
| Orchestration | `orch.submit_next_level(chip_handle, ...)` per chip + `orch.submit_sub(sub_handle, args)` |

## Layout

```text
multi_chip_dispatch/
  main.py
  kernels/
    aiv/vector_add_kernel.cpp       # same kernel as l2/vector_add (copy)
    orchestration/vector_add_orch.cpp  # same orch as l2/vector_add (copy)
  README.md
```

Both kernel files are verbatim copies from `../../l2/vector_add/kernels/` —
kept in this directory so each example is independently readable. If you've
read through the L2 example the kernel / orchestration sources are already
familiar.

## The DAG

```text
                         ┌────────────────────┐
                         │  vector_add on     │
                         │  chip 0            │ ──┐
                         │  (a0, b0) → out0   │   │
                         └────────────────────┘   │
                                                  ▼
                                         ┌───────────────┐
                                         │  SubWorker    │
                                         │  (Python fn)  │
                                         └───────────────┘
                                                  ▲
                         ┌────────────────────┐   │
                         │  vector_add on     │   │
                         │  chip 1            │ ──┘
                         │  (a1, b1) → out1   │
                         └────────────────────┘
```

Three nodes, two edges. The sub waits until **both** chip tasks finish because
its `TaskArgs` includes both `host_out[0]` and `host_out[1]` tagged `INPUT`.
The scheduler infers this from the tags; you never list dependencies
explicitly.

## The lifecycle, step by step

### 1. Pre-init: prepare inputs and register callables

```python
host_a   = [torch.randn(...) for _ in device_ids]
host_b   = [torch.randn(...) for _ in device_ids]
host_out = [torch.zeros(...) for _ in device_ids]

def subworker(sub_args): ...
chip_handle = worker.register(chip_callable)   # ChipCallable: BEFORE init()
sub_handle  = worker.register(subworker)        # Python SubWorker: BEFORE init()
```

Ordinary Torch tensors hold inputs and golden values. After initialization,
explicit shared Buffers supply the chip arguments; children attach their
backing from descriptors rather than inheriting parent virtual addresses.

This example calls **`register()` before `init()`** for every callable — both
the `ChipCallable` dispatched to chips and the Python sub functions — so
startup can seed child registries and pre-warm chip callables before the first
DAG dispatch.

### 2. `init()`: child processes and the C++ scheduler

Worker allocates `MAILBOX_SIZE` shared-memory mailboxes (one per chip, one per
sub), starts child processes, and starts the C++ scheduler. After this call
returns, 4 processes are alive: parent, sub, chip 0, chip 1.

After initialization, copy inputs and outputs into Worker-owned shared Buffers:

```python
from simpler_setup.torch_interop import create_shared_tensor

shared = {
    "host_a": [create_shared_tensor(worker, t) for t in host_a],
    "host_b": [create_shared_tensor(worker, t) for t in host_b],
    "host_out": [create_shared_tensor(worker, t) for t in host_out],
}
```

Each entry is `(Tensor, Torch view)`. Tasks access the same backing while
the original Torch tensor remains independent.

### 3. Orchestration function — submit nodes

```python
def orch_fn(orch, _args, cfg):
    for i in range(len(device_ids)):
        chip_args = TaskArgs()
        chip_args.add_tensor(shared["host_a"][i][0],   TensorArgType.INPUT)
        chip_args.add_tensor(shared["host_b"][i][0],   TensorArgType.INPUT)
        chip_args.add_tensor(shared["host_out"][i][0], TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip_handle, chip_args, cfg, worker=i)

    sub_args = TaskArgs()
    for i in range(len(device_ids)):
        sub_args.add_tensor(shared["host_out"][i][0], TensorArgType.INPUT)
    orch.submit_sub(sub_handle, sub_args)
```

The orch function runs **in the parent process**. Each `orch.submit_*` adds a
node to the DAG; the scheduler (running on C++ threads) actually picks up the
nodes and dispatches them to the chip child processes. The orch function
returns immediately after submitting all nodes — `Worker.run` will then drain
the DAG before returning.

### 4. `Worker.run(orch_fn, ...)` — blocks until DAG drains

```python
worker.run(orch_fn, args=None, config=CallConfig())
```

After this returns, all tags on `host_out` are satisfied: chip tasks have
written, sub task has run.

### 5. Verify + close

Read outputs from the shared views. This example copies them into the
original Torch tensors for golden checks. Drop every shared view before close.

```python
for i in range(len(device_ids)):
    host_out[i].copy_(shared["host_out"][i][1])
    assert torch.allclose(host_out[i], expected[i], rtol=1e-5, atol=1e-5)
shared.clear()
worker.close()
```

## Run

```bash
python examples/workers/l3/multi_chip_dispatch/main.py -p a2a3sim -d 0-1
```

Expected output (sim, macOS / Linux):

```text
[multi_chip_dispatch] devices=[0, 1]
[multi_chip_dispatch] compiling kernels for a2a3sim...
[multi_chip_dispatch] init worker...
[multi_chip_dispatch] running DAG (2 chip tasks + 1 sub)...
[chip_process pid=... dev=0] ready
[chip_process pid=... dev=1] ready
[multi_chip_dispatch] subworker fired (received 2 tensor refs) ✅
[multi_chip_dispatch] chip 0: max |out - expected| = 0.000e+00
[multi_chip_dispatch] chip 1: max |out - expected| = 0.000e+00
[multi_chip_dispatch] all golden checks PASSED ✅
```

On real hardware (`-p a2a3 -d 0-1`) device memory is a separate address space
from host memory, but the runtime handles DMA transparently based on the
`TensorArgType` tags — the user-side code is identical to sim.

## Common pitfalls

- **Passing legacy fork arguments to onboard chips**: a new interpreter cannot
  use parent virtual addresses; create shared Buffers explicitly.
- **Forgetting `OUTPUT_EXISTING` on writable tensors** → runtime treats them
  as new allocations rather than writing into the provided buffer, and the
  sub task reads from a different tensor than the one the host expects.
- **Skipping `worker.close()`** → leaks chip child processes and shared
  memory segments. See
  [issue #604](https://github.com/hw-native-sys/simpler/issues/604) for the
  kind of downstream CI breakage this causes.

## Next step

For multi-host (L4+) composition, where L3 workers themselves become children
of a higher-level `Worker`, see the unit tests under
`tests/ut/py/test_worker/test_l4_recursive.py` until a dedicated L4 example
lands.
