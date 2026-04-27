# `multi_chip_dispatch/` — two chips, one orchestration, one SubWorker

Dispatches the same `vector_add` `ChipCallable` to two chips (each with its own
input tensors), then fires a Python `SubWorker` callable that depends on both
chip outputs. The smallest correct L3 program.

## What's new vs. L2 `vector_add`

| Concept | Where it shows up in `main.py` |
| ------- | ------------------------------ |
| Shared-memory tensors | `torch.randn(...).share_memory_()` — chip children see the same storage |
| `TensorArgType` tags | `INPUT` / `OUTPUT_EXISTING` drive DAG dependency tracking |
| Python SubWorker | `worker.register(fn)` **before** `init()` |
| `Worker(level=3)` config | `device_ids=[0, 1]`, `num_sub_workers=1` |
| Orchestration | `orch.submit_next_level(...)` per chip + `orch.submit_sub(cid, args)` |

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

### 1. Pre-init — shared-memory tensors + callable registration

```python
host_a   = [torch.randn(...).share_memory_() for _ in device_ids]
host_b   = [torch.randn(...).share_memory_() for _ in device_ids]
host_out = [torch.zeros(...).share_memory_() for _ in device_ids]

def subworker(sub_args): ...
sub_cid = worker.register(subworker)   # BEFORE init() — see below
```

`share_memory_()` moves the tensor's storage to a `mmap` region. After
`fork()`, the chip child process has that region mapped at the same virtual
address, so when the kernel writes to `host_out[i]`, the parent's tensor sees
it immediately. No explicit copy back.

**`register()` MUST come before `init()`**. `init()` forks child processes;
the registry is captured by copy-on-write. Anything registered after `init()`
is invisible to the forked children.

### 2. `init()` — fork + C++ scheduler

Worker allocates `MAILBOX_SIZE` shared-memory mailboxes (one per chip, one per
sub), forks child processes, and starts the C++ scheduler. After this call
returns, 4 processes are alive: parent, sub, chip 0, chip 1.

### 3. Orchestration function — submit nodes

```python
def orch_fn(orch, _args, cfg):
    for i in range(len(device_ids)):
        chip_args = TaskArgs()
        chip_args.add_tensor(make_tensor_arg(host_a[i]),   TensorArgType.INPUT)
        chip_args.add_tensor(make_tensor_arg(host_b[i]),   TensorArgType.INPUT)
        chip_args.add_tensor(make_tensor_arg(host_out[i]), TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip_callable, chip_args, cfg, worker=i)

    sub_args = TaskArgs()
    for i in range(len(device_ids)):
        sub_args.add_tensor(make_tensor_arg(host_out[i]), TensorArgType.INPUT)
    orch.submit_sub(sub_cid, sub_args)
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

Because `host_out` is shared memory, reading it in the parent gives the chip
output directly. No `copy_from` needed.

```python
for i in range(len(device_ids)):
    assert torch.allclose(host_out[i], expected[i], rtol=1e-5, atol=1e-5)
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

- **Registering after `init()`** → sub task fires but hits `KeyError` for the
  callable id. Always register first.
- **Not using `share_memory_()`** → chip child sees zeros where it expects
  inputs. `torch` tensors without `share_memory_()` live in each process's
  private heap.
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
