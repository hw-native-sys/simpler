# worker_chip_orch_comm_stream — closed-loop host/L2 handshake over a shared region

The lower-level counterpart to [`worker_chip_message_queue`](../worker_chip_message_queue/).
Same idea — the host drives a task that is already running — but instead of a
queue with arenas and opcodes, the host and the L2 orchestration share a raw
**region**: one payload area plus a small counter area they signal through.

Three rounds of a strict ping-pong, then a stop:

```text
host: write input  ->  bump data_ready  ->  wait completion  ->  read output
L2:   wait data_ready  ->  transform  ->  bump completion
```

See [`docs/l3-l2-orch-comm.md`](../../../../docs/l3-l2-orch-comm.md) for the
mechanism.

## What this exercises

| Concept | How |
| ------- | --- |
| **Creating the region** | `orch.create_worker_chip_region(worker_id=0, payload_bytes=..., counter_bytes=...)`, described to L2 via `region.descriptor_scalars()`. |
| **Named counters at fixed offsets** | `region.counter(0)` is `data_ready`, `region.counter(64)` is `completion`. The offsets are passed to the kernel as scalars — both sides must agree. |
| **Notify / test / wait** | `counter.notify(seq, NotifyOp.Set)` publishes; `counter.test(seq, WaitCmp.GE)` polls without blocking and returns a snapshot; `counter.wait(seq, WaitCmp.GE, timeout=...)` blocks. The example calls `test` first and only falls back to `wait` when it did not already match — the fast path costs no block. |
| **Explicit payload transfer** | `region.payload_write(offset, tensor, nbytes)` and `payload_read(...)`, at offsets the host lays out itself: header at 0, input at 64, output after the input. |
| **A monotonic sequence as the protocol** | Round *n* sets `data_ready` to *n* and waits for `completion >= n`. `GE`, not `EQ`, so a host that fell behind still makes progress. |
| **In-band shutdown** | The stop is round `_ROUNDS + 1` with opcode 2 in the header — the same notify path as a data round, no separate channel. |
| **Passing a typed scalar** | `add_scalar(ctypes.c_float(7.0))` stores the float's bits. A ctypes scalar is read at its own width and zero-extended, so the encoding matches C++ `to_u64()`; a bare Python float narrows to single precision. |

## Run

Single device, all four platforms:

```bash
pytest examples/workers/l3/worker_chip_orch_comm_stream --platform a2a3sim
pytest examples/workers/l3/worker_chip_orch_comm_stream --platform a2a3 --device 0
```

The test file is also the example — `run_closed_loop_stream(platform,
device_id)` is importable directly.

`config.aicpu_thread_num = 2` is required: one AICPU thread polls the counter
while the other runs the transform kernel.

## Verification

After each round the host reads the output area and asserts every one of the
16,384 elements equals `input + 7.0` within `1e-5`. Because the check runs
**inside the loop**, a round that returned the previous round's data is caught
immediately — the inputs differ per round by construction (`round_idx * 1000 +
(i % 251)`), so a stale buffer cannot pass.

## File structure

```text
worker_chip_orch_comm_stream/
├── kernels/
│   ├── aiv/
│   │   └── kernel_worker_chip_transform.cpp
│   └── orchestration/
│       └── worker_chip_orch_comm_orch.cpp
├── test_worker_chip_orch_comm_stream.py
└── README.md
```
