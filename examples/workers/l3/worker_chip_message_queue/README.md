# worker_chip_message_queue — stream requests into a running L2 task

A different shape from every other L3 example. Elsewhere the host builds a DAG,
submits it, and waits. Here the host submits **one long-lived L2 task** and then
feeds it a stream of requests while it runs, reading results back as they
appear — a serving loop, not a batch.

The transport is the L3-L2 SPSC queue (`SPSQ` binding): two arenas (input and
output) plus a 10-scalar endpoint binding the L2 orchestration receives as
plain `TaskArgs` scalars starting at offset 0. See
[`docs/l3-l2-message-queue.md`](../../../../docs/l3-l2-message-queue.md) for
the channel's current design.

## What this exercises

| Concept | How |
| ------- | --- |
| **Creating the channel** | `orch.create_worker_chip_queue(worker_id=0, depth=8, input_arena_bytes=..., output_arena_bytes=...)`. |
| **Handing it to L2** | `queue.chip_task_arg_scalars()` packs the `SPSQ` binding into `TaskArgs` as exactly 10 scalars — the L2 side needs no other setup. |
| **Full-duplex, decoupled** | The host enqueues two requests, drains **three** responses, then enqueues two more. Requests and responses are not paired one-to-one. |
| **Zero-copy reads** | `queue.output.peek(timeout)` → `read_into(message, buf)` → `release(message)`. `release` is what returns arena space; skipping it stalls the producer once the queue fills. |
| **Cooperative shutdown** | `queue.request_stop(timeout)` lets the L2 side finish work already accepted. Responses queued before the stop are still drained afterwards. |
| **Application-level correlation** | The queue's `seq` is transport ordering only. This example carries its own 64-byte headers — `request_id` in, `(request_id, kind, aux)` out — because responses may arrive out of request order and one request may produce several. |

The mode field in each request drives the L2 kernel. Payload-header
`request_id` values are `0, 0, 0, 7`:

| Request | request_id | Mode | Produces |
| ------- | ---------- | ---- | -------- |
| 1 | 0 | 1 | two responses (kinds 10 and 11) |
| 2 | 0 | 2 | one response (kind 20) |
| 3 | 0 | 3 | combined with request 4 |
| 4 | 7 | 3 | one response combining both tiles (kind 30, `aux = 7`) |

Note the expected output order: request 2's kind-20 response comes back before
request 1's two. That is the point — a queue, not a call stack.

## Run

Single device:

```bash
pytest examples/workers/l3/worker_chip_message_queue --platform a2a3sim
pytest examples/workers/l3/worker_chip_message_queue --platform a5sim
pytest examples/workers/l3/worker_chip_message_queue --platform a2a3 --device <id>
pytest examples/workers/l3/worker_chip_message_queue --platform a5 --device <id>
```

The test file is also the example — `run_worker_chip_message_queue_example(platform,
device_id)` is importable directly.

`config.aicpu_thread_num = 2` is required: one AICPU thread drains the queue
while the other runs the transform kernel.

## Verification

Each of the four expected responses is matched **byte-for-byte** against a
host-packed golden — header and payload — and in the exact order given above.
An out-of-order or duplicated response fails, as does a payload whose header
metadata is right but whose tile is wrong.

## File structure

```text
worker_chip_message_queue/
├── kernels/
│   ├── aiv/
│   │   └── kernel_queue_transform.cpp
│   └── orchestration/
│       └── worker_chip_message_queue_orch.cpp
├── test_worker_chip_message_queue.py
└── README.md
```
