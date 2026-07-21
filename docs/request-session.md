# Streaming Request Sessions

`RequestSession` lets an initialized hierarchical `Worker` keep multiple
`Worker.run` calls in flight. It also provides a per-request output stream,
bounded submission, cancellation, and failure isolation.

`max_active_runs` controls the number of dispatcher threads. Each dispatcher
enters a real `Worker.run` call with its own outer scope. Submission mutates the
shared orchestrator under one lock, while callbacks may wait for token messages
independently. Onboard a2a3 HostGraph chip workers expose two mailbox credits,
so two session dispatchers can execute complete runs concurrently. Other
endpoint types remain single-credit.

## PR1379 Multi-Flight Architecture

The bounded multi-flight mailbox has these invariants:

- one resident child message thread owns mailbox state transitions;
- two long-lived run threads execute complete HostGraph runs, each through an
  isolated `ChipWorker`/`DeviceRunner`;
- each run thread uses bank 0 of its private runner-owned arena;
- each run thread owns its Runtime storage and mutable Host run context;
- Host O may overlap an earlier Device S, while a process-level device gate
  serializes Device S because the resident AICPU runtime owns process-global
  executor, affinity, register, and diagnostic state;
- task completion is correlated by slot generation, so completion order may
  differ from submission order;
- control and shutdown operations are exclusive and wait for active task slots
  to drain;
- cancellation never interrupts active device work, but it suppresses output
  and releases the slot after completion;
- onboard HostGraph endpoints advertise two credits; TRB, simulation, SUB, and
  remote endpoints initially remain single-credit.

The target request path is:

```text
parent scheduler -> task slot 0 -> run thread 0 -> runner 0 arena -> completion
                 -> task slot 1 -> run thread 1 -> runner 1 arena -> completion
```

The parent-side endpoint publishes a ready slot without waiting for Device S.
A completion pump reports the matching scheduler completion only after the
child marks that slot done. This is the non-blocking boundary; changing only
`submit_next_level` would leave the current synchronous endpoint bottleneck in
place.

HostApi thread context, scheduler completion latches, the HostGraph epoch/token
pipeline, runner-owned arenas, and concurrent top-level `Worker.run` batching
remain.
There is no cross-request admission lane or split prepare/execute ABI: every
mailbox task is a self-contained run.

## API

Open one session on an initialized L3-or-higher Worker:

```python
def request_orch(orch, request, request_id, emitter, config):
    token = build_and_submit(orch, request, request_id, config)
    emitter.emit(token, final=token.is_final)

with worker.open_request_session(
    request_orch, max_pending=8, max_active_runs=2
) as session:
    stream_a = session.submit(request_a, config=config, request_id=100)
    stream_b = session.submit(request_b, config=config, request_id=101)

    forward_concurrently(stream_a, stream_b)
```

`submit` returns immediately after enqueueing. Request IDs are unsigned 64-bit
integers and must be unique among live requests in the session. The session
rejects a full pending queue with `RequestBackpressureError`; `max_pending`
bounds queued work and does not count requests already owned by dispatchers.

`RequestStream.next(timeout)` returns one item, raises `StopIteration` after a
successful terminal item, and re-raises the request's terminal exception on
failure. `RequestEmitter.emit()` does not return until `next()` has received
that item. This one-item handoff prevents L3 from accumulating several tokens
before returning them to the user. Consumers must drain the stream;
`wait(timeout)` alone cannot complete while a token is waiting for delivery.
A callback exception terminates its request stream, then the dispatcher
continues with later queued requests.

Only one request session may own a Worker. Its dispatcher threads are the only
threads allowed to call `Worker.run` until the session closes. Closing a
session stops new admission, drains every dispatcher, and releases ownership.

## Cancellation

`RequestStream.cancel()` is a soft cancellation:

- a queued request is skipped before `Worker.run`;
- active device work is drained to preserve arena and scope lifetimes;
- later output items are suppressed;
- consumers observe `RequestCancelledError`.

Cancellation does not interrupt an AICPU or AICore operation. Reusing its
buffers before Device S finishes would allow the next Host O to overwrite live
state, so active cancellation deliberately favors memory safety over immediate
device preemption.

Each run thread has a dedicated `ChipWorker`, `DeviceRunner`, runtime buffer,
and device arena. Both select bank 0 because HostGraph async threads do not
inherit the caller's thread-local bank selection. Host O can overlap an
earlier Device S without sharing mutable Host runtime state. Device S remains
bounded to one active launch per child process. Control operations are still
exclusive: create token queues and other L3-L2 resources before publishing the
first task when later requests will reuse them.

The HostGraph token trailer carries `request_id`, sequence, token value, final
state, and status. L3 decodes it into `HostGraphToken`. A depth-one L2-to-L3
token queue plus the acknowledged `emit()` handoff forms this flow-control
chain:

```text
L2 publish → L3 read → user next() → L3 release → L2 may publish next token
```

## Implementation

| Path | Role |
| ---- | ---- |
| `python/simpler/request_session.py` | Session and protocol |
| `python/simpler/worker.py` | Worker ownership and two-slot child execution |
| `src/common/hierarchical/scope.cpp` | Per-run thread-local scope frames |
| `src/common/hierarchical/worker_manager.cpp` | Credit and mailbox-slot protocol |
| `src/common/worker/chip_worker.cpp` | Per-run runtime ownership and arena selection |
| `src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp` | HostGraph epoch/token pipeline |
