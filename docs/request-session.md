# Streaming Request Sessions

`RequestSession` lets an initialized hierarchical `Worker` keep multiple
`Worker.run` calls in flight. It also provides a per-request output stream,
bounded admission, cancellation, and failure isolation.

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
- run thread 0 always uses arena bank 0, and run thread 1 uses bank 1;
- each run thread owns its Runtime storage and mutable run context;
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
parent scheduler -> task slot 0 -> run thread 0 -> arena bank 0 -> completion
                 -> task slot 1 -> run thread 1 -> arena bank 1 -> completion
```

The parent-side endpoint publishes a ready slot without waiting for Device S.
A completion pump reports the matching scheduler completion only after the
child marks that slot done. This is the non-blocking boundary; changing only
`submit_next_level` would leave the current synchronous endpoint bottleneck in
place.

The following legacy cross-request mechanisms remain only until the final
cleanup step:

- `HostRequestAdmissionClient` and `_HostRequestAdmissionService`;
- the prepared-request TaskArgs marker;
- `prepare_from_request_blob` and `execute_prepared`;
- the platform split prepare/execute C ABI;
- the prepared-request map, bank wait state, and HostGraph prepare gate.

HostApi thread context, scheduler completion latches, the HostGraph epoch/token
pipeline, arena banks, and concurrent top-level `Worker.run` batching remain.
The protocol and two-credit execution path are hardware validated. The final
cleanup removes the superseded admission path and split ABI.

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

`submit` returns immediately after admission. Request IDs are unsigned 64-bit
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

## Legacy Admission Path

The section below describes the compatibility path being removed by PR1379.
New request-session code submits a normal complete run and relies on the two
mailbox credits.

The later request prepares its HostGraph through the independent admission
lane before submitting its prepared Device task:

```python
emitter.prepare_host_request(
    worker_id=0,
    request_id=request_id,
    callable_handle=callable_handle,
    args=chip_args,
    config=config,
    arena_bank=1,
)
orch.submit_next_level(callable_handle, chip_args, config, worker=0)
```

`prepare_host_request` sends the callable identity, `CallConfig`, and
`TaskArgs` through an independent L3-to-L2 Host control lane. The L2 admission
thread binds the request and releases Host O construction without launching
Device S. The normal L3 task submission later selects that prepared request and
launches its Device S.

Each chip worker has two HostGraph arena banks. A steady request chain
alternates banks so `O_(n+1)` can overlap `S_n`. A bank remains busy through
the complete run and validation path; preparing a later request on the same
bank waits for that release. Host graph publication has a separate gate:
build, materialization, and upload may run early, while commit waits until the
device runner has installed the matching run state.

These constraints give the pipeline this shape:

```text
request A, bank 0:  Host O_A  | Device S_A --------------------|
request B, bank 1:             Host O_B | wait | Device S_B ---|
request C, bank 0:                         Host O_C | wait | S_C
```

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
| `python/simpler/worker.py` | Worker ownership and admission |
| `src/common/hierarchical/scope.cpp` | Per-run thread-local scope frames |
| `src/common/worker/chip_worker.cpp` | Prepared request state |
| `src/common/platform/onboard/host/c_api_shared.cpp` | Split C ABI |
| `src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp` | Host gates |
