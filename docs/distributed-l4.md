# Distributed L4 to L3 Dispatch

This package adds a Python-first remote L3 transport for L4 `Worker` instances.
It uses gRPC/protobuf for control messages and keeps the existing C++ scheduler
and mailbox layout unchanged.

## API

Start an L3 daemon:

```bash
python -m simpler.distributed.l3_daemon --port 5050 --num-sub-workers 1
```

Attach it to an L4 worker:

```python
from simpler.worker import Worker

w4 = Worker(level=4, num_sub_workers=0)
l3_sub_cid = w4.register(l3_sub)
l3_orch_cid = w4.register(l3_orch)
w4.add_remote_worker("127.0.0.1:5050")
w4.init()
```

`add_remote_worker()` allocates one local mailbox and registers it as a normal
next-level PROCESS worker. A Python shim thread polls that mailbox and forwards
ready tasks to the remote L3 daemon with `L3Worker.Dispatch`.

## Callable Catalog

The L4 side serializes registered callables with `cloudpickle` and pushes them
to the daemon during handshake. Callable ids are preserved, so an L3 orch can
submit L3 sub callables by the same ids that were registered on L4.

Callable payloads are trusted cluster traffic. Do not expose the catalog service
to untrusted clients.

## Daemon Lifecycle

The daemon starts a backend process before accepting gRPC traffic. gRPC handler
threads forward catalog and dispatch requests to that backend process. The
backend owns the inner `Worker`, so Worker child forks do not happen in a
process with active gRPC threads.

## Tensors

`tensor_pool.py` provides the planned inline/handle byte pool surface. Scalar
`TaskArgs` and `ContinuousTensor` metadata are wired through dispatch today;
full remote tensor materialization is isolated behind `TensorPool`.

## Health

Each `RemoteWorkerProxy` starts a heartbeat thread after handshake. Consecutive
heartbeat failures mark the remote unavailable, and later dispatches fail fast.
