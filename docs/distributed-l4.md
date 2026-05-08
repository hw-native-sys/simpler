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

`tensor_pool.py` now provides the Python MVP data-plane bridge:

- tensors up to 4KB are sent inline in `DispatchReq.tensor_refs`;
- larger tensors are staged into the remote L3 backend `TensorPool` with
  `AllocTensor` + streaming `PushTensor`, then dispatched by handle;
- the L3 backend materializes `TensorRef` values into shared mmap-backed
  `ContinuousTensor` buffers before calling the inner `Worker`;
- `OUTPUT`, `INOUT`, and `OUTPUT_EXISTING` tensors are returned in
  `DispatchResp.output_tensors` and copied back into the original L4 buffers;
- tensor dispatches use a per-dispatch inner L3 worker so sub/chip children are
  forked after the mmap exists and inherit the same tensor storage.

This is still not the production zero-copy path. `remote_addr` and `rkey` are
kept in the protocol shape for future SHM/RDMA/Urma backends, while the current
MVP uses Python byte buffers and gRPC streaming for inter-host data movement.
Persistent L3 worker reuse for tensor dispatches still needs a later mailbox or
transport path that can inject new shared mappings into already-forked children.

### Optional HCOMM Backend

The tensor pool now has a narrow transport backend boundary. The default remains
`grpc`, preserving the existing byte-pool + gRPC streaming behavior. An
experimental `hcomm` backend can be selected explicitly:

```bash
SIMPLER_TENSOR_TRANSPORT=hcomm \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=192.168.0.243 \
SIMPLER_HCOMM_CHANNEL_ROLE=client \
SIMPLER_HCOMM_CHANNEL_PORT=60001 \
python -m simpler.distributed.l3_daemon --port 5050 --tensor-transport hcomm
```

`SIMPLER_HCOMM_ENDPOINT_HANDLE` can still be supplied directly. If it is not
set, the backend can create an endpoint from `SIMPLER_HCOMM_ENDPOINT_IP` and the
optional `SIMPLER_HCOMM_ENDPOINT_*` location fields. `--tensor-transport auto`
tries HCOMM only when the required resources are present; otherwise it falls
back to `grpc`.

In this first version, the L3 `TensorPool` registers large tensor buffers with
HCOMM and publishes `TensorHandle.transport = "hcomm"` plus the exported memory
descriptor. The L4 side imports that descriptor when an endpoint handle is
available, then uses `HcommWriteWithNotifyNbi` + `HcommChannelFence` for input
tensor push. The L4 client stages source bytes through a HCOMM-registered host
buffer before issuing the write, so the local source address is covered by a
registered memory region. A pre-created `SIMPLER_HCOMM_CHANNEL_HANDLE` is still
accepted, but the facade can now also call the latest public `HcommChannelCreate`
ABI directly. For CPU RoCE channels, `SIMPLER_HCOMM_CHANNEL_ROLE` selects
`client` or `server`, and `SIMPLER_HCOMM_CHANNEL_PORT` selects the listen/connect
port. A socket handle may still be supplied for environments that pre-create one,
but it is no longer required by the Python facade.

Output tensor writeback still uses the existing gRPC `PullTensor` path, because
the current public CPU RoCE path does not yet provide the full
readback/remote-writeback protocol needed by the L4 output semantics.

Real HCOMM smoke tests are opt-in so normal CI does not require HCOMM hardware:

```bash
SIMPLER_HCOMM_REAL_TEST=1 \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=127.0.0.1 \
python -m pytest tests/ut/py/test_distributed/test_transport_backend.py -q
```

The pre-created channel write smoke additionally requires
`SIMPLER_HCOMM_ENDPOINT_HANDLE`, `SIMPLER_HCOMM_CHANNEL_HANDLE`,
`SIMPLER_HCOMM_REMOTE_ADDR`, and `SIMPLER_HCOMM_REMOTE_NBYTES`.

The single-node HCOMM E2E smoke creates a server/client channel pair through the
latest channel descriptor shape and writes bytes over CPU RoCE. It is also
opt-in because it needs a built HCOMM shared library and a working RoCE/RXE
device:

```bash
SIMPLER_HCOMM_E2E_REAL_TEST=1 \
SIMPLER_HCOMM_LIB=/path/to/libhcomm.so \
SIMPLER_HCOMM_ENDPOINT_IP=192.168.0.243 \
SIMPLER_HCOMM_CHANNEL_PORT=60001 \
python -m pytest tests/ut/py/test_distributed/test_hcomm_e2e_real.py -q
```

Local Soft-RoCE/RXE can be used as a lower-level real-machine smoke test for
the ibverbs data path. This validates that the host has an active RXE device and
that RC queue pairs can exchange data over the selected GID; it does not prove
that HCOMM channel creation is available, because HCOMM still needs its own
endpoint/channel resources and shared library.

```bash
SIMPLER_RXE_REAL_TEST=1 \
SIMPLER_RXE_DEVICE=rxe0 \
SIMPLER_RXE_GID_INDEX=1 \
SIMPLER_RXE_SERVER_IP=192.168.0.243 \
python -m pytest tests/ut/py/test_distributed/test_rxe_real.py -q
```

If `SIMPLER_RXE_GID_INDEX` and `SIMPLER_RXE_SERVER_IP` are omitted, the test
tries to infer the first IPv4-mapped GID from `/sys/class/infiniband/<device>`.
On the current development host, `rxe0` GID index `1` maps to
`::ffff:192.168.0.243` and the RC pingpong smoke passes.

## Health

Each `RemoteWorkerProxy` starts a heartbeat thread after handshake. Consecutive
heartbeat failures mark the remote unavailable, and later dispatches fail fast.
