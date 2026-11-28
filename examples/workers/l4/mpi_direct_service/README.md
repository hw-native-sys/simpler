# MPI Direct P/D service ranks

The MPI Direct supervisor can launch external Prefill and Decode services.
Rank 0 coordinates the job, and each other rank owns one service process.
The service process owns its existing L3 initialization and device resources.

```bash
export SIMPLER_MPI_DIRECT_ROUTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
simpler-mpi-direct --service-config services.json --mpirun mpirun
```

The Router needs the same job token. It connects to rank 0 at
`127.0.0.1:<router_port>`. Install `mpi4py` on every rank host with an MPI
library matching the launcher and supporting `MPI_THREAD_MULTIPLE`.

```json
{
  "controller_host": "127.0.0.1",
  "router_port": 18421,
  "startup_timeout_s": 120,
  "session_timeout_s": 30,
  "services": [
    {"id": "p0", "role": "prefill", "host": "127.0.0.1", "argv": ["pypto-serving", "--pd-role", "prefill"]},
    {"id": "d0", "role": "decode", "host": "127.0.0.1", "bind_port": 18422, "argv": ["pypto-serving", "--pd-role", "decode"]}
  ]
}
```

Service IDs are unique and array order assigns ranks starting at 1. Keep
ranks on each host contiguous. In a multi-host job, use a `controller_host`
reachable by all ranks. Optional top-level `launcher_args` are passed to the
MPI launcher. Each service can supply `cwd` and string-valued `env` entries
for its target host. The supervisor sends each rank its startup record over
the existing MPI startup gate.

## Local control

For a Supervisor-owned deployment, supply each service's `http_url` and a
top-level `coordination` object. It accepts `prepare_timeout_s`,
`drain_timeout_s`, and a `prebuild` array with `prefill`, `decode`, and
optional `device_pairs` entries. A deployment process can start the Router
separately using the same token. The Supervisor launches the MPI job.

In this mode the service child receives `SIMPLER_MPI_DIRECT_COORDINATED=1`.
The ready payload contains the boot, registry digest, descriptor and
application-encoded registry. Rank 0 sends `prepare_pair` to Decode, then
to Prefill after the matching Decode result. The service replies with
`pair_result`, echoing the command ID and its registration identity, and
reports whether receiver preparation, domain binding or device connection
actually completed. Rank 0 sends `snapshot` with the current registered
instances and pair states to each authenticated Router connection.

SIGINT or SIGTERM to the Supervisor requests a cooperative shutdown.
Rank 0 publishes retired pairs and sends `drain` to the Router with a
deadline. The Router answers with `drain_result` and the remaining request
count. Rank 0 stops Prefill before Decode. If Prefill retirement is
unconfirmed, rank 0 reports the failure and withholds Decode teardown.
An independent Router can remain up and reconnect after losing rank 0.

Each child receives `SIMPLER_MPI_DIRECT_HOST`, `SIMPLER_MPI_DIRECT_PORT`,
`SIMPLER_MPI_DIRECT_TOKEN`, `SIMPLER_MPI_DIRECT_RANK`,
`SIMPLER_MPI_DIRECT_INSTANCE_ID`, and `SIMPLER_MPI_DIRECT_ROLE`. The host is
always `127.0.0.1`.

The Router and service processes each use one persistent control socket.
Frames are a four-byte unsigned big-endian length followed by a UTF-8 JSON
object, up to 8 MiB. The client sends `{"type":"hello","token":"..."}` and
receives `{"type":"hello","rank":0}` with the actual rank in the response.

- A child sends `{"type":"status","state":"ready","payload":{...}}` when its
  serving resources are ready. It can also report `draining` or `failed`.
  Rank 0 forwards status frames with the rank, instance ID, and role. The
  bridge reports `starting`, `disconnected`, and `exited` itself.
- The Router can send `{"type":"broadcast","id":"...","scope":"all","payload":{...}}`
  to ready services. Scopes are `all`, `prefill`, and `decode`. Rank 0
  replies after the MPI collective with
  `{"type":"response","id":"...","ok":true,"payload":{"ranks":[1,2]}}`.
  A service can publish a status if it needs to confirm application of the
  message.
- In the legacy Router-owned mode, after draining in-flight requests the Router sends
  `{"type":"shutdown","id":"..."}`. Children receive
  `{"type":"shutdown"}` and exit through their application teardown. Rank 0
  waits for Prefill to exit before signaling Decode. Each `exited` status
  carries `payload.returncode`. The Router gets the matching response
  after all service children exit.

## P/D peer streams

A Prefill child receives `SIMPLER_MPI_DIRECT_PEERS` as a JSON object mapping
Decode instance IDs to this Prefill host's loopback relay ports. It opens its
existing P/D session at `127.0.0.1:<relay_port>`. A Decode child listens at
`127.0.0.1:<SIMPLER_MPI_DIRECT_BIND_PORT>`, taken from its service record.

The two service ranks forward the TCP byte stream in both directions via
MPI point-to-point messages. The relay preserves application framing and
half-close semantics. The application continues to own request handling
and device data transfers.

Run the three-rank CPU smoke on a single host with an MPI-enabled Python:

```bash
python examples/workers/l4/mpi_direct_service/smoke.py --mpirun /path/to/mpirun
```
