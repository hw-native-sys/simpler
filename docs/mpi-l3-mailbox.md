# MPI L3 group mailbox protocol

An MPI L3 group has one L4-owned, named shared-memory mailbox. Only local MPI
rank 0 opens that mailbox. All ranks participate in the same ordered
`dispatch_comm` collectives, while Global CommDomain descriptor exchange uses
a separate `domain_comm`.

```text
L4 / MpiGroupMailboxEndpoint
          |
          | named SharedMemory (one request lane)
          v
local MPI rank 0 / L3
          |
          | dispatch_comm Bcast + Gather
          v
all MPI ranks / L3 -> each rank's local L2 workers
```

MPI groups never start or connect Simpler command/health TCP sockets. Ordinary
`RemoteWorkerSpec` workers still use `RemoteL3SocketTransport` and keep their
existing command and health lanes.

## Startup and shutdown

1. L4 creates the mailbox and writes its name, protocol version, size, and
   world size into the group manifest.
2. L4 starts `mpirun` as a new process group and monitors that direct child.
3. Each rank creates and initializes its own L3 `Worker`.
4. All ranks complete a readiness `allgather`.
5. Rank 0 reopens the mailbox by name and publishes `READY`. Other ranks never
   map it.
6. L4 attaches every stable MPI worker id to the same
   `MpiGroupMailboxChannel`; it does not create `RemoteL3Endpoint` sockets.
7. Shutdown is a mailbox `SHUTDOWN` request, followed by one MPI broadcast.
   Each rank closes its inner worker, communicators are freed, `mpirun` exits,
   and L4 unlinks the mailbox and manifest directory.

If startup, a collective, the mailbox, or `mpirun` fails, the group becomes
terminal. Runtime timeout also kills the complete `mpirun` process group.
There is no TCP fallback.

## Envelope and state

Protocol version 1 has a fixed 256-byte header and two 16 MiB payload regions.
The header contains:

- magic `SMPIBOX\0`
- protocol version and layout size
- MPI world size
- group state: `INITIALIZING`, `READY`, `TERMINAL`, or `CLOSED`
- request state: `IDLE`, `REQUEST_READY`, `TASK_ACCEPTED`, `TASK_DONE`,
  `TASK_FAILED`, `SHUTDOWN_READY`, or `SHUTDOWN_DONE`
- monotonic mailbox `sequence_id`
- opcode: `TASK`, `CONTROL`, `PING`, or `SHUTDOWN`
- target: `GROUP`, `RANK`, or `PER_RANK`
- target rank, payload count, and byte lengths

Rank 0 copies the complete request to private memory before publishing
`TASK_ACCEPTED`. It publishes `TASK_DONE` only after gathering every rank's
status. Duplicate or decreasing sequence ids make the group terminal.

Every gathered error contains `rank`, `error_type`, and `message`. Any target
rank failure fails the group operation. A broken command processor or
collective is terminal; an ordinary task/control application error is returned
to L4 and the communicator may be reused.

## Target and API semantics

- `orch.submit_next_level(..., worker=id)` remains a directed rank operation.
  Every MPI rank receives the envelope in collective order, but only the
  selected rank executes it.
- `orch.submit_next_level_group(args_list, workers=...)` remains one DAG node.
  When `workers` is the complete MPI group, C++ batches all members into one
  `PER_RANK` mailbox request. Rank `workers[i]` uses `args_list[i]`.
- A subset group remains supported as ordered directed requests. It is not
  silently widened to the complete MPI group.
- Group-wide controls use one `GROUP` mailbox request.

The existing remote task codec is reused. It serializes scalar values, tensor
metadata, inline host payloads, and `RemoteTensorRef` descriptors. Bare host or
child virtual addresses without a valid remote sidecar are rejected before
execution; a pointer value is never forwarded as if it were meaningful on
another rank. `PYTHON_SERIALIZED` callable payloads remain unsupported by the
underlying Remote L3 protocol; `PYTHON_IMPORT` and inline `CHIP_CALLABLE`
registration are supported.

## Remote protocol audit and MPI mapping

The wire `FrameType` values remain unchanged:

| Existing frame | MPI mailbox mapping |
| -------------- | ------------------- |
| `HELLO` / ready | rank-local initialization, readiness `allgather`, then rank 0 publishes mailbox `READY` |
| `TASK` | `TASK`; directed `RANK`, or one full-group `PER_RANK` vector |
| `CONTROL` / `CONTROL_REPLY` | `CONTROL`; directed except the group-wide controls below |
| `COMPLETION` | gathered per-rank status; selected/per-rank replies returned to L4 |
| `HEALTH` | `PING` to `GROUP`, gathered before success |
| `SHUTDOWN` | `SHUTDOWN` to `GROUP`, gathered before `SHUTDOWN_DONE` |

All existing remote controls use the mailbox path:

| Number | Control | MPI target |
| -----: | ------- | ---------- |
| 1 | `UNREGISTER_CALLABLE` | directed rank |
| 2 | `PREPARE_REGISTER_CALLABLE` | directed rank |
| 3 | `COMMIT_REGISTER_CALLABLE` | directed rank |
| 4 | `ABORT_REGISTER_CALLABLE` | directed rank |
| 5 | `PREPARE_CALLABLE` | directed rank |
| 6 | `ALLOC_REMOTE_BUFFER` | directed rank |
| 7 | `FREE_REMOTE_BUFFER` | directed rank |
| 8 | `COPY_TO_REMOTE` | directed rank |
| 9 | `COPY_FROM_REMOTE` | directed rank |
| 10 | `EXPORT_BUFFER` | directed rank |
| 11 | `IMPORT_BUFFER` | directed rank |
| 12 | `RELEASE_IMPORT` | directed rank |
| 13 | `COMM_INIT` | directed rank |
| 14 | `ALLOC_DOMAIN` prepare/import/commit/abort | one group request; descriptor work uses `domain_comm` |
| 15 | `RELEASE_DOMAIN` | one group request |
| 16 | `COPY_TO_DOMAIN` | directed rank |
| 17 | `COPY_FROM_DOMAIN` | directed rank |

Remote control number 18 is intentionally not assigned. The local hierarchical
protocol keeps number 18 for committed-device-memory control.

## Threading

Only the main dispatcher thread calls MPI. The existing command processor runs
on a rank-local thread over an in-memory, socket-shaped queue. Global
CommDomain operations cross back to the dispatcher through a queue and
`threading.Event`, so they use `domain_comm` on the MPI-owning thread. This
design does not require `MPI_THREAD_MULTIPLE`.
