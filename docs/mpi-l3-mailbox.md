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
3. Rank 0 reads and verifies startup archives and broadcasts their bytes.
   Each rank installs and activates those artifacts, installs the dispatcher
   and local CHIP registry, runs its optional preparation hook, and initializes
   its own L3 `Worker` and command processor.
4. All ranks exchange status after each of those four startup stages. A failed
   stage prevents every rank from entering the following stage.
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

### Startup artifacts and node preparation

`MpiL3GroupSpec.startup_artifacts` accepts a tuple of
`simpler.startup_artifacts.StartupArtifactSpec` objects. Each descriptor names
a parent-local `.tar.gz` archive, its SHA-256 digest, and an import namespace.
Archives contain Python source under that namespace; for example, `entry.py`
inside an archive bound to `job_bundle` becomes `job_bundle.entry`.
Use `tarfile.USTAR_FORMAT` when producing archives. Only rank 0 opens the archive
paths. Other ranks receive verified bytes and select their own cache using
`SIMPLER_STARTUP_ARTIFACT_CACHE`, or `~/.cache/simpler/startup_artifacts` by default.

`startup_hooks_by_rank` accepts one optional
`simpler.startup_preparation.NodeStartupHook` per rank. Its default supplies no
hook on any rank. Hooks use ordinary `module:qualname` imports, including modules
from startup artifacts, and accept a JSON object of parameters containing only
finite values. Configuration snapshots those parameters defensively.

```python
import hashlib
from pathlib import Path

from simpler.startup_artifacts import StartupArtifactSpec
from simpler.startup_preparation import NodeStartupHook
from simpler.worker import MpiL3GroupSpec

archive = Path("generated/job_bundle.tar.gz")
artifact = StartupArtifactSpec(
    archive,
    hashlib.sha256(archive.read_bytes()).hexdigest(),
    "job_bundle",
)
spec = MpiL3GroupSpec(
    hosts=("node-a", "node-b"),
    platform="a2a3sim",
    device_ids_by_rank=((0,), (0,)),
    startup_artifacts=(artifact,),
    startup_hooks_by_rank=(
        NodeStartupHook("job_bundle.entry:prepare", {"role": "left"}),
        NodeStartupHook("job_bundle.entry:prepare", {"role": "right"}),
    ),
)
```

Each hook receives a `NodeStartupContext` with `rank`, `artifact_roots`
(artifact digest to local `Path`), `inner_worker`, `chip_handles` (hex digest to
the existing local `CallableHandle`), and its `parameters`. Hooks run on the main
dispatcher thread after local registration and before `inner_worker.init()`.
They can bind module state that the worker's children inherit when forked.
CHIP callables continue to travel in the existing inner registry; startup
artifacts introduce no second CHIP payload or metadata format.

A hook returns `None` or an object with `close()`. The session keeps that object
alive until the command processor has drained and the inner worker has closed,
then closes it, including during failed startup. Cleanup continues after an
individual cleanup error and preserves the original startup failure when the
inner worker's inventory proves that no live resources or accepted work remain.
An unconfirmed inner-worker close instead exits the rank with status 124 while
retaining hook state, just as an undrained command processor does.
The returned object may additionally implement `task_completed()`. The command
processor calls it once after each successful `inner_worker.run()` has drained
the complete task DAG, before sending a successful completion reply. It is not
called when execution fails. A callback error produces a failed task completion;
the hook object remains alive until worker close so uncertain state can be
retained safely. The callback receives no compiler-specific objects or arguments.

钩子返回对象还可实现 `task_completed()`。命令处理器在每次成功的
`inner_worker.run()` 完整排空任务 DAG 后、发送成功完成回复前调用一次。执行失败时
不调用；回调异常会生成失败的任务完成回复。钩子对象始终保留到内部 Worker 关闭后，
因此可以安全持有尚未确认完成的状态。该回调没有参数，也不依赖任何编译器对象。
若内部 Worker 关闭失败，只有在其资源清单能证明没有存活资源、已接收任务及活动操作时，
才继续释放钩子状态；否则 rank 保留钩子状态并以状态码 124 退出。

If the command processor remains alive after its bounded drain wait, the rank
exits with status 124 while retaining its worker and hook state until process
exit. Registry, hook, and initialization errors include the supplied artifact
digests alongside the failing rank and stage.

Manifest loading, artifact activation, registries, preparation, and inner
initialization share one local absolute deadline. Payload-read and broadcast time
consume that budget. Until rank 0 has read the JSON budget, the parent's launch
deadline remains the backstop. A stalled hook or collective causes a watchdog to
exit the rank with status 124 without calling MPI from the watchdog thread. The
MPI launcher and parent own whole-job teardown after that fail-stop exit; wedged
user code cannot receive cooperative cleanup. Ordinary stage errors use the status
exchange and ordered cleanup instead. Mailbox `READY` requires all stages to
succeed on every rank.

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
- reserved tail bytes (offsets 80 through 255), zero in version 1

Rank 0 parks on the request-state word between requests — a shared futex on
Linux, woken by every L4 publish — instead of polling it. Rank 0 copies the
complete request to private memory before publishing `TASK_ACCEPTED`. It
publishes `TASK_DONE` only after gathering every rank's status. Duplicate or
decreasing sequence ids make the group terminal.

### Versioning

Version 1 freezes the layout above. The creator zeroes the reserved header
bytes and both attach paths — the L4 channel and the rank-0 reopen — reject a
mailbox whose reserved bytes are non-zero, so a version 1 peer cannot silently
carry fields it does not understand. Any layout or semantic change bumps the
protocol version, and each side rejects a version it does not implement: a
mixed-version pair fails closed at attach instead of misreading the lane.

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
- Group-wide controls use one `GROUP` mailbox request. The caller marks them
  explicitly with the `FRAME_FLAG_GROUP_TARGET` frame flag; a control frame
  without the flag is always a directed rank request, whatever its control
  name.

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
| `CONTROL` / `CONTROL_REPLY` | `CONTROL`; directed unless the frame carries `FRAME_FLAG_GROUP_TARGET` |
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
| 14 | `ALLOC_DOMAIN` prepare/import/commit/abort | group request when the frame carries the group-target flag; otherwise directed rank |
| 15 | `RELEASE_DOMAIN` | group request when the frame carries the group-target flag; otherwise directed rank |
| 16 | `COPY_TO_DOMAIN` | directed rank |
| 17 | `COPY_FROM_DOMAIN` | directed rank |

Remote control number 18 is intentionally not assigned. The local hierarchical
protocol keeps number 18 for committed-device-memory control.

## Threading

The main dispatcher owns ordinary MPI command ordering. The command processor
runs on a rank-local thread over an in-memory, socket-shaped queue. Global
CommDomain requests cross back to the dispatcher through a queue and
`threading.Event`. The bounded domain exchange runs its blocking `allgather` on
a helper thread while the dispatcher waits. Successful exchanges serialize MPI
calls; a domain-exchange timeout uses the existing MPI job-abort path. The startup
watchdog makes no MPI calls and exits only its own rank on timeout. Startup
preparation introduces no additional MPI thread-level requirement. Hooks and worker
initialization remain on the main dispatcher thread.
