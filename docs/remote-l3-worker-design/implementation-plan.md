# Remote L3 Implementation Plan

Deliver remote L3 support in small PRs. Each step should keep existing local
fork/shm behavior working.

Status for the local PR #866 cut:

- Steps 1-5 are implemented for the simulation transport, including an
  independent health lane.
- Step 6 is implemented for dispatcher `PYTHON_IMPORT`, inner manifest/control
  `PYTHON_IMPORT`, and inner manifest/control inline `CHIP_CALLABLE`.
  `PYTHON_SERIALIZED` remains a negotiated future extension, and
  staged chip-callable blobs require a staged-blob adapter before use.
- Steps 7-9 are implemented for the two-process simulation runner and sim
  remote buffers, including export/import/release-import.
- Steps 10-12 remain pending hardware-profile work. Remote CommDomain controls
  remain reserved/unsupported in this cut.

## PR Sequence

1. Endpoint interface and local adapter. **Implemented.**
   - Add `WorkerEndpoint`.
   - Include read-only `WorkerEndpoint::caps()` capability metadata for
     endpoint eligibility. `caps()` returns logical features only and must not
     expose HCOMM/RDMA/socket handles to Scheduler or Orchestrator code.
   - Define `WorkerEndpoint::run()` to return an explicit outcome: success,
     task failure, or endpoint failure.
   - Move current mailbox code into `LocalMailboxEndpoint`.
   - Teach `LocalMailboxEndpoint` to map `MAILBOX_OFF_ERROR == 0` to success
     and non-zero child mailbox errors to task failure.
   - Treat local dispatch exceptions, child crash detection, and timeout paths
     as endpoint failure once those paths are available.
   - Keep `WorkerManager::add_next_level(void *mailbox)` working by wrapping
     the mailbox in a local endpoint.
   - Thread the endpoint outcome through the WorkerThread completion callback
     without yet changing all downstream DAG poisoning behavior.
   - Add local adapter regression tests for existing L3/L4 examples.

2. Worker eligibility metadata. **Implemented.**
   - Assign each NEXT_LEVEL child a stable `worker_id`.
   - Store `callable hashid -> eligible worker ids` in parent runtime
     metadata.
   - Extend submit slots with final eligible worker-id sets computed as
     callable eligibility intersected with tensor/buffer data eligibility.
   - Teach Scheduler/WorkerManager to pick only eligible idle workers.
   - Validate `worker=` affinity against the slot's final eligible set.
   - Keep current docs in sync: describe local fork/shm as
     `LocalMailboxEndpoint` and remote L3 as a framed endpoint, not as another
     mailbox child loop.

3. Remote task sidecars and dependency keys. **Implemented for explicit
   sidecars, owner/imported simulation buffers, and slot-ref capture.**
   - Add `RemoteBufferHandle`, `RemoteTensorRef`, and `RemoteTensorDesc`.
   - Store a `RemoteTaskArgsView` sidecar beside existing `TaskArgs`.
   - Extend `TensorKey` for remote endpoint, buffer id, generation, logical
     start offset, and address kind.
   - Teach `Orchestrator::infer_deps()` to use remote logical keys while
     preserving existing local keys.
   - Reject remote sidecars on local fork/shm endpoints unless an explicit
     import/staging API has converted them into local-addressable tensors.
   - Reject unstaged raw host pointers before a remote slot is committed.
   - Reject remote `OUTPUT` tensors with `data == 0` unless an explicit remote
     allocation API has already produced a `RemoteTensorRef` sidecar.

4. Failed task poisoning. **Implemented.**
   - Add per-member group state/outcome tracking so group failure can skip
     unstarted members while waiting for already-dispatched members to finish.
   - Add failed/poisoned slot handling.
   - Prevent downstream consumers of failed producers from dispatching.
   - Preserve `drain()` cleanup and first-error-wins reporting.

5. Versioned remote frame codec. **Implemented for frame/TASK/COMPLETION/
   CONTROL/CONTROL_REPLY, ordered command-lane tests, and simulation health
   lane.**
   - Add `remote_wire.h/.cpp`.
   - Implement canonical little-endian encode/decode for `CallConfigWire`,
     `ContinuousTensorWire`, frame headers, descriptors, counts, strings, and
     enum values.
   - Implement the `HOST_INLINE` inline byte arena with descriptor
     `inline_payload_offset` / `inline_payload_len` validation.
   - Keep local mailbox `write_blob` / `read_blob` local-only; remote codec must
     not memcpy C++ POD structs as its wire format.
   - Implement encode/decode bounds checks for all frame types.
   - Define and test `CONTROL_REPLY` encode/decode for command success,
     command failure, result payloads, and sequence matching.
   - Define and test the per-endpoint ordered command lane for TASK, CONTROL,
     CONTROL_REPLY, COMPLETION, and SHUTDOWN frames.
   - Define and test an independent `HEALTH` lane or transport keepalive so
     liveness is not queued behind long-running TASK execution.
   - Include tests for corrupt lengths, tensor counts, sequence mismatch, and
     bounded error payloads.
   - Include tests that reject unknown enum values, non-zero reserved fields,
     and truncated multi-byte fields.
   - Include tests that reject non-zero `ContinuousTensorWire.data` in remote
     TASK frames.

6. Remote callable registry. **Implemented for dispatcher `PYTHON_IMPORT`,
   inner manifest/control `PYTHON_IMPORT`, and inner manifest/control inline
   `CHIP_CALLABLE` on the simulation path. Serialized Python remote payloads
   and staged chip blobs remain negotiated extensions.**
   - Implement `RemoteCallable("module:qualname")`.
   - Preserve the public callable identity lifecycle from PR #891:
     `Worker.register()` returns `CallableHandle`, TASK/control frames carry
     the handle hash digest, endpoint-private execution slots stay hidden,
     visibility starts only after register reply, unregister performs
     stale-state cleanup, and TASK/control ordering prevents partially visible
     hashids.
   - Treat import-path callables as the baseline remote mode.
   - Add `PYTHON_IMPORT` as the remote baseline callable kind and
     `REMOTE_TASK_DISPATCHER` as the parent-facing resolver scope for
     `RemoteCallable` handles.
   - Support serialized Python callable payloads only as a negotiated feature
     with serializer version, payload limit, Python ABI/runtime, and
     dependency/runtime-environment compatibility checks. Follow the payload
     contract in `docs/python-callable-serialization.md`.
   - Extend the existing public `register(callable, workers=...)` flow so
     `RemoteCallable` descriptors can target remote endpoints. Do not add a
     separate public remote-only registration API.
   - Require the first implementation to reject `RemoteCallable` registration
     without an explicit non-empty `workers=[...]` list. Leave named remote
     pools and placement policies as future API work, and do not define
     implicit broadcast to all remote endpoints as a contract.
   - Implement multi-endpoint all-or-nothing registration with prepare, commit,
     and abort controls. Keep the hashid invisible until every selected
     endpoint commits, and mark uncertain endpoints failed rather than leaving a
     partially visible hashid.
   - Implement final-unregister tombstones per worker/hashid pair. Do not
     block dispatch through other live handles that still reference the same
     hashid. If failed-register rollback cleanup is uncertain, block only the
     affected worker/hashid pair for the current session.
   - Keep `INNER_L3_WORKER` as a remote-internal install target, not a
     parent-facing handle scope. Never assume a parent TASK hashid names an
     inner chip/sub callable.
   - Make canonical descriptors and their SHA-256 hashids the only public
     callable identity source in each manifest registry scope; target-private
     slots are allocated by the receiving endpoint.
   - Define post-bootstrap registry-scope-aware prepare, commit, abort, and
     unregister frames for Python callables and `ChipCallable` entries.
   - Implement the protocol v1 target/kind matrix:
     dispatcher `PYTHON_IMPORT`, optional negotiated dispatcher
     `PYTHON_SERIALIZED`, inner `PYTHON_IMPORT`, optional negotiated inner
     `PYTHON_SERIALIZED`, and inner `CHIP_CALLABLE`.
   - Reject `CHIP_CALLABLE` for `REMOTE_TASK_DISPATCHER` and reject
     `PYTHON_SERIALIZED` on any remote target unless the session feature set
     explicitly negotiated serialized support.
   - Implement `CHIP_CALLABLE` remote prepare with either bounded inline blob
     bytes or a session-local staged blob token. Validate the PR #891
     descriptor hash, executable blob SHA-256, target platform/runtime, and
     signature hash before commit.
   - On inner registry commit, install into `inner_worker = Worker(level=3)`;
     do not add the hashid to the parent-facing dispatcher registry.
   - Release staged callable bytes after confirmed commit or abort cleanup.

7. Fork-safe simulation session runner. **Implemented for the socket-backed
   simulation transport.**
   - Add `simpler-remote-worker` control entry point.
   - Add per-session `simpler-remote-l3-session` runner.
   - Pass the validated bootstrap manifest from daemon to runner through an
     inherited fd, manifest path in env, or single-threaded pipe before any
     runner transport threads start.
   - Add an explicit runner prestart step equivalent to `inner_worker.init()`
     plus `_start_hierarchical()`: fork L3 chip/sub children, register local
     endpoints, and start the inner Scheduler before any remote transport or
     health threads are started.
   - Start the sim transport only after the local L3 child tree is established,
     then run the post-prestart `HELLO`/ready handshake.
   - Treat `HELLO ready_state=READY` as a scheduling barrier; the parent must
     not schedule an endpoint that is alive but not prestarted.
   - Run TASK frames over the sim transport and return completions.
   - Add localhost two-process integration tests.

8. Remote control-plane parity. **Implemented for the simulation transport;
   Remote CommDomain controls remain reserved/unsupported.**
   - Map existing NEXT_LEVEL controls onto typed remote frames:
     prepare, register, unregister, remote buffer allocation, remote buffer
     free, copy to remote, copy from remote, export buffer, import buffer, and
     release import.
   - Implement versioned request/result codecs for every remote buffer control.
     `EXPORT_BUFFER`, `IMPORT_BUFFER`, and `RELEASE_IMPORT` must use the v1
     schemas in `protocol.md`; do not invent backend-specific ad hoc payloads.
   - Keep `EXPORT_BUFFER` owner-scoped, `IMPORT_BUFFER` importer-scoped, and
     `RELEASE_IMPORT` importer-scoped. The parent-visible dependency identity
     for imported handles remains the original owner buffer identity.
   - Implement rollback for partial import setup: if any importer fails, send
     `RELEASE_IMPORT` to importers that succeeded and mark uncertain
     endpoint/buffer pairs ineligible for the current session.
   - Reserve remote `COMM_INIT`, `ALLOC_DOMAIN`, and `RELEASE_DOMAIN` opcodes
     for the later Remote CommDomain phase; the first task-dispatch cut rejects
     them as unsupported controls.
   - Keep local mailbox sub-command ids local-only.
   - Add tests for post-bootstrap ChipCallable registration and remote buffer
     controls through the session runner, including export/import/release.

9. Remote buffer registry. **Implemented for simulation owner/imported
   buffers.**
   - Add `ALLOC_REMOTE_BUFFER`, `FREE_REMOTE_BUFFER`, `COPY_TO_REMOTE`, and
     `COPY_FROM_REMOTE`.
   - Add owner export records and importer-local mapping records for
     `EXPORT_BUFFER`, `IMPORT_BUFFER`, and `RELEASE_IMPORT`.
   - De-duplicate live imports by
     `(importer_worker_id, owner_worker_id, buffer_id, generation, offset,
     nbytes, access_flags, transport_profile)` when the transport profile
     supports reuse; otherwise return distinct `import_id` values and ref-count
     each mapping independently.
   - Track per-slot capture refs for explicit buffers and imported peer
     buffers.
   - Tie owner physical free to post-drain cleanup after all captured refs and
     importer mappings drop.
   - Tie release-import to post-drain cleanup after all captured refs on the
     imported handle drop.

10. A2 RoCE HCOMM profile.
    - Implement HCOMM endpoint/channel setup with `COMM_PROTOCOL_ROCE`, HCOMM
      RPC command rings, registered staging buffers, and timeout/error paths.
    - Keep bootstrap sockets limited to session setup and HCOMM bring-up.
      After HCOMM RPC is ready, task metadata, controls, completions, and
      shutdown use the HCOMM RPC adapter.
    - Add a hardware-gated smoke test with one remote L3 worker.

11. A3 HCCS HCOMM profile.
    - Implement the same HCOMM adapter contract with the HCCS protocol profile.
    - Reuse the A2 frame, HCOMM RPC, and buffer registry tests.

12. A5 UB HCOMM profile.
    - Add UB export/import metadata through the HCOMM adapter.
    - Implement LD/ST doorbell and completion paths only when the selected
      profile proves the mapping and fence rules are valid.
    - Keep RDMA/HCOMM fallback for bulk transfers.

13. Remote `allocate_domain()` future work.
    - Extend `CommDomainHandle` to carry remote worker ids.
    - Allocate/import windows collectively across remote workers.
    - Preserve deferred release after `drain()`.

## Required Tests Before Hardware Backends

| Test | Expected result |
| ---- | --------------- |
| Local adapter regression | Existing L3/L4 fork/shm behavior unchanged. |
| Endpoint eligibility | Scheduler never picks an ineligible endpoint. |
| Frame fuzz/bounds | Corrupt lengths and counts are rejected. |
| Remote sim hello | Parent bootstraps remote L3 and shuts down cleanly. |
| Manifest handoff | Runner reads manifest before transport starts. |
| Prestart barrier | HELLO READY only after inner L3 scheduler is started. |
| Remote sim task | L4 parent dispatches one L3 orch task successfully. |
| Remote sim error | Remote orch raises; parent raises with host/seq/hashid. |
| Failed dependency | Consumer of failed remote producer is not dispatched. |
| Remote hashid mapping | Daemon resolves an outer remote-orch hashid. |
| Remote dep key | Shared remote buffer serializes through TensorMap. |
| Remote import eligibility | Imported peer handle makes only the importer worker eligible. |
| Remote import dep key | Owner and imported views use the same owner-based TensorMap key. |
| Raw pointer rejection | Unstaged host pointer fails before slot commit. |
| Wire data zero | Non-zero remote TASK tensor data is rejected. |
| HOST_INLINE desc | Inline payloads require a descriptor and bounds checks. |
| Remote buffer copy | Host stages input, remote writes output, host pulls. |
| Input-only free deferral | Released input buffer survives queued consumers. |
| Import release deferral | Released import survives queued consumers, then tears down. |
| Owner free with import | Owner free waits for live imports and captured refs. |
| Import partial fail | Successful imports are released or marked uncertain/ineligible. |
| Timeout | Killed session runner produces bounded failure. |
| Health expiry idle | Expired idle endpoint is removed from schedulable set. |
| Health expiry in-flight | Expiry fabricates failed reply/completion for in-flight work. |
| Dynamic Python register | Import-path callable register/unregister works. |
| Serialized gate | Unnegotiated serialized payload rejects without hashid install. |
| Callable visibility | TASK with hashid works only after register reply. |
| Prepare partial fail | Multi-endpoint register stays invisible and aborts prepared peers. |
| Commit partial fail | Committed/uncertain endpoint hashids are cleaned or blocked. |
| Abort cleanup fail | Only affected worker/hashid pair becomes cleanup-uncertain. |
| Unregister tombstone | Final unregister keeps tombstone until endpoint cleanup. |
| Command-lane order | TASK cannot overtake register/unregister control. |
| Health during task | Health remains live while command lane runs a task. |
| Callable kind gate | Unsupported kind rejects without hashid install. |
| Dynamic inner Python register | Inner Python hashid can run only inside inner Worker. |
| Dynamic inner chip register | Inner ChipCallable hashid runs through inner chip workers. |
| Remote registry scope | Parent TASK resolves only in dispatcher. |
| Inner registry target | Inner Worker install is not parent-facing. |
| Remote control parity | Register/unregister/domain reaches target registry. |

## Hardware-Gated Tests

- A2 RoCE single remote L3 task.
- A2 RoCE remote buffer copy round trip.
- A3 HCCS single remote L3 task.
- A5 UB LD/ST doorbell plus RDMA fallback.
- Remote domain allocation and deferred release across two remote L3 workers
  after Remote CommDomain enters scope.

## Open Decisions

- Exact platform HAL names for HCCS and UB export/import.
- Authentication and isolation for remote daemon sessions.
- Exact remote compatibility metadata required for serialized Python callable
  payloads beyond the local payload contract, serializer version, and Python
  ABI/runtime.
- How much of `CommContext` should remain shared with PTO-ISA once remote UB
  address metadata is added, when Remote CommDomain enters scope.

The current cut lands endpoint abstraction, endpoint eligibility, remote
callable identity, remote sidecars, frame codec, failure poisoning, the
fork-safe simulation runner, and sim buffer import/export. Hardware HCOMM
profiles and Remote CommDomain controls remain future work.

## Failure Poisoning Contract

Worker failures must finish DAG bookkeeping without pretending the producer
succeeded. The contract applies to remote completions, endpoint failures, and
local mailbox child errors reported by `LocalMailboxEndpoint`.

State model:

```text
FREE -> PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
                                \-> FAILED    -> CONSUMED
```

Rules:

- Worker completion carries `success`, `task_failure`, or `endpoint_failure`.
- `success` transitions `RUNNING -> COMPLETED`.
- `task_failure` or `endpoint_failure` transitions `RUNNING -> FAILED`.
- `LocalMailboxEndpoint` converts a non-zero child mailbox error into
  `task_failure`; first-error-wins controls only which root error message is
  retained for `drain()`.
- A `FAILED` producer releases fanout bookkeeping, but consumers are marked
  `FAILED` instead of `READY`.
- Failed consumers are never dispatched.
- `FAILED -> CONSUMED` runs the same cleanup hooks as `COMPLETED -> CONSUMED`:
  TensorMap erase, ring release, remote buffer ref release, and deferred free
  scheduling.
- `drain()` waits for all successful, failed, and skipped slots to become
  `CONSUMED`, then rethrows the first root failure.

Group task rules:

`TaskSlotState` stores per-member execution state for group slots:

```text
GroupMemberState:
  NOT_DISPATCHED
  RUNNING
  SUCCESS
  FAILED
  SKIPPED

GroupMemberOutcome:
  success
  task_failure
  endpoint_failure
  skipped
```

Additional group bookkeeping:

- `member_states[group_size]` and `member_outcomes[group_size]`;
- `group_terminal_count`: members in `SUCCESS`, `FAILED`, or `SKIPPED`;
- `group_dispatched_count`: members that reached `RUNNING`;
- `group_failed`: set when any member reports `task_failure` or
  `endpoint_failure`;
- `group_first_failure_index`: first failed member used for root-error context.

Rules:

- A group slot is successful only if every member reaches `SUCCESS`.
- On dispatch of member `i`, transition
  `NOT_DISPATCHED -> RUNNING` before handing work to the endpoint.
- On successful completion of member `i`, transition `RUNNING -> SUCCESS` and
  increment `group_terminal_count`.
- On failed completion of member `i`, transition `RUNNING -> FAILED`, set
  `group_failed`, record `group_first_failure_index` if unset, and increment
  `group_terminal_count`.
- When `group_failed` becomes true, every member still in `NOT_DISPATCHED`
  transitions to `SKIPPED` and increments `group_terminal_count`. Skipped
  members are never dispatched.
- Members already in `RUNNING` are allowed to complete so their endpoint state
  and buffer refs can be cleaned up.
- The group slot reaches terminal outcome only when
  `group_terminal_count == group_size`.
- If the terminal group has any `FAILED` member, the slot outcome is `FAILED`;
  otherwise it is `COMPLETED`.
- Slot cleanup runs once for the whole group after the group slot reaches its
  terminal outcome.

Error reporting:

- The first root failure message includes remote host, worker id,
  callable hashid, and sequence.
- Poisoned downstream slots should reference the root producer slot instead of
  overwriting the first-error-wins message.
