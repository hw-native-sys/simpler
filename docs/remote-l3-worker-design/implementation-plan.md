# Remote L3 Implementation Plan

Deliver remote L3 support in small PRs. Each step should keep existing local
fork/shm behavior working.

## PR Sequence

1. Endpoint interface and local adapter.
   - Add `WorkerEndpoint`.
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

2. Endpoint eligibility metadata.
   - Assign each NEXT_LEVEL child a stable `endpoint_id`.
   - Store `callable_id -> eligible endpoint ids` in parent runtime metadata.
   - Extend submit slots with final eligible endpoint sets computed as
     callable eligibility intersected with tensor/buffer data eligibility.
   - Teach Scheduler/WorkerManager to pick only eligible idle workers.
   - Validate `worker=` affinity against the slot's final eligible set.
   - Keep current docs in sync: describe local fork/shm as
     `LocalMailboxEndpoint` and remote L3 as a framed endpoint, not as another
     mailbox child loop.

3. Remote task sidecars and dependency keys.
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

4. Failed task poisoning.
   - Add per-member group state/outcome tracking so group failure can skip
     unstarted members while waiting for already-dispatched members to finish.
   - Add failed/poisoned slot handling.
   - Prevent downstream consumers of failed producers from dispatching.
   - Preserve `drain()` cleanup and first-error-wins reporting.

5. Versioned remote frame codec.
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

6. Remote callable registry.
   - Implement `RemoteCallable("module:qualname")`.
   - Preserve the public cid lifecycle from local dynamic Python registration:
     visibility only after register reply, unregister/cid reuse, stale-state
     cleanup, and TASK/control ordering.
   - Treat import-path callables as the baseline remote mode.
   - Support PR #839 serialized Python callable payloads only as a negotiated
     feature with serializer version, payload limit, Python ABI/runtime, and
     dependency/runtime-environment compatibility checks.
   - Implement `register_remote(..., workers=...)`.
   - Implement multi-endpoint all-or-nothing registration with prepare, commit,
     and abort controls. Keep the parent cid invisible until every selected
     endpoint commits, and mark uncertain endpoints failed rather than leaving a
     partially visible cid.
   - Implement unregister tombstones. Do not reuse a cid until every selected
     endpoint confirms cleanup or is removed from eligibility as failed.
   - Split cid handling into an outer remote-orch namespace and an inner L3
     Worker namespace; never assume a parent TASK cid is an inner chip/sub cid.
   - Make parent/session-assigned cid values the only cid source in each
     manifest namespace.
   - Define post-bootstrap namespace-aware prepare, commit, abort, and
     unregister frames for Python callables and `ChipCallable` entries.

7. Fork-safe simulation session runner.
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

8. Remote control-plane parity.
   - Map existing NEXT_LEVEL controls onto typed remote frames:
     prepare, register, unregister, comm init, domain alloc, and domain
     release.
   - Keep local mailbox sub-command ids local-only.
   - Add tests for post-bootstrap ChipCallable registration and dynamic domain
     allocation through the remote session runner.

9. Remote buffer registry.
   - Add `ALLOC_REMOTE_BUFFER`, `FREE_REMOTE_BUFFER`, `COPY_TO_REMOTE`, and
     `COPY_FROM_REMOTE`.
   - Track per-slot capture refs for explicit buffers and imported peer
     buffers.
   - Tie physical free and release-import to post-drain cleanup after all
     captured refs drop.

10. RoCE transport.
    - Implement connection setup, SEND/RECV frames, registered staging buffers,
      and timeout/error paths.
    - Add a hardware-gated smoke test with one remote L3 worker.

11. HCCS transport.
    - Implement the same transport contract through platform HCCS APIs.
    - Reuse the RoCE frame and buffer registry tests.

12. A5 UB transport.
    - Add UB export/import metadata.
    - Implement LD/ST doorbell and completion path with fences.
    - Keep RDMA fallback for bulk transfers.

13. Remote `allocate_domain()`.
    - Extend `CommDomainHandle` to carry remote endpoint ids.
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
| Remote sim error | Remote orch raises; parent raises with host/seq/cid. |
| Failed dependency | Consumer of failed remote producer is not dispatched. |
| Remote cid mapping | Daemon resolves non-zero parent-assigned remote cid. |
| Remote dep key | Shared remote buffer serializes through TensorMap. |
| Raw pointer rejection | Unstaged host pointer fails before slot commit. |
| Wire data zero | Non-zero remote TASK tensor data is rejected. |
| HOST_INLINE desc | Inline payloads require a descriptor and bounds checks. |
| Remote buffer copy | Host stages input, remote writes output, host pulls. |
| Input-only free deferral | Released input buffer survives queued consumers. |
| Timeout | Killed session runner produces bounded failure. |
| Dynamic Python register | Import-path callable register/unregister works. |
| Serialized Python gate | PR #839 payload works only after feature negotiation. |
| Callable visibility | TASK with cid works only after register reply. |
| Register partial failure | Multi-endpoint register is invisible after partial fail. |
| Unregister tombstone | Reused cid waits for cleanup or failed endpoint removal. |
| Command-lane order | TASK cannot overtake register/unregister control. |
| Health during long task | Health remains live while command lane runs a task. |
| Callable kind gate | Unsupported callable kind rejects without cid install. |
| Dynamic inner register | Inner Python and ChipCallable cids can run. |
| Remote cid namespace | Outer TASK cid cannot collide with inner cid. |
| Remote control parity | Register/unregister/domain reaches namespace. |

## Hardware-Gated Tests

- A2 RoCE single remote L3 task.
- A2 RoCE remote buffer copy round trip.
- A3 HCCS single remote L3 task.
- A5 UB LD/ST doorbell plus RDMA fallback.
- Remote domain allocation and deferred release across two remote L3 workers.

## Open Decisions

- Exact platform HAL names for HCCS and UB export/import.
- Authentication and isolation for remote daemon sessions.
- Exact compatibility metadata required for PR #839 serialized Python callable
  payloads beyond serializer version and Python ABI/runtime.
- How endpoint health feeds scheduler-level eligibility after the transport
  reports a failed health lane.
- How much of `CommContext` should remain shared with PTO-ISA once remote UB
  address metadata is added.

The first cut should land endpoint abstraction, endpoint eligibility,
remote callable registration, failure poisoning, and the simulation runner
before any hardware transport code.

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

- The first root failure message includes remote host, endpoint id,
  callable id, and sequence.
- Poisoned downstream slots should reference the root producer slot instead of
  overwriting the first-error-wins message.
