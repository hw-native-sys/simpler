# Remote L3 Implementation Record

Baseline document commit: `565fede20ba0f259c1db0ef8c089c9dcbc2141d6`.

This record tracks implementation against the PR-head remote L3 documents.
It is updated as each documented feature is completed and verified.

## Status

| Step | Documented feature | Status | Notes |
| ---- | ------------------ | ------ | ----- |
| 1 | Endpoint interface and local adapter | In progress | Local adapter and remote sim endpoint are implemented; HCOMM endpoint adapters remain. |
| 2 | Endpoint eligibility metadata | In progress | Callable endpoint sets are intersected with owner/imported remote sidecar eligibility. |
| 3 | Remote task sidecars and dependency keys | In progress | Public `TaskArgs.add_tensor(RemoteTensorRef(...))` API, remote TensorMap keys, and remote payload-sidecar rejection are implemented. |
| 4 | Failed task poisoning | In progress | Remote task-failure poisoning and session-exit endpoint failure are verified; explicit health-expiry-only coverage remains. |
| 5 | Versioned remote frame codec | In progress | TASK/COMPLETION/CONTROL_REPLY/HELLO/CONTROL/HEALTH exist; core fuzz/bounds coverage is present, with more exhaustive corpus testing still possible. |
| 6 | Remote callable registry | In progress | Dispatcher `PYTHON_IMPORT`, inner manifest/control `PYTHON_IMPORT`, and inner manifest/control inline `CHIP_CALLABLE` are implemented; serialized payloads and staged chip blobs remain negotiated extensions. |
| 7 | Fork-safe simulation session runner | In progress | Daemon/session bootstrap and HELLO READY barrier are implemented for sim transport. |
| 8 | Remote control-plane parity | In progress | Registry, alloc/free/copy, export/import/release-import controls are implemented for sim; Remote CommDomain controls are reserved/unsupported. |
| 9 | Remote buffer registry | In progress | Sim owner/imported buffers, TASK materialization, public memory API, opaque handles, slot/import-ref capture, and deferred free/release-import are implemented. |
| 10 | A2 RoCE HCOMM profile | Pending | Hardware-gated profile. |
| 11 | A3 HCCS HCOMM profile | Pending | Hardware-gated profile. |
| 12 | A5 UB HCOMM profile | Pending | Hardware-gated profile. |

## Completed Items

- Added a socket-backed `RemoteL3SocketTransport` and removed the fail-fast
  placeholder endpoint path from the Python init flow.
- Added `HELLO` payload encode/decode and made C++ endpoint registration wait
  for `HELLO READY` before the endpoint becomes schedulable.
- Added parent-side independent sim health-lane monitoring. The session runner
  emits session/endpoint-scoped `HEALTH` frames on a separate socket after
  prestart, and the C++ command-lane waits periodically check health failure
  without depending on command-lane progress.
- Added `simpler-remote-worker` and `simpler-remote-l3-session` entry points.
  The session runner reads the manifest first, constructs and prestarts the
  embedded L3 Worker, then opens command/health sockets.
- Changed the public sidecar API to
  `TaskArgs.add_tensor(RemoteTensorRef(...), tag)`.
- Added typed remote `CONTROL` frames for PYTHON_IMPORT dispatcher
  prepare/commit/abort/unregister, with commit-only TASK visibility in the
  session runner.
- Made register-family remote controls registry-scope-aware. The C++ endpoint
  and nanobind binding now carry `REMOTE_TASK_DISPATCHER` or
  `INNER_L3_WORKER` explicitly instead of hardcoding the dispatcher target.
- Added session-runner inner registry support for post-bootstrap
  `INNER_L3_WORKER` / `PYTHON_IMPORT`, including installation into
  `inner_worker = Worker(level=3)` and a session-local
  `get_inner_handle(hashid)` lookup for remote orchestration code.
- Added bootstrap-manifest inner registry installation for `PYTHON_IMPORT` and
  inline `CHIP_CALLABLE`. The runner rebuilds the same register commands used
  by post-bootstrap controls, validates hash/descriptor/platform/runtime
  before prestart, and publishes only session-local inner handles.
- Added v1 inline `CHIP_CALLABLE` payload decode/validation for
  `INNER_L3_WORKER`, including descriptor hash, executable blob SHA-256, and
  endpoint platform/runtime context checks. `REMOTE_TASK_DISPATCHER` rejects
  `CHIP_CALLABLE`, and remote `PYTHON_SERIALIZED` remains unsupported without
  negotiated features.
- Mapped `WorkerEndpoint::control_prepare(digest)` for remote endpoints to a
  typed `PREPARE_CALLABLE` control. The session runner accepts it only for a
  committed `REMOTE_TASK_DISPATCHER` / `PYTHON_IMPORT` digest.
- Added typed sim `ALLOC_REMOTE_BUFFER`, `FREE_REMOTE_BUFFER`,
  `COPY_TO_REMOTE`, and `COPY_FROM_REMOTE` handling in the session runner.
- Added `Worker.remote_malloc()`, `remote_free()`, `remote_copy_to()`, and
  `remote_copy_from()` public APIs. Remote handles are returned by the Worker
  API and no longer expose transport keys as public attributes.
- Fixed the nanobind `remote_malloc()` return path so the binding reacquires
  the GIL before constructing the Python tuple of handle fields.
- Added parent-side remote buffer slot-ref capture in the Python Orchestrator
  facade. `remote_free()` marks a live handle released and defers the physical
  `FREE_REMOTE_BUFFER` control until captured refs are released after
  `drain()`.
- Added session-side `RemoteTensorDesc` materialization before
  `inner_worker.run()`: `HOST_INLINE` descriptors become local ctypes-backed
  tensors and sim `REMOTE_DEVICE` descriptors resolve through the live session
  buffer registry.
- Removed the descriptor-dropping Python `task_args_from_wire()` helper so the
  session runner has a single registry-backed materialization path.
- Tightened parent/endpoint validation so remote task payload tensors require a
  `RemoteTensorRef` sidecar; bare host pointers and non-materialized remote
  payloads are rejected before remote dispatch.
- Added submit-time intersection between remote callable endpoint eligibility
  and remote tensor/buffer owner or importer eligibility.
- Documented the v1 `EXPORT_BUFFER`, `IMPORT_BUFFER`, and `RELEASE_IMPORT`
  wire schema, imported-handle identity, release deferral, and partial-import
  rollback contract.
- Implemented sim `EXPORT_BUFFER`, `IMPORT_BUFFER`, and `RELEASE_IMPORT`.
  Imports use shared-memory backed mappings in the session runner, imported
  handles remain opaque on the parent, and owner frees wait for live imports
  and slot refs to drain.
- Documented the v1 remote registry target/kind matrix, inner
  `INNER_L3_WORKER` visibility rules, remote `CHIP_CALLABLE` staged/inline
  payload contract, partial-register cleanup outcomes, and health-expiry
  scheduling behavior.

## Verification

- Python focused sidecar/callable tests:
  `tests/ut/py/test_task_interface.py tests/ut/py/test_callable_identity.py`
  passed with `145 passed`.
- Focused endpoint/data eligibility tests:
  remote sidecar owner filtering and non-owner C++ direct-submit rejection
  passed.
- Remote sim daemon/session noop TASK integration:
  noop TASK, prepare-callable control, error completion, post-init dynamic
  registration, unregister/reregister, health, buffer copy, failed dependency,
  session exit, input-free deferral, and `HOST_INLINE` integration passed with
  `11 passed` outside the network-restricted sandbox. The same tests skip
  inside the sandbox when local TCP sockets are denied.
- Remote sim long TASK health integration:
  a remote orch that keeps the command lane busy for 1 second completed while
  the independent health lane stayed live.
- Remote sim buffer copy integration:
  `remote_malloc` + `COPY_TO_REMOTE` + remote TASK materialization/write +
  `COPY_FROM_REMOTE` passed on the sim backend.
- Remote sim input-only free deferral:
  freeing an input buffer immediately after submit kept the remote allocation
  alive until the captured slot ref dropped after drain.
- Remote sim `HOST_INLINE` descriptor integration:
  inline TASK payload materialized into a session-local tensor and fed a
  remote output write.
- C++ wire fuzz/bounds coverage:
  bad frame version/type/flags, truncated control payload, and invalid remote
  descriptor inline fields are rejected.
- Remote sim failed-dependency integration:
  a failed remote producer poisoned a downstream same-buffer remote consumer,
  and the consumer did not dispatch or mutate the buffer.
- Remote sim endpoint-failure integration:
  a session runner exit during TASK returned a bounded endpoint failure to the
  parent instead of hanging `drain()`.
- Remote callable unregister/reregister integration:
  final unregister cleared the dispatcher entry, and a later same-descriptor
  registration became visible only after a new prepare/commit sequence.
- Remote `PREPARE_CALLABLE` integration:
  parent `control_prepare()` reached the session runner as a typed
  `PREPARE_CALLABLE` frame and validated the committed dispatcher digest.
- Remote inner Python import integration:
  parent sent `INNER_L3_WORKER` prepare/commit controls, the session runner
  installed the import-path callable into `inner_worker`, and a remote orch
  resolved the handle with `get_inner_handle()` and submitted a `SUB` task.
- Remote imported-buffer integration:
  owner endpoint export plus peer endpoint import allowed a remote TASK to run
  on the importer endpoint and mutate the owner's shared simulated buffer.
- C++ no-hardware ctest:
  `ctest --test-dir tests/ut/cpp/build-fetch -LE requires_hardware
  --output-on-failure` passed with `39/39` tests.
- Python unit tests:
  `tests/ut/py -m "not requires_hardware" --clone-protocol https` passed with
  `350 passed, 11 skipped, 10 deselected`.
- Editable build with `CCACHE_DISABLE=1`: passed.
- C++ focused UTs built and passed from `tests/ut/cpp/build-fetch`:
  `test_orchestrator`, `test_remote_wire`, `test_remote_endpoint`.
- ST smoke:
  `examples/a2a3/tensormap_and_ringbuffer/vector_example --platform a2a3sim`
  passed with `1 passed`.
- The default `tests/ut/cpp/build` still links against a system GoogleTest
  with a C++11 ABI mismatch; `tests/ut/cpp/build-fetch` uses the fetched
  GoogleTest build and passes the focused C++ UTs.
