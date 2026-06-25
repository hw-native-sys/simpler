# Remote L3 PR Split and Audit Artifacts

This file records the pre-split audit artifacts for the remote L3 worker stack.
It follows `pr-split-and-audit-plan.md` and stops before child branch creation.

## Phase 0 Baseline

| Item | Value |
| ---- | ----- |
| Date | 2026-06-08 |
| Working branch | `remote-l3-worker-design` |
| Feature HEAD | `4c1f4b53` (`CI: retrigger PR checks fifth time`) |
| Upstream tracking branch | `puddingfjz/remote-l3-worker-design` |
| Main baseline | `origin/main` at `ca29019d` |
| Merge base | `ca29019d64493e958c07647bb9cbd32571348588` |
| PR snapshot ref observed locally | `origin/pr/866` at `565fede2` |
| Document baseline | PR-head documents on `remote-l3-worker-design` before split |

### Drop-Only Commits

These commits are empty CI retriggers and should not appear in any child PR:

| Commit | Subject | Evidence |
| ------ | ------- | -------- |
| `362b8593` | `CI: retrigger PR checks` | Empty commit; no file stats |
| `ea314277` | `CI: retrigger PR checks again` | Empty commit; no file stats |
| `e6e4a6c8` | `CI: retrigger PR checks third time` | Empty commit; no file stats |
| `5fff8140` | `CI: retrigger PR checks fourth time` | Empty commit; no file stats |
| `4c1f4b53` | `CI: retrigger PR checks fifth time` | Empty commit; no file stats |

### Fixup-Only Commits

These commits should be squashed into the child PR that introduces the affected
functional code:

| Commit | Subject | Affected area |
| ------ | ------- | ------------- |
| `92709210` | `Fix: stabilize remote L3 CI controls` | Python orchestration/task interface/worker, C++ worker manager, callable identity tests |
| `04de866b` | `Fix: resolve remote L3 CI checks` | bindings, Python protocol/session/worker, C++ orchestrator/wire, Python tests |
| `d6500ea2` | `Fix: match CI clang-format output` | C++ remote endpoint, worker manager, scheduler test formatting |

### Worktree Status

Tracked local changes after Phase 3 audit fixes:

| Path | Status | Notes |
| ---- | ------ | ----- |
| `docs/remote-l3-worker-design.md` | Modified | Adds a link to `pr-split-and-audit-plan.md`; pre-existing audit/doc change |
| `src/common/hierarchical/remote_wire.cpp` | Modified | Phase 3 fix: include `enable_scope_stats` in `CallConfigWire v1` encode/decode |
| `python/simpler/remote_l3_protocol.py` | Modified | Phase 3 fix: decode `enable_scope_stats` from TASK payloads |
| `tests/ut/cpp/hierarchical/test_remote_wire.cpp` | Modified | Phase 3 regression coverage for `enable_scope_stats` TASK round trip |
| `tests/ut/py/test_task_interface.py` | Modified | Phase 3 regression coverage for Python TASK decode of `enable_scope_stats` |
| `tests/ut/py/test_callable_identity.py` | Modified | Phase 3 coverage for `PYTHON_SERIALIZED` and `STAGED_BLOB` unsupported negotiation paths |

Untracked files existed before this artifact was created. They are preserved
unless later audit work explicitly needs one of them.

### Stop Point

This audit has completed Phases 0 through 3 and intentionally stops before
Phase 4 child branch creation / PR split execution. No child PR branches have
been created by this audit.

## Document Compliance Matrix

| Req | Source | Class | Implementation | Tests | Gap/Drift | PR |
| --- | ------ | ----- | -------------- | ----- | --------- | -- |
| R1 Endpoint abstraction keeps local mailbox local-only and routes remote L3 through `WorkerEndpoint` / `RemoteL3Endpoint` | `remote-l3-worker-design.md` §Target Architecture; `implementation-plan.md` step 1; `worker-manager.md` §4 | required in this PR cut | `WorkerEndpoint`, `LocalMailboxEndpoint`, `RemoteL3Endpoint`, `WorkerManager::add_next_level_endpoint`, `Worker::add_remote_l3_socket` | `test_remote_endpoint`, `test_scheduler`, `test_remote_sim_noop_task_roundtrip` | None found | PR 4 / PR 5 |
| R2 Endpoint outcomes distinguish success, task failure, endpoint failure, and skipped group members | `remote-l3-worker-design.md` §Failure Semantics; `implementation-plan.md` steps 1, 4 and Failure Poisoning Contract; `scheduler.md` §§2,6,9 | required in this PR cut | `EndpointOutcome`; endpoint completion mapping; scheduler failure poisoning and group skip state | `RemoteTaskErrorMapsToTaskFailure`, `FailedProducerPoisonsDependentTask`, `GroupFailureWaitsForRunningMembersThenConsumes`, remote sim error/exit tests | None found | PR 4 / PR 5 |
| R3 Scheduler dispatch uses stable worker ids and final eligible worker-id sets; `worker=` affinity is validated | `remote-l3-worker-design.md` §Worker Identity and Callable Routing; `implementation-plan.md` step 2; `scheduler.md` §5 | required in this PR cut | `Orchestrator::validate_worker_eligibility`, `WorkerManager::pick_idle_excluding_eligible`, Python worker-id set intersection | `EndpointEligibilityRestrictsIdleSelection`, `AffinityMustBeInEligibleEndpointSet`, Python remote callable worker-id intersection tests | None found | PR 4 |
| R4 Mixed local/remote pools are allowed only when callable and tensor representations are consumable by the selected worker | `remote-l3-worker-design.md` §Worker Identity and Callable Routing; `buffers-and-transports.md` §TaskArgs Sidecar Contract | required in this PR cut | Python `Orchestrator` only allows `RemoteTensorRef` for `RemoteCallable`; C++ rejects remote sidecars on local workers and non-owner remote-device dispatch without import | `RemoteSidecarRejectsLocalEndpointEligibility`, `RemoteSidecarRejectsNonOwnerEligibleEndpointWithoutImport`, Python RemoteCallable sidecar tests | No blocking drift; add a future end-to-end mixed local+remote smoke during split validation | PR 4 |
| R5 Remote sidecars are hidden metadata aligned by tensor index; local endpoints reject sidecars; remote endpoints reject bare host pointers | `remote-l3-worker-design.md` §Remote TaskArgs Representation; `buffers-and-transports.md` §§Public Memory API, TaskArgs Sidecar Contract; `protocol.md` §TASK Payload | required in this PR cut | `RemoteTaskArgsSidecar`, Python `_remote_sidecar_for`, C++ `validate_remote_sidecars`, `LocalMailboxEndpoint::run`, `RemoteL3Endpoint::build_task_payload` | `TestRemoteTaskArgsSidecar`, `RemoteBarePayloadFailsBeforeSlotCommit`, `BareHostPointerWithoutSidecarIsEndpointFailure` | None found | PR 4 / PR 6 |
| R6 Remote null `OUTPUT` tensors fail fast unless the caller supplies an explicit `RemoteTensorRef` | `remote-l3-worker-design.md` §Remote TaskArgs Representation; `buffers-and-transports.md` §Remote OUTPUT Allocation Policy; `implementation-plan.md` step 3 | required in this PR cut | `Orchestrator::validate_remote_sidecars` requires sidecar for remote OUTPUT; `reserve_outputs_and_slot` skips local HeapRing only when sidecar present | `RemoteOutputSidecarSkipsLocalAutoAllocAndRegistersRemoteKey`, remote sim OUTPUT buffer tests | No code drift; add a narrow negative test for null remote OUTPUT without sidecar when carving PR 4 | PR 4 / PR 6 |
| R7 Remote dependency keys use `(address_kind, owner_worker_id, buffer_id, generation, offset)` and preserve exact-start semantics | `remote-l3-worker-design.md` §Remote TaskArgs Representation; `buffers-and-transports.md` §Dependency Keys; `orchestrator.md` §2 | required in this PR cut | `TensorKey::remote_buffer`, `Orchestrator::infer_deps` remote path, `TensorMap` remote keys | `RemoteInputSidecarUsesRemoteTensorMapKey`, `test_remote_sim_failed_dependency_skips_consumer` | Range-overlap support remains future/open decision | PR 4 |
| R8 Remote callable identity uses canonical hashid descriptors, not target-private slots or cross-worker integer ids | `remote-l3-worker-design.md` §Worker Identity and Callable Routing; `task-flow.md` §Callable Identity; `implementation-plan.md` step 6 | required in this PR cut | `callable_identity.py`, `CallableHandle`, descriptor builders, `Worker._identity_registry` | hash stability, public export, forged/mutated handle, cleanup-uncertain tests | None found | PR 2 / PR 6 |
| R9 `RemoteCallable("module:qualname")` is the required baseline with explicit non-empty `workers=[...]` | `remote-l3-worker-design.md` §Worker Identity and Callable Routing; `protocol.md` §CONTROL Payload; `implementation-plan.md` step 6 | required in this PR cut | `RemoteCallable`, `parse_python_import_target`, `_build_callable_registration`, explicit remote worker validation | target validation and explicit remote worker tests | None found | PR 2 / PR 6 |
| R10 Multi-worker remote registration is all-or-nothing with prepare, commit, abort, cleanup-uncertain blocking, and final-unregister tombstones | `remote-l3-worker-design.md` §Worker Identity and Callable Routing; `protocol.md` §CONTROL Payload; `implementation-plan.md` step 6 | required in this PR cut | `Worker._post_start_register_remote`, `remote_prepare/commit/abort/unregister`, pending unregister tombstones, uncertain hashid guard | post-init register, unregister/reregister, inner register/unregister, uncertain cleanup guard | Partial-failure cleanup path is complex; keep as high-risk review item | PR 2 / PR 5 / PR 6 |
| R11 `INNER_L3_WORKER` is remote-internal; parent TASK frames resolve only in `REMOTE_TASK_DISPATCHER` | `remote-l3-worker-design.md` §§Worker Identity and Callable Routing, Remote Worker Session; `protocol.md` §CONTROL Payload; `task-flow.md` §Callable Identity | required in this PR cut | session dispatcher registry vs inner registry; `_prepare_register_callable`; `get_inner_handle` | dispatcher rejects chip target, inner Python import/chip callable install, inner sub-task integration | None found | PR 5 / PR 6 |
| R12 `PYTHON_SERIALIZED` remote callables reject unless serialized support is explicitly negotiated | `remote-l3-worker-design.md` §§Scope, Worker Identity and Callable Routing; `protocol.md` §CONTROL Payload; `implementation-plan.md` step 6 | required explicit unsupported behavior | `_prepare_register_callable` rejects `CallableKind.PYTHON_SERIALIZED` before install | `test_remote_register_rejects_python_serialized_without_negotiation` | Phase 3 added missing coverage | PR 6 |
| R13 `CHIP_CALLABLE` is valid only for `INNER_L3_WORKER`; inline blobs are supported by sim, staged blobs reject unless negotiated | `protocol.md` §CONTROL Payload; `implementation-plan.md` step 6; `implementation-record.md` §Completed Items | required in this PR cut plus required explicit unsupported behavior for staged blobs | `_prepare_inner_chip_callable`; dispatcher rejects chip target; staged blob reject | inner chip manifest install, dispatcher reject test, `test_remote_inner_chip_callable_rejects_staged_blob_without_negotiation` | Phase 3 added missing staged-blob coverage | PR 5 / PR 6 |
| R14 Bootstrap manifest installs dispatcher and inner registries before `HELLO READY`; unsupported negotiated manifest extensions reject rather than partially install | `remote-l3-worker-design.md` §Remote Worker Session; `protocol.md` §CONTROL Payload; `implementation-plan.md` steps 6-7 | required in this PR cut | `remote_l3_session.run_session`, `_install_manifest_inner_registry`, daemon manifest validation | manifest inner Python/chip install tests; remote sim roundtrip tests | None found | PR 6 |
| R15 Fork ordering preserves prestart before command/health transport threads; `HELLO READY` is a scheduling barrier | `remote-l3-worker-design.md` §Fork-Safe Remote Process Model; `protocol.md` §HELLO Payload; `hierarchical_level_runtime.md` §Process Model | required in this PR cut | daemon writes manifest, session prestarts `inner_worker`, then binds command/health and sends HELLO READY; parent `add_remote_l3_socket` waits for READY | remote unreachable daemon test; remote sim roundtrip/prep tests | None found | PR 6 |
| R16 Remote frame protocol uses versioned canonical little-endian encoding and never memcpy's C++ POD structs | `remote-l3-worker-design.md` §Protocol; `protocol.md` §§Frames, Wire Encoding; `implementation-plan.md` step 5 | required in this PR cut | `remote_wire.cpp` explicit put/get helpers; Python `_Reader`; local mailbox remains same-binary POD IPC only | `FrameRoundTripValidatesHeader`, Python decode tests | Found and fixed `CallConfigWire.enable_scope_stats` drift in Phase 3 | PR 3 |
| R17 Frame codec rejects bad magic/version/type/flags, oversized/truncated payloads, unknown enums, non-zero reserved fields, and malformed counts | `protocol.md` §§Frames, Wire Encoding, Bounds and Fuzz Tests; `implementation-plan.md` step 5 | required in this PR cut | C++ and Python decode validation for headers, counts, enums, reserved fields, payload sizes | remote wire bad header/truncation/reserved tests; Python materialization/decode tests | No blocking drift found | PR 3 |
| R18 TASK payload carries digest, `CallConfigWire`, and `RemoteTaskArgsWire`; tensor wire `data` must be zero; `HOST_INLINE` must be descriptor-backed and bounds-checked | `protocol.md` §§TASK Payload, RemoteTensorDesc, Bounds and Fuzz Tests; `buffers-and-transports.md` §TaskArgs Sidecar Contract | required in this PR cut | `encode_task_payload`, `decode_task_payload`, `RemoteL3Endpoint::build_task_payload`, Python `_materialize_task_args` | non-zero tensor data rejection, HOST_INLINE descriptor bounds, host-inline materialization/roundtrip | Phase 3 fixed and tested missing `enable_scope_stats` field | PR 3 / PR 5 |
| R19 COMPLETION and CONTROL_REPLY match sequence/name/version, bound error text, and fabricate failures on health expiry or process exit | `protocol.md` §§COMPLETION Payload, CONTROL_REPLY Payload, Ordering; `remote-l3-worker-design.md` §Failure Semantics | required in this PR cut | `decode_completion`, `decode_control_reply`, remote endpoint reply validation, session `_format_remote_error`, health monitor | sequence/name mismatch tests, remote sim error completion and process exit tests | None found | PR 3 / PR 5 / PR 6 |
| R20 Each endpoint has one ordered command lane; TASK, state-changing CONTROL, SHUTDOWN, replies, and visibility are sequence ordered | `remote-l3-worker-design.md` §§Remote Worker Session, Protocol; `protocol.md` §Ordering; `buffers-and-transports.md` §HCOMM Adapter Contract | required in this PR cut | `OrderedCommandLane`, endpoint command mutex, session command loop, SHUTDOWN frame path | `OrderedCommandLaneIsSingleFlight`, remote prepare/register tests | None found | PR 3 / PR 5 / PR 6 |
| R21 Health/liveness is independent from command-lane progress; health expiry removes endpoint eligibility and fails pending/in-flight work | `remote-l3-worker-design.md` §Remote Worker Session; `protocol.md` §Ordering; `implementation-plan.md` steps 5,7 | required in this PR cut | separate health socket/thread, remote endpoint health monitor, process-exit endpoint failure | long-task health-lane test, process-exit endpoint failure test | Broader multi-endpoint health-removal coverage should remain in PR 5/6 verification | PR 5 / PR 6 |
| R22 Remote memory APIs expose opaque handles and simulation alloc/free/copy/export/import/release-import controls | `buffers-and-transports.md` §§Buffer Handles, Public Memory API, Required Controls; `implementation-plan.md` steps 8-9 | required in this PR cut | Python `RemoteBufferHandle`, Worker remote memory APIs, C++ controls, session sim buffer registry | opaque handle tests, buffer copy roundtrip, export/import controls roundtrip | None found | PR 5 / PR 6 |
| R23 Owner/imported handle semantics preserve owner identity, worker eligibility, access flags, and import rollback behavior | `remote-l3-worker-design.md` §Buffer Lifecycle; `buffers-and-transports.md` §§Export/Import Handle Semantics, Release Policy; `protocol.md` §Remote Buffer Export and Import Controls | required in this PR cut | export/import descriptors, access flag validation, owner/import worker routing, imported address-space materialization | remote buffer export/import wire tests, imported buffer runs on peer worker | Import rollback is implementation-reviewed but should get split-specific failure injection | PR 5 / PR 6 |
| R24 Owner free and release-import defer physical cleanup until slot refs and imports drain; failed runs use the same post-drain cleanup path | `remote-l3-worker-design.md` §Buffer Lifecycle; `buffers-and-transports.md` §Release Policy; `implementation-plan.md` step 9 | required in this PR cut | Python slot refs, pending remote frees/import releases, `run`/`close` cleanup flush | owner-free-waits-for-import-release, input-free-deferred-until-slot-refs-drop | None found | PR 4 / PR 5 / PR 6 |
| R25 Remote CommDomain controls `COMM_INIT`, `ALLOC_DOMAIN`, and `RELEASE_DOMAIN` are reserved and rejected as unsupported in this cut | `remote-l3-worker-design.md` §Scope; `protocol.md` §CONTROL Payload; `implementation-plan.md` steps 8,13 | required explicit unsupported behavior | session control loop returns error for reserved domain controls; local C++ mailbox CommDomain controls remain local-only | Implementation inspected; no direct UT found | Add direct reserved-control negative in PR 6 verification | PR 5 / PR 6 |
| R26 A2 RoCE, A3 HCCS, and A5 UB HCOMM profiles are documented contracts but pending hardware-profile work, not part of this split cut | `remote-l3-worker-design.md` §§Current Implementation Status, Rollout; `buffers-and-transports.md` §§HCOMM Adapter Contract, A2/A3/A5 Profiles; `implementation-plan.md` steps 10-12 | reserved or future work | daemon accepts only `transport="sim"`; export/import simulation rejects unsupported profiles | Implementation inspected; remote worker manifest validation covers sim-only | Future only; do not include HCOMM adapters in child PRs | Future |
| R27 Exact HCCS/UB HAL names, daemon auth/isolation, serialized compatibility metadata, and future CommContext split remain open decisions | `implementation-plan.md` §Open Decisions | open decision | Documented only | Not applicable | Future only | Future |
| R28 Top-of-stack verification excludes unavailable hardware except through `task-submit`, covers Python UT, C++ UT, pre-commit docs, sim ST, and remote sim integration | `pr-split-and-audit-plan.md` §Review and Verification; `.claude/rules/task-submit-isolation.md` | required in this PR cut | Audit follows local UT only here; hardware remains `task-submit` only | Current run log below | Full child PR verification remains to be run after split | All |

## Phase 3 Drift Fixes

| Item | Finding | Fix | Verification |
| ---- | ------- | --- | ------------ |
| `CallConfigWire.enable_scope_stats` | `protocol.md` requires the field, but C++ omitted it and Python decoded it as `False` | Updated C++ encode/decode and Python decode; added C++ and Python round-trip tests | `test_remote_wire`; targeted Python protocol test |
| `PYTHON_SERIALIZED` unsupported behavior | Implementation rejected it, but coverage was missing | Added direct `_prepare_register_callable` negative test | targeted Python callable identity test |
| `STAGED_BLOB` unsupported behavior | Implementation rejected it, but coverage was missing | Added direct inner `CHIP_CALLABLE` staged-blob negative test | targeted Python callable identity test |

## Split Map

| File or hunk | Owner PR | Reason | Depends on | Tests |
| ------------ | -------- | ------ | ---------- | ----- |
| `docs/remote-l3-worker-design.md`, `docs/remote-l3-worker-design/*`, related docs (`worker-manager.md`, `scheduler.md`, `orchestrator.md`, `task-flow.md`, `hierarchical_level_runtime.md`) | PR 1 | Establish design, protocol, buffer lifecycle, rollout, and audit artifacts | None | docs/pre-commit checks |
| `python/simpler/callable_identity.py`, callable descriptor/hashid portions of `python/simpler/worker.py`, callable-handle portions of tests | PR 2 | Make callable identity reviewable before remote endpoint behavior | PR 1 docs | callable identity Python UT |
| `src/common/hierarchical/remote_wire.*`, `python/simpler/remote_l3_protocol.py`, remote-wire-focused tests | PR 3 | Stable cross-host protocol and codec validation | PR 1 docs; PR 2 enum/identity definitions | C++ `test_remote_wire`; Python protocol decode tests |
| Scheduler and slot-state hunks in `src/common/hierarchical/types.*`, `orchestrator.*`, `scheduler.*`, `worker_manager.*` | PR 4 | Endpoint eligibility, outcomes, sidecars, dependency keys, failure poisoning | PR 2 identity; PR 3 types/protocol where referenced | C++ `test_orchestrator`, `test_scheduler` |
| C++ remote endpoint and binding hunks in `remote_endpoint.*`, `worker_manager.*`, `worker.h`, `python/bindings/worker_bind.h` | PR 5 | Transport endpoint, command lane, controls, and Python C++ facade | PR 3 wire; PR 4 endpoint abstraction | C++ `test_remote_endpoint`, `test_remote_wire`; binding smoke |
| Python remote session/runtime hunks in `python/simpler/worker.py`, `orchestrator.py`, `task_interface.py`, `remote_l3_worker.py`, `remote_l3_session.py` | PR 6 | Remote daemon/session runner, registration choreography, sim memory controls, lifecycle cleanup | PR 2-5 | Python remote sim integration tests |
| Fixup commit `92709210` | Squash into PR 4 / PR 5 / PR 6 hunks | Stabilizes CI controls across Python orchestration, task interface, worker manager, and tests | Matching child PRs | Matching child PR tests |
| Fixup commit `04de866b` | Squash into PR 3 / PR 5 / PR 6 hunks | Resolves protocol/session/binding/orchestrator CI failures | Matching child PRs | Matching child PR tests |
| Fixup commit `d6500ea2` | Squash into PR 5 plus affected C++ tests | clang-format-only formatting for remote endpoint/worker manager/scheduler tests | Matching child PRs | formatting / C++ UT |
| Empty CI retrigger commits | Drop | No functional or document content | None | Not applicable |

## Risk Register

| Risk | Location | Failure mode | Mitigation | Test |
| ---- | -------- | ------------ | ---------- | ---- |
| Protocol drift between C++ endpoint and Python session | `remote_wire.cpp`, `remote_l3_protocol.py`, `protocol.md` | Remote TASK decode shifts fields or drops config flags | Keep PR 3 protocol-first; add round-trip tests for every `CallConfigWire` field | Phase 3 `enable_scope_stats` tests; `test_remote_wire` |
| Partial remote registration cleanup | `Worker._post_start_register_remote`, remote prepare/commit/abort controls | Some endpoints commit while others fail, leaving stale registry state | Preserve prepare/commit/abort/tombstone design; review cleanup-uncertain guard in PR 6 | post-init register/unregister tests; add failure injection in PR 6 |
| Endpoint failure poisoning | `scheduler.cpp`, `remote_endpoint.cpp`, `remote_l3_session.py` | Failed producer or dead endpoint lets consumers dispatch with stale data | First-error-wins, dependency poisoning, health/process-exit failures | scheduler failed producer; remote sim process-exit and failed-dependency tests |
| Remote buffer lifetime | `worker.py`, `task_interface.py`, `remote_l3_session.py` | Owner freed while imported/slot-ref still active | Slot refs, import refs, pending free/release queues | owner-free/import-release and slot-ref-deferred free tests |
| Hidden sidecar integrity | `task_interface.py`, `orchestrator.cpp`, `remote_endpoint.cpp` | Bare host pointer or mismatched descriptor count reaches remote worker | Python sidecar storage, C++ validation, remote endpoint fail-fast | sidecar unit tests, bare host pointer endpoint failure |
| Future controls accidentally treated as supported | `remote_l3_session.py`, HCOMM docs | Reviewer believes CommDomain/HCOMM or serialized/staged paths are ready | Explicit unsupported tests and Future Work list | Phase 3 serialized/staged tests; add CommDomain negative |
| Verification build environment | `tests/ut/cpp/build` | System GoogleTest ABI mismatch blocks C++ UT link | Use `tests/ut/cpp/build-fetch` or rebuild system gtest per troubleshooting doc | Current run log records fallback |

## Per-PR Verification Checklist

| PR | Verification |
| -- | ------------ |
| PR 1 Docs | Review rendered docs; run docs/pre-commit hooks if available |
| PR 2 Callable Identity | Python UT for descriptor hashing, opaque handles, local registration/unregistration, forged/mutated handle rejection |
| PR 3 Protocol | C++ `test_remote_wire`; Python protocol decode/materialization tests; verify no POD memcpy in remote wire |
| PR 4 Scheduler / Eligibility | C++ `test_orchestrator` and `test_scheduler`; negative tests for remote sidecar/local endpoint and null remote OUTPUT |
| PR 5 C++ Remote Endpoint / Bindings | C++ `test_remote_endpoint`; binding import smoke; command lane/control reply checks |
| PR 6 Python Session / Sim Runtime | Python remote sim integration tests in `test_callable_identity.py`; memory alloc/copy/export/import/release tests; reserved-control negative |
| All | No hardware commands locally; hardware verification only through `task-submit` per repo rule |

## Future Work

| Item | Source | Notes |
| ---- | ------ | ----- |
| A2 RoCE remote data plane | `buffers-and-transports.md` A2 profile | HCOMM adapter and hardware verification are out of this split |
| A3 HCCS remote data plane | `buffers-and-transports.md` A3 profile | Exact HAL/API names remain open |
| A5 UB LD/ST profile | `buffers-and-transports.md` A5 profile | UB address-space support is documented but future |
| Remote CommDomain controls | `implementation-plan.md` steps 8,13 | `COMM_INIT`, `ALLOC_DOMAIN`, `RELEASE_DOMAIN` remain reserved/unsupported remotely |
| `PYTHON_SERIALIZED` negotiated support | `protocol.md` CONTROL payload | Requires compatibility metadata and negotiation |
| `CHIP_CALLABLE` `STAGED_BLOB` support | `protocol.md` CONTROL payload | Requires staged-blob adapter and negotiation |
| Remote OUTPUT autoallocation | `buffers-and-transports.md` Remote OUTPUT policy | Current cut requires explicit `RemoteTensorRef` |
| Remote dependency range overlap | `buffers-and-transports.md` Dependency Keys | Current key uses exact-start semantics |
| Daemon auth/isolation | `implementation-plan.md` Open Decisions | Security model is future work |
| Future CommContext split | `implementation-plan.md` Open Decisions | Kept out of current split |

## Current Verification Run Log

| Command | Result | Notes |
| ------- | ------ | ----- |
| `.venv/bin/python -m pytest tests/ut/py/test_task_interface.py::TestRemoteL3SessionTaskArgsMaterialization::test_task_payload_decode_preserves_scope_stats_config tests/ut/py/test_callable_identity.py::test_remote_register_rejects_python_serialized_without_negotiation tests/ut/py/test_callable_identity.py::test_remote_inner_chip_callable_rejects_staged_blob_without_negotiation -q` | Passed | 3 passed; non-fatal pto-isa network update warning appeared in sandbox |
| `cmake --build tests/ut/cpp/build --target test_remote_wire -j2` | Failed at link | Existing system GoogleTest ABI mismatch; compile reached modified C++ objects |
| `cmake --build tests/ut/cpp/build-fetch --target test_remote_wire -j2` | Passed | Uses FetchContent GoogleTest build |
| `ctest --test-dir tests/ut/cpp/build-fetch -R '^test_remote_wire$' --output-on-failure` | Passed | 1/1 test passed |
