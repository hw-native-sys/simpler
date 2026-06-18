# PR Split and Audit Plan

This plan defines how to split the large PR #866 remote L3 branch into
reviewable stacked PRs without losing the document contract or hiding
correctness, race, and resource-lifetime risks across PR boundaries.

## Decision

Use this order:

1. Audit the current whole PR against the PR-head documents.
2. Fix or remove clear design drift in the current branch.
3. Split the corrected branch into stacked PRs.
4. Run deep risk review on each smaller PR.

Do not split first. A split performed before the contract audit can spread one
design error across several smaller PRs, making the error harder to see and
harder to revert.

Goals:

- Preserve the remote L3 documents as the source of truth.
- Make each child PR independently reviewable.
- Keep tests with the feature they verify.
- Separate protocol, scheduler, endpoint, and Python runtime concerns.
- Keep unsupported and future work explicit.
- Avoid transition-only public APIs.

Non-goals:

- Do not redesign the protocol while splitting.
- Do not add hardware HCOMM profiles only to make the split look complete.
- Do not move feature tests into a separate test-only PR.
- Do not preserve empty CI retrigger commits in any split PR.

Inputs:

- `docs/remote-l3-worker-design.md`
- `docs/remote-l3-worker-design/protocol.md`
- `docs/remote-l3-worker-design/buffers-and-transports.md`
- `docs/remote-l3-worker-design/implementation-plan.md`
- `docs/remote-l3-worker-design/implementation-record.md`
- Related contract docs:
  - `docs/orchestrator.md`
  - `docs/scheduler.md`
  - `docs/worker-manager.md`
  - `docs/task-flow.md`
  - `docs/hierarchical_level_runtime.md`

Use PR-head documents, not stale local notes. Drop empty CI retrigger commits
during the split. Fixup formatting and CI stabilization commits into the
functional PR that introduced the affected code.

## Required Artifacts

Create these before opening the child PRs:

- Document compliance matrix.
- Split map.
- Risk register.
- Per-PR verification checklist.
- Future-work list.

Use these verdicts in the compliance matrix:

- `ok`: implemented exactly as documented.
- `gap`: documented requirement is missing or only partially implemented.
- `drift`: implementation adds behavior not justified by the documents.
- `unsupported`: implementation explicitly rejects a documented future or
  reserved path.
- `future`: document says the behavior is outside this PR cut.
- `open`: document leaves a decision unresolved.

The document compliance matrix must record:

- Requirement source document and section.
- Requirement classification:
  - required in this PR cut
  - required explicit unsupported behavior
  - reserved or future work
  - open decision
- Implementation files and symbols.
- Test coverage.
- Missing pieces.
- Extra behavior not described by the documents.
- Whether extra behavior is necessary, harmless, or design drift.
- Target split PR.

Use this matrix shape:

| Req | Source | Class | Implementation | Tests | Gap/Drift | PR |
| --- | ------ | ----- | -------------- | ----- | --------- | -- |
| | | | | | | |

The split map must assign every changed file and major hunk to one child PR.
If one hunk does not belong cleanly to one PR, split the hunk before opening
the child branch.

Use this split-map shape:

| File or hunk | Owner PR | Reason | Depends on | Tests |
| ------------ | -------- | ------ | ---------- | ----- |
| | | | | |

The risk register must include at least:

- Register prepare, commit, abort, unregister cleanup.
- Scheduler endpoint selection and completion ordering.
- Group partial failure and downstream poison propagation.
- Command-lane serialization and sequence matching.
- Health lane and session teardown.
- Remote buffer owner free, import, release-import, and slot refs.
- Fork-before-thread ordering.
- Nanobind GIL release and reacquire boundaries.
- Unsupported reserved controls.

Use this risk-register shape:

| Risk | Location | Failure mode | Mitigation | Test |
| ---- | -------- | ------------ | ---------- | ---- |
| | | | | |

The future-work list must be explicit about what the current split does not
implement. For this PR cut, expected future work includes hardware HCOMM
profiles and Remote CommDomain support unless the documents are changed.

## Workflow

Phase 0: freeze and normalize.

1. Record branch name, feature HEAD, and document baseline.
2. Mark empty CI retrigger commits as drop-only.
3. Mark formatting or CI-fix commits as fixup-only.
4. Confirm there are no unrelated tracked changes.
5. Do not create new feature commits until the document audit is complete.

Expected output:

- A short baseline note with branch, HEAD, docs baseline, drop commits, and
  fixup commits.

Phase 1: inventory document requirements.

1. Read the PR-head remote L3 documents.
2. Extract every required behavior.
3. Classify each behavior as required, unsupported, future, or open decision.
4. Record success behavior and failure behavior.
5. Record which layer owns the behavior:
   protocol, Python API, binding, scheduler, endpoint, session runner, or
   tests.

Stop if a requirement is ambiguous enough that two incompatible
implementations would both look plausible. Update the document or ask for a
decision before coding or splitting.

Phase 2: map current implementation.

1. Map each requirement to concrete files and symbols.
2. Map each requirement to tests.
3. Record missing pieces.
4. Record extra behavior not described by the documents.
5. Decide whether the extra behavior is necessary, harmless, or design drift.
6. Assign each requirement and major hunk to a child PR.

Stop if a requirement has unresolved design drift, depends on a temporary API,
or only works because two future child PRs are merged together.

Phase 3: remove drift before splitting.

1. Delete behavior that is not required and not justified by the documents.
2. Convert future work into explicit unsupported errors where required.
3. Update docs when an accepted schema or lifecycle rule was missing.
4. Add focused tests for required unsupported behavior.

Examples:

- Hardware profiles may stay pending only if the docs say A2 RoCE, A3 HCCS,
  and A5 UB HCOMM are future hardware-profile work.
- Remote CommDomain controls should be explicit unsupported paths in this cut
  if the docs reserve them for a later phase.
- `PYTHON_SERIALIZED` should be rejected unless negotiated if the docs define
  it as a negotiated future extension.

Phase 4: split into stacked PRs.

1. Create child branches in stack order.
2. Move only the files and hunks assigned to each PR.
3. Keep tests with the feature under test.
4. Squash fixup commits into the affected functional PR.
5. Drop empty CI commits.
6. Re-run the per-PR verification checklist after each split.

Suggested split mechanics:

1. Create a clean base branch from the document baseline.
2. Create `remote-l3-docs` for PR 1.
3. Create each later child branch from the previous child branch.
4. Move only assigned files and hunks onto each child branch.
5. Prefer `git restore --source <feature-head> -- <path>` for whole-file
   slices.
6. Use interactive staging or patch application for mixed files such as
   `python/simpler/worker.py` and `worker_manager.cpp`.
7. Build and test each child branch before creating the next branch.
8. At the top of the stack, verify that the combined diff matches the
   corrected feature branch except for dropped CI-only commits.

Stop before opening a child PR if:

- It requires a later child PR to pass its own unit tests.
- It exposes a public API that exists only to make the split convenient.
- It contains behavior classified as unresolved drift.
- It makes a future feature partially visible instead of explicitly
  unsupported.
- It moves tests away from the feature under test.
- Its PR description cannot name the document requirements it satisfies.

## Split Stack

PR 1: remote L3 contract documents.

Scope:

- Remote L3 landing doc.
- Protocol document.
- Buffer and transport document.
- Scheduler, orchestrator, worker-manager, and task-flow contracts.
- This split and audit plan if useful for reviewers.

Acceptance criteria:

- Required, unsupported, reserved, and future work are separated.
- Required controls have schemas or explicit unsupported behavior.
- Hardware profiles are clearly marked as future if not implemented.
- No document promises behavior that later PRs do not implement.

PR 2: callable identity and run contract.

Scope:

- Callable descriptors, digest/hashid identity, and target namespaces.
- `ChipWorker.prepare_callable()` handle semantics.
- `ChipWorker.run()` and `Worker.run()` timing return contract.
- Python import callable descriptors needed by remote registration.
- Focused Python unit tests.

Acceptance criteria:

- Target-private slots are not public routing IDs.
- Prepare rollback and unregister refcount behavior are tested.
- Descriptor hashing is stable.
- `RunTiming` introspection contract is preserved.

PR 3: remote wire codec.

Scope:

- C++ `remote_wire`.
- Python `remote_l3_protocol`.
- Frame, TASK, COMPLETION, CONTROL, CONTROL_REPLY, HELLO, HEALTH, and buffer
  control codecs.

Acceptance criteria:

- Bounds checks reject malformed payloads.
- Unknown enums and non-zero reserved fields are rejected.
- Sequence matching is tested.
- Error payloads are bounded.
- Remote wire format does not memcpy C++ POD structs.

PR 4: scheduler and endpoint abstraction.

Scope:

- `WorkerEndpoint` interface.
- `LocalMailboxEndpoint` adapter.
- Endpoint capability metadata and eligible endpoint sets.
- `WorkerCompletion` outcome propagation.
- Failed task poisoning.
- Remote tensor sidecars and dependency keys, without a live remote session.
- C++ scheduler, orchestrator, and worker-manager tests.

Acceptance criteria:

- Scheduler chooses only eligible idle endpoints.
- Worker affinity is validated against eligibility.
- Group partial failure and downstream poison are tested.
- Slot release and `drain()` behavior are correct after success and failure.

PR 5: C++ remote endpoint and bindings.

Scope:

- `RemoteL3Endpoint`.
- Transport-neutral command lane.
- C++ remote endpoint control methods.
- Nanobind exposure for remote controls and remote buffer descriptors.
- Endpoint-level C++ unit tests.

Acceptance criteria:

- Per-endpoint command serialization is enforced.
- Control and completion replies match sequence numbers.
- Health failure can surface while command lane waits.
- Endpoint failure and task failure are classified distinctly.
- Bindings do not build Python return values while the GIL is released.

PR 6: Python remote session and simulation runtime.

Scope:

- `simpler-remote-worker`.
- `simpler-remote-l3-session`.
- Python `RemoteCallable`, `RemoteBufferHandle`, `RemoteBufferExport`,
  `RemoteTensorRef`, and hidden remote tensor sidecar integration.
- Simulation remote buffer allocation, copy, export, import, and release.
- Remote dispatcher and inner worker registry controls.
- Simulation integration tests.

Acceptance criteria:

- Fork-before-thread invariant is preserved.
- `HELLO READY` is a scheduling barrier.
- Session shutdown and health expiry are tested.
- Partial register and partial import cleanup are tested.
- Owner frees wait for slot refs and imports to drain.
- Reserved Remote CommDomain controls fail explicitly.
- A2 RoCE, A3 HCCS, and A5 UB HCOMM are not included unless they are their
  own hardware-profile PRs.

## Review and Verification

Each child PR review must answer:

1. Which document requirements does this PR satisfy?
2. Which required requirements remain for later stacked PRs?
3. Which future or reserved behaviors are explicitly unsupported?
4. What state is shared across threads or processes?
5. What happens if prepare, task, control, copy, import, or release fails
   halfway through?
6. What cleanup runs after `drain()` throws?
7. What cleanup runs during `Worker.close()`?
8. Which user-visible errors are bounded and deterministic?
9. Which tests prove success paths?
10. Which tests prove failure and cleanup paths?

Minimum risk checklist:

- No lock inversion across scheduler, ring, worker manager, and endpoint code.
- No task slot is released before captured refs are safe to drop.
- No endpoint receives TASK for a prepared, aborted, or partially committed
  digest.
- No partially failed remote register leaves a visible hashid.
- No imported buffer is used after `RELEASE_IMPORT`.
- No owner buffer is physically freed while an import or slot ref is live.
- No health thread can outlive the transport or session object it references.
- No command-lane wait can hide health failure indefinitely.
- No Python session forks after network or C++ worker threads start.
- No nanobind return value is constructed while the GIL is released.

Minimum verification:

- PR 1: markdown and pre-commit docs checks.
- PR 2: callable identity and task interface Python unit tests.
- PR 3: C++ remote wire tests and Python protocol codec tests.
- PR 4: C++ scheduler, orchestrator, and worker-manager tests.
- PR 5: C++ remote endpoint tests and binding smoke tests.
- PR 6: remote simulation integration tests, buffer export/import/release
  tests, health lane tests, and session-exit failure tests.

Top-of-stack verification:

- Python unit tests excluding hardware-only marks.
- C++ unit tests excluding hardware-only labels.
- Pre-commit checks that do not require unavailable network downloads.
- Simulation scene tests for at least one A2/A3 path and one A5 path.
- Hardware ST only through `task-submit --device ...`.

Onboard CI failures must be classified by affected job and stack trace. Do not
treat an onboard runner failure as a remote L3 regression unless the failing
stack enters remote L3 code or a split PR changed the failing runtime path.

The split is done when every document requirement maps to a child PR or a
future-work item, every child PR has tests, unsupported controls fail
explicitly, empty CI commits are absent, and the top of the stack passes the
agreed non-hardware test set.
