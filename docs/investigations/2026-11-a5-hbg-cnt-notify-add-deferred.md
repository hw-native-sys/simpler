# CntNotify ADD completion chains through the SET mapping

On 2026-11-17 the direct ADD mapping was withdrawn. The active A5 implementation
uses unit increments through the retained SET mapping.

Follow-up, 2026-11-17: direct ADD mapping still fails, but the user's A5 probe
validated ADD(1) through the held SET mapping's 32-byte slot plus offset 0x08.
The user authorized enabling the prepared implementation through that access.
Follow-up probes require command value 1 per increment: writing 2 does not add
2 and writing 3 increments once. The restored scheduler repeats command 1 for
each credit. Actual full ADD coverage and the twelve-case replay passed; the
AICore L0 sum regressed 0.202%, within the accepted 0.5% threshold.
The current evidence and validation limits are recorded in the existing
[design document](../zh-cn/design/a5-hbg-graph-task/build-execute-separation.md).
The withdrawal below remains the historical decision, not a hardware limitation.

## Initial withdrawal

Commit `8a2c5e42bb5d918110af0ca3d9a3e45f5b72f922` prepared a real, non-returning
CntNotify ADD path. The installed driver still rejects the AICore ADD mapping
with 507899. CPU validation and a successful hardware GM fallback cannot qualify
that path for production or establish an ADD performance result.

The initial revert covered the preparation commit, including runtime, platform,
DFX, tests and active design changes. Production behavior at that point returned to
`f8a6b675617ff4d36655b380ce75a97ea11ca446`: SET/read logical DONE notifications,
remote GM MIX SET handshakes, the retained Host resource pool and the resource
exhaustion handling fix. The original preparation commit stays in Git history.
Pre-commit also normalizes whitespace in two restored protocol headers; it changes no runtime behavior.
This withdrawal is a capability/readiness decision, not a measured ADD regression.

## Preserved proposal

- A physical counter is shared only along a strict, real DAG ancestor chain.
  Each task has a cumulative threshold; independent tasks gain no dependency.
- Every executed lane contributes physical completion P and condition completion
  Q. Synchronous ordinary returns combine them into ADD(2); deferred returns and
  successful condition checks contribute ADD(1) separately.
- MIX retains the full resource-retirement boundary before publishing physical
  credits. Exactly-once condition ownership and dynamic GM task claims remain.
- Counters only increase during a run. Host planning splits before 32-bit
  overflow. READY preparation resets each chain once; there is no within-run
  reset, bitmap, bank ownership transfer or retired-state snapshot.
- Successors observe the threshold directly. Canonical completion accounting is
  separate and must not gate readiness, task ownership or resource retirement.
- The proposed DFX records actual P/Q contributions and counter observations,
  without inventing a returned FAA ticket or using accounting time as readiness.

The complete proposal is available in the original commit's
`docs/zh-cn/design/a5-hbg-graph-task/build-execute-separation.md`, under
the non-returning ADD completion-chain section. It is not current behavior.

## Evidence and re-entry requirements

The preparation passed focused runtime, Host and DFX CPU checks, A5 compilation
and 16/16 device translation-unit IR checks. A single-device SWA functional run
on automatically allocated device 1 passed numerical validation with the GM
fallback. The ADD mapping failed with 507899 and was attempted once in that
context. At withdrawal, the twelve-case comparison had only exercised the GM fallback.

Local preservation paths:

- `artifacts/cnt-add-rollback-20261117/add-preparation.patch` and
  `design-before-revert.md`: complete patch and design snapshot.
- `artifacts/cnt-add-prep-20261117/`: validation summary, fallback hardware logs,
  installation provenance and the future ADD-coverage check.
- `artifacts/c2a-handoff-20261115/cnt-add-prepared-final/`: frozen source, runtime
  binaries and IR evidence for the prepared candidate.
- `artifacts/c2a-handoff-20261115/cnt-retained-final/`: restored supported runtime.

The subsequent A5 run used the existing driver and held SET mapping. To repeat
that comparison, use a fresh Worker/context, check ADD coverage at SET slot
offset `0x08` with command value 1 per credit, and validate accumulation, late
observation, branch independence, MIX, deferred conditions and replay reset.
Compare the twelve-case AICore L0 sum against the 0.5% aggregate regression
limit. The verified run measured a 0.202% increase.
