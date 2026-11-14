# A5 resident scheduler test contract

The current implementation and planned work are described in the
[unified HBG scheduler design](../../../../docs/zh-cn/design/a5-hbg-graph-task/build-execute-separation.md).
This page records the observable test boundaries; wire layouts and publication
rules have one authoritative definition in the runtime headers.

## Execution and progress

- Discovery of an ordinary task is nonexclusive. Its block reservation and
  prepared Call ownership must still grant each execution exactly once.
- A successor may prepare before its dependencies complete, but may execute
  only after their logical completion. Unready candidates cannot hide an
  independent notification publisher beyond the private candidate window.
- Ordinary MIX uses one formal SSBUF group per cluster. Required resources
  include the active lanes and the AIC coordinator; only active lanes execute
  business Calls. No ordinary MIX admission or sync claim certificate is invented.
- `sync_start` retains ordered admission and local obligation retirement.
  Stage/hold arbitration must precede its block claim. A fast member can advance
  without requiring a global completion barrier; late sync members retain their
  close certificates.
- Kernel return releases physical execution resources. Deferred conditions keep
  logical completion pending through shared CHECK/FINISH service, including
  terminal tasks with no consumers. Pending checks release their leases.
- Fatal errors do not grant new execution responsibility or fabricate successful
  completion. EXIT retires formal cluster responsibility even without a sync log.

## Evidence

Use the current schemas declared by
[`native_protocol.py`](../../../../simpler_setup/tools/hbg/native_protocol.py)
and [`execution_identity.py`](../../../../simpler_setup/tools/hbg/execution_identity.py).
Identity must distinguish GM and SSBUF groups; an ordinary SSBUF generation does
not imply an admission order. Predicate skips retain their actual execution flag.

L4 checks must join actual task/block/lane identities and validate ownership,
stage/claim, arrival/release, reuse, completion and terminal facts. Sequence
integrity alone does not prove those contracts. Missing, dropped or truncated
records cannot stand in for a zero-cost operation or a successful protocol.

Read concrete offsets, sizes and atomic/cached publication domains from
[`scheduler_types.h`](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h),
[`scheduler_cluster_signal.h`](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_cluster_signal.h)
and the corresponding producer/consumer helpers. Preserve their acquire/release
boundaries; do not infer cross-clock latency from timestamps lacking alignment.
