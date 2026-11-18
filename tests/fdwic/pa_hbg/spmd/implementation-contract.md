# A5 resident scheduler test contract

The current implementation and planned work are described in the
[unified HBG scheduler design](../../../../docs/zh-cn/design/a5-hbg-graph-task/build-execute-separation.md).
This page records the observable test boundaries; wire layouts and publication
rules have one authoritative definition in the runtime headers.

## Execution and progress

- Discovery of an ordinary task is nonexclusive. Its block reservation and
  prepared Call ownership must still grant each execution exactly once.
- A successor may prepare before its dependencies complete, but may execute
  only after their logical completion. Each private
  issue cursor advances only after the current task has closed its supply.
- Each cluster's AIV0 publishes its MIX descriptor in SSBUF. Only the active
  computing lanes publish ARRIVED and FIN. The active accountant contributes
  the group's physical credits after all successful FINs.
- `sync_start` uses distinct physical positions. A member committing to work
  checks its earlier conflicting obligations against the Host rank and observed
  progress. A fast member may retire before the entire task is logically done.
- Kernel return releases physical execution resources. Deferred conditions keep
  logical completion pending through shared CHECK/FINISH service, including
  terminal tasks with no consumers. Pending checks release their leases.
- Fatal errors do not grant new execution responsibility or fabricate successful
  completion. EXIT retires formal cluster responsibility for the active generation.

## Evidence

Use the current schemas declared by
[`native_protocol.py`](../../../../simpler_setup/tools/hbg/native_protocol.py)
and [`execution_identity.py`](../../../../simpler_setup/tools/hbg/execution_identity.py).
Identity joins the source record, Host rank, cluster and group generation to
the executing worker. Predicate skips retain their actual execution flag.

L4 checks must join actual task/block/lane identities and validate ownership,
claim, arrival/FIN, reuse, completion and terminal facts. Sequence
integrity alone does not prove those contracts. Missing, dropped or truncated
records cannot stand in for a zero-cost operation or a successful protocol.

Read concrete offsets, sizes and atomic/cached publication domains from
[`scheduler_types.h`](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_types.h),
[`scheduler_cluster_signal.h`](../../../../src/a5/runtime/host_build_graph/runtime/scheduler/scheduler_cluster_signal.h)
and the corresponding producer/consumer helpers. Preserve their acquire/release
boundaries; do not infer cross-clock latency from timestamps lacking alignment.
