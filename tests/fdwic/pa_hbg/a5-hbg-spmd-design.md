# A5 HBG AICore-resident scheduling

## Responsibilities

Host expands the instance DAG, validates execution shapes, prepares the immutable
Calls and emits the ranked AIC and AIV catalogs. AICPU coordinates bootstrap and
exit. Each Scalar worker owns a private catalog cursor and keeps the execution
record for each block it claims or receives through its cluster.

## Supported execution shapes

| Shape | Allocation and execution | Boundary |
| ----- | ------------------------ | -------- |
| Ordinary pure AIC/AIV | Workers of the matching type claim blocks dynamically, in multiple waves if necessary | The owner prepares, runs and retires each block |
| MIX, masks 3/5/6/7 | Each cluster's AIV0 claims a logical block and publishes its descriptor in SSBUF | Only the lanes in the computing mask execute and publish FIN |
| Multi-block `sync_start` | At most one claim per physical position for that instance | The request must fit the distinct AICs, AIVs or complete MIX clusters |
| Single-block public `sync_start` request | Host normalizes away the multi-block sync flag | A MIX block still requires its cluster handshake |
| Dummy / inline service | Logical dependency service | Host-completed nodes are counted once |
| Predicate | Single-block pure kernels can skip invocation and retire | Host rejects multi-block or MIX predicates |
| Supported Graph body | Expanded ENTER, child and EXIT nodes follow their control and data dependencies | Distinct instances retain separate identities |
| Deferred completion | Physical return and registered external conditions contribute separately | Dependent work needs logical completion |

The execution count satisfies `N > 0 && N * popcount(mask) <= INT16_MAX`.
A kernel that waits internally for all blocks must request common residency and
fit the available physical positions.

## Issue, ordering and logical completion

Within each engine catalog, an earlier task's supply must close before a worker
can advance its cursor. Closure certifies that all blocks have owners. The owners
can still be waiting for dependencies or running kernels. An independent task can
therefore start while another worker finishes earlier work.

The Host rank orders conflicting resource commitments. A pure worker checks the
required MIX floor and its own outstanding cluster obligations. MIX members check
earlier conflicting pure `sync_start` obligations before committing to the group.
A real data dependency requires its predecessor's logical completion. A consumer
can receive its Call earlier, then waits for that completion before its final
payload publication, or before publishing ARRIVED for a MIX execution.

## Cluster MIX and identity

AIV0 publishes one group generation per cluster. Each active lane receives its
own Call, publishes ARRIVED, observes all active arrivals and runs from its own
MIX record. After publishing outputs and returning, each lane publishes FIN. The
accountant is AIV0 when it computes, otherwise the lowest active lane. It checks
all successful FINs and contributes group completion before the descriptor is
reused. For mask 5, AIV0 publishes and AIC accounts. For mask 6, both AIVs
compute and AIC has no group obligation.

| Identity | Meaning |
| -------- | ------- |
| `(run_epoch, task_ref, block_idx, subtask_slot)` | Unique kernel execution |
| `(run_epoch, order_key)` | Instance rank within the run |
| Worker and source record | Physical owner and payload lifetime |
| Cluster and group sequence | Descriptor, ARRIVED and FIN generation |

## Implementation map

Paths are relative to `src/a5/runtime/host_build_graph/`.

| Component | Responsibility |
| --------- | -------------- |
| `host/task_catalog_plan.h` | Rank expanded instances and generate conflict thresholds |
| `runtime/scheduler/scheduler_issue.h` | Check catalog heads, claim blocks and certify supply closure |
| `runtime/scheduler/scheduler_order.h` | Compare local obligations and cross-source progress against rank |
| `runtime/scheduler/scheduler_cluster_mix.h` | Publish descriptors, service actual lanes and account for group FINs |
| `runtime/scheduler/scheduler_cluster_signal.h` | Cluster SSBUF fields and publication primitives |
| `runtime/scheduler/scheduler_completion.h` | Account for contributions and publish logical completion |
| `aicore/resident_scheduler_impl.h` | Advance local work and service control and termination |
| `host/scheduler_protocol_export.h` | Export graph, native protocol records and terminal state |

See the [implementation contract](spmd/implementation-contract.md) for the
observable protocol, [validation](spmd/validation.md) for executable coverage
and [performance](spmd/performance.md) for timing limits.
