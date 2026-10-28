# A5 HBG AICore-resident scheduling

## Responsibilities

Host builds the DAG, TensorMap, kernel registrations, memory plan and immutable
task metadata. AICPU performs bootstrap, topology discovery and lifecycle/error
handshakes. Each AIC/AIV runs its own Scalar scheduler and prepares its own kernel
payload. AICore does not build a new graph or allocate a runtime heap.

Dependencies remain completion-driven: the final block completion advances the
existing wake-list and publishes ready successors. Workers do not repeatedly
scan all fan-in flags. Ready queues discover work; they are not a global ordering
proof for kernels that can wait for other participants.

## Supported execution shapes

| Shape | Allocation and execution | Boundary |
| ----- | ------------------------ | -------- |
| Ordinary pure AIC/AIV, one block | Existing ready/direct candidates and entry gate | Existing single-lane predicate behavior retained |
| Ordinary pure AIC/AIV, multiple blocks | Dynamic ownership, multiple waves | Original block count and parameter ABI retained |
| Mix, masks 3/5/6/7, any legal block count | AIC coordinates a GM group; each active lane executes itself | All three local resource lanes participate, including inactive lanes |
| `sync_start`, multiple blocks | At most one claim per execution position | Must fit distinct AIC/AIV positions or complete Mix clusters |
| Public `sync_start` request with N=1 | Existing normalization removes the multi-block sync flag | Mix pairing and resource obligations remain |
| Dummy / inline service | Logical dependency service, no kernel trace or payload | Host-completed nodes are not completed twice |
| Multi-block or Mix predicate | Rejected by Host | Not implemented by this change |
| Graph body task | Whole-run legacy AICPU fallback | Strict resident validation rejects fallback |
| Deferred/asynchronous completion | Resident registration returns false | No automatic detection or fallback for arbitrary async kernels |

Ordinary blocks must finish without an undeclared all-block barrier. A task that
waits for all blocks must request common residency and fit its capacity. Two
pending payload slots do not double physical residency. Public block requests
are checked before narrowing; the execution count must satisfy
`N > 0 && N * popcount(mask) <= INT16_MAX`.

## Shared order without a global completion barrier

```text
predecessor TaskDone -> wake-list -> ready discovery
                                      |
                    publish admission order + frozen members
                                      |
                    each worker's earliest member obligation
                                      |
                    claim block / pair Mix group / execute
                                      |
                     local completion and resource release
                                      |
                 last block -> TaskDone -> release successors
```

Cooperative tasks share a runtime order, not Host submission order. If A is
blocked and independent B is ready, B may be admitted first. After publication,
workers sharing membership must respect that order. Nonmembers can skip a task.
Membership cannot expand after a worker has passed it.

`LocalDone(worker, A)` means the worker cannot receive more A work and has
retired all its execution, publication and resource obligations. `TaskDone(A)`
means all blocks have completed and permits dependent consumers to run. A fast
worker may enter independent B after its own LocalDone(A), while another worker
is still executing A. A real A-to-B dependency still requires TaskDone(A).

Pure tasks include candidates of their own core type. Mix currently allows all
clusters to claim, so its membership includes all three lanes of every candidate
cluster. A block's handshake is local to its cluster, not a chip-wide start
barrier. Shrinking only the member bitmap would be incorrect: an omitted AIV
could enter a later blocking kernel before a preceding Mix needs it.

## GM Mix and identity

One cluster has one in-flight GM group. The AIC claims a block, publishes a
descriptor and aggregates release acknowledgments. Each lane prepares its own
context; inactive lanes perform resource service only. Mask 6 has no AIC kernel.
Logical slots retain their hardware lane mapping, including the fixed AIV0
endpoint used by NO_SPLIT kernels. Scheduler communication does not use SSBUF;
kernel-internal TPipe/L1/event contracts remain unchanged.

| Identity | Meaning |
| -------- | ------- |
| `(run_epoch, task_ref, block_idx, subtask_slot)` | Unique kernel execution |
| `(run_epoch, order_key)` | Published cooperative order |
| Local slot and generation | Payload lifetime, not global order |
| Cluster and group sequence | Mix descriptor/acknowledgment lifetime |

The [implementation contract](spmd/implementation-contract.md) specifies claims,
publication domains, local closure, errors and native protocol evidence.

## Implementation map

Paths below are relative to `src/a5/runtime/host_build_graph/`.

| Component | Responsibility |
| --------- | -------------- |
| `host/runtime_maker.cpp` | Shape/capacity checks, preallocation, upload and trace export |
| `aicpu/aicore_lifecycle.cpp` | Worker discovery, bootstrap and exit |
| `runtime/scheduler/scheduler_admission.h` | Unique admission, publication and member frontier |
| `runtime/scheduler/scheduler_protocol*.h` | Persistent candidates, claims, entry permits and closure |
| `runtime/scheduler/scheduler_gm_mix.h` | Group descriptor, lane service and close seal |
| `runtime/scheduler/scheduler_completion.h` | Block aggregation, logical completion and successor routing |
| `runtime/scheduler/scheduler_memory.h` | Atomic, bypass signal and cache publication primitives |
| `runtime/scheduler/scheduler_error.h` | First-error ownership and contextual failure publication |
| `aicore/aicore_executor.cpp` | Per-core scheduling and final kernel entry checks |
| `host/scheduler_protocol_export.h` | Frozen graph, protocol events and terminal state export |

The implementation retains two local candidates, not a per-core graph replica.
Unchanged topology facts may be cached for a run; permissions and ownership may
not be inferred from stale observations. Runtime failures stop new admissions
without fabricating successful completion or reclaiming still-accessed storage.

See [validation](spmd/validation.md) for executable coverage and
[performance](spmd/performance.md) for retained optimizations and timing limits.
