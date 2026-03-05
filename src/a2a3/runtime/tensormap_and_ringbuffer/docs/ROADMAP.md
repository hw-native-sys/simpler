# PTO2 Runtime Roadmap: Advanced Scheduling Features

This document outlines the planned features and architectural changes for the PTO2 runtime system, specifically focusing on advanced cluster-aware and block-level scheduling semantics.

## 1. In-Cluster Function Group Scheduling

**Goal:** Enable co-scheduling of multiple tasks onto the same physical hardware cluster to leverage local interconnects and optimize data locality.

### Concept
An **in-cluster function group** consists of all incore functions submitted between an `allocate_cluster()` and a `free_cluster()` call (or within a managed scope). The runtime treats this group as a co-scheduled unit: every task in the group executes on the **same physical cluster** (identified by a `clusterID`).

### Required Architectural Changes

#### 1. Task Descriptor Extension
The `PTO2TaskDescriptor` will be extended to record function group membership:
- `cluster_id` (int32_t): ID of the allocated cluster (-1 = unconstrained).
- `group_id` (int32_t): Function group identifier.

#### 2. Orchestration API Additions
```cpp
// Allocate a cluster. Blocks if no cluster is available.
int32_t pto2_rt_allocate_cluster(PTO2Runtime* rt);

// Release a cluster back to the free pool.
void pto2_rt_free_cluster(PTO2Runtime* rt, int32_t cluster_id);

// Submit a task constrained to a specific cluster.
void pto2_rt_submit_task_clustered(PTO2Runtime* rt, int kernel_id,
                                    int worker_type, PTOParam* params,
                                    int n, int32_t cluster_id);
```

#### 3. Scheduler Enhancements
- **Cluster ↔ Core mapping**: A static, platform-specific mapping from `cluster_id` to the set of physical cores (e.g., cluster 0 = {AIC0, AIV0, AIV1}).
- **Cluster-Aware Dispatch**: When popping a task, if `cluster_id >= 0`, the scheduler dispatches it *only* to a core belonging to that specific cluster.
- **Cluster Free Pool**: A ring or bitset tracking free clusters to handle allocation and release.
- **Back-Pressure**: `pto2_rt_allocate_cluster` will implement a spin-wait pattern with deadlock detection, similar to the existing task and heap rings.

---

## 2. `block_incore` (SPMD → MPMD) Task Submission

**Goal:** Support executing a single logical SPMD block function as multiple independent MPMD tasks across available cores.

### Execution Model
At the runtime level, the orchestration layer will **expand** a single `block_incore` call (with a specified `block_dim`) into `block_dim` individual tasks, each with a distinct `block_id`.

```cpp
// Orchestration expansion logic
PTO2_SCOPE(rt) {
    for (int bid = 0; bid < block_dim; bid++) {
        // ... build params with make_scalar_param(bid) ...
        pto2_rt_submit_task(rt, KERNEL_FUNC_ID, PTO2_WORKER_VECTOR, params, 4);
    }
}
```

### Future Optimization Path
While the initial implementation will use O(N) expansion (submitting N individual task descriptors), future optimizations may include:
- **Batch Descriptors**: A single descriptor containing a `block_dim` field.
- **Group-Aware Dispatch**: The scheduler scans one descriptor and expands it into `block_dim` hardware dispatches.
- **Shared-Tensor Optimization**: Reducing TensorMap entries by having one entry per logical tensor instead of per-block tensor.

---

## 3. `block_incore` as InCore Function (Cube + Vector)

**Goal:** Allow a `block_incore` function to be a composite subgraph requiring both AIC (Cube) and AIV (Vector) cores working cooperatively on the same data block.

### Execution Model
When combined with cluster allocation, both the cube and vector tasks of each block are pinned to the **same cluster**. This ensures they execute on co-located cores and can utilize local interconnects (e.g., `PIPE_IN`/`PIPE_OUT`) without round-tripping to Global Memory.

```cpp
// Each block runs its cube and vector kernels on the same cluster
int32_t cid = pto2_rt_allocate_cluster(rt);
PTO2_SCOPE(rt) {
    pto2_rt_submit_task_clustered(rt, CUBE_KERNEL, PTO2_WORKER_CUBE, ..., cid);
    pto2_rt_submit_task_clustered(rt, VEC_KERNEL,  PTO2_WORKER_VECTOR, ..., cid);
}
pto2_rt_free_cluster(rt, cid);
```

### Data Structure Impact Summary
- `PTO2TaskDescriptor`: Add `cluster_id`, `group_id`, `block_id`, `block_dim`.
- `PTO2SharedMemoryHeader`: Add cluster free pool tracking.
- **Scheduler**: Cluster-aware dispatch logic and cluster-to-core mapping tables.
