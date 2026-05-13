# Multi-Communication-Domain Design

This document describes how Simpler currently wires one communication domain
from L3 workers down to PTO-ISA kernels, then proposes an extension for
multiple communication domains.

The design scope is intentionally narrow:

- use HCOMM/HCCL only to create communication resources and windows;
- use PTO-ISA kernels for all data movement and synchronization;
- do not call HCCL collective kernels such as `HcclAllReduce`;
- make rank mapping and per-domain windows explicit.

## Source Map

The design is based on these implementation points:

| Area | Files |
| ---- | ----- |
| Python API | `python/simpler/task_interface.py` |
| L3 examples | `examples/workers/l3/*/main.py` |
| Kernel examples | `examples/workers/l3/*/kernels/aiv/*.cpp` |
| Chip worker | `src/common/worker/chip_worker.{h,cpp}` |
| Comm ABI | `src/common/platform_comm/comm.h` |
| Device context | `src/common/platform_comm/comm_context.h` |
| Sim backend | `src/common/platform_comm/comm_sim.cpp` |
| Hardware backend | `src/a2a3/platform/onboard/host/comm_hccl.cpp` |
| PTO-ISA comm | PTO-ISA `include/pto/comm/` |
| HCCL/HCOMM | CANN HCCL `inc/hccl/` and `src/` |

The PTO-ISA and CANN HCCL source trees are design references only.  The
proposed Simpler path relies on HCCL/HCOMM setup APIs such as
`HcclCommInitRootInfo`, `HcomGetCommHandleByGroup`,
`HcclAllocComResourceByTiling`, and `HcclCreateSubCommConfig`.
Data movement remains in PTO-ISA kernels through address-based instructions
such as `TPUT`, `TGET`, `TNOTIFY`, and `TWAIT`.

## Why Domains

A communication domain is the set of ranks that can address each other's
window slots through one `CommContext`.

The current examples use one domain:

```text
domain "world"
  group ranks: 0, 1
  per-rank window: scratch
  device context: CommContext*
```

Multi-domain support lets one L3 worker tree describe several overlapping or
disjoint groups:

```text
worker indices:    0        1        2        3

domain "tp":       0 ------ 1        0 ------ 1
                  group A            group B

domain "ep":       0 --------------- 1
                            0 --------------- 1
```

Each domain needs two independent pieces of state:

- rank mapping: who am I in this domain, and which peer slot should I use?
- window state: which shared window and `CommContext*` belong to this domain?

## NVIDIA Stack Comparison

The surveyed NVIDIA path separates semantic domains from communicator
resources:

```text
Megatron parallel strategy
  -> TP / DP / PP / CP / EP rank groups
  -> PyTorch ProcessGroup objects
  -> NCCL communicators
  -> NCCL collective or P2P kernels on CUDA streams
```

The useful mapping for Simpler is:

| NVIDIA stack | Simpler proposal |
| ------------ | ---------------- |
| global rank from launcher | L3 `worker=i` index |
| Megatron TP/DP/PP group | `CommDomain(name, worker_indices)` |
| PyTorch `ProcessGroup` | `ChipCommDomainContext` exposed to L3 |
| NCCL communicator | HCCL/HCOMM communicator handle |
| NCCL group-local rank | `CommContext.rankId` / `domain_rank` |
| NCCL group size | `CommContext.rankNum` / `domain_size` |
| NCCL collective kernel | PTO-ISA kernel using `TPUT`/`TGET`/signals |
| NCCL stream scheduling | Simpler L3 orchestration and chip tasks |

The divergence is intentional.  NVIDIA's stack usually hides data movement
inside NCCL collectives such as AllReduce or ReduceScatter.  This design does
not call HCCL collectives.  HCCL/HCOMM only creates a per-domain communicator
and registered windows; PTO-ISA kernels implement the actual protocol.

Two lessons carry over directly:

- domain membership should be explicit and stable before communicator setup;
- each domain should own an independent communicator resource, so TP-like and
  EP-like traffic do not accidentally share rank numbering, windows, or signal
  slots.

## Current Single-Domain Implementation

This section traces the current single communication domain from L3 Python
workers to PTO-ISA kernels.

### End-To-End Path

```text
Python L3 user code
  Worker(comm_plan=CommDomainPlan([CommDomain("default", ...)]))
        |
        v
Worker(level=3)
  forks one chip child per device
        |
        v
chip child / ChipWorker
  comm_init(rank, nranks, rootinfo_path)
  comm_alloc_windows(window_size)
  comm_get_local_window_base()
        |
        v
ChipContext exposed to L3 orchestration
  domains["default"].domain_rank
  domains["default"].domain_size
  domains["default"].device_ctx
  domains["default"].buffer_ptrs["scratch"]
        |
        v
TaskArgs to each chip task
  tensor: scratch pointer in child memory
  scalar: nranks
  scalar: CommContext*
        |
        v
AIV/AIC kernel
  ctx->rankId, ctx->rankNum, ctx->windowsIn[peer]
  PTO-ISA TPUT/TGET/TNOTIFY/TWAIT on derived addresses
```

The `CommContext` data is installed before any kernel launch.  It is not
assembled by the kernel and it is not passed field by field:

1. `comm_alloc_windows()` asks the platform backend to allocate communication
   resources and per-rank windows.
2. The backend fills a `CommContext` with `rankId`, `rankNum`, `winSize`,
   `windowsIn[]`, and `windowsOut[]`.
3. The backend returns the address of that `CommContext` through
   `device_ctx_out`.
4. `ChipWorker.bootstrap_context()` stores that integer in the default
   domain context.
5. L3 orchestration passes `ctx.domains["default"].device_ctx` as a scalar
   task argument.
6. The AIV/AIC kernel casts that scalar back to `__gm__ CommContext *`.

On hardware, `device_ctx_out` is a real device address.  For MESH topology,
HCCL may return a device context directly.  For RING topology, Simpler parses
HCCL's resource structure on the host, fills this repo's `CommContext`, copies
it to device memory with `aclrtMemcpy`, then returns that device pointer.  On
sim, `device_ctx_out` points to the process-local `host_ctx`; sim kernels
dereference that process-local context while the window data itself lives in a
shared mmap segment.

There are two different kinds of access:

- reading `ctx->rankId`, `ctx->rankNum`, and `ctx->windowsIn[i]` reads the
  local `CommContext` object for this chip;
- using `ctx->windowsIn[peer] + offset` accesses a peer's registered
  communication window.

The second access is only valid because HCCL/HCOMM allocated and registered
those windows as communication resources.  The addresses in `windowsIn[]` are
not arbitrary remote pointers.  Kernels should use them through PTO memory or
communication instructions such as `TLOAD`, `TPUT`, `TNOTIFY`, and `TWAIT`.

For MESH, HCCL's returned device context already has the compatible
`CommContext` layout, so the kernel can read the context in place.  The peer
window addresses inside it are device-visible communication-window addresses.
For RING, HCCL returns a different resource shape.  Simpler copies only a
small, normalized `CommContext` to device memory during bootstrap.  That copy
is not on the kernel hot path; kernels reuse the same device context pointer
for all later task launches in that communication session.

The key invariant is offset preservation.  L3 passes one local window pointer
and one `CommContext*`; the kernel derives peer pointers by applying the same
window offset to another rank's window base:

```cpp
template <typename T>
AICORE inline __gm__ T *CommRemotePtr(
    __gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}
```

The single-domain contract is:

- `ctx->rankId` is this chip's rank in the domain.
- `ctx->rankNum` is the domain size.
- `ctx->windowsIn[i]` addresses rank `i` in that same domain.
- every rank uses the same window layout.
- the same byte offset names the same logical buffer on every rank.

The `scratch` buffer is a named slice of each rank's communication window.
It is not special to HCCL; it is an example-owned mailbox area carved from
`local_window_base`.  Kernels use it for temporary communication state:

- payload staging, such as copying local input into `scratch` so peers can
  read it;
- peer mailboxes, such as one slot per source rank in TP all-reduce;
- signal slots used by `TNOTIFY` and `TWAIT`.

The important property is symmetry.  If `scratch` starts at offset `0` in
rank 0's window, it also starts at offset `0` in every other rank's window.
That lets a kernel compute the peer address from a local pointer:

```text
local scratch pointer -> offset from my window base
peer scratch pointer  -> peer window base + same offset
```

`CommContext` alone is not enough for a kernel to find `scratch`.  It only
knows whole-window bases:

```text
ctx->windowsIn[0] = base of rank 0's whole window
ctx->windowsIn[1] = base of rank 1's whole window
```

It does not know the example's layout inside that window:

```text
window base + 0x0000: scratch payload
window base + 0x4000: signal slots
window base + 0x5000: recv buffer
```

Passing `scratch` tells the kernel which local sub-buffer to use.  Then
`CommRemotePtr()` maps that same sub-buffer to another rank:

```text
local scratch = ctx->windowsIn[my_rank] + 0x0000
peer scratch  = ctx->windowsIn[peer]    + 0x0000
```

If the kernel needs the signal area, it starts from a local signal pointer and
maps that pointer instead:

```text
local signal = scratch + signal_offset
peer signal  = CommRemotePtr(ctx, local signal, peer)
```

So `scratch` names the local logical buffer.  `CommRemotePtr()` does not move
data by itself; it only converts a local window pointer into the peer pointer
at the same offset.  PTO instructions then use those pointers to read, write,
or synchronize.

### L3 Integration

The previous L3 implementation owned one `ChipBootstrapConfig` per chip.
The new public surface expresses the same single-domain case as one
`CommDomain` named `"default"`:

- `CommDomain(name="default", worker_indices=[0, 1], window_size=...)`;
- one or more per-domain `ChipBufferSpec` entries, usually a `scratch`
  window;
- optional per-chip host staging information keyed by domain name.

Current single-domain window sizing is explicit.  The example computes a
requested `window_size`, usually `max(sum(buffer nbytes), floor)`, and passes
it to `comm_alloc_windows()`.  After allocation, the child asks the backend
for `actual_window_size`, because HCCL may round the request.  Named buffers
are then carved sequentially from `local_window_base`; bootstrap fails if any
buffer would exceed `actual_window_size`.

`Worker.init()` forks chip children and waits until each child publishes a
`ChipBootstrapResult`.  The parent turns those results into
`worker.chip_contexts`.

An L3 orchestration function consumes a context like this:

```python
ctx = worker.chip_contexts[i]
default = ctx.domains["default"]

chip_args.add_tensor(
    ContinuousTensor.make(
        data=default.buffer_ptrs["scratch"],
        shapes=(scratch_count,),
        dtype=DataType.FLOAT32,
        child_memory=True,
    ),
    TensorArgType.INOUT,
)
chip_args.add_scalar(default.domain_size)
chip_args.add_scalar(default.device_ctx)
orch.submit_next_level(chip_cid, chip_args, cfg, worker=i)
```

The tensor is marked `child_memory=True` because it is already a device/window
pointer owned by the chip child.  The framework must not stage it through host
memory.

`ContinuousTensor` is the compact tensor argument format used by the
orchestration runtime.  It contains:

- `data`: the base address;
- `shapes`: up to five tensor dimensions;
- `dtype`: element type, used to compute byte size;
- `child_memory`: whether `data` is already memory owned by the chip child.

The scratch window is passed as a `ContinuousTensor` instead of a scalar
because kernels need a normal `Tensor` descriptor: `buffer.addr`,
`start_offset`, shape, dtype, and dependency tags.  The orchestration shim
converts it with `from_tensor_arg()`, then the AIV/AIC kernel reaches the
pointer through:

```cpp
__gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
__gm__ float *scratch =
    reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) +
    scratch_tensor->start_offset;
```

`default.device_ctx` is different: it is not a tensor buffer that participates
in scheduling, staging, or shape-aware access.  It is only an opaque
`CommContext*`, so it is passed as a scalar.

### L2 Integration

The chip child owns one `ChipWorker`.  During bootstrap it runs:

```text
ChipWorker.init(device_id, bins)
ChipWorker.bootstrap_context(device_id, cfg, channel)
  comm_init()
  comm_alloc_windows()
  carve named buffer pointers from local_window_base
  optional H2D staging into window slices
```

The current C++ `ChipWorker` enforces one active communication session:

```text
comm_stream_ != nullptr -> "a comm session is already active"
```

That guard is correct for the old model.  Multi-domain support must replace it
with a per-domain session table.

### Platform Backend

The backend-neutral C API is:

```cpp
CommHandle comm_init(int rank, int nranks, void *stream,
                     const char *rootinfo_path);
int comm_alloc_windows(CommHandle h, size_t win_size,
                       uint64_t *device_ctx_out);
int comm_get_local_window_base(CommHandle h, uint64_t *base_out);
int comm_get_window_size(CommHandle h, size_t *size_out);
int comm_barrier(CommHandle h);
int comm_destroy(CommHandle h);
```

In the current single-domain API, `rootinfo_path` is a filesystem rendezvous
key for communicator bootstrap.  It is not the communication window, and it
is not read by kernels.  It only lets all chip child processes agree on the
same HCCL communicator identity before `comm_alloc_windows()` creates the
device-side `CommContext`.

The hardware flow is:

```text
rank 0:
  HcclGetRootInfo()
  write HcclRootInfo bytes to rootinfo_path

rank 1..N-1:
  wait until rootinfo_path exists
  read the same HcclRootInfo bytes

all ranks:
  file barrier "rootinfo_ready"
  HcclCommInitRootInfo(nranks, rootInfo, rank)
  later, file barrier "hccl_init"
  HcclAllocComResourceByTiling(...)
```

So `rootinfo_path` currently carries two responsibilities:

- root-info exchange: rank 0 publishes the opaque `HcclRootInfo` token that
  all ranks pass to `HcclCommInitRootInfo`;
- bootstrap synchronization: the same path prefixes small barrier files so
  ranks do not allocate communication resources before every rank has
  initialized HCCL.

On simulation, there is no HCCL root info.  `comm_sim.cpp` still accepts
`rootinfo_path`, but only uses it to derive a deterministic POSIX shared
memory name.  That shared-memory segment then holds the sim windows for all
ranks.

On hardware, `comm_hccl.cpp` uses HCCL/HCOMM resources:

- `HcclCommInitRootInfo` creates the communicator.
- `HcomGetCommHandleByGroup` resolves the HCOMM handle.
- `HcclAllocComResourceByTiling` creates communication resources and windows.
- MESH topology can return a device `CommContext` directly.
- RING topology is parsed into this repo's `CommContext` and copied to
  device.

On sim, `comm_sim.cpp` creates one POSIX shared-memory segment:

```text
[ header ][ rank 0 window ][ rank 1 window ] ...
```

The sim `CommContext` points to each rank's local mmap view of every peer
slot.  Numeric addresses need not match across processes, but each process's
own `CommContext` remains valid.

### PTO-ISA Use

PTO-ISA communication APIs are address based.  `TPUT`, `TGET`, `TNOTIFY`, and
`TWAIT` receive `GlobalTensor` or `Signal` objects whose pointers already
encode the target.

PTO-ISA does not know the domain name.  The domain is selected before the
PTO-ISA call, when the kernel chooses:

- which `CommContext*` to read;
- which local window pointer to offset from;
- which peer rank index to use.

This means synchronous PTO-ISA APIs do not need to change for multiple
domains.  Simpler must pass the correct context, window, and mapping into
kernels.

### Existing Example Patterns

`allreduce_distributed` uses one scratch window:

```text
input/output: host-backed per-rank tensors
scratch:      HCCL window
scalars:      nranks, CommContext*
kernel:       stage local input into scratch, notify, TLOAD peer scratch
```

`ffn_tp_parallel` chains two L2 tasks per rank:

```text
stage 1: local AIC matmul writes partial_local
stage 2: AIV reduce reads partial_local and uses scratch window for exchange
```

`ep_dispatch_combine` uses a larger window layout:

```text
pub_counts | signals | recv_x | recv_w | recv_idx | signals |
routed_y_buf | signals
```

All three examples use the same domain mechanism: one `CommContext*`, one
per-rank window, and kernel-side offset translation.

## Proposed Runtime Model

Keep the split between parent-level planning and per-chip bootstrap explicit.
The communication-domain plan is a single L3-level object.  It is not copied
into every chip's `ChipBootstrapConfig`.

The public L3 input is one `CommDomainPlan` object.  The plan contains a list
of `CommDomain` objects:

```python
comm_plan = CommDomainPlan(
    domains=[
        CommDomain(
            name="tp0",
            worker_indices=[0, 1],
            window_size=...,
            buffers=[
                ChipBufferSpec(name="scratch", nbytes=...),
                ChipBufferSpec(name="signals", nbytes=...),
            ],
        ),
    ],
)
```

`CommDomain` describes one domain: membership, logical rank order, window
size, and named buffer layout.  `CommDomainPlan` is the public source of truth
for the whole L3 worker.  The parent validates this plan before forking chip
children.

The domain name is the user-facing identity.  L3 orchestration looks up
domains by name, for example `ctx.domains["tp"]`.  The public plan should not
expose a numeric domain identifier in the initial implementation.  The parent
derives internal backend IDs from sorted domain names.  Names only need to be
non-empty and unique; the API should not impose identifier-style spelling
rules.

The order of `worker_indices` defines domain rank.  For
`worker_indices=[2, 3]`, worker `2` is domain rank `0`, and worker `3` is
domain rank `1`.  The plan should not also carry explicit per-worker rank
IDs.

The buffer layout is symmetric for every participant in the domain.  A buffer
name, size, and byte offset must mean the same thing on every domain rank.
This is required because kernels translate local pointers to peer pointers by
preserving the byte offset within the per-rank window.

Buffer names are scoped to one domain.  Two domains may both define a buffer
named `"scratch"`; callers disambiguate through the domain first, for example
`ctx.domains["tp"].buffer_ptrs["scratch"]`.  Within one domain, buffer names
must be unique.

`window_size` remains explicit, matching the current single-domain behavior.
The runtime should not derive it from `sum(buffers.nbytes)` in the initial
implementation.  Bootstrap only validates that the declared buffers fit in the
actual allocated window.

The parent then derives the per-chip `ChipBootstrapConfig` list internally.
Each chip child receives only the domains that the chip participates in:

```text
L3 plan:
  CommDomain("tp0", worker_indices=[0, 1])
  CommDomain("ep0", worker_indices=[1, 3])

derived chip bootstrap configs:
  worker 0:
    domain "tp0": domain_rank=0, domain_size=2
  worker 1:
    domain "tp0": domain_rank=1, domain_size=2
    domain "ep0": domain_rank=0, domain_size=2
  worker 2:
    no domains
  worker 3:
    domain "ep0": domain_rank=1, domain_size=2
```

The derived per-chip config should be concise:

```python
chip_cfg = ChipBootstrapConfig(
    comm=comm_plan.bootstrap_for_worker(worker_idx=0),
)

# chip_cfg.comm contains:
[
    ChipDomainBootstrapConfig(
        name="tp0",
        sub_comm_id=0,
        domain_rank=0,
        domain_size=2,
        window_size=...,
        buffers=[...],
        # backend-only base-window derivation metadata,
        # derived by the parent from CommDomainPlan
    ),
]
```

There should not be a second public domain config that users fill per chip.
Users should not copy the whole `CommDomainPlan` into every chip config.  The
parent is the only place that has the full worker list, so it is the right
place to validate `worker_indices` and derive each chip child's `domain_rank`,
`domain_size`, window layout, internal backend identity, and active domain
list.

Conceptually, `CommDomainPlan` owns the derivation method:

```python
class CommDomainPlan:
    domains: list[CommDomain]

    def bootstrap_for_worker(
        self,
        worker_idx: int,
    ) -> list[ChipDomainBootstrapConfig]:
        ...
```

The method filters domains that contain `worker_idx`.  For each match, it
uses the position of `worker_idx` in `CommDomain.worker_indices` as
`domain_rank`, uses `len(worker_indices)` as `domain_size`, copies the domain
window size and symmetric buffer layout, and attaches backend-only
base-window layout metadata.  The returned list is what the parent
puts into `ChipBootstrapConfig(comm=...)` for that chip.

The public domain name is not the backend communicator identity.  The parent
derives a deterministic internal numeric `sub_comm_id` for each domain, using
sorted domain-name order.  The initial RMA path mainly needs the same sorted
order to compute `window_offset`, while `sub_comm_id` remains available as an
internal backend identity.  `ChipCommDomainContext` exposes the public name
and domain-local rank information, not `sub_comm_id`.

The domain participant list is backend construction metadata in the derived
chip config.  It is not a kernel-visible rank map and it is not exposed
through `ChipCommDomainContext`.  Kernel code communicates only with dense
domain ranks.

All domains are bootstrapped eagerly during `Worker.init()`.  After init
returns, every active `ctx.domains[name]` is ready for L3 orchestration.  The
initial implementation should not add lazy first-use communicator creation.

### Bootstrap Sequence

The handoff from `CommDomainPlan` to chip-worker domain bootstrap should be
mechanical and one-way:

```text
L3 Worker(comm_plan)
  validate CommDomainPlan against device_ids
  derive internal chip bootstrap config for child i:
      ChipBootstrapConfig(
          comm=comm_plan.bootstrap_for_worker(worker_idx=i)
      )
  fork chip child i with that derived config
        |
        v
chip child i / ChipWorker.bootstrap_context()
  create hidden base communicator once, if comm_plan has any domains
      rank      = i
      rank_size = len(device_ids)
  allocate one base window of base_window_size
  for each active ChipDomainBootstrapConfig in sorted domain-name order:
      derive domain CommContext from base context, rank_ids, and window_offset
      local_base = base_local_window + domain.window_offset
      carve domain buffer_ptrs from local_base
      record ChipCommDomainContext(name, domain_rank, domain_size, ...)
  publish all domain contexts to the parent bootstrap mailbox
        |
        v
L3 parent _wait_for_bootstrap()
  assemble ChipContext(worker_index=i, domains={name: context})
```

`ChipBootstrapConfig(comm=...)` is therefore not the public communication
topology.  It is the per-chip, derived execution plan for the chip child.  In
ordinary L3 usage the parent authors it internally from `CommDomainPlan`.  In
explicit bootstrap usage, such as host staging, the caller may still construct
`ChipBootstrapConfig` manually, but its `comm` list must be derived from the
same parent plan with `plan.bootstrap_for_worker(worker_idx)`.

The hidden base communicator is a control-plane object.  It has no exposed
buffers, no `ChipCommDomainContext`, and no entry in `ctx.domains`.  It exists
only so the backend can expose one common L3 rank space and base window.  A
chip that is not a member of a particular domain still joins the hidden base
communicator when the L3 plan has communication domains, but it does not
allocate that domain's window or receive that domain's buffer metadata.

The visible contract is that non-members do not receive a domain session or a
`ctx.domains[name]` entry.  They still join the hidden base communicator when
the plan has any communication domain, because the base communicator is
created over all chip children.

The parent receives:

```python
ChipContext(
    device_id=...,
    worker_index=...,
    domains={
        "tp0": ChipCommDomainContext(...),
        "ep0": ChipCommDomainContext(...),
    },
)
```

A domain context should contain:

```python
ChipCommDomainContext(
    name: str,
    domain_rank: int,
    domain_size: int,
    device_ctx: int,
    local_window_base: int,
    actual_window_size: int,
    buffer_ptrs: dict[str, int],
)
```

`worker_index` is the same logical worker ID already used by
`orch.submit_next_level(..., worker=i)`.  It also matches the index in
`device_ids` and the parent's internally derived bootstrap list.  The public
multi-domain API should use this same indexing model instead of introducing a
second worker reference mechanism.

`worker_indices` belongs to the domain specification that the parent validates
before bootstrap.  It should not be copied into every active domain context
unless a later API needs to expose membership for host-side introspection.

`ChipContext` exposes communication state only through domain lookups.  Even
single-domain code uses `ctx.domains["default"]`.

## Mechanism 1: Domain Membership and Logical Rank

A domain rank is dense in `[0, domain_size)`.  Domain membership is expressed
with the existing L3 `worker=i` index space.

For every domain, publish the domain membership list:

```text
worker_indices[d] = L3 worker index for domain rank d
```

Example:

```text
worker indices:            0   1   2   3
domain "tp1" workers:              [2, 3]

domain rank 0 -> worker index 2
domain rank 1 -> worker index 3
```

`worker_indices` may be non-contiguous and out of order.  The order is
semantic because it defines domain rank.  For example, `[3, 1]` means worker
`3` is domain rank `0`, and worker `1` is domain rank `1`.

Host-side use:

- validate that every participating chip has a unique domain rank;
- validate that every worker index belongs to this L3 worker;
- compute `domain_rank` from this worker's position in `worker_indices`;
- publish no domain context or window on chips not in the domain.

Kernel-side use is intentionally smaller: kernels communicate only in
domain-rank coordinates.  They use `ctx->rankId`, `ctx->rankNum`, and peer
domain ranks.  They do not need worker indices or a rank-map buffer.

## Mechanism 2: Per-Domain Window Views

Each domain owns an independent logical window view, not necessarily an
independent HCCL MC2 allocation.  On hardware the bootstrap allocates one
hidden base communicator window over all L3 chip children, then derives a
small `CommContext` for each visible domain:

```text
base CommContext over L3 workers
  rankId  = worker index
  rankNum = number of chip children
  winSize = sum(domain.window_size for all domains)
  windowsIn[worker] = base window for worker

visible domain "tp0"
  rankId  = domain_rank
  rankNum = domain_size
  winSize = tp0.window_size
  windowsIn[d] = base.windowsIn[worker_indices[d]] + tp0.window_offset
  buffers = slices inside the tp0 window view

visible domain "ep0"
  rankId  = domain_rank
  rankNum = domain_size
  winSize = ep0.window_size
  windowsIn[d] = base.windowsIn[worker_indices[d]] + ep0.window_offset
  buffers = slices inside the ep0 window view
```

The offsets are deterministic and symmetric.  The parent sorts domain names
and assigns each domain a range inside the base window.  All chips reserve
the same total base-window size, even if a chip does not participate in every
domain.  Non-participant chips do not publish a `ctx.domains[name]` entry.

This keeps the kernel contract exactly domain-local:

- `ctx->windowsIn[ctx->rankId]` is the local base for the selected domain;
- byte offset `x - local_base` is meaningful only inside that domain view;
- signal slots cannot collide because each domain occupies a disjoint base
  window range;
- each domain has its own device `CommContext*`, rank space, and buffer map.

The Simpler runtime should not define cross-domain concurrency semantics.
Its job is to pass isolated per-domain contexts and window slices to the
kernel.  Whether two active domains make concurrent progress, contend for
links, or serialize internally is HCCL/HCOMM backend behavior.

The allocation order must be deterministic across ranks.  All chips should
derive offsets from sorted domain-name order.  If a chip does not belong to a
domain, it omits that domain from `ctx.domains` and receives no buffer layout
metadata for that domain.

## Backend Path

`comm_init()` creates the hidden base communicator for the L3 chip set.
`comm_alloc_windows(base, base_window_size)` allocates one base MC2/shared
window per chip.  `ChipWorker.bootstrap_context()` then derives visible
domain contexts from the base `CommContext`; it does not call
`HcclAllocComResourceByTiling` on overlapping domain sub-communicators.

Conceptually the flow is:

```text
base handle from comm_init()
  -> comm_alloc_windows(base, total_base_window_size)
  -> copy/read base CommContext on the host
  -> for each active ChipDomainBootstrapConfig:
       build domain-local CommContext with dense domain ranks
       write that CommContext to device memory
       carve named buffers from base local window + window_offset
```

The derived `ChipDomainBootstrapConfig` carries the backend-only inputs for
that step:

```cpp
struct ChipDomainBootstrapConfig {
    std::string name;
    uint64_t sub_comm_id;             // deterministic domain identity
    std::vector<uint32_t> rank_ids;   // domain membership in L3 worker order
    uint32_t domain_rank;             // this chip's dense rank in the domain
    size_t window_size;
    size_t window_offset;             // offset inside the hidden base window
    size_t base_window_size;          // same on every chip in the L3 plan
    std::vector<ChipBufferSpec> buffers;
};
```

The public domain config does not need a `rootinfo_path`.  Root-info files are
a backend bootstrap transport detail for the hidden base communicator.  The
visible domains are derived from the parent `CommDomainPlan`, not from
independent root-info exchanges.

The implementation still keeps `comm_create_subcomm` available as a low-level
backend capability because HCCL sub-communicators are useful for future
collective or resource paths.  The initial PTO-ISA/HCOMM RMA path should not
allocate MC2 resources on each overlapping sub-communicator because the HCCL
old AICPU MC2 init path accepts only one active hcomId per process.  Base
window slicing avoids that limitation while preserving domain-local kernel
semantics.

Independent root-info communicators and sub-communicators are not needed for
the first PTO-ISA RMA implementation:

- independent root-info would exchange one `HcclRootInfo` per visible domain
  and create unrelated communicators for overlapping subsets;
- sub-communicators would derive HCCL group communicators from a base
  communicator, but would still need a separate MC2 window allocation if used
  as the visible RMA resource;
- base-window slicing creates one backend window and then derives
  kernel-visible domain contexts from rank selection and byte offsets.

All three routes can produce the same `ChipCommDomainContext` shape.  The
first implementation chooses base-window slicing because it matches the
PTO-ISA kernel ABI and avoids repeated overlapping MC2 allocations.

### Why No Exposed Meta Domain

The initial implementation should not expose an allocated meta communication
domain just to mirror PyTorch/NCCL's WORLD process group.

NVIDIA's stack needs a global process group because multiple OS processes are
launched independently and need a rendezvous space before frameworks can make
TP/DP/PP groups.  Simpler's L3 parent already owns the chip-child list before
communication bootstrap.  The parent has `device_ids`, `worker=i` indices, and
all `CommDomain.worker_indices`, so it can validate membership and derive
domain ranks without exposing a WORLD-like data domain to kernels.

A meta domain would add costs without serving the PTO-ISA kernel contract:

- it would allocate a data-domain window that kernels should not use;
- it would need lifecycle and teardown rules despite being only metadata;
- it could be confused with a real data domain in `ctx.domains`;
- it would not remove the need to create per-domain windows, because each
  data domain needs independent `CommContext` and buffer layout.

The base HCCL communicator is still an internal backend detail.  It should
not appear in `ctx.domains` or expose buffers to kernels.  Its window is
allocated once and sliced into the visible data domains.

## Proposed Python API

The normal public L3 constructor accepts one global domain plan.
`ChipBootstrapConfig` remains the per-chip object sent to chip children.  It
may be derived automatically by the parent from that plan, or supplied
explicitly when a caller needs per-chip bootstrap details such as host staging.

The user-facing input is a `CommDomainPlan` built from `CommDomain` entries,
for example `comm_plan=CommDomainPlan(domains=[...])`.  Each domain declares
only symmetric communication properties: membership, window size, and buffer
layout.

`comm_plan=None` and `CommDomainPlan(domains=[])` both mean no communication
domains.  In that mode there are no domain windows and no domain buffers.

### Public Surface Boundary

The new API is one surface, not a compatibility layer around the old
single-domain bootstrap list.

At L3 and above, the source of truth for communication topology is always
`CommDomainPlan`.  Users should not write rank IDs, root-info paths, or old
single-domain bootstrap objects by hand.  `ChipCommBootstrapConfig` is not a
public API.

There are two valid construction paths:

- `Worker(comm_plan=plan)` for ordinary cases.  The parent derives one
  `ChipBootstrapConfig` per chip.
- `Worker(chip_bootstrap_configs=cfgs)` for cases that need per-chip bootstrap
  data.  Each `cfg.comm` must still come from
  `plan.bootstrap_for_worker(worker_idx)`.

The explicit path exists for host staging.  `HostBufferStaging` contains a
concrete `SharedMemory` name, so it is naturally per-chip and sometimes
asymmetric.  That does not belong in `CommDomain`, which describes symmetric
domain membership and window layout.

Single-domain spelling:

```python
worker = Worker(
    level=3,
    device_ids=device_ids,
    comm_plan=CommDomainPlan(
        domains=[
            CommDomain(
                name="default",
                worker_indices=list(range(len(device_ids))),
                window_size=window_size,
                buffers=[ChipBufferSpec("scratch", ...)],
            ),
        ],
    ),
)
```

Multi-domain spelling:

```python
comm_plan = CommDomainPlan(
    domains=[
        CommDomain(
            name="tp",
            worker_indices=[0, 1],
            window_size=tp_window,
            buffers=[ChipBufferSpec("tp_scratch", ...)],
        ),
        CommDomain(
            name="ep",
            worker_indices=[0, 2],
            window_size=ep_window,
            buffers=[ChipBufferSpec("ep_scratch", ...)],
        ),
    ],
)

worker = Worker(
    level=3,
    device_ids=device_ids,
    comm_plan=comm_plan,
)
```

Host-staged single-domain spelling:

```python
counter_shm = SharedMemory(create=True, size=4)
counter_shm.buf[:4] = b"\x00\x00\x00\x00"

plan = CommDomainPlan(
    domains=[
        CommDomain(
            name="default",
            worker_indices=[0, 1],
            window_size=window_size,
            buffers=[
                ChipBufferSpec(
                    name="notify_counter",
                    dtype="int32",
                    count=1,
                    nbytes=4,
                    load_from_host=True,
                ),
            ],
        ),
    ],
)

cfg = ChipBootstrapConfig(
    comm=plan.bootstrap_for_worker(worker_idx=0),
    host_inputs=[
        HostBufferStaging(
            domain_name="default",
            name="notify_counter",
            shm_name=counter_shm.name,
            size=4,
        ),
    ],
)
```

Inside `Worker.init()`, the parent converts `comm_plan` into one
`ChipBootstrapConfig` per chip child.  For chip worker `i`, that derived config
contains a `ChipDomainBootstrapConfig` only for domains whose
`worker_indices` list contains `i`.  The derived config records this chip's
`domain_rank`, `domain_size`, window size, buffer layout, and internal
`sub_comm_id`.  It also carries rank IDs and base-window layout metadata
needed to derive the domain-local `CommContext`.  The chip result published
back to the parent exposes only the domain-local context.

If host staging is used for communication-window buffers, staging entries live
on the per-chip `ChipBootstrapConfig` and must be domain-scoped.  The key is
`(domain_name, buffer_name)`, not just `buffer_name`, because two domains may
both define a buffer named `"scratch"`.  The child should match host input and
output staging against the active `ChipDomainBootstrapConfig` and that
domain's buffer specs.

Host staging is not a second communication mechanism.  It is parent-to-chip
initialization and chip-to-parent readback for buffers that live inside a
communication window.

`load_from_host=True` means:

```text
parent creates SharedMemory and fills bytes
  -> child attaches by shm_name during bootstrap
  -> child copies those bytes into this chip's domain-local window buffer
```

`store_to_host=True` means:

```text
kernel writes a domain-local window buffer
  -> child copies the buffer back to parent SharedMemory after a task
  -> parent reads the bytes from that SharedMemory
```

This is useful for examples such as notify-counter initialization or
window-resident inputs.  Explicit configs are valid only in the new shape:
`ChipBootstrapConfig(comm=plan.bootstrap_for_worker(i), ...)`.

L3 orchestration chooses the domain explicitly:

```python
tp = ctx.domains["tp"]

args.add_tensor(
    ContinuousTensor.make(
        data=tp.buffer_ptrs["tp_scratch"],
        shapes=(tp_count,),
        dtype=DataType.FLOAT32,
        child_memory=True,
    ),
    TensorArgType.INOUT,
)
args.add_scalar(tp.domain_size)
args.add_scalar(tp.device_ctx)
```

Direct lookup is valid when the orchestration expects this worker to
participate in the domain.  If the worker is not a member, normal dictionary
lookup raises `KeyError`; that is the desired fail-fast behavior.  Code that
intentionally handles optional participation can use `"tp" in ctx.domains`
before submitting a TP-domain task.

Overlapping domains show why this API is not just a renamed single-domain
context:

```python
comm_plan = CommDomainPlan(
    domains=[
        CommDomain(
            name="tp",
            worker_indices=[0, 1],
            window_size=tp_window,
            buffers=[ChipBufferSpec("scratch", ...)],
        ),
        CommDomain(
            name="pp",
            worker_indices=[1, 2],
            window_size=pp_window,
            buffers=[ChipBufferSpec("scratch", ...)],
        ),
    ],
)
```

Worker 1 receives both `ctx.domains["tp"]` and `ctx.domains["pp"]`.  It is
rank 1 in `tp` but rank 0 in `pp`, and each domain has its own
`device_ctx`, window base, and `"scratch"` pointer.  Worker 0 has only `tp`;
worker 2 has only `pp`.  Looking up a non-member domain, such as
`worker0_ctx.domains["pp"]`, raises the normal dictionary `KeyError`.

## Kernel Contract

Kernels should treat a domain as an explicit argument group:

```text
tensor: domain window buffer
scalar: domain_size
scalar: domain CommContext*
```

A kernel may consume more than one domain, but only by receiving more than
one explicit argument group:

```text
tp tensor, tp domain_size, tp CommContext*
ep tensor, ep domain_size, ep CommContext*
```

There is no implicit current domain and no global domain lookup inside the
kernel.  This keeps domain selection visible in L3 orchestration and prevents
accidentally using a pointer from one domain with another domain's context.

For domain-local algorithms:

```cpp
int me = static_cast<int>(ctx->rankId);
for (int peer = 0; peer < static_cast<int>(ctx->rankNum); ++peer) {
    if (peer == me) continue;
    auto remote = CommRemotePtr(ctx, local, peer);
    ...
}
```

Never mix a pointer from domain A with a `CommContext*` from domain B.  The
offset calculation will still produce an address, but it will address the
wrong window.

Simpler should not enforce that every member of a domain submits the same
kernel or task pattern.  The runtime validates resources and domain
membership; protocol symmetry is the kernel/orchestration contract.  This
keeps asymmetric patterns such as dispatch/combine, pipeline handoff, and
producer/consumer communication valid.

## Validation Rules

The parent should validate before forking:

- domain names are non-empty and unique;
- every `worker_indices` list is non-empty and has no duplicates;
- every worker index belongs to this L3 worker;
- each domain has an explicit `window_size`;
- buffer names are unique within each domain;
- every domain size is `<= COMM_MAX_RANK_NUM`.

The child should validate during bootstrap:

- the base communicator is created before any domain context is derived;
- each domain entry in the derived config is active for this chip;
- non-member chips do not receive that domain entry;
- `comm_alloc_windows` returned a non-null base `CommContext*`;
- domain context derivation returns a non-null `CommContext*`;
- every named buffer fits in `actual_window_size`;
- the returned `ctx->rankId` and `ctx->rankNum` match the derived
  `domain_rank` and domain size.

The kernel should validate cheap invariants:

- `domain_size == ctx->rankNum`;
- `ctx->rankId < ctx->rankNum`;
- peer domain ranks are in range before indexing `windowsIn`.

## Teardown

Teardown should run in reverse bootstrap order:

```text
release derived domain contexts
comm_destroy(base handle)
finalize ChipWorker
```

A failure destroying one domain should not skip cleanup of later domains.
The caller should receive the first error after best-effort cleanup completes.

## Implementation Slices

1. Add dataclasses for public `CommDomainPlan`, public `CommDomain`,
   returned `ChipCommDomainContext`, plus an internal derived
   `ChipDomainBootstrapConfig`.
2. Change L3 worker construction to accept `comm_plan` as the single
   parent-level domain plan.
3. During `Worker.init()`, validate the parent plan and derive one
   `ChipBootstrapConfig` per chip child by calling
   `CommDomainPlan.bootstrap_for_worker(worker_idx)`.
4. Update `ChipBootstrapConfig` so its `comm` field contains derived
   per-chip domain bootstrap configs, not public `CommDomain` objects.
5. Remove top-level `buffers` from the new communication API.
6. Add backend support for one hidden base window and derived domain
   `CommContext` objects from each domain's `rank_ids` / `domain_rank`.
7. Change `ChipWorker` to own one hidden base session plus derived domain
   contexts.
8. Extend bootstrap mailboxes/results to publish a map of domain contexts
   instead of one scalar `rank/nranks/device_ctx`.
9. Migrate every communication-domain example to `comm_plan`, including
   notification and SDMA examples that use host staging.
10. Update L3 one-domain examples to use `ctx.domains["default"]`.
11. Add a tiny multi-domain rank-map example that exercises domain
    membership, missing-domain lookup, and domain-local ranks.
12. Add a two-domain data example where overlapping domains run real
    communication, computation, and golden checks.
13. Add tests for validation, the one-domain `CommDomainPlan` case, sim
    multi-domain windows, and hardware HCCL context fields.

## Open Questions

- What is the maximum practical number of active domains per chip?
- Should window zeroing be a runtime responsibility per domain, or remain an
  example/kernel protocol detail guarded by barriers?
