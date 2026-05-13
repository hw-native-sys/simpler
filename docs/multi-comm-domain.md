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
  ChipBootstrapConfig(comm=ChipCommBootstrapConfig(...), buffers=[scratch])
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
  rank, nranks, device_ctx, local_window_base, buffer_ptrs["scratch"]
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
4. `ChipWorker.bootstrap_context()` stores that integer as
   `ChipContext.device_ctx`.
5. L3 orchestration passes `ctx.device_ctx` as a scalar task argument.
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

Today, L3 owns one `ChipBootstrapConfig` per chip.  Examples such as
`examples/workers/l3/allreduce_distributed/main.py` declare:

- `ChipCommBootstrapConfig(rank, nranks, rootinfo_path, window_size)`;
- one or more `ChipBufferSpec` entries, usually a `scratch` window;
- optional host staging information.

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

chip_args.add_tensor(
    ContinuousTensor.make(
        data=ctx.buffer_ptrs["scratch"],
        shapes=(scratch_count,),
        dtype=DataType.FLOAT32,
        child_memory=True,
    ),
    TensorArgType.INOUT,
)
chip_args.add_scalar(ctx.nranks)
chip_args.add_scalar(ctx.device_ctx)
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

`ctx.device_ctx` is different: it is not a tensor buffer that participates in
scheduling, staging, or shape-aware access.  It is only an opaque
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
expose a numeric domain identifier in the first PR.  If the backend needs
numeric IDs, the parent can assign them internally from sorted domain names.
Names only need to be non-empty and unique; the API should not impose
identifier-style spelling rules.

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
The runtime should not derive it from `sum(buffers.nbytes)` in the first PR.
Bootstrap only validates that the declared buffers fit in the actual allocated
window.

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
        domain_rank=0,
        domain_size=2,
        window_size=...,
        buffers=[...],
        # backend-only sub-communicator construction metadata,
        # derived by the parent from CommDomainPlan
    ),
]
```

There should not be a second public domain config that users fill per chip.
Users should not copy the whole `CommDomainPlan` into every chip config.  The
parent is the only place that has the full worker list, so it is the right
place to validate `worker_indices` and derive each chip child's `domain_rank`,
`domain_size`, window layout, and active domain list.

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
window size and symmetric buffer layout, and attaches any backend-only
sub-communicator construction metadata.  The returned list is what the parent
puts into `ChipBootstrapConfig(comm=...)` for that chip.

If the HCCL sub-communicator backend needs the domain's participant list, that
list is backend construction metadata in the derived chip config.  It is not a
kernel-visible rank map and it is not exposed through `ChipCommDomainContext`.
Kernel code communicates only with dense domain ranks.

All domains are bootstrapped eagerly during `Worker.init()`.  After init
returns, every active `ctx.domains[name]` is ready for L3 orchestration.  The
first PR should not add lazy first-use communicator creation.

### Bootstrap Sequence

The handoff from `CommDomainPlan` to chip-worker sub-communicators should be
mechanical and one-way:

```text
L3 Worker(comm_plan)
  validate CommDomainPlan against device_ids
  derive chip_bootstrap_configs[i] =
      ChipBootstrapConfig(
          comm=comm_plan.bootstrap_for_worker(worker_idx=i)
      )
  fork chip child i with chip_bootstrap_configs[i]
        |
        v
chip child i / ChipWorker.bootstrap_context()
  create hidden base communicator once, if comm_plan has any domains
      rank      = i
      rank_size = len(device_ids)
      windows   = none
  for each ChipDomainBootstrapConfig in sorted domain-name order:
      sub = create_subcomm(
          base,
          rank_ids=domain.worker_indices,
          sub_comm_rank_id=domain_rank,
      )
      device_ctx = comm_alloc_windows(sub, window_size)
      local_base = comm_get_local_window_base(sub)
      carve domain buffer_ptrs from local_base
      record ChipCommDomainContext(name, domain_rank, domain_size, ...)
  publish all domain contexts to the parent bootstrap mailbox
        |
        v
L3 parent _wait_for_bootstrap()
  assemble ChipContext(worker_index=i, domains={name: context})
```

`ChipBootstrapConfig(comm=...)` is therefore not user-authored.  It is the
per-chip, derived execution plan for the chip child.  The public plan remains
`CommDomainPlan`; the derived config is just how the parent tells one chip
which sub-communicators and windows to create.

The hidden base communicator is a control-plane object.  It has no exposed
buffers, no `ChipCommDomainContext`, and no entry in `ctx.domains`.  It exists
only so the backend can derive domain communicators from a common L3 rank
space.  A chip that is not a member of a particular domain still joins the
hidden base communicator when the L3 plan has communication domains, but it
does not create that domain's sub-communicator, allocate that domain's window,
or receive that domain's buffer metadata.

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
`device_ids` and `chip_bootstrap_configs`.  The public multi-domain API should
use this same indexing model instead of introducing a second worker reference
mechanism.

`worker_indices` belongs to the domain specification that the parent validates
before bootstrap.  It should not be copied into every active domain context
unless a later API needs to expose membership for host-side introspection.

The old scalar fields on `ChipContext` should be replaced by domain lookups.
Even single-domain code should use `ctx.domains["default"]` instead of
`ctx.rank`, `ctx.nranks`, `ctx.device_ctx`, and `ctx.buffer_ptrs` directly.

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
- skip sub-communicator and window creation on chips not in the domain.

Kernel-side use is intentionally smaller: kernels communicate only in
domain-rank coordinates.  They use `ctx->rankId`, `ctx->rankNum`, and peer
domain ranks.  They do not need worker indices or a rank-map buffer.

## Mechanism 2: Per-Domain Windows

Each domain owns its own communicator handle, stream, `CommContext`, and
window allocation:

```text
chip child
  domain "tp0"
    CommHandle h0
    stream s0
    CommContext* ctx0
    window base b0
    buffers: scratch, signals

  domain "ep0"
    CommHandle h1
    stream s1
    CommContext* ctx1
    window base b1
    buffers: send, recv, signals
```

Windows should not be multiplexed across domains in the first PR, even when
two domains have identical `worker_indices`.  Separate windows make these
invariants simple:

- `ctx->windowsIn[ctx->rankId]` is the base for exactly one domain;
- byte offset `x - local_base` is meaningful only inside that domain;
- signal slots cannot collide across domains;
- teardown can release resources domain by domain.

The Simpler runtime should not define cross-domain concurrency semantics.
Its job is to pass isolated per-domain handles, contexts, and windows to the
kernel.  Whether two active domains make concurrent progress, contend for
links, or serialize internally is HCCL/HCOMM backend behavior.

The allocation order must be deterministic across ranks.  All chips should
bootstrap domains in sorted domain-name order.  If a chip does not belong to a
domain, it omits that domain from `ctx.domains` and does not create that
domain's sub-communicator or windows.  It also receives no buffer layout
metadata for that domain.

## Proposed Backend API

The current 5-function C ABI is not enough for the sub-communicator path.
`comm_init()` can still create the hidden base communicator for the L3 chip
set, but each domain also needs a derived sub-communicator before
`comm_alloc_windows()` can build a domain-local `CommContext`.

That means Simpler needs one explicit domain-creation step in the backend
path, either as a new C ABI entry point or as a dedicated method on
`ChipWorker`.  Conceptually the flow becomes:

```cpp
base handle from comm_init()
  -> per-domain subcomm creation from ChipDomainBootstrapConfig
  -> comm_alloc_windows(subcomm_handle, window_size)
  -> CommContext* + local window base
```

The derived `ChipDomainBootstrapConfig` is the right place to carry the
backend-only inputs for that step:

```cpp
struct ChipDomainBootstrapConfig {
    std::string name;
    std::vector<uint32_t> rank_ids;   // domain membership in L3 worker order
    uint32_t sub_comm_rank_id;        // this chip's dense rank in the domain
    size_t window_size;
    std::vector<ChipBufferSpec> buffers;
};
```

`HcclCreateSubCommConfig` is the backend API that matches this shape best:
it takes a global communicator, the selected rank list, and the dense
sub-communicator rank for the local chip.  The HCCL sub-rank-table creation
path assigns sub-rank IDs from the supplied `rank_ids` order, which matches
Simpler's rule that `worker_indices` order defines `domain_rank`.  Simpler
should call this directly rather than routing the design through
`HcomCreateGroup`, because that helper sorts rank IDs while building group
metadata.  It is a registry and lookup layer, not the conceptual API for
user-defined rank ordering.

`ChipWorker` changes from:

```text
void *comm_stream_;
one active CommHandle accepted by caller
```

to:

```text
std::vector<ActiveCommSession> comm_sessions_;
lookup by name or returned handle
```

The C API still returns opaque `CommHandle`s.  The Python-facing bootstrap API
should not expose raw handles; it should expose domain contexts.  The hidden
base communicator should never be visible in `ctx.domains`.

The public domain config does not need a `rootinfo_path`.  Root-info files are
a backend bootstrap transport detail for the hidden base communicator.  The
multi-domain design should use a sub-communicator path:

```text
internal base HcclComm over the L3 chip children
  -> HcclCreateSubCommConfig(worker_indices, domain_rank)
  -> domain subcomm handle
  -> HcclAllocComResourceByTiling(domain subcomm)
```

This path matches the shape of `CommDomain`: `worker_indices` is exactly a
group list in the parent communicator, and the position in that list is
exactly the dense domain rank.  It avoids running an independent root-info
exchange for every domain, keeps one internal rendezvous for the L3 chip set,
and matches the ProcessGroup/NCCL style used by NVIDIA stacks.

If the backend wants to register the sub-communicator by group name, that
lookup remains an internal implementation detail.  The design does not depend
on it.

Independent root-info communicators and sub-communicators differ only in the
backend bootstrap route:

- independent root-info: each domain exchanges its own `HcclRootInfo` and
  creates a standalone communicator whose rank space is already the domain;
- sub-communicator: the runtime creates one internal base communicator, then
  asks HCCL to derive group communicators from selected base ranks.

The sub-communicator route is the better first target here because the L3
parent already knows every chip child and every `CommDomainPlan`.  Repeating
root-info exchange per domain would recreate global coordination that the
parent already has, and it would make overlapping domains look unrelated to
the backend even though they are all subsets of the same L3 chip set.

Both routes would expose the same `ChipCommDomainContext` and `CommContext`
shape to L3 and PTO-ISA kernels.  Choosing sub-communicators is therefore a
backend construction decision, not a kernel ABI decision.

### Why No Exposed Meta Domain

The first PR should not expose an allocated meta communication domain just to
mirror PyTorch/NCCL's WORLD process group.

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

The base HCCL communicator is still an internal backend detail for the
sub-communicator path, but it should not appear in `ctx.domains`, allocate
exposed windows, or expose buffers to kernels.  It is only a control-plane
parent used to derive the real data domains.

## Proposed Python API

The public L3 constructor should accept one global domain plan, while
`ChipBootstrapConfig` remains the per-chip object sent to chip children.

The user-facing input is a `CommDomainPlan` built from `CommDomain` entries,
for example `comm_plan=CommDomainPlan(domains=[...])`.  Top-level per-chip
`buffers=` is removed from the new communication API; each domain declares
its own symmetric buffers.  The old single-domain
`ChipCommBootstrapConfig + buffers` shape is not preserved.

`comm_plan=None` and `CommDomainPlan(domains=[])` both mean no communication
domains.  In that mode there are no domain windows and no domain buffers.

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

Inside `Worker.init()`, the parent converts `comm_plan` into one
`ChipBootstrapConfig` per chip child.  For chip worker `i`, that derived config
contains a `ChipDomainBootstrapConfig` only for domains whose
`worker_indices` list contains `i`.  The derived config records this chip's
`domain_rank`, `domain_size`, window size, and buffer layout.  It may also
carry backend construction metadata needed to create the HCCL sub-communicator,
but the chip result published back to the parent should expose only the
domain-local context.

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

- the base communicator is created before any domain sub-communicator;
- each domain entry in the derived config is active for this chip;
- non-member chips do not receive that domain entry;
- sub-communicator creation returns a non-null domain handle;
- `comm_alloc_windows` returned a non-null `CommContext*`;
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
for domain in reversed(active_domains):
    comm_destroy(domain.handle)
    destroy domain stream
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
6. Add backend support for creating HCCL sub-communicators from the hidden
   base communicator and each domain's `rank_ids` / `sub_comm_rank_id`.
7. Change `ChipWorker` from one active stream to a session table.
8. Extend bootstrap mailboxes/results to publish a map of domain contexts
   instead of one scalar `rank/nranks/device_ctx`.
9. Update one L3 example to use `ctx.domains["default"]`.
10. Add a two-domain smoke example where the same worker indices use two
    windows and kernels prove that data and signals do not cross domains.
11. Add tests for validation, single-domain migration, sim multi-domain
    windows, and hardware HCCL context fields.

## Open Questions

- What is the maximum practical number of active domains per chip?
- Should window zeroing be a runtime responsibility per domain, or remain an
  example/kernel protocol detail guarded by barriers?
