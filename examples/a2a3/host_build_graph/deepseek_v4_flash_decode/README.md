# DeepSeek-V4 FLASH decode on host_build_graph

The `tensormap_and_ringbuffer` DeepSeek-V4 case
(`examples/a2a3/tensormap_and_ringbuffer/deepseek_v4_flash_decode/`) run under
`host_build_graph`: same 43-layer network, same 368 kernels, same fixture, same
comm-window protocol. Only the runtime changes — HBG compiles the orchestration
with the host `g++`, runs it on the host CPU instead of the AICPU, and ships the
built shared-memory image to the device, which then boots scheduler-only.

The case exists to measure **host-side graph construction** and to prove the
device can execute what the host built. It is deliberately not a numerics test.

## What differs from the TMR case

`kernels/orchestration/decode_fwd_graph.cpp` — the file the test points at — is
the TMR orchestration recast as a Graph, with **no runtime-specific rewrite left**:
the whole forward pass is cut into Graph blocks, so a layer's task set becomes a
Graph body (a free function reading its per-layer views, scales and indices
through `GraphTaskArgs`, positionally) and the host records it once per identity
instead of submitting the pass's ~15600 tasks individually. The runtime is
untouched.

Eight Definitions cover all 43 layers:

- `csa_attn_block` (50 nodes) / `csa_moe_block` (32) and `hca_attn_block` (35) /
  `hca_moe_block` (31) — the decoder loop's two alternating layer shapes, layers
  2..41, plus layer 42 replaying `csa_attn_block` and `hca_moe_block`.
  `csa_moe_block` records two Definitions: its routing kernel is `route_hash_1`
  at layer 2 and `route_sort` from layer 4 on, and a body is recorded once per
  key, so the predicate is a Graph config value rather than a host-side `if` the
  first recorded layer would settle for every replay.
- `swa_attn_block` (28) — the two peeled sliding-window attentions of layers 0
  and 1. Their nodes are pairwise alpha-equivalent, so layer 1 replays what layer
  0 recorded.
- `hash_moe_l0_block` (31) / `hash_moe_l1_block` (31) — the peeled MoE scopes.
  These cannot share a Definition: `dispatch_wait` folds the MoE epoch in as a
  constant (32 at layer 0, 64 at layer 1) where the loop's variants take it as a
  scalar.

The read that used to force a rewrite was `recv_count_out[expert][0]`, driving
the ten MoE per-expert tile loops. HBG builds the whole graph before the device
runs anything, so a read of a **task-produced** tensor has no value to return
here, and the ten sites stood a constant in. Both runtimes now express that
control flow as **dispatch predicates** instead: a static per-expert tile
grid, with each of a tile's six tasks predicated on
`recv_count_out[expert][0] > t0`. The scheduler evaluates it at the dispatch
point, on device, where the value is current — so the same source is correct
under both runtimes and the HBG copy no longer encodes a routing the fixture
does not have. The one value the loops computed from the count,
`valid_rows = min(count - t0, 16)`, moves into the `exp_gate_up_act*` kernels
the same way the scale reads did: the orchestration passes `recv_count_out` as
an extra tensor input and each kernel derives the row count from GM in
`kernel_entry`.

The only `get_tensor_data` read left is `ext_num_tokens_per_owner`. It is an
**external** tensor, which the runtime stages with a host view, and it feeds
`set_block_num` — a launch parameter a predicate cannot express, because a
predicate decides whether a task dispatches, not how wide it is. The 30 former
`hc_attn_scale_*` / `hc_ffn_scale_*` reads moved data, not control flow, and are
gone from both runtimes: each `split_pre_post*` / `comb_sinkhorn*` kernel now
takes the scale view as an extra tensor input and reads its elements from GM
itself.

The six former orchestration-side initializations are now identical under both
runtimes. Each `sh_gate_up_act_q*` producer clears its own two padded
`h_tile_i8` rows, while a dedicated AIV seed task clears `mixes_raw` before the
split-K `hc_head_linear` AtomicAdds. The host therefore never writes a GM-heap
device address.

Everything else — submit order, dependencies, scope nesting — is
byte-identical to the TMR source inside the Graph body. The graph keeps the
size and shape of the real one.

`skip_golden` is inherited from the TMR case, which is itself a
completion/smoke case: no full-network torch reference exists upstream either,
and component-level goldens live with the standalone kernels in pypto-lib.

## Status: the host records the Definitions and the device replays them

With the Graph form the host records the eight Definitions and boots with **129
submissions from the submitting thread**, two orders of magnitude below
submitting every task individually — this is measured, on both ranks. The device
replays all of them and both ranks reach `outcome=0`.

`skip_golden` still means this establishes completion, not numbers.

Getting the replay running took three fixes. Each is a way a Graph can differ
from the same body submitted task by task, so they are worth keeping written
down:

1. **The recorder inferred dependencies from the allocation site, not the last
   writer.** A recorded node's fanin came only from tensor args classified
   `INTERNAL` — which names whichever node's packed window holds the bytes, i.e.
   the allocator — plus explicit `set_dependencies`. Every write-then-read
   through an `alloc_tensors` buffer or a boundary view was therefore unordered,
   and a Definition replayed a DAG the body does not have when its tasks are
   submitted individually. Measured on the pre-split single-Definition form of
   this body: 1348 edges against the 2143 the ordinary path computes for the same
   tasks, 543 of 561 comparable nodes short. On device that ran
   `csa_slots_build_valid_qk_plan` before the `topk` that fills its input, so
   `qk_pv_1` gathered KV pages at addresses the bus rejected. The recorder now
   runs the same `compute_task_fanin` / `register_task_outputs` the ring path
   runs, against a tensor map owned by the recording.
2. **Two bodies read 64-bit comm handles back as `int32_t`.** `csa_moe_block` and
   `hca_moe_block` bound `recv_*_ctx`, `*_arrived_ctx` and `routed_y_buf_ctx` as
   `int32_t` while the entry passes `uint64_t`; the peeled `hash_moe_l*_block`
   bodies already had it right. The MoE all-to-all pushed to a truncated window
   address, which surfaces as an AIV MTE bus fault rather than a wrong number.
3. **A host-side branch inside a body was settled at record time** — the
   `route_hash_1` / `route_sort` choice described above.

The `sched_error_code=5 INVALID_ARGS` this section used to report came from a
fourth, separate defect: `bind_graph_topology` bounded the Graph *boundary*
scalar count by `MAX_SCALAR_ARGS`, the per-AICore-task cap (16), rather than the
`GRAPH_MAX_SCALAR_ARGS` (64) the recorder's boundary is built with. Cutting the
pass into blocks brought every boundary under 16 and hid it; it is fixed in the
runtime, so a wider boundary is legal again.

### Why `hc_head_linear` carries a row-tail bound

`x_flat [8,16384]` is a valid 512 KiB allocation, but the kernel's two TLOAD
views were hard-coded as `Shape<...,16,256>` with a 16384-element row stride.
They covered an almost 1 MiB address range and read rows 8-15 out of bounds.
HBG allocates each tensor at its exact size, which surfaces the over-read as an
MTE out-of-range fault; TMR's retained bump allocation keeps it inside a larger
mapping and masks the same kernel bug, so the kernel completes there.

The kernel derives

```text
row_base = (block_idx / 16) * 16
valid_rows = clamp(t_dim - row_base, 0, 16)
```

and uses it for the two `x_flat` views and TLOAD tiles, the dependent
matmul/accumulator tiles, and the `mixes_raw` AtomicAdd store. It is eight for
this invocation, so no instruction addresses a non-existent input row. With that
bound in place the full two-rank device body ran to completion under a
submit-every-task form of this orchestration, both ranks `outcome=0`
(`task_20260817_204500_135435017977`, `task_20260817_210320_91023923204`).

## Runtime gap this case exposed

This gap is independent of the `hc_head_linear` MTE fault:

- **`get_tensor_data` on a task-produced tensor burns its full timeout.** The
  wait can never be satisfied in this runtime — the device does not execute until
  orchestration finishes — yet `wait_for_tensor_ready` spins the whole 15 s before
  failing. The condition is decidable at the call.

## Running

```bash
# standalone (2 dies; wrap in task-submit on a shared box)
python examples/a2a3/host_build_graph/deepseek_v4_flash_decode/\
test_deepseek_v4_flash_decode.py -p a2a3 -d <d0>,<d1>

# pytest
pytest examples/a2a3/host_build_graph/deepseek_v4_flash_decode \
    --platform a2a3 --device <d0>,<d1>
```

The case participates in the default Per-PR collection. It remains
`skip_golden` because no full-network torch reference exists upstream.

To exercise only the host side without launching the device body, set
`SIMPLER_SKIP_DEVICE_RUN=1`. `simpler_launch_run` then completes the run before
any execution claim — orchestration, graph recording, image relocation and the
SM H2D all ran during prepare, the kernel launch and its completion wait do not
happen — and `simpler_finalize_run` still releases the run's resources. The
check sits at the launch entry because the multi-chip subprocess drives a run
through the split `prepare/launch/wait/finalize` entry points and never calls
`simpler_run`. No outputs are produced, so a run under this variable is a timing
harness, not a test.

To collect scheduler diagnostics for a regression, raise the device log level —
the per-task dump is `LOG_INFO` and the default threshold does not open CANN's
INFO stream:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1     # before Worker.init()
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/<run>/ascend"   # dir must pre-exist
```

## Provenance

Kernels, fixture and orchestration come from the TMR case; see its
[README](../../tensormap_and_ringbuffer/deepseek_v4_flash_decode/README.md) for
network shape, regeneration steps and cost. One orchestration file is specific
to this case: `kernels/orchestration/decode_fwd_graph.cpp`, the TMR
orchestration carrying the rewrite in the table above and recast as a Graph —
the forward pass is cut into the eight Definitions listed above, each layer's
task set forming a Graph body whose per-layer views, scales and indices cross
the boundary positionally.
