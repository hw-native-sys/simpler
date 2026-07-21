# TPUSH/TPOP Usage Guidelines (Advisory)

Recommended patterns for MIX kernels that use GM FIFO tile-pipe (`TPipe` +
`TPUSH` / `TPOP` / `TFREE`, commonly with `TileSplitAxis::TILE_UP_DOWN`).
New kernels and refactors should follow these by default; special pipelines
may deviate after review. This is not a hard lint.

Implementation reference: `pto-isa/include/pto/npu/*/TPush.hpp`, `TPop.hpp`.

## Why (overview)

pto-isa's `TPUSH` / `TPOP` / `TFREE` already embed record handling,
allocate/back-pressure, and lane offset computation under
`TILE_UP_DOWN`. Re-implementing the same logic in kernel code is usually
redundant and can diverge from the built-in path. Prefer calling TPUSH/TPOP
directly and leave the cross-core FIFO protocol to the library.

## 1. Prefer TPUSH/TPOP directly — do not manual-record

**Why**: `TPUSH_IMPL` / `TPOP_IMPL` call `prod.record()` / consumer wait
after push/pop when `getRecordStatus()` is true (the default). Cross-core
handshake is already inside TPUSH/TPOP.

- Default `TPUSH_IMPL` flow: `allocate` → `push` (TSTORE to GM FIFO) →
  `tileIndex++` → `record()`.
- **Prefer**: call `TPUSH(...)` / `TPOP(...)` only; do not also call
  `pipe.prod.record()`.
- **Prefer**: do not call `setRecordStatus(false)` to disable built-in
  record; combining with manual record is usually redundant and easy to
  miss a signal.
- Pipeline `set_flag` / `wait_flag` and TPUSH record are **independent**:
  the former synchronizes MTE/M/V pipeline stages; the latter handles
  cross-core FIFO handshake. Keep each where needed — this rule does not
  constrain flag usage.
- If C2V AccTile still has record timing issues on a specific pto-isa
  version, prefer upgrading pto-isa or the platform bridge over keeping
  manual record long-term.

## 2. Prefer default back-pressure — do not bulk-disable status flags

**Why**: TPUSH already calls `prod.allocate()` when needed via
`getAllocateStatus()` (default true). The consumer path with `TFREE` uses
consumer free sync. That back-pressure lives in TPUSH/TPOP/TFREE; turning
it off requires the kernel to guarantee slots never overflow.

- Producer default `isAllocate=true`: when the FIFO is full, `allocate()`
  waits for consumer `TFREE`.
- Consumer default enables free sync: after consume, the slot is released
  to the producer.
- **Prefer**: do not batch `setAllocateStatus(false)` / `setFreeStatus(false)`
  at kernel entry unless pipeline analysis proves `FIFO_DEPTH` and scheduling
  order already guarantee safe slot reuse — and document the rationale in a
  comment.

## 3. Rely on TILE_UP_DOWN auto lane offset — do not manual setEntryOffset

**Why**: under `TileSplitAxis::TILE_UP_DOWN`, `TPUSH` / `TPOP` / `TFREE`
compute `subAIVOffset` via `get_subblockid()` inside
`pushVec2GMFiFo` / `popVecTileFromGMFiFo` and related paths. Calling
`setEntryOffset(get_sub_block_id(args) * …)` on top duplicates or stacks
with the library offset.

- Library path: `subAIVOffset = get_subblockid() * tile_bytes_per_lane`
  (expanded by split axis and tile shape).
- **Prefer**: do not use
  `pipe.cons/prod.setEntryOffset(get_sub_block_id(args) * …)` for lane
  splitting; ensure the platform provides a correct `get_subblockid()`.
- **Known issue (current pinned pto-isa)**: on 1C2V (1 Cube + 2 Vector)
  MIX, CCE `get_subblockid()` returns 0 for both AIVs, so library
  `subAIVOffset` cannot distinguish lanes — a pto-isa / launch identity
  bug, not kernel misuse.
- **Onboard bridge (interim)**: when that applies (or simpler dispatch has
  not yet programmed sub-block registers), add the lane split explicitly on
  the tile-pipe with `setEntryOffset(get_sub_block_id(args) * sub_rows * cols *
  elem_bytes)` (`GlobalContext.sub_block_id`) — see the `run_aiv`
  `setEntryOffset` call sites in
  [`paged_attention_parallel.cpp`](../tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp).
  Remove once pto-isa / dispatch fixes `get_subblockid()`.

## Reference examples

- [`tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp`](../tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp)
- [`examples/a5/tensormap_and_ringbuffer/bgemm/kernels/mix/kernel_bgemm.cpp`](../examples/a5/tensormap_and_ringbuffer/bgemm/kernels/mix/kernel_bgemm.cpp)
