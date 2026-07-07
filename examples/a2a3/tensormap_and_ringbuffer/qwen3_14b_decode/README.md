# `qwen3_14b_decode/` — Qwen3-14B 2-layer decode (load-balanced fused attention)

Self-contained SceneTestCase port of pypto-lib
`models/qwen3/14b/decode_layer.py` entry `decode_fwd_layers` with
`_CHUNK_NLAYERS == 2`: a **fused chunk of two consecutive Qwen3-14B decode
layers** (hidden → hidden, no LM head), with the FP32 inter-layer residual
carry. It lets a simpler developer build and run the case directly — no descent
through pypto-lib / the JIT, no auto-built intermediate artifacts. This replaces
the earlier single-layer `qwen3_14b_decode` example.

## Parameter regime — matches `stress_profile.py`

The fixture mirrors the vLLM serving stress run (`stress_profile.py`):

| Param | Value | Source |
| ----- | ----- | ------ |
| `BATCH` | 16 | `CONCURRENCY` (aligned with decode kernel BATCH=16) |
| `MAX_SEQ` | 5500 | `max_model_len` (KV-pool / RoPE sizing) |
| decode `seq_len` | 3500 | the ~3500-token prompt |
| layers | 2 | fused chunk (`decode_fwd_layers` N=2) |

Per the lib's const-layer-0 stacked-fwd reference, the two layers reuse layer-0
weights (weights + paged KV pool are stacked ×2 along dim 0, one slice per
layer); each layer still reads/writes its own KV pool.

Dataflow per layer (`_decode_layer`): input RMSNorm → split-K SPMD Q/K/V (seed +
atomic-add) → per-head Q/K RMS-norm → RoPE + paged KV write → `fa_work_build` →
persistent grid-stride `fa_fused` → `online_softmax` → split-K out_proj +
residual → post-RMSNorm → SwiGLU FFN → `dcr_xgamma`. `copy_hidden` embeds the
bf16 input; the FP32 residual is carried between layers; `copy_out` does the
single FP32→bf16 round at the chunk tail.

## Provenance — how the C++ was produced

The 36 sources under `kernels/` (orchestration + 35 incores: 8 AIC + 27 AIV) and
the golden `simpler_setup/goldens/qwen3_14b_decode.py` are **harvested pypto
codegen** for `decode_fwd_layers` (`_CHUNK_NLAYERS=2`, `PTO2_MANUAL_MAX_SEQ=5500`):
the orchestration + kernel `.cpp` (license header prepended, otherwise verbatim)
and the `CALLABLE` transcribed from that run's `kernel_config.py`. The golden
ports the per-layer `golden_decode_layer` math (RoPE θ=1e4, controlled scales,
FP32 residual, bf16 cast points) composed over the two layers with FP32 carry +
per-layer KV pools.

The kernels are used essentially as generated, with **one hand-edit** to
`fa_fused_aiv`: the codegen emitted the AIV sub-block id as a `[[block_local]]
static` (`pypto_runtime_subblock_id`) whose **non-branch** `.text` relocation
simpler's strict `.text`-only loader rejects. The fix swaps it for
`setEntryOffset(get_sub_block_id(args) * …)` (legacy workaround). New MIX
kernels should prefer direct `TPUSH`/`TPOP` per
[docs/tpush-tpop.md](../../../docs/tpush-tpop.md). See
Status below.

To regenerate: in pypto-lib, set `_CHUNK_NLAYERS=2`, `PTO2_MANUAL_MAX_SEQ=5500`,
build the ×2-stacked inputs, and `decode_fwd_layers.compile_for_test(...)`; then
harvest `orchestration/` + `kernels/` + `kernel_config.py`. No post-edit needed.

## Running

```bash
# pytest (hardware; wrap in task-submit on shared boxes)
pytest examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode \
    --platform a2a3 --device ${DEVICE}

# standalone
python examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py -p a2a3 -d ${DEVICE}
```

DFX (the `--enable-l2-swimlane` the lib command implies) is opt-in via the
existing flags — no kernel changes needed:

```bash
pytest .../qwen3_14b_decode --platform a2a3 --device ${DEVICE} \
    --enable-l2-swimlane 1 --enable-dep-gen
```

## Status — PASSING

The case passes deterministically on device — output and both layers' KV-cache
match the torch reference (finite `max_abs_diff ≈ 0.03`).

### Sub-block-id handling (the one hand-edit)

The fused-attention tile-pipe (`TPUSH/TPOP<…, TILE_UP_DOWN>`) computes its per-AIV
FIFO offset from the **no-arg** ISA `get_subblockid()`, which under this MIX
dispatch returns **0 for both** AIV0 and AIV1 (confirmed by tensor dump). Left
uncorrected, both lanes collide on the same FIFO half and attention is poisoned →
NaN. The correct lane id (0/1) is only available from simpler's
`get_sub_block_id(args)`.

The codegen's original workaround cached that value once (in `kernel_entry`) into
a per-core `[[block_local]]` static and `#define`d `get_subblockid()` to read it.
That static's `.text` relocation is rejected by `elf_parser` (which rejects
**any** `.rela.text` entry — the guard against unapplied *branch* relocations,
issues #900 / #830 / #831), so the verbatim kernel would not load.

The fix follows the `spmd_paged_attention` pattern: drop the `[[block_local]]`
static and macro, and apply the per-lane split explicitly on the tile-pipe in
`kernel_entry`/`fa_fused_aiv` —
`v36.cons/prod.setEntryOffset(get_sub_block_id(args) * rows * cols * sizeof(T))`.
`get_subblockid()` then stays at its native 0 (the library auto-split contributes
nothing) and the explicit offset carries the lane separation. No per-core static,
no relocation — loads under the strict loader and computes correctly.

The cleaner long-term fix is direct `TPUSH`/`TPOP` with platform-correct
`get_subblockid()` (see [docs/tpush-tpop.md](../../../docs/tpush-tpop.md)
and [`spmd_paged_attention`](../../../../tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/kernels/mix/paged_attention_parallel.cpp)).
