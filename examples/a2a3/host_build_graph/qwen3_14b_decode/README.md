# `qwen3_14b_decode/` — Qwen3-14B 40-layer decode (CANN fused attention), host_build_graph

Self-contained SceneTestCase port of pypto-lib
`models/qwen3/14b/decode_fwd.py` entry `decode_fwd_layers` with
`_CHUNK_NLAYERS == 40`: **the whole Qwen3-14B decode stack as one fused
dispatch** (hidden → hidden, no LM head), with the FP32 inter-layer residual
carry. A simpler developer builds and runs it directly — no descent through
pypto-lib / the JIT, no auto-built intermediate artifacts.

The layer loop is a real loop in the generated orchestration
(`for (int64_t i = 0; i < 40; i += 1)`), not an unrolled one, so a 40-layer
chunk costs one extra literal over a 2-layer chunk rather than 20× the kernels.

## Parameter regime — matches `stress_profile.py`

The fixture mirrors the vLLM serving stress run (`stress_profile.py`):

| Param | Value | Source |
| ----- | ----- | ------ |
| `BATCH` | 16 | `CONCURRENCY` (aligned with decode kernel BATCH=16) |
| `MAX_SEQ` | 5500 | `max_model_len` (KV-pool / RoPE sizing) |
| decode `seq_len` | 3500 | the ~3500-token prompt |
| layers | 40 | full model (`decode_fwd_layers` N=40) |

Per the lib's const-layer-0 stacked-fwd reference, every layer reuses layer-0
weights (weights + paged KV pool are stacked ×40 along dim 0, one slice per
layer); each layer still reads and writes its own KV pool.

Footprint at this regime, bf16:

| component | size |
| --------- | ---: |
| weights (×40) | 24.61 GiB |
| paged KV pool (×40) | 13.44 GiB |
| **total fixture** | **38.05 GiB** |

Die HBM is 64 GiB, so it fits with ~26 GiB of headroom for ring heap and
workspace.

**This runtime needs the ring heap raised to 512 MiB**, which the case carries in
its own `runtime_env`. On `tensormap_and_ringbuffer` each layer's intermediates
live inside that iteration's scope and are freed at its end, so the live set does
not grow with layer count and the defaults carry the graph. `host_build_graph`
builds the entire graph on the host before the device schedules anything, so no
task has completed while the graph is being built, the heap tail never advances
off 0, and all 40 layers' intermediates are live at once. Measured on a2a3:

| `ring_heap` | outcome |
| ----------- | ------- |
| 256 MiB (default) | `Task Allocator Deadlock - Heap Exhausted`, tail=0, ~254 MiB of 256 used |
| 512 MiB | passes |

The task window is *not* the constraint: the graph is ~10.6K tasks, inside the
16384 default.

## Dataflow per layer (`_decode_layer`)

input RMSNorm → split-K SPMD Q/K/V (seed + atomic-add) → **`paged_attention_rope_cce`**
→ split-K out_proj + residual → post-RMSNorm → SwiGLU FFN → `dcr_xgamma`.

`copy_hidden` embeds the bf16 input; the FP32 residual is carried between
layers; `copy_out` does the single FP32→bf16 round at the chunk tail.

The attention stage is one **CANN `FusedInferAttentionScore` extern** that
subsumes what used to be seven generated kernels — it folds per-head Q/K
RMS-norm, RoPE, the paged KV write, the flash-attention inner loop and the
online softmax into a single mixed (AIC + 2×AIV) task, gated by an
`AscendC::SyncAll<false>()` FFTS barrier. `paged_attention_tiling_cce` builds
its runtime tiling metadata first.

Paged KV uses vLLM's **BSND** layout: a page holds `[BLOCK_SIZE, KV_HIDDEN]`
ordered `[page, token, kv_head, dim]`, so `slot_mapping[b]` is directly the
row index. (The previous harvest used NSND; the golden was updated to match.)

## Provenance — how the C++ was produced

| component | source |
| --------- | ------ |
| pypto-lib | `45be52c` |
| pypto | `d64380cb` |
| ptoas | `v0.48` |
| pto-isa | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` (`pto_isa.pin`) |

`kernels/orchestration/` + `kernels/aic/` (18) + `kernels/aiv/` (16) are
harvested pypto codegen for `decode_fwd_layers` (`_CHUNK_NLAYERS=40`,
`PTO2_MANUAL_MAX_SEQ=5500`) — license header prepended, otherwise verbatim.
The `CALLABLE` is transcribed from that run's `kernel_config.py`, and
`simpler_setup/goldens/qwen3_14b_decode.py` ports the per-layer
`golden_decode_layer` math (RoPE θ=1e4, controlled scales, FP32 residual, bf16
cast points) composed over 40 layers with FP32 carry + per-layer KV pools.

**There are no hand-edits.** The previous harvest patched `fa_fused_aiv` to work
around a `[[block_local]] static`; that kernel no longer exists (attention is
the extern now), and the current codegen emits no such construct.

### `kernels/vendor/paged_attention_cce/` — the attention extern

Copied verbatim from pypto-lib
`models/qwen3/14b/kernels/paged_attention_cce/`. The tree must stay intact:
`kernel/fai_body.hpp` reaches its dependencies through relative includes, so
splitting it would mean patching the source and re-patching on every refresh.

- `attention_rope/`, `tiling/`, `kernel/`, `generated/` — PyPTO-authored glue.
- `vendor/fused_infer_attention_score/` — **CANN `FusedInferAttentionScore`,
  Copyright (c) 2025 Huawei Technologies, CANN Open Software License 2.0**
  (~16 k LOC), upstream's own vendored copy, left where upstream put it.
- `attention/` is the non-RoPE variant of the extern. `decode_fwd_layers` does
  not use it; it is kept so a refresh is a plain directory copy.

**Why it sits under `kernels/vendor/`.** Nothing below a `vendor/` directory is
ours to reformat: the repo's header, formatting and language lint all skip that
path (`.pre-commit-config.yaml`, `tests/lint/check_headers.py`). Without that,
`clang-format` rewrites the glue files and `end-of-file-fixer` touches the CANN
headers, and the next refresh diffs against *our* reformatting instead of
against upstream — which is how the drift this example is meant to expose starts.
The carve-out keys on the directory, not on this operator's name, so harvesting
another extern is a matter of dropping it in `kernels/vendor/` with no lint
change at all.

Building these needs **CANN devkit headers** (`$ASCEND_HOME_PATH/aarch64-linux/asc/…`,
`tikcpp/…`), declared per-incore via `extra_include_dirs` in the `CALLABLE`.
`$ASCEND_HOME_PATH` keeps them machine-independent; paths a given CANN layout
does not ship are dropped rather than failing the build.

They also depend on simpler linking incore objects before extracting `.text`
([#1497](https://github.com/hw-native-sys/simpler/pull/1497)): AscendC declares
`g_vecTPipePtr` / `g_kfcClient` as block-local globals, whose relocations no
amount of inlining removes.

### To regenerate

From a simpler worktree with pypto + pypto-lib cloned under `build/` (see the
[`multi-repo-setup`](../../../../.claude/skills/multi-repo-setup/SKILL.md)
skill) and `eval "$(pypto-setup --export)"`:

```python
# PTO2_MANUAL_MAX_SEQ must be set before importing decode_fwd (read at import).
os.environ["PTO2_MANUAL_MAX_SEQ"] = "5500"
D = <import build/pypto-lib/models/qwen3/14b/decode_fwd.py by path>
D._CHUNK_NLAYERS = 40          # read at trace time; rebind before the first call
D.decode_fwd_layers(*inputs, out, config=RunConfig(
    platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=OUT))
```

`codegen_only` needs no device. Then copy `OUT/orchestration/`, `OUT/kernels/`
and `models/qwen3/14b/kernels/paged_attention_cce/` into `kernels/vendor/` here, and
re-transcribe `CALLABLE` from `OUT/kernel_config.py` (which already records
`func_id`, `core_type`, per-kernel `signature`, and `extra_include_dirs`).

One deliberate deviation from `kernel_config.py`: `decode_fwd_layers` declares
`k_cache` / `v_cache` as plain inputs, but the extern writes the current token's
KV into them. The `CALLABLE` marks them `INOUT` so simpler copies the pools back
and the golden can verify all 40 layers' KV writes, not just the hidden output.

## Running

```bash
# pytest (hardware; wrap in task-submit on shared boxes)
pytest examples/a2a3/host_build_graph/qwen3_14b_decode \
    --platform a2a3 --device ${DEVICE}

# standalone
python examples/a2a3/host_build_graph/qwen3_14b_decode/test_qwen3_14b_decode.py -p a2a3 -d ${DEVICE}
```

DFX is opt-in via the existing flags — no kernel changes needed:

```bash
pytest .../qwen3_14b_decode --platform a2a3 --device ${DEVICE} \
    --enable-l2-swimlane 1 --enable-dep-gen
```

Note that `--enable-dep-gen` / `--enable-l2-swimlane` on the full 40-layer graph
can overflow the per-run SHM record buffer ("records dropped"); pypto-lib warns
about the same thing for `decode_fwd.py --fwd-layers`. Capture on a smaller
harvest if you need a clean trace.

## Status — PASSING

Passes on device: output **and all 40 layers' KV caches** match the torch
reference at `RTOL=5e-2 / ATOL=1e-1`.

## Cost

It runs in the general `st-onboard-a2a3` scene-test sweep like any other case.
Measured on this repo's a2a3 box, one device:

| Phase | Wall |
| ----- | ---- |
| kernel compilation — 36 incores + 1 orchestration | **59 s** |
| `generate_inputs` (the 38 GiB fixture) | **13 s** |
| `compute_golden` (40 layers, torch, thread-capped) | **~43 s** |
| host→device upload, device run, comparison | the remainder; the device itself is busy for tens of ms |
| | **~115 s** |

Two details behind those numbers. The 38 GiB is mostly replication —
`torch.cat([w] * 40)` of one layer's weights — so only ~2 GB is actually
generated; and the vendor FAI kernel is the compile outlier at 8.4 s for its AIV
variant against ~1.2 s for an ordinary incore.

The golden is thread-capped by the scene-test framework, and that matters more
than its size: its 3584 small slice operations per layer took **6.35 s per layer
at 320 torch threads against 1.05 s at 4**, so on a many-core host the untamed
thread pool cost 5-8x the work it was doing. Before the cap this case held a
device for **406 s** (ci run 30507320146, job 90761014912) and had to be kept out
of the sweep with a `--ignore` and a step of its own.
