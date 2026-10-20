# `qwen3_14b_decode/` — Qwen3-14B 40-layer decode (CANN fused attention)

> **a5 port.** This is the a5 port of
> [`examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/`](../../a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/).
> The kernels are the a2a3 harvest with the arch deltas listed in
> "Relationship to the a2a3 example" below — the math is untouched, but the
> a2a3 harvest is not byte-portable: a5 spells the L0A fractal, the legal
> `pipe_barrier` pipes and the core-count accessor differently, and the a2a3
> `__DAV_C220_*` guards name a macro ccec does not predefine on a5.

Self-contained standalone-driver port of pypto-lib
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
workspace. The run needs no `runtime_env` tuning — the default ring heap, task
window and dep pool carry the 40-layer graph, because each layer's
intermediates live inside that iteration's scope and are freed at its end, so
the live set does not grow with layer count.

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

The harvest is a2a3 codegen, so re-apply the deltas from "Relationship to the
a2a3 example" afterwards. The two mechanical ones are one pass each:

```bash
cd examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode
grep -rlZ '__DAV_C220_' kernels | xargs -0 sed -i \
    's/__DAV_C220_VEC__/__DAV_VEC__/g; s/__DAV_C220_CUBE__/__DAV_CUBE__/g'
grep -rlZ 'pipe_barrier(PIPE_\(V\|MTE1\|FIX\));' kernels | xargs -0 sed -i \
    '/^[[:space:]]*pipe_barrier(PIPE_\(V\|MTE1\|FIX\));[[:space:]]*$/d'
```

The Left-tile fractal and the `fai_body.hpp` edits (include order, soft Mix
`SYNCALL`) are not mechanical — take them from this tree's history.

One deliberate deviation from `kernel_config.py`: `decode_fwd_layers` declares
`k_cache` / `v_cache` as plain inputs, but the extern writes the current token's
KV into them. The callable marks them `INOUT`; correctness runs explicitly copy
the pools back so the golden can verify all 40 layers' KV writes, not just the
hidden output.

## Running

```bash
# pytest (hardware; wrap in task-submit on shared boxes)
pytest examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode \
    --platform a5 --device ${DEVICE} --manual include

# standalone correctness
python examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py \
    -p a5 -d ${DEVICE}

# benchmark: one fixture upload, then device-only parameter reuse across rounds
python examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py \
    -p a5 -d ${DEVICE} --rounds 100 --skip-golden
```

DFX is opt-in on the standalone driver — no kernel changes needed:

```bash
python .../qwen3_14b_decode/main.py -p a5 -d ${DEVICE} \
    --enable-dep-gen

python .../qwen3_14b_decode/main.py -p a5 -d ${DEVICE} \
    --enable-chip-swimlane 1
```

Note that dependency generation or chip swimlane on the full 40-layer graph can
overflow the per-run SHM record buffer ("records dropped"); pypto-lib warns
about the same thing for `decode_fwd.py --fwd-layers`. Capture on a smaller
harvest if you need a clean trace.

## Status — a5 port

Passes on a5 silicon against the same torch reference as the a2a3 source, at
`RTOL=5e-2 / ATOL=1e-1`: the hidden output **and all 40 layers' K and V caches**
match, with every element inside tolerance (worst element `out` 0.0625,
`k_cache` / `v_cache` 0.03125 — one bf16 quantum at those magnitudes).

## Memory and round behavior

All 20 entry tensors are allocated once with `Worker.malloc`. In
`--skip-golden` benchmark mode, the 19 inputs are generated and uploaded one at
a time, so the host never retains the complete 38 GiB fixture. The flag still
uploads valid model weights, KV cache, and paging metadata; it skips only the
torch reference and explicit D2H comparison. Every round reuses the same device
addresses and does no automatic parameter staging or copy-back.

Correctness mode materializes the host fixture once, uploads that same fixture,
computes the torch golden in place, then reads back `out`, `k_cache`, and
`v_cache` one at a time after the last round. The freshly allocated `out` buffer
is not uploaded because the final `copy_out` task overwrites every element.

## Historical cost

Before the device-resident migration, the measured breakdown below came from
the a2a3 source on one device:

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

## Relationship to the a2a3 example

The kernels are the a2a3 harvest with the arch deltas below. The math — tiling,
loop structure, GM offsets, the KV write's BSND row arithmetic — is identical,
so a refresh is still "re-harvest a2a3, then re-apply this table".

| Delta | Where | Why a5 needs it |
| ----- | ----- | --------------- |
| Left(A) tile `BLayout::RowMajor` → `ColMajor` (×288) | AIC kernels | a5 cube takes L0A in the ColMajor fractal: pto-isa's own `TileLeft` alias is `BLayout::ColMajor` off a2a3, and `CheckMadValid` asserts `!TileLeft::isRowMajor`. SFractal stays `RowMajor`, and the L1 `Mat` tiles the `TEXTRACT` reads from are unchanged |
| drop `pipe_barrier(PIPE_V)` (×135), `pipe_barrier(PIPE_MTE1)` (×32), `pipe_barrier(PIPE_FIX)` (×3) | AIV + AIC kernels | a5 `pipe_barrier` accepts only `MTE3` / `M` / `ALL`; anything else is a compile error. Nothing is lost: a5's own `pto::Event` emits no barrier for a same-pipe V event either, and every cross-pipe order here is already carried by a `set_flag` / `wait_flag` pair |
| `__DAV_C220_{VEC,CUBE}__` → `__DAV_{VEC,CUBE}__` (3 files) | attention extern | `__DAV_C220_*` is the a2a3 (`dav-c220`) ccec predefine. a5 builds as `dav-c310`, so on a5 those guards silently compiled out the whole RoPE/QK-norm/KV-write phase **and** the vendor FAI's cube and vector bodies — the task ran to completion writing nothing. `__DAV_VEC__` / `__DAV_CUBE__` are predefined on both arches and are what every other kernel here already uses |
| `#include <pto/pto-inst.hpp>` moved after the vendor header | `kernel/fai_body.hpp` | `pto` and `KernelCommon` both declare `NUM_32` / `NUM_128` / `NUM_256` at namespace scope and `tensor.h` opens `namespace pto`, so including pto first makes those names ambiguous inside the vendor kernel |
| hard `AscendC::SyncAll<false>` → `pto::SYNCALL<Soft, Mix>` | `kernel/fai_body.hpp`, `kernel/metadata_layout.h`, `tiling/entry.cpp` | the hard barrier is the FFTS flag region, whose base a2a3 gets from `rtGetC2cCtrlAddr`; that call is unsupported on a5 and simpler's a5 runtime sets no FFTS base, so the barrier never releases. The soft barrier's counter (`kSyncAllCounterWord`, one int32 past the kernel's own 48 slots) is zeroed by `clear_barrier` alongside them — it is a monotonic ticket counter, so a non-zero start splits the first round's arrivals across a participant boundary and hangs the half that lands above it |
| `platforms: ["a5"]`, `ring_dep_pool=65536` | test `CASES` | the 40-layer graph's in-flight dependency footprint (~32 k) overflows a5's default 16 k pool |

The vendor `FusedInferAttentionScore` tree is otherwise untouched, including its
`Arch::AtlasA2` budget (UB 192 KiB, L1 512 KiB). a5 has L1 512 KiB and UB 248 KiB,
so that budget is valid there — conservative on UB, not wrong.

One arch-specific knob carries over unchanged: `attention_core_num` in
`kernels/orchestration/decode_fwd_layers.cpp` (the AIC `block_num` the fused
attention task spans). The a2a3 value is **24** = the a2a3 per-device AIC count.
On a5 a device exposes **36 AIC** (2 dies × 18), so 24 is still a valid
`block_num` — it runs, just leaving 12 cores idle. Raising it toward 36 for full
utilization is a tuning follow-up, not a correctness requirement. It must stay
**≥ 16**: the RoPE phase runs on `block_idx * 2 + sub_block_idx` lanes and needs
all `kQwenRopeCores == 32` of them.
