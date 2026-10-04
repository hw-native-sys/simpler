# `qwen3_14b_decode_auto/` — the TensorMap-derived-dependency twin of `qwen3_14b_decode/`

Same network, same numerics, same fixture as
[`qwen3_14b_decode/`](../qwen3_14b_decode/README.md) — read that README for the
model, the parameter regime, the attention extern, provenance and cost. This
directory exists only so the two dependency-derivation modes can be measured
against each other on one workload.

The sibling declares every dependency by hand inside
`SIMPLER_SCOPE(ScopeMode::MANUAL)`, which returns from `compute_task_fanin`
immediately and switches TensorMap off entirely. It therefore exercises no
dependency derivation at all. This one runs under `ScopeMode::AUTO`, so every
WAIT edge is derived from tensor overlap.

## What is actually here

Everything identical to the sibling is **shared, not copied**: the `CALLABLE`
names 27 of its 41 incores as `../qwen3_14b_decode/kernels/…`, so a refresh of
the harvested codegen lands in one place. Only the files that had to change are
local:

| file | why it differs |
| ---- | -------------- |
| `kernels/orchestration/decode_fwd_layers.cpp` | `ScopeMode::AUTO`; banded views; private partials; reducer submits |
| `kernels/aic/{down,out}_proj.cpp` | stores its own partial instead of accumulating |
| `kernels/aic/{gate,up}_proj_4.cpp` | stores its own partial instead of accumulating |
| `kernels/aiv/silu.cpp` | column term dropped from the base pointer |
| `kernels/aiv/residual_rms_cast{,_0,_1,_2,_3}.cpp` | its two stores index relative to the band |
| `kernels/aiv/partials_reduce*.{h,cpp}` | new; sums private split-K partials |

## The two things AUTO needs that MANUAL does not

### 1. Declare each view at the width the task actually touches

A task that declares a whole buffer while writing one column band makes
TensorMap order it against every other band. `down_acc_all` is `[16, 17408]`;
17 k-splits each write their own 1024-wide band but the harvested codegen
declared the parent, so all 17 were serialized. The `.slice()` calls here
narrow each argument to the band its kernel indexes.

**A narrowed view moves the argument's base**, so this only works when the
kernel indexes relative to it. Every kernel listed above that lost a column term
did so for this reason. Two places therefore keep the parent declaration:
`out_proj_0` resolves its band as `(idx / 5) * 512` measured from the buffer
base, and there is no constant to drop because `idx` is `block_idx`-derived.

### 2. Give commutative accumulation somewhere to land

`AtomicAdd` into a shared band is commutative, but `INOUT` cannot say so, so
TensorMap must serialize the writers. Four accumulators — `down_proj` (17
splits), `out_proj`, `gate_proj_4` and `up_proj_4` (5 each) — now write private
partials that a `partials_reduce_*` task sums, which turns an N-deep chain into
one parallel round plus one task.

`out_proj`'s reducer accumulates rather than stores: `out_proj_0`'s SPMD blocks
write the same bands from a `block_idx`-derived offset, so the sum has to join
them. Its split count is per-band, because the direct loop stops at
`N_OUT_DIRECT` and the last band is short — summing a fixed 5 would fold in
slabs no task wrote.

## Measured

One decode step on a2a3, same die; WAIT-edge graph from `--enable-dep-gen`,
critical path = longest path over the WAIT subgraph.

| variant | critical path | per layer | edges |
| ------- | ------------: | --------: | ----: |
| MANUAL sibling | 443 | 11.1 | 23,605 |
| AUTO, parent declarations everywhere | 7,963 | 199.1 | 58,404 |
| AUTO, banded declarations | 1,563 | 39.1 | 66,604 |
| **AUTO, + private split-K partials (this)** | **684** | **17.1** | 87,938 |

Edge count rises as the critical path falls: narrowing a declaration replaces
one long-range edge with several short-range ones. Edge count is not a proxy for
parallelism — the critical path is.

## What the remaining 241 steps are

Both modes walk the same per-layer skeleton; AUTO takes 17 edges where MANUAL
takes 11, and the 6 extra split evenly:

- **120 steps — `gate_proj`, `gate_proj_0..3`.** Five separate SPMD tasks, one
  per k-split, all accumulating into columns `[0, 6144)`. The six blocks *inside*
  each task run in parallel; the five tasks are what serialize. Curable by the
  same private-partial treatment, at the cost of ten near-duplicate kernels whose
  only difference from the sibling's would be `AtomicAdd` → `AtomicNone`.
- **120 steps — the reducers themselves.** MANUAL gets atomic accumulation for
  free: it declares by hand that the 85 `down_proj` tasks are mutually
  independent, so accumulation costs zero critical-path steps. AUTO's best is two
  edges — one parallel round, then the reducer. Closing this needs an argument
  direction that marks a write commutative, so TensorMap can leave the writers
  unordered; no amount of slicing reaches it.

## Running

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode_auto \
    --platform a2a3 --device ${DEVICE} --manual include

# the A/B: add --enable-dep-gen and run the sibling the same way
pytest examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode_auto \
    --platform a2a3 --device ${DEVICE} --manual include --enable-dep-gen
```

Like the sibling it runs in the daily full scene-test sweep, not per-PR CI, and
passes at `RTOL=5e-2 / ATOL=1e-1` against the same golden — output and all 40
layers' KV caches.
