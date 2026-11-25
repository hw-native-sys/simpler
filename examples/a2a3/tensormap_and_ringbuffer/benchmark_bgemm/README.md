# benchmark_bgemm — a shape sweep, not a kernel demo

Tiled batched matmul `C = sum over k of A[k] @ B[k]`, with every dimension that
matters driven from `params`. The arithmetic is ordinary; the directory exists
so you can move one axis at a time and watch what it costs.

| Task | Core | Work |
| ---- | ---- | ---- |
| `GEMM` | AIC | one `tile_size × tile_size` matmul |
| `ADD` | AIV | accumulate that tile into `C` |

## The four axes

| Param | Moves | Effect |
| ----- | ----- | ------ |
| `matmul_add_task_num` | task count | more, smaller units of scheduling — the dispatch-overhead axis |
| `incore_data_size` | tile edge | arithmetic per task, quadratically |
| `incore_loop` | tiles per task | arithmetic per task, linearly, with no extra dispatch |
| `grid_k` | accumulation depth | how many partial products land in the same `C` tile — the dependency-chain axis |

The cases are laid out as single-axis moves from `Case1`, which is what makes
them comparable:

| Case | Default Sim? | Default Onboard? | `task_num` | `data_size` | `loop` | `grid_k` | Reads as |
| ---- | ------------ | ---------------- | ---------- | ----------- | ------ | -------- | -------- |
| `Case1` | no | no | 64 | 128 | 4 | 2 | the reference point |
| `Case2` | no | no | 256 | 128 | 4 | 2 | 4× the tasks |
| `Case0` | **yes** | no | 500 | 128 | 4 | 2 | ~8× the tasks |
| `Case3` | no | no | 64 | 128 | 16 | 2 | 4× the work per task, same task count |
| `Case4` | no | no | 64 | 128 | 4 | 4 | twice the accumulation depth |
| `Bgemm64` | no | no | 32 | 64 | 1 | 4 | small tiles, minimal per-task work |

`Case2` against `Case3` is the interesting pair: both quadruple the total
matmul work relative to `Case1`, one by adding tasks and one by making each
task bigger.

## The default run differs by platform

`Case1`–`Case4` and `Bgemm64` are manual on both platforms. `Case0` is
automatic on `a2a3sim` and manual on `a2a3`. The Sim Per-PR run keeps
`Case0`; the Onboard Per-PR sweep excludes this benchmark. Daily includes
all six cases on both platforms:

```bash
# Sim default — Case0 only, 1 of 6 cases
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim

# Onboard explicit single case — Case0 only
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm \
  --platform a2a3 --device 0 --manual include --case Case0

# all six on either platform
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim --manual include

# Sim manual-only selection — Case1–Case4 + Bgemm64, 5 of 6 cases
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim --manual only
```

If a `--case` selector matches nothing, the error names this as the likely
cause — the selected case is manual and the default mode is `exclude`.

All six run on `a2a3sim` as well as `a2a3`, which makes this one of the few
places you can move these axes without holding a die.

## `C` is INOUT, not OUT

Worth knowing before you copy the `CALLABLE` block. `C` is a zero-initialised
accumulator: the AIV kernel reads it from GM, adds the matmul result, and
stores it back, once per `grid_k` iteration. The host-provided zeros therefore
have to be staged host-to-device, which makes `C` **read-before-write** — an
`INOUT`, not a pure `OUT`. Declaring it `OUT` would skip the H2D stage and
accumulate onto whatever the buffer held.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3 --device 0 --manual include
```

Tolerances are `RTOL = ATOL = 1e-3`, with inputs scaled by `0.01` to keep the
`grid_k`-deep float32 accumulation inside them.

For turning a run into numbers rather than pass/fail, see
[`docs/dfx/README.md`](../../../../docs/dfx/README.md) — in particular the
scheduler-overhead model, which is the thing the task-count axis here is built
to probe.
