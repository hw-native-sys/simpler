# bgemm — one source, two cores, no round trip through GM

Batched tiled matmul `C = A @ B` over a fixed 4×4×4 grid of 64×64 tiles, 2
batches. The arithmetic is unremarkable. What makes this the example to read on
a5 is the directory it keeps its kernel in:

```text
kernels/mix/kernel_bgemm.cpp        # not kernels/aic/ + kernels/aiv/
```

## One file, compiled twice

The same source is registered as **two** incores with different `core_type`s,
and the preprocessor picks the half each build compiles:

| `func_id` | Name | `core_type` | Guard | Does |
| --------- | ---- | ----------- | ----- | ---- |
| 0 | `GEMM` | `aic` | `__DAV_CUBE__` | `TLOAD`, `TMATMUL`, `TPUSH` |
| 1 | `ADD` | `aiv` | `__DAV_VEC__` | `TPOP`, `TADD`, `TSTORE` |

Both entries name the same `source`. Both see the same three-tensor `args[]` —
`args[0] = A`, `args[1] = B`, `args[2] = C` — which is what "cooperative mix"
means here: the halves are not two kernels passing data, they are two views of
one kernel.

**The intermediate `P = A[m,k] @ B[k,n]` never touches GM.** The cube half
`TPUSH`es it into `VEC_FIFO` and the vector half `TPOP`s it straight out. Only
the accumulator `C` is read and written through global memory, once per
`GRID_K` iteration.

That is the whole point of the mix form: a matmul feeding an elementwise
accumulate is the canonical case where a GM round trip for the intermediate is
pure cost.

`"mix"` is a recognised sentinel elsewhere too — `deps_viewer` and
`core_swimlane` collapse a multi-core-type task to it, and
`sched_overhead_analysis` labels it `MIX`.

## `C` is INOUT, not OUT

The vector half `TLOAD`s `C` from GM, adds, and stores it back, `GRID_K` times.
`C` is a zero-initialised accumulator, so those host-provided zeros must be
staged host-to-device — making it **read-before-write**. Declaring it `OUT`
would skip the H2D stage and accumulate onto whatever the buffer happened to
hold.

## Run

```bash
pytest examples/a5/tensormap_and_ringbuffer/bgemm --platform a5sim
pytest examples/a5/tensormap_and_ringbuffer/bgemm --platform a5 --manual include --device 0
```

One case, `default`, on both `a5sim` and `a5` — one of the few a5 examples you
can run without a die. Shapes are compile-time constants (`TILE_* = 64`,
`GRID_* = 4`, `BATCH = 2`) and must match the kernel; the case takes no params.

Wrap hardware runs in `task-submit` on a shared box.

## Compare with

[`examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/`](../../../a2a3/tensormap_and_ringbuffer/benchmark_bgemm/)
is a2a3's matmul example and is **not** a port of this one. It splits AIC and
AIV into separate sources, routes the intermediate through GM, and exists to
sweep shapes rather than to demonstrate the mix form. Same operation, opposite
emphasis.
