# benchmark_bgemm on A5

This example runs the `Case0` batched matrix-multiplication graph on the A5
`tensormap_and_ringbuffer` runtime. Each group submits a GEMM task for every
`grid_k` partition and an ADD task that accumulates the partial result into
`C`.

The A5 port intentionally supports only the 128 × 128 tile used by `Case0`.
Unlike the A2/A3 version, which passes
`[tile_size, grid_k, num_groups, incore_loop]` in a configuration tensor, A5
passes those four topology values as scalar orchestration arguments. The A5
incore kernels derive the number of 128 × 128 tiles from their `ChipTensor`
views, so their callable signatures contain only the data tensors.

`Case0` runs in both simulation and on hardware:

```bash
pytest examples/a5/tensormap_and_ringbuffer/benchmark_bgemm --platform a5sim

pytest examples/a5/tensormap_and_ringbuffer/benchmark_bgemm \
  --platform a5 --device 0 --manual include
```

The host-build-graph counterpart references the same orchestration and incore
sources, ensuring both runtimes exercise the same fixed-tile compute graph.

`C` is declared `INOUT` because every ADD task reads the previous partial sum
before writing the updated value. The host therefore initializes `C` to zero
and stages it to the runtime before execution.
