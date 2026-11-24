# benchmark_bgemm — host_build_graph

This is the `host_build_graph` counterpart of the
[`tensormap_and_ringbuffer` benchmark](../../tensormap_and_ringbuffer/benchmark_bgemm/README.md).
It ports only `Case0`, keeping its parameters, callable signatures, and golden
unchanged. The orchestration and incore sources remain in the
`tensormap_and_ringbuffer` example and are referenced here rather than copied,
so both runtimes exercise the same compute graph.

`Case0` is manual on both platforms. Daily includes it; use an explicit
manual selection for a standalone run.

Run `Case0` in simulation:

```bash
pytest examples/a2a3/host_build_graph/benchmark_bgemm --platform a2a3sim --manual include --case Case0
```

Run `Case0` on hardware:

```bash
pytest examples/a2a3/host_build_graph/benchmark_bgemm \
  --platform a2a3 --device 0 --manual include --case Case0
```
