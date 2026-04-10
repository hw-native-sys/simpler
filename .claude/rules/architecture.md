# Architecture Quick Reference

See [docs/architecture.md](../../docs/architecture.md) for the full diagram, API layers, execution flow, and handshake protocol.

## Key Concepts

- **Three programs**: Host `.so`, AICPU `.so`, AICore `.o` — compiled independently, linked at runtime
- **Three runtimes** under `src/{arch}/runtime/`: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`
- **Two platform backends** under `src/{arch}/platform/`: `onboard/` (hardware), `sim/` (simulation)

## Build System Lookup

| What | Where |
| ---- | ----- |
| Runtime selection | `kernel_config.py` → `RUNTIME_CONFIG.runtime` |
| Per-runtime build config | `src/{arch}/runtime/{runtime}/build_config.py` |
| Runtime build orchestration | `examples/scripts/runtime_builder.py` → `runtime_compiler.py` → cmake |
| Pre-build all runtimes | `examples/scripts/build_runtimes.py` (invoked by `pip install .`) |
| Platform/runtime discovery | `examples/scripts/platform_info.py` |
| Kernel compilation | `python/kernel_compiler.py` (one `.cpp` per `func_id`) |
| Python bindings | `python/bindings/` (nanobind extension for ChipWorker, task types) |
| Pre-built binary lookup | `build/lib/{arch}/{variant}/{runtime}/` |
| Persistent cmake cache | `build/cache/{arch}/{variant}/{runtime}/` |

## Example / Test Layout

```text
my_example/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS list + ORCHESTRATION dict + RUNTIME_CONFIG
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p <platform>`

Add `--build` to recompile runtime from source (incremental). Without it, pre-built binaries from `build/lib/` are used. See [docs/developer-guide.md](../../docs/developer-guide.md#build-workflow) for the full rebuild decision table.
