# Architecture Quick Reference

See [docs/architecture.md](../../docs/architecture.md) for the full diagram, API layers, execution flow, and handshake protocol.

## Key Concepts

- **Three programs**: Host `.so`, AICPU `.so`, AICore `.o` — compiled independently, linked at runtime
- **Three runtimes** under `src/{arch}/runtime/`: `host_build_graph`, `aicpu_build_graph`, `tensormap_and_ringbuffer`
- **Two platform backends** under `src/{arch}/platform/`: `onboard/` (hardware), `sim/` (simulation)

## Build System Lookup

| What | Where |
|------|-------|
| Runtime selection | `kernel_config.py` → `RUNTIME_CONFIG.runtime` |
| Per-runtime build config | `src/{arch}/runtime/{runtime}/build_config.py` |
| Compilation pipeline | `python/runtime_builder.py` → `runtime_compiler.py` → cmake |
| Kernel compilation | `python/kernel_compiler.py` (one `.cpp` per `func_id`) |
| Python bindings | `python/bindings.py` (ctypes wrappers for host `.so`) |

## Example / Test Layout

```
my_example/
  golden.py              # generate_inputs() + compute_golden()
  kernels/
    kernel_config.py     # KERNELS list + ORCHESTRATION dict + RUNTIME_CONFIG
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run with: `python examples/scripts/run_example.py -k <kernels_dir> -g <golden.py> -p <platform>`
