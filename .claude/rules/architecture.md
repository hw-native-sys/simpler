# Architecture Quick Reference

See [docs/chip-level-arch.md](../../docs/chip-level-arch.md) for the full diagram, API layers, execution flow, and handshake protocol. See [docs/hierarchical_level_runtime.md](../../docs/hierarchical_level_runtime.md) for the L0–L6 level model and component composition, and [docs/task-flow.md](../../docs/task-flow.md) for end-to-end data flow through the hierarchical runtime.

## Key Concepts

- **Three programs**: Host `.so`, AICPU `.so`, AICore `.o` — compiled independently, linked at runtime
- **Two runtimes** under `src/{arch}/runtime/`: `host_build_graph`, `tensormap_and_ringbuffer`
- **Two platform backends** under `src/{arch}/platform/`: `onboard/` (hardware), `sim/` (simulation)

## Python Package Layout

| Package | Source | What's in wheel | Use for |
| ------- | ------ | --------------- | ------- |
| `simpler` | `python/simpler/` | `task_interface`, `worker`, `env_manager` only | Stable user API at runtime |
| `simpler_setup` | `simpler_setup/` | All files + `_assets/{src,build/lib}` | Test framework, compilers, path resolution |
| `_task_interface` | `python/bindings/` | nanobind `.so` at wheel root | Internal nanobind module |

The 4 files `kernel_compiler.py`, `runtime_compiler.py`, `toolchain.py`, `elf_parser.py` exist in **both** `python/simpler/` and `simpler_setup/` during transition. The `simpler_setup/` copies are authoritative; the `python/simpler/` copies are excluded from wheel via `pyproject.toml::wheel.exclude`. New code must `import` from `simpler_setup.*`, not `simpler.*`, for these four.

## Build System Lookup

| What | Where |
| ---- | ----- |
| Runtime selection | `@scene_test(runtime="...")` on the SceneTestCase class |
| Per-case knobs (aicpu_thread_num, block_dim) | `CASES[*]["config"]` on the SceneTestCase class |
| Per-runtime build config | `src/{arch}/runtime/{runtime}/build_config.py` |
| Runtime build orchestration | `simpler_setup/runtime_builder.py` → `simpler_setup/runtime_compiler.py` → cmake |
| Pre-build all runtimes | `simpler_setup/build_runtimes.py` (invoked by `pip install .`) |
| Platform/runtime discovery | `simpler_setup/platform_info.py` |
| Kernel compilation | `simpler_setup/kernel_compiler.py` (one `.cpp` per `func_id`) |
| Python bindings | `python/bindings/` (nanobind extension for ChipWorker, task types) |
| Path resolution (wheel vs source tree) | `simpler_setup/environment.py::PROJECT_ROOT` |
| Pre-built binary lookup | `build/lib/{arch}/{variant}/{runtime}/` (source tree) or `simpler_setup/_assets/build/lib/...` (wheel) |
| Persistent cmake cache | `build/cache/{arch}/{variant}/{runtime}/` |

## Example / Test Layout

```text
my_example/
  test_my_example.py     # @scene_test class (CALLABLE + CASES + generate_args + compute_golden)
  kernels/
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run via pytest: `pytest examples tests/st --platform <platform>`, or standalone: `python <example_or_test>/test_*.py -p <platform>`.

Add `--build` to recompile runtime from source (incremental). Without it, pre-built binaries from `build/lib/` are used. See [docs/developer-guide.md](../../docs/developer-guide.md#build-workflow) for the full rebuild decision table.
