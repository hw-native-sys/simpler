# Project Layout Quick Reference

How this repo organizes Python packages, the build system, and example / test directories. For the *architecture* the code implements (hardware tiers, software three-program model), see [ascend.md](ascend.md).

## Python Package Layout

| Package | Source | What's in wheel | Use for |
| ------- | ------ | --------------- | ------- |
| `simpler` | `python/simpler/` | Every module in the package | Runtime user API: `Worker`, the task/callable types, and `trace` for putting your own spans on the host timeline |
| `simpler_setup` | `simpler_setup/` | All files + `_assets/{src,build/lib}` | **Also user-facing**, not internal-only: `KernelCompiler`, `SceneTestCase` / `scene_test`, `TensorArg`, `Scalar`, `TaskArgsBuilder`, `ensure_pto_isa_root`, `make_chip_tensor_arg`, plus the `simpler_setup.tools` CLIs. A kernel cannot be compiled without it |
| `_task_interface` | `python/bindings/` | nanobind `.so` at wheel root | Internal nanobind module |

`simpler` exposes `Worker` and the `task_interface` and `trace` submodules lazily
(PEP 562 `__getattr__`), so `from simpler import Worker` works while
`import simpler` does not require the `_task_interface` extension. Importing `simpler` only configures
the Python logger; it performs no native `dlopen`. Once `_task_interface` is
loaded, it owns the process's host-log state. `Worker.init()` seeds that state
before the first fork, and each host runtime module compiles a private logger
implementation and binds it to the state during module initialization.
`simpler.task_interface.ChipTensor` is the GM-address-bearing device descriptor
a `ChipWorker` consumes; `simpler_setup.TensorArg` is the address-free
scene-test arg spec `NamedTuple`. They are separate types in separate
namespaces.

The 4 files `kernel_compiler.py`, `runtime_compiler.py`, `toolchain.py`, `elf_parser.py` exist in **both** `python/simpler/` and `simpler_setup/` during transition. The `simpler_setup/` copies are authoritative and are the ones new code must `import` from (`simpler_setup.*`, not `simpler.*`). Both copies ship: `wheel.packages` takes `python/simpler` whole, and the `wheel.exclude` that once dropped the `python/simpler` copies was removed in #552 — so the duplication is a source-tree convention, not a packaging one.

## Build System Lookup

| What | Where |
| ---- | ----- |
| Runtime selection | `@scene_test(runtime="...")` on the SceneTestCase class |
| Per-case knobs (aicpu_thread_num, runtime_env) | `CASES[*]["config"]` on the SceneTestCase class |
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
    ccu/                 # CCU program definitions, module exports and manifest (optional)
    orchestration/       # Orchestration C++ source
```

A CCU module may export multiple kernels from one source file. Keep its program
definitions under `kernels/ccu/`.

Run via pytest: `pytest examples tests/st --platform <platform>`, or standalone: `python <example_or_test>/test_*.py -p <platform>`.

Tests load pre-built runtime binaries from `build/lib/`. After changing runtime/platform C++, re-run `pip install --no-build-isolation -e .` to rebuild them (incremental via the cmake cache; there is no rebuild-on-import — `editable.rebuild = false`) before re-running. See [docs/developer-guide.md](../../docs/developer-guide.md#when-to-rebuild) for the full rebuild decision table.
