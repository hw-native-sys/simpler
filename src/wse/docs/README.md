# WSE simulation

`wsesim` is a peer platform of `a2a3sim` and `a5sim`. It implements
`host_build_graph` on the Host and a homogeneous Core/PE executor on the CPU.
There is no AICPU component or WSE-specific Python binding. Applications use the
ordinary `Worker`, `Buffer`, `TaskArgs`, `CallConfig` and `RunHandle` interfaces.

## Build and test

From the repository root, use a project-local virtual environment and the same
build tools as the other simulator targets. Host/PE runtimes use the ordinary
Host C++ compiler with C++17. User simulation kernels use the existing GCC 15
toolchain shared by all simulator platforms.

```sh
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install 'scikit-build-core>=0.10.0' 'nanobind>=2.0.0,<3' 'cmake>=3.20'
pip install --no-build-isolation --config-settings=build.targets=build_package_wsesim -e '.[test]'
python -m pytest examples tests/st --platform wsesim --device 0 --max-parallel 1 \
  --pto-session-timeout 1200 --require-pto-isa --manual exclude
```

Default installation discovers WSE alongside A3/A5 when the Host compilers are available.
`build_package_sim` explicitly builds all supported simulators, including WSE. Individual
platform jobs use their own `build_package_<platform>` target. The WSE job needs
only WSE runtime binaries; foreign kernel images used for rejection tests do not
require loading the foreign runtime.
A5-only changes do not select the WSE scene job. Shared unit-test and lint jobs
retain coverage across platforms.

The installed wheel includes WSE sources, configuration and headers under
`simpler_setup/_assets`, as it does for the other platforms. The platform compiler
uses `simpler_setup.environment.PROJECT_ROOT` to locate them.

WSE kernel callers explicitly pass `core_type=""`; a homogeneous PE has no
AIC/AIV compiler-role macro. The common compiler forwards this value unchanged.
The shared CPU target header identifies these kernels as `wsesim`, while the
pinned compatible PTO operations retain their existing A5 CPU semantics.
Kernel caching uses the existing source-include closure and compiler-logic
identity; unrelated WSE headers do not invalidate a kernel's cache entry.

## Device configuration and memory

The device model is defined by
[`platform/config/default_simulation.json`](../platform/config/default_simulation.json).
The default has 40 Cores, four PEs per Core, and 100 MiB SRAM per Core, including
a 256 KiB runtime reservation. Change this build input and rebuild to use another
fixed device model. Configuration is not changed through Python during execution.

The device owns its SRAM backing for its lifetime. One call to ordinary
`worker.malloc(application_bytes)` acquires a Buffer covering the complete,
contiguous application area. Its per-Core windows have fixed offsets. There is
no individual tensor or per-Core allocation operation. Use Tensor views and the
existing copy offsets to place data, including intermediates, in those windows.
See [SRAM layout and ownership](sram.md) for the exact address and lifetime contract.

## Execution

Every run uses all configured Cores. A task's logical Block count comes from
`CoreTaskArgs.launch_spec.block_num()`, independently of the device width.
The default mapping is `core_id = block_id % core_count`. Cores with no work are
idle; excess Blocks queue on their assigned Core. A Core executes one Block at a
time, and the task completes once all its Blocks have finished.

A Block has one to four logical Threads placed on distinct local PEs. WSE
orchestration may specify Block placement and per-Block arguments, while the
Python caller still uses ordinary `worker.submit(...)` or `worker.run(...)`.
See [Block deployment](deployment.md).

Tensor access sets from all Blocks participate in dependency construction,
including native Graph recording, rebinding and replay. Application Tensor
storage is predeclared; no dynamic `TensorCreateInfo` output allocation is
needed. The vector example demonstrates a four-task DAG with fixed intermediate
views. Tests cover predicates, cross-Core access, counter publication, SharedTile
cooperation, Graph replay and FFN computation.

## Operations and diagnostics

The temporary operation backend lives in `platform/sim/operations`, with public
WSE headers in `platform/include/wse`. It provides an FP32 Tile subset, scoped
SharedTile cooperation, and monotonic counter operations. SharedTile storage
belongs to one active Block; each slot owns its mutex and condition variable; completion invalidates its scopes before releasing
the storage. Other Cores and later Blocks have independent storage.

The current PTO-ISA pin and A5 CPU instruction semantics remain in use for the
compatible PTO kernels. The temporary WSE operations are not a complete native
Linx backend. PTO-ISA #418 is an operation-level integration reference, not a
replacement for WSE scheduling, SRAM ownership or Worker lifetime management.

With `CallConfig.output_prefix`, WSE emits `wse_events.jsonl` and
`wse_sram.json`. Dependency capture additionally emits `deps.json`.
`wse_prepare_error.txt` records a PE run-admission failure, including runtime SRAM
capacity exhaustion, before the execution start gate opens. Legacy device PMU,
chip swimlane, scope-stat and argument-dump collectors are unsupported.

## Implementation boundary

WSE implements the existing runtime C ABI in its own Host sources. The explicit
`wse_host_runner`, `wse_runtime_api`, `wse_native_run_context` and
`wse_launch_transaction` files own WSE execution and memory lifetime. The WSE
Graph submit units retain complete Block access sets while reusing unchanged
common HBG declarations and other graph algorithms.

`host-source-provenance.json` records these derived sources and their maintenance
boundary. `source-copy-manifest.json` records the earlier A5-derived platform and
runtime sources. These are explicit WSE compilation units, not replacement
headers placed ahead of common headers on the include path.

The WSE backend exports `requires_aicpu_executor()` with a zero result. A backend
without this optional query retains the existing mandatory AICPU path.

Shared platform integration is limited to registering `wsesim`, selecting its
compiler inputs, and accepting a runtime without an AICPU binary. Other platform
Worker paths, public data layouts, allocation error handling and cache-input
scope remain unchanged.

## Limits

This is CPU functional simulation, not instruction timing, physical register
allocation or hardware execution. Communication engines, execution-fault
recovery and forced kernel interruption are outside this backend. The supported
user path is direct L2 WSE execution with `host_build_graph`. Whole-device SRAM
ownership does not imply support for arbitrary Host-pointer task inputs; supply
fixed device Tensor views through the ordinary Buffer API.
