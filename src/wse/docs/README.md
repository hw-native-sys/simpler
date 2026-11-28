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
The shared CPU target header identifies the current kernel ABI as `wsesim-v2`.
Recompile prebuilt kernels marked `wsesim` before registering them with this
runtime. The loader checks the marker before registering hooks because the
shared Block slot layout changed. PTO operations retain their A5 CPU semantics.
Kernel caching uses the existing source-include closure and compiler-logic
identity; unrelated WSE headers do not invalidate a kernel's cache entry.

## Device configuration and memory

The device model is defined by
[`platform/config/default_simulation.json`](../platform/config/default_simulation.json).
The default has 40 Cores, four PEs per Core, and 100 MiB SRAM per Core, including
a 256 KiB runtime reservation. Change this build input and rebuild to use another
fixed device model. Configuration is not changed through Python during execution.

Worker initialization establishes a local SRAM window per Core and an optional
global memory window per chip. Use `worker.get_base_addr(coreid=...)` for local
SRAM and `worker.get_base_addr(memory="global")` for chip-global memory. Both support
byte offsets and ordinary Worker copies. See [device memory layout and ownership](sram.md)
for the address and lifetime contract.

## Execution

SS (SuperScalar) is WSE's single-Core unit of computation. Each Block runs
one SS kernel entry, which can perform matrix operations, vector operations,
or both. An SPMD launch runs the same SS program with a distinct `block_id`
for each Block. A Graph expresses dependencies between these launches.

Every run uses all configured Cores. A task's logical Block count comes from
`CoreTaskArgs.launch_spec.block_num()`, independently of the device width.
Host submission supplies a Core ID for every Block. Cores with no work are
idle, and Blocks assigned to the same Core queue there. A Core executes one Block at a
time, and the task completes once all its Blocks have finished.

A Block has one to four logical Threads placed on distinct local PEs. WSE
orchestration specifies Block placement and may provide per-Block arguments. The
Python caller still uses ordinary `worker.submit(...)` or `worker.run(...)`.
See [Block deployment](deployment.md).

At bind, the Host builds a calendar for each Core from ordinary tasks and the
bound Graph invocations. Each kernel Block occupies one row on its assigned
Core. Among Blocks with the same earliest start, the Host favors the longest
remaining dependency path, then the lower task and Block IDs. The functional
simulator assigns each row a unit cost. Graph enter and exit nodes carry
dependencies and occupy zero rows. All Blocks of a producer must retire before
a dependent task can start, even when their Cores differ.

The WSE planner adapts the Host scheduling policy from
[A5 HBG #3262](https://github.com/hw-native-sys/simpler/pull/3262).
The Host binds Graph instances and publishes one calendar per homogeneous
Core. During admission, the PE runtime resolves each Block's arguments into
Core SRAM. Each invocation of a Graph gets its own rows in the run's DAG.
Its execution can overlap independent work subject to the DAG and Core order.

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

With `CallConfig.output_prefix`, WSE emits `calendar.json`, `wse_events.jsonl`
and `wse_sram.json`. The calendar groups planned `task`, `block` and `tick`
rows by Core and lists each node's producer IDs in `waits`. A tick is the
earliest start under the unit-cost model. Execution advances on real Block
completion. Sort `BLOCK_BEGIN` events by `seq` within each Core to compare
actual dispatch with its planned rows. Dependency capture also emits `deps.json`.
`wse_prepare_error.txt` records a PE run-admission failure, including runtime SRAM
capacity exhaustion, before the execution start gate opens. Legacy device PMU,
chip swimlane, scope-stat and argument-dump collectors are unsupported.

## Implementation boundary

WSE implements the existing runtime C ABI in its own Host sources. The explicit
`host_runner`, `runtime_api`, `run_context` and
`launch_transaction` files own WSE execution and memory lifetime. The WSE
Graph submit units retain complete Block access sets while reusing unchanged
common HBG declarations and other graph algorithms.

WSE compiles the A5 PMU, profiling-copy and compiler-info sources into its own
runtime. Its headers also reference shared argument and mailbox definitions.
Include dependencies and the runtime source stamp invalidate caches when shared
inputs change.

WSE predicate scenes reuse four A5 kernel fixtures. The WSE orchestration supplies
the execution Core for each task.
A5-only changes deliberately do not select the independent WSE scene job,
including changes to reused sources. For changes to A5 product sources,
`packaging-matrix` and `ut` build WSE through their default installation, covering
compilation and interface coupling. These build checks cannot detect behavior-only
changes in reused sources or validate WSE use of A5 predicate fixtures.
Such changes still need targeted WSE scene validation; test-only fixture changes do
not select those product-build jobs.

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
user path is direct L2 WSE execution with `host_build_graph`. Supply local device Tensor views from the owning Worker for kernel arguments.
