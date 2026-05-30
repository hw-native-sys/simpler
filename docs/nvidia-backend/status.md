# CUDA Backend Status

This page tracks current implementation status against the CUDA backend
design. It distinguishes verified tracer-bullet behavior from remaining
design work so evaluation results are not mistaken for a complete backend.

## Implemented And Verified

### Platform And Runtime Discovery

- `cuda` maps to the `cuda/onboard` platform variant.
- `RuntimeBuilder(platform="cuda")` discovers both CUDA runtimes:
  `host_schedule` and `persistent_device`.
- `RuntimeBinaries` now exposes a role-keyed view through `role_paths` and
  `path_for_role(...)`. Ascend platforms map the existing `host`, `aicpu`,
  and `aicore` roles directly. CUDA build configs now declare native `host`
  and `device` targets, and `persistent_device` also declares a native
  `scheduler` target. CUDA `RuntimeBinaries` no longer populates legacy
  `aicpu_path` / `aicore_path` aliases; consumers use the role map instead.
- The Python `ChipWorker` wrapper and underlying `_ChipWorker` nanobind
  boundary now accept role-keyed runtime binary maps directly through
  `_ChipWorker.init_roles(...)`. The C++ `ChipWorker` probes the optional
  host-runtime `simpler_init_roles(...)` ABI and passes non-host CUDA roles
  such as `device` and `scheduler` without lowering them into AICPU/AICore
  slots. The legacy positional init remains as a compatibility fallback for
  runtimes that do not export the optional role-keyed entry.

Evidence:

- `tests/ut/py/test_cuda_backend.py`
- `tests/ut/py/test_runtime_builder.py`
- `tests/ut/py/test_runtime_compiler.py`
- `src/cuda/runtime/host_schedule/build_config.py`
- `src/cuda/runtime/persistent_device/build_config.py`
- `src/cuda/platform/onboard/device/CMakeLists.txt`
- `src/cuda/platform/onboard/scheduler/CMakeLists.txt`
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp`

### Host-Schedule Runtime

The CUDA `host_schedule` runtime is implemented as a real CUDA host runtime
slice. It supports:

- device initialization/finalization;
- device allocation and free;
- host-to-device and device-to-host copies;
- PTX module loading through the CUDA Driver API;
- prepared callable registration and unregistration;
- vector-add launch through `run_prepared`;
- a non-blocking stream pool using callable `stream_id` metadata. The pool
  defaults to four streams and can be sized at device init with
  `PTO_CUDA_STREAM_POOL_SIZE` for host-schedule concurrency experiments.

`KernelCompiler(platform="cuda").compile_cuda_host_schedule()` now compiles a
user-authored CUDA task body through the shared wrapper generator and writes a
cached host-schedule callable artifact under
`build/cache/cuda/onboard/host_schedule/callables/`. The generated host
wrapper can lower a task context into the current vector-add launch ABI, so
the artifact can be passed to `prepare_callable` and run with real device
data.

Evidence:

- `tests/ut/py/test_cuda_backend.py` validates vector-add with real CUDA
  device data.
- The stream concurrency smoke validates two independent prepared callables
  on distinct streams.
- The stream-pool sizing smoke validates a callable on `stream_id=5` with
  `PTO_CUDA_STREAM_POOL_SIZE=6` on both local A100 and remote H200.
- `simpler_setup.cuda_callable_compiler.prepare_cuda_host_schedule_callable()`
  builds the shared ctypes manifest for host-schedule compiler artifacts and
  preserves PTX/entry-name buffer lifetimes for `prepare_callable`.
- `PreparedCudaCallable` exposes `buffer_ptr()` / `buffer_size()`, and the L2
  Python `Worker.register(...)` path can prepare those raw CUDA callable
  blobs through `prepare_callable_from_blob`.
- L2 Python `Worker.run(...)` can launch backend-specific raw CUDA argument
  structs that expose `buffer_ptr()` / `buffer_size()`, so the host-schedule
  vector-add path no longer has to call `run_prepared` through `ctypes`.
- `tests/ut/py/test_cuda_kernel_compiler.py` covers the CUDA `KernelCompiler`
  entry point for host-schedule task bodies.
- `tests/ut/py/test_cuda_backend.py` runs one host-schedule callable compiled
  by `KernelCompiler` through `prepare_callable` and validates real CUDA output
  data.
- `tests/ut/py/test_cuda_backend.py` also runs the compiler-backed
  host-schedule vector-add through `Worker(level=2, platform="cuda")`,
  `Worker.register(...)`, device allocation/copy helpers, and `Worker.run(...)`
  with a real `CudaVectorAddArgs` struct.
- `SceneTestCase` L2 compilation accepts `CALLABLE["cuda"]` specs for
  `host_schedule`, compiles them through `KernelCompiler(platform="cuda")`,
  registers the prepared raw callable through the normal L2 `Worker`, builds
  `CudaVectorAddArgs`, `CudaVectorUnaryArgs`, `CudaVectorScaleArgs`,
  `CudaVectorAxpyArgs`, `CudaVectorAffineArgs`, and `CudaVectorTernaryArgs`
  from normal `TaskArgsBuilder` CPU tensors/scalars, and validates real
  copied-back CUDA output data.

Focused stream-pool verification first failed because the fixed four-stream
pool rejected a callable manifest with `stream_id=5`. After adding
`PTO_CUDA_STREAM_POOL_SIZE`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q \
  -k 'stream_pool_size_env or independent_callables_on_multiple_streams' \
  --platform cuda
```

Result: `2 passed, 37 deselected`. The same stream-pool sizing selector
passed on remote H200 after syncing the tree and rebuilding the runtime:
`1 passed, 38 deselected`, with the known PTO-ISA SSH refresh warning printed
before pytest.

The stream-concurrency benchmark now accepts `--stream-pool-size` and records
the host stream-pool setting in report metadata. A paired A100/H200 capture
used `--stream-pool-size 6`, two repeats, and the stream-concurrency
microbenchmark:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
  --stream-concurrency --device 0 --repeats 2 --arch compute_80 \
  --stream-pool-size 6 --label a100-stream-pool6-working \
  --output-dir tmp/cuda-backend/a100-stream-pool6-working
```

The H200 command used `--arch compute_90` after syncing the working tree.
The merged report is under `tmp/cuda-backend/stream-pool6-working/` and
contains JSON, Markdown, and SVG report files. A100 reported median
`pto_stream_parallel/pto_stream_serial = 0.51x`; H200 reported `0.48x`.
The merged Markdown report includes `Host stream pool size: 6`, making the
concurrency configuration explicit in the visual artifact set.

The stream-concurrency workflow now has a paired runner,
`cuda_pair_stream_benchmark.py`, so the A100/H200 capture is repeatable with
the same validation discipline as the selected persistent benchmark gates. The
TDD check first failed because the paired stream runner did not exist, then
passed after adding command construction, sanitized merge examples, and a
capture validation command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'pair_stream_benchmark'
```

Result: `2 passed, 266 deselected`.

The paired A100/H200 run used `--stream-pool-size 6`, two repeats, local
`compute_80`, remote `compute_90`, and tree sync to the H200 host:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_stream_benchmark.py \
    --repeats 2 --stream-pool-size 6 --sync-remote-tree \
    --output-root tmp/cuda-backend/stream-pair-working
```

Artifacts:

- `tmp/cuda-backend/stream-pair-working/a100-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/h200-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/combined-stream-pool6-a36d137b/`
- `tmp/cuda-backend/stream-pair-working/index.md`

The validator required machines `hina` and `dasys-h200x8`, baselines
`pto_stream_serial` and `pto_stream_parallel`, size `2`, two repeats, eight
total rows, source-paper provenance, command examples, and generated report
files.

| GPU | Serial ns | Parallel ns | Parallel/serial |
| --- | --------- | ----------- | --------------- |
| A100 | 113791963 | 58041653 | 0.51x |
| H200 | 89717007 | 46203624 | 0.51x |

### Persistent-Device Runtime

The CUDA `persistent_device` runtime is implemented as a set of tracer-bullet
execution modes:

- direct descriptor-array persistent executor;
- scheduler/worker bounded ready queue;
- bounded ring wraparound with capacity smaller than task count;
- generated-dispatch DAG with fan-in counters;
- five-task DAG-chain runtime graph descriptor;
- six-task scratch-reuse DAG descriptor;
- tensor-tile DAG descriptor with rows/cols/inner/stride metadata;
- tensor-core tile DAG descriptor with a block-wide CUDA WMMA
  `m16n16k8`/TF32/F32 generated-dispatch task body;
- scalar-argument DAG descriptors for single-tensor scale,
  mixed tensor/scalar AXPY-style, and two-scalar affine task bodies.
- third tensor-argument DAG descriptor for a generated-dispatch triad task
  body.
- fourth tensor-argument DAG descriptor for a generated-dispatch quad task
  body.
- generic tensor/scalar argument slots in the persistent DAG descriptor, with
  a generated-dispatch task reading `tensor_args[0]`, `tensor_args[1]`,
  `scalar_args[0]`, and `scalar_args[1]`.
- unary generated-dispatch DAG descriptor for a task body that reads one
  tensor input and leaves the second tensor pointer unused.
- device-side scheduler diagnostics for unsupported generated-dispatch
  `func_id` values, invalid dependent task IDs, out-of-range dependent
  spans, fan-in underflow, duplicate dependents, self-dependents, and
  initial-fan-in mismatch.
- device-side scheduler diagnostics for malformed graphs that have no
  zero-fan-in root, or that publish some ready roots but exhaust work before
  every task completes.
- explicit resource-policy smoke metadata for the current single scheduler
  block, configurable queue/DAG worker blocks, direct-mode worker blocks per
  task, and callable `stream_id`.
- prepared-callable repeat-run lifecycle metadata for direct, queue, and DAG
  modes, with queue counters/flags and DAG graph state reset between launches.

The persistent DAG path compiles generated CUDA source with `nvcc` and stores
the generated source, PTX, and manifest under
`build/cache/cuda/onboard/persistent_device/callables/`.
The smoke and benchmark path now reaches that artifact compiler through
`KernelCompiler(platform="cuda").compile_cuda_persistent_device(...)`, which
accepts task source files plus `func_id` metadata, lowers task-body style
sources through the same `CudaTaskBody` wrapper contract as `host_schedule`,
and composes the generated dispatch entry.
`SceneTestCase` L2 compilation accepts `CALLABLE["cuda"]` specs for
`persistent_device`, compiles task-body sources through the same
`KernelCompiler` entry point, registers the prepared raw callable through the
normal L2 `Worker`, builds `persistent_dag_fork_join_f32`,
`persistent_dag_chain_f32`, `persistent_dag_reuse_f32`, and
`persistent_dag_scalar_scale_f32`, `persistent_dag_scalar_axpy_f32`, and
`persistent_dag_scalar_affine_f32` scalar descriptors,
`persistent_dag_tensor_tile_f32` state objects,
`persistent_dag_triad_f32` third-tensor descriptors,
`persistent_dag_quad_f32` fourth-tensor descriptors,
`persistent_dag_generic_args_f32` generic tensor/scalar argument descriptors,
`persistent_dag_graph_f32` explicit and tensor-flow-inferred graph
descriptors, and
`persistent_dag_unary_square_f32` unary descriptors, and
`persistent_dag_tensor_core_tile_f32` WMMA tensor-core descriptors from normal
`TaskArgsBuilder` CPU tensors and scalars, and validates real copied-back CUDA
output data. The explicit graph descriptor adapter resolves per-task
`scalar0`, `scalar1`, and generic `scalar_args` entries from either numeric
literals or `TaskArgsBuilder` scalar names, so graph descriptors use the same
scalar argument flow as the fixed scalar descriptor adapters. The scene-test
persistent-device compiler path also forwards callable
`stream_id` into the prepared CUDA manifest, so these L2 tests can run on a
selected non-default runtime stream. After each persistent-device scene-test
launch, the L2 path now copies back device scheduler counters and raises on
nonzero scheduler errors or incomplete DAG execution, so scheduler failures
are visible even when a diagnostic test intentionally skips golden comparison.
The no-torch
persistent smoke path also validates a generated-dispatch triad descriptor
with a third tensor pointer field, a quad descriptor with third and fourth
tensor pointer fields, a generic-argument descriptor, and a generated-dispatch
unary-square descriptor with a single tensor input.
The host-schedule scene path also accepts the neutral
`elementwise_binary_f32` adapter for non-addition task bodies that still use
the current `(a, b, out, n)` launch ABI. It accepts `elementwise_unary_f32`
for unary `(a, out, n)` task bodies, `elementwise_scale_f32` for scalar
`(a, out, alpha, n)` task bodies, and `elementwise_axpy_f32` for mixed
tensor/scalar `(a, b, out, alpha, n)` task bodies. It also accepts
`elementwise_affine_f32` for two-scalar affine
`(a, b, out, alpha, beta, n)` task bodies and `elementwise_triad_f32` for
three-input `(a, b, c, out, n)` task bodies, `elementwise_quad_f32` for
four-input `(a, b, c, d, out, n)` task bodies, and
`elementwise_generic_args_f32` for the host-schedule generic tensor/scalar
argument slots. The original two-slot launch ABI remains available, and the
host runtime also accepts the four tensor/scalar slots already represented by
`CudaVectorGenericArgs`.
The no-torch Worker smoke can validate that same non-addition host-schedule
ABI with `--op mul`, unary ABI with `--op square`, scalar ABI with
`--op scale`, mixed tensor/scalar ABI with `--op axpy`, and two-scalar affine
ABI with `--op affine`, three-input ABI with `--op triad`, and four-input ABI
with `--op quad`, which keeps H200 coverage available when the remote Python
environment lacks `torch`.

Evidence:

- `tests/ut/py/test_cuda_backend.py` runs persistent-device smoke tests with
  real CUDA data when `nvcc` is available.
- `tests/ut/py/test_cuda_persistent_codegen.py` covers generated dispatch,
  device scheduler diagnostic fields, tensor descriptor fields, shared
  task-body wrapper generation, host-schedule and persistent-device manifest
  writing, and cache reuse.
- `tests/ut/py/test_cuda_kernel_compiler.py` covers both CUDA
  `KernelCompiler` entry points.
- `simpler_setup/cuda_preflight.py` gives CUDA real-data tests one shared
  preflight path for `nvcc`, `nvidia-smi`, and driver visibility.
- `simpler_setup/cuda_callable_compiler.py` contains the generated-dispatch
  source renderer, shared task-body wrapper renderer, prepared-callable
  manifest helpers, and offline `nvcc` compile helper.

### Evaluation And Reporting

The current evaluation setup covers local A100 and remote H200 runs with:

- `direct_driver`;
- `direct_driver_graph`;
- `pto_host_schedule`;
- `pto_host_schedule_compiler`;
- `pto_host_schedule_unary_square`;
- `pto_host_schedule_quad`;
- `pto_host_schedule_generic_args`;
- `pto_persistent_device`;
- `pto_persistent_queue`;
- `pto_persistent_dag`;
- `pto_persistent_dag_chain`;
- `pto_persistent_dag_reuse`;
- `pto_persistent_dag_scalar_axpy`;
- `pto_persistent_dag_scalar_scale`;
- `pto_persistent_dag_scalar_affine`;
- `pto_persistent_dag_triad`;
- `pto_persistent_dag_quad`;
- `pto_persistent_dag_generic_args`;
- `pto_persistent_dag_graph`;
- `pto_persistent_dag_graph_generic_args4`;
- `pto_persistent_dag_graph_node_attrs`;
- `pto_persistent_dag_graph_node_io`;
- `pto_persistent_dag_graph_node_link`;
- `pto_persistent_dag_graph_named_callable`;
- `pto_persistent_dag_graph_node_op`;
- `pto_persistent_dag_graph_depends_on`;
- `pto_persistent_dag_graph_scalar_axpy`;
- `pto_persistent_dag_graph_scalar_scale`;
- `pto_persistent_dag_graph_scalar_affine`;
- `pto_persistent_dag_graph_reordered`;
- `pto_persistent_dag_graph_chain`;
- `pto_persistent_dag_graph_scratch_reuse`;
- `pto_persistent_dag_graph_diamond`;
- `pto_persistent_dag_graph_tagged`;
- `pto_persistent_dag_graph_tagged_inout`;
- `pto_persistent_dag_graph_role_keyed_inout`;
- `pto_persistent_dag_graph_compact_role_inout`;
- `pto_persistent_dag_graph_pair_inout`;
- `pto_persistent_dag_graph_triad`;
- `pto_persistent_dag_graph_quad`;
- `pto_persistent_dag_graph_unary_square`;
- `pto_persistent_dag_unary_square`;
- `pto_persistent_dag_tensor`;
- `pto_persistent_dag_graph_tensor`;
- `pto_persistent_dag_tensor_core`;
- `pto_persistent_dag_graph_tensor_core`;
- `cublas_sgemm`;
- `cublas_sgemm_graph`;
- same-work batch rows;
- worker-grid batch rows.

The latest full paired capture at commit `4e81fbff` uses the `16x16x16`
tensor descriptor, sizes `1024,65536,1048576`, three repeats, task counts
`2,6,12`, and worker-grid values `32,64,128,256`. It writes artifacts under
`tmp/cuda-backend/current-head-full-parallel-chains-4e81fbff-working/`
`combined-current-4e81fbff/` and validates `1296` combined samples after the
parallel-chains graph row joined the selected matrix. The paired-runner
validator checked
source-paper provenance, sanitized command examples, generated Markdown/SVG
reports, zero scheduler errors, selected tensor throughput reports, graph
topology reports, graph TaskArgs-like metadata reports, expected generated
dispatch sequences, tensor tile descriptors, graph fan-in/dependent arrays,
node attrs/ops metadata, named-callable metadata, task-argument spellings,
scratch-reuse metadata, and the nine-task parallel-chains queue capacity.
This supersedes the older `c183d1ad`, `f99dc6b0`, `9ec5511e`, `cb300e82`,
and `61cf96cd` full captures while keeping the same three-size/three-repeat
comparison role.

Selected current-head full-capture medians show that the compiler-backed
host-schedule row remains within `0.84x-1.06x` of the handwritten
host-schedule row across A100/H200 and vector sizes. The graph task-argument
spelling rows also validate tag, role-keyed, compact, pair-shaped, and
role-map spellings through the same in-place graph topology: dispatch
`1,1,1`, fan-in `0,1,1`, dependents `1,2`, and the same report-visible task
args. At `N=1024`, A100 reported tag/role/compact/pair/role-map medians of
`30720/30720/30720/29696/30720 ns`; H200 reported
`29056/28576/27712/27936/29056 ns`.
The selected named-callable row now appears in the full matrix at all three
sizes with graph fan-in `0,0,2`, dependents `2,2`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`named_callable`, and graph task args
`task0=callable:add,input:a,input:b,output:tmp0;`
`task1=callable:mul,input:a,input:b,output:tmp1;`
`task2=callable:add,input:a,input:b,output:out`. Median device times were
`26624/135168/2361344 ns` on A100 and `25344/132320/1907936 ns` on H200 for
sizes `1024/65536/1048576`.

A compact paired benchmark at commit `945016c3` adds
`pto_persistent_dag_graph_diamond` to the benchmark matrix and validates the
new row on A100 and H200. It uses `N=1024`, one repeat, no batch rows, the
default `16x16x16` tensor descriptor for tensor rows, and writes raw JSON,
Markdown, and SVG reports under
`tmp/cuda-backend/combined-current-945016c3/`. The validator checked `48`
combined rows, required command examples, source-paper provenance, report
files, zero scheduler errors, and the graph-diamond generated-dispatch
sequence. The graph-diamond row reported dispatch `[9,2,1,2,1]`, five
completed tasks, A100 `device_wall_ns=36864`, and H200
`device_wall_ns=31744`.

The previous graph-generic compact paired validation at commit `b2c5c8a4` uses
the default `16x16x16` tensor descriptor so the scalar tensor DAG,
`pto_persistent_dag_tensor_core`, and `cublas_sgemm` rows are all runnable in
the same paired report. It runs `N=1024`, one repeat, `batch_tasks=2`, and
`worker_blocks_per_task=4`, producing `60` combined rows under
`tmp/cuda-backend/combined-current-b2c5c8a4/`. The paired runner validated
required baselines, expected generated-dispatch sequences, command examples,
tensor descriptor metadata, source-paper provenance, and Markdown and SVG
report files. It also validates the host-schedule generic-args benchmark row.
Selected A100 device times for host, host-generic, base-DAG,
persistent-generic, graph-generic4, tensor, tensor-core, cuBLAS, and
grid-batch were `22528/35840/44032/29696/27648/37888/36864/37888/37888 ns`;
H200 reported
`16992/31264/40320/30592/27520/48992/32480/34304/31872 ns`. All PTO
persistent DAG rows reported zero device scheduler errors.

The compact paired validation at artifact label `dbb01406` adds
`pto_persistent_dag_graph_scratch_reuse` to the selected benchmark path after
the previous `06b8c0c6` graph-chain gate. It uses the same `N=1024`, one
repeat, `batch_tasks=2`, `worker_blocks_per_task=4`, and default `16x16x16`
tensor descriptor shape as the compact gate, producing `64` combined rows
under `tmp/cuda-backend/combined-current-dbb01406/`. The validator checked
source-paper provenance, sanitized command examples, report files, tensor
descriptor metadata, dispatch sequences, graph descriptor fan-in/dependent
metadata, zero scheduler errors, and the new graph-scratch-reuse row. The
compact and paired-current presets now require
`scratch_reuse=reused_buffer=tmp0,reuse_task=4` for that row. The captured row
reported
`graph_descriptor.fanin=[0,0,2,1,1,2]`,
`graph_descriptor.dependents=[2,2,3,4,5,5]`, dispatch `[1,2,1,2,1,1]`,
`scratch_reuse.reused_buffer=tmp0`, `scratch_reuse.reuse_task=4`, completed
count `6`, A100 `device_wall_ns=36864`, and H200 `device_wall_ns=38240`.
The graph-chain row remains in the same compact gate and reported A100
`device_wall_ns=35840` and H200 `device_wall_ns=34528`.

The compact paired benchmark gate at artifact label `55a144de` now includes
`pto_persistent_dag_graph_tagged_inout`, validating explicit `input`,
`output`, `inout`, and `output_existing` task-argument tags in the selected
benchmark path. It uses `N=1024`, one repeat, `batch_tasks=2`,
`worker_blocks_per_task=4`, and the default `16x16x16` tensor descriptor,
producing `68` combined rows under
`tmp/cuda-backend/tagged-inout-benchmark-working/combined-current-55a144de/`.
The validator checked required baselines, dispatch sequences, tensor
descriptor metadata, source-paper provenance, command examples, generated
Markdown/SVG reports, graph descriptor fan-in/dependent metadata, graph
task-argument metadata, and zero scheduler errors.
The tagged-inout row reported
dispatch `[1,1,1]`, `graph_descriptor.fanin=[0,1,1]`,
`graph_descriptor.dependents=[1,2]`, completed count `3`,
`graph_task_args.task1=inout:tmp1,input:b`, A100
`device_wall_ns=35840`, and H200 `device_wall_ns=30080`.

The compact paired benchmark gate at artifact label `a46db551` promotes
`pto_persistent_dag_scalar_scale` into the selected benchmark path. It uses
`N=4096`, one repeat, no batch rows, and default `16x16x16` tensor descriptor
metadata. The paired runner synced the working tree to H200, captured A100
and H200 reports, merged `44` rows, and validated required baselines,
source-paper provenance, command examples, report files, and zero scheduler
errors under `tmp/cuda-backend/combined-current-a46db551/`. The scalar-scale
row reported dispatch `[11,2,1]`, `scalar0=2.0`, `device_wall_ns=37888` on
A100, and `device_wall_ns=27744` on H200.

The supplemental tensor-shape sweep at commit `c0ada3ad` runs
`pto_persistent_dag_tensor` on local A100 and remote H200 for `8x4x12`,
`16x16x64`, and `32x16x64` descriptors with `N=4096` and two repeats. All
12 rows passed with dispatch sequence `[3,1,2,1]`; the descriptor tile counts
were `128`, `16`, and `8`, respectively. The raw JSON, Markdown, and SVG
artifacts are under `tmp/cuda-backend/tensor-shape-sweep-c0ada3ad/`. This is
still scalar tiled GEMM scheduler evidence, not tensor-core throughput.

The first tensor-core persistent DAG smoke at commit `390eda4f` runs
`tensor_core_tile` on local A100 and remote H200 with a `16x16x16` descriptor.
The generated dispatch sequence is `[10,1,2,1]`; func_id `10` is a block-wide
WMMA `m16n16k8` task body with TF32 inputs and F32 accumulation. The paired
runner validated both artifacts with zero scheduler errors, tensor descriptor
`16x16x16`, `completed_count=4`, and generated Markdown/SVG report files under
`tmp/cuda-backend/persistent-tensor_core_tile-16x16x16-smoke-390eda4f/`.
This is callable and scheduler evidence for tensor-core task bodies, not a
tuned throughput result.

The first tensor-core selected-baseline benchmark row at commit `0879aa9e`
runs `pto_persistent_dag_tensor_core` on local A100 and remote H200 with the
same `16x16x16` descriptor. The compact selected-baseline report uses
`N=256`, one repeat, no batch rows, and the usual JSON/Markdown/SVG benchmark
outputs. The tensor-core DAG row measured `37888 ns` device time on A100 and
`38656 ns` on H200, compared with `40960 ns` and `43392 ns` for the scalar
`pto_persistent_dag_tensor` row in the same report. The raw artifacts are
under `tmp/cuda-backend/combined-tensor-core-current-0879aa9e/`. Benchmark
reports now also write `cuda-benchmark-dag-deltas.svg`, which plots each
`pto_persistent_dag_*` row's signed device-time increment over the matched
`pto_persistent_dag` scheduler baseline, and
`cuda-benchmark-throughput.svg`, which plots median GF/s for tensor-DAG and
cuBLAS rows with recorded tensor tile descriptors. Benchmark Markdown reports
now include a graph descriptor metadata table, and the primary benchmark SVG
embeds graph topology and task-argument metadata for explicit graph rows.

The first cuBLAS library-backed tensor baseline adds `cublas_sgemm` to the
same compact selected-baseline report shape. It uses CUDA Runtime API events
around a warm cuBLAS `cublasSgemmStridedBatched` call over the configured
`16x16x16` descriptor. The matching `cublas_sgemm_graph` row captures that
same warmed descriptor into a CUDA Graph, instantiates it outside the measured
interval, warms graph replay once, and times `cudaGraphLaunch` with CUDA
events. In the paired A100/H200 capture under
`tmp/cuda-backend/combined-cublas-current-343924df/`, the row measured
`48128 ns` device time on A100 and `58623 ns` on H200. The matched
`pto_persistent_dag_tensor_core` rows in that report measured `33792 ns` and
`32960 ns`. This row is a CUDA library launch/compute comparison point, not a
PTO runtime path.

The first cuBLAS CUDA Graph paired capture is under
`tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/`.
It uses `N=1024`, one repeat, no batch rows, and the default `16x16x16`
descriptor. The paired runner synced the working tree to `bizhaoh200`,
captured A100 and H200 reports, merged `58` rows, and validated report files,
command examples, source-paper provenance, tensor descriptor metadata, PTO
dispatch sequences, and zero scheduler errors. The A100 rows measured
`cublas_sgemm=48128 ns` and `cublas_sgemm_graph=10239 ns`; H200 measured
`9119 ns` and `8543 ns`. The graph row is a launch-path comparison point
around cuBLAS graph replay, not a tuned GEMM throughput claim.

The WMMA tensor-core task body now handles a grid of 16x16 output fragments
instead of only one fragment per descriptor. Focused TDD first failed because
`pto_persistent_dag_tensor_core` rejected `32x16x16`; after generalizing the
descriptor contract and generated-dispatch task body, the paired smoke:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_core_tile --task-count 4 --n 1024 \
    --tensor-rows 32 --tensor-cols 16 --tensor-inner 16 \
    --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-core-wide-working
```

validated A100 and H200 JSON, Markdown, and SVG artifacts at
`tmp/cuda-backend/tensor-core-wide-working/persistent-tensor_core_tile-32x16x16-repeat2-smoke-ef475c2d/`.
Both GPUs reported dispatch `10,1,2,1`, tensor tile `32x16x16`,
`tile_count=2`, repeat completions `[4,4]`, and zero scheduler errors. The
A100 device total was `77824 ns`; H200 was `61152 ns`.

The normal L2 `SceneTestCase` adapter now uses the same descriptor rule as
the benchmark/smoke path for tensor-core tasks: `rows` and `cols` must be
multiples of `16`, and `inner` must be divisible by `8`. Focused TDD first
failed on both the fixed `persistent_dag_tensor_core_tile_f32` builder and
the explicit graph `func_id=10` path because each still required
`rows=16, cols=16`. After the guard was generalized, the no-torch ctypes
SceneTestCase selector ran the normal tensor-core path with a `32x16x16`
descriptor and the graph tensor-core path with the existing `16x16x16`
descriptor on A100 and H200:

```bash
PYTHONPATH=$PWD:$PWD/python PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -rs -k 'tensor_core_tile_with_ctypes_data' --platform cuda
```

Local A100 reported `2 passed, 143 deselected`; the synced H200 run reported
the same result after the known PTO-ISA SSH refresh warning. This closes the
normal scene-test gap for multi-fragment WMMA descriptors; broader
model-kernel families are still an evaluation and tuning task.

The compact paired benchmark then used the same `32x16x16` descriptor with
`N=1024`, one repeat, no batch rows, and the normal selected baseline set:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --tensor-rows 32 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-core-wide-benchmark-working
```

The capture under
`tmp/cuda-backend/tensor-core-wide-benchmark-working/combined-current-ef475c2d/`
validated `58` rows, report files, command examples, source-paper metadata,
tensor descriptors, PTO dispatch sequences, and zero scheduler errors. In the
tensor-throughput table, A100 measured
`pto_persistent_dag_tensor_core=36864 ns`,
`cublas_sgemm=37888 ns`, and `cublas_sgemm_graph=7168 ns`; H200 measured
`33280 ns`, `38591 ns`, and `9216 ns` for the same rows.

The tensor shape sweep script now accepts `--baselines` and `--sizes`, so one
paired A100/H200 sweep can compare scalar tensor DAG, the explicit graph
tensor DAG, WMMA tensor-core DAG, and cuBLAS SGEMM rows across descriptor
shapes and problem sizes. The multi-repeat size-sweep report at commit
`e79edba2` uses a `16x16x16` descriptor with `N=256`, `4096`, and `65536`,
three repeats, and the artifact under
`tmp/cuda-backend/tensor-shape-sweep-e79edba2/`. It writes raw rows,
VDCores/MPK provenance metadata, per-baseline workload descriptions, a median
summary table with normalized GFLOP/s, and SVG charts for median device time
and median GFLOP/s with sample counts. Median device times are: A100
scalar/tensor-core/cuBLAS at
`N=256`
`47104/47104/43007 ns`, `N=4096` `79872/71680/36864 ns`, and `N=65536`
`587616/470368/38911 ns`; H200 at `N=256` `30560/28160/50496 ns`, `N=4096`
`88576/49888/37055 ns`, and `N=65536` `1032896/390368/36127 ns`. Normalized
throughput at `N=65536` is A100 scalar/tensor-core/cuBLAS
`3.57/4.46/53.90` GFLOP/s and H200 `2.03/5.37/58.05` GFLOP/s. The PTO
tensor-core row improves over the scalar tensor row as repeated tile work
grows, while the cuBLAS baseline shows the remaining gap to a tuned CUDA
library path.
A current-head one-repeat graph-tensor sweep at
`tmp/cuda-backend/tensor-shape-sweep-0e84fd26/` adds
`pto_persistent_dag_graph_tensor` to the same `16x16x16` tensor-baseline
comparison at `N=256` and `4096`. The validated median device times were:
A100 scalar/graph/tensor-core/cuBLAS `47104/47104/45056/48128 ns` at
`N=256` and `80896/80896/82944/39935 ns` at `N=4096`; H200
`29568/32800/27040/51711 ns` at `N=256` and
`89472/89152/51872/35904 ns` at `N=4096`. The graph tensor row uses the same
dispatch `3,1,2,1` as the scalar tensor row while exercising the explicit
runtime graph descriptor path.
`cuda_validate_tensor_sweep.py` checked the expected A100/H200 rows,
baselines, sizes, shape, three repeats, report files, throughput report
content, and PTO dispatch sequences before the numbers were copied into docs.
New tensor-sweep captures can also require sanitized local/remote command
examples and source-paper metadata before publishing with
`--require-command-examples` and `--require-source-papers`; the source-paper
gate verifies the referenced files exist under `tmp/sources/`.
Benchmark captures now use the same VDCores/MPK `source_papers` metadata
contract and sanitized command-example metadata contract as tensor sweeps.
They can be gated with `cuda_validate_capture.py` plus
`--require-command-examples`, `--require-zero-scheduler-errors`, and
`--require-source-papers` before new paired-current numbers are published.
The existing real-data paired A100/H200 capture from commit `61cf96cd` was
re-rendered through the updated report path under
`tmp/cuda-backend/combined-current-61cf96cd-command-source-gate/`, and the
paired-current validator passed with `--require-command-examples` and
`--require-source-papers`.

A current-HEAD one-repeat compact tensor sweep at commit `a5fd4bfd` validated
that gate against real local A100 and remote H200 data. The artifact under
`tmp/cuda-backend/tensor-shape-sweep-a5fd4bfd/` includes sanitized local
sample, remote sample, and remote tree-sync commands in generated Markdown and
JSON. The validation required A100/H200 rows, scalar tensor DAG, WMMA tensor
DAG, cuBLAS SGEMM, `N=256`, `16x16x16`, report files, command examples, and
source-paper metadata, and PTO dispatch sequences.

A follow-up working-tree sweep under
`tmp/cuda-backend/tensor-graph-library-baselines-working/`
`tensor-shape-sweep-848c4ee5/` adds the `cublas_sgemm_graph` tensor-sweep
baseline beside scalar tensor, explicit graph tensor, WMMA tensor-core, and
plain cuBLAS rows. The run uses one repeat, `N=256`, and the `16x16x16`
descriptor on local A100 and remote H200. Validated median device times were:
A100 scalar/graph/tensor-core/cuBLAS/cuBLAS-graph
`43008/41984/51200/89088/12288 ns`; H200
`37440/45120/37472/50271/10271 ns`. The cuBLAS Graph row shows the expected
benefit of replaying an already captured library call in this launch-dominated
compact descriptor, while PTO graph tensor remains close to the scalar tensor
DAG and continues to validate the explicit runtime graph descriptor path.

A current-head full compact sweep at commit `219042f5` refreshes that
baseline set with three repeats and both compact preset shapes,
`16x16x16` and `16x16x64`. The artifact under
`tmp/cuda-backend/tensor-sweep-current-working/tensor-shape-sweep-219042f5/`
validated 72 rows with `--preset compact-tensor-baselines`,
`--require-command-examples`, and `--require-source-papers`. That preset now
also requires visible Markdown/SVG throughput content and visible Markdown
baseline-comparison content. Median `16x16x64` device times were A100
scalar/graph/tensor-core/graph-tensor-core/cuBLAS/cuBLAS-graph
`52224/38912/50176/50176/74752/9216 ns`; H200 measured
`32288/32512/32480/32127/51135/10176 ns`. Median `16x16x64` GFLOP/s were
A100 `0.63/0.84/0.65/0.65/0.44/3.56` and H200
`1.01/1.01/1.01/1.02/0.64/3.22`. The regenerated Markdown now compares PTO
tensor rows against matching cuBLAS Graph rows; for example the explicit
graph tensor row is `4.00x` cuBLAS Graph device time on A100 and `3.36x` on
H200 for `16x16x16`, while explicit graph tensor-core is `5.44x` on A100 and
`3.16x` on H200 for `16x16x64`.

A current-head one-repeat size sweep at commit `76422250` extends the same
six selected tensor baselines across `N=256`, `4096`, and `65536` for
`16x16x16`. The artifact under
`tmp/cuda-backend/tensor-size-sweep-working/tensor-shape-sweep-76422250/`
validated 36 rows with required A100/H200 artifacts, report files, visible
throughput content, command examples, VDCores/MPK source-paper metadata, and
PTO dispatch sequences. At `N=65536`, A100 measured scalar/graph/tensor-core/
graph-tensor-core/cuBLAS/cuBLAS-graph device times of
`437184/439392/579360/583840/30719/11264 ns`; H200 measured
`424448/426752/343904/343136/43039/8448 ns`. The explicit graph descriptor
tracks the generated scalar tensor DAG closely at large `N`, while H200's
WMMA tensor-core rows improve over scalar tensor and A100's current WMMA row
does not in this one-repeat capture. cuBLAS Graph remains the fastest
launch-replay baseline, so the next gap is still tuned PTO tensor body work
rather than descriptor, graph, or report plumbing.

Evidence:

- [evaluation.md](evaluation.md) is the evaluation landing page.
- [evaluation-current.md](evaluation-current.md) summarizes the latest paired
  A100/H200 capture.
- [evaluation-history.md](evaluation-history.md) preserves earlier captures.
- `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py` writes JSON,
  Markdown, and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py` writes
  compact smoke Markdown and SVG reports, including persistent-device dispatch
  `func_id` sequences, device scheduler error counters, repeat-run lifecycle
  counters, resource-policy metadata, tensor-core metadata, and scalar and
  tensor task arguments when present.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py` automates
  the local A100 run, remote H200 run, artifact copy, merge, command-example
  metadata capture, combined-artifact validation, and index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py` automates the
  no-torch host-schedule Worker smoke on local A100 and remote H200, then
  renders the compact smoke report and refreshes the artifact index.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py`
  automates no-torch persistent-device DAG smoke captures on local A100 and
  remote H200, including optional remote tree sync, tensor-tile descriptor
  flags, compact report rendering, smoke artifact validation, and
  artifact-index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py`
  automates paired A100/H200 tensor baseline sweeps over model-shaped tensor
  descriptors, records VDCores/MPK provenance and sanitized command examples
  in generated metadata, and writes JSON, Markdown, and SVG artifacts under
  `tmp/cuda-backend/`.
- `.agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py` renders
  the compact benchmark tables, selected benchmark tensor-throughput table,
  and compact tensor-sweep median table used by
  [evaluation-current.md](evaluation-current.md) from raw JSON artifacts,
  including graph scratch-reuse ratios in the DAG-shapes table, explicit
  graph descriptor dispatch/fan-in/task-argument metadata plus scalar/tensor
  descriptor argument maps in the graph-metadata table, and cuBLAS Graph
  replay columns in tensor-sweep summaries.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`
  checks paired benchmark captures for expected machines, selected baselines,
  sizes, repeats, sample count, generated report files, source-paper
  metadata, sanitized command examples, dispatch IDs, tensor-tile shapes, and
  graph descriptor, graph task-argument, and scratch-reuse metadata before
  docs are refreshed. Current paired presets also require visible graph
  topology and task-argument metadata in `cuda-benchmark.md` and
  `cuda-benchmark.svg`.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py` checks
  paired smoke captures for required A100/H200 artifacts, pass status, zero
  scheduler errors, expected runtime/mode, dispatch IDs, repeat-run lifecycle
  counts, tensor-tile descriptor shape, generated smoke report files, and
  visible report graph topology and graph task-argument metadata when
  requested.
- `.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py` indexes
  local `tmp/cuda-backend/` benchmark, tensor-shape sweep, lifecycle matrix,
  and smoke artifacts, including tensor-tile shapes, persistent smoke modes,
  lifecycle scenarios, dispatch sequences, scheduler error counters,
  repeat-run counts, per-launch completion counts, graph descriptor
  fan-in/dependent arrays, graph task-argument keys, and graph task-argument
  metadata.
- `.agents/skills/cuda-backend-eval/SKILL.md` documents the current paired
  A100/H200 recipe.

## Latest Local Verification

The focused CUDA test set was run from the project-local virtual environment.
The CUDA backend tests run separately so the shell exit status is authoritative
even when hardware tests take longer than the default tool wait window:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_persistent_codegen.py -q
```

Result: `34 passed`.

After adding persistent-device scene-test plumbing, the same focused CUDA set
was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_scene_test.py -q
```

Result: `36 passed`.

The CUDA scene-test file was also run on the remote H200 checkout after
pushing this change:

```bash
ssh bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `1 passed, 1 skipped`. The compile/plumbing test passed; the real-data
scene case skipped because the remote Python environment does not have
`torch`.

The remote H200 real-data Worker smoke was run through the no-torch smoke
script:

```bash
ssh bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
     --runner worker --device 0 --n 1024 --block-dim 256 \
     --arch compute_90 --no-build'
```

Result: `status=pass`, `runner=worker`, `ptx_arch=compute_90`,
`ptx_source=kernel-compiler-worker-task-body-compute_90`,
`device_wall_ns=12736`.

After adding persistent-device scene-test plumbing, the CUDA scene-test file
was rerun on the remote H200 checkout:

```bash
ssh bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `2 passed, 2 skipped`. The host-schedule and persistent-device
compile/plumbing tests passed; both real-data scene cases skipped because the
remote Python environment does not have `torch`.

The remote H200 persistent DAG real-data smoke was run through the no-torch
smoke script:

```bash
ssh bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python \
     .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 3 --n 1024 --arch compute_90 \
     --mode dag --queue-capacity 2'
```

Result: `status=pass`, `runtime=persistent_device`,
`ptx_arch=compute_90`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`completed_count=3`, `device_wall_ns=41152`.

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py -q
```

Result: `16 passed`.

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_worker/test_ensure_prepared.py \
  tests/ut/py/test_worker/test_host_worker.py -q
```

Result: `87 passed`.

The host-schedule Worker raw-args smoke was run locally on A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 --no-build
```

Result: `status=pass`, `runner=worker`, `ptx_arch=compute_80`,
`ptx_source=kernel-compiler-worker-task-body-compute_80`.

The no-torch host-schedule Worker multiply smoke was captured on both GPUs
with `--output-json` and rendered through `cuda_smoke_report.py`:

- A100: `status=pass`, `mode=worker/mul`, `ptx_arch=compute_80`
- H200: `status=pass`, `mode=worker/mul`, `ptx_arch=compute_90`
- Report:
  `tmp/cuda-backend/worker-mul-smoke-output-json/cuda-smoke-report.md`
- SVG: `tmp/cuda-backend/worker-mul-smoke-output-json/cuda-smoke-report.svg`

The paired no-torch smoke runner was also verified with remote tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op mul --sync-remote-tree
```

Result: A100 `status=pass`, `mode=worker/mul`, `ptx_arch=compute_80`,
`device_wall_ns=20480`; H200 `status=pass`, `mode=worker/mul`,
`ptx_arch=compute_90`, `device_wall_ns=20000`. The generated artifacts are
under `tmp/cuda-backend/worker-mul-smoke-d5788710/`.

The scalar host-schedule Worker smoke was captured on both GPUs with runtime
rebuild enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op scale --sync-remote-tree --build-runtime
```

Result: A100 `status=pass`, `mode=worker/scale`, `ptx_arch=compute_80`,
`device_wall_ns=8192`; H200 `status=pass`, `mode=worker/scale`,
`ptx_arch=compute_90`, `device_wall_ns=12544`. The generated artifacts are
under `tmp/cuda-backend/worker-scale-smoke-4240a4ba/`.

The unary host-schedule Worker smoke was captured on both GPUs with runtime
rebuild enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op square --sync-remote-tree --build-runtime
```

Result: A100 `status=pass`, `mode=worker/square`, `ptx_arch=compute_80`,
`device_wall_ns=9216`; H200 `status=pass`, `mode=worker/square`,
`ptx_arch=compute_90`, `device_wall_ns=16192`. The generated artifacts are
under `tmp/cuda-backend/worker-square-smoke-4cdde399/`.

The docs and skill updates were checked with targeted `pre-commit` runs and
`git diff --check` before commit.

The paired benchmark capture was refreshed at commit `db0acd4c` after
syncing the local checkout to the remote H200 host:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Result: A100 `label=a100-current-db0acd4c`, `ptx_arch=compute_80`; H200
`label=h200-current-db0acd4c`, `ptx_arch=compute_90`. Both reports use
`ptx_source=nvcc-*`, include generated compiler, unary host-schedule, and
persistent-device rows, and are merged in
`tmp/cuda-backend/combined-current-db0acd4c/`.

The local A100 persistent DAG smoke was run through the persistent-device
`KernelCompiler` entry point with task-body style DAG sources:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Result: `status=pass`, `runtime=persistent_device`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_80`,
`completed_count=3`.

The same task-body style persistent DAG smoke was run on the remote H200
checkout after pushing `6f1497b5`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only >/dev/null && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 3 --n 1024 --arch compute_90 \
     --mode dag --queue-capacity 2'
```

Result: `status=pass`, `runtime=persistent_device`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`completed_count=3`.

After adding persistent-device scheduler diagnostics, the focused CUDA test
set was rerun locally:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_scene_test.py -q
```

Result: `38 passed`.

The local A100 persistent DAG smoke now reports zero device scheduler errors
on the normal generated-dispatch path:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Result: `status=pass`, `device_scheduler_errors={"count": 0, "code": 0,
"task_id": 0}`, `completed_count=3`.

The synthetic invalid-dispatch shape was also run locally to verify that an
unsupported generated-dispatch `func_id` is surfaced before output mismatch
checks:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_func_id
```

Result: expected non-zero exit with `persistent dag scheduler error code=1
task_id=0 count=1`.

The synthetic invalid-dependent shape was run locally to verify that a runtime
graph descriptor cannot release a task ID outside the descriptor array:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=2
task_id=7 count=1`.

The synthetic invalid-dependent-range shape was run locally to verify that a
runtime graph descriptor cannot read outside the dependents array:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_dependent_range
```

Result: expected non-zero exit with `persistent dag scheduler error code=3
task_id=0 count=1`.

The synthetic fan-in-underflow shape was run locally to verify that a runtime
graph descriptor cannot decrement a dependent task's fan-in below zero:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape bad_fanin_underflow
```

Result: expected non-zero exit with `persistent dag scheduler error code=4
task_id=2 count=1`.

The synthetic duplicate-dependent shape was run locally to verify that a
runtime graph descriptor cannot list the same dependent task twice for one
completed task. Without this check, one producer could decrement the same
dependent fan-in twice and make it ready without two distinct predecessors:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 2 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape bad_duplicate_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=8
task_id=1 count=1`.

The synthetic self-dependent shape was run locally to verify that a runtime
graph descriptor cannot make a completed task release itself:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_self_dependent
```

Result: expected non-zero exit with `persistent dag scheduler error code=9
task_id=0 count=1`.

The synthetic initial-fan-in mismatch shape was run locally to verify that a
runtime graph descriptor cannot start from fan-in counters that disagree with
task metadata:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_initial_fanin
```

Result: expected non-zero exit with `persistent dag scheduler error code=5
task_id=0 count=1`.

The synthetic no-root shape was run locally to verify that a runtime graph
descriptor cannot deadlock workers by declaring no zero-fan-in task:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 1 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_no_root
```

Result: expected non-zero exit with `persistent dag scheduler error code=6
task_id=0 count=1`.

The synthetic unreachable-task shape was run locally to verify that a runtime
graph descriptor cannot deadlock workers by publishing one root but leaving
another task behind a dangling fan-in counter:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 2 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 1 --dag-shape bad_unreachable \
    --worker-blocks 2
```

Result: expected non-zero exit with `persistent dag scheduler error code=7
task_id=1 count=1`.

The same scheduler-diagnostic slice was verified on the remote H200 checkout
after pushing this change:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 3 --n 1024 --arch compute_90 \
     --mode dag --queue-capacity 2'
```

Result: `status=pass`, `ptx_arch=compute_90`,
`device_scheduler_errors={"count": 0, "code": 0, "task_id": 0}`,
`completed_count=3`.

The H200 invalid-dispatch check returned the expected diagnostic:
`persistent dag scheduler error code=1 task_id=0 count=1`.

The H200 invalid-dependent check returned the expected diagnostic:
`persistent dag scheduler error code=2 task_id=7 count=1`.

The H200 invalid-dependent-range check returned the expected diagnostic:
`persistent dag scheduler error code=3 task_id=0 count=1`.

The H200 fan-in-underflow check returned the expected diagnostic:
`persistent dag scheduler error code=4 task_id=2 count=1`.

After syncing this working tree to H200, the duplicate-dependent check returned
the expected diagnostic:
`persistent dag scheduler error code=8 task_id=1 count=1`.

The H200 self-dependent check returned the expected diagnostic:
`persistent dag scheduler error code=9 task_id=0 count=1`.

The H200 initial-fan-in mismatch check returned the expected diagnostic:
`persistent dag scheduler error code=5 task_id=0 count=1`.

The H200 no-root check returned the expected diagnostic:
`persistent dag scheduler error code=6 task_id=0 count=1`.

The H200 unreachable-task check returned the expected diagnostic:
`persistent dag scheduler error code=7 task_id=1 count=1`.

The current scheduler-diagnostic matrix was then captured as paired A100/H200
JSON, Markdown, and SVG evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_scheduler_error_matrix.py \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/scheduler-error-matrix-working
```

Result:
`tmp/cuda-backend/scheduler-error-matrix-working/scheduler-error-matrix-35de3303/`
contains `cuda-scheduler-error-matrix.json`,
`cuda-scheduler-error-matrix.md`, and
`cuda-scheduler-error-matrix.svg`. The matrix has `18` rows covering A100 and
H200 for unsupported `func_id`, invalid dependent ID, invalid dependent
range, fan-in underflow, duplicate dependent, self dependent, initial fan-in
mismatch, no root, and unreachable task. Every row reported `status=pass`
with the expected scheduler code, task ID, and `count=1`.

The matrix also passes the scheduler-error matrix validator with
`--preset default`, which checks the required cases, A100/H200 coverage,
source-paper provenance, command examples, and generated report files.

The current unreachable-task slice was also checked on H200 through pytest
after syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
     -q -rs -k "unreachable or smoke_runs_dispatch_dag" --platform cuda'
```

Result: `10 passed, 25 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The persistent callable lifecycle path has repeat-run smoke support that
prepares the callable once and launches it multiple times. Direct mode reuses
the prepared callable, queue mode resets the ready queue counters and flags,
and DAG mode resets the graph state between launches:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape chain --repeat-runs 2
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 4096 --arch compute_80 \
    --mode queue --queue-capacity 2 --repeat-runs 2
```

Result: `tmp/cuda-backend/persistent-chain-repeat2-smoke-4bcd56c4/`
contains A100 and H200 JSON plus Markdown/SVG reports. Local A100 returned
`launch_completed_counts=[5,5]`, `launch_device_wall_ns=[44032,41984]`, and
zero scheduler errors. Remote H200 returned `launch_completed_counts=[5,5]`,
`launch_device_wall_ns=[45760,39616]`, and zero scheduler errors.

The queue-mode lifecycle capture at
`tmp/cuda-backend/persistent-queue-repeat2-smoke-0a4447c0/` verifies that the
ready-queue counters and flags are reset between prepared-callable launches.
Local A100 returned `launch_completed_counts=[4,4]` and
`launch_device_wall_ns=[22528,14336]`. Remote H200 returned
`launch_completed_counts=[4,4]` and `launch_device_wall_ns=[25280,14272]`.

After adding invalid-dependent, dependent-range, fan-in-underflow,
duplicate-dependent, self-dependent, initial-fan-in, no-root, and
unreachable-task scheduler diagnostics, the CUDA backend/codegen/report tests
were rerun locally:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
    -q --platform cuda

.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py -q
```

Result: the CUDA backend suite reported `47 passed`, and the codegen/report
suite reported `276 passed`.

After adding the third-tensor persistent DAG scene-test arg builder, the new
ctypes-backed real-data scene test was checked on remote H200 without requiring
`torch`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PTO_ISA_ROOT=/data/shibizhao/pto-cu/build/pto-isa \
   PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest \
     tests/ut/py/test_cuda_scene_test.py::test_scene_test_runs_cuda_persistent_device_triad_with_ctypes_data -q'
```

Result: `1 passed`.

After adding shared CUDA preflight skip reporting, the local A100-focused test
set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_preflight.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_scene_test.py -q
```

Result: `25 passed`.

After adding the tensor-tile persistent DAG scene-test arg builder, the
focused CUDA suite was rerun locally on A100:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_kernel_compiler.py -q
```

Result: `39 passed`.

The same branch tip was checked on the remote H200 checkout. The scene-test
file passed its compile/plumbing cases and skipped the real-data cases because
the remote Python environment still lacks `torch`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `2 passed, 3 skipped`.

The no-torch tensor-tile persistent DAG smoke was also run on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
     --device 0 --task-count 4 --n 4096 --arch compute_90 \
     --mode dag --queue-capacity 2 --dag-shape tensor_tile'
```

Result: `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[3, 1, 2, 1]`, `completed_count=4`,
`device_scheduler_errors={"count": 0, "code": 0, "task_id": 0}`.

After adding chain and reuse persistent DAG scene-test arg builders, the
focused local CUDA scene/codegen set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_persistent_codegen.py -q
```

Result: `27 passed`. The new real-data chain and reuse scene tests both ran
through the local A100 CUDA L2 Worker path.

The same chain and reuse DAG shapes were checked on the remote H200 through
the no-torch persistent smoke path:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 1024 --arch compute_90 \
    --mode dag --queue-capacity 3 --dag-shape chain

CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_90 \
    --mode dag --queue-capacity 3 --dag-shape scratch_reuse
```

Result: both returned `status=pass` with zero device scheduler errors.

After adding paired persistent-smoke automation, the chain DAG smoke was
captured on local A100 and remote H200 with tree sync and compact report
generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape chain --task-count 5 --queue-capacity 3 --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-chain-smoke-e1fa429b/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `device_wall_ns=29696`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=33152`.

After adding tensor descriptor flags to the paired persistent-smoke runner,
the non-square tensor-tile DAG smoke was captured on local A100 and remote
H200 with tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_tile --task-count 4 --queue-capacity 2 \
    --n 4096 --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-tensor_tile-8x4x12-smoke-ad45b69c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `device_wall_ns=82944`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=49536`. Both rows
reported `dispatch_func_ids=[3,1,2,1]`, tensor shape `8x4x12`, `128` tiles,
and zero device scheduler errors.

After adding tensor descriptor validation to the smoke validator, the same
non-square tensor-tile DAG was captured with prepared-callable repeat reuse:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_tile --task-count 4 --queue-capacity 2 \
    --n 768 --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-tensor_tile-8x4x12-repeat2-smoke-223425b6/`
contains A100/H200 JSON plus Markdown/SVG reports. The paired runner then
validated both artifacts with `expected-tensor-tile=8x4x12`,
`repeat_runs=2`, `launch_completed_counts=[4,4]`, zero scheduler errors, and
the generated report files. The A100 row reported
`launch_device_wall_ns=[48128,36864]`; H200 reported
`launch_device_wall_ns=[54944,27040]`.

The preflight and CUDA scene-test subset was also run on the remote H200
checkout after pushing this change:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   git fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest \
     tests/ut/py/test_cuda_preflight.py \
     tests/ut/py/test_cuda_scene_test.py -q'
```

Result: `6 passed, 2 skipped`. The skips are the real-data scene cases that
still require `torch` in the remote Python environment; the compile/plumbing
and preflight checks passed.

After adding native CUDA `device` build roles, both CUDA runtimes were rebuilt
locally and on H200. `RuntimeBuilder(platform="cuda")` reported
`libcuda_device_runtime.so` for the `device` role in `host_schedule` and
`persistent_device`. The later role-cleanup slice removed the CUDA
`aicpu_path` and `aicore_path` aliases, so CUDA consumers now read the device
artifact through `path_for_role("device")` or `role_paths["device"]`.

The local A100 no-build host-schedule Worker smoke passed after the rebuild:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 --no-build
```

Result: `status=pass`, `mode=worker/add`, `ptx_arch=compute_80`,
`device_wall_ns=40960`.

The two-scalar affine host-schedule Worker smoke was captured on local A100
and remote H200 after adding the ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op affine --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-affine-smoke-dd026085/` contains A100 and
H200 JSON plus Markdown/SVG reports. A100 reported `status=pass`,
`ptx_arch=compute_80`, and `device_wall_ns=16384`; H200 reported
`status=pass`, `ptx_arch=compute_90`, and `device_wall_ns=41760`.

The three-input triad host-schedule Worker smoke was captured on local A100
and remote H200 after adding the `(a, b, c, out, n)` ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op triad --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-triad-smoke-1af18449/` contains A100 and
H200 JSON plus Markdown/SVG reports. A100 reported `status=pass`,
`ptx_arch=compute_80`, and `device_wall_ns=22528`; H200 reported
`status=pass`, `ptx_arch=compute_90`, and `device_wall_ns=42496`.

The local A100 and remote H200 persistent-device DAG smokes also passed after
building the native `device` role:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Result: A100 `status=pass`, `device_wall_ns=30720`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`; H200
`status=pass`, `ptx_arch=compute_90`, `device_wall_ns=24896`, zero scheduler
errors.

After promoting the scalar AXPY DAG to a benchmark baseline, the benchmark
report tests passed locally and the new single-baseline path was checked on
both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_axpy \
    --sizes 1024 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[4,2,1]`, `scalar_args={"scalar0":1.5}`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`; H200
`status=pass`, `ptx_arch=compute_90`, `dispatch_func_ids=[4,2,1]`,
`scalar_args={"scalar0":1.5}`, zero scheduler errors.

After promoting the unary host-schedule ABI to a benchmark baseline, the new
single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_unary_square \
    --sizes 1024 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_80`,
`device_wall_ns=8192`; H200 `status=pass`, `ptx_arch=compute_90`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_90`,
`device_wall_ns=13888`.

After adding the unary row to the default paired benchmark, the validator was
updated to compare CUDA `float32` square results. The previous exact Python
integer-square comparison failed at `N=65536` because CUDA stores
single-precision output:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_unary_square \
    --sizes 65536 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`ptx_source=kernel-compiler-task-body-wrapper-unary-square-compute_80`,
`device_wall_ns=848864`.

After adding a two-scalar persistent DAG descriptor, the focused local CUDA
test set was rerun:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_backend.py -q
```

Result: `133 passed`.

The two-scalar affine DAG smoke was captured on local A100 and remote H200
with tree sync and compact report generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape scalar_affine --task-count 3 --queue-capacity 2 \
    --n 4096 --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-scalar_affine-smoke-469f55cd/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[5,2,1]`,
`scalar_args={"scalar0":1.5,"scalar1":0.5}`, `device_wall_ns=28672`;
the H200 row returned `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[5,2,1]`, the same scalar args, and
`device_wall_ns=35584`. Both rows reported zero device scheduler errors.

After adding a third tensor pointer to the persistent DAG descriptor, the
triad DAG smoke was captured on local A100 and remote H200 with tree sync and
compact report generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape triad --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-triad-smoke-3a3bcdb1/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[6,2,1]`,
`tensor_args={"c":"tmp0"}`, `device_wall_ns=27648`; the H200 row returned
`status=pass`, `ptx_arch=compute_90`, the same dispatch IDs and tensor args,
and `device_wall_ns=24832`. Both rows reported zero device scheduler errors.

After adding a unary generated-dispatch task body, the unary-square DAG smoke
was captured on local A100 and remote H200 with tree sync and compact report
generation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape unary_square --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/persistent-unary_square-smoke-cb01f013/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[7,1,1]`, and
`device_wall_ns=30720`; the H200 row returned `status=pass`,
`ptx_arch=compute_90`, the same dispatch IDs, and `device_wall_ns=31136`.
Both rows reported zero device scheduler errors.

After promoting the unary-square DAG to a benchmark baseline, the new
single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_unary_square \
    --sizes 4096 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[7,1,1]`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`, and
`device_wall_ns=43008`; H200 `status=pass`, `ptx_arch=compute_90`, the same
dispatch IDs, zero scheduler errors, and `device_wall_ns=45184`.

After promoting the triad DAG to a benchmark baseline, the new single-baseline
path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_triad \
    --sizes 4096 --arch compute_80
```

Result: `tmp/cuda-backend/persistent-triad-baseline/` contains A100 and H200
JSON plus Markdown/SVG reports. A100 returned `status=pass`,
`ptx_arch=compute_80`, `dispatch_func_ids=[6,2,1]`,
`tensor_args={"c":"tmp0"}`, and `device_wall_ns=34816`; H200 returned
`status=pass`, `ptx_arch=compute_90`, the same dispatch IDs and tensor args,
and `device_wall_ns=33536`. Both rows reported zero device scheduler errors.

After promoting the two-scalar affine DAG to a benchmark baseline, the focused
report tests passed locally and the new single-baseline path was checked on
both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_affine \
    --sizes 4096 --arch compute_80
```

Result: A100 `status=pass`, `ptx_arch=compute_80`,
`dispatch_func_ids=[5,2,1]`,
`scalar_args={"scalar0":1.5,"scalar1":0.5}`,
`device_scheduler_errors={"count":0,"code":0,"task_id":0}`,
`device_wall_ns=31744`; H200 `status=pass`, `ptx_arch=compute_90`,
`dispatch_func_ids=[5,2,1]`, the same scalar args, zero scheduler errors,
and `device_wall_ns=30560`.

The full paired benchmark was then refreshed with scalar affine, triad, quad,
and unary-square rows in the default persistent baseline set:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-ba99b593/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`. The combined JSON has `666` samples, including
`18` `pto_persistent_dag_scalar_affine` samples and `18`
`pto_persistent_dag_triad` samples, `18`
`pto_persistent_dag_quad` samples, and `18`
`pto_persistent_dag_unary_square` samples. The compact DAG table reports
quad ratios versus `pto_persistent_dag` of `1.05x`, `1.19x`, and `1.19x` on
A100 for `N=1024,65536,1048576`, and `0.98x`, `1.07x`, and `1.01x` on H200
for the same sizes. The quad smoke and paired benchmark golden path now
matches NVCC's generated `mul.f32` plus `fma.rn.f32` sequence for
`a * b + c * d`, so the fourth tensor descriptor row is recorded as
correctness and scheduler-shape evidence. Throughput conclusions still need a
tuned tensor workload.

The combined capture was validated with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.json \
    --preset paired-current
```

Result: `validated tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.json`.

After adding `persistent_dag_unary_square_f32` to the normal SceneTestCase L2
persistent-device argument builders, the CUDA scene-test file was rerun
locally on A100:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py -q
```

Result: `30 passed`. The new unary persistent scene slice also uses a
ctypes-backed real-data case so it can run on the H200 checkout without
`torch`.

The same unary scene-test slice was run on the remote H200 checkout after
syncing the current local tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k unary_square'
```

Result: `2 passed, 28 deselected`.

The unary graph descriptor path was then promoted from a fixed
`persistent_dag_unary_square_f32` builder to an explicit runtime graph shape
named `graph_descriptor_unary_square`. The focused TDD check first failed
because the paired A100/H200 smoke runner rejected that shape at argument
parsing. After adding the smoke shape and graph-metadata expectations, the
focused local checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_descriptor_unary_square or unary_square_dag_shape'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_unary_square or unary_square'
```

Results: `3 passed, 215 deselected` for the smoke workflow tests, and
`4 passed, 71 deselected` for the SceneTestCase tests. The paired A100/H200
repeat-run smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_unary_square --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-unary-square-working
```

Result:
`tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required dispatch `[7,1,1]`,
`graph_descriptor.fanin=[0,1,1]`, `graph_descriptor.dependents=[1,2]`,
`launch_completed_counts=[3,3]`, zero scheduler errors,
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and `grid_dim=4`.

| GPU | Device ns | Host ns | Per-launch device ns |
| --- | --------- | ------- | -------------------- |
| A100 | 75776 | 106800 | `[49152,26624]` |
| H200 | 57056 | 74823 | `[36960,20096]` |

After adding the four-input host-schedule ABI, the no-torch Worker quad smoke
was captured on local A100 and remote H200 with runtime rebuild enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op quad --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-quad-smoke-4327698e/` contains `a100.json`,
`h200.json`, `cuda-smoke-report.md`, and `cuda-smoke-report.svg`. A100
reported `status=pass`, `ptx_arch=compute_80`, and `device_wall_ns=21504`;
H200 reported `status=pass`, `ptx_arch=compute_90`, and
`device_wall_ns=18752`. The local A100 quad smoke was also checked at
`N=65536` to exercise the same CUDA fused multiply-add rounding path used by
`ctx->a[i] * ctx->b[i] + ctx->c[i] * ctx->d[i]`.

After promoting the four-input host-schedule ABI to a benchmark baseline, the
focused single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_quad --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90`. Result:
`tmp/cuda-backend/host-quad-baseline-working/` contains `a100.json` and
`h200.json`. The A100 row reported `status=pass`,
`ptx_source=kernel-compiler-task-body-wrapper-quad-compute_80`, and
`device_wall_ns=8192`; H200 reported `status=pass`,
`ptx_source=kernel-compiler-task-body-wrapper-quad-compute_90`, and
`device_wall_ns=24960`.

The full paired current benchmark was refreshed at commit `c0dc1372`.
Result: `tmp/cuda-backend/combined-current-c0dc1372/` contains the raw JSON,
Markdown report, median-device SVG, and ratio SVG. The combined JSON has
`684` samples, including `18` `pto_host_schedule_quad` samples and `18`
`pto_persistent_dag_quad` samples. The paired-current validator reported:
`validated tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark.json`.

After adding generic persistent DAG tensor/scalar argument slots, the
`generic_args` smoke was run locally on A100 and remotely on H200 with a tree
sync. The graph uses generated-dispatch `func_id` sequence `[9, 2, 1]`; the
first task computes from the base tensor fields plus `tensor_args[0]`,
`tensor_args[1]`, `scalar_args[0]`, and `scalar_args[1]`, and the final task
joins with an independent `a * b` branch. Result:
`tmp/cuda-backend/persistent-generic_args-smoke-7c99f607/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. Both runs reported zero scheduler errors and
argument metadata
`scalar_args[0]=1.5,scalar_args[1]=0.25,tensor_args[0]=tmp0,tensor_args[1]=tmp3`.
The same descriptor shape now also has normal `SceneTestCase` L2 coverage
through `persistent_dag_generic_args_f32`, using ctypes-backed CPU tensors so
the path remains usable on the H200 host without requiring `torch`. The
focused local command passed on A100:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k generic_args_with_ctypes --platform cuda
```

The same command was run on H200 after syncing the working tree to
`bizhaoh200`; it passed with `1 passed, 35 deselected`.

After promoting generic persistent DAG tensor/scalar argument slots to a
benchmark baseline, the focused single-baseline path was checked on both GPUs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_generic_args --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90`. Result:
`tmp/cuda-backend/persistent-generic-args-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The A100 row reported `status=pass`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_80`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=30720`; H200 reported
`status=pass`, `ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=33600`.

The full paired current benchmark was then refreshed at commit `61cf96cd`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-61cf96cd/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`. The combined JSON has `720` samples, including
`18` `pto_persistent_dag_generic_args` samples and `18`
`pto_persistent_dag_graph` samples. All rows reported pass
status. The paired-current validator reported:
`validated tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json`.
The compact DAG table now includes `Graph Descriptor/DAG`; the H200 graph
descriptor ratios versus `pto_persistent_dag` are `0.95x`, `1.05x`, and
`1.00x` for `N=1024,65536,1048576`, while the A100 ratios are `1.08x`,
`1.19x`, and `1.05x`. Treat the DAG-shape rows as correctness and scheduler
shape evidence rather than tuned throughput claims.

The compact paired-current gate was refreshed again at commit `8e868bfe`
after the tensor-core scene-test and persistent `stream_id` plumbing changes.
It uses the WMMA-compatible `16x16x16` tensor descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-8e868bfe/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `56` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-8e868bfe/cuda-benchmark.json`.
This capture validates all 28 selected baselines on A100 and H200, including
`pto_persistent_dag_scalar_scale`, `pto_persistent_dag_graph_diamond`, and
`pto_persistent_dag_graph_tensor`. Selected A100 device times for
host/base-DAG/tensor/tensor-core/cuBLAS/grid-batch were
`29696/48128/38912/36864/53247/49152 ns`; H200 reported
`14880/36512/48960/33632/37631/30176 ns`.

The compact paired-current gate was refreshed at commit `2aedb40f` after
adding the host-schedule generic-args benchmark row. It uses the same command
shape, `N=1024`, one repeat, `batch_tasks=2`, `worker_blocks_per_task=4`, and
the WMMA-compatible `16x16x16` tensor descriptor.

Result: `tmp/cuda-backend/combined-current-2aedb40f/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `58` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-2aedb40f/cuda-benchmark.json`.
This capture validates all 29 selected baselines on A100 and H200, including
`pto_host_schedule_generic_args`. Selected A100 device times for host,
host-generic, base-DAG, persistent-generic, tensor, tensor-core, cuBLAS, and
grid-batch were `29696/43008/44032/29696/41984/41984/49152/40960 ns`; H200
reported `17920/36032/38112/31168/47008/32543/51520/32896 ns`.

The compact paired-current gate was refreshed again at commit `b2c5c8a4`
after promoting `pto_persistent_dag_graph_generic_args4` into the selected
benchmark path. It uses the same command shape, `N=1024`, one repeat,
`batch_tasks=2`, `worker_blocks_per_task=4`, and the WMMA-compatible
`16x16x16` tensor descriptor.

Result: `tmp/cuda-backend/combined-current-b2c5c8a4/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `60` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.json`.
This capture validates `30` selected benchmark rows per GPU, including
`pto_host_schedule_generic_args` and
`pto_persistent_dag_graph_generic_args4`. Selected A100 device times for host,
host-generic, base-DAG, persistent-generic, graph-generic4, tensor,
tensor-core, cuBLAS, and grid-batch were
`22528/35840/44032/29696/27648/37888/36864/37888/37888 ns`; H200 reported
`16992/31264/40320/30592/27520/48992/32480/34304/31872 ns`.

Earlier result: `tmp/cuda-backend/combined-current-d361006f/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`; it also writes
`cuda-benchmark-dag-deltas.svg`. New reports also write
`cuda-benchmark-throughput.svg` for tensor and cuBLAS rows. The combined JSON
has `50` samples and the paired-current validator reported:
`validated tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.json`.
This capture proves the default paired workflow now keeps
`pto_persistent_dag_tensor`, `pto_persistent_dag_tensor_core`, and
`cublas_sgemm` in one validated current-head report on A100 and H200.

The compact paired-current gate was refreshed at commit `0b3c1699` after
adding persistent DAG no-progress diagnostics. It uses the same command shape
and validates command examples, source-paper provenance, and generated report
files:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-0b3c1699/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, and `cuda-benchmark-dag-deltas.svg`. Regenerated
reports also include `cuda-benchmark-throughput.svg`. The combined JSON has
`50` samples and the validator reported:
`validated tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.json`.
Selected A100 device times for
host/base-DAG/tensor/tensor-core/cuBLAS/grid-batch were
`33792/61440/44032/59392/60416/41984 ns`; H200 reported
`14848/31936/44576/36096/40959/28768 ns`.

## Remaining Gaps

### Kernel Compiler Integration

Host-schedule task-body compilation and persistent-device generated dispatch
now have first `KernelCompiler` entry points. Both paths can consume
`CudaTaskBody` style sources. CUDA prepared-callable artifacts can be staged
through the L2 Python `Worker` registration path. The normal scene-test flow
can compile and run host-schedule CUDA vector-add, binary elementwise, unary
square, scalar scale, axpy, two-scalar affine, three-input triad, quad, and
generic tensor/scalar callable specs and persistent-device
fork/join, chain, reuse, scalar scale, scalar AXPY, scalar affine, and
tensor-tile DAG callable specs, plus third-tensor persistent triad and
unary-square callable specs, end to end.
The fourth-tensor persistent quad callable spec also runs end to end through
ctypes-backed real data. The host-schedule quad callable spec also runs
through both the normal `SceneTestCase` L2 path and the no-torch Worker smoke
path. The host-schedule generic-args callable spec now has ctypes-backed
scene tests, so it runs on H200 without requiring `torch`; it lowers the
original two-slot `tensor_args[0:2]` and `scalar_args[0:2]` path, plus the
four-slot `tensor_args[0:4]` and `scalar_args[0:4]` path, through the host
runtime launch ABI. A generic `persistent_dag_graph_f32`
adapter can now lower explicit
runtime graph descriptors with per-task `func_id`, dependency lists, fan-in,
temporary buffers, generic tensor slots, and scalar slots through the same L2
`SceneTestCase` path. The adapter now allocates default-sized temporaries for
graph task `out` names that do not match existing input/output tensors, so the
`temporaries` map is only needed for non-default temporary sizes. It also
supports a first dependency-inference slice: when a graph task omits
`dependents`, the adapter infers its outgoing edges from tensor flow by
binding `a`/`b`/`c`/`d` and `tensor_args` reads to the nearest previous
producer for that tensor name, or to a later producer when the descriptor is
intentionally out of topological order. The inference is per task, so mixed
descriptors can keep explicit dependency lists for some tasks while inferring
omitted edges for the remaining tasks.
The graph adapter also accepts incoming dependency lists through `depends_on`
or `dependencies`, lowering consumer-side task IDs into the same flattened
dependent array. That lets scene tests express graph edges independently from
the task's tensor pointer layout, which is closer to normal task-graph
metadata than CUDA-specific outgoing edge lists.
Those incoming edges may now reference named graph tasks as well as integer
task IDs. The named form keeps descriptor tests stable when tasks are inserted
or reordered and moves the CUDA graph adapter closer to normal named PTO task
graphs.
Outgoing `dependents` may now use the same task names, so both graph edge
directions share one descriptor naming model. The edge fields may be a single
task name/id or a list of task names/ids. The real-data
`graph_with_ctypes_data` scene now uses scalar named outgoing dependents and
passed on local A100 and remote H200; the H200 run printed the known PTO-ISA
SSH refresh warning before pytest reported `1 passed, 104 deselected`.
The graph adapter also accepts a top-level `edges` list with
`{"from": <task>, "to": <task>}` dictionaries, two-item endpoint pairs, or
`"<source> -> <target>"` strings. The ctypes-backed
`edge_list_graph_with_ctypes_data` scene passed locally on A100 with pytest
reporting `5 passed, 102 deselected` for the focused selector and remotely on
H200 with pytest reporting `1 passed, 106 deselected`; the H200 run printed
the known PTO-ISA SSH refresh warning first. The new string-edge selector
first failed at `_graph_edge_endpoints`, then passed on local A100 and remote
H200 with pytest reporting `2 passed, 125 deselected` on both systems.
Top-level edge dictionaries now also accept dep-gen-style `pred` and `succ`
endpoint keys, matching the structural edge vocabulary used in `deps.json`.
The focused builder selector first failed at `_graph_edge_endpoints`, then
passed with `3 passed, 144 deselected` for the adjacent edge forms. The
ctypes-backed `dep_gen_edge_graph_with_ctypes` scene passed on local A100 and
remote H200 with `1 passed, 146 deselected`; the H200 run printed the known
PTO-ISA SSH refresh warning first.
Graph task identity now also accepts dep-gen-style `task_id` as a `name`/`id`
alias, so `deps.json`-like task rows can be referenced directly from
`pred`/`succ` edge dictionaries. The focused builder selector first failed
with `unknown dependency task name: left`, then passed with
`1 passed, 148 deselected`. The adjacent identity/edge selector passed with
`4 passed, 145 deselected`. The ctypes-backed
`dep_gen_task_id_graph_with_ctypes` scene passed on local A100 with
`1 passed, 148 deselected` and on remote H200 with
`1 passed, 148 deselected`; the H200 run printed the known PTO-ISA SSH
refresh warning first.
Real `deps.json` edge rows also include annotation fields such as
`source="creator"` or `source="tensormap"`. The adapter now prefers
`pred`/`succ` over `source`/`target` when both forms appear, so the
annotation is not misread as a task endpoint. The focused builder selector
first failed with `unknown dependency task name: creator`, then passed with
`1 passed, 150 deselected`. The adjacent edge compatibility selector passed
with `5 passed, 146 deselected`, covering dep-gen, `from`/`to`, and
node-link `source`/`target` spellings. The ctypes-backed
`annotated_dep_gen_graph_with_ctypes` scene passed on local A100 and remote
H200 with `1 passed, 150 deselected`; the H200 run printed the known PTO-ISA
SSH refresh warning first.
Dep-gen task rows that only carry identity and metadata, such as `task_id` and
`scope`, can now inherit runnable CUDA descriptor fields from graph-level
`task_defaults`, `task_template`, or `default_task`. The focused selector first
failed with `KeyError: 'func_id'`, then passed locally on A100 with
`2 passed, 151 deselected` and remotely on H200 with
`2 passed, 151 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first. This keeps real `deps.json` task lists closer to their captured
shape while still letting scene tests provide the common callable and TaskArgs
metadata needed to execute the graph.
The adapter can now load that graph descriptor from a JSON file through
`graph_path` or `graph_file`, with inline `graph` metadata applied as an
overlay. The focused selector first failed because only the inline defaults
were seen and the graph had no tasks, then passed locally on A100 with
`2 passed, 153 deselected` and remotely on H200 with
`2 passed, 153 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first. The JSON-file test uses the v2 `deps.json` shape: large string
task IDs, `tasks[]`, `tensors[]`, and annotated `pred`/`succ` edges.
Imported graph tasks can now receive keyed `task_overrides` or `task_metadata`
after defaults and raw task rows are merged. The keyed override path keeps the
raw `deps.json` structural while mapping individual task IDs to different
CUDA `func_id` values and temporary/output bindings. The focused selector
first failed because task `4294967296` still dispatched `func_id=1` and the
ctypes output was plain `a+b`, then passed locally on A100 with
`2 passed, 155 deselected` and remotely on H200 with
`2 passed, 155 deselected`; the H200 run printed the known PTO-ISA SSH refresh
warning first.
List-shaped `task_metadata` sidecars are now accepted too. Each metadata entry
keys itself with `name`, `id`, or `task_id`, then merges through the same
override path. The TDD selector first failed with `graph task overrides must be
a dictionary`, then passed locally on A100 with `2 passed, 157 deselected`.
The same graph-shaped path now accepts `graph.tasks` as a dictionary keyed by
task name, so descriptor specs can keep node names in one place and reference
those names from top-level edges. The ctypes-backed
`task_dict_graph_with_ctypes_data` scene passed locally on A100 with pytest
reporting `2 passed, 107 deselected` and remotely on H200 with pytest
reporting `1 passed, 108 deselected`; the H200 run printed the known PTO-ISA
SSH refresh warning first.
Top-level `graph.edges` now also accepts adjacency dictionaries, mapping each
source task name/id to either one target or a list of targets. The
ctypes-backed `adjacency_graph_with_ctypes_data` scene passed locally on A100
with pytest reporting `2 passed, 109 deselected` and remotely on H200 with
pytest reporting `1 passed, 110 deselected`; the H200 run printed the known
PTO-ISA SSH refresh warning first.
The graph adapter now also accepts `graph.nodes` as an alias for
`graph.tasks`, so the same named-node and edge-map descriptor can use graph
terminology without changing the runtime task array. The ctypes-backed
`node_graph_with_ctypes_data` scene passed locally with `2 passed, 113
deselected`; the full local A100 CUDA scene-test file passed with
`115 passed`; and the H200 selector passed with `1 passed, 114 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
List-shaped graph nodes may now spell node identity as `id`, which normalizes
to the existing `name` field used by edge lookup. The focused selector first
failed with `unknown dependency task name: left`, then passed locally on A100
with `2 passed, 129 deselected` and remotely on H200 with `2 passed, 129
deselected`; the H200 run printed the known PTO-ISA SSH refresh warning
first.
Graph nodes now also accept node-style IO fields: `inputs`, `outputs`,
`output_existing`, `inouts`, and `scalars`. These fields expand into the
existing role-keyed `task_args` lowering path, so graph-node descriptors can
avoid embedding `a`/`b`/`out` ABI field names directly. The ctypes-backed
`node_io_graph_with_ctypes_data` scene passed locally with `2 passed, 115
deselected`; the full local A100 CUDA scene-test file passed with
`117 passed`; and the H200 selector passed with `1 passed, 116 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
Dictionary-valued `inputs` and `outputs` node-port maps now lower through the
same node-IO path. This covers graph schemas that carry edge ports as
`{"lhs": "a", "rhs": "b"}` or `{"value": "tmp0"}` rather than plain lists.
The focused TDD selector first failed because the adapter stringified the
port dictionaries into unknown tensor names; after flattening port values
deterministically, local A100 and synced H200 both reported
`2 passed, 135 deselected` for `-k node_port_dict --platform cuda`.
Graph nodes may now use `op` as an alias for `callable`, resolved through
`graph.callables` before node IO fields are lowered. This lets node
descriptors keep an operation-name spelling while still producing the same
generated-dispatch `func_id` task array. The ctypes-backed
`node_op_graph_with_ctypes_data` scene passed locally with `2 passed, 117
deselected`; the full local A100 CUDA scene-test file passed with
`119 passed`; and the H200 selector passed with `1 passed, 118 deselected`
while still emitting the known PTO-ISA SSH refresh warning.
Graph nodes may now keep non-IO metadata under an `attrs` dictionary. The
adapter merges `attrs` before node IO lowering, so a node can spell
`op`, `inputs`, and `outputs` as graph structure while keeping auxiliary
runtime payloads such as `tensor_args` and `scalar_args` out of the edge
metadata. The ctypes-backed `node_attrs_graph_with_ctypes_data` scene passed
locally with `2 passed, 119 deselected`; the full local A100 CUDA scene-test
file passed with `121 passed`; and the H200 selector passed with `1 passed,
120 deselected` while still emitting the known PTO-ISA SSH refresh warning.
The same graph-node attrs shape is now capturable through the paired
persistent-smoke workflow as `graph_descriptor_node_attrs`. The capture under
`tmp/cuda-backend/persistent-graph_descriptor_node_attrs-repeat2-smoke-b1b3e28c/`
contains A100/H200 JSON plus Markdown/SVG reports with dispatch `[9,2,1]`,
graph fan-in `[0,0,2]`, graph dependents `[2,2]`, repeat completions
`[3,3]`, zero scheduler errors, tensor slots
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, and report metadata
`graph_node_attrs.task0=attrs:tensor_args,scalar_args`. Device times were
`72704 ns` on A100 and `45408 ns` on H200 for `N=1024`.
The paired persistent-smoke validator now also requires the same attrs
metadata in both JSON and generated Markdown/SVG reports. The working-tree
capture under
`tmp/cuda-backend/persistent-node-attrs-smoke-working/`
`persistent-graph_descriptor_node_attrs-repeat2-smoke-761fbdfe/` contains
A100/H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[9,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, tensor slots
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, report-visible graph-node attrs
`task0=attrs:tensor_args,scalar_args`, and zero scheduler errors. A100
reported `device_wall_ns=67584`, `host_wall_ns=99134`; H200 reported
`device_wall_ns=42144`, `host_wall_ns=60174`.
The paired smoke validator now also has first-class scalar/tensor argument
checks, so the attrs path cannot silently lose the payload it is meant to
carry. The stricter working-tree capture under
`tmp/cuda-backend/persistent-node-attrs-args-smoke-working/`
`persistent-graph_descriptor_node_attrs-repeat2-smoke-46c2f848/` validated
the same dispatch/topology, report-visible graph-node attrs, scalar slots,
and tensor slots on A100 and H200. A100 reported `device_wall_ns=62464`,
`host_wall_ns=93490`; H200 reported `device_wall_ns=40672`,
`host_wall_ns=57904`.
The same shape is also promoted into the selected benchmark matrix as
`pto_persistent_dag_graph_node_attrs`. The compact paired A100/H200 capture
under
`tmp/cuda-backend/graph-node-attrs-benchmark-working/combined-current-3d129351/`
validated `78` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[9,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`, and
`graph_node_attrs.task0=attrs:tensor_args,scalar_args`. The benchmark
validator now also requires the row and generated Markdown/SVG reports to
carry `scalar_args[0]=1.5,scalar_args[1]=0.25` and
`tensor_args[0]=tmp0,tensor_args[1]=tmp3`, so the selected benchmark gate
cannot preserve the node-attrs label while losing its descriptor payload.
The compact `cuda_current_summary.py --section graph-metadata` table now
renders those scalar/tensor descriptor maps as well, so copied evaluation
tables keep the payload visible.
Device times were `40960 ns` on A100 and `33152 ns` on H200 for `N=1024`.
A current-head no-batch paired compact gate under
`tmp/cuda-backend/current-head-compact-args-summary-working/combined-current-7191db4e/`
refreshes the selected benchmark matrix after that summary update. It
validated `88` A100/H200 samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, visible graph
topology, visible tensor throughput, and report-visible scalar/tensor
descriptor payloads for `pto_persistent_dag_graph_node_attrs`. The same raw
JSON reports A100/H200 graph-node-attrs device times of `29696/31072 ns`,
graph tensor-core device times of `39936/31936 ns`, and cuBLAS Graph replay
times of `13311/9088 ns` for the default `16x16x16`, `N=1024` compact
descriptor.
The graph-node callable-alias path is now also promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_op`. The compact paired
A100/H200 capture under
`tmp/cuda-backend/graph-node-op-benchmark-working/combined-current-7edfb7df/`
validated `88` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`, and
`graph_node_ops=task0=op:add=1;task1=op:mul=2;task2=op:add=1`. Device times
were `31744 ns` on A100 and `25536 ns` on H200 for `N=1024`.
The graph-node input/output spelling is now promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_io`. The compact paired
A100/H200 capture under
`tmp/cuda-backend/graph-node-io-benchmark-working/combined-current-c0d327d2/`
validated `98` benchmark samples with source-paper provenance, sanitized
command examples, Markdown/SVG reports, zero scheduler errors, dispatch
`[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
`graph_task_arg_key=node_io`, and task args
`task0=input:a,input:b,output:tmp0`,
`task1=input:a,input:b,output:tmp1`, and
`task2=input:a,input:b,output:out`.
Device times were `28672 ns` on A100 and `25632 ns` on H200 for `N=1024`.
The node-link graph descriptor spelling is now promoted into the selected
benchmark matrix as `pto_persistent_dag_graph_node_link`. It covers
`graph.nodes[*].id`, nested node `data`, and `graph.links` through the same
add/mul/add callable sequence used by the graph-node `op` row. The compact
A100/H200 capture under
`tmp/cuda-backend/graph-node-link-compact-current-preset-working/`
`combined-current-8a74e5ab/` validates the current `compact-current` preset:
`102` rows, source-paper provenance, Markdown/SVG reports, graph topology and
task-argument report metadata, tensor-throughput SVG output, sanitized command
examples, zero scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`,
graph dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, and device times of
`35840 ns` on A100 and `31808 ns` on H200 for `N=1024`.
The same node-IO graph descriptor path is now also covered by the paired
persistent-smoke report validator. The working-tree capture under
`tmp/cuda-backend/persistent-node-io-smoke-working/`
`persistent-graph_descriptor_node_io-repeat2-smoke-feddd21b/` contains A100
and H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, report-visible
`graph_task_arg_key=node_io`, task args
`task0=input:a,input:b,output:tmp0;task1=input:a,input:b,output:tmp1;task2=input:a,input:b,output:out`,
and zero scheduler errors. A100 reported `device_wall_ns=68608`,
`host_wall_ns=102364`; H200 reported `device_wall_ns=42784`,
`host_wall_ns=59753`.
The same node-op graph descriptor path is now also covered by the paired
persistent-smoke report validator. The working-tree capture under
`tmp/cuda-backend/persistent-node-op-smoke-working/`
`persistent-graph_descriptor_node_op-repeat2-smoke-32e3b1ae/` contains A100
and H200 JSON plus Markdown/SVG artifacts. The paired validator accepted
dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents `[2,2]`,
repeat completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, report-visible graph-node
ops `task0=op:add=1;task1=op:mul=2;task2=op:add=1`, and zero scheduler errors.
A100 reported `device_wall_ns=65536`, `host_wall_ns=97710`; H200 reported
`device_wall_ns=55296`, `host_wall_ns=78895`.
The incoming-edge path is now covered by both a real-data L2 ctypes scene and
paired persistent-device smoke. The working-tree smoke capture under
`tmp/cuda-backend/depends-on-graph-working/persistent-graph_descriptor_depends_on-repeat2-smoke-06b988b5/`
contains A100/H200 JSON, Markdown, and SVG artifacts for
`graph_descriptor_depends_on` with dispatch `[1,2,1]`, graph fan-in
`[0,0,2]`, graph dependents `[2,2]`, `launch_completed_counts=[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
`grid_dim=4`, and zero scheduler errors. This proves the CUDA runtime can
schedule edges supplied as consumer-side metadata even when the consumer's
tensor pointers stay bound to the original graph inputs.
The same real-data scene fixture now uses named `depends_on` entries. It
passed locally on A100 and remotely on H200 with selector
`depends_on_graph_with_ctypes_data`; the H200 run printed the known PTO-ISA
SSH refresh warning before pytest reported `1 passed, 101 deselected`.
The same graph notation is now promoted into the selected benchmark path as
`pto_persistent_dag_graph_depends_on`. The compact paired capture under
`tmp/cuda-backend/graph-depends-benchmark-working/combined-current-01ddf564/`
validated `84` A100/H200 samples, source-paper provenance, command examples,
Markdown/SVG report files, zero scheduler errors, and graph topology metadata.
The depends-on row recorded dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, and device times of `30720 ns` on A100 and `26112 ns` on
H200 for `N=1024`.
The graph adapter now accepts a role-keyed `task_args` task form as a first
TaskArgs-like lowering slice: `input`, `output`, `output_existing`, and
`inout` roles are lowered to the existing bounded CUDA graph descriptor fields
before temporary allocation, tensor-flow dependency inference, and task struct
construction. The adapter prefers the `role` key and still accepts the older
`tag` spelling for compatibility. This lets a scene test describe a
persistent graph in terms of task-argument roles while still using the current
statically compiled generated-dispatch callable.
The role-keyed `task_args` form also accepts scalar inputs, lowering them
through the same bounded `scalar_args` slots as explicit graph descriptors.
Scalar entries still resolve through normal `TaskArgsBuilder` scalar names
before the descriptor is launched, so a graph descriptor can now keep tensor
roles and scalar inputs in one TaskArgs-like list.
The graph adapter now also resolves named graph callables before lowering
role-keyed `task_args`: a descriptor may define `graph.callables` as either a
dictionary keyed by callable name or a list of callable specs with `name`
fields, then each graph task can use `callable: "name"` or, for list-shaped
registries, the zero-based callable index instead of embedding the raw
generated-dispatch ID. Task-local fields override callable defaults, so the
resulting descriptor still records the same generated dispatch while the
scene-test graph shape is closer to normal PTO
`submit_next_level(callable, TaskArgs, ...)` submissions.
The list-shaped callable registry slice is covered by descriptor tests and by
the real-data scene selector
`named_callable_list_graph_with_ctypes_data`. That selector passed on the
local A100 and remote H200; the H200 run also printed the known PTO-ISA SSH
refresh warning before pytest reported `1 passed, 84 deselected`.
The callable-index slice is covered by
`callable_index_graph_with_ctypes_data`, which passed on the local A100 and
remote H200 with the same known H200 PTO-ISA SSH refresh warning.
The index-only registry slice is covered by
`unnamed_callable_index_graph_with_ctypes_data`; list entries now only need
`name` when graph tasks reference callables by name. That selector passed on
the local A100 and remote H200, again with the known H200 PTO-ISA SSH refresh
warning.
The compact callable-registry slice is covered by
`compact_callable_index_graph_with_ctypes_data`; integer list entries now
lower directly to generated-dispatch `func_id` values for index-referenced
graphs. The same compact form now works for dictionary registries such as
`{"add": 1}` and is covered by
`compact_callable_dict_graph_with_ctypes_data`. The focused selector passed
locally with `2 passed, 111 deselected`, the full local A100 CUDA scene-test
file passed with `113 passed`, and the H200 selector passed with `1 passed,
112 deselected` while still emitting the known PTO-ISA SSH refresh warning.
Callable metadata in graph descriptors may now spell the generated-dispatch
ID as `func_id`, `callable_id`, or `cid`. The aliases normalize before
role-keyed `task_args` and node IO lowering, so descriptor tests can use the
same `callable_id` / `cid` terminology as the normal Worker dispatch path
while preserving the current CUDA task ABI. The focused TDD selector first
failed with `KeyError: 'func_id'`, then passed on local A100 with `2 passed,
127 deselected`; the same selector passed on remote H200 with `2 passed, 127
deselected` after syncing the working tree, with the known PTO-ISA SSH
refresh warning printed first.
The role-keyed task-argument slice is covered by descriptor and real-data
selectors
`role_keyed_task_args` and `role_keyed_inout_graph_with_ctypes_data`. These
first failed because `role` was ignored and `tmp0` was treated as an input
before it existed. After adding shared `role`/`tag` normalization, the focused
local A100 selector reported `2 passed, 91 deselected`; the remote H200
real-data selector reported `1 passed, 92 deselected` with the known PTO-ISA
SSH refresh warning.
Graph task descriptors now also accept `args` as a short alias for
`task_args`, matching the argument slot in
`submit_next_level(callable, TaskArgs, ...)` more closely while preserving
the same role-keyed lowering. The descriptor-only regression first failed
with fan-in `[0,0,0]` because `args` was ignored. After wiring the alias into
the same normalization path, the local A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'args_alias' --platform cuda
```

reported `2 passed, 121 deselected`, covering both the descriptor and a
ctypes-backed real-data persistent-device graph scene. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q -k args_alias --platform cuda'
```

also reported `2 passed, 121 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph task descriptors now also accept role-map `task_args` dictionaries such
as `{"inputs": ["a", "b"], "output": "tmp0"}`. This keeps the descriptor
closer to a bundled `TaskArgs` object while preserving CUDA descriptor field
ordering. The focused regression first failed because the adapter iterated
dictionary keys as task-arg entries. After adding role-map normalization, the
descriptor selector reported `4 passed, 139 deselected`, the local A100 ctypes
selector reported `1 passed, 142 deselected`, and the synced H200 selector
reported `1 passed, 142 deselected` after the known PTO-ISA SSH refresh
warning.
The paired persistent-smoke/report path now also has a dedicated
`graph_descriptor_role_map_inout` shape. The focused report-helper TDD first
failed because both the persistent smoke builder and paired-smoke CLI rejected
that shape. After registering it, the focused selector reported
`2 passed, 296 deselected`. The paired A100/H200 smoke under
`tmp/cuda-backend/role-map-inout-working/`
`persistent-graph_descriptor_role_map_inout-repeat2-smoke-63e71c8a/`
validated `repeat_runs=2`, dispatch `1,1,1`, graph fan-in `0,1,1`,
dependents `1,2`, `graph_task_arg_key=role_map`, generated Markdown/SVG
reports, and zero scheduler errors. A100 reported `device_wall_ns=69632`;
H200 reported `device_wall_ns=49888`.
Graph task-list descriptors now also accept top-level `graph.submits` or
`graph.submissions` as aliases for `graph.tasks`/`graph.nodes`, so a
scene-test descriptor can look like a list of
`submit_next_level(callable, TaskArgs, ...)` calls. Submit entries still use
the same named/indexed callable resolution, `args`/`task_args` role lowering,
temporary allocation, and tensor-flow edge inference as task entries. The
focused regression first failed with
`CUDA persistent_dag_graph_f32 requires at least one graph task` because the
adapter ignored `submits`. After adding the alias, the local A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'submits_alias or args_alias' --platform cuda
```

reported `4 passed, 135 deselected`, covering descriptor construction and
ctypes-backed real-data persistent-device graph scenes. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q \
   -k "submits_alias or args_alias" --platform cuda'
```

also reported `4 passed, 135 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph descriptors now also accept `graph.submit_groups` or
`graph.submission_groups` as a bridge toward
`submit_next_level_group(callable, args_list, config)`. The current CUDA
persistent-device tracer bullet expands each `args_list` entry into one CUDA
DAG task while preserving callable resolution, TaskArgs-like role lowering,
temporary allocation, and tensor-flow dependency inference. This is not yet
the final PTO one-slot group semantics. The focused regression first failed
with `CUDA persistent_dag_graph_f32 requires at least one graph task` because
the adapter ignored `submit_groups`. After adding group expansion, the local
A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'submit_groups or submits_alias' --platform cuda
```

reported `4 passed, 137 deselected`, covering descriptor construction and
ctypes-backed real-data persistent-device graph scenes. The same synced
working tree on H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && CUDA_HOME=/usr/local/cuda \
   PATH=/usr/local/cuda/bin:$PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
   tests/ut/py/test_cuda_scene_test.py -q \
   -k "submit_groups or submits_alias" --platform cuda'
```

also reported `4 passed, 137 deselected`, after the known PTO-ISA SSH refresh
warning.
Graph task descriptors now also accept two-item role/name pairs in the
`task_args` list, such as `("input", "a")`, `("output", "tmp0")`,
`("inout", "tmp0")`, and `("output_existing", "out")`. These pair entries
normalize into the same role-keyed lowering path as expanded dictionaries,
so they preserve temporary allocation, existing-output aliasing, and
tensor-flow dependency inference. The focused TDD selector first failed at
descriptor construction because non-dictionary entries were rejected. After
adding pair normalization, the local A100 selector:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k pair_task_args --platform cuda
```

reported `2 passed, 123 deselected`, covering both descriptor construction
and a ctypes-backed real-data persistent-device graph scene. The same synced
working tree on H200 reported `2 passed, 123 deselected` with the known
PTO-ISA SSH refresh warning. A neighboring local selector for
`pair_task_args`, `role_keyed_task_args`, `args_alias`, and
`compact_role_task_args` reported `6 passed, 119 deselected`, checking that
the new shorthand did not regress the existing graph task-argument spellings.
The role mapping now preserves the lifecycle distinction needed by CUDA
memory planning: role `output` may create a default-sized temporary, but
roles `output_existing` and `inout` must name storage that is already known at
that point in descriptor order. Descriptor construction raises before launch
if either role references an unknown tensor or temporary, avoiding a silent
scratch allocation for values that are supposed to alias existing storage.
The same lifecycle rule is now applied to explicit graph `out_storage`.
Logical `out` names still create default-sized temporaries, but an
`out_storage` alias must point at storage that has already been allocated or
declared. This keeps scratch-buffer reuse explicit and prevents typos in the
physical storage name from allocating a new buffer silently.
The negative lifecycle cases are now covered by descriptor-only regression
tests: unknown `output_existing`, unknown `inout`, and unknown `out_storage`
names fail before task struct construction. The combined tagged role selector
reported `4 passed, 68 deselected`; the scratch-storage selector reported
`2 passed, 71 deselected`; and the valid tagged-inout and scratch-reuse
real-data selectors still reported `1 passed` each on local A100 and remote
H200.

The tagged graph lowering was checked with a failing test first, then local
A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k tagged_task_args -m 'not requires_hardware'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k tagged_graph --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k tagged_graph --platform cuda'
```

Results: the descriptor-only test reported `1 passed, 67 deselected`; the
local A100 real-data tagged graph scene reported `1 passed, 67 deselected`;
and the H200 real-data tagged graph scene reported `1 passed, 67 deselected`
after the known PTO-ISA SSH refresh warning.
After adding scalar entries to tagged graph `task_args`, the descriptor and
local A100 real-data selector was rerun:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'scalar_task_args or tagged_graph' --platform cuda
```

Result: `2 passed, 77 deselected`.
After syncing the branch tree to H200 with `rsync`, the same selector returned
`2 passed, 77 deselected` with the known PTO-ISA SSH refresh warning.

Named-callable graph lowering was then added under TDD. The first real-data
selector failed with `KeyError: 'func_id'` because graph tasks using
`callable: "generic"` still reached task construction without resolving the
callable metadata. After adding callable-name resolution, the local A100
descriptor and real-data selectors passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'named_callable or named_callables'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'unknown_cuda_persistent_graph_callable_name'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k named_callable_graph_with_ctypes_data --platform cuda
```

Results: the first selector reported `2 passed, 80 deselected`, the unknown
callable-name guard reported `1 passed, 81 deselected`, and the local A100
real-data named-callable graph scene reported `1 passed, 81 deselected`. The
same real-data selector passed on the H200 checkout after syncing the touched
files: `1 passed, 81 deselected`, again with the known PTO-ISA SSH refresh
warning.
Tagged `inout` graph lowering was then covered with a failing descriptor test
that first produced fan-in `[0,0,1]`, leaving the in-place update as an
incorrect root. After changing tensor-flow inference to prefer the nearest
previous producer for duplicate logical tensor names, the tagged inout
descriptor and real-data ctypes scene passed locally on A100, and the same
selector passed on H200 after syncing the tree:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k tagged_inout_graph -m 'not requires_hardware'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k tagged_inout_graph \
  --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k tagged_inout_graph --platform cuda'
```

Results: descriptor-only `1 passed, 71 deselected`; local A100 real-data
`1 passed, 71 deselected`; remote H200 real-data `1 passed, 71 deselected`
with the known PTO-ISA SSH refresh warning.
The same tagged `inout` graph shape is now part of the no-torch paired
persistent-smoke workflow as `graph_descriptor_tagged_inout`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_tagged_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/tagged-inout-working
```

Result:
`tmp/cuda-backend/tagged-inout-working/persistent-graph_descriptor_tagged_inout-repeat2-smoke-a8b7819c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`, tagged task args
`input:a,input:b,output:tmp1`, `inout:tmp1,input:b`, and
`input:tmp1,input:a,output_existing:out`, repeat completions `[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
and zero scheduler errors on both GPUs. A100 reported per-launch device times
`[46080,25600]` and H200 reported `[28960,20512]`.
The role-keyed variant of the same in-place graph shape is now covered by
`graph_descriptor_role_keyed_inout` in the paired persistent-smoke workflow:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_role_keyed_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-inout-working
```

Result:
`tmp/cuda-backend/role-keyed-inout-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-5075b400/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`, `graph_task_arg_key=role`,
task args `input:a,input:b,output:tmp1`, `inout:tmp1,input:b`, and
`input:tmp1,input:a,output_existing:out`, repeat completions `[3,3]`,
resource policy `scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`,
and zero scheduler errors on both GPUs. A100 reported per-launch device times
`[43008,26624]` and H200 reported `[41792,25440]`. The generated smoke report
adds a visible `Graph task arg key` column, so the artifact distinguishes the
preferred `role` spelling from the older `tag` spelling.

The graph adapter also accepts compact role-keyed task-argument entries such
as `{"input": "a"}`, `{"output": "tmp0"}`, `{"inout": "tmp0"}`, and
`{"output_existing": "out"}`. This keeps graph descriptors closer to a
TaskArgs-style role map without repeating `tensor` plus `role` in every
entry. Mixed compact/expanded entries are rejected before task construction.
The compact role-entry slice was checked with a failing descriptor test first,
then with local A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'compact_role_task_args or compact_role_graph_with_ctypes_data or \
      mixed_cuda_persistent_compact_role_task_arg' --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k compact_role_graph_with_ctypes_data --platform cuda'
```

Results: the local A100 selector reported `3 passed, 93 deselected`; the H200
real-data selector reported `1 passed, 94 deselected` with the known PTO-ISA
SSH refresh warning.

The same compact role-entry graph shape is now covered by the no-torch paired
persistent-smoke workflow as `graph_descriptor_compact_role_inout`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_compact_role_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-inout-working
```

Result:
`tmp/cuda-backend/compact-role-inout-working/persistent-graph_descriptor_compact_role_inout-repeat2-smoke-1fbef8c4/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=compact`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, and `input:tmp1,input:a,output_existing:out`, repeat
completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, and zero scheduler errors on both GPUs.
A100 reported per-launch device times `[43008,23552]`; H200 reported
`[28896,20224]`. The generated Markdown and SVG smoke reports expose the
compact task-argument key beside the graph topology.

The same role-keyed graph smoke was rerun at current head after lifecycle
matrix indexing landed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_role_keyed_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/current-head-role-keyed-working
```

Artifact:
`tmp/cuda-backend/current-head-role-keyed-working/persistent-graph_descriptor_role_keyed_inout-repeat2-smoke-8030fc57/`

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator again required dispatch
`[1,1,1]`, fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=role`, repeat completions `[3,3]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and zero scheduler
errors. A100 reported per-launch device times `[50176,26624]`; H200 reported
`[21664,20448]`. The refreshed local index for that output root records the
same role-keyed graph metadata in one row.

The same role-keyed graph shape is also part of the selected paired benchmark
matrix as `pto_persistent_dag_graph_role_keyed_inout`. A compact A100/H200
capture:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-benchmark-working
```

validated `72` rows under
`tmp/cuda-backend/role-keyed-benchmark-working/combined-current-a7787008/`.
The capture required dispatch `[1,1,1]`, graph fan-in `[0,1,1]`,
dependents `[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=role`. The role-keyed row reported A100
`device_wall_ns=38912`, `host_wall_ns=52853`; H200
`device_wall_ns=20864`, `host_wall_ns=2446166`. Both rows passed with zero
device scheduler errors.

The compact role-entry spelling is now also part of the selected paired
benchmark matrix as `pto_persistent_dag_graph_compact_role_inout`, beside the
role-keyed row. A failing benchmark/report test first required the new row in
`cuda_benchmark.py`, `cuda_pair_benchmark.py`, `cuda_validate_capture.py`, and
`cuda_current_summary.py`. A no-batch A100/H200 capture:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-benchmark-working
```

validated `74` rows under
`tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f/`.
The capture required dispatch `[1,1,1]`, graph fan-in `[0,1,1]`,
dependents `[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=compact`. The compact role row reported A100
`device_wall_ns=50176`, `host_wall_ns=62766`; H200
`device_wall_ns=27360`, `host_wall_ns=36621`. Both rows passed with zero
device scheduler errors, and the current-summary DAG table now includes a
`Graph Compact Role Inout/DAG` column.
The combined benchmark report was regenerated under
`tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f-report-role-spelling/`
so `cuda-benchmark.md` now has a focused `Graph Role Spelling Rows` section
and `cuda-benchmark.svg` exposes the same tag/role/compact rows in
`graph role spelling:` metadata.

The graph adapter also accepts pair-shaped task-argument entries, where each
entry is written as a two-item role/name pair. This spelling is useful as a
minimal structured form when JSON-like role maps would add unnecessary keys,
but it must still lower to the same TaskArgs role flow as tag, role-keyed,
and compact spellings. A failing benchmark/report selector first required
`graph_descriptor_pair_inout` and
`pto_persistent_dag_graph_pair_inout` in the persistent smoke, paired smoke,
benchmark, validator, and current-summary scripts. The focused selector then
passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q -k 'pair_inout or graph_role_spelling or \
          validate_command_matches_configured_capture or paired_current or \
          compact_current or current_summary'
```

Result: `17 passed, 267 deselected`.

The no-torch paired persistent-smoke workflow now covers the same in-place
graph shape as `graph_descriptor_pair_inout`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_pair_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-pair-inout-smoke-working
```

Result:
`tmp/cuda-backend/persistent-pair-inout-smoke-working/persistent-graph_descriptor_pair_inout-repeat2-smoke-5028d521/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required dispatch `[1,1,1]`,
graph fan-in `[0,1,1]`, dependents `[1,2]`,
`graph_task_arg_key=pair`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, and `input:tmp1,input:a,output_existing:out`, repeat
completions `[3,3]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, and zero scheduler errors on both GPUs.
A100 reported per-launch device times `[40960,25600]`; H200 reported
`[29760,20928]`. Supplemental single-baseline samples under
`tmp/cuda-backend/pair-inout-single-benchmark/` reported A100
`device_wall_ns=49152`, `host_wall_ns=68485`, and H200
`device_wall_ns=28672`, `host_wall_ns=38506` for
`pto_persistent_dag_graph_pair_inout`.

The pair-shaped graph spelling is now also covered by the selected compact
A100/H200 benchmark matrix:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/pair-current-compact-working
```

Result:
`tmp/cuda-backend/pair-current-compact-working/combined-current-c5094aa5/`
contains `cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The paired validator required `92` rows,
source-paper provenance, generated report files, zero scheduler errors, and
the pair row's dispatch `[1,1,1]`, graph fan-in `[0,1,1]`, dependents
`[1,2]`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=pair`. The pair row reported A100
`device_wall_ns=39936`, `host_wall_ns=54972`; H200 reported
`device_wall_ns=25120`, `host_wall_ns=33814`. The current-summary DAG table
now includes `Graph Pair Inout/DAG`; the compact selected capture reported
A100 `0.87x` and H200 `0.68x` versus the base
`pto_persistent_dag` row.

The same tagged graph shape is now also in the paired persistent-smoke report
flow as `graph_descriptor_tagged`, with scalar inputs recorded beside tensor
roles in `graph_task_args`. The current A100/H200 JSON plus Markdown/SVG
artifacts are under
`tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/`.
The paired validator accepted dispatch `9,2,1`, fan-in `[0,0,2]`,
dependents `[2,2]`, repeat completions `[3,3]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, scalar metadata
`scalar_args[0]=1.5`, `scalar_args[1]=0.25`, tagged task args
`input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]`,
`input:a,input:b,output:tmp2`, and
`input:tmp1,input:tmp2,output_existing:out`, and zero scheduler errors on
both GPUs. A100 reported per-launch device times `[44032,24576]`; H200
reported `[24768,18720]`. The regenerated Markdown/SVG smoke report and
artifact index also show graph fan-in/dependents and `Graph task args`, so
this artifact visibly ties the paired hardware result back to the tagged
tensor/scalar task-argument lowering form.

The host-schedule generic-args adapter was checked with a failing test first,
then local A100 and remote H200 real-data ctypes scene tests:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'builds_cuda_elementwise_generic_args' --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'host_schedule_elementwise_generic_args_with_ctypes_data' \
  --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k host_schedule_elementwise_generic_args_with_ctypes_data \
     --platform cuda'
```

Results: the local unit check reported `1 passed, 55 deselected`, the local
A100 real-data scene reported `1 passed, 55 deselected`, and the H200
real-data scene reported `1 passed, 55 deselected` after the known PTO-ISA SSH
refresh warning.

The same generic host-schedule ABI is now covered by the no-torch Worker
smoke runner, so it can be captured without the `SceneTestCase` framework:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op generic_args --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-generic_args-smoke-72c8186c/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The artifact validator accepted both rows with
`runtime=host_schedule` and `mode=worker/generic_args`; the A100 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args-compute_80` and
`device_wall_ns=35840`, while the H200 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args-compute_90` and
`device_wall_ns=15488`.

After widening host-schedule generic args to the four tensor/scalar slots
already present in `CudaVectorGenericArgs`, the new `generic_args4`
ctypes `SceneTestCase` and no-torch Worker smoke were run on A100 and H200.
Focused local checks:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'builds_cuda_elementwise_generic_args' --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'host_schedule_elementwise_generic_args_with_ctypes_data or \
      host_schedule_elementwise_generic_args_four_slots_with_ctypes_data' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'generic_args4_helpers or generic_args_helpers_use_aux_tensor'
```

The paired no-torch Worker smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op generic_args4 --sync-remote-tree --build-runtime
```

Result: `tmp/cuda-backend/worker-generic_args4-smoke-03ed75da/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The artifact validator accepted both rows with
`runtime=host_schedule` and `mode=worker/generic_args4`; the A100 row
reported
`ptx_source=kernel-compiler-worker-task-body-generic_args4-compute_80` and
`device_wall_ns=26624`, while the H200 row reported
`ptx_source=kernel-compiler-worker-task-body-generic_args4-compute_90` and
`device_wall_ns=19552`.

The same host-schedule generic-args path is now a benchmark baseline:
`pto_host_schedule_generic_args`. It compiles a generated task-body wrapper
for the generic tensor/scalar packet and uses the same indexed tensor/scalar
values as the smoke path:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_generic_args \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --label host-generic-args-baseline-a100
```

Result: `tmp/cuda-backend/host-generic-args-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-benchmark.json`, `cuda-benchmark.md`, and
SVG report files. The capture validator accepted both A100 and H200 rows with
`baseline=pto_host_schedule_generic_args`, `N=4096`, source-paper provenance,
and report files. The A100 row reported
`ptx_source=kernel-compiler-task-body-wrapper-generic-args-compute_80` and
`device_wall_ns=32768`; the H200 row reported
`ptx_source=kernel-compiler-task-body-wrapper-generic-args-compute_90` and
`device_wall_ns=17664`.

The graph-descriptor adapter was checked with focused local tests:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k 'graph_args or graph_edges or graph_temporaries or mixed_graph' \
  --platform cuda

.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k graph_with_ctypes --platform cuda
```

Result: the focused descriptor and real-data graph tests passed locally on
A100. The full CUDA scene-test file was also rerun locally and reported
`48 passed`.

The mixed explicit/inferred graph path was also run on the remote H200 after
syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k mixed_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 47 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The graph adapter now also forwards tensor-tile descriptor fields from each
graph task: rows, columns, inner dimension, leading dimensions, and per-tile
strides. This lets an explicit `persistent_dag_graph_f32` descriptor run the
same scalar tiled-GEMM first task as `persistent_dag_tensor_tile_f32`, then
feed residual, gate, and fan-in elementwise tasks. The focused red/green test
first failed with `rows == 0`, then passed after descriptor-field lowering.

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k graph_tensor_tile_args \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q -k 'graph_tensor_tile' \
  --platform cuda
```

Result: the second command reported `2 passed, 52 deselected` on the local
A100 path, covering both the struct descriptor and a real-data ctypes scene.
The full local CUDA scene-test file was also rerun after this adapter and
reported `54 passed`. The same no-torch real-data scene was run on the remote
H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k graph_tensor_tile_with_ctypes_data --platform cuda'
```

Result: `1 passed, 53 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The same explicit graph tensor-tile shape is now part of the persistent smoke
tooling as `--dag-shape graph_tensor_tile`. It records both graph dependency
metadata and tensor-tile descriptor metadata in the smoke JSON, then validates
the paired A100/H200 artifacts with expected dispatch `3,1,2,1`, completed
count `4`, repeat count `2`, tensor descriptor `16x16x16`, and generated
Markdown/SVG report files.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 512 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape graph_tensor_tile \
    --repeat-runs 2 --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --output-json tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/a100.json
```

The H200 run used the same command after syncing the working tree, with
`--arch compute_90` and output
`tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/h200.json`.
The report and validator commands then produced:

```text
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/cuda-smoke-report.md
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/cuda-smoke-report.svg
validated tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/a100.json,
tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/h200.json
```

Result summary: A100 reported `device_wall_ns=98304`, H200 reported
`device_wall_ns=67968`, and both reported zero scheduler errors,
`launch_completed_counts=[4,4]`, dispatch `[3,1,2,1]`, and
`graph_descriptor.dependents=[1,2,3,3]`.

The graph tensor-core smoke then added the same explicit graph descriptor
coverage for the WMMA first task. The focused red test first failed because
`graph_tensor_core_tile` was not accepted as a DAG shape and the paired
runner did not classify it as a tensor-tile shape. After adding the shape, the
focused unit selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k graph_tensor_core_tile
```

Result: `2 passed, 218 deselected`.

Paired A100/H200 evidence used the graph tensor-core shape with two launches:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_tensor_core_tile --task-count 4 --queue-capacity 2 \
    --repeat-runs 2 --n 256 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-core-working
```

The artifact directory is:

```text
tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/
```

Both GPUs validated `runtime=persistent_device`, `mode=dag`, dispatch
`[10,1,2,1]`, graph fan-in `[0,1,1,2]`, dependents `[1,2,3,3]`, tensor tile
`16x16x16`, tensor-core metadata `wmma:m16n16k8:tf32->f32`,
`launch_completed_counts=[4,4]`, zero scheduler errors, and resource policy
`scheduler_blocks=1`, `worker_blocks=4`, `block_dim=256`, `grid_dim=5`.

| GPU | Device ns | Host ns | Per-launch device ns | Status |
| --- | --------- | ------- | -------------------- | ------ |
| A100 | 76800 | 106982 | `49152,27648` | pass |
| H200 | 57472 | 75314 | `31712,25760` | pass |

The same explicit graph tensor-core descriptor is now covered by the normal
L2 `SceneTestCase` path. The descriptor-only red test first failed because
`persistent_dag_graph_f32` accepted a graph task with `func_id=10` and an
incompatible `rows=8` WMMA tile. The graph adapter now applies the same
tensor-core compatibility guard as the fixed tensor-core adapter:
`rows` and `cols` in multiples of `16`, and `inner` divisible by `8`.

Focused local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_tensor_core_tile_args or graph_tensor_core_tile_with_ctypes_data' \
  --platform cuda
```

Result: `3 passed, 75 deselected`, covering graph descriptor construction,
the incompatible tensor-core descriptor rejection, and a no-torch real-data
ctypes scene through `Worker` / `ChipWorker`.

The same no-torch real-data scene passed on remote H200 after syncing the
working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda-12.8 \
   PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k graph_tensor_core_tile_with_ctypes_data --platform cuda'
```

Result: `1 passed, 77 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph tensor-tile path is also exposed as the
`pto_persistent_dag_graph_tensor` benchmark baseline. It uses
`dag_shape=graph_tensor_tile`, receives the same `--tensor-rows`,
`--tensor-cols`, and `--tensor-inner` descriptor flags as the scalar tensor
DAG, and records both graph dependency metadata and tensor descriptor metadata
in the benchmark JSON and Markdown report. A one-repeat A100/H200 sample at
`N=512`, `16x16x16` is under
`tmp/cuda-backend/combined-graph-tensor-current-working/`. The capture
validated source-paper metadata, sanitized command examples, generated report
files, zero scheduler errors, and expected dispatch `[3,1,2,1]`. The sample
device times were `51200 ns` on A100 and `38080 ns` on H200.

The explicit graph tensor-core path is now exposed as the
`pto_persistent_dag_graph_tensor_core` benchmark and tensor-shape sweep
baseline. It uses `dag_shape=graph_tensor_core_tile`, preserves the same
graph descriptor metadata as the smoke path, and records the WMMA first task
as dispatch `[10,1,2,1]`.

The focused benchmark tests first failed because the tensor-shape sweep parser
rejected the new baseline, `run_single_sample` treated it as unknown, and the
publish validators still expected only the fixed tensor-core row. After adding
the baseline to the benchmark, sweep, and validator flows, the focused selector
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_tensor_core or tensor_sweep_validator_compact_preset_keeps_dispatch_commas'
```

Paired A100/H200 evidence used the tensor-shape sweep with one repeat and a
`16x16x16` descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_graph_tensor_core \
    --shapes 16x16x16 --n 256 --repeats 1 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-core-benchmark-working
```

The validated working-tree artifact is:

```text
tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/
```

Validation used:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/graph-tensor-core-benchmark-working/tensor-shape-sweep-debe979d/cuda-tensor-shape-sweep.json \
    --require-artifact a100 --require-artifact h200 \
    --require-baseline pto_persistent_dag_graph_tensor_core \
    --require-size 256 --require-shape 16x16x16 \
    --expected-repeats 1 --expected-result-count 2 \
    --require-dispatch pto_persistent_dag_graph_tensor_core=10,1,2,1 \
    --require-report-files --require-report-throughput \
    --require-command-examples \
    --require-source-papers
```

| GPU | Device ns | Host ns | PTX | Graph fan-in | Status |
| --- | --------- | ------- | --- | ------------ | ------ |
| A100 | 52224 | 73631 | `compute_80` | `0,1,1,2` | pass |
| H200 | 50144 | 64644 | `compute_90` | `0,1,1,2` | pass |

Both rows record graph dependents `[1,2,3,3]`, tensor tile `16x16x16`,
tensor-core metadata `wmma:m16n16k8:tf32->f32`, source-paper metadata, and
zero scheduler errors. The artifact label uses the then-current `HEAD`
because the sweep was captured from an uncommitted working tree.

The persistent scalar-scale scene-test adapter was then added to cover the
single-tensor plus scalar descriptor shape on the persistent-device runtime.
It compiles a generated-dispatch `func_id=11` task body, runs it before the
existing multiply/add fan-in branch, and uses ctypes-backed CPU tensors so the
same selector can run on H200 without `torch`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k scalar_scale --platform cuda

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k scalar_scale --platform cuda'
```

Result: local A100 reported `2 passed, 48 deselected`; remote H200 reported
`2 passed, 48 deselected` after the known PTO-ISA SSH refresh warning. The
full local CUDA scene-test file was also rerun after this adapter and reported
`50 passed`.

The same scalar-scale task body was promoted to the no-torch standalone
persistent DAG smoke so it can be captured without the full scene-test
framework. The smoke uses generated-dispatch `func_id` sequence `[11,2,1]`,
with the first task computing `tmp0 = scalar0 * a`, an independent multiply
branch computing `tmp1 = a * b`, and the final task adding both branches.
Focused local coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'scalar_scale' --platform cuda

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_backend.py::test_cuda_persistent_device_smoke_runs_dispatch_dag_scalar_scale \
    -q --platform cuda
```

Results: `2 passed, 146 deselected` for the shape/paired-runner unit tests,
and `1 passed` for the local A100 real-data CUDA smoke.

The paired A100/H200 smoke was then captured with a tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape scalar_scale --task-count 3 --queue-capacity 2 \
    --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner validated both artifacts with
`runtime=persistent_device`, `mode=dag`, `dag_shape=scalar_scale`,
`completed_count=3`, `dispatch_func_ids=[11,2,1]`, `scalar0=2.0`, zero
scheduler errors, and generated report files. The A100 row reported
`device_wall_ns=40960` and `host_wall_ns=61301`; H200 reported
`device_wall_ns=25856` and `host_wall_ns=34808`.

The same DAG shape was then promoted to the selected benchmark path as
`pto_persistent_dag_scalar_scale`. Focused benchmark/report tests and a local
A100 single-baseline sample were run:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'scalar_scale or worker_and_dag_tables or old_captures_without_scalar_affine or include_all_default_persistent or include_batch_mode' \
    --platform cuda

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_scale \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --label scalar-scale-baseline-a100
```

Results: the focused report tests passed with `6 passed, 144 deselected`; the
local A100 single-baseline sample reported `status=pass`,
`dispatch_func_ids=[11,2,1]`, `scalar0=2.0`, and
`device_wall_ns=47104`.

The same real-data ctypes graph test was run on the remote H200 after syncing
the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k inferred_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 43 deselected`. The remote command also printed the known
PTO-ISA SSH refresh warning before the selected CUDA test passed.

The same real-data graph path was then run without an explicit `temporaries`
map, so `tmp0` and `tmp1` were allocated from task outputs:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k auto_temp_graph_with_ctypes --platform cuda'
```

Result: `1 passed, 45 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph descriptor scalar fields now also resolve scalar argument
names from `TaskArgsBuilder`, not just numeric literals in the descriptor. The
focused TDD selector first failed because `_make_graph_task` called
`float("alpha")` for a graph task `scalar0` field. After adding scalar-name
resolution, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'graph_scalar_scale or scalar_field_names' --platform cuda
```

Result: `2 passed, 64 deselected`. The same real-data graph scalar-scale
ctypes selector passed on remote H200 after syncing the tree:
`1 passed, 65 deselected`, with the known PTO-ISA SSH refresh warning printed
before pytest.

The same scalar-scale graph shape is now covered by the no-torch paired
persistent-smoke workflow as `graph_descriptor_scalar_scale`, so it can be
validated outside `SceneTestCase` while still recording explicit runtime graph
metadata:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_scale --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-scale-working
```

Result:
`tmp/cuda-backend/graph-scalar-scale-working/persistent-graph_descriptor_scalar_scale-repeat2-smoke-15e9038f/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_scalar_scale`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, dispatch `[11,2,1]`,
`graph_descriptor.fanin=[0,0,2]`, `graph_descriptor.dependents=[2,2]`,
`scalar0=2.0`, resource policy `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, `grid_dim=4`, and zero scheduler errors on both GPUs. A100
reported per-launch device times `[33792,19456]`, total
`device_wall_ns=53248`, and `host_wall_ns=82872`. H200 reported per-launch
device times `[39616,20096]`, total `device_wall_ns=59712`, and
`host_wall_ns=84805`.

The remaining fixed scalar variants now have the same no-torch graph
descriptor coverage. The paired persistent-smoke runner captured
`graph_descriptor_scalar_axpy` and `graph_descriptor_scalar_affine` under
`tmp/cuda-backend/graph-scalar-variants-working/`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_axpy --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_affine --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working
```

Both captures validated `runtime=persistent_device`, `mode=dag`,
`repeat_runs=2`, `launch_completed_counts=[3,3]`,
`graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, `grid_dim=4`, and
zero scheduler errors. The AXPY descriptor reported dispatch `[4,2,1]`,
`scalar0=1.5`, A100 `device_wall_ns=62464`, and H200
`device_wall_ns=59072`. The affine descriptor reported dispatch `[5,2,1]`,
`scalar0=1.5`, `scalar1=0.5`, A100 `device_wall_ns=67584`, and H200
`device_wall_ns=59616`.

The tensor-core tile descriptor was then added to the same normal L2
`SceneTestCase` path as `persistent_dag_tensor_core_tile_f32`. Its first task
uses block-wide generated dispatch with `func_id=10`, while the remaining
residual, gate, and fan-in tasks reuse the scalar tensor DAG shape. Focused
local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'tensor_core_tile_args or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `2 passed, 39 deselected`. The full local CUDA scene-test file was
then rerun and reported `41 passed`.

The no-torch ctypes tensor-core scene test was also run on H200 after syncing
the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k "tensor_core_tile_with_ctypes_data" --platform cuda'
```

Result: `1 passed, 40 deselected`. The H200 venv lacks `torch`, so the
torch-backed tensor-core scene test is local-only there; the ctypes version
validates the same L2 `Worker` and `TaskArgsBuilder` path with real CUDA data.
The command also printed the known PTO-ISA SSH refresh warning before passing.

The persistent-device scene-test compiler path now also forwards
`CALLABLE["cuda"]["stream_id"]` into the prepared callable manifest. A focused
local A100 check used `stream_id=1` in the compile/plumbing test and the
no-torch tensor-core ctypes scene test:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'compiles_cuda_persistent_device_callable or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `2 passed, 39 deselected`.

The same non-default-stream tensor-core ctypes scene test was then run on H200
after syncing the working tree. Result: `1 passed, 40 deselected`; the command
printed the known PTO-ISA SSH refresh warning before passing.

The same normal L2 persistent-device scene-test path now also checks
device-side scheduler diagnostics after `Worker.run`. A bad explicit graph
with unsupported `func_id=99` raises
`CUDA persistent DAG scheduler error code=1 task_id=0 count=1` even with
`skip_golden=True`, while a good explicit graph still validates real copied
data. Focused local A100 coverage:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'reports_cuda_persistent_scheduler_errors or graph_with_ctypes or tensor_core_tile_with_ctypes_data' \
    --platform cuda
```

Result: `3 passed, 39 deselected`. The full local CUDA scene-test file was
then rerun and reported `42 passed`.

The diagnostic and good-graph ctypes tests were also run on H200 after
syncing the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -rs -k "reports_cuda_persistent_scheduler_errors or graph_with_ctypes" \
     --platform cuda'
```

Result: `3 passed, 41 deselected`. The command printed the known PTO-ISA SSH
refresh warning before passing.

The explicit graph-descriptor path has now been promoted into the benchmark
scripts as `pto_persistent_dag_graph`, using `dag_shape=graph_descriptor`.
This row uses the generated-dispatch `func_id` sequence `[9,2,1]`, generic
tensor slots `tensor_args[0]=tmp0,tensor_args[1]=tmp3`, scalar slots
`scalar_args[0]=1.5,scalar_args[1]=0.25`, and the explicit runtime graph
metadata `fanin=[0,0,2]` and `dependents=[2,2]`.

Focused local test coverage for the benchmark/report wiring:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q --platform cuda
```

Result: `108 passed`.

The focused single-baseline path was checked on A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph --sizes 4096 \
    --repeats 1 --arch compute_80
```

The same command was run on H200 with `--arch compute_90` and the remote venv
Python. Result:
`tmp/cuda-backend/persistent-graph-baseline-working/` contains `a100.json`,
`h200.json`, `cuda-smoke-report.md`, and `cuda-smoke-report.svg`. The A100
row reported `status=pass`,
`ptx_source=nvcc-persistent-generated-dispatch-compute_80`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=36864`; H200 reported
`status=pass`, `ptx_source=nvcc-persistent-generated-dispatch-compute_90`,
`dispatch_func_ids=[9,2,1]`, and `device_wall_ns=31424`.

The paired persistent-smoke runner also supports `graph_descriptor`, so the
explicit graph path can be captured with the same A100/H200 lifecycle workflow
as the fixed DAG shapes. A repeat-run lifecycle smoke was captured at commit
`5139ba23` with automatic smoke artifact validation enabled:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-5139ba23/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner then ran
`cuda_validate_smoke.py`, which accepted both JSON payloads, required the
`a100` and `h200` artifacts, checked `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, and the generated report files. The A100 row reported
`device_wall_ns=51200` and `host_wall_ns=74021`; H200 reported
`device_wall_ns=51616` and `host_wall_ns=71847`. This validates that the
explicit graph descriptor path can reuse one prepared generated-dispatch
callable across two launches after resetting fan-in, ready flags, counters,
and scratch/output buffers.

The paired persistent-smoke runner now also requires expected generated
dispatch sequences for the existing DAG shapes: `chain`, `fork_join`,
`scratch_reuse`, tensor-tile and tensor-core-tile, scalar AXPY/scale/affine,
triad, quad, unary-square, `generic_args`, and `graph_descriptor`. The focused
unit selector for these paired workflow builders passed with `8 passed, 142
deselected`, after first failing because the validation command omitted those
`--expected-dispatch` checks.

The generic-argument descriptor path was then captured with repeat-run
lifecycle reuse on A100 and H200:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape generic_args --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The paired runner validated
`runtime=persistent_device`, `mode=dag`, `dag_shape=generic_args`,
`repeat_runs=2`, `launch_completed_counts=[3,3]`,
`dispatch_func_ids=[9,2,1]`, zero scheduler errors, and generated report
files. A100 reported per-launch device times `[44032,25600]`, total
`device_wall_ns=69632`, and `host_wall_ns=106096`. H200 reported per-launch
device times `[21088,19808]`, total `device_wall_ns=40896`, and
`host_wall_ns=4953113`. This extends lifecycle evidence beyond the
graph-descriptor repeat-run path to generic indexed tensor/scalar descriptor
slots.

The persistent generic-argument generated-dispatch body now also consumes all
four bounded descriptor slots when they are present. The `generic_args4` smoke
maps `tensor_args[0]=tmp0`, `tensor_args[1]=tmp3`, `tensor_args[2]=a`, and
`tensor_args[3]=b`, with scalar slots `[1.5, 0.25, 0.125, 0.0625]`.
Focused TDD checks first failed because the paired runner omitted the
`generic_args4` dispatch expectation and the scene-test task body ignored
slots 2 and 3. After the fix, local A100 checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'persistent_generic_args_four_slots or \
      persistent_device_generic_args_four_slots or generic_args4_workflow' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape generic_args4 \
    --output-json \
      tmp/cuda-backend/persistent-generic_args4-smoke-working/a100.json
```

The paired A100/H200 repeat-run smoke was then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape generic_args4 --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=generic_args4`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, resource policy `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, `grid_dim=4`, and generated report files. A100 reported
per-launch device times `[33792,23552]`, total `device_wall_ns=57344`, and
`host_wall_ns=81101`. H200 reported per-launch device times `[24192,17888]`,
total `device_wall_ns=42080`, and `host_wall_ns=59414`. The same
`persistent_dag_generic_args_f32` four-slot scene-test path passed on H200
with `1 passed, 59 deselected` after the known PTO-ISA SSH refresh warning.

The explicit graph-descriptor adapter now has the same four-slot generic
argument coverage. The descriptor shape `graph_descriptor_generic_args4`
keeps the graph metadata path (`graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`) while mapping `tensor_args[0]=tmp0`,
`tensor_args[1]=tmp3`, `tensor_args[2]=a`, and `tensor_args[3]=b`, with
scalar slots `[1.5, 0.25, 0.125, 0.0625]`. Focused TDD checks first failed
because the smoke runner did not recognize this DAG shape and the paired
validator had no dispatch expectation. After the fix, local A100 checks
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_generic_args_four_slots or graph_descriptor_generic_args4' \
  --platform cuda

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 \
    --dag-shape graph_descriptor_generic_args4 \
    --output-json \
      tmp/cuda-backend/persistent-graph_descriptor_generic_args4-smoke-working/a100.json
```

The paired A100/H200 repeat-run smoke was then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_generic_args4 --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_generic_args4`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[9,2,1]`, zero scheduler
errors, `graph_descriptor.fanin=[0,0,2]`,
`graph_descriptor.dependents=[2,2]`, resource policy `scheduler_blocks=1`,
`worker_blocks=3`, `block_dim=256`, `grid_dim=4`, and generated report files.
A100 reported per-launch device times `[33792,18432]`, total
`device_wall_ns=52224`, and `host_wall_ns=390302`. H200 reported per-launch
device times `[23936,21344]`, total `device_wall_ns=45280`, and
`host_wall_ns=62740`. The same graph-descriptor four-slot scene-test path
passed on H200 with `2 passed, 60 deselected` after the known PTO-ISA SSH
refresh warning.

The explicit graph-descriptor path also validates fixed tensor-arity task
descriptors for the generated-dispatch triad and quad shapes. Focused TDD
checks first failed because the smoke runner only accepted the fixed `triad`
and `quad` shapes, and the paired validator had no graph-descriptor dispatch
expectations. After adding `graph_descriptor_triad` and
`graph_descriptor_quad`, the focused local checks passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_descriptor_triad or graph_descriptor_quad'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q \
  -k 'graph_descriptor_triad or graph_descriptor_quad' --platform cuda
```

Results: `4 passed, 209 deselected` for the smoke workflow tests, and
`2 passed, 42 deselected` for the real CUDA A100 tests. The paired A100/H200
repeat-run smokes were then captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_triad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_quad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working
```

Result:
`tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_triad-repeat2-smoke-4cd73e6a/`
and
`tmp/cuda-backend/graph-tensor-arity-working/persistent-graph_descriptor_quad-repeat2-smoke-4cd73e6a/`
contain `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validators required `runtime=persistent_device`,
`mode=dag`, `repeat_runs=2`, zero scheduler errors, graph fan-in `[0,0,2]`,
dependents `[2,2]`, `scheduler_blocks=1`, `worker_blocks=3`,
`block_dim=256`, and `grid_dim=4`. Triad validated dispatch `[6,2,1]` with
`tensor_args={"c":"tmp0"}`. Quad validated dispatch `[8,2,1]` with
`tensor_args={"c":"tmp0","d":"tmp3"}`.

| Shape | GPU | Device ns | Host ns | Per-launch device ns |
| ----- | --- | --------- | ------- | -------------------- |
| `graph_descriptor_triad` | A100 | 64512 | 95893 | `[40960,23552]` |
| `graph_descriptor_triad` | H200 | 63776 | 88670 | `[42112,21664]` |
| `graph_descriptor_quad` | A100 | 68608 | 101095 | `[44032,24576]` |
| `graph_descriptor_quad` | H200 | 62496 | 87014 | `[41120,21376]` |

The same shape is now promoted to a benchmark baseline named
`pto_persistent_dag_graph_generic_args4`. Focused TDD checks first failed
because `cuda_benchmark.py`, the paired benchmark runner, and the
paired-current capture validator did not recognize the new row or its expected
dispatch sequence. After the fix, the local benchmark/report tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_generic_args4_dag or current_a100_h200_workflow or \
      configured_capture or paired_current_requires_generic_args_baseline or \
      compact_current_preset or include_persistent_baselines or \
      same_work_batch_modes'
```

A quick A100/H200 single-baseline capture was then run through
`cuda_benchmark.py`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 4096 --repeats 1 --arch compute_80 \
    --single-baseline pto_persistent_dag_graph_generic_args4 \
    --label graph-generic-args4-baseline-a100
```

Result:
`tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/` contains
`a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_generic_args4`,
`dispatch_func_ids=[9,2,1]`, zero scheduler errors, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, and `grid_dim=4`.
A100 reported `device_wall_ns=43008` and `host_wall_ns=58143`; H200 reported
`device_wall_ns=33664` and `host_wall_ns=43163`.

After promoting the graph tagged-inout, graph triad, and graph quad baselines
to the selected benchmark matrix, the paired-current validator then expected
`954` full paired samples or `72` compact paired samples. The focused
benchmark/report TDD selector passed locally after adding the graph-triad and
graph-quad rows:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_triad_dag or graph_quad_dag or \
      paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      current_a100_h200_workflow or \
      validate_command_matches_configured_capture or \
      include_persistent_device_modes or same_work_batch_modes or \
      omits_empty_batch_sweeps'
```

Result: `9 passed, 206 deselected`.

The graph tensor-arity rows were then captured through the compact paired
A100/H200 benchmark gate:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-benchmark-working
```

The paired runner wrote local A100, remote H200, and merged reports under
`tmp/cuda-backend/graph-tensor-arity-benchmark-working/`, then validated the
combined `combined-current-943620bf/cuda-benchmark.json` with the
`compact-current` preset. It required 72 samples, source-paper provenance,
sanitized command examples, generated Markdown/SVG reports, zero scheduler
errors, triad dispatch `[6,2,1]`, quad dispatch `[8,2,1]`, graph fan-in
`[0,0,2]`, and graph dependents `[2,2]`.

| GPU | Base DAG ns | Fixed triad ns | Graph triad ns | Fixed quad ns | Graph quad ns |
| --- | ----------- | -------------- | -------------- | ------------- | ------------- |
| A100 | 47104 | 33792 | 29696 | 39936 | 33792 |
| H200 | 41472 | 35296 | 30880 | 33376 | 28704 |

The combined report directory contains `cuda-benchmark.json`,
`cuda-benchmark.md`, `cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`,
`cuda-benchmark-dag-deltas.svg`, and `cuda-benchmark-throughput.svg`.

The graph scalar-scale DAG is now also part of the selected paired benchmark
matrix as `pto_persistent_dag_graph_scalar_scale`. The row keeps the same
scalar-scale generated-dispatch body as the fixed
`pto_persistent_dag_scalar_scale` row, but represents it through explicit
runtime graph metadata. The paired gate first failed under TDD because the
validator preset, paired benchmark row list, and `run_single_sample(...)`
dispatcher did not know the graph scalar-scale baseline. After wiring the
row, the focused tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'graph_scalar_scale or paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      pair_benchmark_builds_current_a100_h200_workflow or \
      pair_benchmark_validate_command_matches_configured_capture'
```

Result: `5 passed, 258 deselected`.

The local A100 single-row check validated dispatch `[11,2,1]`, graph fan-in
`[0,0,2]`, dependents `[2,2]`, scalar metadata `scalar0=2.0`, and zero device
scheduler errors:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_scale \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-scale-benchmark-working/a100-single
```

The paired A100/H200 compact selected-baseline gate then validated
`80` samples with source-paper provenance, command examples, Markdown/SVG
reports, visible graph topology, tensor throughput rows, zero scheduler
errors, and the new graph scalar-scale topology requirements:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-scale-benchmark-working
```

Artifacts:

- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/a100-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/h200-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/combined-current-993254e8/`
- `tmp/cuda-backend/graph-scalar-scale-benchmark-working/index.md`

| GPU | N | Dispatch | Fan-in | Dependents | Scalar args | Device ns | Status |
| --- | - | -------- | ------ | ---------- | ----------- | --------- | ------ |
| A100 | 1024 | `11,2,1` | `0,0,2` | `2,2` | `scalar0=2.0` | 28672 | pass |
| H200 | 1024 | `11,2,1` | `0,0,2` | `2,2` | `scalar0=2.0` | 27712 | pass |

The generated benchmark Markdown and SVG graph-metadata sections now render a
`Scalar args` column, so the scalar descriptor value is visible outside the
raw JSON artifact.

The graph scalar AXPY and affine DAGs are now selected benchmark rows as
`pto_persistent_dag_graph_scalar_axpy` and
`pto_persistent_dag_graph_scalar_affine`. They use the same generated task
bodies as the fixed scalar DAG rows, but run through explicit graph descriptor
metadata. The first focused TDD run failed because the paired benchmark row
list, capture preset, and `run_single_sample(...)` dispatcher did not include
the two graph scalar rows. After wiring them, the focused regression subset
passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'paired_current_requires_generic_args_baseline or \
      compact_current_preset_matches_docs_gate or \
      pair_benchmark_builds_current_a100_h200_workflow or \
      pair_benchmark_validate_command_matches_configured_capture or \
      omits_empty_batch_sweeps or include_persistent_device_modes or \
      same_work_batch_modes or graph_scalar_axpy_dag or \
      graph_scalar_affine_dag'
```

Result: `9 passed, 257 deselected`.

A100 single-row checks validated the two promoted rows before the paired gate:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_axpy \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-axpy-single

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scalar_affine \
    --sizes 4096 --repeats 1 --arch compute_80 \
    --output-dir tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-affine-single
```

The compact selected-baseline gate then validated `84` A100/H200 samples with
source-paper provenance, command examples, generated Markdown/SVG reports,
visible graph topology, scalar metadata, tensor throughput rows, and zero
scheduler errors:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-benchmark-working
```

Artifacts:

- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/a100-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/h200-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/combined-current-93fc927d/`
- `tmp/cuda-backend/graph-scalar-variants-benchmark-working/index.md`

| GPU | N | Baseline | Dispatch | Fan-in | Dependents | Scalar args | Device ns |
| --- | - | -------- | -------- | ------ | ---------- | ----------- | --------- |
| A100 | 1024 | `pto_persistent_dag_graph_scalar_axpy` | `4,2,1` | `0,0,2` | `2,2` | `scalar0=1.5` | 28672 |
| A100 | 1024 | `pto_persistent_dag_graph_scalar_affine` | `5,2,1` | `0,0,2` | `2,2` | `scalar0=1.5,scalar1=0.5` | 34816 |
| H200 | 1024 | `pto_persistent_dag_graph_scalar_axpy` | `4,2,1` | `0,0,2` | `2,2` | `scalar0=1.5` | 25280 |
| H200 | 1024 | `pto_persistent_dag_graph_scalar_affine` | `5,2,1` | `0,0,2` | `2,2` | `scalar0=1.5,scalar1=0.5` | 25600 |

The selected paired benchmark gate now includes
`pto_persistent_dag_graph_tensor_core`. The paired runner was first updated
under TDD because it still omitted the row from its selected baseline list
even though the benchmark and validator accepted it. The focused tests passed:

The paired-current validator now also rejects stale tensor reports with
`--require-report-tensor-throughput`. That gate checks that
`cuda-benchmark.md` contains the `Tensor Throughput Rows` table and that
`cuda-benchmark-throughput.svg` visibly includes each required tensor/core and
cuBLAS baseline with the requested tensor descriptor shape.
The gate was validated on the current compact paired A100/H200 capture at
artifact label `a9d028de`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-throughput-gate-current-working
```

The paired runner wrote and validated
`tmp/cuda-backend/tensor-throughput-gate-current-working/combined-current-a9d028de/cuda-benchmark.json`
with the `compact-current` preset. It required `84` samples, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports,
visible tensor throughput rows, zero scheduler errors, graph descriptor
topology, graph task-argument metadata, and selected tensor/cuBLAS baseline
rows. Selected rows:

| GPU | Host ns | Base DAG ns | Graph tensor-core ns | cuBLAS graph ns | Grid batch ns |
| --- | ------- | ----------- | -------------------- | --------------- | ------------- |
| A100 | 19456 | 46080 | 37888 | 11264 | 35840 |
| H200 | 13984 | 39904 | 32288 | 9472 | 28128 |

The graph tensor-core row validates dispatch `10,1,2,1`, fan-in
`0,1,1,2`, dependents `1,2,3,3`, tensor tile `16x16x16`, and zero scheduler
errors on both GPUs. The tensor-throughput table reports A100
`0.86 GF/s` for graph tensor-core and `2.91 GF/s` for cuBLAS Graph; H200
reports `1.01 GF/s` for graph tensor-core and `3.46 GF/s` for cuBLAS Graph.

The previous graph-unary compact paired A100/H200 capture is:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-unary-benchmark-working
```

The paired runner wrote and validated
`tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/cuda-benchmark.json`
with the `compact-current` preset. It required 78 samples, source-paper
provenance, sanitized command examples, generated Markdown/SVG reports, zero
scheduler errors, graph tensor-core metadata, tagged scalar graph metadata,
and the graph unary-square row. The graph unary-square row validates dispatch
`[7,1,1]`, graph fan-in `[0,1,1]`, graph dependents `[1,2]`, and a
three-task explicit graph descriptor for the same one-input square task body
used by the fixed unary DAG row.

| GPU | Fixed unary ns | Graph unary ns | Tagged scalar ns | Graph tensor-core ns |
| --- | -------------- | -------------- | ---------------- | -------------------- |
| A100 | 41984 | 36864 | 34816 | 39936 |
| H200 | 32416 | 31968 | 31552 | 40864 |

All selected PTO rows use target-specific PTX (`compute_80` on A100 and
`compute_90` on H200), report zero scheduler errors, and are visible in the
generated Markdown/SVG compact report beside the tagged scalar and tensor-core
graph rows. The previous tagged-scalar compact gate remains under
`tmp/cuda-backend/tagged-scalar-compact-current-working/`.

Graph-descriptor dependency inference now builds the producer map from the
whole descriptor before inferring omitted `dependents`, so the scene-test graph
adapter no longer requires topological task order. A focused unit test first
failed with `fanin=[0,0,0]` for a reordered graph where the final consumer is
task `0`; after the inference change it passed with `fanin=[2,0,0]`,
`dependents=[0,0]`, and dispatch sequence `[1,9,2]`. The corresponding
no-torch ctypes scene test passed locally on A100 with `2 passed, 50
deselected`, and passed on H200 with `1 passed, 51 deselected` after syncing
the working tree; the H200 command printed the known PTO-ISA SSH refresh
warning before passing.

The reordered graph descriptor was also captured through the paired persistent
smoke runner:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_reordered --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required `runtime=persistent_device`,
`mode=dag`, `dag_shape=graph_descriptor_reordered`, `repeat_runs=2`,
`launch_completed_counts=[3,3]`, `dispatch_func_ids=[1,9,2]`, zero scheduler
errors, and generated report files. A100 reported per-launch device times
`[39936,23552]`, total `device_wall_ns=63488`, and `host_wall_ns=91185`.
H200 reported per-launch device times `[25632,20608]`, total
`device_wall_ns=46240`, and `host_wall_ns=63520`.

The DAG-chain graph descriptor was then captured through the same paired
smoke runner, proving the five-task chain dependency shape as explicit runtime
graph metadata instead of only through the fixed `chain` shape:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_chain --task-count 5 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_chain`, `repeat_runs=2`,
`launch_completed_counts=[5,5]`, `dispatch_func_ids=[1,2,1,2,1]`,
`graph_descriptor.fanin=[0,0,2,1,1]`,
`graph_descriptor.dependents=[2,2,3,4]`, zero scheduler errors, and generated
report files. A100 reported per-launch device times `[41984,27648]`, total
`device_wall_ns=69632`, and `host_wall_ns=94042`. H200 reported per-launch
device times `[31712,25632]`, total `device_wall_ns=57344`, and
`host_wall_ns=74979`.

The diamond graph descriptor was then captured through the same paired smoke
runner:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_diamond --task-count 5 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_diamond`, `repeat_runs=2`,
`launch_completed_counts=[5,5]`, `dispatch_func_ids=[9,2,1,2,1]`,
`graph_descriptor.fanin=[0,0,2,2,2]`,
`graph_descriptor.dependents=[2,3,2,3,4,4]`, zero scheduler errors, and
generated report files. A100 reported per-launch device times
`[49152,31744]`, total `device_wall_ns=80896`, and
`host_wall_ns=111293`. H200 reported per-launch device times
`[24096,23520]`, total `device_wall_ns=47616`, and
`host_wall_ns=4912047`.

The scratch-reuse graph descriptor was then added as
`graph_descriptor_scratch_reuse`, so the explicit runtime graph path now
covers the six-task scratch-reuse DAG shape as descriptor data instead of only
through the fixed `scratch_reuse` shape. Focused TDD checks first failed
because the paired runner rejected the new DAG shape and `_make_dag_shape`
could not build it; after the fix, the focused unit selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py -q \
  -k 'scratch_reuse_graph_descriptor'
```

Result: `2 passed, 178 deselected`.

The paired A100/H200 smoke was captured with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scratch_reuse --task-count 6 \
    --queue-capacity 3 --repeat-runs 2 --sync-remote-tree
```

Result:
`tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required
`runtime=persistent_device`, `mode=dag`,
`dag_shape=graph_descriptor_scratch_reuse`, `repeat_runs=2`,
`launch_completed_counts=[6,6]`, `dispatch_func_ids=[1,2,1,2,1,1]`,
`graph_descriptor.fanin=[0,0,2,1,1,2]`,
`graph_descriptor.dependents=[2,2,3,4,5,5]`, zero scheduler errors, and
generated report files. A100 reported per-launch device times
`[55296,33792]`, total `device_wall_ns=89088`, and `host_wall_ns=121166`.
H200 reported per-launch device times `[36640,29504]`, total
`device_wall_ns=66144`, and `host_wall_ns=84089`.

The same scratch-reuse shape is now covered by the L2 `SceneTestCase`
graph-descriptor adapter with logical-output and storage-output separation.
The graph task uses `out="tmp4"` and `out_storage="tmp0"` for the reuse task,
so tensor-flow inference still sees a unique logical producer while the task
descriptor reuses the original `tmp0` buffer after its last consumer. Focused
TDD first failed because the builder allocated a distinct buffer for `tmp4`;
after adding `out_storage`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'reused_output_storage or graph_scratch_reuse_with_ctypes' \
  --platform cuda
```

Result: `2 passed, 62 deselected`. The same real-data ctypes scene selector
passed on remote H200 after syncing the working tree:
`1 passed, 63 deselected`.

Needed:

- broader CUDA scene-test argument builders beyond the current binary
  elementwise, unary square, scalar scale, axpy, affine, triad, quad,
  host-schedule generic-args, persistent scalar/DAG tracer bullets, and
  explicit graph-descriptor scratch-storage reuse.

### Fourth-Tensor Persistent DAG Verification

After adding the fourth tensor pointer to the persistent DAG descriptor, the
new quad DAG shape was verified with both Python coverage and real CUDA data.
The quad graph uses generated-dispatch `func_id` sequence `[8, 2, 1]`; the
first task computes `a * b + c * d`, and the final task adds an independent
`a * b` branch.

Focused local tests:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_benchmark_report.py \
  -q -m "not requires_hardware"

.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k quad_with_ctypes --platform cuda

.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
  -q -k dispatch_dag_quad --platform cuda
```

Results: `148` non-hardware tests passed, the CUDA ctypes scene test passed,
and the CUDA standalone smoke test passed on the local A100.

The same standalone smoke was run on the remote H200 with a tree sync:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_90 \
    --mode dag --queue-capacity 2 --dag-shape quad \
    --output-json tmp/cuda-backend/persistent-quad-smoke-working/h200.json
```

Result artifacts:

- `tmp/cuda-backend/persistent-quad-smoke-working/a100.json`
- `tmp/cuda-backend/persistent-quad-smoke-working/h200.json`
- `tmp/cuda-backend/persistent-quad-smoke-working/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-quad-smoke-working/cuda-smoke-report.svg`

Both A100 and H200 runs reported zero scheduler errors and tensor arguments
`c=tmp0,d=tmp3`.

### Target Role Cleanup

CUDA now builds native `host`, optional `scheduler`, and `device` target roles
when a runtime build config declares them, and runtime consumers can read
binaries through roles instead of direct hardware slot names. The current
compatibility mapping is:

- Ascend: `host`, `aicpu`, and `aicore` map to their existing artifacts.
- CUDA: `host` maps to `libhost_runtime.so`, `device` maps to
  `libcuda_device_runtime.so`, `scheduler` maps to
  `libcuda_scheduler_runtime.so` when present. CUDA no longer exposes
  `aicpu_path` / `aicore_path` aliases for these role-native artifacts.

The Python `ChipWorker.init(...)` wrapper now resolves runtime binary paths
through `path_for_role(...)` / `role_paths` first. CUDA role-only binary maps
with `host` / `device` or `host` / `scheduler` / `device` initialize through
the Python API and are passed through the C++ nanobind boundary as role maps.
The underlying `ChipWorker` now probes the optional C host-runtime
`simpler_init_roles(...)` entry and passes non-host role binaries directly to
runtimes that export it. The loaded `libhost_runtime.so` represents the
`host` role, so it is not copied back through the role map. Runtimes without
the optional entry still fall back to the legacy two-binary `simpler_init`
ABI.

The scheduler-role build slice was verified with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_runtime_builder.py tests/ut/py/test_chip_worker.py -q \
  -k 'cuda_runtime_binaries or role_only_runtime_binaries or \
      role_keyed_init or role_keyed_paths or scheduler_role'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
from simpler_setup.runtime_builder import RuntimeBuilder
bins = RuntimeBuilder(platform="cuda").get_binaries("persistent_device", build=True)
print(sorted(bins.role_paths))
print(bins.path_for_role("scheduler"))
PY
```

Result: the focused selector reported `7 passed, 75 deselected` with
`--platform cuda`; the runtime builder printed
`['device', 'host', 'scheduler']`, `aicpu_path=None`, `aicore_path=None`, and
the scheduler artifact path under
`build/lib/cuda/onboard/persistent_device/`.

After adding `_ChipWorker.init_roles(...)`, the same focused selector was run
on the synced H200 checkout and reported `5 passed, 38 deselected` with the
known PTO-ISA SSH refresh warning. A real persistent graph SceneTest path was
then checked on both GPUs through the Python `Worker` / `ChipWorker` surface:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k persistent_device_graph_with_ctypes_data --platform cuda
```

Result: local A100 reported `1 passed, 72 deselected`; synced H200 reported
`1 passed, 72 deselected` with the known PTO-ISA SSH refresh warning.

The C host-runtime role-keyed ABI slice was then verified with a fake host
runtime that fails legacy `simpler_init(...)` but succeeds
`simpler_init_roles(...)`, plus a CUDA export check:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_chip_worker.py -q \
  -k prefers_role_keyed_runtime_init

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q -k role_keyed_init --platform cuda
```

Result: both selectors reported `1 passed`; the fake runtime received
`device` and `scheduler` entries and no `host` entry, while the rebuilt CUDA
`persistent_device` host runtime exported `simpler_init_roles`.

The same source tree was synced to `bizhaoh200` and checked with a paired
real-data persistent graph smoke:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor --task-count 3 --queue-capacity 2 \
    --repeat-runs 1 --sync-remote-tree \
    --output-root tmp/cuda-backend/scheduler-role-working
```

Result:
`tmp/cuda-backend/scheduler-role-working/persistent-graph_descriptor-smoke-539a05b9/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. The validator required runtime `persistent_device`,
mode `dag`, `dag_shape=graph_descriptor`, dispatch `[9,2,1]`, graph fan-in
`[0,0,2]`, graph dependents `[2,2]`, one repeat launch, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, `grid_dim=4`, report
files, and zero scheduler errors. A100 reported `device_wall_ns=41984`;
H200 reported `device_wall_ns=56864`. Both hosts also built
`libcuda_scheduler_runtime.so` beside `libhost_runtime.so` and
`libcuda_device_runtime.so`.

Needed:

- remove or hide the legacy positional `_ChipWorker.init(...)` path once all
  host runtimes export `simpler_init_roles(...)`; until then it remains the
  compatibility path for Ascend and old runtime-binary objects without
  `role_paths`.

### Persistent Scheduler Generalization

The persistent-device scheduler is proven for small generated descriptors, but
it is not yet a full TensorMap/ringbuffer analogue.

The selected benchmark gate now includes the reordered explicit graph
descriptor as `pto_persistent_dag_graph_reordered`, not only the paired smoke
artifact. The compact no-batch A100/H200 capture under
`tmp/cuda-backend/graph-reordered-benchmark-working/combined-current-e038c96a/`
validated `86` rows, source-paper provenance, generated reports, zero device
scheduler errors, dispatch `1,9,2`, graph fan-in `2,0,0`, and dependents
`0,0`. At `N=1024`, A100 reported `35840 ns` device time and H200 reported
`25856 ns`, so order-independent graph dependency inference is covered by the
selected benchmark matrix as well as by the repeat-run smoke.

The explicit graph-descriptor adapter now accepts node-link style graph nodes
where list-shaped `graph.nodes` entries carry identity in `id` and task
payload under `data`. The adapter flattens `data` before callable, `op`, and
node-IO lowering, so this shape uses the same CUDA task descriptor ABI as the
existing top-level node field form. The focused TDD selector first failed with
`KeyError: 'func_id'` because `data` stayed nested; after adding early
node-data normalization, local A100 passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'node_data or node_id or node_op or callable_id' \
    --platform cuda
```

That run reported `8 passed, 125 deselected`. The synced H200 focused selector
also passed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k node_data --platform cuda
```

That run reported `2 passed, 131 deselected` after the known PTO-ISA SSH
refresh warning.

The same node-link compatibility path now accepts `graph.links` as an alias
for top-level edge lists. A focused TDD selector first failed with graph fan-in
`[0,0,0]` because the links were ignored; after routing `links` through the
same dependency lowering as `edges`, the local A100 selector passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'node_link or node_data or graph_edges or adjacency_edges' \
    --platform cuda
```

That run reported `11 passed, 124 deselected`. The synced H200 focused
selector also passed:

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k node_link --platform cuda
```

That run reported `2 passed, 133 deselected` after the known PTO-ISA SSH
refresh warning.

The node-link descriptor shape is now also available in the paired
persistent-smoke workflow as `graph_descriptor_node_link`, so the `links`
schema can produce A100/H200 JSON plus generated Markdown/SVG evidence. The
new TDD selector first failed because the smoke builder and paired runner did
not know the shape; after wiring the generated add/mul/add descriptor,
metadata expectations, and CLI choices, the focused local report tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'node_link_graph_descriptor'
```

That run reported `2 passed, 284 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_node_link --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-node-link-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-node-link-smoke-working/persistent-graph_descriptor_node_link-repeat2-smoke-3e4ddb00/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, A100 device time `45056 ns`,
and H200 device time `43808 ns`.

The dictionary-valued node-port graph descriptor spelling is now promoted into
the paired persistent-smoke workflow as `graph_descriptor_node_port_dict`.
The new TDD selector first failed because both smoke runners rejected the
unknown DAG shape; after adding the direct-smoke task descriptor, paired
runner expectations, graph-node op metadata, and report-visible port-task-arg
metadata, the focused local tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k node_port_dict --platform cuda
```

That run reported `2 passed, 337 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_node_port_dict --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-node-port-dict-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-node-port-dict-smoke-working/persistent-graph_descriptor_node_port_dict-repeat2-smoke-b336f9ff/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`node_port_dict`, and graph task args
`task0=input.lhs:a,input.rhs:b,output.value:tmp0;`
`task1=input.lhs:a,input.rhs:b,output.value:tmp1;`
`task2=input.lhs:tmp0,input.rhs:tmp1,output.value:out`. Device times were
`61440 ns` on A100 and `41408 ns` on H200.

The dictionary-keyed graph task descriptor spelling is now promoted into the
paired persistent-smoke workflow as `graph_descriptor_task_dict`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
add/mul/add descriptor, paired-runner expectations, graph fan-in/dependent
metadata, and report-visible task-dictionary metadata, the focused local tests
passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_task_dict or task_dict_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 343 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_task_dict --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-task-dict-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-task-dict-smoke-working/persistent-graph_descriptor_task_dict-repeat2-smoke-6566536a/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents
`[2,2]`, graph task arg key `task_dict`, and graph task args
`join=input:a,input:b,output:out;left=input:a,input:b,output:tmp0;`
`right=input:a,input:b,output:tmp1`. Device times were `67584 ns` on A100 and
`43456 ns` on H200.

The submit-shaped graph descriptor spelling is now also available in the
paired persistent-smoke workflow as `graph_descriptor_submits`. The TDD
selector first failed because the direct smoke rejected the unknown DAG shape;
after wiring the generated add/add/add inout descriptor, paired-runner
expectations, graph fan-in/dependent metadata, and report-visible submit
task-arg metadata, the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_submits or submits_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 339 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_submits --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-submits-smoke-working
```

The capture under
`tmp/cuda-backend/persistent-submits-smoke-working/persistent-graph_descriptor_submits-repeat2-smoke-c6130969/`
contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,1,1]`, graph fan-in `[0,1,1]`, graph
dependents `[1,2]`, graph task arg key `submits`, and graph task args
`task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;`
`task2=input:tmp1,input:a,output_existing:out`. Device times were
`74752 ns` on A100 and `48288 ns` on H200.

The submit-group graph descriptor spelling is now available in the paired
persistent-smoke workflow as `graph_descriptor_submit_groups`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
parallel add/add/join descriptor, paired-runner expectations, graph
fan-in/dependent metadata, and report-visible submit-group task-arg metadata,
the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_submit_groups or submit_groups_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 341 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_submit_groups --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-submit-groups-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-submit-groups-smoke-working/persistent-graph_descriptor_submit_groups-repeat2-smoke-dccd3bd9/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,1,1]`, graph fan-in `[0,0,2]`, graph
dependents `[2,2]`, graph task arg key `submit_groups`, and graph task args
`task0=input:a,input:b,output:tmp1;task1=input:a,input:b,output:tmp2;`
`task2=input:tmp1,input:tmp2,output_existing:out`. Device times were
`64512 ns` on A100 and `43616 ns` on H200.

The named graph-callable descriptor spelling is now available in the paired
persistent-smoke workflow as `graph_descriptor_named_callable`. The TDD
selector first failed because `run_persistent_smoke` rejected the unknown DAG
shape and the paired runner rejected the CLI choice. After wiring the
add/mul/add descriptor, paired-runner expectations, graph fan-in/dependent
metadata, report-visible graph-node ops, and callable-name task-arg metadata,
the focused local tests passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_benchmark_report.py \
    tests/ut/py/test_cuda_backend.py \
    -q -k 'graph_descriptor_named_callable or named_callable_graph_descriptor' \
    --platform cuda
```

That run reported `2 passed, 345 deselected`. The paired smoke then validated
local A100 and remote H200 artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_named_callable --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/persistent-named-callable-smoke-working
```

The capture is under:

```text
tmp/cuda-backend/persistent-named-callable-smoke-working/persistent-graph_descriptor_named_callable-repeat2-smoke-4b785a91/
```

It contains `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg`. It validated repeat completions `[3,3]`, zero
scheduler errors, dispatch `[1,2,1]`, graph fan-in `[0,0,2]`, graph dependents
`[2,2]`, graph-node ops
`task0=op:add=1;task1=op:mul=2;task2=op:add=1`, graph task arg key
`named_callable`, and graph task args
`task0=callable:add,input:a,input:b,output:tmp0;`
`task1=callable:mul,input:a,input:b,output:tmp1;`
`task2=callable:add,input:a,input:b,output:out`. Device times were
`66560 ns` on A100 and `42784 ns` on H200.

The same named-callable graph descriptor is now part of the selected paired
benchmark matrix as `pto_persistent_dag_graph_named_callable`. The compact
paired benchmark gate uses `--batch-tasks 0` as the no-batch sentinel, so the
paired runner and `compact-current` validator now normalize that value away
and require `96` A100/H200 samples: `48` selected non-batch rows on each
machine. The focused TDD selector first failed because the compact validator
still expected batch rows and `104` samples for this no-batch command. After
normalizing zero batch sweeps in `cuda_pair_benchmark.py` and splitting the
compact baseline list from the full paired-current list in
`cuda_validate_capture.py`, the selector passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
  -q -k 'compact_current_preset_matches_docs_gate or \
omits_empty_batch_sweeps or treats_zero_batch_sweeps_as_empty or \
validate_command_matches_configured_capture' --platform cuda
```

That run reported `4 passed, 292 deselected`. The compact A100/H200 capture
under:

```text
tmp/cuda-backend/persistent-named-callable-baseline-working/combined-current-95be2b5b/
```

passes:

```bash
PYTHONPATH=$PWD:$PWD/python \
  ROOT=tmp/cuda-backend/persistent-named-callable-baseline-working \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    "$ROOT"/combined-current-95be2b5b/cuda-benchmark.json \
    --preset compact-current
```

It validates source-paper provenance, sanitized command examples, graph
topology/task-argument reports, tensor-throughput reports, and zero scheduler
errors. The named-callable benchmark rows were:

- A100: `pto_persistent_dag_graph_named_callable`, `n=1024`,
  dispatch `1,2,1`, fan-in `0,0,2`, dependents `2,2`, task arg key
  `named_callable`, `33792 ns`.
- H200: `pto_persistent_dag_graph_named_callable`, `n=1024`,
  dispatch `1,2,1`, fan-in `0,0,2`, dependents `2,2`, task arg key
  `named_callable`, `25728 ns`.

The refreshed full paired-current capture under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validates the same selected row across
`N=1024,65536,1048576` with zero scheduler errors. Median named-callable
device times were `26624/135168/2361344 ns` on A100 and
`25344/132320/1907936 ns` on H200 for those sizes.

Needed:

- full graph construction from normal PTO task graphs;
- broader graph-lowering coverage beyond the current
  `persistent_dag_graph_f32` descriptor adapter, which already covers
  automatic default temporary allocation, logical-output/storage-output
  separation for scratch reuse, order-independent tensor-flow dependency
  inference, explicit outgoing and incoming graph edges with scalar or
  list-valued named task dependencies, top-level graph edge lists including
  string `source -> target` entries and dep-gen-style `pred`/`succ` endpoint
  dictionaries including annotated real `deps.json` rows with `source`
  metadata, dep-gen-style `task_id` graph task identities, graph task defaults
  for runnable metadata shared by dep-gen task rows, external JSON graph files
  with inline overlays, keyed task overrides for heterogeneous imported graph
  rows, list-shaped task metadata sidecars, adjacency dictionaries,
  `graph.links` aliases, `graph.nodes` aliases, node `id` identity aliases,
  node-link `data` payloads, node-style IO fields, dictionary-valued node IO
  port maps, node `op` callable aliases, callable metadata `callable_id` /
  `cid` aliases, and paired smoke including node-link `links` and
  dictionary-valued node IO port maps,
  tagged TaskArgs-like graph task lowering including `inout` producer
  chaining, role-map task-argument dictionaries with paired smoke,
  submit-shaped graph descriptors, submit-group descriptor expansion in the
  selected benchmark matrix, explicit unary square graph dispatch, tagged
  graph-descriptor paired smoke, and five-task chain, five-task fan-out/fan-in,
  and six-task scratch-reuse graph descriptor smokes;
- broader lifecycle validation beyond the current scratch-reuse,
  graph-descriptor and generic-argument repeat-run, tensor-core graph, and
  direct/queue/DAG prepared-callable repeat-run smokes. The paired lifecycle
  matrix runner now captures direct, queue, DAG-chain, incoming-edge graph,
  graph-descriptor scratch-reuse, and graph tensor-core repeat-run evidence
  across A100 and H200 in one artifact set
  (`tmp/cuda-backend/lifecycle-tensor-core-working/persistent-lifecycle-matrix-1c683c1c/`).
  The lifecycle matrix validator checks required scenarios, A100/H200
  artifacts, repeat-run completion counts, DAG-chain dispatch, graph
  depends-on dispatch/topology, graph-scratch-reuse dispatch/topology,
  graph tensor-core dispatch/topology, scratch-reuse metadata, tensor-tile
  metadata, report files, VDCores/MPK source-paper provenance, collection
  mode, sanitized reconstruction commands, and zero device scheduler errors.
  The matrix report can be regenerated from the existing per-scenario smoke
  JSON with `--collect-existing-suffix 1c683c1c` without rerunning the GPUs,
  and the validator requires that flag in the regenerated local command
  example. The artifact index reads the lifecycle metadata commit and
  collection mode so regenerated reports remain distinguishable in
  `tmp/cuda-backend/*/index.md`.
  The paired persistent-smoke validator now
  requires `scratch_reuse=reused_buffer=tmp0,reuse_task=4`, and the smoke
  Markdown/SVG report renders that physical alias beside task metadata. In
  the current capture, graph-depends-on validates
  `launch_completed_counts=[3,3]`, dispatch `1,2,1`, graph fan-in `0,0,2`,
  dependents `2,2`, and device times of `63488 ns` on A100 and `41600 ns` on
  H200. The same capture's graph-scratch-reuse scenario validates
  `launch_completed_counts=[6,6]`, dispatch `1,2,1,2,1,1`, graph fan-in
  `0,0,2,1,1,2`, dependents `2,2,3,4,5,5`, scratch reuse of `tmp0` at task
  `4`, and device times of `92160 ns` on A100 and `89056 ns` on H200. The
  graph-tensor-core scenario validates `launch_completed_counts=[4,4]`,
  dispatch `10,1,2,1`, graph fan-in `0,1,1,2`, dependents `1,2,3,3`,
  tensor tile `16x16x16`, `worker_blocks=4`, and device times of `77824 ns`
  on A100 and `56672 ns` on H200, so the remaining lifecycle gap is normal
  PTO graph breadth rather than prepared-callable reset coverage;
- broader resource policy across configurable scheduler blocks, configurable
  queue/DAG worker blocks, direct worker-blocks-per-task, callable stream id
  tracer bullet, and configurable block dimension. The
  paired A100/H200 resource-policy smoke under
  `tmp/cuda-backend/persistent-block128-working/` validates a five-task
  DAG-chain repeat run with `scheduler_blocks=1`, `worker_blocks=2`,
  `worker_blocks_per_task=1`, `stream_id=1`, `block_dim=128`, `grid_dim=3`,
  `repeat_runs=2`, dispatch `1,2,1,2,1`, generated Markdown/SVG reports, and
  zero scheduler errors. The broader resource-policy diamond capture under
  `tmp/cuda-backend/resource-policy-diamond-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-4862b62c/` validates a
  five-task graph descriptor with `worker_blocks=4`, `stream_id=2`,
  `block_dim=512`, `grid_dim=5`, repeat completions `[5,5]`, dispatch
  `9,2,1,2,1`, graph fan-in `0,0,2,2,2`, dependents `2,3,2,3,4,4`,
  scalar/tensor arg metadata, generated Markdown/SVG reports, and zero device
  scheduler errors on A100 and H200. A100 reported `device_wall_ns=72704`;
  H200 reported `device_wall_ns=53728`. The scheduler-distribution diamond
  capture under `tmp/cuda-backend/scheduler-distribution-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-93e0a299/` validates the
  same graph shape with `scheduler_blocks=2`, `scheduler_init_count=2`,
  `worker_blocks=3`, `stream_id=2`, `block_dim=256`, `grid_dim=5`, repeat
  completions `[5,5]`, dispatch `9,2,1,2,1`, graph fan-in `0,0,2,2,2`,
  dependents `2,3,2,3,4,4`, scalar/tensor arg metadata, generated
  Markdown/SVG reports, and zero device scheduler errors on A100 and H200.
  A100 reported `device_wall_ns=77824`; H200 reported
  `device_wall_ns=52768`. The scheduler-loop policy capture under
  `tmp/cuda-backend/scheduler-loop-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-5d7b3961/` moves
  dependent release out of worker blocks and into scheduler blocks through a
  bounded completion ring. It validates the same graph shape with
  `scheduler_loop_count=2`, `scheduler_processed_count=5`,
  `scheduler_init_count=2`, repeat completions `[5,5]`, generated
  Markdown/SVG reports, and zero device scheduler errors on A100 and H200.
  A100 reported `device_wall_ns=97280`; H200 reported
  `device_wall_ns=72928`. The scheduler by-block policy capture under
  `tmp/cuda-backend/scheduler-by-block-policy-working/`
  `persistent-graph_descriptor_diamond-repeat2-smoke-01b85c21/` keeps the
  same graph and resource policy, exposes per-scheduler completion counters,
  and validates `scheduler_processed_by_block=[2,3]` on both A100 and H200.
  A100 reported `device_wall_ns=97280`; H200 reported
  `device_wall_ns=71136`. The scheduler scaling sweep under
  `tmp/cuda-backend/scheduler-scaling-working/` validates the same graph over
  `scheduler_blocks=1,2,4` with paired A100/H200 smoke JSON and a summary
  report under `scheduler-scaling-a5ca4fac/`. A100 reported
  `110592/97280/98304 ns` for `1/2/4` scheduler blocks; H200 reported
  `82240/70368/70752 ns`. The four-scheduler rows expose the current small
  graph's load-balance limit directly: A100 processed completions as
  `[0,2,3,0]` with active schedulers `2/4` and a `60.0%` busiest-scheduler
  completion share, while H200 processed `[2,1,1,1]` with active schedulers
  `4/4` and a `40.0%` busiest share. The parallel-chains capture under
  `tmp/cuda-backend/parallel-chains-working/` adds a nine-task graph with four
  roots, two joins, two parallel consumers, and one final join over
  `scheduler_blocks=4`, `worker_blocks=4`, and `queue_capacity=4`. It validates
  dispatch `1,2,1,2,1,1,2,1,1`, graph fan-in `0,0,0,0,2,2,2,2,2`, graph
  dependents `4,4,5,5,6,7,6,7,8,8`, repeat completions `[9,9]`, and zero
  scheduler errors on A100 and H200. A100 processed completions as
  `[2,1,3,3]`; H200 processed `[3,3,2,1]`. The parallel-chains scheduler
  scaling sweep under
  `tmp/cuda-backend/parallel-chains-scheduler-scaling-working/` validates the
  same graph over `scheduler_blocks=1,2,4` and writes a shape-aware summary
  under `scheduler-scaling-674ebe2e/`. A100 reported
  `155648/123904/115712 ns`; H200 reported `131104/102496/90272 ns`.
  The four-scheduler rows compare at `0.74x` and `0.69x` versus each GPU's
  matching one-scheduler row and keep all four scheduler blocks active on
  both GPUs. The current-head graph-size summary under
  `tmp/cuda-backend/scheduler-graph-size-scaling-working/`
  `scheduler-graph-size-scaling-952bdefd/` regenerates one JSON/Markdown/SVG
  report over both the five-task diamond and nine-task parallel-chain sweeps.
  The report includes task count, device ns/task, and tasks/scheduler columns;
  the four-scheduler per-task rows are `19660 ns` vs. `12856 ns` on A100 and
  `14150 ns` vs. `10030 ns` on H200 for diamond vs. parallel chains. The
  parallel-chains descriptor is now also wired into the selected benchmark
  path as `pto_persistent_dag_graph_parallel_chains`. The full paired
  current-head gate under
  `tmp/cuda-backend/current-head-full-parallel-chains-4e81fbff-working/`
  `combined-current-4e81fbff/` validates 1296 A100/H200 rows, dispatch
  `1,2,1,2,1,1,2,1,1`, queue capacity `9`, fan-in/dependent metadata,
  generated-dispatch PTX, Markdown/SVG reports, tensor-throughput reports,
  source-paper provenance, sanitized command examples, and zero scheduler
  errors. The compact paired gate under
  `tmp/cuda-backend/parallel-chains-compact-current-working/`
  `combined-current-c3274430/` remains the 102-row quick path. The remaining
  policy gap is broader graph families rather than launch resource
  partitioning, root seeding, completion-ring ownership, graph-size
  reporting, or artifact validation;
- broader scheduler error taxonomy beyond the current unsupported-`func_id`
  invalid-dependent-ID, dependent-range, fan-in-underflow,
  duplicate-dependent, self-dependent, initial-fan-in, and
  no-root/unreachable-task
  diagnostics. The current validators, smoke reports, lifecycle matrix
  reports, benchmark validators, and local artifact index render known
  nonzero scheduler codes with stable labels such as `7(unreachable_task)`,
  so negative A100/H200 captures are easier to triage without cross-reading
  raw runtime constants.

The role-map graph descriptor row is now part of the selected benchmark
matrix, not only the paired smoke path. The focused TDD first failed because
`cuda_benchmark.py` rejected
`pto_persistent_dag_graph_role_map_inout`, and the paired/compact validators
did not require that row or its graph metadata. After adding the benchmark
row, the focused selector reported `7 passed, 292 deselected`. The full
paired-current gate under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validated `1278` samples. At `N=1024`, role
spelling medians were tag/role/compact/pair/role-map
`30720/30720/30720/29696/30720 ns` on A100 and
`29056/28576/27712/27936/29056 ns` on H200. The compact paired A100/H200
gate under
`tmp/cuda-backend/role-map-selected-benchmark-working/`
`combined-current-a3c09113/` validated `98` non-batch samples with source
paper provenance, sanitized reconstruction commands, Markdown/SVG reports,
and zero scheduler errors. The role-map row validated dispatch `1,1,1`, graph
fan-in `0,1,1`, dependents `1,2`, and `graph_task_arg_key=role_map`; A100
reported `device_wall_ns=29696`, and H200 reported `device_wall_ns=25440`.

The submit-groups graph descriptor row is now also part of the selected
benchmark matrix as `pto_persistent_dag_graph_submit_groups`. The TDD red
selector first failed because `cuda_benchmark.py` rejected that baseline and
the paired/compact validators still expected `1260` full samples and `98`
compact non-batch samples. After adding the selected row and validator
metadata, the full benchmark-report test file passed with `300` tests. The
full paired A100/H200 gate under
`tmp/cuda-backend/current-head-full-submit-groups-working/`
`combined-current-c183d1ad/` validated `1278` samples with source-paper
provenance, sanitized reconstruction commands, Markdown/SVG reports, and zero
scheduler errors. The submit-groups row validated dispatch `1,1,1`,
graph fan-in `0,0,2`, dependents `2,2`, and
`graph_task_arg_key=submit_groups`; A100 reported
`device_wall_ns=25600`, and H200 reported `device_wall_ns=24160`. The compact
paired A100/H200 gate under
`tmp/cuda-backend/submit-groups-selected-benchmark-working/`
`combined-current-193ccc4d/` remains useful as the focused selector gate and
validated `100` non-batch samples before this full refresh.

### Tuned Tensor Workloads

The tensor DAG row validates descriptor metadata and generated dispatch, but
the GEMM body is a scalar microbenchmark rather than a tuned tensor-core
kernel. The first paired tensor-shape sweep now covers `8x4x12`,
`16x16x64`, and `32x16x64` descriptors on A100 and H200. The first
`tensor_core_tile` smoke and selected-baseline benchmark row now validate a
WMMA generated-dispatch task body on both GPUs for one-fragment and
multi-fragment descriptors. The benchmark report includes a signed
DAG-increment table and SVG, so scheduler-vs-task work separation exists for
current microbenchmarks. The remaining gap is tuned tensor execution and
comparative throughput at model-relevant sizes, not descriptor-shape or first
tensor-core callable plumbing.

Needed:

- tensor-core or library-backed callable body tuning beyond the current
  small multi-fragment WMMA benchmark row;
- broader model-kernel shape families once the tensor-core/library path
  exists;
- real tuned-kernel throughput rows beyond the current scheduler-adjusted
  microbenchmark deltas.

### CI Coverage

CUDA tests are optional and hardware-dependent. They currently provide strong
local evidence but are not guaranteed in every CI environment.

Needed:

- optional CUDA CI runner coverage if infrastructure becomes available.
