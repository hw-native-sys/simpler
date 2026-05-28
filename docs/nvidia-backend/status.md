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
  and `device` targets; the legacy `aicpu_path` and `aicore_path` fields are
  compatibility aliases to the CUDA `device` artifact.

Evidence:

- `tests/ut/py/test_cuda_backend.py`
- `tests/ut/py/test_runtime_builder.py`
- `tests/ut/py/test_runtime_compiler.py`
- `src/cuda/runtime/host_schedule/build_config.py`
- `src/cuda/runtime/persistent_device/build_config.py`
- `src/cuda/platform/onboard/device/CMakeLists.txt`
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
- a small non-blocking stream pool using callable `stream_id` metadata.

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
- scalar-argument DAG descriptors for mixed tensor/scalar AXPY-style and
  two-scalar affine task bodies.
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
  `func_id` values, invalid dependent task IDs, and out-of-range dependent
  spans, fan-in underflow, and initial-fan-in mismatch.
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
`persistent_dag_scalar_axpy_f32` and `persistent_dag_scalar_affine_f32` mixed
tensor/scalar descriptors, `persistent_dag_tensor_tile_f32` state objects,
`persistent_dag_triad_f32` third-tensor descriptors,
`persistent_dag_quad_f32` fourth-tensor descriptors,
`persistent_dag_generic_args_f32` generic tensor/scalar argument descriptors,
`persistent_dag_graph_f32` explicit graph descriptors, and
`persistent_dag_unary_square_f32` unary descriptors from normal `TaskArgsBuilder`
CPU tensors, and validates real copied-back CUDA output data. The no-torch
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
three-input `(a, b, c, out, n)` task bodies, and `elementwise_quad_f32` for
four-input `(a, b, c, d, out, n)` task bodies.
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
- `pto_persistent_device`;
- `pto_persistent_queue`;
- `pto_persistent_dag`;
- `pto_persistent_dag_chain`;
- `pto_persistent_dag_reuse`;
- `pto_persistent_dag_scalar_axpy`;
- `pto_persistent_dag_scalar_affine`;
- `pto_persistent_dag_triad`;
- `pto_persistent_dag_quad`;
- `pto_persistent_dag_generic_args`;
- `pto_persistent_dag_graph`;
- `pto_persistent_dag_unary_square`;
- `pto_persistent_dag_tensor`;
- same-work batch rows;
- worker-grid batch rows.

The latest paired capture at commit `61cf96cd` uses the `8x4x12` tensor
descriptor, sizes `1024,65536,1048576`, three repeats, task counts `2,6,12`,
and worker-grid values `32,64,128,256`. It includes the compiler-backed
host-schedule row, unary square host-schedule row, quad host-schedule row, and
`pto_persistent_dag_scalar_axpy`, `pto_persistent_dag_scalar_affine`, and
`pto_persistent_dag_triad` on both A100 and H200. It also includes
`pto_persistent_dag_quad`, validating fourth tensor task descriptor fields,
`pto_persistent_dag_generic_args`, validating indexed generic tensor/scalar
argument slots, and `pto_persistent_dag_unary_square`, validating unary
persistent DAG arguments in the full paired benchmark path. It also includes
`pto_persistent_dag_graph`, validating the explicit runtime graph descriptor
path in the full paired benchmark path.

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
`pto_persistent_dag` scheduler baseline.

The first cuBLAS library-backed tensor baseline adds `cublas_sgemm` to the
same compact selected-baseline report shape. It uses CUDA Runtime API events
around a warm cuBLAS `cublasSgemmStridedBatched` call over the configured
`16x16x16` descriptor. In the paired A100/H200 capture under
`tmp/cuda-backend/combined-cublas-current-343924df/`, the row measured
`48128 ns` device time on A100 and `58623 ns` on H200. The matched
`pto_persistent_dag_tensor_core` rows in that report measured `33792 ns` and
`32960 ns`. This row is a CUDA library launch/compute comparison point, not a
PTO runtime path.

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
  the local A100 run, remote H200 run, artifact copy, merge, and index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py` automates the
  no-torch host-schedule Worker smoke on local A100 and remote H200, then
  renders the compact smoke report and refreshes the artifact index.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py`
  automates no-torch persistent-device DAG smoke captures on local A100 and
  remote H200, including optional remote tree sync, tensor-tile descriptor
  flags, compact report rendering, smoke artifact validation, and
  artifact-index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py`
  automates paired A100/H200 `pto_persistent_dag_tensor` sweeps over
  model-shaped tensor descriptors and writes JSON, Markdown, and SVG
  artifacts under `tmp/cuda-backend/`.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`
  checks paired benchmark captures for expected machines, selected baselines,
  sizes, repeats, sample count, and generated report files before docs are
  refreshed.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py` checks
  paired smoke captures for required A100/H200 artifacts, pass status, zero
  scheduler errors, expected runtime/mode, dispatch IDs, repeat-run lifecycle
  counts, tensor-tile descriptor shape, and generated smoke report files.
- `.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py` indexes
  local `tmp/cuda-backend/` artifacts, including tensor-tile shapes,
  persistent smoke modes, dispatch sequences, scheduler error counters,
  repeat-run counts, and per-launch completion counts.
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

The H200 initial-fan-in mismatch check returned the expected diagnostic:
`persistent dag scheduler error code=5 task_id=0 count=1`.

The H200 no-root check returned the expected diagnostic:
`persistent dag scheduler error code=6 task_id=0 count=1`.

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
initial-fan-in, and no-root scheduler diagnostics, the focused CUDA test set
was rerun locally:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_persistent_codegen.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_scene_test.py \
  tests/ut/py/test_cuda_benchmark_report.py -q
```

Result: `165 passed`.

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
`persistent_device`; the legacy `aicpu_path` and `aicore_path` attributes
aliased the same device artifact.

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

## Remaining Gaps

### Kernel Compiler Integration

Host-schedule task-body compilation and persistent-device generated dispatch
now have first `KernelCompiler` entry points. Both paths can consume
`CudaTaskBody` style sources. CUDA prepared-callable artifacts can be staged
through the L2 Python `Worker` registration path. The normal scene-test flow
can compile and run host-schedule CUDA vector-add, binary elementwise, unary
square, scalar scale, axpy, two-scalar affine, and three-input triad callable
specs and persistent-device
fork/join, chain, reuse, scalar AXPY, scalar affine, and tensor-tile DAG
callable specs, plus third-tensor persistent triad and unary-square callable
specs, end to end.
The fourth-tensor persistent quad callable spec also runs end to end through
ctypes-backed real data. The host-schedule quad callable spec also runs
through both the normal `SceneTestCase` L2 path and the no-torch Worker smoke
path. A generic `persistent_dag_graph_f32` adapter can now lower explicit
runtime graph descriptors with per-task `func_id`, dependency lists, fan-in,
temporary buffers, generic tensor slots, and scalar slots through the same L2
`SceneTestCase` path.

The graph-descriptor adapter was checked with focused local tests:

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k builds_cuda_persistent_graph_args --platform cuda

.venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
  -q -k graph_with_ctypes --platform cuda
```

Result: both tests passed locally on A100. The full CUDA scene-test file was
also rerun locally and reported `38 passed`.

The same real-data ctypes graph test was run on the remote H200 after syncing
the working tree:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
     -q -k graph_with_ctypes --platform cuda'
```

Result: `1 passed, 37 deselected`. The remote command also printed the known
PTO-ISA SSH refresh warning before the selected CUDA test passed.

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

Needed:

- broader CUDA scene-test argument builders beyond the current binary
  elementwise, unary square, scalar scale, axpy, affine, triad, quad, and
  persistent DAG tracer bullets.

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

CUDA now builds native `host` and `device` target roles when a runtime build
config declares `device`, and runtime consumers can read binaries through
roles instead of direct hardware slot names. The current compatibility mapping
is:

- Ascend: `host`, `aicpu`, and `aicore` map to their existing artifacts.
- CUDA: `host` maps to `libhost_runtime.so`, `device` maps to
  `libcuda_device_runtime.so`, and legacy `aicpu_path` / `aicore_path`
  attributes alias the same CUDA device artifact.

The Python `ChipWorker.init(...)` wrapper now resolves runtime binary paths
through `path_for_role(...)` / `role_paths` first. A CUDA role-only binary map
with `host` and `device` can initialize through the Python API while the
underlying C++ nanobind call still receives its compatibility host,
scheduler, and device path arguments.

Needed:

- optional `scheduler` role once `persistent_device` has a separately named
  scheduler/runtime image;
- removal of legacy `aicpu_path` and `aicore_path` attributes after all
  external and C++ binding boundaries accept role-keyed binaries directly.

### Persistent Scheduler Generalization

The persistent-device scheduler is proven for small generated descriptors, but
it is not yet a full TensorMap/ringbuffer analogue.

Needed:

- graph construction from normal PTO task graphs;
- broader graph-lowering coverage beyond the current explicit
  `persistent_dag_graph_f32` descriptor adapter;
- broader lifecycle validation beyond the current scratch-reuse,
  graph-descriptor repeat-run, and direct/queue/DAG prepared-callable
  repeat-run smokes;
- broader resource policy beyond the current single scheduler block,
  configurable queue/DAG worker blocks, direct worker-blocks-per-task, and
  callable stream id tracer bullet;
- broader scheduler error taxonomy beyond the current unsupported-`func_id`
  invalid-dependent-ID, dependent-range, fan-in-underflow, initial-fan-in, and
  no-root diagnostics.

### Tuned Tensor Workloads

The tensor DAG row validates descriptor metadata and generated dispatch, but
the GEMM body is a scalar microbenchmark rather than a tuned tensor-core
kernel. The first paired tensor-shape sweep now covers `8x4x12`,
`16x16x64`, and `32x16x64` descriptors on A100 and H200. The first
`tensor_core_tile` smoke and selected-baseline benchmark row also validate a
block-wide WMMA generated-dispatch task body on both GPUs. The benchmark
report now includes a signed DAG-increment table and SVG, so scheduler-vs-task
work separation exists for current microbenchmarks. The remaining gap is
tuned tensor execution and comparative throughput, not descriptor-shape or
first tensor-core callable plumbing.

Needed:

- tensor-core or library-backed callable body tuning beyond the current
  single-tile WMMA benchmark row;
- broader model-kernel shape families once the tensor-core/library path
  exists;
- real tuned-kernel throughput rows beyond the current scheduler-adjusted
  microbenchmark deltas.

### CI Coverage

CUDA tests are optional and hardware-dependent. They currently provide strong
local evidence but are not guaranteed in every CI environment.

Needed:

- optional CUDA CI runner coverage if infrastructure becomes available.
