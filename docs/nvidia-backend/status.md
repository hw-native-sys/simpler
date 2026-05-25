# CUDA Backend Status

This page tracks current implementation status against the CUDA backend
design. It distinguishes verified tracer-bullet behavior from remaining
design work so evaluation results are not mistaken for a complete backend.

## Implemented And Verified

### Platform And Runtime Discovery

- `cuda` maps to the `cuda/onboard` platform variant.
- `RuntimeBuilder(platform="cuda")` discovers both CUDA runtimes:
  `host_schedule` and `persistent_device`.
- The current runtime layout still fits the existing host/aicpu/aicore binary
  slots while the CUDA target-role cleanup remains future work.

Evidence:

- `tests/ut/py/test_cuda_backend.py`
- `src/cuda/runtime/host_schedule/build_config.py`
- `src/cuda/runtime/persistent_device/build_config.py`
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
  `CudaVectorAddArgs` from normal `TaskArgsBuilder` CPU tensors, and validates
  real copied-back CUDA output data.

### Persistent-Device Runtime

The CUDA `persistent_device` runtime is implemented as a set of tracer-bullet
execution modes:

- direct descriptor-array persistent executor;
- scheduler/worker bounded ready queue;
- bounded ring wraparound with capacity smaller than task count;
- generated-dispatch DAG with fan-in counters;
- five-task DAG-chain runtime graph descriptor;
- six-task scratch-reuse DAG descriptor;
- tensor-tile DAG descriptor with rows/cols/inner/stride metadata.
- device-side scheduler diagnostics for unsupported generated-dispatch
  `func_id` values.

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
normal L2 `Worker`, builds `persistent_dag_fork_join_f32` and
`persistent_dag_tensor_tile_f32` state objects from normal `TaskArgsBuilder`
CPU tensors, and validates real copied-back CUDA output data.

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
- `pto_persistent_device`;
- `pto_persistent_queue`;
- `pto_persistent_dag`;
- `pto_persistent_dag_chain`;
- `pto_persistent_dag_reuse`;
- `pto_persistent_dag_tensor`;
- same-work batch rows;
- worker-grid batch rows.

The latest paired capture at commit `38ff341e` uses the `8x4x12` tensor
descriptor, sizes `1024,65536,1048576`, three repeats, task counts `2,6,12`,
and worker-grid values `32,64,128,256`. It includes the compiler-backed
host-schedule row on both A100 and H200.

Evidence:

- [evaluation.md](evaluation.md) is the evaluation landing page.
- [evaluation-current.md](evaluation-current.md) summarizes the latest paired
  A100/H200 capture.
- [evaluation-history.md](evaluation-history.md) preserves earlier captures.
- `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py` writes JSON,
  Markdown, and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py` writes
  compact smoke Markdown and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py` automates
  the local A100 run, remote H200 run, artifact copy, merge, and index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py` indexes
  local `tmp/cuda-backend/` artifacts, including tensor-tile shapes.
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

The docs and skill updates were checked with targeted `pre-commit` runs and
`git diff --check` before commit.

The H200 paired benchmark capture was also run after pushing `38ff341e` to
the remote checkout:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git fetch origin design/nvidia-backend \
     >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
   PYTHONPATH=$PWD:$PWD/python \
   .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 2,6,12 \
     --worker-blocks-per-task 32,64,128,256 \
     --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
     --label h200-current-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-current-$(git rev-parse --short HEAD)'
```

Result: `label=h200-current-38ff341e`, `ptx_arch=compute_90`,
`ptx_source=nvcc-compute_90`, with generated compiler and persistent-device
rows included in `tmp/cuda-backend/h200-current-38ff341e/`.

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

## Remaining Gaps

### Kernel Compiler Integration

Host-schedule task-body compilation and persistent-device generated dispatch
now have first `KernelCompiler` entry points. Both paths can consume
`CudaTaskBody` style sources. CUDA prepared-callable artifacts can be staged
through the L2 Python `Worker` registration path. The normal scene-test flow
can compile and run host-schedule CUDA vector-add callable specs and
persistent-device fork/join DAG callable specs end to end.

Needed:

- broader CUDA scene-test argument builders beyond the current
  `vector_add_f32`, `persistent_dag_fork_join_f32`, and
  `persistent_dag_tensor_tile_f32` tracer bullets.

### Target Role Cleanup

CUDA still fits through the legacy host/aicpu/aicore binary shape. The design
calls for target roles such as `host`, `device`, and optional `scheduler` so
CUDA does not have to pretend to own AICPU/AICore artifacts.

Needed:

- a role-keyed binary model;
- Ascend compatibility mapping for `host`, `aicpu`, and `aicore`;
- CUDA mapping for host runtime and device images.

### Persistent Scheduler Generalization

The persistent-device scheduler is proven for small generated descriptors, but
it is not yet a full TensorMap/ringbuffer analogue.

Needed:

- generalized task argument ABI;
- graph construction from normal PTO task graphs;
- lifecycle validation beyond the current scratch-reuse smoke;
- resource policy for scheduler blocks, worker blocks, and stream use;
- broader scheduler error taxonomy beyond the current unsupported-`func_id`
  diagnostic.

### Tuned Tensor Workloads

The tensor DAG row validates descriptor metadata and generated dispatch, but
the GEMM body is a scalar microbenchmark rather than a tuned tensor-core
kernel.

Needed:

- tensor-core or library-backed callable body experiments;
- shape families aligned with real model kernels;
- evaluation rows that distinguish scheduler overhead from compute throughput.

### CI Coverage

CUDA tests are optional and hardware-dependent. They currently provide strong
local evidence but are not guaranteed in every CI environment.

Needed:

- optional CUDA CI runner coverage if infrastructure becomes available.
