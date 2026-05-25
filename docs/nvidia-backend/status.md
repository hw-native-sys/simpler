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
- `tests/ut/py/test_cuda_kernel_compiler.py` covers the CUDA `KernelCompiler`
  entry point for host-schedule task bodies.
- `tests/ut/py/test_cuda_backend.py` runs one host-schedule callable compiled
  by `KernelCompiler` through `prepare_callable` and validates real CUDA output
  data.

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

The persistent DAG path compiles generated CUDA source with `nvcc` and stores
the generated source, PTX, and manifest under
`build/cache/cuda/onboard/persistent_device/callables/`.

Evidence:

- `tests/ut/py/test_cuda_backend.py` runs persistent-device smoke tests with
  real CUDA data when `nvcc` is available.
- `tests/ut/py/test_cuda_persistent_codegen.py` covers generated dispatch,
  tensor descriptor fields, shared task-body wrapper generation, host-schedule
  and persistent-device manifest writing, and cache reuse.
- `simpler_setup/cuda_callable_compiler.py` contains the generated-dispatch
  source renderer, shared task-body wrapper renderer, and offline `nvcc`
  compile helper.

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

The latest paired capture uses the `8x4x12` tensor descriptor, sizes
`1024,65536,1048576`, three repeats, task counts `2,6,12`, and worker-grid
values `32,64,128,256`.

Evidence:

- [evaluation.md](evaluation.md) is the evaluation landing page.
- [evaluation-current.md](evaluation-current.md) summarizes the latest paired
  A100/H200 capture.
- [evaluation-history.md](evaluation-history.md) preserves earlier captures.
- `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py` writes JSON,
  Markdown, and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py` writes
  compact smoke Markdown and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py` indexes
  local `tmp/cuda-backend/` artifacts, including tensor-tile shapes.
- `.agents/skills/cuda-backend-eval/SKILL.md` documents the current paired
  A100/H200 recipe and remote artifact copy step.

## Latest Local Verification

The focused CUDA test set was run from the project-local virtual environment:

```bash
.venv/bin/python -m pytest \
  tests/ut/py/test_cuda_benchmark_report.py \
  tests/ut/py/test_cuda_backend.py \
  tests/ut/py/test_cuda_kernel_compiler.py \
  tests/ut/py/test_cuda_persistent_codegen.py -q
```

Result: `59 passed`.

The docs and skill updates were checked with targeted `pre-commit` runs and
`git diff --check` before commit.

## Remaining Gaps

### Kernel Compiler Integration

Host-schedule task-body compilation has a first `KernelCompiler` entry point.
Persistent-device generated dispatch is still driven by smoke and benchmark
helpers, and the normal scene-test flow does not yet consume CUDA callable
artifacts end to end.

Needed:

- dispatch entry composition for generated persistent-device task wrappers;
- scene-test plumbing from CUDA callable artifacts into `ChipCallable` and
  `prepare_callable`;
- persistent-device callable manifests wired through the normal build/cache
  layout.

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
- error propagation and diagnostics for device-side scheduler failures.

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

- clear skip reporting when CUDA, `nvcc`, or a driver is unavailable;
- optional CUDA CI runner coverage if infrastructure becomes available;
- remote H200 smoke automation that can be invoked without hand-copy steps.
