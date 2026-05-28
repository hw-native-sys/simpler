---
name: cuda-backend-eval
description: Use when implementing, testing, or evaluating the PTO CUDA backend on local A100 GPUs or the remote bizhaoh200 H200 host, including source-paper setup, CUDA smoke tests, and benchmark-result capture.
---

# CUDA Backend Evaluation

## Scope

Use this skill for PTO CUDA backend work under `src/cuda/`, CUDA tests under
`tests/ut/py/test_cuda_backend.py`, local A100 smoke tests, remote H200 smoke
tests through `ssh bizhaoh200`, and paper-aligned evaluation setup.

## Source Discipline

- Keep downloaded paper PDFs and extracted text under `tmp/sources/`.
- Keep source notes under `tmp/` so they are inspectable but not committed.
- Current source files:
  - `tmp/sources/arxiv-2605.03190-vdcores.pdf`
  - `tmp/sources/arxiv-2605.03190-vdcores.txt`
  - `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.pdf`
  - `tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt`

## Local Smoke Test

Run tests through the venv Python module, not a bare `pytest`; this machine may
resolve bare `pytest` to a user-level executable outside `.venv`.

```bash
.venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py -q
```

The current smoke test builds `cuda/host_schedule`, compiles a PTX vector-add
kernel with `nvcc`, allocates real CUDA device buffers, copies real data, runs
the kernel through the runtime C API, and validates copied-back results.

For remote machines without `pytest`, run the standalone smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py
```

Run the host-schedule smoke through the normal L2 Python `Worker` surface
instead of the raw C API:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 --no-build
```

Use `--op mul` on the Worker smoke to validate a non-addition task body
through the same `(a, b, out, n)` host-schedule ABI without requiring `torch`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op mul --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 --no-build \
    --output-json tmp/cuda-backend/worker-mul-smoke/a100.json
```

Use `--op scale` to validate the scalar host-schedule ABI
`(a, out, alpha, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op scale --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-scale-smoke/a100.json
```

Use `--op square` to validate the unary host-schedule ABI `(a, out, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op square --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-square-smoke/a100.json
```

Use `--op axpy` to validate the mixed tensor/scalar host-schedule ABI
`(a, b, out, alpha, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op axpy --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-axpy-smoke/a100.json
```

Use `--op affine` to validate the two-scalar host-schedule ABI
`(a, b, out, alpha, beta, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op affine --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-affine-smoke/a100.json
```

Use `--op triad` to validate the three-input host-schedule ABI
`(a, b, c, out, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op triad --device 0 --n 1024 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-triad-smoke/a100.json
```

Use `--op quad` to validate the four-input host-schedule ABI
`(a, b, c, d, out, n)`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op quad --device 0 --n 65536 --block-dim 256 \
    --arch compute_80 \
    --output-json tmp/cuda-backend/worker-quad-smoke/a100.json
```

Use `cuda_smoke_report.py` to turn captured smoke JSON from A100 and H200 into
Markdown and SVG evidence. Persistent-device reports include dispatch
`func_id` sequences, device-side scheduler error counters, resource policy,
task argument metadata, and repeat-run lifecycle counters when present:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py \
    tmp/cuda-backend/worker-mul-smoke/a100.json \
    tmp/cuda-backend/worker-mul-smoke/h200.json \
    --label worker-mul-smoke \
    --output-dir tmp/cuda-backend/worker-mul-smoke
```

Use `cuda_pair_smoke.py` when the same no-torch Worker smoke should be
captured on local A100 and remote H200 in one command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op mul --sync-remote-tree
```

This writes `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg` under
`tmp/cuda-backend/worker-<op>-smoke-<commit>/`, then refreshes
`tmp/cuda-backend/index.md`. Add `--build-runtime` after changing CUDA runtime
C++ so the local and remote runtime shared objects are rebuilt before the
smoke runs.

Run the persistent-device tracer-bullet smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 2 --n 1024 --arch compute_80
```

Run the scheduler/worker ready-queue persistent smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 1024 --arch compute_80 --mode queue
```

Pass `--worker-blocks` and `--stream-id` to validate the current
persistent-device resource policy: one scheduler block, configurable queue/DAG
worker blocks, direct-mode `--worker-blocks-per-task`, and CUDA callable stream
selection.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_80 \
    --mode queue --queue-capacity 2 --worker-blocks 2 --stream-id 1
```

Run the bounded-ring persistent smoke with wraparound:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_80 \
    --mode queue --queue-capacity 2
```

Run the persistent DAG smoke with dispatch and fan-in counters:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2
```

Use `--dag-shape scalar_axpy` to validate mixed tensor/scalar persistent DAG
task arguments. The first DAG task reads `scalar0` from the task descriptor
and computes `out = scalar0 * a + b` before downstream generated-dispatch
tasks consume its output.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape scalar_axpy
```

Use `--dag-shape scalar_affine` to validate two scalar fields in the
persistent DAG task descriptor. The first DAG task reads `scalar0` and
`scalar1` and computes `out = scalar0 * a + scalar1 * b` before downstream
generated-dispatch tasks consume its output.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape scalar_affine
```

Use `--dag-shape triad` to validate the third tensor pointer field in the
persistent DAG task descriptor. The first DAG task reads `c` from `tmp0` and
computes `out = a * b + c` before downstream generated-dispatch tasks consume
its output.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape triad
```

Use `--dag-shape quad` to validate third and fourth tensor pointer fields in
the persistent DAG task descriptor. The first DAG task reads `c` from `tmp0`
and `d` from `tmp3`, computes `out = a * b + c * d`, then a downstream add
task combines it with an independent `a * b` branch.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape quad
```

Use `--dag-shape generic_args` to validate the generic persistent DAG
argument slots. The first DAG task reads `tensor_args[0]` from `tmp0`,
`tensor_args[1]` from `tmp3`, `scalar_args[0]`, and `scalar_args[1]`, then a
downstream add task combines it with an independent `a * b` branch.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape generic_args
```

Use `--dag-shape graph_descriptor` to validate the same generated-dispatch
task bodies through an explicit runtime graph descriptor. This mirrors the
`persistent_dag_graph_f32` SceneTestCase adapter: each task supplies its
`func_id`, dependency list, fan-in, tensor/scalar slots, and temporary/output
bindings as descriptor data.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape graph_descriptor
```

Run the corresponding no-torch L2 `SceneTestCase` path after changing generic
persistent argument lowering:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k generic_args_with_ctypes --platform cuda
```

Use `--dag-shape unary_square` to validate a generated-dispatch task body
that reads only one tensor input from the persistent DAG descriptor. The
first DAG task computes `tmp0 = a * a`, then downstream add tasks consume its
output.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape unary_square
```

Run the corresponding benchmark baseline directly after changing the
two-scalar DAG descriptor or generated-dispatch benchmark wiring:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_affine \
    --sizes 4096 --arch compute_80
```

Use `cuda_pair_persistent_smoke.py` when the same persistent DAG smoke should
be captured on local A100 and remote H200 with Markdown/SVG evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape chain --task-count 5 --queue-capacity 3 \
    --worker-blocks 2 --stream-id 1 --sync-remote-tree
```

Use `--dag-shape graph_descriptor --repeat-runs 2` with the paired persistent
smoke runner to validate explicit graph-descriptor lifecycle reuse on A100 and
H200. This path prepares the generated-dispatch callable once, resets fan-in,
ready flags, counters, and scratch/output buffers between launches, then runs
the same prepared callable twice.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor --task-count 3 --queue-capacity 2 \
    --repeat-runs 2 --sync-remote-tree
```

This writes `a100.json`, `h200.json`, `cuda-smoke-report.md`, and
`cuda-smoke-report.svg` under
`tmp/cuda-backend/persistent-<shape>-smoke-<commit>/`, validates the paired
smoke artifact, then refreshes `tmp/cuda-backend/index.md`. Use
`--sync-remote-tree` when remote Git fetch is unreliable or the remote
`origin` URL is not accessible. Use `--skip-validation` only for intentional
negative scheduler-diagnostic captures.
The paired runner validates persistent smoke artifacts with the equivalent of:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-d3a86494/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor-repeat2-smoke-d3a86494/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 9,2,1 \
    --require-report-files
```

For tensor-tile smokes, the paired runner also passes
`--expected-tensor-tile ROWSxCOLSxINNER` so the validator rejects artifacts
whose recorded descriptor shape does not match the requested
`--tensor-rows`, `--tensor-cols`, and `--tensor-inner`.

The JSON payload and compact report include `resource_policy` fields for
`scheduler_blocks`, `worker_blocks`, `worker_blocks_per_task`, `stream_id`,
`block_dim`, and `grid_dim`. Scalar DAG payloads also include `scalar_args`
and tensor DAG payloads include `tensor_args`, so descriptor arguments are
visible in the Markdown and SVG reports. The `generic_args` payload also
includes a nested `generic_args` summary showing the indexed generic tensor
and scalar slots used by the task descriptor.
The current two-scalar descriptor capture is under
`tmp/cuda-backend/persistent-scalar_affine-smoke-469f55cd/`.
The current third-tensor descriptor capture is under
`tmp/cuda-backend/persistent-triad-smoke-3a3bcdb1/`.
The current generic-argument descriptor capture is under
`tmp/cuda-backend/persistent-generic_args-smoke-7c99f607/`.
For `--dag-shape tensor_tile`, pass `--tensor-rows`, `--tensor-cols`, and
`--tensor-inner`; the artifact directory includes the descriptor shape, such
as `persistent-tensor_tile-8x4x12-smoke-<commit>/`.

The generated-dispatch DAG smoke also carries device-side scheduler
diagnostics. A normal pass returns `device_scheduler_errors` with zero counts.
Use the synthetic invalid-dispatch shape below to validate propagation of an
unsupported device `func_id` without relying on output mismatches:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_func_id",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic invalid-dependent shape below to validate propagation of a
runtime graph descriptor whose dependent task ID is outside the task array:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_dependent",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic invalid-dependent-range shape below to validate propagation
of a runtime graph descriptor whose dependent range exceeds the dependent
array length:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_dependent_range",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic fan-in-underflow shape below to validate propagation of a
runtime graph descriptor whose fan-in metadata is lower than its number of
producer releases:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=3,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=2,
        dag_shape="bad_fanin_underflow",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic initial-fan-in mismatch shape below to validate propagation
of a runtime graph descriptor whose fan-in array does not match task
`initial_fanin` metadata:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_initial_fanin",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic no-root shape below to validate propagation of a runtime
graph descriptor that has no zero-fan-in task and would otherwise leave worker
blocks waiting for the first ready-queue entry:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(".agents/skills/cuda-backend-eval/scripts").resolve()))
from cuda_persistent_smoke import run_persistent_smoke

try:
    run_persistent_smoke(
        device=0,
        task_count=1,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_no_root",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Run the five-task persistent DAG-chain smoke, which reuses the same generated
dispatch PTX but passes a different runtime task graph:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape chain
```

Pass `--repeat-runs` with direct, queue, or DAG mode to reuse one prepared
persistent callable across multiple launches. Queue mode resets the ready
queue counters/flags between launches, and DAG mode resets the runtime graph
state. This validates callable lifecycle separately from scratch-buffer reuse
inside one graph:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 5 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape chain --repeat-runs 2
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 4096 --arch compute_80 \
    --mode queue --queue-capacity 2 --repeat-runs 2
```

Run the six-task persistent DAG scratch-reuse smoke. This graph reuses `tmp0`
after its last dependent has completed and validates the final reused-buffer
contents:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape scratch_reuse
```

Run the tensor-tile persistent DAG smoke. This graph uses a generated-dispatch
tiled GEMM task with rows/cols/inner/stride descriptor metadata, then residual,
gate, and fan-in elementwise tasks. The default descriptor is 16x16x16:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 4 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape tensor_tile
```

Pass `--tensor-rows`, `--tensor-cols`, and `--tensor-inner` to test a
non-square descriptor. `n` must be a multiple of `rows * cols`; the smoke
allocates separate A, B, and output extents so A/B strides can differ from the
output tile size while later elementwise tasks still run over `n` values.

Run the tensor-core persistent DAG smoke to validate the first block-wide
generated-dispatch task body. This shape uses CUDA WMMA
`m16n16k8` with TF32 inputs and F32 accumulation for the first task, then the
same residual, gate, and fan-in elementwise tasks as `tensor_tile`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_core_tile --task-count 4 --queue-capacity 2 \
    --n 256 --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree
```

The paired runner validates dispatch `10,1,2,1`, tensor descriptor
`16x16x16`, zero scheduler errors, and generated Markdown/SVG report files.
The report includes a `Tensor core` column such as
`wmma:m16n16k8:tf32->f32`. Treat this as a tensor-core callable smoke, not yet
as a tuned throughput result.

The DAG smoke compiles generated CUDA source through
`KernelCompiler(platform="cuda").compile_cuda_persistent_device(...)`. The
returned JSON includes `source_kind: generated-dispatch` when that path is in
use. The built-in DAG tasks are emitted as `CudaTaskBody` style source files
with `PtoTaskContext *ctx`, so this path exercises the same task-body wrapper
contract as `host_schedule`. The `nvcc` path writes `generated_dispatch.cu`,
`pto_callable.ptx`, and `pto_callable.json` under
`build/cache/cuda/onboard/persistent_device/callables/` before the host runtime
loads the PTX bytes.

For host-schedule task-body compiler work, use
`KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)`. It renders
one shared PTO CUDA task body into host-schedule and persistent-device wrappers,
then writes the host-schedule generated source, PTX, and JSON manifest under
`build/cache/cuda/onboard/host_schedule/callables/`. Use
`prepare_cuda_host_schedule_callable(...)` to turn the artifact into the shared
ctypes manifest consumed by the current `prepare_callable` C API. This is still
a compiler/runtime slice; L2 `Worker.register(...)` can prepare that raw
manifest blob, and L2 `Worker.run(...)` can launch raw CUDA argument structs
that expose `buffer_ptr()` / `buffer_size()`. The normal `SceneTestCase` L2
path can now build `CALLABLE["cuda"]` host-schedule specs and run the current
`arg_builder: vector_add_f32`, `arg_builder: elementwise_binary_f32`,
`arg_builder: elementwise_unary_f32`, `arg_builder: elementwise_scale_f32`,
`arg_builder: elementwise_axpy_f32`, and
`arg_builder: elementwise_affine_f32`, and
`arg_builder: elementwise_triad_f32`, and
`arg_builder: elementwise_quad_f32` adapters from CPU
`TaskArgsBuilder` tensors and scalars through real CUDA device buffers.
Use the neutral `elementwise_binary_f32` name when the compiled task body is
not addition but still uses the current `(a, b, out, n)` launch ABI. The same
path can build
`persistent_device` generated-dispatch DAG specs and run the
`arg_builder: persistent_dag_fork_join_f32`,
`arg_builder: persistent_dag_chain_f32`,
`arg_builder: persistent_dag_reuse_f32`,
`arg_builder: persistent_dag_scalar_axpy_f32`,
`arg_builder: persistent_dag_scalar_affine_f32`,
`arg_builder: persistent_dag_tensor_tile_f32`,
`arg_builder: persistent_dag_triad_f32`,
`arg_builder: persistent_dag_quad_f32`,
`arg_builder: persistent_dag_generic_args_f32`,
`arg_builder: persistent_dag_graph_f32`, and
`arg_builder: persistent_dag_unary_square_f32` adapters through the L2
`Worker`.
Use `persistent_dag_graph_f32` when a test should pass an explicit runtime
graph descriptor with per-task `func_id`, `a`/`b`/`c`/`d`/`out`,
`dependents`, optional `initial_fanin`, `tensor_args`, and `scalar_args`
fields instead of selecting one of the fixed tracer-bullet DAG adapters.
For real host-schedule smoke coverage, pass a context definition plus
`host_parameters`/`host_context_initializer` so the generated `__global__`
wrapper matches the current vector-add launch ABI and can be loaded by
`prepare_callable`.

## Microbenchmark Report

Use `cuda_benchmark.py` for the current early-runtime comparison. It runs the
same vector-add PTX kernel through two launch paths:

- `pto_host_schedule`: the PTO CUDA host runtime C API and manifest dispatch.
- `pto_host_schedule_compiler`: the same host runtime path, but the PTX comes
  from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)` and
  the shared task-body wrapper generator.
- `pto_host_schedule_unary_square`: the same generated host runtime path for
  the unary `(a, out, n)` ABI, using a square task body.
- `pto_host_schedule_quad`: the same generated host runtime path for the
  four-input `(a, b, c, d, out, n)` ABI.
- `direct_driver`: a thin CUDA Driver API baseline in Python `ctypes`.
- `direct_driver_graph`: the same Driver API kernel replayed through a CUDA
  Graph, with graph instantiation outside the timed interval.
- `pto_persistent_device`: a descriptor-array persistent executor.
- `pto_persistent_queue`: one scheduler block publishing ready task IDs to a
  bounded device ring queue consumed by worker blocks inside the same launch.
- `pto_persistent_dag`: generated-dispatch-like task selection and fan-in
  counters that release dependent tasks onto the bounded ring.
- `pto_persistent_dag_chain`: five-task generated-dispatch DAG with a
  post-fan-in dependency chain, using the same compiled device binary as the
  smaller DAG and only changing runtime graph descriptors.
- `pto_persistent_dag_reuse`: six-task generated-dispatch DAG with scratch
  buffer reuse after dependency completion, validating that graph lifetime
  rules can be represented by runtime descriptors.
- `pto_persistent_dag_scalar_axpy`: generated-dispatch DAG with a `scalar0`
  task descriptor field, validating mixed tensor/scalar persistent DAG
  arguments in the benchmark path.
- `pto_persistent_dag_triad`: generated-dispatch DAG with a `c` tensor task
  descriptor field, validating three-input persistent DAG arguments in the
  benchmark path.
- `pto_persistent_dag_quad`: generated-dispatch DAG with `c` and `d` tensor
  task descriptor fields, validating four-input persistent DAG arguments in
  the benchmark path.
- `pto_persistent_dag_generic_args`: generated-dispatch DAG with generic
  tensor/scalar descriptor slots, validating variable-arity persistent DAG
  arguments in the benchmark path.
- `pto_persistent_dag_graph`: generated-dispatch DAG using an explicit
  runtime graph descriptor, validating the generic graph-lowering path shared
  with `persistent_dag_graph_f32`.
- `pto_persistent_dag_unary_square`: generated-dispatch DAG with a one-input
  square task body, validating unary persistent DAG arguments in the
  benchmark path.
- `pto_persistent_dag_tensor`: four-task generated-dispatch DAG with a tiled
  GEMM task followed by elementwise residual, gate, and fan-in tasks. The
  benchmark uses the default 16x16x16 descriptor unless
  `--tensor-rows`, `--tensor-cols`, and `--tensor-inner` are supplied.
- `pto_persistent_dag_tensor_core`: four-task generated-dispatch DAG with a
  block-wide WMMA `m16n16k8` TF32 tensor-core task followed by the same
  elementwise residual, gate, and fan-in tasks. Use this for the first
  tensor-core benchmark row; it still measures a single generated task shape,
  not a tuned model kernel.
- `cublas_sgemm`: CUDA Runtime API plus cuBLAS
  `cublasSgemmStridedBatched` over the configured tensor descriptor. Use this
  as the first library-backed tensor baseline for the same report shape as the
  PTO persistent tensor rows; it measures a warm cuBLAS handle and CUDA Runtime
  events, not PTO runtime overhead.
- `pto_host_schedule_batch`, `pto_persistent_device_batch`,
  `pto_persistent_device_grid_batch`, and `pto_persistent_queue_batch`:
  same-work batch rows enabled by `--batch-tasks N`. Pass a
  comma-separated list, such as `--batch-tasks 2,6,12`, to sweep descriptor
  counts in one report. The worker-grid row is enabled with
  `--worker-blocks-per-task M` and assigns multiple CUDA worker blocks to each
  task descriptor. Pass a comma-separated list, such as
  `--worker-blocks-per-task 32,64,128,256`, to sweep grid shapes in one
  report.

The smoke helper caches the built and loaded host runtime per process, so the
benchmark can run repeated PTO and baseline samples without rebuilding a shared
object that is already loaded.

Local A100 example:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 32,64,128,256 \
    --label a100-local --output-dir tmp/cuda-backend/a100
```

Remote H200 example:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 2,6,12 \
     --worker-blocks-per-task 32,64,128,256 \
     --label h200-remote --output-dir tmp/cuda-backend/h200'
```

Current paired A100/H200 benchmark recipe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py
```

This runs the local A100 benchmark, checks out the current branch on
`bizhaoh200`, runs the H200 benchmark, copies the H200 artifact directory back,
merges the two JSON reports, and refreshes `tmp/cuda-backend/index.md`.
The remote checkout step uses bounded Git HTTPS low-speed settings by default;
override them with `--remote-git-low-speed-limit` and
`--remote-git-low-speed-time` if the remote network is unusually slow. The
fetch is also wrapped in `timeout`; override it with
`--remote-git-fetch-timeout`. If the remote checkout is already prepared and
Git HTTPS is unhealthy on `bizhaoh200`, pass `--skip-remote-refresh` to reuse
that checkout and run only the H200 benchmark command before copying artifacts
back. In that mode the runner reads `git rev-parse --short HEAD` from the
remote checkout and uses that remote commit in the H200 artifact name. If the
local and remote commits differ, the combined artifact name includes both
commits. A skip-refresh dry run still performs the read-only remote
`git rev-parse --short HEAD` probe so the printed artifact paths are accurate.

If remote Git remains unhealthy and the local checkout is the source of truth,
pass `--sync-remote-tree` to copy the current local tree to `bizhaoh200` with
`rsync` before running H200. This mode skips remote Git entirely, excludes
`.venv`, `build`, and `tmp`, and labels both A100 and H200 artifacts with the
local commit. It syncs `.git` so the remote benchmark metadata reports the
same commit as the synced source tree.

Use `--dry-run` to print the commands without launching benchmarks. The current
committed summary uses the `61cf96cd` artifact names in
`docs/nvidia-backend/evaluation-current.md`.

For a lighter no-torch real-data check, run the paired Worker smoke instead of
the full benchmark:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op triad --sync-remote-tree --build-runtime
```

It mirrors the benchmark runner's remote refresh, `--skip-remote-refresh`,
`--sync-remote-tree`, and `--dry-run` controls. It also supports
`--build-runtime` for source changes under `src/cuda/`. It captures
host-schedule Worker smoke JSON on A100/H200, renders a compact smoke report,
and refreshes the artifact index.

The script writes:

- `cuda-benchmark.json`: raw samples, metadata, hardware, git commit.
- `cuda-benchmark.md`: short report with interpretation notes.
- `cuda-benchmark.svg`: bar chart of median device time by baseline.
- `cuda-benchmark-ratios.svg`: bar chart of each row's device-time ratio
  against its matched reference.
- `cuda-benchmark-dag-deltas.svg`: bar chart of each `pto_persistent_dag_*`
  row's device-time increment over the matched `pto_persistent_dag` row.

For tensor-DAG and tensor-library experiments, pass `--tensor-rows`,
`--tensor-cols`, and `--tensor-inner` to the benchmark script. These flags
affect `pto_persistent_dag_tensor`, `pto_persistent_dag_tensor_core`, and
`cublas_sgemm`; other baselines keep their normal vector-add work. The
generated Markdown report records the descriptor as `rows x cols x inner`.

Use `--single-baseline pto_persistent_dag_scalar_axpy` for a quick benchmark
path check of the scalar descriptor DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_axpy \
    --sizes 1024 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_triad` for a quick benchmark path
check of the third-tensor descriptor DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_triad \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_quad` for a quick benchmark path
check of the fourth-tensor descriptor DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_quad \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_generic_args` for a quick benchmark
path check of generic tensor/scalar descriptor slots on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_generic_args \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph` for a quick benchmark path
check of the explicit graph-descriptor persistent DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_unary_square` for a quick
benchmark path check of the unary persistent DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_unary_square \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_host_schedule_unary_square` for a quick benchmark
path check of the unary host-schedule ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_unary_square \
    --sizes 1024 --arch compute_80
```

Use `--single-baseline pto_host_schedule_quad` for a quick benchmark path
check of the four-input host-schedule ABI:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_quad \
    --sizes 4096 --arch compute_80
```

Use `cuda_tensor_shape_sweep.py` to run paired A100/H200
samples over model-shaped tensor tile descriptors. By default it runs
`pto_persistent_dag_tensor`; pass `--baselines` to include
`pto_persistent_dag_tensor_core` and `cublas_sgemm` for a scalar-vs-WMMA-vs-
library comparison on compatible descriptors. Pass `--sizes` when the same
baseline/shape set should be swept across multiple problem sizes. Treat the
scalar tiled GEMM rows as shape and scheduler evidence rather than tensor-core
throughput evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --shapes 8x4x12,16x16x64,32x16x64 --n 4096 --repeats 3 \
    --sync-remote-tree
```

Run a compact tensor-baseline comparison sweep with shapes compatible with the
current WMMA task:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_tensor_core,cublas_sgemm \
    --shapes 16x16x16,16x16x64 --n 256 --repeats 3 \
    --sync-remote-tree
```

Run a size sweep for the same descriptor family when launch-dominated compact
rows need to be compared with larger repeated tensor work:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_tensor_core,cublas_sgemm \
    --shapes 16x16x16 --sizes 256,4096,65536 --repeats 3 \
    --sync-remote-tree
```

The sweep writes `cuda-tensor-shape-sweep.json`,
`cuda-tensor-shape-sweep.md`, `cuda-tensor-shape-sweep.svg`, and
`cuda-tensor-shape-throughput.svg` under
`tmp/cuda-backend/tensor-shape-sweep-<commit>/`. The Markdown keeps raw
repeat rows plus a median summary table with normalized GFLOP/s; the SVG
files plot median device time and median GFLOP/s per GPU/N/shape/baseline
with sample counts.

Validate the compact tensor-baseline sweep before copying numbers into docs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/tensor-shape-sweep-<commit>/cuda-tensor-shape-sweep.json \
    --preset compact-tensor-baselines
```

Validate a size sweep by spelling out the required sizes and result count:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/tensor-shape-sweep-<commit>/cuda-tensor-shape-sweep.json \
    --require-artifact a100 --require-artifact h200 \
    --require-baseline pto_persistent_dag_tensor \
    --require-baseline pto_persistent_dag_tensor_core \
    --require-baseline cublas_sgemm \
    --require-size 256 --require-size 4096 --require-size 65536 \
    --require-shape 16x16x16 --expected-repeats 3 \
    --expected-result-count 54 --require-report-files \
    --require-dispatch pto_persistent_dag_tensor=3,1,2,1 \
    --require-dispatch pto_persistent_dag_tensor_core=10,1,2,1
```

Use `--single-baseline pto_persistent_dag_tensor_core` for a quick benchmark
path check of the WMMA tensor-core generated-dispatch DAG:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_tensor_core \
    --sizes 256 --arch compute_80 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16
```

Use `--single-baseline cublas_sgemm` for a quick CUDA library-backed tensor
baseline check on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline cublas_sgemm \
    --sizes 256 --arch compute_80 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16
```

Refresh the local artifact index after adding or merging captures:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py \
    --root tmp/cuda-backend
```

The indexer scans benchmark `cuda-benchmark.json` files, tensor-shape sweep
`cuda-tensor-shape-sweep.json` files, and smoke `cuda-smoke-report.md`
directories, then writes `tmp/cuda-backend/index.md` with each artifact's
kind, metadata, baselines, vector sizes, tensor-tile descriptor shapes,
persistent smoke modes, dispatch sequences, scheduler error counters,
repeat-run counts, per-launch completion counts, and generated report/chart
presence. It is a local audit aid under `tmp/`; do not commit it with raw
benchmark, tensor sweep, or smoke data.

Render compact smoke JSON reports when a result is a smoke validation rather
than a full benchmark:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py \
    tmp/cuda-backend/tensor-descriptor-smoke-38db010e/a100.json \
    tmp/cuda-backend/tensor-descriptor-smoke-38db010e/h200.json \
    --label tensor-descriptor-smoke-38db010e \
    --output-dir tmp/cuda-backend/tensor-descriptor-smoke-38db010e
```

The smoke reporter writes `cuda-smoke-report.md` and
`cuda-smoke-report.svg`, keeping the raw JSON under `tmp/`. Persistent smoke
reports include lifecycle fields such as `repeat_runs` and
`launch_completed_counts` when the JSON payloads carry them.

The report's ratio column uses the matched host-schedule row for the same
machine, vector length, and task count. Same-work batch rows are therefore
relative to `pto_host_schedule_batch`, not the one-task `pto_host_schedule`
row.

The ratio SVG uses the same matched-reference rule as the Markdown table. It
is the clearest visual for launch-overhead comparisons such as
`direct_driver_graph` vs. `pto_host_schedule`, stream parallel-vs-serial, and
same-work persistent batch rows.

Render the compact tables used by `docs/nvidia-backend/evaluation-current.md`
directly from a combined benchmark JSON payload:

```bash
PYTHONPATH=$PWD:$PWD/python:.agents/skills/cuda-backend-eval/scripts \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py \
    tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json
```

Use `--section launch`, `--section unary-square`, `--section worker-grid`, or
`--section dag-shapes` to refresh only one table. This avoids hand-calculating
the current-evaluation summary from raw JSON.

Render the compact tensor-baseline sweep table directly from its raw sweep
JSON:

```bash
PYTHONPATH=$PWD:$PWD/python:.agents/skills/cuda-backend-eval/scripts \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py \
    tmp/cuda-backend/tensor-shape-sweep-47d857e1/cuda-tensor-shape-sweep.json \
    --section tensor-sweep
```

New tensor-sweep artifacts include source-paper provenance for the VDCores and
MPK notes in `tmp/sources/`, the paper-alignment statement for the current
microbenchmark setup, and one workload description per selected baseline.

Validate the paired-current capture before copying numbers into docs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json \
    --preset paired-current
```

The preset checks the expected A100/H200 machines, current selected
baselines, sizes `1024,65536,1048576`, three repeats, `720` combined samples,
and the Markdown/SVG report files.

Use `cuda_validate_smoke.py` for paired smoke artifacts. It checks required
artifacts, pass status, zero device scheduler errors, expected runtime/mode,
dispatch IDs, repeat-run lifecycle counts, tensor-tile descriptor shape when
requested, and generated smoke report files.
`cuda_pair_persistent_smoke.py` runs this validator automatically unless
`--skip-validation` is set.

When worker-grid rows are present, the report includes a
`Best Worker Grid Rows` table that picks the lowest median device time for
each machine, vector length, and task count.

When DAG-shape rows are present, the report includes a `DAG Shape Rows` table
that compares `pto_persistent_dag_*` rows against the three-task
`pto_persistent_dag` row for the same machine and vector length. Use this
table for lifecycle and callable-shape interpretation because those rows do
not have the same task count as the host-schedule baseline.
The adjacent `DAG Increment Rows` table and `cuda-benchmark-dag-deltas.svg`
show the absolute device-time delta after subtracting the matched
`pto_persistent_dag` scheduler baseline. Use that view when distinguishing
scheduler overhead from extra task-body work in tensor, tensor-core, graph,
and other generated-dispatch DAG rows.

The report also includes a PTX-source table by machine and baseline. Treat any
`embedded-sm80-*` row as a fallback path: the local CUDA driver JIT compiled
embedded `sm_80` PTX instead of using `nvcc` to produce fresh PTX for the
requested target architecture.

The default persistent batch row uses one worker block per descriptor. Add
`--worker-blocks-per-task M[,N...]` to include one
`pto_persistent_device_grid_batch` row per value, which separates launch
amortization from intra-task grid parallelism by giving each descriptor
multiple worker blocks. In the current vector-add slice, the
`32,64,128,256` extended sweep at `3eeb399a` showed `256` worker blocks per
descriptor as the best large-vector point on both A100 and H200. Treat that
as evidence for the current microbenchmark, not a tuned default. The
`7194bfc9` task-count sweep adds `--batch-tasks 2,6,12` with
`--worker-blocks-per-task 128,256`; use that report to reason about
descriptor-count scaling before setting a policy.

The `cc6869f7` wider range sweep uses `--sizes 16384,262144,4194304`,
`--batch-tasks 4,8,16`, and `--worker-blocks-per-task 128,256`. It keeps
worker-grid rows below the matched host-schedule batch row on A100 and H200,
but the `4194304` rows become compute-sensitive and no fixed `128` or `256`
blocks/task setting wins across every GPU, vector size, and task count.

The default benchmark includes `direct_driver_graph`. Use it to compare
`host_schedule` launch overhead against CUDA Graph replay for the same
one-kernel callable. It should not be interpreted as a persistent-device
scheduler because the host still instantiates and replays the graph.

Merge local and remote JSON payloads into one comparative report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100/cuda-benchmark.json \
    tmp/cuda-backend/h200/cuda-benchmark.json \
    --label cuda-a100-h200 --output-dir tmp/cuda-backend/combined
```

Run the host-schedule stream-concurrency microbenchmark:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --stream-concurrency --device 0 --repeats 5 --arch compute_80 \
    --label a100-streams --output-dir tmp/cuda-backend/a100-streams
```

For remote H200, use `--arch compute_90`. The scripts discover `nvcc` from
`CUDA_HOME`, `CUDA_PATH`, `PATH`, and common `/usr/local/cuda*` toolkit paths.
If `nvcc` is still unavailable, the script uses an embedded `sm_80` PTX
fallback that the H200 driver JITs.

The stream report compares `pto_stream_parallel` against `pto_stream_serial`.
The current A100/H200 capture at `37bebf44` shows about `0.51x` parallel-vs-
serial wall time on both machines, supporting multiple streams for the
host-schedule runtime when independent callables are launched from separate
host threads.

The DAG-chain capture at `323f4587` adds `pto_persistent_dag_chain` to the
normal `--include-persistent` benchmark. It shows the same generated-dispatch
PTX can run both the three-task fork/join DAG and a five-task post-fan-in
chain by changing only runtime graph descriptors. The chain row is expectedly
slower because it performs more vector work and has two more dependency
levels; use it as a lifecycle/scheduler validation row, not as a throughput
claim.

The scratch-reuse DAG capture at `bcf54a88` adds `pto_persistent_dag_reuse`.
It validates that a runtime graph can reuse a scratch buffer after dependency
completion without recompiling the generated-dispatch PTX. Use this as an
early buffer-lifecycle row; it is still elementwise vector work, not a tensor
kernel workload.

The tensor-tile DAG capture at `8950e029` adds `pto_persistent_dag_tensor`.
It validates generated-dispatch task bodies beyond elementwise kernels by
running the default 16x16x16 tiled GEMM descriptor before residual, gate, and
fan-in tasks. The later descriptor-metadata smoke extends that tensor task
with rows/cols/inner/leading-dimension and per-tile stride fields. The current
smoke helper can also run non-square descriptors and allocates separate A, B,
and output buffer extents. Use these rows as ABI and scheduler validation;
they are still scalar CUDA GEMM microbenchmarks, not tuned tensor-core
workloads.
The `38db010e` A100/H200 descriptor smoke outputs were saved under
`tmp/cuda-backend/tensor-descriptor-smoke-38db010e/`, with a generated smoke
Markdown report and SVG in the same directory.

The CUDA Graph launch-baseline capture at `ba2cdd0e` adds
`direct_driver_graph` to the default benchmark. It showed graph replay faster
than raw Driver API launch on A100 and H200, and faster than PTO
`host_schedule` on every captured row except H200 `N=1024`.

## Hardware Checks

Local A100:

```bash
command -v nvcc
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'hostname; command -v nvcc || true; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader'
```

The CUDA real-data pytest files use
`simpler_setup.cuda_preflight.cuda_skip_reason(require_nvcc=True)` before
running device work. The shared preflight skips with a concrete reason when
`nvcc`, `nvidia-smi`, or a visible NVIDIA driver/GPU is unavailable. Keep new
CUDA hardware tests on the same marker path instead of open-coding
`shutil.which("nvcc")`.

## Paper-Aligned Evaluation Shape

Use the papers to choose workloads and baselines, but do not claim paper-level
results until full LLM serving baselines are actually run.

- VDCores setup: fixed offline decoding, 128-token context, 64 decode steps,
  batch sizes 1-8, models including Qwen3-1.7B/Qwen3-8B and Llama small
  variants, baselines including vLLM, SGLang, Mirage, ThunderKittens, and
  Torch+ThunderKittens.
- MPK setup: offline batched inference, fixed prompt length 64, decode 1024
  tokens, max batch sizes 1-16, baselines vLLM and SGLang, GPUs A100/H100/B200
  in the paper and local A100/remote H200 for this repo.
- For early PTO CUDA runtime work, report only microbenchmarks and smoke tests
  as such: launch latency, host wall time, device event time, copy bandwidth,
  and simple dependency/concurrency tests.

## Result Capture

For repeatable local/remote runs, save raw command output under
`outputs/cuda-backend/` or `tmp/cuda-backend/` with hardware, git commit, CUDA
toolkit, driver, command, and timestamp. Commit only scripts, docs, and tests;
do not commit raw benchmark outputs unless the user asks.
