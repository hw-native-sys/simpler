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

Set `PTO_CUDA_STREAM_POOL_SIZE=<N>` before initializing the CUDA host runtime
when a host-schedule concurrency experiment needs callable `stream_id` values
outside the default four-stream pool. Invalid, zero, or very large values fall
back to the default. Validate the knob on A100/H200 with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
    -q -k 'stream_pool_size_env or independent_callables_on_multiple_streams' \
    --platform cuda
```

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

Use `--op generic_args` to validate the host-schedule generic tensor/scalar
ABI. The generated task body reads indexed tensor slots and scalar slots
through the runtime's generic argument packet:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op generic_args --device 0 --n 1024 \
    --block-dim 256 --arch compute_80 \
    --output-json tmp/cuda-backend/worker-generic_args-smoke/a100.json
```

Use `--op generic_args4` after changing the host-schedule generic packet path
that consumes all four `CudaVectorGenericArgs::tensor_args` and
`scalar_args` slots:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_smoke.py \
    --runner worker --op generic_args4 --device 0 --n 1024 \
    --block-dim 256 --arch compute_80 \
    --output-json tmp/cuda-backend/worker-generic_args4-smoke/a100.json
```

Use `cuda_smoke_report.py` to turn captured smoke JSON from A100 and H200 into
Markdown and SVG evidence. Persistent-device reports include dispatch
`func_id` sequences, device-side scheduler error counters, resource policy,
tensor/scalar task argument metadata, graph descriptor fan-in/dependent
arrays, tagged graph `graph_task_args` metadata, and repeat-run lifecycle
counters when present. Nonzero scheduler error codes are rendered with stable
taxonomy names such as `7(unreachable_task)` in validators, smoke reports, and
artifact indexes. The shared label table lives in
`.agents/skills/cuda-backend-eval/scripts/cuda_scheduler_errors.py`, so add
new device-side scheduler codes there before updating individual reports:

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

Pass `--worker-blocks`, `--stream-id`, and `--block-dim` to validate the current
persistent-device resource policy: one scheduler block, configurable queue/DAG
worker blocks, direct-mode `--worker-blocks-per-task`, and CUDA callable stream
selection plus the manifest block dimension.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_80 \
    --mode queue --queue-capacity 2 --worker-blocks 2 --stream-id 1 \
    --block-dim 128
```

For paired A100/H200 evidence of the same policy, use the persistent runner.
It validates the recorded `resource_policy` fields in both JSON artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape chain --task-count 5 --queue-capacity 3 \
    --worker-blocks 2 --stream-id 1 --block-dim 128 --repeat-runs 2 \
    --sync-remote-tree
```

The paired `block_dim=128` resource-policy capture under
`tmp/cuda-backend/persistent-block128-working/` validated A100 and H200 JSON,
Markdown, and SVG artifacts with `scheduler_blocks=1`, `worker_blocks=2`,
`stream_id=1`, `block_dim=128`, `grid_dim=3`, `repeat_runs=2`, and DAG-chain
dispatch `1,2,1,2,1`.

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

Use `--dag-shape scalar_scale` to validate the single-input scalar persistent
DAG task descriptor. The first DAG task reads `scalar0` and computes
`out = scalar0 * a` before downstream generated-dispatch tasks consume its
output.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape scalar_scale
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

Use `--dag-shape generic_args4` when the persistent DAG generated-dispatch
task body should consume all four bounded generic tensor/scalar descriptor
slots. The current shape maps `tensor_args[2]` to `a`, `tensor_args[3]` to
`b`, and records those slots in the smoke report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape generic_args4
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

Use `--dag-shape graph_descriptor_generic_args4` when the graph descriptor
path should also prove all four bounded generic tensor/scalar slots. This
records both graph metadata and `tensor_args[2]=a`, `tensor_args[3]=b` in the
smoke JSON:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 3 --n 4096 --arch compute_80 \
    --mode dag --queue-capacity 2 \
    --dag-shape graph_descriptor_generic_args4
```

Use `--dag-shape graph_descriptor_triad` and
`--dag-shape graph_descriptor_quad` when the graph descriptor path should
prove the fixed third and fourth tensor task descriptor fields. These reuse
the triad/quad generated-dispatch task bodies while requiring explicit graph
metadata in the JSON and paired validator:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_triad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_quad --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-working
```

The paired runner validates triad dispatch `6,2,1`, quad dispatch `8,2,1`,
graph fan-in `0,0,2`, dependents `2,2`, repeat completions, zero scheduler
errors, tensor-argument metadata, and generated Markdown/SVG report files.

Run the corresponding no-torch L2 `SceneTestCase` path after changing generic
persistent argument lowering:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k generic_args_with_ctypes --platform cuda
```

Run the persistent scalar-scale L2 `SceneTestCase` path after changing
single-tensor scalar descriptor lowering. This selector is no-torch and can
run on the remote H200 venv:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k scalar_scale --platform cuda
```

Run the mixed explicit/inferred graph descriptor path after changing
`persistent_dag_graph_f32` dependency inference. This selector exercises a
graph where one task keeps explicit `dependents` and another task has its
outgoing edge inferred from tensor flow:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k mixed_graph_with_ctypes --platform cuda
```

Run the tagged graph descriptor path after changing TaskArgs-like graph
lowering for tensor roles or scalar inputs. The selector includes a
descriptor-only scalar task-arg case and a no-torch real-data tagged graph
scene that lowers `input`, `output`, `output_existing`, and `scalar`
task-arg entries:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'scalar_task_args or tagged_graph' --platform cuda
```

Run the graph scratch-storage reuse SceneTestCase path after changing
`persistent_dag_graph_f32` temporary allocation or tensor-flow inference. This
selector checks that logical `out` names stay unique while `out_storage`
aliases a later graph output onto an earlier scratch buffer:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'reused_output_storage or graph_scratch_reuse_with_ctypes' \
    --platform cuda
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

Use `--dag-shape graph_descriptor_unary_square` when the same one-input
generated-dispatch DAG should be represented as explicit runtime graph
metadata. The paired validator requires dispatch `7,1,1`,
`graph_descriptor.fanin=0,1,1`, and `graph_descriptor.dependents=1,2`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_unary_square --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-unary-square-working
```

Run the corresponding benchmark baseline directly after changing a scalar DAG
descriptor or generated-dispatch benchmark wiring:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_scalar_scale \
    --sizes 4096 --arch compute_80

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

Use `--dag-shape graph_descriptor_chain --repeat-runs 2` to validate the
five-task DAG-chain dependency shape as explicit graph descriptor metadata
instead of as the fixed `chain` shape:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_chain --task-count 5 --queue-capacity 3 \
    --repeat-runs 2 --sync-remote-tree
```

Use `--dag-shape graph_descriptor_depends_on --repeat-runs 2` to validate
incoming-edge graph notation. This shape records graph fan-in `0,0,2`,
dependents `2,2`, and dispatch `1,2,1`, while the final consumer keeps
`a`/`b` bound to original inputs instead of producer temporaries:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_depends_on --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

The current working-tree paired capture is under
`tmp/cuda-backend/depends-on-graph-working/persistent-graph_descriptor_depends_on-repeat2-smoke-06b988b5/`.

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
    --expected-graph-fanin 0,0,2 --expected-graph-dependents 2,2 \
    --expected-scheduler-blocks 1 --expected-worker-blocks 3 \
    --expected-worker-blocks-per-task 1 --expected-stream-id 0 \
    --expected-block-dim 256 --expected-grid-dim 4 \
    --require-report-files --require-report-graph-topology
```

For generated-dispatch DAG shapes, the paired runner passes
`--expected-dispatch` for the known `func_id` sequence. This covers `chain`,
`fork_join`, `scratch_reuse`, tensor-tile and tensor-core-tile shapes, graph
tensor-core-tile, scalar AXPY/scale/affine, triad, quad, unary-square,
`generic_args`,
`generic_args4`, `graph_descriptor`, `graph_descriptor_chain`,
`graph_descriptor_depends_on`, `graph_descriptor_generic_args4`,
`graph_descriptor_reordered`, `graph_descriptor_diamond`,
`graph_descriptor_scratch_reuse`,
`graph_descriptor_tagged`, `graph_descriptor_tagged_inout`,
`graph_descriptor_unary_square`, `graph_tensor_tile`, and
`graph_tensor_core_tile`. The validator therefore rejects A100/H200 artifacts
that pass numerically through a different generated task path.

For tensor-tile smokes, the paired runner also passes
`--expected-tensor-tile ROWSxCOLSxINNER` so the validator rejects artifacts
whose recorded descriptor shape does not match the requested
`--tensor-rows`, `--tensor-cols`, and `--tensor-inner`.
For explicit graph-descriptor smokes, it also passes
`--expected-graph-fanin` and `--expected-graph-dependents`, so reordered,
depends-on, diamond, scratch-reuse, and graph tensor captures must prove the
recorded runtime graph topology. It also passes
`--require-report-graph-topology`, so the Markdown and SVG smoke reports must
visibly carry the same fan-in and dependent arrays.
For `graph_descriptor_tagged` and `graph_descriptor_tagged_inout`, it
additionally passes `--expected-graph-task-args`, so tagged graph captures
must prove the TaskArgs-like roles that were lowered into the runtime
descriptor. It also passes `--require-report-graph-task-args`, so the
Markdown and SVG smoke reports must show those roles instead of only carrying
them in JSON.

The JSON payload and compact report include `resource_policy` fields for
`scheduler_blocks`, `worker_blocks`, `worker_blocks_per_task`, `stream_id`,
`block_dim`, and `grid_dim`. Scalar DAG payloads also include `scalar_args`
and tensor DAG payloads include `tensor_args`, so descriptor arguments are
visible in the Markdown and SVG reports. The `generic_args` payload also
includes a nested `generic_args` summary showing the indexed generic tensor
and scalar slots used by the task descriptor.
The paired persistent runner now passes the expected resource-policy fields
to `cuda_validate_smoke.py`, so A100/H200 smoke artifacts are rejected when
the CUDA persistent scheduler runs with a different worker-grid or stream
policy than the command requested.
The current paired resource-policy capture is under
`tmp/cuda-backend/persistent-chain-repeat2-smoke-4b220bb7/`.
The current two-scalar descriptor capture is under
`tmp/cuda-backend/persistent-scalar_affine-smoke-469f55cd/`.
The current single-input scalar-scale descriptor capture is under
`tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/`.
The current third-tensor descriptor capture is under
`tmp/cuda-backend/persistent-triad-smoke-3a3bcdb1/`.
The current generic-argument descriptor capture is under
`tmp/cuda-backend/persistent-generic_args-smoke-7c99f607/`.
The current generic-argument repeat-run lifecycle capture is under
`tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/`.
The current four-slot generic-argument repeat-run capture is under
`tmp/cuda-backend/persistent-generic_args4-repeat2-smoke-7bac4e3e/`.
The current graph-descriptor four-slot generic-argument repeat-run capture is
under
`tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`.
Use `--dag-shape graph_descriptor_tagged --repeat-runs 2` to validate the
same three-task graph descriptor shape after lowering tagged TaskArgs-like
entries (`input`, `output`, `output_existing`, and scalar inputs) into the
CUDA descriptor fields. The current paired capture is under
`tmp/cuda-backend/graph-tagged-scalar-working/persistent-graph_descriptor_tagged-repeat2-smoke-a618e624/`.
The paired validator requires the scalar roles in the tagged task metadata:
`task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]`.
The current graph-descriptor DAG-chain repeat-run capture is under
`tmp/cuda-backend/persistent-graph_descriptor_chain-repeat2-smoke-b94b555d/`.
Use `--dag-shape graph_descriptor_reordered --repeat-runs 2` to validate that
graph-descriptor dependencies are inferred from tensor flow across the whole
descriptor, even when the final consumer task appears before its producers.
The current capture is under
`tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/`.
Use `--dag-shape graph_descriptor_diamond --repeat-runs 2` to validate an
explicit runtime graph descriptor with two root producers, two fan-out
consumers, and one final join. This shape reuses the same generated-dispatch
task bodies as `graph_descriptor`, but records fan-in `[0,0,2,2,2]`,
dependents `[2,3,2,3,4,4]`, and dispatch `9,2,1,2,1`. The current capture is
under
`tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/`.
Use `--dag-shape graph_descriptor_scratch_reuse --repeat-runs 2` to validate
that an explicit runtime graph descriptor can represent the scratch-reuse DAG
shape. This records fan-in `[0,0,2,1,1,2]`, dependents
`[2,2,3,4,5,5]`, dispatch `1,2,1,2,1,1`, and `scratch_reuse` metadata for
the reused `tmp0` buffer. The paired validator now requires
`scratch_reuse=reused_buffer=tmp0,reuse_task=4`, and the smoke Markdown/SVG
report renders that alias so the physical storage reuse is visible in the
artifact. The current capture is under
`tmp/cuda-backend/persistent-graph_descriptor_scratch_reuse-repeat2-smoke-d8f6d0bf/`.
For `--dag-shape tensor_tile`, pass `--tensor-rows`, `--tensor-cols`, and
`--tensor-inner`; the artifact directory includes the descriptor shape, such
as `persistent-tensor_tile-8x4x12-smoke-<commit>/`.
Use `--dag-shape graph_tensor_tile` when the smoke should validate the same
tiled-GEMM/residual/gate/fan-in task sequence through the explicit graph
descriptor path. It records both `graph_descriptor` dependency metadata and
the `tensor_tile` descriptor in the JSON and report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_tensor_tile --task-count 4 --queue-capacity 2 \
    --repeat-runs 2 --n 512 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree
```

The current working-tree capture is under
`tmp/cuda-backend/persistent-graph_tensor_tile-16x16x16-repeat2-working/`.

Use `--dag-shape graph_tensor_core_tile` when the smoke should validate the
WMMA tensor-core first task through the explicit graph descriptor path. This
records dispatch `10,1,2,1`, graph fan-in `0,1,1,2`, dependents `1,2,3,3`,
the requested tensor descriptor, tensor-core metadata, repeat completions,
zero scheduler errors, and generated Markdown/SVG report files:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_tensor_core_tile --task-count 4 --queue-capacity 2 \
    --repeat-runs 2 --n 256 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-core-working
```

The current capture is under
`tmp/cuda-backend/graph-tensor-core-working/persistent-graph_tensor_core_tile-16x16x16-repeat2-smoke-40aa2f43/`.

The current graph-descriptor unary-square repeat-run capture is under
`tmp/cuda-backend/graph-unary-square-working/persistent-graph_descriptor_unary_square-repeat2-smoke-02c99b5c/`.

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

Use the synthetic duplicate-dependent shape below to validate propagation of
a runtime graph descriptor that lists the same dependent task twice for one
completed task:

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
        task_count=2,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=2,
        dag_shape="bad_duplicate_dependent",
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use the synthetic self-dependent shape below to validate propagation of a
runtime graph descriptor that makes a completed task release itself:

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
        dag_shape="bad_self_dependent",
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

Use the synthetic unreachable-task shape below to validate propagation of a
runtime graph descriptor that has at least one ready root, but exhausts all
published work before every task completes. This catches cycle/dangling-fan-in
cases that would otherwise leave worker blocks waiting for a future
ready-queue entry:

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
        task_count=2,
        n=1024,
        arch="compute_80",
        mode="dag",
        queue_capacity=1,
        dag_shape="bad_unreachable",
        worker_blocks=2,
    )
except RuntimeError as exc:
    print(exc)
PY
```

Use `cuda_scheduler_error_matrix.py` when all synthetic scheduler diagnostics
should be captured as paired A100/H200 JSON, Markdown, and SVG evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_scheduler_error_matrix.py \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/scheduler-error-matrix-working
```

The current capture under
`tmp/cuda-backend/scheduler-error-matrix-working/scheduler-error-matrix-35de3303/`
validates all nine scheduler diagnostics on A100 and H200:
unsupported `func_id`, invalid dependent ID, invalid dependent range,
fan-in underflow, duplicate dependent, self dependent, initial fan-in
mismatch, no root, and unreachable task.

Validate the captured matrix contract with the dedicated checker. The default
preset requires A100/H200 coverage for all nine diagnostics, source-paper
provenance under `tmp/sources/`, command examples, and Markdown/SVG reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_scheduler_error_matrix.py \
    tmp/cuda-backend/scheduler-error-matrix-working/scheduler-error-matrix-35de3303/cuda-scheduler-error-matrix.json \
    --preset default
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

Use `cuda_persistent_lifecycle_matrix.py` when direct, queue, DAG-chain,
incoming-edge graph, and graph-descriptor scratch-reuse prepared-callable
lifecycle evidence should be captured together on local A100 and remote H200.
The default matrix uses `repeat_runs=2`, `stream_id=1`, direct
`worker_blocks_per_task=2`, and queue/DAG `worker_blocks=2`, then validates
each paired smoke and the combined matrix report before refreshing the
artifact index:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_lifecycle_matrix.py \
    --sync-remote-tree
```

This writes per-scenario smoke artifacts plus
`tmp/cuda-backend/persistent-lifecycle-matrix-<commit>/cuda-lifecycle-matrix.md`,
`cuda-lifecycle-matrix.svg`, and `cuda-lifecycle-matrix.json`.
Lifecycle matrix JSON/Markdown includes source-paper provenance, collection
mode, paper alignment text, sanitized local/remote command examples, graph
topology, scratch-reuse metadata, and tensor-tile metadata. Use
`--collect-existing-suffix
<commit>` to regenerate the combined matrix report and index from existing
per-scenario `a100.json`/`h200.json` files without rerunning A100/H200
hardware; the regenerated report's local sample command must include the same
flag.
Use `--dry-run` to print every paired smoke command plus the final matrix
validator and artifact-index commands without writing the matrix report.
The current paired lifecycle matrix capture is under
`tmp/cuda-backend/lifecycle-tensor-core-working/persistent-lifecycle-matrix-1c683c1c/`.
Validate a lifecycle matrix before copying its fields into docs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_lifecycle_matrix.py \
    tmp/cuda-backend/lifecycle-tensor-core-working/persistent-lifecycle-matrix-1c683c1c/cuda-lifecycle-matrix.json \
    --preset default --require-source-papers --require-command-examples
```

The default preset now requires `graph-depends-on` with dispatch `1,2,1`,
graph fan-in `0,0,2`, and dependents `2,2`. It also requires
`graph-scratch-reuse` with dispatch `1,2,1,2,1,1`, fan-in
`0,0,2,1,1,2`, dependents `2,2,3,4,5,5`, and
`scratch_reuse=reused_buffer=tmp0,reuse_task=4`. It also requires
`graph-tensor-core` with dispatch `10,1,2,1`, fan-in `0,1,1,2`,
dependents `1,2,3,3`, and tensor tile `16x16x16`. Focused one-scenario
lifecycle matrices should pass explicit `--scenario` values to the matrix
runner rather than relying on the default preset.

Run the six-task persistent DAG scratch-reuse smoke. This graph reuses `tmp0`
after its last dependent has completed and validates the final reused-buffer
contents:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    --device 0 --task-count 6 --n 1024 --arch compute_80 \
    --mode dag --queue-capacity 2 --dag-shape scratch_reuse
```

Use `--dag-shape graph_descriptor_scalar_scale` when the scalar-scale
three-task DAG should be represented as explicit runtime graph descriptor
metadata instead of the fixed `scalar_scale` shape:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_scale --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-scale-working
```

The paired runner validates dispatch `11,2,1`, graph fan-in `0,0,2`,
dependents `2,2`, repeat completions, zero scheduler errors, and generated
Markdown/SVG report files.

Use the graph descriptor scalar variants when fixed scalar AXPY/affine DAG
logic needs to be validated as explicit runtime graph metadata:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_axpy --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_scalar_affine --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-scalar-variants-working
```

The paired runner validates dispatch `4,2,1` for AXPY and `5,2,1` for
affine, graph fan-in `0,0,2`, dependents `2,2`, scalar metadata, repeat
completions, zero scheduler errors, and generated Markdown/SVG report files.

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

Run the tensor-core persistent DAG smoke to validate the block-wide
generated-dispatch task body. This shape uses CUDA WMMA `m16n16k8` with TF32
inputs and F32 accumulation for the first task, then the same residual, gate,
and fan-in elementwise tasks as `tensor_tile`. It supports descriptors whose
rows and columns are multiples of `16`, with inner dimension divisible by `8`:

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
Use `--tensor-rows 32 --tensor-cols 16 --tensor-inner 16` when the WMMA task
should execute multiple 16x16 output fragments per descriptor. The paired
capture under `tmp/cuda-backend/tensor-core-wide-working/` validates A100/H200
repeat-run completions, generated reports, and zero scheduler errors for that
shape.

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
`arg_builder: elementwise_quad_f32`, and
`arg_builder: elementwise_generic_args_f32` adapters from CPU
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
`arg_builder: persistent_dag_tensor_core_tile_f32`,
`arg_builder: persistent_dag_triad_f32`,
`arg_builder: persistent_dag_quad_f32`,
`arg_builder: persistent_dag_generic_args_f32`,
`arg_builder: persistent_dag_graph_f32`, and
`arg_builder: persistent_dag_unary_square_f32` adapters through the L2
`Worker`.

After changing the host-schedule generic tensor/scalar ABI, rebuild the CUDA
host runtime and run the no-torch ctypes scene selector locally and on H200:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
from simpler_setup.runtime_builder import RuntimeBuilder
RuntimeBuilder(platform="cuda").get_binaries("host_schedule", build=True)
PY

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k host_schedule_elementwise_generic_args_with_ctypes_data \
    --platform cuda
```

After changing CUDA runtime role discovery or `persistent_device` build
targets, rebuild the runtime and check that the role-keyed binary map includes
the optional scheduler image and that the Python/C++ `ChipWorker` boundary
accepts role-keyed maps. CUDA `RuntimeBinaries` should print `None None` for
the legacy AICPU/AICore path fields; use `role_paths` / `path_for_role`
instead:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
from simpler_setup.runtime_builder import RuntimeBuilder
bins = RuntimeBuilder(platform="cuda").get_binaries("persistent_device", build=True)
print(sorted(bins.role_paths))
print(bins.aicpu_path, bins.aicore_path)
print(bins.path_for_role("scheduler"))
PY

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_chip_worker.py tests/ut/py/test_runtime_builder.py \
    -q -k 'role_keyed_init or role_only_runtime_binaries or \
           role_keyed_paths or scheduler_role or cuda_runtime_binaries'

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_backend.py \
    -q -k role_keyed_init --platform cuda

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k persistent_device_graph_with_ctypes_data --platform cuda
```

Use `persistent_dag_tensor_core_tile_f32` for the normal L2 scene-test path
when the first DAG task should be a block-wide WMMA
`m16n16k8:tf32->f32` task. The current L2 adapter requires `rows=16`,
`cols=16`, and `K` divisible by `8`.
CUDA `persistent_device` scene-test specs can pass `stream_id` in the
`CALLABLE["cuda"]` spec; this is forwarded to the prepared callable manifest
and selects the CUDA runtime stream used by `Worker.run`.
The L2 `SceneTestCase` persistent-device path also reads the device scheduler
counters after each launch. It raises `RuntimeError` for nonzero scheduler
errors or incomplete DAG completion, so negative scheduler-diagnostic tests can
use `skip_golden=True` without silently accepting a failed device schedule.
Use `persistent_dag_graph_f32` when a test should pass an explicit runtime
graph descriptor with per-task `func_id`, `a`/`b`/`c`/`d`/`out`,
`dependents`, optional `initial_fanin`, `tensor_args`, and `scalar_args`
fields instead of selecting one of the fixed tracer-bullet DAG adapters.
Graph tasks may set `name` and then use task names in outgoing `dependents`
or incoming `depends_on` / `dependencies`; integer task IDs still work.
Prefer named edges for new graph-descriptor tests because they are closer to
normal named PTO task graphs and avoid renumbering errors when inserting
tasks.
Graph tasks may alternatively pass role-keyed `task_args` entries with
`input`, `output`, `output_existing`, or `inout` roles. The adapter prefers
the `role` key and still accepts the older `tag` spelling for compatibility.
It lowers the first four inputs to `a`/`b`/`c`/`d`, appends any additional
inputs to `tensor_args`, and lowers the single output to `out`; this is the
preferred test form when checking the first TaskArgs-like lowering slice.
For compact graph specs, a task arg may also be written as a single role-keyed
dictionary such as `{"input": "a"}`, `{"output": "tmp0"}`,
`{"inout": "tmp0"}`, or `{"output_existing": "out"}`. Do not mix this compact
form with `tensor`/`name` plus `role`/`tag` in the same task-arg entry.
Graph tasks may use a `callable` name instead of embedding `func_id` directly
when `graph.callables` maps that name to callable metadata such as
`{"func_id": 9}`. `graph.callables` may be either a dictionary keyed by
callable name or a list of callable specs with `name` fields, matching the
list-shaped callable registries used elsewhere in scene tests. For list-shaped
registries, tasks may reference a callable by `name` or by zero-based list
index. A list entry only needs `name` when tasks reference it by name; pure
index-based graphs may use unnamed callable specs, or the compact integer
form where each list element is the generated-dispatch `func_id`. The
task-local fields override callable defaults, and the adapter resolves the
callable before role-keyed `task_args`, temporary allocation, and tensor-flow
edge inference. Use this form when checking the scene-test step toward normal
PTO task graphs with indexed or named callables and task-argument roles.
Validate the list-shaped callable registry path with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_scene_test.py -q \
  -k 'compact_callable_index_graph_with_ctypes_data' --platform cuda
```

Use `output` only when a graph task creates a new default-sized temporary.
Use `output_existing` or `inout` only for storage already known at that point
in descriptor order, such as scene tensors, explicit temporaries, or
temporaries produced by earlier graph tasks. Unknown `output_existing` or
`inout` names should fail during descriptor construction instead of allocating
scratch storage.
Graph task scalar fields may be numeric literals or scalar names from the
scene's `TaskArgsBuilder`: `scalar0`, `scalar1`, and every `scalar_args`
entry are resolved while building the host task descriptor.
Graph tasks may also pass tensor-tile descriptor fields: `rows`, `cols`,
`inner`, `lda`, `ldb`, `ldc`, `a_batch_stride`, `b_batch_stride`, and
`out_batch_stride`. Use this when the explicit graph descriptor should run a
scalar tiled-GEMM task before downstream residual, gate, and fan-in tasks.
Graph tasks with `func_id=10` run the WMMA tensor-core generated-dispatch
body and must pass a compatible descriptor: `rows=16`, `cols=16`, and
`inner` divisible by `8`.
If every graph task omits `dependents`, the SceneTestCase CUDA adapter infers
task edges from tensor flow: reads bind to the nearest previous producer for
that tensor name, or to a later producer when the descriptor is intentionally
out of topological order. Use this form when testing the first step toward
PTO-style dependency inference while still providing an explicit descriptor.
Graph tasks may also spell incoming task edges as `depends_on` or
`dependencies`. Those fields are lowered into the same flattened dependent
array as explicit outgoing `dependents`, but they do not require the consumer's
tensor arguments to read producer output buffers. Use this form when checking
graph construction that is closer to normal task-graph dependency notation.
Graph tasks whose `out` names are not existing input/output tensors are
allocated as temporary buffers automatically, so tests only need an explicit
`temporaries` map when a temporary needs a size different from the output
tensor size.
Run the tagged graph SceneTestCase path after changing this lowering:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k tagged_graph --platform cuda
```

Run the tagged `inout` graph selector after changing dependency inference for
duplicate logical tensor producers or in-place graph updates:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'tagged_inout_graph or role_keyed_inout_graph' --platform cuda
```

Run the named-callable graph selector after changing callable-name resolution
or the tagged task-argument graph lowering:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k named_callable_graph_with_ctypes_data --platform cuda
```

Use the paired smoke runner for A100/H200 artifact evidence of the same
tagged `inout` graph-descriptor shape:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_tagged_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree
```

Use `graph_descriptor_role_keyed_inout` when the paired smoke report must
prove that the descriptor came from the preferred `role` task-arg spelling,
not only the compatibility `tag` spelling. The validator requires
`graph_task_arg_key=role` in both A100 and H200 JSON payloads:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_role_keyed_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/role-keyed-inout-working
```

Use `graph_descriptor_compact_role_inout` after changing compact role-entry
lowering or report metadata. It validates the same in-place graph shape, but
requires `graph_task_arg_key=compact` in the paired JSON plus Markdown/SVG
reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor_compact_role_inout --task-count 3 \
    --queue-capacity 2 --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-inout-working
```

The selected benchmark path also includes
`pto_persistent_dag_graph_role_keyed_inout` and
`pto_persistent_dag_graph_compact_role_inout`. Use a compact paired capture
when changing graph task-argument lowering or capture validation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --worker-blocks-per-task '' --sync-remote-tree \
    --output-root tmp/cuda-backend/compact-role-benchmark-working
```

Graph tasks may also pass `out_storage` when the logical graph output should
reuse an existing scratch buffer. Keep `out` unique for tensor-flow dependency
inference and set `out_storage` to the physical buffer name, for example
`out="tmp4", out_storage="tmp0"`. Unknown `out_storage` names should fail
during descriptor construction; use plain `out` for graph tasks that create a
new default-sized temporary.
Run the no-torch graph tensor-tile ctypes scene on A100 or H200 with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k graph_tensor_tile_with_ctypes_data --platform cuda
```

Run the no-torch graph tensor-core ctypes scene after changing graph
descriptor tensor-core lowering or validation:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k graph_tensor_core_tile_with_ctypes_data --platform cuda
```

Run the graph scalar-name ctypes scene after changing graph descriptor scalar
lowering:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k 'graph_scalar_scale or scalar_field_names' --platform cuda
```

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
- `pto_persistent_dag_graph_depends_on`: generated-dispatch DAG using an
  explicit graph descriptor whose edges come from incoming `depends_on`
  metadata while the final consumer reads original input tensors.
- `pto_persistent_dag_graph_chain`: five-task generated-dispatch DAG using
  an explicit graph descriptor with the same chain dependency shape as
  `pto_persistent_dag_chain`.
- `pto_persistent_dag_graph_scratch_reuse`: six-task generated-dispatch DAG
  using an explicit graph descriptor with the same scratch-buffer reuse shape
  as `pto_persistent_dag_reuse`.
- `pto_persistent_dag_graph_diamond`: five-task generated-dispatch DAG using
  an explicit graph descriptor with two roots, two fan-out consumers, and a
  final join.
- `pto_persistent_dag_graph_tagged`: three-task generated-dispatch DAG using
  explicit graph task-argument tags for `input`, `output`,
  `output_existing`, and scalar inputs.
- `pto_persistent_dag_graph_tagged_inout`: three-task generated-dispatch DAG
  using explicit graph task-argument tags, including an `inout` producer.
- `pto_persistent_dag_graph_triad`: three-task generated-dispatch DAG using
  an explicit graph descriptor for the fixed third tensor pointer field.
- `pto_persistent_dag_graph_quad`: three-task generated-dispatch DAG using an
  explicit graph descriptor for fixed third and fourth tensor pointer fields.
- `pto_persistent_dag_unary_square`: generated-dispatch DAG with a one-input
  square task body, validating unary persistent DAG arguments in the
  benchmark path.
- `pto_persistent_dag_tensor`: four-task generated-dispatch DAG with a tiled
  GEMM task followed by elementwise residual, gate, and fan-in tasks. The
  benchmark uses the default 16x16x16 descriptor unless
  `--tensor-rows`, `--tensor-cols`, and `--tensor-inner` are supplied.
- `pto_persistent_dag_graph_tensor`: four-task generated-dispatch DAG using
  an explicit runtime graph descriptor for the same tiled GEMM, residual,
  gate, and fan-in task sequence as `pto_persistent_dag_tensor`.
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
- `cublas_sgemm_graph`: CUDA Runtime API plus cuBLAS captured into a CUDA
  Graph. It uses the same tensor descriptor as `cublas_sgemm`, instantiates
  the graph outside the measured interval, warms graph replay once, then times
  `cudaGraphLaunch` with CUDA events.
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
merges the two JSON reports, validates the combined artifact, and refreshes
`tmp/cuda-backend/index.md`.
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

Use `--dry-run` to print the commands without launching benchmarks. The paired
benchmark default tensor descriptor is `16x16x16` so the scalar tensor DAG,
explicit graph tensor DAG, WMMA tensor-core DAG, and cuBLAS rows can run
together. The current committed summary keeps the full `61cf96cd` capture plus
compact current-head gates in `docs/nvidia-backend/evaluation-current.md`.

Use this compact paired gate after changing selected persistent graph
benchmark rows. It validates 84 samples across A100 and H200, including
`pto_persistent_dag_graph_depends_on`, `pto_persistent_dag_graph_triad`,
`pto_persistent_dag_graph_quad`, and
`pto_persistent_dag_graph_compact_role_inout` with dispatch `1,2,1`,
`6,2,1`, `8,2,1`, and `1,1,1`. The depends-on row requires graph fan-in
`0,0,2` and dependents `2,2`; the compact role row requires graph fan-in
`0,1,1`, dependents `1,2`, and `graph_task_arg_key=compact`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree \
    --output-root tmp/cuda-backend/graph-tensor-arity-benchmark-working
```

For a lighter no-torch real-data check, run the paired Worker smoke instead of
the full benchmark:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py \
    --op generic_args --sync-remote-tree --build-runtime
```

It mirrors the benchmark runner's remote refresh, `--skip-remote-refresh`,
`--sync-remote-tree`, and `--dry-run` controls. It also supports
`--build-runtime` for source changes under `src/cuda/`. It captures
host-schedule Worker smoke JSON on A100/H200, renders a compact smoke report,
and refreshes the artifact index.

Use `--op generic_args4` with the same command when validating the four-slot
generic host-schedule ABI.

The script writes:

- `cuda-benchmark.json`: raw samples, metadata, hardware, git commit.
- `cuda-benchmark.md`: short report with interpretation notes and graph
  descriptor metadata when explicit graph rows are present.
- `cuda-benchmark.svg`: bar chart of median device time by baseline, with
  graph topology and task-argument metadata embedded for explicit graph rows.
- `cuda-benchmark-ratios.svg`: bar chart of each row's device-time ratio
  against its matched reference.
- `cuda-benchmark-dag-deltas.svg`: bar chart of each `pto_persistent_dag_*`
  row's device-time increment over the matched `pto_persistent_dag` row.
- `cuda-benchmark-throughput.svg`: bar chart of median GF/s for tensor-DAG
  and cuBLAS rows that record a tensor tile descriptor.

For tensor-DAG and tensor-library experiments, pass `--tensor-rows`,
`--tensor-cols`, and `--tensor-inner` to the benchmark script. These flags
affect `pto_persistent_dag_tensor`, `pto_persistent_dag_tensor_core`,
`pto_persistent_dag_graph_tensor`, `cublas_sgemm`, and
`cublas_sgemm_graph`; other baselines keep their normal vector-add work. The
generated Markdown report records the descriptor as `rows x cols x inner`.

Use `--single-baseline pto_persistent_dag_graph_tensor` for a quick
benchmark path check of the explicit graph tensor-tile DAG on one GPU:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_tensor \
    --sizes 512 --repeats 1 --arch compute_80 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16
```

The current A100/H200 sample report for that row is under
`tmp/cuda-backend/combined-graph-tensor-current-working/`. It validates
source-paper metadata, sanitized command examples, report files, dispatch
`3,1,2,1`, tensor descriptor `16x16x16`, and zero scheduler errors.

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

Use `--single-baseline pto_persistent_dag_graph_generic_args4` for the
explicit graph-descriptor path that also carries all four generic tensor and
scalar slots. The current A100/H200 quick capture is under
`tmp/cuda-backend/persistent-graph-generic-args4-baseline-working/`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_generic_args4 \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph_depends_on` for a quick
benchmark path check of incoming-edge graph notation. This row uses dispatch
`1,2,1`, graph fan-in `0,0,2`, and graph dependents `2,2`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_depends_on \
    --sizes 4096 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph_diamond` for a quick
benchmark path check of the wider explicit graph descriptor with dispatch
`9,2,1,2,1`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_diamond \
    --sizes 1024 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph_chain` for a quick benchmark
path check of the explicit five-task chain graph descriptor with dispatch
`1,2,1,2,1`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_chain \
    --sizes 1024 --arch compute_80
```

The current compact paired benchmark capture with this row is under
`tmp/cuda-backend/combined-current-06b8c0c6/`.

Use `--single-baseline pto_persistent_dag_graph_scratch_reuse` for a quick
benchmark path check of the explicit six-task scratch-reuse graph descriptor
with dispatch `1,2,1,2,1,1`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_scratch_reuse \
    --sizes 1024 --arch compute_80
```

The historical compact paired benchmark capture with this row is under
`tmp/cuda-backend/combined-current-dbb01406/`. The current compact selected
baseline gate is under
`tmp/cuda-backend/graph-unary-benchmark-working/combined-current-f074746a/`
and also includes the graph tensor-core, tagged scalar graph, and graph
unary-square rows. It uses `N=1024`, one repeat,
`batch_tasks=2`, `worker_blocks_per_task=4`, validates source-paper
provenance and zero scheduler errors, requires
`scratch_reuse=reused_buffer=tmp0,reuse_task=4`, and includes Markdown plus
SVG reports.

The compact-role selected benchmark row is also covered by a no-batch paired
A100/H200 capture under
`tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f/`.
It validates 74 samples, including dispatch `1,1,1`, fan-in `0,1,1`,
dependents `1,2`, task args `input:a,input:b,output:tmp1`,
`inout:tmp1,input:b`, `input:tmp1,input:a,output_existing:out`, and
`graph_task_arg_key=compact`.

Use `--single-baseline pto_persistent_dag_graph_tagged` for a quick benchmark
path check of explicit graph task-argument tags that include scalar inputs.
This path validates `input`, `output`, `output_existing`, and scalar mappings
through the persistent DAG benchmark row with dispatch `9,2,1`.
Paired-current benchmark validation now requires this row to carry
`scalar:scalar_args[0]` and `scalar:scalar_args[1]` in
`graph_task_args`, so a numerically passing row that silently loses scalar
roles is rejected before docs are refreshed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_tagged \
    --sizes 1024 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph_tagged_inout` for a quick
benchmark path check of explicit graph task-argument tags. This path validates
`input`, `output`, `inout`, and `output_existing` mappings through the
persistent DAG benchmark row with dispatch `1,1,1`. Paired-current benchmark
validation now requires the tagged row to carry
`task1=inout:tmp1,input:b` in `graph_task_args`, so a numerically passing row
that silently loses the inout tag is rejected before docs are refreshed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_tagged_inout \
    --sizes 1024 --arch compute_80
```

Use the graph triad and graph quad single-baseline rows for quick one-GPU
checks of the explicit graph-descriptor path that carries fixed third and
fourth tensor pointer fields:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_triad \
    --sizes 1024 --repeats 1 --arch compute_80

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_quad \
    --sizes 1024 --repeats 1 --arch compute_80
```

Use `--single-baseline pto_persistent_dag_graph_unary_square` for a quick
benchmark path check of the explicit graph-descriptor path that carries the
same one-input square task body as `pto_persistent_dag_unary_square`. This row
validates dispatch `7,1,1`, graph fan-in `0,1,1`, and dependents `1,2`:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_unary_square \
    --sizes 1024 --repeats 1 --arch compute_80
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

Use `--single-baseline pto_host_schedule_generic_args` for a quick benchmark
path check of the host-schedule generic tensor/scalar argument packet:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_host_schedule_generic_args \
    --sizes 4096 --arch compute_80
```

Use `cuda_tensor_shape_sweep.py` to run paired A100/H200
samples over model-shaped tensor tile descriptors. By default it runs
`pto_persistent_dag_tensor`; pass `--baselines` to include
`pto_persistent_dag_graph_tensor`, `pto_persistent_dag_tensor_core`, and
`pto_persistent_dag_graph_tensor_core`, plus
`cublas_sgemm,cublas_sgemm_graph` for a scalar-vs-explicit-graph-vs-WMMA-vs-
library-vs-CUDA-Graph comparison on compatible descriptors. Pass `--sizes`
when the same baseline/shape set should be swept across multiple problem
sizes. Treat the scalar tiled GEMM rows as shape and scheduler evidence rather
than tensor-core throughput evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --shapes 8x4x12,16x16x64,32x16x64 --n 4096 --repeats 3 \
    --sync-remote-tree
```

Run a compact tensor-baseline comparison sweep with shapes compatible with the
current WMMA task. Use `32x16x16` or another `16`-multiple row/column shape
when the WMMA task should execute multiple output fragments per descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_graph_tensor,pto_persistent_dag_tensor_core,pto_persistent_dag_graph_tensor_core,cublas_sgemm,cublas_sgemm_graph \
    --shapes 16x16x16,16x16x64 --n 256 --repeats 3 \
    --sync-remote-tree
```

Run a size sweep for the same descriptor family when launch-dominated compact
rows need to be compared with larger repeated tensor work:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_graph_tensor,pto_persistent_dag_tensor_core,pto_persistent_dag_graph_tensor_core,cublas_sgemm,cublas_sgemm_graph \
    --shapes 16x16x16 --sizes 256,4096,65536 --repeats 3 \
    --sync-remote-tree
```

The sweep writes `cuda-tensor-shape-sweep.json`,
`cuda-tensor-shape-sweep.md`, `cuda-tensor-shape-sweep.svg`, and
`cuda-tensor-shape-throughput.svg` under
`tmp/cuda-backend/tensor-shape-sweep-<commit>/`. The Markdown keeps raw
repeat rows plus a median summary table with normalized GFLOP/s; the SVG
files plot median device time and median GFLOP/s per GPU/N/shape/baseline
with sample counts. The JSON metadata also records sanitized local and remote
sample command examples so the selected baseline/shape/size setup can be
reconstructed without rerunning the sweep. Publish-time source-paper
validation checks that the referenced VDCores and MPK notes exist under
`tmp/sources/`. The compact validation preset also requires the Markdown
median summary and throughput SVG to expose `Median GF/s`, each required
baseline, and each required tensor shape.

Regenerate reports from an existing tensor-sweep JSON without rerunning the
A100/H200 measurements:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --render-json tmp/cuda-backend/tensor-shape-sweep-<commit>/cuda-tensor-shape-sweep.json
```

Validate the compact tensor-baseline sweep before copying numbers into docs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/tensor-shape-sweep-<commit>/cuda-tensor-shape-sweep.json \
    --preset compact-tensor-baselines \
    --require-command-examples \
    --require-source-papers
```

The current validated compact tensor-baseline artifact is:

```text
tmp/cuda-backend/tensor-sweep-current-working/tensor-shape-sweep-219042f5/
```

Validate a size sweep by spelling out the required sizes and result count:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_validate_tensor_sweep.py \
    tmp/cuda-backend/tensor-shape-sweep-<commit>/cuda-tensor-shape-sweep.json \
    --require-artifact a100 --require-artifact h200 \
    --require-baseline pto_persistent_dag_tensor \
    --require-baseline pto_persistent_dag_graph_tensor \
    --require-baseline pto_persistent_dag_tensor_core \
    --require-baseline pto_persistent_dag_graph_tensor_core \
    --require-baseline cublas_sgemm \
    --require-baseline cublas_sgemm_graph \
    --require-size 256 --require-size 4096 --require-size 65536 \
    --require-shape 16x16x16 --expected-repeats 3 \
    --expected-result-count 108 --require-report-files \
    --require-report-throughput \
    --require-command-examples \
    --require-source-papers \
    --require-dispatch pto_persistent_dag_tensor=3,1,2,1 \
    --require-dispatch pto_persistent_dag_graph_tensor=3,1,2,1 \
    --require-dispatch pto_persistent_dag_tensor_core=10,1,2,1 \
    --require-dispatch pto_persistent_dag_graph_tensor_core=10,1,2,1
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

Use `--single-baseline pto_persistent_dag_graph_tensor_core` for the matching
explicit graph descriptor path with a WMMA first task:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline pto_persistent_dag_graph_tensor_core \
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

Use `--single-baseline cublas_sgemm_graph` for the matching CUDA Graph replay
baseline around a warmed cuBLAS descriptor:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --single-baseline cublas_sgemm_graph \
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
`cuda-tensor-shape-sweep.json` files, lifecycle matrix
`cuda-lifecycle-matrix.json` files, and smoke `cuda-smoke-report.md`
directories, then writes `tmp/cuda-backend/index.md` with each artifact's
kind, metadata, baselines or lifecycle scenarios, vector sizes, tensor-tile
descriptor shapes, persistent smoke modes, dispatch sequences, scheduler
error counters, repeat-run counts, per-launch completion counts, graph
descriptor fan-in/dependent arrays, graph task-argument keys, graph
task-argument metadata, tensor-sweep source-paper IDs, tensor-sweep
command-example presence, lifecycle-matrix source-paper IDs,
lifecycle-matrix collection mode, lifecycle-matrix command-example presence,
named scheduler error codes, and generated report/chart presence. It is a
local audit aid under `tmp/`; do not commit it with raw benchmark, tensor
sweep, lifecycle matrix, or smoke data.

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
    tmp/cuda-backend/combined-current-dbb01406/cuda-benchmark.json
```

Use `--section launch`, `--section unary-square`, `--section worker-grid`,
`--section dag-shapes`, `--section graph-metadata`,
`--section tensor-throughput`, or `--section tensor-sweep` to refresh only one
table. The tensor-throughput and tensor-sweep sections summarize selected
tensor-DAG, graph-tensor, fixed tensor-core, graph tensor-core, cuBLAS, and
cuBLAS Graph rows as median GF/s from recorded tensor descriptors. They also
keep graph tensor-core-only working captures visible even when a scalar tensor
reference row is absent. The DAG-shapes section includes explicit graph
depends-on and scratch-reuse rows when the capture has
`pto_persistent_dag_graph_depends_on` or
`pto_persistent_dag_graph_scratch_reuse`, avoiding hand-calculated
current-evaluation ratios from raw JSON. The graph-metadata section lists
explicit graph descriptor dispatch IDs, fan-in/dependent arrays, task-argument
keying, tagged task arguments, and tensor-tile shape per GPU. The
graph-role-spelling section focuses on tagged, role-keyed, and compact
role-entry graph rows when checking TaskArgs spelling compatibility. The
full benchmark Markdown report also includes the same `Graph Role Spelling
Rows` section, and `cuda-benchmark.svg` carries matching `graph role
spelling:` metadata in its `<desc>` element:

```bash
PYTHONPATH=$PWD:$PWD/python:.agents/skills/cuda-backend-eval/scripts \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py \
    tmp/cuda-backend/compact-role-benchmark-working/combined-current-30a8974f/cuda-benchmark.json \
    --section graph-role-spelling
```

Render the compact tensor-baseline sweep table directly from its raw sweep
JSON:

```bash
PYTHONPATH=$PWD:$PWD/python:.agents/skills/cuda-backend-eval/scripts \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py \
    tmp/cuda-backend/tensor-sweep-current-working/tensor-shape-sweep-219042f5/cuda-tensor-shape-sweep.json \
    --section tensor-sweep
```

New benchmark and tensor-sweep artifacts include source-paper provenance for
the VDCores and MPK notes in `tmp/sources/`, sanitized local/remote command
examples, plus the paper-alignment statement for the current microbenchmark
setup. Tensor sweeps also include one workload description per selected
baseline.

Validate the paired-current capture before copying numbers into docs:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/<combined-capture>/cuda-benchmark.json \
    --preset compact-current
```

The compact current-head gate checks the expected A100/H200 machines,
selected tensor baselines, the graph tensor-core baseline, the host-schedule
generic-args baseline, graph generic-args4 baseline, graph-chain baseline,
graph-depends-on baseline, graph-scratch-reuse baseline,
graph-tagged-inout baseline, graph descriptor fan-in/dependent metadata,
graph-triad and graph-quad baselines, the tagged scalar graph baseline, the
graph unary-square baseline, task-argument tags, visible Markdown/SVG graph
topology and task-argument metadata, visible Markdown/SVG tensor throughput
rows for required tensor/cuBLAS descriptors, size `1024`, one repeat, `84`
combined samples, and the Markdown/SVG report files. The current compact gate
artifact is under
`tmp/cuda-backend/tensor-throughput-gate-current-working/combined-current-a9d028de/`.
Validate older captures with explicit `--require-*` checks if the current
preset has gained new selected rows since that capture.
New paired-runner captures use a dynamic validator command because the
selected benchmark rows can change with runner flags.
`--require-command-examples` checks that
local and remote sample commands are reconstructable without local checkout
paths. `--require-source-papers` checks that the report records the
VDCores/MPK source IDs and that the referenced files exist under
`tmp/sources/`. `--require-zero-scheduler-errors` checks that PTO persistent
DAG rows include device scheduler counters and that each counter set is zero.
`--require-dispatch` checks that generated-dispatch benchmark rows ran the
expected device `func_id` sequence. The paired benchmark runner now adds these
dispatch requirements automatically for known persistent DAG baselines,
including scalar, graph-descriptor, tensor-tile, and tensor-core rows.
`--require-tensor-tile` checks that tensor and cuBLAS benchmark rows recorded
the requested tensor descriptor shape. The paired benchmark runner adds these
requirements automatically from `--tensor-rows`, `--tensor-cols`, and
`--tensor-inner`.
`--require-graph-fanin` and `--require-graph-dependents` check that explicit
runtime graph descriptor rows recorded the expected dependency shape. The
paired benchmark runner adds these requirements automatically for known graph
descriptor benchmark baselines.
It also passes `--require-report-graph-topology` and
`--require-report-graph-task-args`. Current paired presets also pass
`--require-report-graph-role-spelling`, so paired benchmark Markdown and SVG
reports must show the same graph topology, TaskArgs-like metadata, and
focused tag/role/compact graph task-argument spelling rows that the JSON
payload carries. They also pass `--require-report-tensor-throughput`, so
`cuda-benchmark.md` and `cuda-benchmark-throughput.svg` must show the required
tensor/core/cuBLAS baselines and the requested tensor tile shape before the
capture is publishable.

Use `cuda_validate_smoke.py` for paired smoke artifacts. It checks required
artifacts, pass status, zero device scheduler errors, expected runtime/mode,
dispatch IDs, repeat-run lifecycle counts, tensor-tile descriptor shape when
requested, graph-descriptor fan-in/dependent metadata when requested,
`graph_task_args` metadata when requested, generated smoke report files,
visible report graph topology when requested, and visible report graph
task-argument metadata when requested.
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
Tensor and cuBLAS rows also produce a `Tensor Throughput Rows` table and
`cuda-benchmark-throughput.svg`. That view normalizes the recorded
`rows x cols x inner` descriptor and tile count into median GF/s, so use it
when comparing selected tensor baselines by useful arithmetic rather than raw
launch time.

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
    --stream-pool-size 6 \
    --label a100-streams --output-dir tmp/cuda-backend/a100-streams
```

For remote H200, use `--arch compute_90`. Use `--stream-pool-size` to record
the host runtime stream pool size in JSON, Markdown, and SVG report artifacts.
The scripts discover `nvcc` from `CUDA_HOME`, `CUDA_PATH`, `PATH`, and common
`/usr/local/cuda*` toolkit paths. If `nvcc` is still unavailable, the script
uses an embedded `sm_80` PTX fallback that the H200 driver JITs.

The stream report compares `pto_stream_parallel` against `pto_stream_serial`.
The current A100/H200 capture at `37bebf44` shows about `0.51x` parallel-vs-
serial wall time on both machines, supporting multiple streams for the
host-schedule runtime when independent callables are launched from separate
host threads.
The stream-pool-size capture under
`tmp/cuda-backend/stream-pool6-working/` uses `--stream-pool-size 6`; it
reported parallel/serial ratios of `0.51x` on A100 and `0.48x` on H200.

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

If non-interactive SSH does not find `nvcc`, export the H200 toolkit path
before running CUDA pytest selectors:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
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
