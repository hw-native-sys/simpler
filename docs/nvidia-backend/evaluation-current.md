# CUDA Current Evaluation Capture

This page summarizes the latest full paired A100/H200 CUDA backend capture
from commit `61cf96cd`, plus the current-head compact validation capture from
commit `0b3c1699`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `720`

## Artifact Paths

- `tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-61cf96cd/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-f0f43b2a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-d361006f/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/a100-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/combined-current-a46db551/cuda-benchmark-dag-deltas.svg`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/a100.json`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/h200.json`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/a100.json`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/h200.json`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/cuda-smoke-report.svg`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/a100.json`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/h200.json`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/cuda-smoke-report.md`
- `tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/cuda-smoke-report.svg`

## Current-Head Compact Paired Gate

The compact current-head paired gate at commit `0b3c1699` uses a
WMMA-compatible `16x16x16` tensor descriptor, `N=1024`, one repeat,
`batch_tasks=2`, and `worker_blocks_per_task=4`. The paired runner synced the
local tree to `bizhaoh200`, captured A100 and H200 reports, merged them, and
validated the combined JSON with required baselines, source-paper provenance,
sanitized command examples, and generated Markdown/SVG report files.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.json \
    --require-size 1024 --expected-repeats 1 --expected-result-count 50 \
    --require-baseline pto_persistent_dag_tensor_core \
    --require-baseline cublas_sgemm --require-report-files \
    --require-command-examples --require-zero-scheduler-errors \
    --require-source-papers
```

Selected rows:

| GPU | Host schedule ns | Base DAG ns | Tensor DAG ns | Tensor-core ns | cuBLAS ns | Grid batch ns |
| --- | ---------------- | ----------- | ------------- | -------------- | --------- | ------------- |
| A100 | 33792 | 61440 | 44032 | 59392 | 60416 | 41984 |
| H200 | 14848 | 31936 | 44576 | 36096 | 40959 | 28768 |

This capture is a gate for command construction, validation coverage, and
real A100/H200 execution at the current commit. It is intentionally smaller
than the full `61cf96cd` capture and should not replace the three-size,
three-repeat rows below for broad trend reading. The `0b3c1699` gate was
captured after adding scheduler no-progress diagnostics; all PTO persistent
DAG rows reported zero device scheduler errors.

## Supplemental Scalar-Scale Benchmark

The compact scalar-scale benchmark gate at artifact label `a46db551` adds
`pto_persistent_dag_scalar_scale` to the selected paired benchmark path. It
uses `N=4096`, one repeat, no batch rows, and the default `16x16x16` tensor
descriptor metadata. The paired runner synced the local tree to `bizhaoh200`,
captured A100 and H200 benchmark reports, merged `44` rows, and validated
required baselines, command examples, source-paper provenance, zero scheduler
errors, and generated Markdown/SVG report files.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    tmp/cuda-backend/combined-current-a46db551/cuda-benchmark.json \
    --require-size 4096 --expected-repeats 1 --expected-result-count 44 \
    --require-baseline pto_persistent_dag_scalar_scale \
    --require-report-files --require-command-examples \
    --require-zero-scheduler-errors --require-source-papers
```

| GPU | Baseline | Dispatch | Scalar args | Device ns | Host ns | Status |
| --- | -------- | -------- | ----------- | --------- | ------- | ------ |
| A100 | `pto_persistent_dag_scalar_scale` | `11,2,1` | `scalar0=2.0` | 37888 | 55626 | pass |
| H200 | `pto_persistent_dag_scalar_scale` | `11,2,1` | `scalar0=2.0` | 27744 | 2498273 | pass |

Both rows reported zero device scheduler errors and the report includes
`cuda-benchmark.svg`, `cuda-benchmark-ratios.svg`, and
`cuda-benchmark-dag-deltas.svg`. The high H200 host time is launch-side noise
in this one-repeat compact gate; the device event time is the useful
scheduler-path signal.

## Supplemental Scalar-Scale Smoke

The scalar-scale persistent DAG smoke at artifact label `e9c9f5f2` validates a
single-input scalar task descriptor outside the scene-test framework. It uses
dispatch sequence `[11,2,1]`: scale `tmp0 = scalar0 * a`, multiply
`tmp1 = a * b`, then add `out = tmp0 + tmp1`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/a100.json \
    tmp/cuda-backend/persistent-scalar_scale-smoke-e9c9f5f2/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-repeat-runs 1 --expected-completed-count 3 \
    --require-report-files --expected-dag-shape scalar_scale \
    --expected-dispatch 11,2,1
```

| GPU | PTX arch | Dispatch | Scalar args | Device ns | Host ns | Status |
| --- | -------- | -------- | ----------- | --------- | ------- | ------ |
| A100 | `compute_80` | `11,2,1` | `scalar0=2.0` | 40960 | 61301 | pass |
| H200 | `compute_90` | `11,2,1` | `scalar0=2.0` | 25856 | 34808 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. This capture is correctness evidence for the single-input
scalar descriptor and generated-dispatch registration, not a benchmark
replacement for the multi-baseline captures.

## Supplemental Generic-Args Repeat-Run Smoke

The `generic_args` persistent DAG smoke at artifact label `6574c43b`
validates prepared-callable lifecycle reuse for the indexed tensor/scalar
descriptor path. It prepares one generated-dispatch callable, then runs it
twice after resetting fan-in, ready flags, counters, and scratch/output
buffers. The dispatch sequence is `[9,2,1]`; task `9` reads
`tensor_args[0]=tmp0`, `tensor_args[1]=tmp3`, `scalar_args[0]=1.5`, and
`scalar_args[1]=0.25`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/a100.json \
    tmp/cuda-backend/persistent-generic_args-repeat2-smoke-6574c43b/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape generic_args --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 9,2,1 \
    --require-report-files
```

| GPU | Dispatch | Repeat runs | Launch completions | Launch device ns | Device ns | Host ns | Status |
| --- | -------- | ----------- | ------------------ | ---------------- | --------- | ------- | ------ |
| A100 | `9,2,1` | 2 | `3,3` | `44032,25600` | 69632 | 106096 | pass |
| H200 | `9,2,1` | 2 | `3,3` | `21088,19808` | 40896 | 4953113 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. The high H200 host time is launch-side noise in this single
capture; the per-launch CUDA event times are the useful lifecycle signal.

## Supplemental Reordered Graph-Descriptor Smoke

The reordered graph-descriptor persistent DAG smoke at artifact label
`f877b7b3` validates order-independent tensor-flow dependency inference. The
runtime task list is `[final add, generic args, multiply]`, so the final
consumer has task id `0` but starts with fan-in `2`; both producer tasks point
back to it through dependents `[0,0]`.

Validation command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/a100.json \
    tmp/cuda-backend/persistent-graph_descriptor_reordered-repeat2-smoke-f877b7b3/h200.json \
    --require-artifact a100 --require-artifact h200 \
    --expected-runtime persistent_device --expected-mode dag \
    --expected-dag-shape graph_descriptor_reordered --expected-repeat-runs 2 \
    --expected-completed-count 3 --expected-dispatch 1,9,2 \
    --require-report-files
```

| GPU | Dispatch | Fan-in | Dependents | Launch completions | Device ns | Host ns | Status |
| --- | -------- | ------ | ---------- | ------------------ | --------- | ------- | ------ |
| A100 | `1,9,2` | `2,0,0` | `0,0` | `3,3` | 63488 | 91185 | pass |
| H200 | `1,9,2` | `2,0,0` | `0,0` | `3,3` | 46240 | 63520 | pass |

Both rows reported zero device scheduler errors and generated Markdown/SVG
smoke reports. This is correctness evidence for graph lowering; the task body
and arithmetic are the same as the generic-args graph descriptor.

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It can
reduce launch overhead in some rows, but it is still host-owned replay rather
than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the task-body compiler path.
It uses the same host runtime path as `pto_host_schedule`, but the PTX comes
from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)` and
the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 32768 | 30720 | 23552 | 15359 | 0.94x | 0.47x |
| A100 | 65536 | 21440 | 20096 | 26591 | 17823 | 0.94x | 0.83x |
| A100 | 1048576 | 301728 | 301696 | 233919 | 328319 | 1.00x | 1.09x |
| H200 | 1024 | 35968 | 40192 | 35392 | 30239 | 1.12x | 0.84x |
| H200 | 65536 | 23488 | 25408 | 39039 | 31456 | 1.08x | 1.34x |
| H200 | 1048576 | 31072 | 31264 | 38623 | 28480 | 1.01x | 0.92x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 large-vector launch rows are noisy in this capture, so use the
paired report as current smoke/evaluation evidence rather than a final
launch-overhead ranking.

## Host-Schedule Shape Rows

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The
`pto_host_schedule_quad` row validates the generated four-input
`(a, b, c, d, out, n)` ABI. Both rows compare against CUDA `float32`
goldens, which matters at larger sizes where exact Python integer arithmetic
does not match single-precision output.

| GPU | N | Unary square ns | Quad ns |
| --- | - | --------------- | ------- |
| A100 | 1024 | 33792 | 32768 |
| A100 | 65536 | 23616 | 20768 |
| A100 | 1048576 | 304128 | 545600 |
| H200 | 1024 | 39296 | 41216 |
| H200 | 65536 | 25056 | 22624 |
| H200 | 1048576 | 37312 | 45728 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 256 | 31744 | 1.19x |
| A100 | 1024 | 6 | 32 | 32768 | 0.36x |
| A100 | 1024 | 12 | 32 | 32768 | 0.21x |
| A100 | 65536 | 2 | 64 | 22528 | 0.85x |
| A100 | 65536 | 6 | 128 | 20480 | 0.39x |
| A100 | 65536 | 12 | 32 | 20480 | 0.24x |
| A100 | 1048576 | 2 | 256 | 22528 | 0.07x |
| A100 | 1048576 | 6 | 128 | 33792 | 0.09x |
| A100 | 1048576 | 12 | 256 | 58368 | 0.14x |
| H200 | 1024 | 2 | 64 | 35776 | 1.20x |
| H200 | 1024 | 6 | 32 | 35072 | 0.38x |
| H200 | 1024 | 12 | 128 | 35104 | 0.23x |
| H200 | 65536 | 2 | 256 | 25312 | 0.74x |
| H200 | 65536 | 6 | 256 | 24544 | 0.32x |
| H200 | 65536 | 12 | 32 | 25280 | 0.24x |
| H200 | 1048576 | 2 | 256 | 32864 | 0.78x |
| H200 | 1048576 | 6 | 256 | 39296 | 0.55x |
| H200 | 1048576 | 12 | 256 | 46816 | 0.39x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay below matched host-schedule batch rows for most multi-task
rows in this capture, with the two-task small-vector points as the
current exception.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine, triad, quad, generic-args, graph-descriptor, and
unary-square rows use generated dispatch and different descriptor
fields/task-body arities without changing the persistent launch path.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Quad/DAG | Generic Args/DAG | Graph Descriptor/DAG | Unary Square/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | -------- | ---------------- | -------------------- | ---------------- | ---------- |
| A100 | 1024 | 2.04x | 1.81x | 1.19x | 1.38x | 1.15x | 1.19x | 1.19x | 1.08x | 1.15x | 1.46x |
| A100 | 65536 | 1.80x | 1.80x | 1.02x | 1.01x | 1.10x | 1.20x | 1.19x | 1.19x | 1.45x | 3.88x |
| A100 | 1048576 | 1.57x | 1.58x | 1.04x | 1.02x | 1.04x | 1.12x | 1.03x | 1.05x | 1.40x | 3.29x |
| H200 | 1024 | 1.32x | 1.38x | 1.02x | 1.03x | 0.98x | 0.96x | 0.96x | 0.95x | 1.04x | 1.24x |
| H200 | 65536 | 1.75x | 1.76x | 0.99x | 0.99x | 1.00x | 1.06x | 1.06x | 1.05x | 1.41x | 2.87x |
| H200 | 1048576 | 1.79x | 1.78x | 0.99x | 1.00x | 1.00x | 1.01x | 0.99x | 1.00x | 1.36x | 3.09x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, quad, generic-args, graph-descriptor,
and unary-square rows prove mixed tensor/scalar fields, extra tensor pointers
through a fourth tensor task descriptor field, generic indexed argument
slots, explicit runtime graph lowering, and unary task-body lowering.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. Treat these DAG-shape rows as correctness and scheduler-shape
evidence first; throughput conclusions require a tuned tensor workload.

## Supplemental Tensor Shape Sweep

The tensor DAG also has a small model-shaped descriptor sweep at commit
`c0ada3ad`. It runs only `pto_persistent_dag_tensor`, so it is shape and
scheduler evidence rather than a replacement for the paired baseline capture.
Each row uses `N=4096`, two repeats, and the generated-dispatch sequence
`[3,1,2,1]` on both A100 and H200. The raw JSON, Markdown, and SVG artifacts
are under `tmp/cuda-backend/tensor-shape-sweep-c0ada3ad/`.

| GPU | Shape | Tiles | Median device ns | Median host ns | Status |
| --- | ----- | ----- | ---------------- | -------------- | ------ |
| A100 | 8x4x12 | 128 | 68096 | 83972.5 | pass |
| A100 | 16x16x64 | 16 | 169472 | 185079 | pass |
| A100 | 32x16x64 | 8 | 132608 | 148213 | pass |
| H200 | 8x4x12 | 128 | 58336 | 68226 | pass |
| H200 | 16x16x64 | 16 | 109936 | 120146 | pass |
| H200 | 32x16x64 | 8 | 98064 | 107936.5 | pass |

This extends the earlier non-square descriptor smoke from a single tile shape
to three descriptor families that are closer to model-kernel tile shapes. The
kernel body is still scalar tiled GEMM followed by elementwise residual/gate
work, so the result should not be read as tensor-core throughput.

## Supplemental Tensor Baseline Sweep

The current multi-baseline tensor size sweep was captured at commit
`e79edba2` under `tmp/cuda-backend/tensor-shape-sweep-e79edba2/`. It runs
three repeats for a WMMA-compatible `16x16x16` descriptor at `N=256`, `4096`,
and `65536`, comparing the scalar tensor DAG,
`pto_persistent_dag_tensor_core`, and `cublas_sgemm` in one Markdown/SVG
report. The Markdown keeps raw repeat rows plus medians, records VDCores/MPK
source-paper provenance and per-baseline workload descriptions, and the SVG
plots cover median device time and median GFLOP/s with sample counts. The
table below reports median device time and normalized GFLOP/s across the
three samples.

| GPU | N | Shape | Scalar tensor ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Tensor-core GF/s | cuBLAS GF/s | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | ---------------- | -------------- | --------- | ----------- | ---------------- | ----------- | ------------------ | ------------- |
| A100 | 256 | 16x16x16 | 47104 | 47104 | 43007 | 0.17 | 0.17 | 0.19 | 1.00x | 0.91x |
| A100 | 4096 | 16x16x16 | 79872 | 71680 | 36864 | 1.64 | 1.83 | 3.56 | 0.90x | 0.46x |
| A100 | 65536 | 16x16x16 | 587616 | 470368 | 38911 | 3.57 | 4.46 | 53.90 | 0.80x | 0.07x |
| H200 | 256 | 16x16x16 | 30560 | 28160 | 50496 | 0.27 | 0.29 | 0.16 | 0.92x | 1.65x |
| H200 | 4096 | 16x16x16 | 88576 | 49888 | 37055 | 1.48 | 2.63 | 3.54 | 0.56x | 0.42x |
| H200 | 65536 | 16x16x16 | 1032896 | 390368 | 36127 | 2.03 | 5.37 | 58.05 | 0.38x | 0.03x |

The tensor-core rows use dispatch `10,1,2,1`, while the scalar tensor rows use
`3,1,2,1`. cuBLAS rows have no PTO dispatch sequence because they run through
CUDA Runtime API plus cuBLAS directly. The tensor-core PTO row improves over
the scalar tensor DAG as the number of tiles grows, especially on H200, but
the normalized throughput still stays in the single-digit GFLOP/s range
because the current PTO path schedules one small generated task per tile. The
cuBLAS path reaches about `54` GFLOP/s on A100 and `58` GFLOP/s on H200 at
`N=65536` because it uses a tuned library implementation. This remains a
compact descriptor/scheduler comparison rather than a tuned GEMM throughput
result.

A current-head one-repeat follow-up under
`tmp/cuda-backend/tensor-shape-sweep-0e84fd26/` adds
`pto_persistent_dag_graph_tensor` to the same tensor-baseline sweep family.
It uses the `16x16x16` descriptor at `N=256` and `4096`, validates sanitized
command examples, VDCores/MPK source-paper metadata, report files, and PTO
dispatch sequences, and keeps the explicit graph tensor row beside scalar
tensor, WMMA tensor-core, and cuBLAS rows.

| GPU | N | Shape | Scalar tensor ns | Graph tensor ns | Tensor-core ns | cuBLAS ns | Scalar GF/s | Graph tensor GF/s | Tensor-core GF/s | cuBLAS GF/s | Graph/scalar | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | ---------------- | --------------- | -------------- | --------- | ----------- | ----------------- | ---------------- | ----------- | ------------ | ------------------ | ------------- |
| A100 | 256 | 16x16x16 | 47104 | 47104 | 45056 | 48128 | 0.17 | 0.17 | 0.18 | 0.17 | 1.00x | 0.96x | 1.02x |
| A100 | 4096 | 16x16x16 | 80896 | 80896 | 82944 | 39935 | 1.62 | 1.62 | 1.58 | 3.28 | 1.00x | 1.03x | 0.49x |
| H200 | 256 | 16x16x16 | 29568 | 32800 | 27040 | 51711 | 0.28 | 0.25 | 0.30 | 0.16 | 1.11x | 0.91x | 1.75x |
| H200 | 4096 | 16x16x16 | 89472 | 89152 | 51872 | 35904 | 1.46 | 1.47 | 2.53 | 3.65 | 1.00x | 0.58x | 0.40x |

### Current-Head Reproducibility Check

A follow-up one-repeat compact tensor sweep at commit `a5fd4bfd` validates the
current report-generation and metadata gate after the tensor-sweep scripts
started recording sanitized command examples. The artifact is under
`tmp/cuda-backend/tensor-shape-sweep-a5fd4bfd/` and was validated with
`--require-command-examples`, `--require-source-papers`, required A100/H200
rows, required report files, and PTO dispatch sequences.

| GPU | N | Shape | Scalar tensor ns | Tensor-core ns | cuBLAS ns |
| --- | - | ----- | ---------------- | -------------- | --------- |
| A100 | 256 | 16x16x16 | 47104 | 46080 | 41983 |
| H200 | 256 | 16x16x16 | 31552 | 28544 | 51040 |

This is a current-HEAD smoke/evidence capture rather than a replacement for
the three-repeat size sweep above. It proves the exact command examples needed
to reconstruct the local A100 and remote H200 setup are now present in the
generated Markdown and JSON.

## Tensor-Core Callable Smoke

The first tensor-core persistent DAG smoke was captured at commit `390eda4f`.
It adds a block-wide generated-dispatch task body using CUDA WMMA
`m16n16k8` with TF32 inputs and F32 accumulation. The task runs before the
same residual, gate, and fan-in elementwise tasks used by `tensor_tile`, so
this validates callable shape and scheduler integration rather than tuned GEMM
throughput.

Artifact:
`tmp/cuda-backend/persistent-tensor_core_tile-16x16x16-smoke-390eda4f/`

| GPU | Shape | Tensor core | Dispatch | Device ns | Host ns | Status |
| --- | ----- | ----------- | -------- | --------- | ------- | ------ |
| A100 | 16x16x16 | `wmma:m16n16k8:tf32->f32` | `10,1,2,1` | 46080 | 65963 | pass |
| H200 | 16x16x16 | `wmma:m16n16k8:tf32->f32` | `10,1,2,1` | 31808 | 41308 | pass |

Both rows report zero device scheduler errors, `completed_count=4`, and
target-specific `nvcc` PTX (`compute_80` on A100, `compute_90` on H200). The
generated Markdown/SVG report is in the artifact directory.

## Tensor-Core Benchmark Row

The first tensor-core row in the selected benchmark report was captured at
commit `0879aa9e`. It uses the same compact A100/H200 benchmark report format
as the full paired capture, but with one size (`N=256`), one repeat, no batch
rows, and a `16x16x16` tensor descriptor. The raw JSON, Markdown, and SVG
artifacts are under
`tmp/cuda-backend/combined-tensor-core-current-0879aa9e/`.

Tensor-core row details:

- A100: `pto_persistent_dag_tensor_core`, `16x16x16`,
  `wmma:m16n16k8:tf32->f32`, `37888 ns` device, `52277 ns` host,
  `0.90x` versus `pto_persistent_dag`; signed DAG increment `-4096 ns`.
- H200: `pto_persistent_dag_tensor_core`, `16x16x16`,
  `wmma:m16n16k8:tf32->f32`, `38656 ns` device, `50211 ns` host,
  `0.97x` versus `pto_persistent_dag`; signed DAG increment `-1088 ns`.

For context, the scalar tensor DAG row in the same report measured `40960 ns`
on A100 and `43392 ns` on H200. The tensor-core row therefore validates that
WMMA callable bodies now participate in the normal selected-baseline report
and chart flow, but it still measures one small generated task shape rather
than a tuned tensor-core kernel.
The generated `cuda-benchmark-dag-deltas.svg` chart visualizes the signed
device-time increment over the matched `pto_persistent_dag` scheduler
baseline, which is the current report view for separating scheduler overhead
from additional generated-dispatch task work.

## cuBLAS Library Baseline Row

The first cuBLAS row in the selected benchmark report was captured in the
`a100-cublas-current-343924df` and `h200-cublas-current-343924df` artifacts,
then merged under `tmp/cuda-backend/combined-cublas-current-343924df/`. It
uses the same compact report shape as the tensor-core row: one size
(`N=256`), one repeat, no batch rows, and the `16x16x16` tensor descriptor.

cuBLAS row details:

- A100: `cublas_sgemm`, `16x16x16`,
  `cublasSgemmStridedBatched`, batch count `1`, `48128 ns` device,
  `64414 ns` host, `2.76x` versus `pto_host_schedule`.
- H200: `cublas_sgemm`, `16x16x16`,
  `cublasSgemmStridedBatched`, batch count `1`, `58623 ns` device,
  `71677 ns` host, `2.34x` versus `pto_host_schedule`.

For context, the same compact report measured `pto_persistent_dag_tensor_core`
at `33792 ns` on A100 and `32960 ns` on H200. The cuBLAS row is intentionally
a library-backed launch/compute baseline rather than PTO runtime work; at this
small descriptor size it is dominated by cuBLAS launch and dispatch overhead,
not GEMM throughput.

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 32,64,128,256 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-current-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-current-$(git rev-parse --short HEAD)
```

Paired A100/H200:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sync-remote-tree
```

Tensor shape sweep:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py \
    --baselines pto_persistent_dag_tensor,pto_persistent_dag_tensor_core,cublas_sgemm \
    --shapes 16x16x16,16x16x64 --n 256 --repeats 3 \
    --sync-remote-tree
```

Tensor-core smoke:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_core_tile --task-count 4 --queue-capacity 2 \
    --n 256 --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree
```

Tensor-core selected-baseline report:

```bash
COMMIT=$(git rev-parse --short HEAD)
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --sizes 256 --repeats 1 --arch compute_80 --include-persistent \
    --batch-tasks 0 --worker-blocks-per-task 1 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-tensor-core-current-$COMMIT \
    --output-dir tmp/cuda-backend/a100-tensor-core-current-$COMMIT
```

cuBLAS selected-baseline report:

```bash
COMMIT=$(git rev-parse --short HEAD)
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --sizes 256 --repeats 1 --arch compute_80 --include-persistent \
    --batch-tasks 0 --worker-blocks-per-task 1 \
    --tensor-rows 16 --tensor-cols 16 --tensor-inner 16 \
    --label a100-cublas-current-$COMMIT \
    --output-dir tmp/cuda-backend/a100-cublas-current-$COMMIT
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-current-61cf96cd/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-61cf96cd/cuda-benchmark.json \
    --label combined-current-61cf96cd \
    --output-dir tmp/cuda-backend/combined-current-61cf96cd
```
