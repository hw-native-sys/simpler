# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `61cf96cd`. The raw JSON, Markdown, and SVG reports are generated
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

The current multi-baseline tensor shape sweep was captured at commit
`47d857e1` under `tmp/cuda-backend/tensor-shape-sweep-47d857e1/`. It runs
three repeats for two WMMA-compatible descriptors and compares the scalar
tensor DAG, `pto_persistent_dag_tensor_core`, and `cublas_sgemm` in one
Markdown/SVG report. The Markdown keeps raw repeat rows plus medians, while
the refreshed report also records VDCores/MPK source-paper provenance and
per-baseline workload descriptions. The SVG plots median device time with
sample counts. The summary helper includes the problem size so future
throughput-oriented sweeps can compare multiple `N` values without collapsing
rows. Each current row uses `N=256`; the table below reports median device
time across the three samples.

| GPU | N | Shape | Scalar tensor ns | Tensor-core ns | cuBLAS ns | Tensor-core/scalar | cuBLAS/scalar |
| --- | - | ----- | ---------------- | -------------- | --------- | ------------------ | ------------- |
| A100 | 256 | 16x16x16 | 47104 | 47104 | 73728 | 1.00x | 1.57x |
| A100 | 256 | 16x16x64 | 51200 | 50176 | 74752 | 0.98x | 1.46x |
| H200 | 256 | 16x16x16 | 31328 | 26848 | 51456 | 0.86x | 1.64x |
| H200 | 256 | 16x16x64 | 41088 | 34144 | 55711 | 0.83x | 1.36x |

The tensor-core rows use dispatch `10,1,2,1`, while the scalar tensor rows use
`3,1,2,1`. cuBLAS rows have no PTO dispatch sequence because they run through
CUDA Runtime API plus cuBLAS directly. At this small descriptor size, the
PTO persistent rows are competitive with or faster than the cuBLAS baseline,
but this remains a compact launch/scheduler comparison rather than a tuned
GEMM throughput result.

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
    --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
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
