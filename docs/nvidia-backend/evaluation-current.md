# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `c0dc1372`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `684`

## Artifact Paths

- `tmp/cuda-backend/a100-current-c0dc1372/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-c0dc1372/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-c0dc1372/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-c0dc1372/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-c0dc1372/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 7168 | 7168 | 8191 | 9216 | 1.00x | 1.29x |
| A100 | 65536 | 18688 | 18112 | 27103 | 191743 | 0.97x | 10.26x |
| A100 | 1048576 | 29184 | 1860224 | 585568 | 173600 | 63.74x | 5.95x |
| H200 | 1024 | 36640 | 37248 | 36639 | 27200 | 1.02x | 0.74x |
| H200 | 65536 | 22656 | 23648 | 38240 | 28384 | 1.04x | 1.25x |
| H200 | 1048576 | 18880 | 18784 | 24544 | 19424 | 0.99x | 1.03x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 large-vector compiler and CUDA Graph rows are noisy in this capture,
so use the paired report as current smoke/evaluation evidence rather than a
final launch-overhead ranking.

## Host-Schedule Shape Rows

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The
`pto_host_schedule_quad` row validates the generated four-input
`(a, b, c, d, out, n)` ABI. Both rows compare against CUDA `float32`
goldens, which matters at larger sizes where exact Python integer arithmetic
does not match single-precision output.

| GPU | N | Unary square ns | Quad ns |
| --- | - | --------------- | ------- |
| A100 | 1024 | 7168 | 8192 |
| A100 | 65536 | 31872 | 349984 |
| A100 | 1048576 | 26752 | 30784 |
| H200 | 1024 | 40320 | 39072 |
| H200 | 65536 | 25088 | 24896 |
| H200 | 1048576 | 18496 | 22208 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 32 | 7168 | 0.29x |
| A100 | 1024 | 6 | 32 | 8192 | 0.21x |
| A100 | 1024 | 12 | 32 | 8192 | 0.10x |
| A100 | 65536 | 2 | 128 | 9216 | 0.29x |
| A100 | 65536 | 6 | 128 | 9216 | 0.01x |
| A100 | 65536 | 12 | 64 | 11264 | 0.01x |
| A100 | 1048576 | 2 | 256 | 26624 | 0.86x |
| A100 | 1048576 | 6 | 128 | 34816 | 0.51x |
| A100 | 1048576 | 12 | 256 | 57344 | 0.42x |
| H200 | 1024 | 2 | 64 | 37056 | 0.76x |
| H200 | 1024 | 6 | 32 | 36960 | 0.37x |
| H200 | 1024 | 12 | 128 | 35936 | 0.22x |
| H200 | 65536 | 2 | 128 | 26400 | 0.89x |
| H200 | 65536 | 6 | 128 | 26176 | 0.33x |
| H200 | 65536 | 12 | 64 | 25824 | 0.23x |
| H200 | 1048576 | 2 | 256 | 19712 | 0.74x |
| H200 | 1048576 | 6 | 128 | 25760 | 0.43x |
| H200 | 1048576 | 12 | 256 | 38432 | 0.36x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay well below matched host-schedule batch rows for medium and
large vectors in this capture.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine, triad, quad, and unary-square rows use generated dispatch
and different descriptor fields/task-body arities without changing the
persistent launch path.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Quad/DAG | Unary Square/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | -------- | ---------------- | ---------- |
| A100 | 1024 | 1.92x | 1.40x | 0.96x | 1.20x | 1.12x | 1.08x | 1.20x | 1.52x |
| A100 | 65536 | 1.05x | 1.04x | 1.00x | 0.99x | 1.00x | 0.94x | 1.04x | 0.24x |
| A100 | 1048576 | 1.58x | 1.18x | 0.43x | 0.35x | 0.38x | 0.38x | 0.50x | 1.58x |
| H200 | 1024 | 1.33x | 1.35x | 1.04x | 1.09x | 0.97x | 0.97x | 1.08x | 1.19x |
| H200 | 65536 | 1.75x | 1.79x | 0.99x | 1.00x | 1.01x | 1.07x | 1.42x | 2.88x |
| H200 | 1048576 | 1.78x | 1.78x | 0.99x | 0.99x | 0.99x | 1.00x | 1.36x | 3.08x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, quad, and unary-square rows prove
mixed tensor/scalar fields, extra tensor pointers through a fourth tensor
task descriptor field, and unary task-body lowering.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. Treat these DAG-shape rows as correctness and scheduler-shape
evidence first; throughput conclusions require a tuned tensor workload.

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

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-current-c0dc1372/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-c0dc1372/cuda-benchmark.json \
    --label combined-current-c0dc1372 \
    --output-dir tmp/cuda-backend/combined-current-c0dc1372
```
