# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `ba99b593`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `666`

## Artifact Paths

- `tmp/cuda-backend/a100-current-ba99b593/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-ba99b593/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-ba99b593/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-ba99b593/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-ba99b593/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 7168 | 8192 | 7168 | 8191 | 1.14x | 1.14x |
| A100 | 65536 | 21568 | 19008 | 26367 | 23935 | 0.88x | 1.11x |
| A100 | 1048576 | 23648 | 24416 | 31168 | 22304 | 1.03x | 0.94x |
| H200 | 1024 | 30144 | 31583 | 23423 | 15519 | 1.05x | 0.51x |
| H200 | 65536 | 15936 | 17760 | 25439 | 18015 | 1.11x | 1.13x |
| H200 | 1048576 | 30912 | 30912 | 49024 | 28863 | 1.00x | 0.93x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 raw Driver API and CUDA Graph rows are slower than the PTO
host-schedule row in this capture for larger vectors, so use the paired report
as current evidence rather than a final launch-overhead ranking.

## Unary Host-Schedule Row

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The validator compares
against CUDA `float32` square results, which matters at the larger sizes
because exact Python integer squares no longer match single-precision output.

| GPU | N | Unary square ns |
| --- | - | --------------- |
| A100 | 1024 | 8192 |
| A100 | 65536 | 21472 |
| A100 | 1048576 | 25440 |
| H200 | 1024 | 30944 |
| H200 | 65536 | 19616 |
| H200 | 1048576 | 29120 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 64 | 7168 | 0.54x |
| A100 | 1024 | 6 | 32 | 8192 | 0.21x |
| A100 | 1024 | 12 | 32 | 8192 | 0.10x |
| A100 | 65536 | 2 | 64 | 21504 | 0.79x |
| A100 | 65536 | 6 | 32 | 19456 | 0.35x |
| A100 | 65536 | 12 | 128 | 21504 | 0.25x |
| A100 | 1048576 | 2 | 256 | 32768 | 0.92x |
| A100 | 1048576 | 6 | 128 | 40960 | 0.57x |
| A100 | 1048576 | 12 | 256 | 63488 | 0.48x |
| H200 | 1024 | 2 | 256 | 29440 | 1.19x |
| H200 | 1024 | 6 | 64 | 29632 | 0.38x |
| H200 | 1024 | 12 | 128 | 29696 | 0.21x |
| H200 | 65536 | 2 | 32 | 17024 | 0.75x |
| H200 | 65536 | 6 | 64 | 15776 | 0.33x |
| H200 | 65536 | 12 | 64 | 17056 | 0.22x |
| H200 | 1048576 | 2 | 256 | 19808 | 0.65x |
| H200 | 1048576 | 6 | 128 | 26432 | 0.44x |
| H200 | 1048576 | 12 | 256 | 36544 | 0.33x |

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
| A100 | 1024 | 1.45x | 1.50x | 1.00x | 1.05x | 1.05x | 1.05x | 1.20x | 1.70x |
| A100 | 65536 | 1.76x | 1.78x | 1.00x | 1.01x | 1.02x | 1.19x | 1.43x | 3.85x |
| A100 | 1048576 | 1.88x | 1.95x | 1.04x | 1.03x | 1.11x | 1.19x | 1.59x | 4.72x |
| H200 | 1024 | 1.36x | 1.41x | 1.00x | 0.98x | 1.18x | 0.98x | 1.16x | 1.36x |
| H200 | 65536 | 1.78x | 1.79x | 1.00x | 0.99x | 0.99x | 1.07x | 1.45x | 3.02x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 1.01x | 1.37x | 3.08x |

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
    tmp/cuda-backend/a100-current-ba99b593/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-ba99b593/cuda-benchmark.json \
    --label combined-current-ba99b593 \
    --output-dir tmp/cuda-backend/combined-current-ba99b593
```
