# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `764a2420`. The raw JSON, Markdown, and SVG reports are generated
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

- `tmp/cuda-backend/a100-current-764a2420/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-764a2420/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-764a2420/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-764a2420/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-764a2420/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-764a2420/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-764a2420/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-764a2420/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 33792 | 31744 | 23552 | 16383 | 0.94x | 0.48x |
| A100 | 65536 | 18176 | 19648 | 25919 | 18624 | 1.08x | 1.02x |
| A100 | 1048576 | 24288 | 22208 | 23615 | 19295 | 0.91x | 0.79x |
| H200 | 1024 | 29248 | 30912 | 21407 | 20800 | 1.06x | 0.71x |
| H200 | 65536 | 14784 | 16319 | 21632 | 16736 | 1.10x | 1.13x |
| H200 | 1048576 | 16287 | 17248 | 20640 | 22272 | 1.06x | 1.37x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 medium-vector launch rows are noisy in this capture,
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
| A100 | 1024 | 32768 | 32768 |
| A100 | 65536 | 23808 | 19872 |
| A100 | 1048576 | 19648 | 22656 |
| H200 | 1024 | 32288 | 33216 |
| H200 | 65536 | 17632 | 15552 |
| H200 | 1048576 | 16224 | 19072 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 256 | 32768 | 1.52x |
| A100 | 1024 | 6 | 64 | 31744 | 0.36x |
| A100 | 1024 | 12 | 256 | 20480 | 0.14x |
| A100 | 65536 | 2 | 32 | 22528 | 0.78x |
| A100 | 65536 | 6 | 256 | 18432 | 0.33x |
| A100 | 65536 | 12 | 64 | 21504 | 0.25x |
| A100 | 1048576 | 2 | 256 | 26624 | 0.79x |
| A100 | 1048576 | 6 | 256 | 35840 | 0.51x |
| A100 | 1048576 | 12 | 256 | 57344 | 0.45x |
| H200 | 1024 | 2 | 128 | 28960 | 1.45x |
| H200 | 1024 | 6 | 128 | 28480 | 0.38x |
| H200 | 1024 | 12 | 64 | 29120 | 0.20x |
| H200 | 65536 | 2 | 64 | 16160 | 0.79x |
| H200 | 65536 | 6 | 64 | 14912 | 0.30x |
| H200 | 65536 | 12 | 128 | 15584 | 0.21x |
| H200 | 1048576 | 2 | 256 | 18848 | 0.74x |
| H200 | 1048576 | 6 | 128 | 25888 | 0.46x |
| H200 | 1048576 | 12 | 256 | 36416 | 0.33x |

The H200 worker-grid rows keep the same broad signal as prior captures:
larger worker-block counts help larger vectors, but the best count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 worker
grid rows stay below matched host-schedule batch rows for most multi-task
rows in this capture, with the H200 two-task small-vector point as the
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
| A100 | 1024 | 0.82x | 0.92x | 0.63x | 0.61x | 0.61x | 0.68x | 0.68x | 0.68x | 0.74x | 0.95x |
| A100 | 65536 | 1.75x | 1.77x | 1.03x | 1.01x | 1.01x | 1.12x | 1.07x | 1.09x | 1.40x | 3.81x |
| A100 | 1048576 | 1.76x | 1.83x | 1.03x | 1.00x | 1.08x | 1.13x | 1.10x | 1.12x | 1.48x | 4.48x |
| H200 | 1024 | 1.39x | 1.46x | 1.19x | 1.05x | 1.03x | 1.02x | 0.94x | 0.96x | 1.25x | 1.28x |
| H200 | 65536 | 1.80x | 1.82x | 1.00x | 1.00x | 1.01x | 1.07x | 1.06x | 1.08x | 1.45x | 3.00x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 1.01x | 0.99x | 1.00x | 1.36x | 3.10x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, triad, quad, generic-args, graph-descriptor,
and unary-square rows prove mixed tensor/scalar fields, extra tensor pointers
through a fourth tensor task descriptor field, generic indexed argument
slots, explicit runtime graph lowering, and unary task-body lowering.
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
    tmp/cuda-backend/a100-current-764a2420/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-764a2420/cuda-benchmark.json \
    --label combined-current-764a2420 \
    --output-dir tmp/cuda-backend/combined-current-764a2420
```
