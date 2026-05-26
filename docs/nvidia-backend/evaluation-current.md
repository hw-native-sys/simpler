# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `b060039c`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `630`

## Artifact Paths

- `tmp/cuda-backend/a100-current-b060039c/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-b060039c/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-b060039c/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 8192 | 7168 | 8191 | 8191 | 0.88x | 1.00x |
| A100 | 65536 | 30496 | 19136 | 196160 | 418112 | 0.63x | 13.71x |
| A100 | 1048576 | 23264 | 26240 | 31263 | 22016 | 1.13x | 0.95x |
| H200 | 1024 | 30400 | 30368 | 20703 | 16992 | 1.00x | 0.56x |
| H200 | 65536 | 21856 | 19328 | 24960 | 19519 | 0.88x | 0.89x |
| H200 | 1048576 | 18976 | 19232 | 22336 | 19007 | 1.01x | 1.00x |

The compiler row stays in the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.

## Unary Host-Schedule Row

The `pto_host_schedule_unary_square` row validates the generated unary
`(a, out, n)` ABI in the full paired benchmark. The validator compares
against CUDA `float32` square results, which matters at the larger sizes
because exact Python integer squares no longer match single-precision output.

| GPU | N | Unary square ns |
| --- | - | --------------- |
| A100 | 1024 | 7168 |
| A100 | 65536 | 34624 |
| A100 | 1048576 | 21664 |
| H200 | 1024 | 28672 |
| H200 | 65536 | 20736 |
| H200 | 1048576 | 20128 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 32 | 8192 | 0.57x |
| A100 | 1024 | 6 | 64 | 7168 | 0.17x |
| A100 | 1024 | 12 | 32 | 8192 | 0.09x |
| A100 | 65536 | 2 | 128 | 9216 | 0.03x |
| A100 | 65536 | 6 | 64 | 10240 | 0.03x |
| A100 | 65536 | 12 | 32 | 12288 | 0.02x |
| A100 | 1048576 | 2 | 256 | 24576 | 0.77x |
| A100 | 1048576 | 6 | 128 | 36864 | 0.49x |
| A100 | 1048576 | 12 | 256 | 58368 | 0.45x |
| H200 | 1024 | 2 | 32 | 29632 | 1.11x |
| H200 | 1024 | 6 | 64 | 28960 | 0.38x |
| H200 | 1024 | 12 | 32 | 29376 | 0.23x |
| H200 | 65536 | 2 | 64 | 14976 | 0.71x |
| H200 | 65536 | 6 | 128 | 14528 | 0.27x |
| H200 | 65536 | 12 | 64 | 15296 | 0.20x |
| H200 | 1048576 | 2 | 256 | 18432 | 0.68x |
| H200 | 1048576 | 6 | 128 | 24800 | 0.42x |
| H200 | 1048576 | 12 | 256 | 36000 | 0.33x |

The H200 worker-grid rows keep the same broad signal as prior captures: more
worker blocks help larger vectors, but the best worker-block count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 host
batch rows still show run-to-run host-time noise, so ratios should be read as
capture evidence rather than a final resource policy.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine and triad rows use the same runtime graph shape as the base
DAG while reading additional descriptor fields, so they should track the base
DAG closely.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Triad/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | --------- | ---------- |
| A100 | 1024 | 1.40x | 1.50x | 1.00x | 0.95x | 1.00x | 1.65x |
| A100 | 65536 | 1.66x | 1.66x | 1.01x | 0.99x | 1.02x | 3.84x |
| A100 | 1048576 | 1.80x | 1.82x | 1.01x | 1.00x | 1.07x | 4.31x |
| H200 | 1024 | 1.26x | 1.30x | 1.06x | 1.00x | 0.85x | 1.12x |
| H200 | 65536 | 1.77x | 1.80x | 0.99x | 1.00x | 1.00x | 2.96x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 1.00x | 3.07x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY, scalar affine, and triad rows prove mixed tensor/scalar and
extra tensor-pointer descriptor lowering while tracking the base DAG closely.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. The `N=1024` DAG-shape ratios are small-launch rows and
should be treated as scheduling-smoke evidence rather than throughput signal.

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
    tmp/cuda-backend/a100-current-b060039c/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-b060039c/cuda-benchmark.json \
    --label combined-current-b060039c \
    --output-dir tmp/cuda-backend/combined-current-b060039c
```
