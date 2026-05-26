# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `5f80dbdd`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `612`

## Artifact Paths

- `tmp/cuda-backend/a100-current-5f80dbdd/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-5f80dbdd/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-5f80dbdd/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-5f80dbdd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-5f80dbdd/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-5f80dbdd/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-5f80dbdd/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-5f80dbdd/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 7168 | 7168 | 8191 | 8191 | 1.00x | 1.14x |
| A100 | 65536 | 301056 | 804384 | 1597216 | 1643872 | 2.67x | 5.46x |
| A100 | 1048576 | 834368 | 831168 | 208287 | 238784 | 1.00x | 0.29x |
| H200 | 1024 | 29504 | 31104 | 22175 | 15647 | 1.05x | 0.53x |
| H200 | 65536 | 15296 | 16480 | 24576 | 18176 | 1.08x | 1.19x |
| H200 | 1048576 | 19552 | 19872 | 29216 | 20320 | 1.02x | 1.04x |

The compiler row is within the same launch-latency band as the handwritten
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
| A100 | 1024 | 6144 |
| A100 | 65536 | 753312 |
| A100 | 1048576 | 349248 |
| H200 | 1024 | 29600 |
| H200 | 65536 | 18720 |
| H200 | 1048576 | 20768 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 32 | 7168 | 0.50x |
| A100 | 1024 | 6 | 32 | 7168 | 0.17x |
| A100 | 1024 | 12 | 32 | 8192 | 0.11x |
| A100 | 65536 | 2 | 128 | 8192 | 0.03x |
| A100 | 65536 | 6 | 128 | 8192 | 0.01x |
| A100 | 65536 | 12 | 64 | 10240 | 0.03x |
| A100 | 1048576 | 2 | 256 | 19456 | 0.06x |
| A100 | 1048576 | 6 | 128 | 35840 | 0.09x |
| A100 | 1048576 | 12 | 256 | 60416 | 0.14x |
| H200 | 1024 | 2 | 32 | 29504 | 1.26x |
| H200 | 1024 | 6 | 128 | 29920 | 0.38x |
| H200 | 1024 | 12 | 256 | 29632 | 0.20x |
| H200 | 65536 | 2 | 32 | 16448 | 0.73x |
| H200 | 65536 | 6 | 64 | 15296 | 0.29x |
| H200 | 65536 | 12 | 128 | 15328 | 0.19x |
| H200 | 1048576 | 2 | 256 | 20480 | 0.74x |
| H200 | 1048576 | 6 | 128 | 25440 | 0.42x |
| H200 | 1048576 | 12 | 256 | 36896 | 0.33x |

The H200 worker-grid rows keep the same broad signal as prior captures: more
worker blocks help larger vectors, but the best worker-block count is not
monotonic across GPUs, vector sizes, and descriptor counts. The A100 host
batch rows in this capture show large run-to-run noise for `N=65536` and
`N=1048576`, so their very small ratios should be read as capture evidence
rather than policy signal.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.
The scalar affine row uses the same runtime graph shape as scalar AXPY but
reads two scalar descriptor fields, so it should track the base DAG closely.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | ---------- |
| A100 | 1024 | 1.47x | 1.58x | 1.00x | 1.05x | 1.79x |
| A100 | 65536 | 1.71x | 1.72x | 1.01x | 1.00x | 4.01x |
| A100 | 1048576 | 1.69x | 1.73x | 1.01x | 1.00x | 5.62x |
| H200 | 1024 | 1.29x | 1.31x | 1.00x | 0.99x | 1.24x |
| H200 | 65536 | 1.79x | 1.79x | 0.99x | 0.99x | 2.94x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | 1.00x | 3.03x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY and scalar affine rows prove mixed tensor/scalar descriptor
lowering while tracking the base DAG closely, especially for larger vectors.
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
    tmp/cuda-backend/a100-current-5f80dbdd/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-5f80dbdd/cuda-benchmark.json \
    --label combined-current-5f80dbdd \
    --output-dir tmp/cuda-backend/combined-current-5f80dbdd
```
