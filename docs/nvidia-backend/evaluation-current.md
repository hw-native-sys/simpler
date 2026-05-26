# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `db0acd4c`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `594`

## Artifact Paths

- `tmp/cuda-backend/a100-current-db0acd4c/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-db0acd4c/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-db0acd4c/cuda-benchmark-ratios.svg`

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It is
faster than PTO host scheduling for most captured rows, but it is still
host-owned replay rather than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the task-body compiler path.
It uses the same host runtime path as `pto_host_schedule`, but the PTX comes
from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)` and
the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 7168 | 7168 | 8191 | 8191 | 1.00x | 1.14x |
| A100 | 65536 | 586752 | 612192 | 1927744 | 1837983 | 1.04x | 3.13x |
| A100 | 1048576 | 844896 | 935776 | 1488512 | 1232576 | 1.11x | 1.46x |
| H200 | 1024 | 28992 | 30816 | 20896 | 18112 | 1.06x | 0.62x |
| H200 | 65536 | 14400 | 15296 | 24383 | 19680 | 1.06x | 1.37x |
| H200 | 1048576 | 18816 | 19968 | 24032 | 20128 | 1.06x | 1.07x |

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
| A100 | 1024 | 7168 |
| A100 | 65536 | 651296 |
| A100 | 1048576 | 750336 |
| H200 | 1024 | 29984 |
| H200 | 65536 | 18336 |
| H200 | 1048576 | 19072 |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 64 | 7168 | 0.50x |
| A100 | 1024 | 6 | 32 | 8192 | 0.20x |
| A100 | 1024 | 12 | 32 | 8192 | 0.11x |
| A100 | 65536 | 2 | 256 | 8192 | 0.01x |
| A100 | 65536 | 6 | 64 | 9216 | 0.01x |
| A100 | 65536 | 12 | 64 | 10240 | 0.03x |
| A100 | 1048576 | 2 | 256 | 19456 | 0.02x |
| A100 | 1048576 | 6 | 128 | 31744 | 0.03x |
| A100 | 1048576 | 12 | 64 | 54272 | 0.06x |
| H200 | 1024 | 2 | 32 | 29472 | 1.26x |
| H200 | 1024 | 6 | 32 | 30400 | 0.36x |
| H200 | 1024 | 12 | 256 | 29344 | 0.21x |
| H200 | 65536 | 2 | 64 | 14560 | 0.68x |
| H200 | 65536 | 6 | 128 | 14240 | 0.30x |
| H200 | 65536 | 12 | 64 | 15008 | 0.19x |
| H200 | 1048576 | 2 | 256 | 18976 | 0.75x |
| H200 | 1048576 | 6 | 128 | 26048 | 0.45x |
| H200 | 1048576 | 12 | 256 | 36160 | 0.34x |

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
This full capture predates the two-scalar affine benchmark row; the focused
single-baseline A100/H200 validation is recorded in
[status.md](status.md#latest-local-verification).

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Scalar Affine/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ----------------- | ---------- |
| A100 | 1024 | 1.50x | 1.50x | 0.95x | - | 1.65x |
| A100 | 65536 | 1.59x | 1.58x | 0.99x | - | 3.70x |
| A100 | 1048576 | 2.99x | 2.98x | 0.96x | - | 5.51x |
| H200 | 1024 | 1.37x | 1.40x | 1.09x | - | 1.31x |
| H200 | 65536 | 1.79x | 1.80x | 1.00x | - | 2.97x |
| H200 | 1048576 | 1.79x | 1.79x | 0.99x | - | 3.06x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The scalar AXPY row proves mixed tensor/scalar descriptor lowering while
tracking the base DAG closely, especially for larger vectors. The tensor row
also proves the descriptor metadata path for non-square `8x4x12` tiles. The
`N=1024` DAG-shape ratios are small-launch rows and should be treated as
scheduling-smoke evidence rather than throughput signal.

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
    tmp/cuda-backend/a100-current-db0acd4c/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-db0acd4c/cuda-benchmark.json \
    --label combined-current-db0acd4c \
    --output-dir tmp/cuda-backend/combined-current-db0acd4c
```
