# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `6c49c5cf`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`

## Artifact Paths

- `tmp/cuda-backend/a100-current-6c49c5cf/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-6c49c5cf/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-6c49c5cf/cuda-benchmark-ratios.svg`

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It is
faster than PTO host scheduling for every captured row in this run, but it is
still host-owned replay rather than a device-side scheduler.

| GPU | N | PTO host ns | Driver ns | Graph ns | Graph/PTO |
| --- | - | ----------- | --------- | -------- | --------- |
| A100 | 1024 | 46080 | 39935 | 26623 | 0.58x |
| A100 | 65536 | 41568 | 41983 | 27551 | 0.66x |
| A100 | 1048576 | 28672 | 36768 | 24224 | 0.84x |
| H200 | 1024 | 31392 | 26944 | 20191 | 0.64x |
| H200 | 65536 | 23200 | 26784 | 20160 | 0.87x |
| H200 | 1048576 | 20832 | 24768 | 20255 | 0.97x |

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 256 | 47104 | 1.02x |
| A100 | 1024 | 6 | 32 | 46080 | 0.38x |
| A100 | 1024 | 12 | 32 | 47104 | 0.22x |
| A100 | 65536 | 2 | 32 | 35840 | 0.79x |
| A100 | 65536 | 6 | 256 | 31744 | 0.34x |
| A100 | 65536 | 12 | 32 | 31744 | 0.23x |
| A100 | 1048576 | 2 | 256 | 31744 | 0.85x |
| A100 | 1048576 | 6 | 128 | 41984 | 0.53x |
| A100 | 1048576 | 12 | 64 | 63488 | 0.45x |
| H200 | 1024 | 2 | 256 | 29920 | 0.91x |
| H200 | 1024 | 6 | 32 | 30496 | 0.37x |
| H200 | 1024 | 12 | 128 | 31168 | 0.22x |
| H200 | 65536 | 2 | 128 | 17632 | 0.68x |
| H200 | 65536 | 6 | 64 | 15648 | 0.29x |
| H200 | 65536 | 12 | 64 | 15776 | 0.21x |
| H200 | 1048576 | 2 | 256 | 20864 | 0.61x |
| H200 | 1048576 | 6 | 128 | 26752 | 0.45x |
| H200 | 1048576 | 12 | 256 | 38080 | 0.34x |

The strongest launch-amortization rows are the 12-task rows: `0.45x` on A100
and `0.34x` on H200 at `N=1048576`. Small vectors also benefit once task
count is high enough, but the two-task A100 `N=1024` row remains slightly
slower than host batch.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Tensor/DAG |
| --- | - | --------- | --------- | ---------- |
| A100 | 1024 | 1.33x | 1.20x | 1.23x |
| A100 | 65536 | 1.75x | 1.74x | 4.14x |
| A100 | 1048576 | 1.81x | 1.71x | 4.59x |
| H200 | 1024 | 1.29x | 1.56x | 1.31x |
| H200 | 65536 | 1.78x | 1.79x | 3.81x |
| H200 | 1048576 | 1.79x | 1.78x | 3.93x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles.

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

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only >/dev/null && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 2,6,12 \
     --worker-blocks-per-task 32,64,128,256 \
     --tensor-rows 8 --tensor-cols 4 --tensor-inner 12 \
     --label h200-current-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-current-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-current-6c49c5cf/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-6c49c5cf/cuda-benchmark.json \
    --label combined-current-6c49c5cf \
    --output-dir tmp/cuda-backend/combined-current-6c49c5cf
```
