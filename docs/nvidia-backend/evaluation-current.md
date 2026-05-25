# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `d7257c84`. The raw JSON, Markdown, and SVG reports are generated
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

- `tmp/cuda-backend/a100-current-d7257c84/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-d7257c84/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-d7257c84/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-d7257c84/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-d7257c84/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-d7257c84/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-d7257c84/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-d7257c84/cuda-benchmark-ratios.svg`

## Launch Baselines

CUDA Graph replay remains a useful phase-one host-launch baseline. It is
faster than PTO host scheduling for most captured rows, but it is still
host-owned replay rather than a device-side scheduler.

The `pto_host_schedule_compiler` row validates the new task-body compiler
path. It uses the same host runtime path as `pto_host_schedule`, but the PTX
comes from `KernelCompiler(platform="cuda").compile_cuda_host_schedule(...)`
and the shared task wrapper generator.

| GPU | N | PTO host ns | Compiler ns | Driver ns | Graph ns | Compiler/PTO | Graph/PTO |
| --- | - | ----------- | ----------- | --------- | -------- | ------------ | --------- |
| A100 | 1024 | 43008 | 46080 | 43007 | 25599 | 1.07x | 0.60x |
| A100 | 65536 | 32320 | 31424 | 42847 | 29120 | 0.97x | 0.90x |
| A100 | 1048576 | 29568 | 25152 | 37055 | 24351 | 0.85x | 0.82x |
| H200 | 1024 | 37472 | 37728 | 33952 | 27551 | 1.01x | 0.74x |
| H200 | 65536 | 25632 | 25184 | 39039 | 27712 | 0.98x | 1.08x |
| H200 | 1048576 | 18528 | 19264 | 23040 | 17535 | 1.04x | 0.95x |

The compiler row is within the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 256 | 41984 | 0.85x |
| A100 | 1024 | 6 | 32 | 43008 | 0.41x |
| A100 | 1024 | 12 | 256 | 43008 | 0.23x |
| A100 | 65536 | 2 | 32 | 31744 | 0.81x |
| A100 | 65536 | 6 | 64 | 30720 | 0.43x |
| A100 | 65536 | 12 | 32 | 29696 | 0.28x |
| A100 | 1048576 | 2 | 256 | 27648 | 0.72x |
| A100 | 1048576 | 6 | 128 | 41984 | 0.56x |
| A100 | 1048576 | 12 | 256 | 60416 | 0.45x |
| H200 | 1024 | 2 | 64 | 38240 | 0.96x |
| H200 | 1024 | 6 | 32 | 37280 | 0.32x |
| H200 | 1024 | 12 | 128 | 36576 | 0.20x |
| H200 | 65536 | 2 | 128 | 22560 | 0.56x |
| H200 | 65536 | 6 | 32 | 21696 | 0.30x |
| H200 | 65536 | 12 | 128 | 22432 | 0.19x |
| H200 | 1048576 | 2 | 256 | 18528 | 0.67x |
| H200 | 1048576 | 6 | 128 | 22848 | 0.39x |
| H200 | 1048576 | 12 | 256 | 35232 | 0.32x |

The strongest launch-amortization rows are still the 12-task rows: `0.45x`
on A100 and `0.32x` on H200 at `N=1048576`. The best worker-block count is
not monotonic, so these rows support a tunable policy rather than a fixed
default.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Tensor/DAG |
| --- | - | --------- | --------- | ---------- |
| A100 | 1024 | 1.29x | 1.38x | 1.50x |
| A100 | 65536 | 1.77x | 1.73x | 4.07x |
| A100 | 1048576 | 1.78x | 1.72x | 4.58x |
| H200 | 1024 | 1.21x | 1.41x | 1.42x |
| H200 | 65536 | 1.71x | 1.73x | 3.62x |
| H200 | 1048576 | 1.79x | 1.79x | 3.93x |

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
    tmp/cuda-backend/a100-current-d7257c84/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-d7257c84/cuda-benchmark.json \
    --label combined-current-d7257c84 \
    --output-dir tmp/cuda-backend/combined-current-d7257c84
```
