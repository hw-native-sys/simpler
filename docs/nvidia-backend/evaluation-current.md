# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `38ff341e`. The raw JSON, Markdown, and SVG reports are generated
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

- `tmp/cuda-backend/a100-current-38ff341e/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-38ff341e/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-38ff341e/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-38ff341e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-38ff341e/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-38ff341e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-38ff341e/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-38ff341e/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 40960 | 41984 | 45056 | 26623 | 1.02x | 0.65x |
| A100 | 65536 | 32256 | 32992 | 26976 | 19807 | 1.02x | 0.61x |
| A100 | 1048576 | 28640 | 26048 | 33504 | 25760 | 0.91x | 0.90x |
| H200 | 1024 | 29184 | 29888 | 27807 | 20352 | 1.02x | 0.70x |
| H200 | 65536 | 22656 | 25152 | 26303 | 28543 | 1.11x | 1.26x |
| H200 | 1048576 | 20800 | 22176 | 28095 | 22816 | 1.07x | 1.10x |

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
| A100 | 1024 | 2 | 128 | 45056 | 1.10x |
| A100 | 1024 | 6 | 64 | 43008 | 0.40x |
| A100 | 1024 | 12 | 64 | 43008 | 0.24x |
| A100 | 65536 | 2 | 32 | 29696 | 0.72x |
| A100 | 65536 | 6 | 32 | 29696 | 0.43x |
| A100 | 65536 | 12 | 32 | 32768 | 0.31x |
| A100 | 1048576 | 2 | 256 | 27648 | 0.73x |
| A100 | 1048576 | 6 | 256 | 38912 | 0.51x |
| A100 | 1048576 | 12 | 256 | 58368 | 0.43x |
| H200 | 1024 | 2 | 32 | 30624 | 0.94x |
| H200 | 1024 | 6 | 64 | 29952 | 0.37x |
| H200 | 1024 | 12 | 256 | 30656 | 0.22x |
| H200 | 65536 | 2 | 128 | 18592 | 0.77x |
| H200 | 65536 | 6 | 128 | 17824 | 0.32x |
| H200 | 65536 | 12 | 64 | 19136 | 0.18x |
| H200 | 1048576 | 2 | 256 | 21856 | 0.71x |
| H200 | 1048576 | 6 | 128 | 26400 | 0.44x |
| H200 | 1048576 | 12 | 256 | 38080 | 0.34x |

The strongest launch-amortization rows are still the 12-task rows: `0.43x`
on A100 and `0.34x` on H200 at `N=1048576`. The best worker-block count is
not monotonic, so these rows support a tunable policy rather than a fixed
default.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Tensor/DAG |
| --- | - | --------- | --------- | ---------- |
| A100 | 1024 | 1.19x | 1.37x | 1.37x |
| A100 | 65536 | 1.71x | 1.69x | 3.78x |
| A100 | 1048576 | 1.81x | 1.70x | 4.19x |
| H200 | 1024 | 1.28x | 1.33x | 1.34x |
| H200 | 65536 | 1.79x | 1.79x | 2.94x |
| H200 | 1048576 | 1.79x | 1.79x | 3.03x |

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
  'cd /data/shibizhao/pto-cu && \
   timeout 60 git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 \
     fetch origin design/nvidia-backend >/dev/null && \
   git checkout -B design/nvidia-backend FETCH_HEAD >/dev/null && \
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
    tmp/cuda-backend/a100-current-38ff341e/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-38ff341e/cuda-benchmark.json \
    --label combined-current-38ff341e \
    --output-dir tmp/cuda-backend/combined-current-38ff341e
```
