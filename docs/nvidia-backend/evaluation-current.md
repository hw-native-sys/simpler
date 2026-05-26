# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `0e1be392`. The raw JSON, Markdown, and SVG reports are generated
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

- `tmp/cuda-backend/a100-current-0e1be392/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-0e1be392/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-0e1be392/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-0e1be392/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0e1be392/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-0e1be392/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-0e1be392/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-0e1be392/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 37888 | 37888 | 36864 | 22528 | 1.00x | 0.59x |
| A100 | 65536 | 26720 | 28608 | 38752 | 19648 | 1.07x | 0.74x |
| A100 | 1048576 | 24960 | 24832 | 35583 | 22048 | 0.99x | 0.88x |
| H200 | 1024 | 36704 | 35744 | 35296 | 27039 | 0.97x | 0.74x |
| H200 | 65536 | 24416 | 25440 | 35840 | 28896 | 1.04x | 1.18x |
| H200 | 1048576 | 20608 | 20672 | 31520 | 19551 | 1.00x | 0.95x |

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
| A100 | 1024 | 2 | 256 | 33792 | 1.14x |
| A100 | 1024 | 6 | 128 | 32768 | 0.36x |
| A100 | 1024 | 12 | 32 | 35840 | 0.23x |
| A100 | 65536 | 2 | 256 | 26624 | 0.71x |
| A100 | 65536 | 6 | 128 | 23552 | 0.32x |
| A100 | 65536 | 12 | 32 | 25600 | 0.24x |
| A100 | 1048576 | 2 | 256 | 26624 | 0.78x |
| A100 | 1048576 | 6 | 128 | 36864 | 0.51x |
| A100 | 1048576 | 12 | 64 | 58368 | 0.45x |
| H200 | 1024 | 2 | 32 | 37440 | 0.97x |
| H200 | 1024 | 6 | 128 | 36576 | 0.39x |
| H200 | 1024 | 12 | 128 | 37408 | 0.23x |
| H200 | 65536 | 2 | 128 | 24256 | 0.70x |
| H200 | 65536 | 6 | 64 | 23136 | 0.34x |
| H200 | 65536 | 12 | 128 | 22112 | 0.18x |
| H200 | 1048576 | 2 | 256 | 21952 | 0.66x |
| H200 | 1048576 | 6 | 128 | 27104 | 0.44x |
| H200 | 1048576 | 12 | 256 | 38496 | 0.35x |

The strongest large-vector launch-amortization rows are still the 12-task
rows: `0.45x` on A100 and `0.35x` on H200 at `N=1048576`. The best
worker-block count is
not monotonic, so these rows support a tunable policy rather than a fixed
default.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Tensor/DAG |
| --- | - | --------- | --------- | ---------- |
| A100 | 1024 | 1.22x | 1.33x | 1.37x |
| A100 | 65536 | 1.76x | 1.76x | 3.90x |
| A100 | 1048576 | 1.80x | 1.72x | 4.27x |
| H200 | 1024 | 1.21x | 1.30x | 1.19x |
| H200 | 65536 | 1.75x | 1.78x | 2.89x |
| H200 | 1048576 | 1.78x | 1.77x | 3.00x |

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
    tmp/cuda-backend/a100-current-0e1be392/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-0e1be392/cuda-benchmark.json \
    --label combined-current-0e1be392 \
    --output-dir tmp/cuda-backend/combined-current-0e1be392
```
