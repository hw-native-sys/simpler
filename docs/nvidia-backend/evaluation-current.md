# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `47ac2bb5`. The raw JSON, Markdown, and SVG reports are generated
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

- `tmp/cuda-backend/a100-current-47ac2bb5/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-47ac2bb5/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-47ac2bb5/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-47ac2bb5/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-47ac2bb5/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-47ac2bb5/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-47ac2bb5/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-47ac2bb5/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 20480 | 18432 | 8191 | 8191 | 0.90x | 0.40x |
| A100 | 65536 | 18592 | 19968 | 24927 | 17503 | 1.07x | 0.94x |
| A100 | 1048576 | 22304 | 20992 | 22175 | 20608 | 0.94x | 0.92x |
| H200 | 1024 | 29088 | 30080 | 22048 | 17920 | 1.03x | 0.62x |
| H200 | 65536 | 15296 | 16608 | 25280 | 19200 | 1.09x | 1.26x |
| H200 | 1048576 | 18304 | 19104 | 22752 | 19680 | 1.04x | 1.08x |

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
| A100 | 1024 | 2 | 256 | 7168 | 0.50x |
| A100 | 1024 | 6 | 32 | 8192 | 0.19x |
| A100 | 1024 | 12 | 32 | 8192 | 0.09x |
| A100 | 65536 | 2 | 32 | 15360 | 0.58x |
| A100 | 65536 | 6 | 128 | 12288 | 0.22x |
| A100 | 65536 | 12 | 128 | 14336 | 0.17x |
| A100 | 1048576 | 2 | 256 | 22528 | 0.70x |
| A100 | 1048576 | 6 | 128 | 31744 | 0.45x |
| A100 | 1048576 | 12 | 256 | 54272 | 0.43x |
| H200 | 1024 | 2 | 32 | 29120 | 0.90x |
| H200 | 1024 | 6 | 32 | 29152 | 0.37x |
| H200 | 1024 | 12 | 128 | 28768 | 0.22x |
| H200 | 65536 | 2 | 256 | 14880 | 0.56x |
| H200 | 65536 | 6 | 64 | 14752 | 0.31x |
| H200 | 65536 | 12 | 128 | 14240 | 0.19x |
| H200 | 1048576 | 2 | 256 | 19040 | 0.65x |
| H200 | 1048576 | 6 | 128 | 23616 | 0.39x |
| H200 | 1048576 | 12 | 256 | 36480 | 0.34x |

The strongest large-vector launch-amortization rows are still the 12-task
rows: `0.43x` on A100 and `0.34x` on H200 at `N=1048576`. The best
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
| A100 | 1024 | 1.45x | 1.50x | 1.70x |
| A100 | 65536 | 1.76x | 1.75x | 3.88x |
| A100 | 1048576 | 1.81x | 1.71x | 4.26x |
| H200 | 1024 | 1.61x | 1.20x | 1.08x |
| H200 | 65536 | 1.79x | 1.80x | 2.95x |
| H200 | 1048576 | 1.79x | 1.79x | 3.03x |

The key correctness signal is that all DAG variants use generated dispatch
and runtime graph descriptors without changing the persistent launch path.
The tensor row also proves the descriptor metadata path for non-square
`8x4x12` tiles. The H200 `N=1024` DAG-shape ratios are small-launch rows and
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
    tmp/cuda-backend/a100-current-47ac2bb5/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-47ac2bb5/cuda-benchmark.json \
    --label combined-current-47ac2bb5 \
    --output-dir tmp/cuda-backend/combined-current-47ac2bb5
```
