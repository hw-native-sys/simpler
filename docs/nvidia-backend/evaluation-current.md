# CUDA Current Evaluation Capture

This page summarizes the current paired A100/H200 CUDA backend capture from
commit `93636997`. The raw JSON, Markdown, and SVG reports are generated
locally under `tmp/cuda-backend/` and intentionally remain uncommitted.

The capture uses `nvcc` for target-specific PTX on both machines:

- A100: `compute_80`
- H200: `compute_90`
- tensor descriptor: `8x4x12`
- sizes: `1024,65536,1048576`
- repeats: `3`
- batch tasks: `2,6,12`
- worker blocks per task: `32,64,128,256`
- samples in combined JSON: `576`

## Artifact Paths

- `tmp/cuda-backend/a100-current-93636997/cuda-benchmark.json`
- `tmp/cuda-backend/a100-current-93636997/cuda-benchmark.md`
- `tmp/cuda-backend/h200-current-93636997/cuda-benchmark.json`
- `tmp/cuda-backend/h200-current-93636997/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-93636997/cuda-benchmark.json`
- `tmp/cuda-backend/combined-current-93636997/cuda-benchmark.md`
- `tmp/cuda-backend/combined-current-93636997/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-current-93636997/cuda-benchmark-ratios.svg`

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
| A100 | 1024 | 38912 | 29696 | 23552 | 15359 | 0.76x | 0.39x |
| A100 | 65536 | 2439776 | 2440800 | 652544 | 451680 | 1.00x | 0.19x |
| A100 | 1048576 | 391424 | 307968 | 572575 | 943008 | 0.79x | 2.41x |
| H200 | 1024 | 29088 | 28864 | 27456 | 17152 | 0.99x | 0.59x |
| H200 | 65536 | 16224 | 20960 | 29184 | 19168 | 1.29x | 1.18x |
| H200 | 1048576 | 20256 | 18688 | 25183 | 18176 | 0.92x | 0.90x |

The compiler row is within the same launch-latency band as the handwritten
host-schedule PTX. That is the important signal for this slice: the shared
task-body wrapper path can feed the existing host runtime without changing
the launch ABI or adding a separate generated-kernel calling convention.
The A100 host-launch rows in this capture show large run-to-run noise for
`N=65536` and `N=1048576`; use them as capture evidence, not as stable
policy signal.

## Worker Grid Rows

The plain one-block persistent batch rows are still too serial for larger
vectors. The worker-grid row is the current useful persistent-device
throughput slice because it keeps one persistent launch but assigns multiple
worker blocks to each task descriptor.

| GPU | N | Tasks | Best worker blocks/task | Device ns | Vs host batch |
| --- | - | ----- | ----------------------- | --------- | ------------- |
| A100 | 1024 | 2 | 128 | 28672 | 0.90x |
| A100 | 1024 | 6 | 64 | 28672 | 0.31x |
| A100 | 1024 | 12 | 256 | 27648 | 0.16x |
| A100 | 65536 | 2 | 64 | 10240 | 0.24x |
| A100 | 65536 | 6 | 256 | 10240 | 0.00x |
| A100 | 65536 | 12 | 128 | 11264 | 0.00x |
| A100 | 1048576 | 2 | 256 | 18432 | 0.04x |
| A100 | 1048576 | 6 | 128 | 31744 | 0.06x |
| A100 | 1048576 | 12 | 256 | 54272 | 0.11x |
| H200 | 1024 | 2 | 256 | 29248 | 1.20x |
| H200 | 1024 | 6 | 32 | 29504 | 0.37x |
| H200 | 1024 | 12 | 64 | 28928 | 0.23x |
| H200 | 65536 | 2 | 256 | 15296 | 0.73x |
| H200 | 65536 | 6 | 64 | 14656 | 0.27x |
| H200 | 65536 | 12 | 64 | 15552 | 0.21x |
| H200 | 1048576 | 2 | 256 | 18720 | 0.70x |
| H200 | 1048576 | 6 | 128 | 26080 | 0.45x |
| H200 | 1048576 | 12 | 256 | 35968 | 0.33x |

The H200 rows keep the same broad signal as the prior capture: more worker
blocks help large vectors, but the best worker-block count is not monotonic.
The A100 `0.00x` ratios are a side effect of the noisy host-batch reference in
this capture, so they should not be used as a policy result.

## Persistent DAG Shapes

The DAG rows validate the persistent-device scheduler path rather than equal
work throughput. Chain and reuse add dependency levels and extra arithmetic.
The tensor row replaces one elementwise task with tiled GEMM work, so its
large-vector ratio is expected to be several times slower than the simple DAG.

| GPU | N | Chain/DAG | Reuse/DAG | Scalar AXPY/DAG | Tensor/DAG |
| --- | - | --------- | --------- | --------------- | ---------- |
| A100 | 1024 | 1.28x | 1.38x | 1.09x | 1.41x |
| A100 | 65536 | 1.64x | 1.61x | 1.00x | 3.72x |
| A100 | 1048576 | 3.55x | 3.61x | 1.01x | 5.75x |
| H200 | 1024 | 1.41x | 1.46x | 1.04x | 1.34x |
| H200 | 65536 | 1.78x | 1.80x | 1.00x | 2.99x |
| H200 | 1048576 | 1.79x | 1.79x | 1.00x | 3.07x |

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
    tmp/cuda-backend/a100-current-93636997/cuda-benchmark.json \
    tmp/cuda-backend/h200-current-93636997/cuda-benchmark.json \
    --label combined-current-93636997 \
    --output-dir tmp/cuda-backend/combined-current-93636997
```
