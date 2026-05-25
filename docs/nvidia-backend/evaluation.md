# CUDA Backend Evaluation Notes

This page summarizes the current evaluation evidence for the CUDA backend.
The measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They are shaped by the VDCores and MPK papers only at the evaluation
structure level: fixed GPU work, repeated problem sizes, selected launch
baselines, local A100 runs, and remote H200 runs.

The latest captured raw reports are under `tmp/`:

- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/a100.json`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/h200.json`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/cuda-smoke-report.md`
- `tmp/cuda-backend/tensor-descriptor-smoke-38db010e/cuda-smoke-report.svg`
- `tmp/cuda-backend/a100-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/h200-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark.md`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-rangewide-cc6869f7/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/h200-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark.md`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-taskcount-7194bfc9/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/h200-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark.md`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-gridext-3eeb399a/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/h200-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/h200-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/h200-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/h200-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/h200-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark.md`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-tensor-8950e029/cuda-benchmark-ratios.svg`
- `tmp/cuda-backend/a100-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/h200-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark.md`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark.svg`
- `tmp/cuda-backend/combined-graph-ba2cdd0e/cuda-benchmark-ratios.svg`

The tensor descriptor smoke data was captured from commit `38db010e`. The
wider vector/task range data was captured from commit `cc6869f7`. The
task-count sweep data was captured from commit `7194bfc9`. The extended
worker-grid data was captured from commit `3eeb399a`. The earlier worker-grid
data was captured from commit `e430bc1b`. The stream concurrency data was
captured from commit `37bebf44`. The DAG-chain data was captured from commit
`323f4587`. The scratch-reuse DAG data was captured from commit `bcf54a88`.
The tensor-tile DAG data was captured from commit `8950e029`. The CUDA Graph
launch-baseline data was captured from commit `ba2cdd0e`.

## Current Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
- `direct_driver_graph`: same Driver API vector-add kernel replayed through a
  CUDA Graph, with graph instantiation outside the measured interval. This is
  a host-launch amortization baseline, not a device-side scheduler.
- `pto_host_schedule`: PTO CUDA host runtime C API and manifest dispatch.
- `pto_persistent_device`: descriptor-array persistent executor.
- `pto_persistent_queue`: scheduler block plus bounded device ring queue.
- `pto_persistent_dag`: generated-dispatch-like task selection with fan-in
  counters.
- `pto_persistent_dag_chain`: five-task generated-dispatch DAG with a
  post-fan-in dependency chain. It reuses the same compiled device binary as
  `pto_persistent_dag`; the difference is only the runtime graph descriptors.
- `pto_persistent_dag_reuse`: six-task generated-dispatch DAG that reuses a
  scratch buffer after the buffer's last dependent completes. It is a
  lifecycle validation row rather than a throughput row.
- `pto_persistent_dag_tensor`: four-task generated-dispatch DAG with a
  16x16 tiled GEMM task followed by residual, gate, and fan-in elementwise
  tasks.
- `*_batch`: same-work rows with six vector-add task descriptors. These rows
  compare repeated host launches with one persistent launch over the same
  descriptor count.
- `pto_persistent_device_grid_batch`: direct persistent-device batch row with
  a swept number of CUDA worker blocks assigned to each task descriptor.

Ratios are relative to the matched host-schedule row for the same GPU, vector
length, and task count. For batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row.
Generated reports also include a `DAG Shape Rows` table that compares
`pto_persistent_dag_*` rows against `pto_persistent_dag` for the same GPU and
vector length. Use that table for graph-shape interpretation because the
chain, reuse, and tensor DAGs intentionally have different task counts.
The generated `cuda-benchmark-ratios.svg` file visualizes the same matched
reference ratios used by the main Markdown table; use it for launch-overhead
and stream-concurrency comparisons where the rows share a reference task
count.

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | Best grid blocks/task | Best grid ratio | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | --------------------- | --------------- | ------------------------ |
| A100 | 1024 | 73728 | 0.61x | 256 | 0.58x | 0.82x |
| H200 | 1024 | 66879 | 0.42x | 256 | 0.55x | 0.60x |
| A100 | 65536 | 67968 | 1.34x | 128 | 0.47x | 1.42x |
| H200 | 65536 | 61952 | 1.33x | 32 | 0.37x | 1.40x |
| A100 | 1048576 | 76768 | 16.46x | 256 | 0.53x | 16.06x |
| H200 | 1048576 | 67327 | 18.13x | 256 | 0.42x | 18.15x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows show why the worker-grid variant matters: in the
`32,64,128,256` extended sweep, the best large-vector row uses 256 worker
blocks per descriptor on both GPUs. That reduces the A100 direct persistent
batch row from `16.46x` to `0.53x` versus the matched host-schedule batch
row, and reduces the H200 row from `18.13x` to `0.42x`. The middle `N=65536`
rows show the same shape: plain persistent batch and queue batch are slower
than host-schedule batch, while the best worker-grid row is faster. The
middle-size optimum is not monotonic: A100 ties at 128/256 blocks, while H200
is best at 32 blocks in this capture.

## Task-Count Sweep

The `7194bfc9` task-count sweep uses the same vector-add callable and compares
two vector lengths, three descriptor counts, and two grid sizes. The
worker-grid row stays below the matched host-schedule batch row for every
captured task count, while the plain one-block persistent batch row remains
too serial for larger vectors.

| GPU | N | Tasks | Best grid blocks/task | Best grid ratio |
| --- | - | ----- | --------------------- | --------------- |
| A100 | 65536 | 2 | 128 | 0.77x |
| A100 | 65536 | 6 | 128 | 0.38x |
| A100 | 65536 | 12 | 128 | 0.33x |
| A100 | 1048576 | 2 | 256 | 0.77x |
| A100 | 1048576 | 6 | 256 | 0.58x |
| A100 | 1048576 | 12 | 256 | 0.44x |
| H200 | 65536 | 2 | 128 | 0.68x |
| H200 | 65536 | 6 | 256 | 0.36x |
| H200 | 65536 | 12 | 128 | 0.22x |
| H200 | 1048576 | 2 | 256 | 0.79x |
| H200 | 1048576 | 6 | 128 | 0.40x |
| H200 | 1048576 | 12 | 256 | 0.34x |

Increasing descriptor count improves the worker-grid ratio because the
matched host-schedule reference pays more repeated launch overhead. It does
not make the one-block persistent rows acceptable: at `N=1048576`, the A100
plain persistent batch row is still `31.59x`, `16.92x`, and `10.55x` for
2, 6, and 12 tasks respectively.

## Wider Range Sweep

The `cc6869f7` wider range sweep keeps the same generated callables and
extends the descriptor-count sweep to `4,8,16` tasks across `N=16384`,
`262144`, and `4194304`. The worker-grid row stays below the matched
host-schedule batch row for every captured row, but the largest vectors become
compute-sensitive and show a smaller ratio advantage.

| GPU | N | Tasks | Best grid blocks/task | Best grid ratio |
| --- | - | ----- | --------------------- | --------------- |
| A100 | 16384 | 16 | 256 | 0.13x |
| H200 | 16384 | 16 | 128 | 0.13x |
| A100 | 262144 | 16 | 128 | 0.22x |
| H200 | 262144 | 16 | 256 | 0.22x |
| A100 | 4194304 | 16 | 128 | 0.53x |
| H200 | 4194304 | 16 | 128 | 0.51x |

The best grid size is not monotonic. H200 uses 256 blocks/task for the
`N=4194304`, four-task row, but 128 blocks/task for the eight-task and
16-task rows. A100 uses 256 blocks/task for the `N=4194304`, eight-task row,
but 128 blocks/task for the four-task and 16-task rows. This keeps `128` and
`256` as candidates for the current vector microbenchmark, not tuned defaults.

The one-block persistent rows remain too serial for large vectors. At
`N=4194304`, `pto_persistent_device_batch` is `47.18x`, `24.26x`, and
`12.38x` on A100 for 4, 8, and 16 tasks respectively; H200 is `65.28x`,
`37.53x`, and `18.46x`. The scalar tensor DAG row is also only an ABI and
scheduler validation row at this size: it reaches `355.69x` on A100 and
`461.79x` on H200 versus the matched four-task host-schedule batch row.

## PTX Sources

The A100 rows compiled PTX with local `nvcc` for `compute_80`. The H200 rows
compiled PTX with remote `nvcc` for `compute_90`, discovered from the
`/usr/local/cuda*` toolkit path. The report still marks embedded PTX rows when
fallback PTX is used, but the latest H200 report does not use that fallback.

## CUDA Graph Launch Baseline

The `direct_driver_graph` row instantiates a one-kernel CUDA Graph before the
timed interval and measures replay of that graph. This is a host-launch
amortization baseline for repeated `host_schedule` style callables; it is not
a replacement for the persistent-device scheduler because the host still owns
graph construction and replay.

| GPU | N | Host-schedule ns | Driver ns | Driver graph ns | Graph/host | Graph/driver |
| --- | - | ---------------- | --------- | --------------- | ---------- | ------------ |
| A100 | 1024 | 32768 | 38911 | 26623 | 0.81x | 0.68x |
| H200 | 1024 | 26720 | 30848 | 29311 | 1.10x | 0.95x |
| A100 | 65536 | 35648 | 27744 | 18975 | 0.53x | 0.68x |
| H200 | 65536 | 32896 | 37184 | 24351 | 0.74x | 0.65x |
| A100 | 1048576 | 28192 | 27904 | 23360 | 0.83x | 0.84x |
| H200 | 1048576 | 34048 | 40832 | 27071 | 0.80x | 0.66x |

Graph replay is faster than raw Driver API launch on every captured row. It is
also faster than the current PTO `host_schedule` path on five of six rows; the
H200 `N=1024` row is the exception, where `host_schedule` remains lower. This
keeps CUDA Graphs useful for a phase-1 repeated-launch optimization, while
leaving the phase-2 persistent-device work focused on device-side scheduling.

## Stream Concurrency

The host-schedule stream microbenchmark prepares two independent slow
vector-add callables with different `stream_id` values, runs them serially,
then launches them concurrently from host threads. The copied-back results are
validated in both cases.

| GPU | Serial ns | Parallel ns | Parallel vs serial |
| --- | --------- | ----------- | ------------------ |
| A100 | 113838981 | 57789544 | 0.51x |
| H200 | 89797063 | 46229849 | 0.51x |

This supports keeping multiple CUDA streams in the host-schedule runtime:
independent prepared callables can overlap when issued from separate host
threads. It does not solve the persistent-device scheduling problem, where the
CUDA device-side scheduler still has to run inside a persistent kernel.

## DAG Graph Shapes

The `pto_persistent_dag_chain` row validates that the same generated-dispatch
compiled binary can run a different runtime graph descriptor: two initial
tasks fan into an add task, then a multiply task, then a final add task. This
is still a vector microbenchmark, but it is closer to the desired persistent
runtime shape than a flat descriptor array because dependencies and fan-in
counters drive the ready queue.

The `pto_persistent_dag_reuse` row adds one more task and reuses `tmp0` after
the original `tmp0` producer's final dependent has completed. This is still
elementwise vector work, but it validates the lifecycle rule needed by a
persistent-device runtime: buffer lifetime can be represented by graph
dependencies and runtime descriptors while the generated-dispatch binary stays
unchanged.

| GPU | N | DAG ns | DAG-chain ns | Chain/DAG |
| --- | - | ------ | ------------ | --------- |
| A100 | 1024 | 32768 | 34816 | 1.06x |
| H200 | 1024 | 39584 | 46528 | 1.18x |
| A100 | 65536 | 155648 | 270336 | 1.74x |
| H200 | 65536 | 140032 | 244320 | 1.74x |
| A100 | 1048576 | 2333696 | 4242432 | 1.82x |
| H200 | 1048576 | 2010240 | 3581216 | 1.78x |

The chain row is slower than the three-task DAG because it performs more
device work and serializes two more dependency levels. That is expected here;
the useful signal is that graph shape, fan-in, and callable selection are
runtime data while the generated-dispatch device binary stays stable.

| GPU | N | DAG ns | DAG-chain ns | DAG-reuse ns | Reuse/DAG |
| --- | - | ------ | ------------ | ------------ | --------- |
| A100 | 1024 | 25600 | 36864 | 34816 | 1.36x |
| H200 | 1024 | 31456 | 39232 | 39168 | 1.25x |
| A100 | 65536 | 153600 | 268288 | 266240 | 1.73x |
| H200 | 65536 | 139328 | 244128 | 245696 | 1.76x |
| A100 | 1048576 | 2328576 | 4263936 | 3947520 | 1.70x |
| H200 | 1048576 | 2005919 | 3580768 | 3581664 | 1.79x |

The reuse row is close to the chain row because it has the same long multiply
path and one additional add branch. On A100 it is slightly faster than the
chain row for larger vectors because the final add consumes the reused
scratch branch rather than the earlier chain value; this is a microbenchmark
effect, not a claim that reuse is inherently faster.

The tensor row keeps the same persistent-DAG scheduler but extends the task
descriptor ABI with rows, columns, inner dimension, leading dimensions, and
per-tile strides. Its generated-dispatch `func_id=3` computes one or more
16x16 GEMM tiles before residual, gate, and fan-in elementwise tasks. The
following rows compare the older tensor DAG capture against the three-task
elementwise DAG and the one-call host-schedule vector baseline for shape
context only. They are not same-work throughput comparisons.

| GPU | N | Host ns | DAG ns | Tensor DAG ns | Tensor/DAG |
| --- | - | ------- | ------ | ------------- | ---------- |
| A100 | 1024 | 46080 | 45056 | 36864 | 0.82x |
| H200 | 1024 | 36512 | 29120 | 38912 | 1.34x |
| A100 | 65536 | 35072 | 151552 | 586752 | 3.87x |
| H200 | 65536 | 31615 | 139616 | 566656 | 4.06x |
| A100 | 1048576 | 31008 | 2296832 | 9235456 | 4.02x |
| H200 | 1048576 | 28576 | 1997568 | 8649408 | 4.33x |

At large `N`, the tensor DAG is roughly four times slower than the simple DAG
because each output element performs a 16-term dot product before the
elementwise residual, gate, and fan-in tasks. This is expected and confirms
that the persistent-device scheduler can run non-elementwise callable bodies
without changing the launch path. A metadata-carrying tensor DAG smoke after
the descriptor extension validated `N=4096` and 16 tiles with copied-back real
CUDA data on both A100 and H200. The compact smoke report and SVG are rendered
from the raw JSON with `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py`.

| GPU | PTX arch | Device ns | Rows x Cols x Inner | Tile count |
| --- | -------- | --------- | ------------------- | ---------- |
| A100 | `compute_80` | 102400 | 16 x 16 x 16 | 16 |
| H200 | `compute_90` | 70464 | 16 x 16 x 16 | 16 |

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 8,16,32,64 \
    --label a100-wide-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-wide-$(git rev-parse --short HEAD)
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 8,16,32,64 \
     --label h200-wide-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-wide-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100-wide-e430bc1b/cuda-benchmark.json \
    tmp/cuda-backend/h200-wide-e430bc1b/cuda-benchmark.json \
    --label cuda-wide-a100-h200-e430bc1b \
    --output-dir tmp/cuda-backend/combined-wide-e430bc1b
```

DAG-chain capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-dag-323f4587 \
    --output-dir tmp/cuda-backend/a100-dag-323f4587

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-dag-323f4587 \
     --output-dir tmp/cuda-backend/h200-dag-323f4587'
```

Scratch-reuse DAG capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-reuse-bcf54a88 \
    --output-dir tmp/cuda-backend/a100-reuse-bcf54a88

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-reuse-bcf54a88 \
     --output-dir tmp/cuda-backend/h200-reuse-bcf54a88'
```

Tensor-tile DAG capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
    --label a100-tensor-8950e029 \
    --output-dir tmp/cuda-backend/a100-tensor-8950e029

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 --worker-blocks-per-task 64 \
     --label h200-tensor-8950e029 \
     --output-dir tmp/cuda-backend/h200-tensor-8950e029'
```

Extended worker-grid capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 6 \
    --worker-blocks-per-task 32,64,128,256 \
    --label a100-gridext-3eeb399a \
    --output-dir tmp/cuda-backend/a100-gridext-3eeb399a

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 6 \
     --worker-blocks-per-task 32,64,128,256 \
     --label h200-gridext-3eeb399a \
     --output-dir tmp/cuda-backend/h200-gridext-3eeb399a'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-gridext-3eeb399a/cuda-benchmark.json \
    tmp/cuda-backend/h200-gridext-3eeb399a/cuda-benchmark.json \
    --label cuda-gridext-a100-h200-3eeb399a \
    --output-dir tmp/cuda-backend/combined-gridext-3eeb399a
```

Task-count sweep capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 65536,1048576 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 2,6,12 \
    --worker-blocks-per-task 128,256 \
    --label a100-taskcount-7194bfc9 \
    --output-dir tmp/cuda-backend/a100-taskcount-7194bfc9

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 65536,1048576 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 2,6,12 \
     --worker-blocks-per-task 128,256 \
     --label h200-taskcount-7194bfc9 \
     --output-dir tmp/cuda-backend/h200-taskcount-7194bfc9'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-taskcount-7194bfc9/cuda-benchmark.json \
    tmp/cuda-backend/h200-taskcount-7194bfc9/cuda-benchmark.json \
    --label cuda-taskcount-a100-h200-7194bfc9 \
    --output-dir tmp/cuda-backend/combined-taskcount-7194bfc9
```

Wider range capture:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 16384,262144,4194304 --repeats 3 \
    --arch compute_80 --include-persistent --batch-tasks 4,8,16 \
    --worker-blocks-per-task 128,256 \
    --label a100-rangewide-cc6869f7 \
    --output-dir tmp/cuda-backend/a100-rangewide-cc6869f7

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 16384,262144,4194304 --repeats 3 \
     --arch compute_90 --include-persistent --batch-tasks 4,8,16 \
     --worker-blocks-per-task 128,256 \
     --label h200-rangewide-cc6869f7 \
     --output-dir tmp/cuda-backend/h200-rangewide-cc6869f7'

PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json \
    tmp/cuda-backend/a100-rangewide-cc6869f7/cuda-benchmark.json \
    tmp/cuda-backend/h200-rangewide-cc6869f7/cuda-benchmark.json \
    --label cuda-rangewide-a100-h200-cc6869f7 \
    --output-dir tmp/cuda-backend/combined-rangewide-cc6869f7
```

Stream concurrency:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --stream-concurrency --device 0 --repeats 7 --arch compute_80 \
    --label a100-stream-37bebf44 \
    --output-dir tmp/cuda-backend/a100-stream-37bebf44

ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --stream-concurrency --device 0 --repeats 7 --arch compute_90 \
     --label h200-stream-37bebf44 \
     --output-dir tmp/cuda-backend/h200-stream-37bebf44'
```

## Next Evaluation Gaps

- Add model-shaped kernels and more repetitions before treating any
  worker-grid setting as a tuned baseline.
- Replace the fixed 16x16 scalar GEMM body with a CUDA implementation closer
  to the intended tensor-core/tiling backend once the runtime ABI can carry
  richer tensor metadata.
