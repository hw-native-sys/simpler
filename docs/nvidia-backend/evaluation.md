# CUDA Backend Evaluation Notes

This page summarizes the current evaluation evidence for the CUDA backend.
The measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They are shaped by the VDCores and MPK papers only at the evaluation
structure level: fixed GPU work, repeated problem sizes, selected launch
baselines, local A100 runs, and remote H200 runs.

The latest captured raw reports are under `tmp/`:

- `tmp/cuda-backend/a100-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/h200-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.md`
- `tmp/cuda-backend/combined-wide-e430bc1b/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/h200-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.md`
- `tmp/cuda-backend/combined-stream-37bebf44/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/h200-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.md`
- `tmp/cuda-backend/combined-dag-323f4587/cuda-benchmark.svg`
- `tmp/cuda-backend/a100-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/h200-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.md`
- `tmp/cuda-backend/combined-reuse-bcf54a88/cuda-benchmark.svg`

The worker-grid data was captured from commit `e430bc1b`. The stream
concurrency data was captured from commit `37bebf44`. The DAG-chain data was
captured from commit `323f4587`. The scratch-reuse DAG data was captured from
commit `bcf54a88`.

## Current Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
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

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | Best grid blocks/task | Best grid ratio | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | --------------------- | --------------- | ------------------------ |
| A100 | 1024 | 90112 | 0.50x | 16 | 0.49x | 0.62x |
| H200 | 1024 | 70144 | 0.50x | 64 | 0.53x | 0.58x |
| A100 | 65536 | 56928 | 1.64x | 32 | 0.52x | 1.74x |
| H200 | 65536 | 66464 | 1.22x | 64 | 0.32x | 1.27x |
| A100 | 1048576 | 74176 | 16.99x | 64 | 0.68x | 16.90x |
| H200 | 1048576 | 62590 | 19.50x | 64 | 0.57x | 19.58x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows show why the worker-grid variant matters: in the
`8,16,32,64` sweep, the best large-vector row uses 64 worker blocks per
descriptor on both GPUs. That reduces the A100 direct persistent batch row
from `16.99x` to `0.68x` versus the matched host-schedule batch row, and
reduces the H200 row from `19.50x` to `0.57x`. The middle `N=65536` rows show
the same shape: plain persistent batch and queue batch are slower than
host-schedule batch, while the best worker-grid row is faster.

## PTX Sources

The A100 rows compiled PTX with local `nvcc` for `compute_80`. The H200 rows
compiled PTX with remote `nvcc` for `compute_90`, discovered from the
`/usr/local/cuda*` toolkit path. The report still marks embedded PTX rows when
fallback PTX is used, but the latest H200 report does not use that fallback.

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

- Extend the worker-grid sweep beyond 64 blocks per descriptor and add more
  vector lengths before treating the grid row as a tuned baseline.
- Evaluate the new tensor-tile DAG row on A100 and H200, then add the raw
  artifact paths and headline numbers here.
