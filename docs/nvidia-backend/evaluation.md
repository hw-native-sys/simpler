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

The worker-grid data was captured from commit `e430bc1b`. The stream
concurrency data was captured from commit `37bebf44`.

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
- Evaluate the new five-task DAG-chain row on A100 and H200, then add the raw
  artifact paths and headline numbers here.
