# CUDA Backend Evaluation Notes

This page summarizes the current evaluation evidence for the CUDA backend.
The measurements are early runtime microbenchmarks, not end-to-end LLM serving
results. They are shaped by the VDCores and MPK papers only at the evaluation
structure level: fixed GPU work, repeated problem sizes, selected launch
baselines, local A100 runs, and remote H200 runs.

The latest captured raw reports are under `tmp/`:

- `tmp/cuda-backend/a100-batch-8b300fd3/cuda-benchmark.md`
- `tmp/cuda-backend/h200-batch-8b300fd3/cuda-benchmark.md`
- `tmp/cuda-backend/combined-batch-8b300fd3/cuda-benchmark.md`
- `tmp/cuda-backend/combined-batch-8b300fd3/cuda-benchmark.svg`

The data was captured from commit `8b300fd3`; the report generator was later
updated to include PTX-source disclosure at commit `150e9c38`.

## Current Baselines

- `direct_driver`: thin CUDA Driver API launch path for the same vector-add
  PTX kernel.
- `pto_host_schedule`: PTO CUDA host runtime C API and manifest dispatch.
- `pto_persistent_device`: descriptor-array persistent executor.
- `pto_persistent_queue`: scheduler block plus bounded device ring queue.
- `pto_persistent_dag`: generated-dispatch-like task selection with fan-in
  counters.
- `*_batch`: same-work rows with six vector-add task descriptors. These rows
  compare repeated host launches with one persistent launch over the same
  descriptor count.

Ratios are relative to the matched host-schedule row for the same GPU, vector
length, and task count. For batch rows, the reference is
`pto_host_schedule_batch`, not the one-task `pto_host_schedule` row.

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | ------------------------ |
| A100 | 1024 | 134144 | 0.40x | 0.35x |
| H200 | 1024 | 80704 | 0.38x | 0.42x |
| A100 | 1048576 | 75232 | 16.86x | 16.76x |
| H200 | 1048576 | 86208 | 14.27x | 14.16x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows expose the current tracer-bullet limitation:
batch rows match descriptor count, not intra-task grid shape. The persistent
executor currently uses one worker block per descriptor, while
`pto_host_schedule` vector-add uses a full grid.

## PTX Source Caveat

The A100 runs compiled PTX with local `nvcc` for `compute_80`. The H200 runs
used embedded `sm_80` fallback PTX that the H200 driver JIT compiled, because
the remote environment did not provide fresh `nvcc` compilation for
`compute_90` in this path. Treat H200 values as real execution results with
that compilation caveat, not as final Hopper-targeted codegen results.

## Reproduction Commands

Local A100:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_80 \
    --include-persistent --batch-tasks 6 \
    --label a100-batch-$(git rev-parse --short HEAD) \
    --output-dir tmp/cuda-backend/a100-batch-$(git rev-parse --short HEAD)
```

Remote H200:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bizhaoh200 \
  'cd /data/shibizhao/pto-cu && git pull --ff-only && \
   PYTHONPATH=$PWD:$PWD/python \
   python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
     --device 0 --sizes 1024,1048576 --repeats 3 --arch compute_90 \
     --include-persistent --batch-tasks 6 \
     --label h200-batch-$(git rev-parse --short HEAD) \
     --output-dir tmp/cuda-backend/h200-batch-$(git rev-parse --short HEAD)'
```

Merge reports:

```bash
PYTHONPATH=$PWD:$PWD/python \
  python3 .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --merge-json tmp/cuda-backend/a100-batch-8b300fd3/cuda-benchmark.json \
    tmp/cuda-backend/h200-batch-8b300fd3/cuda-benchmark.json \
    --label cuda-batch-a100-h200-8b300fd3 \
    --output-dir tmp/cuda-backend/combined-batch-8b300fd3
```

## Next Evaluation Gaps

- Compile H200 PTX with fresh `compute_90` tooling instead of embedded
  fallback PTX.
- Add a persistent worker-grid variant so large-vector rows compare similar
  intra-task parallelism.
- Add a higher-level task graph workload beyond vector add once the runtime
  ABI is stable enough.
