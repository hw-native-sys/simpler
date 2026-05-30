# Current-Head Layered-Cross Capture

## Summary

Commit `743709f3` is the current CUDA backend benchmark evidence point for
human review. It adds `pto_persistent_dag_graph_layered_cross` to the selected
and full paired A100/H200 benchmark matrix.

| Capture | Artifact root | Samples | Validator preset |
| --- | --- | ---: | --- |
| Full paired matrix | `tmp/cuda-backend/current-head-full-layered-cross-fixed/combined-current-743709f3/` | 1350 | `paired-current` |
| Compact selected gate | `tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/` | 108 | `compact-current` |

## Layered-Cross Descriptor

- Dispatch ids: `1,2,11,1,2,1,6,1,1`
- Fan-in: `0,0,0,2,3,1,2,3,2`
- Dependents: `3,3,4,4,5,4,6,7,6,7,7,8,8`
- Descriptor metadata: `scalar0=2.0`, `c=a`

The row stresses multi-root scheduling, cross-layer fan-in, fan-out, and
scratch lifetime reuse in the persistent CUDA device runtime.

## Validation

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
  tmp/cuda-backend/current-head-full-layered-cross-fixed/combined-current-743709f3/cuda-benchmark.json \
  --preset paired-current
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
  tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/cuda-benchmark.json \
  --preset compact-current
```

## Review Signal

The capture is not a performance claim that persistent-device is universally
faster. It is evidence that the current CUDA backend can run and validate the
same graph descriptor on A100 and H200, preserve graph metadata in artifacts,
and compare against host-launch, CUDA Graph, and cuBLAS baselines.
