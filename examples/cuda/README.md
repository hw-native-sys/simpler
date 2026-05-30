# CUDA Examples

These examples are thin wrappers around the CUDA smoke paths used by the
NVIDIA backend evaluation. They are intentionally close to the benchmark
commands so reviewers can connect examples, docs, and artifacts directly.

## Host-Schedule Vector Ops

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/host_schedule_vector_ops.py \
  --op add --n 1024 --arch compute_80
```

Use `--op` to select the evaluated host-schedule ABI shape:
`add`, `mul`, `scale`, `square`, `axpy`, `affine`, `triad`, `quad`,
`generic_args`, or `generic_args4`.

## Persistent Layered-Cross Graph

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_layered_cross.py \
  --n 1024 --arch compute_80 --scheduler-blocks 3
```

This runs the same `graph_descriptor_layered_cross` smoke shape that feeds the
current `743709f3` benchmark gate.
