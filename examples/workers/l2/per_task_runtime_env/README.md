# `per_task_runtime_env/` — per-task ring sizing on one L2 Worker

Runs the same vector_add kernel three times on one L2 `Worker`, each with a
different `CallConfig.runtime_env` (ring buffer sizing). Ring sizing is a
**per-run** knob carried on `CallConfig` — not a process-wide env export.

## What it shows

`CallConfig.runtime_env` groups the three ring overrides as a distinct config
tier, separate from the top-level execution knobs (`block_dim`, …):

| field | unit | constraint |
| ----- | ---- | ---------- |
| `ring_task_window` | tasks | power of 2, >= 4 |
| `ring_heap` | bytes / ring | >= 1024 |
| `ring_dep_pool` | entries | 4 .. INT32_MAX |

Precedence per value: **`runtime_env` field > `PTO2_RING_*` env var >
compile-time default**. A field left at 0 (or omitted) falls back to the env
var / default.

```python
cfg = CallConfig()
cfg.runtime_env.ring_task_window = 128
cfg.runtime_env.ring_heap = 8 * 1024 * 1024   # bytes per ring
cfg.runtime_env.ring_dep_pool = 256
worker.run(chip_handle, args, cfg)
```

The three runs (`small_ring`, `large_ring`, `env_or_default`) compute the same
vector add and all pass golden — only the ring footprint differs.

## Layout

```text
per_task_runtime_env/
  main.py                 # 3 runs, one CallConfig.runtime_env each
  test_per_task_runtime_env.py
```

The kernel is reused verbatim from the sibling `../vector_add/kernels` — this
example only varies the per-run ring configuration.

## Run

```bash
python examples/workers/l2/per_task_runtime_env/main.py -p a2a3sim -d 0
```

See [`../vector_add/main.py`](../vector_add/main.py) for the full L2 lifecycle
walk-through (kernel compile, `ChipCallable` assembly, device memory, readback).
For dispatching several L2 tasks with distinct ring sizes from one launch, see
[`../../l3/per_task_runtime_env/`](../../l3/per_task_runtime_env/).
