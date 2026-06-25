# `per_task_runtime_env/` — per-task ring sizing on one L2 Worker

Runs the same vector_add kernel several times on one L2 `Worker`, each with a
different `CallConfig.runtime_env` (ring buffer sizing) — covering both the
**scalar** form (one value broadcast to every ring) and the **per-ring** form
(each scope-depth ring sized independently). Ring sizing is a **per-run** knob
carried on `CallConfig` — not a process-wide env export.

## What it shows

`CallConfig.runtime_env` groups the ring overrides as a distinct config tier,
separate from the top-level execution knobs (`block_dim`, …). Each resource is a
single field that accepts **either** a scalar (broadcast to every ring) **or** a
4-entry list (one value per scope-depth ring):

| field | unit | constraint (per value) |
| ----- | ---- | ---------------------- |
| `ring_task_window` | tasks | power of 2 in [4, INT32_MAX] |
| `ring_heap` | bytes / ring | >= 1024 |
| `ring_dep_pool` | entries | 4 .. INT32_MAX |

A list must contain exactly **4 entries**, indexed by scope-depth ring `0..3`
(depth `>=3` maps to ring 3). A `0` entry — or a field left unset — falls
through to the next precedence tier:

```text
per-ring CallConfig entry > per-ring env > scalar env > compile-time default
```

```python
cfg = CallConfig()
# Scalar: one value broadcast to every ring.
cfg.runtime_env.ring_task_window = 128
cfg.runtime_env.ring_heap = 8 * 1024 * 1024   # bytes per ring
cfg.runtime_env.ring_dep_pool = 256

# ...or a 4-entry list: size rings 0..3 independently.
cfg.runtime_env.ring_task_window = [128, 64, 32, 16]
cfg.runtime_env.ring_heap = [8 * 1024 * 1024, 4 * 1024 * 1024, 2 * 1024 * 1024, 1 * 1024 * 1024]
cfg.runtime_env.ring_dep_pool = [256, 128, 64, 64]
worker.run(chip_handle, args, cfg)
```

The runs (`scalar_small`, `scalar_large`, `per_ring`, `env_or_default`) compute
the same vector add and all pass golden — only the ring footprint differs.
Confirm the effective per-ring sizes with `--enable-scope-stats` (the first line
of `scope_stats/scope_stats.jsonl` reports `task_window_max` / `heap_max` /
`dep_pool_max`, indexed by `ring`).

## Layout

```text
per_task_runtime_env/
  main.py                 # 4 runs, one CallConfig.runtime_env each
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
