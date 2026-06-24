# `per_task_runtime_env/` — distinct ring sizes per L2 in one L3 launch

One L3 orchestration dispatches two L2 tasks, each binding its **own** ring
buffers via `CallConfig.runtime_env`. This is the headline use case the
per-task ring sizing enables: heterogeneous L2 tasks in a single launch that
each need a different ring footprint.

## What it shows

Before this knob, every L2 dispatched from one L3 shared the process-wide
`PTO2_RING_*` env and could not be sized independently. Now each
`submit_next_level` gets its own `CallConfig`:

```python
def orch_fn(orch, _args, _cfg):
    for spec in L2_TASKS:                      # one entry per L2 task
        cfg = CallConfig()
        cfg.runtime_env.ring_task_window = spec["ring_task_window"]
        cfg.runtime_env.ring_heap = spec["ring_heap"]       # bytes per ring
        cfg.runtime_env.ring_dep_pool = spec["ring_dep_pool"]
        orch.submit_next_level(chip_handle, chip_args, cfg)  # per-task config
```

The per-task config travels through the mailbox to the chip child, so each L2
binds its rings from its own values. The demo dispatches `l2_small`
(16 / 1 MiB / 64) and `l2_large` (128 / 8 MiB / 256); both run the same
vector_add and pass golden.

### Derive per-task config from the base, don't rebuild it

`runtime_env` lives on `CallConfig` alongside the diagnostics flags
(`enable_scope_stats`, `output_prefix`, …). The per-task config must preserve
any fields the harness injected on the orchestration's base config — otherwise
`--enable-scope-stats` and friends silently collect nothing for that L2. So
`_l2_config(base, spec)` copies those fields from `base` and overrides only
`runtime_env`, rather than starting from a blank `CallConfig()`.

## Layout

```text
per_task_runtime_env/
  main.py                 # 2 submit_next_level, one runtime_env each
  test_per_task_runtime_env.py
```

The kernel is reused verbatim from `../../l2/vector_add/kernels`.

## Run

```bash
python examples/workers/l3/per_task_runtime_env/main.py -p a2a3sim -d 0
```

The two L2 tasks run serially on one device. See
[`../multi_chip_dispatch/`](../multi_chip_dispatch/) for the multi-device DAG
primitives (`worker=i` pinning, `submit_sub`), and
[`../../l2/per_task_runtime_env/`](../../l2/per_task_runtime_env/) for the
single-L2 version.
