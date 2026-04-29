# Worker API Examples

This directory demonstrates how to drive PTO runtime **directly through the
`Worker` class**, without going through the `@scene_test` framework.

If you want to **write and test kernels** with golden comparison, automatic
case parametrization, and pytest integration, use `@scene_test` instead — see
the examples under `examples/a2a3/` and `examples/a5/`.

If you want to **understand exactly what the framework does for you** — how a
`ChipCallable` is built from source `.cpp` files, how `TaskArgs` map to device
memory, how `Worker(level=N)` composes chips and sub-workers into a DAG — the
examples here show the full lifecycle step by step.

## Audience

These examples are written for users who are seeing the `Worker` API for the
first time. Every non-obvious line has a comment explaining **why**, and each
example's README walks through the code block by block.

If you already know `@scene_test` and just want a quick syntactic map to the
raw API, skim [`l2/hello_worker/main.py`](l2/hello_worker/main.py) first — it
is the smallest possible correct program.

## Layout

```text
workers/
  l2/                       # Single-chip examples (one NPU device)
    hello_worker/           # Worker(level=2).init().close(), no kernels
    vector_add/             # One AIV kernel, TaskArgs, golden check
  l3/                       # Multi-chip examples (host-level DAG)
    multi_chip_dispatch/    # Worker(level=3) + orchestration + SubWorker
```

Why no `tensormap_and_ringbuffer/` layer? Because every example here hard-codes
`runtime="tensormap_and_ringbuffer"` in its `Worker(...)` call — that is the
default user-facing runtime. The other runtime (`host_build_graph`) is
covered by scene tests under `tests/st/`, not here.

## Prerequisites

Examples assume you have built and installed the package in a venv:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation .
```

`pip install .` pre-builds runtime binaries into `build/lib/`, which every
example loads on `Worker.init()`. See
[`docs/developer-guide.md`](../../docs/developer-guide.md) for the full build
pipeline.

## Running

Each example has a `main.py` with uniform CLI:

```bash
python examples/workers/l2/hello_worker/main.py -p a2a3sim -d 0
python examples/workers/l2/vector_add/main.py -p a2a3sim -d 0
python examples/workers/l3/multi_chip_dispatch/main.py -p a2a3sim -d 0-1
```

Flags:

- `-p / --platform`: `a2a3sim` (simulator, no NPU needed), `a2a3` (real
  hardware), `a5sim`, `a5`. Matches the `--platform` flag on scene tests.
- `-d / --device`: device id for L2, or device range for L3 (e.g. `0-1`).

Simulator (`a2a3sim`) works on any Linux host with gcc; hardware platforms
require an Ascend NPU box with `ASCEND_HOME_PATH` set.

## Related documentation

- [`docs/hierarchical_level_runtime.md`](../../docs/hierarchical_level_runtime.md) — the L0–L6 level model
- [`docs/chip-level-arch.md`](../../docs/chip-level-arch.md) — what L2 sees
- [`docs/task-flow.md`](../../docs/task-flow.md) — end-to-end data flow
- [`python/simpler/worker.py`](../../python/simpler/worker.py) — Worker source (all comments are useful)
