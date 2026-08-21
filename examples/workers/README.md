# Worker API Examples

This directory demonstrates how to drive the simpler runtime **directly through the
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
    worker_malloc/          # malloc/copy_to/copy_from/free round-trip, no run()
    vector_add/             # One AIV kernel, TaskArgs, golden check
  l3/                       # Multi-chip examples (host-level DAG)
    multi_chip_dispatch/    # Worker(level=3) + orchestration + SubWorker
    child_memory/           # orch.malloc + child_memory=True, weight reuse across tasks
  l4/                       # Multi-machine examples (one L3 here, one over TCP or mpirun)
    vector_add_mixed_l3/    # Worker(level=4) + add_remote_worker, golden checked on both sides
    global_tload_mixed_l3/  # Global CommDomain build + cross-machine peer TLOAD on both ranks
    compute_then_tload_mixed_l3/  # compute round on both L2s, then peer TLOAD through the same domain
    global_tload_mpirun_l3/ # one mpirun launches an L3 rank per machine; MPI descriptor exchange
    vector_add_mpi_direct_l3/ # L4 joins MPI rank 0 and controls both real L3 ranks through direct P2P
```

Why no `tensormap_and_ringbuffer/` layer? Because every example here hard-codes
`runtime="tensormap_and_ringbuffer"` in its `Worker(...)` call — that is the
default user-facing runtime. The other runtime (`host_build_graph`) is
covered by scene tests under `tests/st/`, not here.

## L4: examples that span two machines

An L4 example is the only kind here that needs **two hosts**. The parent runs
on one, holding a forked local L3, and attaches a second L3 running on the
other over TCP:

```text
L4 parent on machine B ─┬─ local  L3 on machine B  → its own NPUs
                        └─ remote L3 on machine A  → machine A's NPUs
```

Two processes, then, not one: a **daemon** on the peer and a **parent** here.
The daemon is `python -m simpler.remote_l3_worker --host H --port P` — generic,
identical for every example, nothing to write. All an example ships is the
parent side.

`global_tload_mpirun_l3` is the one exception to the daemon shape: there the
parent owns a single `mpirun` that launches an L3 rank on each machine
(`add_mpirun_worker_group`), so no daemon runs on the peer — see that
example's README for its extra prerequisites (`mpirun` + `mpi4py` on both
machines).

`vector_add_mpi_direct_l3` is the direct-MPI alternative: one supervisor
launches L4 as MPI rank 0 and the two real L3 workers as ranks 1 and 2. It also
needs `mpirun` and `mpi4py` on both machines, but task/control traffic is P2P
between L4 and each L3 rather than passing through the PR2 group mailbox.

### What a new L4 example needs

```text
l4/<your_example>/
  README.md
  kernels/aiv/*.cpp
  kernels/orchestration/*.cpp
  main.py                   # entry point: argparse + main() delegating to run()
  test_<your_example>.py    # @scene_level(SceneTestLevel.NETWORK1) wrapper collected by network1 CI
  run_parent.sh             # maps environment variables onto main.py's flags
```

Copy [`vector_add_mixed_l3/`](l4/vector_add_mixed_l3/) and work outwards from
it. Three things are load-bearing:

- **`main.py` must not be named `test_*.py`.** `pyproject.toml` sets
  `testpaths = ["tests", "examples"]`, so pytest imports anything matching that
  name. A file with no test functions collects as zero tests and looks
  harmless — until someone adds one, and the single-machine scene-test job
  starts trying to run a two-machine example.
- **The remote imports your module by path.** `REMOTE_ORCH_TARGET` is a
  `"package.module:function"` string the *peer* resolves, so renaming or moving
  the file means updating that string in the same commit — nothing on the local
  side will fail if you forget.
- **Exit non-zero when the golden check fails.** CI reads the exit code and
  nothing else.

### The environment-variable contract

`run_parent.sh` exists to turn environment variables into `main.py`'s flags for
manual two-machine runs. Pick a prefix and read these five:

| Variable | Meaning |
| -------- | ------- |
| `<PREFIX>_REMOTE` | The peer's daemon, as `host:port` |
| `<PREFIX>_LOCAL_DEVICES` | Device ids the local L3 owns |
| `<PREFIX>_REMOTE_DEVICES` | Device ids the peer's L3 owns |
| `<PREFIX>_SESSION_TIMEOUT` | Seconds to wait on the remote session |
| `<PREFIX>_SESSION_LISTEN_HOST` | Interface the parent's session runner binds |

Anything else — platform, runtime — defaults inside `run_parent.sh`.

### Running it in CI

The `st-network1-onboard-a2a3` job runs L4 examples across a pair of a2a3 machines.
The job runs one `pytest examples tests/st --level 4` sweep, so adding yours
means adding a `test_*.py` wrapper carrying
`@scene_level(SceneTestLevel.NETWORK1)`. Keep `network1_remote_device_count` when the peer
side needs more than one remote device; it declares remote resource demand,
not selection. Do not edit `_st-network1.yml`. The wiring and the log artifact are described in
[`docs/ci.md`](../../docs/ci.md#multi-machine-network1-jobs).

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
python examples/workers/l2/worker_malloc/main.py -p a2a3sim -d 0
python examples/workers/l2/vector_add/main.py -p a2a3sim -d 0
python examples/workers/l3/multi_chip_dispatch/main.py -p a2a3sim -d 0-1
python examples/workers/l3/child_memory/main.py -p a2a3sim -d 0
```

Flags:

- `-p / --platform`: `a2a3sim` (simulator, no NPU needed), `a2a3` (real
  hardware), `a5sim`, `a5`. Matches the `--platform` flag on scene tests.
- `-d / --device`: device id for L2, or device range for L3 (e.g. `0-1`).

Simulator (`a2a3sim`) works on any Linux host with gcc; hardware platforms
require an Ascend NPU box with `ASCEND_HOME_PATH` set.

L2 and L3 examples follow that uniform CLI. **L4 examples do not** — they need
a peer's address and a device split on each side, so they take `--remote`,
`--local-devices` and `--remote-devices` instead of `-p`/`-d`. Each L4 example
now also ships a `test_*.py` wrapper for pytest collection; CI uses that wrapper
and `run_parent.sh` remains the manual entry point. See each L4 example's README
for the two-machine sequence.

## Related documentation

- [`docs/hierarchical-level-runtime.md`](../../docs/hierarchical-level-runtime.md) — the L0–L6 level model
- [`docs/chip-level-arch.md`](../../docs/chip-level-arch.md) — what L2 sees
- [`docs/task-flow.md`](../../docs/task-flow.md) — end-to-end data flow
- [`python/simpler/worker.py`](../../python/simpler/worker.py) — Worker source (all comments are useful)
