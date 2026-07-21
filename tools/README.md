# Developer Scripts

Repo-local scripts that are **not** shipped in the wheel. They assume a full
source checkout and known repo layout.

End-user profiling / debug CLIs live in
[`simpler_setup/tools/`](../simpler_setup/tools/) and ship with the wheel —
invoke them via `python -m simpler_setup.tools.<name>`.

## critical_path.py

Post-processing analysis over an L2-swimlane run. Given a directory produced by
a `--enable-l2-swimlane` run, it reconstructs the critical path of the device
execution and writes a per-task compute/stall report.

It recursively discovers every directory under the given path that holds the
full triple — a records file (`l2_swimlane_records.json`, or `l2_perf_records.json`
for older runtimes) plus `deps.json` and `name_map.json` — so it is not
restricted to a `dfx_outputs/rank*/d*` layout. For each such rank/device it
builds a happens-before DAG from the dependency graph oriented by observed
timestamps, then computes two critical paths:

- **Static CPM** — the longest duration-weighted path, i.e. the
  dependency-limited latency floor (unlimited cores).
- **Observed** — the as-executed backward "blame" walk from the last-finishing
  task; each task's compute plus the scheduling stall before it (`data-wait` =
  waiting on a producer, `core-wait` = waiting for a core to free) tiles the
  makespan exactly.

It writes `critical_path_report.md` into the given directory: an overview
(makespan, CPM floor, compute vs. stall split), a per-kernel-family table, and a
full per-task listing. Pure post-processing — no C++, no device; O(N log N).

```bash
# Analyze a case run produced with --enable-l2-swimlane
python tools/critical_path.py <run-dir>

# Options: report filename, kernel-family rows, tick tolerance, echo summary
python tools/critical_path.py <run-dir> \
    --report critical_path_report.md --top 25 --stdout
```

It auto-discovers all ranks, so you can equally point it at a single
`dfx_outputs/rank0/d0` directory or at the whole run tree.

## benchmark_rounds.sh

Batch-run a predefined set of ST examples on hardware, parse `orch_start` /
`orch_end` / `sched_end` timestamps from the device log, and report per-round
elapsed time.

```bash
# Use defaults (device 0, 10 rounds)
./tools/benchmark_rounds.sh

# Specify device / rounds / runtime
./tools/benchmark_rounds.sh -p a2a3 -d 4 -n 20 -r tensormap_and_ringbuffer
```

Requires `SIMPLER_DFX=1` in the runtime; device log must include the
`orch_*` / `sched_*` lines. The `TMR_EXAMPLE_CASES` map at the top of the
script controls which examples/cases are run.

## verify_packaging.sh

Exercises all 5 install paths × 2 entry points from a fully clean state.
CI calls this directly; see [docs/python-packaging.md](../docs/python-packaging.md).
Must run from the repo root inside an activated venv.

```bash
source .venv/bin/activate
bash tools/verify_packaging.sh
```

## cann-examples/

Standalone runnable references for the CANN host-side ACL APIs. Each
subdirectory is its own minimal CMake project — build and run on a host
with `ASCEND_HOME_PATH` set.

### cann-examples/query

Host-side device-info CLI. Subcommands wrap individual clusters of CANN
APIs (`aclrtGetDeviceCount`, `aclrtGetSocName`, `aclrtGetStreamResLimit`,
`aclrtGetMemInfo`, `aclrtGetVersion`). Treat the source as a runnable
reference for "how do I ask the driver for X?".

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
cd tools/cann-examples/query
cmake -B build .
cmake --build build

./build/query              # full overview
./build/query devices      # device count and IDs
./build/query device 0     # SoC name, AIC/AIV core counts, HBM total
./build/query mem 0        # HBM free / total / used
./build/query version      # CANN runtime version
```

### cann-examples/aicpu-device-query

Runs `halGetDeviceInfo` queries from **inside an AICPU OS process** —
resolves the "used in device" HAL queries (`AICPU + OS_SCHED`,
`AICPU + PF_*`, etc.) that always fail from host code. Uploads a small
inner SO via the same dispatcher bootstrap path the production runtime
uses; results come back through GM. Documents the resolution of the
a3 AICPU 8 → 6 split and the a5 AICPU 9 → 6 split — see the tool's own
[README](./cann-examples/aicpu-device-query/README.md) for build/run
instructions and what it confirmed.

### cann-examples/aicpu-kernel-launch

The minimum end-to-end demonstration of launching a custom AICPU kernel
from a host process using the production dispatcher bootstrap path —
no sudo, no tar.gz pre-deployment. Strips out everything specific to
this repo's runtime (ringbuffer setup, tensormap encoding, ChipWorker
fork, etc.); the inner kernel writes a magic value, an echoed token, and
one `halGetDeviceInfo` result so the readback proves end-to-end
correctness. Read this first if you want to add new AICPU work to this
repo. See the tool's own
[README](./cann-examples/aicpu-kernel-launch/README.md) for the
pipeline diagram, I/O contract, and Path A vs Path B (#822) notes.

### cann-examples/aicpu-mmio-probes

AICPU-side MMIO microbenchmarks. No AICore involvement. Measures STR
DMB cost (single + burst), STR + LDR round trip, single-thread LDR COND
serialization (same core / rotating cores), and multi-thread parallel
scaling. Reproduces Phase 4 + Phase 12 of
[`docs/hardware/mmio-performance.md`](../docs/hardware/mmio-performance.md);
the multi-thread test is the one that directly refutes "polling COND
from AICPU is sequential". See the tool's own
[README](./cann-examples/aicpu-mmio-probes/README.md) for build and
expected output.

### cann-examples/aicore-notification-perf

End-to-end measurement of the **two AICore→AICPU notification paths**:
`GM + dcci` vs `COND register (MMIO Device-nGnRE)`. Runs an AICore
producer and an AICPU consumer concurrently on two streams, computes
single-event E2E latency and idle-state polling LDR rate for both
paths. Reproduces Phase 13 + Phase 14 of
[`docs/investigations/2026-06-cond-vs-gm-notification.md`](../docs/investigations/2026-06-cond-vs-gm-notification.md)
standalone — no dependency on this repo's runtime. Use as a template
when adding a new notification mechanism that needs head-to-head
comparison with the existing two. See the tool's own
[README](./cann-examples/aicore-notification-perf/README.md) for the
pipeline diagram, build steps, and expected numbers.
