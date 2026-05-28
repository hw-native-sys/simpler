# Developer Scripts

Repo-local scripts that are **not** shipped in the wheel. They assume a full
source checkout and known repo layout.

End-user profiling / debug CLIs live in
[`simpler_setup/tools/`](../simpler_setup/tools/) and ship with the wheel —
invoke them via `python -m simpler_setup.tools.<name>`.

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

Requires `PTO2_PROFILING=1` in the runtime; device log must include the
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
a3 AICPU 8 → 6 split — see the tool's own
[README](./cann-examples/aicpu-device-query/README.md) for build/run
instructions and what it confirmed.
