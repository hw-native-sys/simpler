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
