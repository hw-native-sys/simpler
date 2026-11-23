# DSpark full decode at 8K

This case measures one complete synthetic DSpark decode dispatch using
PyPTO-Lib's V4.1 Flash target and drafter. Use the aligned fork branch
[MaxwellF1/pypto-lib dev/hbg](https://github.com/MaxwellF1/pypto-lib/tree/dev/hbg)
or a Lib checkout with the equivalent A5 runtime selection and model code.
Record the Lib, PyPTO, Simpler and PTO ISA revisions and the PTOAS and CANN
versions. Set up the selected PyPTO and Simpler in the active Python environment
using the [multi-repo setup](../../multi-repo-setup/SKILL.md).

- EP2 / TP2 uses `--ep 2 --dp 2 --lm-head-tp 2` and the reservation
  `--device auto --device-num 2`.
- EP8 / TP8 uses `--ep 8 --dp 8 --lm-head-tp 8` and the reservation
  `--device auto --device-num 8`.

Both use B12 per card, start position 8192, 72 target rows per card, and
`--experts-per-rank 12 --draft-experts-per-rank 4 --balanced-routing`.
The local expert counts use the 32-way expert sharding proportions of the
models (384/32 target, 128/32 drafter), independent of the benchmark EP group.
The helper creates the 8K fixture in memory. The model entry point retains
its short-context default. The fixture verifies execution with synthetic
inputs and supplies no independent numerical golden.

## Run

Set `SIMPLER_ROOT` and `LIB` to the two checkout paths. Use a new absolute
output directory on the experiment host with capacity for the compiled
artifacts and frozen tensors. Distributed tensor replay also needs sufficient
Host shared memory.

```bash
SCRIPT="$SIMPLER_ROOT/.claude/skills/dsv41-scheduler-benchmark/scripts/dspark_benchmark.py"
PYTHON="$(command -v python)"
EP=2
OUT=/path/to/new/dspark-ep2-8k
"$PYTHON" "$SCRIPT" prepare --lib "$LIB" --output "$OUT" --ep "$EP"
```

Preparation compiles and assembles the chip callables, verifies the artifact
runtime, and freezes the input tensors outside the allocation. Use
`inspect` in place of `prepare` to check the fixture and CLI topology alone.
When preparation succeeds, check the queue and request the topology within
the available card allowance. Pass through the same environment and
toolchain used for preparation. The replay reads the actual device IDs from
`TASK_DEVICE`.
If the replay reports a native cache miss, run `prime` on the same
prepared output on CPU and check that its cache verification passes before
requesting another reservation.

```bash
env_args=()
for key in PYTHONPATH PTOAS_ROOT PTO_ISA_ROOT PYPTO_SRC PTO_ISA_COMMIT; do
  if [[ -v $key ]]; then env_args+=(--env "$key"); fi
done
printf -v replay '%q ' "$PYTHON" "$SCRIPT" run --lib "$LIB" --output "$OUT"
task-submit --device auto --device-num "$EP" --timeout 3600 --max-time 1800 \
  "${env_args[@]}" --run "$replay"
"$PYTHON" "$SCRIPT" summarize --lib "$LIB" --output "$OUT"
```

Set `EP=8` and use a separate output directory for the eight-card run. If
the repo whitelist is too small, use the daily runner's `--ignore-whitelist`
option where the host's queue and affinity rules permit it. To prepare an
explicit TMR control, use a fresh output directory and pass
`--runtime tensormap_and_ringbuffer` to `prepare`.
Keep inputs, software pins and physical devices matched across runtime arms.
Replay uses the saved artifact runtime and rejects native compilation on a
cache miss. It restores the frozen InOut tensors before every warmup and
measured dispatch, so the cached request position starts at 8192 each time.
The restore runs before the device timing window.

## Read the timing

The helper reads warmup and rounds from the selected Lib manifest (currently
10 and 100). It enables the same device timing collection as daily CI and uses
Lib's `benchmark_payload` and `effective_us` functions. The reducer sums the
dispatch times within each rank and round, computes a median across rounds
for each fixed rank, and reports the lowest of those medians. The summary
also includes every rank's median. Preserve `prepared.json`, `raw-result.json`
and `summary.json` on the experiment host. Report missing effective windows
as unavailable. Device wall is collected for the same dispatches and uses the
same lowest-median rank aggregation, with its own metric name. The summary
also reports the median of the per-round slowest-rank sum. This uses rank
durations and does not require synchronized device clocks.
