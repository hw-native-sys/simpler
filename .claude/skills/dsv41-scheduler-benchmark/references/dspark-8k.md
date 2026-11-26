# DSpark full decode at 8K

This case measures one complete synthetic DSpark decode dispatch using
PyPTO-Lib's V4.1 Flash target and drafter. Use Lib main for current upstream
measurements or the requested dev branch for branch comparisons. Verify that
the Lib HBG selection resolves to A5 Percounter on the selected PyPTO checkout.
Record the full Lib, PyPTO, loaded Simpler, and PTO ISA commits, any override of
PyPTO's Simpler pin, and the PTOAS and CANN versions. Set up the selected PyPTO
and Simpler in the active Python environment
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

## Graph Calendar key comparison on EP2

Use [dspark_calendar_hash.py](../scripts/dspark_calendar_hash.py) with the frozen
8K EP2 Percounter fixture. Its checked source transformation wraps the local
drafter Markov normalization and LM-head row selection in a Graph immediately
before the existing TP push. The workload still executes the complete target
and drafter on both ranks. Replays cycle through prefixes of 1, 6, 11 and 12
active requests per rank. The Markov normalization kernel normalizes 5, 30,
55 and 60 rows respectively and zeroes normalized padding. Physical tensor
shapes, task dependencies, kernel registrations and the 30/64-block launch
specifications stay fixed inside the Graph. Both arms use the same business
kernels.

Run `prepare` outside the device allocation for each arm, passing `--lib`,
`--fixture` (the directory with `prepared.json` and frozen inputs), `--scratch`,
`--mode exact` or `--mode bucket`, and a distinct `--output` directory. The
adapter checks the generated prologue before wrapping it and primes the binary
cache for each arm. Each replay starts a fresh Worker and restores the frozen
InOuts and active prefix before every dispatch. Set `--first-active-requests`
to rotate the first input used to train the bucket. A replay creates a fresh
`first-N` directory under its prepared arm and records the input sequence,
integer outputs per input and raw timings there. Keep any failed replay for
diagnosis by archiving its `first-N` directory before retrying.

After releasing the two devices, run `compare` with `--exact EXACT/first-N`,
`--bucket BUCKET/first-N` and `--output REPORT.json`. It checks that both ranks
executed every round with matching input sequences and outputs. The exact arm
publishes four Graph calendars per rank and the bucket arm publishes one.
Whole-decode AICore wall P50 and P95 use the slower rank in each measured
round and are also reported per input. `--reference-outputs` checks the
12-request boundary against the integer outputs of an unwrapped replay of
the full-B12 frozen fixture.

Both arms report the host AICPU enqueue skew. Large skew can dominate the
slower rank's wall time while it waits for its peer. For an aligned timing
comparison, prepare the repository's test-only device-start gate around the
same runtime binaries outside the device allocation. The gate build manifest
records unchanged runtime executable sections.
[Device start gate build and validation](../../../../tests/fdwic/dsv41_scheduler/README.md#prepare-and-run)

```bash
GATE_TOOL="$SIMPLER_ROOT/tests/fdwic/dsv41_scheduler/production/device_start_gate.py"
"$PYTHON" "$GATE_TOOL" --runtime-dir "$SIMPLER_ROOT/build/lib/a5/onboard" --output-dir "$GATE_DIR"
GATE_RUNTIME="$GATE_DIR/runtime/a5/onboard"
"$PYTHON" "$GATE_TOOL" validate "$GATE_RUNTIME" percounter
```

With `ADAPTER` pointing to `dspark_calendar_hash.py`, submit each arm and
first input as a separate device job on the same physical pair. Alternate
exact-first and bucket-first across the four pairs. Use first inputs 1, 6,
11 and 12, and retain the recorded run order, commands, raw samples and reports.
The gate option selects the validated runtime snapshot for execution. The
compiler keeps the original runtime source and cache identity.

```bash
# Set ARM to EXACT or BUCKET for this job, FIRST to the first request count,
# and DEVICE_IDS to the physical pair used for the entire comparison.
printf -v replay '%q ' "$PYTHON" "$ADAPTER" run --lib "$LIB" --output "$ARM" \
  --gate-runtime "$GATE_RUNTIME" --first-active-requests "$FIRST"
task-submit --device "$DEVICE_IDS" --timeout 3600 --max-time 1800 \
  "${env_args[@]}" --run "$replay"
```

After releasing the devices, use `compare --require-start-gate` to verify both
ranks' matching peer epochs for every dispatch and require identical runtime
and gate binaries between the arms. Each report includes the spread in local
gate-to-AICore-start offsets. `within_performance_limits` requires the bucket
relative to exact change to stay within +5% P50 and +10% P95 for every input.
Check this field for every first-input rotation.

## Capture and export Perfetto

Reuse the prepared output for the same runtime, fixture and topology. Set
`EP` to the prepared card count and retain the environment from the replay
example. Within the card allowance, submit a separate trace run, then export
on the experiment host after the reservation finishes.

```bash
LEVEL=2
printf -v trace '%q ' "$PYTHON" "$SCRIPT" trace --lib "$LIB" --output "$OUT" --trace-level "$LEVEL"
task-submit --device auto --device-num "$EP" --timeout 3600 --max-time 1800 \
  "${env_args[@]}" --run "$trace"
"$PYTHON" "$SCRIPT" export-trace --output "$OUT" --trace-level "$LEVEL"
```

L2 includes Worker View, Scheduler View, and AICPU Lifecycle. Use L3 for
scalar scheduler phases or L4 for detailed CntNotify, GM, and SSBUF activity.
See the [calendar DFX levels](../../../../docs/dfx/chip-swimlane-profiling.md#a5-percounter-calendar).
Use the converter from the Simpler checkout that ran the capture. Trace replay
uses PyPTO's own graph and timing passes and restores frozen InOut tensors
before both dispatches, including the decode position and KV caches.

Each trace run preserves the resulting `dfx_outputs` under `$OUT/traces/`.
A partial capture left in the build directory after a failed replay can be
archived separately before retrying. To convert existing recordings, pass
`--trace-root` pointing to a tree containing the matching `rank*/d*/` records.
Keep the records, dependency files and kernel name maps together.

The exporter checks each candidate against its dependency graph for complete
task and block coverage. It tries conversion in order of increasing Worker
span and checks the converted Worker View against the source clock. The
report records skipped candidates and their reasons. For L2 it writes
`$OUT/swimlanes/fastest-l2.perfetto.json` and its source summary
`fastest-l2.json` for L2. Report the exported span separately from the
fastest fixed-rank median in the performance table. Keep the raw and
converted files on the experiment host.
