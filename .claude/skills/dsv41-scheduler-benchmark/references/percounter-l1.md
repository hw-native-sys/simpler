# Capture Percounter swimlanes and export Perfetto

Use a Simpler revision that includes [#3073](https://github.com/hw-native-sys/simpler/pull/3073).
For FDWIC use `run.sh` with level `1` and `--runtime=percounter`.
For a live lib case follow [pypto-lib decode](pypto-lib.md), select the
percounter runtime for the chip worker, and pass `enable_chip_swimlane=1`
in the PyPTO run config. The daily benchmark disables profiling. Use a separate
correctness run with `PYPTO_BENCH=0` and set the swimlane flag on the config
passed to `golden.run`. Use L1 for kernel lanes. For the [A5 calendar
scheduler](../../../../docs/dfx/chip-swimlane-profiling.md#a5-percounter-calendar),
set `enable_chip_swimlane=2` to include scheduler transitions and AICPU
lifecycle. L3 adds scalar scheduler phases and L4 adds detailed memory and
notification operations. Capture at the selected level in a separate
board replay and export after releasing the reservation.

For a pinned PyPTO checkout whose codegen exposes HBG but has no percounter
selector, prepare the lib case with HBG per-task submission. Copy the generated
runtime directory to a fresh remote location and pass that directory as
`runtime_dir` to the same lib case. In the copy, set the chip
`kernel_config.py` runtime to `percounter`. Prebuild against the matching
Simpler percounter runtime outside the device allocation. Reuse the case's
golden inputs and outputs. If the harness overrides runtime lookup, verify the
host, AICPU and AICore binaries come from the pinned percounter build.
Read the scheduler mode from the chip worker after the profiled dispatch.

Verify scheduler mode 5 and
`metadata.runtime == "percounter"` in the timing capture. Collect the matching
dependency graph with dep-gen. In revisions using the shared HBG dep-gen
writer, `deps.json` can say `host_build_graph` even when the timing record says
`percounter`. Confirm the shared TaskId layout and the paired dispatch before
joining them. PyPTO's onboard swimlane runner can collect
dependencies and timings in separate passes. Keep their inputs and submitted
task identities matched, and use fresh inputs for passes that mutate arguments.
If the timing pass fails while reopening the device, keep the dependency export
and run the L1 timing pass in a fresh process using the same inputs and runtime.
Pair the files only after confirming the task IDs and execution counts match.

Keep the matching `chip_swimlane_records.json`, `deps.json`, and
`name_map*.json` together in the capture directory for the selected dispatch.
For a distributed PyPTO entry point, look under `dfx_outputs/rank*/d*/`.
Use the tools from the same Simpler checkout. With `CAPTURE` set to that
directory, export a Perfetto trace and run the critical-path tool:

```bash
python -m simpler_setup.tools.swimlane_converter \
  "$CAPTURE/chip_swimlane_records.json" \
  --deps-json "$CAPTURE/deps.json" \
  --func-names "$CAPTURE/name_map.json" \
  -o "$CAPTURE/merged_swimlane.json"
python -m simpler_setup.tools.critical_path "$CAPTURE" --stdout
```

Set the name-map path to the file generated for the selected capture. Open
`merged_swimlane.json` in Perfetto. When the merged trace is present, the tool
also writes `CPM_static.json` and `CPM_observed.json` with the corresponding
paths highlighted and `critical_path_report.md` with the timing breakdown.

For a distributed workload, select the shortest Worker View across valid
recordings of the same fixture and level. Use the first block start and last
block end from each capture and its own device clock. The L0 performance
reducer identifies the best fixed-rank median independently. Keep both rank
identities in the report. The [DSpark exporter](dspark-8k.md#capture-and-export-perfetto)
can scan existing distributed L2 recordings with `--trace-root`.

Before quoting CPM, verify every executable task joins to the dependency graph
and the per-task AIC/AIV event counts agree with block count and active slots.
Check for edges dropped during the timing join. Preserve dependencies through
kernel-less barriers by projecting their predecessor-to-successor edges onto
the timed tasks in a separate analysis directory. A creator-only endpoint
missing from the task table may be an inline host allocation. Verify this
against the actual submit path before treating it as a zero-duration root.
Unknown endpoints, unmatched dispatches, or unobserved asynchronous completion
make a complete critical path unavailable. Percounter L1 supplies task and
core record keys but does not provide the separate HBG block/subtask identity
table.

The standard Static CPM weights tasks by their measured kernel envelopes,
which can include stagger between blocks. For a conditional bound with
scheduling removed, use verified completion edges and kernel interval lengths.
The FDWIC `production/ideal_bound.py` model also uses per-task and global
AIC/AIV capacity bounds. Its bundle CLI validates an HBG-specific fixture.
Apply its model to independently validated percounter inputs rather than
passing percounter artifacts to that CLI.
