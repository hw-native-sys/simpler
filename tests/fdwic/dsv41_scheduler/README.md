# DSV4.1 cases using the production AICore scheduler

The runner uses Simpler's production `ChipWorker` and HBG graph builder, and runs
the graph on the AICore scheduler selected by `--runtime=percounter` or
`--runtime=host_build_graph` (the default). It compiles the included original model kernel sources and loads their binaries. There is no test
scheduler, replacement arithmetic, timed busy loop, or simulated notification.
The previous reference scanner has been removed. Its historical measurements
are **not a production performance baseline**.

The source forward uses EP2, 72 valid tokens per rank (80 padded rows),
64 local experts, and the original zero-weight inputs. This is a scheduling
experiment, not checkpoint accuracy validation. Isolated Attention and Engram
cases run rank 0 on **one device**. Only `moe`, `layers_19_21`, and the original
full forward run on **two devices** with real communication and deferred waits.

The matrix contains 16 configurations: three initial PyPTO-lib cases, three
half-sync DSL cases, three capacity-bounded all-sync DSL cases, and seven latest
PyPTO-lib cases. The checked-in `workloads/<flavor>/source.json` manifests own
this selection; both exporters and the run list use them.
Measurement availability and the timing contract are maintained in [RESULTS.md](RESULTS.md).

## Manual-only execution

This self-contained benchmark is maintained and run manually. Changes confined
to `tests/fdwic/dsv41_scheduler/` do not trigger automatic runtime or scene-test
CI. Pre-commit still checks the changed files. Changes to runtime code, shared
infrastructure, CI configuration, or unclassified paths retain their normal
checks. Documentation remains covered by the documentation workflow.

The suite's policy and lower-bound checks live in its own `tests/` directory,
outside the shared `tests/ut` corpus. They are marked manual for repository-wide
pytest collection. Run them explicitly without runtime bootstrap:

```bash
source .venv/bin/activate
export DSV41=tests/fdwic/dsv41_scheduler
pip install pytest -r "$DSV41/production/requirements-bound.txt"
PYTHONPATH=.:python python -m pytest --confcutdir="$DSV41/tests" "$DSV41/tests" -q
```

NPU benchmarks continue to use the automatically allocating launcher described
below. A clean automatic CI run does not establish this manual suite's hardware
correctness or update its performance results; run and record those explicitly.

## Cases

Layer indices are zero-based. The three source task organizations are captured separately.
The tables in `workloads/` identify the original cuts; they are not executable
inputs and cannot replace tensor parameters or initialization data.

| Case | Initial PyPTO-lib | DSL half | DSL all-sync | Latest PyPTO-lib | Count |
| ---- | ----------------- | -------- | ------------ | ---------------- | ----- |
| swa | - | - | - | yes | 1 |
| c2a_full | yes | yes | yes | yes | 4 |
| c1a_reindex | - | - | - | yes | 1 |
| c1a_reuse | - | - | - | yes | 1 |
| moe | yes | yes | yes | yes | 4 |
| engram | - | - | - | yes | 1 |
| layers_19_21 | yes | yes | yes | yes | 4 |

The initial and DSL groups retain their complex-DAG, MoE and multi-layer stress
cases. Latest SWA, reindex, reuse and Engram supply the remaining short-path,
index-refresh, cache-consumer and oversized-block coverage. Standalone c2a_reuse
and c1a_full are removed; layer20 remains inside layers_19_21. Full-forward
sources and kernels remain necessary for real prefixes and the multi-layer case.

An execution is one `(task, block, subtask)` invocation. Original scalar values,
tensor shapes, strides, offsets, aliasing, argument directions, explicit
dependencies, scope modes, block counts, and sync-start flags are recorded at
the production Host submission interface. DSL case export then applies the
selected synchronization policy below; the full-forward capture remains original. Device dispatch, ready queues,
completion, MIX admission, and `defer_wait` use the unchanged production code.

## Latest native PyPTO-lib group

`pypto_lib_latest` comes from the unchanged
[`dev/hbg` full forward at `a6a9a2d49047`](https://github.com/hw-native-sys/pypto-lib/tree/a6a9a2d49047ce67a85857c6b15f9604a7287714/models/deepseek_v4_1_flash).
The original `pypto_lib` identifier stays stable and is displayed as the initial
PyPTO-lib implementation. The latest group keeps the seven selected regions above;
it has its own kernels, tensor layouts, native task boundaries, dependencies,
sync-start flags and deferred communication. Its captured HBG orchestration has
no Graph submissions. No half/all synchronization policy is applied to it.

Inputs retain 12 requests, six positions per request (0 through 5), 72 valid
tokens per rank, 80 padded rows, EP2, 64 local experts, 40 weight banks, zero
weights and the original deterministic routing bias. Comparing the per-rank
input manifests, 67 common arguments have identical geometry, storage and
initial values. Seven use the latest native contracts:

| Argument | Initial contract | Latest native contract |
| -------- | ---------------- | ---------------------- |
| `hc_attn_scale`, `hc_ffn_scale` | Host storage | Device storage; same FP32 values and shapes |
| `window_cache_pool` | 512 FP8 bytes per cache entry | 544 packed UINT8 bytes including native metadata |
| `compressed_cache_pool` | 256 UINT8 bytes per entry | 320 UINT8 bytes per entry |
| `state_cache_pool` | Four state slots | Eight state slots |
| `gate_weight` | FP32 | BF16; zero values retained |
| `lm_head_weight` | `[129280, 5120]` | `[5120, 129280]`; outside the selected regions |

The added DSpark target-hidden output is retained by the full-forward signature;
its producer tasks are outside layers 19–21. The layers 19–21 slice includes the
native encoder-to-decoder transition preparation between layers 19 and 20.
MoE now uses grouped routed-expert kernels with device-side counts and routes;
its dispatch-data and combine waits preserve their deferred completion semantics. Logical
regions align across groups, but changed fusion, arithmetic kernels and cache
formats mean cross-group timing changes are not pure scheduler speedups.
Compare the HBG/percounter columns within each row for the fixed workload.

The native sources and compact input initializers are checked in under
`kernels/pypto_lib_latest/` and `workloads/pypto_lib_latest/`. Build and prepare
with `--flavor pypto_lib_latest` to regenerate the seven selected cases. To refresh
topology tables from a full 40-layer capture, place its `deps.json` and matching
`name_map.json` together and run `export_workloads.py CAPTURE --flavor pypto_lib_latest`.
The exporter recognizes alternating native attention/MoE HC boundaries and
checks all 40 pairs before refreshing only the selected replay regions. Native MoE remains a native descriptor.

## Native MoE Host controls

The selected `pypto_lib_latest/moe` inherits
[PR3038](https://github.com/hw-native-sys/simpler/pull/3038). It executes the
original `dev/hbg` `l3_moe`/`l2_moe` orchestration, including the actual
`get_tensor_data<int32_t>(num_tokens)` Host read. Counts and routes produced by
communication remain device-side reads in AICore tasks. Frozen latest-MoE plans
are rejected. Initial PyPTO-lib and DSL MoE remain fixed-input replay stress cases.

One compiled native program must pass the original numerical goldens for
per-rank token counts `0/6`, `6/0` and `72/72`, including all three outputs.
The full-count case records one warmup and ten measured rounds, Host timing,
per-rank Device timing and the per-round maximum across ranks. Compile-only
results never count as a hardware pass; the auditor rejects missing count cases,
control-read evidence, Host/Device samples or rank data.

Native MoE uses the model's original numerical-golden inputs, including nonzero
weights and count-dependent forced routing. Its refreshed performance row is a
new workload baseline and must not be interpreted as a speedup or regression
against the former frozen zero-weight slice. The other15 rows retain their data.
Host controls execute during graph construction, outside Device wall time.
The other replay cases preserve device execution for their captured controls;
they do not test dynamic Host `get_tensor_data` reads.

Prepare with a clean `dev/hbg` checkout and its compatible PyPTO/Simpler Python
environment. `prepare.py`/`export_cases.py` accept `--pypto-lib-root` and optional
`--native-moe-bundle` to reuse a verified native bundle. `production/moe.py` can
prepare just this case with `--compile-only --lib-root PATH`; add `--reuse-bundle`
to copy a compatible existing program. Runtime execution reuses the packed
native callable, checks its source and binding identity, and loads the supplied
`RUNTIME_LIB` with `--runtime=host_build_graph` or `--runtime=percounter`.
Runtime hashes and binding provenance are recorded separately. The original
model checkout is needed for numerical validation; standalone replay cases
continue to use only the included sources and inputs.

## Boundary preparation and validation

Local cases (`swa`, `c2a_full`, `c1a_reindex`,
`c1a_reuse`, and `engram`) use materialized rank-0 inputs:

1. During preparation, run the original prefix and case together to save reference
   values. Then run the prefix separately and save the actual cut inputs.
   Prefixes that contain MoE retain EP2; local prefixes such as SWA use one device.
2. Validate an isolated replay against the fused reference before publishing the
   snapshot. Reject slices that reference communication contexts or windows.
3. Reserve one device, restore the saved inputs before each sample, and execute
   only the selected case. No communicator, peer process, Host rank barrier, or
   prefix is created during these single-device runs. Compare writable bytes
   directly with the saved reference; uniform data uses one repeated byte.

The communicating cases retain two ranks and regenerate the original prefix
before each sample. The launcher reads the bundle's topology before requesting
resources and rejects mismatched or duplicate reserved devices. Unprepared local
bundles are rejected instead of silently running as EP2.

The prefix includes the real producers needed by reuse and cross-layer cases;
dependencies are not declared ready without producing their data. Buffers that
cross the cut keep their original allocation sizes. The runner omits unused
tails of external weights beyond every access in the prefix and case. Original
kernel views, offsets, strides, and accessed bytes are preserved. Consequently
the Host upload for SWA need not include all 40 layers' weights.

Separating launches changes the scheduling context and may change cache warmth.
Original binaries and data geometry are necessary but do not prove identical
kernel durations: compare L1 intervals with the source forward before using a
case to draw whole-network conclusions. Prefix timing is never included in a
case's scalar scheduling measurements.

## DSL half-SPMD sync-start policy

`production/prepare.py` applies this policy automatically when exporting the
three `pypto_lib_dsl` cases; it also exports the all-sync variant. PyPTO-lib cases retain their original flags. There
are four groups: `pypto_lib` (initial), `pypto_lib_dsl` (half),
`pypto_lib_dsl_all_sync64` (all up to 64), and `pypto_lib_latest` (native latest).

For each case and rank, count compute submissions with `block_num > 1` as SPMD
(one logical task, including MIX). The target is `ceil(SPMD / 2)`. Preserve all
existing sync-start flags, then enable only enough unsynchronized tasks to reach
the target. If existing flags already exceed half, retain them all. Candidates
are sampled deterministically across submission order, rather than selected at
random. Reapplying the policy does not change the plan.

New flags are limited to 32 blocks for AIC/MIX and 64 blocks for pure AIV,
matching this A5 fixture's 32 AIC / 64 AIV capacity. Oversized tasks remain
unchanged, including MoE `dispatch_push` and the three
`dispatch_push` tasks in layers 19–21. If insufficient tasks fit the capacity,
the report records the actual count below the target.

Export invokes `record-tool --half-sync-start INPUT FIRST_TASK LAST_TASK OUTPUT`.
Only sync-start fields within the selected case change; the prefix, suffix,
block counts, kernels, arguments and dependencies retain their original bytes.
Both rank plans are processed. `case.json` records the policy, before/after
counts, capacity exclusions and selected source task IDs per rank. The
original files under `workloads/` remain source-capture references.

Regenerate prepared bundles after updating the exporter. Existing bundles do
not acquire this policy automatically. The standard preparation and run
commands below produce 16 configurations by default. Use `--sync-policy half`
to prepare only the half-sync DSL variant, or `--sync-policy all-cap64` for
only the all-sync DSL variant. Both PyPTO-lib groups retain their native flags.

The all-sync variant uses `record-tool --all-sync-start-cap64` and enables
every capacity-eligible multi-block compute task in the selected slice.
The limit is inclusive: 64 for AIV, 32 for AIC/MIX. In this fixture that
enables 167 of 171 SPMD tasks; four oversized dispatch tasks remain unchanged. The half-sync group enables 87 of 171 tasks.
Both variants retain the original flavor in `case.json` and identify the
selected variant separately, so source workload tables and the lower-bound
analyzers continue to use the same task identities.

## Conditional ideal lower bound

The performance table uses a resource-constrained model that preserves
`sync_start` common admission as well as data dependencies. Half-sync and
all-sync cases have separate L1 captures and separate bounds. The older
critical-path/work-volume relaxation is only a preliminary lower bound;
it does not describe their different common-start constraints.

See [BOUND.md](BOUND.md) for the model, optional solver installation,
reproduction commands and how to distinguish a proven bound from a feasible
schedule or a proved model optimum.

## Performance presentation contract

Use one table, with each case followed by initial PyPTO-lib, half-sync DSL,
all-sync DSL, then latest PyPTO-lib. Keep only the current result set, average milliseconds with six
decimal places, and the following fixed columns (see [RESULTS.md](RESULTS.md)):

Use the exact six-column header shown in [RESULTS.md](RESULTS.md).

The HBG column contains the selected HBG revision; always state its exact
commit beside the table, independently of the fixed column label. The current
published HBG values use `534c5656a`, and percounter uses `21d2e9a103d5`.
Do not add a duplicate candidate column. When preserving an existing group while
adding another, state which measurements were reused and which were refreshed. The percentage is `(min(HBG, percounter) / bound - 1) * 100`,
computed before rounding and displayed with two decimal places. Missing
bounds and communication cases use an em dash. Mark unclosed solver bounds
and their percentages with a dagger; never substitute an incumbent as an
exact lower bound. Do not publish raw samples or P99 columns in this table.

## Included sources and data

Everything specific to this workload lives in this directory:

```text
kernels/
  pypto_lib/manifest.json            # Function IDs, signatures, core types, source hashes
  pypto_lib/vendor/                  # Original generated kernels and projection headers
  pypto_lib_dsl/manifest.json
  pypto_lib_dsl/vendor/              # Original kernels, DSL entry.cpp and device_source.hpp
workloads/<flavor>/
  source.json                       # Selected cuts or native descriptors
  <case>/{tasks,edges}.tsv            # Human-readable topology reference
  vendor/orchestration.cpp.gz        # Exact original C++ full-forward orchestration
  vendor/provenance.json             # Uncompressed source digest and origin
  inputs/{rank0,rank1}.txt            # Tensor layouts, initializers and scalar bindings
  inputs/vendor/data/               # Compact original input bytes, shared by digest
production/                         # Native runner, capture/replay adapters, build helpers
```

There are 131 kernel registrations for initial PyPTO-lib, 109 for PyPTO-lib-dsl,
and 82 for latest PyPTO-lib.
The original DSL forward is mixed: 83 registrations use 52 DSL-generated
`entry.cpp` / `device_source.hpp` pairs under `vendor/external/`; the other 26
use the original PTO-generated sources under `vendor/aic/` and `vendor/aiv/`
(for example, Engram and communication/wait kernels). `external` retains the
original registration category; these sources are now included locally.
The DSL-generated kernels have not been replaced with PTO implementations.
Shared sources are stored once; registrations retain their original function
IDs, argument directions, and AIC/AIV targets. Generated source is kept verbatim
under `vendor/`. The original Host source is losslessly compressed because the
larger file exceeds this repository's individual-file size limit. It is
expanded into `build/` before compilation and can be read with `gzip -dc`.

Uniform zero/one initializers are stored as one element plus a repeat count;
nonuniform metadata is stored in full. This avoids checking in tens of GB of
materialized weights without changing the bytes uploaded to the devices.
No old `logs/`, `artifacts/`, model repository, or JIT cache is required.

## Build from source

Use a Linux A5 environment with CANN, CCEC, a C++17 host compiler, CMake, Ninja,
Python 3.10 or newer, PTO-ISA headers, and an already built **matching Simpler
production runtime**. Use the repository's normal setup instructions to obtain
that runtime if needed; its usual location is `build/lib/a5/onboard`.
Retain its `host_build_graph/{libhost_runtime.so,libaicpu_kernel.so,aicore_kernel.o}`
and the sibling `../dispatcher/libsimpler_aicpu_dispatcher.so` when copying a
runtime build. The runtime, native runner, and capture/replay adapters must
use compatible Simpler headers. This workflow never rebuilds the scheduler automatically.

From the Simpler repository root:

```bash
source /path/to/cann/set_env.sh
export DSV41=tests/fdwic/dsv41_scheduler
bash "$DSV41/build.sh" --pto-isa "$PWD/build/pto-isa"
```

This builds the small Host tools and all three sets of model kernels. Python only
coordinates compilation/packing using `simpler_setup`; neither PyPTO-lib,
PyPTO JIT, torch, nor Simpler's Python bindings is required for replay cases. Native MoE additionally uses the original model and golden harness.
The generated DSL C++ uses CANN headers directly; the DSL Python package is
not needed to rebuild it.

Optional controls:

- `--flavor pypto_lib`, `--flavor pypto_lib_dsl`, or `--flavor pypto_lib_latest` builds one organization.
- `--flavor all` is the default; `--flavor both` selects the initial PyPTO-lib and DSL sources.
- `--host-only` rebuilds just the native runner and Host adapters.
- `DSV41_BUILD_JOBS=4` controls compiler concurrency.
- `DSV41_PYTHON=/path/to/python` selects the build helper's Python interpreter.
- `DSV41_BUILD_DIR=/path/to/build` changes the output directory. Use the same
  value for preparation and execution.

The default build directory is `$DSV41/build/portable`. Model compilation reuses
Simpler's existing content-addressed cache, including local/PTO header and compiler
identities. SDK headers use the toolchain/package identity for the DSL entries,
following the existing compiler cache policy for SDK-provided headers. Build output includes `forward/<flavor>/build.json` with compiler
flags and kernel registrations, plus `name_map.json` for the production trace converter.
Capture console output if exact compiler commands are needed:

```bash
bash "$DSV41/build.sh" --pto-isa "$PWD/build/pto-isa" > /tmp/dsv41-build.log 2>&1
```

## Prepare and run

Preparation executes each included original full forward once on two A5 devices
and captures its Host submissions. It generates the selected replay bundles, materializes their local inputs, and prepares native MoE outside the device reservation. Start with a new output directory:

```bash
export DSV41_RUNTIME="$PWD/build/lib/a5/onboard"
python3 "$DSV41/production/prepare.py" "$DSV41_RUNTIME" "$DSV41/build/prepared" \
  --pypto-lib-root /path/to/pypto-lib-dev-hbg

bash "$DSV41/run.sh" \
  "$DSV41/build/prepared/cases/pypto_lib_latest/swa" \
  "$DSV41_RUNTIME" "$DSV41/outputs/repro/pypto_lib_latest/swa" all 1 3
```

Pass the same `--flavor` selection to preparation when building only a subset.
To run exactly the selected 16 scenarios, including when an older prepared directory contains extra cases:

```bash
while IFS= read -r selected; do
  bash "$DSV41/run.sh" \
    "$DSV41/build/prepared/cases/$selected" \
    "$DSV41_RUNTIME" "$DSV41/outputs/repro/$selected" 0 1 10
done < <(python3 "$DSV41/production/list_cases.py")
```

The wrapper uses `task-submit` to reserve the bundle's required number of devices,
using `task-submit --device auto --device-num N`, or validates `TASK_DEVICE`
within an existing reservation. Use the actual assigned IDs from `TASK_DEVICE`
and record them with the results.
Do not wrap all cases in a single EP2 reservation: that would hold an unused card
while running Attention. Existing
outputs are never overwritten. `DSV41_MAX_TIME` controls the queue's per-job
runtime limit (default 300 seconds); avoid grouping all cases under a single
short reservation. `DSV41_RUNNER` optionally selects a frozen native executable.

Append `--runtime=percounter` to select the percounter runtime directory.
Build it explicitly first: `python -m simpler_setup.build_runtimes --platforms a5 --runtime percounter`
(or set `SIMPLER_BUILD_RUNTIMES=percounter` during installation).
The default runtime build does not include percounter.

The three arguments after the output directory are profiling level, warmup
count, and measured round count. `all 1 3` performs one warmup, three L0
measurements, one L1 capture, and one L3 capture. `0`, `1`, and `3` select only
that production profiling level. Outputs are split by rank and run;
`result.json` confirms completion and boundary validation, and
`source-task-map.txt` maps replay task IDs to the original forward.

The launcher prepares each native run before the cross-rank launch barrier,
then launches, waits and finalizes through the production native lifecycle.
The peer's Host preparation is therefore outside the measured Device interval.
For a single selected profiling level, warmup uses the same level.

HBG and percounter use the same prepared bundles and original model kernels.
Build both runtimes from one source revision and run each pair in the same
automatic device allocation. Select percounter with `--runtime=percounter` after the round count; the default
is `--runtime=host_build_graph`. The runner loads that runtime's subdirectory
under the supplied library root. HBG requires reported mode 1/2; percounter
requires reported mode 5. The result also records the selected runtime. Do not
mix runtime components across builds. The revision recorded in [RESULTS.md](RESULTS.md)
passed all six multi-device MoE/layers variants with percounter. Validate the
selected revision rather than carrying an older unsupported label forward;
failed or untested runs never supply successful timing values.

Percounter L1 uses the standard shared task-buffer rotation and Host recycling
paths; no diagnostic source copy or replacement recorder is needed. For example,
one SWA L1 capture after two warmups is:

```bash
bash "$DSV41/run.sh" "$PREPARED/cases/pypto_lib_latest/swa" \
  "$DSV41_RUNTIME" "$RUN_DIR/pypto_lib_latest/swa" 1 2 1 --runtime=percounter
```

Its L1 records include kernel execution and final input acquisition, not earlier
payload preparation or dependency waits. This selection does not make unsupported
task kinds runnable.

The local `plan-rank*.bin` records use current production C++ types. They are
**derived build artifacts**, not portable runtime protocols. Regenerate them
from the included original C++ after changing the runtime ABI, Host adapters,
model sources, or inputs. Rebuilding kernels and preparing a new directory
needs no model Python environment. Changing only a cut reuses the captured
full forward and the same replay adapter:

```bash
python3 "$DSV41/production/export_cases.py" \
  "$DSV41/build/portable/forward/pypto_lib_latest" \
  "$DSV41/build/prepared/capture/pypto_lib_latest" \
  "$DSV41/build/new-cases" --flavor pypto_lib_latest \
  --pypto-lib-root /path/to/pypto-lib-dev-hbg
```

Before running an exported local bundle, materialize its inputs and reference:

```bash
python3 "$DSV41/production/materialize.py" \
  "$DSV41/build/new-cases/pypto_lib_latest/swa" "$DSV41_RUNTIME" \
  "$DSV41/outputs/preparation/pypto_lib_latest/swa"
```

`prepare.py` performs this conversion automatically. Snapshot data is specific to
the included fixture, source capture, and compatible runtime ABI. Regenerate it
when those inputs change; it is not a model checkpoint or a numerical accuracy
reference for arbitrary inputs.

The Host adapter occupies the original orchestration slot, with trailing ELF
padding, so model kernels retain their original offsets in the uploaded
callable image. Packing rejects an adapter larger than that slot. This does
not promise identical absolute device addresses or cache state across launches.

## Audit and compare

Audit the generated graphs and convert L1/L3 traces with the production tools:

```bash
for flavor in pypto_lib pypto_lib_dsl pypto_lib_latest; do
  PYTHONPATH=.:python python3 "$DSV41/analyze.py" \
    "$DSV41/outputs/repro/$flavor" \
    "$DSV41/build/prepared/cases/$flavor" \
    "$DSV41/build/prepared/capture/$flavor" \
    "$DSV41/build/portable/forward/$flavor/name_map.json" \
    "$DSV41/outputs/repro/report-$flavor" --convert
done
```

The report checks physical `(task, block, subtask)` invocation sets, tensor
arguments, internal dependencies, and capture completeness. It includes
per-kernel duration comparisons, L0 Device timing, and L3 phases. The Host launch barrier uses bounded busy polling so a timer sleep does not
become device-side peer wait. Binding still occurs after that barrier; Host
`chip.run.runner_run` timestamps expose any remaining rank launch skew. Device timing
includes cross-rank waits and launch skew; it is not pure Scalar scheduling
time. Recompiling with different CANN/PTO versions can change kernel durations;
compare against the full-forward capture from that same build.

Keep binary bundles, compiler logs, runtime provenance, validation results, and
traces under ignored `build/` and `outputs/` directories. The source/data package
supports functional and scheduling reproduction, not identical timing across
machines or compiler versions.
