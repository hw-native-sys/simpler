# DSV4.1 cases using the production AICore scheduler

The runner uses Simpler's production `ChipWorker`, HBG graph builder, and AICore
resident scheduler. It compiles the included original model kernel sources and loads their binaries. There is no test
scheduler, replacement arithmetic, timed busy loop, or simulated notification.
The previous reference scanner has been removed. Its historical measurements
are **not a production performance baseline**.

The source forward uses EP2, 72 valid tokens per rank (80 padded rows),
64 local experts, and the original zero-weight inputs. This is a scheduling
experiment, not checkpoint accuracy validation. Isolated Attention and Engram
cases run rank 0 on **one device**. Only `moe`, `layers_19_21`, and the original
full forward run on **two devices** with real communication and deferred waits.

The matrix contains 18 configurations: nine original PyPTO-lib cases and nine
DSL cases with approximately half their SPMD tasks requiring sync-start.
Measurement availability and the timing contract are maintained in [RESULTS.md](RESULTS.md).

## Cases

Layer indices are zero-based. The two task organizations are captured separately.
The tables in `workloads/` identify the original cuts; they are not executable
inputs and cannot replace tensor parameters or initialization data.

| Case | Original region | PyPTO-lib tasks / executions | DSL tasks / executions |
| ---- | --------------- | ---------------------------- | ---------------------- |
| `swa` | Layer 0 attention | 25 / 631 | 19 / 339 |
| `c2a_full` | Layer 2 attention | 43 / 908 | 30 / 666 |
| `c2a_reuse` | Layer 3 attention | 24 / 583 | 17 / 391 |
| `c1a_full` | Layer 20 attention | 38 / 818 | 25 / 550 |
| `c1a_reindex` | Layer 24 attention | 34 / 801 | 23 / 625 |
| `c1a_reuse` | Layer 21 attention | 24 / 583 | 17 / 391 |
| `moe` | Layer 0 MoE | 479 / 3,069 | 32 / 913 |
| `engram` | Layer 1 Engram | 12 / 960 | 12 / 960 |
| `layers_19_21` | Three consecutive layers | 1,527 / 11,218 | 159 / 4,168 |

An execution is one `(task, block, subtask)` invocation. Original scalar values,
tensor shapes, strides, offsets, aliasing, argument directions, explicit
dependencies, scope modes, block counts, and sync-start flags are recorded at
the production Host submission interface. DSL case export then applies the
half-SPMD policy below; the full-forward capture remains original. Device dispatch, ready queues,
completion, MIX admission, and `defer_wait` use the unchanged production code.

## Boundary preparation and validation

Local cases (`swa`, `c2a_full`, `c2a_reuse`, `c1a_full`, `c1a_reindex`,
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
nine `pypto_lib_dsl` cases. PyPTO-lib cases retain their original flags. There
is no third group of DSL variants.

For each case and rank, count compute submissions with `block_num > 1` as SPMD
(one logical task, including MIX). The target is `ceil(SPMD / 2)`. Preserve all
existing sync-start flags, then enable only enough unsynchronized tasks to reach
the target. If existing flags already exceed half, retain them all. Candidates
are sampled deterministically across submission order, rather than selected at
random. Reapplying the policy does not change the plan.

New flags are limited to 32 blocks for AIC/MIX and 64 blocks for pure AIV,
matching this A5 fixture's 32 AIC / 64 AIV capacity. Oversized tasks remain
unchanged, including MoE `dispatch_push`, Engram `engram_widen`, and the three
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
commands below produce exactly 18 configurations.

## Conditional ideal lower bound

Following [PR #3061](https://github.com/hw-native-sys/simpler/pull/3061), derive a
zero-scheduler-cost model from measured kernel durations and the captured
completion-dependency DAG. This model applies only to isolated single-device
cases without deferred or early dependency completion. Communication cases
are not calculated. Keep the L0 benchmark columns unchanged; the model uses
a separate HBG L1 capture of the same bundle and runtime source.

For each task, let `W_AIC` and `W_AIV` be the sums of its measured block/lane
execution durations. Its optimistic duration is
`d = max(longest block/lane duration, W_AIC / 32, W_AIV / 64)`.
Use `d` as the node weight of the DAG and compute its longest path `CP`.
The run's conditional lower bound is
`max(CP, total W_AIC / 32, total W_AIV / 64)`.
Verified Host allocation nodes have zero duration. Average the per-run bounds
over the ten measured L1 runs, excluding warmup.

This is a bound under **fixed measured durations**, not an achieved schedule
or a hardware throughput limit. Scheduling, publication, start barriers and
teardown have zero modeled cost; resource fragmentation and additional
cross-task contention are relaxed. Existing waits inside a measured kernel
remain in its duration. L1 instrumentation and cache/concurrency conditions
can change those durations, so the result is not a strict lower bound on the
separate L0 measurements and is not scaled by an L0/L1 ratio.

From the suite directory, after building the matching runtime and preparing
an isolated bundle:

```bash
bash production/run.sh "$BUNDLE" "$DSV41_RUNTIME" "$CAPTURE" 1 1 10
PYTHONPATH="$PWD/../../..:$PWD/../../../python" python3 production/ideal_bound.py \
  "$BUNDLE" "$CAPTURE/rank0" "$CAPTURE/ideal-bound.json"
```

The reducer verifies numerical success, L1 invocation identities, every
block/lane execution, stable topology and dependency endpoint coverage.
Missing or cyclic dependencies are errors; only mapped Host allocations can
supply missing zero-cost nodes. The inferred bound must fit within the
observed kernel envelope. Raw captures and per-round values remain local;
[RESULTS.md](RESULTS.md) contains only the mean in the shared table.

## Included sources and data

Everything specific to this workload lives in this directory:

```text
kernels/
  pypto_lib/manifest.json            # Function IDs, signatures, core types, source hashes
  pypto_lib/vendor/                  # Original generated kernels and projection headers
  pypto_lib_dsl/manifest.json
  pypto_lib_dsl/vendor/              # Original kernels, DSL entry.cpp and device_source.hpp
workloads/<flavor>/
  source.json                       # Nine case cuts and expected task/execution counts
  <case>/{tasks,edges}.tsv            # Human-readable topology reference
  vendor/orchestration.cpp.gz        # Exact original C++ full-forward orchestration
  vendor/provenance.json             # Uncompressed source digest and origin
  inputs/{rank0,rank1}.txt            # Tensor layouts, initializers and scalar bindings
  inputs/vendor/data/               # Compact original input bytes, shared by digest
production/                         # Native runner, capture/replay adapters, build helpers
```

There are 131 kernel registrations for PyPTO-lib and 109 for PyPTO-lib-dsl.
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

This builds the small Host tools and both sets of model kernels. Python only
coordinates compilation/packing using `simpler_setup`; neither PyPTO-lib,
PyPTO JIT, torch, nor Simpler's Python bindings is required. Execution is C++.
The generated DSL C++ uses CANN headers directly; the DSL Python package is
not needed to rebuild it.

Optional controls:

- `--flavor pypto_lib` or `--flavor pypto_lib_dsl` builds one organization.
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
and captures its Host submissions. It then generates nine replay bundles per
organization and materializes the seven local cases using their original prefixes. Start with a new output directory:

```bash
export DSV41_RUNTIME="$PWD/build/lib/a5/onboard"
python3 "$DSV41/production/prepare.py" "$DSV41_RUNTIME" "$DSV41/build/prepared"

bash "$DSV41/run.sh" \
  "$DSV41/build/prepared/cases/pypto_lib/swa" \
  "$DSV41_RUNTIME" "$DSV41/outputs/repro/pypto_lib/swa" all 1 3
```

Add `--flavor pypto_lib` or `--flavor pypto_lib_dsl` to preparation when only
one set was built. To run all 18 scenarios:

```bash
for flavor in pypto_lib pypto_lib_dsl; do
  for case in swa c2a_full c2a_reuse c1a_full c1a_reindex c1a_reuse moe engram layers_19_21; do
    bash "$DSV41/run.sh" \
      "$DSV41/build/prepared/cases/$flavor/$case" \
      "$DSV41_RUNTIME" "$DSV41/outputs/repro/$flavor/$case" all 1 3
  done
done
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
mix runtime components across builds. Percounter multi-device communication is not yet
supported: mark its MoE and layers_19_21 entries unavailable.
[RESULTS.md](RESULTS.md) lists both L0 means.

Percounter L1 uses the standard shared task-buffer rotation and Host recycling
paths; no diagnostic source copy or replacement recorder is needed. For example,
one SWA L1 capture after two warmups is:

```bash
bash "$DSV41/run.sh" "$PREPARED/cases/pypto_lib/swa" \
  "$DSV41_RUNTIME" "$RUN_DIR/pypto_lib/swa" 1 2 1 --runtime=percounter
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
  "$DSV41/build/portable/forward/pypto_lib" \
  "$DSV41/build/prepared/capture/pypto_lib" \
  "$DSV41/build/new-cases" --flavor pypto_lib
```

Before running an exported local bundle, materialize its inputs and reference:

```bash
python3 "$DSV41/production/materialize.py" \
  "$DSV41/build/new-cases/pypto_lib/swa" "$DSV41_RUNTIME" \
  "$DSV41/outputs/preparation/pypto_lib/swa"
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
for flavor in pypto_lib pypto_lib_dsl; do
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
