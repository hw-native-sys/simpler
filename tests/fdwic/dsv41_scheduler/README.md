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

The seven cases are selected by `workloads/pypto_lib_latest/source.json`.
The build, export and run commands use this model by default.
Measurement availability and the timing contract are maintained in [RESULTS.md](RESULTS.md).
Refresh both HBG and percounter for every case.

Retain scheduler optimizations only when the sum of all seven case means improves,
using AICore L0 wall with one warmup and ten measured rounds per case. Include
both communicating cases; each two-device round starts after a Device
rendezvous and contributes the maximum of the ranks' local durations. L1/L3
are diagnostic and do not replace this acceptance metric.

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

| Case | Devices | Timed work |
| ---- | ------: | ---------- |
| swa | 1 | Layer 0 sliding-window attention |
| c2a_full | 1 | Layer 2 full C2A attention |
| c1a_reindex | 1 | Layer 24 C1A index refresh |
| c1a_reuse | 1 | Layer 21 C1A index reuse |
| moe | 2 | Standalone native MoE with Host token-count control |
| engram | 1 | Layer 1 Engram quantization, projection and gating |
| layers_19_21 | 2 | Layers 19 through 21, including the encoder-to-decoder transition |

Layer indices are zero-based. The six slices use the original full-forward
prefix to produce their boundary inputs.

An execution is one `(task, block, subtask)` invocation. Original scalar values,
tensor shapes, strides, offsets, aliasing, argument directions, explicit
dependencies, scope modes, block counts, and sync-start flags are recorded at
the production Host submission interface and preserved during replay. Device dispatch, ready queues,
completion, MIX admission, and `defer_wait` use the unchanged production code.

## Latest native PyPTO-lib group

The seven latest cases use the DeepSeek V4.1 Flash model at
[PyPTO-lib `4cb7442261`](https://github.com/hw-native-sys/pypto-lib/tree/4cb74422616359b3f95c598b1f9cdd31cf4e4173/models/deepseek_v4_1_flash).
Six are slices of the same 40-layer HBG capture. The seventh runs the upstream
`l3_moe` entry. All use the model's original task boundaries and synchronization.

The forward fixture uses EP2, 64 local experts per rank, 40 weight banks and 12
requests of six positions each (72 valid tokens, padded to 80). CPU tensors
`moe_num_tokens` and `attention_num_tokens` drive the generated Host orchestration.
The full forward returns `dspark_target_hidden`. Each replay checks its writable
buffers against the original prefix and slice on the same inputs.

Attention ends at the fused output projection and HC post. The layer-1 Engram
slice contains quantization, projection and gating. The layers 19–21 slice
includes the encoder-to-decoder transition. In each MoE communication phase,
one block per destination handles publication and one wait task per source
registers a counter condition when the source is remote.

The build compiles the checked-in vector-function LLVM IR with Bisheng and
links it with each matching AIV caller. The manifests reference plain `.ll`
sources and share identical functions between forward and standalone MoE.
The IR was lowered with the upstream compiler's pinned PTOAS 0.72.

The native sources and compact input initializers are checked in under
`kernels/pypto_lib_latest/` and `workloads/pypto_lib_latest/`. Build and prepare
with `--flavor pypto_lib_latest` to regenerate the seven selected cases. To refresh
topology tables from a full 40-layer capture, place its `deps.json` and matching
`name_map.json` together and run `export_workloads.py CAPTURE --flavor pypto_lib_latest`.
The exporter recognizes the fused attention output and the following MoE boundary and
checks all 40 pairs before refreshing only the selected replay regions. Native MoE remains a native descriptor.

## Native MoE Host controls

The selected `pypto_lib_latest/moe` inherits
[PR3038](https://github.com/hw-native-sys/simpler/pull/3038). It executes the
original generated `dev/hbg` `l2_moe` C++ Host orchestration, including the actual
`get_tensor_data<int32_t>(num_tokens)` Host read. Counts and routes produced by
communication remain device-side reads in AICore tasks. Frozen latest-MoE plans
are rejected.

The native MoE case measures `72/72` tokens with one warmup and ten measured
rounds. This suite is for scheduler performance: native MoE skips numerical
golden computation, output comparison and the `0/6`/`6/0` functional checks.
Its result explicitly records `numerical_validation: false`. It still requires
successful execution, the original Host control-read evidence, complete L0
AICore endpoints and per-round rank maxima after the Device start rendezvous.
The replay cases retain their inexpensive reference and output checks.

Native MoE's fixed weights and scales occupy about 2.48 GB (2.31 GiB) across
both ranks. Initial bundle preparation uses the included Torch-only input
generator and a local cache under `build/native-weights/`, keyed by model
revision, Torch version and tensor layout. It exports the original inputs to
raw local files; outputs use compact zero initializers. The prepared bundle
is reused by daily tests without importing Torch or regenerating weights.
The cache and prepared input files remain in ignored build directories and
are not committed. See [whole-test execution timings](TEST_TIME.md).

Native MoE uses the model's original numerical-golden inputs, including nonzero
weights and count-dependent forced routing. Its refreshed performance row is a
new workload baseline and must not be interpreted as a speedup or regression
against the former frozen zero-weight slice.
Host controls execute during graph construction, outside Device wall time.
The other replay cases preserve device execution for their captured controls;
they do not test dynamic Host `get_tensor_data` reads.

All seven prepared cases run with the C++ `native-runner`, standard Python and
Simpler/CANN libraries. No PyPTO, PyPTO-lib checkout, Torch or Simpler Python
bindings are needed during execution. Only initial native MoE input preparation
requires CPU Torch; the numerical helpers and generated Host/kernel sources
are included here. Existing prepared bundles avoid that preparation step.

`build.sh --flavor pypto_lib_latest` also builds the native MoE callable under
`build/portable/native-moe`. `prepare.py`/`export_cases.py` prepare its input
bundle; `--native-moe-bundle` reuses an existing standalone bundle.
`production/moe.py BUNDLE OUTPUT --compile-only --build-dir PATH` prepares
only this case; `--reuse-bundle PATH` copies a compatible standalone bundle.
Old Python-runner bundles must be prepared again once. Runtime execution
validates the callable digest, input sizes and live HOST `num_tokens` argument,
then runs the original Host callable with the supplied HBG or percounter runtime.
Each rank
submits one native call per round; the former outer Python `l3_moe` ordering
token is unnecessary with these serial per-rank submissions.

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

## Conditional ideal lower bound

The performance table uses a resource-constrained model that preserves
`sync_start` common admission as well as data dependencies. Each case uses its own L1 capture. The older
critical-path/work-volume relaxation is only a preliminary lower bound;
it does not describe their different common-start constraints.

See [BOUND.md](BOUND.md) for the model, optional solver installation,
reproduction commands and how to distinguish a proven bound from a feasible
schedule or a proved model optimum.

## Two-device timing boundary

Every two-device performance round must rendezvous on Device before the
AICPU `RunWall` timer starts. Both ranks must have enqueued the actual run
before releasing that rendezvous; a Host file barrier alone, or a separate
`HcclBarrier` followed by independent Host launches, does not establish this
boundary. Single-device cases keep their existing timing path.

Retain the AICPU entry-to-exit Device interval after the start gate as a
diagnostic, including scheduling, computation, workload communication and
shutdown. The primary comparison uses the AICore kernel window defined below.
Do not subtract estimated Host launch skew or compare counters across devices.
Run both comparison arms with the same start gate. Historical unsynchronized
two-device values remain separately labelled and cannot serve as a synchronized
baseline.

## Performance presentation contract

Performance comparisons use **L0 AICore wall** for both HBG and percounter:
the last actual kernel end minus the first actual kernel begin across active
AIC/AIV cores. Include internal scheduling and dependency gaps; exclude
AICPU startup and teardown. Use one warmup and ten measured rounds. For two
devices, rendezvous on Device before every launch, compute each device's own
duration, then take the maximum rank duration in each round before averaging.
Never subtract raw counters across devices or across AICore/AICPU clocks.

The replay launcher writes `l0-timing.json` for level `0` and `all`; its `mean_ms`
is the primary result. Native MoE writes `performance.aicore_wall_us` and
`performance.aicore_us_mean` in its L0 `result.json`, and retains the raw
`benchmark.strace.log`. Both require complete kernel endpoint fields from
`chip.run.runner_run.aicore_wall`; missing, duplicate or invalid endpoints fail
the run instead of falling back to Device wall or the old percounter loop
window. Host and Device wall fields remain auxiliary diagnostics.

Historical Device-wall measurements in existing reports retain their original
labels and cannot be reused as AICore-wall reference values. A refreshed table
must identify the runtime revision and measurement boundary for each column.

Use one row per native case. Report average milliseconds with six decimal
places using the six columns in [RESULTS.md](RESULTS.md). Identify the runtime
revisions for each measurement batch and which rows were refreshed.
Compute the percentage as `(min(HBG, percounter) / limit - 1) * 100`
before rounding to two decimal places. In the final row, sum the unrounded case
means for each runtime and the case limits. Apply the same formula to these totals. Use an em dash for unavailable values.
Mark unclosed solver bounds and their percentages with a dagger.

## Included sources and data

Everything specific to this workload lives in this directory:

```text
kernels/
  pypto_lib_latest/                  # Full-forward registrations and kernel sources
  native_moe/                        # Standalone MoE registrations and kernel sources
workloads/<flavor>/
  source.json                       # Selected cuts or native descriptors
  <case>/{tasks,edges}.tsv            # Human-readable topology reference
  vendor/                           # Plain C++ host orchestration sources
  vendor/provenance.json             # Source digest and origin
  inputs/{rank0,rank1}.txt            # Tensor layouts, initializers and scalar bindings
  inputs/vendor/data/               # Compact original input bytes, shared by digest
production/                         # Native runner, capture/replay adapters, build helpers
```

The full forward registers 81 kernels and standalone MoE registers 26.
Both preserve the original function IDs, argument directions and AIC/AIV
targets. Their plain C++ Host entry points are under `workloads/*/vendor/`.

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

This builds the small Host tools, all three sets of model kernels and the
standalone native MoE callable. Python only coordinates compilation/packing
using `simpler_setup`; PyPTO-lib, PyPTO JIT, Torch and Simpler Python bindings
are not needed to build these sources. Initial native MoE input preparation
uses CPU Torch; all prepared cases execute without it.
Optional controls:

- `--host-only` rebuilds just the native runner and Host adapters.
- `--runtime percounter` builds model kernels against Percounter's Context ABI.
  Keep its build directory separate from the default HBG artifacts.
- `DSV41_BUILD_JOBS=4` controls compiler concurrency.
- `DSV41_PYTHON=/path/to/python` selects the build helper's Python interpreter.
- `DSV41_BUILD_DIR=/path/to/build` changes the output directory. Use the same
  value for preparation and execution.

The default build directory is `$DSV41/build/portable`. Model compilation reuses
Simpler's existing content-addressed cache, including local/PTO header and compiler
identities. Build output includes `forward/pypto_lib_latest/build.json` with compiler
flags and kernel registrations, plus `name_map.json` for the production trace converter.
Capture console output if exact compiler commands are needed:

```bash
mkdir -p "$DSV41/build"
bash "$DSV41/build.sh" --pto-isa "$PWD/build/pto-isa" > "$DSV41/build/dsv41-build.log" 2>&1
```

## Prepare and run

Preparation executes the included original full forward once on two A5 devices
and captures its Host submissions. It generates the selected replay bundles,
materializes their local inputs, and prepares native MoE outside the device
reservation. Build the test-only device-start wrapper **before preparation**,
including when the selected replay case will later run on one device.

The wrapper compiles only its Host shim and small standalone gate. It copies
the original Host implementation, AICPU/AICore binaries, dispatcher and pin
metadata into an independent snapshot; rebuilding the original runtime cannot
change that snapshot. The renamed Host implementation retains identical
executable sections. Use new output directories and build a separate wrapper
for each runtime build being compared:

```bash
export DSV41_ORIGINAL_RUNTIME="$PWD/build/lib/a5/onboard"
python3 "$DSV41/production/device_start_gate.py" \
  --runtime-dir "$DSV41_ORIGINAL_RUNTIME" --output-dir "$DSV41/build/device-start-gate"
export DSV41_RUNTIME="$PWD/$DSV41/build/device-start-gate/runtime/a5/onboard"
python3 "$DSV41/production/device_start_gate.py" validate "$DSV41_RUNTIME" host_build_graph

python3 "$DSV41/production/prepare.py" "$DSV41_RUNTIME" "$DSV41/build/prepared"

bash "$DSV41/run.sh" \
  "$DSV41/build/prepared/cases/pypto_lib_latest/swa" \
  "$DSV41_RUNTIME" "$DSV41/outputs/repro/pypto_lib_latest/swa" all 1 3
```

`--cann` and `--pto-isa` select the installed SDK and PTO headers. The builder
uses that SDK's compiler to emit and decode the device bitcode. It also retains
the runtime root's `pto_isa_build.json`, so normal runtime pin validation still
applies. To reuse the same gate for another runtime build, pass `--gate-kernel`
pointing to the first wrapper's `build/device_start_gate.o`. The manifest records
source file metadata, copied component sizes and single-TU IR locations; it is
build evidence, not a hardware-validation result. It performs no file-content
hash verification. Treat the snapshot as immutable; rebuild old symlink-based
wrappers with this builder.

All two-device runs require this wrapper. Preparation validates it before
creating its output directory or launching captures. The launcher and direct
native MoE execution check the completed manifest and require regular snapshot
files with the recorded sizes for the selected runtime, dispatcher and pin.
Missing, linked or size-changed components fail with rebuild instructions; the
launcher never compiles while holding devices. Native MoE `--compile-only`
remains available with an unwrapped runtime. Single-device Attention replay
needs no gate.

To run exactly the seven native scenarios, including when an older prepared directory contains extra cases:

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
measurements, one L1 capture, and one L3 capture. `0`, `1`, `2`, `3`, and `4` select only
that production profiling level. Outputs are split by rank and run;
`result.json` confirms completion and boundary validation, and
`source-task-map.txt` maps replay task IDs to the original forward.

The launcher prepares each native run before the cross-rank Host launch barrier.
For two-device cases, the wrapper additionally queues a device-side rendezvous
before the unchanged production launch. AICPU Device timing starts after this
gate completes, excluding peer Host preparation and launch skew. A Host barrier
alone does not establish this timing boundary. The wrapper logs per-rank gate
epochs and cycle counts as evidence. Launch, wait and finalization retain the
production native lifecycle; model kernels and task scheduling are unchanged.
For a single selected profiling level, warmup uses the same level.

HBG and percounter use the same inputs and original model sources. Compile the
model kernels with the selected runtime's headers (`build.sh --runtime percounter`
for Percounter), and pack each runtime's prepared bundle with those binaries.
Percounter stores invocation Context in SSBUF; HBG uses argument suffix pointers,
so their compiled model kernels cannot be interchanged.
Pass `--runtime percounter` to `production/prepare.py` as well, using the matching
`--build-dir`; `production/materialize.py` accepts the same runtime selection.
Build both runtimes from one source revision and run each pair in the same
automatic device allocation. Select percounter with `--runtime=percounter` after the round count; the default
is `--runtime=host_build_graph`. The runner loads that runtime's subdirectory
under the supplied library root. HBG requires reported mode 1/2; percounter
requires reported mode 5. The result also records the selected runtime. Do not
mix runtime components across builds. The revision recorded in [RESULTS.md](RESULTS.md)
covers the two-device native MoE and layers 19–21 cases with percounter. Validate the
selected revision rather than carrying an older unsupported label forward;
failed or untested runs never supply successful timing values.

Percounter L1 uses the standard shared task-buffer rotation and Host recycling
paths; no diagnostic source copy or replacement recorder is needed. For example,
one SWA L1 capture after two warmups is:

```bash
bash "$DSV41/run.sh" "$PREPARED/cases/pypto_lib_latest/swa" \
  "$DSV41_RUNTIME" "$RUN_DIR/pypto_lib_latest/swa" 1 2 1 --runtime=percounter
```

Its L1 records include kernel execution and final input acquisition. Use level 2
for local execution dispatch/return timing, level 3 for percounter scheduling and
Scalar memory operations, and level 4 for all instrumented function intervals plus Host
orchestration. SSBUF context binding, dependency waits, MIX rendezvous and
sync-start work appear in L3/L4. The standard converter produces a nested AICore
Scheduler lane without changing the Worker kernel intervals. This selection does
not make unsupported task kinds runnable.

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
  "$DSV41/build/new-cases" --flavor pypto_lib_latest
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
PYTHONPATH=.:python python3 "$DSV41/analyze.py" \
  "$DSV41/outputs/repro/pypto_lib_latest" \
  "$DSV41/build/prepared/cases/pypto_lib_latest" \
  "$DSV41/build/prepared/capture/pypto_lib_latest" \
  "$DSV41/build/portable/forward/pypto_lib_latest/name_map.json" \
  "$DSV41/outputs/repro/report-pypto_lib_latest" --convert
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
