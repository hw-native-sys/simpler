# DSV4.1 cases using the production AICore scheduler

The runner uses Simpler's production `ChipWorker`, HBG graph builder, and AICore
resident scheduler. It compiles the included original model kernel sources and loads their binaries. There is no test
scheduler, replacement arithmetic, timed busy loop, or simulated notification.
The previous reference scanner has been removed. Its historical measurements
are **not a production performance baseline**.

The configuration is fixed at EP2, 72 valid tokens per rank (80 padded rows),
64 local experts, and the original zero-weight inputs. This is a scheduling
experiment, not checkpoint accuracy validation. Both ranks execute real
communication and deferred waits.

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
the production Host submission interface. Device dispatch, ready queues,
completion, MIX admission, and `defer_wait` use the unchanged production code.

## Boundary preparation and validation

Each case runs in three steps:

1. Once, execute the original prefix and case together to obtain reference
   values for writable entry and boundary buffers.
2. Restore mutable original inputs and execute the real prefix into persistent
   boundary buffers. This preparation is outside the measured case.
3. Execute only the selected case through the production runtime and compare
   writable buffers with the reference. Repeat preparation before each sample.

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
flags and binary hashes, plus `name_map.json` for the production trace converter.
Capture console output if exact compiler commands are needed:

```bash
bash "$DSV41/build.sh" --pto-isa "$PWD/build/pto-isa" > /tmp/dsv41-build.log 2>&1
```

## Prepare and run

Preparation executes each included original full forward once on two A5 devices
and captures its Host submissions. It then generates nine replay bundles per
organization. Start with a new output directory:

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

The wrapper uses `task-submit` to reserve two devices, or `TASK_DEVICE` within
an existing reservation. On a machine without that service, explicitly set
`DSV41_DEVICES=0,1` to two available A5 devices assigned to this run. Existing
outputs are never overwritten. `DSV41_MAX_TIME` controls the queue's per-job
runtime limit (default 300 seconds); avoid grouping all cases under a single
short reservation. `DSV41_RUNNER` optionally selects a frozen native executable.

The three arguments after the output directory are profiling level, warmup
count, and measured round count. `all 1 3` performs one warmup, three L0
measurements, one L1 capture, and one L3 capture. `0`, `1`, and `3` select only
that production profiling level. Outputs are split by rank and run;
`result.json` confirms completion and boundary validation, and
`source-task-map.txt` maps replay task IDs to the original forward.

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
per-kernel duration comparisons, L0 Device timing, and L3 phases. Device timing
includes cross-rank waits and launch skew; it is not pure Scalar scheduling
time. Recompiling with different CANN/PTO versions can change kernel durations;
compare against the full-forward capture from that same build.

Keep binary bundles, compiler logs, runtime hashes, validation results, and
traces under ignored `build/` and `outputs/` directories. The source/data package
supports functional and scheduling reproduction, not identical timing across
machines or compiler versions.
