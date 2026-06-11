# aicpu-thread-spread

Asks CANN to launch **N AICPU threads** via `rtsLaunchCpuKernel` and
reads back what `cpu_id` each thread landed on. Answers the question
"if I ask for N threads, which user cpus does CANN actually distribute
them to?" — a hard prerequisite for any device-side affinity / packing
design that wants to do post-hoc selection over the AICPU user pool.

Reuses the same dispatcher bootstrap path as
[`aicpu-device-query`](../aicpu-device-query/README.md); the only
difference is the inner SO records `sched_getcpu()` instead of running
HAL queries, and the launch is multi-threaded.

## What it answered

On a5 device 0 of one dev box (`Ascend950PR_9599`, OCCUPY=0x1f8, 6 user
cores at cpu_id 3..8):

| `aicpu_num` | cpu_ids hit (with duplicates) | observation |
| ----------- | ----------------------------- | ----------- |
| 1 | 8 | default sink cpu |
| 6 | 3 4 5 6 7 8 | exactly fills the user pool |
| 7 | 3 4 5 6 7 8 **8** | over-launch doubles up on cpu_id 8 |
| 8 | 3 4 5 6 7 **8 8 8** | three threads share cpu_id 8 |
| 14 | 3 3 4 4 5 5 6 6 7 7 **8 8 8 8** | every cpu doubled, sink quadrupled |

Conclusions (recorded in
[`src/a5/docs/hardware.md`](../../../src/a5/docs/hardware.md#cann-aicpu-thread-dispatch-under-varying-launch-budgets)):

1. CANN dispatch set = OCCUPY exactly — threads never land on a non-OCCUPY cpu_id
2. Over-launch (N > popcount(OCCUPY)) doubles up on a "sink" cpu rather than expanding the set
3. `launch_count = popcount(OCCUPY)` is the sweet spot for device-side affinity gates

This only confirms behavior on the **0x1f8 SKU** (Scenario A). The
0x7ffe SKU (Scenario B, 14 user cores) needs the same test on a chip
that exposes it — see "Run" below.

## Architecture

Three pieces, same wiring as
[`aicpu-device-query`](../aicpu-device-query/) (which reuses the
production dispatcher):

```text
+---------------------+        rtAicpuKernelLaunchExWithArgs (KFC, libaicpu_extend_kernels)
|   host launcher     | ---->  bootstrap: dispatcher SO dlopens, writes inner SO bytes to
|  aicpu_thread_spread|        /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<dev>.so
+---------------------+
          |
          | rtsBinaryLoadFromFile (JSON), rtsFuncGetByName, rtsLaunchCpuKernel(aicpu_num=N)
          v
+----------------------+
| libaicpu_thread_     |   N AICPU OS threads enter simpler_aicpu_spread:
|   spread.so          |     - sched_getcpu()
| (inner SO)           |     - claim slot via static atomic
+----------------------+     - write {thread_idx, cpu_id} to GM
          |
          | D2H aclrtMemcpy
          v
+---------------------+
|   host launcher     | pretty-print: arrival order + cpu_id histogram
+---------------------+
```

`simpler_aicpu_init` is launched single-threaded once per probe call to
reset the static slot counter — same idiom production uses
([`src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp`](../../../src/a5/runtime/host_build_graph/aicpu/aicpu_executor.cpp)
resets `thread_idx_` in `deinit`).

The dispatcher SO comes from a normal `pip install .` runtime build —
build the runtime first if you have not. Either `a5` or `a2a3` works;
both arches accept the same inner SO body.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 1. cross-compile device SO
(cd tools/cann-examples/aicpu-thread-spread/device && \
   mkdir -p build && cd build && \
   cmake .. -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
            -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++ && \
   cmake --build .)

# 2. host launcher
(cd tools/cann-examples/aicpu-thread-spread/host && \
   mkdir -p build && cd build && cmake .. && cmake --build .)
```

## Run

```bash
REPO=$PWD
export SIMPLER_DISPATCHER_SO=$REPO/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so
export SIMPLER_AICPU_SPREAD_SO=$REPO/tools/cann-examples/aicpu-thread-spread/device/build/libaicpu_thread_spread.so

# Always run hardware work via task-submit on shared dev boxes (see
# .claude/rules/running-onboard.md).
task-submit --device auto --device-num 1 \
    --run "for N in 1 6 7 8 14; do
             echo '====== launch_count='\$N' ======'
             $REPO/tools/cann-examples/aicpu-thread-spread/host/build/aicpu_thread_spread \$TASK_DEVICE \$N
           done"
```

Each run prints arrival order (`thread_idx -> cpu_id`) and a sorted
histogram of cpu_ids hit. To know your device's OCCUPY before reading
the spread, run
[`aicpu-device-query`](../aicpu-device-query/README.md) first.

## Validating Scenario B (0x7ffe SKU)

On a chip that reports `AICPU + OCCUPY = 0x7ffe` device-side (14 user
cpu_ids 1..14), the expected outcome is:

- `aicpu_num=7` lands on a subset (likely cpu_id 1..7 or a similar
  low-half bias) — the spread tool will show which
- `aicpu_num=14` should reach all 14 user cpu_ids, each exactly once
- `aicpu_num=15+` should over-saturate the sink cpu the same way

If the 14-thread launch does **not** spread to all 14 cpu_ids on a
0x7ffe SKU, that's a stronger constraint than the dispatch-equals-OCCUPY
rule observed on 0x1f8 — record the actual spread and update both this
README and `src/a5/docs/hardware.md`.

## Scope and limits

- Only the AICPU OS user pool is probed — kernel cpu placement on the
  host or AICore is unrelated.
- Single-launch per pretty-print run; static slot counter is reset by
  `simpler_aicpu_init` between launches. Concurrent `--device` clients
  would each get an independent process and an independent SO instance.
- Inner SO uses local device id `0` (`self_did = 0`) where the
  dispatcher convention requires it — same as `aicpu-device-query`.
- This tool is a diagnostic spike. The host launcher reuses
  `aicpu-device-query`'s bootstrap boilerplate verbatim; if that tool's
  load mechanism breaks, this one will too.
