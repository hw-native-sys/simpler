# aicpu-mmio-probes

AICPU-side MMIO microbenchmarks against the chip's AIC_CTRL register
window. AICore is not involved — every number measures **what happens
when the AICPU CPU issues a load / store at this MMIO address**.

Reproduces three measurements from
[`docs/hardware/mmio-performance.md`](../../../docs/hardware/mmio-performance.md):

- **Phase 4** — STR DMB burst throughput (~5 ns / STR posted) and
  STR + LDR round-trip (~250–300 ns; the LDR drains the in-flight queue
  because `Device-nGnRE` `nR` ordering forbids overlap).
- **Phase 12-A** — Single AICPU thread, 10000 LDR COND at the same
  core. Per-LDR ~95 ns; the bus round trip dominates.
- **Phase 12-B** — Single AICPU thread, 10000 LDR COND rotating across
  N AIC cores. Same per-LDR cost as A — target-switching is free, but
  `nR` still forbids outstanding LDRs.
- **Phase 12-C** — M ∈ [1..4] AICPU pthreads, each spinning a distinct
  AIC core's COND for 10000 iter. Per-thread cost stays ~95 ns
  regardless of M — proves the LDR bus is per-target, not chip-shared.
  This is the result that refutes the colloquial claim "polling COND
  from AICPU is sequential" (which is true for *one thread*, false
  across threads).

For the GM-vs-COND notification path comparison (Phase 13 + 14), see
[`tools/cann-examples/aicore-notification-perf/`](../aicore-notification-perf/) —
that one needs an AICore producer running concurrently.

## Why this exists

The same numbers can be re-measured from the experimental phases on the
`experiment/dmb-64bit-probe` branch in the main runtime, but doing
that requires rebasing 600+ lines of probe code onto a moving
`scheduler_cold_path.cpp` and re-deriving the `hammer_go` handshake
state machine. This tool keeps the measurement primitive — and only
the measurement primitive — alive as a buildable, runnable artifact.

If you want to **add a new MMIO probe** (e.g. burst LDR pattern, or a
new register offset), this is the right starting point: edit
`device/probes.cpp`, extend `shared/probes_types.h` with new result
fields, update the host pretty-printer. The dispatcher / bootstrap
plumbing in `host/launch.cpp` stays untouched.

## Pipeline

```text
host launch.cpp                                        |  AICPU probes.cpp (block_dim=1)
-------------------------------------------------------|---------------------------------
halMemCtl(REG_AIC_CTRL) -> aic_ctrl_reg_base           |
aclrtMalloc dev_args, dev_result                       |
H2D dispatcher + inner SO + DeviceArgs(bootstrap)      |
rtAicpuKernelLaunchExWithArgs(KFC, libaicpu_extend...) |
fingerprint inner SO + JSON descriptor                 |
rtsBinaryLoadFromFile + rtsFuncGetByName               |
rewrite DeviceArgs(run) with reg_base + result_addr    |
rtsLaunchCpuKernel(run_handle)                         |
                                                       |  simpler_aicpu_run entered
                                                       |  RunPhase4: burst STR, round trip
                                                       |  RunPhase12 single-thread: A + B
                                                       |  RunPhase12 multi-thread: pthread M=1..4
                                                       |  write MmioProbeResult; return
sync stream                                            |
D2H result; pretty-print Phase 4 / 12 table            |
```

## Files

| Path | Purpose |
| ---- | ------- |
| `shared/probes_types.h` | `MmioProbeResult`, `MmioProbeDeviceArgs`, register offsets |
| `device/probes.cpp` | AICPU SO, two-export contract (`simpler_aicpu_init`, `simpler_aicpu_run`), pthread Phase 12-C |
| `device/CMakeLists.txt` | aarch64 cross-compile to `.so`, links pthread |
| `host/launch.cpp` | Host orchestration — Path-A dispatcher bootstrap + halMemCtl + pretty-print |
| `host/CMakeLists.txt` | Host build |

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# AICPU probe SO (aarch64 cross compile).
cd device
cmake -B build -S . \
    -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build build
# Output: device/build/libaicpu_mmio_probes.so

# Host launcher.
cd ../host
cmake -B build -S .
cmake --build build
# Output: host/build/launch_mmio_probes
```

## Run

```bash
# The dispatcher SO is built by this repo's runtime pipeline.
export SIMPLER_DISPATCHER_SO=/path/to/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
export AICPU_MMIO_PROBES_SO=$(pwd)/device/build/libaicpu_mmio_probes.so

# Lock device via task-submit on the shared dev box.
task-submit --device auto --device-num 1 \
    --run "./host/build/launch_mmio_probes \$TASK_DEVICE 3"
```

Arguments:

- `device_id` — required.
- `n_aic_cores` — how many AIC cores to probe (default 3, matches the
  `vector_example` block_dim used in experiments; cap is
  `kProbeMaxCores = 8` in `shared/probes_types.h`).

## Expected output (a3, ~50 MHz sys counter)

```text
=== aicpu-mmio-probes result ===
  probe_rc           = 0
  magic              = 0xabcd1234  OK
  AICPU pid          = <small int>

  --- Phase 4: STR DMB ---
    burst N=1000  total=~250 ticks (~5000 ns)  per=0.25 ticks (~5 ns/STR)
    STR+LDR round trip = ~13 ticks (~260 ns)

  --- Phase 12: LDR COND serialization ---
    A: 1 thread, same core    per=4.76 ticks (~95 ns/LDR)
    B: 1 thread, rotate 3 cores  per=4.77 ticks (~95 ns/LDR)
    C: M=1 threads (each own core)
         thread=0 ... per=4.75 ticks (~95 ns/LDR)
    C: M=2 threads (each own core)
         thread=0 ... per=4.76 ticks (~95 ns/LDR)
         thread=1 ... per=4.91 ticks (~98 ns/LDR)
    C: M=3 threads (each own core)
         thread=0 ... per=4.75 ticks (~95 ns/LDR)
         thread=1 ... per=4.91 ticks (~98 ns/LDR)
         thread=2 ... per=4.38 ticks (~87 ns/LDR)
```

The per-thread cost holding flat as M grows from 1 → 3 is the
parallel-scaling result. If you ever see per-thread cost growing
linearly with M, something on the bus has gone wrong (or you ported to
silicon that does serialise multi-thread LDR — at which point the
chip-architecture doc needs an update).

## Caveats

- **a2a3 register offsets** (`DMB=0xA0`, `COND=0x4C8`) are hard-coded
  in `shared/probes_types.h`. The a5 chip uses `0xD0` / `0x5108`; flip
  the constants there for an a5 port.
- **kProbeMaxConcurrentReaders = 4** in `shared/probes_types.h`. A
  fuller scaling sweep up to 8 or 16 threads needs to grow that and
  the on-device result array — but watch the AICPU OS pthread limits
  (a3's AICPU OS scheduler historically owns cpu_id 0 only; see
  `tools/cann-examples/aicpu-device-query`).
- **Single block** (block_dim = 1, runs on one AICPU thread group).
  This tool measures the CPU-side cost of MMIO ops, not scheduler
  throughput. A real "polling N cores per round" benchmark needs to
  pair with the production runtime's AICore handshake.
- **Don't run on a critical chip.** If you've just been running a tool
  that hangs the chip (the AICore-side MMIO probe in
  `2026-06-aicore-mmio-to-spr.md`), the device shows up `Critical` in
  `npu-smi info` and `task-submit --device auto` may still pick it.
  Use an explicit healthy device id.
