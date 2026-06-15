# aicore-notification-perf

End-to-end measurement of the **two AICore → AICPU notification paths**:

- `GM + dcci` (Normal cacheable, coherency-routed) — AICore writes a
  field in GM and flushes its L1 with `dcci`; AICPU's coherent L1 fetches
  the new line.
- `COND register` (Device-nGnRE MMIO) — AICore writes via `set_cond`
  SPR instruction; AICPU reads via MMIO LDR at
  `aic_ctrl_reg_base + core_stride * core_idx + COND_OFFSET`.

The tool reproduces Phase 13 (idle-state LDR rate) and Phase 14
(end-to-end "write → AICPU first sees" latency) from
[`docs/investigations/2026-06-cond-vs-gm-notification.md`](../../../docs/investigations/2026-06-cond-vs-gm-notification.md)
in standalone form, with no dependency on this repo's runtime
(scheduler, ringbuffer, task dispatch).

## Why this exists

The same numbers can be re-measured from the experimental phases on
the `experiment/dmb-64bit-probe` branch in the main runtime, but doing
that requires rebasing 600+ lines of probe code onto a moving
`scheduler_cold_path.cpp` and re-deriving the `hammer_go` handshake
state machine. This tool keeps the measurement primitive — and only
the measurement primitive — alive as a buildable, runnable artifact.

If you want to **add a new notification mechanism** (e.g. SDMA event,
mailbox, future fast-path register) and compare it against the
existing two, this is the right starting point: copy this directory,
add another mode to `producer.cce`, add another subtest to
`consumer.cpp`. The host launcher stays unchanged.

## Pipeline

```text
host launch.cpp                 |  AICPU consumer.cpp (block_dim=1)         |  AICore producer.cce (block_dim=1)
--------------------------------|--------------------------------------------|-----------------------------------
halMemCtl(REG_AIC_CTRL) -> base |                                            |
aclrtMalloc handshake, result   |                                            |
register producer.o             |                                            |
bootstrap consumer.so (Path A)  |                                            |
launch producer on aicore stream|                                            |  spin: dcci(go); if go == 0 wait
launch consumer on aicpu stream |  simpler_aicpu_run entered                 |
                                |  hank.mode = 0; hank.go = 1                |
                                |                                            |  see go=1, mode=0 -> GM path
                                |  for j in N: wait p_seq change, compute    |    p_tw = sys_cnt; p_seq++; dcci
                                |  hank.go = 0; sweep 10000 LDR on p_seq     |
                                |  hank.mode = 1; hank.go = 1; *cond_addr=0  |
                                |                                            |  see mode=1 -> COND path
                                |  for j in N: wait *cond_addr change        |    p_tw = sys_cnt; dcci(tw); set_cond
                                |  hank.go = 0; sweep 10000 LDR on COND      |  see go=0 -> return
                                |  write NotifPerfResult; return             |
sync streams                    |                                            |
D2H result; print table         |                                            |
```

## Files

| Path | Purpose |
| ---- | ------- |
| `shared/handshake.h` | `NotifPerfHandshake`, `NotifPerfResult`, `NotifPerfDeviceArgs` — shared across all three programs |
| `device-aicore/producer.cce` | AICore inner kernel, mode-aware tight loop |
| `device-aicore/CMakeLists.txt` | CCEC compile to `.o` |
| `device-aicpu/consumer.cpp` | AICPU SO, two-export contract (`simpler_aicpu_init`, `simpler_aicpu_run`) |
| `device-aicpu/CMakeLists.txt` | aarch64 cross-compile to `.so` |
| `host/launch.cpp` | Host orchestration |
| `host/CMakeLists.txt` | Host build |

## Build

Each of the three pieces compiles with its own CMake. The aarch64
cross toolchain is needed for the AICPU SO; `ccec` is needed for the
AICore `.o`; the host launcher uses the system compiler.

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# AICore producer (a3 dav-c220 by default; override CCE_AICORE_ARCH for a5).
cd device-aicore
cmake -B build -S . -DCCE_AICORE_ARCH=dav-c220-cube
cmake --build build
# Output: device-aicore/build/notif_perf_producer.o

# AICPU consumer (aarch64 cross compile).
cd ../device-aicpu
cmake -B build -S . \
    -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build build
# Output: device-aicpu/build/libnotif_perf_consumer.so

# Host launcher (native).
cd ../host
cmake -B build -S .
cmake --build build
# Output: host/build/launch_notif_perf
```

## Run

The host launcher needs three artifacts:

1. The CANN dispatcher SO that this repo's runtime builds at
   `build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so`. Set
   `SIMPLER_DISPATCHER_SO` to that path.
2. The consumer SO from the build above (`libnotif_perf_consumer.so`).
3. The producer `.o` from the build above (`notif_perf_producer.o`).

```bash
# In an activated venv where simpler is installed (so dispatcher exists).
export SIMPLER_DISPATCHER_SO=/path/to/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so
export NOTIF_PERF_CONSUMER_SO=$(pwd)/device-aicpu/build/libnotif_perf_consumer.so
export NOTIF_PERF_PRODUCER_O=$(pwd)/device-aicore/build/notif_perf_producer.o

# Always lock the device via task-submit on the shared dev box (see
# .claude/rules/task-submit-isolation.md).
task-submit --device auto --device-num 1 \
    --run "./host/build/launch_notif_perf \$TASK_DEVICE 0 100"
```

Arguments:

- `device_id` — required.
- `target_core_idx` — which AIC core's COND register the consumer polls (default 0). The producer always runs with `block_dim=1` on the first AIC; this index selects the COND MMIO offset on the AICPU read side.
- `n_samples` — per E2E subtest (default 100).

## Expected output (a3, ~50 MHz sys counter)

```text
=== notification-perf result ===
  consumer_rc        = 0
  magic              = 0xc0decafe  OK
  observed_p_seq     = <large; producer ran for the test duration>

  --- Phase 14: E2E AICore->AICPU latency ---
  GM   N=100  avg=~52 ticks (~1040 ns)  min=~49 (~980 ns)  max=~69 (~1380 ns)
  COND N=100  avg=~30 ticks (~600 ns)   min=~9  (~180 ns)  max=~101 (~2020 ns)

  --- Phase 13 supplemental: idle-state LDR rate (10000 LDRs) ---
  GM   LDR ticks total = ~1500   (~30000 ns)   per LDR ~ 3 ns
  COND LDR ticks total = ~52000  (~1040000 ns) per LDR ~ 104 ns
```

These match the headline numbers cited in
[`docs/hardware/mmio-performance.md`](../../../docs/hardware/mmio-performance.md).

## Interpretation

| Path | Best for | Why |
| ---- | -------- | --- |
| COND | Single-event latency (FIN signalling, etc.) | `set_cond` retires in ~10 ns; AICPU's next LDR (~100 ns) catches it |
| GM + dcci | Wide polling sweeps (low producer rate) | AICPU L1 stays warm → ~3 ns / LDR when nothing changed |

For the design decision behind production picking COND for FIN, see
the [investigation doc](../../../docs/investigations/2026-06-cond-vs-gm-notification.md).

## Caveats

- **block_dim = 1** — the producer runs on a single AIC. A more
  realistic scheduler-style measurement (one AICPU thread polling
  many AICores) needs a multi-block producer + multi-core consumer.
  The current `NotifPerfHandshake` design assumes one producer; extend
  to an array of handshakes (one per block) and a multi-core consumer
  loop to scale.
- **a2a3 register offsets** — `kNotifPerfRegSprCondOffset = 0x4C8`
  matches `src/a2a3/platform/include/common/platform_config.h`. The
  a5 chip uses `0x5108`; flip the constant in `shared/handshake.h`
  (and the CCE arch flag) for a5.
- **Throttle** — `NotifPerfHandshake.throttle_iter` (default 50 ≈ 1 µs
  spin) prevents the producer from racing ahead so far that the
  consumer's polling loop drops events. Tune up for a slower consumer,
  down for a faster one. With throttle = 0 the producer's E2E latency
  reading collapses (consumer sees only the last value of a long
  burst, not each transition).
- **Producer / consumer system counter** — both AICore `get_sys_cnt()`
  and AICPU's `mrs cntvct_el0` read the same `CNTVCT_EL0` system
  counter on a3 / a5, so the subtraction `t_obs - tw` is well-defined.
  If you port to a chip that doesn't share the counter, you need a
  different latency-measurement strategy.
