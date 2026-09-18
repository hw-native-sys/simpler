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
host launch.cpp                 |  AICPU consumer.cpp (block_dim=1)          |  AICore producer.cce (block_dim=1)
--------------------------------|--------------------------------------------|-----------------------------------
halMemCtl(REG_AIC_CTRL) -> base |                                            |
aclrtMalloc handshake, result   |                                            |
register producer.o             |                                            |
bootstrap consumer.so (Path A)  |                                            |
launch producer on aicore stream|                                            |  spin: dcci(line 0); wait go == 1
launch consumer on aicpu stream |  simpler_aicpu_run entered                 |    (bounded, ~5 s)
                                |  reset line 1 + line 2; clean line 2       |
                                |  hank.mode = 0; hank.go = 1; clean line 0  |
                                |                                            |  see go=1 -> publish core_id (line 2)
                                |  poll core_id_valid (bounded, ~1 s)        |
                                |                                            |  mode=0 -> GM path
                                |  for j in N: wait p_seq change, compute    |    p_tw = sys_cnt; p_seq++; dcci line 1
                                |  hank.mode = 1; clean line 0; *cond_addr=0 |
                                |                                            |  see mode=1 -> COND path
                                |  for j in N: wait *cond_addr change        |    p_tw = sys_cnt; dcci line 1; set_cond
                                |  hank.go = 0; clean line 0                 |  see go=0 -> write producer_rc, return
                                |  sweep 10000 LDR on p_seq, then on COND    |
                                |  write NotifPerfResult; return             |
sync streams                    |                                            |
D2H result; print table         |                                            |
```

`go` is raised once and lowered once: the producer's loop has no re-entry path,
so bouncing it between the two subtests would end the producer for good and leave
the COND subtest waiting on a core that has already returned. The mode switch is
what moves between subtests.

Each of the three cache lines has exactly one writer — line 0 the consumer's
control words, line 1 and line 2 the producer's. The producer publishes with
`dcci(..., CACHELINE_OUT)`, which writes back a whole line, so a control word
sharing a line with a producer-written field gets restored to the producer's
stale copy every iteration. That is not theoretical: with `go` and `p_seq` on one
line, the `go = 0` ending the run was undone often enough to hang 3 of 6 runs.

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
- `target_core_idx` — fallback only. The producer publishes the core it actually
  landed on and the consumer polls that one; `block_dim=1` does **not** pin the
  kernel to the first AIC (0x8015 → core 21 observed), so a hardcoded index polls
  a register that never changes and the COND subtest times out. This argument is
  used only when the producer never reports. The printed
  `producer core_id = 0x… raw -> … used [reported by producer | FALLBACK …]` line
  says which happened.
- `n_samples` — per E2E subtest (default 100).

## Expected output (a3, ~50 MHz sys counter)

Measured on CANN 9.0.0, driver 26.0.rc1, one locked a2a3 die:

```text
=== notification-perf result ===
  consumer_rc        = 0
  magic              = 0xc0decafe  OK
  observed_p_seq     = 726 (must be > 0)
  producer_rc        = 0 (OK, stopped by the consumer)

  --- Phase 14: E2E AICore->AICPU latency ---
  GM   N=100  avg=23 ticks (~460 ns)  min=19 (~380 ns)  max=32 (~640 ns)
  COND N=100  avg=15 ticks (~300 ns)  min=6 (~120 ns)  max=58 (~1160 ns)
  producer core_id   = 0x8015 raw -> 21 used (masked)  [reported by producer]

  --- Phase 13 supplemental: idle-state LDR rate (10000 LDRs) ---
  GM   LDR ticks total = 344  (~6880 ns)  per LDR ~ 0 ns
  COND LDR ticks total = 53536  (~1070720 ns)  per LDR ~ 107 ns
```

**The core id is not stable between runs** — two runs of the same binaries
reported `0x8015 -> 21` and `0x8 -> 8`. That is why the consumer polls the core
the producer names rather than an index passed on the command line.

`consumer_rc = 0`, `producer_rc = 0` and a non-zero `N` on **both** rows is the
pass condition. Three failure shapes to read rather than guess at:

- `COND no valid samples (0 observed before their timestamp)` — the register never
  changed, so the wait timed out. Check the `producer core_id` line; a `FALLBACK`
  there means the producer never reported and the poll went to `target_core_idx`.
- `COND no valid samples (N observed before their timestamp)` with N > 0 — the
  notification was seen before the timestamp it refers to, i.e. a publication
  ordering fault rather than a harness problem.
- `producer_rc` non-zero — the producer ended on its own budget instead of on the
  consumer's `go = 0`, so the handshake broke rather than the measurement being
  merely noisy. Both budgets exist so this prints instead of the producer spinning
  and the host blocking in `aclrtSynchronizeStream` with nothing to show.

The COND path being faster than GM is the point of the tool. The relationship,
not the absolute figures, is what carries over between chips and CANN versions.

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
