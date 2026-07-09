# Sim Hangs / `rc=-1` Under CPU Oversubscription

## TL;DR

On **sim**, every AICPU and AICore is a host thread, not silicon. When more
runtime threads are runnable than there are CPUs — high `--max-parallel` on a
few-vCPU box, or several scene-test cases sharing one runner — the
busy-spin/handshake/timeout logic that is correct on hardware can **livelock or
false-timeout**. Symptoms: a stuck test that eventually hits the pytest session
timeout (`rc=124`), or a `simpler_run failed with code -1` cascade. The work is
not wrong; the threads it depends on just never got a CPU slice in time.

Mitigation: lower `--max-parallel` (e.g. `--max-parallel 2`) on CPU-constrained
runners. Onboard is unaffected — AICPU/AICore are separate physical cores there.

## Symptom

Two distinct signatures, both only on sim and only under load:

1. **Init-handshake livelock → hang.** The case makes no progress and the run
   is killed by `--pto-session-timeout` (process exit `124`):

   ```text
   [pytest] TIMEOUT: session exceeded 600s (10min) limit
   HUNG standalone test_... (rt=tensormap_and_ringbuffer, dev=N) elapsed=490.1s
   Process completed with exit code 124.
   ```

2. **Deinit-timeout cascade → `rc=-1`.** A teardown wall-clock timeout is
   misread as a wedged core:

   ```text
   Core X deinit timed out
   aicpu_execute: Thread execution failed with rc=-1
   RuntimeError: chip_process dev=N: simpler_run failed with code -1
   ```

Both appear intermittently in `st-sim-*` on `ubuntu-latest` (≈4 vCPU) and are
easy to trip on any box running concurrent sim tests.

## Root Cause

`--max-parallel` is the number of scene-test **cases** the scheduler runs
concurrently (`auto` = `min(devices, nproc)`). Each case forks one chip
subprocess per device, and each chip spawns one host thread per simulated
AICore (`block_dim` threads) plus AICPU + scheduler threads. So N concurrent
cases on an M-vCPU runner put roughly `N × devices × block_dim` runnable threads
on M cores — easily 10–100× oversubscription.

Under that pressure two HW-correct patterns misbehave:

- **Init handshake (livelock).** `SchedulerContext::handshake_all_cores` and
  `aicore_execute` synchronize startup with **bare busy-spins**
  (`while (hank->aicore_done == 0) {}`, `while (read_reg(RegId::DATA_MAIN_BASE) == 0) {}`).
  With no yield, the spinning threads monopolize the CPUs and starve the very
  threads they wait on, so the handshake never completes. A `gdb` snapshot of a
  wedged chip subprocess (8× `domain_rank_map` on 2 CPUs) showed the AICPU stuck
  in `handshake_all_cores` while ~100 AICore host threads busy-spun in
  `aicore_execute`.

- **Deinit timeout (false positive).** `platform_deinit_aicore_regs()` waits up
  to a wall-clock budget for an AICore exit ACK. A 1 s budget conflates "the op
  is stuck" with "the OS scheduler hasn't scheduled my thread yet."

## Why Onboard Does Not Hit This

On real hardware the AICPU and each AICore are separate physical cores that
genuinely run in parallel, so a bare spin is the *correct* low-latency wait and
a 1 s deinit budget is a real hang detector. `SPIN_WAIT_HINT()` is `((void)0)`
on onboard — yielding a hardware AICPU spin-wait would add unacceptable latency
(see `.claude/rules/codestyle.md`, spin-wait rule). The pathology is specific to
the sim host-thread model.

## Mitigation

- **Throttle parallelism**: `pytest ... --max-parallel 2` (or lower) on a
  CPU-constrained runner. This is the flag's intended use — it shrinks the
  oversubscription without changing `--device`.
- **Shrink the per-case footprint**: a smaller `block_dim` in the case config
  means fewer AICore threads per chip.

## Fix

The runtime now yields/extends these waits on sim while staying a no-op on
hardware:

- **Init handshake**: the four startup spins in `aicore_execute`
  (`aicore_executor.cpp`) and `SchedulerContext::handshake_all_cores`
  (`scheduler_cold_path.cpp`) call `SPIN_WAIT_HINT()` — `sched_yield()` on sim,
  `((void)0)` on onboard — matching the steady-state poll loops in the same
  files.
- **Deinit timeout**: `inner_get_deinit_timeout_ticks()` gives sim a 10 s budget
  vs onboard's 1 s.

Both landed together in PR #893; the init-handshake half also relates to #884
(a2a3sim `dynamic_register` instability).

## How to Reproduce / Debug

Constrain CPUs and oversubscribe (a Docker container with `--cpus=2` reproduces
reliably, far below CI's ~4 vCPU):

```bash
# N concurrent instances of one case, on 2 CPUs
docker run --cpus=2 <image> bash -c '
  source .venv/bin/activate
  for c in $(seq 1 8); do
    python -c "from examples.workers.l3.domain_rank_map.main import run; run(\"a5sim\",[0,1,2])" &
  done; wait'
```

To capture a livelock backtrace, attach `gdb` to a wedged chip subprocess
(needs `--cap-add=SYS_PTRACE` or matching uid):

```bash
gdb -p <python-chip-pid> -batch -ex "set pagination off" -ex "thread apply all bt"
```

Look for many AICore threads in `aicore_execute` and the AICPU thread in
`SchedulerContext::handshake_all_cores` — the livelock signature.

## References

- PR #893 — sim init-handshake yield + deinit timeout widening.
- Issue #884 — a2a3sim dynamic_register instability.
- Spin-wait rule (no yield on hardware AICPU): `.claude/rules/codestyle.md`.
