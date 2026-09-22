# Chasing down a stall (100, and code 8)

← [Device Error Codes](../device-error-codes.md)

## The timeout race decides what you see

Three watchdogs compete, and whichever fires first determines whether you get a
clean `-100` with a `sub_class`, or a masked `507018`:

- the scheduler no-progress timeout (`SIMPLER_SCHEDULER_TIMEOUT_MS`),
- the STARS op-execute timeout (`SIMPLER_OP_EXECUTE_TIMEOUT_US`, ~45 s, kills `aicpu-sd`),
- the host stream-sync timeout (`SIMPLER_STREAM_SYNC_TIMEOUT_MS`).

**A 45 s op-execute kill is not proof of a deadlock** — the kernel may simply be
slow. To find out which, order the race deliberately. The STs do exactly this, and
the technique transfers:

- To make the **scheduler** win (get a `sub_class`), squeeze it below the others:
  `SCHEDULER=2000ms < OP_EXECUTE=3s < STREAM_SYNC=4000ms` (`aicore_hang` case).
- To let a **slow-but-alive** path finish, push the others out of the way:
  `SCHEDULER=30000ms`, `OP_EXECUTE=30s`, `STREAM_SYNC=40000ms` — this is how the
  `tensor_wait_timeout` case lets the 15 s tensor-data wait land code 8 instead of
  being reaped first.

**These three are read once, at `Worker.init()`.** Changing them between `run()`
calls on the same Worker does nothing — you must rebuild the Worker (or use a
separate process) per value. Defaults and the rationale are in
[`../local-timeout-defaults.md`](../local-timeout-defaults.md).

## S1: find the stuck kernel

The `sub_class=` line gives you `stuck_task_id` and `stuck_core`. Map the task id
back to your orchestration and look at that kernel for an infinite loop, a wait on
a signal that never arrives, or simply too much work.

When the task id is not enough, the device log carries a per-task snapshot of the
moment the scheduler gave up. It follows the `[SHUTDOWN_SNAPSHOT … reason=scheduler_timeout]`
line and is printed at **the default log threshold**, so a stall that was already
reaped can be read without reproducing it:

```text
[STALL thread=0 idle_iterations=...] TASK ring=1 task_id=42 state=RUNNING \
    fanin_refcount=0/2 kernels=[aic:3 aiv0:7 aiv1:-1] \
    running_on=[owner_thread=0 cores=[core=5(AIC) core=6(AIV0)]]
```

`kernels=[...]` are the kernel ids in the task's three sub-core slots and
`cores=[...]` the physical cores running it — that maps "stuck task" to "which
kernel, on which core". The `SUMMARY` line beside them gives `completed=x/y` and the
iteration progress last happened on.

Lowering the threshold to **DEBUG** adds the *periodic* rounds the scheduler emits
every `STALL_LOG_INTERVAL` idle iterations while a stall is still building. Those
rounds are what show whether the pending set was changing or frozen from the start —
the shutdown snapshot alone shows only the final state.

Setting the level:

- **Worker directly**: `logging.getLogger("simpler").setLevel("DEBUG")` **before**
  `worker.init()` — the level is snapshotted at init and pushed to the device, so
  setting it between `run()` calls has no effect.
- **pytest / scene test**: `--log-level debug`.

`DEBUG` is the most verbose level (`DEBUG`, `INFO`, `TIMING`, `WARN`, `ERROR`;
default `TIMING`). Device logs land
in the shared `~/ascend/log/debug/device-<id>/` by default, where several processes
interleave; redirect them per-run with `ASCEND_PROCESS_LOG_PATH` (the directory must
exist) before reading. See the "Device logs" section of
[`running-onboard.md`](../../../.claude/rules/running-onboard.md).

## Code 8, specifically

Only `tensormap_and_ringbuffer` raises this code. Its frequency-scaled
tensor-data wait defaults to 15 s onboard and 30 s in simulation, selected from
`PLATFORM_ONBOARD_TENSOR_DATA_WAIT_TIMEOUT_MS` or
`PLATFORM_SIM_TENSOR_DATA_WAIT_TIMEOUT_MS`. It means either the producer never
completed, or a consumer never released its fanout reference. Check for a hung
producer first (that is S1 above), then verify the consumer really declares the
dependency and exits. If the kernel is merely slow, raising the corresponding
platform default will prove it. Both defaults are compile-time constants — no
environment variable or `runtime_env` knob reads them. To raise one, edit
`src/{arch}/platform/include/common/platform_config.h` and rebuild the runtimes
(`pip install --no-build-isolation -e .`).

`host_build_graph` has no such wait: its orchestration finishes before the device
starts, so `get_tensor_data` / `set_tensor_data` reject a tensor with a producer
with `INVALID_ARGS` (code 5) instead of waiting.

## If the fault line is an addressing error, not a stall

A core that faults stops answering, so a watchdog reaps the op afterwards and the
tail looks identical to a stall. If the device log carries a `VEC`/`CUBE`
instruction error rather than only a timeout, you are in
[Chasing down an AICore fault](aicore-fault.md), not here.
