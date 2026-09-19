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
- To make the **tensor-data wait** win (get code 8 and its producer locator),
  squeeze `TENSOR_DATA=1000ms` below all three — it sits outside their ordering
  group, so lowering it alone leaves them valid (`tensor_wait_timeout` case).

**These four are read once, at `Worker.init()`.** Changing them between `run()`
calls on the same Worker does nothing — you must rebuild the Worker (or use a
separate process) per value. Defaults and the rationale are in
[`../local-timeout-defaults.md`](../local-timeout-defaults.md).

## S1: find the stuck kernel

The `sub_class=` line gives you `stuck_task_id` and `stuck_core`. Map the task id
back to your orchestration and look at that kernel for an infinite loop, a wait on
a signal that never arrives, or simply too much work.

When the task id is not enough, lower the log threshold to **DEBUG** and the device log
prints a task snapshot at the moment of the stall:

```text
[STALL thread=0 idle_iterations=...] TASK ring=1 task_id=42 state=RUNNING \
    fanin_refcount=0/2 kernels=[aic:3 aiv0:7 aiv1:-1] \
    running_on=[owner_thread=0 cores=[core=5(AIC) core=6(AIV0)]]
```

`kernels=[...]` are the kernel ids in the task's three sub-core slots and
`cores=[...]` the physical cores running it — that maps "stuck task" to "which
kernel, on which core".

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

Only `tensormap_and_ringbuffer` raises this code. Its tensor-data wait defaults to
15 s (`TENSOR_DATA_TIMEOUT_MS`, frequency-scaled) and is overridden per run by
`SIMPLER_TENSOR_DATA_TIMEOUT_MS`. It means either the producer
never completed, or a consumer never
released its fanout reference. Check for a hung producer first (that is S1 above),
then verify the consumer really declares the dependency and exits. If the kernel is
merely slow, raising the timeout will prove it.

Code 8 only reaches you if this wait expires before the other watchdogs reap the
op. Its 15 s default sits below the 20 s scheduler no-progress budget but above
the op-execute timeout CI tightens to 3 s, so under CI's values a stall reports
code 100 or a bare 507018 instead. Lower `SIMPLER_TENSOR_DATA_TIMEOUT_MS` below
both when you want the producer locator that code 8 carries.

`host_build_graph` has no such wait: its orchestration finishes before the device
starts, so `get_tensor_data` / `set_tensor_data` reject a tensor with a producer
with `INVALID_ARGS` (code 5) instead of waiting.

## If the fault line is an addressing error, not a stall

A core that faults stops answering, so a watchdog reaps the op afterwards and the
tail looks identical to a stall. If the device log carries a `VEC`/`CUBE`
instruction error rather than only a timeout, you are in
[Chasing down an AICore fault](aicore-fault.md), not here.
