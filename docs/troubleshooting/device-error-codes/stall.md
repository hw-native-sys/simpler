# Chasing down a stall (100, and code 8)

← [Device Error Codes](../device-error-codes.md)

## The timeout race decides what you see

Three timeout mechanisms can affect whether a stall surfaces as a classified
`-100` with a `sub_class`, a Host wait timeout, or an execution error such as
`507018`. Their configured budgets alone do not identify which mechanism fired:

- the scheduler no-progress timeout (`SIMPLER_SCHEDULER_TIMEOUT_MS`),
- the STARS op-execute timeout (`SIMPLER_OP_EXECUTE_TIMEOUT_US`, ~45 s, kills `aicpu-sd`),
- the host stream-sync timeout (`SIMPLER_STREAM_SYNC_TIMEOUT_MS`).

**A 45 s op-execute kill is not proof of a deadlock** — the kernel may simply be
slow. The STs configure different budgets to exercise the diagnostic paths:

- To give the **scheduler** an earlier budget for publishing a `sub_class`:
  `SCHEDULER=2000ms`, requested `OP_EXECUTE=3s`, `STREAM_SYNC=6000ms`
  (`aicore_hang` case). ACL may adjust the operator value; owned initialization
  checks that the reported value remains strictly between the other budgets.
  This numeric ordering does not establish which timer actually fires first.
- To let a **slow-but-alive** path finish, push the others out of the way:
  `SCHEDULER=30000ms`, `OP_EXECUTE=30s`, `STREAM_SYNC=40000ms` — this is how the
  `tensor_wait_timeout` case lets the 15 s tensor-data wait land code 8 instead of
  being reaped first.

**These three are read once, at `Worker.init()`.** Changing them between `run()`
calls on the same Worker does nothing — you must rebuild the Worker (or use a
separate process) per value. Defaults and the rationale are in
[`../local-timeout-defaults.md`](../local-timeout-defaults.md).

## S1: find the stuck kernel

The `sub_class=` line gives you `stuck_task_id` and `stuck_core`. The task id is
printed in hex, and `0xffffffffffffffff` is the reserved sentinel meaning the
classifier found no RUNNING task to name. Map the task id back to your
orchestration and look at that kernel for an infinite loop, a wait on a signal
that never arrives, or simply too much work.

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

At **INFO** or **DEBUG**, each scheduler thread emits one local `CLUSTER` snapshot
when its continuous no-progress time reaches half the scheduler timeout. The
snapshot includes that thread's cores, running tasks, kernels and COND state.
Progress on that thread rearms its warning; other threads neither suppress nor
rearm it. With INFO disabled, the logging backend filters out these messages.

For A5 `tensormap_and_ringbuffer`, each busy core also reports `cond_raw`,
`cond_tok`, `running_tok`, `pending_tok`, `cond_match` and `dmb_low`.
`cond_match` compares the sampled COND token with the copied scheduler tokens;
it is `running`, `pending` or `other`. A FIN can belong to either slot, and a
running FIN with an ordinary pending task can remain visible while the scheduler
waits for pending ACK/FIN. FIN alone does not establish a completion-poll bug.

`dmb_low` is the low 32 bits of the publication register, not the high-word
doorbell. Seeing the pending token there proves neither pickup nor execution:
the dispatcher can read a payload and wait on argument preparation or its gate
before posting ACK. `snapshot=best_effort` also matters: registers are sampled
separately, and a shutdown dump reads other scheduler threads' mutable state
without a quiescence acknowledgement. These fields are observations, not an
atomic state snapshot or a diagnosis of a GM-read stall. They do not add pending
task/payload traversal or change the completion protocol.

Local warnings do not scan shared task rings, acquire the AsyncWaitList lock, or
coordinate through a shared warning latch. Global `TASK`/`SUMMARY` and `ASYNC_WAIT`
reports are reserved for the final shutdown snapshot, whose winner emits all
threads' cluster state and bounded async-wait details at **WARN**. Tensor-data
wait timeouts use this same final snapshot with `reason=tensor_data_timeout`;
scheduler timeouts use `reason=scheduler_timeout`. A5 RDMA diagnostic details also
use WARN. The original fatal error remains at ERROR.

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

### TMR running and pending publication records

For A2/A3 and A5 `tensormap_and_ringbuffer`, each owner's local INFO snapshot
also emits `publication_records=owner_local implicit_wait_metadata=unavailable`
and one `SLOT` line for each valid running or pending token on its own cores:

```text
[STALL thread=0] SLOT core=0 slot=running token=1 bank=1 task_id=0x0 block=0/1 ...
[STALL thread=0] SLOT core=0 slot=pending token=2 bank=0 task_id=0x1 block=0/1 ...
```

The owner that wins the scheduler-timeout shutdown also emits its own records
at WARN before the global snapshot. An owner that reaches its tensor-data
timeout fatal branch also emits its own WARN records before that thread calls
emergency retirement. Owners that exit on an already published completion flag
do not reach this branch. These records cover only the reporting owner, not
every owner; the all-owner shutdown scan does not emit owner-local records.
HBG diagnostics are unchanged.

`core` is the scheduler core id; `block=i/n` names the task's logical block.
The token selects a reusable bank, and the task id comes from the published
payload's control fields. `function_bin_addr` and `src_payload` are recorded
addresses, not dereferenced memory. `src_payload != 0` indicates the gated
payload form; it does not establish whether its doorbell has opened.
`slot=running` describes scheduler placement, not proof of kernel execution;
`slot=pending` proves neither pickup nor completion. The helper does not read
kernel arguments or follow task-slot pointers.

The example can locate a candidate producer behind a waiter on the same core,
but the runtime has no implicit wait-event or producer metadata with which to
prove that relationship. Read the
[blocking-wait contract](../../aicore-kernel-programming/blocking-waits.md)
and correlate the kernel's signal accesses and submitted dependencies. A
timeout or this placement alone cannot prove a wait cycle or safe recovery.

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
