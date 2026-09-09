# A2/A3 Scheduling Tails

Continuous AICPU polling can exhaust the device Linux real-time CPU budget.
On two A3 hosts, the executing CANN workers used `SCHED_FIFO` at priority 10,
with `sched_rt_runtime_us=950000` and `sched_rt_period_us=1000000`. Dual-slot
Qwen decode produced roughly one long completion interval per second.

During a representative tail, an AICPU scheduler thread consumed 38.65 ms of
CPU time over 82.73 ms of wall time and recorded one involuntary context
switch. All three scheduler threads paused together. The typical frame
duration was unchanged. Replacing FIFO with `SCHED_OTHER` for the scheduler
threads removed the tails; restarting with FIFO restored them. All three
groups passed the full token golden and native dual-slot validation.

| Scheduler policy | Effective P50 | Effective max | Samples above 50 ms |
| ---------------- | ------------: | ------------: | ------------------: |
| FIFO baseline | 38.689 ms | 82.709 ms | 4/122 |
| SCHED_OTHER | 38.690 ms | 39.044 ms | 0/122 |
| FIFO repeated | 38.684 ms | 82.965 ms | 5/122 |

These measurements establish a scheduling-policy dependency consistent with
real-time bandwidth throttling. The device did not expose `/proc/sched_debug`,
so a kernel throttle event was not captured directly. The Linux bandwidth
controls are described in the [kernel documentation](https://docs.kernel.org/scheduler/sched-rt-group.html).

## Execution Policy

A2/A3 onboard execution selects `SCHED_OTHER` for every thread that survives
the affinity gate before entering the runtime. This covers TMR orchestration
and scheduling, and HBG's scheduling participants. The thread policy is
checked on every invocation because a CANN worker can be reused or have its
policy reset between calls. Already-normal threads need no setter call.

The policy belongs to the CANN worker thread and persists after the Simpler
entry returns. Switching back to FIFO can require privileges unavailable to
device code. Callers sharing a CANN worker process must account for this
lifetime; the policy is not scoped to one graph or model. Ordinary scheduling
can introduce contention latency when other runnable threads share the CPU.

If the policy query or update fails, the runtime emits one warning per loaded SO
and continues with its current policy. Returning from only one participant
would leave its peers waiting at the runtime barriers. Such a run is not
guaranteed to avoid real-time throttling.

Completion intervals, host runner time and device windows must all be checked:
a pause before device-phase entry or after its completion appears in the
runner window but can be absent from the device timing. Single-slot gaps can
reduce budget pressure, but a single-slot workload is not inherently exempt.
