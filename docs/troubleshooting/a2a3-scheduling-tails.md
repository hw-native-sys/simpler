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

## Rare Warmed CSA HEAD Stalls

[Issue #2744](https://github.com/hw-native-sys/simpler/issues/2744) tracks
microsecond-scale stalls before TMR orchestration starts on warmed CSA calls.
For this workload, count each formal call with reported HEAD >= 10 us as an
event. Exclude complete-case warmups and golden calls from its denominator,
retain every formal sample, and report event counts and maxima per group.
The threshold is specific to this CSA baseline, whose median HEAD is about
7 us. A run without a peak is not by itself evidence of a mitigation.

The main periodic peaks are strongly associated with device OS timer work:
on the original A3 card, eight short windows with an interrupt increment each
contained a 3.98-11.34 us execution gap, while 56 windows without an interrupt
increment contained no gap above 2 us. Their approximately 4 ms phase matched
six peaks in a separate 2,000-call diagnostic cohort. The original peaks have
no same-call interrupt trace; hard-interrupt and subsequent RCU costs cannot
be separated. One composite peak remains unexplained. This mechanism is
distinct from the second-scale real-time throttling described above.

### Runtime Mitigation

On the eligible A2/A3 TMR LaunchEnvelope route, the orchestrator adopts the
arguments before waiting for the leader's setup publication. Leader setup
does not access the entry-argument storage, so these preparations can overlap:

```text
Leader:  shared setup ----------------------> publish setup done
Old Orch: wait for setup done --------------> adopt arguments -> refs/checks
New Orch: adopt arguments -> wait if needed -> refs/checks
```

This reduces the serial preparation exposed after setup. It neither disables
OS interrupts nor removes the per-call argument copy. Adoption remains inside
Preamble and RunWall. Preamble is a cross-thread envelope that can overlap
orchestration and scheduling; adding it to HEAD would double-count overlap.

Early preparation requires a stable orchestrator index, multiple participants,
non-serial execution, an exact LaunchEnvelope source and a registered
callable. Other routes retain late preparation. The result is local to one
invocation: Ready avoids a second adoption, Rejected waits for initialization
and uses the original argument-error exit, and an initialization failure
remains authoritative. Reference reconstruction, callback/count checks, arena
and shared-memory reset, ready publication and cleanup retain their ordering.

### Validation and Limits

The frozen `8f42cef9b3f0ebbda800e239aab0949e95552f25` runtime with ISA
`217bb98952a8f9004b95fe938da97d8d5f68f702` was compared on the same A3 card,
with fixed inputs and matching host/core artifacts. Execution order was
control-0, candidate-0, candidate-1, control-1. Each worker performed five
complete-case warmups followed by 1,000 consecutive formal calls; each version
therefore has two independent 1,000-call groups.

| Group | Control HEAD >= 10 us | Candidate HEAD >= 10 us | Control HEAD max | Candidate HEAD max |
| ----- | --------------------: | ----------------------: | ---------------: | -----------------: |
| 0 | 3/1000 | 0/1000 | 15.46 us | 5.98 us |
| 1, reverse order | 2/1000 | 0/1000 | 13.44 us | 5.88 us |

The maximum excess over each worker's own median also fell from 8.50/6.42 us
to 1.20/1.26 us. A separate DS regression cohort observed 1/300 control events
and 0/300 candidate events, with maxima 15.18/5.88 us; its counts remain
separate. These results support mitigation of the target warmed HEAD peaks
under the measured conditions, without an OS or image change.

The complete nine-workload benchmark and eleven-workload DS cohorts passed
their golden and cleanup checks. Whole-Device and host latency changes were
mixed: DS CSA Device mean increased by 1.110 us and the alternating benchmark
increased by 0.850-2.595 us across three groups. These are retained costs,
not a claim of general speedup. The targeted cohort's whole-Device maxima
432.36/432.26 us had ordinary HEAD values of 7.68/4.24 us and were not the
HEAD peak events.

These are frozen-build observations, not new measurements of every later
integration commit or proof that interrupts disappear. Reported HEAD is
rebased to the earliest subphase; RunWall has its own raw origin, so its
absolute kernel-entry saving cannot be inferred directly from the HEAD
delta. Other execution-period high values and untraced interrupt costs remain
separate from the target HEAD mitigation.
