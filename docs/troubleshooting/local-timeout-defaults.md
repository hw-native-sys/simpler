# Local Runtime Timeouts

Local runs use production-friendly timeout defaults. Onboard platforms wait up
to 20 s for AICPU scheduler no-progress, 45 s for STARS op-execute timeout,
and 50 s for host stream synchronization. Sim platforms use the same 20 s
scheduler timeout and do not have STARS or ACL stream-sync timeouts.

The scheduler budget is a single constant (`PLATFORM_SCHEDULER_TIMEOUT_MS` in
each arch's `platform_config.h`) shared by onboard and sim, because both run
the same no-progress watchdog. It is sized to outlast a slow CPU-sim kernel on
an oversubscribed host while still firing well before the 45 s STARS op-execute
timeout onboard.

This means a real local hang can take much longer to surface than it does in
CI. CI restores the old fast-fail values with environment overrides:

```bash
export SIMPLER_SCHEDULER_TIMEOUT_MS=2000
export SIMPLER_OP_EXECUTE_TIMEOUT_US=3000000
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=4000
```

For sim-only runs, CI sets only:

```bash
export SIMPLER_SCHEDULER_TIMEOUT_MS=5000
```

Use the same variables locally when you want faster failure while debugging a
suspected hang. For onboard runs, keep the ordering valid:

```text
scheduler timeout < op-execute timeout < stream-sync timeout
stream-sync timeout > scheduler timeout + 1.5 s
```

Invalid values or invalid onboard ordering are ignored with a warning and the
compiled defaults are used instead. See [args-dump](../dfx/args-dump.md#8-faq-and-debug-guide)
for the timeout chain and dump-recovery details.

## Tensor-data waits

Both A2A3 and A5 `tensormap_and_ringbuffer` runtimes default to **30 s in CPU
simulation** and **15 s onboard** for scalar tensor-data access. Override the
budget before `Worker.init()`:

```bash
export SIMPLER_TENSOR_DATA_TIMEOUT_MS=30000
```

The value must be an integer from 1 through 2147483647 milliseconds. Zero,
negative, malformed and out-of-range values warn and fall back to the selected
backend default. Zero does not disable the deadline. On CPU-constrained hosts,
measure the dependency wait and size this override for the workload; the default
is not a completion guarantee for every CPU allocation.

The Host parses this setting through `runtime_timeout_config.h`. Onboard it
travels in `InitArgs.tensor_data_timeout_ms` to `simpler_aicpu_init`; simulation
passes it to a required setter in the resident AICPU SO. It is latched at Worker
initialization and is not reread between runs. Onboard initialization publishes
it together with the DMA workspace addresses. Recreate the Worker to change it.
Rebuild matching Host and AICPU libraries together; the existing DMA fields retain
their offsets, but mixed library revisions are not a supported compatibility
contract.

This budget covers each producer wait separately. A write starts another budget
for that producer's outstanding consumers. It includes dependency-chain latency,
not just the target kernel's execution time. Unrelated task completions do not
reset it. The scheduler instead keeps a progress timestamp per scheduling thread:
completion handling, dispatch and reclamation can refresh that thread's timer.
On expiry it reports a stall when the thread owns a running task, or when
unfinished work remains and no thread owns a running task. An idle thread with
a running sibling renews its timer instead. A task completing elsewhere does
not unconditionally refresh every scheduler thread's deadline.

Tensor waits are independent of the scheduler/op/stream ordering above: invalid
ordering restores those three defaults while retaining a valid tensor override.
Increasing this setting does not extend the outer watchdogs; any of them can
terminate a run first. The scalar wait observes existing orchestration/scheduler
errors, aborts access, and preserves their codes. Otherwise expiry reports code 8.
`host_build_graph` does not use this wait; it rejects scalar access requiring a
producer with code 5.
