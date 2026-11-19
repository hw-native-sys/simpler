# Local Runtime Timeouts

Local runs use production-friendly timeout defaults. Onboard platforms request
20 s for AICPU scheduler no-progress, 45 s for STARS op-execute timeout,
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
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=6000
```

For sim-only CI runs, CI sets:

```bash
export SIMPLER_SCHEDULER_TIMEOUT_MS=10000
```

Use the same variables locally when you want faster failure while debugging a
suspected hang. For onboard runs, keep the ordering valid:

```text
scheduler timeout < op-execute timeout < stream-sync timeout
stream-sync timeout > scheduler timeout + 1.5 s
```

Invalid requested values or invalid requested ordering are ignored with a warning
and the compiled defaults are used instead. On an owned onboard context, ACL can
adjust the operator timeout. Initialization validates its API-reported value
against the effective scheduler and stream budgets, including compiled defaults.
If the setter succeeds but the resulting order is invalid, initialization logs
a warning and continues with unchanged budgets. A setter failure retains its
original error code and rejects initialization; close the Worker before retrying
that failed initialization. Requested and API-reported values are logged
separately; neither is a measurement of the actual watchdog firing time.

Borrowed kernel contexts do not call the device-global operator-timeout setter.
Their local requested budgets do not validate the caller's external watchdog.

## How reported-timeout ordering warnings are handled

A successful ACL setter call establishes the API-reported operator setting.
The recommended ordering gives the scheduler an opportunity to classify a stall
and publish diagnostics before an operator or Host wait expires. An invalid
numeric ordering does not by itself prove that execution will fail or that
resources cannot be reclaimed, so it emits a warning instead of refusing
initialization. A Host wait can expire before the reported operator watchdog
when its stream budget is smaller, reducing the available diagnostic window.
Numeric ordering does not guarantee the order in which the timers fire.

The warning includes the requested and API-reported operator values,
the effective scheduler and stream budgets, the failed relationship and the
configuration to adjust. For example, a 3,000,000 us request reported as
4,294,967 us cannot fit below a 4,000 ms stream budget. Increase
`SIMPLER_STREAM_SYNC_TIMEOUT_MS` above the reported value before creating a
Worker when that diagnostic ordering is needed. Initialization itself continues
with the original stream budget; the runtime never enlarges it automatically.

Actual synchronization timeouts and device errors still fail the operation and
retain stopped admission, draining, reset or quarantine as required by the
existing lifecycle. Resource release still requires completion evidence; a
configuration warning is not permission to ignore a later failure.

Setter failures are separate initialization errors. Their original ACL code and
configuration advice pass into the Worker exception before cleanup. Matching
runtime modules export the initialization-diagnostic query; a null or empty
detail preserves the original numeric error. A module missing the query is
rejected during symbol loading, before creating its device context.

## What the Host wait budget measures

`SIMPLER_STREAM_SYNC_TIMEOUT_MS` is applied to individual Host synchronization
calls. A normally fenced run waits for its AICPU and AICore completion Events,
with a separate budget for each boundary. An unfenced fallback waits for each
whole Stream in turn. The budget can also be used by recovery and cleanup waits;
it is not one shared deadline from kernel launch, nor a bound on the total
duration of `run()` or `close()`.

The millisecond API parameter does not establish millisecond return-time
accuracy. The scheduler's no-progress timer, operator execution watchdog and
Host waits have different timing boundaries. Configured numeric ordering does
not prove the order in which they fire. This validation does not measure Host
wait precision or attribute a `507018` to a particular watchdog; the operator's
reported adjustment does not establish the same adjustment for Host waits.

See [args-dump](../dfx/args-dump.md#8-faq-and-debug-guide)
for the timeout chain and dump-recovery details.
