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

Invalid values or invalid onboard ordering are ignored with a warning and the
compiled defaults are used instead. See [args-dump](../dfx/args-dump.md#8-faq-and-debug-guide)
for the timeout chain and dump-recovery details.
