# Local Runtime Timeouts

Local runs use production-friendly timeout defaults. Onboard platforms wait up
to 10 s for AICPU scheduler no-progress, 45 s for STARS op-execute timeout,
and 50 s for host stream synchronization. Sim platforms use a 10 s scheduler
timeout and do not have STARS or ACL stream-sync timeouts.

This means a real local hang can take much longer to surface than it does in
CI. CI restores the old fast-fail values with environment overrides:

```bash
export PTO2_SCHEDULER_TIMEOUT_MS=2000
export PTO2_OP_EXECUTE_TIMEOUT_US=3000000
export PTO2_STREAM_SYNC_TIMEOUT_MS=4000
```

For sim-only runs, CI sets only:

```bash
export PTO2_SCHEDULER_TIMEOUT_MS=5000
```

Use the same variables locally when you want faster failure while debugging a
suspected hang. For onboard runs, keep the ordering valid:

```text
scheduler timeout < op-execute timeout < stream-sync timeout
stream-sync timeout > scheduler timeout + 1.5 s
```

Invalid values or invalid onboard ordering are ignored with a warning and the
compiled defaults are used instead. See [args-dump](../dfx/args-dump.md#8-faq--debug-guide)
for the timeout chain and dump-recovery details.
