# Shared SDMA queue wrap and retained-run reuse

Real peer TGET validation on A5 HBG covers two/four ranks with diagnostics disabled/enabled. It uses existing Worker and CommDomain ownership and the default SDK queue configuration, without new Host APIs or fixed physical cores.

A sync-start producer spans `rt_available_aiv_count()` AIV blocks. The two AIV lanes in each physical cluster share its default SDK queue. Each execution stays below the default capacity of 64 conditions.

1. Each AIV submits 33 transfers, producing 66 submissions per physical queue and crossing its 64 internal slots.
2. The next run retains the same communication domain and submits once per AIV: two more consecutive post IDs per queue.
3. Consumers depend on the entire producer task, inspect its public completion records, and add 7 to all transferred values.

The oracle groups the records registered by production `send_request_entry` by their actual completion-counter address. It checks:

- One physical cluster and exactly two logical AIV blocks per queue, covering every available cluster.
- Consecutive post IDs within and across runs, with unchanged retained contexts and completion-record addresses.
- Distinct epoch/rank/transfer data, complete transferred and consumed values, and exactly one submission/consumption per item.
- Unchanged inputs, unused regions, payload guards and metadata guards.

The first transfer is 4 MiB; the others contain eight floats in guarded views. The 128-lane bound only reserves fixture storage; the actual topology comes from the run. All numerical assertions remain active when diagnostics are disabled.

This establishes cumulative slot wrap, shared-queue submission and retained-workspace continuity. It does not infer 66 simultaneously in-flight requests or actual SDK backpressure from cumulative counts or kernel duration. Separate SDMA completion cases verify a real PENDING window and independent business progress.

Hardware only: the current PTO-ISA SDMA implementation has no CPU simulation backend. Pass the repository's A5 architecture precheck, then run this directory's pytest inside a `task-submit --device auto --device-num 4` allocation using `$TASK_DEVICE`. Preserve the CANN library environment. Each `queue-reuse.json` retains per-rank, per-queue and per-epoch post-ID evidence.
