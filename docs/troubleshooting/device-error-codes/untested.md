# Codes with no end-to-end test

← [Device Error Codes](../device-error-codes.md)

The entries below describe missing end-to-end scene coverage, including
runtime-specific gaps. Code 1 is tested in TMR; its entry applies only to HBG.
Reachability and narrower regression coverage are recorded here (the defensive
branches are also discussed in `tests/st/runtime_fatal_codes/test_runtime_fatal_codes.py`):

- **103 ASYNC_REGISTRATION_FAILED** is structurally pre-empted. The AICore side caps
  at `MAX_COMPLETIONS_PER_TASK` (64) and latches **102** on the 65th condition, so
  the scheduler-side "more than 64" check that would raise 103 never sees enough
  messages. 103 is reachable only by corrupting the slab (UB) or a malformed message.
- **10 SCOPE_TASKS_OVERFLOW** has no raise site at all in `host_build_graph`: an hbg
  scope keeps no task list, so nothing can saturate one. In
  `tensormap_and_ringbuffer` it exists but cannot be reached — `scope_tasks_cap` is the
  sum of the per-ring windows, but each ring physically holds only `window - 1` tasks,
  so all rings together top out at `cap - CHIP_MAX_RING_DEPTH`, strictly below the cap.
  The rings always fill first and latch 1 or 3. Shrinking the window shrinks both, so
  the gap holds. The number stays reserved in both runtimes' code tables.
- **1 SCOPE_DEADLOCK** likewise has no raise site in `host_build_graph`. It reports
  "one scope submitted more tasks than the ring holds", which presumes reclaim at scope
  end; hbg is whole-graph-resident and reclaims nothing, so overrunning the ring latches
  **2 HEAP_RING_DEADLOCK** or the ring's own exhaustion code instead. Reserved in hbg's
  table, raised only by `tensormap_and_ringbuffer`.
- **11 TENSORMAP_OVERFLOW** needs the 65536-entry pool (`CHIP_TENSORMAP_POOL_SIZE`,
  compile-time, not tunable via `runtime_env`) flooded *and* a stalled producer
  holding entries.
- **S3 ready-but-all-idle**: no stable public-API reproducer after orchestration-side
  dependency wiring. The CPU scheduler tests hold a genuinely ready task before
  dispatch and verify S3, including when another task awaits async completion.
- **S6 async-completion-wait**: reachable when a deferred task's AICore work has
  finished but logical completion is pending. CPU scheduler tests cover mailbox
  handoff, wait-list residence, mixed running work, timeout publication and
  eventual completion/dependency release. Hardware transport-stall reproduction
  is separate from this diagnostic regression coverage.
- **S4 / S5 / unknown**: not expressible through the public API. `set_dependencies()`
  can only name already-submitted tasks, so a dependency cycle cannot be written; a
  stuck producer is RUNNING (→ S1), never pure WAIT (S4). `total_tasks_` counts
  *submitted* tasks, so once they retire `completed == total` and the S5 watchdog
  never fires. They are kept as defensive labels: if a future bug produces the state,
  it gets named instead of mislabelled.

The name tables are held complete for these anyway
(`tests/ut/cpp/common/runtime_status/test_error_code_names.cpp`) — an unreachable code is exactly
the one that would otherwise print as a bare number on the day it finally fires.

If you do hit 10, 11, 103, S4, S5 or unknown, it is a runtime bug. Keep the device
log and report it.

Code 1 is the one entry above that is unreachable in only *one* runtime. Latched by
`tensormap_and_ringbuffer` it is an ordinary capacity error carrying its own remedy
("raise `ring_task_window` or split the scope"), so treat it as a runtime bug only
when `host_build_graph` reports it.
