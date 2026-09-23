# A2/A3 worker retirement: CLOSE must precede return

The A2/A3 persistent worker must not return after merely acknowledging EXIT.
It must wait until the AICPU has closed the group's fast-path register windows
and explicitly released the workers through global memory (GM).

This applies to both `tensormap_and_ringbuffer` and `host_build_graph`.
It does not add borrowed-stream/L1 support or change the A5 exit protocol.

## What fast path means here

The AICPU accesses each AICore's register window through MMIO. It writes
`DATA_MAIN_BASE` to dispatch work and reads `COND` for ACK/FIN status. The
AICore reads its own dispatch SPR and writes its own COND SPR.
`FAST_PATH_ENABLE` (offset `0x18`) is written with `0xE` to open and `0xF`
to close the window. This is a device-side dispatch mechanism, not a torch
allocator setting, host task-queue option, or ACL event.

## The missing ordering edge

The old protocol established these two chains:

```text
AICore: observe EXIT -> write COND=EXITED -> return
AICPU:                 observe EXITED   -> write IDLE -> CLOSE
```

They have a common predecessor, not an ordering between their tails. The
following execution was therefore legal in the software protocol:

1. AICore observes EXIT and publishes EXITED.
2. AICore returns while its fast-path window is still open.
3. AICPU observes EXITED, writes IDLE, and closes the window.

EXITED proves that the worker has stopped accepting tasks. It cannot also
prove that the AICPU has finished managing the window: that work occurs
*after* the ACK. The missing edge is `CLOSE -> worker return`.

A host mutex, an event between streams, or waiting for an entire launch to
complete cannot retroactively order these two operations inside that launch.
Multiple in-flight graph executions are not required for this race. Nor does
an asynchronous error reported at a later native operator prove that the
native operator caused it.

## Why the race is reachable

The AICPU program and the AICore kernel are **two ops on two independent
streams**. `DeviceRunnerBase` creates `stream_aicpu_` and `stream_aicore_`
separately, and `DeviceRunner::launch_run` submits one kernel to each. Stream
order constrains only the ops within a stream, and the launch path inserts no
cross-stream event — it states outright that it "intentionally performs no
stream synchronization". The launch and reap halves are also separable, and a
prepared successor is allowed to overlap its predecessor's execution
(`allow_prepared_successor`).

So a worker that returns early makes its AICore-stream op complete early, while
the AICPU-stream op of the same round is still writing that core's window. The
AICore stream is then free to start the next round's kernel. That is what the
return gate closes: it makes "the AICore op finished" imply "the AICPU closed
every window it owned".

The neighbouring hazard already has a guard: `launch_run` clears each worker's
`aicore_done` before the AICore kernel launches, because a prior run's report
would otherwise "open a window on that run's physical_core_id". That guard
covers a stale *report* being read by the new AICPU; the return gate covers the
other direction, a stale *window write* landing on the new worker.

## Retirement

Each AICPU thread retires the cores it owns as it leaves the dispatch loop,
before it reaches the completion latch. Core ownership is a partition —
`assign_cores_to_threads` hands every cluster to exactly one scheduler thread —
so those retirements run concurrently and never name the same core.

Retirement must not hang off the completion latch. `ThreadCompletionGate` is a
last-one-out counter, not a barrier: no participant waits, and a thread that
returns early simply never arrives. Before this handoff existed, missing the
latch cost a leaked runtime context; with workers blocked on gates, it would
cost every core on the chip.

`platform_retire_aicore_group` retires one claimed set in passes:

1. Send EXIT to every core in the set before waiting for any ACK.
2. Collect EXITED, using one shared deadline.
3. Re-read the still-silent cores once. The shared deadline bounds the all-dead
   case to a single timeout, but a core whose turn came after it expired never
   got a wait of its own; this second read decides such a core on its own
   evidence rather than on a peer's timeout.
4. Reset dispatch to IDLE and close every acknowledged window before starting
   any readback. Then read back each acknowledged window individually and
   drain all those reads before publishing any GM return gate. This lets the
   group's posted writes overlap without removing per-window completion evidence.
5. Store `AICORE_POST_CLOSE_RELEASE` to each closed core's gate. The stores are
   relaxed: step 4's drain is what orders them after the CLOSE they belong to,
   and the gates are independent of one another.
6. AICore observes that word and only then returns.

The AICore wait is unbounded on purpose. A bounded wait cannot help: below the
45 s op-execute timeout it would return while the AICPU may still be writing —
reopening exactly this race — and at or above it, STARS reaps the op first and
the timeout is unreachable. The guarantee is instead that the release always
happens, which is why retirement sits on every thread's exit path rather than
behind a latch. An AICore has no logging path, so the AICPU device log is the
only place a blocked worker is named: grep for
`AICore retirement: core N not released`. The host's `print_handshake_results`
covers `workers[]` only and does not read `teardown_gates[]`, so there is no
host-side view of the gate today.

The A2/A3 onboard worker uses `ld_dev` for an uncached/bypass read and
`dsb(DSB_DDR)` before return. The simulation uses an atomic acquire load.
`ld_dev` is not being treated as a general-purpose atomic RMW or as a C++
acquire operation; this is a single-writer, per-core handoff. The platform's
MMIO completion ordering and visibility of the GM publication are distinct
parts of the protocol. A release store alone does not flush posted MMIO.

Emergency shutdown uses the same retirement helper and an ownership latch,
so normal finalization does not subsequently write to already released
workers' registers. A core that misses the deadline is not closed or released;
responsive peers still retire. The existing host recovery path remains
responsible for unresponsive cores and cores whose windows never opened.
This patch does not introduce a new reset operation or recovery policy.

## Cache-line and generation ownership

The return gate lives in `teardown_gates[]`, a separate array beside
`workers[]` in the device-copied image — not inside `Handshake`:

| Storage | Writer/access | Purpose |
| ------- | ------------- | ------- |
| `workers[i]` — one 64-byte line | Existing cached report/task publication protocol; the AICore flushes the whole line with `dcci(..., CACHELINE_OUT)` | Startup identity, ready report, dispatch-payload pointer |
| `teardown_gates[i]` — one 64-byte line | AICPU atomic stores; AICore bypass reads | Post-close permission to return |

The gate must not sit in a line the AICore writes back, or a stale cached
report writeback would overwrite the AICPU's release. Keeping the two in
separate arrays makes that hold by construction rather than by field offset:
no cached write or cache maintenance targeting `workers[i]` can reach the
gate line. `static_assert(sizeof(Handshake) == 64)` pins the report line, and
`AicoreTeardownControl` is `alignas(64)` with its own size assertion.

`Handshake` therefore keeps its 64-byte layout and stride. HBG shares that
definition with A5, and A5 is unaffected: its handshake image is byte-for-byte
what it was, and its execution protocol and the public Python API are
unchanged. A5 does link the `teardown_gates[]` array — it grows the device
image by one cache line per worker — but never reads or writes it.

The boot leader zeroes the whole gate array before publishing handshake setup
and before any register window opens; the write barrier that follows is what
orders the reset ahead of both, so the preceding launch's value of 1 cannot
release a new launch. Each cell is 4 bytes inside a 64-byte-aligned block, so a
straggler still bypass-loading its gate cannot read a torn value. Reuse assumes
the existing runtime lifecycle: the previous launch has completed before the
same storage is re-armed. This is not a claim that one runtime image supports
concurrent independent runs.

## Evidence and reproduction boundaries

The executor regression links each **production** A2/A3 AICore execution loop
against simulated register storage. With CLOSE deliberately withheld, all
four AIC/AIV checks fail on the unmodified main implementation at
`fab1a41e2fd5bbefb9eb18e59609876e63297e98`: the worker returns immediately.
With the handoff, those checks pass. Companion tests check eventual return,
32 launches reusing the same gate, group ACK ordering, partial timeout, and
invalid-target rejection.

```bash
cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build --parallel 2 \
  --target test_a2a3_tensormap_and_ringbuffer_retirement test_a2a3_host_build_graph_retirement
ctest --test-dir tests/ut/cpp/build -R '^test_a2a3_.*_retirement$' --output-on-failure
```

These are software-ordering tests; they do not emulate silicon register
retirement hazards. The motivating downstream L1 ACLGraph investigation
separately found that removing only the post-close wait changed a passing
1280-replay run into a matching vector timeout at replay 2. The fault PC was
at the persistent AIV wrapper's end, not inside the later native operator.
That is downstream integration evidence, not a claim that upstream main
already supports or reproduces that L1 graph workload. No undocumented
internal hardware state-machine failure is asserted here.

The PR's validation record distinguishes new upstream tests from this
historical downstream evidence and lists any unavailable toolchains.
