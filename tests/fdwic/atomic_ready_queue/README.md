# A5 ticket ready queue hardware acceptance

This probe calls the production `scheduler_ticket_reserve`,
`scheduler_ticket_publish` and `scheduler_ticket_try_pop` primitives directly
from real A5 kernels. It uses the existing HBG scene compiler and resident
runtime; no replacement queue implementation or simulator is used.

The queue objects below are private probe data. They are not additional
runtime ready queues and do not change task routing or model kernels.

## Device protocol and assertions

Each case runs three dependent logical tasks against the same INOUT backing.
HBG task dependencies retire every previous invocation before the next one can
enter. Every invocation also uses atomic epoch arrivals before initialization.
The queue controls, entry publication words, gates and claim counters occupy
isolated 64-byte lines. Payload lines have a single writer followed by a unique
claiming reader, with explicit cache publication/acquisition.

| Stage | Explicit event ordering | Acceptance |
| ----- | ----------------------- | ---------- |
| Publication hole | Producer 0 reserves entries 0–1 and waits; producer 1 publishes entries 2–3; a designated consumer observes NOT_READY before releasing producer 0 | Head remains 0 while the hole exists; subsequent consumption is exactly 0, 1, 2, 3 |
| Concurrent batches | Every producer reserves its first batch before any producer publishes; consumers claim while producers publish subsequent batches | Every ref has one claim; all eight payload words match; total count, sum and XOR match |
| Quiescent reuse | All participants acknowledge the completed phase; later invocations also depend on the preceding INOUT task | Same backing starts empty each epoch; no previous payload version or queue ref is consumed |
| Final entry | All participants confirm head is zero and rendezvous before attempting to claim the single entry | Exactly one winner and head equals 1 |
| Bounds | Empty-capacity operations and over-capacity reserve/publish use separate views | Correct result codes, no out-of-bounds publication and unchanged guard entry |

All waits observe atomic events. Five-second deadlines only bound failures;
no calibrated delays, sleeps or elapsed-time guesses establish an ordering.
Participant entry/return clocks supplement the explicit arrival checks.

## Cases and resources

All cases require one A5 device and a normal Release HBG runtime built from
the same source as the primitive header. Do not use DEV or simulation for
acceptance.

| Case | Participants | Producers | Concurrent consumers | Stress refs per invocation |
| ---- | ------------ | --------- | -------------------- | -------------------------- |
| AIC 8 blocks | 8 AIC | 2 | 6 | 32 |
| AIV 16 blocks | 16 AIV | 5 | 11 | 80 |
| MIX 8 clusters | 8 AIC + 16 AIV | 8 | 16 | 128 |
| MIX 32 clusters | 32 AIC + 64 AIV | 32 | 64 | 512 |

Producer/consumer roles follow logical participant rank, so the mixed cases
exercise both AIC and AIV code paths and cross-type publication. All tasks
request `sync_start` and fit the available physical execution positions.
The kernels include production headers directly; their individual memory
operations are not a substitute for runtime L3 measurements.

## Build and run

Use the project-local venv and the repository's pinned PTO-ISA. The scene
compiler builds the three lane callables and orchestration with its normal
onboard toolchain. It does not rebuild the scheduler runtime.

From the repository root, after installing/building a matching Release runtime:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
mkdir -p outputs/atomic_ready_queue/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/atomic_ready_queue/ascend"
task-submit --device auto --device-num 1 --timeout 900 --max-time 600 \
    --run ".venv/bin/python -m pytest tests/fdwic/atomic_ready_queue/test_atomic_ready_queue.py --platform a5 --device \$TASK_DEVICE --pto-session-timeout 540 -q"
```

The task timeout is a failure bound, not part of the protocol. The main agent
coordinates compilation and the device reservation with other work in the
shared checkout. No hardware result is established merely by collection or
successful compilation.

## Evidence

Each successfully returned invocation is checked for a fresh resident runtime
mode and exact device-produced records. The comparator writes a unique
`outputs/atomic_ready_queue/` directory containing:

- `state.i64.bin`: raw private queue/control/payload backing.
- `records.i64.bin`: raw per-participant and per-invocation summary rows.
- `result.json`: case identity, mode, count/sum/XOR and PASS/FAIL.

Raw data is saved before numerical/record assertions, including failures that
return to the host. A device timeout may prevent those copies; retain the
scene log and redirected device logs in that case. Keep the runtime/kernel
binaries and exact source revision with the enclosing acceptance run.

This probe proves neither whole-model performance nor the absence of all
hardware memory failures. The two-queue implementation still needs the
production SPMD/MIX/async regressions and model acceptance matrix.
