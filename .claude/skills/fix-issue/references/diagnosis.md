# Diagnose a defect

Use this before changing code for a reported defect whose cause is not already
established. The main [fix-issue](../SKILL.md) skill decides whether the work is
a bounded repair or a `develop-task`; this reference is the diagnosis loop that
produces the failing repro, the cause, and the regression test.

## Phase 1 — Build a feedback loop

Everything else depends on a signal that goes red on this defect. Build one
before forming a theory; reading code first is the failure mode this reference
prevents.

Try loops in roughly this order:

1. A pytest test at the narrowest public seam (`tests/ut/`).
2. A single `@scene_test` case, run standalone with `--manual include` on the
   sim platform.
3. A replay of a captured artifact: args dump, device log, DFX output, or a
   saved input that reaches the failing path.
4. A differential run: the same input through sim and onboard, old and new
   runtime, or two configs.
5. A fuzz or shape/dtype sweep when the failure is "sometimes wrong".
6. `git bisect run` against a known-good and a known-bad commit.
7. `task-submit` for a device-only loop, with `ASCEND_PROCESS_LOG_PATH` set to a
   per-run directory so the device log is isolated.
8. A small human-in-the-loop script when a person must act; have it record the
   action and its output.

The loop is usable when it is red-capable, deterministic (or has a high enough
reproduction rate to debug), and fast enough to repeat. Prefer one that runs
unattended; a loop a person must drive each iteration is slower but still a
loop, so use it rather than proceeding without a signal.

For a non-deterministic bug, raise the rate with repetition, parallel runs,
stress, or narrower timing windows; do not declare it undebuggable after one
run.

If no loop is possible, stop and say so. List what you tried and ask for the
environment, a captured artifact, or permission to add temporary
instrumentation. Do not proceed to hypotheses without a signal.

## Phase 2 — Reproduce and minimise

Run the loop and confirm it produces the user's exact symptom, not a nearby
failure. Capture the error, wrong output, or timing so the fix can be checked
against it.

Then shrink the repro one input, caller, config, or step at a time, re-running
after each cut. Stop when every remaining element is load-bearing. The minimal
repro is both a smaller hypothesis space and the regression test.

## Phase 3 — Rank hypotheses

Write three to five falsifiable hypotheses before testing any of them. Each
must state a prediction: "if X is the cause, changing Y makes the bug
disappear; changing Z makes it worse." A hypothesis with no prediction is a
guess; discard or sharpen it.

Show the ranked list to the user before instrumenting. They often know which
one was already ruled out or which change just landed. Proceed with your
ranking if they are unavailable.

## Phase 4 — Instrument

Map every probe to a specific prediction. Change one variable at a time.

Prefer, in order: a debugger or REPL inspection; a targeted log at the boundary
that distinguishes the hypotheses; never "log everything and grep".

Tag temporary diagnostics with a unique prefix such as `[DEBUG-a4f2]` so cleanup
is one grep. On AICPU hot paths, never log per task or per scope; gate
diagnostics to a high-water mark, a sample interval, or the cold/stall path.
Unconditional logging can trip the op-execute timeout and mask the bug; see
[codestyle.md](../../../rules/codestyle.md) rule 7.

For a performance regression, logs are usually the wrong instrument. Establish
a baseline first with `benchmark`, `perf-example-device`, `profile`, or
`dfx-analyze`; then bisect or diff the measurement. Measure before changing the
code.

## Phase 5 — Fix and lock it down

Add the regression test before the fix when a correct seam exists. A correct
seam exercises the real failure pattern at the call site; a too-shallow unit
test gives false confidence. If no correct seam exists, record that as a
finding: the architecture is preventing the regression barrier.

Then apply the fix, watch the regression test pass, and re-run the original
Phase 1 loop against the unminimised scenario. The fix is not confirmed until
the original symptom is gone.

## Phase 6 — Clean up and record

- Re-run the original repro; confirm it no longer reproduces.
- Confirm the regression test passes, or document why no seam exists.
- Remove all `[DEBUG-...]` instrumentation and throwaway prototypes.
- State the confirmed cause in the commit or PR message.
- If the bug revealed a missing seam or hidden coupling, hand the architectural
  finding to [develop-task](../../develop-task/SKILL.md) after the fix, not
  before.

## simpler-specific triage

- **Onboard timeout or `507018`** — do not call it a deadlock from the host
  error. Read the device log and classify the mechanism first; see
  [running-onboard.md](../../../rules/running-onboard.md#507018-triage-classify-the-mechanism-before-concluding).
- **Test passes alone, fails in the sweep** — find the cause before changing
  the test. It is a real defect when a test leaves shared state behind, or when
  the code under test is order-dependent; it is an isolation requirement only
  when the case is already quarantined for a known resource conflict. Check for
  a quarantine marker (`sdma`, `sdma_fault`, `manual`) and bisect the sweep to
  the test that precedes the failure; see
  [testing](../../testing/SKILL.md#running-tests).
- **Wrong platform or a `507899` cascade** — run
  [onboard-arch-precheck](../../onboard-arch-precheck/SKILL.md) before treating
  it as a runtime bug.
- **Unlocked hardware run** — if `task-submit` is available, every device-owning
  run goes through it; an unlocked contradiction with CI is an environment
  difference until ruled out. See
  [running-onboard.md](../../../rules/running-onboard.md).
