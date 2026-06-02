# a5 AICPU filter affinity gate: Scenario B fail-fast guard not added

**Date**: 2026-06-02
**Verdict**: dropped — behavior is empirically deterministic per SKU; documented instead of code-guarded

## Question

PR #963 ships a filter-style AICPU affinity gate on a5 (`platform_aicpu_affinity_gate_filter`)
driven by a host-computed `ALLOWED_CPUS` table. The design assumes:

1. CANN's AICPU dispatch set == the device-side `halGetDeviceInfo(AICPU, OCCUPY)`
   bitmap exactly. Empirically verified on the Scenario A SKU
   (Ascend950PR, OCCUPY=0x1f8) using `tools/cann-examples/aicpu-thread-spread/`.
2. Launching `popcount(OCCUPY)` threads spreads exactly one thread per
   user cpu_id, so every cpu in `ALLOWED_CPUS` (a subset of OCCUPY)
   gets a representative thread, and the orch slot at
   `exec_idx == allowed_count - 1` is always populated.

CodeRabbit asked, on the PR: what if a future SKU or CANN version
breaks (1) or (2)? The orch slot would be empty, the scheduler would
wait for `orchestrator_done_` that never gets set, and the symptom
upstream would be a stream-sync timeout
(`aclrtSynchronizeStream rc=507000`) — slow to diagnose without
context.

Proposed change: add a host-side post-launch check that every
`exec_idx` slot got populated (need a device→host signal of which
slots survived the gate), fail-fast if not.

## What was tried

Nothing yet beyond the design discussion. The "fail-fast" path would
need a fresh device→host signal channel:

- the gate runs inside the AICPU kernel,
- the orch-empty stall happens later in the scheduler's drain (not the
  kernel that's spinning at the gate barrier),
- nothing today reports the surviving `exec_idx` set back to the host
  in time for a pre-execution check.

So it's not a one-line guard. Estimated effort: small (~50-100 lines)
but requires a Runtime field for "exec_idx populated bitmap" written
by the CAS-winner thread of the filter gate, and a host-side post-sync
check.

## Result

- Scenario A SKU (OCCUPY=0x1f8, 6 user cpus): runtime produces
  `ALLOWED_CPUS = {5,6,7,8,3}`, `launch_count = 6`, every slot
  populated, all 17 a5 ST tests pass.
- Scenario B SKU (OCCUPY=0x7ffe, 14 user cpus): not available on the
  dev box this PR was developed on. Predicted layout from the same
  algorithm:
  `ALLOWED_CPUS = {11,12,13,14,7}`, `launch_count = 14`. **Unmeasured.**

## Why not (now)

- The CANN AICPU dispatcher's "set == OCCUPY" property has held on
  every a5 sample we have. It's a behavior of the kernel-launch dispatch
  policy, not a property that would change per workload or per launch.
- If the property ever stops holding, the symptom is loud enough
  (`507000` after the AICPU sync timeout) to spot quickly, and the
  failed run's stderr already contains the diagnostic gate logs
  (`AICPU filter gate: allowed[i] = cpu_id N role=...` followed by
  per-thread `idx=X cpu=Y exec_idx=Z ACTIVE/DROPPED`). The empty slot
  shows up by inspection: an `allowed[i]` with no matching
  `ACTIVE(sched)` / `ACTIVE(orch)` line.
- The probability-weighted cost (failure × diagnosis-time) is below
  the work of the fail-fast plumbing and its ongoing maintenance.

If you reach for this doc because you hit `507000` on an a5 onboard
test, the diagnosis path is:

1. Grep the failing run's stderr for `AICPU filter gate:` —
   that's the gate's per-launch dump.
2. Cross-check `allowed[i] = cpu_id N` lines against the per-thread
   `thread idx=X cpu=Y exec_idx=Z ACTIVE/DROPPED` lines.
3. For each `i`, exactly one thread should show `exec_idx=i ACTIVE`.
   If any `i` has zero ACTIVE matches, CANN didn't dispatch to that
   cpu_id and the slot is empty — that's the Scenario-B-style
   regression this investigation documented.
4. Cross-check device-side OCCUPY with
   `tools/cann-examples/aicpu-device-query/` to confirm the static
   topology still matches what the runtime probe sees.
5. Cross-check CANN's actual dispatch with
   `tools/cann-examples/aicpu-thread-spread/` — that's the empirical
   "what cpu_ids did N threads land on" tool.
6. If `aicpu-thread-spread` confirms CANN dispatch ⊊ OCCUPY, update
   `compute_allowed_cpus`
   (in `src/a5/platform/onboard/host/aicpu_topology_probe.cpp`) to
   pick `ALLOWED_CPUS` from the *reachable* set, not OCCUPY.

## When to reconsider

Implement the fail-fast guard if **any** of these become true:

- A Scenario B SKU run shows an empty `exec_idx` slot in the gate
  diagnostic (`AICPU filter gate:` lines), proving the
  `dispatch ⊆ OCCUPY` assumption is tighter than assumed.
- A new CANN version changes the AICPU dispatch policy and the same
  empty-slot pattern shows up on a previously working SKU.
- A future test makes the gate enter a hot path (it currently runs
  once per Worker::run; if it ever runs per-iteration on a streaming
  workload, the `507000`-after-timeout cost becomes proportional to
  hot-loop count, not per-launch).

## References

- PR [#963](https://github.com/hw-native-sys/simpler/pull/963) — the
  affinity gate implementation.
- [`src/a5/docs/hardware.md`](../../src/a5/docs/hardware.md) §"CANN
  AICPU thread dispatch under varying launch budgets" — the empirical
  data behind the design.
- [`tools/cann-examples/aicpu-thread-spread/`](../../tools/cann-examples/aicpu-thread-spread/README.md)
  — diagnostic spike tool for reproducing CANN dispatch shape on any
  SKU.
- [`tools/cann-examples/aicpu-device-query/`](../../tools/cann-examples/aicpu-device-query/README.md)
  — device-side `halGetDeviceInfo` probe for verifying OCCUPY.
- CodeRabbit review thread on PR #963 (the proposal that led to this
  doc).
