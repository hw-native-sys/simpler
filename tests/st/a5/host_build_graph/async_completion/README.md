<!--
Copyright (c) PyPTO Contributors.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-----------------------------------------------------------------------------------------------------------
-->

# A5 resident async completion: behavioral acceptance cases

These cases check functional behavior, with no performance thresholds. They are
explicitly selected with `--manual include` while resident async integration is
under development. An unavailable deferred context fails an executed case;
there is no synchronous fallback or expected-failure marker.

## Coverage

| File | Boundary and observable verdict |
| ---- | ------------------------------- |
| `test_sdma_completion.py` | Two real peers; empty, one 32-byte tile, and 4 MiB SDMA. Check transferred data, dependent computation, guards, and an independent task observing a real event still pending after the submit kernel returned. |
| `test_condition_boundaries.py` | 0/1/63/64 conditions; 65 must fail; `Ge` below/equal/above threshold, zero and signed-int32 maximum; the second condition arrives later; predicate skip must release the successor. |
| `test_dependency_completion.py` | Ordinary fanin in both completion orders; two-block AIC, AIV and MIX masks 3/5/6/7; last computation versus last condition; a two-block sync-start MIX task. |
| `test_waiter_progress.py` | C-1/C/C+1 independent waiters, with the publisher submitted later. C comes from `rt_available_aiv_count()`. A bounded AIC rescue makes a blocking implementation fail without leaving an unbounded kernel spin. |
| `test_consumer_visibility.py` | Every candidate AIC first reads the old payload; AIV publishes new data; a synchronous AIC consumer must read the new value without test-side payload invalidation. |
| `test_run_reuse.py` | Two invocations of the same single-level Graph definition with different bindings, including consumers outside Graph EXIT; retained notification windows across increasing and repeated epochs. |
| `test_pending_close.py` | Close drains an accepted unfinished run; a never-satisfied condition reports a scheduler error and cannot run its consumer. Both negative/lifecycle cases first require a successful async probe. |

The numerical oracle derives expected executions from the input task graph.
Every planned physical execution and consumer must occur exactly once; inactive
lanes and guard regions must stay unchanged. Test-owned observations record
early entry, missing context and bounded-producer failures. Each successful run
also requires fresh swimlane evidence identifying the AICore resident producer.
No assertion reads a scheduler directory, lease, check queue or join counter.

The SDMA probe uses the PTO-ISA event's public completion-record API. It neither
derives private queue addresses nor atomically modifies an external completion
record. A numerically correct large transfer without an observed post-return
pending interval does **not** qualify that scenario. Transfer size and delays
are fixtures for exercising ordering, not speed requirements.

## Run

Use the repository's installed development environment and its pinned PTO-ISA.
From the repository root:

```bash
python -m pytest tests/st/a5/host_build_graph/async_completion \
  --platform a5sim --runtime host_build_graph --device 0 \
  --manual include --max-parallel 1 --pto-session-timeout 600
```

Simulation selects the COUNTER/lifecycle cases. Real SDMA requires two A5
devices, the normal SDMA workspace, and the repository's hardware queue:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5 || exit 1
task-submit --device auto --device-num 2 --timeout 1200 --max-time 1200 \
  --run 'python -m pytest tests/st/a5/host_build_graph/async_completion \
    --platform a5 --runtime host_build_graph --device "$TASK_DEVICE" \
    --manual include --max-parallel 1 --pto-session-timeout 600'
```

Every case owns its L3 Worker. Inputs/results crossing a fork use the public
buffer APIs; signal and payload storage live in retained CommDomains. Tests
never close a pooled Worker owned by another case. Runtime watchdog overrides
in the never-ready case are scoped by pytest's `monkeypatch` fixture.

## Integration status and remaining qualification

The initial baseline is `main` at `d57258765e9a145a8bd2545deff8ebd9e3f52a7d`,
with PTO-ISA `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`. Its resident executor
does not provide the deferred context needed by the positive cases. Compile
success and the predicate-skip control therefore do not establish async
correctness. Keep these cases explicitly selected until the implementation
passes the acceptance suite; then remove the `manual` markers.

The workload adapter currently uses the public `get_async_ctx`,
`save_expected_notification_counter` and `register_completion_condition`
interfaces present on that baseline. The capability manifest, resource
bindings, reserve-before-submit and return/seal interfaces in the design are
not available there. When those public interfaces land, wire them into
`helpers.py`, `kernels/waiter.h` and `kernels/aiv/sdma_submit.cpp` before claiming
qualification. Preserve the task graphs, boundary inputs and numerical/causal
verdicts; do not fabricate a capability field, patch scheduler state, or teach
the oracle an implementation's queue layout.

This first batch does not qualify every backend failure phase, dirty-cache
ownership, unsafe overlapping epochs, complete sync-start capacity boundaries,
or cross-peer COUNTER publication. The close case covers an accepted unfinished
run; it does not prove close began in a particular internal check phase.
Private state-machine races and instruction-level protocol tests belong in
the implementation's unit tests. Neither suite substitutes for A5 hardware
validation, and performance remains outside this batch.
