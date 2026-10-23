# A5 URMA baseline fix: implementation and evidence

## Baseline and root causes

The implementation started from simpler
`8cb3cebd7f9a9c694540fd30601e3462d07572d3` and pto-isa
`c0d7148e95ef73bd12a73165fdce4b723a3b7e72`, matching the audited baseline.
The initial implementation did not change the pin. The local changes have since
been rebased onto main `2522f02bfead6071a70603d0ccb35431ca03139b`, which pins pto-isa
`b1d460ee8c001585ede36b68ba3678dc5a226bbf`. Pre-existing untracked design/training
files and the original branch were preserved. Unless explicitly labeled
post-rebase, the results below describe the original baseline, not the new
mainline/pto-isa combination.

The source audit was confirmed by deterministic CPU regressions, without
assuming a particular hardware failure:

| Contract | Old implementation evidence | Correction |
| -------- | --------------------------- | ---------- |
| Workspace ABI | Executable comparison with actual pto-isa: rank count offsets 8/4, SQ pointer offsets 16/8 | Updated native layout; every field, size and alignment compared with pto-isa types |
| Completion | Wrapped zero target returned READY when the CQE owner still indicated pending | Preserve jetty identity and independent BB/CQE targets, including valid zero handles; acquire successful CQE prefixes before publishing tails |
| Lifetime | All five baseline-source lifecycle scenarios aborted on manager finalization while a communicator was live | Retain partial managers and released domains; prove stop/drain, destroy child and parent communications, then reclaim backing |

The ABI and initial completion regressions were recorded before changing those
implementations. The lifecycle regression was replayed against an immutable
`git show` extraction of the audited source, using the native production-source
fixture. These are CPU/source-contract failures, not reproduced board faults.
The original hardware implementation was not run to induce a fault.

Two additional integration issues were identified. The original pto-isa types header
imported AICore-only tensor definitions. The new mainline pin makes the original
header Host-compatible, so the generated type-only view was removed during
rebase. Both Host and AICore now use the original pto-isa declarations. pto-isa's fixed HCCL
registration tags require domain isolation: each domain uses a public HCCL
subcommunicator, while retaining the existing pto-isa manager. No manager code or
private workspace configuration is replaced.

## Support and lifetime

See [the usage and support contract](../a5-urma.md). The supported path is A5
TMR Device URMA with PER_PEER, one row/jetty per peer, serialized producers,
contiguous equal-dtype/equal-size tensors and dense rank domains. The hardware
suite validates two-rank TGET plus a consumer. Shared-pool and multi-jetty
configurations, nonzero derived-window offsets, reused retained allocation IDs,
and full rings/slabs are explicitly rejected. Multi-producer arbitration,
notify, Host DDR and new HBG functionality are outside this change.

Normal destruction order is:

1. Close admission and complete the peer handshake.
2. Synchronize the device and verify SQ BB and CQE submitted/completed counters.
3. Destroy domain HCCL communicators and the base communicator.
4. Finalize pto-isa managers, then free contexts, imported mappings and local windows.

Logical domain release moves ownership to the base communicator's retained
collection. Failed Init also leaves ownership there because HCCL may already
hold registrations. A failed stop/drain/handshake/destroy quarantines the owner
and returns an error. Quarantine is deliberately retained for process lifetime;
it requires device reset/process termination for device-resource reclamation.
Normal domain retention is reclaimed at successful communicator destruction.

## Recorded validation results

| Validation | Result |
| ---------- | ------ |
| Actual pto-isa header ABI probe | PASS: rank count 4/4, SQ pointer 8/8 |
| URMA contract and execution-gate Python tests | 26 passed |
| Build-option Python regression (earlier combined run) | 75 passed, including the then-current 14 URMA tests |
| C++ tests without hardware | 244/245 passed in sandbox; the socket-only failure passed in isolation with local sockets allowed |
| Focused URMA C++ tests | 3 executables / 11 GTests passed |
| Default OFF A5 simulation sweep | 84 selected: 40 resource jobs, 14 HBG and 30 TMR tests passed |
| Explicit ON A5DT execution | Final four required items passed in 112.01 s on locked devices 0,2; zero skipped/missing |
| Default OFF A5DT sweep | HBG 27 passed / 1 skipped; TMR 48 passed. Resource phase: 38 passed / 4 URMA skips / 1 baseline failure |

The A5DT environment uses Ascend950DT_9581, CANN 9.3.0 and driver
25.6.rc2.b023, Python 3.11.6 and GCC 12.3.1. The local CPU/default build environment uses
Python 3.11.14 and CANN 9.0.0 on A3 silicon; no A5 hardware test ran there.

## Validation commands

All Python work used project-local venvs. A5DT validation used a detached,
independent checkout at the audited HEAD and pin; existing remote workspaces
and branches were not modified. Every device invocation used architecture
precheck and `task-submit` device locks.

```bash
# CPU ABI, production-host fault injection, and strict CI execution gate
.venv/bin/python -m pytest tests/ut/py/test_urma_contract.py \
  tests/ut/py/test_urma_execution_gate.py -q
# Build-option regression also run with the repository-local ccache directory
CCACHE_DIR="$PWD/build/ccache" .venv/bin/python -m pytest \
  tests/ut/py/test_runtime_builder.py -q

cmake -S tests/ut/cpp -B tests/ut/cpp/build
cmake --build tests/ut/cpp/build -j8
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure

# Default OFF simulation sweep, both runtimes
SIMPLER_ENABLE_PTO_URMA_WORKSPACE=OFF SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=OFF \
  .venv/bin/python -m pytest examples tests/st --platform a5sim \
  --exclude-level 4 --device 0-15 --pto-session-timeout 1200
```

Explicit URMA build, kernel compilation and queued hardware commands are in
[the usage guide](../a5-urma.md#build-and-verify). The strict runner records the
loaded backend, pto-isa build pin, binary SHA-256 values, selected/executed tests,
failures and JUnit results. The native teardown item checks `comm_destroy`'s
return value before any device reset, so quarantine cannot satisfy that test.

Local evidence is under `outputs/urma-baseline/` (generated artifacts, not
tracked source). Important files are `old-abi.log`, `old-scheduler.log`,
`old-host-lifecycle.log`, `new-abi.log`, `new-contract-tests.log`, `new-cpp-tests.log`,
`cpp-regression.log`, `cpp-socket-retest.log`, `python-tests.log`,
`default-a5sim.log` and `ci-change-detection.log`. Remote hardware logs and
verification JSON are copied into `outputs/urma-baseline/a5dt/`.
`a5dt/urma-on-final.json` records the final binary hashes and executed items;
`a5dt/urma-CMakeCache.txt` records ON/OFF options. The remote source hashes
were compared with the local final source manifest with no mismatches.

## CI and limits

PR and Daily use the same `_st-urma-a5.yml` workflow and strict runner. PR gating
reuses `a5_changed && st_affected`. The canonical change-detection script was
executed locally for platform source, pin, CMake, runner, example and workflow
changes; each enables the URMA job. Docs-only and UT-only edits retain the
existing category behavior. The default A5 lane explicitly sets both URMA and
RDMA OFF.

The strict gate has real pytest tests for skipped and empty collections, plus
unit coverage for failed setup/call/teardown, xfail and missing execution.
It also checks explicit overlay configuration and loaded artifact identity.
GitHub Actions execution remains unverified; local execution of the validation
entry point is reported separately. Daily coverage takes effect after the
workflow is merged to the default branch.

CPU tests cover 32-bit sequence wrap, owner wrap, independent BB/CQE counters,
pending/ready/error prefixes, repeated registration/polling, capacity rejection,
normal/partial destruction and failure quarantine. Hardware tests do not force
32-bit counter wrap or inject HCCL/driver faults. TPUT and transfers exceeding
256 MiB retain the adapter contract but do not have board acceptance in this
suite. No performance claim is made.

## Default-path baseline failure

The OFF sweep returned failure for
`test_comm_domain_under_visible_devices` on devices 1,2. An isolated rerun on
0,7 also failed. Running the underlying `async_notify` test directly exposed
`domain_alloc_via_ipc: MemSetAccess -> 107001` on logical device/rank 1 with
`ASCEND_RT_VISIBLE_DEVICES=0,7`.

For attribution, the independent remote checkout's Host communication source
was temporarily replaced with the exact audited `git show HEAD:.../comm_hccl.cpp`,
rebuilt with URMA/RDMA OFF and rerun on the same devices 0,7. It failed at the
same MemSetAccess call. The final source was then restored and rebuilt for
URMA verification. This proves that the observed default-path failure also
exists with the baseline Host implementation; its remapping/SDK root cause
has not been diagnosed or fixed in this URMA task. The full default hardware
sweep is therefore **not green**. Its four URMA skips and the HBG prepared-callable
skip are not counted as hardware URMA acceptance.

See `outputs/urma-baseline/a5dt/default-a5-device.log`, `default-isolated.log`
and `default-baseline-isolated.log` for the full and isolated evidence. The
independent issue is also recorded in the local ignored `KNOWN_ISSUES.md`.

## Mainline rebase and PR #2650 comparison

The working changes are on `fix/a5-urma-baseline`, based directly on main
`2522f02bfead6071a70603d0ccb35431ca03139b`. Because the changes were uncommitted,
they were saved in a task-scoped stash and applied onto a fresh branch at main;
the original branch's unrelated early-enqueue commit was not replayed. A tar
backup, patch and hashes are retained under `outputs/urma-baseline/pre-rebase/`.
All six pre-existing user files have unchanged hashes. There were no textual
merge conflicts. The generated Host header workaround was removed and the
new `TaskId::make` factory replaced the test's raw integer constructor.

Post-rebase validation: 33 Python URMA contract/gate tests and three C++ URMA
executables (11 GTests) pass. The ABI test includes the actual new pto-isa header.
The original header-generation guard and obsolete TaskId constructor failed
before these adaptations. Hardware builds and acceptance of this patch on the new pin, and the full
post-rebase simulation suite, have not been repeated. Main also changes A5 VMM access
descriptors to ACL-logical device IDs, so the old visible-device failure must
not be attributed to current main without another hardware run.

The CI comparison also exposed an integration gap: `.claude/` was entirely
non-code even though the URMA workflow executes its architecture-check script.
The canonical detector now treats executable helpers as code while preserving
Markdown-only skips. Seven tests execute its actual shell decision body for
helper, documentation, source, workflow and UT changes. The script-only case
reported `a5_changed=false, st_affected=false` before the correction. The first
workflow precheck also sources CANN before looking up its platform metadata.

[PR #2650](https://github.com/hw-native-sys/simpler/pull/2650) was inspected at
head `e22f05a79837f587e0c99da9c7af429f48210ca6`; it was open and unmerged. It
uses the same new pto-isa pin as main. It covers a2a3 SDMA and a5 SDMA/URMA/RDMA,
whereas this work changes the existing a5 TMR Device URMA path.

| Area | PR #2650 | This work | Disposition if #2650 lands |
| ---- | -------- | --------- | -------------------------- |
| Workspace interface | Uses public `Workspace`, `WorkspaceRequest`, `CreateWorkspace`, `DestroyWorkspace`, `AbandonWorkspace`; reads addr/bytes from the facade | Retains the manager and verifies private layout against real pto-isa types | Prefer the facade; remove scheduler layout decoding after public completion/retirement contracts suffice |
| Completion | Registers `{addr, expected, kind}` records, iterates records for multi-jetty events, polls CQE owner/status | Preserves separate BB/CQE targets and jetty identity; validates supported shape; handles zero/wrap and publishes tails after successful prefixes | Replace old encoding with records; preserve the behavioral tests and resolve retirement, wrap and capacity contracts |
| Teardown | Calls `DestroyWorkspace` before `HcclCommDestroy`; domain release also destroys workspace immediately | Stops/drains, destroys child/base communication resources, then finalizes/frees; retains failed or released owners | Keep lifetime fix, adapt ownership to the facade once failure ownership is safe |
| Multiple domains | Continues using the base communicator for domain workspaces | Uses public child communicators to isolate pto-isa's fixed registration tags; released domains remain retained | Keep isolation and retention behavior |
| Failed Init | Calls facade and then `AbandonWorkspace` on failure | Takes ownership before Init so partial registrations/backing survive until communication destruction | pto-isa facade failure contract needs adjustment before equivalent migration |
| Build | Adds CANN path to cache identity, workspace support definitions and library-cache handling | Explicit overlay/backend/pin validation | Complementary; keep #2650's build fixes |
| Regression/CI | Source checks, a CQE poller C++ test, reported demo runs; no workflow changes | ABI, wrap/prefix/error, release/rollback tests; four mandatory board items; PR/Daily URMA ON job | Keep and adapt tests plus CI; do not treat existing CI success as dedicated URMA execution |

Three source-level gaps prevent treating #2650 as completion of this baseline
task. They are not claims of reproduced board faults:

1. Its `retire_urma_cqe_record` is empty and its poller only reads the final CQE.
   pto-isa's record getters also only produce an address/owner expectation. SQ/CQ
   tails are published by pto-isa's Wait/Test polling path; the posting path can
   invoke a blocking drain when the ring is full. An explicit nonblocking
   retirement/queue-capacity contract is needed before replacing our tail
   bookkeeping while preserving deferred completion. The remaining Wait calls
   on registration errors also need an explicit failure policy.
2. At pto-isa `b1d460ee`, `CompletionRecordCount` returns zero for `handle == 0` and
   rejects zero BB targets; `GetUrmaCompletionRecordAt` rejects zero CQE targets.
   These values occur at 32-bit sequence wrap. Our wrap tests remain necessary;
   moving to the public record API requires fixing this upstream or explicitly
   rejecting a bounded configuration before it reaches the boundary.
3. The pto-isa facade deletes the manager internally when Init fails, before the
   caller can retain it. `AbandonWorkspace` only clears the returned object;
   it cannot undo that destruction. A CPU probe using the real pinned
   `workspace.hpp` and a fault-injected manager recorded
   `status=3 finalized_while_communicator_live=1 retained_impl=(nil)`.
   This verifies facade call ordering, not a hardware failure. The manager's
   own Finalize contract still requires communication destruction first.

The recommended integration is to adopt #2650's public API direction and carry
the lifetime, isolation, boundary tests and strict CI work forward. Do not
merge both scheduler implementations together. Public ownership/retirement
gaps should be resolved in pto-isa with a referenced fixed revision rather than
copying its manager, editing private workspace data or adding synchronous Wait.

## Unmodified mainline hardware comparison

A separate clean checkout of main `2522f02bfead6071a70603d0ccb35431ca03139b`
with pto-isa `b1d460ee8c001585ede36b68ba3678dc5a226bbf` built successfully
with URMA ON and RDMA OFF. On Ascend950DT_9581 with CANN 9.3.0, the existing
`urma_deferred_completion_demo` returned **1 passed, 0 skipped in 5.85 s** on
queue-locked devices 0,1. Both simpler and the actual pto-isa checkout remained
clean. Neither this patch nor PR #2650 was included.

The run used the project's local venv, architecture precheck, kernel compilation,
and the following queued test command:

```bash
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 2 \
  --run ".venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo --platform a5 --runtime tensormap_and_ringbuffer --device \$TASK_DEVICE --require-pto-isa --pto-session-timeout 1200 -v -s --junitxml=outputs/main-urma/verification.xml"
```

Build options, binary hashes, the full run log and JUnit are retained under
`outputs/urma-main-2522f02b/evidence/`. This is evidence that main's existing
peer-read/consumer smoke passes. It does not invalidate the deterministic
ABI/completion/lifetime regressions or constitute hardware acceptance of this
patch after rebase.
