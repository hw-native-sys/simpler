# CCU Allocated Window Validation

A5 validation uses Linux aarch64, CANN 9.3.0 and the pinned PTO-ISA checkout.
Each section identifies the source snapshot it covers.

## User-Owned Mission Lifecycle

The update starts from PR head `85b8c095d50604622ab15d50d0ef92779b9bd8e3`.
A5 hardware validation ran through `task-submit` in an independent checkout
and venv with rebuilt bindings and both runtimes. Normal two-card tasks used
tree `e28461df4001c443bb0f8d4b28c466337091d67a`. Fault and three-card tasks
used tree `af8b93606b4a35354a114050c2bdba63f27501c5`, which corrects only the
fault fixture's timeout override. Local and remote Git trees matched.

| Case | Task | Devices | Result |
| ---- | ---- | ------- | ------ |
| Registration reuse, disabled domain, two serialized requests within one L2 run, callable replacement | `task_20261112_200826_3568623089` | 6,7 | Four cases passed |
| Prepare overlap, TMR real W16, explicit exit-only and mixed requests | `task_20261112_200720_22791316025` | 6,7 | Three cases passed |
| HBG real W16 and two TMR W1000 cases | `task_20261112_200856_31289113566` | 6,7 | Three cases passed |
| Missing exit, missing trigger and missing done | `task_20261112_201120_45057421981` | 6,7 | Three cases passed |
| Three-rank all-peer, reversed subset and shared registration, 16 runs each | `task_20261112_200856_31232130493` | 1,6,7 | All three cases passed |

The same-L2 test checks both distinct outputs from one mission before its stream
retires. Each W1000 case exercises 1,000 Host runs and checks the complete 32 KiB
regions and untouched ranges. The exit-only variant mixes business and exit-only
requests. The Host timeout case completes ordinary tasks before the CCU exit
deadline expires. The missing-trigger and missing-done cases observe a completed
AIV task with a pending CCU completion at the L2 scheduler timeout. All three
fault cases verify closed admission, retained window ownership and controlled
chip-child exit.

Focused validation also passed 98 Python cases, nine C++ CTest targets and
full PR-file Linux pre-commit, including clang-tidy and pyright. The historical
round-171 done timeout remains unexplained.

## Historical Explicit Registration Validation

The final tested source tree is `bff35ca416399318f50bfddd53c01eef6615a708`, based on
main `8a78542bda22d69d51cf88b5ba79de46e19`, before this report update. The initial
two-card snapshot was `9759996d03fff0117ad8b4f0dd918a0bb1085573`. The final snapshot
adds a Python subset-control fix and its regression tests; native sources are
unchanged. Native modules, bindings and both A5 runtimes were rebuilt in an
independent checkout and venv. Tests explicitly call
`Worker.register_ccu_callable(handle)` before allocating W.

All hardware cases use queue-managed automatic device allocation through
`task-submit`, with build and runtime logs in the validation checkout.

| Case | Task | Devices | Result |
| ---- | ---- | ------- | ------ |
| Native registration/reuse/replacement and disabled domain | `task_20261111_220508_174580524178` | 0,3 | Four cases passed |
| TMR real W, 16 runs | Same task | 0,3 | Passed |
| Partial/no-kernel submission, 16 runs | Same task | 0,3 | Passed |
| Prepare B while A is held at trigger | Same task | 0,3 | Passed |
| HBG real W, 16 runs | Same task | 0,3 | Passed |
| Missing done, retained ownership and controlled child exit | Same task | 0,3 | Passed |
| TMR real W, 1000 runs | `task_20261111_222148_25721162027` | 0,3 | Passed, 33.83 s |
| Three-rank full/reversed subset and shared registration, 16 runs each | `task_20261111_223734_345538910917` | 1,2,4 | All three cases passed |

The two-card task passed all nine cases and exited successfully. The W1000 task
also exited successfully. These are separate Host `worker.run` calls, not an
inner-L2 submission loop. Tests check complete 32 KiB regions, dynamic lengths,
offsets and tokens, W reuse and fresh W allocation.

CPU validation on the final snapshot passed 266 Python tests, with three skips and
13 Torch-dependent tests deselected, and all 12 focused C++ targets. Full PR-file
Linux pre-commit passed, including clang-tidy and pyright. The previous CI head's
10 Python failures were reproduced as test-fixture incompatibilities with explicit
registration and corrected here. Its packaging failure occurred while
fetching the pinned PTO-ISA dependency, before product tests.

The first three-card task `task_20261111_220658_1836231697` passed full-member W
communication but failed both subset cases: a control-path check incorrectly
compared full-context registration count with subset size. CPU regression
reproduced both failures before the fix. Controls now validate and select each
registration by L2 worker index; all three hardware cases passed on retry above.
The historical round-171 timeout remains unexplained; this W1000 pass does not
diagnose it.

CI additionally exposed 21 URMA fixture setup errors from missing CCU cleanup
link dependencies. The independent CPU environment reproduced all 21 errors.
Linking the real owner/lifecycle code and rejecting unexpected CCU token requests
in the URMA-only mock restored all 108 URMA/workspace tests. This changes test
support only; hardware-tested production sources remain unchanged.

## Historical Communication-Context Registration

The feature source tree is `bc138c024e375a4eeb87c2b15687fd3fea97f668`, based on
main `3e263b58ff01fb9315fd3a53136c6344801e82ab`. Local and independent remote Git
trees matched before this evidence-only update. Native modules, bindings, both
A5 runtimes and the simulation runtimes were rebuilt in the remote environment.
The final test snapshot is `426a6612ff7d406fd2f97d27896b7f1dd23104f8`;
production sources match the snapshot above. Only test corrections distinguish
it: domain-rank setup for subsets and expectations for the extended wire layout.

Registrations belong to the callable and full L3/L2 communication context;
`Worker.register_ccu_callable(handle)` performs that step before any window is
allocated, and changing W does not repeat registration. Explicit CCU enablement, one selected
domain per orchestration run, independent prepare resources and local AICPU READY
remain required. This extends the domain-scoped topology accepted in the current
Issue discussion; it is not claimed to have reviewer approval. The finite
runtime EXECUTE/EXIT envelope is also an explicit extension of the Issue's
user-owned control-flow contract. See [the ABI contract](ccu-native-abi.md).

All hardware cases below ran through `task-submit` on A5 with automatic device
allocation. They compare complete 32 KiB regions, including source and guard
bytes, and test dynamic lengths/offsets, retained W and fresh W allocation.

| Case | Task | Devices | Result |
| ---- | ---- | ------- | ------ |
| TMR real W, 16 runs | `task_20261111_185742_20549017654` | 6,7 | Passed |
| TMR real W, 1000 runs | `task_20261111_190039_4030766316` | 6,7 | Passed, 32.99 s |
| K0 only, K1 only, neither, both; 16 runs | `task_20261111_185742_20403617153` | 6,7 | Passed |
| HBG real W, 16 runs | `task_20261111_190020_37226322235` | 6,7 | Passed |
| Prepare B while A is held at trigger | `task_20261111_190534_77126729754` | 3,4 | Passed |
| Native registration/reuse, disabled domain, callable replacement | `task_20261111_190118_49460021357` | 6,7 | Four cases passed |
| Missing done, retained ownership and controlled child exit | `task_20261111_190118_49472428606` | 3,4 | Passed |
| Three-rank A={0,1}, B={0,2}, shared registration | `task_20261111_192009_161972718904` | 1,4,5 | Passed, 16 runs |
| Three-rank full/reordered subset and all-peer access | `task_20261111_192011_162137916928` | 1,4,5 | Both cases passed, 16 runs each |

The partial-submission case checks that skipped kernels perform no business
writes and that every run stops normally. The overlap case holds actual CCU
streams at the trigger gate, verifies unchanged A parameters/Events/generation
after B preparation, rejects premature B activation, then completes both runs.
It also rejects disabled registration and a wrong SDK identity before successful
retry, with one native registration before W allocation.

CPU validation includes 262 focused Python passes, three skips and 13 deselected
tests, and 14 focused C++ targets including the actual HBG resident observer and
the callable-artifact collision regression. Full PR-file Linux pre-commit passed,
including clang-tidy and pyright. Broader interface regression passed with
542 Python tests and 83 skips. The complete C++ build succeeded; 494 of 499
CTest targets passed initially. Four failing targets had stale layout/padding
expectations, and one lacked the A5 simulation libraries. After correcting
the tests and building those libraries, all five failing targets passed on
targeted rerun; all 499 targets therefore have passing results across these runs.

Baseline tests first reproduced duplicate domain-scoped registrations, incorrect
domain-to-L2 peer mapping and early HBG completion. An initial A5 run then exposed
an empty registration dispatch caused by using Python-child IDs for device L3
workers; the corrected topology regression reproduced six failures before the
fix. A separate overlap harness failure passed the CCU mask as a commit-flag
address; its corrected test uses a valid 64-bit flag and passed above.
The initial three-rank tasks `task_20261111_185742_20563618495` and
`task_20261111_190020_3731278510` failed because setup passed a worker ID to
`copy_to_domain`, which requires a domain rank. Correcting that test call
produced the successful three-rank results above without a backend change.

The historical round-171 done timeout remains unexplained. The successful
current W1000 run is new acceptance evidence, not a diagnosis of that old failure.

## Data Path

`allocate_domain(workspace_engines=("ccu",))` allocates W, obtains its CCU token,
exchanges owner addresses/ranges/tokens and uploads a rank-ordered descriptor
table referenced by CommContext. It does not register a callable or launch a
kernel; each L2 execution owns launch preparation and activation.

AIV submits window references and scalars. Runtime validates the requested byte
ranges, resolves addresses and tokens from the domain, and publishes them to P
before triggering. The native kernel loads the parameters and executes CCU
`Write` through the HCOMM channel, waits for DMA completion, synchronizes with
its peer, and records done. A dependent AIV copies its local W for Host checking.
No AIV writes the peer window in this test. P's token and W's token are separate.

## Historical Prototype Results

The tested feature source tree is `77923019df366d137f0fe8dc8e4244e3301929c4`,
on base `da46d7a91`. The local and remote Git trees matched before publication.

- Two-rank W transfer, 16 Host runs: passed; devices 2,6; task
  `task_20261111_141025_15066207996`.
- Two-rank W transfer, 1000 Host runs: passed in 85.97 seconds; devices 5,6;
  task `task_20261111_141637_185337911056`.
- Domain-zero binding and ordinary/replacement callable regression, 16 runs:
  all three cases passed in the latter task; exit 0 and device locks released.
- Focused Python regression: 209 passed, 2 skipped, 13 deselected.
- CCU device submission, workspace, module, registration and run lifecycle
  C++ suites: all five passed.
- Linux changed-file pre-commit: passed, including clang-tidy and pyright.

The test uses a 4 MiB W per rank and checks the complete 32 KiB named data
region after each run. Each rank supplies different payload bytes; two kernels
write to separate peer ranges. Transfer lengths vary from 64 to 1024 bytes and
offsets vary in 64-byte steps. Checking the complete region detects incorrect
payloads and changes outside the intended destinations. Halfway through the
test, W is released and freshly allocated; other runs reuse retained W.

These are separate Host `worker.run` calls, not repeated submissions inside one
L2 execution. The existing binding supplies a ring channel; all-peer channels,
eight-rank collectives, Host-source W and same-mission request queues are not
validated by this test. Earlier local scalar output and cross-rank notification
tests did not establish this W data path. The previously observed unexplained
extended-run CCU-done timeout is not disproved by these successful runs.

## Historical PR2986 Integration Results

The integrated feature tree is `931150b7a20483f6f2eddcf91abfa8c9b9487c84`, based
on upstream main `0569c7796`. An independent checkout, venv, native binding,
simulation runtime and A5 TMR runtime were built. This version also preserves
upstream workspace configuration mappings and exposes `CcuConfig`.

- Task `task_20261111_142612_26095995963`, devices 5,6: 1000 real W transfers
  passed in 86.82 seconds. Domain-zero binding and ordinary/replacement callable
  regression cases all passed at 16 rounds. Exit 0; device locks released.
- 243 focused Python tests passed, 2 skipped, 13 deselected.
- Six C++ suites passed: device submission, workspace, native module,
  registration, owner and run lifecycle.
- Full Linux pre-commit passed, including clang-tidy and pyright.

The source tree IDs above precede this report-only addition. Native source
and test files match the validated snapshots. The earlier PR head's CI failed
while downloading GitHub Actions and fetching the repository; its logs show
network timeouts and early EOF before code checks ran.
