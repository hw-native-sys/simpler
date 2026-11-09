# CCU deferred completion on A5

This scene uses the public Worker path, an embedded ABI 2 builder module,
the PR1 Session provider, and the PR2 TMR completion engine on two A5 devices.
The `smoke` case runs four rounds; `thousand_rounds` runs 1000 rounds per rank.

Each round initializes domain-owned source, destination and scratch tensors,
submits a CCU request from AIV, and returns without waiting. A dependent AIV
consumer checks the received data, the unchanged source and scratch, and both
destination guards. The next round depends on that consumer. Host validation
checks every round's status, root, count and offset, then checks domain cleanup.

The two embedded entries implement broadcasts rooted at rank 0 and rank 1.
Rounds alternate 64/1024 elements, cycle eight 64-byte-aligned offsets and use
different source data on each rank. Both entries receive three positional
tensors and one scalar; users do not supply callable IDs or platform handles.

The test builds `kernels/host/ccu_positional.cpp` with the native `CXX` compiler
(default `c++`), `-std=c++17 -O2 -shared -fPIC -I src/common`. It embeds the
resulting bytes in `CcuFunction`; no module path is passed to a child worker.
The usual scene compiler builds the orchestration and AIV kernels. All three
PRs and their runtime/extension build products must be present before running.

After the repository A5 architecture precheck, use the hardware queue:

```bash
task-submit --device auto --device-num 2 --timeout 600 --max-time 600 \
  --run '.venv/bin/python -m pytest tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion --platform a5 --device "$TASK_DEVICE" -v'
```

Select `-k smoke` for the initial run. Use the project-local `.venv` with the
combined runtime and Python extension installed. No fake backend or direct SDK
launch substitutes for the normal Worker path.

Domain creation and reuse select the required registered callable handles with
`ccu_callables=(registered,)` before the first task submission. Registration alone
does not prepare a callable on every domain.

## Hardware cases

| Case | Checks |
| ---- | ------ |
| `smoke` | Four rounds, two embedded entries, both broadcast roots, data and guards |
| `thousand_rounds` | 1000 rounds per rank with changing addresses, lengths and data; no per-round Host preparation |
| `invalid_and_cross_domain` | Foreign-domain addresses, oversized ranges, misalignment and invalid argument metadata are rejected; a valid request still succeeds afterward |
| `test_ccu_retained_domains_reuse_release_and_recreate` | Cross-run reuse, distinct domain data, release A while B remains valid, recreation with a new identity, and rejection of old public handles/references |
| `test_ccu_id_zero_isolated_after_new_callable_directory_publication` | Two callables both use ID 0 with different broadcast roots; late preparation republishes the directory without invalidating the first instance |
| `test_unused_failing_builder_isolated_from_selected_callable` | Registered A runs and reuses its domain while unselected B never invokes its builder; explicitly selecting B triggers real rank-local rejection, preserves ownership until reclamation, and allows A to recover on a new domain |
| `test_single_rank_prepare_failure_rolls_back_before_recovery` | The real rank-1 builder rejects preparation; both ranks roll back before a new callable/domain completes normal work |
| `test_ccu_is_pending_after_aiv_fin` | Delayed peer submission leaves the request pending when a full-width sync-start AIV cohort runs after the sender's FIN; the dependent consumer checks completed data |
| `test_ccu_failed_request_reports_error_and_reclaims_domain` | Unequal peer scalars produce error 105; the exact failed task's successor is published but undispatched, then the domain and Worker close cleanly |

The unused-builder isolation case is an additional regression; the historical
eight-case hardware evidence below does not include it.

The pending witness and consumer write disjoint result slices, so they have no
ordinary result-tensor dependency. The control flag is only an instantaneous
observation. The main evidence is the post-FIN pending snapshot together with
the consumer's correct data. The failure case reads a captured scheduler record
for the failed task and its successors: the scan must be complete, a successor
must exist, and its dispatched block count must be zero. An unchanged Host
sentinel is not evidence because failed calls skip ordinary tensor copy-back.

## Failure and cleanup contract

A confirmed CCU operation failure remains a failed call even when all device
work has stopped. Only exact error 105 with explicit AICore retirement proof,
no other callback error and successful terminal publication can complete the
SDK kernel normally. Host reads the original error after independent stream
completion and CCU drain; it withholds outputs and preserves the exception.
An early observation wait must still publish that exception and keep the child
control channel available for domain cleanup when native destruction and last
use are proven. Error 106, unconfirmed readers, failed retirement or failed
publication keep the existing non-success and ownership-retention behavior.
A closed CCU Session does not authorize reset while its parent communicator
remains live.

## Validation status (2026-11-08)

| Validation | Evidence |
| ---------- | -------- |
| CPU and Python | Eight CTest groups and 38 focused Python tests passed for the final candidate; the C++ code matches `9fd5b9cc` |
| Final eight-case A5 run | `e8d6081b5264b87a879a7f6faa883d6468950e11`, task `task_20261108_173610_121532929008`, devices 2 and 3: all eight cases PASS, exit 0 |
| Final run logs | `extended-e8d6081b-full-20261108-173610/` |

| Final hardware case | Reported case duration |
| ------------------- | ---------------------- |
| `smoke` | PASS, 12.0 s |
| `thousand_rounds` | PASS, 15.0 s |
| `invalid_and_cross_domain` | PASS, 5.5 s |
| Retained domains and recreation | PASS, 12.0 s |
| Two callable ID-0 instances | PASS, 12.0 s |
| One-rank preparation failure and recovery | PASS, 17.0 s |
| Pending after AIV FIN | PASS, 12.1 s |
| Confirmed FAILED, domain reclamation and normal close | PASS, 13.0 s |

Both ranks in the FAILED case reported a complete successor scan and no
successor dispatch. The case also confirmed physical domain reclamation and
normal Worker close while preserving the expected error 105.

Earlier combined snapshot `68e00c466` passed four and 1000 rounds through
Worker/TMR with normal cleanup. Its AIV wrapper reads Tensor metadata field by
field from GM, fixing the earlier first-send descriptor-copy failure. On
`9fd5b9cc`, the seven non-FAILED cases passed in 87.36 seconds; the FAILED case
proved stream completion, error 105 and an undispatched successor, but exposed
the early-wait cleanup-control bug. That partial run is not eight-case acceptance.

This suite covers the listed two-rank cases, not arbitrary hardware failures,
forced SDK close failures, more than two members or concurrent requests in one
Session. Unconfirmed shutdown remains a separate boundary. Standalone PR1
results remain component evidence. See the
[integration contract](../../../../../docs/design/ccu-pr3-requirements.md).
