# A5 HBG payload prefix writeback

## Decision

Retain full `DispatchPayload` writeback. The bounded-prefix candidate does not
establish a useful performance benefit on Qwen, despite reducing the number of
cache-line operations for ordinary tasks with few arguments.

The accepted control is `224068d1d`; the experimental code is preserved in
`d0c3aa795`. The experiment changes no payload layout or asynchronous protocol.

## Candidate and measurement

`DispatchPayload` occupies eight 64-byte lines. The candidate writes the control
line, the actual tensor/scalar argument prefix, and the final line containing
the fixed context pointers and `GlobalContext`. Ordinary dispatch passes the
checked argument count through local variables. Other callers retain the
48-argument default and full writeback. Publication barriers remain in place.

Device 5 runs the frozen control followed by the candidate under one exclusive
allocation. Both use the same 37 computation kernels and the same 40-layer,
batch-16, sequence-3500, seed-1234 Qwen inputs. Each arm has five warmup rounds
and 50 steady rounds; L1/L3/L4 captures are separate from those timings.

| Metric (ms) | Control | Candidate |
| ----------- | ------- | --------- |
| Device wall median | 35.687778 | 35.830123 |
| Device wall mean | 35.696875 | 35.824212 |
| Device wall minimum | 35.500135 | 35.652486 |
| Device wall maximum | 35.867297 | 35.977134 |
| L1 first-kernel to last-kernel window | 36.521333 | 36.409968 |
| L1 causal-path kernel time | 16.805221 | 16.775220 |
| L1 causal-path gap | 19.716112 | 19.634748 |

The uninstrumented median increases by 0.142345 ms, while the independently
instrumented gap decreases by only 0.081364 ms. This sequential pair does not
prove a small regression or improvement. It provides no reason to retain the
additional argument-count plumbing. The measurements do not identify whether
dynamic loop control, generated code layout, or cache behavior explains the result.

## Validation and evidence

Five CPU targets cover payload contracts, dispatch, MIX, asynchronous control,
and ready pools. Seven device scene tests and one standalone synchronous
dual-AIV diagnostic pass. The candidate passes full output/K/V validation and
the L4 protocol audit. No full-runtime performance matrix was run.

Only the AICore runtime binary changes; Host and AICPU hashes match the control.
The AICore build takes 187.11 seconds with a concurrency ceiling of 16.
Frozen libraries, source diff, per-round samples, audits and reports are retained
locally under `outputs/a5-hbg-schedule-8ms/payload-prefix/`.

## Shared PreparedCall producer and consumer experiment, 2026-11-13

The Build/Execute implementation `aabab11ec` uses a separate 512-byte immutable
PreparedCall. A second candidate maintained its header, populated argument
prefix, and fixed context-pointer tail on both the builder and receiver. Bits
40–47 of the atomic publication word carried the prefix extent; only a successful
Claim authorized decoding that extent and refreshing the data. Both trailing
DSBs and the existing Binding publication remained in place.

The existing DSV4.1 suite used one warmup and ten measured runs per case, with
full `chip.run.runner_run.device_wall` timing. The figures below are sums of
fourteen non-communication case means, not one forward duration:

| Version | Sum (ms) |
| ------- | -------: |
| Pre-change main `2f61d0c6c` | 19.9902648 |
| Shared Call with full maintenance | 20.5123812 |
| Shared Call with bounded maintenance | 20.6372252 |

The candidate increased this aggregate by 0.6086% against the shared-Call control
and by 3.2364% against pre-change main. The samples use different allocation
times, and the control's SWA/lib row also used a different device; this does not
prove a small regression mechanism. It supplies no evidence for retaining the
added extent protocol. All eighteen numerical checks passed. The four
communication cases remain reported but excluded from the performance gate.
Six existing CPU targets passed 1681 GoogleTest cases;
the A5 IR scan covered 14 translation units without findings.

The candidate is withdrawn. Existing SWA/lib L3 evidence places most first-launch
handoff time in admission publication and worker adoption, while Call receive
maintenance occupies a much smaller part of those intervals. That evidence
does not bound other blocks or predict an L0 percentage; it explains why an
instruction-count reduction alone was insufficient grounds for this experiment.
Further proposals must first identify the actual critical path in swimlanes.

Evidence is under `artifacts/build-execute-separation/`: `valid-lines-rejected.patch`,
`benchmark18-valid-lines/`, `runtime-valid-lines/`, `cpp-valid-lines-results.json`,
`ir-valid-lines.log`, and `critical-gap-audit/`. The shared-Call control is retained
in `runtime-targeted-dcci/` and `benchmark18/`.
