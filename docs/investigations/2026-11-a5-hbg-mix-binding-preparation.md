# A5 HBG early MIX Binding publication

**Date**: 2026-11-16
**Verdict**: withdrawn at the user's request

Commit `e97cbc0a4` moved Binding writeback before the arrival check for GM
and SSBUF MIX, retained 16 bytes of metadata in each local group state,
and removed a second context/error check within one GM preparation call.
The corresponding Ready publication skipped its previous Binding writeback.

The twelve-case paired AICore L0 mean sum was 20.338691 -> 20.452534 ms
(+0.560%). Latest layers 19-21 regressed 1.381% individually. Each arm used
one warmup and ten samples on matching automatically allocated devices;
two-device samples used a Device rendezvous and per-round rank maximum.
Eleven replay cases validated numerically; native MoE was performance-only.
The candidate passed 880 existing CPU checks and all 16 A5 device IR scans.

This change did not establish an overall latency benefit. It was initially
retained under the allowance for redundant-read reductions with at most 1%
aggregate regression, but the user subsequently rejected this implementation.
That specific decision overrides the general allowance for this candidate.
No new benchmark was run for the rollback.

The rollback restores the previous Binding publication timing, metadata
reads and context checks, including the previous local-state sizes. It keeps
`a5d46bfd6`'s single-writer signal stores and the existing MIX ACK-observation
reuse. The installed HBG binaries are restored from the verified build of
that retained revision; the withdrawn candidate's measurements remain history.

Do not restore this patch solely because its regression was below 1%.
A revised preparation design needs a new rationale and comparable L0 evidence;
a shorter publication scope alone is not evidence of a whole-run gain.

Evidence is local under `artifacts/single-writer-signal-20261116/`:
`SUMMARY.json` retains the case results and current decisions, `mix/` contains
validation logs, and `mix-before-publication.json` / `mix-after-publication.json`
record the L3 scope analysis. Frozen source, binaries and paired raw results
are under `artifacts/c2a-handoff-20261115/mix-preparation-ahead-20261116*`.
