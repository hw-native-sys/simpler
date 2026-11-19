# Local closure from an owned sync_start block

Retain a separate optimization on baseline `2c8ef58df`: after its sole valid
block has retired, a sync_start position can close its local obligation from
the existing `sync_claimed` proof. Pure members still require empty slots; a
MIX coordinator also requires its group to be IDLE. The valid-count RMW must
succeed before setting the latch. No private or shared storage is added.

This proves local retirement, never global supply exhaustion. Zero-work
positions still strongly confirm the global cursor, the final valid claimant
still publishes the formal claim seal, and MIX peers still consume their
coordinator's actual close seal. Error, epoch and physical retirement gates
remain. Unlike the rejected terminal-count cache, no terminal cursor value is
memoized or inferred.

DFX marks the member with `MEMBER_FROM_CLAIM` and an unobserved cursor sentinel.
The auditor requires that member's unique real block claim and connects it to
closure. It does not invent a new global read or order another core's next
claim after this closure. Non-owners and MIX peers cannot use this proof.

The existing four-mask MIX close test fails before the change and passes after:
the completed coordinator's cursor FAA count changes from one to zero, while
its close-seal CAS remains one. Admission, GM MIX and admission-progress CPU
suites pass; protocol and causality tests pass 804 cases. The native A5 build
takes 72.01 seconds with a clean 16/16 IR scan.

Twelve paired AICore L0 means sum to 21.3901756 -> 21.3160697 ms (-0.346448%).
Eight cases improve and four regress. Each pair uses automatically allocated
devices, one warmup and ten samples; two-device runs rendezvous on Device
before launch and take the per-round maximum rank-local span. Eleven replay
cases pass numerical validation; native MoE is timing-only.

Evidence: `artifacts/atomic-reduction-20261115/owned-close-proof12.json` and
`artifacts/c2a-handoff-20261115/owned-close-proof/`. The latter's native inputs
match the tested runtime. Its `validation-manifest.json` and
`validation-source/` capture the later Python audit and CPU test corrections,
which do not change native build inputs.

A paired DSL MoE L3 capture records 500 -> 197 returned FAA calls inside
visible cursor-read scopes on each rank. Total returned FAA inside the kernel
window changes 6446 -> 6394 and 6595 -> 6483, or 13041 -> 12877 (-1.258%).
The visible target-scope count falls more than the total FAA count. Raw
records omit sub-1us parent scopes and do not carry atomic addresses: increased
traffic elsewhere and changes in scope attribution cannot be separated from
these counts alone. Both ranks
validate numerically and every scalar profile is complete. Evidence:
`artifacts/atomic-continuous-20261116/owned-close-l3/comparison.json`.

[Task-state experiment index](../2026-11-a5-hbg-terminal-claim-cache.md)
