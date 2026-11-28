# Design and Code Quality Review

Use this guide after establishing the current system's pain points and reading
the proposed mechanism. Assess how the change affects the surrounding system,
including its callers and analogous implementations. Apply the dimensions to
all code-changing PR types. Let the changed behavior, callers, supported
configurations, and deployment boundaries determine the depth and any additional
checks. Use primary documentation matched to the reviewed versions where an
external contract affects a conclusion.

## Assessment and Evidence

Quality effects are part of the change's benefits and costs. For each dimension,
compare the baseline with the proposed implementation and explain the resulting
impact on users, operators, or maintainers. Better architecture, reduced coupling,
clearer code, and stronger performance are benefits. Architectural degradation,
tighter coupling, more complexity, and resource regressions are costs, including
when they are intentional trade-offs for another benefit.

Record gains, losses, or no material change; a dimension may have both gains and
losses. Mark unverified effects or inapplicable dimensions explicitly. Cite code,
contracts, tests, or measurements, and distinguish author claims from additional
effects found during review. Existing issues matter only when the PR introduces
or worsens them, or its stated goal requires addressing them.

Use one evidence chain: **baseline pain → design change → quality effect →
benefit/cost → evidence**. Feed these results into the solution explanation and
combined assessment in report §§2–3, and weigh the trade-offs in §5. Shared
evidence can support several dimensions without counting one gain repeatedly.

For a concern, connect **changed code → criterion → concrete impact → suggested
action**. Distinguish repository requirements from design trade-offs and personal
preferences. Do not force a rewrite or speculative abstraction just to satisfy a
checklist. Route conclusions and findings into report §§2–4 as SKILL.md specifies.

### When an extension spreads a rule

Check whether an extension increases the number of places that implement the
same invariant, even when each edit follows the current design. Find its
producers, consumers, and validators. Name a plausible change to the rule and
show which sites would need to change together and what happens if one is
missed. When the PR expands this maintenance obligation, identify the owner
that should enforce the invariant and propose a bounded refactor or a concrete
reason to defer it. Base the decision on the coupling and effect of a missed
update. Identify which complexity is required by the product or hardware
contract and which comes from the current design. Judge severity by the
resulting risk and impact.

When the needed refactor spans more subsystems than the feature, check whether
it can land as a behavior-preserving prerequisite with independent verification.
Use the feature review to verify how the new callers use that prerequisite.

Inspect construction, conversions, and every public operation of a new type.
Find callers that still interpret the representation and check invalid inputs,
including values loaded through serialization. Review the interface against its
intended guarantees even when the implementation follows a prior design.

#### Example: LaneMask as a scheduling prerequisite

1. **Find the design pressure in the feature.** The initial spare-AIV change in
   #2676 (`564d2238`) spread raw `uint8_t` masks and hard-coded combinations
   through A5 TMR dispatch and queue accounting. The same lane semantics were
   already duplicated in three `ActiveMask` definitions, and passed as raw
   bytes across Host and device metadata. The review proposed a shared
   `LaneMask` migration before the feature. It assigned queue pointer tagging
   to `TaggedPtr` and demand accounting to `LaneDemandCounter`.
   [Initial feature review](https://github.com/hw-native-sys/simpler/pull/2676#pullrequestreview-5625250431)
2. **Review the prerequisite on its own contract.** In #2692, `1979c331` had
   unified the types and still exposed the raw bit API from the initial design.
   An invalid `SubtaskSlot` could also map to `AIV1`. The review traced actual
   callers to membership, activation, active-lane rank, and wire validation.
   Final head `8c2ed4c3` exposed those semantic operations and validated
   wire-loaded bytes at Graph intake. It kept the one-byte layout and tested
   the public interface, all 256 wire values, and invalid slots.
   [Prerequisite review](https://github.com/hw-native-sys/simpler/pull/2692#pullrequestreview-5646934772) ·
   [Re-review](https://github.com/hw-native-sys/simpler/pull/2692#pullrequestreview-5648286587)
3. **Verify the feature uses the new boundary.** #2676 resumed on top of
   #2692. The queue captures the original lane tag on enqueue and uses that
   tag on pop and requeue. Reading the reused task slot at those points could
   account for a different mask. The feature has regression tests for slot
   reuse and requeue. Raw encoding stays confined to the internal tagging and
   accounting implementation.
   [Queue ownership review](https://github.com/hw-native-sys/simpler/pull/2676#pullrequestreview-5625439276) ·
   [Feature re-review](https://github.com/hw-native-sys/simpler/pull/2676#pullrequestreview-5676675335)

Use the same approach to find shared invariants, review the type's actual
capabilities, and trace them through the dependent feature.

## Review Dimensions

1. **Architecture and responsibility.** Does the behavior belong at this layer?
   Does each component own a coherent responsibility, and does the change use
   the system's existing extension points? Check the relevant design contract
   and the impact on callers, runtimes, and platforms. Compare with keeping the
   current design or extending an existing mechanism where those are realistic
   alternatives. Explain why the chosen design resolves the identified pain.

2. **Decoupling and dependencies.** Trace dependency direction, exposed internals,
   mutable shared state, and initialization/order dependencies. Can a caller use
   the interface without knowing the callee's layout or private sequencing?
   Does changing one component now require coordinated edits elsewhere? Name
   the dependency and its impact; more interfaces or files alone do not prove
   better decoupling, and unnecessary indirection can make the design worse.

3. **Performance, resources, and scalability.** Identify the hot path and the
   relevant workload/scale before judging efficiency. Check algorithmic growth,
   allocations/copies, memory footprint and locality, locking/serialization,
   batching, backpressure, and idle versus per-task costs where applicable.
   Where work can accumulate, check queue and retry bounds, overload behavior,
   and recovery. Evaluate latency, throughput, and resource trade-offs against
   the intended use and any agreed budget. Follow the evidence rules below.

4. **Simplicity and readability.** Can a reader follow the control flow and
   invariants without reconstructing hidden state? Inspect nesting, repeated
   branches, overly broad functions, boolean-mode combinations, dead code, and
   unnecessary abstraction. Prefer a clear expression of the current problem;
   fewer lines, clever expressions, or a generic framework are not automatic
   improvements. Explain a concrete comprehension or maintenance cost when
   calling code unnecessarily complex.

5. **Reuse and consolidation.** For incremental logic, search existing callers
   and sibling implementations for equivalent helpers, validation, conversion,
   and protocols. Compare inputs, behavior, error and ordering guarantees,
   ownership, and platform constraints. If an existing abstraction fits, reuse
   it. If similar implementations enforce a shared rule, identify the common
   behavior and its owning layer, and check whether consolidation can preserve
   each caller's contract. Cite the implementations and the expected benefit or
   cost. When the contracts differ, identify the difference and any consistency
   obligations that remain.

6. **Naming and vocabulary.** Check that names express domain role, units,
   ownership, and state precisely where relevant, and use the repository's
   canonical terminology consistently across interfaces. Look for misleading
   abbreviations, one name with multiple meanings, and multiple names for one
   concept. Apply the local style rules; label an otherwise valid alternative
   spelling as optional rather than a blocker.

7. **Data structures and invariants.** Does the representation model the domain
   and its access patterns? Check field meaning, ownership, initialization,
   ranges/units, sentinel values, legal state transitions, and consistency among
   fields. Can invalid combinations be prevented by the type or checked at the
   boundary? Examine redundant fields, container choice, capacity/overflow, and
   reset/reuse behavior. For shared or serialized layouts, check field widths,
   alignment, versioning, and layout assertions against the actual contract.
   Do not prescribe host-side abstractions for a device wire format.

8. **Concurrency and resource lifetime.** Trace allocation, publication, access,
   completion, cancellation, and reclamation. Identify each synchronization
   relationship, lock order, and lifetime owner, including exceptional paths.
   Check contention, stale completions, reuse/generation boundaries, and shutdown
   where relevant. A race finding needs concrete concurrent paths and an
   interleaving; resource findings need a specific ownership or cleanup trace.

9. **Failure handling and recovery.** Inspect invalid input, partial
   initialization, failed allocation, timeout, retry, and cancellation paths that
   the change affects. Does failure preserve invariants, clean up resources,
   and reach the caller with enough context? Check that retries cannot silently
   duplicate effects and that partial work can be safely resumed or cleaned up.

10. **Compatibility and change scope.** Inspect source/API/ABI/protocol behavior,
    caller expectations, supported architectures, dependency versions, and
    configuration defaults. Check all supported variants that the change reaches.
    Identify coordinated updates required by their contracts and verify whether
    they are supplied. Check whether unrelated changes obscure the stated goal
    or increase the cost of reverting it.

11. **Test quality and testability.** Do tests observe the promised behavior and
    detect a plausible broken implementation? Check meaningful assertions,
    boundary/failure cases, regression triggers, affected platform coverage,
    isolation, and reproducibility. For concurrent behavior, require explicit
    state/event ordering rather than timing guesses. Avoid tests that only mirror
    implementation details, and distinguish an untested relevant path from a
    blanket demand to add tests to every edit.

12. **Documentation and operability.** Are the public contracts, examples,
    comments, and user-visible errors consistent with the new behavior? Can an
    operator distinguish relevant failure modes using existing diagnostics?
    Check that telemetry avoids sensitive data and hot-path overhead. Add
    diagnostics only when they close a concrete gap; follow repository comment
    rules rather than importing a conflicting external style convention.

### Additional checks for affected paths

- **Security and privacy.** For external inputs, privileges, sensitive data,
  or new execution paths, trace trust boundaries, validation and authorization
  decisions, and safe defaults. Check secrets and sensitive data throughout
  storage, transport, and diagnostics. Tie findings to the changed exposure.

- **Build, dependencies, and delivery.** For changes to builds, dependencies,
  generators, or packaging, trace the source and versions of inputs to the
  artifact users install. Check pins, generated outputs, and supported build
  and install variants. Look for ambient files or caches that could hide a
  missing dependency. Verify provenance where the delivery contract uses it.

- **Rollout, data integrity, and recovery.** For persisted state, migrations,
  or deployment changes, trace the supported upgrade sequence and an
  interruption at each durable step. Check for data loss, duplicate writes,
  and unsafe mixed states. Identify how operators detect a failed rollout and
  perform a safe rollback or recovery. Apply the repository's supported
  compatibility model at each boundary.

- **Numerical correctness and reproducibility.** For changed arithmetic or
  kernels, check ranges, overflow, precision and accumulators, exceptional
  values, and any determinism guarantees. Compare against a reference with
  justified tolerances across the affected shapes, dtypes, and backends.

- **Accessibility and internationalization.** For affected user interfaces,
  check keyboard interaction, focus, accessible names, and status or error
  feedback. For locale-sensitive changes, check formatting, text handling,
  and relevant time zones. Use the product's supported languages and access
  paths to select cases.

## Performance Evidence

- Name the baseline/head revisions, workload, scale, platform, configuration,
  metric, and any known acceptance budget. If the target is absent, report the
  observed trade-off without claiming that performance is "good enough."
- Separate **measured results**, **analytical estimates**, and **unverified
  hypotheses**. Complexity or operation counts can motivate a check; they do
  not establish actual latency, throughput, or the size of an improvement.
- For material performance claims or plausible hot-path regressions, use an
  appropriate focused benchmark/profile and compare like-for-like environments.
  Account for warmup, repetitions, variability, and tail behavior when relevant.
  Trace whether a local improvement moves work or cost to another phase.
- Use existing benchmark/profiling skills and built-in DFX tools when relevant;
  obey the platform precheck and device-lock rules for hardware runs. If evidence
  cannot be collected, identify the exact gap, proposed check, and effect on the
  recommendation. Do not run hardware benchmarks for an unrelated docs edit.

## Applying Simpler's Design Rules

Load the current `.claude/rules/` and the affected subsystem's design contract.
In particular, check:

- **Layering:** host orchestration, chip runtime, and core task responsibilities,
  plus the platform/runtime boundary; compare affected architecture siblings.
- **Representations:** modern C++ and RAII for host-owned state; contiguous POD
  for host/device wire data, internal offsets/indices rather than host pointers,
  and the required layout guards. Allocate host containers to the actual need.
- **Domain names:** the canonical `Worker` / `Chip` / `Core` vocabulary and
  matching public Python/C++ semantics, following the current codestyle rules.
- **Critical paths:** cache visibility, barriers, alignment, polling/MMIO,
  scheduling overhead, dispatch-path waits, and AICPU logging where affected.
  Consult established hardware constraints and `docs/investigations/` before
  suggesting an alternative that may already have been ruled out.

These are applications of existing repository contracts. Cite the actual rule
or subsystem contract when reporting a violation; this guide does not create a
new permission to change unrelated code or relax a hardware constraint.

## Reference Rationale

The user's review requirements and local contracts define this checklist.
These primary sources inform the complementary checks:

- [Google: What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
  emphasizes overall design, surrounding context, complexity, names, and useful
  tests alongside functionality.
- [Google: The standard of code review](https://google.github.io/eng-practices/review/reviewer/standard.html)
  grounds decisions in technical evidence and code health while distinguishing
  engineering concerns from personal preferences.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
  discusses explicit interfaces (I.1), representation invariants (C.2), resource
  ownership (R.1), and measured performance claims (Per.6). Apply these within
  this repository's host/device constraints, not as blanket language mandates.
- [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) informs security review.
  [SLSA Build Provenance](https://slsa.dev/spec/v1.2/build-provenance) describes
  how to trace an artifact to build inputs and dependencies.
- Google SRE's [Canarying Releases](https://sre.google/workbook/canarying-releases/),
  [Data Integrity](https://sre.google/sre-book/data-integrity/), and
  [Handling Overload](https://sre.google/sre-book/handling-overload/) inform
  rollout, recovery, and capacity checks.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  covers precision and backend-dependent floating-point behavior.
  [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/) gives testable web accessibility
  criteria for affected web interfaces.
