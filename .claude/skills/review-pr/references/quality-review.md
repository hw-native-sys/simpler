# Design and Code Quality Review

Use this guide after establishing the current system's pain points and reading
the proposed mechanism. Assess how the change affects the surrounding system,
including its callers and analogous implementations. Apply the dimensions to
all code-changing PR types; depth follows the change's actual risks.

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
   Evaluate latency, throughput, and resource trade-offs against the intended
   use and any agreed budget. Follow the evidence rules below.

4. **Simplicity and readability.** Can a reader follow the control flow and
   invariants without reconstructing hidden state? Inspect nesting, repeated
   branches, overly broad functions, boolean-mode combinations, dead code, and
   unnecessary abstraction. Prefer a clear expression of the current problem;
   fewer lines, clever expressions, or a generic framework are not automatic
   improvements. Explain a concrete comprehension or maintenance cost when
   calling code unnecessarily complex.

5. **Reuse and redundancy.** Use `rg` to find equivalent helpers, protocols,
   validation, conversion logic, and data models before proposing additions.
   Cite the nearest implementation and compare semantics, ownership, lifetime,
   and constraints. Look for multiple sources of truth and derived state that
   must be kept synchronized. Reuse when the abstraction fits; do not merge
   superficially similar paths whose contracts differ. For intentional copies,
   explain the constraint and how required consistency is maintained.

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

9. **Failure handling and trust boundaries.** Inspect invalid input, partial
   initialization, failed allocation, timeout, retry, and cancellation paths that
   the change affects. Does failure preserve invariants, clean up resources,
   and reach the caller with enough context? Check idempotency where retries
   can duplicate effects. For exposed trust boundaries, assess bounds, authority,
   unsafe execution, and sensitive data handling; do not turn unrelated internal
   edits into an unrequested security audit.

10. **Compatibility and change scope.** Inspect source/API/ABI/protocol behavior,
    caller expectations, supported architectures, dependency versions, and
    configuration defaults. Identify migrations or coordinated changes required
    by the diff and whether they are supplied. When rollout or rollback matters,
    explain how old/new states interact. Check whether unrelated changes obscure
    the stated goal or increase the cost of reverting it.

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
