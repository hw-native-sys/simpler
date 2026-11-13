# Test-first development

Use this when a change adds a capability or fixes a defect that has a testable
behavior seam. The main [testing](../SKILL.md) skill still decides which suite,
platform, and command to run; this reference decides what test to write first
and how to keep it useful after a refactor.

[discipline.md](../../../rules/discipline.md#3-bug-fixes-start-with-a-failing-repro)
already requires a failing repro for a fixable defect. This is the same
discipline for capability work and for defects whose cause is not yet known.

## What a good test asserts

Test behavior through a public interface, not implementation structure. A test
should read like a specification and keep passing when the implementation is
refactored without changing behavior.

- Assert the observable result at the seam, not a private helper's call count.
- Take expected values from an independent source: a worked example, a known
  golden tensor, the spec, or a hand-derived literal. Do not recompute the
  expectation with the same algorithm the code uses; that test passes by
  construction.
- One behavior per test. A failure should name the behavior that broke, not
  require reading the implementation to interpret it.

## Choose the seam first

A seam is the public boundary where behavior is observable. Write down the
seams under test before writing any test; do not test internals.

Choose the narrowest seam that exercises the real change:

| Change | Preferred seam |
| ------ | -------------- |
| Python compilation pipeline or bindings | `tests/ut/` pytest at the public `simpler_setup` or `simpler` API |
| Pure C++ module | `tests/ut/cpp/` GoogleTest at the module's public header |
| Runtime, platform, or kernel behavior | one `@scene_test` case under `examples/` or `tests/st/` |
| Cross-layer behavior | a scene test; add a unit test only for a separable sub-behavior |

Prefer simulation for the red-green loop. `a2a3sim` and `a5sim` run without a
device and give a fast, deterministic signal. When the behavior under test
reaches real silicon, confirm onboard once the loop is green; the main
[testing](../SKILL.md) skill decides whether that applies. Do not make a
hardware-only seam the place where you discover the design.

## Loop

1. Write one failing test at the chosen seam. Run it; confirm it fails for the
   expected reason, not an import error or a missing fixture.
2. Make the smallest production change that makes it pass. Do not add
   speculative behavior for later tests.
3. Run the test again; confirm green.
4. Refactor only on green, with the test as the safety net. Never refactor
   while the test is red: you would be changing structure and behavior at once.
5. Repeat with the next vertical slice.

Work in vertical slices: one test, one implementation, repeat. Do not write all
tests first and then all implementation; that tests imagined behavior and
commits you to a structure before you understand the code.

## Anti-patterns

- **Implementation-coupled** — mocking an internal collaborator, testing a
  private method, or asserting call order. The tell: the test breaks on a
  refactor that did not change behavior.
- **Tautological** — the assertion recomputes the expected value the way the
  implementation does. Use an independent expectation.
- **Horizontal slicing** — writing the whole test suite before the
  implementation. It tests the shape of the code, not the behavior.
- **Testing the mock** — a test whose only assertion is that a stub was called.
  It proves the wiring, not the capability.
- **Over-broad scene test** — a new scene test for behavior already covered at
  a narrower seam. Extend the existing seam or add a case, not a parallel test.

## Mocking boundaries

Mock at system boundaries only: the CANN/driver boundary, time, randomness,
filesystem side effects, and external services. Do not mock the project's own
runtime modules, `ChipWorker`, or the platform backend to make a unit test
convenient; use the sim platform or a scene test instead. If a unit test needs
an internal mock to observe behavior, the seam is wrong; move the test outward.

At a real boundary, prefer dependency injection or a small stub with one
behavior over one generic mock that branches on its arguments.

## After the loop

Run the scope-appropriate commands from [testing](../SKILL.md), including the
runtime rebuild decision when C++ changed. A green unit test does not replace
the scene test for behavior that crosses the runtime or platform boundary.
