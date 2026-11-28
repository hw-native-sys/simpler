# Repository checks for PR review

Run the checks selected in the main workflow at the pinned review revisions.

## Change breakdown

Use `git diff --numstat` on the reviewed range. Count files and added,
deleted, and total changed lines by category. Report a total and identify
substantial Uncategorized changes. Classify each path by the first match:

| Category | Paths and file types |
| -------- | -------------------- |
| Test/Examples | `examples/`, test directories (including fixtures, snapshots, and test data), test files, and `conftest.py`. |
| Build | Build-system and packaging inputs, such as CMake, Make, Autotools, Meson, Bazel, package manifests, and lockfiles. |
| Docs | Documentation directories, documentation and image files, README, CHANGELOG, CONTRIBUTING, LICENSE, and NOTICE. |
| Core | Source under the repository's `src/`, `python/`, or `simpler_setup/`. |
| Uncategorized | Remaining paths, including unclassified tooling and repository configuration. |

Count churn as additions plus deletions. These are review-size signals:

- **Total > 1,000 changed lines:** flag review and merge burden and suggest a
  split by concern, with the actual total.
- **Core > 1,000 changed lines:** give the actual Core total, recommend a split,
  and include a mechanism brief and practical commit or subsystem review order.
  A large Core change with no size justification weighs toward requesting changes
  or discussion.

## PTO-ISA pin

Use `pto_isa.pin` from the reviewed head. A 40-hex SHA is active. An empty value
or `head`, `latest`, or `none` selects latest. The source of truth for these
values is [pto_isa.py](../../../../simpler_setup/pto_isa.py).

Inspect added and removed `#include` lines that reference PTO-ISA headers.
Compare paths by basename. A different path for the same basename signals a
relocation. An addition with no matching removal signals a newly referenced
header. Record whether the pin file itself changed.

| Result | Report in Appendix B |
| ------ | -------------------- |
| Unpinned | Record that the pin check was skipped. |
| Active pin, no new or relocated include | Report the pinned SHA as an advisory. |
| Active pin, new or relocated include | Report the SHA and exact include changes as a recommendation. Verify that the pin supplies the headers and carry the check into Should-fix. |

For a pin change, verify the intended SHA and the onboard A2/A3
`host_runtime.so` rebuild against `SIMPLER_PTO_ISA_BUILD_COMMIT`. CI uses the
managed pin checkout for the build and passes the pin to tests with
`--pto-isa-commit`. See the [developer guide](../../../../docs/developer-guide.md).

## Feature coverage

Review changed scene cases under `examples/` and `tests/st/`, coverage
implementation changes, and affected onboard/framework workflows. Match the
actual lane's platform, environment, selection flags, and
`--coverage-partition`. The
[CI guide](../../../../docs/ci.md#feature-coverage-verification) contains the
`collection_inventory` command and coverage obligations. Use available
collection and execution evidence in a read-only review.

- Program is the default. Kernel and auxiliary declare `runtime_product`.
  Only Kernel declares `execution_mode("eager"|"capture_replay")`. Verify the
  scene-class defaults as well as explicit markers.
- Each selected case resolves to exactly one runtime identity. The
  `coverage_runtime` marker supports runtime-independent probes and carries
  no feature claim.
- Program uses a borrowed context or limited resources. Kernel is
  worker-owned. Auxiliary probes carry their own obligations. Framework
  consumers declare Kernel. Check that each mandatory group still has an
  eligible case after removals, renames, or changes to selection.
- Verify exact platform-scoped `coverage_skip` reasons against JUnit and the
  documented exceptions. Framework coverage has no skip exceptions. An
  expected failure or approved skip contributes no passing execution receipt.
- For changes to coverage machinery, verify that partial reports keep
  source-checked counts and retries count each node ID once. A failed attempt
  stays a failure. Distinguish collection evidence from execution receipts.

Missing or conflicting declarations that break collection are Must-fix.
Use [test_coverage.py](../../../../simpler_setup/test_coverage.py) and
[test_environments.py](../../../../simpler_setup/test_environments.py) for the
actual selection and receipt contracts.

## A5 device stack realignment

Check changes to device code under `src/a5/`. The
[A5 compiler defect](https://github.com/hw-native-sys/simpler/issues/2973)
concerns aligned stack objects whose addresses escape into outlined calls.
Use existing evidence matching the reviewed head and build configuration.
When building is authorized, run:

```bash
./tools/a5-ir-check/a5_ir_check.py <build-dir>
```

Record the exit status. `0` covers a clean scanned configuration with no skipped
translation units. `1` identifies candidate stack shapes that need a source
change or justification. `2` or a skipped translation unit leaves an evidence
gap. The scanner checks a precondition of the defect. Its result is specific
to the compiler, build, and inlining configuration, and escape detection has
limits such as `phi` and `select`. Use A5 hardware evidence for a claimed
device-functional fix.
