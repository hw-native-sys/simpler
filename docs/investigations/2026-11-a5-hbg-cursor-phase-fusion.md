# A5 HBG cursor phase fallthrough

## Question

Can a resolver cursor avoid repeated outer-loop dispatch by falling through
START, SELECT, APPEND and PUBLISH while its work and cancellation-check budgets
permit?

## Experiment

The experiment retained the existing work-unit count and cursor state on every
yield. A phase limit was the minimum of the remaining work boundary and next
cancellation observation. It was applied after the cursor already cached its
CSR row per invocation and held scalar positions locally until return.

## Result

On A5 with the same DSV4.1 PyPTO-lib SWA case and device, full L3 resolver median
increased from 2.050 to 2.251 microseconds. L0 Device median over 100 measured
iterations was effectively flat: 873.103 versus 873.418 microseconds. Full
completion medians were 14.785 and 16.892 microseconds; those spans also contain
successor admission and variable atomic waiting, so they do not isolate phase
dispatch cost.

The profiled AIV resolver grew from 5,496 to 5,839 four-byte instructions. Full
AICore cold builds took 125.113 and 137.782 seconds. Related unit tests, SWA and
the two-device asynchronous COUNTER case passed; actual SP/LOCAL checks passed.
These were sequential measurements, not interleaved samples.

## Decision

The phase-fallthrough change was reverted. It did not demonstrate a performance
benefit and increased generated code. Caching immutable CSR metadata and local
cursor positions is a separate change and remains in the measured control.

Revisit only with a materially different generated control-flow shape and new
hardware evidence. Do not assume fewer source-level switch entries imply less
work in the CCEC output.

Local evidence is recorded under `outputs/swa-next-20261112/` in
`completion-cursorflow-comparison.json`, `cursorflow-only.patch` and the
corresponding provenance and validation logs. The source base was `72fba2e75`.
