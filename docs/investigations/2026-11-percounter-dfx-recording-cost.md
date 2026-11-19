# Percounter DFX recording cost

The percounter L1–L4 implementation keeps individual GM and SSBUF operations
in L3. Its acceptance metric is the complete AICore wall, including final
record publication. The first-to-last kernel envelope is a separate metric.

## Measured alternatives

These measurements used the DSV4.1 PyPTO SWA frozen inputs, A5 device 0,
two warmups and ten measured iterations on base `3bcaf8d4f`. Every measured
arm passed the fused-prefix numerical check. The L0 reference was approximately
535 microseconds. Times below include instrumentation and retirement.

| Variant | AICore wall (microseconds) | Decision |
| ------- | -------------------------: | -------- |
| Original DFX implementation | 674.42 | Replace per-kernel publication |
| Deferred task timing, inline primitive wrappers | 660.50 | Retain completion-before-record ordering |
| Whole-cache retirement clean, no unused AICPU stream polling | 651.70 | Retain |
| Fully inline generic memory encoder | 660.85 | Reject: no improvement |
| Private trace writes using `st_dev` | 690.85 | Reject: slower |
| Remove redundant logical-record counter | 655.73 | Derive logical count on Host; insufficient alone |
| Specialized 16-byte primitive encoder | 653.87 | Insufficient |
| Omit result-dependent timestamp wait (diagnostic only) | 651.42 | Reject: no useful gain and changes timing semantics |
| Outline uncommon 16-byte encoding branches | 658.73 | Insufficient |
| Lossless 8-byte relative timestamp encoding | 623.01 | Retain |

The compact representation retains every clock tick and operation. A row that
cannot fit its offset, duration or depth uses full-width fields. It does not
sample operations or move them to L4. The paired DSL SWA measurement was
1129.59 microseconds, against a 1047.65 microsecond L0 reference.

L1 publishes cached task records at buffer handoff and final retirement. A
30-iteration reverse-order comparison measured 535.04 microseconds at L0 and
541.96 at L1, or +1.29%. Its 49,155-kernel buffer-rotation test passed.

## Remaining limit

The SWA L3 overhead is approximately +16.4% for PyPTO and +7.8% for DSL on
that base. The requested preferred +5% target is not achieved, and PyPTO also
exceeds the +10% ceiling. These results justify shipping a complete trace
implementation with an explicit performance limitation, not claiming the
performance target passed. Rebased code needs its own validation; these
measurements must not be relabeled as results from a newer base.

Changing store instructions or inlining the whole encoder is not a supported
optimization without new evidence. Future work should isolate recording and
publication costs while preserving operation timing and complete event capture.
