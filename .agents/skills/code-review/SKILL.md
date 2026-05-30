---
name: code-review
description: Use when reviewing repository changes for architecture, verification, capability, or documentation drift
---

# PTO CUDA Code Review Workflow

Review changes for:

1. Runtime/platform boundary clarity across `src/cuda/`, `simpler_setup/`,
   Python bindings, examples, tests, and docs.
2. Verification completeness and failure-path correctness.
3. Benchmark provenance, artifact, and viewer-data consistency.
4. Documentation alignment with the implemented contract.
