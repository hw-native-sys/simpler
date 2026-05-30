---
name: testing
description: Use when validating code changes, repository tooling changes, or verification claims
---

# PTO CUDA Testing Workflow

Use this workflow when validating code changes:

1. Run the smallest relevant `pytest` target first.
2. Expand to the local regression suite for adjacent modules.
3. If repository tooling changed, run `pre-commit run --all-files`.
4. Record the exact command and result before claiming success.
