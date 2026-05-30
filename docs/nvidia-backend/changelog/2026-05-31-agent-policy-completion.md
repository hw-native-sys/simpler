# 2026-05-31 Agent Policy Completion

## What Changed

- Added the missing Spine-style `.agents/` structure for this repository:
  specialized agents, GitHub workflow skills, shared GitHub procedures, and
  the ultimate-goal template.
- Added PTO/CUDA-specific coding guidance and dispatch rules instead of
  copying Spine-specific architecture language.
- Extended the review-readiness guard so future changes must keep ultimate
  goal mode, GitHub PR workflows, and specialized review agents present.

## Review Impact

The NVIDIA backend branch now has the same operating pattern as the referenced
`.agents/` layout: PR-sized mode for ordinary work, ultimate-goal dispatcher
mode for multi-session work, and worker mode for child slices.

## Verification

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```
