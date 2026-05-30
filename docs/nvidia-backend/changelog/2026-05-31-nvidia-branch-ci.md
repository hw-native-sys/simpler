# 2026-05-31 NVIDIA Branch CI

## What Changed

- Replaced the branch CI workflow with a focused NVIDIA review job.
- Removed Ascend sim and hardware jobs from this branch workflow so the NVIDIA
  backend PR does not schedule a2a3/a5 CI.
- Added a review-artifact test that guards this branch-specific CI shape.

## Current CI Scope

The branch CI now runs:

- `.agents/checks/check_nvidia_review_ready.py`
- `tests/ut/py/test_nvidia_review_artifacts.py`
- Python syntax checks for the review guard and CUDA example wrappers

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```
