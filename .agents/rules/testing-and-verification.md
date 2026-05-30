# Testing And Verification Rules

- Add or update tests for every new public or review-facing contract.
- Prefer focused pytest tests close to the changed surface.
- Do not claim a slice works without fresh command output.
- Verification artifacts and benchmark metadata are part of the feature.
- When touching review artifacts, run:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

- When touching CUDA runtime behavior, also run the cheapest matching command
  from `.agents/skills/cuda-backend-eval/SKILL.md`.
