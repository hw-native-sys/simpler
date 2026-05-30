# Quality And Evidence Rules

- New abstractions must have names that explain their boundary without reading
  unrelated files.
- Prefer small review units: one runtime path, one benchmark setup, or one
  documentation slice per change.
- Add focused tests or guard checks for review artifacts that are intended to
  stay synchronized over time.
- Run the cheapest relevant checks before finishing:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

- If a full CUDA benchmark cannot be rerun, say so and cite the latest raw
  artifact path being used as evidence.
