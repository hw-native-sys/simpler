# Remote CUDA Evaluation Rules

- Prefer the paired CUDA evaluation scripts in
  `.agents/skills/cuda-backend-eval/scripts/`.
- First try remote Git refresh when the remote checkout has working Git access.
- If remote Git fetch, pull, or checkout fails, use tree sync instead of
  blocking evaluation:

```bash
rsync -a --delete \
  --exclude=.venv --exclude=build --exclude=tmp \
  --exclude=__pycache__ --exclude=.pytest_cache \
  ./ bizhaoh200:<remote-pto-cu>/
```

- After tree sync, run the same benchmark or smoke command from the remote
  checkout with `CUDA_HOME`, CUDA `PATH`, and `PYTHONPATH` set explicitly.
- Record whether the artifact came from Git refresh or tree sync in the
  changelog or history entry for that capture.
- Never claim paired A100/H200 validation from a local-only run.
