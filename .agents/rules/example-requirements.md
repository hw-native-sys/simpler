# Example Requirements Rules

- Examples are part of the public review surface.
- CUDA examples should stay runnable from the repository root with
  `PYTHONPATH=$PWD:$PWD/python`.
- CUDA examples should use the same evaluated smoke paths or public runtime
  surfaces that the benchmark docs describe.
- Do not create a second example framework for CUDA unless the user asks for
  it explicitly.
- Keep example README commands synchronized with smoke scripts and evaluation
  viewer commands.
