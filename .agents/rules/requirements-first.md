# Requirements-First Rules

- Start substantial CUDA backend work from an explicit requirement, design
  note, or tracked `docs/in_progress/` umbrella goal.
- Treat requirements as a simplification tool, not a feature inventory.
- Prefer the smallest design that satisfies the requested runtime behavior and
  can be proven with focused tests or benchmark artifacts.
- Do not keep duplicate runtime concepts, compatibility aliases, or knobs that
  do not change real behavior.
- If a value can be derived from a runtime descriptor, manifest, or benchmark
  artifact, derive it instead of storing another source of truth.
- Before adding a new abstraction, check whether better naming or deleting an
  old concept solves the review problem more directly.
