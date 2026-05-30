# Core Development Rules

- Follow `CLAUDE.md` and `.claude/rules/` for upstream PTO runtime
  architecture and terminology.
- Keep CUDA backend work scoped to the runtime, platform, docs, tests,
  examples, and `.agents/` surfaces needed by the task.
- Do not bypass the normal `ChipWorker`, `TaskArgs`, `CallConfig`, runtime
  builder, or platform discovery contracts without documenting the reason.
- Keep host-schedule and persistent-device abstractions easy to read from file
  names, class names, function names, and benchmark method names.
- Prefer explicit error surfaces over silent fallback behavior.
- Update tests and docs in the same slice when contracts change.
- Keep `.agents/` workflow docs, changelogs, and evaluation viewer data in
  sync with implementation.
