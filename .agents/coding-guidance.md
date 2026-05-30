# PTO CUDA Agent Coding Guidance

This file is the primary `.agents/` engineering manual for this repository.
`CLAUDE.md` and `.claude/rules/` remain the source of truth for the upstream
runtime architecture; this file adds review, dispatch, and evidence rules for
the NVIDIA backend work.

## Repository Direction

This repository currently explores a CUDA/NVIDIA backend for PTO runtime as a
personal-interest research branch. Treat the CUDA backend as an early backend
prototype until maintainers explicitly promote it.

The active CUDA surfaces are:

- `src/cuda/platform/`: CUDA platform ABI and host runtime integration.
- `src/cuda/runtime/host_schedule/`: host-scheduled CUDA launch path.
- `src/cuda/runtime/persistent_device/`: persistent device scheduler path.
- `.agents/skills/cuda-backend-eval/`: CUDA smoke, benchmark, and validation
  workflows.
- `docs/nvidia-backend/`: architecture, evaluation, history, and review
  evidence.
- `examples/cuda/`: runnable examples aligned with evaluated smoke paths.

## Hard Gates

- Keep `CLAUDE.md`, `.claude/rules/`, and `.agents/rules/` aligned.
- Do not describe CUDA behavior as implemented unless code evidence exists.
- Keep evaluation claims tied to raw artifacts under `tmp/` and validator
  commands.
- Do not commit raw benchmark dumps from `tmp/cuda-backend/`.
- Keep current evaluation pages short; archive history under
  `docs/nvidia-backend/history/`.
- Use `--sync-remote-tree` or explicit `rsync` fallback when remote Git fails.
- Run `.agents/checks/check_nvidia_review_ready.py` before claiming review
  readiness.

## Architecture Expectations

- Preserve existing PTO terminology: platform variant, runtime, `ChipWorker`,
  host runtime, AICPU scheduler, AICore worker, `TaskArgs`, and `CallConfig`.
- Be explicit about the CUDA/AICPU mismatch: CUDA persistent-device scheduling
  must live in the compiled CUDA device binary.
- Keep host scheduling and persistent-device scheduling separate in docs,
  examples, and benchmark method names.
- Prefer small, reviewable slices that update code, docs, examples, benchmark
  data, and changelog together.

## Ultimate Goal Mode

Use `.agents/rules/ultimate-goal-dispatch.md` and
`.agents/templates/ultimate-goal.md` when a goal is too large for one PR or
one Codex session. The dispatcher must keep an umbrella note, dispatch log,
worker prompts, and verification evidence under `docs/in_progress/`.

Workers own one child slice and must not dispatch nested workers. If a worker
finds more parallel work, it records the proposed child slice for the
dispatcher.

## Verification

For review-readiness changes, run:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

For CUDA runtime changes, use the relevant commands from
`.agents/skills/cuda-backend-eval/SKILL.md`.
