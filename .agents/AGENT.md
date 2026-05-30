# AI Assistant Rules For PTO CUDA

This repository keeps Codex-facing project guidance under `.agents/`.

## Structure

- `.agents/coding-guidance.md` is the primary Codex engineering manual for
  this branch.
- `.agents/rules/` stores stable project conventions.
- `.agents/skills/` stores executable workflow guides.
- `.agents/agents/` stores specialized review/verification roles.
- `.agents/lib/` stores shared GitHub workflow procedures.
- `.agents/templates/` stores reusable goal and dispatch templates.
- `.agents/checks/` stores lightweight review-readiness checks.

## Priority

1. Follow `CLAUDE.md` and `.claude/rules/` first.
2. Follow `.agents/coding-guidance.md`.
3. Follow `.agents/rules/` for CUDA backend review, evaluation, dispatch, and
   evidence discipline.
4. Use `.agents/skills/cuda-backend-eval/` for CUDA smoke and benchmark work.
5. Use `.agents/templates/ultimate-goal.md` with
   `.agents/rules/ultimate-goal-dispatch.md` when work spans multiple PRs or
   Codex sessions.
6. Run `.agents/checks/check_nvidia_review_ready.py` before claiming that the
   NVIDIA backend branch is ready for human review.

## Repository Purpose

This branch explores a CUDA/NVIDIA backend for PTO runtime as a personal
interest project. Treat it as early backend research until the maintainers
explicitly decide otherwise.

## Current Review Risks

- CUDA lacks an Ascend-style AICPU, so persistent-device scheduling must be
  part of the compiled CUDA device binary.
- Host-schedule and persistent-device rows exercise different launch models;
  docs must not blur host CUDA Graph replay with device-side scheduling.
- Benchmark claims must point to code evidence, validation commands, and raw
  artifacts under `tmp/`.
