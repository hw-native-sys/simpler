---
name: git-commit
description: Complete git commit workflow including pre-commit review, staging, message generation, and verification. Use when creating commits or preparing changes for commit.
---

# Git Commit Workflow

## Prerequisites

Check what changed to determine review/test needs:

```bash
git diff --name-only
git diff --cached --name-only
```

| File Types Changed | Run Tests |
| ------------------ | --------- |
| C++ (`.cpp`, `.h`) | Yes |
| Python (`.py`) | Yes |
| Build system (`CMakeLists.txt`) | Yes |
| Docs only (`.md`) | Skip |
| Config only (`.gitignore`, `.json`) | Skip |

## Pre-Commit Testing

When tests are needed, follow the testing skill for the full strategy: see [`.claude/skills/testing/SKILL.md`](../testing/SKILL.md) — "Pre-Commit Testing Strategy" section.

## Stage Changes

Stage related changes together, never stage build artifacts (`build/`, `*.o`, `outputs/`).

```bash
git add path/to/file1.cpp path/to/file2.h
git diff --staged  # Review before committing
```

## Commit Message Format

### Subject Line

`Type: concise description` (under 72 characters, imperative mood, no period)

**Types**:

| Type | Usage |
| ---- | ----- |
| **Add** | new feature or file |
| **Fix** | bug fix |
| **Update** | enhancement to existing feature |
| **Refactor** | restructuring without behavior change |
| **Support** | tooling, profiling, CI infrastructure |
| **Sim** | simulation-specific changes |
| **CI** | CI/CD pipeline changes |

### Body (required for multi-file changes)

Separate from subject by a blank line. Explain **what** changed and **why**. Use bullet points for multiple items. Wrap at 72 characters.

**When to include a body**:
- Changes touch 3+ files
- The "why" is not obvious from the subject alone
- There are trade-offs, side effects, or migration notes

**Good examples**:

```text
Add: paged attention kernel with batch support

- Implement batch-level KV-cache slot allocation in orchestration
- Add softmax_prepare kernel for cross-batch score normalization
- Golden test covers batch_size=4 with variable sequence lengths
```

```text
Fix: resolve cross-pipeline race in softmax_prepare

PIPE_V and PIPE_S barriers were missing between score accumulation
and softmax normalization, causing sporadic mismatches on hardware
when AICore pipelines overlapped.
```

```text
Support: restructure .ai-instructions into .claude/skills and rules

- Move coding conventions to .claude/rules/ (auto-loaded)
- Move git-commit and testing workflows to .claude/skills/ (on-demand)
- Rename test-sim/test-device commands to test-example-* for clarity
- Add test-runtime-* commands for runtime-scoped testing
- Add pre-commit testing strategy with npu-smi and nproc detection
```

**Simple changes (body optional)**:

```text
Fix: typo in vector_example golden.py
```

**Bad examples**:

```text
x  Added new feature.              # Past tense, has period
x  fix bug                         # Lowercase type
x  WIP                             # Not descriptive
x  Chore: update gitignore         # Invalid type
x  Support: update stuff           # Vague, no body for multi-file change
```

## Co-Author Policy

**Never** add AI co-author lines (e.g., `Co-Authored-By:`). Commits reflect human authorship only.

## Post-Commit Verification

```bash
git log -1              # Check message
git show HEAD --stat    # Verify staged files
```

**Fix unpushed commits only**:

```bash
git commit --amend -m "Corrected message"
```

## Checklist

- [ ] Only relevant files staged (no build artifacts)
- [ ] Tests passed (if code changed) or skipped (if docs/config only)
- [ ] Message format: `Type: description` (under 72 chars, imperative, no period)
- [ ] Body included for multi-file changes (what + why)
- [ ] No AI co-authors
