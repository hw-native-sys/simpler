# Virtual Environment Isolation

## Why

Multiple AI agents may work on this repo concurrently (parallel sessions, worktrees). Installing directly with `pip install .` into the user's global or base environment causes package conflicts and non-reproducible state.

## Rule

**Always create and use a project-local virtual environment before running `pip install .` or any `pip` command.**

### Steps

1. **Create venv** (once per working directory — skip if `.venv/` already exists and is usable):

   ```bash
   python3 -m venv --system-site-packages .venv
   ```

   `--system-site-packages` ensures system-level packages (e.g. driver bindings) remain accessible.

2. **Activate** before any pip or Python command:

   ```bash
   source .venv/bin/activate
   ```

3. **Install the project**:

   ```bash
   # Production / CI install
   pip install --no-build-isolation .

   # Editable install for development (auto-rebuilds C++ on import)
   pip install --no-build-isolation -e .
   ```

   `--no-build-isolation` is required because scikit-build-core consumes the venv's already-installed `scikit-build-core`, `nanobind`, and `cmake` directly. Without the flag, pip spins up a temporary isolated build env that doesn't see them, slowing the install and risking version drift.

4. **Run tests / examples** inside the activated venv.

### Worktree Scenario

When working in a git worktree (`.claude/worktrees/` or any other worktree path), the same rule applies — create `.venv` **inside the worktree directory**, not in the original repo. Each worktree gets its own independent venv.

### Quick Reference

```bash
# First time in a directory (or worktree)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation .            # or: -e . for editable

# Subsequent runs — just activate
source .venv/bin/activate
```

## Do NOT

- Run `pip install .` without an activated local venv
- Share a single venv across multiple worktrees
- Use `--user` installs as a substitute for venv isolation
- Drop `--no-build-isolation` — scikit-build-core needs the venv's `nanobind`/`cmake` directly
