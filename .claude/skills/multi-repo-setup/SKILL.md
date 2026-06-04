---
name: multi-repo-setup
description: Set up a cross-repo investigation when a workload from another repo (pypto, pypto-lib, etc.) needs to be run, especially when you want to swap in simpler-main HEAD or the current worktree's simpler instead of the version that repo pins. Clones-or-updates each external repo every invocation so stale local clones don't lie about CI parity. MUST invoke before chasing "X doesn't work on simpler" reports where X lives outside this repo.
---

# Multi-Repo Investigation Setup

The skill lives in the simpler repo, so `$PWD` when you invoke it is
already a simpler worktree — nothing to clone for simpler itself.

Every other repo gets cloned-or-updated to a canonical local path each
time. The default is to follow each repo's own pinning. Simpler dev
often diverges from that to swap in either simpler-main HEAD or the
current worktree.

## When to invoke

Invoke before:

- running an external-repo workload (`python <ext>/some_test.py`,
  pytest in another repo, etc.) for the first time in a session, or
- reporting "X doesn't work on simpler" where X lives in another repo.

Skip for work that stays inside `simpler/` (its own pytest, examples,
unit tests).

## Step 1: The repo graph (URLs)

For an Ascend workload investigation, the cast is usually some subset
of these. `simpler` is `$PWD` — the other rows list a GitHub URL and a
canonical local clone path Step 2 ensures exists and is up-to-date.

| repo | role | GitHub URL | local clone |
| ---- | ---- | ---------- | ----------- |
| simpler | host runtime + DFX (this repo) | <https://github.com/hw-native-sys/simpler> | `$PWD` (the current worktree) |
| pypto | compiler + Python frontend; vendors a simpler submodule at `runtime/` | <https://github.com/hw-native-sys/pypto> | `$HOME/workspace/pypto` |
| pypto-lib | model workloads (qwen3, deepseek, etc.) — imports pypto + simpler | <https://github.com/hw-native-sys/pypto-lib> | `$HOME/workspace/pypto-lib` |
| pto-isa | ISA spec; pinned by pypto-lib via env (`PTO_ISA_COMMIT`) | <https://github.com/hw-native-sys/pto-isa> | `$HOME/workspace/pto-isa` |
| PTOAS | release binary (`ptoas`); located via `PTOAS_ROOT` env | <https://github.com/hw-native-sys/PTOAS> (releases) | `/tmp/ptoas-v0XX/` (extracted tarball) |

Drop rows your investigation doesn't need. Add rows for ad-hoc repos in
the same format.

## Step 2: Clone-or-update every external repo

Run this each invocation — never trust the existing clone to be
current. Same shell function handles both "doesn't exist yet" and
"already there, just sync to origin/main":

```bash
ensure_repo() {
  local url=$1 dir=$2
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch origin --quiet
    git -C "$dir" reset --hard origin/main
  else
    git clone "$url" "$dir"
  fi
  git -C "$dir" submodule update --init --recursive --depth 1
}

ensure_repo https://github.com/hw-native-sys/pypto      "$HOME/workspace/pypto"
ensure_repo https://github.com/hw-native-sys/pypto-lib  "$HOME/workspace/pypto-lib"
# add more as needed
```

Notes:

- `reset --hard origin/main` overwrites any local changes in the clone.
  If you have edits there, stash / commit them before invoking this skill.
- `submodule update --init --recursive` matches the pin recorded in
  each repo's commit — that's the version the repo's CI runs against.
- For PTOAS, download the release tarball mentioned in pypto-lib's
  `.github/workflows/daily_ci.yml` (the `PTOAS_VERSION` env), don't
  clone — it ships as a binary release.

## Step 3: Override — pick which simpler the workload sees

The default is what pypto's submodule pins. Simpler dev usually wants
ONE of these instead:

### Override A: simpler `origin/main` HEAD

You want the latest merged simpler against the external workload (e.g.
confirm a recently merged PR didn't break a downstream case).

```bash
ensure_repo https://github.com/hw-native-sys/simpler "$HOME/workspace/simpler-main"
source <your-venv>/bin/activate
pip install --no-build-isolation "$HOME/workspace/simpler-main"
```

### Override B: this worktree's simpler (your PR branch)

You want to test in-flight changes (current PR / dev branch) against the
external workload.

```bash
source .venv/bin/activate
pip install --no-build-isolation .
```

### Either way: verify which simpler actually loaded

A previous session may have left a user-site editable hook
(`_simpler_editable.pth`) that shadows your venv install. Always
verify:

```bash
python -c "import simpler, simpler_setup; \
  print('simpler      :', simpler.__file__); \
  print('simpler_setup:', simpler_setup.__file__)"
```

Both paths must point at the simpler you intended. If either points at
`~/.local/...` or another worktree, clean up and reinstall:

```bash
rm -f ~/.local/lib/python*/site-packages/_simpler_editable.{pth,py}
pip uninstall -y simpler
pip install --no-build-isolation <the-simpler-you-want>
```

### Note on `-e`

Prefer plain `pip install --no-build-isolation .` over `-e .` unless
you actively edit the package and need re-import to pick up changes
without reinstall. Editable installs leak the `_simpler_editable.pth`
hook into user-site, which survives sessions and shadows the next
venv install. Non-editable installs don't.

## Step 4: Install pypto / pypto-lib

For pypto, plain install too — let pypto's vendored `runtime/` stay
unbuilt because you've already provided simpler via Step 3:

```bash
pip install --no-build-isolation "$HOME/workspace/pypto"
# Do NOT `pip install "$HOME/workspace/pypto/runtime"` — that would
# overwrite the simpler you installed in Step 3 with pypto's older pin.
```

pypto-lib is import-only (no install). Set `PYTHONPATH` and run
scripts out of its tree directly:

```bash
export PYTHONPATH="$HOME/workspace/pypto-lib"
```

Re-verify the loaded simpler after every install — pypto's install can
re-resolve dependencies in ways that change what wins.

## Step 5: If the workload fails — start with "is it CI-gated?"

The most informative first check is whether the workload is gated by
that repo's CI workflow.

| case | meaning | how to triage |
| ---- | ------- | ------------- |
| **CI-gated** | the repo's CI runs this exact script today; it was passing as of the last green CI run | A failure now usually points at a **recent code change** — your in-flight simpler changes, or a commit landed in pypto / pypto-lib since the last CI run. Bisect against `origin/main` of each repo. |
| **not CI-gated** | the script is in the repo but no workflow invokes it | Read the docstring first. Files like this are often "intent" / "EXPECTED / INTENT program" / experimental drafts — they may be documented as expected-to-fail. Treat as workload bug, not simpler bug, unless proven otherwise. |

Quick check:

```bash
F=<workload>.py
grep -nE "python .*$(basename $F)" "$HOME/workspace/"*/.github/workflows/*.yml
# Match → CI-gated. No match → not CI-gated.
```

Also worth checking: pypto's CI sometimes pulls pypto-lib and runs a
specific subset (see pypto's `.github/workflows/ci.yml` step "Run
pypto-lib ... example"). That subset is the actual cross-repo gate.

### Common surface errors → first suspect

| symptom | likely layer | first check |
| ------- | ------------ | ----------- |
| `ModuleNotFoundError: No module named 'pypto'` | pypto not installed in this venv | reinstall pypto from your local clone |
| `import simpler` resolves to wrong path | user-site `.pth` hook shadowing venv | remove `_simpler_editable.{pth,py}` from user-site |
| `FileNotFoundError: kernel_config.py not found in ...` | `PTOAS_ROOT` unset → pypto auto-skips ptoas → no `kernel_config.py` emitted | `export PTOAS_ROOT=/tmp/ptoas-v040` (or whichever release) |
| script exits 0 but no device run | compile-only smoke fallback (golden data dir missing) | pass `--data-dir <golden>` or `--smoke` explicitly |
| `aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018` | several — device log is the only ground truth | read `~/ascend/log/debug/device-N/device-<pid>_*.log` |
| `BFloat16 did not match Float` at validate | golden data shape mismatch (data older than code) | regenerate golden via the workload's `gen_*_golden.py` |

When the surface is `507018` / `507899` / `507046`, **do not stop at
the host log**. The host only reports CANN's verdict; the actual AICPU
state lives in `~/ascend/log/debug/device-N/`:

```bash
LOG=~/ascend/log/debug/device-$DEVICE_ID/device-$PID_*.log
grep -oE "task_id=[0-9]+ state=RUNNING" "$LOG" | sort -u
grep "state=RUNNING" "$LOG" | head -1 | grep -oE 'kernels=\[[^]]+\]'
grep "completed=" "$LOG" | head -1
```

If the same task hangs across every retry on every chip, it's the
workload (or your code change), not chip contention.

## Anti-patterns

- ❌ **Trusting an existing local clone without `git fetch`**. Your
  clone is whatever you last fetched, possibly weeks behind. Step 2
  exists precisely to make this not a thing.
- ❌ **Using `-e` "just to be safe"**. Editable installs leak a
  user-site finder hook that survives sessions and shadows the next
  venv install. Plain install is the default; reach for `-e` only
  when you'll actively edit.
- ❌ **Blaming chip contention before reading the device log**. The
  device log either shows the contention signature (sibling-die cores
  with `cond_reg_state=ack` from another owner) or it doesn't.
- ❌ **Treating any failing workload as "simpler broke it"**. Step 5's
  CI-gate check separates "your simpler change broke a CI case" (real)
  from "this file was always expected to fail" (not your problem).
- ❌ **Skipping `.claude/rules/task-submit-isolation.md`** on onboard
  hardware runs. Multi-repo flows don't waive the per-die lock.
