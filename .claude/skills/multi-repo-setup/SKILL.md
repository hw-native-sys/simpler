---
name: multi-repo-setup
description: Set up a cross-repo investigation when a workload from another repo (pypto, pypto-lib, pypto-serving, etc.) needs to be run, especially when you want to swap in simpler-main HEAD or the current worktree's simpler instead of the version that repo pins. Documents the repo graph (who imports who), clones-or-updates each external repo every invocation so stale local clones don't lie about CI parity, and points at each repo's own skills/docs for workload-specific steps. MUST invoke before chasing "X doesn't work on simpler" reports where X lives outside this repo.
---

# Multi-Repo Investigation Setup

This skill answers one question: **given a workload that lives in another
repo, how do the repos relate, and how do I clone + set up each one so the
workload runs against the simpler I want?** It stops at "the repos are
installed and the workload's own entry point is reachable" — for what to
*run* and how to read its output, defer to that repo's own skills/docs
(see [Step 5](#step-5-hand-off-to-each-repos-own-skillsdocs)) or, for the
qwen3 workload specifically, to
[`multi-repo-qwen-setup`](../multi-repo-qwen-setup/SKILL.md).

The skill lives in the simpler repo, so `$PWD` when you invoke it is
already a simpler worktree — nothing to clone for simpler itself. Every
other repo gets cloned-or-updated to a canonical local path each time. The
default is to follow each repo's own pinning. Simpler dev often diverges
from that to swap in either simpler-main HEAD or the current worktree.

## When to invoke

Invoke before:

- running an external-repo workload (`python <ext>/some_test.py`,
  pytest in another repo, etc.) for the first time in a session, or
- reporting "X doesn't work on simpler" where X lives in another repo.

Skip for work that stays inside `simpler/` (its own pytest, examples,
unit tests).

## Step 1: The repo graph (who imports who)

For an Ascend workload investigation, the cast is usually some subset
of these. `simpler` is `$PWD` — the other repos are cloned under the
worktree's **`build/`** (gitignored, so they never get committed and stay
co-located with the simpler you're testing). `$BUILD` =
`$(git rev-parse --show-toplevel)/build` (Step 2 defines it).

| repo | role | imports | GitHub URL | local clone |
| ---- | ---- | ------- | ---------- | ----------- |
| simpler | host runtime + DFX (this repo) | — | <https://github.com/hw-native-sys/simpler> | `$PWD` (the current worktree) |
| pto-isa | ISA spec | — | [pto-isa][pto-isa] | `$BUILD/pto-isa` |
| pypto | compiler + Python frontend; vendors a simpler submodule at `runtime/` | simpler | <https://github.com/hw-native-sys/pypto> | `$BUILD/pypto` |
| pypto-lib | model kernels (qwen3, deepseek, etc.) | pypto + simpler | <https://github.com/hw-native-sys/pypto-lib> | `$BUILD/pypto-lib` |
| pypto-serving | serving stack + standalone model runners (qwen3 `npu_generate.py`) | pypto-lib + pypto + simpler | <https://github.com/hw-native-sys/pypto-serving> | `$BUILD/pypto-serving` |

[pto-isa]: https://github.com/hw-native-sys/pto-isa
| PTOAS | `ptoas` assembler — **provided globally**, no clone | — | (on dev box / CI) | `/usr/local/bin/ptoas-bin` via `pypto-setup` |

The import chain is a stack: **pto-isa** (build-time spec) → **simpler**
(runtime) → **pypto** (compiler, vendors simpler) → **pypto-lib** (kernels)
→ **pypto-serving** (serving + runners). A change low in the stack
(simpler) can break anything above it; that's why Step 3 lets you pin which
simpler the whole stack sees.

Drop rows your investigation doesn't need. Add rows for ad-hoc repos in
the same format.

## Step 2: Clone-or-update every external repo

Clones live under the worktree's `build/` (gitignored — co-located with
the simpler you test, never committed). Run this each invocation — never
trust the existing clone to be current. Same shell function handles both
"doesn't exist yet" and "already there, just sync to origin/main":

```bash
BUILD="$(git rev-parse --show-toplevel)/build"   # gitignored; holds the external clones
mkdir -p "$BUILD"

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

ensure_repo https://github.com/hw-native-sys/pto-isa        "$BUILD/pto-isa"
ensure_repo https://github.com/hw-native-sys/pypto          "$BUILD/pypto"
ensure_repo https://github.com/hw-native-sys/pypto-lib      "$BUILD/pypto-lib"
ensure_repo https://github.com/hw-native-sys/pypto-serving  "$BUILD/pypto-serving"
# add more as needed; drop rows your investigation doesn't touch
```

Notes:

- `reset --hard origin/main` overwrites any local changes in the clone.
  If you have edits there, stash / commit them before invoking this skill.
- `submodule update --init --recursive` matches the pin recorded in
  each repo's commit — that's the version the repo's CI runs against.
- PTOAS / CANN / gcc-15 are **provided globally** — no download. Step 2.5's
  `pypto-setup --export` points `PTOAS_ROOT` at them.

## Step 2.5: Toolchain env — `pypto-setup`

The dev box and CI runners ship ptoas, CANN, and gcc-15 globally; the
`pypto-setup` helper hands you the env in one line:

```bash
eval "$(pypto-setup --export)"          # exports PTOAS_ROOT, ASCEND_HOME_PATH, GCC15_ROOT, PATH
```

Run `pypto-setup` (no args) to see every global component, its path, and
whether it's present. Why each matters:

- **`PTOAS_ROOT`** unset → the JIT compile silently sets `skip_ptoas=True` →
  no `kernel_config.py` emitted → the runtime can't assemble kernels onboard.
- **PTO-ISA checkout unavailable** → the simpler runtime build fails with
  `PTO-ISA not available`; manually clone pto-isa into `build/pto-isa`.

## Step 3: Override — pick which simpler the workload sees

The default is what pypto's submodule pins. Simpler dev usually wants
ONE of these instead:

### Override A: simpler `origin/main` HEAD

You want the latest merged simpler against the external workload (e.g.
confirm a recently merged PR didn't break a downstream case).

```bash
ensure_repo https://github.com/hw-native-sys/simpler "$BUILD/simpler-main"
source <your-venv>/bin/activate
pip install --no-build-isolation "$BUILD/simpler-main"
```

### Override B: this worktree's simpler (your PR branch)

You want to test in-flight changes (current PR / dev branch) against the
external workload. The install builds the onboard runtime against the
`pto_isa.pin` managed checkout. A stale simpler whose ABI
doesn't match the pypto compiler surfaces as **507018** on the first run.

```bash
source .venv/bin/activate
pip install --no-build-isolation .
```

**Re-editing simpler later → the runtime `.so` may NOT recompile.** Non-editable
`pip install .` (and `--force-reinstall`) rebuilds the *Python wheel* but can
**silently skip the runtime build** (the runtime builder skips when outputs
already exist), so a fresh C++ edit to `src/{arch}/runtime|platform/*` does not
take effect — you keep running the old binary (real footgun: a runtime fix or
diagnostic you added appears to do nothing). After editing simpler's C++, either
verify the installed `.so` actually changed, or rebuild it via the cmake cache
and sync to **both** load locations (`build/lib` AND the installed
`.venv/.../simpler_setup/_assets/build/lib`):

```bash
BD=build/cache/{arch}/onboard/tensormap_and_ringbuffer/{aicpu|host|aicore}
cmake --build "$BD" -j                       # incremental; recompiles changed .cpp
SO="$BD/libaicpu_kernel.so"                   # or libhost_runtime.so / aicore_kernel.o
for d in $(find .venv build/lib -path "*onboard*tensormap_and_ringbuffer*$(basename "$SO")"); do cp "$SO" "$d"; done
strings "$SO" | grep '<your-marker>'          # confirm your change is in the binary
```

(CI/dev normally "use `-e .`" — editable rebuilds incrementally on reinstall; but
multi-repo prefers non-editable to avoid the `_simpler_editable.pth` leak, so you
must take the verify-or-cmake-build step above instead.)

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

## Step 4: Install the compiler / kernel / serving layers

Install only the layers your workload imports (the Step 1 chain). For each,
prefer plain `--no-build-isolation` install and let the vendored simpler
stay unbuilt — you've already provided simpler via Step 3.

### pypto (compiler)

```bash
pip install --no-build-isolation "$BUILD/pypto"
# Do NOT `pip install "$BUILD/pypto/runtime"` — that would
# overwrite the simpler you installed in Step 3 with pypto's older pin.
```

### pypto-lib (kernels) — import-only

```bash
export PYTHONPATH="$BUILD/pypto-lib"   # no install; run scripts out of its tree
```

### pypto-serving (serving + runners)

Install or `PYTHONPATH`-import per that repo's README (see Step 5). Its
qwen3 runner is the entry point the
[`multi-repo-qwen-setup`](../multi-repo-qwen-setup/SKILL.md)
skill drives.

Re-verify the loaded simpler after every install — an install can
re-resolve dependencies in ways that change what wins.

## Step 5: Hand off to each repo's own skills/docs

This skill gets the repos cloned, the toolchain env exported, and the
simpler-of-your-choice installed. **What to run inside each repo — its
test entry points, flags, golden-data setup, DFX knobs — is documented in
that repo, not here.** After Step 4, for each external repo you're using,
consult, in this order:

1. **That repo's `.claude/skills/`** (if it has one) — task-specific
   workflows maintained by that repo's owners.
2. **That repo's top-level `README.md`** and any `docs/` — setup and
   run recipes.
3. **The workload script's own docstring / `--help`** — often the most
   precise statement of intent and flags.

```bash
ls "$BUILD/<repo>/.claude/skills" 2>/dev/null   # repo-owned skills, if any
sed -n '1,40p' "$BUILD/<repo>/README.md"        # repo setup/run recipe
python "$BUILD/<repo>/<workload>.py" --help      # the script's own contract
```

Do not duplicate those instructions into simpler's skills — link to them.
The one exception maintained here is the qwen3 workload, whose cross-repo
timing/ring/timeout gotchas are simpler-specific enough to warrant the
dedicated [`multi-repo-qwen-setup`](../multi-repo-qwen-setup/SKILL.md)
skill.

## Step 6: If the workload fails — start with "is it CI-gated?"

The most informative first check is whether the workload is gated by
that repo's CI workflow.

| case | meaning | how to triage |
| ---- | ------- | ------------- |
| **CI-gated** | the repo's CI runs this exact script today; it was passing as of the last green CI run | A failure now usually points at a **recent code change** — your in-flight simpler changes, or a commit landed in pypto / pypto-lib / pypto-serving since the last CI run. Bisect against `origin/main` of each repo. |
| **not CI-gated** | the script is in the repo but no workflow invokes it | Read the docstring first. Files like this are often "intent" / "EXPECTED / INTENT program" / experimental drafts — they may be documented as expected-to-fail. Treat as workload bug, not simpler bug, unless proven otherwise. |

Quick check:

```bash
F=<workload>.py
grep -nE "python .*$(basename $F)" "$BUILD/"*/.github/workflows/*.yml
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
| `FileNotFoundError: kernel_config.py not found in ...` | `PTOAS_ROOT` unset → pypto auto-skips ptoas → no `kernel_config.py` emitted | `eval "$(pypto-setup --export)"` (Step 2.5) |
| PTO-ISA unavailable | no checkout | clone into `build/pto-isa` |
| script exits 0 but no device run | compile-only smoke fallback (golden data dir missing) | pass `--data-dir <golden>` or `--smoke` explicitly |
| `aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018` | binary skew (simpler runtime vs pypto compiler ABI), OR device log is the only ground truth | rebuild simpler against the matching pypto (Step 3/4); then read `~/ascend/log/debug/device-N/device-<pid>_*.log` |
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
workload (or your code change), not chip contention. For the full
`507018` mechanism-classification table (deadlock vs op-timeout vs
stall), see [`.claude/rules/running-onboard.md`](../../rules/running-onboard.md).

## Anti-patterns

- ❌ **Trusting an existing local clone without `git fetch`**. Your
  clone is whatever you last fetched, possibly weeks behind. Step 2
  exists precisely to make this not a thing.
- ❌ **Re-documenting another repo's run steps here**. Link to that
  repo's own `.claude/skills/` / README (Step 5) — duplicated steps go
  stale the moment that repo moves.
- ❌ **Using `-e` "just to be safe"**. Editable installs leak a
  user-site finder hook that survives sessions and shadows the next
  venv install. Plain install is the default; reach for `-e` only
  when you'll actively edit.
- ❌ **Blaming chip contention before reading the device log**. The
  device log either shows the contention signature (sibling-die cores
  with `cond_reg_state=ack` from another owner) or it doesn't.
- ❌ **Treating any failing workload as "simpler broke it"**. Step 6's
  CI-gate check separates "your simpler change broke a CI case" (real)
  from "this file was always expected to fail" (not your problem).
- ❌ **Skipping `.claude/rules/running-onboard.md`** on onboard
  hardware runs. Multi-repo flows don't waive the per-die lock.
