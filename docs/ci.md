# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Merge by runner, not by language** — Python and applicable C++ unit tests share setup cost and run as steps within a single job per runner tier. A runner with no platform-specific C++ tests does not carry an empty CTest step; currently `ut-a5` is Python-only.
2. **Runner matches hardware tier** — no-hardware tests run on `ubuntu-latest`; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim` jobs run on github-hosted runners alongside unit tests.
5. **Skip irrelevant platforms, and irrelevant suites** — `detect-changes` gates `st-sim-*` and `st-onboard-*` by platform, so pure-a5 PRs skip a2a3 scene-test runs and vice versa. **UT jobs are still not gated by platform** — unit tests cover shared contracts and the cost of a falsely-skipped regression outweighs the savings. They *are* gated by test category: a diff confined to `tests/st/` or `examples/` cannot break a unit test, and one confined to `tests/ut/` cannot break a scene test, because the two suites execute disjoint trees (`pytest examples tests/st` vs `pytest tests/ut` plus the C++ ctest) and no unit test reads `examples/` or `tests/st/`. Shared test infrastructure — the root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/` — belongs to neither category and runs both.
6. **Non-code PRs run pre-commit and docs, and nothing else** — `detect-changes` sets `non_code_only` when *no* changed file falls outside the `NON_CODE` set. Skipping the UT jobs is riskless there because nothing in that set can change what the code does, and each member already has its own gate: markdownlint inside `pre-commit` reads the markdown, `docs.yml` (unconditional on every PR) builds the site with `--strict`, and `pre-commit` itself is ungated so a `.pre-commit-config.yaml` change is fully exercised by it.

## Full Job Matrix

The complete test-type × hardware-tier matrix. Empty cells have no tests yet; only non-empty jobs exist in `ci.yml`.

| Category | github-hosted (no hardware) | a2a3 runner | a5 runner |
| -------- | --------------------------- | ----------- | --------- |
| **ut** | `ut` (py + cpp) | `ut-a2a3` (py + cpp) | `ut-a5` (py) |
| **st** | `st-sim-a2a3`, `st-sim-a5` | `st-onboard-a2a3`, `st-network1-onboard-a2a3`, `st-deepseek-onboard-a2a3` | `st-onboard-a5` |

## GitHub Actions Jobs

`ci.yml` and `ci-self-cpu.yml` are now thin topology callers: they own
triggers, `needs:`, gate `if:` expressions, runner/setup inputs, and matrix
shape. The executable job bodies live in reusable workflows:
`_detect-changes.yml`, `_pre-commit.yml`, `_ut-no-hardware.yml`,
`_packaging.yml`, `_profiling-flags-smoke.yml`, `_st-sim-a2a3.yml`,
`_st-sim-a5.yml`, `_ut-npu-a2a3.yml`, `_ut-npu-a5.yml`,
`_st-npu-a2a3.yml`, `_st-npu-a5.yml`, `_st-network1.yml`, and
`_st-deepseek-a2a3.yml`. The scene-test and
NPU unit-test bodies are split one workflow per architecture so each job
renders only its own steps. Shared step scaffolding that is safe to run
after checkout lives in composite actions under `.github/actions/`
(`apt-install`, `cache-pip`, `setup-gcc-15`, `setup-venv`, and the three `network1-*` actions).
`setup-gcc-15` accepts a complete, pre-provisioned GCC 15 toolchain on any
Linux runner; automatic installation of missing Linux tools is Ubuntu-only,
while macOS installation uses Homebrew. `_packaging.yml` deliberately retains
its separate compiler setup: Linux packaging accepts the platform compiler and
also installs ccache, while its macOS fallback is outside the shared action's
strict GCC 15 contract. Only the compiler is required there — ninja comes from
PyPI with the venv, and both the workflow and `tools/verify_packaging.sh` run
without ccache — so a package manager that cannot serve ccache emits a warning
and the job continues. The packaging matrix preserves any explicit `CC`/`CXX`
selection and resolves its required `gcc`/`g++` pair once when neither is set,
so every install mode and its nested runtime builds use one compiler identity.
The job logs the default pair's resolved paths and compiler versions, and
reports an explicit runner-provisioning error if that pair is absent.
`apt-install` gives every apt operation a wall-clock
bound and attempts to repair an interrupted dpkg transaction before another apt
operation can run. It uses the runner's existing package indexes by default, avoiding a
full mirror refresh for ordinary CI dependencies. `setup-gcc-15` is the one
exception: after installing its pinned Toolchain PPA source, it refreshes only
that `.sources` file, retrying a failed refresh once with a 60-second bound per
attempt. That isolated refresh disables list cleanup, preserving the runner's
Ubuntu indexes for the following install. A failed refresh is a warning; the
required package install, bounded at four minutes and retried once for this
PPA workload, decides whether the job can continue. An install from cached
indexes that fails receives one bounded system-refresh retry. The acquire
timeouts remain a per-connection safeguard, not a wall-clock guarantee.

```text
PullRequest
  ├── pre-commit             (ubuntu-latest)
  ├── packaging-matrix       (ubuntu + macOS)        — [needs !examples_only && !tests_only]
  ├── profiling-flags-smoke  (ubuntu-latest)         — (a2a3_changed || a5_changed) && !examples_only && !tests_only
  ├── ut                     (ubuntu + macOS)        — Python + C++ UT, no hardware [needs ut_affected]
  ├── detect-changes         (reusable; ubuntu-latest in ci.yml) — outputs non_code_only, a{2a3,5}_changed, {st,ut}_affected, examples_only, tests_only
  ├── st-sim-a2a3            (ubuntu + macOS)        — a2a3_changed && st_affected
  ├── st-sim-a5              (ubuntu + macOS)        — a5_changed && st_affected
  ├── ut-a2a3                (a2a3 self-hosted)      — Python + C++ UT, a2a3 hardware [needs ut_affected]
  ├── st-onboard-a2a3        (a2a3 self-hosted)      — a2a3_changed && st_affected
  ├── st-network1-onboard-a2a3    (a2a3pod pair)          — a2a3_changed && st_affected
  ├── st-deepseek-onboard-a2a3    (a2a3 self-hosted)      — a2a3_changed && st_affected
  ├── ut-a5                  (a5 self-hosted)        — Python UT, a5 hardware [needs ut_affected]
  └── st-onboard-a5          (a5 self-hosted)        — a5_changed && st_affected
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut` | `ubuntu-latest`, `macos-latest` | `pytest tests/ut` + `ctest -LE requires_hardware` |
| `st-sim-a2a3` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a2a3sim` |
| `st-sim-a5` | `ubuntu-latest`, `macos-latest` | `pytest examples tests/st --platform a5sim` |
| `ut-a2a3` | a2a3 self-hosted | `pytest tests/ut --platform a2a3` + `ctest -L "^requires_hardware(_a2a3)?$" --resource-spec-file ...` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (host + cross-compiled device SO, link smoke only) |
| `st-onboard-a2a3` | a2a3 self-hosted | `pytest examples tests/st -m "not sdma" --platform a2a3 --exclude-level 4 --device ...`, then a separate `-m sdma` step, then adaptive-parallel DFX feature smokes |
| `ut-a5` | a5 self-hosted | `pytest tests/ut --platform a5` + build `tools/cann-examples/query` and run `query version` (no device) + build `tools/cann-examples/aicpu-device-query` and `tools/cann-examples/aicpu-kernel-launch` (link smoke only) |
| `st-onboard-a5` | a5 self-hosted | `pytest examples tests/st --platform a5 --exclude-level 4 --device ...`, including SDMA tests, then adaptive-parallel DFX feature smokes |
| `st-network1-onboard-a2a3` | a pair of `a2a3pod` machines | `pytest examples tests/st --level 4 --platform a2a3 --device ... --max-parallel 1`, one L3 daemon on the peer |
| `st-deepseek-onboard-a2a3` | a2a3 self-hosted | `pytest <the two deepseek_v4_flash_decode dirs> --platform a2a3 --manual only --device ...`, both 2-device cases concurrently, in parallel with `st-onboard-a2a3` |

### Multi-machine network1 jobs

`st-network1-onboard-a2a3` is the only job spanning two machines. The runner it
lands on becomes the L4 parent and drives its peer entirely over ssh; the peer
runs no workflow code at all. Everything that identifies either machine —
addresses, the device split, ports, the staging root, proxies — comes from a
`.env` the runner carries, so `_st-network1.yml` holds no machine-specific value.
Adding or re-addressing a machine is an edit to that file. Only a machine
hosting a runner needs one.

Two names in this job are provisioned on the machines rather than in this repo,
so they still carry the older word: the `a2a3pod` runner label and the `POD_*`
keys a machine's `.env` may use. The config loader accepts either prefix and
exports one, so a machine converts independently; a runner label cannot be
read either way, because a job matches every label in its list.

Its body splits by what is per-run and what is per-pytest session:

| Action | Called | What it does |
| ------ | ------ | ------------ |
| `network1-stage` | once | rsync this run's tree onto the peer and build it there |
| `network1-run-pytest` | once | start the peer's L3 daemon, set the network1 pytest environment, run `pytest examples tests/st --level 4`, stop the daemon and pull its logs |
| `network1-teardown` | once, `if: always()` | remove the run's tree from the peer |

Staging and the peer-side build are the job's whole cost, and every network1 test
runs against that same tree and venv, so they happen once. The pytest command
owns selection through the scene-test level axis: adding an L4 network1 example
means adding a `test_*.py` wrapper with `@scene_level(SceneTestLevel.NETWORK1)`, not
editing `_st-network1.yml`. `network1_remote_device_count` stays as the peer-resource
declaration; `network1` is the runner topology, not a pytest selection marker.

Network1 logs go to `output/network1-ci-<run>-<attempt>/pytest/` while the job is running.
Parent-side `ASCEND_PROCESS_LOG_PATH` is split per pytest nodeid by
`st_network1_logs`; peer-side daemon/device logs are grouped for the pytest session.
The directory is uploaded on a best-effort basis: artifact-service failures are
ignored so they cannot override the `network1` test result. Preserve relevant
diagnostics in the inline job log as well when investigating a failure.

Writing an example — the files, the entry module, the `run(...)` entry point,
the `test_*.py` network1 wrapper, and the manual `run_parent.sh` — is covered in
[`examples/workers/README.md`](../examples/workers/README.md).

### Daily full scene-test sweep

[`daily.yml`](../.github/workflows/daily.yml) runs the full regular + manual
scene-test corpus with `--manual include` once per day and supports manual
re-runs through `workflow_dispatch`. Simulation runs on Ubuntu and macOS for
both architectures; onboard runs on the A2/A3 and A5 self-hosted pools. The
same DFX smoke steps used by Per-PR run once in each Daily platform job, and
the A2/A3 `network1` corpus runs through the existing two-machine workflow. The
main Per-PR scene-test steps keep the default `--manual exclude`, so moving an
ordinary case to Daily does not require a second workflow exclusion list.
The dedicated dep-gen, chip-swimlane, PMU, args-dump, and scope-stats steps
instead use
`include` for the normal Per-PR and Daily modes because they own the full
corpus under their target paths; a `manual_mode` of `only` remains `only` in
those steps. Marking a case under one of those paths manual therefore removes
its duplicate main-step execution without removing its dedicated Per-PR
coverage. Scope-stats needs its own step for a reason the others do not: its
scene test runs in the main sweep either way, but `_validate_scope_stats_artifact`
returns early unless `--enable-scope-stats` is supplied, so without the dedicated
step the run happens and the `scope_stats.jsonl` it should have produced is never
asserted on.

Use `"manual": True` on an individual `SceneTestCase.CASES` entry and
`@pytest.mark.manual` on a standalone pytest test. The reusable scene-test
workflows accept `manual_mode`; Per-PR callers use its `exclude` default and the
Daily caller passes `include`.

For platform-specific pruning, the same `manual` value accepts a platform list,
for example `"manual": ["a2a3sim", "a5sim"]`; standalone tests use
`@pytest.mark.manual(["a2a3sim", "a5sim"])`. This removes only the listed Sim
executions from Per-PR. It retains same-architecture onboard coverage only when
the case also declares that onboard platform; otherwise that architecture's
path becomes Daily-only and must be called out in the PR's coverage analysis.

### Nightly sanitizer sweep

A **separate** workflow, [`sanitizers.yml`](../.github/workflows/sanitizers.yml),
runs on a nightly `schedule` — kept out of `ci.yml` so the cron fires only the
sanitizer jobs, never the PR/self-hosted pipeline. Its
`sanitizer-sim` job builds the sim runtime + kernels with ASAN or TSAN
(`pip install --config-settings=cmake.define.SIMPLER_SANITIZER=...`) and runs a
**scoped** subset under the matching `LD_PRELOAD` (a2a3sim/a5sim, ubuntu-only).
`dlopen_count` tests are excluded everywhere (they assert exact dlopen accounting
that the sanitizers perturb by interposing `dlopen`). The full suite is avoided
because ASAN/TSAN slow the sim enough that oversubscription-heavy cases livelock
on a 4-vCPU runner — so the scope is parallelism-limited per sanitizer:

- **ASAN** (~1.7x): `prepared_callable` + `dynamic_register` (where present),
  `--max-parallel 2`, skipping `parallel_broadcast`.
- **TSAN** (~5-15x): livelocks the chip-fork L3 cases even when run serially, so it
  runs only the light `prepared_callable` L2 tests, `--max-parallel 1`, with
  `TSAN_OPTIONS=halt_on_error=0:exitcode=0` (report races without aborting *or*
  failing the job — TSAN's default `exitcode=66` would otherwise redden the cell on
  every race; the job gates on hang/crash, triaging the reported races into a
  suppressions file is a follow-up).

Both sanitizer jobs gate (no `continue-on-error`). Not a PR gate; see
[sanitizers.md](sanitizers.md) for the design + usage.

### Parallel ST runs on hardware

For self-hosted jobs with multiple NPUs, pass a `--device` range (and
optionally pytest's `-x` for fail-fast) to get the full dispatcher
benefit — device bin-packing for L3, xdist fanout for L2, and a shared
`ChipWorker` per `(runtime, device)`:

```bash
# Recommended CI invocation — a2a3 deselects SDMA and network1 tests, as the job does,
# and runs SDMA as a second pass afterwards
pytest examples tests/st -m "not sdma" --platform a2a3 --exclude-level 4 --device 4-7 -x
pytest examples tests/st -m sdma --platform a2a3 --device 4-5 -x

# A5 runners run the non-network1 corpus, including SDMA tests
pytest examples tests/st --platform a5 --exclude-level 4 --device 0-7 -x
```

`-x` (`--exitfirst`) is appropriate for CI, where aborting on first
failure saves runner minutes. Local development usually wants the opposite
(let every failure surface) — just drop the flag. The short form is the
same in both pytest and standalone on purpose; see
[testing.md §CLI Design Principles](testing.md#cli-design-principles).

`pytest-xdist` is pulled in via the `test` extra. See
[testing.md §Parallel Test Execution](testing.md#parallel-test-execution-and-resource-reuse)
for the full hierarchy, fail-fast semantics, and the
profiling-vs-parallelism trade-off.

### Targeted runtime builds

The sim, onboard, and network1 jobs select one explicit
`build_package_<platform>` CMake target during package install. That aggregate
target depends on both the `_task_interface` binding and the selected platform's
runtime target, so scikit-build-core makes one `cmake --build` call and CMake can
build both dependencies in parallel while omitting unused platforms. The
selection is not a cached CMake variable, so a later ordinary package install
in the same worktree still uses the default `ALL` target and auto-detects every
available platform. The network1 stage passes the same target to its peer.
Pre-commit selects its build from the existing files in the merge-base diff,
the same file set its hooks inspect. `tests/lint/clang_tidy_paths.py` supplies
the shared path policy for the clang-tidy hook and this build selection. A
clang-tidy-eligible C/C++ change builds
the combined `build_package_sim` target so clang-tidy has both simulator
compile databases. Files excluded by the clang-tidy hook — vendored
`3rdparty/`, Python bindings, kernels, and AICore sources — do not trigger that
build. Python, documentation, and other recognized non-C++ changes also skip
it. Changes to `.pre-commit-config.yaml`, and all unrecognized paths, remain
conservative and build the combined target. Self-hosted CPU runs still
create a lightweight venv so the pre-commit action never installs into the
runner's system Python; only diffs selected for clang-tidy preparation install
the package and prepare its compiler stand-ins. Each sanitizer matrix cell
selects its own platform. The profiling-flags smoke installs only the
`_task_interface` binding because its matrix builds every runtime configuration
itself.

### Sim jobs on CPU-constrained runners

Sim jobs (`st-sim-a2a3`, `st-sim-a5`) run on `ubuntu-latest`, whose standard
GitHub-hosted runner currently has **4 vCPUs**. `--device 0-15` is still the
right choice for the **pool size** (some L3 cases need several virtual ids), but
the default `--max-parallel auto` caps the in-flight subprocess count to
`min(nproc, len(--device))` — on a 4-core runner that becomes `4`. Note
`os.cpu_count()` reports the host's logical CPUs and ignores any cgroup CPU
quota, so this is the true core count, not a container limit.

```bash
# Sim: --max-parallel auto resolves to 4 on a standard ubuntu-latest runner
pytest examples tests/st --platform a2a3sim --device 0-15

# Throttle further on a CPU-starved runner: 4 concurrent cases (each forking
# several chip subprocesses with many threads) can oversubscribe 4 cores and
# trigger the sim handshake/deinit failures in
# troubleshooting/sim-oversubscription-hang.md. --max-parallel 2 trades
# throughput for stability.
pytest examples tests/st --platform a2a3sim --device 0-15 --max-parallel 2
```

On hardware jobs the `auto` default is `len(--device)` because each subprocess
is device-bound (host CPU mostly waits on the NPU), so hardware runners do
not need `--max-parallel` manually.

### Scheduling constraints

- Sim scene tests and no-hardware unit tests run on github-hosted runners (no hardware).
- `detect-changes` is implemented once in [`.github/workflows/_detect-changes.yml`](../.github/workflows/_detect-changes.yml) and computes four axes from the PR diff — non-code (`non_code_only`), architecture (`a2a3_changed` / `a5_changed`), test category (`st_affected` / `ut_affected`), and corpus (`examples_only` / `tests_only`) — **all of them derived from one `NON_CODE` set**: `docs/`, `.docs/`, `.claude/`, `mkdocs.yml`, `.github/workflows/docs.yml`, `.gitignore`, `.pre-commit-config.yaml`, and any `*.md` file anywhere. Membership follows a file's *effect*, not its path — `mkdocs.yml` and `docs.yml` are docs tooling that happens to live outside `docs/`. An arch flag is `false` only when every changed file is in the opposite platform's tree (`src/{arch}/`, `examples/{arch}/`, `tests/{st,ut/cpp}/{arch}/`) or in `NON_CODE`. Anything else — shared C++ (`src/common/`), Python (`python/`, `simpler_setup/`), build files (`CMakeLists.txt`, `pyproject.toml`), shared test infra (`tests/ut/py/`, `tests/lint/`), tooling (`tools/`), or any CI implementation workflow (`.github/workflows/ci.yml`, `.github/workflows/ci-self-cpu.yml`, `.github/workflows/_*.yml`) — flips both flags to `true`. CI implementation workflows are deliberately excluded from `NON_CODE`: a change to the gates or reusable job bodies must run everything, including whatever it just switched off.
- **Test-category axis:** `ST_ONLY='^(tests/st/|examples/)'` and `UT_ONLY='^tests/ut/'`, applied in the same shape as the arch patterns — a category is unaffected only when *every* changed file is exclusively the other's. `st_affected` gates the four scene-test jobs; `ut_affected` gates `ut`, `ut-a2a3`, `ut-a5`. Anything belonging to neither (root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/`) flips both, so shared infrastructure always runs both suites.
- **Corpus axis:** `EXAMPLES_ONLY='^examples/'` and `TESTS_ONLY='^(tests/)'`, same shape again, one per side of the product jobs' payload. `packaging-matrix` and `profiling-flags-smoke` build and install the product and then exercise it with a fixed, tiny payload — one entry-point script (a `tests/st/` file) and one `vector_example` — so neither reads either suite as a corpus, and a diff confined to `examples/` or to `tests/` cannot reach them: `wheel.packages` is `["simpler_setup", "python/simpler"]`, so both partitions are provably absent from the product. A payload file changed under either is still exercised by the scene-test job that reads the same corpus.
- **Gated jobs (scene tests):** `st-sim-{a2a3,a5}`, `st-onboard-{a2a3,a5}` run iff their platform's flag **and** `st_affected` are `true`.
- **Platform-independent jobs (all UT + packaging):** `ut`, `ut-a2a3`, `ut-a5`, `packaging-matrix` ignore the *platform* flags — unit tests exercise shared contracts (nanobind bindings, RuntimeBuilder, ring buffers, etc.) and the risk of silently skipping a regression outweighs the CI minutes saved. The `tests/ut/cpp/{arch}/` entry in the gating regex only *attributes* an arch-specific C++ UT change to that platform (so it does not spuriously flip the other arch's scene-test flag); it does not gate the UT jobs themselves. The three UT jobs do respect `ut_affected`, which is a statement about test category rather than silicon. `packaging-matrix` respects the corpus axis, which is a statement about which corpus a job reads: neither `examples/` nor `tests/` is in `wheel.packages`, so a diff confined to either partition cannot change what the packaging job builds.
- **`non_code_only` is the same `NON_CODE` set, not a narrower one.** It is `true` when no changed file falls outside it. Nothing in the set can change what the code does, and no workflow consumes any of it beyond its own gate: `pre-commit` is ungated so it always exercises `.pre-commit-config.yaml`, `docs.yml` is unconditional on every PR so it always exercises `mkdocs.yml` / `docs/`, and **no workflow invokes anything under `.claude/`** (`grep -rn '\.claude' .github/workflows/` finds only comments). An **empty diff short-circuits the whole step**: attribution is impossible, so a single guard sets `non_code_only=false`, every arch and category flag `true`, and both corpus flags (`examples_only`, `tests_only`) `false`, then returns — running the full matrix, packaging and the profiling smoke included. That guard is deliberately one place; testing emptiness per flag is what previously left `non_code_only` false while both arch flags also came out false, running UT and packaging but skipping every scene test.

  The arch flags subtract `NON_CODE` before deciding, so a non-code-only change already makes both `false`. An arch-gated job therefore needs no separate non-code check. See [`.claude/rules/ci-change-detection.md`](../.claude/rules/ci-change-detection.md) for the invariants these gates must keep.

- **SDMA tests run as their own step inside `st-onboard-a2a3`.** The ordinary sweep deselects them with `-m "not sdma"` and `--exclude-level 4`, and a later step runs `-m sdma`. Ordering is what the two SDMA paths share: the SDMA step is always second, so no fault-injection case can land on a device that has already provisioned SDMA. Device acquisition differs by host arch — on aarch64 the SDMA step takes its own `task-submit --device auto --device-num 2`, so the two steps are disjoint in devices as well; on x86_64 there is no `task-submit` and both steps use the same `${DEVICE_RANGE}`, leaving ordering as the only separation. Provisioning the SDMA workspace creates device-only STARS streams that live in the device fault domain, so an AICore fault on a device that has provisioned SDMA costs minutes instead of milliseconds — the sweep's `aicore_op_timeout` fault injection must therefore never share a device with them ([#1425](https://github.com/hw-native-sys/simpler/issues/1425)). Selection for SDMA remains by marker on both sides, so the two cannot drift apart; the split can be dropped once #1425 is fixed. Network1 tests are selected by `--level 4` in `st-network1-onboard-a2a3` and explicitly excluded from ordinary onboard ST lanes.

### CPU emergency lane (`ci-self-cpu.yml`) and the `/run-cpu` button

When GitHub-hosted runners are congested, a repo admin can validate a PR on the
repo-level self-hosted runners via the emergency lane, bypassing GitHub-hosted
queueing entirely:

- **Trigger**: comment `/run-cpu` on the PR (`ci-self-cpu-button.yml`), or
  `gh workflow run ci-self-cpu.yml -f repository=<repo> -f ref=<sha>` manually
  (covers forks and arbitrary SHAs). The button gate is
  `permission(commenter) == 'admin'` only (`getCollaboratorPermissionLevel`).
  `issue_comment` from fork PRs runs with a read-only token, so the permission
  check must work under it — verify on first fork-PR use.
- **What it runs**: checks out `repository@ref` (the PR head), then T1 — the
  no-hardware Linux jobs (`pre-commit`, `ut`, `packaging`, `profiling-flags-smoke`,
  `st-sim-{a2a3,a5}`) on `[self-hosted, cpu]` — and T3 — the NPU jobs
  (`ut-a2a3`, `st-onboard-a2a3`, `ut-a5`, `st-onboard-a5`) on
  `[self-hosted, a2a3/a5]`. T2 (macOS) is intentionally absent. The lane calls
  the same reusable job-body workflows as `ci.yml`, passing
  `setup_variant=self-cpu`, `repository`, `ref`, and self-hosted runner labels
  where the main CI passes `setup_variant=github` and GitHub-hosted runners.
  Gate outputs still come from the canonical `detect-changes` workflow, executed
  on `[self-hosted, cpu]` in this lane.
- **cpu runner contract**: dnf-installed `cmake ninja-build gcc-c++ clang-tools-extra graphviz gtest-devel python3-devel`, plus a pip-installable torch aarch64 CPU wheel. `setup-venv` installs a PyPI `ninja` into the venv on every lane, which shadows the dnf-provided `ninja-build` on `PATH` whenever the venv is active — deliberately, so CMake's generator selection is identical on this lane and on the github-hosted ones. Same tool, different provenance; the dnf package stays in the contract for builds that run outside the venv. When the pre-commit selector requests clang-tidy preparation — for eligible C/C++, `.pre-commit-config.yaml`, or an unknown path — the job creates `g++-15` as a stand-in for the Ubuntu Toolchain PPA compiler. On the agents `g++` resolves to a conda GCC 15 prefix rather than `/usr/bin/g++`, so that diff's sim artifacts are built with GCC 15 and `compile_commands.json` names that prefix's `<triple>-g++`; `tests/lint/clang_tidy.py` drops the triple before replaying a command, without which clang-tidy adopts it as a target and resolves no C++ standard library at all. `ci.yml` lints with clang-tidy 18 and HCE 2.0 packages only LLVM 12, so an agent additionally provides 18 on `PATH` as `clang-tidy-18`, installed together with its clang builtin headers — a clang-tidy whose prefix carries no `lib/clang/<major>/include` resolves no resource dir and fails every `#include <stddef.h>`. The pre-commit job shadows the distro `clang-tidy` only on that build path; Python-only and other lint-only diffs do not build sim artifacts or create either compiler shim.
- The lane run is standalone — it attaches no checks to the PR; results are read from the run.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [testing.md](testing.md#hardware-classification) for the full table including per-category mechanisms (pytest markers, ctest labels, folder structure).

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `ubuntu-latest` | `ut`, `st-sim-*` |
| Platform-specific (a2a3) | `[self-hosted, a2a3]` | `ut-a2a3`, `st-onboard-a2a3` |
| Platform-specific (a5) | `[self-hosted, a5]` | `ut-a5`, `st-onboard-a5` |

On a self-hosted runner, every step that touches an NPU — pytest and ctest
alike — must hold its devices exclusively while it runs. There are two a2a3
runner pools, branched at run time on the host arch (`uname -m`):

- **ARM64 a2a3 runners** share the host with interactive users, so the step
  runs through `task-submit --device <list> --run "..."`, whose per-device
  lock keeps a CI job from colliding with someone's local run (and vice
  versa).
- **X64 a2a3 runners** do not use `task-submit` — their cards are exclusive to
  the runner — so the step runs `pytest`/`ctest` directly with
  `--device ${DEVICE_RANGE}`.

a5 runners always use `task-submit` and run the full non-network1 scene-test corpus,
including SDMA tests, on both x86_64 and ARM64. Steps that only build (cmake,
`RuntimeBuilder`, the
`cann-examples` smokes) take no lock on either arch. The same device-lock rule
applies to local onboard work — see
[.claude/rules/running-onboard.md](../.claude/rules/running-onboard.md).

The two onboard scene-test jobs (`st-onboard-a2a3`, `st-onboard-a5`) compile
their selected test batch before entering `task-submit`. The lock-free warm-up
populates the `build/cache/kernels/` cache; the subsequent pytest invocation
keeps the existing batch-level device allocation and reconstructs each
`ChipCallable` from that cache. The warm-up does not acquire one lock per case,
and runners with exclusive devices continue to execute pytest directly.
Both onboard jobs pass `--compile-workers 8`, which caps the entire warm-up at
eight compiler processes. Class-level concurrency and the parallel artifacts
inside a large callable share that budget instead of multiplying it. Without an
override the automatic budget reserves two logical CPUs and caps at eight. The
sim jobs run on ephemeral GitHub-hosted runners with no restored cache, so they
compile cold every time and get no warm-up step.

The DFX smokes reuse the runner's device allocation after the main scene-test
sweep. The `run-onboard-dfx-smokes` CI action distributes `dep_gen`, chip
swimlane, PMU, and args dump round-robin across the allocated devices and keeps
`-p no:xdist` inside each pytest process. Those are the four a5 smokes; a2a3
adds a fifth smoke for host-build-graph `dep_gen`. A device runs at most one
smoke at a time, so a one-device allocation runs serially while multiple
devices run as many independent lanes as both the allocation and platform's
smoke count permit. With four or fewer a2a3 devices, the fifth smoke waits to
reuse its round-robin device. Each smoke retains its own log and exit status,
so one failure does not suppress the remaining DFX results; the aggregate step
fails after all logs have been reported.

A warm-up that cannot compile a class reports it and keeps going: the pass only
fills a cache, so the locked pytest run that follows is what recompiles the
class and attributes the failure to the case that owns it, instead of one
unbuildable kernel costing the whole batch its results.

`actions/checkout` cleans ignored files before each job, so the onboard jobs
restore and save `build/cache/kernels/` through `actions/cache`. Cache keys are
partitioned by target architecture, runner OS/architecture, and PTO-ISA pin.
The callable key covers the contents of its orchestration, incore, and
transitively included sources, the compiler identities and effective fixed
flags, a digest of the modules that decide artifact bytes
(`kernel_compiler.py`, `toolchain.py`, `compile_paths.py`, `elf_parser.py`), the binding's
serialized-callable ABI, a manual schema constant, and the owning test class's
qualified name. Each incore key covers the source closure, stable
compiler-visible paths, and the compilation inputs that affect that kernel
binary. The compiler runs from the checkout root and receives checkout-local
paths relative to it, so `__FILE__` and `__BASE_FILE__` stay stable when CI
runs on a different runner; paths outside the checkout remain absolute. The key
excludes orchestration, callable ABI, `func_id`,
signature, and owning test class, so callables that reference the same kernel
path can share its artifact.

This cache complements rather than replaces `ccache`: the packaging workflow
uses `ccache` for supported CMake compiler invocations, while scene-test cache
entries also retain serialized callables and final incore bytes, including the
CCEC/linker and ELF-section processing used by onboard builds.

Entries are content-addressed and therefore never overwritten, so a run prunes
entries whose last use is more than 14 days old before it exits. Without that,
the directory grows by one entry per kernel change forever and the saved
`actions/cache` archive — one new entry per push, restored by prefix — would
crowd the repository's 10 GB cache budget and evict the pip and other caches
the other jobs depend on.

The pip download cache is keyed by runner OS, runner architecture, and
`pyproject.toml`. Source-only changes therefore reuse the same dependency cache
instead of creating another roughly 200 MB archive; pip still resolves and
validates requested versions on every install.

## Test Sources

### `tests/ut/` — Python unit tests (ut-py)

Python unit tests. Run via pytest, filtered by `--platform` + `requires_hardware` marker.

| File | Content | Hardware? |
| ---- | ------- | --------- |
| `test_task_interface.py` | nanobind extension API tests | No |
| `test_runtime_builder.py` (mocked classes) | RuntimeBuilder discovery, error handling, build logic | No |
| `test_runtime_builder.py::TestRuntimeBuilderIntegration` | Real compilation across platform × runtime | Yes (`@pytest.mark.requires_hardware`) |

### `tests/ut/cpp/` — C++ unit tests (ut-cpp)

GoogleTest-based tests for pure C++ modules. Run via ctest, filtered by label `-LE` exclusion.

| Runner | Command |
| ------ | ------- |
| No hardware | `ctest --test-dir tests/ut/cpp/build -LE requires_hardware` |
| a2a3 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a2a3)?$"` |
| a5 | `ctest --test-dir tests/ut/cpp/build -L "^requires_hardware(_a5)?$"` |

The no-hardware job follows ctest with two checks that read the build rather
than run it, catching what ctest structurally cannot.
`tests/lint/check_ut_cpp_axis.py`: a case built for one arch or one runtime
passes exactly like one built for every configuration.
`tests/lint/check_ut_cpp_stub_linkage.py`: a stub the support archive lost to a
weak fallback links fine and then hangs, so ctest only reaches it by timing
out. The first reads the build's compile lines, so that job configures with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. See
[testing.md](testing.md#how-many-times-a-case-is-built).

### `examples/` — Small examples (sim + onboard)

Small, fast examples that run on both simulation and real hardware. Organized as `examples/{arch}/{runtime}/{name}/`. Discovered and executed by pytest via each example's `test_*.py` (`@scene_test` format).

### `tests/st/` — Scene tests (onboard-biased)

Large-scale, feature-rich hardware tests. Too slow or using instructions unsupported by the simulator. Organized as `tests/st/{arch}/{runtime}/{name}/`. Platform compatibility is declared per test via `@scene_test(platforms=[...])`.

### Shared structure

Both `examples/` and `tests/st/` cases follow the same layout:

```text
{name}/
  test_{name}.py                 # @scene_test class (generate_args, compute_golden)
  kernels/
    orchestration/*.cpp
    aic/*.cpp                    # optional
    aiv/*.cpp                    # optional
```

Cases are discovered by pytest via `test_*.py` files. Each test module ends with `if __name__ == "__main__": SceneTestCase.run_module(__name__)` so it can also run standalone as `python test_*.py -p <platform>`.

## Selection Scheme

A single `--platform` flag controls hardware/non-hardware splitting across all three categories.

### ut-py (pytest marker)

```python
@pytest.mark.requires_hardware                  # any hardware
class TestRuntimeBuilderIntegration:
    ...

@pytest.mark.requires_hardware("a2a3")          # a2a3 specifically
class TestA2A3Feature:
    ...
```

Selection:

```bash
# No hardware (no-hw tests run, requires_hardware tests skip)
pytest tests/ut

# Hardware (no-hw tests skip, hw + platform-specific tests run)
pytest tests/ut --platform a2a3
```

### ut-cpp (ctest label)

```cmake
# any hardware
set_tests_properties(test_runtime_integration PROPERTIES LABELS "requires_hardware")
# a2a3-specific
set_tests_properties(test_a2a3_feature PROPERTIES LABELS "requires_hardware_a2a3")
```

Selection uses `-LE` (label exclude) on no-hw runner and `-L` (label include) on device runners:

```bash
ctest -LE requires_hardware                 # no-hardware runner: only unlabeled
ctest -L "^requires_hardware(_a2a3)?$"      # a2a3 runner: hw + a2a3-specific
ctest -L "^requires_hardware(_a5)?$"        # a5 runner: hw + a5-specific
```

### st (`@scene_test`)

```python
@scene_test(level=2, platforms=["a2a3sim", "a2a3"], runtime="tensormap_and_ringbuffer")
class TestVectorExample(SceneTestCase):
    ...
```

| `--platform` | Behavior |
| ------------ | -------- |
| `a2a3sim` | Run if `"a2a3sim"` in `platforms` |
| `a2a3` | Run if `"a2a3"` in `platforms` |
| *(none)* | Auto-parametrize over all `*sim` entries in `platforms` |

No `--platform` means "run all sims" — tests with no sim in their `platforms` list are skipped. No additional markers are used.

## Platform notes

- **macOS libomp collision**: on macOS, the root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before `import pytest` to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one Python process (see [troubleshooting/macos-libomp-collision.md](troubleshooting/macos-libomp-collision.md)). Standalone `python test_*.py` bypasses conftest — rely on the env var being exported by the shell or `tools/verify_packaging.sh`.
- **sim hangs / `rc=-1` under CPU oversubscription**: on a few-vCPU runner, high `--max-parallel` (or many concurrent sim cases) oversubscribes the host CPUs, where sim's busy-spin handshake can livelock (hang → `rc=124`) or the deinit timeout can false-trip (`simpler_run failed with code -1`). Mitigate with `--max-parallel 2`; onboard is unaffected (see [troubleshooting/sim-oversubscription-hang.md](troubleshooting/sim-oversubscription-hang.md)).
- **local runs time out more slowly than CI**: compiled defaults are lenient for serving workloads, while CI sets tighter `SIMPLER_*_TIMEOUT_*` env values to fail fast. Use the same env values locally when debugging suspected hangs (see [troubleshooting/local-timeout-defaults.md](troubleshooting/local-timeout-defaults.md)).
- **`st-onboard-a2a3` mass 507899 is not OOM**: a whole-suite collapse of `507899`/`507018`/`register_callable -1` is an AICPU device-fault cascade (`simpler_aicpu_exec` exception), not memory exhaustion. Diagnosis recipe and the per-device preinstall-name fix are in [troubleshooting/a2a3-507899-aicpu-shared-so-fault.md](troubleshooting/a2a3-507899-aicpu-shared-so-fault.md).
