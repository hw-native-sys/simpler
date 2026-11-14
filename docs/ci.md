# CI Pipeline

## Overview

The CI pipeline maps test categories (st, ut-py, ut-cpp) × hardware tiers to GitHub Actions jobs. See [testing.md](testing.md) for full test organization and hardware classification.

Design principles:

1. **Merge by runner, not by language** — Python and applicable C++ unit tests share setup cost and run as steps within a single job per runner tier. A runner with no platform-specific C++ tests does not carry an empty CTest step; currently `ut-a5` is Python-only.
2. **Runner matches hardware tier** — no-hardware PR jobs run on `[self-hosted, cpu]`, with Ubuntu/macOS coverage in Daily; platform-specific tests run on self-hosted runners with the matching label (`a2a3`, `a5`).
3. **`--platform` is the only filter** — pytest uses `--platform` + the `requires_hardware` marker; ctest uses label `-LE` exclusion. No `-m st`, no `-m "not requires_hardware"`.
4. **sim = no hardware** — `a2a3sim`/`a5sim`/`wsesim` PR jobs run on the self-hosted CPU pool alongside unit tests.
5. **Skip irrelevant platforms, and irrelevant suites** — `detect-changes` gates `st-sim-*` and `st-onboard-*` by platform, so a PR confined to one of A2/A3, A5 or WSE skips the other platforms' scene-test runs. **UT jobs are still not gated by platform** — unit tests cover shared contracts and the cost of a falsely-skipped regression outweighs the savings. They *are* gated by test category: a diff confined to `tests/st/` or `examples/` cannot break a unit test, and one confined to `tests/ut/` cannot break a scene test, because the two suites execute disjoint trees (`pytest examples tests/st` vs `pytest tests/ut` plus the C++ ctest) and no unit test reads `examples/` or `tests/st/`. Shared test infrastructure — the root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/` — belongs to neither category and runs both.
6. **Non-code PRs automatically run pre-commit and docs, and nothing else** — `detect-changes` sets `non_code_only` when *no* changed file falls outside the `NON_CODE` set. Skipping the UT jobs is riskless there because nothing in that set can change what the code does, and each member already has its own gate: markdownlint inside `pre-commit` reads the markdown, `docs.yml` (unconditional on every PR) builds the site with `--strict`, and `pre-commit` itself is ungated so a `.pre-commit-config.yaml` change is fully exercised by it.

A5 hardware validation for ordinary PRs is requested by a maintainer or admin with `/run-a5`.
A5 simulation, CPU tests and profiling remain automatic under their existing
change gates. Daily retains its A5 jobs. A2/A3 multi-machine Network1 validation
runs only in Daily, including manually dispatched Daily runs.

## Full Job Matrix

The complete test-type × hardware-tier matrix across automatic CI, the A5 comment workflow and Daily. A5 hardware jobs in this table use `/run-a5` for ordinary PR validation.

| Category | CPU (no hardware) | a2a3 runner | a5 runner |
| -------- | ----------------- | ----------- | --------- |
| **ut** | `ut` (py + cpp) | `ut-a2a3` (py + cpp) | `ut-a5` (py) |
| **st** | `st-sim-a2a3`, `st-sim-a5`, `st-sim-wse` | `st-onboard-a2a3`, `st-network1-onboard-a2a3` (Daily only), `st-deepseek-onboard-a2a3`, `st-framework-a2a3` | `st-onboard-a5`, `st-framework-a5` |

## GitHub Actions Jobs

`ci.yml`, `ci-a5-comment.yml` and `daily.yml` are thin topology callers: they own
triggers, `needs:`, gate `if:` expressions, runner/setup inputs, and matrix
shape. The executable job bodies live in reusable workflows:
`_detect-changes.yml`, `_pre-commit.yml`, `_ut-no-hardware.yml`,
`_packaging.yml`, `_profiling-flags-smoke.yml`, `_st-sim-a2a3.yml`,
`_st-sim-a5.yml`, `_st-sim-wse.yml`, `_ut-npu-a2a3.yml`, `_ut-npu-a5.yml`,
`_st-npu-a2a3.yml`, `_st-npu-a5.yml`, `_st-network1.yml`,
`_st-deepseek-a2a3.yml`, and `_st-framework.yml`. The simulation,
ordinary onboard, and NPU unit-test bodies are split by architecture so each
job renders only its own steps. The framework environment shares `_st-framework.yml`;
callers supply matching `platform` and `runs_on` inputs. Shared step scaffolding
that is safe to run after checkout lives in composite actions under `.github/actions/`
(`apt-install`, `cache-pip`, `cache-ccache`, `setup-gcc-15`, `setup-python`, `setup-venv`, `upload-diagnostics`, and the three `network1-*` actions).
Optional onboard, Network1, and framework diagnostic uploads use the local
`upload-diagnostics` composite action. Its definition is fetched from the
workflow's `github.workflow_sha` into `.ci-support` at retention time, so an
older tested PR head need not contain the action. The tested checkout stays
at its admitted revision. Both helper checkout and upload retain
`continue-on-error: true`; upload runs only when helper checkout succeeds.
The runner loads the remote uploader when the final retention step runs,
rather than during job initialization.
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
attempt. This source-file refresh treats transient download errors as failed
attempts so they use the retry budget. It disables list cleanup, preserving the runner's
Ubuntu indexes for the following install. A failed refresh is a warning; the
required package install, bounded at four minutes and retried once for this
PPA workload, decides whether the job can continue. An install from cached
indexes that fails receives one bounded system-refresh retry. The acquire
timeouts remain a per-connection safeguard, not a wall-clock guarantee.

```text
PullRequest
  ├── pre-commit             (self-hosted CPU)
  ├── packaging-matrix       (self-hosted CPU, Python ≥3.10)          — [needs !examples_only && !tests_only]
  ├── profiling-flags-smoke  (self-hosted CPU)         — (a2a3_changed || a5_changed) && !examples_only && !tests_only
  ├── ut                     (self-hosted CPU, Python ≥3.10)          — Python + C++ UT, no hardware [needs ut_affected]
  ├── detect-changes         (reusable; self-hosted CPU in ci.yml) — outputs non_code_only, {a2a3,a5,wse}_changed, {st,ut}_affected, examples_only, tests_only
  ├── st-sim-a2a3            (self-hosted CPU)          — a2a3_changed && st_affected
  ├── st-sim-a5              (self-hosted CPU)          — a5_changed && st_affected
  ├── st-sim-wse             (self-hosted CPU)          — wse_changed && st_affected
  ├── ut-a2a3                (a2a3 self-hosted)      — Python + C++ UT, a2a3 hardware [needs ut_affected]
  ├── st-onboard-a2a3        (a2a3 self-hosted)      — a2a3_changed && st_affected
  ├── st-framework-a2a3 (a2a3 self-hosted)     — a2a3_changed && st_affected
  └── st-deepseek-onboard-a2a3    (a2a3 self-hosted)      — a2a3_changed && st_affected

Admin PR comment /run-a5
  └── gate-and-resolve       (ubuntu-24.04)          — open PR to this repo main; pinned merge
      ├── ut-a5             (a5 self-hosted)        — Python UT and CANN examples
      ├── st-onboard-a5     (a5 self-hosted)        — onboard ST and DFX smokes
      └── st-framework-a5 (a5 self-hosted)     — framework eager ST
```

| Job | Runner | What it runs |
| --- | ------ | ------------ |
| `ut` | self-hosted CPU, Python ≥3.10 | `pytest tests/ut` + `ctest -LE requires_hardware` |
| `st-sim-a2a3` | self-hosted CPU | `pytest examples tests/st --platform a2a3sim` |
| `st-sim-a5` | self-hosted CPU | `pytest examples tests/st --platform a5sim` |
| [`st-sim-wse`](../src/wse/docs/README.md#build-and-test) | self-hosted CPU | `pytest examples tests/st --platform wsesim` |
| `ut-a2a3` | a2a3 self-hosted | Hardware Python UT: `device_count` cases use resource scheduling, RuntimeBuilder integration runs before the xdist fanout, and remaining cases use one worker per allocated device. Then `ctest -L "^requires_hardware(_a2a3)?$" --resource-spec-file ...` and CANN example smokes. |
| `st-onboard-a2a3` | a2a3 self-hosted | `pytest examples tests/st -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4 --device ...`, then separate SDMA-demo and `sdma_fault` device tasks, then adaptive-parallel DFX feature smokes |
| `st-framework-a2a3` | a2a3 self-hosted | `pytest examples tests/st --platform a2a3 --test-environment torch_npu --manual exclude --require-pto-isa --device ... --max-parallel 1 -p no:xdist` in one exclusive task, then verify plan/JUnit |
| `ut-a5` | a5 self-hosted | Hardware Python UT: `device_count` cases use resource scheduling, RuntimeBuilder integration runs before the xdist fanout, and remaining cases use one worker per allocated device. Then CANN example smokes. |
| `st-onboard-a5` | a5 self-hosted | Default SDMA+URMA / RDMA=OFF build and `pytest examples tests/st --platform a5 --exclude-level 4 --device ...`, including SDMA, the active URMA regressions and mixed SDMA/URMA cases, then adaptive-parallel DFX feature smokes |
| `st-framework-a5` | a5 self-hosted | `pytest examples tests/st --platform a5 --test-environment torch_npu --manual exclude --require-pto-isa --device ... --max-parallel 1 -p no:xdist` in one exclusive task, then verify plan/JUnit |
| `st-network1-onboard-a2a3` (Daily only) | a pair of `a2a3pod` machines | `pytest examples tests/st --level 4 --platform a2a3 --device ... --max-parallel 1`, one L3 daemon on the peer |
| `st-deepseek-onboard-a2a3` | a2a3 self-hosted | `pytest <the two deepseek_v4_flash_decode dirs> --platform a2a3 --manual only --device ...`, both 2-device cases concurrently, in parallel with `st-onboard-a2a3` |

### Multi-machine network1 jobs

`st-network1-onboard-a2a3` runs only in `daily.yml`, on its schedule or through
`workflow_dispatch`, and is the only job spanning two machines. The runner it
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
| `network1-run-pytest` | once | submit the peer daemon and local pytest through `task-submit`, then collect logs |
| `network1-teardown` | once, `if: always()` | remove the peer tree after both tasks end and logs are copied |

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

Both hosts require `task-submit`; configured device lists alone do not reserve cards.
The peer daemon starts in its own device task. Once its port opens, the local
pytest starts in another device task. The daemon keeps its allocation while
the local task queues and runs. A port already open before submission is treated
as a stale listener and fails the job.

The first workflow step records this job's start. Queue waiting can use the
remaining 60-minute job, except for 90 seconds reserved for task cleanup and
log collection. Pytest calculates its timeout when its task actually starts,
so waiting in the queue leaves it whatever time remains. Both task bodies
refuse to run if they start after that deadline. The runner records task IDs;
cleanup cancels pending tasks or kills running ones and waits for terminal status.
If terminal status cannot be confirmed, it retains the peer tree for diagnosis.

MPI ranks started over SSH source a run-specific file with their host task's
`TASKQUEUE_TASK_ID` and device environment. The task daemon uses this marker
to clean up ranks that have detached from the task's original process group.
`npu-smi` and process snapshots are captured inside the allocated tasks; raw
work logs and task IDs are kept under `output/network1-ci-<run>-<attempt>/`.

Writing an example — the files, the entry module, the `run(...)` entry point,
the `test_*.py` network1 wrapper, and the manual `run_parent.sh` — is covered in
[`examples/workers/README.md`](../examples/workers/README.md).

### Daily full scene-test sweep

[`daily.yml`](../.github/workflows/daily.yml) runs the full regular + manual
scene-test corpus with `--manual include` once per day and supports manual
re-runs through `workflow_dispatch`. It also runs packaging and no-hardware
unit tests on macOS. A2/A3, A5 and WSE simulation runs on Ubuntu and macOS;
onboard runs on the A2/A3 and A5 self-hosted pools. The
same DFX smoke steps used by Per-PR run once in each Daily platform job, and
the A2/A3 `network1` corpus runs exclusively in Daily through the existing two-machine workflow. The
main Per-PR scene-test steps keep the default `--manual exclude`, so moving an
ordinary case to Daily does not require a second workflow exclusion list.
WSE runs its dependency-capture smoke; its legacy PMU, chip-swimlane,
args-dump and scope-stats collectors are unsupported.
The dedicated A2/A3 and A5 dep-gen, chip-swimlane, PMU, args-dump, and scope-stats steps
instead use
`include` for the normal Per-PR and Daily modes because they own the full
corpus under their target paths; a `manual_mode` of `only` remains `only` in
those steps. Marking a case under one of those paths manual therefore removes
its duplicate main-step execution without removing its dedicated Per-PR
coverage. Scope-stats needs its own step for a reason the others do not:
`_validate_scope_stats_artifact` returns early unless `--enable-scope-stats` is
supplied. Its Sim cases run in that dedicated step, which asserts the
`scope_stats.jsonl` artifact.

Use `"manual": True` on an individual `SceneTestCase.CASES` entry and
`@pytest.mark.manual` on a standalone pytest test. The reusable scene-test
workflows accept `manual_mode`; Per-PR callers use its `exclude` default and the
Daily caller passes `include`.

The Sim host-build-graph invalid-input test keeps every distinct owner, output,
dependency, and block-count error path in Per-PR, with two wide-count boundaries.
Daily adds the remaining wide-count values.
For Sim, the Per-PR sweep keeps onephase Sum and Max Allreduce and the
bidirectional-ring non-Sum rejection check. Daily adds onephase Min/Prod,
twophase and ring non-Sum, and the ibing rejection check; their onboard
executions remain in Per-PR. Storage readiness, local domain tensors, and SPSC
fault checks also remain in the Sim Per-PR sweep. A2/A3 Sim's L3 child-memory
and expert-dispatch examples and deferred-notify case run in Daily; A2/A3
onboard and A5 Sim retain their Per-PR executions.
For A5 Sim, predicated multi-block refusal retains one gate value in Per-PR;
Daily and A5 onboard exercise both values.
The A5 resident-failure PR-sized sweep retains the slot-1 shape for each timeout
mode; Daily also exercises the all-lane shape.
The A5 onboard `/run-a5` sweep uses the bounded PR-sized selection;
Daily covers the remaining mask and failed-lane combinations.
The two-rank onboard non-Sum Allreduce variants remain in Per-PR: each class
takes only a few seconds, and its algorithm path is distinct.

A sweep's budget follows what it selects, so `manual_mode` is not the only
input a Daily caller sets. `_st-npu-a5.yml` takes `session_timeout_seconds`
(default 1200, sizing the Per-PR sweep) and derives the `task-submit`
execution and queue limits from it; the Daily caller raises it, because
`include` selects a corpus the default never sized. `job_timeout_minutes` has
to stay above the queue limit that implies. A sweep that outgrows its budget
does not fail on the case at fault: the session timeout fires on whatever was
still running and reports that one, so read the `HUNG` group rather than the
assertion.

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
pytest examples tests/st -m "not sdma and not sdma_fault" --platform a2a3 --exclude-level 4 --device 4-7 -x
pytest examples tests/st -m "sdma and not sdma_fault" --platform a2a3 --device 4-5 -x
pytest examples tests/st -m sdma_fault --platform a2a3 --device 4 -x

# A5 runners run the non-network1 corpus, including SDMA tests
pytest examples tests/st --platform a5 --exclude-level 4 --device 0-7 -x
```

`-x` (`--exitfirst`) is appropriate for CI, where aborting on first
failure saves runner minutes. Local development usually wants the opposite
(let every failure surface) — just drop the flag. The short form is the
same in both pytest and standalone on purpose; see
[CLI design principles](testing/cli.md#cli-design-principles).

`pytest-xdist` is pulled in via the `test` extra. See
[Parallel test execution](testing/parallel-execution.md#parallel-test-execution-and-resource-reuse)
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
conservative and build the combined target. Self-hosted CPU runs install and
run pre-commit directly in a lightweight venv; only diffs selected for clang-tidy
preparation install the package and prepare its compiler stand-ins. Each
sanitizer matrix cell selects its own platform. The profiling-flags smoke installs only the
`_task_interface` binding because its matrix builds every runtime configuration
itself.

### Sim jobs on CPU-constrained runners

PR simulation jobs (`st-sim-a2a3`, `st-sim-a5`, `st-sim-wse`) share the self-hosted CPU pool.
WSE passes `--max-parallel` explicitly because its width is variant-dependent:
one worker on the 4-vCPU hosted runner Daily uses, sixteen on the self-hosted
pool. The virtual-device pool and automatic parallelism below apply to A2/A3 and A5.
Daily also runs Ubuntu and macOS cells; a standard GitHub-hosted Ubuntu runner has **4 vCPUs**. `--device 0-15` is still the
right choice for the **pool size** (some L3 cases need several virtual ids), but
the default `--max-parallel auto` caps the in-flight subprocess count to
`min(nproc, len(--device))` — on a 4-core runner that becomes `4`. Note
`os.cpu_count()` reports the host's logical CPUs and ignores any cgroup CPU
quota, so this is the true core count, not a container limit.

```bash
# Sim: --max-parallel auto resolves to 4 on a standard ubuntu-24.04 runner
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

- PR simulation and no-hardware unit tests run on the self-hosted CPU pool; Daily retains GitHub-hosted coverage.
- Change detection and pre-commit start with 100 commits of checkout history. They use the admitted PR base SHA when available, fetching only a missing base commit. If the merge-base is outside the shallow history, they deepen the head and base histories in two bounded rounds of 100 and 400 commits, without fetching every branch or tag. Unresolved history makes the detector run the full matrix; pre-commit fails with an explicit error rather than emitting an empty base SHA.
- `detect-changes` is implemented once in [`.github/workflows/_detect-changes.yml`](../.github/workflows/_detect-changes.yml) and computes four axes from the PR diff — non-code (`non_code_only`), architecture (`a2a3_changed` / `a5_changed` / `wse_changed`), test category (`st_affected` / `ut_affected`), and corpus (`examples_only` / `tests_only`) — **all of them derived from one `NON_CODE` set**: `docs/`, `.docs/`, `.claude/`, `mkdocs.yml`, `.github/workflows/docs.yml`, `.gitignore`, `.pre-commit-config.yaml`, and any `*.md` file anywhere. Membership follows a file's *effect*, not its path — `mkdocs.yml` and `docs.yml` are docs tooling that happens to live outside `docs/`. An arch flag is `false` only when every changed file is in the other platforms' trees (`src/{arch}/`, `examples/{arch}/`, `tests/{st,ut/cpp}/{arch}/`) or in `NON_CODE`. Anything else — shared C++ (`src/common/`), Python (`python/`, `simpler_setup/`), build files (`CMakeLists.txt`, `pyproject.toml`), shared test infra (`tests/ut/py/`, `tests/lint/`), tooling (`tools/`), or any CI implementation workflow (`.github/workflows/ci.yml`, `.github/workflows/_*.yml`) — flips all three architecture flags to `true`. CI implementation workflows are deliberately excluded from `NON_CODE`: a change to the gates or reusable job bodies must run the complete automatic matrix. Explicit `/run-a5` hardware suites remain manual.
- **Test-category axis:** `ST_ONLY='^(tests/st/|examples/)'` and `UT_ONLY='^tests/ut/'`, applied in the same shape as the arch patterns — a category is unaffected only when *every* changed file is exclusively the other's. `st_affected` gates automatic scene-test jobs; `ut_affected` gates automatic `ut` and `ut-a2a3`. Explicit `/run-a5` requests run all three A5 hardware suites without change detection. Anything belonging to neither (root `conftest.py`, `pyproject.toml`, `simpler_setup/`, `tests/lint/`) flips both, so shared infrastructure always runs both suites.
- **Corpus axis:** `EXAMPLES_ONLY='^examples/'` and `TESTS_ONLY='^(tests/)'`, same shape again, one per side of the product jobs' payload. `packaging-matrix` and `profiling-flags-smoke` build and install the product and then exercise it with a fixed, tiny payload — one entry-point script (a `tests/st/` file) and one `vector_example` — so neither reads either suite as a corpus, and a diff confined to `examples/` or to `tests/` cannot reach them: `wheel.packages` is `["simpler_setup", "python/simpler"]`, so both partitions are provably absent from the product. A payload file changed under either is still exercised by the scene-test job that reads the same corpus.
- **Gated jobs (scene tests):** automatic `st-sim-{a2a3,a5,wse}` and `st-onboard-a2a3` run iff their platform's flag **and** `st_affected` are `true`. Explicit `/run-a5` requests are not filtered by these gates.
- **Platform-independent jobs (UT + packaging):** automatic `ut`, `ut-a2a3` and `packaging-matrix` ignore the *platform* flags — unit tests exercise shared contracts (nanobind bindings, RuntimeBuilder, ring buffers, etc.) and the risk of silently skipping a regression outweighs the CI minutes saved. The `tests/ut/cpp/{arch}/` entry in the gating regex only *attributes* an arch-specific C++ UT change to that platform (so it does not spuriously flip the other platforms' scene-test flags); it does not gate the UT jobs themselves. The gated UT jobs do respect `ut_affected`, which is a statement about test category rather than silicon. `packaging-matrix` respects the corpus axis, which is a statement about which corpus a job reads: neither `examples/` nor `tests/` is in `wheel.packages`, so a diff confined to either partition cannot change what the packaging job builds.
- **`non_code_only` is the same `NON_CODE` set, not a narrower one.** It is `true` when no changed file falls outside it. Nothing in the set can change what the code does, and no workflow consumes any of it beyond its own gate: `pre-commit` is ungated so it always exercises `.pre-commit-config.yaml`, `docs.yml` is unconditional on every PR so it always exercises `mkdocs.yml` / `docs/`, and **no workflow invokes anything under `.claude/`** (`grep -rn '\.claude' .github/workflows/` finds only comments). An **empty diff short-circuits the whole step**: attribution is impossible, so a single guard sets `non_code_only=false`, every arch and category flag `true`, and both corpus flags (`examples_only`, `tests_only`) `false`, then returns — running the full matrix, packaging and the profiling smoke included. That guard is deliberately one place; testing emptiness per flag is what previously left `non_code_only` false while the arch flags also came out false, running UT and packaging but skipping every scene test.

  The arch flags subtract `NON_CODE` before deciding, so a non-code-only change already makes all three `false`. An arch-gated job therefore needs no separate non-code check. See [`.claude/rules/ci-change-detection.md`](../.claude/rules/ci-change-detection.md) for the invariants these gates must keep.

- **SDMA tests run in isolated tasks inside `st-onboard-a2a3`.** Within the default test environment, three selections partition the corpus (`examples tests/st`): the ordinary sweep takes `-m "not sdma and not sdma_fault"` with `--exclude-level 4`, one queued task runs `-m "sdma and not sdma_fault"` for the demos, and a separate one-device task runs `-m sdma_fault` for the AICore fault-injection case. The three are disjoint and exhaustive over both markers, and all three name the same paths, so a case cannot land in two tasks or in none. Provisioning the SDMA workspace creates 48 device-only STARS streams that live in the device fault domain, and tearing them down once any AICore task has poisoned that card blocks minutes rather than milliseconds — whether the teardown is a device reset or an explicit stream destroy ([#1425](https://github.com/hw-native-sys/simpler/issues/1425)). The cost therefore falls on whichever process still holds those streams when the fault happens, which is why the fault-injection case gets a task containing nothing else. Separate `task-submit` tasks preserve that on every onboard runner, independent of pytest ordering. Network1 tests are selected by `--level 4` in `st-network1-onboard-a2a3` and explicitly excluded from ordinary onboard ST lanes.

### Feature coverage verification

`--environment-receipts --verify-feature-coverage --junitxml <result.xml>`
checks feature declarations and execution evidence. The default onboard and
framework reusable workflows enable these flags; this integration still requires
validation on real CI runners. The scene
corpus declares its features, and collect-only inventories can check the actual
selection without running fixtures. CPU contract tests exercise the real collection hooks,
dispatcher, xdist workers and JUnit verifier.

Workflow triggers, environment setup, device leases and SDMA isolation use the
same contracts across the environment suites. Default A2/A3 writes independent `main`, `sdma`
and `sdma_fault` plans and receipts under `outputs/default-ci`; A5 writes `main`.
Framework results remain under `outputs/framework-ci`. Each job's final
verification step runs with `always()`, rechecks the original JUnit files against
the plan, and writes `coverage-summary.md`, `coverage-summary.json` and the GitHub step summary. It fails
if any expected partition is missing or its scope differs from the invocation.
For A2/A3 manual-only runs, the two SDMA steps are outside scope; Daily retains
their existing default manual exclusion. An empty framework manual-only run is
not successful qualification.

Artifacts retain plans, parent/child receipts, retry manifests and summaries,
including partial evidence on failure. Default artifact names include platform,
host architecture, run ID and attempt. Artifact upload failure does not override
the test verdict. Sim, Network1, standalone model tests and DFX smokes keep their
existing validation paths and are not covered by these new summaries.

A5 onboard runs with a JUnit report also place CANN process and device logs in
an `ascend/` directory beside that report. Child processes inherit this location,
so the existing default-environment artifact retains the native failure details.
An explicit `ASCEND_PROCESS_LOG_PATH` takes precedence.

Program is the default product: a test declares
`runtime_product("kernel"|"auxiliary")` only when it is not Program. Kernel tests
also declare `execution_mode("eager"|"capture_replay")`; Program and auxiliary
groups are mode-free because Program only supports eager. Runtime identity comes
from the existing runtime marker, scene class or parameter instance; conflicting
sources are errors. `SceneTestCase` defaults to Program through
`_st_runtime_product`; its `_st_execution_mode` default applies only when the
product is Kernel. A marker overrides those class defaults for an exceptional
method. Plain Program tests need no product or mode marker.
`coverage_runtime` supplies classification for runtime-independent probes or
legacy tests without a runtime selection marker, preserving `--runtime` filtering.
Filenames do not classify graph capture. Borrowed-context parameter instances in
the stateful eager tests are Kernel; owned-context instances are Program. The
common stream-capture probe is separate from HBG end-to-end capture qualification.
Auxiliary probes do not claim delivered Kernel features or satisfy mandatory
feature groups; they never narrow the corresponding Kernel capture gap.

Coverage groups are `program/<runtime>`, `auxiliary/<runtime>` and
`kernel/<runtime>/eager|capture_replay`. The policy in
`simpler_setup/test_coverage.py` requires Program and Kernel eager
for both runtimes in default onboard sweeps, plus native HBG capture on A2/A3 and A5.
Framework sweeps require Kernel eager for both runtimes and the graph-capture
groups declared in the applicable corpus, including their mandatory members.
The stock Torch-NPU native-capture tests declare Kernel capture/replay separately
from eager argument dumps. A capture group absent from the corpus remains a gap.
Undelivered capture
combinations remain explicit gaps. Manual-only, runtime/level and the existing
SDMA partitions narrow the obligations. Each workflow passes
`--coverage-partition main|sdma|sdma_fault|all` explicitly; the verifier never
infers the partition from the `-m` spelling. Ad-hoc `-k`/`-m` filtering cannot remove
mandatory members from the collected applicable corpus. Deleting a test and its
declaration entirely cannot be detected unless a required group disappears.

An optional default-environment test may declare
`coverage_skip(platforms=["a2a3"], reason="exact skip reason")`. This is a
reviewable exemption for those platforms, not a runtime capability probe.
The JUnit skip reason must match exactly; xfail is not an allowed skip. Exempt
members cannot replace a group's required members. Framework skip policy stays
strict. Summary counts distinguish selected tests, entry into the test body,
passes, failures, skips and missing results. Setup errors count as failures without
body entry; teardown errors fail a case even when its body passed. Setup skips do
not count as executed or passed. Poison-sink retries retain both attempts and count
each recovered node ID once; a failed attempt cannot become a recovered pass.

Failed verification preserves source-checked partial counts in the job summary and
its JSON document, with `verification_status` set to `failed` or `incomplete`.
Missing or invalid case evidence is never a pass. Foreign receipts and unapproved
duplicates fail verification; an invalid source plan has no attributable counts.
The JSON includes environment identity, source SHA, partition scope and evidence
paths; the summary links to the CI run and its artifacts. A diagnostic report
does not change the nonzero verification exit status or recreate a cached green
partition summary.

Collect the real corpus with its usual platform, environment and selection flags:

```bash
python -m pytest examples tests/st --collect-only --platform a2a3 \
  --exclude-level 4 -m "not sdma and not sdma_fault" \
  --environment-receipts --verify-feature-coverage --coverage-partition main \
  -p simpler_setup.tools.collection_inventory --collection-inventory inventory.json
```

The inventory includes exact selected node IDs, duplicate/classification errors,
feature declarations and policy obligations. Compare node IDs before and after
a migration for each existing CI partition. Collection is not execution evidence;
an empty manual-only framework selection remains empty, not a passing qualification.

The scoped skip audit preserves these existing exceptions:

| Existing condition | Coverage treatment |
| ------------------ | ------------------ |
| Early-enqueue classes' unused `test_run` placeholder | Explicit auxiliary test and exact skip reason; actual sequence tests remain required. |
| HBG unpublished-dlopen validation runs only on sim | Exact skip exception on its onboard architecture. |
| A2/A3 HBG graph-execution AICore SO-reload probe runs only on sim | Exact reason on A2/A3; reported as skipped, never qualified. |
| Quarantined A5 chunked URMA fetch | Exact existing quarantine reason on A2/A3 and A5: collection preserves explicit skip markers before platform filtering. It remains skipped and cannot satisfy required execution. |
| Visible-device remapping receives the identity two-device grant `[0, 1]` | Exact reason allowed on A2/A3 and A5; other skips fail verification. |
| A5 RDMA workspace build option disabled | Exact exception for that smoke only. |
| A5 manual CSA frozen fixture absent | Exact exception; reported as skipped, never qualified. |
| Daily PMU counting-window case without `--enable-pmu 1` | Exact A2/A3 skip; the separate DFX smoke enables PMU and verifies actual counting. |
| Sim-only branches, unavailable `/proc/self/maps`, missing matching run-completion case | No new onboard exemption; unexpected skips fail. |
| L4 MPI/Network1 prerequisites missing | Outside these non-L4 sweeps; no blanket exemption for future L4 qualification. |

MXFP cases are generated conditionally from available torch dtypes before pytest
collection. Their absence cannot be detected by this inventory or by runtime-group
coverage alone. Compare inventories under the CI dependency versions before enabling
mandatory verification there. Platform-scoped exemptions permit the specified skip;
they do not independently establish that the capability is absent.

### Framework environment qualification (a2a3 and a5)

The environment suite is named `st-framework-<arch>` and uses `_st-framework.yml`
and `setup-framework/`. Execution modes belong to test declarations and coverage
reports, so future qualified capture cases share this environment and device lease.
The suite includes Kernel eager and native graph capture/replay declarations;
actual qualification requires their source-bound passing execution receipts.

Comment-trigger workflows are loaded from the default branch, whereas their local
actions and identity scripts come from the checked-out PR. The existing
`setup-framework-eager/action.yml` and `identity.py` paths therefore delegate to
the canonical setup/identity implementation and preserve the caller's output path.
They contain no independent dependency pins or execution suite. Remove these entry
points only after the default-branch callers use the canonical paths and queued
invocations of the older callers have drained. An older caller's passing result
qualifies only its own verification contract; candidate coverage summaries require
a caller that enables feature verification.

When deploying this naming migration, administrators must inspect both branch
protection and repository/organization rulesets for required check contexts. Replace
any `st-framework-eager-<arch> / st-framework-eager-<arch>` requirement with the
observed `st-framework-<arch> / st-framework-<arch>` context from the candidate CI
run, preserving the required-check app identity and enforcement settings. These
settings are external to Git and cannot be migrated by a YAML rename. Do not add
an extra suite merely to emit the retired name. Recheck merge requirements before
merging the migration; a public ruleset query does not inspect legacy protection.

`st-framework-a2a3` runs automatically on affected PRs. A5 framework
eager validation for ordinary PRs is part of `/run-a5`. Both run in Daily.
Automatic callers use the corresponding `a2a3_changed && st_affected` gate;
the explicit A5 command runs without a change gate. Each uses its architecture runner pool.
All callers reuse `_st-framework.yml` with an explicit platform. Setup validates `a2a3|a5`, builds `build_package_<platform>`,
and records the matching onboard runtime roots. The exclusive task checks actual
silicon before backend import and passes the same platform to identity recording
and pytest. Artifacts are named separately by platform; a green A3 job does not
qualify A5.
Tests declare `@pytest.mark.test_environment("torch_npu")`. The job discovers
them across `examples tests/st` with `--test-environment torch_npu`; it does
not enumerate directories or cases. Ordinary sweeps and compile warm-up select
the implicit `default` environment, including in Daily. Framework dependencies
live only in `.venv-framework`, separate from the Program/fork sweep's `.venv`.
Environment selection is independent of `manual`. The reusable job accepts the
same `manual_mode` contract as other scene-test jobs: PR callers default
to `exclude`, while Daily passes `include`. The existing
framework cases are automated, but a future manual case in this environment is
discovered and included by Daily without another workflow edit.

The dedicated environment uses `--system-site-packages` and installs local,
pinned wheels for CPU Torch 2.7.1 and torch_npu 2.7.1.post4. The pins constrain
subsequent test/SDK installation too. The profile selects an already installed
toolkit by its actual manifest. Selector and preflight share a minimum CANN
version line of 9.0, comparing numeric major/minor values without an architecture
allowlist. Separated release/build labels are opaque: 9.0.RC1, 9.1.T500,
9.2.0-beta.2, 9.10 and 10.0 meet the minimum. Missing or malformed versions and
version lines below 9.0 are rejected. This is an admission policy, not a claim
that all admitted releases and hardware combinations have been tested.

CANN 9.0.0 on x86_64 has framework CI coverage. CANN 9.2.0-beta.2 passed the
[ARM64 A3 framework experiment](https://github.com/hw-native-sys/simpler/actions/runs/40190729557/job/121165491940)
with these unchanged Torch pins. Those execution results remain the measured
compatibility evidence; A5 and other admitted combinations need their own actual
validation. Adding a platform route is not a claim that its tuple has passed.

An already active toolkit meeting the minimum is preferred. Otherwise setup scans its
existing roots in deterministic order, without using directory names as version
claims. The selected `set_env.sh` is used for installation, compilation and inside
the exclusive task; the system default symlink is not changed. If no installation meeting the minimum
exists, setup fails with the inspected toolkit inventory and required
minimum version line. Preflight checks the actual active manifest against
the same rule. Unknown versions, missing dependencies, and failed imports are
errors rather than skipped qualification. Runner labels alone do not establish
this compatibility. Setup prints Python, host architecture, CANN manifest,
distribution versions and module locations; backend imports happen inside the
exclusive device task with automatic backend loading disabled beforehand.

The ScopeStats consumer adds four queued calls with off/on/on/off selection
and independent output prefixes. TMR checks nested scope records and allocation
frontiers; HBG checks metadata only. Those cases enter the same collection plan
without a workflow case list.

The diagnostic-selection consumer queues four calls on one Worker: diagnostics
off, swimlane level 1 / PMU event 2, level 3 / event 4, then off again. It changes
the original CallConfig selectors and output prefixes after each accepted submit,
then checks the submitted selections, original export destinations and per-run
attribution. PMU and swimlane collection can wait; this does not qualify other
DFX collectors or claim overlapping execution.

One `task-submit` allocation of one device runs the selected consumers serially
for TMR/HBG × `TASK_QUEUE_ENABLE=0/1`. Same-stream and cross-stream consumers
perform three calls; the alias consumer performs three updates for each of
its full-alias and partial-overlap modes. Callback-failure recovery checks three
accepted runs, including the run whose Python callback reports an error, before
init-thread close without individual waits. Public L2 close uses the common
accepted-handle drain before workspace teardown, preserving late close-only
lane errors and retryable cleanup. Typed-value consumers perform three
DEVICE calls, plus three explicit HOST/NONE-control calls on HBG; TMR checks
clean HOST refusal followed by valid DEVICE work. Persistent-state consumers
perform twelve DEVICE cursor/count-controlled weighted updates to one cache,
including zero acceptance and revisited rows. Framework snapshots check the
entire backing against an independent multi-step reference, guards and unchanged
weights. No host control readback decides later kernel work. A session-scoped fixture
compiles the shared host bridge once. Argument-dump contract cases queue four
OFF/FULL/HYBRID/OFF attempts. A2/A3 (both runtimes) and A5 TMR accept all four,
each with two dependent tasks and a runtime-owned intermediate, and verify exact
task-time payloads, materialized geometry, per-call export ownership, Torch
outputs and guards. A5 HBG requires typed unsupported refusals for FULL/HYBRID,
unchanged inputs/outputs and no dump files, plus healthy OFF calls and reuse
after close. Its distinct refusal summary cannot count as verified payloads;
the required per-task capture remains unimplemented under
[#2666](https://github.com/hw-native-sys/simpler/issues/2666). Refusal tests do not
deliver that capability. All combinations
enter the same dynamic plan without a workflow case list. The client wait is bounded at 15 minutes,
including queueing; the execution watchdog is also 15 minutes. Individual
subprocess timeouts do not reserve additional aggregate execution time.

Pre-admission recovery consumers check stale Buffer registration and wrong
caller context refusals. Each refusal is followed by healthy work on the same
Worker; the wrong-context callback restores the caller context before its
healthy submit, and a third healthy request follows. Final stream reuse is
also checked. These are pre-admission checks, not native partial launch or
device-fault recovery.

The final pytest collection freezes a nonempty source/environment/nodeid plan
next to the JUnit file. Every case carries its original nodeid and plan/source
identity into JUnit. One shared verifier requires exactly that selected set to
pass, with no duplicates, foreign cases, skips, errors or failures. A new test
declaring the same environment automatically enters the plan. This verifies
execution of the selected set; the tests themselves declare the qualification
matrix. Strict receipts currently support direct pytest execution only; a
profile selecting resource-dispatched, SceneTestCase or xdist execution fails
explicitly rather than accepting incomplete child results. The default
environment retains its existing dispatcher and does not claim this strict
JUnit aggregation. Logs contain the
source/native build identities, runtime and framework module hashes, compiled
bridge identity and consumer results; when artifact upload succeeds,
`framework-<platform>-*` retains JUnit and per-case logs even on test failure.
Artifact upload is best-effort and does not override the verified test result.
This qualifies the reference
consumers' same-stream and cross-stream eager ordering, caller-owned lifetimes,
bounded aliased INOUT assertions, ordinary callback-error recovery and per-call
value semantics and bounded DEVICE-controlled persistent cache state. It does
not qualify native failed-drain recovery, pending
framework-callback close or early HOST-backing release. It does not establish
overlap, arbitrary strides/cross-tile aliasing, PyPTO dispatch or capture/replay.
Qualification is limited to the actual platform and environment in each receipt.

### A5 hardware validation (`/run-a5`)

A repo maintainer or admin can comment exactly `/run-a5` on an open PR targeting this repo's
`main` branch. Surrounding whitespace is accepted; extra arguments or text are
ignored. `ci-a5-comment.yml` checks the commenter's current `admin` permission
or `maintain` role (returned as `role_name`, with `permission: write`)
with `getCollaboratorPermissionLevel` before resolving the PR's head repository
and base SHA, then pinning their GitHub-generated merge commit in the upstream
repository, including fork PRs. The gate verifies that the
merge commit has exactly the observed base and head as its two parents. Pending
or stale snapshots are re-read for up to 60 seconds; a persistent mismatch or
conflict fails the gate before any suite is scheduled. The gate summary records all three
SHAs. Testing the merge includes current main's CI interfaces even for a PR whose
head predates their introduction. This qualifies integration with that base,
not the standalone head. Non-PR comments and unverified or
requests without maintain or admin access cannot schedule hardware work.

Every admitted request runs `ut-a5` with CANN examples, `st-onboard-a5` with DFX
smokes, and `st-framework-a5` through the existing reusable workflows.
No change detector filters an explicit request, including a documentation-only
PR. A new admitted request cancels the previous same-PR job for each suite;
unmatched and denied comments cannot cancel admitted jobs. Pushes do not
retrigger these suites; comment again to test a new head. The gate refuses any
run attempt after the first, so "Re-run all jobs" cannot admit a head the
comment never saw; "Re-run failed jobs" keeps the original gate outputs and
retests the admitted SHA. The gate and report jobs only call the GitHub API and
run on GitHub-hosted `ubuntu-24.04`, so a burst of comments never queues on the
self-hosted pools.

For a mainline baseline, create a test PR from the latest `main` with only a
documentation change, then request `/run-a5`. Keep its runtime, tests and CI
configuration unchanged until the baseline completes. Use the report's tested
merge and base SHAs when diagnosing failures; a later push needs a new command.

The final A5 PR comment records the tested merge, original head and base, each suite's
result and the workflow run link. Skipped and cancelled jobs retain those results
and are not reported as passing. The issue-comment workflow itself uses the
default branch's definition, so this new command cannot execute on GitHub until
`ci-a5-comment.yml` is merged into the default branch. Local mocked-API tests
validate its scripts and topology before that deployment.

### Self-hosted CPU runner contract

All eight no-hardware jobs in `ci.yml` (`detect-changes`, `pre-commit`, `ut`,
`packaging-matrix`, `profiling-flags-smoke`, `st-sim-a2a3`, `st-sim-a5`, `st-sim-wse`) use
`[self-hosted, cpu]`. The runner label accepts either ARM64 or X64 hosts.
Default installation discovers and builds WSE alongside A2/A3 and A5 when the
Host compilers are available. The WSE scene workflow selects only its runtime
with `--config-settings=build.targets=build_package_wsesim`.
The Python-consuming CPU jobs verify that the provisioned `python3` is at
least Python 3.10 before installing dependencies or running Python tools;
newer interpreters are accepted. GitHub-hosted variants retain Python setup
with the requested version, defaulting to 3.10. Only hosted jobs fetch and
run the local `setup-python` wrapper from the workflow's fixed SHA in
`.ci-support`, so CPU jobs do not prepare its remote setup action and old
tested PR heads need not contain the wrapper. CI static analysis also targets
Python 3.10. Daily retains separate GitHub-hosted Ubuntu/macOS coverage. The
A5 comment gate/report and docs jobs also use GitHub-hosted runners.

Provision Python 3.10 or newer, GCC 15, CMake, Ninja, GoogleTest, Graphviz, clang-tidy 18 and a matching CPU Torch wheel on each CPU runner. The distribution package names depend on the host OS. `setup-venv` installs a PyPI `ninja` into the venv on every lane, which shadows the dnf-provided `ninja-build` on `PATH` whenever the venv is active — deliberately, so CMake's generator selection is identical on CPU and GitHub-hosted runners. Same tool, different provenance; the dnf package stays in the contract for builds that run outside the venv. When the pre-commit selector requests clang-tidy preparation — for eligible C/C++, `.pre-commit-config.yaml`, or an unknown path — the job creates `g++-15` as a stand-in for the Ubuntu Toolchain PPA compiler. On the agents `g++` resolves to a conda GCC 15 prefix rather than `/usr/bin/g++`, so that diff's sim artifacts are built with GCC 15 and `compile_commands.json` names that prefix's `<triple>-g++`; `tests/lint/clang_tidy.py` drops the triple before replaying a command, without which clang-tidy adopts it as a target and resolves no C++ standard library at all. `ci.yml` lints with clang-tidy 18 and HCE 2.0 packages only LLVM 12, so an agent additionally provides 18 on `PATH` as `clang-tidy-18`, installed together with its clang builtin headers — a clang-tidy whose prefix carries no `lib/clang/<major>/include` resolves no resource dir and fails every `#include <stddef.h>`. The pre-commit job shadows the distro `clang-tidy` only on that build path; Python-only and other lint-only diffs do not build sim artifacts or create either compiler shim. On APT-based self-hosted CPU runners, the same bounded APT action used by the GitHub-hosted lane installs missing build/lint tools before verification; an existing `clang-tidy-18` satisfies the lint-tool requirement. Non-APT runners must provide the tools in advance, and verification still fails if any required tool is unavailable.

Pip and compiler caches persist on these hosts; CPU jobs use their on-disk
caches without uploading or downloading them through `actions/cache`.

## Hardware Classification

Three hardware tiers, applied to all test categories. See [hardware classification](testing/environments.md#hardware-classification) for per-category selection mechanisms, and [Test Sources](#test-sources) for folder layouts.

| Tier | CI Runner | Job examples |
| ---- | --------- | ------------ |
| No hardware | `[self-hosted, cpu]` (PR), `ubuntu-24.04` / `macos-latest` (Daily) | `ut`, `st-sim-*` |
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
simulation jobs have no separate warm-up step. PR simulation jobs can reuse
the CPU runner's on-disk caches; Daily GitHub-hosted jobs start with a cold
local kernel cache.

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

This cache complements rather than replaces `ccache`: the packaging and
no-hardware-UT workflows use `ccache` for supported CMake compiler invocations,
while scene-test cache entries also retain serialized callables and final
incore bytes, including the CCEC/linker and ELF-section processing used by
onboard builds.

Entries are content-addressed and therefore never overwritten, so a run prunes
entries whose last use is more than 14 days old before it exits. Without that,
the directory grows by one entry per kernel change forever and the saved
`actions/cache` archive — one new entry per push, restored by prefix — would
crowd the repository's 10 GB cache budget and evict the pip and other caches
the other jobs depend on.

GitHub-hosted jobs restore and save the pip download cache, keyed by runner OS, runner architecture, and
`pyproject.toml`. Source-only changes therefore reuse the same dependency cache
instead of creating another roughly 200 MB archive; pip still resolves and
validates requested versions on every install.

Self-hosted CPU jobs use persistent pip and ccache directories without remote
cache restore or upload. Pre-commit runs directly in the selected Python
environment and reuses its local hook cache; it does not download the
`pre-commit/action` wrapper or that wrapper's remote cache action.

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
[unit-test build coverage](testing/unit-tests.md#how-many-times-a-case-is-built).

The two checks are scoped differently, by what decides their answer. The axis
check reads which configurations each case is built for, which the include
paths this repository's CMakeLists compose decide, so it runs on the Linux leg
only — and Linux is the leg that can establish it, because its build graph is
the superset: a non-Linux configure drops `test_thread_scheduling.cpp`, whose
wrapped `sched_*` symbols are a GNU ld feature. The stub-linkage check runs on
every leg, because which archive member resolves a symbol, and whether a weak
definition wins, is decided by the platform's linker instead.

That job also uses `ccache` through `CMAKE_CXX_COMPILER_LAUNCHER`, with the
directory restored and saved by the `cache-ccache` composite action. The
launcher is passed only when `ccache` is on `PATH`, so a runner whose package
manager could not serve it still builds; the step prints `ccache --show-stats`
after the build, which is the only place a restored-but-unusable cache is
distinguishable from a cold one. Self-hosted runners keep their home directory
between runs, so self-hosted CPU runs use the on-disk cache and skip the
upload.

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
ctest -LE requires_hardware                # no-hardware runner: unlabeled and no_hardware
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
| *(none)* | Skip with `--platform required` |

Select a platform explicitly for each invocation. The runner matches it against
each case's supported platforms; it does not automatically parametrize over
simulators. No additional markers are used.

## Platform notes

- **macOS libomp collision**: on macOS, the root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before `import pytest` to work around a duplicate-libomp abort triggered by homebrew numpy and pip torch coexisting in one Python process (see [troubleshooting/macos-libomp-collision.md](troubleshooting/macos-libomp-collision.md)). Standalone `python test_*.py` bypasses conftest — rely on the env var being exported by the shell or `tools/verify_packaging.sh`.
- **sim hangs / `rc=-1` under CPU oversubscription**: on a few-vCPU runner, high `--max-parallel` (or many concurrent sim cases) oversubscribes the host CPUs, where sim's busy-spin handshake can livelock (hang → `rc=124`) or the deinit timeout can false-trip (`simpler_run failed with code -1`). Mitigate with `--max-parallel 2`; onboard is unaffected (see [troubleshooting/sim-oversubscription-hang.md](troubleshooting/sim-oversubscription-hang.md)).
- **local runs time out more slowly than CI**: compiled defaults are lenient for serving workloads, while CI sets tighter `SIMPLER_*_TIMEOUT_*` env values to fail fast. Use the same env values locally when debugging suspected hangs (see [troubleshooting/local-timeout-defaults.md](troubleshooting/local-timeout-defaults.md)).
- **`st-onboard-a2a3` mass 507899 is not OOM**: a whole-suite collapse of `507899`/`507018`/`register_callable -1` is an AICPU device-fault cascade (`simpler_aicpu_exec` exception), not memory exhaustion. Diagnosis recipe and the per-device preinstall-name fix are in [troubleshooting/a2a3-507899-aicpu-shared-so-fault.md](troubleshooting/a2a3-507899-aicpu-shared-so-fault.md).
