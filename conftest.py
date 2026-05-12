# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Root conftest — CLI options, markers, ST platform filtering, runtime isolation, and ST fixtures.

Runtime isolation: CANN's AICPU framework caches the user .so per device context.
Switching runtimes on the same device within one process causes hangs. When multiple
runtimes are collected and --runtime is not specified, pytest_runtestloop spawns a
subprocess per runtime so each gets a clean CANN context. See docs/testing.md.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import subprocess
import sys
import time
import typing

# Make simpler's V0..V9 and NUL acceptable to pytest's `--log-level` validator.
# pytest does `int(getattr(logging, level.upper(), level))`, so the value must
# exist as a module attribute on `logging` (not just registered via
# `addLevelName`). Set both — the addLevelName side gives nice formatter output
# (`%(levelname)s` shows `V3` instead of `Level 18`); the setattr side is what
# pytest's CLI parser actually consumes.
for _v in range(10):
    logging.addLevelName(15 + _v, f"V{_v}")
    setattr(logging, f"V{_v}", 15 + _v)
logging.addLevelName(60, "NUL")
setattr(logging, "NUL", 60)
# `pytest --log-level null` upcases to "NULL" before the getattr lookup, so
# expose both spellings.
setattr(logging, "NULL", 60)

# macOS libomp collision workaround — must run before any import that may
# transitively load numpy or torch (i.e. before pytest collects scene test
# goldens). See docs/troubleshooting/macos-libomp-collision.md.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

from simpler_setup import parallel_scheduler as _ps  # noqa: E402
from simpler_setup.log_config import configure_logging  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.scene_test import clear_compile_cache  # noqa: E402

# Exit code used when the session watchdog fires. Matches the GNU `timeout`
# convention so shell wrappers (e.g. CI) can distinguish timeout from other
# failures.
TIMEOUT_EXIT_CODE = 124


def _parse_device_range(s: str) -> list[int]:
    """Parse a --device spec into a sorted list of ints.

    Delegates to :func:`simpler_setup.parallel_scheduler.device_range_to_list`
    so both conftest and standalone share the same parser (supports ``0``,
    ``0-7``, ``0,2,5``, and mixed ``0,2-4,7``).
    """
    return _ps.device_range_to_list(s)


class DevicePool:
    """Device allocator for pytest fixtures.

    Manages a fixed set of device IDs. Tests allocate IDs before use
    and release them after. Works identically for sim and onboard.
    """

    def __init__(self, device_ids: list[int]):
        self._available = list(device_ids)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._available):
            return []
        allocated = self._available[:n]
        self._available = self._available[n:]
        return allocated

    def release(self, ids: list[int]) -> None:
        self._available.extend(ids)


_device_pool: DevicePool | None = None


def pytest_addoption(parser):
    """Register CLI options."""
    parser.addoption("--platform", action="store", default=None, help="Target platform (e.g., a2a3sim, a2a3)")
    parser.addoption("--device", action="store", default="0", help="Device ID or range (e.g., 0, 4-7)")
    parser.addoption(
        "--case",
        action="append",
        default=None,
        help="Case selector; repeatable. Forms: 'Foo' (any class), 'ClassA::Foo', 'ClassA::' (whole class).",
    )
    parser.addoption(
        "--manual",
        action="store",
        choices=["exclude", "include", "only"],
        default="exclude",
        help="Manual case handling: exclude (default), include, only",
    )
    parser.addoption("--runtime", action="store", default=None, help="Only run tests for this runtime")
    parser.addoption(
        "--level",
        action="store",
        type=int,
        default=None,
        choices=[2, 3],
        help="Only run tests for this SceneTestCase level (2 or 3); default: all levels",
    )
    parser.addoption(
        "--max-parallel",
        action="store",
        default="auto",
        help=(
            "Max in-flight subprocesses (make-style); decouples the device pool size "
            "from parallelism. 'auto' = min(nproc, len(--device)) on sim, "
            "len(--device) on hardware. Use '--max-parallel 2' to throttle sim on a "
            "CPU-constrained CI runner without shrinking --device. pytest reserves "
            "lowercase short options for itself, so no '-j' short is registered — "
            "use the long form in both pytest and standalone."
        ),
    )
    parser.addoption("--rounds", type=int, default=1, help="Run each case N times (default: 1)")
    parser.addoption(
        "--skip-golden", action="store_true", default=False, help="Skip golden comparison (benchmark mode)"
    )
    parser.addoption(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Enable perf swimlane collection (first round only)",
    )
    parser.addoption("--dump-tensor", action="store_true", default=False, help="Dump per-task tensor I/O at runtime")
    parser.addoption(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="Enable dep_gen capture (SubmitTrace ring, first round only)",
    )
    parser.addoption(
        "--enable-pmu",
        nargs="?",
        const=2,
        default=0,
        type=int,
        metavar="EVENT_TYPE",
        help="Enable PMU collection. Bare flag = PIPE_UTILIZATION(2). "
        "Pass event type to override (e.g. --enable-pmu 4)",
    )
    parser.addoption("--build", action="store_true", default=False, help="Compile runtime from source")
    parser.addoption(
        "--pto-isa-commit",
        action="store",
        default=None,
        help="Pin pto-isa clone to this commit before running tests",
    )
    parser.addoption(
        "--clone-protocol",
        action="store",
        default="ssh",
        choices=["ssh", "https"],
        help="Protocol for cloning pto-isa when --pto-isa-commit is set",
    )
    parser.addoption(
        "--require-pto-isa",
        action="store_true",
        default=False,
        help="Abort the session immediately if PTO-ISA can't be resolved/cloned, "
        "instead of deferring to the per-test lazy path. CI scene-test jobs pass "
        "this so a transient clone failure fails fast rather than fanning out into "
        "device subprocesses that each re-clone into a poisoned directory.",
    )
    # Distinct from pytest-timeout's per-test --timeout (which `.[test]` pulls
    # in on the a2a3 hardware runner); this is session-level.
    parser.addoption(
        "--pto-session-timeout",
        action="store",
        type=int,
        default=0,
        help=(f"Abort whole pytest session after N seconds (0 = disabled; exit code {TIMEOUT_EXIT_CODE} on timeout)"),
    )


def _install_session_timeout(timeout_s: int) -> None:
    # Module-level `_ps` import is intentional (rather than a function-local
    # one): doing `from simpler_setup import parallel_scheduler` inside a
    # signal handler can deadlock on the import lock if the module hasn't
    # been imported yet. Hoisting it to the top guarantees the handler only
    # touches an already-loaded module.
    def _handler(signum, frame):
        print(
            f"\n{'=' * 40}\n[pytest] TIMEOUT: session exceeded {timeout_s}s ({timeout_s // 60}min) limit\n{'=' * 40}",
            flush=True,
        )

        # If the dispatcher is mid-flight, surface every stuck child:
        # 1. SIGUSR1 each pid so its faulthandler dumps all-thread tracebacks
        #    (Python + C frames) into its own stdout — pumped into output_lines.
        # 2. Briefly let pumps drain those bytes before we tear everything down.
        # 3. Print each in-flight job's tail buffer in a HUNG group so the log
        #    contains the actual cause, not just the timeout banner.
        # 4. SIGTERM/SIGKILL the children so they don't outlive us as orphans
        #    holding NPU device state.
        state = _ps._active_state
        if state is not None and state.running:
            for p in list(state.running):
                try:
                    if hasattr(signal, "SIGUSR1"):
                        p.send_signal(signal.SIGUSR1)
                except (ProcessLookupError, OSError):
                    pass

            time.sleep(2.0)

            now = time.monotonic()
            for p, rj in list(state.running.items()):
                elapsed = now - rj.start_time
                tail = "".join(rj.output_lines[-200:])
                print(
                    f"::group::HUNG {rj.job.label} pid={p.pid} devices={rj.device_ids} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if tail:
                    print(tail, end="" if tail.endswith("\n") else "\n", flush=True)
                print("::endgroup::", flush=True)
                print(
                    f"*** HUNG: {rj.job.label} (devices={rj.device_ids}) — expand group above ***",
                    flush=True,
                )

            try:
                _ps._terminate_all(state)
            except Exception:  # noqa: BLE001
                pass

        os._exit(TIMEOUT_EXIT_CODE)

    # signal.alarm / SIGALRM are Unix-only; skip silently on platforms without
    # them so --pto-session-timeout is a no-op rather than a crash (e.g. Windows).
    if hasattr(signal, "alarm") and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)


def _install_child_faulthandler() -> None:
    """In dispatched child pytest processes, let SIGUSR1 dump all-thread stacks.

    The parent dispatcher's session-timeout handler sends SIGUSR1 to every
    in-flight child before tearing the run down. ``faulthandler.register``
    runs in the C signal handler, so it works even when the main thread is
    blocked inside a native call that doesn't release the GIL (NPU runtime,
    nanobind into C++) — exactly the case Python-level watchdogs miss.

    Always-on ``faulthandler.enable()`` also gives us a stack on real crashes
    (SIGSEGV/SIGABRT) instead of a silent exit.
    """
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, chain=False, all_threads=True)
        except (ValueError, RuntimeError):
            # Fails when stdout/stderr can't be duped (rare in child subprocs);
            # leave faulthandler.enable() in place and continue.
            pass


def pytest_configure(config):
    """Register custom markers and apply global config."""
    config.addinivalue_line("markers", "platforms(list): supported platforms for standalone ST functions")
    config.addinivalue_line("markers", "requires_hardware: test needs Ascend toolchain and real device")
    config.addinivalue_line("markers", "device_count(n): number of NPU devices needed")
    config.addinivalue_line(
        "markers",
        "runtime(name): runtime this standalone test targets; used by runtime-isolation subprocess "
        "filtering so non-@scene_test tests only run under their matching runtime",
    )

    log_level = config.getoption("--log-level", default=None)
    if log_level:
        configure_logging(log_level)

    commit = config.getoption("--pto-isa-commit")
    clone_protocol = config.getoption("--clone-protocol")
    # Pre-clone / refresh PTO-ISA up front so that (a) the requested
    # --clone-protocol is honored before SceneTestCase's lazy default-ssh
    # resolve, and (b) the local clone is fetched to origin/HEAD so a
    # --pto-isa-commit request doesn't miss a recently-published commit.
    # Short-circuits when $PTO_ISA_ROOT already points to a user-managed clone.
    #
    # Pre-clone is an optimization, not a requirement: jobs that don't actually
    # need PTO-ISA (e.g. pytest tests/ut on a runner without SSH keys) must not
    # be aborted when the eager clone fails. If an actual scene test later needs
    # PTO-ISA, scene_test.py's lazy path will re-raise the original error.
    #
    # --require-pto-isa flips that: callers that know PTO-ISA is mandatory
    # (CI scene-test jobs) want the session to die here rather than fan out
    # into device subprocesses that each re-attempt the clone.
    try:
        root = ensure_pto_isa_root(
            verbose=True,
            commit=commit,
            clone_protocol=clone_protocol,
            update_if_exists=True,
        )
    except OSError as e:
        if config.getoption("--require-pto-isa"):
            pytest.exit(f"PTO-ISA required but unavailable: {e}", returncode=pytest.ExitCode.USAGE_ERROR)
        print(f"[pytest] PTO-ISA pre-clone skipped: {e}", file=sys.stderr)
        root = None
    if root:
        os.environ["PTO_ISA_ROOT"] = root

    timeout = config.getoption("--pto-session-timeout")
    if timeout and timeout > 0:
        _install_session_timeout(timeout)

    # Always register SIGUSR1 → faulthandler. In dispatched child pytest
    # processes this is what the parent's session-timeout handler relies on
    # to extract a stack from a hung run. In the parent dispatcher itself
    # it's harmless and lets a developer query "what is this process doing?"
    # interactively with `kill -USR1 <pid>`.
    _install_child_faulthandler()

    # xdist worker: bind this process to a single device id from the --device range.
    # The dispatcher (or the user) supplies --device 0-7; xdist spawns N workers
    # labelled gw0..gwN-1. We slice device_ids[worker_index] so each worker owns
    # exactly one device. L2 Worker is session-scoped inside xdist children, so
    # all tests on this worker share one ChipWorker init().
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and worker_id.startswith("gw"):
        try:
            idx = int(worker_id[2:])
        except ValueError:
            idx = 0
        device_spec = config.getoption("--device", default="0")
        ids = _parse_device_range(device_spec)
        if 0 <= idx < len(ids):
            config.option.device = str(ids[idx])

    # Profiling + parallelism is safe: each test case sets its own per-task
    # `output_prefix` on CallConfig (see scene_test.py::_build_config), so
    # diagnostic artifacts land in distinct directories with no shared
    # filenames or rename dance.


def pytest_collection_modifyitems(session, config, items):  # noqa: PLR0912
    """Filter ST tests by --platform / --runtime / --level; order L3 before L2.

    Static filter mismatches (wrong level, wrong runtime, wrong platform)
    are **deselected** rather than marked ``pytest.skip`` so they don't
    inflate the "N skipped" count in each subprocess's terminal summary —
    the L2 subprocess alone re-collects ~50 items per runtime, and the
    skipped variant produced one SKIPPED line per item under ``-v``.
    Deselection goes through ``config.hook.pytest_deselected`` (the same
    path pytest's ``-k`` / ``-m`` use), which reports "M deselected"
    instead of per-item output.

    User-actionable problems (``--platform required``) stay as real skips
    so the reason still surfaces in the default pytest summary.
    """
    platform = config.getoption("--platform")
    runtime_filter = config.getoption("--runtime")
    level_filter = config.getoption("--level")

    keep: list = []
    deselected: list = []

    for item in items:
        # Pre-existing skip markers (e.g. explicit ``@pytest.mark.skip``)
        # stay put — the user asked for a visible skip, not a silent drop.
        if any(m.name == "skip" for m in item.iter_markers()):
            keep.append(item)
            continue

        cls = getattr(item, "cls", None)

        # Under --level, non-SceneTestCase items don't participate in
        # level-based dispatch at all. Resource phase collects them
        # separately in the parent; in a level-filtered child they're
        # simply not this phase's concern.
        if level_filter is not None and cls is None:
            deselected.append(item)
            continue

        if cls is not None and hasattr(cls, "CASES") and isinstance(cls.CASES, list):
            # SceneTestCase class item.
            if not platform:
                # User error: surface it as a real skip so the reason is visible.
                item.add_marker(pytest.mark.skip(reason="--platform required"))
                keep.append(item)
                continue
            if not any(platform in c.get("platforms", []) for c in cls.CASES):
                deselected.append(item)
                continue
            if runtime_filter and getattr(cls, "_st_runtime", None) != runtime_filter:
                deselected.append(item)
                continue
            if level_filter is not None and getattr(cls, "_st_level", None) != level_filter:
                deselected.append(item)
                continue
            keep.append(item)
            continue

        # Non-class pytest function (standalone resource tests and such).
        platforms_marker = item.get_closest_marker("platforms")
        if platforms_marker:
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
                keep.append(item)
                continue
            if platform not in platforms_marker.args[0]:
                deselected.append(item)
                continue

        # runtime-isolation filter for non-@scene_test tests: if the item
        # declares ``@pytest.mark.runtime("X")`` and a --runtime filter is
        # active, deselect when they don't match. Prevents
        # test_explicit_fatal_reports and friends from running under every
        # runtime's subprocess.
        runtime_marker = item.get_closest_marker("runtime")
        if runtime_marker and runtime_marker.args and runtime_filter and runtime_marker.args[0] != runtime_filter:
            deselected.append(item)
            continue

        keep.append(item)

    if deselected:
        items[:] = keep
        config.hook.pytest_deselected(items=deselected)

    # Sort: L3 tests first (they fork child processes that inherit main process CANN state,
    # so they must run before L2 tests pollute the CANN context).
    def sort_key(item):
        cls = getattr(item, "cls", None)
        level = getattr(cls, "_st_level", 0) if cls else 0
        return (0 if level >= 3 else 1, item.nodeid)

    items.sort(key=sort_key)

    # L3 perf collection is not supported yet: a single L3 case forks N chip-processes
    # that all write l2_perf_records_<ts>.json to the same directory with
    # second-precision timestamps, so they trample each other. Block the
    # combination up front; waiting for a proper device-id-in-filename fix.
    if config.getoption("--enable-l2-swimlane", default=False):
        l3_items = [
            i
            for i in items
            if getattr(getattr(i, "cls", None), "_st_level", None) == 3
            and not any(m.name == "skip" for m in i.iter_markers())
        ]
        if l3_items:
            sample = ", ".join(sorted({i.nodeid for i in l3_items})[:3])
            more = "" if len(l3_items) <= 3 else f" (+{len(l3_items) - 3} more)"
            raise pytest.UsageError(
                f"--enable-l2-swimlane is not supported for L3 tests yet — "
                f"multi-chip-process filename collision unresolved. "
                f"L3 items in this session: {sample}{more}. "
                f"Either drop --enable-l2-swimlane or scope to L2 with --level 2."
            )


# ---------------------------------------------------------------------------
# Test dispatcher: Resource phase (device-aware parallel subprocesses for L3
# classes *and* standalone resource-marked functions) + L2 phase (per-runtime
# subprocess). Activated only when neither --runtime nor --level is set by
# the caller. Dispatcher-spawned children set both, so they fall through to
# pytest's default runtestloop without recursing.
# ---------------------------------------------------------------------------


class _ResourceJob(typing.NamedTuple):
    """One device-allocating subprocess job fed into Resource phase.

    ``kind`` drives the ``--level 3`` filter added to the child command (for
    L3 classes). The dispatch itself (bin-pack over ``--device`` pool,
    ``run_jobs`` scheduling, fail-fast semantics) is identical.
    """

    kind: str  # "l3" or "standalone"
    nodeid: str
    label: str  # class name for "l3", function name for "standalone"
    runtime: str
    device_count: int


def _collect_st_runtimes(items, level=None):
    """Return sorted list of unique runtimes from items, optionally filtered by level."""
    runtimes = set()
    for item in items:
        cls = getattr(item, "cls", None)
        if not cls:
            continue
        rt = getattr(cls, "_st_runtime", None)
        lvl = getattr(cls, "_st_level", None)
        if rt and (level is None or lvl == level):
            runtimes.add(rt)
    return sorted(runtimes)


def _collect_resource_jobs(items, platform):
    """Collect every item that needs a dedicated device-allocating subprocess.

    Two job kinds share one phase:

      - ``l3``:         one per L3 ``SceneTestCase`` class.
        ``device_count`` is the max across the class's platform-matching
        non-manual cases.
      - ``standalone``: one per non-class pytest function that declares its
        resource needs via ``@pytest.mark.device_count(n)`` +
        ``@pytest.mark.runtime("...")`` (and optional
        ``@pytest.mark.platforms([...])``).

    Both are dispatched through the same ``parallel_scheduler.run_jobs``
    bin-pack, so merging them reduces the dispatcher to a single phase in
    front of L2.
    """
    jobs: list[_ResourceJob] = []

    # L3 SceneTestCase classes (one job per class, keyed on nodeid).
    l3_by_nodeid: dict[str, _ResourceJob] = {}
    for item in items:
        if any(m.name == "skip" for m in item.iter_markers()):
            continue
        cls = getattr(item, "cls", None)
        if not cls or getattr(cls, "_st_level", None) != 3:
            continue
        rt = getattr(cls, "_st_runtime", None)
        if not rt:
            continue
        max_dev = 1
        saw_case = False
        for case in getattr(cls, "CASES", []):
            if platform and platform not in case.get("platforms", []):
                continue
            if case.get("manual"):
                continue
            saw_case = True
            max_dev = max(max_dev, int(case.get("config", {}).get("device_count", 1)))
        if saw_case:
            l3_by_nodeid[item.nodeid] = _ResourceJob(
                kind="l3", nodeid=item.nodeid, label=cls.__name__, runtime=rt, device_count=max_dev
            )
    jobs.extend(l3_by_nodeid.values())

    # Standalone pytest functions with device_count + runtime markers.
    standalone_by_nodeid: dict[str, _ResourceJob] = {}
    for item in items:
        if any(m.name == "skip" for m in item.iter_markers()):
            continue
        if getattr(item, "cls", None) is not None:
            continue
        dev_marker = item.get_closest_marker("device_count")
        if dev_marker is None:
            continue
        rt_marker = item.get_closest_marker("runtime")
        if rt_marker is None or not rt_marker.args:
            continue
        platforms_marker = item.get_closest_marker("platforms")
        if platforms_marker and platform and platform not in platforms_marker.args[0]:
            continue
        dev_count = int(dev_marker.args[0]) if dev_marker.args else 1
        standalone_by_nodeid[item.nodeid] = _ResourceJob(
            kind="standalone",
            nodeid=item.nodeid,
            label=item.name,
            runtime=rt_marker.args[0],
            device_count=dev_count,
        )
    jobs.extend(standalone_by_nodeid.values())

    return jobs


def _base_pytest_argv(session):
    """Inherit the user's original pytest invocation args."""
    base = [sys.executable, "-m", "pytest"]
    for arg in session.config.invocation_params.args:
        base.append(str(arg))
    return base


def _resolve_max_parallel(cfg, platform: str, device_ids: list[int]) -> int:
    """Parse the -j/--max-parallel CLI value; 'auto' → platform-aware default."""
    raw = cfg.getoption("--max-parallel", default="auto")
    if raw in (None, "", "auto"):
        return _ps.default_max_parallel(platform or "", device_ids)
    try:
        val = int(raw)
    except (TypeError, ValueError) as e:
        raise pytest.UsageError(f"--max-parallel must be 'auto' or an integer, got {raw!r}") from e
    if val < 1:
        raise pytest.UsageError(f"--max-parallel must be >= 1, got {val}")
    return val


def _emit_group(header: str, body: str) -> None:
    """Print a GitHub Actions collapsible group around ``body``.

    ``::group::`` / ``::endgroup::`` are workflow commands — Actions
    renders them as a fold, other shells treat them as plain text so
    running pytest locally still reads sensibly.
    """
    print(f"::group::{header}", flush=True)
    if body:
        print(body, end="" if body.endswith("\n") else "\n", flush=True)
    print("::endgroup::", flush=True)


def _dispatch_test_phases(session, resource_specs):  # noqa: PLR0912
    """Run Resource → L2 phases.

    The Resource phase dispatches every item that needs a dedicated
    device-allocating subprocess — L3 ``SceneTestCase`` classes *and*
    standalone functions marked with ``@pytest.mark.device_count`` +
    ``@pytest.mark.runtime``. They share the same ``run_jobs`` bin-pack
    and fail-fast gate, so they are one phase, not two.

    ``resource_specs`` is pre-collected by ``pytest_runtestloop`` (which
    already has to inspect the list to decide whether to dispatch) so
    this function does not walk ``session.items`` a second time.
    """
    cfg = session.config
    device_spec = cfg.getoption("--device", default="0")
    device_ids = _parse_device_range(device_spec)
    # pytest registers -x as an alias of --exitfirst; both resolve via this name.
    fail_fast = bool(cfg.getoption("--exitfirst", default=False))
    platform = cfg.getoption("--platform")
    max_parallel = _resolve_max_parallel(cfg, platform or "", device_ids)

    base_args = _base_pytest_argv(session)
    cwd = session.config.invocation_params.dir

    # ----- Phase 1: Resource (L3 classes + standalone resource functions) -----
    resource_failed = False
    if resource_specs:
        jobs = []
        for spec in resource_specs:
            label = f"{spec.kind} {spec.label} (rt={spec.runtime}, dev={spec.device_count})"

            def _build(ids, _nodeid=spec.nodeid, _rt=spec.runtime, _kind=spec.kind):
                # Narrow the child to the specific nodeid, not the inherited
                # directory args (examples tests/st). Passing the directories
                # would re-collect every SceneTestCase and run them alongside
                # this job in the same subprocess, which has only this job's
                # allocated devices — e.g. TestL3Group (needs 2) would fail
                # inside TestL3ChildMemory's 1-device subprocess.
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    _nodeid,
                    "--runtime",
                    _rt,
                    "--device",
                    _ps.format_device_range(ids),
                ]
                if _kind == "l3":
                    # L3 jobs run inside --level 3 child mode; standalone jobs
                    # are non-SceneTestCase functions and do not participate
                    # in level-based dispatch.
                    cmd.extend(["--level", "3"])
                if platform:
                    cmd.extend(["--platform", platform])
                return cmd

            jobs.append(
                _ps.Job(
                    label=label,
                    device_count=spec.device_count,
                    build_cmd=_build,
                    cwd=str(cwd),
                )
            )

        def _on_done(res):
            tag = "PASS" if res.returncode == 0 else f"FAIL rc={res.returncode}"
            header = f"{res.label} [{tag} {res.duration_s:.1f}s, devices={res.device_ids}]"
            _emit_group(header, res.output)
            if res.returncode != 0:
                # Out-of-group summary so a reviewer scanning the collapsed
                # log still sees the failure without having to expand.
                print(
                    f"*** FAIL: {res.label} (devices={res.device_ids}) — expand group above ***",
                    flush=True,
                )

        print(
            f"\nResource phase: {len(jobs)} case(s), pool={device_ids}, max_parallel={max_parallel}",
            flush=True,
        )
        try:
            results = _ps.run_jobs(
                jobs,
                device_ids,
                max_parallel=max_parallel,
                fail_fast=fail_fast,
                on_job_done=_on_done,
            )
        except ValueError as e:
            print(f"\n*** Resource phase ABORTED: {e} ***\n", flush=True)
            session.testsfailed = 1
            return True
        resource_failed = any(r.returncode != 0 for r in results)
        if any(r.returncode == TIMEOUT_EXIT_CODE for r in results):
            print("\n*** Resource phase: TIMED OUT ***\n", flush=True)
            os._exit(TIMEOUT_EXIT_CODE)

        # Fail-fast: stop before L2 phase if any Resource job failed.
        if resource_failed and fail_fast:
            session.testsfailed = 1
            return True

    # ----- Phase 2: L2 per-runtime subprocess -----
    l2_runtimes = _collect_st_runtimes(session.items, level=2)
    l2_failed = False
    # When we have more than one device, enable pytest-xdist so the L2 phase
    # spreads classes across devices. Each xdist worker slices --device 0-7
    # down to one id in its own pytest_configure (above) and the st_worker
    # fixture is session-scoped inside the worker — one ChipWorker per (runtime,
    # device), reused across every class assigned to that worker.
    xdist_available = False
    if max_parallel > 1:
        try:
            import xdist  # noqa: F401,PLC0415

            xdist_available = True
        except ImportError:
            print(
                "\n[warning] -j > 1 but pytest-xdist not installed; "
                "falling back to serial L2 phase. pip install pytest-xdist to enable.\n",
                flush=True,
            )
    for rt in l2_runtimes:
        cmd = base_args + ["--runtime", rt, "--level", "2"]
        if xdist_available:
            cmd += ["-n", str(max_parallel), "--dist", "loadfile"]
        # L2 subprocesses run serially (one runtime at a time) so we don't
        # need to buffer their stdout — we can stream it directly through
        # the group markers. ``::group::`` on its own line before the run
        # opens the fold; ``::endgroup::`` after closes it.
        label = f"L2 {rt}" + (f" [-n {max_parallel}]" if xdist_available else "")
        start = time.monotonic()
        print(f"::group::{label}", flush=True)
        result = subprocess.run(cmd, check=False, cwd=cwd)
        duration = time.monotonic() - start
        tag = "PASS" if result.returncode == 0 else f"FAIL rc={result.returncode}"
        print(f"--- L2 {rt}: {tag} {duration:.1f}s ---", flush=True)
        print("::endgroup::", flush=True)

        if result.returncode == TIMEOUT_EXIT_CODE:
            print(f"*** L2 {rt}: TIMED OUT ***", flush=True)
            os._exit(TIMEOUT_EXIT_CODE)
        if result.returncode != 0:
            l2_failed = True
            print(f"*** FAIL: L2 {rt} — expand group above ***", flush=True)
            if fail_fast:
                break

    session.testsfailed = 1 if (resource_failed or l2_failed) else 0
    if not (resource_failed or l2_failed):
        session.testscollected = sum(1 for _ in session.items)
    return True  # returning True prevents default runtestloop


def pytest_runtestloop(session):
    """Dispatch Resource + L2 phases unless caller is already in child mode.

    Child mode (both --runtime and --level set, or --collect-only) skips the
    dispatcher and falls through to pytest's default runtestloop.
    """
    runtime_filter = session.config.getoption("--runtime")
    level_filter = session.config.getoption("--level")

    # Child mode: if the caller filters by runtime or level, it wants direct
    # control — don't re-enter the multi-phase dispatcher (which would cause
    # nested dispatch, device pool exhaustion, and timeout).
    if runtime_filter is not None or level_filter is not None:
        return

    # User explicitly asked for collect-only / scoped-run — don't orchestrate.
    if session.config.getoption("--collect-only", default=False):
        return

    # If there are no items, nothing to orchestrate.
    if not session.items:
        return

    # If only L2 items exist in a single runtime and no resource-dispatched
    # jobs (L3 classes or standalone resource functions) are collected, the
    # dispatcher would reduce to a single L2 subprocess — not worth the
    # fork overhead vs. letting pytest run directly. Skip dispatching in
    # that trivial case. Collect the specs once and hand them to the
    # dispatcher to avoid walking ``session.items`` twice.
    level_filter_explicit = level_filter is not None
    platform = session.config.getoption("--platform")
    runtimes_all = _collect_st_runtimes(session.items)
    resource_specs = _collect_resource_jobs(session.items, platform)
    if not resource_specs and len(runtimes_all) <= 1 and not level_filter_explicit:
        return

    return _dispatch_test_phases(session, resource_specs)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Drop session-lifetime nanobind references before interpreter shutdown.

    ``simpler_setup.scene_test._compile_cache`` accumulates one
    ``ChipCallable`` per ``SceneTestCase`` compiled during the run. At
    interpreter exit the order in which Python clears module globals
    versus the nanobind module destructor is undefined, which on macOS
    surfaces as ``nanobind: leaked N instances of type
    _task_interface.ChipCallable`` on stderr. Clearing the cache here
    (session scope ends after every fixture teardown, including the L2
    worker pool) lets those instances die while nanobind is still
    available.
    """
    clear_compile_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device_pool(request):
    """Session-scoped device pool parsed from --device."""
    global _device_pool  # noqa: PLW0603
    if _device_pool is None:
        raw = request.config.getoption("--device")
        _device_pool = DevicePool(_parse_device_range(raw))
    return _device_pool


@pytest.fixture(scope="session")
def st_platform(request):
    """Platform from --platform CLI flag."""
    p = request.config.getoption("--platform")
    if not p:
        pytest.skip("--platform required for ST tests")
    return p


@pytest.fixture(scope="session")
def _l2_worker_pool(request, st_platform):
    """Session-scoped L2 worker pool keyed by (runtime, device_id).

    Under xdist, each worker process owns one device (slicing done in
    pytest_configure), so this pool typically ends up with one entry per
    runtime. Tests on the same worker that share a runtime reuse the same
    ``ChipWorker`` — amortizing the init cost (three dlopens + device
    acquire) over every class on that device.
    """
    pool: dict[tuple[str, int], object] = {}
    yield pool
    # Session teardown: close every Worker we minted.
    for w in pool.values():
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass
    pool.clear()


@pytest.fixture()
def st_worker(request, st_platform, device_pool, _l2_worker_pool):
    """Per-test Worker.

    L2: session-scoped, reused across classes with the same (runtime, device).
    L3: per-test (registers sub-callables at init, can't be reused).
    """
    cls = request.node.cls
    if cls is None or not hasattr(cls, "_st_level"):
        pytest.skip("st_worker requires SceneTestCase")

    level = cls._st_level
    runtime = cls._st_runtime
    build = request.config.getoption("--build", default=False)

    if level == 2:
        # L2 share: reuse any Worker already created for this runtime in the
        # current process. Under xdist, each worker process is sliced to a
        # single device so there's at most one matching entry. On first call
        # we allocate a device from the pool and immediately release it back —
        # the pool is a process-scoped counter for other fixtures (e.g.
        # st_device_ids) that also draw from it; retaining the id would drain
        # the pool and break any non-st_worker test that runs afterward on the
        # same xdist worker.
        for (rt, dev_id), existing in _l2_worker_pool.items():
            if rt == runtime:
                yield existing
                return

        ids = device_pool.allocate(1)
        if not ids:
            pytest.fail(f"no devices available in --device pool (requested 1, pool has {len(device_pool._available)})")
        dev_id = ids[0]
        device_pool.release(ids)
        key = (runtime, dev_id)
        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=2, device_id=dev_id, platform=st_platform, runtime=runtime, build=build)
        w._st_device_id = dev_id
        w.init()
        _l2_worker_pool[key] = w
        yield w
        # No close here — pool handles teardown at session end.

    elif level == 3:
        max_devices = max((c.get("config", {}).get("device_count", 1) for c in cls.CASES), default=1)
        max_subs = max((c.get("config", {}).get("num_sub_workers", 0) for c in cls.CASES), default=0)
        ids = device_pool.allocate(max_devices)
        if not ids:
            pytest.fail(
                f"need {max_devices} devices but --device pool has {len(device_pool._available)}; widen --device range"
            )

        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(
            level=3,
            device_ids=ids,
            num_sub_workers=max_subs,
            platform=st_platform,
            runtime=runtime,
            build=build,
        )
        w._st_device_id = ids[0]  # expose primary device to test_run for profiling snapshots

        # Register SubCallable entries from cls.CALLABLE
        sub_ids = {}
        chip_cids = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                cid = w.register(entry["callable"])
                sub_ids[entry["name"]] = cid
            elif "orchestration" in entry:
                from simpler_setup.scene_test import _compile_chip_callable_from_spec  # noqa: PLC0415

                name = entry["name"]
                cache_key = (cls.__qualname__, name, st_platform, runtime)
                chip = _compile_chip_callable_from_spec(entry, st_platform, runtime, cache_key)
                cid = w.register(chip)
                chip_cids[name] = cid
                chip_cids[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        cls._st_sub_ids = sub_ids
        cls._st_chip_cids = chip_cids

        w.init()
        yield w
        w.close()
        device_pool.release(ids)


@pytest.fixture()
def st_device_ids(request, device_pool):
    """Allocate device IDs. Use @pytest.mark.device_count(n) to request multiple."""
    marker = request.node.get_closest_marker("device_count")
    n = marker.args[0] if marker else 1
    ids = device_pool.allocate(n)
    if not ids:
        pytest.fail(f"need {n} devices")
    yield ids
    device_pool.release(ids)
