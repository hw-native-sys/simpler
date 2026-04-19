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

import os
import signal
import subprocess
import sys

# macOS libomp collision workaround — must run before any import that may
# transitively load numpy or torch (i.e. before pytest collects scene test
# goldens). See docs/macos-libomp-collision.md.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

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
    from simpler_setup.parallel_scheduler import device_range_to_list  # noqa: PLC0415

    return device_range_to_list(s)


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
        "--enable-profiling", action="store_true", default=False, help="Enable profiling (first round only)"
    )
    parser.addoption("--dump-tensor", action="store_true", default=False, help="Dump per-task tensor I/O at runtime")
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
    def _handler(signum, frame):
        print(
            f"\n{'=' * 40}\n"
            f"[pytest] TIMEOUT: session exceeded {timeout_s}s "
            f"({timeout_s // 60}min) limit, aborting\n"
            f"{'=' * 40}",
            flush=True,
        )
        os._exit(TIMEOUT_EXIT_CODE)

    # signal.alarm / SIGALRM are Unix-only; skip silently on platforms without
    # them so --pto-session-timeout is a no-op rather than a crash (e.g. Windows).
    if hasattr(signal, "alarm") and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_s)


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
        os.environ["PTO_LOG_LEVEL"] = log_level

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
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    try:
        root = ensure_pto_isa_root(
            verbose=True,
            commit=commit,
            clone_protocol=clone_protocol,
            update_if_exists=True,
        )
    except OSError as e:
        print(f"[pytest] PTO-ISA pre-clone skipped: {e}", file=sys.stderr)
        root = None
    if root:
        os.environ["PTO_ISA_ROOT"] = root

    timeout = config.getoption("--pto-session-timeout")
    if timeout and timeout > 0:
        _install_session_timeout(timeout)

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
        # Each xdist worker gets its own perf output dir so parallel profiling
        # runs don't fight over the same perf_swimlane_*.json filename (the
        # runtime's timestamp is second-precision). Anchor to config.rootpath
        # so the C++ runtime (which resolves the path against its own CWD) and
        # Python post-processing always point at the same filesystem location
        # regardless of where pytest was invoked. Only set if the parent
        # hasn't already scoped us into a subprocess dir.
        if "SIMPLER_PERF_OUTPUT_DIR" not in os.environ:
            os.environ["SIMPLER_PERF_OUTPUT_DIR"] = str(config.rootpath / "outputs" / f"perf_{worker_id}")
        # else: more xdist workers than devices — fall through with original range;
        # DevicePool will fail clearly if the test tries to allocate.

    # Note: profiling + parallelism used to be blocked here because perf files
    # shared a process-global directory. The test dispatcher now scopes each
    # subprocess to its own SIMPLER_PERF_OUTPUT_DIR (see _dispatch_test_phases and
    # the xdist slicing above) and flatten_perf_subdirs reassembles outputs/
    # at the end, so the combination is now safe.


def pytest_collection_modifyitems(session, config, items):
    """Skip ST tests based on --platform, --runtime, --level filters; order L3 before L2."""
    platform = config.getoption("--platform")
    runtime_filter = config.getoption("--runtime")
    level_filter = config.getoption("--level")

    # Orchestrator L3 children set PTO_TARGET_NODEID to the single case they
    # were dispatched for. Pytest's --case filter runs inside test_run (too
    # late — other classes' st_worker fixtures already fired at setup). Skip
    # everything except the target nodeid so the child stays narrow even when
    # the parent invocation had broad positional args like ``examples tests/st``.
    target_nodeid = os.environ.get("PTO_TARGET_NODEID")
    if target_nodeid:
        for item in items:
            if item.nodeid != target_nodeid:
                item.add_marker(pytest.mark.skip(reason=f"dispatcher target is {target_nodeid}"))

    # Sort: L3 tests first (they fork child processes that inherit main process CANN state,
    # so they must run before L2 tests pollute the CANN context).
    def sort_key(item):
        cls = getattr(item, "cls", None)
        level = getattr(cls, "_st_level", 0) if cls else 0
        return (0 if level >= 3 else 1, item.nodeid)

    items.sort(key=sort_key)

    for item in items:
        cls = getattr(item, "cls", None)
        if cls and hasattr(cls, "CASES") and isinstance(cls.CASES, list):
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif not any(platform in c.get("platforms", []) for c in cls.CASES):
                item.add_marker(pytest.mark.skip(reason=f"No cases for {platform}"))
            elif runtime_filter and getattr(cls, "_st_runtime", None) != runtime_filter:
                item.add_marker(
                    pytest.mark.skip(reason=f"Runtime {getattr(cls, '_st_runtime', '?')} != {runtime_filter}")
                )
            elif level_filter is not None and getattr(cls, "_st_level", None) != level_filter:
                item.add_marker(pytest.mark.skip(reason=f"Level {getattr(cls, '_st_level', '?')} != {level_filter}"))
            continue
        platforms_marker = item.get_closest_marker("platforms")
        if platforms_marker:
            if not platform:
                item.add_marker(pytest.mark.skip(reason="--platform required"))
            elif platform not in platforms_marker.args[0]:
                item.add_marker(pytest.mark.skip(reason=f"Not supported on {platform}"))

        # runtime-isolation filter for non-@scene_test tests: if the item declares
        # `@pytest.mark.runtime("X")` and a --runtime filter is active, skip when
        # they don't match. Prevents test_explicit_fatal_reports and friends from
        # running under every runtime's subprocess.
        runtime_marker = item.get_closest_marker("runtime")
        if runtime_marker and runtime_marker.args and runtime_filter and runtime_marker.args[0] != runtime_filter:
            item.add_marker(pytest.mark.skip(reason=f"Runtime {runtime_marker.args[0]} != {runtime_filter}"))

    # L3 profiling is not supported yet: a single L3 case forks N chip-processes
    # that all write perf_swimlane_<ts>.json to the same directory with
    # second-precision timestamps, so they trample each other. Block the
    # combination up front; waiting for a proper device-id-in-filename fix.
    if config.getoption("--enable-profiling", default=False):
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
                f"--enable-profiling is not supported for L3 tests yet — "
                f"multi-chip-process filename collision unresolved. "
                f"L3 items in this session: {sample}{more}. "
                f"Either drop --enable-profiling or scope to L2 with --level 2."
            )


# ---------------------------------------------------------------------------
# Test dispatcher: L3 phase (device-aware parallel subprocesses) + L2 phase
# (per-runtime subprocess). Activated only when neither --runtime nor --level
# is set by the caller. Dispatcher-spawned children set both, so they fall
# through to pytest's default runtestloop without recursing.
# ---------------------------------------------------------------------------


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


def _collect_l3_cases(items, platform):
    """Collect one job per L3 class (not per case).

    Returns a list of tuples ``(nodeid, cls_name, runtime, max_device_count)``
    where ``max_device_count`` is the maximum ``device_count`` across the
    class's matching cases. Per-class dispatch matches the ``st_worker``
    fixture's contract (it allocates ``max(CASES.device_count)`` for the whole
    class) — dispatching per-case with a smaller device budget would trip the
    fixture whenever the class also has a case that needs more devices.

    Cases within a class still run in the child process via the existing
    ``test_run`` case loop, reusing the Worker (layer-4 reuse).
    """
    by_nodeid: dict[str, tuple[str, str, int]] = {}
    for item in items:
        cls = getattr(item, "cls", None)
        if not cls or getattr(cls, "_st_level", None) != 3:
            continue
        if any(m.name == "skip" for m in item.iter_markers()):
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
                continue  # --manual exclude is the default; children honor the flag
            saw_case = True
            max_dev = max(max_dev, int(case.get("config", {}).get("device_count", 1)))
        if saw_case:
            by_nodeid[item.nodeid] = (cls.__name__, rt, max_dev)
    return [(nodeid, cls_name, rt, dev) for nodeid, (cls_name, rt, dev) in by_nodeid.items()]


def _base_pytest_argv(session):
    """Inherit the user's original pytest invocation args."""
    base = [sys.executable, "-m", "pytest"]
    for arg in session.config.invocation_params.args:
        base.append(str(arg))
    return base


def _resolve_max_parallel(cfg, platform: str, device_ids: list[int]) -> int:
    """Parse the -j/--max-parallel CLI value; 'auto' → platform-aware default."""
    from simpler_setup.parallel_scheduler import default_max_parallel  # noqa: PLC0415

    raw = cfg.getoption("--max-parallel", default="auto")
    if raw in (None, "", "auto"):
        return default_max_parallel(platform or "", device_ids)
    try:
        val = int(raw)
    except (TypeError, ValueError) as e:
        raise pytest.UsageError(f"--max-parallel must be 'auto' or an integer, got {raw!r}") from e
    if val < 1:
        raise pytest.UsageError(f"--max-parallel must be >= 1, got {val}")
    return val


def _dispatch_test_phases(session):
    """Run L3 phase (device-parallel) then L2 phase (per-runtime subprocess)."""
    from simpler_setup import parallel_scheduler as _ps  # noqa: PLC0415

    cfg = session.config
    device_spec = cfg.getoption("--device", default="0")
    device_ids = _parse_device_range(device_spec)
    # pytest registers -x as an alias of --exitfirst; both resolve via this name.
    fail_fast = bool(cfg.getoption("--exitfirst", default=False))
    platform = cfg.getoption("--platform")
    max_parallel = _resolve_max_parallel(cfg, platform or "", device_ids)

    base_args = _base_pytest_argv(session)
    cwd = session.config.invocation_params.dir

    # ----- Phase 1: L3 classes (device-bin-packed subprocesses, one per class) -----
    l3_cases = _collect_l3_cases(session.items, platform)
    l3_failed = False
    if l3_cases:
        # Static check happens inside run_jobs; we translate errors into session failure.
        jobs = []
        for nodeid, cls_name, rt, dev_count in l3_cases:
            label = f"L3 {cls_name} (rt={rt}, dev={dev_count})"

            def _build(ids, _nodeid=nodeid, _rt=rt):
                return base_args + [
                    _nodeid,
                    "--runtime",
                    _rt,
                    "--level",
                    "3",
                    "--device",
                    _ps.format_device_range(ids),
                ]

            # PTO_TARGET_NODEID makes the child skip every item except this
            # nodeid — defends against inherited positional args (``examples``,
            # ``tests/st``) collecting unrelated classes whose fixtures would
            # then fire at setup and fail on the narrower child device pool.
            # SIMPLER_PERF_OUTPUT_DIR scopes this L3 case's perf files to its own
            # subdir so concurrent L3 cases can't collide on filename (the
            # runtime's timestamp is second-precision). Anchor to cfg.rootpath
            # so the C++ runtime and Python post-processing agree regardless
            # of the child's CWD. Use a nodeid-derived sanitized label so the
            # dir name stays readable for post-mortem.
            safe_nodeid = nodeid.replace("/", "_").replace(":", "_").replace(".", "_")
            child_env = {
                **os.environ,
                "PTO_TARGET_NODEID": nodeid,
                "SIMPLER_PERF_OUTPUT_DIR": str(cfg.rootpath / "outputs" / f"perf_l3_{safe_nodeid}"),
            }
            jobs.append(_ps.Job(label=label, device_count=dev_count, build_cmd=_build, cwd=str(cwd), env=child_env))

        def _on_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"\n--- {res.label}: {tag} on devices {res.device_ids} ---\n", flush=True)

        print(
            f"\n{'=' * 60}\n  L3 phase: {len(jobs)} case(s), "
            f"pool={device_ids}, max_parallel={max_parallel}\n{'=' * 60}\n",
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
            print(f"\n*** L3 phase ABORTED: {e} ***\n", flush=True)
            session.testsfailed = 1
            return True
        l3_failed = any(r.returncode != 0 for r in results)
        if any(r.returncode == TIMEOUT_EXIT_CODE for r in results):
            print("\n*** L3 phase: TIMED OUT ***\n", flush=True)
            os._exit(TIMEOUT_EXIT_CODE)

        # Fail-fast: stop before L2 phase if any L3 failed.
        if l3_failed and fail_fast:
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
        print(
            f"\n{'=' * 60}\n  L2 Runtime: {rt}"
            + (f" [-n {max_parallel}]" if xdist_available else "")
            + f"\n{'=' * 60}\n",
            flush=True,
        )
        result = subprocess.run(cmd, check=False, cwd=cwd)
        if result.returncode == TIMEOUT_EXIT_CODE:
            print(f"\n*** L2 runtime {rt}: TIMED OUT ***\n", flush=True)
            os._exit(TIMEOUT_EXIT_CODE)
        if result.returncode != 0:
            l2_failed = True
            print(f"\n*** L2 runtime {rt}: FAILED ***\n", flush=True)
            if fail_fast:
                break
        else:
            print(f"\n--- L2 runtime {rt}: PASSED ---\n", flush=True)

    # Flatten per-subprocess outputs/perf_*/ subdirs back to outputs/ so
    # downstream tools (swimlane_converter.py, CI artifact upload) find
    # everything in the historical location. Anchor to config.rootpath (not
    # invocation_params.dir) so a user running pytest from a subdirectory
    # still flushes files into the project's top-level outputs/.
    _ps.flatten_perf_subdirs(cfg.rootpath / "outputs")

    session.testsfailed = 1 if (l3_failed or l2_failed) else 0
    if not (l3_failed or l2_failed):
        session.testscollected = sum(1 for _ in session.items)
    return True  # returning True prevents default runtestloop


def pytest_runtestloop(session):
    """Dispatch L3+L2 phases unless caller is already in child mode.

    Child mode (both --runtime and --level set, or --collect-only) skips the
    dispatcher and falls through to pytest's default runtestloop.
    """
    runtime_filter = session.config.getoption("--runtime")
    level_filter = session.config.getoption("--level")

    # Child mode: the dispatcher's spawned subprocesses carry both flags.
    if runtime_filter is not None and level_filter is not None:
        return

    # User explicitly asked for collect-only / scoped-run — don't orchestrate.
    if session.config.getoption("--collect-only", default=False):
        return

    # If there are no items, nothing to orchestrate.
    if not session.items:
        return

    # If only L2 items exist in a single runtime, the dispatcher reduces to a
    # single L2 subprocess — not worth the extra fork overhead vs. letting
    # pytest run directly. Skip dispatching in that trivial case.
    level_filter_explicit = level_filter is not None
    runtimes_all = _collect_st_runtimes(session.items)
    has_l3 = any(getattr(getattr(i, "cls", None), "_st_level", None) == 3 for i in session.items)
    if not has_l3 and len(runtimes_all) <= 1 and not level_filter_explicit:
        return

    return _dispatch_test_phases(session)


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
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                cid = w.register(entry["callable"])
                sub_ids[entry["name"]] = cid
        cls._st_sub_ids = sub_ids

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
