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
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import typing

# Make simpler's TIMING and NUL levels acceptable to pytest's `--log-level` validator.
# pytest does `int(getattr(logging, level.upper(), level))`, so the value must
# exist as a module attribute on `logging` (not just registered via
# `addLevelName`). Set both — the addLevelName side gives nice formatter output
# (`%(levelname)s` shows `TIMING` instead of `Level 25`); the setattr side is what
# pytest's CLI parser actually consumes.
logging.addLevelName(25, "TIMING")
setattr(logging, "TIMING", 25)
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
from simpler_setup.log_config import DEFAULT_LOG_LEVEL, configure_logging  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.scene_test import SceneTestLevel, clear_compile_cache, is_manual_for_platform  # noqa: E402

# Exit code used when the session watchdog fires. Matches the GNU `timeout`
# convention so shell wrappers (e.g. CI) can distinguish timeout from other
# failures.
TIMEOUT_EXIT_CODE = 124
_SCENE_LEVEL_CHOICES = [int(level) for level in SceneTestLevel]


def _parse_device_range(s: str) -> list[int]:
    """Parse a --device spec into a sorted list of ints.

    Delegates to :func:`simpler_setup.parallel_scheduler.device_range_to_list`
    so both conftest and standalone share the same parser (supports ``0``,
    ``0-7``, ``0,2,5``, and mixed ``0,2-4,7``).
    """
    return _ps.device_range_to_list(s)


def _normalize_cli_scene_level(level: int | None) -> SceneTestLevel | None:
    if level is None:
        return None
    return SceneTestLevel(level)


def _item_scene_level(item) -> SceneTestLevel | None:
    cls = getattr(item, "cls", None)
    if cls is not None:
        level = getattr(cls, "_st_level", None)
        if level is not None:
            return SceneTestLevel(level)
    function = getattr(item, "function", None)
    level = getattr(function, "_st_level", None)
    if level is not None:
        return SceneTestLevel(level)
    return None


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


class Network1Peer(typing.NamedTuple):
    endpoint: str
    remote_device_ids: tuple[int, ...]
    session_timeout_s: float
    session_listen_host: str


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
        help="Manual test handling: exclude (default), include, only",
    )
    parser.addoption("--runtime", action="store", default=None, help="Only run tests for this runtime")
    parser.addoption(
        "--level",
        action="store",
        type=int,
        default=None,
        choices=_SCENE_LEVEL_CHOICES,
        help="Only run tests for this scene-test level (2, 3, or 4); default: all levels",
    )
    parser.addoption(
        "--exclude-level",
        action="store",
        type=int,
        default=None,
        choices=_SCENE_LEVEL_CHOICES,
        help="Exclude tests carrying this scene-test level (2, 3, or 4)",
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
        "--enable-chip-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        help="Enable chip swimlane. Bare flag=level 4 (full). "
        "1=AICore timing, 2=+dispatch/fanout, 3=+sched phases, 4=+orch phases",
    )
    parser.addoption(
        "--dump-args",
        nargs="?",
        const=1,
        type=int,
        default=0,
        help="Dump per-task args at runtime. Level: 0=off, 1=partial (only "
        "args selected via Arg::dump(...), default when given without a value), 2=full (all args), "
        "3=hybrid (all tasks' JSON metadata; args marked via Arg::dump(...) also write payload; "
        "used by simpler_setup.tools.core_swimlane for Core swimlane simulator replay).",
    )
    parser.addoption(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="Enable dep_gen capture (disabled when --rounds > 1)",
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
    parser.addoption(
        "--enable-scope-stats",
        action="store_true",
        default=False,
        help="Enable per-scope peak collection and emit <output_prefix>/scope_stats/scope_stats.jsonl "
        "(per-scope ring-fill peaks).",
    )
    parser.addoption(
        "--enable-swimlane-overhead",
        action="store_true",
        default=False,
        help="Add the 8 Overhead Analysis counter tracks (per-engine "
        "idle/ready/overhead + system all/has overhead) to the swimlane JSON. "
        "Requires --enable-chip-swimlane + deps.json (re-run with --enable-dep-gen if absent).",
    )
    parser.addoption(
        "--sanitizer",
        action="store",
        default="none",
        help=(
            "Run against sanitizer-built binaries. Preset (asan/ubsan/tsan) or raw "
            "-fsanitize tokens. Must match the SIMPLER_SANITIZER the runtime was "
            "pip-installed with, and needs the matching runtime preloaded "
            "(e.g. LD_PRELOAD=$(g++ -print-file-name=libasan.so))."
        ),
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


def _collect_descendant_pids(pid: int) -> list[int]:
    """Return all descendant pids of ``pid``, BFS via Linux ``/proc``.

    L3 ``Worker`` forks ChipWorker / SubWorker / next-level children
    (``python/simpler/worker.py::_start_hierarchical``). When a sim test
    deadlocks inside one of those forked grandchildren, sending SIGUSR1 only
    to the dispatched pytest pid is useless — that process is calmly waiting
    in ``waitpid``; the real deadlock site sees no signal. Walking the tree
    via ``/proc/<pid>/task/<tid>/children`` lets the timeout handler hit
    every descendant so faulthandler (which is inherited across ``fork``)
    fires in the one that's actually stuck.

    Returns ``[]`` on platforms without ``/proc`` (macOS) or if the pid is
    already gone. Best-effort: races with grandchild exit are silently
    ignored.
    """
    from collections import deque  # noqa: PLC0415 — local import keeps the signal-handler import surface minimal

    out: list[int] = []
    visited: set[int] = {pid}
    queue: deque[int] = deque([pid])
    while queue:
        cur = queue.popleft()
        try:
            task_dir = f"/proc/{cur}/task"
            tids = os.listdir(task_dir)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for tid in tids:
            try:
                with open(f"{task_dir}/{tid}/children") as f:
                    raw = f.read()
            except (FileNotFoundError, PermissionError):
                continue
            for tok in raw.split():
                try:
                    child = int(tok)
                except ValueError:
                    continue
                if child not in visited:
                    visited.add(child)
                    out.append(child)
                    queue.append(child)
    return out


def _drain_until_quiet(state: object, max_wait_s: float = 10.0) -> None:
    """Wait until the pump's output stops growing (bounded), so a signaled
    process's faulthandler dump lands before the next signal or the SIGTERM.
    A fixed sleep races slow signal delivery: a starved process can take
    longer than a few seconds to run its dump, and the dump then dies with
    the SIGTERM. Called from the session-timeout handler between signals."""
    drain_deadline = time.monotonic() + max_wait_s
    quiet_rounds = 0
    prev_lines = -1
    while time.monotonic() < drain_deadline:
        cur_lines = sum(len(rj.output_lines) for rj in state.running.values())
        if cur_lines == prev_lines:
            quiet_rounds += 1
            if quiet_rounds >= 5:  # ~1 s of no new output
                break
        else:
            quiet_rounds = 0
            prev_lines = cur_lines
        time.sleep(0.2)


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
        # 1. SIGUSR1 each pid AND its descendants so faulthandler (inherited
        #    across fork in L3 Worker's ChipWorker/SubWorker children) dumps
        #    all-thread tracebacks (Python + C frames) into the child's
        #    stdout — pumped into output_lines.
        # 2. Briefly let the pump thread drain those bytes (``join`` with a
        #    short timeout) before reading the tail buffer; otherwise bytes
        #    sit in the OS pipe and are dropped when SIGTERM closes it.
        # 3. Print each in-flight job's tail buffer in a HUNG group so the log
        #    contains the actual cause, not just the timeout banner.
        # 4. SIGTERM/SIGKILL the children so they don't outlive us as orphans
        #    holding NPU device state.
        state = _ps._active_state
        if state is not None and state.running:
            descendants: dict[int, list[int]] = {}
            for p in list(state.running):
                kin = _collect_descendant_pids(p.pid) if hasattr(signal, "SIGUSR1") else []
                descendants[p.pid] = kin
                if not hasattr(signal, "SIGUSR1"):
                    continue
                # Signal the dispatched pytest itself, then every descendant
                # (in BFS order — closer kin first is fine, ordering doesn't
                # affect the dump). Each signal is followed by a drain until
                # the output settles: concurrent faulthandler dumps to the
                # same pipe interleave at the byte level, splitting frame
                # names, so a signaled process must finish dumping before the
                # next one starts.
                for target_pid in (p.pid, *kin):
                    try:
                        os.kill(target_pid, signal.SIGUSR1)
                    except (ProcessLookupError, OSError):
                        pass
                    _drain_until_quiet(state)

            now = time.monotonic()
            for p, rj in list(state.running.items()):
                elapsed = now - rj.start_time
                # ``join`` here only yields the GIL so the pump's pending
                # ``output_lines.append`` lands before we read the list. Short
                # timeout — pump will block on the next ``readline()`` since
                # the child is still alive.
                pump = getattr(rj, "pump_thread", None)
                if pump is not None:
                    pump.join(timeout=0.05)
                tail = "".join(rj.output_lines[-200:])
                kin = descendants.get(p.pid, [])
                kin_str = f" descendants={kin}" if kin else ""
                print(
                    f"::group::HUNG {rj.job.label} pid={p.pid} devices={rj.device_ids} elapsed={elapsed:.1f}s{kin_str}",
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


def _configure_sanitizer(config):
    """Wire the `--sanitizer` option: drive kernel compile + require the preload.

    The runtime `.so` are sanitizer-built at install time
    (`pip install --config-settings=cmake.define.SIMPLER_SANITIZER=...`); this
    only has to (a) compile the per-test kernels/orchestration to match and
    (b) fail early if the runtime isn't preloaded.
    """
    from simpler_setup import sanitizers as san  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    selection = config.getoption("--sanitizer", default="none")
    tokens = san.resolve(selection)
    if not tokens:
        return
    try:
        san.validate(tokens)
    except ValueError as e:
        raise pytest.UsageError(f"--sanitizer={selection}: {e}") from e
    KernelCompiler._sanitizers = tokens

    lib = san.preload_lib(tokens)
    if lib and not san.is_runtime_loaded(lib):
        platform = config.getoption("--platform", default="") or ""
        raise pytest.UsageError(
            f"--sanitizer={selection} needs the {lib} runtime preloaded "
            f"(the instrumented .so are dlopen'd into this Python). Re-run with:\n"
            f"  {san.preload_command(tokens, platform)} pytest --sanitizer {selection} ..."
        )


def _validate_diagnostic_flags(config) -> None:
    # Imported by full path: `simpler_setup.scene_test` as an attribute is the
    # @scene_test decorator, not this module.
    from simpler_setup.scene_test import _validate_diagnostic_flags as validate  # noqa: PLC0415

    try:
        validate(
            chip_swimlane=config.getoption("--enable-chip-swimlane", default=0),
            swimlane_overhead=config.getoption("--enable-swimlane-overhead", default=False),
        )
    except ValueError as e:
        raise pytest.UsageError(str(e)) from e


def _validate_level_filters(config) -> None:
    if config.getoption("--level", default=None) is not None and (
        config.getoption("--exclude-level", default=None) is not None
    ):
        raise pytest.UsageError("--level and --exclude-level cannot be used together")


def pytest_configure(config):
    """Register custom markers and apply global config."""
    config.addinivalue_line("markers", "platforms(list): supported platforms for standalone ST functions")
    config.addinivalue_line("markers", "requires_hardware: test needs Ascend toolchain and real device")
    config.addinivalue_line("markers", "device_count(n): number of NPU devices needed")
    config.addinivalue_line(
        "markers", "manual(platforms=None): test runs only when --manual is include or only on selected platforms"
    )
    config.addinivalue_line(
        "markers",
        "network1_remote_device_count(n): number of remote NPU devices needed on the peer machine",
    )
    config.addinivalue_line(
        "markers",
        "sdma: the test exercises PTO-ISA async SDMA. SceneTestCase pytest "
        "fixtures and standalone runners build its Worker with enable_sdma=True "
        "unless worker_workspace=False selects a platform-provisioned "
        "communication-domain workspace. "
        "Worker-global provisioning creates 48 "
        "device-only STARS streams that sit in the device fault domain, which "
        "makes a later AICore fault on that device cost minutes instead of "
        "~0.3 s (#1425). Two consequences follow from the one marker: such a "
        "test never shares an L2 Worker (the pool key carries the flag) and it "
        "sorts after every ordinary test, so fault-injection cases run on a "
        "device that has never provisioned. The a2a3 CI additionally runs them "
        "in a step of their own via -m sdma until #1425 is fixed",
    )
    config.addinivalue_line(
        "markers",
        "resource_last: run this test in the final Resource subphase after L2",
    )
    config.addinivalue_line(
        "markers",
        "runtime(name): runtime this standalone test targets; used by runtime-isolation subprocess "
        "filtering so non-@scene_test tests only run under their matching runtime",
    )

    _validate_level_filters(config)
    _validate_diagnostic_flags(config)
    _configure_sanitizer(config)

    # Configure logging unconditionally (not only when --log-level is passed) so
    # simpler's own WARNINGs — e.g. the device-log-timing "no device log written"
    # diagnostic — reach stderr by default under pytest, matching the standalone
    # CLI path. Without this the root logger has no handler, so pytest's log
    # capture swallows the message and a passing run shows nothing. An explicit
    # --log-level still overrides the default threshold.
    log_level = config.getoption("--log-level", default=None)
    configure_logging(log_level or DEFAULT_LOG_LEVEL)

    # Pre-clone / refresh PTO-ISA up front so scene-test children inherit the
    # pinned managed checkout resolved from pto_isa.pin.
    # Pre-clone is an optimization, not a requirement: jobs that don't actually
    # need PTO-ISA (e.g. pytest tests/ut on a runner without SSH keys) must not
    # be aborted when the eager clone fails. If an actual scene test later needs
    # PTO-ISA, scene_test.py's lazy path will re-raise the original error.
    #
    # --require-pto-isa flips that: callers that know PTO-ISA is mandatory
    # (CI scene-test jobs) want the session to die here rather than fan out
    # into device subprocesses that each re-attempt the clone.
    try:
        # Eager clone only — do not export PTO_ISA_ROOT into the ambient env
        # (#1403). Downstream host builds receive the path via -DPTO_ISA_ROOT=.
        ensure_pto_isa_root(verbose=True)
    except OSError as e:
        if config.getoption("--require-pto-isa"):
            pytest.exit(f"PTO-ISA required but unavailable: {e}", returncode=pytest.ExitCode.USAGE_ERROR)
        print(f"[pytest] PTO-ISA pre-clone skipped: {e}", file=sys.stderr)

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


def _manual_mode_matches(is_manual: bool, manual_mode: str) -> bool:
    return manual_mode == "include" or is_manual == (manual_mode == "only")


def _manual_marker_applies(marker, platform: str | None) -> bool:
    if marker is None:
        return False

    unknown_kwargs = set(marker.kwargs) - {"platforms"}
    if unknown_kwargs:
        names = ", ".join(sorted(unknown_kwargs))
        raise pytest.UsageError(f"@pytest.mark.manual got unsupported keyword argument(s): {names}")
    if marker.args and "platforms" in marker.kwargs:
        raise pytest.UsageError(
            "@pytest.mark.manual platforms must be passed either positionally or by keyword, not both"
        )

    if "platforms" in marker.kwargs:
        platforms = marker.kwargs["platforms"]
    elif marker.args:
        platforms = marker.args[0] if len(marker.args) == 1 else marker.args
    else:
        return True
    if platforms is None or platform is None:
        return True
    return is_manual_for_platform(platforms, platform)


def pytest_collection_modifyitems(session, config, items):  # noqa: PLR0912
    """Filter ST tests by --platform / --runtime / level axes; order L3 before L2.

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
    level_filter = _normalize_cli_scene_level(config.getoption("--level"))
    exclude_level_filter = _normalize_cli_scene_level(config.getoption("--exclude-level"))
    manual_mode = config.getoption("--manual", default="exclude")

    keep: list = []
    deselected: list = []

    for item in items:
        # Pre-existing skip markers (e.g. explicit ``@pytest.mark.skip``)
        # stay put — the user asked for a visible skip, not a silent drop.
        if any(m.name == "skip" for m in item.iter_markers()):
            keep.append(item)
            continue

        cls = getattr(item, "cls", None)

        item_level = _item_scene_level(item)

        if cls is not None and hasattr(cls, "CASES") and isinstance(cls.CASES, list):
            # SceneTestCase class item.
            if not platform:
                # User error: surface it as a real skip so the reason is visible.
                item.add_marker(pytest.mark.skip(reason="--platform required"))
                keep.append(item)
                continue
            if not any(
                platform in case.get("platforms", [])
                and _manual_mode_matches(is_manual_for_platform(case.get("manual"), platform), manual_mode)
                for case in cls.CASES
            ):
                deselected.append(item)
                continue
            if runtime_filter and getattr(cls, "_st_runtime", None) != runtime_filter:
                deselected.append(item)
                continue
            if level_filter is not None and item_level != level_filter:
                deselected.append(item)
                continue
            if exclude_level_filter is not None and item_level == exclude_level_filter:
                deselected.append(item)
                continue
            keep.append(item)
            continue

        # Standalone pytest test (resource functions and ordinary tests).
        is_manual = _manual_marker_applies(item.get_closest_marker("manual"), platform)
        if not _manual_mode_matches(is_manual, manual_mode):
            deselected.append(item)
            continue

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

        if level_filter is not None and item_level != level_filter:
            deselected.append(item)
            continue
        if exclude_level_filter is not None and item_level == exclude_level_filter:
            deselected.append(item)
            continue

        keep.append(item)

    if deselected:
        items[:] = keep
        config.hook.pytest_deselected(items=deselected)

    # Sort: L3 tests first (they fork child processes that inherit main process CANN state,
    # so they must run before L2 tests pollute the CANN context).
    def sort_key(item):
        level = _item_scene_level(item) or 0
        # SDMA last, for the same class of reason L3 goes first: provisioning
        # the workspace leaves 48 STARS streams in the device's fault domain,
        # so every fault-injection case must have already run on a device that
        # never provisioned (#1425). Keyed off the marker, not the class, since
        # the fault-injection tests are plain functions with no _st_level.
        sdma_last = 1 if item.get_closest_marker("sdma") else 0
        return (sdma_last, 0 if level >= 3 else 1, item.nodeid)

    items.sort(key=sort_key)

    # L3 perf collection is not supported yet: a single L3 case forks N chip-processes
    # that all write chip_swimlane_records_<ts>.json to the same directory with
    # second-precision timestamps, so they trample each other. Block the
    # combination up front; waiting for a proper device-id-in-filename fix.
    if config.getoption("--enable-chip-swimlane", default=0) and config.getoption("--rounds", default=1) <= 1:
        l3_items = [
            i
            for i in items
            if _item_scene_level(i) == SceneTestLevel.NODE and not any(m.name == "skip" for m in i.iter_markers())
        ]
        if l3_items:
            sample = ", ".join(sorted({i.nodeid for i in l3_items})[:3])
            more = "" if len(l3_items) <= 3 else f" (+{len(l3_items) - 3} more)"
            raise pytest.UsageError(
                f"--enable-chip-swimlane is not supported for L3 tests yet — "
                f"multi-chip-process filename collision unresolved. "
                f"L3 items in this session: {sample}{more}. "
                f"Either drop --enable-chip-swimlane or scope to L2 with --level 2."
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
    run_last: bool


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


def _collect_resource_jobs(items, platform, manual_mode="exclude"):
    """Collect every item that needs a dedicated device-allocating subprocess.

    Two job kinds share one phase:

      - ``l3``:         one per L3 ``SceneTestCase`` class.
        ``device_count`` is the max across the class's platform-matching
        cases selected by ``manual_mode``.
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
            if not _manual_mode_matches(is_manual_for_platform(case.get("manual"), platform), manual_mode):
                continue
            saw_case = True
            max_dev = max(max_dev, int(case.get("config", {}).get("device_count", 1)))
        if saw_case:
            l3_by_nodeid[item.nodeid] = _ResourceJob(
                kind="l3",
                nodeid=item.nodeid,
                label=cls.__name__,
                runtime=rt,
                device_count=max_dev,
                run_last=item.get_closest_marker("resource_last") is not None,
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
            run_last=item.get_closest_marker("resource_last") is not None,
        )
    jobs.extend(standalone_by_nodeid.values())

    return jobs


def _strip_value_options(args, options):
    stripped = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        text = str(arg)
        if text in options:
            skip_next = True
            continue
        if any(text.startswith(f"{option}=") for option in options):
            continue
        stripped.append(text)
    return stripped


def _resource_child_command(spec, device_ids, platform, manual_mode):
    command = [
        sys.executable,
        "-m",
        "pytest",
        spec.nodeid,
        "--runtime",
        spec.runtime,
        "--device",
        _ps.format_device_range(device_ids),
    ]
    if spec.kind == "l3":
        command.extend(["--level", "3"])
    if platform:
        command.extend(["--platform", platform])
    command.extend(["--manual", manual_mode])
    return command


def _base_pytest_argv(session, *, strip_options=()):
    """Inherit the user's original pytest invocation args."""
    base = [sys.executable, "-m", "pytest"]
    args = _strip_value_options(session.config.invocation_params.args, set(strip_options))
    for arg in args:
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


def _github_actions_escape(value: object) -> str:
    """Escape a value for the GitHub Actions workflow-command payload."""
    return str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _emit_resource_failure_summary(
    results: list[_ps.JobResult],
    *,
    emit_annotations: bool = True,
    heading: str = "Resource phase failed",
) -> None:
    """Print failed Resource child jobs outside collapsible groups."""
    failed = [r for r in results if r.returncode != 0]
    if not failed:
        return

    print(f"\n*** {heading}: {len(failed)} child job(s) ***", flush=True)
    for res in failed:
        devices = ",".join(str(d) for d in res.device_ids)
        nodeid = res.nodeid or "<unknown>"
        if emit_annotations:
            message = _github_actions_escape(f"{nodeid} ({res.label}) rc={res.returncode} devices=[{devices}]")
            print(f"::error title=Resource phase failed::{message}", flush=True)

        print(
            f"- nodeid={nodeid}",
            flush=True,
        )
        print(f"  label={res.label}", flush=True)
        print(f"  rc={res.returncode} devices={res.device_ids} duration={res.duration_s:.1f}s", flush=True)
        print("  full output is in the Resource child group above", flush=True)


def _run_resource_phase(resource_specs, device_ids, max_parallel, fail_fast, platform, manual_mode, cwd, heading):
    jobs = []
    for spec in resource_specs:
        label = f"{spec.kind} {spec.label} (rt={spec.runtime}, dev={spec.device_count})"

        def _build(ids, _spec=spec):
            return _resource_child_command(_spec, ids, platform, manual_mode)

        jobs.append(
            _ps.Job(
                label=label,
                device_count=spec.device_count,
                build_cmd=_build,
                cwd=str(cwd),
                nodeid=spec.nodeid,
            )
        )

    def _on_done(res):
        tag = "PASS" if res.returncode == 0 else f"FAIL rc={res.returncode}"
        nodeid = res.nodeid or "<unknown>"
        header = f"{res.label} nodeid={nodeid} [{tag} {res.duration_s:.1f}s, devices={res.device_ids}]"
        _emit_group(header, res.output)
        if res.returncode != 0:
            print(
                f"*** FAIL: {nodeid} ({res.label}, devices={res.device_ids}) — expand group above ***",
                flush=True,
            )

    print(
        f"\n{heading}: {len(jobs)} case(s), pool={device_ids}, max_parallel={max_parallel}",
        flush=True,
    )
    results = _ps.run_jobs(
        jobs,
        device_ids,
        max_parallel=max_parallel,
        fail_fast=fail_fast,
        on_job_done=_on_done,
    )
    if any(r.returncode == TIMEOUT_EXIT_CODE for r in results):
        print(f"\n*** {heading}: TIMED OUT ***\n", flush=True)
        os._exit(TIMEOUT_EXIT_CODE)
    return results


def _dispatch_test_phases(session, resource_specs):  # noqa: PLR0912
    """Run Resource → L2 → final Resource phases.

    The Resource phase dispatches every item that needs a dedicated
    device-allocating subprocess — L3 ``SceneTestCase`` classes *and*
    standalone functions marked with ``@pytest.mark.device_count`` +
    ``@pytest.mark.runtime``. Items marked ``resource_last`` run after L2 in
    the same pytest invocation and enclosing device allocation.

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
    manual_mode = cfg.getoption("--manual", default="exclude")
    max_parallel = _resolve_max_parallel(cfg, platform or "", device_ids)

    base_args = _base_pytest_argv(session, strip_options=("--exclude-level",))
    cwd = session.config.invocation_params.dir
    ordinary_resource_specs = [spec for spec in resource_specs if not spec.run_last]
    final_resource_specs = [spec for spec in resource_specs if spec.run_last]

    # ----- Phase 1: Resource (L3 classes + standalone resource functions) -----
    resource_failed = False
    resource_results: list[_ps.JobResult] = []
    if ordinary_resource_specs:
        try:
            results = _run_resource_phase(
                ordinary_resource_specs,
                device_ids,
                max_parallel,
                fail_fast,
                platform,
                manual_mode,
                cwd,
                "Resource phase",
            )
        except ValueError as e:
            print(f"\n*** Resource phase ABORTED: {e} ***\n", flush=True)
            session.testsfailed = 1
            return True
        resource_results = results
        resource_failed = any(r.returncode != 0 for r in results)
        if resource_failed:
            _emit_resource_failure_summary(results)

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
        # Per-runtime sink for the in-process poison guards (issue #1110). Each
        # xdist worker appends the classes it poison-skips; we re-run them in a
        # fresh subprocess after this one exits so they don't silently lose
        # coverage. mkstemp gives an unpredictable, exclusively-created path
        # (this dev box is shared — a predictable /tmp name is exposed to symlink
        # / TOCTOU attacks and pid-recycle collisions); the children just append.
        sink_fd, sink_path = tempfile.mkstemp(prefix="simpler_l2_poison_", suffix=f"_{rt}.jsonl")
        os.close(sink_fd)
        run_env = {**os.environ, _L2_POISON_SINK_ENV: sink_path}
        # L2 subprocesses run serially (one runtime at a time) so we don't
        # need to buffer their stdout — we can stream it directly through
        # the group markers. ``::group::`` on its own line before the run
        # opens the fold; ``::endgroup::`` after closes it.
        label = f"L2 {rt}" + (f" [-n {max_parallel}]" if xdist_available else "")
        start = time.monotonic()
        print(f"::group::{label}", flush=True)
        result = subprocess.run(cmd, check=False, cwd=cwd, env=run_env)
        duration = time.monotonic() - start
        tag = "PASS" if result.returncode == 0 else f"FAIL rc={result.returncode}"
        print(f"--- L2 {rt}: {tag} {duration:.1f}s ---", flush=True)
        print("::endgroup::", flush=True)

        if result.returncode == TIMEOUT_EXIT_CODE:
            print(f"*** L2 {rt}: TIMED OUT ***", flush=True)
            os._exit(TIMEOUT_EXIT_CODE)

        # Restore coverage for classes the in-process poison guard skipped: a
        # fresh subprocess gets a clean card. Collect strictly from the sink the
        # guards registered, never from "outcome == skipped", so legitimate
        # skips are untouched.
        deferred = _read_l2_poison_sink(sink_path)
        try:
            os.unlink(sink_path)
        except FileNotFoundError:
            pass
        if deferred and _l2_poison_retry(base_args, rt, deferred, cwd) != 0:
            l2_failed = True
            print(f"*** FAIL: L2 {rt} poison-retry — expand group above ***", flush=True)
            if fail_fast:
                break

        if result.returncode != 0:
            l2_failed = True
            print(f"*** FAIL: L2 {rt} — expand group above ***", flush=True)
            if fail_fast:
                break

    # ----- Phase 3: Resource jobs that must leave no ordinary ST successor -----
    if final_resource_specs and not (l2_failed and fail_fast):
        try:
            results = _run_resource_phase(
                final_resource_specs,
                device_ids,
                max_parallel,
                fail_fast,
                platform,
                manual_mode,
                cwd,
                "Final Resource phase",
            )
        except ValueError as e:
            print(f"\n*** Final Resource phase ABORTED: {e} ***\n", flush=True)
            session.testsfailed = 1
            return True
        resource_results.extend(results)
        final_resource_failed = any(r.returncode != 0 for r in results)
        resource_failed = resource_failed or final_resource_failed
        if final_resource_failed:
            _emit_resource_failure_summary(results, heading="Final Resource phase failed")

    if resource_failed:
        _emit_resource_failure_summary(
            resource_results,
            emit_annotations=False,
            heading="Resource phase failed recap",
        )

    session.testsfailed = 1 if (resource_failed or l2_failed) else 0
    if not (resource_failed or l2_failed):
        session.testscollected = sum(1 for _ in session.items)
    return True  # returning True prevents default runtestloop


def pytest_runtestloop(session):
    """Dispatch Resource + L2 phases unless caller is already in child mode.

    Child mode (runtime-filtered runs, L2/L3 level-filtered runs, or
    --collect-only) skips the dispatcher and falls through to pytest's
    default runtestloop. Level 4 network1 selection is not child mode; it still
    uses the Resource dispatcher.
    """
    runtime_filter = session.config.getoption("--runtime")
    level_filter = session.config.getoption("--level")

    # Child mode: if the caller filters by runtime, or by the SceneTestCase
    # levels the dispatcher itself uses for children, it wants direct control.
    if runtime_filter is not None:
        return
    if _normalize_cli_scene_level(level_filter) in (SceneTestLevel.CHIP, SceneTestLevel.NODE):
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
    manual_mode = session.config.getoption("--manual", default="exclude")
    runtimes_all = _collect_st_runtimes(session.items)
    resource_specs = _collect_resource_jobs(session.items, platform, manual_mode)
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
# L2 worker-pool generation rotation and self-healing
#
# The L2 ``st_worker`` pool hands ONE ``ChipWorker`` to every SceneTestCase on
# a (runtime, device) for the life of an xdist worker process. That amortizes
# init cost, but it also means a single device-runtime error (an AICore
# op-timeout reaped by STARS, or a kernel-launch failure) poisons the shared
# ACL/device context: every later test on that worker then fails at
# ``halResMap`` (rc=62) / ``rtMalloc`` (507899) — one failure cascades into the
# whole worker's remaining tests. An a2a3 Worker that has attempted an SDMA
# dispatch has a second, non-error terminal condition for pooling:
# CANN does not expose a fence proving its remote channels fully retired, so it
# must be closed/reset before an ordinary or fault-injection class can reuse the
# physical device. The Worker records actual SDMA dispatch attempts; merely
# registering an unused SDMA callable does not rotate the pool.
#
# The driver-side guard
# (``DeviceRunner::run`` fail-fast on ``device_unusable_``) keeps that cascade
# from wedging, but the Worker stays poisoned.
#
# On a5 the poison survives close()+device-reset for the life of the process
# (an in-process Worker.init rebuild fails with rtStreamCreate 507899; a
# force-reset is unsafe on this shared box), so there is no in-process
# recovery — only a fresh worker process gets a clean device. The native status
# query tells the ``st_worker`` finalizer when to drop a poisoned or
# SDMA-exposed pooled Worker; the hook below also stashes setup/call exceptions
# as a diagnostic fallback. The L2 branch rebuilds where the arch allows it and otherwise
# skips the remaining tests for that runtime (``_l2_poisoned``). Golden
# mismatches / ordinary assertions leave a non-SDMA Worker reusable.
# ---------------------------------------------------------------------------

# Plain attribute name on the test item — NOT pytest.StashKey()/item.stash,
# which are pytest>=7.0 only; the test extra pins `pytest>=6.0`, and StashKey
# at import time would AttributeError on pytest 6.x.
_ST_CALL_EXCINFO = "_st_call_excinfo"


# hookwrapper (not the 8.0+ `wrapper=True`) so this works across the whole
# declared pytest range (test extra pins only `pytest>=6.0`). We stash
# call.excinfo for the side effect and don't touch the result.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    yield
    # A fixture that depends on st_worker may dispatch during setup. Preserve
    # that error before st_worker's teardown finalizer decides whether its
    # Worker can return to the session pool. Native lifecycle status below is
    # authoritative; this text classification remains a diagnostic fallback.
    if call.when in {"setup", "call"} and call.excinfo is not None:
        setattr(item, _ST_CALL_EXCINFO, call.excinfo)


# Only these error codes mean the device/ACL context is sticky-errored — i.e.
# the worker is poisoned and must be rebuilt/skipped. Matching on the bare
# "<op> failed with code" prefix would also catch ordinary kernel/run failures
# (every non-zero rc is wrapped that way), turning a normal failing test into a
# spurious worker rebuild and, downstream, a misleading runtime-wide skip. So
# we extract the trailing <N> and match only the known poison codes:
#   207001 ACL_ERROR_RT_MEMORY_ALLOCATION  (the CI cascade trigger)
#   507000 ACL_ERROR_RT_INTERNAL_ERROR  (a5 op-timeout at AICPU stream sync)
#   507015 ACL_ERROR_RT_AICORE_EXCEPTION
#   507018 ACL_ERROR_RT_AICPU_EXCEPTION
#   507046 ACL_ERROR_RT_STREAM_SYNC_TIMEOUT
#   507899 sticky-error rtMalloc / rtStreamCreate on the poisoned context
#       -1 a5 DeviceRunner fail-fast sentinel ("marked unusable; refusing to run")
# The names above are CANN's own (acl/error_codes/rt_error_codes.h); what each one
# means and how to chase it is in docs/troubleshooting/device-error-codes.md, and the
# host log now prints that meaning next to the code.
# Worker surfaces these as "run/run_prepared/prepare_callable/simpler_init
# failed with code <N>" — never as an AssertionError, so golden mismatches are
# already excluded.
_DEVICE_POISON_CODES = frozenset({207001, 507000, 507015, 507018, 507046, 507899, -1})
_DEVICE_ERROR_CODE_RE = re.compile(r"\b(?:run|run_prepared|prepare_callable|simpler_init) failed with code (-?\d+)\b")
_DEVICE_QUARANTINE_MARKERS = (
    "device is quarantined after a prior Worker could not confirm reset",
    "device is already owned by another live ChipWorker in this process",
    "device reset was not confirmed, so this process quarantined the device",
    "cleanup could not confirm device reset, so this process quarantined the device",
)


class _L2WorkerPool(dict):
    """Reusable Workers plus closed-terminal Workers retained for safe cleanup.

    A Worker is removed from the reusable mapping before close, but a close
    failure can be the retryable active-operation drain path. Keep a strong
    reference in ``retired`` so GC cannot jump directly to the C++ destructor
    and bypass Python's drain fence; session teardown retries it once.
    """

    def __init__(self):
        super().__init__()
        self.retired: list[object] = []

    def retire(self, key):
        worker = self.pop(key, None)
        if worker is None:
            return
        try:
            worker.close()
        except BaseException:  # noqa: BLE001
            self.retired.append(worker)
            raise


# A final session-teardown close can still fail (for example an operation that
# never drained). Retain such terminal objects until interpreter/process exit;
# recreating or dropping them early would be less safe than an intentional
# process-lifetime leak.
_L2_TERMINAL_RETAINED_WORKERS: list[object] = []


def _is_device_runtime_error_msg(msg: str) -> bool:
    match = _DEVICE_ERROR_CODE_RE.search(msg)
    return (match is not None and int(match.group(1)) in _DEVICE_POISON_CODES) or any(
        marker in msg for marker in _DEVICE_QUARANTINE_MARKERS
    )


def _is_device_runtime_error(excinfo) -> bool:
    if excinfo is None or not issubclass(excinfo.type, RuntimeError):
        return False
    return _is_device_runtime_error_msg(str(excinfo.value))


def _register_l2_pool_recycle(request, pool, key, poisoned_runtimes):
    """Drop + close a pooled L2 Worker after a device-runtime error.

    Recycling prevents a poisoned-context cascade. A passing test and a
    non-device failure (golden mismatch, assertion) keep the Worker pooled.

    NOTE: an AICore op-timeout poisons the device context. `DeviceRunner::
    finalize()` force-resets the card (`aclrtResetDeviceForce`, per-card safe
    under the exclusive task-submit lock), so the rebuilt Worker.init normally
    lands on a clean card and the next test runs. Force-reset fires when
    `device_unusable_` was set by the runner's launch/sync error paths. If the
    rebuild still fails (for example rtStreamCreate returns 507899),
    and the st_worker L2 branch falls back to skipping the rest of the runtime
    (see _l2_poisoned). The dispatcher re-runs those poison-skipped classes in a
    fresh subprocess (= clean card) so they keep coverage — see
    _register_l2_poison_skip / issue #1110. Either way one device error stops
    cascading into a worker-wide storm of confusing 507899 failures."""

    def _recycle():
        worker = pool.get(key)
        if worker is None:
            return
        device_error = _is_device_runtime_error(getattr(request.node, _ST_CALL_EXCINFO, None))
        if not device_error:
            return
        try:
            pool.retire(key)
        except Exception:  # noqa: BLE001
            poisoned_runtimes.add(key[0])

    request.addfinalizer(_recycle)


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
def st_network1_peer():
    """Network1 endpoint and remote device pool from the network1 runner environment."""
    endpoint = os.environ.get("NETWORK1_REMOTE_ENDPOINT")
    if not endpoint:
        pytest.skip("NETWORK1_REMOTE_ENDPOINT is required for network1 tests")
    remote_devices = os.environ.get("NETWORK1_REMOTE_DEVICES")
    if not remote_devices:
        pytest.skip("NETWORK1_REMOTE_DEVICES is required for network1 tests")
    try:
        session_timeout_s = float(os.environ.get("NETWORK1_L3_SESSION_TIMEOUT_S", "120"))
    except ValueError as e:
        pytest.fail(f"NETWORK1_L3_SESSION_TIMEOUT_S must be a float: {e}")
    # The remote peer connects back to the parent session runner.
    return Network1Peer(
        endpoint=endpoint,
        remote_device_ids=tuple(_parse_device_range(remote_devices)),
        session_timeout_s=session_timeout_s,
        session_listen_host=os.environ.get("NETWORK1_L3_SESSION_LISTEN_HOST", "0.0.0.0"),  # noqa: S104
    )


@pytest.fixture()
def st_network1_remote_device_ids(request, st_network1_peer):
    """Allocate remote device IDs from the network1 peer's default device pool.

    Every network1 test gets the same leading slice, so this is collision-free only
    while the network1 job serializes the sweep with ``--max-parallel 1``.
    """
    marker = request.node.get_closest_marker("network1_remote_device_count")
    n = marker.args[0] if marker else 1
    if n > len(st_network1_peer.remote_device_ids):
        available = len(st_network1_peer.remote_device_ids)
        pytest.fail(f"need {n} remote devices but NETWORK1_REMOTE_DEVICES only has {available} entries")
    return list(st_network1_peer.remote_device_ids[:n])


@pytest.fixture()
def st_network1_logs(request, monkeypatch):
    """Per-test parent log directory for network1 scene tests."""
    if _item_scene_level(request.node) != SceneTestLevel.NETWORK1:
        pytest.fail("st_network1_logs requires SceneTestLevel.NETWORK1")
    run_dir = os.environ.get("RUN_DIR")
    if not run_dir:
        if not os.environ.get("NETWORK1_REMOTE_ENDPOINT") or not os.environ.get("NETWORK1_REMOTE_DEVICES"):
            pytest.skip("network1 runner environment is required for network1 tests")
        pytest.fail("RUN_DIR is required for network1 tests")
    machine = os.environ.get("NETWORK1_MACHINE", "parent")
    nodeid = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.nodeid)
    log_path = os.path.join(run_dir, "pytest", f"parent-{machine}", "ascend", nodeid)
    os.makedirs(log_path, exist_ok=True)
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", log_path)
    return log_path


@pytest.fixture(scope="session")
def _l2_worker_pool(request, st_platform):
    """Session-scoped L2 worker pool keyed by (runtime, device_id).

    Under xdist, each worker process owns one device (slicing done in
    pytest_configure), so this pool typically ends up with one entry per
    runtime. Tests on the same worker that share a runtime reuse the same
    ``ChipWorker`` — amortizing the init cost (three dlopens + device
    acquire) over every class on that device.
    """
    pool = _L2WorkerPool()
    yield pool
    # Session teardown: close every Worker we minted.
    workers = list(pool.values()) + pool.retired
    pool.clear()
    pool.retired.clear()
    for w in workers:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            _L2_TERMINAL_RETAINED_WORKERS.append(w)


_L2_POISON_SINK_ENV = "SIMPLER_L2_POISON_SINK"


def _register_l2_poison_skip(node):
    """Record a poison-skipped L2 class so the dispatcher can re-run it in a
    fresh subprocess (= clean card), restoring coverage the in-process skip
    would otherwise drop (issue #1110).

    Called from the *two* ``st_worker`` poison guards only — legitimate skips
    (``@pytest.mark.skip``, platform-required, ``requires SceneTestCase``,
    ``No cases matched``) never reach this sink and so are never retried. The
    record is a class selector (``ClassName::``) because one ``test_run`` node
    runs every case of its class; ``--case ClassName::`` reselects them all.

    The sink is a per-runtime JSONL file shared across the xdist-worker → L2
    subprocess → dispatcher process boundaries; ``O_APPEND`` makes concurrent
    worker writes atomic. No-op when the env var is unset (a direct ``pytest``
    run not under the dispatcher) — registration is then simply skipped.

    Best-effort: this runs immediately before an intended ``pytest.skip`` in the
    poison guards, so an I/O failure here must not raise — that would convert
    the skip into a setup ERROR. A failed write only costs the coverage retry
    (the class is not re-run in a fresh process), which is no worse than the
    pre-#1110 behavior; we warn so the dropped retry is diagnosable."""
    sink = os.environ.get(_L2_POISON_SINK_ENV)
    if not sink:
        return
    cls = getattr(node, "cls", None)
    if cls is None:
        return
    record = json.dumps({"selector": f"{cls.__name__}::", "nodeid": node.nodeid})
    try:
        fd = os.open(sink, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (record + "\n").encode())
        finally:
            os.close(fd)
    except OSError as e:
        print(
            f"*** WARN: could not register L2 poison-skip for {cls.__name__} "
            f"(sink write failed: {e}); class will not be retried ***",
            flush=True,
        )


def _read_l2_poison_sink(path):
    """Read poison-skipped class selectors from the sink, deduped and
    order-preserved (both poison guards may register the same class)."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return []
    selectors = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sel = json.loads(line).get("selector")
        except json.JSONDecodeError:
            continue
        if sel:
            selectors.append(sel)
    return list(dict.fromkeys(selectors))


def _l2_poison_retry(base_args, rt, selectors, cwd):
    """Re-run poison-skipped L2 classes for ``rt`` in a fresh subprocess.

    A new process gets a clean card — the op-timeout poison does not survive a
    process exit — so classes the in-process skip dropped get real coverage
    (issue #1110), independent of whether the in-place force-reset recovery
    worked. Runs *without* the sink env so a retried class that itself
    re-poisons is not collected again: this is a single bounded pass, not a
    recursive loop. Returns the subprocess return code."""
    cmd = base_args + ["--runtime", rt, "--level", "2"]
    for sel in selectors:
        cmd += ["--case", sel]
    env = {k: v for k, v in os.environ.items() if k != _L2_POISON_SINK_ENV}
    label = f"L2 {rt} poison-retry ({len(selectors)} class(es), fresh process)"
    print(f"::group::{label}", flush=True)
    result = subprocess.run(cmd, check=False, cwd=cwd, env=env)
    tag = "PASS" if result.returncode == 0 else f"FAIL rc={result.returncode}"
    print(f"--- {label}: {tag} ---", flush=True)
    print("::endgroup::", flush=True)
    return result.returncode


@pytest.fixture(scope="session")
def _l2_poisoned():
    """Runtimes whose L2 device context was poisoned by a device-runtime error
    and could not be re-initialized in this process. ``st_worker`` skips
    remaining tests for a poisoned runtime instead of letting them fail with
    confusing 507899 cascades. Lives for the xdist worker process; only a fresh
    process (process exit fully releases the device) recovers — which is exactly
    how the standalone phase + a new xdist worker get a clean device."""
    return set()


@pytest.fixture()
def st_worker(request, st_platform, device_pool, _l2_worker_pool, _l2_poisoned):  # noqa: PLR0912
    """Per-test Worker.

    L2: session-scoped, reused across classes with the same (runtime, device).
    L3: per-test (registers sub-callables at init, can't be reused).
    """
    cls = request.node.cls
    if cls is None or not hasattr(cls, "_st_level"):
        pytest.skip("st_worker requires SceneTestCase")

    level = cls._st_level
    runtime = cls._st_runtime
    from simpler_setup.scene_test import _class_wants_sdma  # noqa: PLC0415

    wants_sdma = _class_wants_sdma(cls)

    if level == 2:
        # A prior test on this runtime poisoned the device and the rebuild below
        # could not re-init it in-process (a5 op-timeout: rtStreamCreate 507899).
        # Skip the rest for this runtime so one device error stops cascading —
        # the triggering failure is already reported; a fresh worker process is
        # the only real recovery.
        if runtime in _l2_poisoned:
            _register_l2_poison_skip(request.node)
            pytest.skip(
                f"L2 device context for runtime '{runtime}' was poisoned by an earlier device-runtime error on "
                f"this worker process and cannot be re-initialized in-process; skipping to avoid a misleading "
                f"507899 cascade (a fresh worker process recovers)."
            )

        # L2 share: reuse any Worker already created for this runtime in the
        # current process. Under xdist, each worker process is sliced to a
        # single device so there's at most one matching entry. On first call
        # we allocate a device from the pool and immediately release it back —
        # the pool is a process-scoped counter for other fixtures (e.g.
        # st_device_ids) that also draw from it; retaining the id would drain
        # the pool and break any non-st_worker test that runs afterward on the
        # same xdist worker.
        # The SDMA capability is part of the Worker's identity, not a per-run
        # option: an enable_sdma Worker holds 48 STARS streams for its whole
        # life, so it must never be handed to a test that did not ask for them,
        # nor a plain Worker to one that did. It therefore takes a slot in the
        # pool key and gates reuse. Ordering puts every sdma test after the
        # rest, so the swap happens once, at the end.
        for (rt, dev_id, pooled_sdma), existing in _l2_worker_pool.items():
            if rt == runtime and pooled_sdma == wants_sdma:
                _register_l2_pool_recycle(request, _l2_worker_pool, (rt, dev_id, pooled_sdma), _l2_poisoned)
                yield existing
                return

        ids = device_pool.allocate(1)
        if not ids:
            pytest.fail(f"no devices available in --device pool (requested 1, pool has {len(device_pool._available)})")
        dev_id = ids[0]
        device_pool.release(ids)
        key = (runtime, dev_id, wants_sdma)

        # At most one runtime-specific Worker may own a device: finalization
        # resets resources that would invalidate every other Worker on it.
        for stale_key in list(_l2_worker_pool):
            if stale_key[1] != dev_id:
                continue
            try:
                _l2_worker_pool.retire(stale_key)
            except Exception:
                _l2_poisoned.add(stale_key[0])
                raise

        from simpler.worker import Worker  # noqa: PLC0415

        w = Worker(level=2, device_id=dev_id, platform=st_platform, runtime=runtime, enable_sdma=wants_sdma)
        w._st_device_id = dev_id
        # First rebuild after a poison-and-heal lands here. On arches where the
        # device re-inits cleanly this just works; on a5 the op-timeout poison
        # survives close()+reset, so Worker.init raises a device-runtime error —
        # mark the runtime poisoned and skip (this test and every later one for
        # the runtime) instead of surfacing a raw 507899 setup ERROR.
        try:
            w.init()
        except RuntimeError as e:
            if _is_device_runtime_error_msg(str(e)):
                _l2_poisoned.add(runtime)
                _register_l2_poison_skip(request.node)
                try:
                    w.close()
                except Exception:  # noqa: BLE001
                    pass
                pytest.skip(
                    f"L2 Worker.init for runtime '{runtime}' failed with a device-runtime error ({e}); the device "
                    f"context is not recoverable in-process after an earlier AICore error — skipping remaining "
                    f"'{runtime}' L2 tests (a fresh worker process recovers)."
                )
            raise
        _l2_worker_pool[key] = w
        _register_l2_pool_recycle(request, _l2_worker_pool, key, _l2_poisoned)
        yield w
        # No close here on success — the pool handles teardown at session end.
        # On a device-runtime failure, the finalizer registered above closes
        # this Worker and drops it from the pool so the next test rebuilds.

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
            enable_sdma=wants_sdma,
        )
        w._st_device_id = ids[0]  # expose primary device to test_run for profiling snapshots

        # Register SubCallable entries from cls.CALLABLE
        sub_handles = {}
        chip_handles = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                handle = w.register(entry["callable"])
                sub_handles[entry["name"]] = handle
            elif "orchestration" in entry:
                from simpler_setup.scene_test import (  # noqa: PLC0415
                    _compile_chip_callable_from_spec,
                    l3_compile_cache_key,
                )

                name = entry["name"]
                cache_key = l3_compile_cache_key(cls.__module__, cls.__qualname__, name, st_platform, runtime)
                chip = _compile_chip_callable_from_spec(entry, st_platform, runtime, cache_key)
                handle = w.register(chip)
                chip_handles[name] = handle
                chip_handles[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        cls._st_sub_handles = sub_handles
        cls._st_chip_handles = chip_handles

        w.init()
        yield w
        w.close()
        device_pool.release(ids)


@pytest.fixture()
def st_device_ids(request, device_pool, _l2_worker_pool):
    """Allocate device IDs. Use @pytest.mark.device_count(n) to request multiple."""
    marker = request.node.get_closest_marker("device_count")
    n = marker.args[0] if marker else 1
    ids = device_pool.allocate(n)
    if not ids:
        pytest.fail(f"need {n} devices")
    # Standalone/manual Worker tests can be selected into the same direct
    # pytest process as SceneTestCase classes. A pooled ChipWorker owns the
    # physical device, so retire it before handing that id to an independent
    # fixture (normal CI dispatch already separates these phases/processes).
    try:
        for key in list(_l2_worker_pool):
            if key[1] in ids:
                _l2_worker_pool.retire(key)
        yield ids
    finally:
        device_pool.release(ids)
