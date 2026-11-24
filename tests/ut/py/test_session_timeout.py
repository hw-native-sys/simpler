# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the dispatcher session-timeout / HUNG-dump machinery.

Covers ``conftest._collect_descendant_pids`` and an end-to-end repro for the
case that motivated this code: a sim test whose deadlock lives in an
``os.fork``'d child of the dispatched pytest. Sending SIGUSR1 only to the
dispatched pid (the pre-fix behavior) produced an empty HUNG group; sending
it through the descendant tree gets faulthandler tracebacks from the actual
deadlock site.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _assert_process_stopped(pid):
    assert Path("/proc/self/stat").is_file(), "Process-state verification requires Linux procfs"
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = stat.read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return
        if state in {"Z", "X", "x"}:
            return
        time.sleep(0.01)
    pytest.fail(f"Process {pid} is still alive: {state}")


def test_scheduler_output_log_failure_cannot_pass(tmp_path):
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    job = scheduler.Job(
        label="unwritable-log",
        device_count=1,
        build_cmd=lambda _: [sys.executable, "-c", "print('diagnostic', flush=True)"],
        output_path=str(tmp_path),  # Opening a directory as a log must fail.
        isolated_process_group=True,
    )
    with pytest.raises(OSError, match="Cannot write subprocess log"):
        scheduler.run_jobs([job], [1])
    assert scheduler._active_state is None


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-state inspection")
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_scheduler_exception_reaps_isolated_descendants(tmp_path, exception):
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    marker = tmp_path / "descendant"
    descendant = (
        "import os,signal; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"marker=Path({str(marker)!r}); pending=marker.with_name('.preparing-'+marker.name); "
        "pending.write_text(str(os.getpid())); pending.replace(marker); signal.pause()"
    )
    slow = f"import subprocess,sys; subprocess.run([sys.executable, '-c', {descendant!r}])"
    fast = (
        "import time; from pathlib import Path; "
        f"marker=Path({str(marker)!r}); end=time.monotonic()+10\n"
        "while not marker.exists() and time.monotonic()<end: time.sleep(0.01)\n"
        "assert marker.exists()\n"
    )
    jobs = [
        scheduler.Job(
            label="slow", device_count=1, build_cmd=lambda _: [sys.executable, "-c", slow], isolated_process_group=True
        ),
        scheduler.Job(
            label="fast", device_count=1, build_cmd=lambda _: [sys.executable, "-c", fast], isolated_process_group=True
        ),
    ]

    def stop(_):
        raise exception("stop")

    try:
        with pytest.raises(exception):
            scheduler.run_jobs(jobs, [1, 3], on_job_done=stop)
        assert marker.exists()
        _assert_process_stopped(int(marker.read_text()))
        assert scheduler._active_state is None
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGALRM, signal.SIGINT])
def test_scheduler_signal_during_launch_cannot_escape_tracking(monkeypatch, signum):
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    processes = []
    original_popen = subprocess.Popen
    previous = signal.getsignal(signum)

    def stop(_signum, _frame):
        raise RuntimeError("launch interrupted")

    def interrupt_after_spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        os.kill(os.getpid(), signum)
        return process

    job = scheduler.Job(
        label="launch-race",
        device_count=1,
        isolated_process_group=True,
        build_cmd=lambda _: [sys.executable, "-c", "import signal; signal.pause()"],
    )
    signal.signal(signum, stop)
    monkeypatch.setattr(scheduler.subprocess, "Popen", interrupt_after_spawn)
    try:
        with pytest.raises(RuntimeError, match="launch interrupted"):
            scheduler.run_jobs([job], [1])
        assert len(processes) == 1
        assert processes[0].poll() is not None
        assert signal.getsignal(signum) is stop
        assert scheduler._active_state is None
    finally:
        signal.signal(signum, previous)
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-state inspection")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGALRM, signal.SIGINT])
def test_scheduler_signal_during_reap_cannot_orphan_descendants(tmp_path, monkeypatch, signum):
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    marker = tmp_path / "descendant"
    descendant = (
        "import os,signal; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"marker=Path({str(marker)!r}); pending=marker.with_name('.preparing-'+marker.name); "
        "pending.write_text(str(os.getpid())); pending.replace(marker); signal.pause()"
    )
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        f"marker=Path({str(marker)!r}); end=time.monotonic()+10\n"
        "while not marker.exists() and time.monotonic()<end: time.sleep(0.01)\n"
        "assert marker.exists()\n"
    )
    previous = signal.getsignal(signum)
    original_killpg = os.killpg
    interrupted = False

    def stop(_signum, _frame):
        raise RuntimeError("reap interrupted")

    def interrupt_before_group_kill(pgid, sig):
        nonlocal interrupted
        if sig == signal.SIGKILL and not interrupted:
            interrupted = True
            os.kill(os.getpid(), signum)
        original_killpg(pgid, sig)

    job = scheduler.Job(
        label="reap-race",
        device_count=1,
        isolated_process_group=True,
        build_cmd=lambda _: [sys.executable, "-c", parent],
    )
    signal.signal(signum, stop)
    monkeypatch.setattr(scheduler.os, "killpg", interrupt_before_group_kill)
    try:
        with pytest.raises(RuntimeError, match="reap interrupted"):
            scheduler.run_jobs([job], [1])
        assert interrupted
        _assert_process_stopped(int(marker.read_text()))
        assert scheduler._active_state is None
        assert signal.getsignal(signum) is stop
    finally:
        signal.signal(signum, previous)
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="POSIX signal masks")
def test_scheduler_children_preserve_signal_mask():
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    child = "import signal; print(sorted(map(int, signal.pthread_sigmask(signal.SIG_BLOCK, set()))))"
    job = scheduler.Job(
        label="signal-mask",
        device_count=1,
        isolated_process_group=True,
        build_cmd=lambda _: [sys.executable, "-c", child],
    )
    (result,) = scheduler.run_jobs([job], [1])
    assert result.returncode == 0
    assert result.output.strip() == str(sorted(map(int, original_mask)))
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="POSIX signal masks")
@pytest.mark.parametrize("pending_termination", [False, True])
def test_scheduler_signal_during_handler_restore_restores_every_handler(monkeypatch, pending_termination):
    from simpler_setup import parallel_scheduler as scheduler  # noqa: PLC0415

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    original_signal = signal.signal
    interrupted = False
    delivered = []
    original_popen = subprocess.Popen

    def stop(_signum, _frame):
        raise RuntimeError("restore interrupted")

    def record(signum, _frame):
        delivered.append(signum)

    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        if pending_termination:
            os.kill(os.getpid(), signal.SIGTERM)
        return process

    def interrupt_restore(sig, handler):
        nonlocal interrupted
        result = original_signal(sig, handler)
        if sig == signal.SIGINT and handler is stop and not interrupted:
            interrupted = True
            os.kill(os.getpid(), sig)
        return result

    expected = {sig: record if sig == signal.SIGTERM else stop for sig in signals}
    for sig, handler in expected.items():
        original_signal(sig, handler)
    monkeypatch.setattr(signal, "signal", interrupt_restore)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    try:
        job = scheduler.Job(label="restore-race", device_count=1, build_cmd=lambda _: [sys.executable, "-c", "pass"])
        with pytest.raises(RuntimeError, match="restore interrupted"):
            scheduler.run_jobs([job], [1])
        assert interrupted
        assert all(signal.getsignal(sig) is expected[sig] for sig in signals)
        assert delivered == ([signal.SIGTERM] if pending_termination else [])
        assert scheduler._active_state is None
    finally:
        for sig, handler in previous.items():
            original_signal(sig, handler)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-state inspection")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGALRM])
def test_scheduler_launch_signal_with_exiting_handler_reaps_child(tmp_path, signum):
    marker = tmp_path / "spawned"
    program = textwrap.dedent(f"""
        import os, signal, subprocess, sys
        from pathlib import Path
        from simpler_setup import parallel_scheduler as scheduler

        original_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            Path({str(marker)!r}).write_text(str(process.pid))
            os.kill(os.getpid(), {int(signum)})
            return process

        def stop(signum, frame):
            scheduler._terminate_all(scheduler._active_state)
            os._exit(128 + signum)

        signal.signal({int(signum)}, stop)
        scheduler.subprocess.Popen = spawn
        job = scheduler.Job(
            label="launch-exit", device_count=1, isolated_process_group=True,
            build_cmd=lambda _: [sys.executable, "-c", "import signal; signal.pause()"],
        )
        scheduler.run_jobs([job], [1])
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=_ROOT, capture_output=True, timeout=15, check=False
        )
        assert result.returncode == 128 + signum, result.stderr.decode()
        assert marker.exists()
        _assert_process_stopped(int(marker.read_text()))
    finally:
        if marker.exists():
            try:
                os.killpg(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc only")
def test_collect_descendant_pids_sees_fork_tree():
    """Fork a child that forks a grandchild; both must appear in the tree."""
    rc, wc = os.pipe()
    rgc, wgc = os.pipe()
    parent_pid = os.getpid()

    child = os.fork()
    if child == 0:
        try:
            os.close(rc)
            os.close(rgc)
            grandchild = os.fork()
            if grandchild == 0:
                os.close(wc)
                os.write(wgc, b"gc\n")
                os.close(wgc)
                # Block forever — parent kills us at end of test
                signal.pause()
                os._exit(0)
            os.write(wc, str(grandchild).encode() + b"\n")
            os.close(wc)
            os.close(wgc)
            signal.pause()
        finally:
            os._exit(0)

    os.close(wc)
    os.close(wgc)
    grandchild_pid: int | None = None
    try:
        # Read grandchild's pid from the pipe (proves both are alive)
        grandchild_pid = int(os.read(rc, 64).decode().strip())
        _ = os.read(rgc, 8)  # grandchild handshake

        cf = _load_root_conftest()
        kin = cf._collect_descendant_pids(parent_pid)
        assert child in kin, f"child {child} missing from {kin}"
        assert grandchild_pid in kin, f"grandchild {grandchild_pid} missing from {kin}"
    finally:
        for p in (child, grandchild_pid):
            if p is None:
                continue
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc only")
def test_collect_descendant_pids_returns_empty_for_dead_pid():
    """Walking ``/proc`` for a pid that's already gone returns ``[]``, not raise."""
    p = subprocess.Popen([sys.executable, "-c", "import os; os._exit(0)"])
    p.wait()
    cf = _load_root_conftest()
    # Even if the pid got recycled, /proc/<pid>/task should be empty / missing
    # and we just want no exception.
    kin = cf._collect_descendant_pids(p.pid)
    assert isinstance(kin, list)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux /proc only")
def test_session_timeout_surfaces_forked_child_traceback(tmp_path):
    """End-to-end repro: a pytest test that forks a deadlocked child must
    produce a HUNG group with the child's faulthandler traceback and a
    ``descendants=[...]`` annotation in the group header.

    Before the fix, SIGUSR1 only reached the dispatched pytest pid (which
    was calmly waiting in ``os.waitpid``); the forked grandchild — where
    the real deadlock lives — never saw the signal, so faulthandler never
    dumped from it.

    Note on ``-s``: pytest's default ``--capture=fd`` dup2's the test's
    stderr onto a capture pipe that is never flushed if the test hangs.
    Production dispatched pytests inherit this issue; the descendants fix
    alone won't fully resolve their HUNG body. ``-s`` here lets the test
    isolate and verify the descendants/pump-drain logic. Routing
    faulthandler to a pre-capture fd is a separate concern.
    """
    # The "job" pytest invocation — a tiny test that forks and blocks both
    # parent and child. The child has a unique sentinel function in its
    # faulthandler traceback so we can grep for it deterministically.
    ready_path = tmp_path / "grandchild-ready"
    target = tmp_path / "test_deadlock_target.py"
    target.write_text(
        textwrap.dedent("""
        # Self-register faulthandler at module load so the SIGUSR1 trampoline
        # is wired before fork (the production setup does this via the
        # project conftest's _install_child_faulthandler; this file lives in
        # tmp_path outside the rootdir so we replicate it inline).
        import faulthandler
        import os
        import signal
        import time

        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, chain=False, all_threads=True)


        def _grandchild_deadlock_sentinel():
            # Signal the driver that the sentinel is reached (the driver's
            # watcher triggers the timeout dump only after this marker
            # exists), then block on sleep — sleep is signal-interruptible,
            # so the SIGUSR1 trampoline preempts it and faulthandler runs.
            open(os.environ["DEADLOCK_READY"], "w").close()
            time.sleep(3600)


        def test_forks_and_blocks():
            pid = os.fork()
            if pid == 0:
                # Re-register in the child too; ``faulthandler.register``
                # state should survive fork but be explicit.
                faulthandler.register(signal.SIGUSR1, chain=False, all_threads=True)
                try:
                    _grandchild_deadlock_sentinel()
                finally:
                    os._exit(0)
            # Parent waits forever; the dispatcher's session timeout fires
            # before this completes.
            os.waitpid(pid, 0)
    """).strip()
        + "\n"
    )

    # Driver: emulate the dispatcher minimally — run the target pytest as a
    # subprocess via ``parallel_scheduler.run_jobs``, install the real
    # session-timeout handler from the project conftest, and let the alarm
    # fire. The same handler the dispatcher uses in CI is exercised.
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(f"""
        import os
        import signal
        import sys
        import threading
        import time
        sys.path.insert(0, {str(_ROOT)!r})
        sys.path.insert(0, {str(_ROOT / "python")!r})

        from simpler_setup import parallel_scheduler as _ps

        import importlib.util
        spec = importlib.util.spec_from_file_location("_rc", {str(_ROOT / "conftest.py")!r})
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)

        # The timeout is a fail-safe bound only (matching the driver's
        # subprocess timeout); the dump is triggered by the watcher once the
        # grandchild provably reaches the sentinel, so a loaded runner's slow
        # pytest startup cannot race it.
        rc._install_session_timeout(60)
        os.environ["DEADLOCK_READY"] = {str(ready_path)!r}

        # ``-s`` (capture=no) is critical: pytest's default fd-capture
        # dup2's the test's stderr onto its own capture pipe, so a
        # SIGUSR1 → faulthandler dump written to fd=2 never reaches the
        # parent's pump. CI's scene scheduler likewise runs with -s.
        job = _ps.Job(
            label="deadlock-repro",
            build_cmd=lambda devs: [
                sys.executable, "-m", "pytest", "-q", "-s",
                "-p", "no:cacheprovider",
                {str(target)!r},
            ],
            device_count=1,
        )
        def _fire_timeout_when_ready():
            # Run the real conftest handler the moment the grandchild is at
            # the sentinel, not on a timer racing the child's startup.
            while not os.path.exists(os.environ["DEADLOCK_READY"]):
                time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGALRM)

        threading.Thread(target=_fire_timeout_when_ready, daemon=True).start()
        _ps.run_jobs([job], device_ids=[0])
    """).strip()
        + "\n"
    )

    result = subprocess.run(
        [sys.executable, str(driver)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr

    # The timeout actually fired:
    assert "[pytest] TIMEOUT: session exceeded" in combined, combined[-3000:]
    # HUNG group exists and now reports descendants (the forked child + grandchild):
    assert "HUNG deadlock-repro" in combined, combined[-3000:]
    assert "descendants=[" in combined, combined[-3000:]
    # The actual fix: faulthandler from the grandchild reaches the HUNG body.
    # Look for the deadlock-sentinel frame in the traceback.
    assert "_grandchild_deadlock_sentinel" in combined, combined[-3000:]
