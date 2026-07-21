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
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


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
            # Block on sleep — sleep is signal-interruptible, so the SIGUSR1
            # trampoline preempts it and faulthandler runs.
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
        import sys
        sys.path.insert(0, {str(_ROOT)!r})
        sys.path.insert(0, {str(_ROOT / "python")!r})

        from simpler_setup import parallel_scheduler as _ps

        import importlib.util
        spec = importlib.util.spec_from_file_location("_rc", {str(_ROOT / "conftest.py")!r})
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)

        # 3 s is plenty: target test forks immediately and the grandchild
        # sleeps; nothing finishes on its own.
        rc._install_session_timeout(3)

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
