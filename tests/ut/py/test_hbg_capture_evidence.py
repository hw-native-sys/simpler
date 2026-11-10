# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU regressions for native capture failure evidence; no device is loaded."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def capture_module(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_hbg_capture_evidence", _ROOT / "tests/st/a2a3/kernel_capture/test_hbg_capture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    return module


@pytest.mark.parametrize("returncode", [0, 1])
def test_failed_replay_preserves_native_and_device_logs(capture_module, tmp_path, returncode):
    driver = tmp_path / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "logs = pathlib.Path(os.environ['ASCEND_PROCESS_LOG_PATH']) / 'debug/device-0'\n"
        "logs.mkdir(parents=True, mode=0o700)\n"
        "(logs / 'device.log').write_text('TIMEOUT_EXIT after_idle_iterations=1\\n')\n"
        "(logs / 'device.log').chmod(0o600)\n"
        "print('fault-replay=1 sync=507011')\n"
        "print('fault-replay=1 executed=16')\n"
        "print('fault-replay=2 sync=507046')\n"
        "print('native failure', file=sys.stderr)\n"
        f"sys.exit({returncode})\n"
    )
    driver.chmod(0o755)
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "0"))
    for _ in range(2):
        with pytest.raises(AssertionError):
            capture_module.test_hbg_model_replays_after_host_finalize((tmp_path, [str(driver)]), request, 12)
    runs = list((tmp_path / "outputs/hbg-capture").glob("scenario-12-*"))
    assert len(runs) == 2
    assert (runs[0] / "stdout.log").read_bytes() == (runs[1] / "stdout.log").read_bytes()
    assert "fault-replay=1 executed=16" in (runs[0] / "stdout.log").read_text()
    assert "fault-replay=2 executed" not in (runs[0] / "stdout.log").read_text()
    assert (runs[0] / "stderr.log").read_text() == "native failure\n"
    assert (runs[0] / "ascend/debug/device-0/device.log").read_text() == "TIMEOUT_EXIT after_idle_iterations=1\n"
    for path in (runs[0], *runs[0].rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o755 if path.is_dir() else 0o644)


def test_timed_out_replay_keeps_partial_streams(capture_module, tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        if "stdout" in kwargs:
            kwargs["stdout"].write("fault-replay=1 sync=507011\n")
            kwargs["stderr"].write("pending second replay\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(capture_module.subprocess, "run", timeout)
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "0"))
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        capture_module.test_hbg_model_replays_after_host_finalize((tmp_path, ["driver"]), request, 12)
    assert failure.value.timeout == 180
    runs = list((tmp_path / "outputs/hbg-capture").glob("scenario-12-*"))
    assert len(runs) == 1
    assert (runs[0] / "stdout.log").read_text() == "fault-replay=1 sync=507011\n"
    assert (runs[0] / "stderr.log").read_text() == "pending second replay\n"


@pytest.mark.parametrize("operation", ["chmod", "is_dir", "is_symlink", "rglob", "rglob_iter"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_normalization_error_preserves_primary_result(
    capture_module, tmp_path, monkeypatch, caplog, operation, timed_out
):
    run = tmp_path / "run"
    run.mkdir()
    (run / "other.log").write_text("other")
    (run / "other.log").chmod(0o600)
    method = "rglob" if operation == "rglob_iter" else operation
    original = getattr(Path, method)
    injected = OSError(f"injected {operation} failure")

    def fail_normalization(path, *args, **kwargs):
        if path == run:
            if operation == "rglob_iter":

                def interrupted_walk():
                    yield run / "other.log"
                    raise injected

                return interrupted_walk()
            raise injected
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, fail_normalization)
    command = [
        sys.executable,
        "-c",
        "import sys; print('partial stdout'); print('partial stderr', file=sys.stderr); sys.exit(7)",
    ]
    timeout = subprocess.TimeoutExpired(command, 180)
    if timed_out:

        def expire(command, **kwargs):
            kwargs["stdout"].write("partial stdout\n")
            kwargs["stderr"].write("partial stderr\n")
            raise timeout

        monkeypatch.setattr(capture_module.subprocess, "run", expire)
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            capture_module._run_capture_probe(command, run, {})
        assert caught.value is timeout
    else:
        result = capture_module._run_capture_probe(command, run, {})
        assert result.returncode == 7
        assert result.stdout == "partial stdout\n"
        assert result.stderr == "partial stderr\n"
    assert (run / "stdout.log").read_text() == "partial stdout\n"
    assert (run / "stderr.log").read_text() == "partial stderr\n"
    if operation in ("chmod", "is_symlink", "rglob_iter"):
        assert (run / "other.log").stat().st_mode & 0o777 == 0o644
    assert str(injected) in caplog.text
    assert "permission normalization" in caplog.text


def test_normalization_skips_symlink(capture_module, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    target = tmp_path / "external.log"
    target.write_text("external")
    target.chmod(0o600)
    (run / "linked.log").symlink_to(target)
    result = capture_module._run_capture_probe([sys.executable, "-c", "pass"], run, {})
    assert result.returncode == 0
    assert target.stat().st_mode & 0o777 == 0o600
