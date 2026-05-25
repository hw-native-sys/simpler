# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for RuntimeCompiler build-cache retry behavior."""

from __future__ import annotations

import logging
import subprocess


def test_cached_cmake_retry_does_not_log_first_failure_as_error(tmp_path, monkeypatch, caplog):
    from simpler_setup import runtime_compiler  # noqa: PLC0415
    from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

    compiler = object.__new__(RuntimeCompiler)
    build_dir = tmp_path / "aicpu"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("stale cache\n")
    attempts = []

    def fake_run(cmd, cwd, check, capture_output, text):  # noqa: ARG001
        attempts.append((cmd, cwd))
        if cmd[:2] == ["cmake", "cuda-aicpu-platform"] and len(attempts) == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stale source dir")
        if cmd[:2] == ["cmake", "cuda-aicpu-platform"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["cmake", "--build"]:
            (build_dir / "libaicpu_kernel.so").write_bytes(b"placeholder")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(runtime_compiler.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime_compiler.multiprocessing, "cpu_count", lambda: 64)

    with caplog.at_level(logging.WARNING, logger="simpler_setup.runtime_compiler"):
        result = compiler._run_compilation(
            "cuda-aicpu-platform",
            ["-DCUSTOM_SOURCE_DIRS=/tmp/stale"],
            "libaicpu_kernel.so",
            platform="AICPU",
            build_dir=str(build_dir),
        )

    assert result == build_dir / "libaicpu_kernel.so"
    assert [cmd for cmd, _ in attempts] == [
        ["cmake", "cuda-aicpu-platform", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCUSTOM_SOURCE_DIRS=/tmp/stale"],
        ["cmake", "cuda-aicpu-platform", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCUSTOM_SOURCE_DIRS=/tmp/stale"],
        ["cmake", "--build", ".", "--parallel", "32", "--verbose"],
    ]
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any("clearing cache and retrying once" in record.message for record in caplog.records)
