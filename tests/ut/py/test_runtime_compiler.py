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


def test_cuda_runtime_compiler_accepts_device_target(tmp_path, monkeypatch):
    from simpler_setup.runtime_compiler import BuildTarget, RuntimeCompiler  # noqa: PLC0415
    from simpler_setup.toolchain import GxxToolchain  # noqa: PLC0415

    compiler = object.__new__(RuntimeCompiler)
    compiler.device_target = BuildTarget(
        toolchain=GxxToolchain(),
        root_dir="/tmp/cuda-device-platform",
        binary_name="libcuda_device_runtime.so",
    )

    captured = {}

    def fake_gen_cmake_args(include_dirs, source_dirs):
        captured["include_dirs"] = include_dirs
        captured["source_dirs"] = source_dirs
        return ["-DDEVICE=1"]

    compiler.device_target.gen_cmake_args = fake_gen_cmake_args
    built_binary = tmp_path / "build" / "libcuda_device_runtime.so"
    built_binary.parent.mkdir()
    built_binary.write_bytes(b"device")
    run_args = {}

    def fake_run_compilation(source, args, binary, platform, build_dir):
        run_args.update(
            {
                "source": source,
                "args": args,
                "binary": binary,
                "platform": platform,
                "build_dir": build_dir,
            }
        )
        return built_binary

    monkeypatch.setattr(
        compiler,
        "_run_compilation",
        fake_run_compilation,
    )

    output_dir = tmp_path / "out"
    result = compiler.compile(
        "device",
        ["/tmp/include"],
        ["/tmp/source"],
        build_dir=str(tmp_path / "cache"),
        output_dir=output_dir,
    )

    assert captured == {"include_dirs": ["/tmp/include"], "source_dirs": ["/tmp/source"]}
    assert run_args == {
        "source": "/tmp/cuda-device-platform",
        "args": ["-DDEVICE=1"],
        "binary": "libcuda_device_runtime.so",
        "platform": "DEVICE",
        "build_dir": str(tmp_path / "cache" / "device"),
    }
    assert result == output_dir / "libcuda_device_runtime.so"
    assert result.read_bytes() == b"device"
