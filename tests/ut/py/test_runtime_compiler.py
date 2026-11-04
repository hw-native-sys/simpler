# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for RuntimeCompiler's CMake invocation."""

import logging

import pytest

from simpler_setup.toolchain import Toolchain


class _StubToolchain(Toolchain):
    def __init__(self, *, is_host):
        self.is_host = is_host

    def get_compile_flags(self, **kwargs):
        return []

    def get_cmake_args(self):
        return []


def test_host_build_target_always_receives_shared_cmake_dir(tmp_path):
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    target = runtime_compiler.BuildTarget(_StubToolchain(is_host=True), str(tmp_path), "libtest.so")

    args = target.gen_cmake_args([], [])

    assert f"-DSIMPLER_CMAKE_DIR={runtime_compiler.PROJECT_ROOT / 'cmake'}" in args
    assert not any(arg.startswith("-DSIMPLER_SANITIZERS=") for arg in args)


def test_device_build_target_also_receives_shared_cmake_dir(tmp_path):
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    target = runtime_compiler.BuildTarget(_StubToolchain(is_host=False), str(tmp_path), "kernel.o")

    args = target.gen_cmake_args([], [])

    # Device targets load cmake/profiling_config.cmake for the generated profiling
    # header, so they need the shared module path too. The four profiling macros must
    # hold one value across a platform: SIMPLER_DFX adds fields to a struct the host
    # and the AICPU both compile.
    assert f"-DSIMPLER_CMAKE_DIR={runtime_compiler.PROJECT_ROOT / 'cmake'}" in args
    # Sanitizers stay host-only: a device toolchain cannot carry the host runtime.
    assert not any(arg.startswith("-DSIMPLER_SANITIZERS=") for arg in args)


def test_build_output_is_verbose_only_for_debug_logging(tmp_path, monkeypatch, caplog):
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    RuntimeCompiler = runtime_compiler.RuntimeCompiler
    compiler = RuntimeCompiler.__new__(RuntimeCompiler)
    binary_path = tmp_path / "libtest.so"
    binary_path.touch()
    calls = []

    def record_step(cmd, cwd, platform, step_name):
        calls.append((cmd, cwd, platform, step_name))

    monkeypatch.setattr(compiler, "_run_build_step", record_step)

    compiler._run_compilation(str(tmp_path), [], binary_path.name, build_dir=str(tmp_path))
    assert "--verbose" not in calls[-1][0]

    calls.clear()
    with caplog.at_level(logging.DEBUG, logger=runtime_compiler.logger.name):
        compiler._run_compilation(str(tmp_path), [], binary_path.name, build_dir=str(tmp_path))
    assert "--verbose" in calls[-1][0]
    assert calls[-1][1] == str(tmp_path)


def test_instance_cache_is_scoped_by_sanitizer_configuration(monkeypatch):
    from simpler_setup import runtime_compiler, toolchain  # noqa: PLC0415

    RuntimeCompiler = runtime_compiler.RuntimeCompiler
    monkeypatch.setattr(RuntimeCompiler, "_instances", {})
    monkeypatch.setattr(RuntimeCompiler, "_ensure_host_compilers", lambda self: None)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)

    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", "")
    plain = RuntimeCompiler.get_instance("a2a3sim")
    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", "address")
    sanitized = RuntimeCompiler.get_instance("a2a3sim")

    assert plain is not sanitized
    assert plain.host_target.toolchain.cxx_path == "g++"
    assert sanitized.host_target.toolchain.cxx_path == "g++-15"


@pytest.mark.parametrize("budget, expected", [("2", 2), ("128", 32), (None, 32), ("", 32)])
def test_build_respects_configured_job_budget(tmp_path, monkeypatch, budget, expected):
    from simpler_setup import runtime_build_budget, runtime_compiler  # noqa: PLC0415

    monkeypatch.setattr(runtime_build_budget.multiprocessing, "cpu_count", lambda: 64)
    if budget is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", budget)
    compiler = runtime_compiler.RuntimeCompiler.__new__(runtime_compiler.RuntimeCompiler)
    binary = tmp_path / "libtest.so"
    binary.touch()
    calls = []
    monkeypatch.setattr(compiler, "_run_build_step", lambda cmd, *_: calls.append(cmd))

    compiler._run_compilation(str(tmp_path), [], binary.name, build_dir=str(tmp_path))

    build = calls[-1]
    assert build[build.index("--parallel") + 1] == str(expected)


@pytest.mark.parametrize("budget", ["0", "-1", "invalid"])
def test_invalid_build_job_budget_fails_before_launch(tmp_path, monkeypatch, budget):
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", budget)
    compiler = runtime_compiler.RuntimeCompiler.__new__(runtime_compiler.RuntimeCompiler)
    calls = []
    monkeypatch.setattr(compiler, "_run_build_step", lambda *args: calls.append(args))

    with pytest.raises(ValueError, match="CMAKE_BUILD_PARALLEL_LEVEL"):
        compiler._run_compilation(str(tmp_path), [], "libtest.so", build_dir=str(tmp_path))

    assert calls == []
