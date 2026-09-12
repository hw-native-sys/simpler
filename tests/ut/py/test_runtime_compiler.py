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
from pathlib import Path

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


@pytest.mark.parametrize("platform", ["a2a3", "a5", "a2a3sim", "a5sim"])
@pytest.mark.parametrize("requested", [False, True])
def test_kernel_aicore_side_product_staging(tmp_path, monkeypatch, platform, requested):
    """Collect the requested onboard ELF without replacing the program result."""
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    compiler = runtime_compiler.RuntimeCompiler.__new__(runtime_compiler.RuntimeCompiler)
    compiler.platform = platform
    compiler._sanitizers = ""
    binary_name = "libaicore_kernel.so" if platform.endswith("sim") else "aicore_kernel.o"
    compiler.aicore_target = runtime_compiler.BuildTarget(_StubToolchain(is_host=False), str(tmp_path), binary_name)

    def fake_build(source, args, name, *, platform, build_dir):
        binary = Path(build_dir) / name
        binary.write_bytes(b"program")
        (Path(build_dir) / "aicore_kernel_mode.o").write_bytes(b"kernel")
        return str(binary)

    monkeypatch.setattr(compiler, "_run_compilation", fake_build)
    output_dir = tmp_path / "out"
    result = compiler.compile(
        "aicore",
        [],
        [],
        build_dir=str(tmp_path / "cache"),
        output_dir=output_dir,
        kernel_aicore_dest=output_dir if requested else None,
    )

    assert result == output_dir / binary_name
    assert result.read_bytes() == b"program"
    staged = output_dir / "aicore_kernel_mode.o"
    if requested and not platform.endswith("sim"):
        assert staged.read_bytes() == b"kernel"
    else:
        assert not staged.exists()


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_missing_requested_kernel_aicore_is_not_masked_by_staged_file(tmp_path, monkeypatch, platform):
    """A stale exported ELF cannot make an incomplete new build appear successful."""
    from simpler_setup import runtime_compiler  # noqa: PLC0415

    compiler = runtime_compiler.RuntimeCompiler.__new__(runtime_compiler.RuntimeCompiler)
    compiler.platform = platform
    compiler._sanitizers = ""
    compiler.aicore_target = runtime_compiler.BuildTarget(
        _StubToolchain(is_host=False), str(tmp_path), "aicore_kernel.o"
    )

    def fake_build(source, args, name, *, platform, build_dir):
        binary = Path(build_dir) / name
        binary.write_bytes(b"program")
        return str(binary)

    monkeypatch.setattr(compiler, "_run_compilation", fake_build)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    staged = output_dir / "aicore_kernel_mode.o"
    staged.write_bytes(b"old kernel")
    with pytest.raises(FileNotFoundError, match="TMR kernel-mode AICore binary not found"):
        compiler.compile(
            "aicore",
            [],
            [],
            build_dir=str(tmp_path / "cache"),
            output_dir=output_dir,
            kernel_aicore_dest=output_dir,
        )
    assert staged.read_bytes() == b"old kernel"
