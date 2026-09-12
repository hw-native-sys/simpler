# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for KernelCompiler host toolchain selection."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5", "a5sim"])
@pytest.mark.parametrize(("sanitizers", "expected_cxx"), [("", "g++"), ("address", "g++-15")])
def test_host_orchestration_matches_runtime_sanitizer_toolchain(monkeypatch, platform, sanitizers, expected_cxx):
    from simpler_setup import kernel_compiler, toolchain  # noqa: PLC0415

    class FakeDeviceToolchain:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kernel_compiler.KernelCompiler, "_sanitizers", sanitizers)
    monkeypatch.setattr(kernel_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(kernel_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(kernel_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)

    compiler = kernel_compiler.KernelCompiler(platform)

    assert compiler.host_gxx.cxx_path == expected_cxx


@pytest.mark.parametrize(
    ("target_platform", "target_runtime", "expects_host_logger"),
    [
        ("a2a3sim", "host_build_graph", True),
        ("a2a3sim", "tensormap_and_ringbuffer", True),
        ("a2a3", "host_build_graph", True),
        ("a2a3", "tensormap_and_ringbuffer", False),
        ("a5sim", "host_build_graph", True),
        ("a5sim", "tensormap_and_ringbuffer", True),
        ("a5", "host_build_graph", True),
        ("a5", "tensormap_and_ringbuffer", False),
    ],
)
def test_host_orchestration_is_a_self_contained_log_consumer(
    monkeypatch, target_platform, target_runtime, expects_host_logger
):
    from simpler_setup import kernel_compiler, toolchain  # noqa: PLC0415

    class FakeDeviceToolchain:
        is_host = False

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kernel_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(kernel_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(kernel_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)

    compiler = kernel_compiler.KernelCompiler(target_platform)
    include_dirs, sources = compiler.get_orchestration_cache_inputs(target_runtime)
    source_names = {source.rsplit("/", 1)[-1] for source in sources}

    if expects_host_logger:
        assert {"host_log.cpp", "unified_log_host.cpp"} <= source_names
        assert any(include_dir.endswith("src/common/log/include") for include_dir in include_dirs)
        assert "-pthread" in compiler._orchestration_link_flags(compiler._orchestration_toolchain(target_runtime))
    else:
        assert "host_log.cpp" not in source_names
        assert "unified_log_host.cpp" not in source_names
        assert "-pthread" not in compiler._orchestration_link_flags(compiler._orchestration_toolchain(target_runtime))

    # The Graph recorder pool belongs to the runtime, and these sources are shared with the
    # aicore and aicpu targets, so nothing that creates host threads may be compiled into
    # the orchestration .so -- see the note on "orchestration" in each runtime's
    # build_config.py. The pool used to live here behind a framework_* entry point.
    assert "graph_recorder_pool.cpp" not in source_names


def test_direct_compiler_command_uses_checkout_relative_paths(monkeypatch, tmp_path):
    from simpler_setup import kernel_compiler  # noqa: PLC0415
    from simpler_setup.environment import PROJECT_ROOT  # noqa: PLC0415

    compiler = kernel_compiler.KernelCompiler("a2a3sim")
    source = PROJECT_ROOT / "simpler_setup" / "incore" / "pipe_sync.h"
    captured = {}
    monkeypatch.setattr(compiler, "_make_temp_path", lambda **_kwargs: str(tmp_path / "kernel.so"))

    def capture_compile(cmd, *_args, **_kwargs):
        captured["cmd"] = cmd
        return b"compiled"

    monkeypatch.setattr(compiler, "_compile_to_bytes", capture_compile)

    assert compiler._compile_incore_sim(str(source), core_type="aiv") == b"compiled"
    assert captured["cmd"][-1] == str(Path("simpler_setup") / "incore" / "pipe_sync.h")
    assert f"-I{Path('simpler_setup') / 'incore'}" in captured["cmd"]


def test_compiler_subprocess_runs_from_checkout_root(monkeypatch):
    from simpler_setup import kernel_compiler  # noqa: PLC0415
    from simpler_setup.environment import PROJECT_ROOT  # noqa: PLC0415

    compiler = kernel_compiler.KernelCompiler("a2a3sim")
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(kernel_compiler.subprocess, "run", fake_run)
    compiler._run_subprocess(["compiler"], "test")

    assert captured["cwd"] == PROJECT_ROOT


def test_compile_cache_token_preserves_every_incore_toolchain_token(monkeypatch):
    from simpler_setup import kernel_compiler  # noqa: PLC0415

    compiler = kernel_compiler.KernelCompiler.__new__(kernel_compiler.KernelCompiler)
    compiler._orchestration_toolchain = lambda _runtime: SimpleNamespace(cxx_path="orch-cxx")
    compiler._orchestration_compile_flags = lambda _toolchain: ["-orch"]
    compiler._orchestration_link_flags = lambda _toolchain: ["-link"]
    tokens = {
        "aic": {"core_type": "aic", "identity": {"name": "ccec"}, "flags": ["-aic"], "linker": "ld-aic"},
        "aiv": {"core_type": "aiv", "identity": {"name": "aicore-cc"}, "flags": ["-aiv"], "linker": "ld-aiv"},
    }
    compiler.incore_compile_cache_token = lambda core_type: tokens[core_type]
    monkeypatch.setattr(kernel_compiler, "_executable_cache_identity", lambda _path: {"name": "orch-cxx"})

    token = compiler.compile_cache_token("host_build_graph", ["aiv", "aic"])

    assert token["incore"] == {"variants": [tokens["aic"], tokens["aiv"]]}


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
@pytest.mark.parametrize("core_type", ["aic", "aiv"])
@pytest.mark.parametrize("pto_isa_root", [None, "isa"])
def test_simulator_ffts_header_is_arch_specific(monkeypatch, tmp_path, platform, core_type, pto_isa_root):
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    compiler = KernelCompiler(platform)
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" void kernel_entry() {}\n')
    captured = {}

    def capture_compile(command, *_args, **_kwargs):
        captured["command"] = command
        return b"compiled"

    monkeypatch.setattr(compiler, "_compile_to_bytes", capture_compile)
    assert compiler._compile_incore_sim(str(source), core_type=core_type, pto_isa_root=pto_isa_root) == b"compiled"
    command = captured["command"]
    assert ("-include" in command) == (platform == "a2a3sim")
    assert ("-D__DAV_CUBE__" in command) == (core_type == "aic")
    assert ("-D__DAV_VEC__" in command) == (core_type == "aiv")
    if platform == "a2a3sim":
        assert command[command.index("-include") + 1].endswith("incore/ffts_sim.h")


def test_simulator_ffts_header_changes_invalidate_compiled_artifacts(monkeypatch, tmp_path):
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    header = tmp_path / "ffts_sim.h"
    compiler = KernelCompiler("a2a3sim")
    monkeypatch.setattr(compiler, "_sim_ffts_header", lambda: header)
    header.write_text("first implementation")
    original = compiler.incore_compile_cache_token("aic")
    header.write_text("second implementation")
    updated = compiler.incore_compile_cache_token("aic")
    assert original["sim_ffts"] != updated["sim_ffts"]
    assert KernelCompiler("a5sim").incore_compile_cache_token("aic")["sim_ffts"] is None
