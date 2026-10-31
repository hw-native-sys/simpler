# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the runtime pre-build entry point."""

import pytest


def test_build_all_prebuilds_sim_context_once(monkeypatch, tmp_path):
    from simpler_setup import build_runtimes  # noqa: PLC0415
    from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

    calls = []

    class FakeRuntimeBuilder:
        _LIB_DIR = None
        _CACHE_DIR = None

        def __init__(self, platform):
            self.platform = platform

        def ensure_sim_context(self, *, build):
            calls.append(("sim_context", self.platform, build))

        def get_binaries(self, name, *, build, build_shared, profiling_config):
            calls.append(("runtime", self.platform, name, build, build_shared, profiling_config))

    monkeypatch.setattr(build_runtimes, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(build_runtimes, "discover_runtimes", lambda arch: ["first", "second"])
    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", [])

    build_runtimes.build_all(
        lib_dir=tmp_path / "lib",
        cache_dir=tmp_path / "cache",
        platforms=["a2a3sim"],
    )

    assert calls.count(("sim_context", "a2a3sim", True)) == 1
    assert sorted(call for call in calls if call[0] == "runtime") == [
        ("runtime", "a2a3sim", "first", True, False, None),
        ("runtime", "a2a3sim", "second", True, False, None),
    ]


@pytest.mark.parametrize(
    "platforms",
    [
        ["a2a3", "a2a3sim"],
        ["a2a3sim", "a2a3"],
        ["a5", "a5sim"],
        ["a5sim", "a5"],
    ],
)
def test_mixed_sanitizer_builds_use_one_host_toolchain_regardless_of_order(monkeypatch, tmp_path, platforms):
    from simpler_setup import build_runtimes, pto_isa, runtime_compiler, toolchain  # noqa: PLC0415

    RuntimeCompiler = runtime_compiler.RuntimeCompiler
    calls = []

    class FakeDeviceToolchain:
        def __init__(self, *args, **kwargs):
            pass

    class FakeRuntimeBuilder:
        _LIB_DIR = None
        _CACHE_DIR = None

        def __init__(self, platform):
            self.platform = platform
            self.compiler = RuntimeCompiler.get_instance(platform)

        def ensure_sim_context(self, *, build):
            calls.append(("sim_context", self.platform, self.compiler.host_target.toolchain.cxx_path))

        def get_binaries(self, name, *, build, build_shared, profiling_config):
            calls.append(("runtime", self.platform, self.compiler.host_target.toolchain.cxx_path))

    monkeypatch.setattr(RuntimeCompiler, "_instances", {})
    monkeypatch.setattr(RuntimeCompiler, "_sanitizers", "")
    monkeypatch.setattr(RuntimeCompiler, "_ensure_host_compilers", lambda self: None)
    monkeypatch.setattr(runtime_compiler.env_manager, "ensure", lambda name: None)
    monkeypatch.setattr(runtime_compiler, "CCECToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(runtime_compiler, "Aarch64GxxToolchain", FakeDeviceToolchain)
    monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose: tmp_path / "pto-isa")
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)
    monkeypatch.setattr(build_runtimes, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(build_runtimes, "discover_runtimes", lambda arch: ["runtime"])
    monkeypatch.setattr(build_runtimes, "platform_embeds_pto_isa", lambda platform: False)

    build_runtimes.build_all(
        lib_dir=tmp_path / "lib",
        cache_dir=tmp_path / "cache",
        platforms=platforms,
        sanitizer="asan",
    )

    assert [call for call in calls if call[0] == "sim_context"] == [
        ("sim_context", next(p for p in platforms if p.endswith("sim")), "g++-15")
    ]
    assert {compiler for _, _, compiler in calls} == {"g++-15"}


def test_profiling_config_completes_defaults_and_validates_dependencies():
    from simpler_setup.build_runtimes import _profiling_config  # noqa: PLC0415

    assert _profiling_config(
        {
            "SIMPLER_DFX": None,
            "SIMPLER_ORCH_PROFILING": 1,
            "SIMPLER_SCHED_PROFILING": None,
            "SIMPLER_TENSORMAP_PROFILING": None,
        }
    ) == {
        "SIMPLER_DFX": "1",
        "SIMPLER_ORCH_PROFILING": "1",
        "SIMPLER_SCHED_PROFILING": "0",
        "SIMPLER_TENSORMAP_PROFILING": "0",
    }

    with pytest.raises(ValueError, match="TENSORMAP_PROFILING requires"):
        _profiling_config(
            {
                "SIMPLER_DFX": 1,
                "SIMPLER_ORCH_PROFILING": 0,
                "SIMPLER_SCHED_PROFILING": 0,
                "SIMPLER_TENSORMAP_PROFILING": 1,
            }
        )


@pytest.mark.parametrize("budget", [None, "", "1", "2", "8", "32"])
def test_nested_runtime_builds_share_budget_and_preserve_parallelism(monkeypatch, tmp_path, budget):
    import threading  # noqa: PLC0415

    from simpler_setup import build_runtimes, runtime_build_budget, runtime_compiler, toolchain  # noqa: PLC0415
    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    if budget is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", budget)
    monkeypatch.setattr(runtime_build_budget.multiprocessing, "cpu_count", lambda: 64)
    Compiler = runtime_compiler.RuntimeCompiler
    monkeypatch.setattr(Compiler, "_instances", {})
    monkeypatch.setattr(Compiler, "_sanitizers", "")
    monkeypatch.setattr(Compiler, "_ensure_host_compilers", lambda self: None)
    monkeypatch.setattr(toolchain, "_is_gcc", lambda path: True)
    monkeypatch.setattr(RuntimeBuilder, "ensure_sim_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(RuntimeBuilder, "_LIB_DIR", tmp_path / "lib")
    monkeypatch.setattr(RuntimeBuilder, "_CACHE_DIR", tmp_path / "cache")

    platforms = ["a2a3sim", "a5sim"]
    expected = {
        (platform, runtime, target)
        for platform in platforms
        for runtime in build_runtimes.discover_runtimes(platform.removesuffix("sim"))
        for target in ["host", "aicpu", "aicore"]
    }
    parallel_targets = min(int(budget), len(expected)) if budget else len(expected)
    # All expected targets must reach CMake together by default. With a small
    # explicit limit, only that many may hold compiler jobs simultaneously.
    started = threading.Barrier(parallel_targets)
    active_jobs = peak_jobs = active_builds = peak_builds = 0
    calls = []
    lock = threading.Lock()

    def build_step(self, command, cwd, platform, step_name):
        nonlocal active_jobs, peak_jobs, active_builds, peak_builds
        if "--build" not in command:
            return
        jobs = int(command[command.index("--parallel") + 1])
        with lock:
            active_jobs += jobs
            active_builds += 1
            peak_jobs = max(peak_jobs, active_jobs)
            peak_builds = max(peak_builds, active_builds)
        try:
            # The last, incomplete group need not rendezvous.
            if len(expected) % parallel_targets == 0:
                started.wait(timeout=5)
        finally:
            with lock:
                active_jobs -= jobs
                active_builds -= 1

    def compile_target(self, target, *args, **kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        binary = output_dir / f"lib{target}.so"
        binary.touch()
        with lock:
            calls.append((self.platform, output_dir.name, target))
        return self._run_compilation(str(output_dir), [], binary.name, build_dir=str(output_dir))

    monkeypatch.setattr(Compiler, "_run_build_step", build_step)
    monkeypatch.setattr(Compiler, "compile", compile_target)

    build_runtimes.build_all(tmp_path / "lib", tmp_path / "cache", platforms=platforms)

    assert set(calls) == expected
    assert len(calls) == len(expected)
    assert all(
        (tmp_path / "lib" / platform.removesuffix("sim") / "sim" / runtime / f"lib{target}.so").is_file()
        for platform, runtime, target in expected
    )
    if budget:
        assert peak_jobs <= int(budget)
        if int(budget) in (1, 2):
            assert peak_builds == int(budget)
        elif int(budget) >= len(expected):
            assert peak_builds == len(expected)
            assert peak_jobs == int(budget)
    else:
        assert peak_builds == len(expected)
        assert peak_jobs == 32 * len(expected)
