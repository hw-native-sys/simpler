# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for RuntimeBuilder class."""

import logging
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# --- Discovery tests (no compilation needed) ---


class TestRuntimeBuilderDiscovery:
    """Test runtime discovery from src/runtime/ subdirectories."""

    def test_discovers_real_runtimes(self, default_test_platform):
        """RuntimeBuilder discovers host_build_graph from the real project tree."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        runtimes = builder.list_runtimes()
        assert "host_build_graph" in runtimes

    def test_runtime_dir_resolves_to_project_root(self, default_test_platform, test_arch):
        """runtime_dir resolves to src/{arch}/runtime/ under the project root."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.runtime_dir == builder.runtime_root / "src" / test_arch / "runtime"
        assert builder.runtime_dir.is_dir()

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_discovers_configs_in_runtime_dir(
        self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch
    ):
        """RuntimeBuilder discovers implementations in the runtime directory."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        # Set up fake runtime tree with architecture-specific structure
        rt_dir = tmp_path / "src" / test_arch / "runtime" / "my_runtime"
        rt_dir.mkdir(parents=True)
        (rt_dir / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["my_runtime"]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_ignores_dirs_without_build_config(
        self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch
    ):
        """Directories without build_config.py are not listed."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        (rt_dir / "has_config").mkdir(parents=True)
        (rt_dir / "has_config" / "build_config.py").write_text("BUILD_CONFIG = {}\n")
        (rt_dir / "no_config").mkdir(parents=True)
        # __pycache__ should also be ignored
        (rt_dir / "__pycache__").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["has_config"]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_empty_runtime_dir(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Empty src/{arch}/runtime/ directory yields no runtimes."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_missing_runtime_dir(self, MockCompiler, tmp_path, monkeypatch, default_test_platform):
        """Non-existent src/{arch}/runtime/ directory yields no runtimes."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_multiple_runtimes_sorted(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Multiple implementations are returned in sorted order."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        for name in ["zeta", "alpha", "beta"]:
            d = rt_dir / name
            d.mkdir(parents=True)
            (d / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["alpha", "beta", "zeta"]


# --- Error handling tests ---


class TestRuntimeBuilderErrors:
    """Test get_binaries() error handling without invoking real compilation."""

    def test_unknown_runtime_raises(self, default_test_platform):
        """get_binaries() raises ValueError for a non-existent runtime name."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="is not available for platform"):
            builder.get_binaries("nonexistent_runtime", build=True)

    def test_unknown_runtime_lists_available(self, default_test_platform):
        """ValueError message includes available runtime names."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="host_build_graph"):
            builder.get_binaries("nonexistent_runtime", build=True)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_empty_registry_shows_none(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """ValueError message shows '(none)' when no runtimes exist."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)
        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match=r"\(none\)"):
            builder.get_binaries("anything", build=True)


class TestRuntimeBuilderPtoIsaValidation:
    """Test PTO-ISA metadata validation is scoped to embedding runtimes."""

    @pytest.mark.parametrize(
        ("platform", "should_validate", "expected_key"),
        [
            ("a2a3", True, "a2a3/onboard/test_rt"),
            ("a2a3sim", False, None),
            ("a5", True, "a5/onboard/test_rt"),
            ("a5sim", False, None),
        ],
    )
    def test_pto_isa_validation_scoped_to_embedding_platforms(
        self, tmp_path, monkeypatch, platform, should_validate, expected_key
    ):
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.platform_info import TARGETS, parse_platform  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        class _Target:
            def __init__(self, name):
                self._name = name

            def get_binary_name(self):
                return f"lib{self._name}.so"

        calls = []

        def _fail_if_called(lib_dir, runtime_key=None):
            calls.append((lib_dir, runtime_key))
            raise RuntimeError("pto-isa validation called")

        monkeypatch.setattr(pto_isa, "validate_runtime_pto_isa_current_pin", _fail_if_called)

        builder = RuntimeBuilder.__new__(RuntimeBuilder)
        builder.platform = platform
        builder._arch, builder._variant = parse_platform(platform)
        builder._LIB_DIR = tmp_path / "lib"
        builder._runtime_compiler = type(
            "Compiler",
            (),
            {f"{target}_target": _Target(target) for target in TARGETS},
        )()

        if should_validate:
            with pytest.raises(RuntimeError, match="pto-isa validation called"):
                builder._lookup_binaries("test_rt", tmp_path / "out")
            assert calls == [(builder._LIB_DIR, expected_key)]
        else:
            with pytest.raises(FileNotFoundError, match="Pre-built runtime binaries not found"):
                builder._lookup_binaries("test_rt", tmp_path / "out")
            assert calls == []


# --- Build integration tests (mocked compilation) ---


class TestRuntimeBuilderGetBinaries:
    """Test get_binaries(build=True) logic with mocked RuntimeCompiler."""

    @pytest.fixture(autouse=True)
    def _patch_runtime_root(self, monkeypatch, tmp_path):
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

    def _make_runtime(self, tmp_path, test_arch):
        """Create a fake runtime with a valid build_config.py."""
        rt_dir = tmp_path / "src" / test_arch / "runtime" / "test_rt"
        for sub in ["aicore", "aicpu", "host", "runtime"]:
            (rt_dir / sub).mkdir(parents=True)

        config_content = textwrap.dedent("""\
            BUILD_CONFIG = {
                "aicore": {
                    "include_dirs": ["aicore", "runtime"],
                    "source_dirs": ["runtime"]
                },
                "aicpu": {
                    "include_dirs": ["runtime"],
                    "source_dirs": ["aicpu", "runtime"]
                },
                "host": {
                    "include_dirs": ["runtime"],
                    "source_dirs": ["host", "runtime"]
                }
            }
        """)
        (rt_dir / "build_config.py").write_text(config_content)
        return rt_dir

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_returns_runtime_binaries(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """get_binaries(build=True) returns RuntimeBinaries with three paths."""
        from simpler_setup.runtime_builder import RuntimeBinaries, RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        # compile() returns Path when output_dir is set
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        result = builder.get_binaries("test_rt", build=True)

        assert isinstance(result, RuntimeBinaries)
        assert result.host_path.name == "libhost.so"
        assert result.aicpu_path.name == "libaicpu.so"
        assert result.aicore_path.name == "libaicore.so"

    def _isolated_builder(self, MockCompiler, tmp_path, monkeypatch, platform):
        """A builder whose staging tree is tmp_path rather than the real build/lib.

        ``_LIB_DIR`` is a class attribute bound to the real PROJECT_ROOT at import,
        so the ``_patch_runtime_root`` fixture does not redirect it. The SDMA
        warmup tests both read and write that tree, so they must shadow it per
        instance or they would inspect (and overwrite) the developer's actual
        staged ELF.
        """
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "a" * 40)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(pto_isa, "write_pto_isa_build_metadata", lambda *args: None)

        builder = RuntimeBuilder(platform=platform)
        builder._LIB_DIR = tmp_path / "lib"
        return builder, mock_instance

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_sdma_warmup_path_is_none_when_nothing_staged(self, MockCompiler, tmp_path, monkeypatch):
        """A missing warmup ELF degrades to None instead of a nonexistent path.

        Unlike the dispatcher, an absent warmup ELF is not fatal: it only costs
        first-TPREFETCH_ASYNC latency, so it must not raise FileNotFoundError.
        """
        self._make_runtime(tmp_path, "a2a3")
        builder, _ = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a2a3")

        assert builder.get_binaries("test_rt", build=True).sdma_warmup_path is None

    def _make_warmup_source(self, tmp_path, arch):
        """Create the arch's warmup kernel source, the 'this arch builds one' marker."""
        source = tmp_path / "src" / arch / "platform" / "onboard" / "aicore" / "sdma_warmup_kernel.cpp"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("// fake warmup kernel\n")
        return source

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_warns_when_an_arch_with_warmup_source_staged_nothing(self, MockCompiler, tmp_path, monkeypatch, caplog):
        """A build/staging regression must be visible, not just slower.

        Every consumer of the warmup ELF degrades silently when it is missing, so
        an arch that carries the source but staged no object is the one case that
        has to say something.
        """
        self._make_runtime(tmp_path, "a2a3")
        self._make_warmup_source(tmp_path, "a2a3")
        builder, _ = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a2a3")

        with caplog.at_level(logging.WARNING, logger="simpler_setup.runtime_builder"):
            assert builder.get_binaries("test_rt", build=True).sdma_warmup_path is None

        assert "SDMA warmup ELF not staged" in caplog.text

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_no_warning_for_an_arch_that_builds_no_warmup(self, MockCompiler, tmp_path, monkeypatch, caplog):
        """Absent source means the arch never builds one, which is not a regression."""
        self._make_runtime(tmp_path, "a5")
        builder, _ = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a5")

        with caplog.at_level(logging.WARNING, logger="simpler_setup.runtime_builder"):
            assert builder.get_binaries("test_rt", build=True).sdma_warmup_path is None

        assert "SDMA warmup ELF not staged" not in caplog.text

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_resolves_staged_sdma_warmup_elf(self, MockCompiler, tmp_path, monkeypatch):
        """A staged sdma_warmup_kernel.o is surfaced through RuntimeBinaries."""
        self._make_runtime(tmp_path, "a2a3")
        builder, _ = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a2a3")

        staged = builder._LIB_DIR / "a2a3" / "sdma_warmup" / "sdma_warmup_kernel.o"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"\x7fELF")

        assert builder.get_binaries("test_rt", build=True).sdma_warmup_path == staged

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_sim_never_claims_a_sdma_warmup_elf(self, MockCompiler, tmp_path, monkeypatch):
        """Sim has no device SDMA, so it ignores a staged ELF and stages nothing."""
        self._make_runtime(tmp_path, "a2a3")
        builder, mock_instance = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a2a3sim")

        staged = builder._LIB_DIR / "a2a3" / "sdma_warmup" / "sdma_warmup_kernel.o"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"\x7fELF")

        result = builder.get_binaries("test_rt", build=True)

        assert result.sdma_warmup_path is None
        aicore_call = next(call for call in mock_instance.compile.call_args_list if call.args[0] == "aicore")
        assert aicore_call.kwargs["sdma_warmup_dest"] is None

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_passes_sdma_warmup_dest_only_to_aicore(self, MockCompiler, tmp_path, monkeypatch):
        """The warmup ELF is a side product of the aicore build, so only that
        target is told where to stage it."""
        self._make_runtime(tmp_path, "a2a3")
        builder, mock_instance = self._isolated_builder(MockCompiler, tmp_path, monkeypatch, "a2a3")

        builder.get_binaries("test_rt", build=True)

        dests = {call.args[0]: call.kwargs["sdma_warmup_dest"] for call in mock_instance.compile.call_args_list}
        assert dests["aicpu"] is None
        assert dests["host"] is None
        assert dests["aicore"] == builder._LIB_DIR / "a2a3" / "sdma_warmup"

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_calls_compiler_three_times(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """get_binaries(build=True) invokes compiler.compile() exactly 3 times."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.get_binaries("test_rt", build=True)

        assert mock_instance.compile.call_count == 3
        targets = sorted(call.args[0] for call in mock_instance.compile.call_args_list)
        assert targets == ["aicore", "aicpu", "host"]
        expected_defaults = {
            "SIMPLER_DFX": "1",
            "SIMPLER_ORCH_PROFILING": "0",
            "SIMPLER_SCHED_PROFILING": "0",
            "SIMPLER_TENSORMAP_PROFILING": "0",
        }
        for call in mock_instance.compile.call_args_list:
            expected = dict(expected_defaults)
            if call.args[0] == "host":
                expected["SIMPLER_RUNTIME_NAME"] = "test_rt"
            assert call.kwargs["cmake_defines"] == expected

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_reuses_prebuilt_shared_libraries(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """A batch build can locate shared libraries without rebuilding them."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        with patch.object(builder, "ensure_sim_context", return_value=tmp_path / "libcpu_sim_context.so") as sim:
            builder.get_binaries("test_rt", build=True, build_shared=False)

        sim.assert_called_once_with(build=False)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_passes_profiling_config_to_every_target(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """All runtime binaries use one complete profiling configuration."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        config = {
            "SIMPLER_DFX": "1",
            "SIMPLER_ORCH_PROFILING": "1",
            "SIMPLER_SCHED_PROFILING": "0",
            "SIMPLER_TENSORMAP_PROFILING": "0",
        }

        RuntimeBuilder(platform=default_test_platform).get_binaries("test_rt", build=True, profiling_config=config)

        assert mock_instance.compile.call_count == 3
        for call in mock_instance.compile.call_args_list:
            expected = dict(config)
            if call.args[0] == "host":
                expected["SIMPLER_RUNTIME_NAME"] = "test_rt"
            assert call.kwargs["cmake_defines"] == expected

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_resolves_paths_relative_to_config(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """Include/source dirs are resolved relative to the build_config.py directory."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        rt_dir = self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.get_binaries("test_rt", build=True)

        # Check any call: include_dirs should be resolved absolute paths
        for call in mock_instance.compile.call_args_list:
            include_dirs = call.args[1]
            for d in include_dirs:
                assert Path(d).is_absolute()
                assert str(rt_dir.resolve()) in d

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_propagates_compiler_error(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """If RuntimeCompiler.compile() raises, get_binaries() propagates the exception."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = RuntimeError("cmake failed")

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(RuntimeError, match="cmake failed"):
            builder.get_binaries("test_rt", build=True)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_a2a3_onboard_direct_build_writes_pto_isa_metadata(self, MockCompiler, tmp_path, monkeypatch):
        """get_binaries(build=True) records PTO-ISA provenance for direct onboard builds."""
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, "a2a3")
        calls = []

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "a" * 40)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(
            pto_isa,
            "write_pto_isa_build_metadata",
            lambda lib_dir, pto_isa_root, runtime_keys: calls.append((lib_dir, pto_isa_root, runtime_keys)),
        )

        builder = RuntimeBuilder(platform="a2a3")
        builder.get_binaries("test_rt", build=True)

        assert calls == [(RuntimeBuilder._LIB_DIR, "/tmp/pto-isa", ["a2a3/onboard/test_rt"])]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_a2a3_onboard_host_build_passes_pto_isa_cmake_define(self, MockCompiler, tmp_path, monkeypatch):
        """Host runtime ccache key includes the pinned PTO-ISA commit."""
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        pin = "a" * 40
        self._make_runtime(tmp_path, "a2a3")

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: pin)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(pto_isa, "write_pto_isa_build_metadata", lambda *args: None)

        builder = RuntimeBuilder(platform="a2a3")
        builder.get_binaries("test_rt", build=True)

        host_call = next(call for call in mock_instance.compile.call_args_list if call.args[0] == "host")
        aicore_call = next(call for call in mock_instance.compile.call_args_list if call.args[0] == "aicore")
        aicpu_call = next(call for call in mock_instance.compile.call_args_list if call.args[0] == "aicpu")
        # Every target of a platform is compiled with the same profiling configuration,
        # so each call carries it alongside whatever else that target needs.
        profiling = {
            "SIMPLER_DFX": "1",
            "SIMPLER_ORCH_PROFILING": "0",
            "SIMPLER_SCHED_PROFILING": "0",
            "SIMPLER_TENSORMAP_PROFILING": "0",
        }
        assert host_call.kwargs["cmake_defines"] == {
            **profiling,
            "PTO_ISA_ROOT": "/tmp/pto-isa",
            "SIMPLER_PTO_ISA_BUILD_COMMIT": pin,
            "SIMPLER_RUNTIME_NAME": "test_rt",
        }
        # aicore needs the checkout path too, for the SDMA warmup kernel's
        # vector-only target, but not the commit stamp (that keys the host ccache).
        assert aicore_call.kwargs["cmake_defines"] == {**profiling, "PTO_ISA_ROOT": "/tmp/pto-isa"}
        assert aicpu_call.kwargs["cmake_defines"] == profiling

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_partial_profiling_config_is_completed_from_the_defaults(self, MockCompiler, tmp_path, monkeypatch):
        """A caller may name one macro; cmake must still receive all four.

        cmake/profiling_config.cmake defaults only the variables that are undefined,
        and a name left out of a partial mapping is still defined in the target's cache
        from whatever configured it last — so a partial mapping that reached cmake
        unchanged would take stale values for the names it omits.
        """
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, "a2a3")
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "a" * 40)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(pto_isa, "write_pto_isa_build_metadata", lambda *args: None)

        builder = RuntimeBuilder(platform="a2a3")
        builder.get_binaries("test_rt", build=True, profiling_config={"SIMPLER_ORCH_PROFILING": "1"})

        for call in mock_instance.compile.call_args_list:
            defines = call.kwargs["cmake_defines"]
            assert defines["SIMPLER_ORCH_PROFILING"] == "1"
            assert defines["SIMPLER_DFX"] == "1"
            assert defines["SIMPLER_SCHED_PROFILING"] == "0"
            assert defines["SIMPLER_TENSORMAP_PROFILING"] == "0"

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_a5_default_build_writes_pto_isa_metadata(self, MockCompiler, tmp_path, monkeypatch):
        """a5 onboard records PTO-ISA provenance for its default SDMA workspace."""
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, "a5")
        calls = []

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "a" * 40)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(
            pto_isa,
            "write_pto_isa_build_metadata",
            lambda lib_dir, pto_isa_root, runtime_keys: calls.append((lib_dir, pto_isa_root, runtime_keys)),
        )

        builder = RuntimeBuilder(platform="a5")
        builder.get_binaries("test_rt", build=True)

        assert calls == [(RuntimeBuilder._LIB_DIR, "/tmp/pto-isa", ["a5/onboard/test_rt"])]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_a5_default_host_build_passes_pto_isa_cmake_define(self, MockCompiler, tmp_path, monkeypatch):
        """a5 host ccache key includes the pinned PTO-ISA commit."""
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        pin = "b" * 40
        self._make_runtime(tmp_path, "a5")
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: pin)
        monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda verbose=False: "/tmp/pto-isa")
        monkeypatch.setattr(pto_isa, "write_pto_isa_build_metadata", lambda *args: None)

        builder = RuntimeBuilder(platform="a5")
        builder.get_binaries("test_rt", build=True)

        host_call = next(call for call in mock_instance.compile.call_args_list if call.args[0] == "host")
        assert host_call.kwargs["cmake_defines"]["SIMPLER_PTO_ISA_BUILD_COMMIT"] == pin
        assert "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE" not in host_call.kwargs["cmake_defines"]
        assert host_call.kwargs["cmake_defines"]["PTO_ISA_ROOT"] == "/tmp/pto-isa"

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_sim_direct_build_does_not_write_pto_isa_metadata(self, MockCompiler, tmp_path, monkeypatch):
        """get_binaries(build=True) only writes PTO-ISA metadata for embedding onboard."""
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, "a2a3")

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")
        mock_instance.compile_sim_context.return_value = tmp_path / "build" / "lib" / "libcpu_sim_context.so"
        monkeypatch.setattr(pto_isa, "write_pto_isa_build_metadata", lambda *args: pytest.fail("unexpected metadata"))

        builder = RuntimeBuilder(platform="a2a3sim")
        builder.get_binaries("test_rt", build=True)


# --- _invalidate_cache_if_stale unit tests ---


class TestInvalidateCacheIfStale:
    """Test cmake cache invalidation behavior under varying git state."""

    def test_clears_when_commit_mismatches(self, tmp_path):
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("old_sha\n")
        (cache_dir / "stale_artifact.o").write_text("stale")

        _invalidate_cache_if_stale(cache_dir, "new_sha")

        assert not (cache_dir / "stale_artifact.o").exists()
        assert (cache_dir / _GIT_COMMIT_FILE).read_text().strip() == "new_sha"

    def test_keeps_when_commit_matches(self, tmp_path):
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("same_sha\n")
        artifact = cache_dir / "kept_artifact.o"
        artifact.write_text("fresh")

        _invalidate_cache_if_stale(cache_dir, "same_sha")

        assert artifact.exists()

    def test_clears_when_commit_unavailable(self, tmp_path):
        """No discoverable git HEAD should force a clean rebuild, not silently skip."""
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("old_sha\n")
        (cache_dir / "stale_artifact.o").write_text("stale")

        _invalidate_cache_if_stale(cache_dir, "")

        assert not (cache_dir / "stale_artifact.o").exists()
        assert not (cache_dir / _GIT_COMMIT_FILE).exists()
        assert cache_dir.is_dir()


class TestAbbrevStamp:
    """Stamp abbreviation must keep a pto-isa-only change visible in the log."""

    def test_pure_runtime_sha_truncated(self):
        from simpler_setup.runtime_builder import _abbrev_stamp  # noqa: PLC0415

        assert _abbrev_stamp("0123456789abcdef0123456789abcdef01234567") == "0123456789ab"

    def test_composite_keeps_both_segments_visible(self):
        """A pto-isa-only bump shares the runtime prefix; both shas must show."""
        from simpler_setup.runtime_builder import _abbrev_stamp  # noqa: PLC0415

        runtime = "0123456789abcdef0123456789abcdef01234567"
        old = _abbrev_stamp(f"{runtime}:pto-isa=aaaaaaaaaaaaaaaa")
        new = _abbrev_stamp(f"{runtime}:pto-isa=bbbbbbbbbbbbbbbb")
        assert old == "0123456789ab:pto-isa=aaaaaaaaaaaa"
        assert new == "0123456789ab:pto-isa=bbbbbbbbbbbb"
        assert old != new  # the bump is not hidden behind an identical prefix


class TestBuildCacheStamp:
    """Test cmake cache stamp composition (runtime HEAD + pto-isa commit)."""

    def _make_builder(self, platform):
        from simpler_setup.platform_info import parse_platform  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder.__new__(RuntimeBuilder)
        builder.platform = platform
        builder._arch, builder._variant = parse_platform(platform)
        return builder

    def test_a2a3_onboard_folds_in_pto_isa_commit(self, monkeypatch):
        """a2a3 onboard with a resolved pto-isa commit → composite stamp."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "_get_git_head", lambda _root: "runtime_sha")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "isa_sha")

        builder = self._make_builder("a2a3")
        assert builder._build_cache_stamp() == "runtime_sha:pto-isa=isa_sha"

    def test_a5_default_folds_in_pto_isa_commit(self, monkeypatch):
        """a5 default SDMA workspace folds the pto-isa pin into the cache stamp."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
        monkeypatch.setattr(rb_module, "_get_git_head", lambda _root: "runtime_sha")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "isa_sha")

        builder = self._make_builder("a5")
        assert builder._build_cache_stamp() == "runtime_sha:pto-isa=isa_sha"

    def test_non_a2a3_onboard_uses_pure_runtime_sha(self, monkeypatch):
        """Other arch/variant ignores pto-isa → stamp keyed on runtime HEAD only."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "_get_git_head", lambda _root: "runtime_sha")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: pytest.fail("unexpected pin read"))

        builder = self._make_builder("a2a3sim")
        assert builder._build_cache_stamp() == "runtime_sha"

    def test_empty_runtime_head_yields_empty_stamp(self, monkeypatch):
        """No runtime HEAD → empty stamp, preserving the 'unavailable → clean rebuild' path."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "_get_git_head", lambda _root: "")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: pytest.fail("unexpected pin read"))

        builder = self._make_builder("a2a3")
        assert builder._build_cache_stamp() == ""


class TestResolveBuildPtoIsaCommit:
    """Test PTO-ISA pin resolution used by runtime build cache keys."""

    def _make_builder(self, platform):
        from simpler_setup.platform_info import parse_platform  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder.__new__(RuntimeBuilder)
        builder.platform = platform
        builder._arch, builder._variant = parse_platform(platform)
        return builder

    def test_non_a2a3_onboard_returns_empty(self, monkeypatch):
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: pytest.fail("unexpected pin read"))

        builder = self._make_builder("a2a3sim")
        assert builder._resolve_build_pto_isa_commit() == ""

    def test_a5_default_reads_pin(self, monkeypatch):
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "isa_sha")

        builder = self._make_builder("a5")
        assert builder._resolve_build_pto_isa_commit() == "isa_sha"

    def test_a5_urma_overlay_on_reads_pin(self, monkeypatch):
        """URMA overlay also embeds pto-isa, so it reads the pin too (#1392)."""
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "ON")
        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "isa_sha")
        builder = self._make_builder("a5")
        assert builder._resolve_build_pto_isa_commit() == "isa_sha"

    def test_a2a3_onboard_reads_pin(self, monkeypatch):
        from simpler_setup import pto_isa  # noqa: PLC0415

        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", lambda: "isa_sha")
        builder = self._make_builder("a2a3")
        assert builder._resolve_build_pto_isa_commit() == "isa_sha"

    def test_pin_error_propagates(self, monkeypatch):
        from simpler_setup import pto_isa  # noqa: PLC0415

        def _raise_bad_pin():
            raise RuntimeError("bad pin")

        monkeypatch.setattr(pto_isa, "read_pto_isa_pin", _raise_bad_pin)

        builder = self._make_builder("a2a3")
        with pytest.raises(RuntimeError, match="bad pin"):
            builder._resolve_build_pto_isa_commit()


class TestPlaceBinary:
    """place_binary must never rewrite the destination in place.

    A build that replaces a .so already dlopened by the running process — which
    every `pip install -e .` does — corrupts that mapping if the
    destination is truncated and rewritten. The fault surfaces far away, as a
    SIGSEGV inside the dynamic linker on an unrelated dlopen or at interpreter
    exit, so the invariant is asserted here rather than left to a crash.
    """

    def _library(self, path: Path) -> Path:
        """Compile a trivial shared object at `path`."""
        import subprocess  # noqa: PLC0415

        source = path.with_suffix(".c")
        source.write_text("int probe(void) { return 7; }\n", encoding="utf-8")
        # gcc is a fixed test-only toolchain dependency, not user input.
        subprocess.run(  # noqa: S603
            ["gcc", "-shared", "-fPIC", "-o", str(path), str(source)],  # noqa: S607
            check=True,
        )
        return path

    def test_replaces_by_rename_not_in_place(self, tmp_path):
        from simpler_setup.runtime_compiler import place_binary  # noqa: PLC0415

        dest = self._library(tmp_path / "libprobe.so")
        src = self._library(tmp_path / "libnewer.so")
        before = dest.stat().st_ino

        place_binary(src, dest)

        assert dest.stat().st_ino != before, "destination was rewritten in place, not renamed over"

    def test_leaves_an_open_mapping_of_the_old_inode_valid(self, tmp_path):
        import ctypes  # noqa: PLC0415

        from simpler_setup.runtime_compiler import place_binary  # noqa: PLC0415

        dest = self._library(tmp_path / "libprobe.so")
        handle = ctypes.CDLL(str(dest))
        assert handle.probe() == 7

        place_binary(self._library(tmp_path / "libnewer.so"), dest)

        # The already-loaded copy keeps its own inode, so calling into it after
        # the replacement is still defined behaviour.
        assert handle.probe() == 7

    def test_leaves_no_temporary_behind_when_post_process_fails(self, tmp_path):
        from simpler_setup.runtime_compiler import place_binary  # noqa: PLC0415

        dest = self._library(tmp_path / "libprobe.so")
        src = self._library(tmp_path / "libnewer.so")

        def _fail(_staged):
            raise RuntimeError("strip failed")

        with pytest.raises(RuntimeError, match="strip failed"):
            place_binary(src, dest, post_process=_fail)

        assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


# --- Full integration tests (real compilation) ---


@pytest.mark.requires_hardware
class TestRuntimeBuilderIntegration:
    """Integration tests that actually compile all platform x runtime combinations.

    Test parametrization is handled dynamically by conftest.py based on:
    - Available platforms discovered from src/*/platform/{onboard,sim}/
    - Available runtimes per architecture from src/{arch}/runtime/*/build_config.py
    - --platform filter if specified on command line
    """

    @pytest.fixture(autouse=True)
    def _reset_compiler_singleton(self):
        """Reset RuntimeCompiler singleton-per-platform cache so each test gets fresh instances."""
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        yield
        RuntimeCompiler._instances.clear()

    def test_get_binaries_returns_valid_paths(self, platform, runtime_name):
        """get_binaries(build=True) produces RuntimeBinaries with existing files."""
        from simpler_setup.runtime_builder import RuntimeBinaries, RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=platform)
        result = builder.get_binaries(runtime_name, build=True)

        assert isinstance(result, RuntimeBinaries)
        for label, path in [
            ("host", result.host_path),
            ("aicpu", result.aicpu_path),
            ("aicore", result.aicore_path),
        ]:
            assert path.is_file(), f"{label} binary not found: {path}"
            assert path.stat().st_size > 0, f"{label} binary is empty: {path}"
