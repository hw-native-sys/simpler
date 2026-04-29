# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for RuntimeBuilder class."""

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
