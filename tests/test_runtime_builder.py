"""Tests for RuntimeBuilder class."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add python/ to path so we can import runtime_builder
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


# --- Discovery tests (no compilation needed) ---


class TestRuntimeBuilderDiscovery:
    """Test runtime discovery from src/runtime/ subdirectories."""

    def test_discovers_real_runtimes(self):
        """RuntimeBuilder discovers host_build_graph from the real project tree."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(runtime_root=PROJECT_ROOT)
        runtimes = builder.list_runtimes()
        assert "host_build_graph" in runtimes

    def test_default_runtime_root(self):
        """Default runtime_root resolves to the project root."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder()
        # runtime_root should be parent of python/ which is the project root
        assert builder.runtime_dir == builder.runtime_root / "src" / "runtime"
        assert builder.runtime_dir.is_dir()

    def test_custom_runtime_root_with_configs(self, tmp_path):
        """RuntimeBuilder discovers implementations in a custom root."""
        from runtime_builder import RuntimeBuilder

        # Set up fake runtime tree
        rt_dir = tmp_path / "src" / "runtime" / "my_runtime"
        rt_dir.mkdir(parents=True)
        (rt_dir / "build_config.py").write_text(
            "BUILD_CONFIG = {}\n"
        )

        builder = RuntimeBuilder(runtime_root=tmp_path)
        assert builder.list_runtimes() == ["my_runtime"]

    def test_ignores_dirs_without_build_config(self, tmp_path):
        """Directories without build_config.py are not listed."""
        from runtime_builder import RuntimeBuilder

        rt_dir = tmp_path / "src" / "runtime"
        (rt_dir / "has_config").mkdir(parents=True)
        (rt_dir / "has_config" / "build_config.py").write_text("BUILD_CONFIG = {}\n")
        (rt_dir / "no_config").mkdir(parents=True)
        # __pycache__ should also be ignored
        (rt_dir / "__pycache__").mkdir(parents=True)

        builder = RuntimeBuilder(runtime_root=tmp_path)
        assert builder.list_runtimes() == ["has_config"]

    def test_empty_runtime_dir(self, tmp_path):
        """Empty src/runtime/ directory yields no runtimes."""
        from runtime_builder import RuntimeBuilder

        (tmp_path / "src" / "runtime").mkdir(parents=True)

        builder = RuntimeBuilder(runtime_root=tmp_path)
        assert builder.list_runtimes() == []

    def test_missing_runtime_dir(self, tmp_path):
        """Non-existent src/runtime/ directory yields no runtimes."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(runtime_root=tmp_path)
        assert builder.list_runtimes() == []

    def test_multiple_runtimes_sorted(self, tmp_path):
        """Multiple implementations are returned in sorted order."""
        from runtime_builder import RuntimeBuilder

        rt_dir = tmp_path / "src" / "runtime"
        for name in ["zeta", "alpha", "beta"]:
            d = rt_dir / name
            d.mkdir(parents=True)
            (d / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(runtime_root=tmp_path)
        assert builder.list_runtimes() == ["alpha", "beta", "zeta"]


# --- Build error handling tests ---


class TestRuntimeBuilderBuildErrors:
    """Test build() error handling without invoking real compilation."""

    def test_build_unknown_runtime_raises(self):
        """build() raises ValueError for a non-existent runtime name."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(runtime_root=PROJECT_ROOT)
        with pytest.raises(ValueError, match="not found"):
            builder.build("nonexistent_runtime")

    def test_build_unknown_runtime_lists_available(self):
        """ValueError message includes available runtime names."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(runtime_root=PROJECT_ROOT)
        with pytest.raises(ValueError, match="host_build_graph"):
            builder.build("nonexistent_runtime")

    def test_build_empty_registry_shows_none(self, tmp_path):
        """ValueError message shows '(none)' when no runtimes exist."""
        from runtime_builder import RuntimeBuilder

        (tmp_path / "src" / "runtime").mkdir(parents=True)
        builder = RuntimeBuilder(runtime_root=tmp_path)
        with pytest.raises(ValueError, match=r"\(none\)"):
            builder.build("anything")


# --- Build integration tests (mocked compilation) ---


class TestRuntimeBuilderBuild:
    """Test build() logic with mocked BinaryCompiler."""

    def _make_runtime(self, tmp_path):
        """Create a fake runtime with a valid build_config.py."""
        rt_dir = tmp_path / "src" / "runtime" / "test_rt"
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

    @patch("runtime_builder.BinaryCompiler")
    def test_build_returns_three_binaries(self, MockCompiler, tmp_path):
        """build() returns (host_binary, aicpu_binary, aicore_binary)."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path)

        mock_instance = MockCompiler.return_value
        mock_instance.compile.side_effect = [
            b"aicore_bin",   # first call: aicore
            b"aicpu_bin",    # second call: aicpu
            b"host_bin",     # third call: host
        ]

        builder = RuntimeBuilder(runtime_root=tmp_path)
        result = builder.build("test_rt")

        assert result == (b"host_bin", b"aicpu_bin", b"aicore_bin")

    @patch("runtime_builder.BinaryCompiler")
    def test_build_calls_compiler_three_times(self, MockCompiler, tmp_path):
        """build() invokes compiler.compile() exactly 3 times (aicore, aicpu, host)."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path)

        mock_instance = MockCompiler.return_value
        mock_instance.compile.return_value = b"binary"

        builder = RuntimeBuilder(runtime_root=tmp_path)
        builder.build("test_rt")

        assert mock_instance.compile.call_count == 3
        platforms = [call.args[0] for call in mock_instance.compile.call_args_list]
        assert platforms == ["aicore", "aicpu", "host"]

    @patch("runtime_builder.BinaryCompiler")
    def test_build_resolves_paths_relative_to_config(self, MockCompiler, tmp_path):
        """Include/source dirs are resolved relative to the build_config.py directory."""
        from runtime_builder import RuntimeBuilder

        rt_dir = self._make_runtime(tmp_path)

        mock_instance = MockCompiler.return_value
        mock_instance.compile.return_value = b"binary"

        builder = RuntimeBuilder(runtime_root=tmp_path)
        builder.build("test_rt")

        # Check the first call (aicore): include_dirs should be resolved paths
        aicore_call = mock_instance.compile.call_args_list[0]
        include_dirs = aicore_call.args[1]
        for d in include_dirs:
            assert Path(d).is_absolute()
            assert str(rt_dir.resolve()) in d

    @patch("runtime_builder.BinaryCompiler")
    def test_build_propagates_compiler_error(self, MockCompiler, tmp_path):
        """If BinaryCompiler.compile() raises, build() propagates the exception."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path)

        mock_instance = MockCompiler.return_value
        mock_instance.compile.side_effect = RuntimeError("cmake failed")

        builder = RuntimeBuilder(runtime_root=tmp_path)
        with pytest.raises(RuntimeError, match="cmake failed"):
            builder.build("test_rt")
