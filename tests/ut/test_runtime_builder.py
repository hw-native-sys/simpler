"""Tests for RuntimeBuilder class."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Add python/ to path so we can import runtime_builder
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


# --- Discovery tests (no compilation needed) ---


class TestRuntimeBuilderDiscovery:
    """Test runtime discovery from src/runtime/ subdirectories."""

    def test_discovers_real_runtimes(self, default_test_platform):
        """RuntimeBuilder discovers host_build_graph from the real project tree."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=default_test_platform)
        runtimes = builder.list_runtimes()
        assert "host_build_graph" in runtimes

    def test_discovers_aicpu_build_graph(self, default_test_platform):
        """RuntimeBuilder discovers aicpu_build_graph from the real project tree."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=default_test_platform)
        runtimes = builder.list_runtimes()
        assert "aicpu_build_graph" in runtimes

    def test_runtime_dir_resolves_to_project_root(self, default_test_platform, test_arch):
        """runtime_dir resolves to src/{arch}/runtime/ under the project root."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.runtime_dir == builder.runtime_root / "src" / test_arch / "runtime"
        assert builder.runtime_dir.is_dir()

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_discovers_configs_in_runtime_dir(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """RuntimeBuilder discovers implementations in the runtime directory."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        # Set up fake runtime tree with architecture-specific structure
        rt_dir = tmp_path / "src" / test_arch / "runtime" / "my_runtime"
        rt_dir.mkdir(parents=True)
        (rt_dir / "build_config.py").write_text(
            "BUILD_CONFIG = {}\n"
        )

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["my_runtime"]

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_ignores_dirs_without_build_config(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Directories without build_config.py are not listed."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        (rt_dir / "has_config").mkdir(parents=True)
        (rt_dir / "has_config" / "build_config.py").write_text("BUILD_CONFIG = {}\n")
        (rt_dir / "no_config").mkdir(parents=True)
        # __pycache__ should also be ignored
        (rt_dir / "__pycache__").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["has_config"]

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_empty_runtime_dir(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Empty src/{arch}/runtime/ directory yields no runtimes."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_missing_runtime_dir(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform):
        """Non-existent src/{arch}/runtime/ directory yields no runtimes."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_multiple_runtimes_sorted(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Multiple implementations are returned in sorted order."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        for name in ["zeta", "alpha", "beta"]:
            d = rt_dir / name
            d.mkdir(parents=True)
            (d / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["alpha", "beta", "zeta"]


# --- Build error handling tests ---


class TestRuntimeBuilderBuildErrors:
    """Test build() error handling without invoking real compilation."""

    def test_build_unknown_runtime_raises(self, default_test_platform):
        """build() raises ValueError for a non-existent runtime name."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="is not available for platform"):
            builder.build("nonexistent_runtime")

    def test_build_unknown_runtime_lists_available(self, default_test_platform):
        """ValueError message includes available runtime names."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="host_build_graph"):
            builder.build("nonexistent_runtime")

    @patch("runtime_builder.RuntimeCompiler")
    @patch("runtime_builder.KernelCompiler")
    def test_build_empty_registry_shows_none(self, MockKernel, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """ValueError message shows '(none)' when no runtimes exist."""
        import runtime_builder as rb_module
        from runtime_builder import RuntimeBuilder

        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)
        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match=r"\(none\)"):
            builder.build("anything")


# --- Build integration tests (mocked compilation) ---


class TestRuntimeBuilderBuild:
    """Test build() logic with mocked RuntimeCompiler."""

    @pytest.fixture(autouse=True)
    def _patch_runtime_root(self, monkeypatch, tmp_path):
        import runtime_builder as rb_module
        monkeypatch.setattr(rb_module, "__file__", str(tmp_path / "python" / "rb.py"))

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

    @patch("runtime_builder.KernelCompiler")
    @patch("runtime_builder.RuntimeCompiler")
    def test_build_returns_three_binaries(self, MockCompiler, MockKernel, tmp_path, default_test_platform, test_arch):
        """build() returns (host_binary, aicpu_binary, aicore_binary)."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = [
            b"aicore_bin",   # first call: aicore
            b"aicpu_bin",    # second call: aicpu
            b"host_bin",     # third call: host
        ]

        builder = RuntimeBuilder(platform=default_test_platform)
        result = builder.build("test_rt")

        assert result == (b"host_bin", b"aicpu_bin", b"aicore_bin")

    @patch("runtime_builder.KernelCompiler")
    @patch("runtime_builder.RuntimeCompiler")
    def test_build_calls_compiler_three_times(self, MockCompiler, MockKernel, tmp_path, default_test_platform, test_arch):
        """build() invokes compiler.compile() exactly 3 times (aicore, aicpu, host)."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.return_value = b"binary"

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.build("test_rt")

        assert mock_instance.compile.call_count == 3
        platforms = [call.args[0] for call in mock_instance.compile.call_args_list]
        assert platforms == ["aicore", "aicpu", "host"]

    @patch("runtime_builder.KernelCompiler")
    @patch("runtime_builder.RuntimeCompiler")
    def test_build_resolves_paths_relative_to_config(self, MockCompiler, MockKernel, tmp_path, default_test_platform, test_arch):
        """Include/source dirs are resolved relative to the build_config.py directory."""
        from runtime_builder import RuntimeBuilder

        rt_dir = self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.return_value = b"binary"

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.build("test_rt")

        # Check the first call (aicore): include_dirs should be resolved paths
        aicore_call = mock_instance.compile.call_args_list[0]
        include_dirs = aicore_call.args[1]
        for d in include_dirs:
            assert Path(d).is_absolute()
            assert str(rt_dir.resolve()) in d

    @patch("runtime_builder.KernelCompiler")
    @patch("runtime_builder.RuntimeCompiler")
    def test_build_propagates_compiler_error(self, MockCompiler, MockKernel, tmp_path, default_test_platform, test_arch):
        """If RuntimeCompiler.compile() raises, build() propagates the exception."""
        from runtime_builder import RuntimeBuilder

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = RuntimeError("cmake failed")

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(RuntimeError, match="cmake failed"):
            builder.build("test_rt")


# --- Full integration tests (real compilation) ---


@pytest.mark.requires_hardware
class TestRuntimeBuilderIntegration:
    """Integration tests that actually compile all platform × runtime combinations.

    Test parametrization is handled dynamically by conftest.py based on:
    - Available platforms discovered from src/*/platform/{onboard,sim}/
    - Available runtimes per architecture from src/{arch}/runtime/*/build_config.py
    - --platform filter if specified on command line
    """

    @pytest.fixture(autouse=True)
    def _reset_compiler_singleton(self):
        """Reset RuntimeCompiler singleton-per-platform cache so each test gets fresh instances."""
        from runtime_compiler import RuntimeCompiler

        yield
        RuntimeCompiler._instances.clear()

    def test_build_returns_three_binaries(self, platform, runtime_name):
        """build() produces a 3-tuple of non-empty bytes."""
        from runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=platform)
        result = builder.build(runtime_name)

        assert isinstance(result, tuple)
        assert len(result) == 3

        host_binary, aicpu_binary, aicore_binary = result
        for label, binary in [("host", host_binary), ("aicpu", aicpu_binary), ("aicore", aicore_binary)]:
            assert isinstance(binary, bytes), f"{label} binary is not bytes"
            assert len(binary) > 0, f"{label} binary is empty"
