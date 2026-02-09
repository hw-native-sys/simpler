"""Tests for KernelCompiler.compile_orchestration dispatch."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


@pytest.fixture
def sim_compiler():
    """Create a KernelCompiler for a2a3sim (no Ascend SDK needed)."""
    from kernel_compiler import KernelCompiler
    return KernelCompiler(platform="a2a3sim")


class TestCompileOrchestrationDispatch:
    """Test that compile_orchestration dispatches to the correct private method."""

    def test_host_build_graph_calls_shared_lib(self, sim_compiler):
        """host_build_graph dispatches to _compile_orchestration_shared_lib."""
        with patch.object(sim_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            result = sim_compiler.compile_orchestration("host_build_graph", "/tmp/fake.cpp")

        assert result == b"so_bytes"
        mock.assert_called_once()
        call_kwargs = mock.call_args
        assert "/tmp/fake.cpp" == call_kwargs.args[0]

    def test_tensormap_and_ringbuffer_calls_shared_lib(self, sim_compiler):
        """tensormap_and_ringbuffer dispatches to _compile_orchestration_shared_lib."""
        with patch.object(sim_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            result = sim_compiler.compile_orchestration("tensormap_and_ringbuffer", "/tmp/fake.cpp")

        assert result == b"so_bytes"
        mock.assert_called_once()
        # On a2a3sim, no extra_sources or extra_link_flags
        call_kwargs = mock.call_args
        assert call_kwargs.kwargs.get("extra_sources") is None
        assert call_kwargs.kwargs.get("extra_link_flags") is None

    def test_aicpu_build_graph_calls_shared_lib(self, sim_compiler):
        """aicpu_build_graph dispatches to _compile_orchestration_shared_lib."""
        with patch.object(sim_compiler, "_compile_orchestration_shared_lib", return_value=b"plugin_bytes") as mock:
            result = sim_compiler.compile_orchestration("aicpu_build_graph", "/tmp/fake.cpp")

        assert result == b"plugin_bytes"
        mock.assert_called_once()
        # On a2a3sim, no extra_cxxflags
        call_kwargs = mock.call_args
        assert call_kwargs.kwargs.get("extra_cxxflags") is None

    def test_unknown_runtime_raises(self, sim_compiler):
        """Unknown runtime_name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown runtime"):
            sim_compiler.compile_orchestration("nonexistent_runtime", "/tmp/fake.cpp")

    def test_include_dirs_contain_runtime_and_platform(self, sim_compiler):
        """Include dirs passed to the private method contain runtime + platform dirs."""
        with patch.object(sim_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            sim_compiler.compile_orchestration("host_build_graph", "/tmp/fake.cpp")

        include_dirs = mock.call_args.kwargs["extra_include_dirs"]
        # Should contain runtime dir and platform dirs
        assert any("host_build_graph" in d for d in include_dirs)
        assert any("platform" in d for d in include_dirs)

    def test_extra_include_dirs_merged(self, sim_compiler):
        """Caller-provided extra_include_dirs are merged with runtime/platform dirs."""
        with patch.object(sim_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            sim_compiler.compile_orchestration(
                "host_build_graph", "/tmp/fake.cpp",
                extra_include_dirs=["/extra/inc"],
            )

        include_dirs = mock.call_args.kwargs["extra_include_dirs"]
        assert "/extra/inc" in include_dirs


class TestCompileOrchestrationTensormapA2a3:
    """Test tensormap_and_ringbuffer dispatch on a2a3 (extra sources + link flags)."""

    @pytest.fixture
    def a2a3_compiler(self):
        """Create a KernelCompiler for a2a3 (requires ASCEND_HOME_PATH)."""
        if not os.getenv("ASCEND_HOME_PATH"):
            pytest.skip("ASCEND_HOME_PATH not set")
        from kernel_compiler import KernelCompiler
        return KernelCompiler(platform="a2a3")

    def test_a2a3_adds_extra_sources_and_link_flags(self, a2a3_compiler):
        """On a2a3, tensormap_and_ringbuffer adds extra sources and link flags."""
        with patch.object(a2a3_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            a2a3_compiler.compile_orchestration("tensormap_and_ringbuffer", "/tmp/fake.cpp")

        call_kwargs = mock.call_args
        assert call_kwargs.kwargs.get("extra_sources") is not None
        assert call_kwargs.kwargs.get("extra_link_flags") is not None

    def test_a2a3_aicpu_adds_cxxflags_and_includes(self, a2a3_compiler):
        """On a2a3, aicpu_build_graph adds AICPU plugin flags and include paths."""
        with patch.object(a2a3_compiler, "_compile_orchestration_shared_lib", return_value=b"so_bytes") as mock:
            a2a3_compiler.compile_orchestration("aicpu_build_graph", "/tmp/fake.cpp")

        call_kwargs = mock.call_args
        extra_cxxflags = call_kwargs.kwargs.get("extra_cxxflags")
        assert extra_cxxflags is not None
        assert "-std=gnu++17" in extra_cxxflags
        assert "-static-libstdc++" in extra_cxxflags
        # AICPU include paths should be merged into extra_include_dirs
        include_dirs = call_kwargs.kwargs.get("extra_include_dirs")
        assert any("toolchain" in d for d in include_dirs)
