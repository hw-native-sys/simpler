"""Unit tests for python/kernel_compiler.py — Kernel and orchestration compilation."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import env_manager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _clear_env_manager_cache():
    """Clear env_manager cache before each test."""
    env_manager._cache.clear()
    yield
    env_manager._cache.clear()


@pytest.fixture
def mock_ascend_home(tmp_path):
    """Set up fake ASCEND_HOME_PATH with compiler stubs."""
    ascend = tmp_path / "ascend"
    (ascend / "bin" / "ccec").mkdir(parents=True)
    (ascend / "bin" / "ccec").rmdir()
    (ascend / "bin").mkdir(parents=True, exist_ok=True)
    (ascend / "bin" / "ccec").touch()
    (ascend / "bin" / "ld.lld").touch()
    (ascend / "tools" / "hcc" / "bin").mkdir(parents=True)
    (ascend / "tools" / "hcc" / "bin" / "aarch64-target-linux-gnu-g++").touch()
    (ascend / "tools" / "hcc" / "bin" / "aarch64-target-linux-gnu-gcc").touch()
    env_manager._cache["ASCEND_HOME_PATH"] = str(ascend)
    return str(ascend)


@pytest.fixture
def sim_compiler(tmp_path):
    """Create a KernelCompiler for a2a3sim (no ASCEND_HOME_PATH needed)."""
    env_manager._cache["ASCEND_HOME_PATH"] = None
    from kernel_compiler import KernelCompiler
    return KernelCompiler(platform="a2a3sim")


# =============================================================================
# Platform include directory tests
# =============================================================================

class TestPlatformIncludeDirs:
    """Tests for get_platform_include_dirs()."""

    def test_a2a3sim_include_dirs(self, sim_compiler):
        """a2a3sim platform include dirs point to a2a3/platform/include."""
        dirs = sim_compiler.get_platform_include_dirs()
        assert len(dirs) >= 1
        assert any("a2a3" in d and "platform" in d and "include" in d for d in dirs)

    def test_a5sim_include_dirs(self):
        """a5sim platform include dirs point to a5/platform/include."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from kernel_compiler import KernelCompiler
        kc = KernelCompiler(platform="a5sim")
        dirs = kc.get_platform_include_dirs()
        assert any("a5" in d and "platform" in d and "include" in d for d in dirs)


# =============================================================================
# Orchestration include directory tests
# =============================================================================

class TestOrchestrationIncludeDirs:
    """Tests for get_orchestration_include_dirs()."""

    def test_a2a3_includes_runtime_dir(self, sim_compiler):
        """Orchestration includes contain the runtime-specific directory."""
        dirs = sim_compiler.get_orchestration_include_dirs("host_build_graph")
        assert any("host_build_graph" in d and "runtime" in d for d in dirs)

    def test_a5_includes_runtime_dir(self):
        """A5 orchestration includes point to a5 runtime directory."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from kernel_compiler import KernelCompiler
        kc = KernelCompiler(platform="a5sim")
        dirs = kc.get_orchestration_include_dirs("host_build_graph")
        assert any("a5" in d and "host_build_graph" in d for d in dirs)


# =============================================================================
# Platform to architecture mapping tests
# =============================================================================

class TestPlatformToArchMapping:
    """Tests for platform → architecture directory mapping."""

    def test_a2a3_maps_to_a2a3(self, sim_compiler):
        """a2a3sim maps to a2a3 architecture directory."""
        assert "a2a3" in str(sim_compiler.platform_dir)

    def test_a5sim_maps_to_a5(self):
        """a5sim maps to a5 architecture directory."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from kernel_compiler import KernelCompiler
        kc = KernelCompiler(platform="a5sim")
        assert "a5" in str(kc.platform_dir)

    def test_unknown_platform_raises(self):
        """Unknown platform raises ValueError."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from kernel_compiler import KernelCompiler
        with pytest.raises(ValueError, match="Unknown platform"):
            KernelCompiler(platform="z9000")


# =============================================================================
# Toolchain fallback tests
# =============================================================================

class TestToolchainFallback:
    """Tests for _get_toolchain() fallback behavior."""

    def test_fallback_on_runtime_error(self, sim_compiler):
        """When C++ library raises RuntimeError, falls back to platform map."""
        from toolchain import ToolchainType

        def failing_strategy():
            raise RuntimeError("Library not loaded")

        result = sim_compiler._get_toolchain(
            failing_strategy,
            {"a2a3sim": ToolchainType.HOST_GXX}
        )
        assert result == ToolchainType.HOST_GXX

    def test_fallback_missing_platform_raises(self, sim_compiler):
        """Fallback with unknown platform raises ValueError."""
        def failing_strategy():
            raise RuntimeError("Library not loaded")

        with pytest.raises(ValueError, match="No toolchain fallback"):
            sim_compiler._get_toolchain(failing_strategy, {"other": 0})


# =============================================================================
# Compilation error handling tests
# =============================================================================

class TestCompilationErrors:
    """Tests for compilation error handling."""

    def test_compile_to_bytes_missing_output(self, sim_compiler, tmp_path):
        """Missing output file after compilation raises RuntimeError."""
        output_path = str(tmp_path / "nonexistent.o")

        # Mock subprocess to succeed but produce no output file
        with patch("kernel_compiler.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            with pytest.raises(RuntimeError, match="output file not found"):
                sim_compiler._compile_to_bytes(
                    ["g++", "-o", output_path, "dummy.cpp"],
                    output_path,
                    "Test",
                )

    def test_subprocess_failure_includes_stderr(self, sim_compiler):
        """Compilation failure error includes stderr content."""
        with patch("kernel_compiler.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="error: undefined reference to 'foo'"
            )
            with pytest.raises(RuntimeError, match="undefined reference"):
                sim_compiler._run_subprocess(
                    ["g++", "bad.cpp"],
                    "Test",
                )


# =============================================================================
# Orchestration config loading tests
# =============================================================================

class TestOrchestrationConfig:
    """Tests for _get_orchestration_config()."""

    def test_missing_config_returns_empty(self, sim_compiler):
        """Non-existent build_config.py returns empty lists."""
        inc, src = sim_compiler._get_orchestration_config("nonexistent_runtime")
        assert inc == []
        assert src == []

    def test_config_without_orchestration_key(self, sim_compiler, tmp_path):
        """build_config.py without 'orchestration' key returns empty lists."""
        # The real host_build_graph runtime has no orchestration key in build_config
        inc, src = sim_compiler._get_orchestration_config("host_build_graph")
        assert inc == []
        assert src == []
