# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for python/kernel_compiler.py -- Kernel and orchestration compilation."""

from unittest.mock import MagicMock, patch

import pytest
from simpler import env_manager

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
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

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
        from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

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
        from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

        kc = KernelCompiler(platform="a5sim")
        dirs = kc.get_orchestration_include_dirs("host_build_graph")
        assert any("a5" in d and "host_build_graph" in d for d in dirs)


# =============================================================================
# Platform to architecture mapping tests
# =============================================================================


class TestPlatformToArchMapping:
    """Tests for platform -> architecture directory mapping."""

    def test_a2a3_maps_to_a2a3(self, sim_compiler):
        """a2a3sim maps to a2a3 architecture directory."""
        assert "a2a3" in str(sim_compiler.platform_dir)

    def test_a5sim_maps_to_a5(self):
        """a5sim maps to a5 architecture directory."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

        kc = KernelCompiler(platform="a5sim")
        assert "a5" in str(kc.platform_dir)

    def test_unknown_platform_raises(self):
        """Unknown platform raises ValueError."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown platform"):
            KernelCompiler(platform="z9000")


# =============================================================================
# Toolchain selection tests (via compile_incore public API)
# =============================================================================


class TestToolchainSelection:
    """Tests for toolchain selection behavior via public API."""

    def test_unknown_platform_compile_raises(self):
        """Unknown platform raises ValueError at construction time."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown platform"):
            KernelCompiler(platform="z9000_nonexistent")


# =============================================================================
# Compilation error handling tests (via public compile methods)
# =============================================================================


class TestCompilationErrors:
    """Tests for compilation error handling via public API."""

    def test_compile_incore_missing_source_raises(self, sim_compiler, tmp_path):
        """Compiling a non-existent source file raises an error."""
        bad_source = str(tmp_path / "nonexistent_kernel.cpp")
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            sim_compiler.compile_incore(bad_source, core_type="aiv")

    def test_compile_orchestration_subprocess_failure(self, sim_compiler, tmp_path):
        """Compilation failure propagates error with stderr content."""
        source = tmp_path / "dummy.cpp"
        source.write_text("int main() {}")
        with patch("simpler_setup.kernel_compiler.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error: undefined reference to 'foo'")
            with pytest.raises(RuntimeError, match="undefined reference"):
                sim_compiler.compile_orchestration(
                    "host_build_graph",
                    str(source),
                )


# =============================================================================
# Orchestration config loading tests (via get_orchestration_include_dirs)
# =============================================================================


class TestOrchestrationConfig:
    """Tests for orchestration config behavior via public API."""

    def test_nonexistent_runtime_include_dirs(self, sim_compiler):
        """Non-existent runtime still returns base include dirs (no crash)."""
        dirs = sim_compiler.get_orchestration_include_dirs("nonexistent_runtime")
        # Should return at least the platform includes, not crash
        assert isinstance(dirs, list)
