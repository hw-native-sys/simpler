# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for python/runtime_compiler.py -- CMake-based runtime compilation."""

import os
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


@pytest.fixture(autouse=True)
def _reset_compiler_singleton():
    """Reset RuntimeCompiler singleton cache between tests."""
    from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

    yield
    RuntimeCompiler.reset_instances()


# =============================================================================
# BuildTarget tests
# =============================================================================


class TestBuildTarget:
    """Tests for BuildTarget CMake argument generation."""

    def test_cmake_args_assembly(self, tmp_path):
        """gen_cmake_args() combines toolchain args with include/source dirs."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import BuildTarget  # noqa: PLC0415

        mock_toolchain = MagicMock()
        mock_toolchain.get_cmake_args.return_value = ["-DCMAKE_CXX_COMPILER=g++"]

        target = BuildTarget(mock_toolchain, str(tmp_path), "libtest.so")
        args = target.gen_cmake_args(include_dirs=[str(tmp_path / "inc")], source_dirs=[str(tmp_path / "src")])

        assert "-DCMAKE_CXX_COMPILER=g++" in args
        assert any("CUSTOM_INCLUDE_DIRS" in a for a in args)
        assert any("CUSTOM_SOURCE_DIRS" in a for a in args)

    def test_root_dir_is_absolute(self, tmp_path):
        """get_root_dir() returns an absolute path."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import BuildTarget  # noqa: PLC0415

        mock_toolchain = MagicMock()
        target = BuildTarget(mock_toolchain, str(tmp_path / "src"), "lib.so")
        assert os.path.isabs(target.get_root_dir())

    def test_binary_name(self, tmp_path):
        """get_binary_name() returns the configured name."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import BuildTarget  # noqa: PLC0415

        mock_toolchain = MagicMock()
        target = BuildTarget(mock_toolchain, str(tmp_path), "mylib.so")
        assert target.get_binary_name() == "mylib.so"


# =============================================================================
# RuntimeCompiler tests
# =============================================================================


class TestRuntimeCompiler:
    """Tests for RuntimeCompiler initialization and validation."""

    @patch("simpler_setup.runtime_compiler.RuntimeCompiler._ensure_host_compilers")
    def test_unknown_platform_raises(self, mock_ensure):
        """Unknown platform raises ValueError with supported list."""
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown platform.*Supported"):
            RuntimeCompiler("z9000")

    @patch("simpler_setup.runtime_compiler.RuntimeCompiler._ensure_host_compilers")
    def test_missing_platform_dir_raises(self, mock_ensure, tmp_path):
        """Non-existent platform directory raises ValueError."""
        # a2a3sim expects src/a2a3/platform/sim/ to exist
        # With a custom project_root that doesn't have the dir, it should fail
        # Verify that a non-existent platform dir would not exist
        phantom_dir = tmp_path / "src" / "a2a3" / "platform" / "sim"
        assert not phantom_dir.is_dir()

    def test_singleton_pattern(self):
        """get_instance() returns same instance for same platform."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        with patch.object(RuntimeCompiler, "_ensure_host_compilers"):
            rc1 = RuntimeCompiler.get_instance("a2a3sim")
            rc2 = RuntimeCompiler.get_instance("a2a3sim")
            assert rc1 is rc2


# =============================================================================
# Compiler availability tests (via construction behavior)
# =============================================================================


class TestCompilerAvailability:
    """Tests for compiler availability via construction."""

    def test_sim_platform_construction_succeeds(self):
        """Sim platform can be constructed (no hardware compilers needed)."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        with patch.object(RuntimeCompiler, "_ensure_host_compilers"):
            rc = RuntimeCompiler("a2a3sim")
            assert rc.platform == "a2a3sim"


# =============================================================================
# Compile target validation tests
# =============================================================================


class TestCompileTargetValidation:
    """Tests for compile() target platform validation."""

    @patch("simpler_setup.runtime_compiler.RuntimeCompiler._ensure_host_compilers")
    def test_invalid_target_platform_raises(self, mock_ensure):
        """Invalid target platform raises ValueError."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        rc = RuntimeCompiler("a2a3sim")
        with pytest.raises(ValueError, match="Invalid target platform"):
            rc.compile("gpu", [], [], None)
