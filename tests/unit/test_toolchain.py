"""Unit tests for python/toolchain.py — Toolchain configuration and flag generation."""

import os
from unittest.mock import patch, MagicMock

import pytest

import env_manager
from toolchain import (
    ToolchainType,
    CCECToolchain,
    Gxx15Toolchain,
    GxxToolchain,
    Aarch64GxxToolchain,
)


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
    """Provide a fake ASCEND_HOME_PATH with expected compiler directories."""
    ascend = tmp_path / "ascend_toolkit"
    # Create ccec paths for A2A3
    (ascend / "bin").mkdir(parents=True)
    (ascend / "bin" / "ccec").touch()
    (ascend / "bin" / "ld.lld").touch()
    # Create ccec paths for A5
    (ascend / "tools" / "bisheng_compiler" / "bin").mkdir(parents=True)
    (ascend / "tools" / "bisheng_compiler" / "bin" / "ccec").touch()
    (ascend / "tools" / "bisheng_compiler" / "bin" / "ld.lld").touch()
    # Create aarch64 cross-compiler paths
    (ascend / "tools" / "hcc" / "bin").mkdir(parents=True)
    (ascend / "tools" / "hcc" / "bin" / "aarch64-target-linux-gnu-g++").touch()
    (ascend / "tools" / "hcc" / "bin" / "aarch64-target-linux-gnu-gcc").touch()

    env_manager._cache["ASCEND_HOME_PATH"] = str(ascend)
    return str(ascend)


# =============================================================================
# CCECToolchain tests
# =============================================================================

class TestCCECToolchain:
    """Tests for CCECToolchain compile flags and cmake args."""

    def test_compile_flags_a2a3_aiv(self, mock_ascend_home):
        """A2A3 platform with aiv core type produces dav-c220-vec flags."""
        tc = CCECToolchain(platform="a2a3")
        flags = tc.get_compile_flags(core_type="aiv")
        flag_str = " ".join(flags)
        assert "dav-c220-vec" in flag_str

    def test_compile_flags_a2a3_aic(self, mock_ascend_home):
        """A2A3 platform with aic core type produces dav-c220-cube flags."""
        tc = CCECToolchain(platform="a2a3")
        flags = tc.get_compile_flags(core_type="aic")
        flag_str = " ".join(flags)
        assert "dav-c220-cube" in flag_str

    def test_compile_flags_a5_aiv(self, mock_ascend_home):
        """A5 platform with aiv core type produces dav-c310-vec flags."""
        tc = CCECToolchain(platform="a5")
        flags = tc.get_compile_flags(core_type="aiv")
        flag_str = " ".join(flags)
        assert "dav-c310-vec" in flag_str

    def test_compile_flags_a5_aic(self, mock_ascend_home):
        """A5 platform with aic core type produces dav-c310-cube flags."""
        tc = CCECToolchain(platform="a5")
        flags = tc.get_compile_flags(core_type="aic")
        flag_str = " ".join(flags)
        assert "dav-c310-cube" in flag_str

    def test_unknown_platform_raises(self, mock_ascend_home):
        """Unknown platform raises ValueError."""
        with pytest.raises(ValueError, match="Unknown platform"):
            CCECToolchain(platform="unknown")

    def test_missing_ccec_compiler_raises(self, tmp_path):
        """Missing ccec binary raises FileNotFoundError."""
        ascend = tmp_path / "empty_toolkit"
        (ascend / "bin").mkdir(parents=True)
        # No ccec binary created
        env_manager._cache["ASCEND_HOME_PATH"] = str(ascend)

        with pytest.raises(FileNotFoundError, match="ccec compiler not found"):
            CCECToolchain(platform="a2a3")

    def test_cmake_args_contain_bisheng(self, mock_ascend_home):
        """CMake args include BISHENG_CC and BISHENG_LD."""
        tc = CCECToolchain(platform="a2a3")
        args = tc.get_cmake_args()
        assert any("BISHENG_CC" in a for a in args)
        assert any("BISHENG_LD" in a for a in args)


# =============================================================================
# Gxx15Toolchain tests
# =============================================================================

class TestGxx15Toolchain:
    """Tests for Gxx15Toolchain compile flags."""

    def test_compile_flags_aiv_defines(self):
        """aiv core type adds -D__DAV_VEC__."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = Gxx15Toolchain()
        flags = tc.get_compile_flags(core_type="aiv")
        assert "-D__DAV_VEC__" in flags

    def test_compile_flags_aic_defines(self):
        """aic core type adds -D__DAV_CUBE__."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = Gxx15Toolchain()
        flags = tc.get_compile_flags(core_type="aic")
        assert "-D__DAV_CUBE__" in flags

    def test_compile_flags_no_core_type(self):
        """Empty core type adds neither __DAV_VEC__ nor __DAV_CUBE__."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = Gxx15Toolchain()
        flags = tc.get_compile_flags(core_type="")
        assert "-D__DAV_VEC__" not in flags
        assert "-D__DAV_CUBE__" not in flags

    def test_compile_flags_contain_cpu_sim(self):
        """Simulation flags include -D__CPU_SIM."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = Gxx15Toolchain()
        flags = tc.get_compile_flags()
        assert "-D__CPU_SIM" in flags

    def test_cmake_args_respect_env_vars(self):
        """CMake args use CC/CXX env vars when set."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = Gxx15Toolchain()
        with patch.dict(os.environ, {"CC": "my-gcc", "CXX": "my-g++"}):
            args = tc.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=my-gcc" in args
        assert "-DCMAKE_CXX_COMPILER=my-g++" in args


# =============================================================================
# GxxToolchain tests
# =============================================================================

class TestGxxToolchain:
    """Tests for GxxToolchain."""

    def test_cmake_args_with_ascend(self, mock_ascend_home):
        """With ASCEND_HOME_PATH, cmake args include it."""
        tc = GxxToolchain()
        args = tc.get_cmake_args()
        assert any("ASCEND_HOME_PATH" in a for a in args)

    def test_cmake_args_without_ascend(self):
        """Without ASCEND_HOME_PATH, cmake args do not include it."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = GxxToolchain()
        args = tc.get_cmake_args()
        assert not any("ASCEND_HOME_PATH" in a for a in args)

    def test_compile_flags_contain_std17(self):
        """Compile flags include C++17 standard."""
        env_manager._cache["ASCEND_HOME_PATH"] = None
        tc = GxxToolchain()
        flags = tc.get_compile_flags()
        assert "-std=c++17" in flags


# =============================================================================
# Aarch64GxxToolchain tests
# =============================================================================

class TestAarch64GxxToolchain:
    """Tests for Aarch64GxxToolchain."""

    def test_cmake_args_cross_compile(self, mock_ascend_home):
        """CMake args include aarch64 cross-compiler paths."""
        tc = Aarch64GxxToolchain()
        args = tc.get_cmake_args()
        assert any("aarch64-target-linux-gnu-gcc" in a for a in args)
        assert any("aarch64-target-linux-gnu-g++" in a for a in args)

    def test_missing_compiler_raises(self, tmp_path):
        """Missing aarch64 compiler raises FileNotFoundError."""
        ascend = tmp_path / "no_hcc"
        (ascend / "tools" / "hcc" / "bin").mkdir(parents=True)
        # No compiler binaries created
        env_manager._cache["ASCEND_HOME_PATH"] = str(ascend)

        with pytest.raises(FileNotFoundError, match="aarch64"):
            Aarch64GxxToolchain()


# =============================================================================
# ToolchainType tests
# =============================================================================

class TestToolchainType:
    """Tests for ToolchainType enum."""

    def test_enum_values(self):
        """ToolchainType values match compile_strategy.h."""
        assert ToolchainType.CCEC == 0
        assert ToolchainType.HOST_GXX_15 == 1
        assert ToolchainType.HOST_GXX == 2
        assert ToolchainType.AARCH64_GXX == 3
