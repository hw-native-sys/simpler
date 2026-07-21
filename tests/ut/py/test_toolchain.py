# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for host toolchain CMake argument generation."""

import pytest


class TestParseCompilerEnv:
    """Conda activate scripts inject flags into CC/CXX (e.g.,
    ``gcc -pthread -B <env>/compiler_compat``). Verify we split them cleanly."""

    def test_unset_returns_default(self, monkeypatch):
        from simpler_setup.toolchain import _parse_compiler_env  # noqa: PLC0415

        monkeypatch.delenv("CC", raising=False)
        assert _parse_compiler_env("CC", "gcc") == ("gcc", [])

    def test_empty_returns_default(self, monkeypatch):
        from simpler_setup.toolchain import _parse_compiler_env  # noqa: PLC0415

        monkeypatch.setenv("CC", "")
        assert _parse_compiler_env("CC", "gcc") == ("gcc", [])

    def test_bare_name(self, monkeypatch):
        from simpler_setup.toolchain import _parse_compiler_env  # noqa: PLC0415

        monkeypatch.setenv("CC", "gcc-12")
        assert _parse_compiler_env("CC", "gcc") == ("gcc-12", [])

    def test_conda_style_injection(self, monkeypatch):
        from simpler_setup.toolchain import _parse_compiler_env  # noqa: PLC0415

        monkeypatch.setenv("CC", "gcc -pthread -B /data/envs/lyf/compiler_compat")
        path, flags = _parse_compiler_env("CC", "gcc")
        assert path == "gcc"
        assert flags == ["-pthread", "-B", "/data/envs/lyf/compiler_compat"]

    def test_quoted_path_with_spaces(self, monkeypatch):
        from simpler_setup.toolchain import _parse_compiler_env  # noqa: PLC0415

        monkeypatch.setenv("CC", "'/opt/my compilers/gcc' -O2")
        path, flags = _parse_compiler_env("CC", "gcc")
        assert path == "/opt/my compilers/gcc"
        assert flags == ["-O2"]


class TestGxxToolchainCmakeArgs:
    """Verify the host g++ toolchain emits CMake args that work under conda."""

    @pytest.fixture
    def toolchain(self, monkeypatch):
        # Avoid probing a real compiler; _is_gcc is only used for compile flags.
        monkeypatch.setattr("simpler_setup.toolchain._is_gcc", lambda _p: True)
        from simpler_setup.toolchain import GxxToolchain  # noqa: PLC0415

        return GxxToolchain()

    def test_plain_env(self, toolchain, monkeypatch):
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc" in args
        assert "-DCMAKE_CXX_COMPILER=g++" in args
        # No flags injected → no CMAKE_C_FLAGS arg.
        assert not any(a.startswith("-DCMAKE_C_FLAGS=") for a in args)
        assert not any(a.startswith("-DCMAKE_CXX_FLAGS=") for a in args)

    def test_conda_env_splits_flags(self, toolchain, monkeypatch):
        monkeypatch.setenv("CC", "gcc -pthread -B /data/envs/lyf/compiler_compat")
        monkeypatch.setenv("CXX", "g++ -pthread -B /data/envs/lyf/compiler_compat")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc" in args
        assert "-DCMAKE_CXX_COMPILER=g++" in args
        assert "-DCMAKE_C_FLAGS=-pthread -B /data/envs/lyf/compiler_compat" in args
        assert "-DCMAKE_CXX_FLAGS=-pthread -B /data/envs/lyf/compiler_compat" in args


class TestGxxToolchainPreferG15Pins:
    """Under a sanitizer, prefer_g15 ABI-pins the host compiler to g++-15 so its
    sanitizer runtime (libtsan.so.2 / libasan.so) matches the lib*san the
    pytest/CI run-step preloads. An env CC/CXX naming a different GCC (e.g. the
    system g++ whose libtsan SONAME is .so.0) must NOT override that pin —
    scikit-build-core exports CXX during `pip install`, and the mismatch fails
    at dlopen with "cannot allocate memory in static TLS block"."""

    @pytest.fixture
    def toolchain(self, monkeypatch):
        monkeypatch.setattr("simpler_setup.toolchain._is_gcc", lambda _p: True)
        from simpler_setup.toolchain import GxxToolchain  # noqa: PLC0415

        return GxxToolchain(prefer_g15=True)

    def test_env_cxx_does_not_override_pin(self, toolchain, monkeypatch):
        # scikit-build-core / a plain dev shell exporting the system g++.
        monkeypatch.setenv("CC", "gcc")
        monkeypatch.setenv("CXX", "g++")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=g++-15" in args

    def test_conda_flags_preserved_but_compiler_pinned(self, toolchain, monkeypatch):
        # Conda's injected -B compiler_compat flags must survive, but the
        # compiler binary is still forced to the pinned g++-15.
        monkeypatch.setenv("CC", "gcc -pthread -B /data/envs/lyf/compiler_compat")
        monkeypatch.setenv("CXX", "g++ -pthread -B /data/envs/lyf/compiler_compat")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=g++-15" in args
        assert "-DCMAKE_C_FLAGS=-pthread -B /data/envs/lyf/compiler_compat" in args
        assert "-DCMAKE_CXX_FLAGS=-pthread -B /data/envs/lyf/compiler_compat" in args

    def test_custom_path_to_g15_is_preserved(self, toolchain, monkeypatch):
        # An env CC/CXX that already names g++-15 at a non-PATH location is the
        # right ABI already — keep it instead of forcing the bare name, which
        # might not be on PATH.
        monkeypatch.setenv("CC", "/opt/custom/bin/gcc-15")
        monkeypatch.setenv("CXX", "/opt/custom/bin/g++-15")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=/opt/custom/bin/gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=/opt/custom/bin/g++-15" in args

    def test_gnu_triplet_g15_is_preserved(self, toolchain, monkeypatch):
        # A versioned cross/triplet name is still GCC-15 — keep it.
        monkeypatch.setenv("CC", "aarch64-linux-gnu-gcc-15")
        monkeypatch.setenv("CXX", "aarch64-linux-gnu-g++-15")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++-15" in args

    def test_clang15_is_not_mistaken_for_pinned_gcc(self, toolchain, monkeypatch):
        # "g++-15" is a substring of "clang++-15", but clang's sanitizer ABI
        # differs — the pin must override it back to the GCC default.
        monkeypatch.setenv("CC", "clang-15")
        monkeypatch.setenv("CXX", "clang++-15")
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=g++-15" in args


class TestGxx15ToolchainCmakeArgs:
    """Same split behavior for the simulation-kernel toolchain."""

    @pytest.fixture
    def toolchain(self):
        from simpler_setup.toolchain import Gxx15Toolchain  # noqa: PLC0415

        return Gxx15Toolchain()

    def test_plain_env_defaults_to_version_15(self, toolchain, monkeypatch):
        # Defaults must match self.cxx_path so CMake uses the same compiler
        # that KernelCompiler uses for direct invocation.
        monkeypatch.delenv("CC", raising=False)
        monkeypatch.delenv("CXX", raising=False)
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc-15" in args
        assert "-DCMAKE_CXX_COMPILER=g++-15" in args

    def test_conda_env_splits_flags(self, toolchain, monkeypatch):
        monkeypatch.setenv("CC", "gcc -pthread -B /data/envs/lyf/compiler_compat")
        monkeypatch.delenv("CXX", raising=False)
        args = toolchain.get_cmake_args()
        assert "-DCMAKE_C_COMPILER=gcc" in args
        # CXX falls back to self.cxx_path = g++-15
        assert "-DCMAKE_CXX_COMPILER=g++-15" in args
        assert "-DCMAKE_C_FLAGS=-pthread -B /data/envs/lyf/compiler_compat" in args
        assert not any(a.startswith("-DCMAKE_CXX_FLAGS=") for a in args)


class TestShlexJoinSafety:
    """Flags containing spaces must survive CMake's shell-style re-parse of CMAKE_C_FLAGS."""

    def test_path_with_spaces_is_quoted(self, monkeypatch):
        from simpler_setup.toolchain import GxxToolchain  # noqa: PLC0415

        monkeypatch.setattr("simpler_setup.toolchain._is_gcc", lambda _p: True)
        monkeypatch.setenv("CC", "gcc -B '/opt/my compilers/compat'")
        monkeypatch.delenv("CXX", raising=False)
        args = GxxToolchain().get_cmake_args()
        flags_arg = next(a for a in args if a.startswith("-DCMAKE_C_FLAGS="))
        # shlex.join must re-quote the path so CMake re-parses it as one token.
        assert "'/opt/my compilers/compat'" in flags_arg
