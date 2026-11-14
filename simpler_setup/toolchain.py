# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TypedDict

from simpler import env_manager
from simpler.ccu import ccu_sdk_roots, runtime_build_identity

from .environment import PROJECT_ROOT


# Must match compile_strategy.h
class ToolchainType(IntEnum):
    """Toolchain types matching the C enum in compile_strategy.h."""

    CCEC = 0  # ccec (Ascend AICore compiler)
    HOST_GXX_15 = 1  # g++-15 (host, simulation kernels)
    HOST_GXX = 2  # g++ (host, orchestration .so)
    AARCH64_GXX = 3  # aarch64-target-linux-gnu-g++ (cross-compile)


def _is_gcc(cxx_path: str) -> bool:
    """Return True if *cxx_path* is a real GCC (not clang masquerading as g++)."""
    try:
        out = subprocess.run(
            [cxx_path, "--version"], check=False, capture_output=True, text=True, timeout=5
        ).stdout.lower()
        return "clang" not in out
    except (OSError, subprocess.SubprocessError):
        return False


def _parse_compiler_env(var_name: str, default: str) -> tuple[str, list[str]]:
    """Split a CC/CXX env var into (compiler_path, extra_flags).

    Conda activate scripts set CC/CXX to a multi-token string such as
    ``gcc -pthread -B <env>/compiler_compat``. CMake rejects that as
    ``CMAKE_C_COMPILER`` because it expects a single executable path, so we
    separate the leading token from the injected flags.
    """
    raw = os.environ.get(var_name, "")
    tokens = shlex.split(raw) if raw else []
    if not tokens:
        return default, []
    return tokens[0], tokens[1:]


def _matches_pinned_name(actual: str, pinned: str) -> bool:
    """True if *actual* (a CC/CXX value) names the *pinned* compiler.

    Matches the bare name or a versioned/triplet variant — ``g++-15``,
    ``/opt/tc/bin/g++-15``, ``aarch64-linux-gnu-g++-15`` — but NOT a substring
    false-positive like ``clang++-15`` (``"g++-15" in "clang++-15"`` is True),
    whose sanitizer-runtime ABI differs from GCC's.
    """
    base = os.path.basename(actual)
    return base == pinned or base.endswith(f"-{pinned}")


def _host_compiler_cmake_args(default_cc: str, default_cxx: str, pin_compiler: bool = False) -> list[str]:
    """CMake ``-D`` args for a host GCC/G++ toolchain.

    Reads CC/CXX from the environment and splits off any conda-injected flags
    (e.g. ``-pthread -B <env>/compiler_compat``) into CMAKE_{C,CXX}_FLAGS. The
    flags are re-joined with ``shlex.join`` so tokens containing spaces survive
    CMake's shell-style re-parse of the flag string.

    When ``pin_compiler`` is set, an env CC/CXX naming a *different* GCC is
    overridden by the caller's ``default_cc``/``default_cxx`` — but any
    env-injected flags are still preserved, and an env compiler that already
    names the pinned version (e.g. a custom path ``/opt/tc/bin/g++-15``) is kept
    so non-PATH installs still work. This is required under a sanitizer: the
    host compiler is ABI-pinned to g++-15 so its sanitizer runtime
    (libtsan.so.2 / libasan.so) matches the lib*san the run-step preloads.
    scikit-build-core exports CXX during ``pip install``; letting it name a
    different GCC (whose libtsan SONAME is .so.0) produces a runtime mismatch
    that fails at dlopen with "cannot allocate memory in static TLS block".
    """
    cc, cc_flags = _parse_compiler_env("CC", default_cc)
    cxx, cxx_flags = _parse_compiler_env("CXX", default_cxx)
    if pin_compiler:
        # Keep an env compiler that already names the pinned version (bare,
        # custom-path, or triplet) so non-PATH installs work; override anything
        # else — a different GCC, or clang++-15 whose runtime ABI differs.
        if not _matches_pinned_name(cc, os.path.basename(default_cc)):
            cc = default_cc
        if not _matches_pinned_name(cxx, os.path.basename(default_cxx)):
            cxx = default_cxx
    args = [
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]
    if cc_flags:
        args.append(f"-DCMAKE_C_FLAGS={shlex.join(cc_flags)}")
    if cxx_flags:
        args.append(f"-DCMAKE_CXX_FLAGS={shlex.join(cxx_flags)}")
    return args


class Toolchain:
    """Base class for all compile toolchains.

    A Toolchain represents a compiler identity: which compiler binary to use,
    what flags to pass, and what CMake -D arguments to generate.

    The Ascend SDK path is managed by env_manager. Call
    env_manager.ensure("ASCEND_HOME_PATH") before creating toolchains that
    need the Ascend SDK (CCECToolchain, Aarch64GxxToolchain, GxxToolchain
    with Ascend includes).

    Used by:
    - KernelCompiler: calls get_compile_flags() for direct single-file invocation
    - BuildTarget (in runtime_compiler.py): calls get_cmake_args() for CMake builds
    """

    # Host toolchains can carry a host sanitizer runtime; device toolchains
    # (ccec / aarch64 cross) run on the NPU and cannot. Overridden to True by
    # the host compilers (Gxx, Gxx15) below.
    is_host: bool = False

    cxx_path: str

    def __init__(self):
        self.ascend_home_path = env_manager.get("ASCEND_HOME_PATH")

    def get_compile_flags(self, **kwargs) -> list[str]:
        """Return base compiler flags for direct invocation."""
        raise NotImplementedError

    def get_cmake_args(self) -> list[str]:
        """Return compiler-specific CMake -D arguments."""
        raise NotImplementedError


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self, platform: str = "a2a3"):
        super().__init__()
        self.platform = platform

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for CCEC toolchain")

        self.cxx_path = os.path.join(self.ascend_home_path, "bin", "ccec")
        self.linker_path = os.path.join(self.ascend_home_path, "bin", "ld.lld")

        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"ccec compiler not found: {self.cxx_path}")
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(f"ccec linker not found: {self.linker_path}")

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> list[str]:
        # A5 uses dav-c310 architecture, A2A3 uses dav-c220
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        flags = [
            "-c",
            "-O3",
            "-g",
            "-x",
            "cce",
            "-Wall",
            "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=false",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
        ]
        # A5 hardware only: enable VF alias analysis between loop iterations.
        # a5sim uses Gxx15Toolchain (KernelCompiler leaves self.ccec unset), so this
        # LLVM option does not apply there. Stricter inter-iteration AA constrains
        # BiSheng VF fusion scale/correctness; BiSheng plans to make this always-on
        # later, so pin it for A5 CCEC builds now.
        if self.platform == "a5":
            # The A5 text-only loader cannot retain CANN's switch jump tables in .rodata.
            flags.extend(["-fno-jump-tables", "-mllvm", "-cce-vf-aa-between-iters=true"])
        return flags

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DBISHENG_CC={self.cxx_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    is_host = True

    def __init__(self):
        super().__init__()
        self.cxx_path = "g++-15"

    def get_compile_flags(self, core_type: str = "", **kwargs) -> list[str]:
        flags = [
            "-shared",
            "-O2",
            "-fPIC",
            "-std=c++23",
            "-fpermissive",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-DNDEBUG",
        ]
        # g++ does not define __DAV_VEC__/__DAV_CUBE__ like ccec does,
        # so we must add them explicitly based on core_type.
        if core_type == "aiv":
            flags.append("-D__DAV_VEC__")
        elif core_type == "aic":
            flags.append("-D__DAV_CUBE__")
        return flags

    def get_cmake_args(self) -> list[str]:
        # Default to gcc-15/g++-15 to match self.cxx_path used for direct compilation.
        return _host_compiler_cmake_args("gcc-15", self.cxx_path)


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation.

    ``prefer_g15`` switches the binary to g++-15/gcc-15. Used under a sanitizer:
    onboard and sim runtimes, helpers, and orchestration must share the SAME
    host compiler as the sim kernels (which are always g++-15), because mixing
    g++ and g++-15 sanitizer runtimes is an ABI mismatch that fails at `.so`
    load.
    """

    is_host = True

    def __init__(self, prefer_g15: bool = False):
        super().__init__()
        self.cxx_path = "g++-15" if prefer_g15 else "g++"
        self._prefer_g15 = prefer_g15
        self._gcc = _is_gcc(self.cxx_path)

    def get_compile_flags(self, **kwargs) -> list[str]:
        flags = ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]
        # -fno-gnu-unique: prevent STB_GNU_UNIQUE binding so dlclose actually
        # unloads the SO.  GCC-only; clang does not produce STB_GNU_UNIQUE.
        if self._gcc:
            flags.append("-fno-gnu-unique")
        return flags

    def get_cmake_args(self) -> list[str]:
        # Under prefer_g15 (sanitizer build) the compiler is ABI-pinned: env
        # CC/CXX must not redirect it away from g++-15, or the built .so link a
        # different lib*san than the run-step preloads. See pin_compiler above.
        args = _host_compiler_cmake_args(
            "gcc-15" if self._prefer_g15 else "gcc",
            self.cxx_path,
            pin_compiler=self._prefer_g15,
        )
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args


class Aarch64GxxToolchain(Toolchain):
    """aarch64 cross-compiler for device code."""

    def __init__(self):
        super().__init__()

        if self.ascend_home_path is None:
            raise RuntimeError("ASCEND_HOME_PATH is required for aarch64 toolchain")

        self.cxx_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-g++",
        )
        self.cc_path = os.path.join(
            self.ascend_home_path,
            "tools",
            "hcc",
            "bin",
            "aarch64-target-linux-gnu-gcc",
        )
        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(f"aarch64 C compiler not found: {self.cc_path}")
        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(f"aarch64 C++ compiler not found: {self.cxx_path}")
        self._gcc = _is_gcc(self.cxx_path)

    def get_compile_flags(self, **kwargs) -> list[str]:
        flags = [
            "-shared",
            "-fPIC",
            "-O3",
            "-g",
            "-std=c++17",
        ]
        # -fno-gnu-unique: prevent STB_GNU_UNIQUE binding so dlclose actually unloads the SO.
        if self._gcc:
            flags.append("-fno-gnu-unique")
        return flags

    def get_cmake_args(self) -> list[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]


_CCU_ABI_HEADER = "platform_comm/ccu/ccu_native_abi.h"
_CCU_SDK_HEADERS = (
    "include/acl/acl.h",
    "include/hcomm/ccu/ccu_res.h",
    "include/hcomm/ccu/ccu_primitives.hpp",
    "include/hcomm/ccu/ccu_types.h",
    "include/hccl/hccl_types.h",
    "include/hccl/hcomm_primitives.h",
    "pkg_inc/hcomm/ccu/ccu_primitives_impl.h",
)
_CCU_LIBRARIES = ("asccomm_ccu", "hcomm")
_CCU_SDK_INCLUDE_DIRS = ("include", "include/hccl", "pkg_inc/hcomm/ccu", "include/hcomm/ccu")
_CCU_ELF_MACHINES = {"x86_64": 62, "aarch64": 183}


class _CcuCacheInputs(TypedDict):
    cxx_path: str
    compile_flags: tuple[str, ...]
    include_dirs: tuple[str, ...]
    link_flags: tuple[str, ...]
    library_paths: tuple[str, ...]
    host_target: str
    pto_isa_commit: str
    manifest_identity_header: str
    dependency_digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _CcuBuildInputs:
    """Ordered compiler and linker inputs for one CCU build."""

    cxx_path: str
    compile_flags: tuple[str, ...]
    include_dirs: tuple[Path, ...]
    link_flags: tuple[str, ...]
    library_paths: tuple[Path, ...]
    sdk_build_id: bytes
    pto_isa_build_id: bytes
    pto_isa_commit: str
    host_target: str
    # Each pair records the lookup path and SHA-256 of its resolved contents.
    dependency_digests: tuple[tuple[str, str], ...]

    def manifest_identity_header(self) -> str:
        """Return a C++17 header defining the IDs to copy into the manifest."""
        return _ccu_manifest_identity_header(self.sdk_build_id, self.pto_isa_build_id)

    @property
    def cache_token(self) -> _CcuCacheInputs:
        """Return the content identities and ordered inputs for this build.

        For a module cache key, also hash each source and its dependencies.
        Use a compiler dependency scan that includes system headers (e.g. -M).
        """
        return {
            "cxx_path": self.cxx_path,
            "compile_flags": self.compile_flags,
            "include_dirs": tuple(map(str, self.include_dirs)),
            "link_flags": self.link_flags,
            "library_paths": tuple(map(str, self.library_paths)),
            "host_target": self.host_target,
            "pto_isa_commit": self.pto_isa_commit,
            "manifest_identity_header": self.manifest_identity_header(),
            "dependency_digests": self.dependency_digests,
        }


def _ccu_manifest_identity_header(sdk_build_id: bytes, pto_isa_build_id: bytes) -> str:
    sdk = ", ".join(map(str, sdk_build_id))
    isa = ", ".join(map(str, pto_isa_build_id))
    return (
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace simpler::ccu {\n"
        f"inline constexpr std::uint8_t sdk_build_id[32] = {{{sdk}}};\n"
        f"inline constexpr std::uint8_t pto_isa_build_id[32] = {{{isa}}};\n"
        "}  // namespace simpler::ccu\n"
    )


def _ccu_executable(name: str, *, required: bool = True) -> Path | None:
    path = shutil.which(name)
    if path:
        # A compiler wrapper may choose its underlying compiler from argv[0].
        return Path(path).absolute()
    if required:
        searched = os.environ.get("PATH", os.defpath) if os.path.dirname(name) == "" else str(Path(name).parent)
        raise FileNotFoundError(f"CCU Host tool {name!r} not found; searched {searched}")
    return None


def _ccu_query(command: list[str], *, stderr: bool = False) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"CCU Host compiler query failed: {command!r}\n{exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot query CCU Host compiler: {command!r}: {exc}") from exc
    return (result.stderr if stderr else result.stdout).strip()


def _ccu_compiler_dependencies(driver: list[str], flags: tuple[str, ...]) -> Iterator[Path]:
    # GCC reports the underlying driver even when argv[0] selects it via ccache.
    for line in _ccu_query([*driver, "-v", "-E", "-x", "c++", os.devnull, "-o", os.devnull], stderr=True).splitlines():
        if line.startswith("COLLECT_GCC="):
            actual = _ccu_executable(line.removeprefix("COLLECT_GCC="))
            if actual is not None:
                yield actual

    linker = "ld"
    for flag in flags:
        if flag.startswith("-fuse-ld="):
            choice = flag.split("=", 1)[1]
            linker = choice if "/" in choice else f"ld.{choice}"
        elif flag.startswith("--ld-path="):
            linker = flag.split("=", 1)[1]
    for name in dict.fromkeys((linker, "cc1plus", "collect2", "as", "lto-wrapper")):
        program = _ccu_query([*driver, f"-print-prog-name={name}"])
        path = _ccu_executable(program, required=name == linker)
        if path is not None:
            yield path

    path = Path(_ccu_query([*driver, "-print-file-name=liblto_plugin.so"]))
    if path.is_file() and path != Path("liblto_plugin.so"):
        yield path


def _ccu_expand_response_files(arguments: Sequence[str], parents: tuple[Path, ...] = ()) -> Iterator[str]:
    for argument in arguments:
        if not argument.startswith("@") or not (path := Path(argument[1:])).is_file():
            yield argument
            continue
        resolved = path.resolve()
        if resolved in parents:
            raise ValueError(f"Recursive CCU compiler response file: {path}")
        # GCC response files allow backslash escapes inside either quote style.
        words, word = [], []
        quote = ""
        escaped = started = False
        for char in path.read_text():
            if escaped:
                word.append(char)
                escaped = False
            elif char == "\\":
                escaped = started = True
            elif quote:
                if char == quote:
                    quote = ""
                else:
                    word.append(char)
            elif char in "\"'":
                quote = char
                started = True
            elif char in " \t\r\n\f\v":
                if started:
                    words.append("".join(word))
                    word = []
                    started = False
            else:
                word.append(char)
                started = True
        if started:
            words.append("".join(word))
        yield from _ccu_expand_response_files(words, (*parents, resolved))


def _ccu_probe_link_command(command: list[str]) -> list[str]:
    result = []
    arguments = iter(_ccu_expand_response_files(command))
    for argument in arguments:
        if argument == "-Xlinker":
            forwarded = [next(arguments)]
        elif argument.startswith("-Wl,"):
            forwarded = argument[4:].split(",")
        else:
            result.append(argument)
            continue
        for flag in _ccu_expand_response_files(forwarded):
            if flag != "--no-undefined-version":
                result.extend(("-Xlinker", flag))
    # Symbol definitions belong to the real sources, absent from this probe.
    return [*result, "-Wl,--unresolved-symbols=ignore-all"]


def _ccu_link_dependencies(command: list[str], output: Path) -> tuple[Path, ...]:
    # GNU ld traces opened archives even when the probe extracts no members.
    # Linkers that trace only extracted members cannot identify unused runtimes.
    version = _ccu_query([*command, "-Wl,--version"])
    if not any(line.startswith("GNU ld ") for line in version.splitlines()):
        raise ValueError("CCU Host builds require GNU ld (bfd) for a complete runtime-library fingerprint")
    trace = _ccu_query([*_ccu_probe_link_command(command), "-Wl,--trace"])
    paths = tuple(
        path
        for line in trace.splitlines()
        if (path := Path(line.strip())).is_file() and not path.resolve().is_relative_to(output.parent)
    )
    if not output.is_file() or not paths:
        raise RuntimeError("CCU Host linker probe did not produce an artifact and dependency trace")
    return paths


def _ccu_sdk_file(roots: tuple[Path, Path], relative: str, machine: str) -> Path:
    candidates = tuple(root / relative for root in roots)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"CANN CCU dependency {relative} not found; searched {', '.join(map(str, candidates))}")
    installation = roots[0]
    resolved = path.resolve()
    if not resolved.is_relative_to(installation) or any(
        resolved.is_relative_to(installation / f"{other}-linux") for other in _CCU_ELF_MACHINES if other != machine
    ):
        raise ValueError(f"CANN CCU dependency {path} resolves outside the selected {machine} SDK: {resolved}")
    return path


def _ccu_sdk_headers(directory: Path, ancestors: tuple[Path, ...] = ()) -> Iterator[Path]:
    resolved = directory.resolve()
    if resolved in ancestors:
        return
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            yield from _ccu_sdk_headers(path, (*ancestors, resolved))
        elif path.is_file():
            yield path


def _ccu_file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_ccu_build_inputs(
    toolchain: CCUToolchain,
    *,
    extra_compile_flags: Sequence[str] = (),
    extra_link_flags: Sequence[str] = (),
    extra_include_dirs: Sequence[str] = (),
) -> _CcuBuildInputs:
    """Discover a native CCU module toolchain for the current Linux Host.

    Set ASCEND_HOME_PATH to the CANN installation. The caller owns compiler
    invocation, source dependencies, output files and caching. KernelCompiler
    supplies the selected toolchain and instrumentation flags.
    The pinned PTO-ISA checkout is prepared on demand and may require network
    access when the local checkout is unavailable.
    """
    from .pto_isa import read_pto_isa_pin  # noqa: PLC0415 -- only needed when building CCU

    roots, include_dirs, library_paths = _ccu_sdk_layout()
    machine = platform.machine()
    abi_header = include_dirs[0] / _CCU_ABI_HEADER
    compiler = _ccu_executable(toolchain.cxx_path)
    assert compiler is not None
    compile_flags = (*toolchain.get_compile_flags(), *extra_compile_flags)
    link_flags = (*toolchain.get_link_flags(), *extra_link_flags)
    flags = (*compile_flags, *link_flags)
    for flag in flags:
        option, _, value = flag.partition("=")
        if option in ("-march", "-mcpu", "-mtune") and value.split("+", 1)[0] == "native":
            raise ValueError(f"CCU Host option {flag} depends on the build CPU; pass an explicit CPU target")
    driver = [str(compiler), *flags]
    target = _ccu_query([*driver, "-dumpmachine"])
    if not target.startswith(f"{machine}-") or "linux" not in target.split("-"):
        raise ValueError(f"CCU Host compiler {compiler} targets {target!r}; expected {machine} Linux for the SDK")
    # Probe with the same generated header and include paths as the build,
    # including forced includes supplied through caller-owned flags.
    sdk_id, isa_id = runtime_build_identity()
    with tempfile.TemporaryDirectory(prefix="ccu-probe-") as directory:
        (Path(directory) / "ccu_manifest_identity.h").write_text(
            _ccu_manifest_identity_header(sdk_id, isa_id), encoding="ascii"
        )
        probe_driver = [*driver, f"-I{directory}", *(f"-I{path}" for path in (*include_dirs, *extra_include_dirs))]
        # The triplet alone does not reflect target modifiers such as -m32.
        macros = _ccu_query([*probe_driver, "-dM", "-E", "-x", "c++", os.devnull]).splitlines()
        if not {f"#define __{machine}__ 1", "#define __LP64__ 1", "#define __linux__ 1"}.issubset(macros):
            raise ValueError(f"CCU Host compiler {compiler} flags must select 64-bit {machine} Linux for the SDK")
        compiler_files = tuple(_ccu_compiler_dependencies(probe_driver, flags))
        output = Path(directory) / "probe.so"
        link_files = _ccu_link_dependencies(
            [
                str(compiler),
                *compile_flags,
                f"-I{directory}",
                *(f"-I{path}" for path in (*include_dirs, *extra_include_dirs)),
                "-x",
                "c++",
                os.devnull,
                "-x",
                "none",
                *link_flags,
                *(str(path) for path in library_paths),
                "-o",
                str(output),
            ],
            output,
        )
    # The hccl and public ccu include directories are already under include/.
    header_dirs = (
        root / relative
        for root in roots
        for relative in ("include", "pkg_inc/hcomm/ccu")
        if root / relative in include_dirs
    )
    dependencies = dict.fromkeys(
        (
            compiler,
            *compiler_files,
            *link_files,
            *library_paths,
            abi_header,
            *(header for directory in header_dirs for header in _ccu_sdk_headers(directory)),
        )
    )
    digests = {}
    dependency_digests = []
    for path in dependencies:
        resolved = path.resolve()
        if resolved not in digests:
            digests[resolved] = _ccu_file_digest(resolved)
        dependency_digests.append((str(path), digests[resolved]))
    return _CcuBuildInputs(
        cxx_path=str(compiler),
        compile_flags=compile_flags,
        include_dirs=include_dirs,
        link_flags=link_flags,
        library_paths=tuple(library_paths),
        sdk_build_id=sdk_id,
        pto_isa_build_id=isa_id,
        pto_isa_commit=read_pto_isa_pin(),
        host_target=target,
        dependency_digests=tuple(dependency_digests),
    )


def _ccu_sdk_layout() -> tuple[tuple[Path, Path], tuple[Path, ...], tuple[Path, ...]]:
    installation, arch = ccu_sdk_roots()
    installation = installation.resolve()
    roots = (installation, installation / arch.name)
    machine = platform.machine()
    if machine not in _CCU_ELF_MACHINES:
        raise ValueError(f"Unsupported CCU Host architecture: {machine}")
    common = PROJECT_ROOT / "src/common"
    abi_header = common / _CCU_ABI_HEADER
    if not abi_header.is_file():
        raise FileNotFoundError(f"simpler CCU ABI header not found; searched {abi_header}")
    for header in _CCU_SDK_HEADERS:
        _ccu_sdk_file(roots, header, machine)

    library_paths = []
    for name in _CCU_LIBRARIES:
        path = _ccu_sdk_file(roots, f"lib64/lib{name}.so", machine)
        with path.open("rb") as stream:
            elf = stream.read(20)
        if elf[:6] != b"\x7fELF\x02\x01" or elf[16:20] != b"\x03\x00" + _CCU_ELF_MACHINES[machine].to_bytes(
            2, "little"
        ):
            raise ValueError(f"CANN CCU library {path} is not a {machine} Linux ELF shared library")
        library_paths.append(path)

    sdk_includes = {}
    for root in roots:
        for relative in _CCU_SDK_INCLUDE_DIRS:
            directory = root / relative
            if directory.is_dir():
                sdk_includes.setdefault(directory.resolve(), directory)
    include_dirs = (common, *sdk_includes.values())
    return roots, include_dirs, tuple(library_paths)


class CCUToolchain(Toolchain):
    """Host C++ toolchain for native CCU modules on A5.

    SDK paths are discovered when queried. KernelCompiler shares its Host
    compiler with this toolchain, including the sanitizer compiler selection.
    """

    is_host = True

    def __init__(self, host_toolchain: GxxToolchain):
        super().__init__()
        self.host_toolchain = host_toolchain

    @property
    def cxx_path(self) -> str:
        return self.host_toolchain.cxx_path

    def get_compile_flags(self, **kwargs) -> list[str]:
        return [flag for flag in self.host_toolchain.get_compile_flags() if flag != "-shared"]

    def get_link_flags(self) -> list[str]:
        return ["-shared"]

    def get_cmake_args(self) -> list[str]:
        return self.host_toolchain.get_cmake_args()

    def get_include_dirs(self) -> list[str]:
        return list(map(str, _ccu_sdk_layout()[1]))

    def get_library_paths(self) -> list[str]:
        """Return the exact validated libraries, in link order."""
        return list(map(str, _ccu_sdk_layout()[2]))

    def _build_inputs(
        self,
        *,
        extra_compile_flags: Sequence[str] = (),
        extra_link_flags: Sequence[str] = (),
        extra_include_dirs: Sequence[str] = (),
    ) -> _CcuBuildInputs:
        return _discover_ccu_build_inputs(
            self,
            extra_compile_flags=extra_compile_flags,
            extra_link_flags=extra_link_flags,
            extra_include_dirs=extra_include_dirs,
        )
