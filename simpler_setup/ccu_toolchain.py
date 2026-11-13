# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build configuration for Linux Host CCU modules."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from simpler.ccu import ccu_sdk_roots, runtime_build_identity

from .environment import PROJECT_ROOT
from .toolchain import GxxToolchain

_ABI_HEADER = "platform_comm/ccu/ccu_native_abi.h"
_SDK_HEADERS = (
    "include/acl/acl.h",
    "include/hcomm/ccu/ccu_res.h",
    "include/hcomm/ccu/ccu_primitives.hpp",
    "include/hcomm/ccu/ccu_types.h",
    "include/hccl/hccl_types.h",
    "include/hccl/hcomm_primitives.h",
    "pkg_inc/hcomm/ccu/ccu_primitives_impl.h",
)
_LIBRARIES = ("asccomm_ccu", "hcomm")
_SDK_INCLUDE_DIRS = ("include", "include/hccl", "pkg_inc/hcomm/ccu", "include/hcomm/ccu")
_ELF_MACHINES = {"x86_64": 62, "aarch64": 183}
_SANITIZER_LIBRARIES = {"address": "asan", "undefined": "ubsan", "thread": "tsan", "leak": "lsan"}


@dataclass(frozen=True)
class CcuToolchain:
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
        sdk = ", ".join(map(str, self.sdk_build_id))
        isa = ", ".join(map(str, self.pto_isa_build_id))
        return (
            "#pragma once\n\n"
            "#include <cstdint>\n\n"
            "namespace simpler::ccu {\n"
            f"inline constexpr std::uint8_t sdk_build_id[32] = {{{sdk}}};\n"
            f"inline constexpr std::uint8_t pto_isa_build_id[32] = {{{isa}}};\n"
            "}  // namespace simpler::ccu\n"
        )

    @property
    def build_identity(self) -> str:
        """Return the SHA-256 identity of the inputs listed by this toolchain.

        For a module cache key, also hash each source and its dependencies.
        Use a compiler dependency scan that includes system headers (e.g. -M).
        """
        inputs = {
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
        return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _executable(name: str, *, required: bool = True) -> Path | None:
    path = shutil.which(name)
    if path:
        # A compiler wrapper may choose its underlying compiler from argv[0].
        return Path(path).absolute()
    if required:
        searched = os.environ.get("PATH", os.defpath) if os.path.dirname(name) == "" else str(Path(name).parent)
        raise FileNotFoundError(f"CCU Host tool {name!r} not found; searched {searched}")
    return None


def _query(command: list[str], *, stderr: bool = False) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"CCU Host compiler query failed: {command!r}\n{exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot query CCU Host compiler: {command!r}: {exc}") from exc
    return (result.stderr if stderr else result.stdout).strip()


def _compiler_dependencies(driver: list[str], flags: tuple[str, ...]) -> Iterator[Path]:
    # GCC reports the underlying driver even when argv[0] selects it via ccache.
    for line in _query([*driver, "-v", "-E", "-x", "c++", os.devnull, "-o", os.devnull], stderr=True).splitlines():
        if line.startswith("COLLECT_GCC="):
            actual = _executable(line.removeprefix("COLLECT_GCC="))
            if actual is not None:
                yield actual

    linker = "ld"
    runtimes = []
    for flag in flags:
        if flag.startswith("-fuse-ld="):
            choice = flag.split("=", 1)[1]
            linker = choice if "/" in choice else f"ld.{choice}"
        elif flag.startswith("--ld-path="):
            linker = flag.split("=", 1)[1]
        elif flag.startswith("-fsanitize="):
            for name in flag.split("=", 1)[1].split(","):
                if name in _SANITIZER_LIBRARIES:
                    runtime = _SANITIZER_LIBRARIES[name]
                    extension = "a" if f"-static-lib{runtime}" in flags else "so"
                    runtimes.append(f"lib{runtime}.{extension}")
    for name in dict.fromkeys((linker, "cc1plus", "collect2", "as", "lto-wrapper")):
        program = _query([*driver, f"-print-prog-name={name}"])
        path = _executable(program, required=name == linker)
        if path is not None:
            yield path

    libstdcxx = "libstdc++.a" if "-static-libstdc++" in flags else "libstdc++.so"
    # GCC can use libgcc.a even when the shared runtime is linked.
    for name in dict.fromkeys(
        (
            "liblto_plugin.so",
            libstdcxx,
            "libgcc_s.so",
            "libgcc.a",
            "libgcc_eh.a",
            "crtbeginS.o",
            "crtendS.o",
            "crti.o",
            "crtn.o",
            *runtimes,
        )
    ):
        path = Path(_query([*driver, f"-print-file-name={name}"]))
        if path.is_file() and path != Path(name):
            yield path


def _sdk_file(roots: tuple[Path, Path], relative: str, machine: str) -> Path:
    candidates = tuple(root / relative for root in roots)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"CANN CCU dependency {relative} not found; searched {', '.join(map(str, candidates))}")
    installation = roots[0]
    resolved = path.resolve()
    if not resolved.is_relative_to(installation) or any(
        resolved.is_relative_to(installation / f"{other}-linux") for other in _ELF_MACHINES if other != machine
    ):
        raise ValueError(f"CANN CCU dependency {path} resolves outside the selected {machine} SDK: {resolved}")
    return path


def _sdk_headers(directory: Path, ancestors: tuple[Path, ...] = ()) -> Iterator[Path]:
    resolved = directory.resolve()
    if resolved in ancestors:
        return
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            yield from _sdk_headers(path, (*ancestors, resolved))
        elif path.is_file():
            yield path


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_ccu_toolchain(
    host_toolchain: GxxToolchain | None = None,
    *,
    extra_compile_flags: Sequence[str] = (),
    extra_link_flags: Sequence[str] = (),
) -> CcuToolchain:
    """Discover a native CCU module toolchain for the current Linux Host.

    Set ASCEND_HOME_PATH to the CANN installation. The caller owns compiler
    invocation, source dependencies, output files and caching. Pass the selected
    Host toolchain and instrumentation flags when building with sanitizers.
    The pinned PTO-ISA checkout is prepared on demand and may require network
    access when the local checkout is unavailable.
    """
    from .pto_isa import read_pto_isa_pin  # noqa: PLC0415 -- only needed when building CCU

    installation, arch = ccu_sdk_roots()
    installation = installation.resolve()
    roots = (installation, installation / arch.name)
    machine = platform.machine()
    if machine not in _ELF_MACHINES:
        raise ValueError(f"Unsupported CCU Host architecture: {machine}")
    common = PROJECT_ROOT / "src/common"
    abi_header = common / _ABI_HEADER
    if not abi_header.is_file():
        raise FileNotFoundError(f"simpler CCU ABI header not found; searched {abi_header}")
    for header in _SDK_HEADERS:
        _sdk_file(roots, header, machine)

    library_paths = []
    for name in _LIBRARIES:
        path = _sdk_file(roots, f"lib64/lib{name}.so", machine)
        with path.open("rb") as stream:
            elf = stream.read(20)
        if elf[:6] != b"\x7fELF\x02\x01" or elf[16:20] != b"\x03\x00" + _ELF_MACHINES[machine].to_bytes(2, "little"):
            raise ValueError(f"CANN CCU library {path} is not a {machine} Linux ELF shared library")
        library_paths.append(path)

    host = host_toolchain if host_toolchain is not None else GxxToolchain()
    compiler = _executable(host.cxx_path)
    assert compiler is not None
    compile_flags = tuple(flag for flag in host.get_compile_flags() if flag != "-shared") + tuple(extra_compile_flags)
    link_flags = ("-shared", *extra_link_flags)
    flags = (*compile_flags, *link_flags)
    for flag in flags:
        option, _, value = flag.partition("=")
        if option in ("-march", "-mcpu", "-mtune") and value.split("+", 1)[0] == "native":
            raise ValueError(f"CCU Host option {flag} depends on the build CPU; pass an explicit CPU target")
    driver = [str(compiler), *flags]
    target = _query([*driver, "-dumpmachine"])
    if not target.startswith(f"{machine}-") or "linux" not in target.split("-"):
        raise ValueError(f"CCU Host compiler {compiler} targets {target!r}; expected {machine} Linux for the SDK")
    # The triplet alone does not reflect target modifiers such as -m32.
    macros = _query([*driver, "-dM", "-E", "-x", "c++", os.devnull]).splitlines()
    if not {f"#define __{machine}__ 1", "#define __LP64__ 1", "#define __linux__ 1"}.issubset(macros):
        raise ValueError(f"CCU Host compiler {compiler} flags must select 64-bit {machine} Linux for the SDK")
    compiler_files = _compiler_dependencies(driver, flags)
    sdk_includes = {}
    for root in roots:
        for relative in _SDK_INCLUDE_DIRS:
            directory = root / relative
            if directory.is_dir():
                sdk_includes.setdefault(directory.resolve(), directory)
    include_dirs = (common, *sdk_includes.values())
    # The hccl and public ccu include directories are already under include/.
    header_dirs = (
        root / relative
        for root in roots
        for relative in ("include", "pkg_inc/hcomm/ccu")
        if root / relative in sdk_includes.values()
    )
    dependencies = dict.fromkeys(
        (
            compiler,
            *compiler_files,
            *library_paths,
            abi_header,
            *(header for directory in header_dirs for header in _sdk_headers(directory)),
        )
    )
    digests = {}
    dependency_digests = []
    for path in dependencies:
        resolved = path.resolve()
        if resolved not in digests:
            digests[resolved] = _file_digest(resolved)
        dependency_digests.append((str(path), digests[resolved]))
    sdk_id, isa_id = runtime_build_identity()
    return CcuToolchain(
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
