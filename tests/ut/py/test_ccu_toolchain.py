# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from simpler.ccu import runtime_build_identity
from simpler.task_interface import ChipCallable

from simpler_setup import ccu_toolchain, environment, pto_isa
from simpler_setup.ccu_toolchain import get_ccu_toolchain
from simpler_setup.toolchain import GxxToolchain


@pytest.fixture
def toolchain_files(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    project = tmp_path / "project"
    isa = tmp_path / "isa"
    compiler = tmp_path / "g++"
    files = (
        *((sdk / header, header.encode()) for header in ccu_toolchain._SDK_HEADERS),
        (
            project / "src/common" / ccu_toolchain._ABI_HEADER,
            (environment.PROJECT_ROOT / "src/common" / ccu_toolchain._ABI_HEADER).read_bytes(),
        ),
        (project / "pto_isa.pin", b"a" * 40),
        (isa / "include/pto/pto-inst.hpp", b"pinned ISA"),
        (compiler, b"compiler driver"),
        (tmp_path / "ld", b"linker"),
    )
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    lib = sdk / "lib64"
    lib.mkdir()
    elf = b"\x7fELF\x02\x01" + bytes(10) + b"\x03\x00" + (183).to_bytes(2, "little")
    for name in ccu_toolchain._LIBRARIES:
        (lib / f"lib{name}.so").write_bytes(elf + name.encode())
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.setattr(ccu_toolchain, "PROJECT_ROOT", project)
    monkeypatch.setattr(pto_isa, "PROJECT_ROOT", project)
    monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda: str(isa))

    def which(name):
        if name in ("g++", "ld"):
            return str(tmp_path / name)
        return None

    def query(command, *, stderr=False):
        if stderr:
            return ""
        if command[-1] == "-dumpmachine":
            return "aarch64-linux-gnu"
        if "-dM" in command:
            return "#define __aarch64__ 1\n#define __LP64__ 1\n#define __linux__ 1"
        kind, name = command[-1].split("=", 1)
        path = tmp_path / name
        return str(path) if kind == "-print-file-name" and path.is_file() else name

    monkeypatch.setattr(ccu_toolchain.shutil, "which", which)
    monkeypatch.setattr(ccu_toolchain, "_query", query)
    monkeypatch.setattr("simpler_setup.toolchain._is_gcc", lambda _: True)
    return sdk, project, compiler


def test_toolchain_config_and_manifest_header(toolchain_files):
    sdk, project, compiler = toolchain_files
    toolchain = get_ccu_toolchain()
    assert toolchain.cxx_path == str(compiler)
    assert toolchain.compile_flags == ("-fPIC", "-O3", "-g", "-std=c++17", "-fno-gnu-unique")
    assert toolchain.link_flags == ("-shared",)
    assert toolchain.include_dirs == (
        project / "src/common",
        sdk / "include",
        sdk / "include/hccl",
        sdk / "pkg_inc/hcomm/ccu",
        sdk / "include/hcomm/ccu",
    )
    assert toolchain.library_paths == (sdk / "lib64/libasccomm_ccu.so", sdk / "lib64/libhcomm.so")
    assert (toolchain.sdk_build_id, toolchain.pto_isa_build_id) == runtime_build_identity()
    for name, value in (("sdk_build_id", toolchain.sdk_build_id), ("pto_isa_build_id", toolchain.pto_isa_build_id)):
        assert f"{name}[32] = {{{', '.join(map(str, value))}}};" in toolchain.manifest_identity_header()
    assert toolchain.build_identity == get_ccu_toolchain().build_identity


def test_toolchain_cache_identity_changes(toolchain_files):
    sdk, project, compiler = toolchain_files
    baseline = get_ccu_toolchain().build_identity
    instrumented = get_ccu_toolchain(
        GxxToolchain(), extra_compile_flags=("-O1", "-fsanitize=address"), extra_link_flags=("-fsanitize=address",)
    )
    assert instrumented.compile_flags[-2:] == ("-O1", "-fsanitize=address")
    assert instrumented.link_flags[-1] == "-fsanitize=address"
    assert instrumented.build_identity != baseline
    assert (
        get_ccu_toolchain(extra_compile_flags=("-O2", "-O1")).build_identity
        != get_ccu_toolchain(extra_compile_flags=("-O1", "-O2")).build_identity
    )

    nested = sdk / "include/hcomm/ccu/nested.h"
    nested.touch()
    previous = baseline
    for path in (
        compiler,
        compiler.with_name("ld"),
        sdk / "lib64/libhcomm.so",
        sdk / "pkg_inc/hcomm/ccu/ccu_primitives_impl.h",
        nested,
        project / "src/common" / ccu_toolchain._ABI_HEADER,
        project / "pto_isa.pin",
    ):
        path.write_bytes(b"b" * 40 if path.name == "pto_isa.pin" else path.read_bytes() + b"changed")
        identity = get_ccu_toolchain().build_identity
        assert identity != previous
        previous = identity


@pytest.mark.parametrize("missing", ["pkg_inc/hcomm/ccu/ccu_primitives_impl.h", "lib64/libasccomm_ccu.so"])
def test_toolchain_reports_missing_sdk_inputs(toolchain_files, missing):
    sdk, _, _ = toolchain_files
    (sdk / missing).unlink()
    with pytest.raises(FileNotFoundError, match="CANN CCU dependency") as error:
        get_ccu_toolchain()
    assert str(sdk / missing) in str(error.value)


def test_toolchain_reports_missing_compiler(toolchain_files, monkeypatch):
    monkeypatch.setattr(ccu_toolchain.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="g\\+\\+.*searched"):
        get_ccu_toolchain()


def test_toolchain_rejects_compiler_target_mismatch(toolchain_files, monkeypatch):
    monkeypatch.setattr(ccu_toolchain, "_query", lambda _: "x86_64-linux-gnu")
    with pytest.raises(ValueError, match="expected aarch64 Linux"):
        get_ccu_toolchain()


def test_toolchain_validates_target_options_in_link_flags(toolchain_files, monkeypatch):
    original_query = ccu_toolchain._query

    def query(command, *, stderr=False):
        if "-dM" in command and "-mabi=ilp32" in command:
            return "#define __aarch64__ 1\n#define __linux__ 1"
        return original_query(command, stderr=stderr)

    monkeypatch.setattr(ccu_toolchain, "_query", query)
    with pytest.raises(ValueError, match="flags must select 64-bit aarch64 Linux"):
        get_ccu_toolchain(extra_link_flags=("-mabi=ilp32",))


def test_toolchain_returns_the_sdk_library_it_fingerprints(toolchain_files, tmp_path):
    sdk, _, _ = toolchain_files
    library = sdk / "lib64/libhcomm.so"
    shadow = tmp_path / "other/libhcomm.so"
    shadow.parent.mkdir()
    shadow.write_bytes(library.read_bytes() + b"other")

    config = get_ccu_toolchain(extra_compile_flags=(f"-L{shadow.parent}",))
    assert config.library_paths[-1] == library
    assert (str(config.library_paths[-1]), hashlib.sha256(library.read_bytes()).hexdigest()) in (
        config.dependency_digests
    )


@pytest.mark.parametrize("flag", ["-march=native", "-mcpu=native+crc", "-mtune=native"])
def test_toolchain_rejects_cpu_dependent_flags(toolchain_files, flag):
    with pytest.raises(ValueError, match="explicit CPU target"):
        get_ccu_toolchain(extra_compile_flags=(flag,))


@pytest.mark.parametrize(
    ("flags", "archive"),
    [
        (("-static-libstdc++",), "libstdc++.a"),
        (("-static-libgcc",), "libgcc_eh.a"),
        (("-fsanitize=address", "-static-libasan"), "libasan.a"),
    ],
)
def test_static_runtime_update_invalidates_build_identity(toolchain_files, flags, archive):
    _, _, compiler = toolchain_files
    path = compiler.with_name(archive)
    path.write_bytes(b"archive")
    original = get_ccu_toolchain(extra_link_flags=flags)
    path.write_bytes(b"updated archive")
    assert original.build_identity != get_ccu_toolchain(extra_link_flags=flags).build_identity


def test_toolchain_rejects_foreign_sdk_library(toolchain_files):
    sdk, _, _ = toolchain_files
    library = sdk / "lib64/libhcomm.so"
    library.write_bytes(library.read_bytes()[:18] + (62).to_bytes(2, "little"))
    with pytest.raises(ValueError, match="libhcomm.so.*aarch64 Linux ELF"):
        get_ccu_toolchain()


@pytest.mark.parametrize("split_public_headers", [False, True])
def test_toolchain_supports_split_sdk_layout(toolchain_files, split_public_headers):
    sdk, _, _ = toolchain_files
    host = sdk / "aarch64-linux"
    host.mkdir()
    (sdk / "pkg_inc").rename(host / "pkg_inc")
    (host / "lib64").mkdir()
    (sdk / "lib64/libasccomm_ccu.so").rename(host / "lib64/libasccomm_ccu.so")
    (host / "include").mkdir()
    (sdk / "include/hccl").rename(host / "include/hccl")
    if split_public_headers:
        (sdk / "include/hcomm").rename(host / "include/hcomm")

    config = get_ccu_toolchain()
    assert sdk / "include" in config.include_dirs
    assert host / "include/hccl" in config.include_dirs
    assert host / "pkg_inc/hcomm/ccu" in config.include_dirs
    assert config.library_paths == (host / "lib64/libasccomm_ccu.so", sdk / "lib64/libhcomm.so")
    assert (config.sdk_build_id, config.pto_isa_build_id) == runtime_build_identity()
    private_header = host / "pkg_inc/hcomm/ccu/ccu_primitives_impl.h"
    private_header.write_bytes(b"updated header")
    assert config.build_identity != get_ccu_toolchain().build_identity


def test_toolchain_fingerprints_symlinked_sdk_headers(toolchain_files):
    sdk, _, _ = toolchain_files
    host = sdk / "aarch64-linux"
    host.mkdir()
    for name in ("include", "pkg_inc", "lib64"):
        (sdk / name).rename(host / name)
        (sdk / name).symlink_to(host / name, target_is_directory=True)
    first = get_ccu_toolchain()
    new_header = host / "include/hcomm/ccu/additional.hpp"
    new_header.write_bytes(b"another SDK header")
    second = get_ccu_toolchain()
    assert second.build_identity != first.build_identity
    assert (str(sdk / "include/hcomm/ccu/additional.hpp"), hashlib.sha256(b"another SDK header").hexdigest()) in (
        second.dependency_digests
    )


def test_manifest_header_compiles_and_modules_round_trip(toolchain_files, tmp_path, monkeypatch):
    actual_compiler = next(
        (path for path in (Path("/usr/bin/g++"), Path("/usr/local/bin/g++")) if path.is_file()), None
    )
    if actual_compiler is None:
        pytest.skip("requires a native g++")
    _, project, _ = toolchain_files
    assets = tmp_path / "installed/simpler_setup/_assets"
    shutil.copytree(project, assets)
    monkeypatch.setattr(environment, "__file__", str(assets.parent / "environment.py"))
    monkeypatch.setattr(ccu_toolchain, "PROJECT_ROOT", environment._resolve_project_root())
    monkeypatch.chdir(tmp_path)
    toolchain = get_ccu_toolchain()
    assert toolchain.include_dirs[0] == assets / "src/common"
    (tmp_path / "ccu_manifest_identity.h").write_text(toolchain.manifest_identity_header())
    source = tmp_path / "module.cpp"
    source.write_text(
        "#include <array>\n"
        '#include "ccu_manifest_identity.h"\n'
        '#include "platform_comm/ccu/ccu_native_abi.h"\n'
        'extern "C" const uint8_t *module_id() { return simpler::ccu::sdk_build_id; }\n'
        'extern "C" int module_index() { return INDEX; }\n'
    )
    depfile = tmp_path / "module.d"
    compile_args = [str(actual_compiler), *toolchain.compile_flags, f"-I{toolchain.include_dirs[0]}"]
    subprocess.run([*compile_args, "-M", "-MF", str(depfile), "-MT", "module", str(source)], check=True)
    assert any(dependency.endswith("/array") for dependency in depfile.read_text().split())
    modules = []
    for index in range(2):
        module = tmp_path / f"module{index}.so"
        subprocess.run(
            [
                *compile_args,
                *toolchain.link_flags,
                f"-DINDEX={index}",
                str(source),
                "-o",
                str(module),
            ],
            check=True,
        )
        loaded = ctypes.CDLL(str(module))
        loaded.module_id.restype = ctypes.POINTER(ctypes.c_uint8 * 32)
        assert bytes(loaded.module_id().contents) == toolchain.sdk_build_id
        assert loaded.module_index() == index
        modules.append(module.read_bytes())

    chip = ChipCallable.build(
        signature=[], func_name="orchestration", binary=b"orchestration", children=[], ccu_modules=modules
    )
    raw = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    assert ChipCallable.from_bytes(raw).ccu_modules == modules
