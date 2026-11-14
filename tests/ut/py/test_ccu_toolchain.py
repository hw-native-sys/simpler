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
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from simpler import env_manager
from simpler.ccu import runtime_build_identity
from simpler.task_interface import ChipCallable

from simpler_setup import environment, pto_isa, toolchain
from simpler_setup.kernel_compiler import KernelCompiler


@pytest.fixture
def sdk_files(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    project = tmp_path / "project"
    isa = tmp_path / "isa"
    compiler = tmp_path / "g++"
    files = (
        *((sdk / header, header.encode()) for header in toolchain._CCU_SDK_HEADERS),
        (
            project / "src/common" / toolchain._CCU_ABI_HEADER,
            (environment.PROJECT_ROOT / "src/common" / toolchain._CCU_ABI_HEADER).read_bytes(),
        ),
        (project / "pto_isa.pin", b"a" * 40),
        (isa / "include/pto/pto-inst.hpp", b"pinned ISA"),
        *(
            (sdk / path, b"compiler placeholder")
            for path in (
                "bin/ccec",
                "bin/ld.lld",
                "tools/hcc/bin/aarch64-target-linux-gnu-g++",
                "tools/hcc/bin/aarch64-target-linux-gnu-gcc",
            )
        ),
        (compiler, b"compiler driver"),
        (tmp_path / "ld", b"linker"),
    )
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    lib = sdk / "lib64"
    lib.mkdir()
    elf = b"\x7fELF\x02\x01" + bytes(10) + b"\x03\x00" + (183).to_bytes(2, "little")
    for name in toolchain._CCU_LIBRARIES:
        (lib / f"lib{name}.so").write_bytes(elf + name.encode())
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    monkeypatch.setitem(env_manager._cache, "ASCEND_HOME_PATH", str(sdk))
    monkeypatch.setattr(toolchain, "PROJECT_ROOT", project)
    monkeypatch.setattr(pto_isa, "PROJECT_ROOT", project)
    monkeypatch.setattr(pto_isa, "ensure_pto_isa_root", lambda: str(isa))

    return sdk, project, compiler


@pytest.fixture
def toolchain_files(sdk_files, tmp_path, monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "aarch64")

    def which(name):
        if name in ("g++", "g++-15", "ld"):
            return str(tmp_path / name)
        return None

    def query(command, *, stderr=False):
        if stderr:
            return ""
        if command[-1] == "-Wl,--version":
            return "GNU ld (GNU Binutils) 2.37"
        if command[-1] == "-Wl,--trace":
            Path(command[command.index("-o") + 1]).touch()
            return "\n".join(str(path) for path in (tmp_path / "ld", *tmp_path.glob("lib*")))
        if command[-1] == "-dumpmachine":
            return "aarch64-linux-gnu"
        if "-dM" in command:
            return "#define __aarch64__ 1\n#define __LP64__ 1\n#define __linux__ 1"
        kind, name = command[-1].split("=", 1)
        path = tmp_path / name
        return str(path) if kind == "-print-file-name" and path.is_file() else name

    monkeypatch.setattr(toolchain.shutil, "which", which)
    monkeypatch.setattr(toolchain, "_ccu_query", query)
    monkeypatch.setattr("simpler_setup.toolchain._is_gcc", lambda _: True)
    return sdk_files


def test_toolchain_config_and_manifest_header(toolchain_files):
    sdk, project, compiler = toolchain_files
    kernels = KernelCompiler("a5")
    config = kernels.ccu_compile_cache_token()
    assert kernels.ccu is not None
    assert kernels.ccu.cxx_path == kernels.host_gxx.cxx_path
    assert kernels.get_ccu_include_dirs() == list(config["include_dirs"])
    assert kernels.get_ccu_library_paths() == list(config["library_paths"])
    assert config["cxx_path"] == str(compiler)
    assert config["compile_flags"] == ("-fPIC", "-O3", "-g", "-std=c++17", "-fno-gnu-unique")
    assert config["link_flags"] == ("-shared",)
    assert config["include_dirs"] == (
        str(project / "src/common"),
        str(sdk / "include"),
        str(sdk / "include/hccl"),
        str(sdk / "pkg_inc/hcomm/ccu"),
        str(sdk / "include/hcomm/ccu"),
    )
    assert config["library_paths"] == (str(sdk / "lib64/libasccomm_ccu.so"), str(sdk / "lib64/libhcomm.so"))
    for name, value in zip(("sdk_build_id", "pto_isa_build_id"), runtime_build_identity(), strict=True):
        assert f"{name}[32] = {{{', '.join(map(str, value))}}};" in config["manifest_identity_header"]
    assert config == KernelCompiler("a5").ccu_compile_cache_token()


def test_ccu_uses_kernel_compiler_sanitizer_selection(toolchain_files, monkeypatch):
    _, _, executable = toolchain_files
    executable.with_name("g++-15").write_bytes(b"sanitizer compiler")
    monkeypatch.setattr(KernelCompiler, "_sanitizers", "undefined")
    compiler = KernelCompiler("a5")
    config = compiler.ccu_compile_cache_token()
    assert compiler.ccu is not None
    assert compiler.ccu.cxx_path == compiler.host_gxx.cxx_path == "g++-15"
    assert config["cxx_path"] == str(executable.with_name("g++-15"))
    assert "-fsanitize=undefined" in config["compile_flags"]
    assert "-fsanitize=undefined" in config["link_flags"]


def test_toolchain_cache_identity_changes(toolchain_files):
    sdk, project, compiler = toolchain_files
    baseline = KernelCompiler("a5").ccu_compile_cache_token()
    instrumented = KernelCompiler("a5").ccu_compile_cache_token(
        extra_compile_flags=("-O1", "-fsanitize=address"), extra_link_flags=("-fsanitize=address",)
    )
    assert instrumented["compile_flags"][-2:] == ("-O1", "-fsanitize=address")
    assert instrumented["link_flags"][-1] == "-fsanitize=address"
    assert instrumented != baseline
    assert KernelCompiler("a5").ccu_compile_cache_token(extra_compile_flags=("-O2", "-O1")) != KernelCompiler(
        "a5"
    ).ccu_compile_cache_token(extra_compile_flags=("-O1", "-O2"))

    nested = sdk / "include/hcomm/ccu/nested.h"
    nested.touch()
    previous = baseline
    for path in (
        compiler,
        compiler.with_name("ld"),
        sdk / "lib64/libhcomm.so",
        sdk / "pkg_inc/hcomm/ccu/ccu_primitives_impl.h",
        nested,
        project / "src/common" / toolchain._CCU_ABI_HEADER,
        project / "pto_isa.pin",
    ):
        path.write_bytes(b"b" * 40 if path.name == "pto_isa.pin" else path.read_bytes() + b"changed")
        identity = KernelCompiler("a5").ccu_compile_cache_token()
        assert identity != previous
        previous = identity


@pytest.mark.parametrize("missing", ["pkg_inc/hcomm/ccu/ccu_primitives_impl.h", "lib64/libasccomm_ccu.so"])
def test_toolchain_reports_missing_sdk_inputs(toolchain_files, missing):
    sdk, _, _ = toolchain_files
    (sdk / missing).unlink()
    with pytest.raises(FileNotFoundError, match="CANN CCU dependency") as error:
        KernelCompiler("a5").ccu_compile_cache_token()
    assert str(sdk / missing) in str(error.value)


def test_toolchain_reports_missing_compiler(toolchain_files, monkeypatch):
    monkeypatch.setattr(toolchain.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="g\\+\\+.*searched"):
        KernelCompiler("a5").ccu_compile_cache_token()


def test_toolchain_rejects_linker_without_complete_archive_trace(toolchain_files, monkeypatch):
    original_query = toolchain._ccu_query

    def query(command, *, stderr=False):
        if command[-1] == "-Wl,--version":
            return "LLD 19.1.0 (compatible with GNU linkers)"
        return original_query(command, stderr=stderr)

    monkeypatch.setattr(toolchain, "_ccu_query", query)
    with pytest.raises(ValueError, match="GNU ld.*runtime-library fingerprint"):
        KernelCompiler("a5").ccu_compile_cache_token()


def test_toolchain_rejects_compiler_target_mismatch(toolchain_files, monkeypatch):
    monkeypatch.setattr(toolchain, "_ccu_query", lambda _: "x86_64-linux-gnu")
    with pytest.raises(ValueError, match="expected aarch64 Linux"):
        KernelCompiler("a5").ccu_compile_cache_token()


def test_toolchain_validates_target_options_in_link_flags(toolchain_files, monkeypatch):
    original_query = toolchain._ccu_query

    def query(command, *, stderr=False):
        if "-dM" in command and "-mabi=ilp32" in command:
            return "#define __aarch64__ 1\n#define __linux__ 1"
        return original_query(command, stderr=stderr)

    monkeypatch.setattr(toolchain, "_ccu_query", query)
    with pytest.raises(ValueError, match="flags must select 64-bit aarch64 Linux"):
        KernelCompiler("a5").ccu_compile_cache_token(extra_link_flags=("-mabi=ilp32",))


def test_toolchain_returns_the_sdk_library_it_fingerprints(toolchain_files, tmp_path):
    sdk, _, _ = toolchain_files
    library = sdk / "lib64/libhcomm.so"
    shadow = tmp_path / "other/libhcomm.so"
    shadow.parent.mkdir()
    shadow.write_bytes(library.read_bytes() + b"other")

    config = KernelCompiler("a5").ccu_compile_cache_token(extra_compile_flags=(f"-L{shadow.parent}",))
    assert config["library_paths"][-1] == str(library)
    assert (str(config["library_paths"][-1]), hashlib.sha256(library.read_bytes()).hexdigest()) in (
        config["dependency_digests"]
    )


@pytest.mark.parametrize("flag", ["-march=native", "-mcpu=native+crc", "-mtune=native"])
def test_toolchain_rejects_cpu_dependent_flags(toolchain_files, flag):
    with pytest.raises(ValueError, match="explicit CPU target"):
        KernelCompiler("a5").ccu_compile_cache_token(extra_compile_flags=(flag,))


@pytest.mark.parametrize(
    ("flags", "archive"),
    [
        (("-static-libstdc++",), "libstdc++.a"),
        (("-static-libgcc",), "libgcc_eh.a"),
        (("-fsanitize=address", "-static-libasan"), "libasan.a"),
        (("-fsanitize=signed-integer-overflow", "-static-libubsan"), "libubsan.a"),
    ],
)
def test_static_runtime_update_invalidates_build_identity(toolchain_files, flags, archive):
    _, _, compiler = toolchain_files
    path = compiler.with_name(archive)
    path.write_bytes(b"archive")
    original = KernelCompiler("a5").ccu_compile_cache_token(extra_link_flags=flags)
    path.write_bytes(b"updated archive")
    assert original != KernelCompiler("a5").ccu_compile_cache_token(extra_link_flags=flags)


def test_toolchain_rejects_foreign_sdk_library(toolchain_files):
    sdk, _, _ = toolchain_files
    library = sdk / "lib64/libhcomm.so"
    library.write_bytes(library.read_bytes()[:18] + (62).to_bytes(2, "little"))
    with pytest.raises(ValueError, match="libhcomm.so.*aarch64 Linux ELF"):
        KernelCompiler("a5").ccu_compile_cache_token()


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

    config = KernelCompiler("a5").ccu_compile_cache_token()
    assert str(sdk / "include") in config["include_dirs"]
    assert str(host / "include/hccl") in config["include_dirs"]
    assert str(host / "pkg_inc/hcomm/ccu") in config["include_dirs"]
    assert config["library_paths"] == (str(host / "lib64/libasccomm_ccu.so"), str(sdk / "lib64/libhcomm.so"))
    private_header = host / "pkg_inc/hcomm/ccu/ccu_primitives_impl.h"
    private_header.write_bytes(b"updated header")
    assert config != KernelCompiler("a5").ccu_compile_cache_token()


def test_toolchain_fingerprints_symlinked_sdk_headers(toolchain_files):
    sdk, _, _ = toolchain_files
    host = sdk / "aarch64-linux"
    host.mkdir()
    for name in ("include", "pkg_inc", "lib64"):
        (sdk / name).rename(host / name)
        (sdk / name).symlink_to(host / name, target_is_directory=True)
    first = KernelCompiler("a5").ccu_compile_cache_token()
    new_header = host / "include/hcomm/ccu/additional.hpp"
    new_header.write_bytes(b"another SDK header")
    second = KernelCompiler("a5").ccu_compile_cache_token()
    assert second != first
    assert (str(sdk / "include/hcomm/ccu/additional.hpp"), hashlib.sha256(b"another SDK header").hexdigest()) in (
        second["dependency_digests"]
    )


@pytest.fixture
def native_toolchain_files(sdk_files, tmp_path):
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "x86_64"):
        pytest.skip("native CCU modules require aarch64 or x86_64 Linux")
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a native g++")
    assert compiler is not None
    sdk, _, _ = sdk_files
    source = tmp_path / "empty.cpp"
    source.write_text('extern "C" void sdk_placeholder() {}\n')
    for name in ("asccomm_ccu", "hcomm"):
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(sdk / f"lib64/lib{name}.so")], check=True)
    return sdk_files


def test_compile_ccu_concurrent_modules_round_trip(native_toolchain_files, tmp_path, monkeypatch):
    _, project, _ = native_toolchain_files
    assets = tmp_path / "installed/simpler_setup/_assets"
    shutil.copytree(project, assets)
    monkeypatch.setattr(environment, "__file__", str(assets.parent / "environment.py"))
    monkeypatch.setattr(toolchain, "PROJECT_ROOT", environment._resolve_project_root())
    monkeypatch.chdir(tmp_path)
    compiler = KernelCompiler("a5")
    assert compiler.get_ccu_include_dirs()[0] == str(assets / "src/common")
    includes = tmp_path / "user includes"
    includes.mkdir()
    (includes / "helper.h").write_text("int helper();\n")
    source = tmp_path / "module.cpp"
    source.write_text(
        "#include <array>\n"
        '#include "ccu_manifest_identity.h"\n'
        '#include "platform_comm/ccu/ccu_native_abi.h"\n'
        '#include "helper.h"\n'
        'extern "C" const uint8_t *module_sdk_id() { return simpler::ccu::sdk_build_id; }\n'
        'extern "C" const uint8_t *module_isa_id() { return simpler::ccu::pto_isa_build_id; }\n'
        'extern "C" int module_index() { return helper() + INDEX; }\n'
    )
    helper = tmp_path / "helper.cpp"
    helper.write_text("int helper() { return 10; }\n")
    build = tmp_path / "build"
    build.mkdir()

    def compile_module(index):
        return compiler.compile_ccu(
            "module.cpp",
            extra_sources=["helper.cpp"],
            extra_include_dirs=["user includes"],
            extra_compile_flags=[f"-DINDEX={index}"],
            extra_link_flags=["-Wl,-z,defs"],
            build_dir="build",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        modules = list(executor.map(compile_module, range(2)))
    assert list(build.iterdir()) == []
    sdk_id, isa_id = runtime_build_identity()
    for index, binary in enumerate(modules):
        assert binary.startswith(b"\x7fELF")
        module = tmp_path / f"module{index}.so"
        module.write_bytes(binary)
        loaded = ctypes.CDLL(str(module))
        for export, expected in (("module_sdk_id", sdk_id), ("module_isa_id", isa_id)):
            function = getattr(loaded, export)
            function.restype = ctypes.POINTER(ctypes.c_uint8 * 32)
            assert bytes(function().contents) == expected
        assert loaded.module_index() == 10 + index

    chip = ChipCallable.build(
        signature=[], func_name="orchestration", binary=b"orchestration", children=[], ccu_modules=modules
    )
    raw = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    assert ChipCallable.from_bytes(raw).ccu_modules == modules

    config = compiler.ccu_compile_cache_token(extra_include_dirs=["user includes"])
    assert config["extra_include_dirs"] == (str(includes),)
    (build / "ccu_manifest_identity.h").write_text(config["manifest_identity_header"])
    depfile = build / "module.d"
    subprocess.run(
        [
            config["cxx_path"],
            *config["compile_flags"],
            *(f"-I{path}" for path in (*config["include_dirs"], *config["extra_include_dirs"], build)),
            "-M",
            "-MF",
            str(depfile),
            "-MT",
            "module",
            str(source),
        ],
        check=True,
    )
    assert any(dependency.endswith("/array") for dependency in depfile.read_text().split())


@pytest.mark.parametrize("search_flag", ["driver", "split", "linker", "response"])
def test_selected_runtime_archive_invalidates_cache(native_toolchain_files, tmp_path, search_flag):
    runtime = tmp_path / "runtime with spaces"
    runtime.mkdir()
    archive = runtime / "libstdc++.a"
    runtime_source = runtime / "runtime.cpp"
    runtime_object = runtime / "runtime.o"
    source = tmp_path / "module.cpp"
    source.write_text(
        'extern "C" int runtime_value();\nextern "C" int selected_runtime() { return runtime_value(); }\n'
    )
    flags = ["-static-libstdc++"]
    if search_flag == "driver":
        flags += [f"-L{runtime}"]
    elif search_flag == "split":
        flags += ["-L", str(runtime)]
    elif search_flag == "linker":
        flags += [f"-Wl,-L,{runtime}"]
    else:
        response = tmp_path / "link.rsp"
        response.write_text(f'-L"{runtime}"\n')
        flags += [f"@{response}"]
    compiler = KernelCompiler("a5")
    tokens, values = [], []
    for version in (11, 22):
        runtime_source.write_text(f'extern "C" int runtime_value() {{ return {version}; }}\n')
        subprocess.run(["g++", "-fPIC", "-c", str(runtime_source), "-o", str(runtime_object)], check=True)
        subprocess.run(["ar", "rcs", str(archive), str(runtime_object)], check=True)
        tokens.append(compiler.ccu_compile_cache_token(extra_link_flags=flags))
        artifact = tmp_path / f"module{version}.so"
        artifact.write_bytes(compiler.compile_ccu(str(source), extra_link_flags=flags))
        values.append(ctypes.CDLL(str(artifact)).selected_runtime())
    assert values == [11, 22]
    assert tokens[0] != tokens[1]
    assert str(archive) in dict(tokens[0]["dependency_digests"])
    assert tokens[1] == compiler.ccu_compile_cache_token(extra_link_flags=flags)


@pytest.mark.parametrize("use_extra_include_dir", [False, True])
def test_compile_ccu_relative_flags_use_caller_directory(
    native_toolchain_files, tmp_path, monkeypatch, use_extra_include_dir
):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config" if use_extra_include_dir else tmp_path
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.h").write_text(
        '#include "ccu_manifest_identity.h"\n'
        '#include "platform_comm/ccu/ccu_native_abi.h"\n'
        "#define CONFIGURED_VALUE 37\n"
    )
    include_dirs = ["config"] if use_extra_include_dir else []
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "module.cpp").write_text('extern "C" int configured_value() { return CONFIGURED_VALUE; }\n')
    compiler = KernelCompiler("a5")
    flags = ("-include", "config.h")
    config = compiler.ccu_compile_cache_token(extra_compile_flags=flags, extra_include_dirs=include_dirs)
    assert config["compile_flags"][-2:] == flags
    binary = compiler.compile_ccu("sources/module.cpp", extra_compile_flags=flags, extra_include_dirs=include_dirs)
    module = tmp_path / "configured.so"
    module.write_bytes(binary)
    assert ctypes.CDLL(str(module)).configured_value() == 37


def test_compile_ccu_failure_cleans_private_build_dir(native_toolchain_files, tmp_path):
    source = tmp_path / "broken.cpp"
    source.write_text("#error invalid CCU module\n")
    build = tmp_path / "build"
    build.mkdir()
    with pytest.raises(RuntimeError, match="invalid CCU module"):
        KernelCompiler("a5").compile_ccu(str(source), build_dir=str(build))
    assert list(build.iterdir()) == []


def test_compile_ccu_reports_missing_extra_source(toolchain_files, tmp_path):
    source = tmp_path / "module.cpp"
    source.touch()
    with pytest.raises(FileNotFoundError, match="CCU source file not found.*missing.cpp"):
        KernelCompiler("a5").compile_ccu(str(source), extra_sources=[str(tmp_path / "missing.cpp")])


@pytest.mark.parametrize("platform", ["a2a3", "a2a3sim", "a5sim", "wsesim"])
def test_ccu_rejects_unsupported_platform(toolchain_files, platform):
    compiler = KernelCompiler(platform)
    assert compiler.ccu is None
    with pytest.raises(ValueError, match="Native CCU modules require platform 'a5'"):
        compiler.get_ccu_include_dirs()
    with pytest.raises(ValueError, match="Native CCU modules require platform 'a5'"):
        compiler.ccu_compile_cache_token()
    with pytest.raises(ValueError, match="Native CCU modules require platform 'a5'"):
        compiler.compile_ccu("unused.cpp")


def test_incore_configuration_does_not_require_ccu_sdk(toolchain_files):
    sdk, _, _ = toolchain_files
    (sdk / "include/hcomm/ccu/ccu_res.h").unlink()
    compiler = KernelCompiler("a5")
    assert compiler.ccec is not None
    assert "--cce-aicore-arch=dav-c310-vec" in compiler.ccec.get_compile_flags(core_type="aiv")
    assert compiler.get_incore_include_dirs()
    with pytest.raises(FileNotFoundError, match="ccu_res.h"):
        compiler.get_ccu_include_dirs()
