# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise A5 communication workspace ownership and rollback with host-side CANN mocks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize("rdma", [False, True])
def test_a5_host_options_select_rdma(tmp_path, rdma):
    source = (ROOT / "src/a5/platform/onboard/host/CMakeLists.txt").read_text()
    selection = source[: source.index("# Pin-resolved checkout path")]
    (tmp_path / "CMakeLists.txt").write_text(
        selection + '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected" "${SIMPLER_ENABLE_PTO_RDMA_WORKSPACE}")\n'
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
            f"-DSIMPLER_CMAKE_DIR={ROOT / 'cmake'}",
            f"-DCUSTOM_INCLUDE_DIRS={ROOT / 'src/common'}",
            f"-DSIMPLER_ENABLE_PTO_RDMA_WORKSPACE={'ON' if rdma else 'OFF'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "build/selected").read_text() == ("ON" if rdma else "OFF")


@pytest.fixture(scope="module", params=[False, True])
def host_resource_binary(request, tmp_path_factory):
    return build_host_resource_binary(tmp_path_factory.mktemp("comm_workspace_host"), request.param)


def _ccu_sdk_environment(cann):
    # Host fault injection needs SDK symbols, not an installed device driver.
    if not (cann / "devlib/libascend_hal.so").is_file():
        pytest.skip("HCOMM failure injection requires the CANN devlib driver stub")
    env = os.environ.copy()
    paths = [str(cann / "lib64"), str(cann / "devlib")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return env


def build_host_resource_binary(build, rdma, *, ccu_backend=False):
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/hccl/hccl_comm.h").exists():
        pytest.skip("production host ownership fixture requires CANN headers, no device")
    for engine in ("sdma", "urma", "rdma"):
        header = build / f"pto/comm/async/{engine}/{engine}_workspace_manager.hpp"
        header.parent.mkdir(parents=True)
        header.write_text('#include "comm_workspace_managers.h"\n')
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    defines = []
    backend_sources = []
    backend_libraries = []
    backend_env = None
    if ccu_backend:
        if not (cann / "include/hccl/hccl_ccu_res.h").exists():
            pytest.skip("HCOMM failure injection requires the CCU SDK, no device")
        backend_env = _ccu_sdk_environment(cann)
        defines += [
            "-DSIMPLER_TEST_CCU_BACKEND=1",
            "-Wl,--export-dynamic-symbol=simpler_ccu_test_manifest_variant",
            f"-I{cann}/include/hccl",
            f"-I{cann}/include/hcomm/ccu",
            f"-I{cann}/pkg_inc/hcomm/ccu",
            f"-I{cann}/pkg_inc",
        ]
        host = ROOT / "src/a5/platform/onboard/host"
        common = ROOT / "src/common/platform_comm/ccu"
        backend_sources = [
            str(host / name)
            for name in (
                "ccu_domain_registration_hcomm.cpp",
                "ccu_run_lifecycle_hcomm.cpp",
            )
        ] + [str(common / name) for name in ("ccu_native_module.cpp", "ccu_domain_registration.cpp")]
        backend_libraries = [
            f"-L{cann}/lib64",
            f"-Wl,-rpath,{cann}/lib64",
            f"-Wl,-rpath-link,{cann}/lib64:{cann}/devlib",
            "-lasccomm_ccu",
            "-lhcomm",
            "-lascendcl",
            "-lruntime",
            "-lc_sec",
        ]
    if rdma:
        defines.append("-DSIMPLER_ENABLE_PTO_RDMA_WORKSPACE=1")
        defines.append("-DPTO_COMM_WORKSPACE_RDMA_SUPPORTED=1")
        defines.append("-Wl,--export-dynamic-symbol=GetRoceIpFromXml")
    isa = Path(ensure_pto_isa_root())
    binary = build / "host_resources"
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *defines,
            f"-I{build}",
            f"-I{isa}/include",
            f"-I{cann}/include",
            f"-I{cann}/pkg_inc/base",
            f"-I{ROOT}/src/a5/platform/include",
            f"-I{ROOT}/src/common/platform/include",
            f"-I{ROOT}/src/common",
            f"-I{ROOT}/src/common/task_interface",
            f"-I{ROOT}/src/a5/platform/onboard/host",
            f"-I{ROOT}/tests/ut/py/support",
            str(Path(__file__).with_name("comm_workspaces_host.cpp")),
            str(ROOT / "src/common/platform_comm/ccu/ccu_domain_owner.cpp"),
            str(ROOT / "src/common/platform_comm/ccu/ccu_run_lifecycle.cpp"),
            *backend_sources,
            *backend_libraries,
            "-ldl",
            "-lrt",
            "-o",
            str(binary),
        ],
        env=backend_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Host resource fixture compilation failed:\n{result.stdout}\n{result.stderr}", pytrace=False)
    return binary


def test_host_fixture_reports_compiler_diagnostics(tmp_path, monkeypatch):
    compiler_dir = tmp_path / "bin"
    compiler_dir.mkdir()
    compiler = compiler_dir / "c++"
    compiler.write_text("#!/bin/sh\nprintf 'fixture compiler diagnostic\\n' >&2\nexit 7\n")
    compiler.chmod(0o755)
    header = tmp_path / "include/hccl/hccl_comm.h"
    header.parent.mkdir(parents=True)
    header.touch()
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setenv("PATH", str(compiler_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(sys.modules[__name__], "ensure_pto_isa_root", lambda: str(tmp_path))
    with pytest.raises(pytest.fail.Exception, match="fixture compiler diagnostic"):
        build_host_resource_binary(build, False)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
def test_workspace_domains_keep_resources_isolated_and_rollback_failures(host_resource_binary):
    subprocess.run([str(host_resource_binary)], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
def test_basic_window_issues_no_explicit_access(host_resource_binary):
    subprocess.run([str(host_resource_binary), "basic_window"], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize(
    "scenario",
    [
        "barrier_complete",
        "barrier_same_context",
        "barrier_context_error",
        "barrier_restore_error",
        "barrier_submit_error",
        "barrier_sync_error",
    ],
)
def test_barrier_restores_caller_context_and_confirms_stream_completion(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize(
    "scenario",
    [
        "host_local",
        "host_urma",
        "host_urma_unsupported",
        "host_peer_prepare_failed",
        "host_ready_publish_failed",
        "host_aiv_preflight_failed",
        "host_urma_config",
        "host_coexist",
        "host_register",
        "host_pointer",
        "host_peer_register",
        "host_peer_pointer",
        "host_workspace",
        "host_context",
        "host_collision",
        "host_invalid",
        "host_unregister",
        "host_local_unregister",
        "host_unregister_abort",
        "host_unregister_destroy",
    ],
)
def test_host_domain_backing_and_native_failure_ownership(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize(
    "scenario",
    [
        "host_mte",
        "host_mte_missing",
        "host_mte_rank",
        "host_mte_size",
        "host_mte_register",
        "host_mte_pointer",
        "host_mte_context",
        "host_mte_throw",
        "host_mte_unregister",
    ],
)
def test_host_mte_peer_windows_and_failure_ownership(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("scenario", ["sdma_destroy", "urma_release", "urma_destroy", "destroy_exception"])
def test_workspace_cleanup_failure_retains_owner(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("host_resource_binary", [True], indirect=True)
def test_rdma_workspace_free_failure_retains_owner(host_resource_binary):
    subprocess.run([str(host_resource_binary), "rdma_release"], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("host_resource_binary", [True], indirect=True)
@pytest.mark.parametrize("engine", ["sdma", "urma", "rdma"])
@pytest.mark.parametrize("failure", ["return", "throw"])
def test_init_rollback_free_failure_retains_owner(host_resource_binary, engine, failure):
    subprocess.run([str(host_resource_binary), f"init_{engine}_{failure}"], check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("host_resource_binary", [True], indirect=True)
@pytest.mark.parametrize("scenario", ["init_rdma_dbcos_fail", "init_urma_alloc_fail"])
def test_init_partial_failure_retains_owner(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.parametrize("ambient", [None, "ON", "OFF"])
def test_a5_urma_has_no_build_selection(ambient, monkeypatch):
    from simpler_setup.runtime_builder import _a5_async_workspace_overlay_defines  # noqa: PLC0415

    if ambient is None:
        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", ambient)
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE" not in _a5_async_workspace_overlay_defines()


@pytest.fixture(scope="module")
def ccu_failure_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("comm_ccu_failure")
    binary = build_host_resource_binary(build, False, ccu_backend=True)
    module = build / "native_module.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{ROOT}/src/common",
            str(ROOT / "tests/ut/cpp/common/platform/support/ccu_native_test_module.cpp"),
            "-o",
            str(module),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    return binary, module, _ccu_sdk_environment(cann)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize("failure", ["channels_only", "destroy_failure"])
def test_legacy_ccu_channel_rollback_retains_unconfirmed_resources(ccu_failure_binary, failure):
    binary, module, env = ccu_failure_binary
    subprocess.run([str(binary), str(module), failure], env=env, check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
def test_domain_and_registration_communicators_use_domain_mode(ccu_failure_binary):
    binary, module, env = ccu_failure_binary
    subprocess.run([str(binary), str(module), "modes"], env=env, check=True, timeout=20)


@pytest.mark.requires_toolkit
@pytest.mark.platforms(["a5"])
@pytest.mark.parametrize(
    "failure",
    [
        "retry_pending",
        "retry_abort_error",
        "retry_query_error",
        "retry_unknown",
        "retry_pool_error",
        "retry_pool_pending",
    ],
)
def test_ccu_abort_retries_only_pending_sdk_queries(ccu_failure_binary, failure):
    binary, module, env = ccu_failure_binary
    subprocess.run([str(binary), str(module), failure], env=env, check=True, timeout=20)
