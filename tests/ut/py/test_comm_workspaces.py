# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Domain configuration and production A5 host resource ownership (no device)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from simpler import CcuConfig, RdmaConfig, SdmaConfig, UrmaConfig
from simpler.buffer import AddressSpace
from simpler.comm_domain import DomainConfig

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


def test_ccu_workspace_mapping_uses_same_configuration_as_name_sequence():
    named = DomainConfig.capture([0, 1], 4096, [], True, (), ("ccu",))
    mapped = DomainConfig.capture([0, 1], 4096, [], True, (), {"ccu": CcuConfig()})
    assert named == mapped
    assert mapped.workspace_mask() == 8


@pytest.mark.parametrize(
    "engines,mask", [((), 0), ([], 0), (["sdma"], 1), (["urma"], 2), (["urma", "sdma"], 3), (["rdma"], 4), (["ccu"], 8)]
)
def test_domain_workspace_selection_is_part_of_configuration(engines, mask):
    config = DomainConfig.capture([0, 1], 4096, [], True, (), engines)
    assert config.workspace_mask() == mask
    if engines:
        reordered = DomainConfig.capture([0, 1], 4096, [], True, (), list(reversed(engines)))
        assert not config.difference(reordered)
    synchronous = DomainConfig.capture([0, 1], 4096, [], True, ())
    assert synchronous.workspace_engines == () and synchronous.workspace_mask() == 0
    if engines:
        assert config.difference(synchronous).startswith("workspace_engines:")
    else:
        assert not config.difference(synchronous)


@pytest.mark.parametrize("engines", [["unknown"], ["sdma", "sdma"], ["rdma", "sdma"], ["rdma", "urma"]])
def test_invalid_workspace_requests_are_rejected_before_allocation(engines):
    with pytest.raises(ValueError):
        DomainConfig.capture([0], 4096, [], False, (), engines)


@pytest.mark.parametrize("engines", ["sdma", None])
def test_workspace_names_require_a_sequence(engines):
    with pytest.raises(TypeError, match="sequence"):
        DomainConfig.capture([0], 4096, [], False, (), engines)


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
                "ccu_domain_backend_hcomm.cpp",
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
            str(ROOT / "tests/ut/py/support/comm_workspaces_host.cpp"),
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


def test_workspace_domains_keep_resources_isolated_and_rollback_failures(host_resource_binary):
    subprocess.run([str(host_resource_binary)], check=True, timeout=20)


def test_basic_window_issues_no_explicit_access(host_resource_binary):
    subprocess.run([str(host_resource_binary), "basic_window"], check=True, timeout=20)


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
        "host_workspace",
        "host_context",
        "host_collision",
        "host_invalid",
        "host_unregister",
        "host_unregister_abort",
        "host_unregister_destroy",
    ],
)
def test_host_domain_backing_and_native_failure_ownership(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.parametrize("scenario", ["sdma_destroy", "urma_release", "urma_destroy", "destroy_exception"])
def test_workspace_cleanup_failure_retains_owner(host_resource_binary, scenario):
    subprocess.run([str(host_resource_binary), scenario], check=True, timeout=20)


@pytest.mark.parametrize("host_resource_binary", [True], indirect=True)
def test_rdma_workspace_free_failure_retains_owner(host_resource_binary):
    subprocess.run([str(host_resource_binary), "rdma_release"], check=True, timeout=20)


@pytest.mark.parametrize("kind", ["CommWorkspaceKind::COUNT", "static_cast<CommWorkspaceKind>(UINT32_MAX)"])
def test_invalid_static_workspace_kind_is_rejected_by_compiler(tmp_path, kind):
    source = tmp_path / "invalid_workspace.cpp"
    source.write_text(
        '#include "platform_comm/comm_context.h"\n'
        f"int main() {{ CommContext ctx{{}}; return comm_workspace<{kind}>(ctx).addr; }}\n"
    )
    result = subprocess.run(
        ["c++", "-std=c++17", f"-I{ROOT / 'src/common'}", "-c", str(source), "-o", str(tmp_path / "invalid.o")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid communication workspace kind" in result.stderr


@pytest.mark.parametrize("ambient", [None, "ON", "OFF"])
def test_a5_urma_has_no_build_selection(ambient, monkeypatch):
    from simpler_setup.runtime_builder import _a5_async_workspace_overlay_defines  # noqa: PLC0415

    if ambient is None:
        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", ambient)
    assert "SIMPLER_ENABLE_PTO_URMA_WORKSPACE" not in _a5_async_workspace_overlay_defines()


@pytest.mark.parametrize(
    "platform,runtime,message",
    [
        ("a2a3sim", "tensormap_and_ringbuffer", "requires A5"),
        ("a5sim", "tensormap_and_ringbuffer", "requires A5"),
        ("a5", "host_build_graph", "tensormap_and_ringbuffer"),
    ],
)
def test_unsupported_urma_domains_are_rejected_before_control(platform, runtime, message):
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform=platform, runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match=message):
        worker._domain_config([0, 1], 4096, [], False, ("urma",))


@pytest.mark.parametrize("members", [(1, 2), (2, 1)])
def test_device_urma_configuration_preserves_ordered_members(members):
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[1, 2, 3], num_sub_workers=0)
    config = worker._domain_config(members, 4096, [], False, ("sdma", "urma"))
    assert config.members == members
    assert config.workspace_engines == ("sdma", "urma")


def test_host_urma_configuration_keeps_dense_prefix_requirement():
    from simpler.task_interface import AddressSpace  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match="dense prefix"):
        worker._domain_config([1, 0], 4096, [], False, ("urma",), AddressSpace.HOST)


def test_single_rank_urma_configuration_is_delegated_to_provider():
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0], num_sub_workers=0)
    config = worker._domain_config([0], 4096, [], False, ("urma",))
    assert config.members == (0,)
    assert config.workspace_engines == ("urma",)


@pytest.mark.parametrize(
    "names,configs",
    [
        (("sdma",), {"sdma": SdmaConfig()}),
        (("urma",), {"urma": UrmaConfig()}),
        (("rdma",), {"rdma": RdmaConfig()}),
        (("urma", "sdma"), {"sdma": SdmaConfig(), "urma": UrmaConfig()}),
    ],
)
def test_default_engine_configurations_equal_legacy_names(names, configs):
    legacy = DomainConfig.capture([0, 1], 4096, [], True, (), names)
    configured = DomainConfig.capture([0, 1], 4096, [], True, (), configs)
    assert configured == legacy
    assert configured.difference(legacy) == ""


@pytest.mark.parametrize("memory_source", [AddressSpace.DEVICE, AddressSpace.HOST])
def test_engine_configuration_is_snapshotted_and_changes_reuse_identity(memory_source):
    engines = {"urma": UrmaConfig(jetties_per_aiv=2, sq_depth=512)}
    saved = DomainConfig.capture([0, 1], 4096, [], True, (), engines, memory_source)
    engines["urma"] = UrmaConfig()
    replacement = DomainConfig.capture([0, 1], 4096, [], True, (), engines, memory_source)
    assert saved.urma_config() == UrmaConfig(jetties_per_aiv=2, sq_depth=512)
    assert saved.difference(replacement).startswith("workspace_configs:")
    with pytest.raises(AttributeError):
        saved.urma_config().sq_depth = 1024


@pytest.mark.parametrize(
    "engines",
    [
        {"urma": SdmaConfig()},
        {"sdma": UrmaConfig()},
        {"rdma": None},
        {"urma": {"jetties_per_aiv": 2}},
    ],
)
def test_engine_configuration_type_must_match_name(engines):
    with pytest.raises(TypeError, match="requires"):
        DomainConfig.capture([0, 1], 4096, [], True, (), engines)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jetties_per_aiv": 0},
        {"jetties_per_aiv": 9},
        {"jetties_per_aiv": -1},
        {"sq_depth": -1},
        {"sq_depth": 3},
        {"sq_depth": 2**31},
    ],
)
def test_invalid_urma_queue_sizes_fail_before_domain_allocation(kwargs):
    with pytest.raises(ValueError):
        UrmaConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"jetties_per_aiv": True}, {"jetties_per_aiv": 2.5}, {"sq_depth": "512"}])
def test_urma_queue_sizes_require_integers(kwargs):
    with pytest.raises(TypeError):
        UrmaConfig(**kwargs)


def test_engine_configs_are_available_when_native_bindings_are_missing():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['_task_interface'] = None; "
            "from simpler import UrmaConfig, SdmaConfig, RdmaConfig; "
            "assert UrmaConfig().jetties_per_aiv == 1; SdmaConfig(); RdmaConfig(); "
            "assert sys.modules['_task_interface'] is None",
        ],
        check=True,
    )


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


@pytest.mark.parametrize("failure", ["channels_only", "destroy_failure"])
def test_legacy_ccu_channel_rollback_retains_unconfirmed_resources(ccu_failure_binary, failure):
    binary, module, env = ccu_failure_binary
    subprocess.run([str(binary), str(module), failure], env=env, check=True, timeout=20)
