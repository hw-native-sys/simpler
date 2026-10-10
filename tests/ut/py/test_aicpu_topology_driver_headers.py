# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile topology consumers against old and public driver declarations without CANN."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_headers(root: Path, kind: str | None) -> None:
    toolkit = root / "cann/include/driver"
    driver = root / "driver/include"
    toolkit.mkdir(parents=True, exist_ok=True)
    driver.mkdir(parents=True, exist_ok=True)
    dsmi_api = (
        "enum { DSMI_MAIN_CMD_SOC_INFO = 14 };\n"
        'extern "C" int dsmi_get_device_info(unsigned, unsigned, unsigned, void *, unsigned *);\n'
    )
    hal_api = (
        'enum { MODULE_TYPE_SYSTEM = 0 };\nextern "C" int halGetDeviceInfoByBuff(unsigned, int, int, void *, int *);\n'
    )
    (toolkit / "dsmi_common_interface.h").write_text("#pragma once\n" + dsmi_api)
    hal = toolkit / "ascend_hal_base.h"
    dsmi = driver / "dsmi_common_interface.h"
    if kind is None:
        hal.write_text("#pragma once\n" + hal_api)
        dsmi.unlink(missing_ok=True)
        return
    if kind == "enum":
        hal.write_text("#pragma once\nenum { INFO_TYPE_CPU_TOPO = 73 };\n" + hal_api)
        selector = "enum { DSMI_SOC_INFO_SUB_CMD_CPU_TOPO = 11 };"
    else:
        hal.write_text("#pragma once\n#define INFO_TYPE_CPU_TOPO 73\n" + hal_api)
        selector = "#define DSMI_SOC_INFO_SUB_CMD_CPU_TOPO 11"
    dsmi.write_text(
        "#pragma once\n"
        + dsmi_api
        + selector
        + "\n#define DSMI_MAX_CPU_TOPO_NUM 12\n"
        + "struct dsmi_single_cpu_topology_info { unsigned long long cpu_mask;\n"
        + "  unsigned char cpu_id, is_share, phy_cpu_id, hyperthread_id; };\n"
        + "struct dsmi_cpu_topology_info {\n"
        + "  unsigned int total_nums;\n"
        + "  dsmi_single_cpu_topology_info single_cpu_topo_info[DSMI_MAX_CPU_TOPO_NUM];\n"
        + "};\n"
    )


def _build_consumer(
    root: Path, *, public: bool, native: bool | None = None, capacity_override: int | None = None
) -> None:
    if not shutil.which("cmake"):
        pytest.skip("cmake is required for the driver header compile test")
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(topology_header_test LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        f'include("{REPO_ROOT.as_posix()}/cmake/aicpu_topology_driver.cmake")\n'
        "add_executable(consumer consumer.cpp unrelated.cpp)\n"
        f'add_library(probe OBJECT "{REPO_ROOT}/src/a5/platform/onboard/host/aicpu_topology_probe.cpp")\n'
        'target_include_directories(probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated" '
        '"${ASCEND_HOME_PATH}/include")\n'
        f'target_include_directories(probe PRIVATE "{REPO_ROOT}/src/common/platform/include" '
        f'"{REPO_ROOT}/src/common/log/include")\n'
        "simpler_configure_aicpu_topology_driver(consumer)\n"
        'target_include_directories(consumer PRIVATE "${ASCEND_HOME_PATH}/include")\n'
    )
    (root / "driver/include/ascend_hal_error.h").write_text('#error "driver include leaked"\n')
    (root / "cann/include/ascend_hal_error.h").write_text("#pragma once\n")
    (root / "unrelated.cpp").write_text("#include <ascend_hal_error.h>\n")
    if native is None:
        native = public
    hal, dsmi, capacity = (73, 11, 12) if public else (59, 2, 64)
    if capacity_override is not None:
        capacity = capacity_override
    assertions = (
        "static_assert(std::is_same_v<pto::driver::CpuTopology, dsmi_cpu_topology_info>);\n"
        "static_assert(std::is_same_v<pto::driver::SingleCpuTopology, dsmi_single_cpu_topology_info>);\n"
        if native
        else "static_assert(sizeof(pto::driver::SingleCpuTopology) == 16);\n"
        f"static_assert(sizeof(pto::driver::CpuTopology) == {8 + capacity * 16});\n"
    )
    (root / "consumer.cpp").write_text(
        '#include "aicpu_topology_driver.h"\n'
        f"static_assert(pto::driver::kCpuTopoHalInfoType == {hal});\n"
        f"static_assert(pto::driver::kCpuTopoDsmiSubcommand == {dsmi});\n"
        f"static_assert(pto::driver::kCpuTopoCapacity == {capacity});\n" + assertions + "int main() {}\n"
    )
    for command in [
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(root / "build"),
            f"-DASCEND_HOME_PATH={root / 'cann'}",
            f"-DASCEND_DRIVER_PATH={root / 'driver'}",
        ],
        ["cmake", "--build", str(root / "build")],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["macro", "enum"])
def test_driver_declarations_override_legacy_toolkit(tmp_path: Path, kind: str) -> None:
    _write_headers(tmp_path, kind)
    _build_consumer(tmp_path, public=True)


def test_reconfigure_refreshes_driver_capabilities(tmp_path: Path) -> None:
    for kind in (None, "enum", None):
        _write_headers(tmp_path, kind)
        _build_consumer(tmp_path, public=kind is not None)


@pytest.mark.parametrize(
    "member", ["cpu_mask", "cpu_id", "is_share", "phy_cpu_id", "hyperthread_id", "single_cpu_topo_info", "total_nums"]
)
def test_incomplete_native_topology_uses_legacy(tmp_path: Path, member: str) -> None:
    _write_headers(tmp_path, "enum")
    header = tmp_path / "driver/include/dsmi_common_interface.h"
    header.write_text(header.read_text().replace(member, "missing_" + member))
    _build_consumer(tmp_path, public=True, native=False)


@pytest.mark.parametrize("member", ["cpu_mask", "cpu_id", "is_share", "phy_cpu_id", "hyperthread_id"])
def test_incompatible_native_member_uses_legacy(tmp_path: Path, member: str) -> None:
    _write_headers(tmp_path, "enum")
    header = tmp_path / "driver/include/dsmi_common_interface.h"
    header.write_text(re.sub(rf"\b{member}\b", "*" + member, header.read_text()))
    _build_consumer(tmp_path, public=True, native=False)


def test_native_capacity_mismatch_uses_legacy(tmp_path: Path) -> None:
    _write_headers(tmp_path, "enum")
    header = tmp_path / "driver/include/dsmi_common_interface.h"
    header.write_text(header.read_text().replace("[DSMI_MAX_CPU_TOPO_NUM]", "[3]"))
    _build_consumer(tmp_path, public=True, native=False)


@pytest.mark.parametrize("mode", ["explicit", "sibling", "standard"])
def test_shared_driver_root_resolution(tmp_path: Path, mode: str) -> None:
    driver = tmp_path / ("custom-driver" if mode == "explicit" else "driver")
    if mode == "standard":
        driver = Path("/usr/local/Ascend/driver")
    else:
        driver.mkdir()
    script = tmp_path / "resolve.cmake"
    script.write_text(
        f'set(ASCEND_HOME_PATH "{tmp_path}/cann")\n'
        + (f'set(ASCEND_DRIVER_PATH "{driver}")\n' if mode == "explicit" else "")
        + f'include("{REPO_ROOT}/cmake/ascend_driver_path.cmake")\n'
        + f'if(NOT ASCEND_DRIVER_PATH STREQUAL "{driver}")\n'
        + 'message(FATAL_ERROR "wrong driver root")\nendif()\n'
    )
    subprocess.run(["cmake", "-P", str(script)], check=True, capture_output=True, text=True)


def test_oversized_driver_topology_falls_back(tmp_path: Path) -> None:
    _write_headers(tmp_path, "enum")
    header = tmp_path / "driver/include/dsmi_common_interface.h"
    header.write_text(header.read_text().replace("DSMI_MAX_CPU_TOPO_NUM 12", "DSMI_MAX_CPU_TOPO_NUM 128"))
    _build_consumer(tmp_path, public=True, capacity_override=128)
    host = REPO_ROOT / "src/a5/platform/onboard/host"
    log = REPO_ROOT / "src/common/log"
    with (tmp_path / "CMakeLists.txt").open("a") as cmake:
        cmake.write(
            "set_target_properties(consumer PROPERTIES ENABLE_EXPORTS ON)\n"
            f'target_sources(consumer PRIVATE $<TARGET_OBJECTS:probe> "{host}/aicpu_affinity_select.cpp" '
            f'"{log}/host_log.cpp" "{log}/unified_log_host.cpp")\n'
            f'target_include_directories(consumer PRIVATE "{host}" "{log}" "{log}/include" '
            f'"{REPO_ROOT}/src/a5/platform/include" "{REPO_ROOT}/src/common/platform/include")\n'
            "target_link_libraries(consumer PRIVATE ${CMAKE_DL_LIBS} pthread)\n"
        )
    (tmp_path / "consumer.cpp").write_text(
        '#include "aicpu_topology_driver.h"\n'
        '#include "aicpu_topology_probe.h"\n'
        'extern "C" int halGetDeviceInfoByBuff(unsigned, int, int, void *buffer, int *) {\n'
        "  static_cast<pto::driver::CpuTopology *>(buffer)->total_nums = 65;\n"
        "  return 0;\n}\n"
        "int main() {\n"
        "  pto::a5::AicpuDeviceOccupancy occupancy{};\n"
        "  occupancy.occupy = 2; occupancy.occupy_valid = true;\n"
        "  pto::a5::AicpuTopology topology;\n"
        "  if (!pto::a5::probe_aicpu_topology(0, occupancy, topology)) return 1;\n"
        "  return topology.source != pto::a5::AicpuTopologySource::kOccupyFallback ||\n"
        "         topology.os_schedulable_cpus.size() != 1 ||\n"
        "         topology.os_schedulable_cpus[0].cpu_id != 1;\n}\n"
    )
    for command in [
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        ["cmake", "--build", str(tmp_path / "build"), "-j", "2"],
        [str(tmp_path / "build/consumer")],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
