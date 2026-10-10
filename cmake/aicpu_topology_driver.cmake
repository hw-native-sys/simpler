# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HAL declarations ship with CANN; newer DSMI topology declarations also ship
# in the separately installed driver package. Prefer that package when readable.
set(_SIMPLER_TOPOLOGY_DRIVER_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(simpler_configure_aicpu_topology_driver target)
    include("${_SIMPLER_TOPOLOGY_DRIVER_CMAKE_DIR}/ascend_driver_path.cmake")
    set(_dsmi_candidates "${ASCEND_DRIVER_PATH}/include")
    list(APPEND _dsmi_candidates
        "${ASCEND_HOME_PATH}/include/driver"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver"
        "${ASCEND_HOME_PATH}/pkg_inc/driver"
    )
    set(_dsmi_include_dir "")
    foreach(_candidate IN LISTS _dsmi_candidates)
        if(EXISTS "${_candidate}/dsmi_common_interface.h")
            set(_dsmi_include_dir "${_candidate}")
            break()
        endif()
    endforeach()
    if(NOT _dsmi_include_dir)
        message(FATAL_ERROR "No readable DSMI headers found in the driver or CANN package")
    endif()
    message(STATUS "AICPU topology DSMI header: ${_dsmi_include_dir}/dsmi_common_interface.h")

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${ASCEND_HOME_PATH}/include")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(_headers "#include <driver/ascend_hal_base.h>\n#pragma push_macro(\"DLLEXPORT\")\n#undef DLLEXPORT\n#include \"${_dsmi_include_dir}/dsmi_common_interface.h\"\n#pragma pop_macro(\"DLLEXPORT\")\n")
    # Recheck on configure: changing the installed packages must not reuse cached capabilities.
    foreach(_check IN ITEMS SIMPLER_DRIVER_HEADERS_COMPILE SIMPLER_HAS_HAL_CPU_TOPO
            SIMPLER_HAS_DSMI_CPU_TOPO_SUBCOMMAND SIMPLER_HAS_DSMI_CPU_TOPO_CAPACITY
            SIMPLER_HAS_DSMI_CPU_TOPOLOGY)
        unset(${_check})
        unset(${_check} CACHE)
    endforeach()
    check_cxx_source_compiles("${_headers}int main() { return 0; }" SIMPLER_DRIVER_HEADERS_COMPILE)
    if(NOT SIMPLER_DRIVER_HEADERS_COMPILE)
        message(FATAL_ERROR "Selected HAL/DSMI headers do not compile; see the CMake check log")
    endif()
    check_cxx_source_compiles("${_headers}int value = INFO_TYPE_CPU_TOPO;" SIMPLER_HAS_HAL_CPU_TOPO)
    check_cxx_source_compiles("${_headers}unsigned value = DSMI_SOC_INFO_SUB_CMD_CPU_TOPO;"
        SIMPLER_HAS_DSMI_CPU_TOPO_SUBCOMMAND)
    check_cxx_source_compiles("${_headers}unsigned value = DSMI_MAX_CPU_TOPO_NUM;"
        SIMPLER_HAS_DSMI_CPU_TOPO_CAPACITY)
    set(SIMPLER_CPU_TOPO_DSMI_CAPACITY 64)
    if(SIMPLER_HAS_DSMI_CPU_TOPO_CAPACITY)
        set(SIMPLER_CPU_TOPO_DSMI_CAPACITY DSMI_MAX_CPU_TOPO_NUM)
    endif()
    check_cxx_source_compiles("${_headers}
#include <cstdint>
#include <type_traits>
struct CpuData { uint64_t mask; uint8_t id, shared, physical, thread; };
int main() {
    static_assert(std::extent_v<decltype(dsmi_cpu_topology_info::single_cpu_topo_info)> == ${SIMPLER_CPU_TOPO_DSMI_CAPACITY});
    dsmi_cpu_topology_info topo{};
    uint32_t count{topo.total_nums};
    const dsmi_single_cpu_topology_info &cpu = topo.single_cpu_topo_info[0];
    CpuData data{cpu.cpu_mask, cpu.cpu_id, cpu.is_share, cpu.phy_cpu_id, cpu.hyperthread_id};
    return count + data.id;
}"
        SIMPLER_HAS_DSMI_CPU_TOPOLOGY)

    set(SIMPLER_CPU_TOPO_HAL_SELECTOR "59")
    if(SIMPLER_HAS_HAL_CPU_TOPO)
        set(SIMPLER_CPU_TOPO_HAL_SELECTOR "INFO_TYPE_CPU_TOPO")
    endif()
    set(SIMPLER_CPU_TOPO_DSMI_SUBCOMMAND "2")
    if(SIMPLER_HAS_DSMI_CPU_TOPO_SUBCOMMAND)
        set(SIMPLER_CPU_TOPO_DSMI_SUBCOMMAND "DSMI_SOC_INFO_SUB_CMD_CPU_TOPO")
    endif()
    configure_file(
        "${_SIMPLER_TOPOLOGY_DRIVER_CMAKE_DIR}/aicpu_topology_driver.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/aicpu_topology_driver.h"
        @ONLY
    )
    target_include_directories(${target} BEFORE PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()
