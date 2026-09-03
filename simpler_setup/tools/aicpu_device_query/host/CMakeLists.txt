# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(query_device_hal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

add_executable(query_device_hal query_device_hal.cpp rtt_probe_host.cpp)

set(SIMPLER_ROOT "" CACHE PATH "Simpler source or installed _assets root")
if(NOT SIMPLER_ROOT)
    get_filename_component(SIMPLER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
endif()
if(NOT EXISTS "${SIMPLER_ROOT}/src/a5/platform/onboard/host/aicpu_topology_probe.cpp")
    message(FATAL_ERROR "SIMPLER_ROOT=${SIMPLER_ROOT} does not contain the required packaged/source src tree")
endif()
set(A5_TOPOLOGY_DIR "${SIMPLER_ROOT}/src/a5/platform/onboard/host")
target_sources(query_device_hal PRIVATE
    "${A5_TOPOLOGY_DIR}/aicpu_topology_probe.cpp"
    "${SIMPLER_ROOT}/src/common/log/host_log.cpp"
    "${SIMPLER_ROOT}/src/common/log/unified_log_host.cpp"
)
add_custom_command(TARGET query_device_hal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${A5_TOPOLOGY_DIR}/aicpu_cpu_topo_fallback.json"
        "$<TARGET_FILE_DIR:query_device_hal>/aicpu_cpu_topo_fallback.json"
)

# runtime/rt.h transitively needs profiling.
target_include_directories(query_device_hal PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime/runtime
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling
    ${A5_TOPOLOGY_DIR}
    ${SIMPLER_ROOT}/src/a5/platform/include
    ${SIMPLER_ROOT}/src/common/platform/include
    ${SIMPLER_ROOT}/src/common/log/include
    ${SIMPLER_ROOT}/src/common/log
)

set(ASCEND_DRIVER_LIB_DIR "/usr/local/Ascend/driver/lib64/driver"
    CACHE PATH "Directory containing libascend_hal.so")
target_link_directories(query_device_hal PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64
    ${ASCEND_DRIVER_LIB_DIR}
)

# ascendcl for aclrt*, runtime for rt*/rts* (rtAicpuKernelLaunchExWithArgs,
# rtsBinaryLoadFromFile, rtsFuncGetByName, rtsLaunchCpuKernel).
target_link_libraries(query_device_hal PRIVATE ascendcl runtime ascend_hal ${CMAKE_DL_LIBS} pthread)
