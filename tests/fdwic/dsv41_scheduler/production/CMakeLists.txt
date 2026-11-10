# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(dsv41_native_runner LANGUAGES CXX)
if(NOT SIMPLER_ROOT)
    get_filename_component(SIMPLER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
endif()
set(COMMON "${SIMPLER_ROOT}/src/common")
add_executable(native-runner
    runner.cpp
    ${COMMON}/worker/chip_worker.cpp
    ${COMMON}/worker/runtime_module.cpp
    ${COMMON}/worker/chip_run_lane.cpp
    ${COMMON}/worker/device_fault_monitor_host.cpp
    ${COMMON}/log/host_log.cpp
    ${COMMON}/log/unified_log_host.cpp
    ${COMMON}/task_interface/assert_compat.cpp
)
target_compile_features(native-runner PRIVATE cxx_std_17)
target_include_directories(native-runner PRIVATE
    ${COMMON}/task_interface
    ${COMMON}/worker
    ${COMMON}/hierarchical
    ${COMMON}/platform/include
    ${COMMON}/platform/include/common
    ${COMMON}/platform/include/host
    ${COMMON}/log/include
)
target_link_libraries(native-runner PRIVATE ${CMAKE_DL_LIBS} pthread rt)

add_executable(record-tool record_tool.cpp)
target_compile_features(record-tool PRIVATE cxx_std_17)
target_include_directories(record-tool PRIVATE ${COMMON} ${COMMON}/task_interface)

# These are Host orchestration libraries only. AICore/AICPU executors come
# from an existing runtime build; model kernels are compiled separately.
set(ORCH_INCLUDES
    ${SIMPLER_ROOT}/src/a5/runtime/host_build_graph/runtime
    ${SIMPLER_ROOT}/src/a5/runtime/host_build_graph/common
    ${SIMPLER_ROOT}/src/a5/runtime
    ${COMMON}/task_interface ${COMMON} ${COMMON}/host_build_graph
    ${COMMON}/host_build_graph/host/orchestration
    ${COMMON}/platform/include ${COMMON}/platform/include/common
    ${COMMON}/log/include
    ${SIMPLER_ROOT}/src/a5/platform/include
)
add_library(orch-support OBJECT
    ${COMMON}/host_build_graph/shared/assert_support.cpp
    ${COMMON}/log/host_log.cpp
    ${COMMON}/log/unified_log_host.cpp
)
target_compile_features(orch-support PRIVATE cxx_std_17)
target_include_directories(orch-support PRIVATE ${ORCH_INCLUDES})
set_target_properties(orch-support PROPERTIES POSITION_INDEPENDENT_CODE ON)
foreach(name capture replay)
    add_library(${name} SHARED ${name}.cpp $<TARGET_OBJECTS:orch-support>)
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_include_directories(${name} PRIVATE ${ORCH_INCLUDES})
    target_compile_options(${name} PRIVATE -fno-gnu-unique)
    target_link_options(${name} PRIVATE -Wl,--build-id)
    target_link_libraries(${name} PRIVATE ${CMAKE_DL_LIBS} pthread)
endforeach()
target_sources(replay PRIVATE ${COMMON}/host_build_graph/host/orchestration/orch_binding.cpp)
