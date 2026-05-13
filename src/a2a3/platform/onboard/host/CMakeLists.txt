# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build Host runtime using host/* + common/* + custom includes and sources (for `Runtime`)
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
# ASCEND_HOME_PATH is set when calling cmake
cmake_minimum_required(VERSION 3.16.3)

project(host_runtime LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/worker")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime")
endif()

set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE_DEFAULT OFF)
if(DEFINED ENV{SIMPLER_ENABLE_PTO_SDMA_WORKSPACE} AND "$ENV{SIMPLER_ENABLE_PTO_SDMA_WORKSPACE}" STREQUAL "1")
    set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE_DEFAULT ON)
endif()
option(
    SIMPLER_ENABLE_PTO_SDMA_WORKSPACE
    "Enable PTO-ISA SDMA workspace support for async SDMA completion"
    ${SIMPLER_ENABLE_PTO_SDMA_WORKSPACE_DEFAULT}
)
if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    if(NOT DEFINED ENV{PTO_ISA_ROOT})
        message(FATAL_ERROR "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE requires PTO_ISA_ROOT")
    endif()
    list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "$ENV{PTO_ISA_ROOT}/include")
endif()

# Build complete source list: core host sources + sources from CUSTOM_SOURCE_DIRS
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_regs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/l2_perf_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/tensor_dump_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/pmu_collector.cpp"
)
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND HOST_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(STATUS "CUSTOM_SOURCE_DIRS not set, using only core sources")
endif()

# Create shared library
list(LENGTH HOST_RUNTIME_SOURCES NUM_HOST_RUNTIME_SOURCES)
message(STATUS "Host runtime: ${NUM_HOST_RUNTIME_SOURCES} source files")
message(VERBOSE "Host runtime sources: ${HOST_RUNTIME_SOURCES}")
add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})

# C++ standard (applied only to C++ files)
set_target_properties(host_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compile options (matching AICPU pattern)
target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
)

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_compile_definitions(host_runtime PRIVATE SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1)
endif()

# Include directories - always include local headers
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        # Shared platform_comm headers (comm.h / comm_context.h) live in
        # src/common so a2a3 (HCCL) and a5/a2a3 sim (POSIX-shm) can use the
        # same contract.
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
    PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver
)

target_link_directories(host_runtime
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
)

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_link_directories(host_runtime PRIVATE ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
endif()

# CANN 9.x exposes the working non-V2 HCCL entry points through libhcomm.
# Link it explicitly so comm_hccl.cpp can follow the same initialization path
# as the pto-isa communication tests.
find_library(HCOMM_LIB NAMES hcomm PATHS ${ASCEND_HOME_PATH}/lib64 NO_DEFAULT_PATH)
if(HCOMM_LIB)
    set(HCCL_LINK_TARGETS hcomm)
    message(STATUS "Using HCCL library: hcomm")
else()
    message(FATAL_ERROR "libhcomm not found under ${ASCEND_HOME_PATH}/lib64")
endif()

# Link against CANN runtime libraries
# ascend_hal is dynamically loaded at runtime via dlopen in device_runner
# when performance profiling is enabled
target_link_libraries(host_runtime
    PRIVATE
        runtime
        ascendcl
        ${HCCL_LINK_TARGETS}
        dl
)

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_link_libraries(host_runtime PRIVATE nnopbase)
endif()

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")
