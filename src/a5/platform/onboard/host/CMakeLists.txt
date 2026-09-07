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

include("${SIMPLER_CMAKE_DIR}/host_log_sources.cmake")

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/host")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/worker")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime")
endif()
set(ASCEND_ARCH_HOME "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")

option(SIMPLER_ENABLE_PTO_URMA_WORKSPACE "Enable a5 PTO URMA workspace overlay" OFF)
# SDMA is the default a5 async-workspace backend. URMA remains an explicit
# alternative because CommContext exposes one workSpace/workSpaceSize pair.
# SIMPLER_ENABLE_PTO_SDMA_WORKSPACE is an internal compile marker, not a
# user-configurable option.
if(SIMPLER_ENABLE_PTO_URMA_WORKSPACE)
    set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE OFF)
else()
    set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE ON)
endif()

# Pin-resolved checkout path is passed by RuntimeBuilder as -DPTO_ISA_ROOT=
# (#1403). Do not read $ENV{PTO_ISA_ROOT} — ambient exports are not the pin.
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR
        "a5 onboard host_runtime PTO async workspace requires "
        "-DPTO_ISA_ROOT=<pin-resolved path> (needs pto-isa host headers)")
endif()
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${PTO_ISA_ROOT}/include")

# Build complete source list: core host sources + sources from CUSTOM_SOURCE_DIRS
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES ${SIMPLER_HOST_LOG_SOURCES})
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_topology_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiling_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_regs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/chip_swimlane_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/host_phase_records.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/clock_correlation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/host/pmu_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/dep_gen_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/scope_stats_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/args_dump_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp"
)
# Add common/aicpu_loader/host sources (LoadAicpuOp)
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/aicpu_loader/host/load_aicpu_op.cpp"
)
# Add common/platform/onboard/host sources (shared between a2a3 / a5 onboard).
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_phase_capture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/clock_correlation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/c_api_shared.cpp"
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
include("${SIMPLER_CMAKE_DIR}/profiling_config.cmake")
simpler_configure_profiling(host_runtime)
add_custom_command(TARGET host_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_cpu_topo_fallback.json"
        "$<TARGET_FILE_DIR:host_runtime>/aicpu_cpu_topo_fallback.json"
)

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

# Platform name baked into the shared get_platform() impl in
# src/common/platform/shared/host/platform_compile_info.cpp.
if(NOT DEFINED SIMPLER_RUNTIME_NAME OR SIMPLER_RUNTIME_NAME STREQUAL "")
    message(FATAL_ERROR "host_runtime requires -DSIMPLER_RUNTIME_NAME=<runtime>")
endif()
target_compile_definitions(host_runtime PRIVATE
    SIMPLER_PLATFORM_NAME="a5"
    SIMPLER_RUNTIME_NAME="${SIMPLER_RUNTIME_NAME}"
)

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_compile_definitions(host_runtime PRIVATE SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1)
endif()
if(SIMPLER_ENABLE_PTO_URMA_WORKSPACE)
    target_compile_definitions(host_runtime PRIVATE SIMPLER_ENABLE_PTO_URMA_WORKSPACE=1)
endif()

# Bake the resolved pto-isa commit into the compile command because the
# selected async-workspace backend embeds pto-isa headers into this .so. A
# pto-isa pin bump leaves runtime HEAD
# — and often header mtimes — untouched, so without this define ccache
# (compiler_check=mtime) can serve a stale object. Mirror of the a2a3 block
# (#1351).
set(SIMPLER_PTO_ISA_BUILD_COMMIT "" CACHE STRING
    "PTO-ISA commit used for host_runtime cache key")
if(NOT "${SIMPLER_PTO_ISA_BUILD_COMMIT}" STREQUAL "")
    target_compile_definitions(host_runtime PRIVATE
        SIMPLER_PTO_ISA_BUILD_COMMIT="${SIMPLER_PTO_ISA_BUILD_COMMIT}")
endif()

# Include directories - always include local headers
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        # Shared platform_comm headers (comm.h / comm_context.h) live in
        # src/common so a5 (HCCL) and a5/a2a3 sim (POSIX-shm) can use the
        # same contract.
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
    PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/runtime
        # pkg_inc/runtime/runtime exposes rts_kernel.h + kernel.h (CANN 7.0+
        # rtsLaunchCpuKernel API used by LoadAicpuOp).
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_HOME_PATH}/pkg_inc/profiling
        ${ASCEND_ARCH_HOME}/include
        ${ASCEND_ARCH_HOME}/pkg_inc
        ${ASCEND_ARCH_HOME}/pkg_inc/runtime
        ${ASCEND_ARCH_HOME}/pkg_inc/runtime/runtime
        ${ASCEND_ARCH_HOME}/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
        ${ASCEND_ARCH_HOME}/asc/include
        ${ASCEND_ARCH_HOME}/include/driver
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver
)

# Prefer libhccl when the CANN package ships it. Newer CANN packages may only
# ship libhcomm, matching pto-isa's own A5 comm tests, so fall back to hcomm.
find_library(HCCL_LIB
    NAMES hccl
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_ARCH_HOME}/lib64
    NO_DEFAULT_PATH
)
find_library(HCOMM_LIB
    NAMES hcomm
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_ARCH_HOME}/lib64
    NO_DEFAULT_PATH
)
if(HCCL_LIB)
    if(HCOMM_LIB)
        set(HCCL_LINK_TARGETS ${HCCL_LIB} ${HCOMM_LIB})
        message(STATUS "Using HCCL libraries: ${HCCL_LIB};${HCOMM_LIB}")
    else()
        set(HCCL_LINK_TARGETS ${HCCL_LIB})
        message(STATUS "Using HCCL library: ${HCCL_LIB}")
    endif()
elseif(HCOMM_LIB)
    set(HCCL_LINK_TARGETS ${HCOMM_LIB})
    message(WARNING "libhccl not found; using HCCL library: ${HCOMM_LIB}")
else()
    message(FATAL_ERROR
        "Neither libhccl nor libhcomm found under ${ASCEND_HOME_PATH}/lib64 or ${ASCEND_ARCH_HOME}/lib64")
endif()

# Link against CANN runtime libraries
# ascend_hal is dynamically loaded at runtime via dlopen in device_runner
# when performance profiling is enabled
# HcclGetRootInfo/HcclCommInitRootInfo are declared weak in CANN's hccl_comm.h.
# Under a toolchain that defaults to --as-needed (Ubuntu/conda gcc on x86), a
# weak undefined ref does not mark libhcomm "needed", so its DT_NEEDED is
# dropped and the symbols resolve to NULL at runtime -> SIGSEGV on the first
# HcclGetRootInfo call. Force the dependency so it survives any default. See #1018.
target_link_libraries(host_runtime
    PRIVATE
        runtime
        ascendcl
        -Wl,--no-as-needed ${HCCL_LINK_TARGETS} -Wl,--as-needed
        c_sec
        nnopbase
        dl
        pthread
)

target_link_directories(host_runtime
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        ${ASCEND_ARCH_HOME}/lib64
        ${ASCEND_ARCH_HOME}/runtime/lib64
)

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")

# Apply compiler sanitizers to this host-compiled target. No-op unless
# SIMPLER_SANITIZERS is set at configure time, which runtime_compiler only
# passes for host toolchains (sim + onboard host). See cmake/sanitizers.cmake.
if(SIMPLER_SANITIZERS)
    include("${SIMPLER_CMAKE_DIR}/sanitizers.cmake")
    simpler_apply_sanitizers(host_runtime)
endif()
