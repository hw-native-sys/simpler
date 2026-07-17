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
if(NOT DEFINED ENV{PTO_ISA_ROOT})
    message(FATAL_ERROR
        "a5 onboard host_runtime requires PTO_ISA_ROOT "
        "(URMA workspace setup uses pto-isa host headers)")
endif()
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "$ENV{PTO_ISA_ROOT}/include")

set(ASCEND_ARCH_HOME "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")

# SIMPLER_ENABLE_PTO_SDMA_WORKSPACE: opt-in overlay that wires the PTO-ISA
# async-SDMA workspace into comm_alloc_windows. When ON it pulls in pto-isa
# headers and CANN-9.0+ aclnn symbols (aclnnShmemSdmaStarsQuery*) and links
# libnnopbase, so it requires a CANN drop that actually exposes working a5
# SDMA primitives. Default OFF because the available a5 CANN drops break
# (AICPU exception -> 507018, or HCCL_E_INTERNAL). See
# docs/a5-sdma-overlay.md for the full isolation status and re-enable steps
# (#1315).
# Flip ON via -DSIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON (or the env var of the
# same name, forwarded by simpler_setup/runtime_builder.py).
option(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE "Enable a5 PTO SDMA workspace overlay" OFF)
if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    if(NOT DEFINED ENV{PTO_ISA_ROOT})
        message(FATAL_ERROR
            "a5 onboard host_runtime with SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON requires PTO_ISA_ROOT "
            "(needs pto-isa headers + CANN 9.0+ with a5 SDMA primitives)")
    endif()
    list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "$ENV{PTO_ISA_ROOT}/include")
endif()

# Build complete source list: core host sources + sources from CUSTOM_SOURCE_DIRS
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_topology_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiling_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_regs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/l2_swimlane_collector.cpp"
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
# See .docs/ONBOARD_HOST_COMMON_REFACTOR.md for the migration plan.
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/c_api_shared.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/l3_l2_orch_comm_service.cpp"
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

# Platform name baked into the shared get_platform() impl in
# src/common/platform/shared/host/platform_compile_info.cpp.
target_compile_definitions(host_runtime PRIVATE SIMPLER_PLATFORM_NAME="a5")

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_compile_definitions(host_runtime PRIVATE SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1)
endif()

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
if(HCCL_LIB AND HCOMM_LIB)
    set(HCCL_LINK_TARGETS ${HCCL_LIB} ${HCOMM_LIB})
    message(STATUS "Using HCCL libraries: ${HCCL_LIB};${HCOMM_LIB}")
elseif(HCOMM_LIB)
    set(HCCL_LINK_TARGETS ${HCOMM_LIB})
    message(WARNING "libhccl not found; using HCCL library: ${HCOMM_LIB}")
else()
    message(FATAL_ERROR
        "libhcomm not found under ${ASCEND_HOME_PATH}/lib64 or ${ASCEND_ARCH_HOME}/lib64")
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
)

target_link_directories(host_runtime
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        ${ASCEND_ARCH_HOME}/lib64
        ${ASCEND_ARCH_HOME}/runtime/lib64
)

if(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE)
    target_link_directories(host_runtime PRIVATE ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
    target_link_libraries(host_runtime PRIVATE nnopbase)
endif()

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")

# Apply compiler sanitizers to this host-compiled target. No-op unless
# SIMPLER_SANITIZERS is set at configure time, which runtime_compiler only
# passes for host toolchains (sim + onboard host). See cmake/sanitizers.cmake.
if(SIMPLER_SANITIZERS)
    include("${SIMPLER_CMAKE_DIR}/sanitizers.cmake")
    simpler_apply_sanitizers(host_runtime)
endif()
