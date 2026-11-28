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

# The SDMA provider is compiled into every a2a3 onboard host runtime.
# Its workspace manager requires PTO-ISA headers and CANN 9.0+ aclnn symbols.
# Pin-resolved checkout path is passed by RuntimeBuilder as -DPTO_ISA_ROOT=
# (#1403). Do not read $ENV{PTO_ISA_ROOT} — ambient exports are not the pin.
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR
        "a2a3 onboard host_runtime requires -DPTO_ISA_ROOT=<pin-resolved path> "
        "(needs pto-isa headers + CANN 9.0+)")
endif()
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${PTO_ISA_ROOT}/include")
option(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE "Enable PTO RDMA domain workspaces" OFF)
if(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE)
    find_path(PTO_HCCP_INCLUDE_DIR hccp.h
        HINTS ENV PTO_HCCP_INCLUDE_DIR
        PATHS ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/pkg_inc/hccp)
    find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
    find_path(HCCL_INCLUDE_DIR hccl/hccl.h PATHS ${ASCEND_HOME_PATH}/include)
    if(NOT PTO_HCCP_INCLUDE_DIR OR NOT IBVERBS_INCLUDE_DIR OR NOT HCCL_INCLUDE_DIR)
        message(FATAL_ERROR "A3 RDMA requires matching HCCP headers (PTO_HCCP_INCLUDE_DIR) and libibverbs development headers")
    endif()
    list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS
        "${PTO_ISA_ROOT}/pkg_inc" "${PTO_ISA_ROOT}/kernels/include"
        "${PTO_HCCP_INCLUDE_DIR}" "${IBVERBS_INCLUDE_DIR}" "${HCCL_INCLUDE_DIR}"
        "${ASCEND_HOME_PATH}/pkg_inc/aicpu" "${ASCEND_HOME_PATH}/pkg_inc/driver"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/driver")
endif()

# Build complete source list: core host sources + sources from CUSTOM_SOURCE_DIRS
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES ${SIMPLER_HOST_LOG_SOURCES})
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_topology_probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_affinity_select.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/context_preparation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiling_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_regs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/chip_swimlane_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/host_phase_records.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/kernel_execution_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/run_resource_growth.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/args_dump_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/args_dump_retained_runs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/scope_stats_retained_runs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_ddr_mapping.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/host/pmu_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/dep_gen_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/dep_gen_retained_runs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/host_graph_exporter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/scope_stats_collector.cpp"
)
# Add common/aicpu_loader/host sources (LoadAicpuOp)
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/aicpu_loader/host/load_aicpu_op.cpp"
)
# Add common/platform/onboard/host sources (shared between a2a3 / a5 onboard).
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/kernel_platform_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_phase_capture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/host/device_fault_monitor_binding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/aicore_pmu_lease.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/device_runner_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/host/run_retention_probe.cpp"
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
if(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE)
    list(REMOVE_ITEM HOST_RUNTIME_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp")
endif()
add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})
include("${SIMPLER_CMAKE_DIR}/profiling_config.cmake")
simpler_configure_profiling(host_runtime)

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
    SIMPLER_PLATFORM_NAME="a2a3"
    SIMPLER_RUNTIME_NAME="${SIMPLER_RUNTIME_NAME}"
    # Host-only CCEC builds omit __NPU_ARCH__; PTO needs the A2/A3 host contract.
    PTO_NPU_ARCH_A2A3
    PTO_COMM_WORKSPACE_SDMA_SUPPORTED=1
    PTO_COMM_WORKSPACE_URMA_SUPPORTED=0
)
if(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE)
    target_compile_definitions(host_runtime PRIVATE
        SIMPLER_ENABLE_PTO_RDMA_WORKSPACE PTO_COMM_WORKSPACE_RDMA_SUPPORTED=1
        PTO_RDMA_SUPPORTED PTO_RDMA_BACKEND_IN_DIE_SUPPORTED)
else()
    target_compile_definitions(host_runtime PRIVATE PTO_COMM_WORKSPACE_RDMA_SUPPORTED=0)
endif()

set(SIMPLER_PTO_ISA_BUILD_COMMIT "" CACHE STRING
    "PTO-ISA commit used for host_runtime cache key")

# Bake the resolved pto-isa commit into the compile command. pto-isa headers
# (e.g. kSdmaMaxChan in sdma_workspace_manager.hpp) are compiled into this .so,
# but a pto-isa update leaves the runtime repo HEAD — and the pto-isa header
# mtimes — untouched, so a plain reinstall serves a stale object from ccache.
# This define perturbs the ccache key so a pto-isa bump forces a real recompile.
# It is intentionally unused in code (cache-bust only). See issue #1139.
# RuntimeBuilder passes the current pto_isa.pin for every a2a3 onboard build.
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
        # src/common so a2a3 (HCCL) and a5/a2a3 sim (POSIX-shm) can use the
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
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver
)

target_link_directories(host_runtime PRIVATE ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)

# CANN 9.x exposes the working non-V2 HCCL entry points through libhcomm.
# Link it explicitly so comm_hccl.cpp can follow the same initialization path
# as the pto-isa communication tests.
find_library(
    HCOMM_LIB
    NAMES hcomm
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    NO_DEFAULT_PATH
)
if(HCOMM_LIB)
    set(HCCL_LINK_TARGETS ${HCOMM_LIB})
    message(STATUS "Using HCCL library: ${HCOMM_LIB}")
else()
    message(FATAL_ERROR "libhcomm not found under ${ASCEND_HOME_PATH}/lib64 or ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
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
        dl
        pthread
        rt
)

target_link_libraries(host_runtime PRIVATE nnopbase)

target_link_directories(host_runtime
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
)

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")

if(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE)
    find_program(SIMPLER_CCEC_COMPILER ccec HINTS
        "${ASCEND_HOME_PATH}/bin" "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/bin")
    if(NOT SIMPLER_CCEC_COMPILER)
        message(FATAL_ERROR "A3 PTO domain headers require CANN ccec --cce-host-only")
    endif()
    # PTO's domain header carries device-qualified tensor declarations; this object is host-only.
    set(_rdma_comm_object "${CMAKE_CURRENT_BINARY_DIR}/comm_hccl_rdma.o")
    add_custom_command(OUTPUT "${_rdma_comm_object}"
        COMMAND "${SIMPLER_CCEC_COMPILER}" -xcce --cce-host-only -std=c++17 -fPIC
            "$<TARGET_PROPERTY:host_runtime,COMPILE_OPTIONS>"
            "-I$<JOIN:$<TARGET_PROPERTY:host_runtime,INCLUDE_DIRECTORIES>,;-I>"
            "-D$<JOIN:$<TARGET_PROPERTY:host_runtime,COMPILE_DEFINITIONS>,;-D>"
            -MMD -MF "${_rdma_comm_object}.d"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp" -o "${_rdma_comm_object}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/comm_hccl.cpp"
        DEPFILE "${_rdma_comm_object}.d"
        COMMAND_EXPAND_LISTS VERBATIM)
    set_source_files_properties("${_rdma_comm_object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    target_sources(host_runtime PRIVATE "${_rdma_comm_object}")
endif()

# Apply compiler sanitizers to this host-compiled target. No-op unless
# SIMPLER_SANITIZERS is set at configure time, which runtime_compiler only
# passes for host toolchains (sim + onboard host). See cmake/sanitizers.cmake.
if(SIMPLER_SANITIZERS)
    include("${SIMPLER_CMAKE_DIR}/sanitizers.cmake")
    simpler_apply_sanitizers(host_runtime)
endif()
