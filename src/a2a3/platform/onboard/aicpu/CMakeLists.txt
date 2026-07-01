# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build AICPU kernel using aicpu/*.cpp files, common/* sources, and custom includes
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
# ASCEND_HOME_PATH is set when calling cmake
cmake_minimum_required(VERSION 3.16.3)

project(aicpu_kernel LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/include")
# Pick up spin_hint.h next to the cross-arch onboard/aicpu sources.
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/aicpu")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build AICPU kernel")
endif()

# Build complete source list from CUSTOM_SOURCE_DIRS and core sources.
# Sources come from three buckets:
#   1. `${CMAKE_CURRENT_SOURCE_DIR}/*.cpp` — onboard-aicpu sources that are
#      still arch-specific (a2a3-only files like kernel.cpp).
#   2. `common/platform/onboard/aicpu/*.cpp` — onboard-aicpu sources shared
#      by both arches (cache_ops, device_log, device_time, device_malloc,
#      orch_so_file, platform_aicpu_affinity).
#   3. arch-shared (between onboard and sim) sources under `../../shared/aicpu/`
#      and `common/platform/shared/aicpu/` (unified log, scope_stats, tensor_dump).
set(AICPU_KERNEL_SOURCES "")
file(GLOB AICPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND AICPU_KERNEL_SOURCES ${AICPU_SOURCES})
file(GLOB COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/aicpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/aicpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/onboard/aicpu/*.cpp"
)
list(APPEND AICPU_KERNEL_SOURCES ${COMMON_SOURCES})
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICPU_KERNEL_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build AICPU kernel")
endif()

list(LENGTH AICPU_KERNEL_SOURCES NUM_AICPU_KERNEL_SOURCES)
message(STATUS "AICpu kernel: ${NUM_AICPU_KERNEL_SOURCES} source files")
message(VERBOSE "AICpu kernel sources: ${AICPU_KERNEL_SOURCES}")
add_library(aicpu_kernel SHARED ${AICPU_KERNEL_SOURCES})

# TraCR
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tools/tracr.cmake)
tracr_enable(aicpu_kernel)

# Option to make the Orchestrator run independently (i.e. finalize before letting the schedulers run)
option(INDEP_ORCH "Run Orchestrator independent from the Schedulers" OFF)
if(DEFINED ENV{INDEP_ORCH})
    set(INDEP_ORCH $ENV{INDEP_ORCH})
endif()
if(INDEP_ORCH)
    target_compile_definitions(aicpu_kernel PRIVATE INDEP_ORCH)
endif()

option(WERROR "Treat compiler warnings as errors" ON)
if(DEFINED ENV{SIMPLER_DISABLE_WARNINGS_AS_ERRORS})
    set(WERROR OFF)
endif()

# Compile options (common to both C and C++)
target_compile_options(aicpu_kernel
    PRIVATE
        -Wall
        -Wextra
        -rdynamic
        -O3
        -fPIC
        $<$<BOOL:${WERROR}>:-Werror>
        -Wno-error=variadic-macros
        -fno-common
        -g
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
        $<$<COMPILE_LANGUAGE:C>:-std=gnu11>
)

target_include_directories(aicpu_kernel
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
    PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/toolchain
        ${ASCEND_HOME_PATH}/pkg_inc/base
)

target_link_directories(aicpu_kernel
    PRIVATE
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/
        ${ASCEND_HOME_PATH}/lib64
)

# Output name
set_target_properties(aicpu_kernel PROPERTIES OUTPUT_NAME aicpu_kernel)

# Build dispatcher SO — bootstrap-only upload helper. The dispatcher has NO
# runtime-specific code; libaicpu_extend_kernels loads it once via
# rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC), invokes
# DynTileFwkBackendKernelServerInit, which writes the bundled inner SO bytes
# (passed via the extended DeviceArgs at offsets 120/128) to
# /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so
# under a content-fingerprint + device-id basename. After bootstrap the host registers the
# preinstall file via rtsBinaryLoadFromFile + rtsFuncGetByName and launches
# per-task through rtsLaunchCpuKernel; the dispatcher SO itself is never
# referenced again.
#
# Output name is fixed ("simpler_aicpu_dispatcher"). See
# src/common/aicpu_loader/{device/aicpu_dispatcher.h,README.md} for the
# extended DeviceArgs layout and the FNV-1a/Build-ID fingerprint protocol.
set(AICPU_DISPATCHER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/aicpu_loader/device/aicpu_dispatcher.cpp"
)
add_library(aicpu_dispatcher SHARED ${AICPU_DISPATCHER_SOURCES})

target_compile_options(aicpu_dispatcher
    PRIVATE
        -Wall
        -Wextra
        -rdynamic
        -O3
        -fPIC
        -g
        $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
)

target_include_directories(aicpu_dispatcher
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
        # src/common is needed so `#include "utils/elf_build_id.h"` resolves;
        # host_runtime.so already has this on its include path (see host
        # CMakeLists), and the dispatcher uses the same header to fingerprint
        # the inner SO bytes by their ELF Build-ID rather than a 64-byte FNV
        # over the (mostly-shared) ELF header.
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
        ${ASCEND_HOME_PATH}/include
)

target_link_libraries(aicpu_dispatcher PRIVATE dl)

set_target_properties(aicpu_dispatcher PROPERTIES
    LINK_FLAGS "-Wl,--build-id"
    OUTPUT_NAME "simpler_aicpu_dispatcher"
)
