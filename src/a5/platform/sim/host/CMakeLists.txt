# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build Host runtime for a5sim (Simulation)
#
# This builds the simulated host runtime that uses threads instead of
# actual device execution.
#
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake

cmake_minimum_required(VERSION 3.16.3)

project(host_runtime_sim LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/worker")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime (simulation)")
endif()

# Build complete source list
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/host_log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/unified_log_host.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/l2_perf_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/pmu_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/tensor_dump_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../aicpu/platform_aicpu_affinity.cpp"
    # Shared POSIX-shm sim comm backend (same source as a2a3 sim).
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform_comm/comm_sim.cpp"
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
add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})

# C++ standard (applied only to C++ files)
set_target_properties(host_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compile options
target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
        $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>
)

# Include directories
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        # Shared platform_comm headers so the common comm_sim.cpp resolves
        # its #include "platform_comm/comm.h" / "platform_comm/comm_context.h".
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/sim_context
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Link against pthread (for std::thread)
target_link_libraries(host_runtime
    PRIVATE
        pthread
        dl
)

# POSIX shm_open / shm_unlink live in libSystem on macOS but require -lrt on Linux.
if(UNIX AND NOT APPLE)
    target_link_libraries(host_runtime PRIVATE rt)
endif()

# Allow undefined symbols from libcpu_sim_context.so (loaded with RTLD_GLOBAL at runtime).
# On macOS, the linker requires -undefined dynamic_lookup; on Linux/gcc this is the default.
if(APPLE)
    target_link_options(host_runtime PRIVATE -undefined dynamic_lookup)
endif()

set_target_properties(host_runtime PROPERTIES
    OUTPUT_NAME "host_runtime"
    # Force .so suffix on all platforms (macOS defaults to .dylib)
    # This matches the hardcoded paths in Python/C++ code
    SUFFIX ".so"
)
