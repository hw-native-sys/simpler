# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build AICPU runtime for a2a3sim (Simulation)
#
# This builds the AICPU executor as a host-compatible shared library (.so)
# that can be dlopen'd by the simulation runtime.
#
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake

cmake_minimum_required(VERSION 3.16.3)

project(aicpu_sim LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/include")
# Pick up spin_hint.h next to the cross-arch sim/aicpu sources.
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/sim/aicpu")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build AICPU (simulation)")
endif()

# Build complete source list from CUSTOM_SOURCE_DIRS and platform sources.
# Sources come from three buckets:
#   1. `${CMAKE_CURRENT_SOURCE_DIR}/*.cpp` — sim-aicpu sources that are still
#      arch-specific (e.g. inner_platform_regs.cpp, which holds per-arch
#      register layouts).
#   2. `common/platform/sim/aicpu/*.cpp` — sim-aicpu sources shared by both
#      arches (cache_ops, device_log, device_time, device_malloc,
#      orch_so_file, platform_aicpu_affinity).
#   3. arch-shared sources under `../../shared/aicpu/` and
#      `common/platform/shared/aicpu/`.
set(AICPU_SOURCES "")
file(GLOB PLATFORM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND AICPU_SOURCES ${PLATFORM_SOURCES})
file(GLOB COMMON_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/aicpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/shared/aicpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform/sim/aicpu/*.cpp"
)
list(APPEND AICPU_SOURCES ${COMMON_SOURCES})

# Then, collect sources from CUSTOM_SOURCE_DIRS (runtime sources)
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND AICPU_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_SOURCE_DIRS to build AICPU (simulation)")
endif()

# Create shared library (host-compatible for dlopen)
add_library(aicpu_kernel SHARED ${AICPU_SOURCES})

# TraCR
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tools/tracr.cmake)
tracr_enable(aicpu_kernel)

# TODO: move this somewhere such that EVERY platform launches this once. Placing this here is hacky...
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../tools/tracr_postprocessing_script.cmake)

# Optional: to make the Orchestrator run independently (i.e. finalize before letting the schedulers run)
option(INDEP_ORCH "Run Orchestrator independent from the Schedulers" OFF)
if(DEFINED ENV{INDEP_ORCH})
    set(INDEP_ORCH $ENV{INDEP_ORCH})
endif()
if(INDEP_ORCH)
    target_compile_definitions(aicpu_kernel PRIVATE INDEP_ORCH)
endif()

# Compile options (host g++/gcc)
target_compile_options(aicpu_kernel
    PRIVATE
        -Wall
        -Wextra
        -Werror
        $<$<CXX_COMPILER_ID:GNU>:-Wno-error=class-memaccess>
        -fPIC
        -O3
        -g
        $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
        $<$<COMPILE_LANGUAGE:C>:-std=gnu11>
)

# Include directories
target_include_directories(aicpu_kernel
    PRIVATE
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
)

# Link against pthread
target_link_libraries(aicpu_kernel
    PRIVATE
        pthread
)

set_target_properties(aicpu_kernel PROPERTIES
    OUTPUT_NAME "aicpu_kernel"
    # Force .so suffix on all platforms (macOS defaults to .dylib)
    # This matches the hardcoded paths in Python/C++ code
    SUFFIX ".so"
)

# Apply compiler sanitizers to this host-compiled target. No-op unless
# SIMPLER_SANITIZERS is set at configure time, which runtime_compiler only
# passes for host toolchains (sim + onboard host). See cmake/sanitizers.cmake.
if(SIMPLER_SANITIZERS)
    include("${SIMPLER_CMAKE_DIR}/sanitizers.cmake")
    simpler_apply_sanitizers(aicpu_kernel)
endif()
