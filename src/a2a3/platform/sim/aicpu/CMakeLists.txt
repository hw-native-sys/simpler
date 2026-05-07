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
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")

if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build AICPU (simulation)")
endif()

# Build complete source list from CUSTOM_SOURCE_DIRS and platform sources
set(AICPU_SOURCES "")

# First, collect all .cpp files from the current platform directory
file(GLOB PLATFORM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND AICPU_SOURCES ${PLATFORM_SOURCES})
# Add common source list like 1) unified log device implementation; 2) platform register implementation
file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../src/aicpu/*.cpp")
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
