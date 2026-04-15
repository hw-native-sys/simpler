# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build AICPU Dispatcher SO - Two-layer architecture for runtime-specific AICPU kernels
cmake_minimum_required(VERSION 3.16.3)

project(aicpu_dispatcher LANGUAGES C CXX)

# Dispatcher SO sources
set(AICPU_DISPATCHER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_dispatcher.cpp"
)

# Create shared library
add_library(aicpu_dispatcher SHARED ${AICPU_DISPATCHER_SOURCES})

# C++ standard
set_target_properties(aicpu_dispatcher PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compile options (matching AICPU pattern)
target_compile_options(aicpu_dispatcher
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
)

# Include directories
target_include_directories(aicpu_dispatcher
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."  # For common/unified_log.h
)

# Link against dl for dlopen/dlsym
target_link_libraries(aicpu_dispatcher
    PRIVATE
        dl
)

# Set output name
set_target_properties(aicpu_dispatcher PROPERTIES OUTPUT_NAME "aicpu_dispatcher")
