# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.15)

# Support both scikit-build and native CMake builds
if(DEFINED SKBUILD_PROJECT_NAME)
    project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})
    set(SKBUILD_MODE ON)
else()
    project(simpler VERSION 0.1.0)
    set(SKBUILD_MODE OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Include Python bindings
add_subdirectory(python/bindings)

# Protocol used by `build_runtimes.py` when an onboard a2a3 build needs to
# clone pto-isa and PTO_ISA_ROOT is not pre-set. Default `ssh` matches
# conftest's default and self-hosted CI runners' deploy-key setup. Override
# at `pip install` time via
#   pip install --config-settings=cmake.define.SIMPLER_PTO_CLONE_PROTOCOL=https .
set(SIMPLER_PTO_CLONE_PROTOCOL "ssh" CACHE STRING
    "Protocol for cloning pto-isa during install (ssh or https)")

# Pre-build runtime binaries (persistent build dirs for incremental compilation)
add_custom_target(build_runtimes ALL
    COMMAND ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/simpler_setup/build_runtimes.py
        --lib-dir ${CMAKE_SOURCE_DIR}/build/lib
        --cache-dir ${CMAKE_SOURCE_DIR}/build/cache
        --clone-protocol ${SIMPLER_PTO_CLONE_PROTOCOL}
    COMMENT "Building runtime binaries (incremental)..."
)

if(SKBUILD_MODE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
            DESTINATION simpler_setup/_assets/src)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/
            DESTINATION simpler_setup/_assets/build/lib
            OPTIONAL)
endif()
