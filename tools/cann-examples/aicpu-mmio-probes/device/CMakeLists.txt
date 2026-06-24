# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Cross-compiles probes.so for the AICPU device side. Caller must set
# CMAKE_C_COMPILER / CMAKE_CXX_COMPILER to the aarch64 cross toolchain
# (e.g. ${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++).
# Mirrors aicpu-kernel-launch/device/CMakeLists.txt.

cmake_minimum_required(VERSION 3.16)
project(aicpu_mmio_probes LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

add_library(aicpu_mmio_probes SHARED probes.cpp)

target_include_directories(aicpu_mmio_probes PRIVATE
    "${ASCEND_HOME_PATH}/include"
)

target_compile_options(aicpu_mmio_probes PRIVATE
    -Wall -Wextra -O2 -fPIC -fno-common -g
)

# pthread is required for the Phase 12-C multi-thread scaling test.
find_package(Threads REQUIRED)
target_link_libraries(aicpu_mmio_probes PRIVATE Threads::Threads)

set_target_properties(aicpu_mmio_probes PROPERTIES
    LINK_FLAGS "-Wl,--build-id"
    OUTPUT_NAME "aicpu_mmio_probes"
)
