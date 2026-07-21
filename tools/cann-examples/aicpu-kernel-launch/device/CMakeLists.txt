# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Cross-compiles hello_aicpu.so for the AICPU device side.
# Caller must set CMAKE_C_COMPILER / CMAKE_CXX_COMPILER to the aarch64
# cross toolchain (e.g. ${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++).

cmake_minimum_required(VERSION 3.16)
project(hello_aicpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

add_library(hello_aicpu SHARED hello_aicpu.cpp)

target_include_directories(hello_aicpu PRIVATE
    "${ASCEND_HOME_PATH}/include"
)

target_compile_options(hello_aicpu PRIVATE
    -Wall -Wextra -O2 -fPIC -fno-common -g
)

# Device-side libascend_hal.so so halGetDeviceInfo resolves at load time
# (we link rather than dlopen — same convention as aicpu-device-query).
target_link_directories(hello_aicpu PRIVATE
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/aarch64/"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device/"
)

target_link_libraries(hello_aicpu PRIVATE ascend_hal)

# Build-id required: the host launcher fingerprints the inner SO by ELF
# Build-ID and uses the result to derive the preinstall basename. Without
# --build-id the launcher falls back to FNV-1a, which still works but
# diverges from the production dispatcher's hash convention.
set_target_properties(hello_aicpu PROPERTIES
    LINK_FLAGS "-Wl,--build-id"
    OUTPUT_NAME "hello_aicpu"
)
