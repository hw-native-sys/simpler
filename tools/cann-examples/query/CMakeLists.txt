# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Standalone CMake project — not part of the main runtime build.
# Build:
#   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
#   cmake -B build .
#   cmake --build build
#   ./build/query

cmake_minimum_required(VERSION 3.16)
project(cann_query LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR
        "ASCEND_HOME_PATH must be set to the CANN toolkit root. "
        "Example: export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ascend_driver_path.cmake")

add_executable(query query.cpp)

# `runtime/rt.h` transitively includes `prof_common.h`, which lives under
# pkg_inc/profiling — both pkg_inc roots are needed here.
target_include_directories(query PRIVATE
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
)

target_link_directories(query PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/runtime/lib64"
    "${ASCEND_DRIVER_PATH}/lib64/driver"
)

# Direct link instead of dlopen: query depends on the Ascend driver at link
# time; if HAL or DSMI is missing the binary will fail to load on startup.
target_link_libraries(query PRIVATE
    ascendcl       # aclInit, aclrt*, aclGetSocName
    runtime        # rtGetAiCpuCount
    ascend_hal     # halGetDeviceInfo, halGetDeviceInfoByBuff
    drvdsmi_host   # dsmi_get_device_info
)
