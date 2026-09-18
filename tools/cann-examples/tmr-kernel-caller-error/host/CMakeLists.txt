# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(tmr_caller_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
include("${ROOT}/cmake/host_log_sources.cmake")
add_library(tmr_caller_probe SHARED probe.cpp
    "${ROOT}/src/common/aicpu_loader/host/load_aicpu_op.cpp"
    "${ROOT}/src/common/task_interface/assert_compat.cpp"
    ${SIMPLER_HOST_LOG_SOURCES})
set(RUNTIME "${ROOT}/src/a2a3/runtime/tensormap_and_ringbuffer")
target_include_directories(tmr_caller_probe PRIVATE
    "${ROOT}/src/common" "${ROOT}/src/common/task_interface" "${ROOT}/src/common/worker"
    "${ROOT}/src/common/platform/include" "${ROOT}/src/a2a3/platform/include"
    "${ROOT}/src/common/log/include" "${ROOT}/src/common/aicpu_loader/host"
    "${RUNTIME}/runtime" "${RUNTIME}/common" "${RUNTIME}/.."
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/pkg_inc/base" "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" "${ASCEND_HOME_PATH}/pkg_inc/driver"
    "${ASCEND_HOME_PATH}/pkg_inc/profiling")
target_link_directories(tmr_caller_probe PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(tmr_caller_probe PRIVATE ascendcl runtime pthread dl)
target_compile_options(tmr_caller_probe PRIVATE -Wall -Wextra -Werror -O2)
