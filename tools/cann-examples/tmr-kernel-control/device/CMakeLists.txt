# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(tmr_control_probe_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
if(NOT ARCH MATCHES "^(a2a3|a5)$")
    message(FATAL_ERROR "Pass -DARCH=a2a3 or -DARCH=a5 matching the target silicon")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
add_library(tmr_control_probe_device SHARED probe.cpp gate_probe.cpp
    "${ROOT}/src/${ARCH}/platform/shared/aicpu/platform_regs.cpp"
    "${ROOT}/src/${ARCH}/platform/onboard/aicpu/inner_platform_regs.cpp"
    "${ROOT}/src/common/platform/onboard/aicpu/cache_ops.cpp"
    "${ROOT}/src/common/platform/onboard/aicpu/device_time.cpp"
    "${ROOT}/src/common/platform/onboard/aicpu/device_log.cpp"
    "${ROOT}/src/common/platform/onboard/aicpu/platform_aicpu_affinity.cpp"
    "${ROOT}/src/common/platform/shared/aicpu/unified_log_device.cpp"
)
set(RUNTIME "${ROOT}/src/${ARCH}/runtime/tensormap_and_ringbuffer")
target_include_directories(tmr_control_probe_device PRIVATE
    "${ROOT}/src/common" "${ROOT}/src/common/task_interface"
    "${ROOT}/src/common/platform/include" "${ROOT}/src/${ARCH}/platform/include"
    "${ROOT}/src/common/platform/onboard/aicpu" "${ROOT}/src/common/log/include"
    "${RUNTIME}/runtime" "${RUNTIME}/common" "${RUNTIME}/.."
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc" "${ASCEND_HOME_PATH}/pkg_inc/base"
)
target_compile_options(tmr_control_probe_device PRIVATE -Wall -Wextra -Werror -O2 -fPIC)
target_link_options(tmr_control_probe_device PRIVATE -Wl,--build-id)
