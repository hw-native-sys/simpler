# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(tmr_caller_probe_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
add_library(tmr_caller_probe_device SHARED probe.cpp)
target_include_directories(tmr_caller_probe_device PRIVATE
    "${ROOT}/src/common/platform/include"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc")
target_compile_options(tmr_caller_probe_device PRIVATE -Wall -Wextra -Werror -O2 -fPIC)
target_link_options(tmr_caller_probe_device PRIVATE -Wl,--build-id)

# Use the checkout's real bootstrap, not a stale binary from another branch.
add_library(simpler_aicpu_dispatcher SHARED
    "${ROOT}/src/common/aicpu_loader/device/aicpu_dispatcher.cpp")
target_include_directories(simpler_aicpu_dispatcher PRIVATE "${ROOT}/src/common")
target_compile_options(simpler_aicpu_dispatcher PRIVATE -Wall -Wextra -O2 -fPIC)
target_link_options(simpler_aicpu_dispatcher PRIVATE -Wl,--build-id)
target_link_libraries(simpler_aicpu_dispatcher PRIVATE dl)
