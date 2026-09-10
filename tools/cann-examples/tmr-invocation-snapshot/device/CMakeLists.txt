# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)
project(snapshot_probe_device LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_library(snapshot_probe_device SHARED snapshot.cpp)
target_include_directories(snapshot_probe_device PRIVATE
    "${ROOT}/src/common" "${ROOT}/src/common/task_interface"
)
target_compile_options(snapshot_probe_device PRIVATE -Wall -Wextra -O2 -fPIC)
target_link_options(snapshot_probe_device PRIVATE -Wl,--build-id)
