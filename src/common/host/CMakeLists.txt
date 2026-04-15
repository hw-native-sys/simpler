# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build host-side AICPU operation loader
cmake_minimum_required(VERSION 3.16.3)

project(host_common LANGUAGES C CXX)

# Host common sources
set(HOST_COMMON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/load_aicpu_op.cpp"
)

# This library is included directly in host_runtime, not built separately
# Sources are added to HOST_RUNTIME_SOURCES in platform CMakeLists.txt
