# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(cuda_device_runtime LANGUAGES CXX)

set(CUDA_DEVICE_RUNTIME_SOURCES "")
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND CUDA_DEVICE_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
endif()

add_library(cuda_device_runtime SHARED ${CUDA_DEVICE_RUNTIME_SOURCES})
set_target_properties(cuda_device_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "cuda_device_runtime"
    SUFFIX ".so"
)
target_compile_options(cuda_device_runtime PRIVATE -Wall -Wextra -fPIC -O2)
