# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(cpu_sim_context CXX)

add_library(cpu_sim_context SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu_sim_context.cpp"
)

target_include_directories(cpu_sim_context
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    # libsimpler_log public header (LOG_* macros used here for diagnostics).
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../log/include"
)

target_compile_options(cpu_sim_context PRIVATE
    -fPIC
    -O3
    -g
    -Wall
    -Wextra
)

target_link_libraries(cpu_sim_context PRIVATE pthread)


# Allow undefined HostLogger / unified_log_* symbols — resolved at runtime
# via libsimpler_log.so loaded with RTLD_GLOBAL by ChipWorker.
if(APPLE)
    target_link_options(cpu_sim_context PRIVATE -undefined dynamic_lookup)
endif()
set_target_properties(cpu_sim_context PROPERTIES
    OUTPUT_NAME "cpu_sim_context"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)
