# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# libsimpler_log.so — process-wide host logging library.
#
# Built once per arch/variant (a5sim, a5, a2a3sim, a2a3) and loaded via
# RTLD_GLOBAL by every consumer .so (host_runtime, cpu_sim_context, the
# nanobind binding) so HostLogger has a single instance per process.
#
# Public ABI (consumed by host AND device .cpp): include/common/unified_log.h.
# Device-side .cpp files do NOT link this library — they only need the header
# (their own translation unit provides the unified_log_* implementation).
cmake_minimum_required(VERSION 3.16)
project(simpler_log CXX)

add_library(simpler_log SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/host_log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unified_log_host.cpp"
)

target_include_directories(simpler_log
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_options(simpler_log PRIVATE
    -fPIC
    -O3
    -g
    -Wall
    -Wextra
    $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>
)

target_link_libraries(simpler_log PRIVATE pthread)

set_target_properties(simpler_log PROPERTIES
    OUTPUT_NAME "simpler_log"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
