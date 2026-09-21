# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/log/.

# Every case here compiles the host logger sources alongside itself, so the
# logger's own directory and its public headers are the whole contract.
set(log_includes ${SIMPLER_SRC}/common/task_interface ${SIMPLER_LOG_DIR} ${SIMPLER_LOG_DIR}/include)

macro(log_case src)
    simpler_ut_plain_case(${src}
        SOURCES ${HOST_LOG_TEST_SOURCES}
        INCLUDES ${log_includes}
        LABEL no_hardware ${ARGN})
endmacro()

log_case(test_host_log_nonblocking.cpp
    DEFINES SIMPLER_HOST_LOG_TEST_HOOKS=1
    TIMEOUT 8)

# A dlopened module that compiles the host logger owns a private buffered
# stream on the shared log file, so unloading it must not strand that buffer.
# Two such modules, one per case, built from support/ sources.
foreach(consumer host_log_unload_consumer host_log_consumer)
    add_library(test_${consumer} SHARED
        ${CMAKE_SOURCE_DIR}/common/support/${consumer}.cpp
        ${HOST_LOG_TEST_SOURCES}
    )
    target_include_directories(test_${consumer} PRIVATE ${SIMPLER_LOG_DIR} ${SIMPLER_LOG_DIR}/include)
    target_link_libraries(test_${consumer} PRIVATE pthread)
endforeach()

log_case(test_host_log_dso_unload.cpp
    DEFINES TEST_HOST_LOG_UNLOAD_CONSUMER_PATH="$<TARGET_FILE:test_host_log_unload_consumer>"
    LINK ${CMAKE_DL_LIBS} DEPENDS test_host_log_unload_consumer)

log_case(test_host_log_cross_dso.cpp
    DEFINES TEST_HOST_LOG_CONSUMER_PATH="$<TARGET_FILE:test_host_log_consumer>"
    LINK ${CMAKE_DL_LIBS} DEPENDS test_host_log_consumer)

# Sim device logging goes through the bound HostLogger threshold, envelope and
# output. Pure host code — no CANN, no hardware. It compiles the sim adapter
# rather than the whole host logger, so it takes neither of the directory
# defaults.
simpler_ut_add_target(NAME test_sim_device_log
    SOURCES test_sim_device_log.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/sim/aicpu/device_log.cpp
        ${SIMPLER_LOG_DIR}/host_log.cpp
    INCLUDES ${SIMPLER_COMMON_PLATFORM_DIR}/include ${SIMPLER_LOG_DIR}/include
    LABEL no_hardware)

simpler_ut_glob_cases(log_case)
