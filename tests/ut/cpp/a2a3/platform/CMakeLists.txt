# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a2a3/platform/.

set(A2A3_ONBOARD_HOST_DIR ${SIMPLER_SRC}/a2a3/platform/onboard/host)

# The arch is this directory's, so a case here names only what it covers and
# the target name picks the arch up from the mirror path.
macro(a2a3_platform_case src)
    string(REGEX REPLACE "^test_(.*)\\.cpp$" "\\1" _a2a3_platform_stem "${src}")
    simpler_ut_common_includes(a2a3 host _a2a3_platform_contract)
    simpler_ut_add_target(NAME test_a2a3_${_a2a3_platform_stem}
        SOURCES ${src}
        INCLUDES ${_a2a3_platform_contract}
        LABEL no_hardware ${ARGN})
endmacro()

# aicpu_affinity_select.cpp holds the arithmetic half of the probe and reaches
# for neither CANN nor libdl, so it compiles here as-is; its sibling
# aicpu_topology_probe.cpp owns the driver queries and includes ascend_hal.h,
# which this CANN-free build cannot see. The test provides its own no-op logger
# stubs, so no logger sources are linked.
a2a3_platform_case(test_aicpu_affinity_select.cpp
    SOURCES ${A2A3_ONBOARD_HOST_DIR}/aicpu_affinity_select.cpp
    INCLUDES ${A2A3_ONBOARD_HOST_DIR})

# The wrapped sched_* symbols are a GNU ld feature, so this builds on Linux only.
# sched_getscheduler/sched_setscheduler are also not declared on non-Linux
# headers, so this must stay out of simpler_ut_glob_cases()'s default sweep
# below rather than just skip the LINK_OPTIONS.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    a2a3_platform_case(test_thread_scheduling.cpp
        LINK_OPTIONS -Wl,--wrap=sched_getscheduler -Wl,--wrap=sched_setscheduler)
else()
    simpler_ut_exclude_case(test_thread_scheduling.cpp)
endif()

simpler_ut_glob_cases(a2a3_platform_case)
