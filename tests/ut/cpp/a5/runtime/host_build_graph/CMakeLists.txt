# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a5/runtime/host_build_graph/.
#
# The arch is this directory's, so no case here states it.
macro(a5_hbg_case src)
    hbg_case(${src} ARCHS a5 ${ARGN})
endmacro()

a5_hbg_case(test_hbg_scheduler_contracts.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/host_build_graph/aicpu)

a5_hbg_case(test_hbg_scheduler_bootstrap.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/host_build_graph/aicpu
    SOURCES  ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

# Drives the real orchestrator submit path.
a5_hbg_case(test_hbg_submit_poison.cpp ORCH)

# Incremental activation inits a real SchedulerState (ready queues), so it links
# the scheduler and shared runtime out-of-line members.
a5_hbg_case(test_graph_activation.cpp
    SOURCES ${HBG_SHARED_RUNTIME_SOURCES})

# The epoch the scheduler reports under, and the terminal record the legacy
# path still writes. Both read state this arch's hbg AICPU headers declare and
# call the shared runtime's out-of-line members.
simpler_ut_cases(a5_hbg_case
    test_hbg_report_epoch.cpp
    test_hbg_legacy_terminal_record.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/host_build_graph/aicpu
    SOURCES  ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

# An AICore error arriving after the resident graph has been torn down. The
# case asserts the late path is not linked in at all, so the build discards
# unreferenced sections rather than letting one survive behind a dead branch.
if(APPLE)
    set(_gc_link_option -Wl,-dead_strip)
else()
    set(_gc_link_option -Wl,--gc-sections)
endif()
a5_hbg_case(test_hbg_resident_late_aicore_error.cpp
    INCLUDES        ${SIMPLER_SRC}/a5/runtime/host_build_graph/aicpu
    SOURCES         ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS    ${_gc_link_option})

simpler_ut_glob_cases(a5_hbg_case)
