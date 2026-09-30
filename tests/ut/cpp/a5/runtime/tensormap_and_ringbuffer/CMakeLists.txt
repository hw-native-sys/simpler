# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a5/runtime/tensormap_and_ringbuffer/.

# Target names carry the a5_ prefix because the a2a3 targets and
# hierarchical/test_tensormap already own the bare names.

# Every case here covers this arch's tensormap_and_ringbuffer runtime.
macro(a5_tmr_case)
    tmr_case(${ARGV} ARCHS a5)
endmacro()

# Exercise the production cold path with an observable platform retirement sink.
a5_tmr_case(test_scheduler_retirement.cpp
    SOURCES
        ${A5_RUNTIME_DIR}/scheduler/scheduler_cold_path.cpp
        ${A5_RUNTIME_DIR}/shared/runtime.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/aicpu_device_config.cpp
    DEFINES SIMPLER_DFX=0
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    TIMEOUT 20)
if(APPLE)
    target_link_options(test_a5_tmr_scheduler_retirement PRIVATE -Wl,-dead_strip)
else()
    target_link_options(test_a5_tmr_scheduler_retirement PRIVATE -Wl,--gc-sections)
endif()

# Everything not named above is a plain case: one file, this runtime's whole
# object library, nothing else to say. Dropping a new test_*.cpp into this
# directory is all it takes — declaring one by hand above is what removes it
# from this sweep, so there is no separate list to keep in step.
simpler_ut_glob_cases(a5_tmr_case)
