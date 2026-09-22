# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a2a3/runtime/tensormap_and_ringbuffer/.

# Every case here covers this arch's tensormap_and_ringbuffer runtime.
macro(a2a3_tmr_case)
    tmr_case(${ARGV} ARCHS a2a3)
endmacro()

# #includes scope_stats_collector_aicpu.cpp into itself, reaching it by the same
# "aicpu/..." prefix as the header beside it. It therefore defines what the
# runtime library also carries, and takes the contract without the objects.
a2a3_tmr_case(test_scope_stats_collector.cpp NO_OBJS
    SOURCES  ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
    LINK     @arch@_ut_support
    INCLUDES ${SIMPLER_SRC}/common/platform/shared)

# Everything not named above is a plain case: one file, this runtime's whole
# object library, nothing else to say. Dropping a new test_*.cpp into this
# directory is all it takes — declaring one by hand above is what removes it
# from this sweep, so there is no separate list to keep in step.
simpler_ut_glob_cases(a2a3_tmr_case)
