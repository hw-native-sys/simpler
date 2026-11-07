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

a5_tmr_case(test_ready_queue_profiling.cpp
    NO_OBJS
    SOURCES
        ${A5_RUNTIME_DIR}/shared/runtime_init.cpp
        ${CMAKE_SOURCE_DIR}/support/device_time_stub.cpp
    DEFINES SIMPLER_DFX=1 SIMPLER_SCHED_PROFILING=1 SIMPLER_ORCH_PROFILING=1
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK a5_ut_support
    TIMEOUT 20)
if(APPLE)
    target_link_options(test_a5_tmr_ready_queue_profiling PRIVATE -Wl,-dead_strip)
else()
    target_link_options(test_a5_tmr_ready_queue_profiling PRIVATE -Wl,--gc-sections)
endif()

a5_tmr_case(test_mix_spare_lane.cpp
    NO_OBJS
    SOURCES
        ${A5_RUNTIME_DIR}/ring_buffer.cpp
        ${A5_RUNTIME_DIR}/shared/shared_memory.cpp
        ${A5_RUNTIME_DIR}/shared/runtime_init.cpp
        ${A5_RUNTIME_DIR}/scheduler/scheduler.cpp
        ${A5_RUNTIME_DIR}/scheduler/scheduler_dispatch.cpp
        ${A5_RUNTIME_DIR}/scheduler/scheduler_completion.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
    DEFINES SIMPLER_DFX=0
    LINK a5_ut_support
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    TIMEOUT 30)
if(APPLE)
    target_link_options(test_a5_tmr_mix_spare_lane PRIVATE -Wl,-dead_strip)
else()
    target_link_options(test_a5_tmr_mix_spare_lane PRIVATE -Wl,--gc-sections)
endif()

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

# Exercise the failed submission slab through the real FIN completion path.
a5_tmr_case(test_urma_completion_kernel.cpp
    SOURCES
        ${A5_RUNTIME_DIR}/scheduler/scheduler_completion.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_SRC}/a5/platform/shared/aicpu/pmu_collector_aicpu.cpp
    COMPILE_OPTIONS -ffunction-sections -fdata-sections)
if(APPLE)
    target_link_options(test_a5_tmr_urma_completion_kernel PRIVATE -Wl,-dead_strip)
else()
    target_link_options(test_a5_tmr_urma_completion_kernel PRIVATE -Wl,--gc-sections)
endif()

# Real polling and completion code with observable COND and doorbell sinks.
# Each target compiles its production TUs with the same profiling definitions
# as the fixture; the shared object library has a different Context layout.
function(a5_completion_batch_case suffix dfx sched_profiling)
    set(target test_a5_tmr_completion_batch${suffix})
    simpler_ut_add_target(NAME ${target}
        ARCH a5 RUNTIME tensormap_and_ringbuffer TIER aicpu
        SOURCES test_completion_batch.cpp
            ${A5_RUNTIME_DIR}/scheduler/scheduler.cpp
            ${A5_RUNTIME_DIR}/scheduler/scheduler_completion.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
        DEFINES SIMPLER_PLATFORM_NAME="a5sim" SIMPLER_DFX=${dfx}
            SIMPLER_SCHED_PROFILING=${sched_profiling}
        LINK a5_ut_support
        COMPILE_OPTIONS -ffunction-sections -fdata-sections
        LABEL no_hardware PRIVATE_WORKDIR TIMEOUT 20)
    if(APPLE)
        target_link_options(${target} PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(${target} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()
a5_completion_batch_case("" 0 0)
a5_completion_batch_case(_dfx 1 0)
a5_completion_batch_case(_profiling 1 1)

# Everything not named above is a plain case: one file, this runtime's whole
# object library, nothing else to say. Dropping a new test_*.cpp into this
# directory is all it takes — declaring one by hand above is what removes it
# from this sweep, so there is no separate list to keep in step.
simpler_ut_glob_cases(a5_tmr_case)
