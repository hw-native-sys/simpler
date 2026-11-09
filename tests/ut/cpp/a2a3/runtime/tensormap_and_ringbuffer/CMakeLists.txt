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

a2a3_tmr_case(test_entry_cache_maintenance.cpp
    SOURCES ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer)

# Exercise production entry/teardown with only hardware boundaries replaced.
set(_entry_cache_sources
    ${SIMPLER_SRC}/common/platform/shared/aicpu/platform_entry_args.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/aicpu_device_config.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/dep_gen_collector_aicpu.cpp
    ${SIMPLER_SRC}/common/platform/sim/aicpu/platform_aicpu_affinity.cpp)
a2a3_tmr_case(test_entry_cache_deinit.cpp NO_OBJS
    SOURCES ${_entry_cache_sources}
        ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
        ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp
    INCLUDES ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer
    DEFINES SIMPLER_DFX=0
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS ${_retirement_gc_link_option}
    LINK a2a3_ut_support ${CMAKE_DL_LIBS})
# The onboard kernel entry uses Linux scheduling APIs.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    a2a3_tmr_case(test_entry_cache_projection.cpp NO_OBJS
        SOURCES ${_entry_cache_sources}
            ${SIMPLER_SRC}/a2a3/platform/onboard/aicpu/kernel.cpp
            ${SIMPLER_SRC}/a2a3/platform/shared/aicpu/platform_regs.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/chip_swimlane_collector_aicpu.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
            ${SIMPLER_SRC}/a2a3/platform/shared/aicpu/pmu_collector_aicpu.cpp
            ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
            ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
        DEFINES SIMPLER_DFX=0
        COMPILE_OPTIONS -ffunction-sections -fdata-sections
        LINK_OPTIONS ${_retirement_gc_link_option}
        LINK a2a3_ut_support)
else()
    simpler_ut_exclude_case(test_entry_cache_projection.cpp)
endif()

if(APPLE)
    set(_entry_preparation_gc -Wl,-dead_strip)
else()
    set(_entry_preparation_gc -Wl,--gc-sections)
endif()
a2a3_tmr_case(test_entry_args_preparation.cpp NO_OBJS
    SOURCES
        ${A2A3_RUNTIME_DIR}/shared/runtime.cpp
        ${A2A3_RUNTIME_DIR}/scheduler/scheduler_cold_path.cpp
        ${A2A3_RUNTIME_DIR}/scheduler/scheduler.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/aicpu_device_config.cpp
    INCLUDES ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/aicpu
    LINK a2a3_ut_support ${CMAKE_DL_LIBS}
    DEFINES SIMPLER_DFX=0 SIMPLER_ORCH_PROFILING=0
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS ${_entry_preparation_gc}
    TIMEOUT 20)

simpler_ut_add_target(NAME test_a2a3_tmr_drain_admission ARCH a2a3 RUNTIME tensormap_and_ringbuffer TIER aicpu
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_drain_admission.cpp
        ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp
        ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
        ${SIMPLER_SRC}/a2a3/platform/sim/aicpu/inner_platform_regs.cpp
    DEFINES SIMPLER_DFX=0 LINK a2a3_ut_support
    COMPILE_OPTIONS -ffunction-sections -fdata-sections LINK_OPTIONS ${_entry_preparation_gc} TIMEOUT 30)

# Everything not named above is a plain case: one file, this runtime's whole
# object library, nothing else to say. Dropping a new test_*.cpp into this
# directory is all it takes — declaring one by hand above is what removes it
# from this sweep, so there is no separate list to keep in step.
simpler_ut_glob_cases(a2a3_tmr_case)
