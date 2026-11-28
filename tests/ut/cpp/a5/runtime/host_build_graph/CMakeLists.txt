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

set(A5_HBG_DIR ${SIMPLER_SRC}/a5/runtime/host_build_graph)

a5_hbg_case(test_hbg_args_dump_service.cpp
    INCLUDES ${A5_HBG_DIR}/aicpu
    SOURCES
        ${A5_HBG_DIR}/aicpu/args_dump_service.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/args_dump_aicpu.cpp)

a5_hbg_case(test_hbg_task_image.cpp
    INCLUDES ${A5_HBG_DIR}/aicpu
    SOURCES ${SIMPLER_SRC}/common/host_build_graph/shared/shared_memory.cpp)

a5_hbg_case(test_hbg_ccu_completion.cpp NO_OBJS LINK a5_ut_support TIMEOUT 20
    SOURCES ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

a5_hbg_case(test_hbg_calendar_execution.cpp NO_OBJS TIMEOUT 20
    INCLUDES ${A5_HBG_DIR}/aicore ${SIMPLER_SRC}/a5/platform/sim/aicore
    DEFINES __CPU_SIM)

a5_hbg_case(test_hbg_calendar_scalar_memory.cpp NO_OBJS TIMEOUT 20
    SOURCES scalar_memory_unprofiled.cpp test_urma_completion.cpp test_urma_requests.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/sim/aicore
    DEFINES __CPU_SIM)

a5_hbg_case(test_hbg_calendar_phases.cpp NO_OBJS LINK a5_ut_support TIMEOUT 20
    INCLUDES ${A5_HBG_DIR})

a5_hbg_case(test_hbg_graph_validation.cpp
    SOURCES ${A5_HBG_DIR}/host/graph_validation.cpp)

a5_hbg_case(test_hbg_scheduler_contracts.cpp
    INCLUDES ${A5_HBG_DIR}/aicpu)

a5_hbg_case(test_hbg_nonce_exit.cpp NO_OBJS TIMEOUT 20
    SOURCES
        ${A5_HBG_DIR}/aicore/entry.cpp
        ${A5_HBG_DIR}/aicore/profiled.cpp
        ${A5_HBG_DIR}/aicore/task_profiled.cpp
        ${A5_HBG_DIR}/aicore/unprofiled.cpp
        ${SIMPLER_SRC}/a5/platform/shared/aicpu/platform_regs.cpp
        ${SIMPLER_SRC}/a5/platform/sim/aicpu/inner_platform_regs.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/sim/aicpu/device_time.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/sim/aicpu/cache_ops.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/sim/aicore ${SIMPLER_COMMON_PLATFORM_DIR}/sim/aicpu)

if(APPLE)
    set(_gc_link_option -Wl,-dead_strip)
else()
    set(_gc_link_option -Wl,--gc-sections)
endif()

# The register-layer tests exercise the ACK and release sequence. This case
# exercises selection and retirement of its owners.
a5_hbg_case(test_hbg_retirement_wiring.cpp
    INCLUDES ${A5_HBG_DIR}/aicpu
    SOURCES
        ${A5_HBG_DIR}/aicpu/aicore_lifecycle.cpp
        ${HBG_COMMON_SHARED_DIR}/runtime.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_SRC}/a5/platform/shared/aicpu/pmu_collector_aicpu.cpp
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS ${_gc_link_option})

a5_hbg_case(test_hbg_submit_poison.cpp ORCH)

a5_hbg_case(test_hbg_report_epoch.cpp
    SOURCES ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

a5_hbg_case(test_hbg_calendar_late_aicore_error.cpp
    INCLUDES ${A5_HBG_DIR}/aicpu
    SOURCES
        ${A5_HBG_DIR}/aicpu/steps.cpp
        ${A5_HBG_DIR}/aicpu/aicore_lifecycle.cpp
        ${A5_HBG_DIR}/aicpu/args_dump_service.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/args_dump_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/aicpu_device_config.cpp
        ${HBG_COMMON_SHARED_DIR}/runtime.cpp
        ${SIMPLER_SRC}/common/host_build_graph/device/graph_execution.cpp
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS ${_gc_link_option})

simpler_ut_glob_cases(a5_hbg_case)
