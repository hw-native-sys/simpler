# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Object libraries and helpers for every host_build_graph target.

# host_build_graph object libraries, the hbg counterpart of a2a3_tmr_objs and
# a5_tmr_objs above. Two per arch, because the hbg tests split along one line:
# whether they drive the real orchestrator. <arch>_hbg_objs carries what every
# hbg target links; <arch>_hbg_orch_objs adds the orchestrator and the stubs
# standing in for what it calls. A target needing the real dep-gen and
# host-phase TUs links the plain one and lists those itself, since
# hbg_orch_stubs.cpp defines the same symbols.
foreach(arch a2a3 a5)
    simpler_ut_includes(${arch} host_build_graph aicpu _hbg_includes)

    # Not a glob of platform/shared/aicpu: two of the eight files there are
    # what every hbg target needs, and pulling the rest in would collide. They
    # define globals that inline functions in the platform headers reach —
    # g_platform_run_result_base / _epoch and the scope-stats device module
    # state — so any target compiling those headers is undefined without them.
    # args_dump_aicpu belongs to the orch library below, and the remaining
    # four are linked only by the targets that exercise them, some of which
    # substitute a stub for the real implementation.
    #
    # The CANN platform stand-ins come from the arch's support archive rather
    # than being compiled in, so a case that brings a real one of them simply
    # never pulls the archive's copy.
    add_library(${arch}_hbg_objs OBJECT
        ${SIMPLER_SRC}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
    )
    target_include_directories(${arch}_hbg_objs PUBLIC ${_hbg_includes})
    target_link_libraries(${arch}_hbg_objs PUBLIC ${arch}_ut_support)

    add_library(${arch}_hbg_orch_objs OBJECT
        ${HBG_ORCH_SHARED_SOURCES}
        ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
    )
    target_include_directories(${arch}_hbg_orch_objs PUBLIC ${_hbg_includes})
endforeach()

# Tests for src/common/host_build_graph/ — the parts of that runtime shared
# between the architectures. Each case is declared once and built for every
# arch that covers it; a case naming one arch is covered under that arch only.

# Every case here covers src/common/host_build_graph, so the runtime is fixed.
macro(hbg_case)
    simpler_ut_runtime_case(hbg ${ARGV})
endmacro()

macro(hbg_cases)
    simpler_ut_cases(hbg_case ${ARGV})
endmacro()

# These drive the real orchestrator, so they link the orchestrator object
# library. That is the whole of what they have beyond a plain case.
hbg_cases(
    test_hbg_async_wait_init.cpp
    test_hbg_graph_submit_failure.cpp
    test_hbg_graph_definition_arena.cpp
    test_hbg_slot_claim.cpp
    test_hbg_ed_qualification.cpp
    test_hbg_graph_ed_qualification.cpp
    test_hbg_graph_recording_bounds.cpp
    # Calls the real ready_queue_init_data_from_layout. The sizing TU it needs
    # comes with the orchestrator library, which globs host/.
    test_hbg_ready_queue_seed.cpp
    ORCH)

hbg_case(test_hbg_graph_cache.cpp
    SOURCES ${SIMPLER_SRC}/common/host_build_graph/device/graph_execution.cpp)

# TaskId is arch-agnostic (src/common/host_build_graph/task_id.h) and
# header-only, so one arch's include path exercises the whole handle.
hbg_case(test_hbg_task_id.cpp ARCHS a2a3)

# Drives the real hbg bind/validate against a fake HostApi. bind takes the
# resolved host-orch entry points as a parameter, so this needs no orchestration
# .so — the test supplies its own two function pointers.
#
# Lists the orchestrator and shared runtime members directly rather than linking
# the orch library: that library carries hbg_orch_stubs.cpp, whose dep-gen and
# host-phase stubs collide with the real TUs this test needs for
# dep_gen_host_graph_begin_capture and the bind-phase records.
hbg_case(test_hbg_bind_ledger.cpp
    SOURCES
        ${SIMPLER_SRC}/@arch@/runtime/host_build_graph/host/runtime_maker.cpp
        ${HBG_HOST_DIR}/orchestrator.cpp
        ${HBG_SHARED_RUNTIME_SOURCES}
        ${HBG_HOST_DIR}/runtime_core.cpp
        ${HBG_HOST_DIR}/dep_gen_host_graph.cpp
        ${HBG_HOST_DIR}/graph_recorder_pool.cpp
        ${HBG_HOST_DIR}/host_phase_trace.cpp
        ${HBG_HOST_DIR}/ready_queue_sizing.cpp
        ${HBG_HOST_DIR}/host_tensor_access.cpp
        ${SIMPLER_SRC}/common/platform/shared/host/platform_compile_info.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
        # support/unified_log_stubs.cpp already defines unified_log_*; only
        # HostLogger is missing, so this takes host_log.cpp for that alone.
        ${SIMPLER_SRC}/common/log/host_log.cpp)

# Which maker states the caller device buffers a run produces. a2a3's does; a5's makes no such
# statement, so the cases covering the declaration assert the opposite fact there rather than
# skipping. Named after the property, not the arch, so the assertions read as what they check.
if(TARGET test_a2a3_hbg_bind_ledger)
    target_compile_definitions(test_a2a3_hbg_bind_ledger
        PRIVATE SIMPLER_TEST_HBG_MAKER_DECLARES_CALLER_WRITES=1)
endif()

# The scheduler pair links its arch's scheduler TUs plus the AICPU collectors
# the scheduler raises into. a2a3 reaches the sim register shim directly; a5
# needs a stub for the completion path its scheduler does not provide here.
hbg_case(test_hbg_scheduler_drain.cpp
    SOURCES
        ${SIMPLER_SRC}/@arch@/runtime/host_build_graph/runtime/scheduler/scheduler_completion.cpp
        ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
    A2A3_SOURCES ${SIMPLER_SRC}/a2a3/platform/sim/aicpu/inner_platform_regs.cpp
    A5_SOURCES   ${CMAKE_CURRENT_SOURCE_DIR}/support/scheduler_drain_a5_stubs.cpp)

hbg_case(test_hbg_stall_dump_level.cpp
    SOURCES
        ${SIMPLER_SRC}/@arch@/runtime/host_build_graph/runtime/scheduler/scheduler_cold_path.cpp
        ${SIMPLER_SRC}/@arch@/runtime/host_build_graph/runtime/scheduler/scheduler_completion.cpp
        ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
        # Defines get_dev_l2_cache_offset(), which the cold path reads to seed
        # every core's GlobalContext.
        ${SIMPLER_SRC}/common/platform/shared/aicpu/aicpu_device_config.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_SRC}/common/platform/shared/aicpu/device_phase_aicpu.cpp
    A2A3_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/support/stall_dump_level_a2a3_stubs.cpp
        ${SIMPLER_SRC}/a2a3/platform/sim/aicpu/inner_platform_regs.cpp
    A5_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/support/scheduler_drain_a5_stubs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/support/stall_dump_level_a5_stubs.cpp)

# The accumulator is header-only and arch-agnostic, and the test reaches it by
# the host_build_graph/ prefix rather than by bare name, so src/common alone
# puts it on the path — no runtime contract, hence not an hbg_case.
simpler_ut_add_target(NAME test_async_poll_phase_accumulator
    SOURCES test_async_poll_phase_accumulator.cpp
    INCLUDES ${SIMPLER_SRC}/common
    LABEL no_hardware)

# NativeRunContext::publish_acceptance is the only path that marks a task
# accepted, and it is keyed on the launch receipt's identity.
hbg_case(test_native_run_acceptance.cpp
    SOURCES  ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES ${SIMPLER_SRC}/common/worker)

# The host accessor's refusals, over a fake HostApi table: a caller device buffer a live run still
# holds is deferred, an address no region covers stays an invalid argument.
hbg_case(test_host_tensor_access_deferral.cpp
    SOURCES ${HBG_HOST_DIR}/host_tensor_access.cpp)

# Everything else here is a case of this runtime with nothing beyond the
# directory's own contract, built for both arches. Declaring one above takes it
# out of this sweep, so the exceptions are the declarations themselves.
simpler_ut_glob_cases(hbg_case)
