# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Object libraries and helpers for every tensormap_and_ringbuffer target.

# ---------------------------------------------------------------------------
# A2A3 runtime sources and stubs for ring-buffer / tensormap tests.
#
# Bundled into an OBJECT library so the runtime .cpp files compile once and
# the resulting .o files are reused across every test executable. PUBLIC
# include dirs / compile options propagate to consumers via
# target_link_libraries, so test executables get the same flags they would
# have had if the sources were listed directly.
#
# The object libraries live here rather than in the directory they are named
# for because CMake requires a target to be defined before a directory that
# links it, and each is linked from three: the arch's own runtime directory
# and common/<runtime>/, plus common/platform/ for the tmr pair. Defining one
# inside a2a3/ would force common/ to be descended into after it, and its a5
# twin would demand the same of a5/.
# ---------------------------------------------------------------------------

add_library(a2a3_tmr_objs OBJECT
    ${A2A3_RUNTIME_DIR}/ring_buffer.cpp
    ${A2A3_RUNTIME_DIR}/shared/shared_memory.cpp
    ${A2A3_RUNTIME_DIR}/scheduler/scheduler.cpp
    ${A2A3_RUNTIME_DIR}/shared/tensormap.cpp
    ${A2A3_RUNTIME_DIR}/shared/runtime_init.cpp
    ${A2A3_RUNTIME_DIR}/orchestrator.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
    # Compiled in, not taken from the archive: orchestrator.cpp below defines
    # get_sys_cnt_aicpu weakly as `return 0`, which satisfies the reference and
    # stops the linker searching an archive for the real one. Its absence is a
    # hang rather than a link error — the 500 ms reclaim backstop is counted in
    # ticks that never advance.
    ${CMAKE_SOURCE_DIR}/support/device_time_stub.cpp
)
simpler_ut_includes(a2a3 tensormap_and_ringbuffer aicpu _a2a3_tmr_inc)
target_include_directories(a2a3_tmr_objs PUBLIC ${_a2a3_tmr_inc} ${SIMPLER_SRC}/a2a3/runtime)
# The CANN platform stand-ins, including the get_reg_ptr that
# SchedulerState::ring_one_doorbell resolves, come from the arch's support
# archive rather than being compiled in here.
target_link_libraries(a2a3_tmr_objs PUBLIC a2a3_ut_support)

# ---------------------------------------------------------------------------
# A5 runtime sources, mirroring a2a3_tmr_objs. Bundled into an OBJECT library
# so the runtime .cpp files compile once and the resulting .o files are
# reused across every a5 runtime test executable.
# ---------------------------------------------------------------------------

add_library(a5_tmr_objs OBJECT
    ${A5_RUNTIME_DIR}/ring_buffer.cpp
    ${A5_RUNTIME_DIR}/shared/shared_memory.cpp
    ${A5_RUNTIME_DIR}/scheduler/scheduler.cpp
    ${A5_RUNTIME_DIR}/shared/tensormap.cpp
    ${A5_RUNTIME_DIR}/shared/runtime_init.cpp
    ${A5_RUNTIME_DIR}/orchestrator.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/device_run_result_base_aicpu.cpp
    ${SIMPLER_SRC}/common/platform/shared/aicpu/args_dump_aicpu.cpp
    # Same weak get_sys_cnt_aicpu as a2a3's; see the note there.
    ${CMAKE_SOURCE_DIR}/support/device_time_stub.cpp
)
simpler_ut_includes(a5 tensormap_and_ringbuffer aicpu _a5_tmr_inc)
target_include_directories(a5_tmr_objs PUBLIC ${_a5_tmr_inc} ${SIMPLER_SRC}/a5/runtime)
target_link_libraries(a5_tmr_objs PUBLIC a5_ut_support)

#
# Tests for src/common/tensormap_and_ringbuffer/ — the parts of that runtime
# shared between the architectures.
#
# The sources here are arch-independent, but the runtime they cover compiles
# into each arch's image against that arch's headers, so a case is built once
# per arch.

# Every case here covers src/common/tensormap_and_ringbuffer, so the runtime is
# fixed; each is declared once and built for every arch that covers it.
macro(tmr_case)
    simpler_ut_runtime_case(tmr ${ARGV})
endmacro()

tmr_case(test_dep_gen_replay.cpp
    SOURCES  ${TMR_SHARED_HOST_DIR}/dep_gen_replay.cpp
    INCLUDES ${TMR_SHARED_HOST_DIR})

tmr_case(test_trb_runtime_temp_buffer.cpp
    SOURCES
        ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp
        ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
        ${SIMPLER_SRC}/common/platform/shared/host/platform_compile_info.cpp
    INCLUDES ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/host)

# Everything else here is a case of this runtime with nothing beyond the
# directory's own contract, built for both arches.
simpler_ut_glob_cases(tmr_case)
