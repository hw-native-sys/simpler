# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/platform/ and the per-arch platform trees it mirrors.
#
# Platform sources have no build of their own — they compile into each runtime's
# image, against one arch's headers — so how many times a case here is built is
# a property of what it reaches, and every declaration says which axis it
# repeats along. SINGLE is the claim that the case opens nothing under
# src/<arch>/platform or src/<arch>/runtime, so one build is every build; it is
# checked by preprocessing the case and reading the files the preprocessor
# opened, not assumed from how the case was declared before.

# The directory's own shape: src/common/worker on top of the common contract,
# since the host-side platform headers reach the worker types by bare name, and
# the arch's support archive, which supplies whatever CANN platform symbol a
# case leaves undefined. A case compiling a real one of those instead never
# pulls the archive's copy, so it says nothing about it.
macro(platform_case src)
    simpler_ut_platform_case(${src}
        INCLUDES ${SIMPLER_SRC}/common/worker
        LINK @arch@_ut_support
        ${ARGN})
endmacro()

macro(platform_single_case src)
    platform_case(${src} SINGLE ${ARGN})
endmacro()

macro(platform_cases)
    simpler_ut_cases(platform_case ${ARGV})
endmacro()

# ---------------------------------------------------------------------------
# Cases carrying production sources of their own
# ---------------------------------------------------------------------------

# Kernel-mode phase machine and restricted-vocabulary tables, table-driven with
# fake ops: init/close balance, poison, sticky CLOSING with retry, and the
# separate runtime-error vs teardown-error slots.
platform_single_case(test_kernel_execution_state.cpp
    SOURCES ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/kernel_execution_state.cpp)

# Sim kernel threads publish one sticky completion state. A gated unit test
# proves submission and completion are separate without relying on kernel
# duration or host scheduling.
platform_single_case(test_sim_run_completion.cpp
    INCLUDES ${SIMPLER_COMMON_PLATFORM_DIR}/sim/host)

# RegisterCallableArgs lives in the arch platform headers, and a2a3's
# kernel_args.h is not a5's, so the POD shape this checks is checked per arch.
platform_case(test_runtime_orch_so.cpp PER_ARCH)

# Both compile the host logger alongside themselves, so they take the platform
# stubs without the log group — the one entry unified_log_host.cpp replaces.
platform_cases(
    test_buffer_pool_manager.cpp
    test_profiler_base.cpp
    PER_ARCH
    SOURCES ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_LOG_DIR})

# MemoryAllocator byte-counter test: links the SIM impl (std::malloc/free,
# no CANN) the same way test_buffer_pool_manager pulls unified_log.
platform_single_case(test_memory_allocator.cpp
    SOURCES ${SIMPLER_COMMON_PLATFORM_DIR}/sim/host/memory_allocator.cpp ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_LOG_DIR})

# Each of these covers a header-only piece of the arch's platform surface and
# needs nothing compiled in but the CANN platform stand-in.
platform_cases(
    test_profiler_device_engine.cpp
    test_worker_chip_message_queue.cpp
    test_worker_chip_orch_endpoint.cpp
    test_region_instance_view.cpp
    PER_ARCH)

platform_case(test_scope_stats_collector.cpp PER_ARCH
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/scope_stats_collector.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp
    TIER aicpu)

# AICPU-side dep_gen writer. The collector stores the DepFlags byte without
# interpreting it and reaches no runtime tree of its own, so one build per arch
# is every build — the runtime axis does not apply to it.
platform_case(test_dep_gen_collector_aicpu.cpp PER_ARCH
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/dep_gen_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp)

# dep_gen run identity + buffer ownership, device producer and host collector in
# one binary so the whole chain (acquire -> record -> flush -> host copy ->
# return) is production code.
platform_case(test_dep_gen_run_identity.cpp PER_ARCH
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/dep_gen_collector.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/dep_gen_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp
    DEFINES SIMPLER_RUNTIME_NAME="test_runtime"
    TIER aicpu)

# One run's terminal record: the wire layout every arch/runtime pair publishes,
# the AICPU publisher, and the fold that picks the code. The sim cache_ops give
# the publisher's flushes a host-side no-op implementation.
platform_single_case(test_run_terminal_record.cpp
    SOURCES ${SIMPLER_COMMON_PLATFORM_DIR}/sim/aicpu/cache_ops.cpp)

# Per-callable_id orch SO file naming regression (see rtStreamSynchronize
# 507018 root cause). Compiles the onboard `create_orch_so_file` against the
# test source so it runs on no-hw runners too.
platform_single_case(test_orch_so_file.cpp
    SOURCES ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/aicpu/orch_so_file.cpp)

# Onboard uses CANN for DEBUG/INFO/WARN/ERROR but keeps TIMING on simpler's
# finer threshold. The dlog stub makes the CANN levels available without SDK
# headers or hardware.
platform_single_case(test_onboard_device_log_level.cpp
    SOURCES ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/aicpu/device_log.cpp
    INCLUDES ${CMAKE_SOURCE_DIR}/support)

# ---------------------------------------------------------------------------
# Cases whose shape the factory cannot express
# ---------------------------------------------------------------------------

# Not a case: a static_assert compiled once per {arch, variant}, asserting that
# spin_hint.h aliases PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS to the default from
# its own variant's platform_config.h. The assertion is the build — an alias
# pointing at the other variant's default fails to compile, and there is
# nothing to run or register. The variant axis is this check's alone; no case
# in the tree varies along it.
foreach(arch a2a3 a5)
    foreach(variant onboard sim)
        string(TOUPPER ${variant} variant_uc)
        add_library(${arch}_${variant}_spin_selector_check OBJECT
            ${CMAKE_CURRENT_SOURCE_DIR}/spin_hint_selector_check.cpp
        )
        target_compile_definitions(${arch}_${variant}_spin_selector_check PRIVATE
            EXPECTED_TENSOR_DATA_WAIT_TIMEOUT_MS=PLATFORM_${variant_uc}_TENSOR_DATA_WAIT_TIMEOUT_MS
        )
        target_include_directories(${arch}_${variant}_spin_selector_check PRIVATE
            ${SIMPLER_SRC}/${arch}/platform/include
            ${SIMPLER_COMMON_PLATFORM_DIR}/${variant}/aicpu
            ${SIMPLER_COMMON_PLATFORM_DIR}/include
        )
    endforeach()
endforeach()

# Two loadable stand-ins for a host runtime, built from one source under two
# names so the test can hold two distinct RTLD_LOCAL images at once. Each
# compiles the production binding translation unit.
foreach(client_suffix a b)
    add_library(test_device_fault_client_${client_suffix} SHARED
        ${CMAKE_SOURCE_DIR}/common/support/device_fault_client.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/device_fault_monitor_binding.cpp
    )
    target_include_directories(test_device_fault_client_${client_suffix} PRIVATE
        ${SIMPLER_COMMON_PLATFORM_DIR}/include
        ${SIMPLER_SRC}/common
    )
    target_link_libraries(test_device_fault_client_${client_suffix} PRIVATE pthread)
endforeach()

platform_single_case(test_device_fault_cross_dso.cpp
    DEFINES
        TEST_DEVICE_FAULT_CLIENT_A_PATH="$<TARGET_FILE:test_device_fault_client_a>"
        TEST_DEVICE_FAULT_CLIENT_B_PATH="$<TARGET_FILE:test_device_fault_client_b>"
    LINK    ${CMAKE_DL_LIBS}
    DEPENDS test_device_fault_client_a test_device_fault_client_b)

# Context-lifetime AICore argument owner, and the publication helper beside it.
# Both pin where the uploaded prefix of a Runtime image stops: they assert it
# ends at offsetof(DeviceRuntimeLaunchDesc, teardown_gates), the device-
# initialized tail the host must not overwrite. Each arch reads that layout out
# of its own kernel_args.h, so the pair is built per arch.
#
# ARCHS a2a3 on the tmr halves: a5's tensormap_and_ringbuffer Runtime has no
# device-initialized tail at all — its header states extent equals copy size —
# so it declares no teardown_gates and the boundary these cases pin does not
# exist there. host_build_graph has the tail on both arches.
set(KERNEL_ARGS_INCLUDES ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/host ${SIMPLER_SRC}/common/worker)

tmr_case(test_kernel_persistent_args.cpp ARCHS a2a3
    SOURCES  ${KERNEL_PERSISTENT_ARGS_SRC}
             ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES ${KERNEL_ARGS_INCLUDES}
    DEFINES  SIMPLER_UT_TRB_RUNTIME)

hbg_case(test_kernel_persistent_args.cpp
    SOURCES  ${KERNEL_PERSISTENT_ARGS_SRC} ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES ${KERNEL_ARGS_INCLUDES})

# Compile the actual onboard helper/allocator without an SDK. support/kernel_args
# stands in for the CANN headers and must precede the gtest prefix, which on a
# machine with CANN installed carries acl/acl.h and runtime/rt.h of its own.
set(KERNEL_ARGS_HELPER_TEST_SOURCES
    ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/host/device_runner_helpers.cpp
    ${SIMPLER_SRC}/@arch@/platform/onboard/host/memory_allocator.cpp)

tmr_case(test_kernel_args_helper.cpp ARCHS a2a3
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES}
            ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES})

hbg_case(test_kernel_args_helper.cpp
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES} ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES})

# The three device lengths of a Runtime descriptor, answered by whichever
# runtime and arch the executable was compiled for. What it covers is the
# runtime trees, not this directory's: the symbols come from a bare
# "runtime.h", which src/common/host_build_graph/ and
# src/<arch>/runtime/tensormap_and_ringbuffer/runtime/ each answer with their
# own. Those two sit at different levels of src, so the pair has no shared
# directory to mirror, and the case stays beside the platform helper that
# publishes the prefixes it measures.
tmr_case(test_runtime_workers_boundary.cpp
    SOURCES ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    DEFINES SIMPLER_UT_TRB_RUNTIME)

hbg_case(test_runtime_workers_boundary.cpp
    SOURCES ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

# The publication side, and a case of this directory's own: it drives the real
# onboard helper and allocator, and takes the CANN stand-in its siblings above
# do. The Runtime it publishes supplies the prefixes, nothing more.
tmr_case(test_runtime_workers_publication.cpp
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES}
            ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES})

hbg_case(test_runtime_workers_publication.cpp
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES} ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES})

# Which route a run's entry values take, and that the route is chosen after
# prepare captured them. Same pairing as the publication case above, and for the
# same reason: one source, one build per runtime, each resolving "runtime.h" to
# its own. The trb build additionally covers the descriptor's wire offsets and
# the raw-byte decode, so it defines SIMPLER_UT_TRB_RUNTIME.
tmr_case(test_launch_entry_args.cpp
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES}
            ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES}
    DEFINES SIMPLER_UT_TRB_RUNTIME)

hbg_case(test_launch_entry_args.cpp
    SOURCES ${KERNEL_ARGS_HELPER_TEST_SOURCES} ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES_BEFORE ${CMAKE_SOURCE_DIR}/support/kernel_args ${KERNEL_ARGS_INCLUDES})

# A loadable stand-in for a compiled AICore child kernel. The sim
# registration writes each child's binary to a temp file, dlopens it and
# resolves kernel_entry, so a case wanting that path to succeed needs bytes
# that really are a shared object exporting that symbol. MODULE, because
# nothing links it — the case reads the file and hands the registration its
# contents.
add_library(test_fake_child_kernel MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/support/fake_child_kernel.cpp)
set_target_properties(test_fake_child_kernel PROPERTIES PREFIX "" POSITION_INDEPENDENT_CODE ON)

# Registration ownership and the per-run function-table reference, driven
# through the real sim runner pool: its upload, record, bind, unregister and
# release are production code here. Sim is the variant whose registration needs
# no SDK — its "upload" is a host scratch plus a real dlopen of each child — so
# this reaches src/common/platform/sim and the arch's own collectors, and takes
# the hbg Runtime because a bind installs the reference into that descriptor.
set(SIM_CALLABLE_POOL_SOURCES
    ${SIMPLER_COMMON_PLATFORM_DIR}/sim/host/device_runner_base.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/sim/host/memory_allocator.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/sim/sim_context/cpu_sim_context.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/args_dump_collector.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/chip_swimlane_collector.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/scope_stats_collector.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/host_phase_records.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/platform_compile_info.cpp
    ${SIMPLER_SRC}/@arch@/platform/shared/host/pmu_collector.cpp
    ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp
    ${SIMPLER_SRC}/common/platform_comm/comm_sim.cpp
    # support/unified_log_stubs.cpp already defines unified_log_*; only
    # HostLogger is missing, so this takes host_log.cpp for that alone.
    ${SIMPLER_SRC}/common/log/host_log.cpp)
set(SIM_CALLABLE_POOL_INCLUDES
    ${SIMPLER_COMMON_PLATFORM_DIR}/sim/host
    ${SIMPLER_COMMON_PLATFORM_DIR}/sim/sim_context
    ${SIMPLER_SRC}/common/worker)

hbg_case(test_sim_callable_pool.cpp
    SOURCES  ${SIM_CALLABLE_POOL_SOURCES} ${HBG_COMMON_SHARED_DIR}/runtime.cpp
    INCLUDES ${SIM_CALLABLE_POOL_INCLUDES}
    # chip_swimlane_collector.cpp refuses to compile without the runtime name,
    # and this case's runtime is fixed by the macro it is declared with.
    DEFINES
        SIMPLER_RUNTIME_NAME="host_build_graph"
        FAKE_KERNEL_SO_PATH="$<TARGET_FILE:test_fake_child_kernel>"
    LINK    ${CMAKE_DL_LIBS}
    DEPENDS test_fake_child_kernel)

# The same case against the other runtime: the registration-owned tables are a
# contract all four (arch, runtime) pairs carry, and the runner resolves its
# bare "runtime.h" to whichever Runtime it was built for. Declared twice rather
# than once with @runtime@ because the two runtimes keep their shared runtime.cpp
# at different levels of src, so there is no single path template — the same
# reason test_runtime_workers_publication above is a pair.
tmr_case(test_sim_callable_pool.cpp
    SOURCES
        ${SIM_CALLABLE_POOL_SOURCES}
        ${SIMPLER_SRC}/@arch@/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    INCLUDES ${SIM_CALLABLE_POOL_INCLUDES}
    DEFINES
        SIMPLER_RUNTIME_NAME="tensormap_and_ringbuffer"
        FAKE_KERNEL_SO_PATH="$<TARGET_FILE:test_fake_child_kernel>"
    LINK    ${CMAKE_DL_LIBS}
    DEPENDS test_fake_child_kernel)

# One source, three builds differing only in how capture is switched on: the
# compile-time flag, the environment variable, and both. A separate process
# verifies the private pre-bind fallback before another test target binds its
# singleton to process-owned state.
function(add_device_phase_capture_test name host_strace device_strace expected)
    simpler_ut_add_target(NAME ${name}
        SOURCES
            test_device_phase_capture.cpp
            ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/host/device_phase_capture.cpp
            ${SIMPLER_LOG_DIR}/host_log.cpp
        INCLUDES
            ${SIMPLER_COMMON_PLATFORM_DIR}/onboard/host
            ${SIMPLER_SRC}/common/task_interface
            ${SIMPLER_LOG_DIR}
            ${SIMPLER_LOG_DIR}/include
        DEFINES
            SIMPLER_HOST_STRACE=${host_strace}
            EXPECT_CAPTURE_ENABLED=${expected}
        LABEL no_hardware)
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "SIMPLER_DEVICE_STRACE_ENABLE=${device_strace}")
endfunction()

add_device_phase_capture_test(test_device_phase_capture 1 1 1)
add_device_phase_capture_test(test_device_phase_capture_env_off 1 0 0)
add_device_phase_capture_test(test_device_phase_capture_compile_off 0 1 0)

# ---------------------------------------------------------------------------
# Cases that repeat along an axis
# ---------------------------------------------------------------------------

platform_case(test_runtime_timeout_config.cpp PER_ARCH)

platform_case(test_pmu_collector.cpp PER_ARCH
    SOURCES
        ${SIMPLER_SRC}/@arch@/platform/shared/host/pmu_collector.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)

# a5 declares write_reg out-of-line with a 64-bit value where a2a3 has it
# inline; the test supplies the missing definition and needs to know which.
# There is no arch macro in the headers — the counter counts are constexpr,
# not #defines — so the build says it.
platform_case(test_pmu_run_identity.cpp PER_ARCH
    SOURCES
        ${SIMPLER_SRC}/@arch@/platform/shared/host/pmu_collector.cpp
        ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)
if(TARGET test_a5_pmu_run_identity)
    target_compile_definitions(test_a5_pmu_run_identity PRIVATE SIMPLER_TEST_PMU_ARCH_A5=1)
endif()

# Cross-run retention, off the same two sides: the real producer publishes into
# the real collector's threads, and the background writer merges and renames.
# Same arch discriminator as the case above, and for the same reason.
platform_case(test_pmu_retained_runs.cpp PER_ARCH
    SOURCES
        ${SIMPLER_SRC}/@arch@/platform/shared/host/pmu_collector.cpp
        ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)
if(TARGET test_a5_pmu_retained_runs)
    target_compile_definitions(test_a5_pmu_retained_runs PRIVATE SIMPLER_TEST_PMU_ARCH_A5=1)
endif()

# The two production boundaries the case above does not reach: a pending run's
# frozen event/column set while a successor resets the live configuration, and
# the default path's begin-before-start order. Same arch discriminator, same
# reason.
platform_case(test_pmu_frozen_config.cpp PER_ARCH
    SOURCES
        ${SIMPLER_SRC}/@arch@/platform/shared/host/pmu_collector.cpp
        ${SIMPLER_SRC}/@arch@/platform/shared/aicpu/pmu_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)
if(TARGET test_a5_pmu_frozen_config)
    target_compile_definitions(test_a5_pmu_frozen_config PRIVATE SIMPLER_TEST_PMU_ARCH_A5=1)
endif()

platform_case(test_args_dump_collector.cpp PER_ARCH
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/args_dump_collector.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)

platform_case(test_args_dump_run_identity.cpp PER_ARCH
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/args_dump_collector.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/args_dump_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)

# The two production sides every chip_swimlane case compiles — the host
# collector and the AICPU producer module — with the arch's sim regs and
# profiling copy under them. @arch@ expands per combination, in the factory.
# The six cases below all take this list; it is one list because they cover
# one pair of sides.
set(CHIP_SWIMLANE_CASE_SOURCES
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/chip_swimlane_collector.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/chip_swimlane_collector_aicpu.cpp
    ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
    ${SIMPLER_SRC}/@arch@/platform/sim/aicpu/inner_platform_regs.cpp
    ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)

# The collector pair carries both real sides: the host collector fills the bank
# and the AICPU producer module drives init -> complete_task -> flush, so a test
# can read the result back through the production host copy. The others read
# the retained per-run terminal snapshots, the AICore and handoff accounting,
# and the retained runs' receipt, seal, budget and control paths off those
# same two sides. Both axes: the swimlane record reaches the arch's
# platform_regs and scheduler_profiling as well as the runtime's task_id.
#
# The collector has a consumer outside ctest —
# tests/ut/py/test_swimlane_export_run_identity.py runs these binaries to feed
# the real Python parser real exporter bytes — and PYUT_EXPORT is what gives
# that consumer each build's path. That is a difference in who reads the
# target, not in the case's shape, which is why it stands alone: grouping
# asserts the cases are the same kind, and the other five are read by ctest
# alone.
platform_case(test_chip_swimlane_collector.cpp
    PER_ARCH PER_RUNTIME TIER aicpu
    PYUT_EXPORT chip_swimlane_collector
    SOURCES ${CHIP_SWIMLANE_CASE_SOURCES})

platform_cases(
    test_chip_swimlane_run_terminal.cpp
    test_chip_swimlane_run_terminal_consistency.cpp
    test_chip_swimlane_aicore_accounting.cpp
    test_chip_swimlane_handoff_accounting.cpp
    test_chip_swimlane_retained_runs.cpp
    PER_ARCH PER_RUNTIME TIER aicpu
    SOURCES ${CHIP_SWIMLANE_CASE_SOURCES})

# The exported run file, off the same two sides. Not PER_ARCH, and the reason
# is the case rather than what it covers: its golden is a captured artifact
# carrying that capture's own `clock_freq_hz` and `platform`, both of which the
# writer takes from the arch it was compiled for. Only a2a3 has been captured,
# so an a5 build compares a5's output against a2a3's file and fails on those
# two values.
#
# The writer itself is correct on a5 — the other five cases above cover it
# there, and an a5 build of this one differs from its golden in exactly those
# two fields and nothing else. Capturing an a5 golden is what this needs; until
# then a5 is uncovered here.
platform_case(test_chip_swimlane_run_export.cpp
    PER_RUNTIME TIER aicpu
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/chip_swimlane_collector.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/aicpu/inner_platform_regs.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/host/profiling_copy.cpp)

# The same two sides, reading the transport rather than the record: support/
# profiling_copy_fault.cpp defines the profiling_copy_* entry points the arch's
# sim copy supplies, and arms one device-to-host transfer to fail. Both are
# ordinary definitions, so this case takes that file in place of the production
# one its siblings above compile.
platform_case(test_chip_swimlane_run_terminal_transport.cpp
    PER_ARCH PER_RUNTIME TIER aicpu
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/support/profiling_copy_fault.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/chip_swimlane_collector.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/chip_swimlane_collector_aicpu.cpp
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/aicpu/device_run_result_base_aicpu.cpp
        ${SIMPLER_SRC}/@arch@/platform/sim/aicpu/inner_platform_regs.cpp
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/support)

platform_case(test_host_phase_records.cpp PER_ARCH PER_RUNTIME
    SOURCES
        ${SIMPLER_COMMON_PLATFORM_DIR}/shared/host/host_phase_records.cpp)

# BARE: this one compiles the arch's sim AICore header, which reaches
# "common/platform_config.h" — a name both arches define. The case's own arch
# must come before the shared directory, which the common contract cannot
# express, so the case names its three directories in order instead.
platform_case(test_chip_swimlane_aicore.cpp PER_ARCH PER_RUNTIME BARE
    INCLUDES
        ${SIMPLER_SRC}/@arch@/platform/sim/aicore
        ${SIMPLER_SRC}/@arch@/platform/include
        ${SIMPLER_COMMON_PLATFORM_DIR}/include
        # a2a3's inner_kernel.h reaches aicore_teardown.h; last, so it cannot
        # shadow anything the arch directories above own.
        ${SIMPLER_SRC}/common/task_interface)

# ---------------------------------------------------------------------------
# Cases reaching neither the arch platform tree nor any runtime's image, so one
# build is every build. They are named here rather than swept up by a default,
# because SINGLE is a judgement about what a case reaches — the same judgement
# PER_ARCH and PER_RUNTIME are — and check_ut_cpp_axis.py verifies it by
# measurement: a case listed here that does open one of those trees is a
# reported gap, not a silent single build.
# ---------------------------------------------------------------------------
simpler_ut_cases(platform_single_case
    test_acl_hal_device.cpp
    test_arena_replacement_transaction.cpp
    test_device_fault_monitor.cpp
    test_device_health_state.cpp
    test_execution_mode_latch.cpp
    test_file_marker_handshake.cpp
    test_host_api.cpp
    test_kernel_entry_validation.cpp
    test_queued_stream_waits.cpp
    test_retained_scheduler_storage.cpp
    test_retained_temp_bump.cpp
    test_run_completion_fence.cpp
    test_run_drain_decision.cpp
    test_run_outcome_decision.cpp
    test_run_shadow_consumer_evidence.cpp
    test_run_stream_pair.cpp
    test_teardown_recorder.cpp
    test_worker_chip_orch_comm.cpp
    test_workspace_manager.cpp
    test_workspace_staging.cpp)

# ---------------------------------------------------------------------------
# Both axes are open in this directory and nothing about a file says which of
# them apply, so a case here is declared or it is refused — there is no default
# to fall into. Everywhere else the directory fixes the answer and its glob can
# supply it; here the answer is the case's own, and a glob that chose SINGLE
# for an unrecognised file would hand an unjudged case the very single build
# the keyword exists to make deliberate.
# ---------------------------------------------------------------------------
macro(platform_case_without_axes src)
    message(FATAL_ERROR
        "common/platform/${src} is built by no declaration in this "
        "CMakeLists.txt.\n\n"
        "This directory's cases cover src/common/platform, which the product "
        "compiles once per arch and once per runtime, so how many times the "
        "case is built is a property of what it reaches and cannot be "
        "defaulted. Declare it with the axes it varies over:\n\n"
        "  platform_case(${src} PER_ARCH)            reaches src/<arch>/platform\n"
        "  platform_case(${src} PER_RUNTIME)         reaches a per-runtime image source\n"
        "  platform_case(${src} PER_ARCH PER_RUNTIME)\n"
        "  platform_single_case(${src})              reaches neither\n\n"
        "Build it, then confirm the choice with `g++ -E -H` per "
        "docs/testing/adding-a-cpp-unit-test.md — "
        "tests/lint/check_ut_cpp_axis.py checks it the same way.")
endmacro()
simpler_ut_glob_cases(platform_case_without_axes)
