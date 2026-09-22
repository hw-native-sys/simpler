# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/hierarchical/ — the L3+ host-side scheduler.
#
# This tree compiles into the nanobind extension rather than into any runtime
# image, so these targets are built once and carry no runtime in their name.

# ---------------------------------------------------------------------------
# Distributed runtime sources under test, bundled into an OBJECT library so
# they compile once and link into every hierarchical test.
# ---------------------------------------------------------------------------
set(HIERARCHICAL_SRC_DIR ${SIMPLER_SRC}/common/hierarchical)
set(WORKER_SRC_DIR ${SIMPLER_SRC}/common/worker)

add_library(hierarchical_objs OBJECT
    ${HIERARCHICAL_SRC_DIR}/types.cpp
    ${HIERARCHICAL_SRC_DIR}/tensormap.cpp
    ${HIERARCHICAL_SRC_DIR}/ring.cpp
    ${HIERARCHICAL_SRC_DIR}/scope.cpp
    ${HIERARCHICAL_SRC_DIR}/remote_wire.cpp
    ${HIERARCHICAL_SRC_DIR}/remote_endpoint.cpp
    ${HIERARCHICAL_SRC_DIR}/orchestrator.cpp
    ${HIERARCHICAL_SRC_DIR}/worker_manager.cpp
    ${HIERARCHICAL_SRC_DIR}/scheduler.cpp
    ${HIERARCHICAL_SRC_DIR}/worker.cpp
    ${WORKER_SRC_DIR}/chip_run_lane.cpp
    ${WORKER_SRC_DIR}/chip_worker.cpp
    # ChipWorker binds the process's device-fault monitor into every host
    # runtime it loads, so the owner that defines it comes along. In production
    # this pair is exactly the same: both are compiled into the module the
    # interpreter never unloads.
    ${WORKER_SRC_DIR}/device_fault_monitor_host.cpp
    # ChipWorker owns the process log state in production through the binding
    # DSO. These tests link ChipWorker directly, so provide the same logger
    # implementation while retaining their custom host-span capture adapter.
    ${SIMPLER_SRC}/common/log/host_log.cpp
    # Host-side assert_impl / get_stacktrace for the unified ChipTensor's
    # always_assert (init_external and the view ops). Mirrors the binding's CMake.
    # ODR invariant: assert_impl / get_stacktrace are ALSO defined in the runtime
    # orchestration/common.cpp. No target may link both that and this file, or
    # the linker sees duplicate symbols. This target links neither runtime
    # orchestration TU, so only this host-side definition is present.
    ${SIMPLER_SRC}/common/task_interface/assert_compat.cpp
)
target_include_directories(hierarchical_objs PUBLIC
    ${HIERARCHICAL_SRC_DIR}
    ${SIMPLER_SRC}/common/log/include
    ${SIMPLER_SRC}/common/platform/include
    ${SIMPLER_SRC}/common/task_interface
    ${WORKER_SRC_DIR}
)

# Every case here drives the L3+ scheduler, so it links the object library and
# the default host-span adapter. worker_manager.cpp uses shm_open/shm_unlink,
# which is librt on glibc < 2.34 and libSystem on macOS.
macro(hierarchical_case src)
    simpler_ut_plain_case(${src}
        SOURCES ${SIMPLER_SRC}/common/log/unified_log_host.cpp
        OBJS hierarchical_objs LABEL no_hardware ${ARGN})
    if(UNIX AND NOT APPLE)
        string(REGEX REPLACE "\\.cpp$" "" _hierarchical_case_name "${src}")
        target_link_libraries(${_hierarchical_case_name} PRIVATE rt)
    endif()
endmacro()

# Brings its own host-span capture adapter, so it takes no default one.
simpler_ut_plain_case(test_scheduler.cpp OBJS hierarchical_objs LABEL no_hardware)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_scheduler PRIVATE rt)
endif()

# Exercise ChipWorker's actual dlopen/dlsym admission before a sentinel factory,
# then the four built sim runtimes through the unchanged public kernel ABI.
# The production binding TU comes along because this library stands in for a
# host runtime, and ChipWorker binds the process's device-fault monitor into
# every host runtime it loads. Binding is mandatory, so a stand-in that did
# not export the binder would be rejected exactly as a stale build is.
add_library(pipeline_contract_runtime SHARED
    ${CMAKE_SOURCE_DIR}/support/pipeline_contract_runtime.cpp
    ${SIMPLER_SRC}/common/platform/shared/host/device_fault_monitor_binding.cpp
)
target_include_directories(pipeline_contract_runtime PRIVATE
    ${SIMPLER_SRC}/common
    ${SIMPLER_SRC}/common/log/include
    ${SIMPLER_SRC}/common/platform/include
)
hierarchical_case(test_pipeline_contract_loader.cpp
    DEPENDS pipeline_contract_runtime
    DEFINES
        PIPELINE_FIXTURE_PATH="$<TARGET_FILE:pipeline_contract_runtime>"
        SIM_CONTEXT_PATH="${SIMPLER_REPO_ROOT}/build/lib/libcpu_sim_context.so"
        RUNTIME_LIBRARY_ROOT="${SIMPLER_REPO_ROOT}/build/lib")

simpler_ut_glob_cases(hierarchical_case)
