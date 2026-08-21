# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# C++ unit tests for src/common/hierarchical using GoogleTest.
# Run with: cmake --build . --target run_ut_cpp
# Or directly: ctest --test-dir build/ut_cpp

cmake_minimum_required(VERSION 3.15)
project(hierarchical_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Gate for tests that may need CANN headers/libs at build time or that only
# make sense on Ascend hardware runners.  Default OFF so no-hw runners
# (ubuntu-latest / macos-latest, no CANN installed) can build the tree
# without touching anything CANN-adjacent.  CI's hw jobs
# (ut-a2a3 / ut-a5) pass -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.
#
# Rule of thumb: any new test that either
#   (a) links against CANN libs (libascendcl, libhcomm, libruntime), or
#   (b) needs to dlopen an onboard libhost_runtime.so to be meaningful,
# goes under `if(SIMPLER_ENABLE_HARDWARE_TESTS)` below.
option(SIMPLER_ENABLE_HARDWARE_TESTS
       "Build tests that require Ascend hardware toolchain to build or run"
       OFF)

# ---------------------------------------------------------------------------
# GoogleTest — fast path uses the system install (apt libgtest-dev / brew
# googletest / yum-distributed /usr/local). Fall back to FetchContent on
# runners that don't preinstall it (currently the a2a3 self-hosted runner).
# Either way the rest of the file sees GTEST_LIB / GTEST_MAIN_LIB /
# GTEST_INCLUDE_DIRS with a consistent meaning.
#
# Both the system install and FetchContent build gtest with the compiler
# default libstdc++ ABI (cxx11), matching the tests in this directory. If
# you hand-compile gtest with `-D_GLIBCXX_USE_CXX11_ABI=0` and drop it in
# `/usr/local/lib*`, find_library will pick it up and link will fail with
# `undefined reference to testing::internal::EqFailure(...)` — see
# docs/troubleshooting/ut-cpp-gtest-abi.md for the workaround.
# ---------------------------------------------------------------------------
if(APPLE)
    set(GTEST_SEARCH_PATHS /opt/homebrew/lib /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /opt/homebrew/include /usr/local/include)
else()
    set(GTEST_SEARCH_PATHS /usr/local/lib64 /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /usr/local/include)
endif()

find_library(GTEST_LIB      gtest      PATHS ${GTEST_SEARCH_PATHS})
find_library(GTEST_MAIN_LIB gtest_main PATHS ${GTEST_SEARCH_PATHS})

if(NOT GTEST_LIB OR NOT GTEST_MAIN_LIB)
    message(STATUS "System GoogleTest not found — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    # FetchContent exposes the gtest / gtest_main targets directly; the
    # per-test functions below use these names uniformly with the system
    # path (where they resolve to the absolute .so/.a path via find_library).
    set(GTEST_LIB gtest)
    set(GTEST_MAIN_LIB gtest_main)
    set(GTEST_INCLUDE_DIRS "")  # include dirs are carried by the gtest target
endif()

# ---------------------------------------------------------------------------
# A2A3 runtime sources and stubs for ring-buffer / tensormap tests.
#
# Bundled into an OBJECT library so the runtime .cpp files compile once and
# the resulting .o files are reused across every test executable. PUBLIC
# include dirs / compile options propagate to consumers via
# target_link_libraries, so test executables get the same flags they would
# have had if the sources were listed directly.
# ---------------------------------------------------------------------------
set(A2A3_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime)

add_library(a2a3_rt_objs OBJECT
    ${A2A3_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A2A3_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A2A3_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a2a3_rt_objs PUBLIC
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)

function(add_a2a3_runtime_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        a2a3_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a2a3_hbg_runtime_test name src)
    add_executable(${name}
        ${src}
        ${ARGN}
        ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/common
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a5_hbg_runtime_test name src)
    add_executable(${name}
        ${src}
        ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a2a3_orchestrator_runtime_test name src)
    add_executable(${name}
        ${src}
        ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime
    )
    target_link_libraries(${name} PRIVATE
        a2a3_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

# ---------------------------------------------------------------------------
# Distributed runtime sources under test, bundled into an OBJECT library so
# they compile once and link into every hierarchical test.
# ---------------------------------------------------------------------------
set(HIERARCHICAL_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/hierarchical)
set(WORKER_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/worker)

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
    # ChipWorker owns the process log state in production through the binding
    # DSO. These tests link ChipWorker directly, so provide the same logger
    # implementation while retaining their custom host-span capture adapter.
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/host_log.cpp
    # Host-side assert_impl / get_stacktrace for the unified ChipTensor's
    # always_assert (init_external and the view ops). Mirrors the binding's CMake.
    # ODR invariant: assert_impl / get_stacktrace are ALSO defined in the runtime
    # orchestration/common.cpp. No target may link both that and this file, or
    # the linker sees duplicate symbols. This target links neither runtime
    # orchestration TU, so only this host-side definition is present.
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface/assert_compat.cpp
)
target_include_directories(hierarchical_objs PUBLIC
    ${HIERARCHICAL_SRC_DIR}
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${WORKER_SRC_DIR}
)

function(add_hierarchical_test name src)
    set(options CUSTOM_HOST_SPAN_ADAPTER)
    cmake_parse_arguments(TEST "${options}" "" "" ${ARGN})
    if(TEST_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown add_hierarchical_test arguments: ${TEST_UNPARSED_ARGUMENTS}")
    endif()
    add_executable(${name} ${src})
    if(NOT TEST_CUSTOM_HOST_SPAN_ADAPTER)
        target_sources(${name} PRIVATE
            ${CMAKE_SOURCE_DIR}/../../../src/common/log/unified_log_host.cpp)
    endif()
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        hierarchical_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    # worker_manager.cpp uses shm_open/shm_unlink; librt on glibc < 2.34, libSystem on macOS.
    if(UNIX AND NOT APPLE)
        target_link_libraries(${name} PRIVATE rt)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a2a3_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        # src/common is needed so `#include "utils/device_arena.h"` resolves.
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a5_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        # src/common is needed so `#include "utils/device_arena.h"` resolves.
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# A5 runtime sources, mirroring a2a3_rt_objs. Bundled into an OBJECT library
# so the runtime .cpp files compile once and the resulting .o files are
# reused across every a5 runtime test executable.
# ---------------------------------------------------------------------------
set(A5_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime)

add_library(a5_rt_objs OBJECT
    ${A5_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${A5_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A5_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${A5_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A5_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a5_rt_objs PUBLIC
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)

function(add_a5_runtime_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        a5_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_a5_orchestrator_runtime_test name src)
    add_executable(${name}
        ${src}
        ${A5_RUNTIME_DIR}/pto_orchestrator.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime
    )
    target_link_libraries(${name} PRIVATE
        a5_rt_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

function(add_task_interface_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_common_utils_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        # src/common is needed so utils/elf_build_id.h, utils/device_arena.h,
        # etc. resolve via the "utils/..." prefix.
        ${CMAKE_SOURCE_DIR}/../../../src/common
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

enable_testing()

# ---------------------------------------------------------------------------
# Hierarchical runtime tests (src/common/hierarchical/)
# ---------------------------------------------------------------------------
add_hierarchical_test(test_tensormap  hierarchical/test_tensormap.cpp)
add_hierarchical_test(test_ring  hierarchical/test_ring.cpp)
add_hierarchical_test(test_scope      hierarchical/test_scope.cpp)
add_hierarchical_test(test_orchestrator hierarchical/test_orchestrator.cpp)
add_hierarchical_test(test_scheduler hierarchical/test_scheduler.cpp CUSTOM_HOST_SPAN_ADAPTER)
add_hierarchical_test(test_chip_run_lane hierarchical/test_chip_run_lane.cpp)
add_hierarchical_test(test_remote_wire hierarchical/test_remote_wire.cpp)
add_hierarchical_test(test_remote_endpoint hierarchical/test_remote_endpoint.cpp)
add_hierarchical_test(test_pipeline_contract hierarchical/test_pipeline_contract.cpp)

# Run-stream pair state machine: publication-aware AICore stream reuse,
# submitter-gated retirement, and failed-destroy ownership. Header-only with
# injected create/destroy, so it needs no runtime objects and no device.
add_executable(test_run_stream_pair hierarchical/test_run_stream_pair.cpp)
target_include_directories(test_run_stream_pair PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/worker
)
target_link_libraries(test_run_stream_pair PRIVATE ${GTEST_MAIN_LIB} ${GTEST_LIB} pthread)
add_test(NAME test_run_stream_pair COMMAND test_run_stream_pair)
set_tests_properties(test_run_stream_pair PROPERTIES LABELS "no_hardware")

add_executable(test_native_run_execution common/test_native_run_execution.cpp)
target_include_directories(test_native_run_execution PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/worker
)
target_link_libraries(test_native_run_execution PRIVATE ${GTEST_MAIN_LIB} ${GTEST_LIB} pthread)
add_test(NAME test_native_run_execution COMMAND test_native_run_execution)
set_tests_properties(test_native_run_execution PROPERTIES LABELS "no_hardware")

# NativeRunContext::publish_acceptance is the only path that marks a task
# accepted, and it is keyed on the launch receipt's identity.
add_executable(test_native_run_acceptance
    common/test_native_run_acceptance.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime/shared/runtime.cpp
)
target_include_directories(test_native_run_acceptance PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/worker
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/aicpu
)
target_link_libraries(test_native_run_acceptance PRIVATE ${GTEST_MAIN_LIB} ${GTEST_LIB} pthread)
add_test(NAME test_native_run_acceptance COMMAND test_native_run_acceptance)
set_tests_properties(test_native_run_acceptance PROPERTIES LABELS "no_hardware")
# A HostApi value keeps one runner and one run's resource selection even when
# callers invoke it from different host threads.
add_common_utils_test(test_host_api common/test_host_api.cpp)

# Sim kernel threads publish one sticky completion state. A gated unit test
# proves submission and completion are separate without relying on kernel
# duration or host scheduling.
add_executable(test_sim_run_completion common/test_sim_run_completion.cpp)
target_include_directories(test_sim_run_completion PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/host
    ${CMAKE_SOURCE_DIR}/../../../src/common/worker
)
target_link_libraries(test_sim_run_completion PRIVATE ${GTEST_MAIN_LIB} ${GTEST_LIB} pthread)
add_test(NAME test_sim_run_completion COMMAND test_sim_run_completion)
set_tests_properties(test_sim_run_completion PROPERTIES LABELS "no_hardware")

# ---------------------------------------------------------------------------
# Types / task_interface tests (src/common/task_interface/)
# ---------------------------------------------------------------------------
add_task_interface_test(test_buffer types/test_buffer.cpp)
add_task_interface_test(test_child_memory types/test_child_memory.cpp)
add_task_interface_test(test_chip_max_tensor_args types/test_chip_max_tensor_args.cpp)
add_task_interface_test(test_call_config types/test_call_config.cpp)

# Contract test for upload_chip_callable_buffer caller-buffer immutability.
# Pulls both task_interface (callable.h) and src/common (utils/fnv1a_64.h),
# so it can't use add_task_interface_test as-is.
add_executable(test_chip_callable_upload_immutable
    types/test_chip_callable_upload_immutable.cpp
)
target_include_directories(test_chip_callable_upload_immutable PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
)
target_link_libraries(test_chip_callable_upload_immutable PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_chip_callable_upload_immutable COMMAND test_chip_callable_upload_immutable)
set_tests_properties(test_chip_callable_upload_immutable PROPERTIES LABELS "no_hardware")

# ---------------------------------------------------------------------------
# Common utilities tests (src/common/utils/)
# ---------------------------------------------------------------------------
add_common_utils_test(test_elf_build_id common/test_elf_build_id.cpp)
add_common_utils_test(test_runtime_orch_so common/test_runtime_orch_so.cpp)
# RegisterCallableArgs lives in the arch platform headers (identical across
# arches); pull in a2a3's so `common/kernel_args.h` resolves for this no-hardware
# POD-shape test.
target_include_directories(test_runtime_orch_so PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
)
add_common_utils_test(test_device_arena common/test_device_arena.cpp)
add_common_utils_test(test_thread_completion_gate common/test_thread_completion_gate.cpp)
add_common_utils_test(test_fatal_shutdown_latch common/test_fatal_shutdown_latch.cpp)
add_common_utils_test(test_buffer_pool_manager common/test_buffer_pool_manager.cpp)
target_sources(test_buffer_pool_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/host_log.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/unified_log_host.cpp
)
target_include_directories(test_buffer_pool_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
)
# MemoryAllocator byte-counter test: links the SIM impl (std::malloc/free,
# no CANN) the same way test_buffer_pool_manager pulls unified_log.
add_common_utils_test(test_memory_allocator common/test_memory_allocator.cpp)
target_sources(test_memory_allocator PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/host/memory_allocator.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/host_log.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/unified_log_host.cpp
)
target_include_directories(test_memory_allocator PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
)
add_common_utils_test(test_profiler_base common/test_profiler_base.cpp)
target_sources(test_profiler_base PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/host_log.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/unified_log_host.cpp
)
target_include_directories(test_profiler_base PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
)

add_common_utils_test(test_profiler_device_engine common/test_profiler_device_engine.cpp)
target_sources(test_profiler_device_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(test_profiler_device_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
)

add_common_utils_test(test_worker_chip_orch_comm common/test_worker_chip_orch_comm.cpp)
add_common_utils_test(test_file_marker_handshake common/test_file_marker_handshake.cpp)

add_executable(test_trb_runtime_temp_buffer
    common/test_trb_runtime_temp_buffer.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/platform_compile_info.cpp
)
target_compile_definitions(test_trb_runtime_temp_buffer PRIVATE SIMPLER_PLATFORM_NAME="a2a3sim")
target_include_directories(test_trb_runtime_temp_buffer PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/host
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_trb_runtime_temp_buffer PRIVATE
    a2a3_rt_objs
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_trb_runtime_temp_buffer COMMAND test_trb_runtime_temp_buffer)
set_tests_properties(test_trb_runtime_temp_buffer PROPERTIES LABELS "no_hardware")

add_executable(test_a5_trb_runtime_temp_buffer
    common/test_trb_runtime_temp_buffer.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/platform_compile_info.cpp
)
target_compile_definitions(test_a5_trb_runtime_temp_buffer PRIVATE SIMPLER_PLATFORM_NAME="a5sim")
target_include_directories(test_a5_trb_runtime_temp_buffer PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/host
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_a5_trb_runtime_temp_buffer PRIVATE
    a5_rt_objs
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_a5_trb_runtime_temp_buffer COMMAND test_a5_trb_runtime_temp_buffer)
set_tests_properties(test_a5_trb_runtime_temp_buffer PROPERTIES LABELS "no_hardware")

add_executable(test_worker_chip_message_queue
    common/test_worker_chip_message_queue.cpp
    stubs/test_stubs.cpp
)
target_include_directories(test_worker_chip_message_queue PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
)
target_link_libraries(test_worker_chip_message_queue PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_worker_chip_message_queue COMMAND test_worker_chip_message_queue)
set_tests_properties(test_worker_chip_message_queue PROPERTIES LABELS "no_hardware")

add_executable(test_worker_chip_orch_endpoint
    common/test_worker_chip_orch_endpoint.cpp
    stubs/test_stubs.cpp
)
target_include_directories(test_worker_chip_orch_endpoint PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
)
target_link_libraries(test_worker_chip_orch_endpoint PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_worker_chip_orch_endpoint COMMAND test_worker_chip_orch_endpoint)
set_tests_properties(test_worker_chip_orch_endpoint PROPERTIES LABELS "no_hardware")

add_common_utils_test(test_runtime_timeout_config common/test_runtime_timeout_config.cpp)
target_include_directories(test_runtime_timeout_config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/a2a3/platform/include
)
add_common_utils_test(test_runtime_timeout_config_a5 common/test_runtime_timeout_config.cpp)
target_include_directories(test_runtime_timeout_config_a5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/a5/platform/include
)
# The PTO2_ERROR_* codes the name table switches on live in the runtime's status header;
# a2a3's and a5's copies are identical, so compiling against one covers both.
add_common_utils_test(test_error_code_names common/test_error_code_names.cpp)
target_include_directories(test_error_code_names PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
)

add_executable(test_scope_stats_collector
    common/test_scope_stats_collector.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/scope_stats_collector.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/sim/host/profiling_copy.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(test_scope_stats_collector PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_scope_stats_collector PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_scope_stats_collector COMMAND test_scope_stats_collector)
set_tests_properties(test_scope_stats_collector PROPERTIES LABELS "no_hardware")

add_executable(test_host_phase_records
    common/test_host_phase_records.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/host_phase_records.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(test_host_phase_records PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_host_phase_records PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_host_phase_records COMMAND test_host_phase_records)
set_tests_properties(test_host_phase_records PROPERTIES LABELS "no_hardware")

add_executable(test_chip_swimlane_aicore
    common/test_chip_swimlane_aicore.cpp
)
target_include_directories(test_chip_swimlane_aicore PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/sim/aicore
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
)
target_link_libraries(test_chip_swimlane_aicore PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_chip_swimlane_aicore COMMAND test_chip_swimlane_aicore)
set_tests_properties(test_chip_swimlane_aicore PROPERTIES LABELS "no_hardware")

# AICPU-side dep_gen writer. The shared-memory layout is byte-identical across
# platforms, so compiling against a2a3's platform_config covers a5 too.
add_executable(test_dep_gen_collector_aicpu
    common/test_dep_gen_collector_aicpu.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/dep_gen_collector_aicpu.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(test_dep_gen_collector_aicpu PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    ${CMAKE_SOURCE_DIR}/../../../src/common
)
target_link_libraries(test_dep_gen_collector_aicpu PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_dep_gen_collector_aicpu COMMAND test_dep_gen_collector_aicpu)
set_tests_properties(test_dep_gen_collector_aicpu PROPERTIES LABELS "no_hardware")

function(add_pmu_collector_test name platform)
    add_executable(${name}
        common/test_pmu_collector.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/${platform}/platform/shared/host/pmu_collector.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/${platform}/platform/sim/host/profiling_copy.cpp
        ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/${platform}/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_pmu_collector_test(test_pmu_collector_a2a3 a2a3)
add_pmu_collector_test(test_pmu_collector_a5 a5)

function(add_args_dump_collector_test name platform)
    add_executable(${name}
        common/test_args_dump_collector.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/args_dump_collector.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/${platform}/platform/sim/host/profiling_copy.cpp
        ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/${platform}/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
        ${CMAKE_SOURCE_DIR}/../../../src/common
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

add_args_dump_collector_test(test_args_dump_collector_a2a3 a2a3)
add_args_dump_collector_test(test_args_dump_collector_a5 a5)

# Per-callable_id orch SO file naming regression (see rtStreamSynchronize
# 507018 root cause). Compiles the a2a3 onboard `create_orch_so_file`
# against the test source so it runs on no-hw runners too.
add_executable(test_orch_so_file
    common/test_orch_so_file.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/onboard/aicpu/orch_so_file.cpp
)
target_include_directories(test_orch_so_file PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/include
)
target_link_libraries(test_orch_so_file PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_orch_so_file COMMAND test_orch_so_file)
set_tests_properties(test_orch_so_file PROPERTIES LABELS "no_hardware")

# ---------------------------------------------------------------------------
# A2A3 tests (src/a2a3/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a2a3_test(test_a2a3_fatal a2a3/test_a2a3_fatal.cpp)
add_a2a3_test(test_task_timing_slots a2a3/test_task_timing_slots.cpp)
add_a2a3_test(test_acl_hal_device common/test_acl_hal_device.cpp)

# PTO2 runtime-linked tests
add_a2a3_runtime_test(test_task_allocator   a2a3/test_task_allocator.cpp)
add_a2a3_runtime_test(test_scope_deadlock_detection common/test_scope_deadlock_detection.cpp)
set(HBG_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime)
set(A5_HBG_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime)
add_a2a3_hbg_runtime_test(test_hbg_single_ring_layout common/test_hbg_single_ring_layout.cpp)
target_sources(test_hbg_single_ring_layout PRIVATE
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_single_ring_layout common/test_hbg_single_ring_layout.cpp)
target_sources(test_a5_hbg_single_ring_layout PRIVATE
    ${A5_HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp)
add_a2a3_hbg_runtime_test(test_hbg_task_allocator a2a3/test_hbg_task_allocator.cpp)
add_a2a3_hbg_runtime_test(test_hbg_tensormap a2a3/test_hbg_tensormap.cpp)
add_a2a3_hbg_runtime_test(test_hbg_dep_gen_host_graph a2a3/test_dep_gen_host_graph.cpp)
add_a2a3_hbg_runtime_test(test_hbg_tensor_access a2a3/test_hbg_tensor_access.cpp)

# The func_id -> CoreCallable dispatch table holds exactly the active callable's
# mappings. An entry that outlives the ChipCallable buffer it points into is
# dereferenced by the scheduler and called by the AICore (issue #1489), so this
# links the Runtime implementation to pin the contract without a device.
add_a2a3_hbg_runtime_test(test_dispatch_table a2a3/test_dispatch_table.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp)
add_a2a3_hbg_runtime_test(test_hbg_core_tracker common/test_hbg_core_tracker.cpp)
# The seed test calls the real ready_queue_init_data_from_layout, so it links the
# shared init TU (and the orchestrator/SM members that TU refers to).
add_a2a3_hbg_runtime_test(test_hbg_ready_queue_seed common/test_hbg_ready_queue_seed.cpp)
add_a2a3_hbg_runtime_test(test_hbg_mailbox_init common/test_hbg_mailbox_init.cpp)
add_a2a3_hbg_runtime_test(test_hbg_self_relative_ptr common/test_hbg_self_relative_ptr.cpp)
add_a2a3_hbg_runtime_test(test_hbg_sm_compaction common/test_hbg_sm_compaction.cpp)
target_sources(test_hbg_ready_queue_seed PRIVATE
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
# The submit-poison test drives the real orchestrator submit path, so it links the
# orchestrator + shared runtime out-of-line members (mirrors a2a3_rt_objs for hbg).
add_a2a3_hbg_runtime_test(test_hbg_submit_poison a2a3/test_hbg_submit_poison.cpp)
target_sources(test_hbg_submit_poison PRIVATE
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a2a3_hbg_runtime_test(test_hbg_graph_submit_failure common/test_hbg_graph_submit_failure.cpp)
target_sources(test_hbg_graph_submit_failure PRIVATE
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a2a3_hbg_runtime_test(test_hbg_graph_async_submit common/test_hbg_graph_async_submit.cpp)
# Drives the real orchestrator submit and Graph placement paths, so it links the
# same out-of-line members as the graph-submit test.
add_a2a3_hbg_runtime_test(test_hbg_slot_claim common/test_hbg_slot_claim.cpp)
target_sources(test_hbg_slot_claim PRIVATE
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a5_hbg_runtime_test(test_a5_hbg_graph_submit_failure common/test_hbg_graph_submit_failure.cpp)
target_sources(test_a5_hbg_graph_submit_failure PRIVATE
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a5_hbg_runtime_test(test_a5_hbg_graph_async_submit common/test_hbg_graph_async_submit.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_slot_claim common/test_hbg_slot_claim.cpp)
target_sources(test_a5_hbg_slot_claim PRIVATE
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a5_hbg_runtime_test(test_a5_hbg_core_tracker common/test_hbg_core_tracker.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_ready_queue_seed common/test_hbg_ready_queue_seed.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_mailbox_init common/test_hbg_mailbox_init.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_self_relative_ptr common/test_hbg_self_relative_ptr.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_sm_compaction common/test_hbg_sm_compaction.cpp)
target_sources(test_a5_hbg_ready_queue_seed PRIVATE
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${A5_HBG_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${A5_HBG_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a5_hbg_runtime_test(test_a5_hbg_aicore_completion_mailbox a5/test_aicore_completion_mailbox.cpp)
set(HBG_A5_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime)
# The submit-poison test drives the real orchestrator submit path, so it links the
# orchestrator + shared runtime out-of-line members (mirrors a2a3's test_hbg_submit_poison).
add_a5_hbg_runtime_test(test_a5_hbg_submit_poison a5/test_hbg_submit_poison.cpp)
target_sources(test_a5_hbg_submit_poison PRIVATE
    ${HBG_A5_RUNTIME_DIR}/orchestrator_core/pto_orchestrator.cpp
    ${HBG_A5_RUNTIME_DIR}/orchestrator_core/pto_ring_buffer.cpp
    ${HBG_A5_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_A5_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_A5_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_A5_RUNTIME_DIR}/shared/runtime.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
)
add_a2a3_hbg_runtime_test(test_graph_cache common/test_hbg_graph_cache.cpp)
target_sources(test_graph_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/host_build_graph/graph_execution.cpp
)
add_a5_hbg_runtime_test(test_a5_graph_cache common/test_hbg_graph_cache.cpp)
target_sources(test_a5_graph_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/common/host_build_graph/graph_execution.cpp
)
# Incremental-activation tests init a real PTO2SchedulerState (ready queues), so
# they link the scheduler + shared runtime out-of-line members.
add_a2a3_hbg_runtime_test(test_graph_activation a2a3/test_graph_activation.cpp)
target_sources(test_graph_activation PRIVATE
    ${HBG_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${HBG_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${HBG_RUNTIME_DIR}/shared/runtime.cpp
)
add_a5_hbg_runtime_test(test_a5_graph_activation a5/test_graph_activation.cpp)
target_sources(test_a5_graph_activation PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime/shared/pto_shared_memory.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime/shared/pto_tensormap.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime/shared/pto_runtime2_init.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/host_build_graph/runtime/shared/runtime.cpp
)
# PTO2TensorMap's out-of-line members (reserve_layout / init / valid_count) live
# in this .cpp; no other add_a2a3_hbg_runtime_test target needs them.
target_sources(test_hbg_tensormap PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime/shared/pto_tensormap.cpp
)
target_sources(test_hbg_dep_gen_host_graph PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/host/dep_gen_host_graph.cpp
)
target_sources(test_hbg_tensor_access PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/host/host_tensor_access.cpp
)
add_a2a3_runtime_test(test_dep_list_pool    a2a3/test_dep_list_pool.cpp)
add_a2a3_runtime_test(test_scheduler_state  a2a3/test_scheduler_state.cpp)
add_a2a3_runtime_test(test_task_state       a2a3/test_task_state.cpp)
add_a2a3_runtime_test(test_ready_queue      a2a3/test_ready_queue.cpp)
add_a2a3_runtime_test(test_shared_memory    a2a3/test_shared_memory.cpp)
add_a2a3_runtime_test(test_a2a3_tensormap   a2a3/test_tensormap.cpp)
add_a2a3_runtime_test(test_fanin_pool       a2a3/test_fanin_pool.cpp)
add_a2a3_orchestrator_runtime_test(test_a2a3_orchestrator_fanin a2a3/test_orchestrator_fanin.cpp)
add_a2a3_orchestrator_runtime_test(test_wiring a2a3/test_wiring.cpp)
add_a2a3_orchestrator_runtime_test(test_a2a3_args_dump a2a3/test_args_dump.cpp)
add_a2a3_runtime_test(test_aicore_completion_mailbox a2a3/test_aicore_completion_mailbox.cpp)
add_a2a3_runtime_test(test_a2a3_scope_stats_collector a2a3/test_scope_stats_collector.cpp)

# ---------------------------------------------------------------------------
# A5 tests (src/a5/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a5_test(test_a5_fatal a5/test_a5_fatal.cpp)

# A5 trb runtime UTs — mirror of a2a3 trb runtime UTs, link against a5_rt_objs.
# Target names carry the a5_ prefix because hierarchical/test_tensormap (and
# the unprefixed a2a3 runtime targets test_scheduler_state / test_ready_queue
# / ...) already own those bare names.
add_a5_runtime_test(test_a5_task_allocator   a5/test_task_allocator.cpp)
add_a5_runtime_test(test_a5_scope_deadlock_detection common/test_scope_deadlock_detection.cpp)
add_a5_runtime_test(test_a5_dep_list_pool    a5/test_dep_list_pool.cpp)
add_a5_runtime_test(test_a5_scheduler_state  a5/test_scheduler_state.cpp)
add_a5_runtime_test(test_a5_task_state       a5/test_task_state.cpp)
add_a5_runtime_test(test_a5_ready_queue      a5/test_ready_queue.cpp)
add_a5_runtime_test(test_a5_shared_memory    a5/test_shared_memory.cpp)
add_a5_runtime_test(test_a5_tensormap        a5/test_tensormap.cpp)
add_a5_runtime_test(test_a5_fanin_pool       a5/test_fanin_pool.cpp)
add_a5_orchestrator_runtime_test(test_a5_orchestrator_fanin a5/test_orchestrator_fanin.cpp)
add_a5_orchestrator_runtime_test(test_a5_wiring a5/test_wiring.cpp)
add_a5_orchestrator_runtime_test(test_a5_args_dump a5/test_args_dump.cpp)
add_a5_runtime_test(test_a5_aicore_completion_mailbox a5/test_aicore_completion_mailbox.cpp)

# Host logger silent/off behavior — no runtime deps, just compile the same
# self-contained host logger sources used by production consumers.
set(SIMPLER_LOG_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/log)
set(HOST_LOG_TEST_SOURCES
    ${SIMPLER_LOG_DIR}/host_log.cpp
    ${SIMPLER_LOG_DIR}/unified_log_host.cpp
)
add_executable(test_host_log_off
    a5/test_host_log_off.cpp
    ${HOST_LOG_TEST_SOURCES}
)
target_include_directories(test_host_log_off PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${SIMPLER_LOG_DIR}
    ${SIMPLER_LOG_DIR}/include
)
target_link_libraries(test_host_log_off PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_host_log_off COMMAND test_host_log_off)

# A separate process verifies the private pre-bind fallback before another
# test target binds its singleton to process-owned state.
add_executable(test_host_log_unbound
    common/test_host_log_unbound.cpp
    ${HOST_LOG_TEST_SOURCES}
)
target_include_directories(test_host_log_unbound PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${SIMPLER_LOG_DIR}
    ${SIMPLER_LOG_DIR}/include
)
target_link_libraries(test_host_log_unbound PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_host_log_unbound COMMAND test_host_log_unbound)

set(COMMON_PLATFORM_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/platform)
function(add_device_phase_capture_test name host_strace device_strace expected)
    add_executable(${name}
        common/test_device_phase_capture.cpp
        ${COMMON_PLATFORM_DIR}/onboard/host/device_phase_capture.cpp
        ${SIMPLER_LOG_DIR}/host_log.cpp
    )
    target_compile_definitions(${name} PRIVATE
        SIMPLER_HOST_STRACE=${host_strace}
        EXPECT_CAPTURE_ENABLED=${expected}
    )
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${COMMON_PLATFORM_DIR}/onboard/host
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${SIMPLER_LOG_DIR}
        ${SIMPLER_LOG_DIR}/include
    )
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "SIMPLER_DEVICE_STRACE_ENABLE=${device_strace}"
        LABELS "no_hardware"
    )
endfunction()

add_device_phase_capture_test(test_device_phase_capture 1 1 1)
add_device_phase_capture_test(test_device_phase_capture_env_off 1 0 0)
add_device_phase_capture_test(test_device_phase_capture_compile_off 0 1 0)

# Onboard uses CANN for DEBUG/INFO/WARN/ERROR but keeps TIMING on simpler's
# finer threshold. The dlog stub makes the CANN levels available without SDK
# headers or hardware.
add_executable(test_onboard_device_log_level
    common/test_onboard_device_log_level.cpp
    ${COMMON_PLATFORM_DIR}/onboard/aicpu/device_log.cpp
)
target_include_directories(test_onboard_device_log_level PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/stubs
    ${COMMON_PLATFORM_DIR}/include
    ${SIMPLER_LOG_DIR}/include
)
target_link_libraries(test_onboard_device_log_level PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_onboard_device_log_level COMMAND test_onboard_device_log_level)
set_tests_properties(test_onboard_device_log_level PROPERTIES LABELS "no_hardware")

# Sim device-log record atomicity: concurrent threads and forked workers must
# not interleave partial records on the shared stderr. Pure host code — compile
# the sim backend alongside the test; no CANN, no hardware.
add_executable(test_sim_device_log
    common/test_sim_device_log.cpp
    ${COMMON_PLATFORM_DIR}/sim/aicpu/device_log.cpp
)
target_include_directories(test_sim_device_log PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${COMMON_PLATFORM_DIR}/include
    ${SIMPLER_LOG_DIR}/include
)
target_link_libraries(test_sim_device_log PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_sim_device_log COMMAND test_sim_device_log)
set_tests_properties(test_sim_device_log PROPERTIES LABELS "no_hardware")

# a2a3 host-side AICPU affinity selection (compute_allowed_cpus). Pure logic —
# no CANN headers, no hardware: compile the probe .cpp alongside the test. The
# test provides its own no-op logger stubs, so no logger sources are linked.
# The dlopen/dlsym HAL path is never exercised by these tests (only
# compute_allowed_cpus is), but the .cpp references libdl symbols, so link it.
# Regression barrier for issue #1045 (all survivors in one cluster).
set(A2A3_ONBOARD_HOST_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/onboard/host)
set(SIMPLER_COMMON_PLATFORM_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/platform)
add_executable(test_a2a3_aicpu_affinity_select
    a2a3/test_aicpu_affinity_select.cpp
    ${A2A3_ONBOARD_HOST_DIR}/aicpu_topology_probe.cpp
)
target_include_directories(test_a2a3_aicpu_affinity_select PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${A2A3_ONBOARD_HOST_DIR}
    ${SIMPLER_LOG_DIR}/include
    ${SIMPLER_COMMON_PLATFORM_DIR}/include
)
target_link_libraries(test_a2a3_aicpu_affinity_select PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    ${CMAKE_DL_LIBS}
    pthread
)
add_test(NAME test_a2a3_aicpu_affinity_select COMMAND test_a2a3_aicpu_affinity_select)
set_tests_properties(test_a2a3_aicpu_affinity_select PROPERTIES LABELS "no_hardware")

# a5 CPU_TOPO fallback and affinity selection. Pure logic; packaged fallback
# entries enforce their own verified SoC, host-architecture, and OCCUPY constraints.
set(A5_ONBOARD_HOST_DIR ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/onboard/host)
add_executable(test_a5_aicpu_topology_fallback
    a5/test_aicpu_topology_fallback.cpp
    ${A5_ONBOARD_HOST_DIR}/aicpu_topology_probe.cpp
)
add_custom_command(TARGET test_a5_aicpu_topology_fallback POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${A5_ONBOARD_HOST_DIR}/aicpu_cpu_topo_fallback.json"
        "$<TARGET_FILE_DIR:test_a5_aicpu_topology_fallback>/aicpu_cpu_topo_fallback.json"
)
target_include_directories(test_a5_aicpu_topology_fallback PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${A5_ONBOARD_HOST_DIR}
    ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
    ${SIMPLER_LOG_DIR}/include
    ${SIMPLER_COMMON_PLATFORM_DIR}/include
)
target_link_libraries(test_a5_aicpu_topology_fallback PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    ${CMAKE_DL_LIBS}
    pthread
)
add_test(NAME test_a5_aicpu_topology_fallback COMMAND test_a5_aicpu_topology_fallback)
set_tests_properties(test_a5_aicpu_topology_fallback PROPERTIES LABELS "no_hardware")

# Hardware-gated tests.  Block is only entered when the project is configured
# with -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.  CI's no-hw `ut` job does not pass
# this flag, so nothing here is compiled there — this is the structural
# invariant that lets future hw tests freely link against CANN libraries
# without breaking the no-hw build.
if(SIMPLER_ENABLE_HARDWARE_TESTS)
    # libhost_runtime.so is loaded via dlopen at runtime — it is the
    # subject under test and mirrors how ChipWorker selects a runtime
    # backend in production.  libascendcl.so, by contrast, is generic
    # CANN infrastructure: we link it directly so the test code uses
    # <acl/acl.h> types and values normally instead of going through
    # dlsym boilerplate.
    #
    # ctest label still gates execution via
    #   -L "^requires_hardware(_a2a3)?$"
    # so we can both (a) only build on hw runners, and (b) still filter
    # by runtime role when multiple hw tests coexist.
    set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../..")
    set(HOST_RUNTIME_LIB
        "${PROJECT_ROOT}/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
    )
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR
            "SIMPLER_ENABLE_HARDWARE_TESTS=ON requires ASCEND_HOME_PATH to be set "
            "(usually via `source $ASCEND_HOME_PATH/bin/setenv.bash`)")
    endif()
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    find_library(ASCENDCL_LIB ascendcl
        PATHS ${ASCEND_HOME_PATH}/lib64 NO_DEFAULT_PATH REQUIRED)
    find_library(RUNTIME_LIB runtime
        PATHS
            ${ASCEND_HOME_PATH}/lib64
            ${ASCEND_HOME_PATH}/runtime/lib64
            ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        NO_DEFAULT_PATH REQUIRED)

    # Black-box lifecycle UT for the comm_* C API on Path D. See the file
    # header for the contract it pins down.
    add_executable(test_comm_lifecycle hardware/test_comm_lifecycle.cpp)
    target_include_directories(test_comm_lifecycle PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${PROJECT_ROOT}/src/a2a3/platform/include
        ${PROJECT_ROOT}/src/common/platform/include
        # Shared platform_comm headers (comm.h / comm_context.h) live under
        # src/common so both a2a3 HCCL and a5/a2a3 sim can use them.
        ${PROJECT_ROOT}/src/common
        ${ASCEND_HOME_PATH}/include
    )
    target_compile_definitions(test_comm_lifecycle PRIVATE
        PTO_HOST_RUNTIME_LIB_PATH="${HOST_RUNTIME_LIB}"
    )
    target_link_libraries(test_comm_lifecycle PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        ${ASCENDCL_LIB}
        pthread
        dl
    )
    add_test(NAME test_comm_lifecycle COMMAND test_comm_lifecycle)
    set_tests_properties(test_comm_lifecycle PROPERTIES
        LABELS "requires_hardware_a2a3"
        RESOURCE_GROUPS "2,npus:1"
    )

endif()
