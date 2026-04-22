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
# googletest) on GH-hosted runners.  Self-hosted (a2a3/a5) runners don't
# have GoogleTest preinstalled, so we fall back to FetchContent from
# upstream.  Either way the rest of the file sees GTEST_LIB / GTEST_MAIN_LIB
# / GTEST_INCLUDE_DIRS with a consistent meaning.
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
    # Match the ABI the tests themselves compile with
    # (-D_GLIBCXX_USE_CXX11_ABI=0 — see add_hierarchical_test / add_comm_api_test).
    # Without this the FetchContent-built libgtest.a ends up with the default
    # cxx11 ABI and link fails against our tests.
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

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
# PTO2 runtime sources and stubs for a2a3 ring-buffer / tensormap tests
# ---------------------------------------------------------------------------
set(A2A3_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime)
set(PTO2_STUB_SOURCES ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp)
set(PTO2_RUNTIME_SOURCES
    ${A2A3_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${A2A3_RUNTIME_DIR}/pto_shared_memory.cpp
    ${A2A3_RUNTIME_DIR}/pto_scheduler.cpp
    ${A2A3_RUNTIME_DIR}/pto_tensormap.cpp
)

set(PTO2_COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
)

function(add_a2a3_pto2_runtime_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;EXTRA_SOURCES" ${ARGN})
    set(_all_sources ${ARG_SOURCES} ${PTO2_STUB_SOURCES})
    foreach(src ${ARG_SOURCES} ${ARG_EXTRA_SOURCES})
        if(EXISTS ${src})
            list(APPEND _all_sources ${src})
        endif()
    endforeach()
    add_executable(${name} ${_all_sources})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${PTO2_COMMON_INCLUDE_DIRS}
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "no_hardware")
endfunction()

# ---------------------------------------------------------------------------
# Distributed runtime sources under test
# ---------------------------------------------------------------------------
set(HIERARCHICAL_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/hierarchical)

set(WORKER_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/worker)

set(HIERARCHICAL_SOURCES
    ${HIERARCHICAL_SRC_DIR}/types.cpp
    ${HIERARCHICAL_SRC_DIR}/tensormap.cpp
    ${HIERARCHICAL_SRC_DIR}/ring.cpp
    ${HIERARCHICAL_SRC_DIR}/scope.cpp
    ${HIERARCHICAL_SRC_DIR}/orchestrator.cpp
    ${HIERARCHICAL_SRC_DIR}/worker_manager.cpp
    ${HIERARCHICAL_SRC_DIR}/scheduler.cpp
    ${HIERARCHICAL_SRC_DIR}/worker.cpp
    ${WORKER_SRC_DIR}/chip_worker.cpp
)

# ---------------------------------------------------------------------------
# Helper: add one test executable
# ---------------------------------------------------------------------------
function(add_hierarchical_test name src)
    add_executable(${name} ${src} ${HIERARCHICAL_SOURCES})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${HIERARCHICAL_SRC_DIR}
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${WORKER_SRC_DIR}
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a2a3_pto2_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a5_pto2_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/orchestration
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a5/runtime/tensormap_and_ringbuffer/common
        ${CMAKE_SOURCE_DIR}/../../../src/a5/platform/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

# ---------------------------------------------------------------------------
# Hierarchical runtime tests (src/common/hierarchical/)
# ---------------------------------------------------------------------------
add_hierarchical_test(test_tensormap  hierarchical/test_tensormap.cpp)
add_hierarchical_test(test_ring  hierarchical/test_ring.cpp)
add_hierarchical_test(test_scope      hierarchical/test_scope.cpp)
add_hierarchical_test(test_orchestrator hierarchical/test_orchestrator.cpp)
add_hierarchical_test(test_scheduler  hierarchical/test_scheduler.cpp)
add_hierarchical_test(test_worker_manager hierarchical/test_worker_manager.cpp)
function(add_task_interface_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# Types / task_interface tests (src/common/task_interface/)
# ---------------------------------------------------------------------------
add_task_interface_test(test_child_memory types/test_child_memory.cpp)

# ---------------------------------------------------------------------------
# PTO2 A2A3 tests (src/a2a3/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a2a3_pto2_test(test_a2a3_pto2_fatal pto2_a2a3/test_a2a3_pto2_fatal.cpp)
add_a2a3_pto2_test(test_runtime_status  pto2_a2a3/test_runtime_status.cpp)
add_a2a3_pto2_test(test_core_types      pto2_a2a3/test_core_types.cpp)
add_a2a3_pto2_test(test_dispatch_payload pto2_a2a3/test_dispatch_payload.cpp)
add_a2a3_pto2_test(test_handshake       pto2_a2a3/test_handshake.cpp)
add_a2a3_pto2_test(test_submit_types    pto2_a2a3/test_submit_types.cpp)
add_a2a3_pto2_runtime_test(test_pto_types
    SOURCES types/test_pto_types.cpp
)
add_a2a3_pto2_runtime_test(test_tensor
    SOURCES types/test_tensor.cpp
)

# ---------------------------------------------------------------------------
# PTO2 A5 tests (src/a5/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a5_pto2_test(test_a5_pto2_fatal pto2_a5/test_a5_pto2_fatal.cpp)

# PTO2 runtime-linked tests (ring buffer, tensormap, dep pool, etc.)
add_a2a3_pto2_runtime_test(test_ring_buffer
    SOURCES pto2_a2a3/test_ring_buffer.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_ring_buffer_edge
    SOURCES pto2_a2a3/test_ring_buffer_edge.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_tensormap_edge
    SOURCES pto2_a2a3/test_tensormap_edge.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_coupling
    SOURCES pto2_a2a3/test_coupling.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES} ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
)
add_a2a3_pto2_runtime_test(test_coupling_stub
    SOURCES pto2_a2a3/test_coupling_stub.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_ready_queue
    SOURCES pto2_a2a3/test_ready_queue.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_scheduler_state
    SOURCES pto2_a2a3/test_scheduler_state.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_scheduler_edge
    SOURCES pto2_a2a3/test_scheduler_edge.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_boundary_edge
    SOURCES pto2_a2a3/test_boundary_edge.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
add_a2a3_pto2_runtime_test(test_shared_memory
    SOURCES pto2_a2a3/test_shared_memory.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
)
# Runtime lifecycle tests — link the full PTO2 runtime including
# pto_runtime2.cpp + pto_orchestrator.cpp so create/destroy is exercised end-to-end.
add_a2a3_pto2_runtime_test(test_runtime_lifecycle
    SOURCES pto2_a2a3/test_runtime_lifecycle.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES}
                  ${A2A3_RUNTIME_DIR}/pto_runtime2.cpp
                  ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
)

# Orchestrator submit-path tests
add_a2a3_pto2_runtime_test(test_orchestrator_submit
    SOURCES pto2_a2a3/test_orchestrator_submit.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES} ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
)

# Orchestrator fatal-path tests — exercise the real pto2_orch_report_fatal
# against a fully-initialized orchestrator + shared memory pair.
add_a2a3_pto2_runtime_test(test_orchestrator_fatal
    SOURCES pto2_a2a3/test_orchestrator_fatal.cpp
    EXTRA_SOURCES ${PTO2_RUNTIME_SOURCES} ${A2A3_RUNTIME_DIR}/pto_orchestrator.cpp
)

# ---------------------------------------------------------------------------
# Platform sim layer tests — only the sim variants are pure userspace and
# can be unit-tested off-hardware.  Hardware (onboard) variants that wrap
# rtMalloc/rtFree / device registers are exercised by the hardware CI.
# ---------------------------------------------------------------------------
set(A2A3_PLATFORM_SIM_DIR  ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/sim)
set(A2A3_PLATFORM_HOST_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/src/host)
set(A2A3_PLATFORM_INCLUDE  ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include)

add_executable(test_platform_memory_allocator
    platform/test_platform_memory_allocator.cpp
    ${A2A3_PLATFORM_SIM_DIR}/host/memory_allocator.cpp
    ${PTO2_STUB_SOURCES}
)
target_include_directories(test_platform_memory_allocator PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${A2A3_PLATFORM_INCLUDE}
)
target_compile_options(test_platform_memory_allocator PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(test_platform_memory_allocator PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_platform_memory_allocator COMMAND test_platform_memory_allocator)
set_tests_properties(test_platform_memory_allocator PROPERTIES LABELS "no_hardware")

add_executable(test_platform_host_log
    platform/test_platform_host_log.cpp
    ${A2A3_PLATFORM_HOST_DIR}/host_log.cpp
)
target_include_directories(test_platform_host_log PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${A2A3_PLATFORM_HOST_DIR}
)
target_compile_options(test_platform_host_log PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(test_platform_host_log PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_platform_host_log COMMAND test_platform_host_log)
set_tests_properties(test_platform_host_log PROPERTIES LABELS "no_hardware")

# host_build_graph Runtime test — needs the HBG runtime.cpp source
set(HBG_RUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/host_build_graph/runtime)
add_executable(test_runtime_graph
    pto2_a2a3/test_runtime_graph.cpp
    ${HBG_RUNTIME_DIR}/runtime.cpp
    ${PTO2_STUB_SOURCES}
)
target_include_directories(test_runtime_graph PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${HBG_RUNTIME_DIR}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
)
target_compile_options(test_runtime_graph PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(test_runtime_graph PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_runtime_graph COMMAND test_runtime_graph)
set_tests_properties(test_runtime_graph PROPERTIES LABELS "no_hardware")

# Hardware-gated tests.  Block is only entered when the project is configured
# with -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.  CI's no-hw `ut` job does not pass
# this flag, so nothing here is compiled there — this is the structural
# invariant that lets future hw tests freely link against CANN libraries
# without breaking the no-hw build.
if(SIMPLER_ENABLE_HARDWARE_TESTS)
    # Hardware-only UT for the HCCL backend of the comm_* C API.
    #
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

    function(add_comm_api_test name src)
        add_executable(${name} ${src})
        target_include_directories(${name} PRIVATE
            ${GTEST_INCLUDE_DIRS}
            ${PROJECT_ROOT}/src/a2a3/platform/include
            # Shared platform_comm headers (comm.h / comm_context.h) moved
            # under src/common so both a2a3 HCCL and a5/a2a3 sim can use them.
            ${PROJECT_ROOT}/src/common
            ${ASCEND_HOME_PATH}/include
        )
        target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
        target_compile_definitions(${name} PRIVATE
            PTO_HOST_RUNTIME_LIB_PATH="${HOST_RUNTIME_LIB}"
        )
        target_link_libraries(${name} PRIVATE
            ${GTEST_MAIN_LIB}
            ${GTEST_LIB}
            ${ASCENDCL_LIB}
            pthread
            dl
        )
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES
            LABELS "requires_hardware_a2a3"
            RESOURCE_GROUPS "2,npus:1"
        )
    endfunction()

    add_comm_api_test(test_hccl_comm hardware/test_hccl_comm.cpp)
endif()
