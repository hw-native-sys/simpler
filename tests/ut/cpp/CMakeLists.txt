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
    ${A2A3_RUNTIME_DIR}/pto_tensormap.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a2a3_rt_objs PUBLIC
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/common
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
)
target_compile_options(a2a3_rt_objs PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)

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
    ${HIERARCHICAL_SRC_DIR}/orchestrator.cpp
    ${HIERARCHICAL_SRC_DIR}/worker_manager.cpp
    ${HIERARCHICAL_SRC_DIR}/scheduler.cpp
    ${HIERARCHICAL_SRC_DIR}/worker.cpp
    ${WORKER_SRC_DIR}/chip_worker.cpp
)
target_include_directories(hierarchical_objs PUBLIC
    ${HIERARCHICAL_SRC_DIR}
    ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
    ${WORKER_SRC_DIR}
)
target_compile_options(hierarchical_objs PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)

function(add_hierarchical_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE
        hierarchical_objs
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
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
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
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
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/common/log/include
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

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

function(add_common_utils_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${GTEST_INCLUDE_DIRS}
        # utils/elf_build_id.h lives under src/common, exposed as "utils/..."
        ${CMAKE_SOURCE_DIR}/../../../src/common
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

enable_testing()

# ---------------------------------------------------------------------------
# Hierarchical runtime tests (src/common/hierarchical/)
# ---------------------------------------------------------------------------
add_hierarchical_test(test_tensormap  hierarchical/test_tensormap.cpp)
add_hierarchical_test(test_ring  hierarchical/test_ring.cpp)
add_hierarchical_test(test_scope      hierarchical/test_scope.cpp)
add_hierarchical_test(test_orchestrator hierarchical/test_orchestrator.cpp)
add_hierarchical_test(test_scheduler  hierarchical/test_scheduler.cpp)

# ---------------------------------------------------------------------------
# Types / task_interface tests (src/common/task_interface/)
# ---------------------------------------------------------------------------
add_task_interface_test(test_child_memory types/test_child_memory.cpp)

# ---------------------------------------------------------------------------
# Common utilities tests (src/common/utils/)
# ---------------------------------------------------------------------------
add_common_utils_test(test_elf_build_id common/test_elf_build_id.cpp)
add_common_utils_test(test_runtime_orch_so common/test_runtime_orch_so.cpp)

# Per-callable_id orch SO file naming regression (see rtStreamSynchronize
# 507018 root cause). Compiles the a2a3 onboard `create_orch_so_file`
# against the test source so it runs on no-hw runners too.
add_executable(test_orch_so_file
    common/test_orch_so_file.cpp
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/onboard/aicpu/orch_so_file.cpp
)
target_include_directories(test_orch_so_file PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../../src/a2a3/platform/include
)
target_compile_options(test_orch_so_file PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
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

# PTO2 runtime-linked tests
add_a2a3_runtime_test(test_task_allocator   a2a3/test_task_allocator.cpp)
add_a2a3_runtime_test(test_dep_list_pool    a2a3/test_dep_list_pool.cpp)
add_a2a3_runtime_test(test_scheduler_state  a2a3/test_scheduler_state.cpp)
add_a2a3_runtime_test(test_task_state       a2a3/test_task_state.cpp)
add_a2a3_runtime_test(test_ready_queue      a2a3/test_ready_queue.cpp)
add_a2a3_runtime_test(test_shared_memory    a2a3/test_shared_memory.cpp)
add_a2a3_runtime_test(test_a2a3_tensormap   a2a3/test_tensormap.cpp)
add_a2a3_runtime_test(test_fanin_pool       a2a3/test_fanin_pool.cpp)
add_a2a3_runtime_test(test_spsc_queue       a2a3/test_spsc_queue.cpp)
add_a2a3_runtime_test(test_wiring           a2a3/test_wiring.cpp)

# ---------------------------------------------------------------------------
# A5 tests (src/a5/runtime/tensormap_and_ringbuffer/)
# ---------------------------------------------------------------------------
add_a5_test(test_a5_fatal a5/test_a5_fatal.cpp)

# Host logger silent/off behavior — no runtime deps, just compile host_log.cpp
# alongside the test (faster than dlopen'ing libsimpler_log.so for a unit test).
set(SIMPLER_LOG_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/log)
add_executable(test_host_log_off
    a5/test_host_log_off.cpp
    ${SIMPLER_LOG_DIR}/host_log.cpp
)
target_include_directories(test_host_log_off PRIVATE
    ${GTEST_INCLUDE_DIRS}
    ${SIMPLER_LOG_DIR}
    ${SIMPLER_LOG_DIR}/include
)
target_compile_options(test_host_log_off PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(test_host_log_off PRIVATE
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_host_log_off COMMAND test_host_log_off)

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
    # libsimpler_log.so must be loaded RTLD_GLOBAL before host_runtime dlopen
    # so undefined HostLogger / unified_log_* symbols in host_runtime resolve.
    # Mirrors what ChipWorker::init does in production. The library is
    # process-global (one .so per host toolchain, not per arch/variant).
    set(SIMPLER_LOG_LIB
        "${PROJECT_ROOT}/build/lib/libsimpler_log.so"
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
            SIMPLER_LOG_LIB_PATH="${SIMPLER_LOG_LIB}"
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
