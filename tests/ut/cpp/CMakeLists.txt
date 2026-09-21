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

# Executables land in the build directory root regardless of which subdirectory
# declares them. tests/lint/check_ut_cpp_stub_linkage.py reads every test
# binary's symbol table and finds them by globbing that one directory, so a
# per-subdirectory layout would hide from it exactly the targets it exists to
# check.
#
# A consumer outside ctest does not rely on this: it reaches a binary through
# $<TARGET_FILE:...>, recorded for it by PYUT_EXPORT, which resolves to
# wherever the file lands.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# SIMPLER_SRC (the tree under test) and simpler_discover_runtimes().
include(${CMAKE_CURRENT_LIST_DIR}/cmake/runtimes.cmake)
# simpler_ut_includes() / simpler_ut_common_includes() — the include contract.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/contract.cmake)
# simpler_ut_add_test() — registration for targets built once per runtime.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ut_test.cmake)

# Directory variables shared by more than one subdirectory. A subdirectory
# inherits the parent scope but cannot write back into it, so anything two of
# them need is set here, before the first add_subdirectory.
set(SIMPLER_LOG_DIR ${SIMPLER_SRC}/common/log)
set(HOST_LOG_TEST_SOURCES
    ${SIMPLER_LOG_DIR}/host_log.cpp
    ${SIMPLER_LOG_DIR}/unified_log_host.cpp
)
set(SIMPLER_COMMON_PLATFORM_DIR ${SIMPLER_SRC}/common/platform)

# Directory and source-list variables the runtime subdirectories read. CMake
# passes variables down but not across, so these cannot live beside the object
# libraries in common/<runtime>/: a2a3/ and a5/ are siblings of common/, and a
# helper's body resolves its variables in the scope that calls it.
set(A2A3_RUNTIME_DIR ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/runtime)
set(TMR_SHARED_HOST_DIR ${SIMPLER_SRC}/common/tensormap_and_ringbuffer/host)
set(A5_RUNTIME_DIR ${SIMPLER_SRC}/a5/runtime/tensormap_and_ringbuffer/runtime)
set(HBG_RUNTIME_DIR ${SIMPLER_SRC}/a2a3/runtime/host_build_graph/runtime)
set(A5_HBG_RUNTIME_DIR ${SIMPLER_SRC}/a5/runtime/host_build_graph/runtime)
set(HBG_COMMON_SHARED_DIR ${SIMPLER_SRC}/common/host_build_graph/shared)
set(HBG_HOST_DIR ${SIMPLER_SRC}/common/host_build_graph/host)
# The hbg source lists are globs of the directories src/common/host_build_graph
# partitions by which runtime target compiles them: shared/ is built by both the
# host and the device target, host/ only by the host one. Linking a file a given
# test does not call costs nothing, whereas naming files individually means the
# next one added to those directories is silently absent from every test that
# should have had it.
file(GLOB HBG_SHARED_RUNTIME_SOURCES CONFIGURE_DEPENDS ${HBG_COMMON_SHARED_DIR}/*.cpp)
list(APPEND HBG_SHARED_RUNTIME_SOURCES ${HBG_HOST_DIR}/tensormap.cpp)

file(GLOB _hbg_orch_host_sources CONFIGURE_DEPENDS ${HBG_HOST_DIR}/*.cpp)
# The orchestrator's two capture sinks are the exception: support/hbg_orch_stubs.cpp
# supplies them instead, so a test driving the orchestrator does not drag in the
# dep-gen collector and the host-phase record pool behind them. The definitions
# are ordinary rather than weak, so a target takes one set or the other.
list(REMOVE_ITEM _hbg_orch_host_sources
    ${HBG_HOST_DIR}/dep_gen_host_graph.cpp
    ${HBG_HOST_DIR}/host_phase_trace.cpp
)
set(HBG_ORCH_SHARED_SOURCES
    ${_hbg_orch_host_sources}
    ${CMAKE_SOURCE_DIR}/support/hbg_orch_stubs.cpp
    ${HBG_SHARED_RUNTIME_SOURCES}
)
list(REMOVE_DUPLICATES HBG_ORCH_SHARED_SOURCES)

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


enable_testing()

# Read by common/platform (the tmr half) and a2a3 (the hbg half), which build
# the same test source against their own runtime.
set(KERNEL_PERSISTENT_ARGS_SRC
    ${SIMPLER_SRC}/common/platform/onboard/host/kernel_persistent_args.cpp
    ${SIMPLER_SRC}/common/platform/shared/host/kernel_execution_state.cpp)

# Hardware-gated tests.  Only resolved when the project is configured with
# -DSIMPLER_ENABLE_HARDWARE_TESTS=ON.  CI's no-hw `ut` job does not pass this
# flag, so nothing gated on it is compiled there — this is the structural
# invariant that lets hw tests link against CANN libraries freely without
# breaking the no-hw build.  The lookups sit here rather than beside the one
# gated directory (common/platform_comm/) because that directory is reached
# through common/, which is descended into below.
if(SIMPLER_ENABLE_HARDWARE_TESTS)
    set(PROJECT_ROOT "${SIMPLER_REPO_ROOT}")
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
endif()

# The shared stand-ins come first: every other directory links one of them.
add_subdirectory(support)

# Each directory declares the targets covering the tree it mirrors under src/.
add_subdirectory(common)
add_subdirectory(a2a3)
add_subdirectory(a5)

# Last, so every target has registered the cases it consumed.
simpler_ut_assert_no_orphan_cases()

# Likewise: every PYUT_EXPORT has been recorded by now.
simpler_ut_write_pyut_manifest()
