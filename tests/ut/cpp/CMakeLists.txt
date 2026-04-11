# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# C++ unit tests for src/common/distributed using GoogleTest.
# Run with: cmake --build . --target run_ut_cpp
# Or directly: ctest --test-dir build/ut_cpp

cmake_minimum_required(VERSION 3.15)
project(dist_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# GoogleTest (pre-installed at /usr/local)
# ---------------------------------------------------------------------------
find_library(GTEST_LIB      gtest      PATHS /usr/local/lib64 REQUIRED)
find_library(GTEST_MAIN_LIB gtest_main PATHS /usr/local/lib64 REQUIRED)

# ---------------------------------------------------------------------------
# Distributed runtime sources under test
# ---------------------------------------------------------------------------
set(DIST_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/distributed)

set(DIST_SOURCES
    ${DIST_SRC_DIR}/dist_types.cpp
    ${DIST_SRC_DIR}/dist_tensormap.cpp
    ${DIST_SRC_DIR}/dist_ring.cpp
    ${DIST_SRC_DIR}/dist_scope.cpp
    ${DIST_SRC_DIR}/dist_orchestrator.cpp
    ${DIST_SRC_DIR}/dist_sub_worker.cpp
    ${DIST_SRC_DIR}/dist_chip_process.cpp
    ${DIST_SRC_DIR}/dist_scheduler.cpp
    ${DIST_SRC_DIR}/dist_worker.cpp
)

# ---------------------------------------------------------------------------
# Helper: add one test executable
# ---------------------------------------------------------------------------
function(add_dist_test name src)
    add_executable(${name} ${src} ${DIST_SOURCES})
    target_include_directories(${name} PRIVATE
        /usr/local/include
        ${DIST_SRC_DIR}
    )
    target_compile_options(${name} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(${name} PRIVATE
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_a2a3_tmr_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        /usr/local/include
        ${CMAKE_SOURCE_DIR}/../../../src/common/task_interface
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/runtime
        ${CMAKE_SOURCE_DIR}/../../../src/a2a3/runtime/tensormap_and_ringbuffer/orchestration
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

add_dist_test(test_dist_tensormap  test_dist_tensormap.cpp)
add_dist_test(test_dist_ring       test_dist_ring.cpp)
add_dist_test(test_dist_scope      test_dist_scope.cpp)
add_dist_test(test_dist_orchestrator test_dist_orchestrator.cpp)
add_dist_test(test_dist_scheduler  test_dist_scheduler.cpp)
add_a2a3_tmr_test(test_a2a3_tmr_tensor_offsets test_a2a3_tmr_tensor_offsets.cpp)
