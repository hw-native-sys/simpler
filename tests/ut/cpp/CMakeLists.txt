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

# ---------------------------------------------------------------------------
# GoogleTest
# ---------------------------------------------------------------------------
if(APPLE)
    set(GTEST_SEARCH_PATHS /opt/homebrew/lib /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /opt/homebrew/include /usr/local/include)
else()
    set(GTEST_SEARCH_PATHS /usr/local/lib64 /usr/local/lib)
    set(GTEST_INCLUDE_DIRS /usr/local/include)
endif()

find_library(GTEST_LIB      gtest      PATHS ${GTEST_SEARCH_PATHS} REQUIRED)
find_library(GTEST_MAIN_LIB gtest_main PATHS ${GTEST_SEARCH_PATHS} REQUIRED)

# ---------------------------------------------------------------------------
# Distributed runtime sources under test
# ---------------------------------------------------------------------------
set(HIERARCHICAL_SRC_DIR ${CMAKE_SOURCE_DIR}/../../../src/common/hierarchical)

set(HIERARCHICAL_SOURCES
    ${HIERARCHICAL_SRC_DIR}/types.cpp
    ${HIERARCHICAL_SRC_DIR}/tensormap.cpp
    ${HIERARCHICAL_SRC_DIR}/ring.cpp
    ${HIERARCHICAL_SRC_DIR}/scope.cpp
    ${HIERARCHICAL_SRC_DIR}/orchestrator.cpp
    ${HIERARCHICAL_SRC_DIR}/worker_manager.cpp
    ${HIERARCHICAL_SRC_DIR}/scheduler.cpp
    ${HIERARCHICAL_SRC_DIR}/worker.cpp
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

add_hierarchical_test(test_tensormap  test_tensormap.cpp)
add_hierarchical_test(test_ring  test_ring.cpp)
add_hierarchical_test(test_scope      test_scope.cpp)
add_hierarchical_test(test_orchestrator test_orchestrator.cpp)
add_hierarchical_test(test_scheduler  test_scheduler.cpp)
add_a2a3_pto2_test(test_a2a3_pto2_fatal test_a2a3_pto2_fatal.cpp)
add_a5_pto2_test(test_a5_pto2_fatal test_a5_pto2_fatal.cpp)
