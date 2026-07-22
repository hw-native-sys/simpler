# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

get_filename_component(SRC_ROOT ${CMAKE_SOURCE_DIR}/../../../src ABSOLUTE)
set(REPLAY_GRAPH_DIR ${SRC_ROOT}/a2a3/runtime/replay_graph)
set(REPLAY_GRAPH_RUNTIME_DIR ${REPLAY_GRAPH_DIR}/runtime)

add_library(a2a3_replay_graph_rt_objs OBJECT
    ${REPLAY_GRAPH_RUNTIME_DIR}/pto_graph_execution.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/pto_ring_buffer.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_shared_memory.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/scheduler/pto_scheduler.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_tensormap.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/shared/pto_runtime2_init.cpp
    ${REPLAY_GRAPH_RUNTIME_DIR}/pto_orchestrator.cpp
    ${SRC_ROOT}/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp
    ${SRC_ROOT}/common/platform/shared/aicpu/args_dump_aicpu.cpp
    ${CMAKE_SOURCE_DIR}/stubs/test_stubs.cpp
)
target_include_directories(a2a3_replay_graph_rt_objs PUBLIC
    ${REPLAY_GRAPH_DIR}/orchestration
    ${REPLAY_GRAPH_DIR}/runtime
    ${REPLAY_GRAPH_DIR}/common
    ${SRC_ROOT}/a2a3/runtime
    ${SRC_ROOT}/a2a3/platform/include
    ${SRC_ROOT}/common/platform/sim/aicpu
    ${SRC_ROOT}/common/platform/include
    ${SRC_ROOT}/common/task_interface
    ${SRC_ROOT}/common/log/include
    ${SRC_ROOT}/common
)

add_executable(test_rg_single_shot test_single_shot.cpp)
target_include_directories(test_rg_single_shot PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_rg_single_shot PRIVATE
    a2a3_replay_graph_rt_objs
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_rg_single_shot COMMAND test_rg_single_shot)
set_tests_properties(test_rg_single_shot PROPERTIES LABELS "no_hardware")

add_executable(test_rg_graph_api test_graph_api.cpp)
target_include_directories(test_rg_graph_api PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_rg_graph_api PRIVATE
    a2a3_replay_graph_rt_objs
    ${GTEST_MAIN_LIB}
    ${GTEST_LIB}
    pthread
)
add_test(NAME test_rg_graph_api COMMAND test_rg_graph_api)
set_tests_properties(test_rg_graph_api PROPERTIES LABELS "no_hardware")
