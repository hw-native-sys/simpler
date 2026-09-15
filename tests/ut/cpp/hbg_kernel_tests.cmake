# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


add_a2a3_hbg_runtime_test(test_hbg_host_graph_build common/test_hbg_host_graph_build.cpp)
add_a5_hbg_runtime_test(test_a5_hbg_host_graph_build common/test_hbg_host_graph_build.cpp)
foreach(arch IN ITEMS a2a3 a5)
    if(arch STREQUAL "a2a3")
        set(hbg_build_target test_hbg_host_graph_build)
    else()
        set(hbg_build_target test_a5_hbg_host_graph_build)
    endif()
    target_compile_definitions(${hbg_build_target} PRIVATE SIMPLER_PLATFORM_NAME="${arch}sim")
    target_include_directories(${hbg_build_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../src/common/worker
    )
    target_sources(${hbg_build_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/kernel_execution_state.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/kernel_device_resources.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/sim/host/memory_allocator.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/${arch}/runtime/host_build_graph/host/runtime_maker.cpp
        ${HBG_SHARED_RUNTIME_SOURCES}
        ${HBG_HOST_DIR}/graph_definition_pack.cpp
        ${HBG_HOST_DIR}/orchestrator.cpp
        ${HBG_HOST_DIR}/runtime_core.cpp
        ${HBG_HOST_DIR}/host_tensor_access.cpp
        ${HBG_HOST_DIR}/host_phase_trace.cpp
        ${HBG_HOST_DIR}/dep_gen_host_graph.cpp
        ${HBG_HOST_DIR}/graph_recorder_pool.cpp
        ${HBG_HOST_DIR}/ready_queue_sizing.cpp
        ${HBG_HOST_DIR}/kernel_graph_template.cpp
        ${HBG_HOST_DIR}/kernel_graph_slot.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/host_build_graph/device/kernel_graph_slot_registry.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/host_build_graph/device/kernel_graph_restore.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/aicpu/args_dump_aicpu.cpp
        ${CMAKE_SOURCE_DIR}/../../../src/common/platform/shared/host/platform_compile_info.cpp
    )
    target_sources(${hbg_build_target} PRIVATE ${CMAKE_SOURCE_DIR}/../../../src/common/log/host_log.cpp)
    target_link_libraries(${hbg_build_target} PRIVATE ${CMAKE_DL_LIBS})
endforeach()
