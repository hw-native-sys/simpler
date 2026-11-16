# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(percounter_wire_header "${SIMPLER_SRC}/a5/runtime/percounter/runtime/scheduler/wire_types.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${percounter_wire_header}")
file(READ "${percounter_wire_header}" percounter_wire_source)
foreach(capacity IN ITEMS 1 2 4)
    set(payload_target test_a5_percounter_payload)
    if(NOT capacity EQUAL 1)
        string(APPEND payload_target "_x${capacity}")
    endif()
    simpler_ut_add_target(NAME ${payload_target}
        SOURCES test_payload.cpp
        ARCH a5 RUNTIME percounter
        INCLUDES
            ${SIMPLER_SRC}/a5/runtime/percounter/runtime
            ${SIMPLER_SRC}/a5/runtime/percounter/aicore
            ${SIMPLER_SRC}/a5/runtime/host_build_graph/runtime
            ${SIMPLER_SRC}/a5/runtime/host_build_graph/common
            ${SIMPLER_SRC}/common/host_build_graph
            ${SIMPLER_SRC}/a5/platform/sim/aicore
        DEFINES __CPU_SIM SIMPLER_PLATFORM_NAME="a5sim"
        LABEL no_hardware TIMEOUT 20 PRIVATE_WORKDIR)
    if(NOT capacity EQUAL 1)
        # Compile the real protocol at each supported slot capacity. The copy is
        # regenerated from the source header; production defaults stay unchanged.
        string(REGEX REPLACE "PERCOUNTER_PENDING_CAPACITY = [0-9]+;"
            "PERCOUNTER_PENDING_CAPACITY = ${capacity};" capacity_wire_source "${percounter_wire_source}")
        set(capacity_include "${CMAKE_CURRENT_BINARY_DIR}/x${capacity}")
        file(CONFIGURE OUTPUT "${capacity_include}/scheduler/wire_types.h"
            CONTENT "${capacity_wire_source}" @ONLY)
        target_include_directories(${payload_target} BEFORE PRIVATE "${capacity_include}")
    endif()
endforeach()

simpler_ut_add_target(NAME test_a5_percounter_scalar_memory
    SOURCES scalar_memory_unprofiled.cpp test_scalar_memory.cpp
    ARCH a5 RUNTIME percounter
    INCLUDES
        ${SIMPLER_SRC}/a5/runtime/percounter/runtime
        ${SIMPLER_SRC}/a5/runtime/percounter/aicore
        ${SIMPLER_SRC}/a5/runtime/host_build_graph/runtime
        ${SIMPLER_SRC}/a5/runtime/host_build_graph/common
        ${SIMPLER_SRC}/common/host_build_graph
        ${SIMPLER_SRC}/a5/platform/sim/aicore
    DEFINES __CPU_SIM SIMPLER_PLATFORM_NAME="a5sim"
    LABEL no_hardware TIMEOUT 20 PRIVATE_WORKDIR)
