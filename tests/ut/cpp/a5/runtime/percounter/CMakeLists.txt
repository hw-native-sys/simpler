# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

simpler_ut_add_target(NAME test_a5_percounter_payload
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
