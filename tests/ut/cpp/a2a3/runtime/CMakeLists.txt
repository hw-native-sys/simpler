# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for the parts of src/a2a3/runtime/ that every runtime carries, so a
# case here is declared once for each of them.
macro(a2a3_runtime_case src)
    hbg_case(${src} ARCHS a2a3 ${ARGN})
    tmr_case(${src} ARCHS a2a3 ${ARGN})
endmacro()

# NO_OBJS: this case compiles platform/shared/aicpu/platform_regs.cpp, and the
# runtime object libraries carry support/platform_regs_stub.cpp, which defines the
# same get_reg_ptr. It takes the include contract from those libraries' arch and
# runtime instead, and names the AICore executor of the runtime it is built for.
#
# TIMEOUT: the case waits on retirement and then joins the worker. A protocol
# regression leaves the retirement it waits for unpublished, so the assertion is
# never reached and the worker never exits — a hang, not a failure. Twenty
# seconds bounds it well under ctest's default, which is what keeps a local
# `ctest` from stopping there indefinitely.
set(RETIREMENT_SOURCES
    ${SIMPLER_SRC}/a2a3/platform/shared/aicpu/platform_regs.cpp
    ${SIMPLER_SRC}/a2a3/platform/sim/aicpu/inner_platform_regs.cpp
    ${SIMPLER_SRC}/common/platform/sim/aicpu/device_time.cpp)
set(RETIREMENT_INCLUDES ${SIMPLER_SRC}/a2a3/platform/sim/aicore)

hbg_case(test_aicore_retirement.cpp ARCHS a2a3 NO_OBJS TIMEOUT 20
    SOURCES  ${SIMPLER_SRC}/a2a3/runtime/host_build_graph/aicore/aicore_executor.cpp
             ${RETIREMENT_SOURCES}
    INCLUDES ${RETIREMENT_INCLUDES})
tmr_case(test_aicore_retirement.cpp ARCHS a2a3 NO_OBJS TIMEOUT 20
    SOURCES  ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp
             ${RETIREMENT_SOURCES}
    INCLUDES ${RETIREMENT_INCLUDES})

simpler_ut_glob_cases(a2a3_runtime_case)

# ---------------------------------------------------------------------------
# host_build_graph
# ---------------------------------------------------------------------------

add_subdirectory(host_build_graph)
add_subdirectory(tensormap_and_ringbuffer)
