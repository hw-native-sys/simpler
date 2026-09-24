# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a5/platform/.

set(A5_ONBOARD_HOST_DIR ${SIMPLER_SRC}/a5/platform/onboard/host)

# The arch is this directory's, so a case here names only what it covers and
# the target name picks the arch up from the mirror path.
macro(a5_platform_case src)
    string(REGEX REPLACE "^test_(.*)\\.cpp$" "\\1" _a5_platform_stem "${src}")
    simpler_ut_common_includes(a5 host _a5_platform_contract)
    simpler_ut_add_target(NAME test_a5_${_a5_platform_stem}
        SOURCES ${src}
        INCLUDES ${_a5_platform_contract}
        LABEL no_hardware ${ARGN})
endmacro()

# Host logger silent/off behavior — no runtime deps, just compile the same
# self-contained host logger sources used by production consumers. It carries no
# arch token: the logger is arch-agnostic and this is where the case happens to
# live.
simpler_ut_add_target(NAME test_host_log_off
    SOURCES test_host_log_off.cpp ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_SRC}/common/task_interface ${SIMPLER_LOG_DIR} ${SIMPLER_LOG_DIR}/include
    LABEL no_hardware)

# a5 CPU_TOPO fallback and affinity selection. Pure logic; packaged fallback
# entries enforce their own verified SoC, host-architecture and OCCUPY constraints.
a5_platform_case(test_aicpu_topology_fallback.cpp
    SOURCES ${A5_ONBOARD_HOST_DIR}/aicpu_topology_probe.cpp
    INCLUDES ${A5_ONBOARD_HOST_DIR}
    LINK ${CMAKE_DL_LIBS})
# The probe reads this table from beside its own binary.
add_custom_command(TARGET test_a5_aicpu_topology_fallback POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${A5_ONBOARD_HOST_DIR}/aicpu_cpu_topo_fallback.json"
        "$<TARGET_FILE_DIR:test_a5_aicpu_topology_fallback>/aicpu_cpu_topo_fallback.json"
)

a5_platform_case(test_aicore_retirement.cpp
    SOURCES
        ${SIMPLER_SRC}/a5/platform/shared/aicpu/platform_regs.cpp
        ${SIMPLER_SRC}/a5/platform/sim/aicpu/inner_platform_regs.cpp
        ${SIMPLER_SRC}/common/platform/sim/aicpu/device_time.cpp
        ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_SRC}/common/platform/sim/aicpu)
set_tests_properties(test_a5_aicore_retirement PROPERTIES TIMEOUT 20)

simpler_ut_glob_cases(a5_platform_case)
