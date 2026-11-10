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

a5_platform_case(test_ccu_execution_abi.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/include)

find_package(OpenSSL REQUIRED)
a5_platform_case(test_ccu_session.cpp
    SOURCES ${A5_ONBOARD_HOST_DIR}/ccu/ccu_session.cpp
    INCLUDES ${A5_ONBOARD_HOST_DIR}/ccu ${SIMPLER_SRC}/a5/platform/include
    LINK OpenSSL::Crypto ${CMAKE_DL_LIBS})
include("${CMAKE_CURRENT_SOURCE_DIR}/support/ccu_session_tests.cmake")
get_filename_component(_ccu_test_root "${SIMPLER_SRC}/.." ABSOLUTE)
ccu_session_test_modules(test_a5_ccu_session "${_ccu_test_root}")
set(_ccu_sdk "$ENV{ASCEND_HOME_PATH}")
if(SIMPLER_ENABLE_HARDWARE_TESTS AND EXISTS "${_ccu_sdk}/include/hcomm/ccu/ccu_primitives.hpp")
    a5_platform_case(test_ccu_service_program.cpp
        SOURCES ${A5_ONBOARD_HOST_DIR}/ccu/ccu_service_program.cpp
        INCLUDES ${A5_ONBOARD_HOST_DIR}/ccu ${_ccu_sdk}/include ${_ccu_sdk}/include/hccl
            ${_ccu_sdk}/include/hcomm/ccu ${_ccu_sdk}/pkg_inc/hcomm/ccu)
    set(ASCEND_HOME_PATH "${_ccu_sdk}")
    include("${A5_ONBOARD_HOST_DIR}/ccu/ccu_build_config.cmake")
    a5_platform_case(test_ccu_session_sdk.cpp
        SOURCES ${A5_ONBOARD_HOST_DIR}/ccu/ccu_session_sdk.cpp
        INCLUDES ${A5_ONBOARD_HOST_DIR}/ccu ${CCU_GENERATED_INCLUDE_DIR} ${_ccu_sdk}/include ${_ccu_sdk}/include/hccl
            ${_ccu_sdk}/include/hcomm/ccu ${_ccu_sdk}/pkg_inc/hcomm/ccu
        LINK ${CMAKE_DL_LIBS})
else()
    simpler_ut_exclude_case(test_ccu_service_program.cpp)
    simpler_ut_exclude_case(test_ccu_session_sdk.cpp)
endif()

a5_platform_case(test_ccu_request_state.cpp
    SOURCES ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_request_state.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/include)

a5_platform_case(test_ccu_request_validation.cpp
    SOURCES ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_request_validation.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/include)

a5_platform_case(test_ccu_execution_api.cpp
    SOURCES
        ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_execution_api.cpp
        ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_execution_registry.cpp
        ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_request_state.cpp
        ${SIMPLER_SRC}/a5/platform/onboard/host/ccu/ccu_request_validation.cpp
    INCLUDES ${SIMPLER_SRC}/a5/platform/include)

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
    SOURCES ${A5_ONBOARD_HOST_DIR}/aicpu_affinity_select.cpp
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
        ${SIMPLER_SRC}/common/platform/sim/aicpu/cache_ops.cpp
        ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_SRC}/common/platform/sim/aicpu)
set_tests_properties(test_a5_aicore_retirement PROPERTIES TIMEOUT 20)

# The AICore half of the return gate: it compiles the real a5sim
# aicore/inner_kernel.h (so the sim include directory is on the path) and the
# same production platform_regs.cpp retire path as the case above, which is what
# publishes the gate the worker waits on.
a5_platform_case(test_return_gate_wait.cpp
    SOURCES
        ${SIMPLER_SRC}/a5/platform/shared/aicpu/platform_regs.cpp
        ${SIMPLER_SRC}/a5/platform/sim/aicpu/inner_platform_regs.cpp
        ${SIMPLER_SRC}/common/platform/sim/aicpu/device_time.cpp
        ${SIMPLER_SRC}/common/platform/sim/aicpu/cache_ops.cpp
        ${HOST_LOG_TEST_SOURCES}
    INCLUDES ${SIMPLER_SRC}/common/platform/sim/aicpu ${SIMPLER_SRC}/a5/platform/sim)
set_tests_properties(test_a5_return_gate_wait PROPERTIES TIMEOUT 20)

# The real sim wrapper is loaded and unloaded with each runtime's Runtime ABI.
# The executor is an identity-checking fixture; pthread key accounting uses
# the process's actual resource limit and calls the production wrapper.
foreach(runtime ${SIMPLER_RUNTIMES})
    simpler_runtime_tag(${runtime} runtime_tag)
    simpler_ut_includes(a5 ${runtime} host tls_contract)
    set(tls_fixture a5_${runtime_tag}_sim_tls_fixture)
    add_library(${tls_fixture} SHARED
        ${SIMPLER_SRC}/a5/platform/sim/aicore/kernel.cpp
        support/sim_aicore_tls_fixture.cpp)
    target_include_directories(${tls_fixture} PRIVATE
        ${tls_contract}
        ${SIMPLER_SRC}/a5/platform/sim/aicore)
    target_link_libraries(${tls_fixture} PRIVATE pthread ${CMAKE_DL_LIBS})
    simpler_ut_add_target(NAME test_a5_${runtime_tag}_sim_aicore_tls
        SOURCES test_sim_aicore_tls.cpp
        DEFINES SIM_TLS_FIXTURE_PATH="$<TARGET_FILE:${tls_fixture}>"
        LINK ${CMAKE_DL_LIBS}
        DEPENDS ${tls_fixture}
        LABEL no_hardware
        TIMEOUT 30)
endforeach()

simpler_ut_glob_cases(a5_platform_case)
