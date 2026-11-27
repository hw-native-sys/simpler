# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Uses the CANN headers/libraries but replaces the allocation release calls;
# the production release entry point runs on the CPU with synthetic windows.
get_property(_pto_isa_root GLOBAL PROPERTY SIMPLER_UT_PTO_ISA_ROOT)
set(_release_stubs "${CMAKE_CURRENT_BINARY_DIR}/release_stubs")
foreach(_engine sdma urma)
    file(WRITE "${_release_stubs}/pto/comm/async/${_engine}/${_engine}_workspace_manager.hpp"
         "#include \"comm_workspace_managers.h\"\n")
endforeach()
simpler_ut_plain_case(test_a5_domain_release.cpp
    SOURCES ${HOST_LOG_TEST_SOURCES}
            ${SIMPLER_SRC}/common/platform_comm/ccu/ccu_domain_owner.cpp
            ${SIMPLER_SRC}/common/platform_comm/ccu/ccu_run_lifecycle.cpp
    INCLUDES_BEFORE ${_release_stubs}
    INCLUDES ${SIMPLER_SRC} ${SIMPLER_SRC}/common
             ${SIMPLER_SRC}/a5/platform/include
             ${SIMPLER_COMMON_PLATFORM_DIR}/include ${_pto_isa_root}/include ${ASCEND_HOME_PATH}/include
             ${SIMPLER_REPO_ROOT}/tests/ut/py/support
             ${SIMPLER_LOG_DIR} ${SIMPLER_LOG_DIR}/include
    LINK ${ASCENDCL_LIB} ${ASCEND_HOME_PATH}/lib64/libhccl.so ${CMAKE_DL_LIBS}
    COMPILE_OPTIONS -ffunction-sections -fdata-sections
    LINK_OPTIONS -Wl,--gc-sections
    LABEL requires_toolkit)
