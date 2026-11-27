# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

macro(platform_comm_case src)
    simpler_ut_plain_case(${src}
        INCLUDES
            ${SIMPLER_SRC}/a2a3/platform/include
            ${SIMPLER_COMMON_PLATFORM_DIR}/include
            # Shared platform_comm headers (comm.h / comm_context.h) live under
            # src/common so both a2a3 HCCL and a5/a2a3 sim can use them.
            ${SIMPLER_SRC}/common
            ${ASCEND_HOME_PATH}/include
        LINK ${ASCENDCL_LIB} ${CMAKE_DL_LIBS}
        LABEL requires_hardware_a2a3
        RESOURCES "2,npus:1"
        ${ARGN})
endmacro()

platform_comm_case(test_comm_lifecycle.cpp
    DEFINES PTO_HOST_RUNTIME_LIB_PATH="${HOST_RUNTIME_LIB}")

simpler_ut_glob_cases(platform_comm_case)
