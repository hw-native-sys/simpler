# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/platform_comm/ — the comm_* C API.
#
# These need an Ascend toolchain to build and a device to run, so the parent
# only enters this directory under -DSIMPLER_ENABLE_HARDWARE_TESTS=ON,
# which is the structural invariant that lets everything here link CANN libraries
# freely without breaking the no-hardware build. ASCEND_HOME_PATH, ASCENDCL_LIB
# and RUNTIME_LIB are resolved by the parent before it descends.
#
# libhost_runtime.so is loaded via dlopen at runtime — it is the subject under
# test and mirrors how ChipWorker selects a runtime backend in production.
# libascendcl.so, by contrast, is generic CANN infrastructure: it is linked
# directly so the test code uses <acl/acl.h> types and values normally instead of
# going through dlsym boilerplate.
#
# The ctest label still gates execution via -L "^requires_hardware(_a2a3)?$", so
# a target can both (a) only build on hw runners and (b) still be filtered by
# runtime role when multiple hw tests coexist.

# Black-box lifecycle UT for the comm_* C API on Path D. See the file
# header for the contract it pins down.
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
