# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#

macro(ccu_host_case src)
    simpler_ut_plain_case(${src}
        INCLUDES ${SIMPLER_SRC}/common
        LABEL no_hardware ${ARGN})
endmacro()

ccu_host_case(test_ccu_registry.cpp)
ccu_host_case(test_ccu_publication.cpp)
ccu_host_case(test_ccu_instance_run_lease.cpp)
ccu_host_case(test_ccu_shutdown.cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(ccu_module_fixture MODULE ccu_module_fixture.cpp)
    add_library(ccu_wrong_abi_fixture MODULE ccu_wrong_abi_fixture.cpp)
    target_include_directories(ccu_module_fixture PRIVATE ${SIMPLER_SRC}/common)
    ccu_host_case(test_ccu_module.cpp
        DEFINES CCU_MODULE_FIXTURE_PATH="$<TARGET_FILE:ccu_module_fixture>"
            CCU_WRONG_ABI_FIXTURE_PATH="$<TARGET_FILE:ccu_wrong_abi_fixture>"
        DEPENDS ccu_module_fixture ccu_wrong_abi_fixture
        LINK ${CMAKE_DL_LIBS})
else()
    set_property(GLOBAL APPEND PROPERTY SIMPLER_UT_SKIPPED_CASES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_ccu_module.cpp)
endif()
