# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
find_program(CCU_CCEC ccec HINTS "${ASCEND_HOME_PATH}/bin" REQUIRED)
find_program(CCU_LD ld.lld HINTS "${ASCEND_HOME_PATH}/bin" NO_DEFAULT_PATH REQUIRED)
get_filename_component(CCU_PLATFORM_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CCU_TRIGGER_SOURCE "${CCU_PLATFORM_DIR}/aicore/ccu_trigger.cpp")
set(CCU_TRIGGER_PART "${CMAKE_CURRENT_BINARY_DIR}/ccu_trigger_vec.o")
set(CCU_TRIGGER_ELF "${CMAKE_CURRENT_BINARY_DIR}/ccu_trigger.o")
add_custom_command(OUTPUT "${CCU_TRIGGER_PART}"
    COMMAND "${CCU_CCEC}" -c -O2 -std=c++17 -x cce --cce-aicore-only
        --cce-aicore-arch=dav-c310-vec -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -MD -MF "${CCU_TRIGGER_PART}.d" -o "${CCU_TRIGGER_PART}" "${CCU_TRIGGER_SOURCE}"
    DEPENDS "${CCU_TRIGGER_SOURCE}" DEPFILE "${CCU_TRIGGER_PART}.d" VERBATIM)
add_custom_command(OUTPUT "${CCU_TRIGGER_ELF}"
    COMMAND "${CCU_LD}" -m aicorelinux -Ttext=0 -static -n -e simpler_ccu_trigger
        -o "${CCU_TRIGGER_ELF}" "${CCU_TRIGGER_PART}"
    DEPENDS "${CCU_TRIGGER_PART}" VERBATIM)
add_custom_target(ccu_trigger ALL DEPENDS "${CCU_TRIGGER_ELF}")
