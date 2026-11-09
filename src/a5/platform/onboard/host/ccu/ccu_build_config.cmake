# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
file(SHA256 "${ASCEND_HOME_PATH}/lib64/libhcomm.so" CCU_HCOMM_FINGERPRINT)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ASCEND_HOME_PATH}/lib64/libhcomm.so")
get_filename_component(CCU_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../common" ABSOLUTE)
set(CCU_ABI_FINGERPRINT "${CCU_HCOMM_FINGERPRINT}:${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}")
foreach(header ccu_types ccu_builder ccu_host ccu_device)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CCU_COMMON_DIR}/platform_comm/ccu/${header}.h")
    file(SHA256 "${CCU_COMMON_DIR}/platform_comm/ccu/${header}.h" CCU_HEADER_FINGERPRINT)
    string(APPEND CCU_ABI_FINGERPRINT ":${CCU_HEADER_FINGERPRINT}")
endforeach()
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/ccu_service_program.cpp" CCU_SERVICE_FINGERPRINT)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccu_service_program.cpp")
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/ccu_v1_arithmetic.h" CCU_ARITHMETIC_FINGERPRINT)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccu_v1_arithmetic.h")
string(SHA256 CCU_SDK_FINGERPRINT "${CCU_ABI_FINGERPRINT}:${CCU_SERVICE_FINGERPRINT}:${CCU_ARITHMETIC_FINGERPRINT}")
set(CCU_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ccu-generated")
file(MAKE_DIRECTORY "${CCU_GENERATED_INCLUDE_DIR}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/ccu_build_config.h.in"
    "${CCU_GENERATED_INCLUDE_DIR}/ccu_build_config.h" @ONLY)
