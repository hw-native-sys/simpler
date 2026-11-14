# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(simpler_ccu_broadcast LANGUAGES CXX)

set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
if(NOT CANN_ROOT)
    message(FATAL_ERROR "Source the target CANN set_env.sh before building this example")
endif()
set(SDK_INCLUDE_HINTS
    "${CANN_ROOT}/include"
    "${CANN_ROOT}/pkg_inc"
    "${CANN_ROOT}/include/experiment"
    "${CANN_ROOT}/include/experimental"
    "${CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
)
find_path(ACL_INCLUDE acl/acl.h PATHS ${SDK_INCLUDE_HINTS} NO_DEFAULT_PATH)
find_path(HCCL_INCLUDE hccl/hccl_res.h PATHS ${SDK_INCLUDE_HINTS} NO_DEFAULT_PATH)
find_path(CCU_INCLUDE hcomm/ccu/ccu_primitives.hpp PATHS ${SDK_INCLUDE_HINTS} NO_DEFAULT_PATH)
if(NOT ACL_INCLUDE OR NOT HCCL_INCLUDE OR NOT CCU_INCLUDE)
    message(FATAL_ERROR
        "This example requires CANN 9.3 HCOMM CCU C/DSL headers, including hcomm/ccu/ccu_primitives.hpp.")
endif()
set(SDK_LIBRARY_HINTS "${CANN_ROOT}/lib64" "${CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
find_library(ASCENDCL_LIBRARY ascendcl PATHS ${SDK_LIBRARY_HINTS} NO_DEFAULT_PATH)
find_library(RUNTIME_LIBRARY runtime PATHS ${SDK_LIBRARY_HINTS} NO_DEFAULT_PATH)
find_library(SECUREC_LIBRARY c_sec PATHS ${SDK_LIBRARY_HINTS} NO_DEFAULT_PATH)
# Ccu* and HcommCcu* public entry points are provided by libhcomm.
find_library(HCOMM_LIBRARY hcomm PATHS ${SDK_LIBRARY_HINTS} NO_DEFAULT_PATH)
find_library(HCCL_LIBRARY hccl PATHS ${SDK_LIBRARY_HINTS} NO_DEFAULT_PATH)
if(NOT ASCENDCL_LIBRARY OR NOT RUNTIME_LIBRARY OR NOT HCOMM_LIBRARY OR NOT HCCL_LIBRARY OR NOT SECUREC_LIBRARY)
    message(FATAL_ERROR "CANN ascendcl/runtime/hcomm/hccl/c_sec libraries were not found below ASCEND_HOME_PATH")
endif()
add_library(ccu_broadcast_session SHARED host/ccu_session.cpp)
target_compile_features(ccu_broadcast_session PRIVATE cxx_std_17)
target_compile_options(ccu_broadcast_session PRIVATE -Wall -Wextra)
target_link_options(ccu_broadcast_session PRIVATE -Wl,--no-undefined)
target_include_directories(ccu_broadcast_session SYSTEM PRIVATE
    ${ACL_INCLUDE} ${HCCL_INCLUDE} ${CCU_INCLUDE}
    "${HCCL_INCLUDE}/hccl" "${CCU_INCLUDE}/hcomm" "${CCU_INCLUDE}/hcomm/ccu"
    "${CANN_ROOT}/pkg_inc/hcomm/ccu" "${CANN_ROOT}/include/experiment" "${CANN_ROOT}/include/experiment/runtime")
target_link_libraries(ccu_broadcast_session PRIVATE
    -Wl,--no-as-needed ${ASCENDCL_LIBRARY} ${RUNTIME_LIBRARY} ${HCOMM_LIBRARY}
    ${HCCL_LIBRARY} ${SECUREC_LIBRARY} -Wl,--as-needed dl pthread)
