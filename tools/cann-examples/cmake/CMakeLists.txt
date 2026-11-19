# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(aicore_reg_mapping_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(SIMPLER_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
include("${SIMPLER_REPO_ROOT}/cmake/host_log_sources.cmake")

add_executable(test_aicore_reg_mapping test_aicore_reg_mapping.cpp ${SIMPLER_HOST_LOG_SOURCES})
set_target_properties(test_aicore_reg_mapping PROPERTIES ENABLE_EXPORTS ON)
target_include_directories(test_aicore_reg_mapping PRIVATE
    "$ENV{ASCEND_HOME_PATH}/include"
    "${SIMPLER_REPO_ROOT}/src/common/platform/include"
    "${SIMPLER_REPO_ROOT}/src/common/log/include"
)
target_link_libraries(test_aicore_reg_mapping PRIVATE dl pthread)

enable_testing()
add_test(NAME aicore_reg_mapping COMMAND test_aicore_reg_mapping)
