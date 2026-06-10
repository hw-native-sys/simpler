# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../build/output/bin/")

message(STATUS "TraCR: REAL_SOURCE_DIR: '${CMAKE_CURRENT_LIST_DIR}'")

# Paraver format configuration file
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/tracr/postprocessing/paraver/state.cfg
    ${BUILD_DIR}/state.cfg
    COPYONLY
)

add_executable(tracr_process ${CMAKE_CURRENT_LIST_DIR}/tracr/postprocessing/tracr_process.cpp)

tracr_enable(tracr_process)

# Set the output directory for the compiled executable
set_target_properties(tracr_process PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
)
