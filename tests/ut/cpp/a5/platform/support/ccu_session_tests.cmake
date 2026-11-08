# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
function(ccu_session_test_modules target root)
    target_sources(${target} PRIVATE "${root}/tests/ut/cpp/a5/platform/support/ccu_session_fake.cpp")
    add_library(ccu_positional_module SHARED "${root}/tests/ut/cpp/a5/platform/support/ccu_positional_module.cpp")
    target_compile_features(ccu_positional_module PRIVATE cxx_std_17)
    target_include_directories(ccu_positional_module PRIVATE "${root}/src/common")
    target_compile_definitions(${target} PRIVATE CCU_TEST_POSITIONAL_MODULE="$<TARGET_FILE:ccu_positional_module>")
    add_dependencies(${target} ccu_positional_module)
endfunction()
