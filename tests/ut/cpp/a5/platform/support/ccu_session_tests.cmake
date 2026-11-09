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
    foreach(kind user bad_abi)
        add_library(ccu_test_${kind}_module SHARED "${root}/tests/ut/cpp/a5/platform/support/ccu_user_module.cpp")
        target_compile_features(ccu_test_${kind}_module PRIVATE cxx_std_17)
        target_include_directories(ccu_test_${kind}_module PRIVATE "${root}/src/common")
    endforeach()
    target_compile_definitions(ccu_test_bad_abi_module PRIVATE CCU_TEST_BAD_ABI=1)
    target_compile_definitions(${target} PRIVATE
        CCU_TEST_USER_MODULE="$<TARGET_FILE:ccu_test_user_module>"
        CCU_TEST_BAD_ABI_MODULE="$<TARGET_FILE:ccu_test_bad_abi_module>")
    add_dependencies(${target} ccu_test_user_module ccu_test_bad_abi_module)
endfunction()
