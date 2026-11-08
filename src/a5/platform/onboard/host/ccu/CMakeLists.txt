# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(simpler_ccu LANGUAGES C CXX)
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}/include/hcomm/ccu/ccu_primitives.hpp")
    message(FATAL_ERROR "CCU requires the HCOMM development SDK")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/ccu_trigger.cmake")
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
include("${CMAKE_CURRENT_LIST_DIR}/ccu_build_config.cmake")
add_library(simpler_ccu SHARED ccu_session.cpp ccu_session_sdk.cpp ccu_service_program.cpp)
target_compile_features(simpler_ccu PUBLIC cxx_std_17)
target_include_directories(simpler_ccu PUBLIC "${CCU_COMMON_DIR}"
    PRIVATE "${CCU_GENERATED_INCLUDE_DIR}" "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/hcomm/ccu"
        "${ASCEND_HOME_PATH}/include/hccl" "${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu")
target_compile_options(simpler_ccu PRIVATE -Wall -Wextra -Werror)
target_link_options(simpler_ccu PRIVATE -Wl,--no-undefined)
target_link_directories(simpler_ccu PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(simpler_ccu PRIVATE -Wl,--no-as-needed hcomm ascendcl -Wl,--as-needed c_sec dl pthread OpenSSL::Crypto)
add_dependencies(simpler_ccu ccu_trigger)
install(TARGETS simpler_ccu LIBRARY DESTINATION lib)
install(FILES "${CCU_TRIGGER_ELF}" DESTINATION lib)
install(DIRECTORY "${CCU_COMMON_DIR}/platform_comm/ccu" DESTINATION include/platform_comm)

option(SIMPLER_CCU_BUILD_TESTS "Build CCU CPU tests and the two-rank A5 harness" OFF)
if(SIMPLER_CCU_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    get_filename_component(SIMPLER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." ABSOLUTE)
    add_executable(test_ccu_session "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/test_ccu_session.cpp" ccu_session.cpp)
    target_compile_features(test_ccu_session PRIVATE cxx_std_17)
    target_compile_options(test_ccu_session PRIVATE -Wall -Wextra -Werror)
    target_include_directories(test_ccu_session PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${CCU_COMMON_DIR}")
    target_link_libraries(test_ccu_session PRIVATE GTest::gtest_main OpenSSL::Crypto ${CMAKE_DL_LIBS} pthread)
    add_test(NAME ccu_session COMMAND test_ccu_session)
    include("${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/support/ccu_session_tests.cmake")
    ccu_session_test_modules(test_ccu_session "${SIMPLER_ROOT}")
    add_executable(test_ccu_session_onboard
        "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/support/ccu_session_onboard.cpp")
    target_compile_features(test_ccu_session_onboard PRIVATE cxx_std_17)
    target_include_directories(test_ccu_session_onboard PRIVATE "${ASCEND_HOME_PATH}/include")
    target_link_directories(test_ccu_session_onboard PRIVATE "${ASCEND_HOME_PATH}/lib64")
    target_link_libraries(test_ccu_session_onboard PRIVATE simpler_ccu ascendcl hcomm hccl pthread)
    set(CCU_ONBOARD_SOURCE "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/support/ccu_session_onboard_kernel.cpp")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ccu_session_onboard.o"
        COMMAND "${CCU_CCEC}" -c -O2 -std=c++17 -x cce --cce-aicore-only
            --cce-aicore-arch=dav-c310-vec -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -I "${CCU_COMMON_DIR}" "${CCU_ONBOARD_SOURCE}"
            -o "${CMAKE_CURRENT_BINARY_DIR}/ccu_session_onboard_vec.o"
        COMMAND "${CCU_LD}" -m aicorelinux -Ttext=0 -static -n -e ccu_session_onboard
            -o "${CMAKE_CURRENT_BINARY_DIR}/ccu_session_onboard.o"
            "${CMAKE_CURRENT_BINARY_DIR}/ccu_session_onboard_vec.o"
        DEPENDS "${CCU_ONBOARD_SOURCE}" "${CCU_COMMON_DIR}/platform_comm/ccu/ccu_device.h"
            "${CCU_COMMON_DIR}/platform_comm/ccu/ccu_types.h" VERBATIM)
    add_custom_target(ccu_session_onboard_kernel ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ccu_session_onboard.o")
    add_executable(test_ccu_service_program "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/test_ccu_service_program.cpp"
        ccu_service_program.cpp)
    target_compile_features(test_ccu_service_program PRIVATE cxx_std_17)
    target_compile_options(test_ccu_service_program PRIVATE -Wall -Wextra -Werror)
    target_include_directories(test_ccu_service_program PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${CCU_COMMON_DIR}"
        "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/hccl"
        "${ASCEND_HOME_PATH}/include/hcomm/ccu" "${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu")
    target_link_libraries(test_ccu_service_program PRIVATE GTest::gtest_main)
    add_test(NAME ccu_service_program COMMAND test_ccu_service_program)
    add_executable(test_ccu_session_sdk "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/test_ccu_session_sdk.cpp" ccu_session_sdk.cpp)
    target_compile_features(test_ccu_session_sdk PRIVATE cxx_std_17)
    target_compile_options(test_ccu_session_sdk PRIVATE -Wall -Wextra -Werror)
    target_include_directories(test_ccu_session_sdk PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${CCU_COMMON_DIR}"
        "${CCU_GENERATED_INCLUDE_DIR}"
        "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/hccl"
        "${ASCEND_HOME_PATH}/include/hcomm/ccu" "${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu")
    target_link_libraries(test_ccu_session_sdk PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS})
    add_test(NAME ccu_session_sdk COMMAND test_ccu_session_sdk)
endif()
