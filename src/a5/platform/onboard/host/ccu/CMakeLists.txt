# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
project(simpler_ccu LANGUAGES C CXX)
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}/include/hcomm/ccu/ccu_primitives.hpp")
    message(FATAL_ERROR "CCU requires the HCOMM development SDK")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/ccu_trigger.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ccu_build_config.cmake")
add_library(simpler_ccu SHARED ccu_execution_api.cpp ccu_execution_registry.cpp ccu_request_state.cpp
    ccu_request_validation.cpp ccu_device.cpp ccu_session.cpp ccu_session_sdk.cpp ccu_service_program.cpp)
target_compile_features(simpler_ccu PUBLIC cxx_std_17)
target_include_directories(simpler_ccu PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    PUBLIC "${CCU_COMMON_DIR}"
    PRIVATE "${CCU_GENERATED_INCLUDE_DIR}" "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/hcomm/ccu"
        "${ASCEND_HOME_PATH}/include/hccl" "${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu")
target_compile_options(simpler_ccu PRIVATE -Wall -Wextra -Werror)
target_link_options(simpler_ccu PRIVATE -Wl,--no-undefined)
target_link_directories(simpler_ccu PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(simpler_ccu PRIVATE -Wl,--no-as-needed hcomm ascendcl -Wl,--as-needed c_sec dl pthread OpenSSL::Crypto)
add_dependencies(simpler_ccu ccu_trigger)
install(TARGETS simpler_ccu LIBRARY DESTINATION lib)
install(FILES "${CCU_TRIGGER_ELF}" DESTINATION lib)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../../../include/host/ccu_execution.h" DESTINATION include/host)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../../../include/common/ccu_execution_types.h" DESTINATION include/common)
install(DIRECTORY "${CCU_COMMON_DIR}/platform_comm/ccu" DESTINATION include/platform_comm)
install(FILES "${CCU_GENERATED_INCLUDE_DIR}/ccu_build_config.h" DESTINATION include/platform_comm/ccu)

option(SIMPLER_CCU_BUILD_TESTS "Build CCU protocol and recording tests without a device" OFF)
if(SIMPLER_CCU_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    get_filename_component(SIMPLER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." ABSOLUTE)
    foreach(case execution_api execution_abi request_state request_validation)
        add_executable(test_ccu_${case} "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/test_ccu_${case}.cpp")
        target_compile_features(test_ccu_${case} PRIVATE cxx_std_17)
        target_include_directories(test_ccu_${case} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../include")
        target_link_libraries(test_ccu_${case} PRIVATE GTest::gtest_main pthread)
        add_test(NAME ccu_${case} COMMAND test_ccu_${case})
        set_tests_properties(ccu_${case} PROPERTIES TIMEOUT 30)
    endforeach()
    target_sources(test_ccu_execution_api PRIVATE ccu_execution_api.cpp ccu_execution_registry.cpp
        ccu_request_state.cpp ccu_request_validation.cpp)
    target_sources(test_ccu_request_state PRIVATE ccu_request_state.cpp)
    target_sources(test_ccu_request_validation PRIVATE ccu_request_validation.cpp)
    add_executable(test_ccu_session "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/test_ccu_session.cpp" ccu_session.cpp)
    target_compile_features(test_ccu_session PRIVATE cxx_std_17)
    target_compile_options(test_ccu_session PRIVATE -Wall -Wextra -Werror)
    target_include_directories(test_ccu_session PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${CCU_COMMON_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/../../../include")
    target_link_libraries(test_ccu_session PRIVATE GTest::gtest_main OpenSSL::Crypto ${CMAKE_DL_LIBS} pthread)
    add_test(NAME ccu_session COMMAND test_ccu_session)
    include("${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/support/ccu_session_tests.cmake")
    ccu_session_test_modules(test_ccu_session "${SIMPLER_ROOT}")
    set(CCU_COMPILE_PROBE "${SIMPLER_ROOT}/tests/ut/cpp/a5/platform/support/ccu_device_compile.cpp")
    file(GLOB CCU_PUBLIC_HEADERS "${CCU_COMMON_DIR}/platform_comm/ccu/*.h")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aiv.o"
        COMMAND "${CCU_CCEC}" -c -O2 -std=c++17 -x cce --cce-aicore-only
            --cce-aicore-arch=dav-c310-vec -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -I "${CCU_COMMON_DIR}" -I "${CCU_COMMON_DIR}/task_interface"
            "${CCU_COMPILE_PROBE}" -o "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aiv.o"
        DEPENDS "${CCU_COMPILE_PROBE}" ${CCU_PUBLIC_HEADERS} VERBATIM)
    add_custom_target(ccu_device_aiv_check ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aiv.o")
    find_program(CCU_AARCH64_CXX aarch64-linux-gnu-g++)
    if(CCU_AARCH64_CXX)
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aicpu.o"
            COMMAND "${CCU_AARCH64_CXX}" -c -O2 -std=c++17 -Wall -Wextra -Werror
                -I "${CCU_COMMON_DIR}" -I "${CCU_COMMON_DIR}/task_interface"
                "${CCU_COMPILE_PROBE}" -o "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aicpu.o"
            DEPENDS "${CCU_COMPILE_PROBE}" ${CCU_PUBLIC_HEADERS} VERBATIM)
        add_custom_target(ccu_device_aicpu_check ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ccu_device_aicpu.o")
    endif()
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
        "${CMAKE_CURRENT_LIST_DIR}/../../../include" "${CCU_GENERATED_INCLUDE_DIR}"
        "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/hccl"
        "${ASCEND_HOME_PATH}/include/hcomm/ccu" "${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu")
    target_link_libraries(test_ccu_session_sdk PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS})
    add_test(NAME ccu_session_sdk COMMAND test_ccu_session_sdk)
    foreach(case ccu_device_contract comm_workspaces)
        add_executable(test_${case} "${SIMPLER_ROOT}/tests/ut/cpp/common/platform/test_${case}.cpp")
        target_compile_features(test_${case} PRIVATE cxx_std_17)
        target_include_directories(test_${case} PRIVATE "${SIMPLER_ROOT}/src/common")
        target_link_libraries(test_${case} PRIVATE GTest::gtest_main pthread)
        add_test(NAME ${case} COMMAND test_${case})
    endforeach()
endif()
