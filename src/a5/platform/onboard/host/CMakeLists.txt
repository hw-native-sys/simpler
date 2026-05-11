# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Build Host runtime using host/* + common/* + custom includes and sources (for `Runtime`)
# CMAKE_C_COMPILER and CMAKE_CXX_COMPILER are set when calling cmake
# CUSTOM_INCLUDE_DIRS and CUSTOM_SOURCE_DIRS are set when calling cmake
# ASCEND_HOME_PATH is set when calling cmake
cmake_minimum_required(VERSION 3.16.3)

project(host_runtime LANGUAGES C CXX)

# Build complete include list
set(CMAKE_CUSTOM_INCLUDE_DIRS "")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/task_interface")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/worker")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common")
list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/log/include")
if(DEFINED CUSTOM_INCLUDE_DIRS)
    foreach(INC_DIR ${CUSTOM_INCLUDE_DIRS})
        list(APPEND CMAKE_CUSTOM_INCLUDE_DIRS "${INC_DIR}")
    endforeach()
else()
    message(FATAL_ERROR "MUST set CUSTOM_INCLUDE_DIRS to build Host runtime")
endif()

# Build complete source list: core host sources + sources from CUSTOM_SOURCE_DIRS
set(HOST_RUNTIME_SOURCES "")
list(APPEND HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/device_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiling_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/platform_compile_info.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_regs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/l2_perf_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/pmu_collector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/host/tensor_dump_collector.cpp"
)
if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND HOST_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
else()
    message(STATUS "CUSTOM_SOURCE_DIRS not set, using only core sources")
endif()

# Create shared library
list(LENGTH HOST_RUNTIME_SOURCES NUM_HOST_RUNTIME_SOURCES)
message(STATUS "Host runtime: ${NUM_HOST_RUNTIME_SOURCES} source files")
message(VERBOSE "Host runtime sources: ${HOST_RUNTIME_SOURCES}")

# =============================================================================
# Chevron launch object (bisheng -xcce, classic CCE frontend)
#
# chevron_launch.cpp is a single translation unit that holds BOTH the AICore
# kernel definition and the host-side <<<>>> launch site. bisheng with -xcce
# (plus -Xhost-start/-Xhost-end) compiles it into a fatbin object: the device
# code is embedded inside the .o, and the <<<>>> call is expanded into the
# corresponding rtKernelLaunch* host code; bisheng also injects a constructor
# that registers the embedded fatbin at dlopen time. The mscatter testcase
# in pto-isa uses the same -xcce path successfully — but mscatter links its
# kernel .so through bisheng (driver passes --cce-fatobj-link itself);
# host_runtime.so here is linked by g++, so we don't pass that bisheng-only
# flag. GNU ld preserves the named ELF section that holds the fatbin, and
# the registration constructor runs at dlopen, so the chevron launch still
# wires up correctly.
#
# `aicore_execute` lives in the active runtime's aicore_executor.h as an
# inline __aicore__ function, so chevron_launch.cpp pulls it directly into
# its own TU via #include — no separate co-link step needed. The runtime's
# `aicore/` dir is exposed to chevron via build_config.py
# (host.include_dirs), which lands in CUSTOM_INCLUDE_DIRS / CHEVRON_INCLUDE_FLAGS.
#
# This is now the default a5 launch path; `SIMPLER_USE_LEGACY_LAUNCH=1`
# falls back to rtKernelLaunchWithHandleV2 at runtime. Platforms whose
# bisheng cannot produce this object (older CANN that rejects -xcce +
# -Xhost-start/-Xhost-end) are excluded by detect_buildable_platforms() so
# a5 onboard is not built there at all.
# =============================================================================
set(CHEVRON_LAUNCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../aicore/chevron_launch.cpp")
set(CHEVRON_LAUNCH_OBJ "${CMAKE_CURRENT_BINARY_DIR}/chevron_launch.o")

find_program(CHEVRON_BISHENG_CC
    NAMES bisheng
    PATHS "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ccec_compiler/bin"
    NO_DEFAULT_PATH
    REQUIRED
)
message(STATUS "Chevron launch compiler: ${CHEVRON_BISHENG_CC}")

set(CHEVRON_INCLUDE_FLAGS "")
foreach(INC_DIR ${CMAKE_CUSTOM_INCLUDE_DIRS})
    list(APPEND CHEVRON_INCLUDE_FLAGS "-I${INC_DIR}")
endforeach()
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../include")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../aicore")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${ASCEND_HOME_PATH}/include")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${ASCEND_HOME_PATH}/pkg_inc")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${ASCEND_HOME_PATH}/pkg_inc/runtime")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${ASCEND_HOME_PATH}/pkg_inc/profiling")
list(APPEND CHEVRON_INCLUDE_FLAGS "-I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime")

# Mix-arch (dav-c310) so bisheng emits both AIC and AIV variants of the same
# source from the __DAV_VEC__ branch in chevron_launch.cpp. -mllvm knobs +
# -DREGISTER_BASE mirror the mscatter testcase build that is known to work.
add_custom_command(
    OUTPUT ${CHEVRON_LAUNCH_OBJ}
    COMMAND ${CHEVRON_BISHENG_CC}
        -c -fPIC -O3 -g -std=c++17 -Wall
        -xcce
        -Xhost-start -Xhost-end
        --cce-aicore-arch=dav-c310
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -DREGISTER_BASE
        ${CHEVRON_INCLUDE_FLAGS}
        -o ${CHEVRON_LAUNCH_OBJ} ${CHEVRON_LAUNCH_SRC}
    DEPENDS ${CHEVRON_LAUNCH_SRC}
    COMMENT "Compiling chevron_launch.cpp with -xcce (mix dav-c310)"
    VERBATIM
)

list(APPEND HOST_RUNTIME_SOURCES ${CHEVRON_LAUNCH_OBJ})
set_source_files_properties(${CHEVRON_LAUNCH_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(host_runtime SHARED ${HOST_RUNTIME_SOURCES})

# C++ standard (applied only to C++ files)
set_target_properties(host_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Compile options (matching AICPU pattern)
target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
)

# Include directories - always include local headers
target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CUSTOM_INCLUDE_DIRS}
    PRIVATE
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/runtime
        ${ASCEND_HOME_PATH}/pkg_inc/profiling
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver
)

target_link_directories(host_runtime
    PRIVATE
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
)

# Link against CANN runtime libraries
# ascend_hal is dynamically loaded at runtime via dlopen in device_runner
# when performance profiling is enabled.
#
# Note: host_runtime.so is linked by g++, not bisheng, so we cannot pass
# bisheng-specific flags like --cce-fatobj-link here. The AICore fatbin
# embedded in chevron_launch.o by bisheng -xcce lives in a regular ELF
# section that GNU ld preserves through the shared-library link, and the
# runtime registers it via the constructor bisheng emitted into the same
# object — no link-time stitching is required on our side.
target_link_libraries(host_runtime
    PRIVATE
        runtime
        ascendcl
        dl
)

set_target_properties(host_runtime PROPERTIES OUTPUT_NAME "host_runtime")
