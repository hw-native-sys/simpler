# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# The tracr.cmake directory
set(TRACR_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# This BUILD_TRACR is a Environment variable used to toggle the build of TraCR
# Use: BUILD_TRACR=ON pip install --no-build-isolation -e '.[test]'
# Default is 'OFF'
option(BUILD_TRACR "Enable TraCR" OFF)
if(DEFINED ENV{BUILD_TRACR})
    set(BUILD_TRACR $ENV{BUILD_TRACR})
endif()

function(tracr_enable target)
    message(STATUS "Enabling TraCR '${BUILD_TRACR}' for target: ${target}")

    if (NOT TARGET ${target})
        message(FATAL_ERROR "Target '${target}' does not exist.")
    endif()

    # Create the TraCR include directory path
    set(TRACR_INCLUDE_DIR
        ${TRACR_ROOT_DIR}/tracr/include
    )

    # Check if it even exists
    if (NOT EXISTS "${TRACR_INCLUDE_DIR}/tracr/tracr.hpp")
        message(FATAL_ERROR
            "tracr.hpp not found at ${TRACR_INCLUDE_DIR}/tracr/tracr.hpp"
        )
    endif()

    # Append the nlohmann json path as well
    set(TRACR_INCLUDE_DIR
        ${TRACR_INCLUDE_DIR}
        ${TRACR_ROOT_DIR}
        ${TRACR_ROOT_DIR}/tracr/extern
    )

    # --- include the directories ---
    # SYSTEM: TraCR and its vendored third-party headers (e.g. extern/nlohmann
    # json) are external to simpler and don't compile cleanly under the build's
    # -Wall -Wextra -Werror (modern clang flags nlohmann's deprecated literal
    # operators). Marking them system suppresses warnings from those headers.
    target_include_directories(${target} SYSTEM PRIVATE
        ${TRACR_INCLUDE_DIR}
    )

    # --- compiler flags of TraCR ---
    if (BUILD_TRACR)
        # Flag to enable/disable TraCR calls at compile time
        target_compile_definitions(${target} PRIVATE ENABLE_TRACR)

        # TraCR threads capacity (default is 1<<20 ~= 1 million traces per thread = ~17MB per thread buffer size)
        set(TRACR_CAPACITY "" CACHE STRING "Optional TraCR buffer capacity (empty = use internal default)")

        if(NOT "${TRACR_CAPACITY}" STREQUAL "")
            message(STATUS "TraCR adding capacity: ${TRACR_CAPACITY}")

            if(NOT TRACR_CAPACITY MATCHES "^[0-9]+$")
                message(FATAL_ERROR "TRACR_CAPACITY must be a positive integer")
            endif()

            target_compile_definitions(${target} PRIVATE
                TRACR_CAPACITY=${TRACR_CAPACITY}
            )
        endif()

        # As the traces are collected on the Ascend device,
        # there is no need to store them on the device filesystem.
        target_compile_definitions(${target} PRIVATE TRACR_DISABLE_FLUSH USE_HW_COUNTER)

        # TraCR full size buffer modes:
        # default (none):              Abort if buffer is full
        # TRACR_POLICY_PERIODIC:       If buffer is full, overwrite from the beginning
        # TRACR_POLICY_IGNORE_IF_FULL: If buffer is full, ignore incoming traces
        # if (TRACR_POLICY)
        #     target_compile_definitions(${target} PRIVATE TRACR_POLICY_PERIODIC)
        # endif()
        set(TRACR_POLICY "" CACHE STRING "TraCR policy (empty = use C++ default)")

        set_property(CACHE TRACR_POLICY PROPERTY STRINGS
            ""  # default: abort if full
            TRACR_POLICY_PERIODIC
            TRACR_POLICY_IGNORE_IF_FULL
        )

        if(NOT "${TRACR_POLICY}" STREQUAL "")
            if(TRACR_POLICY STREQUAL "TRACR_POLICY_PERIODIC")
                message(STATUS "TraCR adding policy: 'TRACR_POLICY_PERIODIC'")
                target_compile_definitions(${target} PRIVATE TRACR_POLICY_PERIODIC)
            elseif(TRACR_POLICY STREQUAL "TRACR_POLICY_IGNORE_IF_FULL")
                message(STATUS "TraCR adding policy: 'TRACR_POLICY_IGNORE_IF_FULL'")
                target_compile_definitions(${target} PRIVATE TRACR_POLICY_IGNORE_IF_FULL)
            else()
                message(FATAL_ERROR "Unknown TRACR_POLICY: ${TRACR_POLICY}")
            endif()
        else()
            message(STATUS "No TraCR policy given: using C++ default")
        endif()

        # Flag to enable TraCR debugging prints (TODO: Not yet working!)
        # if (TRACR_DEBUG)
        #     target_compile_definitions(${target} PRIVATE ENABLE_TRACR_DEBUG)
        # endif()
    endif()
endfunction()
