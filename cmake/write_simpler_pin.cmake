# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT DEFINED SIMPLER_PIN_SOURCE_DIR OR NOT DEFINED SIMPLER_PIN_OUTPUT_DIR)
    message(FATAL_ERROR "SIMPLER_PIN_SOURCE_DIR and SIMPLER_PIN_OUTPUT_DIR are required")
endif()

set(_simpler_pin "${SIMPLER_PIN_OUTPUT_DIR}/simpler.pin")
# A previous install must not leave a stale pin when the source is now a tarball.
file(REMOVE "${_simpler_pin}")

find_program(_simpler_git git)
if(NOT _simpler_git)
    return()
endif()

# A tarball unpacked within an unrelated Git checkout is not a Git source tree.
execute_process(
    COMMAND "${_simpler_git}" -C "${SIMPLER_PIN_SOURCE_DIR}" rev-parse --show-toplevel
    RESULT_VARIABLE _simpler_root_result
    OUTPUT_VARIABLE _simpler_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT _simpler_root_result EQUAL 0)
    return()
endif()
get_filename_component(_simpler_root "${_simpler_root}" REALPATH)
get_filename_component(_simpler_source_root "${SIMPLER_PIN_SOURCE_DIR}" REALPATH)
if(NOT _simpler_root STREQUAL _simpler_source_root)
    return()
endif()

execute_process(
    COMMAND "${_simpler_git}" -C "${SIMPLER_PIN_SOURCE_DIR}" rev-parse --verify HEAD
    RESULT_VARIABLE _simpler_revision_result
    OUTPUT_VARIABLE _simpler_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
string(LENGTH "${_simpler_revision}" _simpler_revision_length)
if(NOT _simpler_revision_result EQUAL 0 OR
   NOT _simpler_revision_length EQUAL 40 OR
   NOT _simpler_revision MATCHES "^[0-9a-f]+$")
    return()
endif()

execute_process(
    COMMAND "${_simpler_git}" -C "${SIMPLER_PIN_SOURCE_DIR}" status --porcelain --untracked-files=all
    RESULT_VARIABLE _simpler_status_result
    OUTPUT_VARIABLE _simpler_status
    ERROR_QUIET)
if(NOT _simpler_status_result EQUAL 0)
    return()
endif()
if(NOT _simpler_status STREQUAL "")
    set(_simpler_clean false)
else()
    # Ignored files in build inputs can change the package without changing HEAD.
    # Limit this check to inputs the wheel uses; root build/ and virtualenvs are
    # intentionally ignored, while generated caches below src/ are not shipped.
    execute_process(
        COMMAND "${_simpler_git}" -C "${SIMPLER_PIN_SOURCE_DIR}" status --porcelain --ignored --untracked-files=normal
            -- src cmake simpler_setup python/simpler python/bindings
            ":(exclude)**/__pycache__" ":(exclude)**/*.pyc" ":(exclude)**/*.pyo"
            ":(exclude)src/**/compile_commands.json"
        RESULT_VARIABLE _simpler_ignored_result
        OUTPUT_VARIABLE _simpler_ignored
        ERROR_QUIET)
    if(NOT _simpler_ignored_result EQUAL 0)
        return()
    endif()
    if(_simpler_ignored STREQUAL "")
        set(_simpler_clean true)
    else()
        set(_simpler_clean false)
    endif()
endif()

file(MAKE_DIRECTORY "${SIMPLER_PIN_OUTPUT_DIR}")
file(WRITE "${_simpler_pin}" "revision=${_simpler_revision}\nclean=${_simpler_clean}\n")
