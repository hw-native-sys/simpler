# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Resolve the Ascend driver package, which ships libascend_hal.so and
# libdrvdsmi_host.so. It sits as a SIBLING of the CANN toolkit install
# (e.g. /usr/local/Ascend/{cann-9.0.0, driver}) and its libs are under
# lib64/driver, not lib64 — a tool that only searches under ASCEND_HOME_PATH
# fails to link with `cannot find -lascend_hal`.
#
# Every tool here that links ascend_hal includes this file, rather than each
# carrying its own guess. Three different spellings had accumulated: this
# resolution, a hardcoded /usr/local/Ascend/driver, and nothing at all — and the
# tool with nothing at all did not build.
#
# Resolution order:
#   1. cmake -DASCEND_DRIVER_PATH=/opt/ascend/driver
#   2. sibling of ASCEND_HOME_PATH, if it exists
#   3. the standard install at /usr/local/Ascend/driver
#
# Sets ASCEND_DRIVER_PATH, and fails the configure naming what is missing rather
# than leaving it to a link error that names only the library.

if(NOT DEFINED ASCEND_HOME_PATH)
    message(FATAL_ERROR "ascend_driver_path.cmake requires ASCEND_HOME_PATH to be set first")
endif()

if(NOT DEFINED ASCEND_DRIVER_PATH)
    get_filename_component(_ASCEND_PARENT "${ASCEND_HOME_PATH}" DIRECTORY)
    set(_DERIVED_DRIVER_PATH "${_ASCEND_PARENT}/driver")
    if(EXISTS "${_DERIVED_DRIVER_PATH}")
        set(ASCEND_DRIVER_PATH "${_DERIVED_DRIVER_PATH}")
    else()
        set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
    endif()
endif()

if(NOT EXISTS "${ASCEND_DRIVER_PATH}/lib64/driver/libascend_hal.so")
    message(FATAL_ERROR
        "libascend_hal.so not found under ${ASCEND_DRIVER_PATH}/lib64/driver/. "
        "Set ASCEND_DRIVER_PATH explicitly (cmake -DASCEND_DRIVER_PATH=...).")
endif()

message(STATUS "ASCEND_HOME_PATH   = ${ASCEND_HOME_PATH}")
message(STATUS "ASCEND_DRIVER_PATH = ${ASCEND_DRIVER_PATH}")
