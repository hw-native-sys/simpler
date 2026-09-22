# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/common/runtime_status/.
#
# The SIMPLER_ERROR_* codes the name table switches on live in a runtime's own
# common/runtime_status.h. Only tensormap_and_ringbuffer has one, and its a2a3
# and a5 copies are the same file byte for byte, so one build covers every
# context this table is compiled in.
macro(runtime_status_case src)
    simpler_ut_plain_case(${src}
        INCLUDES
            ${SIMPLER_SRC}/common
            ${SIMPLER_SRC}/common/platform/include
            ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/common
        LABEL no_hardware ${ARGN})
endmacro()

simpler_ut_glob_cases(runtime_status_case)
