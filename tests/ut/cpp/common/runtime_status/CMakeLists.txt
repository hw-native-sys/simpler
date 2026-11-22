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
# The A2/A3 TMR header also exposes the optional kernel-validation codes.
macro(runtime_status_case src)
    simpler_ut_plain_case(${src}
        INCLUDES
            ${SIMPLER_SRC}/common
            ${SIMPLER_SRC}/common/platform/include
            ${SIMPLER_SRC}/a2a3/runtime/tensormap_and_ringbuffer/common
        LABEL no_hardware ${ARGN})
endmacro()

simpler_ut_glob_cases(runtime_status_case)
