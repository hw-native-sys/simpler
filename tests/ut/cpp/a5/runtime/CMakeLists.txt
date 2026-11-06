# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for the parts of src/a5/runtime/ that every runtime carries.

# Tests for the parts of src/a5/runtime/ that every runtime carries, so a case
# here is declared once for each of them. a2a3 has its own mailbox case under
# a2a3/runtime/tensormap_and_ringbuffer/, covering that runtime only.
macro(a5_runtime_case src)
    hbg_case(${src} ARCHS a5 ${ARGN})
    tmr_case(${src} ARCHS a5 ${ARGN})
endmacro()

simpler_ut_glob_cases(a5_runtime_case)

add_subdirectory(host_build_graph)
add_subdirectory(tensormap_and_ringbuffer)
