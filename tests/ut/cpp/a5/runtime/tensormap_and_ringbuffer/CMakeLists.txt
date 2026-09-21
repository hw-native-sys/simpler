# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a5/runtime/tensormap_and_ringbuffer/.

# Target names carry the a5_ prefix because the a2a3 targets and
# hierarchical/test_tensormap already own the bare names.

# Every case here covers this arch's tensormap_and_ringbuffer runtime.
macro(a5_tmr_case)
    tmr_case(${ARGV} ARCHS a5)
endmacro()

# Everything not named above is a plain case: one file, this runtime's whole
# object library, nothing else to say. Dropping a new test_*.cpp into this
# directory is all it takes — declaring one by hand above is what removes it
# from this sweep, so there is no separate list to keep in step.
simpler_ut_glob_cases(a5_tmr_case)
