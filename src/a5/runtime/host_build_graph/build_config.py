# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Runtime build configuration
# All paths are relative to this file's directory (src/runtime/)
#
# "aicore" appears in the include_dirs of both the aicore and host targets:
# aicore_execute() lives in aicore/aicore_executor.h as an inline __aicore__
# function so kernel.cpp (legacy AICore launch) and chevron_launch.cpp
# (chevron launch, compiled into the host SO via bisheng -xcce as a single
# host+device TU) can each pull it into their own TU without a separate
# .cpp to co-link.

BUILD_CONFIG = {
    "aicore": {"include_dirs": ["runtime", "aicore"], "source_dirs": ["aicore", "runtime"]},
    "aicpu": {"include_dirs": ["runtime"], "source_dirs": ["aicpu", "runtime"]},
    "host": {
        "include_dirs": ["runtime", "orchestration", "aicore"],
        "source_dirs": ["host", "runtime"],
    },
    "orchestration": {"include_dirs": ["runtime", "orchestration"], "source_dirs": []},
}
