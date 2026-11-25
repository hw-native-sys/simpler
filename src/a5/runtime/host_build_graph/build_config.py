# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# A5 host_build_graph records tasks on the Host, places calendar rows there,
# and materializes Graph invocations on the AICPU.

SHARED = "../../../common/host_build_graph"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["aicore", "runtime", "common", "..", SHARED],
        "source_dirs": ["aicore"],
    },
    "aicpu": {
        "include_dirs": ["aicpu", "runtime", "common", "..", SHARED],
        "source_dirs": ["aicpu", "runtime", f"{SHARED}/device", f"{SHARED}/shared/assert"],
    },
    "host": {
        "include_dirs": ["host/image", "host", "runtime", "common", "..", SHARED],
        "source_dirs": ["host", f"{SHARED}/host", f"{SHARED}/shared"],
    },
    "orchestration": {
        "include_dirs": ["runtime", "common", "..", SHARED, f"{SHARED}/host/orchestration"],
        "source_dirs": [f"{SHARED}/host/orchestration"],
        "source_files": [f"{SHARED}/shared/assert/assert_support.cpp"],
    },
}
