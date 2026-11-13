# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Host serializes an immutable graph; independent PEs initialize and execute it.
# Tile operation state uses the existing CPU simulator context.
SHARED = "../../../common/host_build_graph"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime", "common", "..", "../../platform/include", SHARED],
        "source_dirs": ["pe"],
    },
    "host": {
        "include_dirs": ["runtime", "common", "..", "../../platform/include", SHARED],
        "source_dirs": [
            "host",
            f"{SHARED}/host",
            f"{SHARED}/shared",
            f"{SHARED}/full_arena",
        ],
    },
    "orchestration": {
        "include_dirs": ["runtime", "common", "..", "../../platform/include", SHARED, f"{SHARED}/host/orchestration"],
        "source_dirs": [f"{SHARED}/host/orchestration"],
        # assert_impl is a C++ symbol, so unlike the extern "C" unified_log_error it
        # cannot be left for dlopen to resolve against the loader: the .so has to
        # carry its own copy. Named as a file because the directory holding it also
        # holds the arena sources, which the .so does not need.
        "source_files": [f"{SHARED}/shared/assert_support.cpp"],
    },
}
