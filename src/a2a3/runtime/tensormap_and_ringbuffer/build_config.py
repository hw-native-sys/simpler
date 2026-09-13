# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Tensormap and Ringbuffer Runtime build configuration
# All paths are relative to this file's directory (src/runtime/tensormap_and_ringbuffer/)
#
# This is a device-orchestration runtime where:
# - AICPU thread 3 runs the orchestrator (builds task graph on device)
# - AICPU threads 0/1/2 run schedulers (dispatch tasks to AICore)
# - AICore executes tasks via an aligned DispatchPayload + pre-built dispatch_args
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).
#
# src/common/tensormap_and_ringbuffer holds the sources shared with the other
# architecture. Its .cpp files sit under a "host" subdirectory naming the target
# that compiles them; headers stay flat there. Those sources still include this
# runtime's headers by bare name (dep_gen_replay.cpp takes dep_compute.h,
# tensormap.h and tensor.h) and resolve them to whichever architecture's
# "runtime" directory is on the include path, which is what lets one source
# produce per-arch object code.
SHARED = "../../../common/tensormap_and_ringbuffer"

BUILD_CONFIG = {
    "aicore": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["aicore", "orchestration"]},
    "aicpu": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["aicpu", "runtime", "orchestration"]},
    "host": {
        "include_dirs": ["runtime", "common", ".."],
        "source_dirs": ["host", "runtime/shared", "orchestration", f"{SHARED}/host"],
    },
    "orchestration": {"include_dirs": ["runtime", "orchestration", "common", ".."], "source_dirs": ["orchestration"]},
}
