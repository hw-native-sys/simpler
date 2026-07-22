# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Replay Graph Runtime build configuration
# All paths are relative to this file's directory (src/a2a3/runtime/replay_graph/)
#
# This is a device-orchestration runtime where:
# - The orchestrator publishes device-built graphs at explicit boundaries
# - Scheduler threads execute a published graph while the next graph is built
# - AICore executes tasks via an aligned PTO2DispatchPayload + pre-built dispatch_args
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).

BUILD_CONFIG = {
    "aicore": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["aicore", "orchestration"]},
    "aicpu": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["aicpu", "runtime", "orchestration"]},
    "host": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["host", "runtime/shared", "orchestration"]},
    "orchestration": {"include_dirs": ["runtime", "orchestration", "common", ".."], "source_dirs": ["orchestration"]},
}
