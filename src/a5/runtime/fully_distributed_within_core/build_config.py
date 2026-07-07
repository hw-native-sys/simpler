# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# fully_distributed_within_core runtime build configuration
# All paths are relative to this file's directory (src/runtime/fully_distributed_within_core/)
#
# Goal: orchestration + scheduling + execution run on the AI cores themselves in
# SPMD fashion, removing AICPU from orchestration/scheduling. See the design spec:
#   docs/fully_distributed_within_core.md
#
# This tree is currently re-based on the tensormap_and_ringbuffer runtime so it
# is discoverable and compiles; it reuses TensorMap, MixedKernels/ActiveMask,
# L0TaskArgs, the pto_orchestration_api submit API, and kernel-address
# resolution. The distributed model (claim race + per-core TensorMap + private
# task ring + global completion-flag ring) is layered on incrementally per the
# spec; the AICPU is reduced to an init/teardown stub.
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).

# The decentralized SPMD engine (dist_engine.{cpp,h}) is shared verbatim across
# arches from src/common/runtime/fully_distributed_within_core/. It is compiled
# into the AICPU .so only (dist_core_main runs on AICore worker threads via a
# function pointer). Arch-specific headers (runtime.h, pto_runtime2.h, ...) still
# resolve through each arch's own include_dirs, so the same source builds per-arch.
DIST_COMMON = "../../../common/runtime/fully_distributed_within_core"

BUILD_CONFIG = {
    "aicore": {
        # "../../../common" puts src/common on the CCEC include path so the
        # engine's transitive includes (utils/device_arena.h via pto_runtime2.h)
        # resolve — matching what the AICPU build already gets through the
        # platform CMake's ../../../../common flag. DIST_COMMON brings in the
        # shared dist_engine.{cpp,h}; onboard compiles dist_core_main into the
        # AICore binary (CCEC) rather than reaching it via a host function
        # pointer (sim). See docs/fully_distributed_within_core.md §16.
        "include_dirs": ["runtime", "common", "..", "../../../common", DIST_COMMON],
        "source_dirs": ["aicore", "orchestration", DIST_COMMON],
    },
    "aicpu": {
        "include_dirs": ["runtime", "common", "..", DIST_COMMON],
        "source_dirs": ["aicpu", "runtime", "orchestration", DIST_COMMON],
    },
    "host": {
        # DIST_COMMON is on the host include path so device_runner.cpp can read
        # DIST_SEG_RESERVE_BYTES from dist_engine.h (the fdwc shared-segment
        # reserve the host pre-allocates for AICore-MPU-visible DistGlobal/heap).
        "include_dirs": ["runtime", "common", "..", DIST_COMMON],
        "source_dirs": ["host", "runtime/shared", "orchestration"],
    },
    "orchestration": {"include_dirs": ["runtime", "orchestration", "common", ".."], "source_dirs": ["orchestration"]},
}
