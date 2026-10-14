# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# host_build_graph Runtime build configuration
# All paths are relative to this file's directory (src/runtime/host_build_graph/)
#
# This is the host-orchestration variant of tensormap_and_ringbuffer: it shares
# the same scheduler, ring buffers, and shared-memory layout, differing only in
# WHEN the orchestrator runs.
# - Host runs the orchestrator to completion, populating SM + arena, then H2Ds
#   the image to device (vs tensormap, where AICPU thread N-1 orchestrates on
#   device concurrently with the scheduler threads)
# - AICPU threads 0..N-1 all run schedulers (no on-device orchestrator thread)
# - AICore executes tasks via an aligned DispatchPayload + pre-built dispatch_args
#
# The orchestration .so is built from src/common/host_build_graph/host/orchestration:
# orchestration runs on the host for this runtime, so its API headers and the
# context binding it exports live under "host" like everything else host-only. The
# .so also compiles shared/assert_support.cpp, named file-wise below: assert_impl is
# what every always_assert expands into, and being a C++ symbol it cannot be left
# for dlopen to resolve the way extern "C" unified_log_error is.
#
# src/common/host_build_graph holds the sources shared with the other architecture.
# Its .cpp files sit under a "host", "device" or "shared" subdirectory naming the
# targets that compile them, because every source_dirs entry is collected by a
# recursive glob: "host" code uses the STL containers the AICPU target forbids, so it
# has to live outside any directory that target names.
#
# Headers stay flat there, except in host/orchestration: the API header includes
# "tensor.h", which must resolve along the include path to the arch-local
# runtime/tensor.h shim for the `using Tensor` alias that shim declares. A quoted
# include searches its own directory first, so a header sitting next to the flat
# tensor.h here would bind to this directory's definition and lose that alias.
#
# That directory is also on every target's include path, which is what lets the
# runtime-agnostic platform and worker sources reach this runtime's runtime.h and
# types.h by bare name: they resolve here when compiled for host_build_graph and to
# tensormap_and_ringbuffer's own copies when compiled for that runtime. "runtime"
# precedes it, so an arch-local header still wins over a shared one of the same name.
SHARED = "../../../common/host_build_graph"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime", "common", "..", SHARED],
        "source_dirs": ["aicore"],
    },
    "aicpu": {
        "include_dirs": ["runtime", "common", "..", SHARED],
        "source_dirs": [
            "aicpu",
            "runtime",
            f"{SHARED}/device",
            f"{SHARED}/shared",
        ],
    },
    "host": {
        "include_dirs": ["runtime", "common", "..", SHARED],
        "source_dirs": [
            "host",
            f"{SHARED}/host",
            f"{SHARED}/shared",
        ],
    },
    "orchestration": {
        "include_dirs": ["runtime", "common", "..", SHARED, f"{SHARED}/host/orchestration"],
        "source_dirs": [f"{SHARED}/host/orchestration"],
        # assert_impl is a C++ symbol, so unlike the extern "C" unified_log_error it
        # cannot be left for dlopen to resolve against the loader: the .so has to
        # carry its own copy. Named as a file because the directory holding it also
        # holds the arena sources, which the .so does not need.
        "source_files": [f"{SHARED}/shared/assert_support.cpp"],
    },
}
