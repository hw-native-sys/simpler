# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# percounter Runtime build configuration
# All paths are relative to this file's directory (src/a5/runtime/percounter/)
#
# percounter consumes the same device graph image host_build_graph does — the same
# ChipTaskSlotState, the same task kinds, the same orchestration .so. What differs is
# who dispatches: AICore lanes schedule themselves from host-built per-task counters
# instead of taking work from an AICPU dispatcher or a resident scheduler core.
#
# So this runtime compiles host_build_graph's sources rather than carrying copies, and
# overrides the entry points it replaces with strong symbols against host_build_graph's
# weak ones. The directories below therefore name a sibling runtime's tree; nothing is
# duplicated, and a fix landing in host_build_graph reaches this runtime unchanged.
#
# A runtime is whatever directory under src/{arch}/runtime/ holds a build_config.py
# (platform_info.discover_runtimes), and each one builds its own aicore .o, aicpu .so
# and host .so. One device runs one runtime per CANN process context, so a change here
# cannot reach a host_build_graph run.
#
# This file's own directories come first on every include path: a header placed here
# shadows the same-named one under host_build_graph, which is how this runtime replaces
# a header without editing its sibling. A .cpp cannot be shadowed that way — two
# definitions of one symbol is a link error, not an override — so a replaced
# translation unit relies on the weak marker on the sibling's definition instead.
#
# HBG's own build_config explains why "runtime" must precede SHARED and why the
# orchestration .so names assert_support.cpp file-wise; both apply here unchanged.
# Out of the default build set: a build that names no runtime leaves this one
# alone, and `--runtime percounter` (or SIMPLER_BUILD_RUNTIMES=percounter)
# builds it. Discovery is unaffected -- the runtime is still here and still
# selectable; what this controls is only whether an unqualified build compiles
# a third a5 runtime, which is what every a5 CI job does.
#
# The cost it keeps off those jobs is a whole runtime's compile: this tree
# shares host_build_graph's sources but not its include paths, so its objects
# are its own. The price is that nothing then proves this runtime still builds
# after a change to the sibling it compiles; whatever job turns it back on is
# what keeps it honest.
BUILD_BY_DEFAULT = False

# Present under a5 and not under a2a3, and meant to be.
#
# The unit tests reject that state unless the runtime claims it here. The
# arch-independent cases -- the platform tree's, which compile into every
# runtime's image -- are built once per runtime under both archs, so a runtime
# one arch lacks would send them looking for a directory that is not there.
# They are therefore built for the runtimes both archs carry, and this one is
# covered by its own cases under a5/runtime/percounter/ instead.
#
# Saying nothing is what a runtime added to one arch by mistake does, and that
# is the case worth stopping: the build would otherwise drop those cases for it
# and say so in a line nobody reads.
ARCH_EXCLUSIVE = True

HBG = "../host_build_graph"
SHARED = "../../../common/host_build_graph"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["aicore", "runtime", f"{HBG}/runtime", f"{HBG}/common", "..", SHARED],
        "source_dirs": ["aicore"],
    },
    "aicpu": {
        "include_dirs": ["aicpu", "runtime", f"{HBG}/aicpu", f"{HBG}/runtime", f"{HBG}/common", "..", SHARED],
        "source_dirs": [
            "aicpu",
            f"{HBG}/aicpu",
            f"{HBG}/runtime",
            f"{SHARED}/device",
            f"{SHARED}/shared",
        ],
    },
    "host": {
        "include_dirs": ["host", "runtime", f"{HBG}/host", f"{HBG}/runtime", f"{HBG}/common", "..", SHARED],
        "source_dirs": [
            "host",
            f"{HBG}/host",
            f"{SHARED}/host",
            f"{SHARED}/shared",
        ],
    },
    "orchestration": {
        "include_dirs": [f"{HBG}/runtime", f"{HBG}/common", "..", SHARED, f"{SHARED}/host/orchestration"],
        "source_dirs": [f"{SHARED}/host/orchestration"],
        "source_files": [f"{SHARED}/shared/assert_support.cpp"],
    },
}
