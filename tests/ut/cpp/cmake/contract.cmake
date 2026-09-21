# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# The include contract for C++ unit tests: the one place that says which
# directories a target compiled against (arch, runtime, tier) may reach.
#
# It mirrors the two halves the product build composes:
#   tier half     <- src/<arch>/platform/<variant>/<tier>/CMakeLists.txt
#   runtime half  <- src/<arch>/runtime/<runtime>/build_config.py include_dirs
#
# Those two halves are still written out per platform-variant and per runtime on
# the product side; this file is a third statement of the same contract, not a
# derivation of it. Keeping it to one function is what makes a missing entry a
# single-site fix instead of a sweep over every target.

# Directories a target reaches regardless of which runtime it is compiled for.
# `tier` is aicpu for targets that link AICPU-side sources (they need the sim
# AICPU shims), host otherwise.
function(simpler_ut_common_includes arch tier out_var)
    set(_dirs "")
    if(arch)
        list(APPEND _dirs "${SIMPLER_SRC}/${arch}/platform/include")
    endif()
    if(tier STREQUAL "aicpu")
        list(APPEND _dirs "${SIMPLER_SRC}/common/platform/sim/aicpu")
    endif()
    list(APPEND _dirs
        "${SIMPLER_SRC}/common/platform/include"
        "${SIMPLER_SRC}/common/task_interface"
        "${SIMPLER_SRC}/common/log/include"
        # src/common carries the "utils/..." and "platform/..." prefixed headers.
        "${SIMPLER_SRC}/common"
        # Test-private fixtures are shared across test directories, so they are
        # included by their path from here rather than by bare name. Last among
        # the directories this function contributes, so a fixture never shadows
        # a header of the tree under test — simpler_ut_includes() appends
        # src/common/<runtime> after it, which is the one directory a fixture
        # could still shadow and the reason no file here carries a bare name
        # that tree also uses.
        "${CMAKE_SOURCE_DIR}"
    )
    set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

# Full contract for a target compiled against one runtime.
#
# The runtime's own directories precede src/common/<runtime>, so an arch-local
# header wins over a shared one of the same name — the ordering the runtime's
# build_config.py documents and relies on.
#
# src/common/<runtime> is load-bearing and not optional: the runtime-agnostic
# platform and worker sources reach this runtime's runtime.h / types.h / task_id.h
# by bare name, and resolve to the other runtime's copy when it is absent.
function(simpler_ut_includes arch runtime tier out_var)
    set(_rt "${SIMPLER_SRC}/${arch}/runtime/${runtime}")
    if(NOT IS_DIRECTORY "${_rt}")
        message(FATAL_ERROR "No such runtime tree: ${_rt}")
    endif()
    set(_dirs "${_rt}/orchestration")
    # Only host_build_graph splits out a host/ directory.
    if(IS_DIRECTORY "${_rt}/host")
        list(APPEND _dirs "${_rt}/host")
    endif()
    list(APPEND _dirs "${_rt}/runtime" "${_rt}/common")

    simpler_ut_common_includes("${arch}" "${tier}" _common)
    list(APPEND _dirs ${_common} "${SIMPLER_SRC}/common/${runtime}")

    set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()
