# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Source-tree root and runtime discovery for the C++ unit tests.
#
# tests/ut/cpp is configured as its own CMake project root (cmake -S tests/ut/cpp),
# so CMAKE_SOURCE_DIR points here rather than at the repository. SIMPLER_REPO_ROOT
# and SIMPLER_SRC are the one place that crosses back out to the tree under test.

get_filename_component(SIMPLER_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(SIMPLER_SRC "${SIMPLER_REPO_ROOT}/src")

if(NOT IS_DIRECTORY "${SIMPLER_SRC}")
    message(FATAL_ERROR "SIMPLER_SRC does not resolve to a directory: ${SIMPLER_SRC}")
endif()

# A directory under src/<arch>/runtime/ is a runtime exactly when it carries a
# build_config.py. This is the same criterion as
# simpler_setup/platform_info.py::discover_runtimes — two implementations of one
# rule, not two copies of one name list, so adding a runtime requires no edit on
# either side.
function(simpler_discover_runtimes arch out_var)
    file(GLOB _configs CONFIGURE_DEPENDS "${SIMPLER_SRC}/${arch}/runtime/*/build_config.py")
    set(_runtimes "")
    foreach(_config ${_configs})
        get_filename_component(_dir "${_config}" DIRECTORY)
        get_filename_component(_name "${_dir}" NAME)
        list(APPEND _runtimes "${_name}")
    endforeach()
    list(SORT _runtimes)
    if(NOT _runtimes)
        message(FATAL_ERROR "No runtime found under ${SIMPLER_SRC}/${arch}/runtime")
    endif()
    set(${out_var} "${_runtimes}" PARENT_SCOPE)
endfunction()

# SIMPLER_RUNTIMES: the runtimes every arch-independent target is built for.
#
# A platform or log source has no build of its own — it compiles into each
# runtime's image, and resolves task_id.h / runtime.h to that runtime's copy.
# A target covering such a source is therefore built once per runtime, and this
# list is the only place that says how many that is.
#
# The two archs must agree: a runtime present under one and not the other leaves
# every target built from the missing side uncovered, with nothing to report it.
# The check below turns that into a configure-time error.
simpler_discover_runtimes(a2a3 _a2a3_runtimes)
simpler_discover_runtimes(a5 _a5_runtimes)
if(NOT _a2a3_runtimes STREQUAL _a5_runtimes)
    message(FATAL_ERROR
        "Runtime sets differ between archs — a2a3 has [${_a2a3_runtimes}], "
        "a5 has [${_a5_runtimes}]. Add the missing build_config.py before "
        "building the unit tests.")
endif()
set(SIMPLER_RUNTIMES "${_a2a3_runtimes}")

# A discovered runtime must have unit tests of its own.
#
# Discovery alone is not enough to make adding a runtime free. The cases that
# repeat per runtime — the platform tree's, which compile into every runtime's
# image — start building for the new one immediately, and fail on a bare-name
# header they now resolve into a directory that does not exist. Left to the
# compiler, that arrives as
#
#     scheduler_profiling.h:21: fatal error: task_id.h: No such file or directory
#
# from a file the author never touched, with nothing connecting it to the
# runtime they added.
#
# The requirement is checked rather than listed: what the tree needs is the
# mirror directory, so the check asks whether the mirror directory is there. A
# hand-kept list of supported runtimes would be one more place to edit, and
# could disagree with the tree it claims to describe.
foreach(_rt ${SIMPLER_RUNTIMES})
    if(NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../common/${_rt}")
        message(FATAL_ERROR
            "Runtime '${_rt}' has a build_config.py but no unit tests.\n"
            "Create tests/ut/cpp/common/${_rt}/ for the parts both arches share, "
            "and tests/ut/cpp/{a2a3,a5}/runtime/${_rt}/ for each arch's own, then "
            "add_subdirectory them. Each needs the object library its cases link, "
            "the way common/tensormap_and_ringbuffer/ declares <arch>_tmr_objs.\n"
            "Until then the per-runtime platform cases build for '${_rt}' and "
            "cannot compile: they resolve task_id.h into src/common/${_rt}/.")
    endif()
endforeach()

# The short tag a runtime goes by in target names, and back.
#
# `hbg` and `tmr` are the tree's own abbreviations, used throughout src/
# comments and docs, and every test target built for a runtime carries one.
# These two functions are the single place the two spellings are tied together;
# neither guesses.
#
# A runtime with no tag keeps its full name rather than failing: a runtime is
# discovered from its build_config.py alone, and adding one must not need an
# edit here to build.
function(simpler_runtime_tag runtime out_var)
    if(runtime STREQUAL "host_build_graph")
        set(${out_var} hbg PARENT_SCOPE)
    elseif(runtime STREQUAL "tensormap_and_ringbuffer")
        set(${out_var} tmr PARENT_SCOPE)
    else()
        set(${out_var} "${runtime}" PARENT_SCOPE)
    endif()
endfunction()

# The runtime a short tag names, found by asking every runtime for its tag.
#
# Matching against the discovered set rather than testing for one name and
# falling through: a fallthrough answers for tags it has never seen, so a third
# runtime's cases would silently take the second one's include contract. No
# match is an error, because there is no right answer to guess.
function(simpler_runtime_from_tag tag out_var)
    foreach(_candidate ${SIMPLER_RUNTIMES})
        simpler_runtime_tag("${_candidate}" _candidate_tag)
        if(tag STREQUAL _candidate_tag)
            set(${out_var} "${_candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR
        "No runtime goes by the tag '${tag}'. Known: [${SIMPLER_RUNTIMES}], "
        "whose tags come from simpler_runtime_tag() in this file.")
endfunction()
