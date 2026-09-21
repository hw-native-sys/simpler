# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Test registration for targets built once per runtime.
#
# A per-runtime target is one of N binaries built from the same source, and they
# run concurrently under `ctest -j`. Anything such a test writes to a path
# relative to its working directory is therefore shared between all N of them,
# and the collision is invisible: the loser reads back an artifact a sibling
# wrote, or none at all, which reads as a flake rather than as shared state.
#
# Registering through this function gives each target a working directory of its
# own, so a relative path written by one is unreachable from another. This is the
# isolation the exporting tests assume: test_chip_swimlane_collector.cpp writes
# "under the process CWD, so a caller can run this binary with its working
# directory set to a scratch dir", and tests/ut/py/test_swimlane_export_run_identity.py
# invokes it that way (cwd=tmp_path, one per runtime).
function(simpler_ut_add_test name)
    set(_workdir "${CMAKE_CURRENT_BINARY_DIR}/test_workdir/${name}")
    file(MAKE_DIRECTORY "${_workdir}")
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY "${_workdir}")
endfunction()

# The one way a unit-test executable is built and registered.
#
# Everything a target can differ by is a keyword here, so a directory states
# what its cases need rather than picking from a set of near-identical helpers
# named after the combinations:
#
#   NAME             target name
#   SOURCES          the case plus anything it compiles in
#   ARCH / RUNTIME   which include contract applies; omitted when a target
#                    inherits its paths from an OBJECT library instead
#   TIER             aicpu for targets linking AICPU-side sources, else host
#   OBJS             OBJECT libraries to link, whose PUBLIC include dirs carry
#                    the contract for targets that name no ARCH
#   INCLUDES         directories beyond the contract
#   INCLUDES_BEFORE  directories that must precede every other, including the
#                    gtest prefix, for a case whose own headers stand in for
#                    ones an SDK may also have installed there
#   DEFINES          compile definitions
#   COMPILE_OPTIONS  extra compile flags
#   LINK             libraries beyond gtest and pthread
#   LINK_OPTIONS     extra link flags
#   DEPENDS          targets that must be built first, for a case that dlopens one
#   LABEL            ctest label
#   TIMEOUT          ctest timeout in seconds
#   RESOURCES        ctest RESOURCE_GROUPS, for a case that claims a device
#   PRIVATE_WORKDIR  register through simpler_ut_add_test, which every target
#                    built more than once from the same source needs so their
#                    relative-path writes cannot reach each other
function(simpler_ut_add_target)
    cmake_parse_arguments(T "PRIVATE_WORKDIR" "NAME;ARCH;RUNTIME;TIER;LABEL;TIMEOUT;RESOURCES"
                            "SOURCES;OBJS;INCLUDES;INCLUDES_BEFORE;DEFINES;COMPILE_OPTIONS;LINK;LINK_OPTIONS;DEPENDS" ${ARGN})
    if(NOT T_NAME OR NOT T_SOURCES)
        message(FATAL_ERROR "simpler_ut_add_target needs NAME and SOURCES")
    endif()
    if(T_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "simpler_ut_add_target: unknown arguments: ${T_UNPARSED_ARGUMENTS}")
    endif()

    add_executable(${T_NAME} ${T_SOURCES})

    # Record which case files this target consumed, so the directory's glob
    # knows what is already declared and takes only the rest.
    foreach(_s ${T_SOURCES})
        get_filename_component(_n "${_s}" NAME)
        if(_n MATCHES "^test_.*\\.cpp$")
            set_property(GLOBAL APPEND PROPERTY SIMPLER_UT_DECLARED_CASES
                         "${CMAKE_CURRENT_SOURCE_DIR}/${_n}")
        endif()
    endforeach()

    set(_dirs ${GTEST_INCLUDE_DIRS} ${T_INCLUDES})
    if(T_ARCH AND T_RUNTIME)
        if(NOT T_TIER)
            set(T_TIER host)
        endif()
        simpler_ut_includes(${T_ARCH} ${T_RUNTIME} ${T_TIER} _contract)
        list(APPEND _dirs ${_contract})
    endif()
    target_include_directories(${T_NAME} PRIVATE ${_dirs})
    if(T_INCLUDES_BEFORE)
        target_include_directories(${T_NAME} BEFORE PRIVATE ${T_INCLUDES_BEFORE})
    endif()

    if(T_DEFINES)
        target_compile_definitions(${T_NAME} PRIVATE ${T_DEFINES})
    endif()

    if(T_COMPILE_OPTIONS)
        target_compile_options(${T_NAME} PRIVATE ${T_COMPILE_OPTIONS})
    endif()

    target_link_libraries(${T_NAME} PRIVATE
        ${T_OBJS}
        ${GTEST_MAIN_LIB}
        ${GTEST_LIB}
        ${T_LINK}
        pthread
    )
    if(T_LINK_OPTIONS)
        target_link_options(${T_NAME} PRIVATE ${T_LINK_OPTIONS})
    endif()
    if(T_DEPENDS)
        add_dependencies(${T_NAME} ${T_DEPENDS})
    endif()

    if(T_PRIVATE_WORKDIR)
        simpler_ut_add_test(${T_NAME})
    else()
        add_test(NAME ${T_NAME} COMMAND ${T_NAME})
    endif()
    if(T_LABEL)
        set_tests_properties(${T_NAME} PROPERTIES LABELS "${T_LABEL}")
    endif()
    if(T_TIMEOUT)
        set_tests_properties(${T_NAME} PROPERTIES TIMEOUT ${T_TIMEOUT})
    endif()
    if(T_RESOURCES)
        set_tests_properties(${T_NAME} PROPERTIES RESOURCE_GROUPS "${T_RESOURCES}")
    endif()
endfunction()

# One runtime case, built once per arch.
#
# The sources here are arch-independent while the runtime they cover compiles
# into each arch's image against that arch's headers, so a case covered under
# one arch only is covered under one arch only. Declaring it once removes the
# chance of that happening by omission.
#
#   ORCH            also link <arch>_<runtime>_orch_objs, where the runtime has
#                   one, for a case driving the real orchestrator
#   SOURCES         extra sources; @arch@ in an entry expands per iteration
#   INCLUDES        extra include directories, @arch@ likewise
#   INCLUDES_BEFORE directories that must precede every other, @arch@ likewise
#   LINK            libraries beyond gtest, @arch@ likewise
#   DEFINES         extra compile definitions
#   COMPILE_OPTIONS extra compile flags
#   LINK_OPTIONS    extra link flags
#   A2A3_SOURCES    sources only that arch's build needs
#   A5_SOURCES
#   NO_OBJS         skip the runtime object library, for a case that supplies its
#                   own definition of something the library's stubs also define
#   ARCHS           override the default pair, for a case that genuinely covers one
#   TIMEOUT         ctest timeout in seconds, for a case that can hang rather than
#                   fail — one whose wait has a counterpart that may stop arriving
#                   leaves its assertion unreached and its thread unjoined, and
#                   without a limit of its own falls back to ctest's default
#
# The target name is derived, never given: test_<arch>_<runtime>_<case>, where <case>
# is the file name with its test_ prefix and any redundant arch or runtime token
# stripped. So the directory and the file name decide it, and no two cases can
# collide by someone naming them.
function(simpler_ut_runtime_case runtime src)
    # The include contract keys off the full directory name, not the short tag.
    simpler_runtime_from_tag("${runtime}" runtime_full)
    cmake_parse_arguments(C "ORCH;NO_OBJS" "TIMEOUT"
                            "SOURCES;A2A3_SOURCES;A5_SOURCES;ARCHS;INCLUDES;INCLUDES_BEFORE;LINK;DEFINES;COMPILE_OPTIONS;LINK_OPTIONS" ${ARGN})
    if(C_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "simpler_ut_runtime_case(${src}): unknown arguments: ${C_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT C_ARCHS)
        set(C_ARCHS a2a3 a5)
    endif()
    string(REGEX REPLACE "^test_(a2a3_|a5_|hbg_)*(.*)\\.cpp$" "\\2" stem "${src}")
    foreach(arch ${C_ARCHS})
        set(objs "")
        if(NOT C_NO_OBJS)
            set(objs ${arch}_${runtime}_objs)
        endif()
        if(C_ORCH)
            list(APPEND objs ${arch}_${runtime}_orch_objs)
        endif()
        string(TOUPPER ${arch} ARCH_UP)
        # @arch@ inside a SOURCES entry resolves here, once per iteration.
        string(CONFIGURE "${C_SOURCES};${C_${ARCH_UP}_SOURCES}" sources @ONLY)
        string(CONFIGURE "${C_INCLUDES}" includes @ONLY)
        string(CONFIGURE "${C_INCLUDES_BEFORE}" includes_before @ONLY)
        string(CONFIGURE "${C_LINK}" link @ONLY)
        set(contract_arch "")
        set(contract_rt "")
        if(C_NO_OBJS)
            set(contract_arch ${arch})
            set(contract_rt ${runtime_full})
        endif()
        simpler_ut_add_target(NAME test_${arch}_${runtime}_${stem}
            SOURCES ${src} ${sources}
            ARCH ${contract_arch} RUNTIME ${contract_rt}
            OBJS ${objs}
            INCLUDES ${includes}
            INCLUDES_BEFORE ${includes_before}
            LINK ${link}
            DEFINES SIMPLER_PLATFORM_NAME="${arch}sim" ${C_DEFINES}
            COMPILE_OPTIONS ${C_COMPILE_OPTIONS}
            LINK_OPTIONS ${C_LINK_OPTIONS}
            TIMEOUT ${C_TIMEOUT}
            LABEL no_hardware PRIVATE_WORKDIR)
    endforeach()
endfunction()

# One platform case, built once for each combination it varies over.
#
# Platform sources have no build of their own: they compile into each runtime's
# image, and each arch's platform tree is its own code. So a case covering them
# repeats along the axes it actually reaches, and which those are is a property
# of the case, not a choice:
#
#   PER_ARCH     the case reaches src/<arch>/platform, whose two copies are not
#                the same file, so it is built for each arch against that arch's
#                include contract
#   PER_RUNTIME  the case reaches a source that ships inside every runtime's
#                image and resolves its bare-name headers to that runtime's
#                copy, so it is built once per runtime and told which one
#                through SIMPLER_RUNTIME_NAME
#   SINGLE       the case reaches neither tree, so one build is every build
#   BARE         skip the common include contract; the case names every
#                directory it needs, for one reaching an arch's headers that
#                the contract's own arch would shadow
#
# PYUT_EXPORT names this case in the manifest a pytest reads to find the
# binaries it drives, keyed by the combination each was built for. A case with
# a consumer outside ctest carries it; see simpler_ut_write_pyut_manifest().
#
# LINK takes the libraries the case needs beyond gtest, @arch@ included, which
# is how a directory hands every one of its cases that arch's support archive
# without the factory knowing which archive that is.
#
# PER_ARCH and PER_RUNTIME combine: a case reaching both trees is built for
# every pair, because neither axis subsumes the other.
#
# The axes are stated, never inferred: a case whose coverage nobody has judged
# yet must not compile, because a silent default turns an unjudged case into a
# single build that then looks deliberate. The judgement is mechanical — build
# the case, preprocess its translation units with `g++ -E -H`, and read which
# trees the preprocessor opened. Opening neither is what SINGLE asserts.
#
# Naming follows the axes: test_<arch>_<case>, test_<runtime>_<case>,
# test_<arch>_<runtime>_<case>, or test_<case> for SINGLE. @arch@ and @runtime@
# in SOURCES and INCLUDES expand per combination.
function(simpler_ut_platform_case src)
    cmake_parse_arguments(C "PER_ARCH;PER_RUNTIME;SINGLE;BARE" "TIER;PYUT_EXPORT"
                            "SOURCES;INCLUDES;INCLUDES_BEFORE;DEFINES;LINK;DEPENDS;A2A3_SOURCES;A5_SOURCES"
                            ${ARGN})
    if(C_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "simpler_ut_platform_case(${src}): unknown arguments: ${C_UNPARSED_ARGUMENTS}")
    endif()
    if(C_SINGLE AND (C_PER_ARCH OR C_PER_RUNTIME))
        message(FATAL_ERROR
            "simpler_ut_platform_case(${src}): SINGLE is the absence of an axis")
    endif()
    if(NOT C_SINGLE AND NOT C_PER_ARCH AND NOT C_PER_RUNTIME)
        message(FATAL_ERROR
            "simpler_ut_platform_case(${src}): state PER_ARCH, PER_RUNTIME, "
            "both, or SINGLE")
    endif()
    string(REGEX REPLACE "^test_(.*)\\.cpp$" "\\1" stem "${src}")
    if(NOT C_TIER)
        set(C_TIER host)
    endif()

    # The combinations to build, as "<arch>|<runtime>" pairs. An empty runtime
    # means the case is not built per runtime; the arch is never empty, since a
    # platform source always compiles against one arch's headers even when
    # nothing it reaches differs between them.
    set(_archs a2a3)
    if(C_PER_ARCH)
        set(_archs a2a3 a5)
    endif()
    set(_combos "")
    foreach(_a ${_archs})
        if(C_PER_RUNTIME)
            foreach(_r ${SIMPLER_RUNTIMES})
                list(APPEND _combos "${_a}|${_r}")
            endforeach()
        else()
            list(APPEND _combos "${_a}|")
        endif()
    endforeach()

    foreach(_combo ${_combos})
        string(REPLACE "|" ";" _pair "${_combo}")
        list(GET _pair 0 arch)
        list(GET _pair 1 runtime)

        string(TOUPPER ${arch} ARCH_UP)
        set(extra ${C_${ARCH_UP}_SOURCES})
        set(contract "")
        if(NOT C_BARE)
            simpler_ut_common_includes("${arch}" "${C_TIER}" contract)
        endif()
        set(defines ${C_DEFINES})
        if(runtime)
            list(APPEND defines SIMPLER_RUNTIME_NAME="${runtime}")
        endif()

        string(CONFIGURE "${C_SOURCES}" sources @ONLY)
        string(CONFIGURE "${C_INCLUDES}" includes @ONLY)
        string(CONFIGURE "${C_INCLUDES_BEFORE}" includes_before @ONLY)
        string(CONFIGURE "${C_LINK}" link @ONLY)
        # The runtime's own directory goes last: it is what makes a bare-name
        # header resolve to this runtime's copy, and it must not shadow a
        # directory the case named for itself.
        if(runtime)
            list(APPEND includes ${SIMPLER_SRC}/common/${runtime})
        endif()

        # The name carries only the axes the case varies over, so it says what
        # distinguishes this build from its siblings and nothing more. The
        # runtime goes in by its short tag, as every other target's does.
        set(name test)
        if(C_PER_ARCH)
            set(name ${name}_${arch})
        endif()
        if(C_PER_RUNTIME)
            simpler_runtime_tag("${runtime}" _tag)
            set(name ${name}_${_tag})
        endif()
        set(name ${name}_${stem})

        # A private working directory is what keeps several builds of one source
        # from reaching each other's relative-path writes, so it goes to the
        # targets that have siblings. A SINGLE case has none.
        set(isolate PRIVATE_WORKDIR)
        if(C_SINGLE)
            set(isolate "")
        endif()

        simpler_ut_add_target(NAME ${name}
            SOURCES ${src} ${sources} ${extra}
            INCLUDES ${contract} ${includes}
            INCLUDES_BEFORE ${includes_before}
            DEFINES ${defines}
            LINK ${link}
            DEPENDS ${C_DEPENDS}
            LABEL no_hardware ${isolate})

        if(C_PYUT_EXPORT)
            simpler_ut_export_to_pyut("${C_PYUT_EXPORT}" "${arch}" "${runtime}" "${name}")
        endif()
    endforeach()
endfunction()

# A case that needs nothing but its own file and the directory's contract. The
# target takes the file's name, which is what lets a glob declare it.
function(simpler_ut_plain_case src)
    string(REGEX REPLACE "\\.cpp$" "" _name "${src}")
    simpler_ut_add_target(NAME ${_name} SOURCES ${src} ${ARGN})
endfunction()

# Several cases of one kind, declared together.
#
# A directory usually has groups of cases that differ only in which file they
# are: the same extra sources, the same axis, the same contract. Declared one
# at a time, the shape is written once per case, the copies can drift, and
# nothing says the group is a group — the next case joins by someone noticing
# the pattern and copying a line, including the part they might forget.
#
# Grouping them says it instead, and the shape has one place to change.
#
# This is an assertion that the cases are the same *kind*, not merely that
# their arguments happen to match today. Two cases can both be PER_ARCH for
# unrelated reasons — one because kernel_args.h differs between the arches,
# another because platform_config.h does — and grouping those would claim a
# relationship that is not there. Where the reasons differ, declare separately
# and let each carry its own.
#
# `declare` is the directory's own case macro. Every leading *.cpp is a case;
# from the first other argument on, everything is the shape they share — so a
# .cpp named inside SOURCES is part of the shape rather than a case of its own.
function(simpler_ut_cases declare)
    set(_srcs "")
    set(_shape "")
    set(_in_shape FALSE)
    foreach(_arg ${ARGN})
        if(NOT _in_shape AND _arg MATCHES "\\.cpp$")
            list(APPEND _srcs "${_arg}")
        else()
            set(_in_shape TRUE)
            list(APPEND _shape "${_arg}")
        endif()
    endforeach()
    if(NOT _srcs)
        message(FATAL_ERROR "simpler_ut_cases(${declare}): no case files before the shared arguments")
    endif()
    foreach(_src ${_srcs})
        cmake_language(CALL ${declare} ${_src} ${_shape})
    endforeach()
endfunction()

# Record a directory's cases as accounted for without building them, for one
# the configuration did not descend into.
#
# A directory left out by an option still holds case files, and they are not
# orphans: the tree knows about them and a configuration that turns the option
# on builds them. Registering them here keeps that distinction — the guard
# below reports files nothing knows about, not files this configuration chose
# to skip.
function(simpler_ut_skip_dir dir)
    file(GLOB _skipped CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/test_*.cpp)
    set_property(GLOBAL APPEND PROPERTY SIMPLER_UT_SKIPPED_CASES ${_skipped})
endfunction()

# A case file whose declaration above is itself conditional — e.g. on
# CMAKE_SYSTEM_NAME — and that has nothing to build on this configuration.
#
# simpler_ut_glob_cases() below only excludes what SIMPLER_UT_DECLARED_CASES
# already lists, so a file left undeclared by a condition that did not hold is
# still unclaimed by the time the glob runs in this same directory, and the
# glob builds it with default settings regardless of why it was left out.
# Registering it here without creating a target keeps it out of that sweep.
function(simpler_ut_exclude_case src)
    set_property(GLOBAL APPEND PROPERTY SIMPLER_UT_DECLARED_CASES
                 ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endfunction()

# Build every case in this directory that no explicit declaration above has
# already claimed.
#
# This is what makes a plain case cost nothing: drop test_foo.cpp into the
# directory and it is compiled and registered, with the include contract its
# directory implies. A case needing anything beyond that is declared by hand
# earlier in the file, which takes it out of this sweep automatically — there
# is no list of exceptions to maintain, because declaring one *is* the record.
#
# `declare` names the macro that builds one case here: hbg_case, tmr_case, or
# any wrapper a directory defines for its own default shape.
function(simpler_ut_glob_cases declare)
    file(GLOB _found CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
    get_property(_declared GLOBAL PROPERTY SIMPLER_UT_DECLARED_CASES)
    if(_declared)
        list(REMOVE_ITEM _found ${_declared})
    endif()
    foreach(_src ${_found})
        get_filename_component(_name "${_src}" NAME)
        cmake_language(CALL ${declare} ${_name} ${ARGN})
    endforeach()
endfunction()

# A case a pytest drives directly, recorded so that pytest never spells the
# target name itself.
#
# A target's name is assembled here, from the case file and the combination it
# is built for. A consumer outside ctest has to address the binary somehow, and
# the obvious way — reassembling the same name on its own side — is a second
# implementation of this file's naming rule that nothing ties to it: the two
# drift, the consumer finds no file, and a test that reports "not built" by
# skipping goes quiet instead of failing.
#
# So the name is not published, the path is, and the key is the thing the
# consumer actually knows: which case, for which combination. Renaming a target
# or moving where binaries land changes the recorded path and nothing else.
function(simpler_ut_export_to_pyut handle arch runtime target)
    set_property(GLOBAL APPEND PROPERTY SIMPLER_UT_PYUT_BINARIES
                 "${handle}\t${arch}\t${runtime}\t$<TARGET_FILE:${target}>")
endfunction()

# Write the manifest the exported cases were recorded into.
#
# file(GENERATE) resolves $<TARGET_FILE:...> at generate time, so the manifest
# exists after configure and before anything is built. A reader therefore
# learns the intended set of binaries from it and whether they exist from the
# filesystem — two different questions, and keeping them apart is what lets a
# consumer tell an unbuilt tree (skip) from a declaration that lost a
# combination (fail).
#
# Call once, after the last add_subdirectory.
function(simpler_ut_write_pyut_manifest)
    get_property(_entries GLOBAL PROPERTY SIMPLER_UT_PYUT_BINARIES)
    set(_records "")
    foreach(_entry ${_entries})
        string(REPLACE "\t" ";" _fields "${_entry}")
        list(GET _fields 0 _handle)
        list(GET _fields 1 _arch)
        list(GET _fields 2 _runtime)
        list(GET _fields 3 _path)
        list(APPEND _records
             "  {\"handle\": \"${_handle}\", \"arch\": \"${_arch}\", \"runtime\": \"${_runtime}\", \"path\": \"${_path}\"}")
    endforeach()
    list(JOIN _records ",\n" _body)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/pyut_binaries.json" CONTENT "[\n${_body}\n]\n")
endfunction()

# Every case file in the tree belongs to a target.
#
# A case is compiled and registered by the glob in its own directory, so one
# sitting in a directory that has no glob is compiled by nothing. That state is
# silent in every direction: the tree configures, builds, and passes, and ctest
# reports the cases that registered — a case that registered nothing subtracts
# no assertion from any other, so the suite is green and smaller by exactly the
# coverage nobody can see is gone.
#
# The other checks cannot reach it either. Case naming and axis coverage are
# both read off the targets that exist, and an unbuilt case has none.
#
# The directories carrying a simpler_ut_glob_cases call are the leaves that
# mirror src/. Intermediate levels hold no cases of their own, and a case
# landing in one — or in a directory left over from a layout this tree no
# longer has — is the failure above. This asserts that the set is empty.
#
# Call once, after the last add_subdirectory, so every target has registered.
function(simpler_ut_assert_no_orphan_cases)
    file(GLOB_RECURSE _all CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/test_*.cpp)
    get_property(_built   GLOBAL PROPERTY SIMPLER_UT_DECLARED_CASES)
    get_property(_skipped GLOBAL PROPERTY SIMPLER_UT_SKIPPED_CASES)

    set(_orphans "")
    foreach(_f ${_all})
        # The build directory may sit inside the source tree, and a fetched
        # GoogleTest lands under it.
        string(FIND "${_f}" "${CMAKE_BINARY_DIR}/" _under_build)
        if(_under_build EQUAL 0)
            continue()
        endif()
        if(_f IN_LIST _built OR _f IN_LIST _skipped)
            continue()
        endif()
        file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}" "${_f}")
        list(APPEND _orphans "    ${_rel}")
    endforeach()

    if(_orphans)
        list(LENGTH _orphans _n)
        list(JOIN _orphans "\n" _listing)
        message(FATAL_ERROR
            "${_n} case file(s) are built by no target:\n\n${_listing}\n\n"
            "Each sits in a directory with no simpler_ut_glob_cases call, so it "
            "compiles into nothing and registers with ctest as nothing. The "
            "suite would still build and still pass, missing exactly these.\n\n"
            "Move each into the directory mirroring the src/ tree its case "
            "covers, which is what decides its include contract and how many "
            "times it is built. A case needing more than its own file is "
            "declared by hand in that directory's CMakeLists.txt.\n\n"
            "See docs/testing/adding-a-cpp-unit-test.md.")
    endif()
endfunction()
