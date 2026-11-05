# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a5/runtime/percounter/.
#
# The runtime tag stays `hbg`: percounter compiles host_build_graph's sources and
# reads its headers, so a case here wants that runtime's include contract and
# object library. Only the percounter headers under test come from this runtime's
# own tree, which each case names below.
#
# The arch is this directory's, so no case here states it.
macro(a5_percounter_case src)
    hbg_case(${src} ARCHS a5 ${ARGN})
endmacro()

a5_percounter_case(test_hbg_percounter_graph.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/percounter/host ${SIMPLER_SRC}/a5/runtime/percounter/runtime)

a5_percounter_case(test_hbg_percounter_graph_prepare.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/percounter/host ${SIMPLER_SRC}/a5/runtime/percounter/aicpu
             ${SIMPLER_SRC}/a5/runtime/percounter/runtime
    SOURCES  ${SIMPLER_SRC}/common/host_build_graph/device/graph_execution.cpp)

a5_percounter_case(test_hbg_percounter_tables.cpp
    INCLUDES ${SIMPLER_SRC}/a5/runtime/percounter/host ${SIMPLER_SRC}/a5/runtime/percounter/runtime)
