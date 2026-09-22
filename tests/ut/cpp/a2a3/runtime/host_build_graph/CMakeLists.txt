# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Tests for src/a2a3/runtime/host_build_graph/.
#
# The arch is this directory's, so no case here states it.
macro(a2a3_hbg_case src)
    hbg_case(${src} ARCHS a2a3 ${ARGN})
endmacro()

# ChipTensorMap's out-of-line members (reserve_layout / init / valid_count) live
# in this .cpp; no other hbg case here needs them.
a2a3_hbg_case(test_hbg_tensormap.cpp
    SOURCES ${HBG_HOST_DIR}/tensormap.cpp)

a2a3_hbg_case(test_dep_gen_host_graph.cpp
    SOURCES ${SIMPLER_SRC}/common/host_build_graph/host/dep_gen_host_graph.cpp)

a2a3_hbg_case(test_hbg_tensor_access.cpp
    SOURCES ${SIMPLER_SRC}/common/host_build_graph/host/host_tensor_access.cpp)

# The func_id -> CoreCallable dispatch table holds exactly the active callable's
# mappings. An entry that outlives the ChipCallable buffer it points into is
# dereferenced by the scheduler and called by the AICore, so this links the
# Runtime implementation to pin the contract without a device.
a2a3_hbg_case(test_dispatch_table.cpp
    SOURCES ${HBG_COMMON_SHARED_DIR}/runtime.cpp)

# Drives the real orchestrator submit path.
a2a3_hbg_case(test_hbg_submit_poison.cpp ORCH)

# Incremental activation inits a real SchedulerState (ready queues), so it links
# the scheduler and shared runtime out-of-line members.
a2a3_hbg_case(test_graph_activation.cpp
    SOURCES ${HBG_SHARED_RUNTIME_SOURCES})

simpler_ut_glob_cases(a2a3_hbg_case)
