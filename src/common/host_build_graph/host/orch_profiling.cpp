/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "host_build_graph/orch_profiling.h"

#include <stdint.h>

#include "host_build_graph/orchestrator.h"

namespace simpler::hbg {

#if SIMPLER_ORCH_PROFILING
uint64_t g_orch_alloc_ns = 0;
uint64_t g_orch_args_ns = 0;
uint64_t g_orch_lookup_ns = 0;
uint64_t g_orch_insert_ns = 0;
uint64_t g_orch_fanin_ns = 0;
int64_t g_orch_submit_count = 0;
#endif

#if SIMPLER_ORCH_PROFILING || SIMPLER_DFX
uint32_t g_orch_submit_idx = 0;
#endif

}  // namespace simpler::hbg

#if SIMPLER_ORCH_PROFILING
OrchProfilingData orchestrator_get_profiling() {
    OrchProfilingData d;
    d.alloc_ns = g_orch_alloc_ns;
    d.args_ns = g_orch_args_ns;
    d.lookup_ns = g_orch_lookup_ns;
    d.insert_ns = g_orch_insert_ns;
    d.fanin_ns = g_orch_fanin_ns;
    d.submit_count = g_orch_submit_count;

    // Reset
    g_orch_alloc_ns = g_orch_args_ns = 0;
    g_orch_lookup_ns = g_orch_insert_ns = 0;
    g_orch_fanin_ns = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    return d;
}
#endif
