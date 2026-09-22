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

/**
 * The two capture sinks host_build_graph's orchestrator raises into.
 *
 * A production build links the real ones from host/dep_gen_host_graph.cpp and
 * host/host_phase_trace.cpp. A test that drives the orchestrator wants neither —
 * it asserts on submitted tasks, not on captured graphs or phase records — and
 * linking either would pull in the collector and the record pool behind it.
 *
 * These are ordinary definitions rather than weak ones, so a target gets exactly
 * one of the two sets: this file rides on HBG_ORCH_SHARED_SOURCES, which
 * test_hbg_dep_gen_host_graph (the one test that wants the real capture) does not
 * use. Adding this file to a target that already links a real sink is a duplicate
 * symbol at link time, which is the intended way to find out.
 */

#include <cstdint>

#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/host_phase_trace.h"

bool dep_gen_host_graph_enabled() { return false; }

void dep_gen_host_graph_begin_task(
    uint64_t, bool, bool, const int32_t[3], int32_t, int32_t, const TensorRef *, const TensorArgType *
) {}

void dep_gen_host_graph_end_task() {}

void dep_gen_host_graph_add_explicit_edge(uint64_t) {}

void dep_gen_host_graph_add_creator_edge(uint64_t, int32_t, const simpler::hbg::Tensor &) {}

void dep_gen_host_graph_add_tensormap_edge(
    uint64_t, int32_t, const simpler::hbg::Tensor &, const ChipTensorMapEntry &, OverlapStatus
) {}

// 0 is the "this bind records nothing" answer host_phase_trace.h documents, which
// is what keeps the orchestrator's hooks down to two calls returning a constant.
uint64_t host_phase_now_ns() { return 0; }

void host_phase_record(uint64_t, uint64_t, uint32_t, uint64_t, uint32_t) {}
