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

#pragma once

#include <stdint.h>

#include "host_build_graph/dep_compute.h"
#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/task_id.h"
#include "host_build_graph/tensormap.h"
#include "tensor.h"

// The ordinary submit path's primitives, shared by the two translation units that submit
// into one OrchestratorState: host/orchestrator.cpp, which defines them and runs the
// ordinary path, and host/graph_submit.cpp, which runs the Graph path's outer shell and
// recorded sub-tasks through the same resources. Nothing outside those two includes this
// header -- the orchestrator's public surface is orchestrator.h.
//
// The set below is what the Graph path reaches for. The orchestrator's instrumentation is
// not part of it and lives in orch_profiling.h, which both units include separately.
//
// The names sit in simpler::hbg so the host runtime does not export a symbol as generic as
// `calculate_output_layout`: the tensormap_and_ringbuffer runtime has a file-static
// function of that exact name, and of the four others here, in its own orchestrator. The
// using-declarations at the bottom keep both call sites spelling them unqualified.

namespace simpler::hbg {

// Raises the two edge kinds compute_task_fanin can discover, for the capture
// instantiation. Shared by the ordinary submit path and the outer GRAPH task so
// both describe an edge the same way.
struct DepGraphAnnotate {
    void creator(int32_t arg_idx, const simpler::hbg::Tensor &consumer, TaskId producer) const {
        dep_gen_host_graph_add_creator_edge(producer, arg_idx, consumer);
    }
    void tensormap(
        int32_t arg_idx, const simpler::hbg::Tensor &consumer, const ChipTensorMapEntry &entry, OverlapStatus overlap
    ) const {
        dep_gen_host_graph_add_tensormap_edge(entry.producer_task_id, arg_idx, consumer, entry, overlap);
    }
};

template <typename Args>
bool require_device_arguments(OrchestratorState *orch, const Args &args) {
    if (orch->is_fatal()) return false;
    if (args.has_error()) return true;
    for (int i = 0; i < args.tensor_count(); ++i) {
        if (args.tag(i) != TensorArgType::OUTPUT && args.tensor(i).ref().address_space != AddressSpace::DEVICE) {
            orch->report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
                "device task argument %d is HOST; pass a separate HOST/H2D or DEVICE/NONE argument", i
            );
            return false;
        }
    }
    return true;
}

int32_t orch_mark_fatal(OrchestratorState *orch, int32_t error_code);

OutputLayout calculate_output_layout(const CoreTaskArgs &args);

bool ensure_tensormap_capacity(OrchestratorState *orch, int32_t needed);

void next_fanin_seen_epoch(OrchestratorState *orch);

bool append_fanin_or_fail(OrchestratorState &orch, TaskId producer_task_id, int32_t *fanin_slots, int32_t &fanin_count);

}  // namespace simpler::hbg

using simpler::hbg::append_fanin_or_fail;
using simpler::hbg::calculate_output_layout;
using simpler::hbg::DepGraphAnnotate;
using simpler::hbg::ensure_tensormap_capacity;
using simpler::hbg::next_fanin_seen_epoch;
using simpler::hbg::orch_mark_fatal;
using simpler::hbg::require_device_arguments;
