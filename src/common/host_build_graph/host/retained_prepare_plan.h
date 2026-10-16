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

#include <cstdint>
#include <utility>
#include <vector>

#include "host_build_graph/runtime.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"

// The callback's result belongs to the slot, while each commit attempt has a
// fresh Runtime and HostApi. No attempt pointer is retained here. Definition
// sources and the SM mirror belong to the same exclusive slot; publication
// storage and tensor leases move between this owner and the current attempt.
template <class Layout>
struct RetainedHostPreparePlan {
    bool committed{false};
    DeviceArena host_arena;
    Layout layout{};
    void *host_sm{nullptr};
    uint64_t sm_size{0};
    uint64_t task_capacity{0};
    uint64_t heap_bytes{0};
    uint64_t image_bytes{0};
    int32_t total_tasks{0};
    sm_layout::BindUsage bind_usage{};
    void *staging_backing{nullptr};
    size_t staging_bytes{0};
    GraphSectionSource graph_section_source{GraphSectionSource::None};
    uint64_t graph_section_base{0};
    uint32_t graph_section_bytes{0};
    simpler::dfx::host_graph::HostGraphExport diagnostic_graph;
    int diagnostic_outcome{static_cast<int>(simpler::dfx::host_graph::TakeOutcome::NotCaptured)};

    Runtime::RunImagePublication publication{};
    std::vector<TensorLease> leases;

    void park(Runtime &runtime) {
        graph_section_source = runtime.get_graph_section_source();
        graph_section_base = runtime.get_graph_section_base();
        graph_section_bytes = runtime.get_graph_section_bytes();
        publication = runtime.take_pending_publication();
        leases = std::move(runtime.tensor_leases());
    }
    void resume(Runtime &runtime) {
        runtime.publish_graph_section(graph_section_source, graph_section_base, graph_section_bytes);
        runtime.restore_pending_publication(std::move(publication));
        publication = {};
        runtime.tensor_leases() = std::move(leases);
    }
};
