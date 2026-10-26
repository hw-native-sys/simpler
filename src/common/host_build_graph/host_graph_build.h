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

#include "host_build_graph/host/graph_task_definition_binding.h"
#include "host_build_graph/shared_memory.h"

class DeviceArena;
class HostTensorAccessor;
struct HostApi;
class Runtime;
struct RuntimeContext;

namespace hbg {

enum class RuntimeArchitecture : uint32_t { A2A3 = 1, A5 = 2 };

// Records belong to this build; mirror and retained staging belong to its
// caller's pipeline slot. That exclusive host lease outlives every consumer.
struct GraphBuild {
    RuntimeArchitecture architecture{};
    GraphHostStatePtr graph_state;
    SharedMemoryHandle host_sm_handle;
    void *sm_mirror{nullptr};
    uint64_t task_capacity{0};
    int32_t total_tasks{0};
    GraphTaskDefinitionBindings definitions;
    ReadyQueuePopulations ready_queue_populations{};
    ReadyQueueCapacities ready_queue_capacities{};
    sm_layout::BindUsage bind_usage{};
    uint64_t image_bytes{0};
    uint64_t heap_bytes{0};
    uint64_t scheduler_state_bytes{0};
    bool build_complete{false};
};

// Host construction only: no execution-region commitment, metadata publication
// or device copy. Arguments have already been staged by the caller. The entry
// points are retained with the callable's orchestration library.
int32_t build_graph(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt, uint64_t sm_size,
    uint64_t task_capacity, void *host_orch_func_ptr, const ChipTaskArgs &args, GraphBuild &build
);

}  // namespace hbg
