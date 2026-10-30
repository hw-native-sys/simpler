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
struct OrchContext;
class Runtime;
struct RuntimeArenaLayout;
struct RuntimeContext;
template <class Layout>
struct RetainedHostPreparePlan;

// The orchestration .so exports these (submit_task form).
typedef void (*OrchestrationEntryFunc)(const ChipTaskArgs &);
typedef void (*OrchestrationBindFunc)(OrchContext *);
typedef void (*OrchestrationFinishFunc)();

// Resolved orchestration .so entry points. Callable registration allocates one
// of these (the entry, plus the .so's own framework_bind_orch_context, which sets
// the .so-private context pointer its inline rt_submit_* read) and stores its
// pointer in CallableArtifacts::host_orch_func_ptr. Owned for the callable's
// lifetime alongside host_dlopen_handle.
struct HostOrchEntryPoints {
    OrchestrationEntryFunc entry{nullptr};
    OrchestrationBindFunc bind{nullptr};
    // Lets the .so release what it holds for a bind, once that bind's entry has returned.
    // Optional: a .so built before this entry point existed holds nothing per bind.
    OrchestrationFinishFunc finish{nullptr};
};

namespace hbg {

// Records belong to this build; mirror and retained staging belong to its
// caller's pipeline slot. That exclusive host lease outlives every consumer.
struct GraphBuild {
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
    bool build_complete{false};
};

// Host construction only: no execution-region commitment, metadata publication
// or device copy. Arguments have already been staged by the caller. The entry
// points are retained with the callable's orchestration library.
int32_t build_graph(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt, uint64_t sm_size,
    uint64_t task_capacity, void *host_orch_func_ptr, const ChipTaskArgs &args, GraphBuild &build
);

// Build the graph and fill the retained plan the host orchestration will commit.
// The plan and its staging belong to the caller's pipeline slot.
int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt,
    DeviceArena &host_arena, const RuntimeArenaLayout &layout, uint64_t sm_size, uint64_t task_capacity,
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2, RetainedHostPreparePlan<RuntimeArenaLayout> &plan
);

}  // namespace hbg
