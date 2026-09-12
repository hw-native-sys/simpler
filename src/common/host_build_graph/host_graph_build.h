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

#include "host_build_graph/graph_host_state.h"
#include "host_build_graph/kernel_resource_plan.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/ready_queue_sizing.h"
#include "host_build_graph/shared_memory.h"

class DeviceArena;
class HostTensorAccessor;
struct HostApi;
class Runtime;
struct RuntimeContext;
struct RuntimeArenaLayout;

namespace simpler::hbg {

// Resolved symbols retained with CallableArtifacts::host_dlopen_handle.
struct HostOrchEntryPoints {
    void (*entry)(const ChipTaskArgs &){nullptr};
    void (*bind)(RuntimeContext *){nullptr};
};

struct GraphDefinitionPlan;

// Caller-owned buffers leased together for one build and all its consumers.
struct LeasedWorkspace {
    RuntimeContext *runtime_context{nullptr};
    void *sm_mirror{nullptr};
    uint64_t sm_bytes{0};
    uint64_t task_capacity{0};
    GraphDefinitionArena definitions{};
};

// Owns the orchestration and Definition records, borrowing the caller's SM mirror
// and Definition staging. Their pipeline-slot lease must outlive every upload;
// neither buffer may be reused for another build while this result is in use.
// Build/upload require exclusive workspace access; queries may overlap only
// other queries against the completed, stable build.
// This internal C++ state is not a wire or plugin ABI. build_complete is a
// validity bit: upload does not consume the build or create an Uploaded state.
// Heap addresses remain virtual; upload binds a fresh compact image to the device.
struct GraphBuild {
    GraphBuild();
    ~GraphBuild();
    GraphBuild(const GraphBuild &) = delete;
    GraphBuild &operator=(const GraphBuild &) = delete;
    GraphBuild(GraphBuild &&) = delete;
    GraphBuild &operator=(GraphBuild &&) = delete;

    LeasedWorkspace workspace;
    OrchestratorState orchestrator;
    GraphHostStatePtr graph_state;
    SharedMemoryHandle host_sm_handle;
    std::unique_ptr<GraphDefinitionPlan> definitions;
    int32_t total_tasks{0};
    ReadyQueuePopulations ready_queue_populations{};
    ReadyQueueCapacities ready_queue_capacities{};
    sm_layout::BindUsage bind_usage{};
    uint64_t image_bytes{0};
    uint64_t heap_bytes{0};
    uint64_t definition_bytes{0};
    bool build_complete{false};
};

// Build is the Host construction and sizing portion of a program bind.
// Upload completes the bind by assigning device addresses and transferring data.
// Build consumes host workspace and already-staged arguments. It neither commits
// device execution regions nor uploads SM or Graph Definitions.
int32_t build_graph(
    Runtime *runtime, HostTensorAccessor &tensor_access, LeasedWorkspace workspace,
    const HostOrchEntryPoints &entry_points, const ChipTaskArgs &args, GraphBuild &build
);

// Capture-external query. Does not allocate device resources, bind addresses or
// upload. The output owns no buffers and is unchanged on failure. It consumes
// the completed H1 measurements rather than rescanning graph Definitions.
int32_t get_graph_resource_requirements(
    const GraphBuild &build, const RuntimeArenaLayout &layout, GraphResourceRequirements &requirements
);

// Program-only allocation and synchronous H2D boundary; not a kernel capture path.
// Repeated uploads use the same leased host workspace and may rebind device bases.
// Each upload compacts from the virtual-address mirror, never from a previous image.
// rt and host_arena must retain the context initialized for this build; they
// cannot serve another build until all uploads of this result finish.
int32_t upload_for_program_mode(
    Runtime *runtime, const HostApi *api, RuntimeContext *rt, DeviceArena &host_arena, const RuntimeArenaLayout &layout,
    GraphBuild &build
);

}  // namespace simpler::hbg
