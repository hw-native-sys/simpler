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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler_native_graph.h"
#include "scheduler/scheduler_graph_binding.h"

// Owns the immutable recording bytes. Nothing refers to GraphHostState or its
// staging arena, so prepare retry and park/resume may retain this value safely.
struct NativeGraphDefinitionPlan {
    GraphDefinition definition{};
    std::vector<std::byte> image;
    std::vector<SchedulerGraphChildPlan> children;
    std::vector<uint16_t> sinks;
    uint64_t args_bytes{0};
    uint64_t execution_count{0};
};

// Invalid input leaves out unchanged. Child argument spans partition the
// Definition pools in recording order, including children with no arguments.
bool native_graph_plan_definition(
    const void *image, size_t bytes, uint64_t graph_key, NativeGraphDefinitionPlan *out, std::string *error
);

bool native_graph_validate_definition(const void *image, size_t bytes, uint64_t graph_key, std::string *error);

struct NativeGraphHostPlan {
    std::vector<std::byte> original_definition_section;
    std::vector<NativeGraphDefinitionPlan> definitions;
    std::vector<SchedulerNodeBinding> bindings;
    std::vector<int64_t> completion_refs;
    std::vector<SchedulerGraphInstance> instances;
};

// definition_indices uses UINT32_MAX for a non-Graph top-level task. The
// resulting instances contain indices only; commit supplies actual GM addresses.
bool native_graph_plan_nodes(
    const std::vector<uint32_t> &definition_indices, NativeGraphHostPlan *plan, std::string *error
);

// Validate device-wide barriers against the launched worker population.
bool native_graph_plan_worker_capacity(const NativeGraphHostPlan &plan, uint64_t workers, std::string *error);

struct NativeGraphReadonlyImage {
    std::vector<std::byte> bytes;
    uint64_t header_offset{0};
};

// Reserves the existing run-control prefix without writing its live contents.
// Commit patches the header's allocation/epoch/inbox fields and the instances'
// external addresses before publishing the entire scheduler backing.
bool native_graph_plan_readonly_image(
    const NativeGraphHostPlan &plan, NativeGraphReadonlyImage *out, std::string *error
);

// Sources are Host mirrors; instance addresses remain device addresses. The
// caller retains and publishes argument_image at the instance's args_base.
SchedulerGraphBindingError native_graph_bind_arguments(
    const SchedulerGraphView &graph, int64_t ref, const simpler::hbg::TensorData *boundary_tensors,
    const uint64_t *boundary_scalars, void *argument_image
);
