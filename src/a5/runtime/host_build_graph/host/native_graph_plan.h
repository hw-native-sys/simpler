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

// Owns the immutable recording bytes. Nothing refers to GraphHostState or its
// staging arena, so prepare retry and park/resume may retain this value safely.
struct NativeGraphPlan {
    Graph definition{};
    std::vector<std::byte> image;
    std::vector<SchedulerGraphChildPlan> children;
    std::vector<uint16_t> sinks;
    uint64_t args_bytes{0};
};

// Invalid input leaves out unchanged. Child argument spans partition the
// Graph pools in recording order, including children with no arguments.
bool native_graph_plan_definition(
    const void *image, size_t bytes, uint64_t graph_key, NativeGraphPlan *out, std::string *error
);

bool native_graph_validate_definition(const void *image, size_t bytes, uint64_t graph_key, std::string *error);

struct NativeGraphHostPlan {
    std::vector<std::byte> original_definition_section;
    std::vector<NativeGraphPlan> definitions;
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
// Commit patches the header's allocation/epoch fields and the instances'
// external addresses before publishing the entire scheduler backing.
bool native_graph_plan_readonly_image(
    const NativeGraphHostPlan &plan, NativeGraphReadonlyImage *out, std::string *error
);

enum class NativeGraphBindingError : uint32_t { NONE, GRAPH, INSTANCE, TENSOR, SCALAR, PREDICATE };

struct NativeGraphArguments {
    const simpler::hbg::TensorData *tensors{nullptr};
    const uint64_t *scalars{nullptr};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint64_t heap_base{0};
    uint64_t args_base{0};
};

// Sources are Host mirrors; heap_base and args_base remain device addresses.
NativeGraphBindingError native_graph_bind_arguments(
    const NativeGraphPlan &plan, uint32_t child_index, const NativeGraphArguments &arguments, void *argument_image
);
