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
struct NativeGraphDefinitionPlan {
    GraphDefinition definition{};
    std::vector<std::byte> image;
    std::vector<SchedulerGraphChildPlan> children;
    std::vector<uint16_t> sinks;
    uint64_t args_bytes{0};
    uint64_t execution_count{0};
    bool resident_supported{true};
};

// Invalid input leaves out unchanged. Legal shapes not currently supported by
// the resident executor still receive full validation before fallback is chosen.
bool native_graph_plan_definition(
    const void *image, size_t bytes, uint64_t graph_key, NativeGraphDefinitionPlan *out, std::string *error
);

struct NativeGraphHostPlan {
    std::vector<std::byte> legacy_section;
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

// Validate device-wide barriers even when the whole run needs the legacy path.

struct NativeGraphReadonlyImage {
    std::vector<std::byte> bytes;
    uint64_t header_offset{0};
};

// Serializes the graph behind an aligned prefix; offset zero denotes no native graph.
bool native_graph_plan_readonly_image(
    const NativeGraphHostPlan &plan, NativeGraphReadonlyImage *out, std::string *error
);
