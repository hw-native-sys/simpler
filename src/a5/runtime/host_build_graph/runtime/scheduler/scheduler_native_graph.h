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
#include <type_traits>

#include "host_build_graph/graph_definition.h"
#include "scheduler_types.h"

inline constexpr uint32_t SCHEDULER_NATIVE_GRAPH_MAGIC = 0x35474248;
inline constexpr uint32_t SCHEDULER_NATIVE_GRAPH_VERSION = 2;
inline constexpr uint32_t SCHEDULER_NATIVE_NO_INSTANCE = UINT32_MAX;
inline constexpr uint64_t SCHEDULER_NATIVE_NO_PREDICATE = UINT64_MAX;

enum class SchedulerNodeKind : uint8_t {
    TOP_KERNEL = 0,
    CHILD_KERNEL = 1,
    CONTROL_DUMMY = 2,
    GRAPH_ENTER = 3,
    GRAPH_EXIT = 4,
    HOST_DONE = 5,
};

struct SchedulerNodeBinding {
    uint32_t source_index;
    uint32_t instance_index;
    SchedulerNodeKind kind;
};

// All offsets name regions of the scheduler backing, including the existing
// run control prefix. Absolute addresses are confined to per-run instances.
struct alignas(64) SchedulerNativeGraphHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t image_bytes;
    uint64_t node_count;
    uint64_t original_task_count;
    uint64_t definition_count;
    uint64_t instance_count;
    uint64_t bindings_offset;
    uint64_t completion_refs_offset;
    uint64_t templates_offset;
    uint64_t instances_offset;
    uint64_t control_inboxes_offset;
    uint64_t control_inbox_count;
    uint64_t readonly_end_offset;
    uint64_t run_epoch;
    uint32_t header_bytes;
};

struct alignas(64) SchedulerGraphTemplate {
    uint64_t definition_offset;
    uint64_t definition_bytes;
    uint64_t child_plans_offset;
    uint64_t fanin_offsets_offset;
    uint64_t fanin_indices_offset;
    uint64_t sinks_offset;
    uint64_t task_count;
    uint64_t edge_count;
    uint64_t sink_count;
    uint64_t args_bytes;
    uint64_t execution_count;
    uint64_t required_heap;
    uint64_t legacy_execution_storage_bytes;
    uint64_t graph_key;
};

struct alignas(64) SchedulerGraphChildPlan {
    SchedulerTaskMetadata metadata;
    uint64_t tensors_offset;
    uint64_t scalars_offset;
    uint64_t predicate_offset;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint64_t execution_prefix;
};

struct alignas(64) SchedulerGraphInstance {
    uint32_t definition_index;
    uint32_t original_shell_id;
    uint32_t enter_ref;
    uint32_t child_begin;
    uint32_t child_count;
    uint32_t exit_ref;
    uint32_t boundary_tensor_count;
    uint32_t boundary_scalar_count;
    uint64_t boundary_tensors_address;
    uint64_t boundary_scalars_address;
    uint64_t heap_base;
    uint64_t args_base;
};

static_assert(sizeof(SchedulerNodeBinding) == 12);
static_assert(offsetof(SchedulerNodeBinding, kind) == 8);
static_assert(std::is_standard_layout_v<SchedulerNodeBinding> && std::is_trivially_copyable_v<SchedulerNodeBinding>);
static_assert(sizeof(SchedulerNativeGraphHeader) == 128);
static_assert(offsetof(SchedulerNativeGraphHeader, image_bytes) == 8);
static_assert(offsetof(SchedulerNativeGraphHeader, bindings_offset) == 48);
static_assert(offsetof(SchedulerNativeGraphHeader, readonly_end_offset) == 96);
static_assert(
    std::is_standard_layout_v<SchedulerNativeGraphHeader> && std::is_trivially_copyable_v<SchedulerNativeGraphHeader>
);
static_assert(sizeof(SchedulerGraphTemplate) == 128);
static_assert(offsetof(SchedulerGraphTemplate, required_heap) == 88);
static_assert(
    std::is_standard_layout_v<SchedulerGraphTemplate> && std::is_trivially_copyable_v<SchedulerGraphTemplate>
);
static_assert(sizeof(SchedulerGraphChildPlan) == 64);
static_assert(offsetof(SchedulerGraphChildPlan, tensors_offset) == 16);
static_assert(offsetof(SchedulerGraphChildPlan, execution_prefix) == 48);
static_assert(
    std::is_standard_layout_v<SchedulerGraphChildPlan> && std::is_trivially_copyable_v<SchedulerGraphChildPlan>
);
static_assert(sizeof(SchedulerGraphInstance) == 64);
static_assert(offsetof(SchedulerGraphInstance, boundary_tensors_address) == 32);
static_assert(offsetof(SchedulerGraphInstance, args_base) == 56);
static_assert(
    std::is_standard_layout_v<SchedulerGraphInstance> && std::is_trivially_copyable_v<SchedulerGraphInstance>
);
