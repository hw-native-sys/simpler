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

#include <stddef.h>
#include <stdint.h>

#include "common/core_type.h"
#include "dispatch_payload.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

// A task's three records share one ChipTaskStorage, so AICore reaches both the
// ones it reads from a single base: the array strides by the storage, and each
// record sits at a fixed offset inside it. These are literals because the AICore
// .o cannot include runtime_types.h; scheduler.h checks them against the types at
// compile time, and the contract UT again at run time.
inline constexpr uint32_t SCHEDULER_GRAPH_TASK_STORAGE_STRIDE = 320;
inline constexpr uint32_t SCHEDULER_GRAPH_DESCRIPTOR_OFFSET = 0;
// TaskPayload includes the early-dispatch cache line in this wire layout.
inline constexpr uint32_t SCHEDULER_GRAPH_PAYLOAD_OFFSET = 128;
inline constexpr uint32_t SCHEDULER_GRAPH_TASK_ID_OFFSET = 0;
inline constexpr uint32_t SCHEDULER_GRAPH_KERNEL_IDS_OFFSET = 8;
inline constexpr uint32_t SCHEDULER_GRAPH_FANIN_COUNT_OFFSET = 8;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_OFFSET = 128;
inline constexpr int32_t SCHEDULER_GRAPH_MAX_FANIN = 128;
inline constexpr uint32_t SCHEDULER_CORE_CALLABLE_RESOLVED_ADDR_OFFSET = 136;
inline constexpr int32_t SCHEDULER_GRAPH_INVALID_KERNEL_ID = -1;

enum class SchedulerGraphResult : uint64_t {
    OK = 0,
    EMPTY = 1,
    INVALID_TASK_COUNT = 2,
    INVALID_TASK_ID = 3,
    UNSUPPORTED_SHAPE = 5,
    INVALID_ARGUMENTS = 6,
    INVALID_CALLABLE = 7,
    INVALID_FANIN_COUNT = 8,
    INVALID_FANIN_ID = 9,
    TIMEOUT = 10,
};

inline constexpr bool
scheduler_resident_v0_task_shape_supported(uint32_t active_subtasks, uint32_t logical_block_num, bool sync_start) {
    return active_subtasks >= 1 && active_subtasks <= 3 && logical_block_num == 1 && !sync_start;
}

struct SchedulerGraphView {
    // Base of the ChipTaskStorage array. One address, because a task's descriptor
    // and payload are members of one entry rather than peers in two arrays.
    uint64_t storage_address;
    // Held so the view keeps its 32-byte wire size. A producer writes 0; the two
    // entry points below reject a non-zero read, so a later field cannot be
    // reintroduced over a value some earlier producer already put here.
    uint64_t reserved;
    uint64_t task_count;
    // Kept in the wire view for diagnostics. HBG task ids directly index the
    // whole-graph-resident tables and never wrap, so this value must not be
    // used to mask an id (the configured capacity need not be a power of two).
    uint64_t task_window_last_index;
};

struct SchedulerTaskInfo {
    int64_t task_id;
    int32_t kernel_id;
    int32_t subtask_slot;
    CoreType core_type;
};

struct SchedulerTaskShape {
    int64_t task_id;
    int32_t kernel_ids[3];
    uint8_t active_mask;
    uint8_t reserved[3];
};

struct SchedulerDispatchPredicate {
    uint64_t addr;
    int64_t target;
    uint8_t elem_size;
    uint8_t op;
};

static_assert(sizeof(SchedulerGraphView) == 32, "read-only graph view layout changed");
static_assert(offsetof(SchedulerGraphView, storage_address) == 0, "storage address offset changed");
static_assert(sizeof(SchedulerTaskInfo) == 24, "root classification result layout changed");
static_assert(sizeof(SchedulerTaskShape) == 24, "task shape result layout changed");
static_assert(sizeof(SchedulerDispatchPredicate) == 24, "dispatch predicate layout changed");

inline __aicore__ bool
scheduler_dispatch_predicate_metadata_valid(uint64_t address, uint8_t element_size, uint8_t operation) {
    if (operation == 0) return true;
    const bool valid_width = element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8;
    return operation <= 6 && valid_width && address != 0 && (address & (static_cast<uint64_t>(element_size) - 1)) == 0;
}

inline __aicore__ __gm__ uint8_t *scheduler_graph_storage(const SchedulerGraphView &graph, int64_t task_id) {
    uint64_t slot = static_cast<uint64_t>(task_id);
    return reinterpret_cast<__gm__ uint8_t *>(graph.storage_address) + slot * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE;
}
inline __aicore__ __gm__ uint8_t *scheduler_graph_descriptor(const SchedulerGraphView &graph, int64_t task_id) {
    return scheduler_graph_storage(graph, task_id) + SCHEDULER_GRAPH_DESCRIPTOR_OFFSET;
}
inline __aicore__ __gm__ uint8_t *scheduler_graph_payload(const SchedulerGraphView &graph, int64_t task_id) {
    return scheduler_graph_storage(graph, task_id) + SCHEDULER_GRAPH_PAYLOAD_OFFSET;
}

inline __aicore__ int32_t
scheduler_graph_fanin_id(const SchedulerGraphView &graph, int64_t task_id, int32_t fanin_index) {
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
    __gm__ uint8_t *field = payload + TASKPAYLOAD_FANIN_DELTA_OFFSET;
    int32_t delta = *reinterpret_cast<__gm__ int32_t *>(field);
    return reinterpret_cast<__gm__ int32_t *>(field + delta)[fanin_index];
}

inline __aicore__ SchedulerGraphResult
scheduler_classify_task_shape(const SchedulerGraphView &graph, int64_t task_id, SchedulerTaskShape *shape) {
    if (shape == nullptr) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
    *shape = {
        -1,
        {SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID},
        0,
        {0, 0, 0}
    };
    if (graph.task_count == 0) return SchedulerGraphResult::EMPTY;
    if (graph.reserved != 0) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.storage_address == 0 || task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        return SchedulerGraphResult::INVALID_TASK_COUNT;
    }

    __gm__ uint8_t *descriptor = scheduler_graph_descriptor(graph, task_id);
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
    uint64_t stored_task_id = *reinterpret_cast<__gm__ uint64_t *>(descriptor + SCHEDULER_GRAPH_TASK_ID_OFFSET);
    if (stored_task_id != static_cast<uint64_t>(task_id)) return SchedulerGraphResult::INVALID_TASK_ID;

    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    if (fanin_count < 0 || fanin_count > SCHEDULER_GRAPH_MAX_FANIN) {
        return SchedulerGraphResult::INVALID_FANIN_COUNT;
    }
    __gm__ uint8_t *fanin_field = payload + TASKPAYLOAD_FANIN_DELTA_OFFSET;
    if (fanin_count > 0 && *reinterpret_cast<__gm__ int32_t *>(fanin_field) == 0) {
        return SchedulerGraphResult::INVALID_FANIN_ID;
    }
    for (int32_t i = 0; i < fanin_count; ++i) {
        int32_t producer = scheduler_graph_fanin_id(graph, task_id, i);
        if (producer < 0 || producer >= task_id) return SchedulerGraphResult::INVALID_FANIN_ID;
        for (int32_t prior = 0; prior < i; ++prior) {
            if (scheduler_graph_fanin_id(graph, task_id, prior) == producer) {
                return SchedulerGraphResult::INVALID_FANIN_ID;
            }
        }
    }

    __gm__ int32_t *kernel_ids = reinterpret_cast<__gm__ int32_t *>(descriptor + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET);
    for (int32_t slot = 0; slot < 3; ++slot) {
        if (kernel_ids[slot] == SCHEDULER_GRAPH_INVALID_KERNEL_ID) continue;
        if (kernel_ids[slot] < 0) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
        shape->kernel_ids[slot] = kernel_ids[slot];
        shape->active_mask |= static_cast<uint8_t>(1U << slot);
    }
    if (shape->active_mask == 0) return SchedulerGraphResult::UNSUPPORTED_SHAPE;

    shape->task_id = task_id;
    return SchedulerGraphResult::OK;
}

struct SchedulerTaskPayloadArguments {
    __gm__ uint8_t *tensors{nullptr};
    __gm__ uint64_t *scalars{nullptr};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
};

inline __aicore__ SchedulerGraphResult scheduler_parse_task_payload_arguments(
    const SchedulerGraphView &graph, int64_t task_id, SchedulerTaskPayloadArguments *arguments
) {
    if (arguments == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.reserved != 0) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.storage_address == 0 || task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        return SchedulerGraphResult::INVALID_TASK_COUNT;
    }
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
    int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(payload + TASKPAYLOAD_TENSOR_COUNT_OFFSET);
    int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(payload + TASKPAYLOAD_SCALAR_COUNT_OFFSET);
    if (tensor_count < 0 || tensor_count > MAX_TENSOR_ARGS || scalar_count < 0 || scalar_count > MAX_SCALAR_ARGS ||
        tensor_count + scalar_count > SPMD_LOCAL_CONTEXT_INDEX) {
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    }

    __gm__ uint8_t *tensors_field = payload + TASKPAYLOAD_TENSORS_DELTA_OFFSET;
    __gm__ uint8_t *scalars_field = payload + TASKPAYLOAD_SCALARS_DELTA_OFFSET;
    int32_t tensors_delta = *reinterpret_cast<__gm__ int32_t *>(tensors_field);
    int32_t scalars_delta = *reinterpret_cast<__gm__ int32_t *>(scalars_field);
    if ((tensor_count > 0 && tensors_delta == 0) || (scalar_count > 0 && scalars_delta == 0)) {
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    }
    arguments->tensors = tensors_field + tensors_delta;
    arguments->scalars = reinterpret_cast<__gm__ uint64_t *>(scalars_field + scalars_delta);
    arguments->tensor_count = tensor_count;
    arguments->scalar_count = scalar_count;
    return SchedulerGraphResult::OK;
}

inline __aicore__ SchedulerGraphResult scheduler_materialize_task_payload_resolved(
    const SchedulerGraphView &graph, const SchedulerTaskInfo &task, uint64_t function_bin_address,
    __gm__ DispatchPayload *dispatch_payload, const SchedulerTaskPayloadArguments *shared_arguments = nullptr,
    bool write_arguments = true
) {
    if (dispatch_payload == nullptr || function_bin_address == 0) {
        return SchedulerGraphResult::INVALID_CALLABLE;
    }
    SchedulerTaskPayloadArguments parsed{};
    if (shared_arguments == nullptr) {
        const SchedulerGraphResult status = scheduler_parse_task_payload_arguments(graph, task.task_id, &parsed);
        if (status != SchedulerGraphResult::OK) return status;
        shared_arguments = &parsed;
    }
    dispatch_payload->function_bin_addr = function_bin_address;
    if (write_arguments) {
        int32_t n = 0;
        for (int32_t i = 0; i < shared_arguments->tensor_count; ++i) {
            dispatch_payload->args[n++] = reinterpret_cast<uint64_t>(
                shared_arguments->tensors + static_cast<uint64_t>(i) * TASKPAYLOAD_TENSOR_STRIDE
            );
        }
        for (int32_t i = 0; i < shared_arguments->scalar_count; ++i)
            dispatch_payload->args[n++] = shared_arguments->scalars[i];
    }

    return SchedulerGraphResult::OK;
}

inline __aicore__ void scheduler_materialize_mix_arguments(
    const SchedulerTaskPayloadArguments &arguments, __gm__ DispatchPayload *const *targets, uint8_t count
) {
    for (int32_t i = 0; i < arguments.tensor_count; ++i) {
        const uint64_t value =
            reinterpret_cast<uint64_t>(arguments.tensors + static_cast<uint64_t>(i) * TASKPAYLOAD_TENSOR_STRIDE);
        for (uint8_t lane = 0; lane < count; ++lane)
            targets[lane]->args[i] = value;
    }
    for (int32_t i = 0; i < arguments.scalar_count; ++i) {
        const uint64_t value = arguments.scalars[i];
        for (uint8_t lane = 0; lane < count; ++lane)
            targets[lane]->args[arguments.tensor_count + i] = value;
    }
}
