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

#include "common/lane_mask.h"

#include <stddef.h>
#include <stdint.h>

#include "common/core_type.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "scheduler_native_graph_access.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

// Field offsets in the shared HBG task image. The Host layout declarations
// check these against ChipTaskStorage and TaskPayload at compile time.
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

inline __aicore__ bool scheduler_node_is_control(SchedulerNodeKind kind) {
    return kind == SchedulerNodeKind::GRAPH_ENTER || kind == SchedulerNodeKind::GRAPH_EXIT ||
           kind == SchedulerNodeKind::CONTROL_DUMMY;
}

struct SchedulerGraphView {
    // Base of the ChipTaskStorage array. One address, because a task's descriptor
    // and payload are members of one entry rather than peers in two arrays.
    uint64_t storage_address;
    // Reserved in the common prefix. Every producer writes zero.
    uint64_t reserved;
    uint64_t task_count;
    // Kept in the wire view for diagnostics. HBG task refs directly index the
    // whole-graph-resident tables and never wrap, so this value must not be
    // used to mask a ref (the configured capacity need not be a power of two).
    uint64_t task_window_last_index;
    uint64_t scheduler_state_address;
    uint64_t native_header_offset;
    uint64_t scheduler_state_bytes;
    uint32_t original_task_count;
    uint32_t view_bytes;
};

inline __aicore__ constexpr SchedulerGraphView
scheduler_ordinary_graph_view(uint64_t storage_address, uint32_t task_count, uint64_t window_last_index) {
    return {storage_address, 0, task_count, window_last_index, 0, 0, 0, task_count, sizeof(SchedulerGraphView)};
}

inline __aicore__ bool scheduler_graph_view_format_valid(const SchedulerGraphView &graph) {
    return graph.reserved == 0 && graph.view_bytes == sizeof(SchedulerGraphView);
}

inline __aicore__ bool scheduler_graph_storage_extent_valid(const SchedulerGraphView &graph) {
    if (!scheduler_graph_view_format_valid(graph) || graph.task_count > INT32_MAX) return false;
    const uint64_t storage_count = graph.original_task_count;
    if (storage_count != 0 &&
        (graph.storage_address == 0 || storage_count - 1 > graph.task_window_last_index ||
         storage_count > (UINT64_MAX - graph.storage_address) / SCHEDULER_GRAPH_TASK_STORAGE_STRIDE))
        return false;
    return true;
}

inline __aicore__ bool scheduler_graph_view_valid(const SchedulerGraphView &graph) {
    if (!scheduler_graph_storage_extent_valid(graph)) return false;
    if (graph.native_header_offset == 0) return graph.original_task_count == graph.task_count;
    return scheduler_native_header_checked(
               graph.scheduler_state_address, graph.scheduler_state_bytes, graph.native_header_offset, graph.task_count,
               graph.original_task_count
           ) != nullptr;
}

// A worker owns this access state for one observed READY image. Construction
// copies only the wire view; native reads still follow the bootstrap observe.
// The header and template are immutable until all workers leave that run.
class SchedulerGraphAccess : public SchedulerGraphView {
public:
    // Wire-view callers get a fresh validation scope, including reused backing.
    inline __aicore__ SchedulerGraphAccess(const SchedulerGraphView &view) :  // NOLINT(runtime/explicit)
        SchedulerGraphView(view) {}
    inline __aicore__ SchedulerGraphAccess(const SchedulerGraphAccess &other) :
        SchedulerGraphView(other) {}
    SchedulerGraphAccess &operator=(const SchedulerGraphAccess &) = delete;

    inline __aicore__ __gm__ const SchedulerNativeGraphHeader *native_header() const {
        if (header_ == nullptr && scheduler_graph_view_format_valid(*this)) {
            header_ = scheduler_native_header_checked(
                scheduler_state_address, scheduler_state_bytes, native_header_offset, task_count, original_task_count
            );
        }
        return header_;
    }

    inline __aicore__ __gm__ const SchedulerGraphTemplate *native_template(uint32_t index) const {
        const auto *header = native_header();
        if (header == nullptr || index >= header->definition_count) return nullptr;
        const auto *entry = reinterpret_cast<__gm__ const SchedulerGraphTemplate *>(
                                scheduler_state_address + header->templates_offset
                            ) +
                            index;
        if (definition_ == entry) return definition_;
        definition_ = scheduler_native_template_checked(scheduler_state_address, *header, index);
        return definition_;
    }

private:
    mutable __gm__ const SchedulerNativeGraphHeader *header_{nullptr};
    mutable __gm__ const SchedulerGraphTemplate *definition_{nullptr};
};
static_assert(sizeof(SchedulerGraphAccess) == sizeof(SchedulerGraphView) + 2 * sizeof(void *));

inline __aicore__ bool scheduler_graph_view_valid(const SchedulerGraphAccess &graph) {
    if (!scheduler_graph_storage_extent_valid(graph)) return false;
    return graph.native_header_offset == 0 ? graph.original_task_count == graph.task_count :
                                             graph.native_header() != nullptr;
}

// The three predicates every reader of a task ref goes through. They exist so
// that the one assumption behind them -- a ref is an index into tables sized by
// task_count -- is stated in three places rather than at each of the forty-odd
// sites that would otherwise open-code it. Widening the encoding to carry a
// parent (which supporting a graph task's sub-tasks needs) changes these and the
// two `_at` accessors in scheduler_ready.h, not a search for bare subscripts.

// True when `ref` names a task this graph holds. A caller that is about to
// subscript a table with the ref has to pass this first; the cast is the only
// place the signed-to-unsigned narrowing lives.
inline __aicore__ bool scheduler_task_ref_valid(const SchedulerGraphView &graph, int64_t ref) {
    return ref >= 0 && static_cast<uint64_t>(ref) < graph.task_count;
}

// True when a fanin edge runs the way the host built it: a producer is submitted
// before its consumer, so its ref is strictly smaller. Bootstrap's forward scan
// depends on this, and a graph that violates it would have the scan waiting on a
// producer that cannot have completed.
inline __aicore__ bool scheduler_task_ref_precedes(int64_t producer, int64_t consumer) {
    return producer >= 0 && producer < consumer;
}

// True when `ref` names no task. Every field that can hold a ref uses the same
// negative sentinel, so this reads the absence rather than testing a sign.
inline __aicore__ bool scheduler_task_ref_is_null(int64_t ref) { return ref < 0; }

struct SchedulerTaskInfo {
    int64_t task_ref;
    int32_t kernel_id;
    int32_t subtask_slot;
    CoreType core_type;
};

struct SchedulerTaskShape {
    int64_t task_ref;
    int32_t kernel_ids[3];
    LaneMask active_mask;
    uint8_t reserved[3];
};

struct SchedulerDispatchPredicate {
    uint64_t addr;
    int64_t target;
    uint8_t elem_size;
    uint8_t op;
};

static_assert(sizeof(SchedulerGraphView) == 64, "read-only graph view layout changed");
static_assert(offsetof(SchedulerGraphView, storage_address) == 0, "storage address offset changed");
static_assert(offsetof(SchedulerGraphView, scheduler_state_address) == 32);
static_assert(offsetof(SchedulerGraphView, native_header_offset) == 40);
static_assert(offsetof(SchedulerGraphView, scheduler_state_bytes) == 48);
static_assert(offsetof(SchedulerGraphView, original_task_count) == 56);
static_assert(offsetof(SchedulerGraphView, view_bytes) == 60);
static_assert(std::is_trivially_copyable_v<SchedulerGraphView> && std::is_standard_layout_v<SchedulerGraphView>);
static_assert(sizeof(SchedulerTaskInfo) == 24, "root classification result layout changed");
static_assert(sizeof(SchedulerTaskShape) == 24, "task shape result layout changed");
static_assert(offsetof(SchedulerTaskShape, active_mask) == 20);
static_assert(std::is_trivially_copyable_v<SchedulerTaskShape> && std::is_standard_layout_v<SchedulerTaskShape>);
static_assert(sizeof(SchedulerDispatchPredicate) == 24, "dispatch predicate layout changed");

inline __aicore__ bool
scheduler_dispatch_predicate_metadata_valid(uint64_t address, uint8_t element_size, uint8_t operation) {
    if (operation == 0) return true;
    const bool valid_width = element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8;
    return operation <= 6 && valid_width && address != 0 && (address & (static_cast<uint64_t>(element_size) - 1)) == 0;
}

inline __aicore__ __gm__ uint8_t *scheduler_graph_storage(const SchedulerGraphView &graph, int64_t task_ref) {
    if (graph.native_header_offset != 0 || !scheduler_task_ref_valid(graph, task_ref)) return nullptr;
    uint64_t slot = static_cast<uint64_t>(task_ref);
    return reinterpret_cast<__gm__ uint8_t *>(graph.storage_address) + slot * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE;
}
inline __aicore__ __gm__ uint8_t *scheduler_graph_descriptor(const SchedulerGraphView &graph, int64_t task_ref) {
    auto *storage = scheduler_graph_storage(graph, task_ref);
    return storage == nullptr ? nullptr : storage + SCHEDULER_GRAPH_DESCRIPTOR_OFFSET;
}
inline __aicore__ __gm__ uint8_t *scheduler_graph_payload(const SchedulerGraphView &graph, int64_t task_ref) {
    auto *storage = scheduler_graph_storage(graph, task_ref);
    return storage == nullptr ? nullptr : storage + SCHEDULER_GRAPH_PAYLOAD_OFFSET;
}

inline __aicore__ int32_t
scheduler_graph_fanin_id(const SchedulerGraphView &graph, int64_t task_ref, int32_t fanin_index) {
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_ref);
    __gm__ uint8_t *field = payload + TASKPAYLOAD_FANIN_DELTA_OFFSET;
    int32_t delta = *reinterpret_cast<__gm__ int32_t *>(field);
    return reinterpret_cast<__gm__ int32_t *>(field + delta)[fanin_index];
}

struct SchedulerNativeNode {
    __gm__ const SchedulerNativeGraphHeader *header{nullptr};
    __gm__ const SchedulerNodeBinding *binding{nullptr};
    __gm__ const SchedulerGraphInstance *instance{nullptr};
    __gm__ const SchedulerGraphTemplate *definition{nullptr};
};

inline __aicore__ bool
scheduler_native_node_checked(const SchedulerGraphAccess &graph, int64_t ref, SchedulerNativeNode *out) {
    if (out == nullptr || !scheduler_graph_view_format_valid(graph)) return false;
    const uint64_t base = graph.scheduler_state_address;
    auto *header = graph.native_header();
    if (header == nullptr) return false;
    auto *binding = scheduler_native_binding_checked(base, *header, ref);
    if (binding == nullptr) return false;
    *out = {header, binding, nullptr, nullptr};
    if (binding->instance_index == SCHEDULER_NATIVE_NO_INSTANCE) return true;
    auto *instance = scheduler_native_instance_checked(base, *header, binding->instance_index);
    auto *definition = instance == nullptr ? nullptr : graph.native_template(instance->definition_index);
    if (definition == nullptr || instance->child_count != definition->task_count) return false;
    auto *completions = reinterpret_cast<__gm__ const int64_t *>(base + header->completion_refs_offset);
    if (completions[instance->original_shell_id] != instance->exit_ref) return false;
    out->instance = instance;
    out->definition = definition;
    return true;
}

inline __aicore__ SchedulerGraphView scheduler_original_storage_view(const SchedulerGraphView &graph) {
    SchedulerGraphView original = graph;
    original.task_count = graph.original_task_count;
    original.native_header_offset = 0;
    return original;
}

inline __aicore__ SchedulerGraphResult
scheduler_node_kind(const SchedulerGraphAccess &graph, int64_t ref, SchedulerNodeKind *kind) {
    if (kind == nullptr || !scheduler_task_ref_valid(graph, ref)) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.native_header_offset == 0) {
        *kind = SchedulerNodeKind::TOP_KERNEL;
        return SchedulerGraphResult::OK;
    }
    SchedulerNativeNode node{};
    if (!scheduler_native_node_checked(graph, ref, &node)) return SchedulerGraphResult::INVALID_ARGUMENTS;
    *kind = node.binding->kind;
    return SchedulerGraphResult::OK;
}

inline __aicore__ __gm__ const SchedulerGraphChildPlan *
scheduler_native_child_plan_checked(const SchedulerGraphView &graph, const SchedulerNativeNode &node) {
    if (node.definition == nullptr || node.instance == nullptr ||
        node.binding->source_index >= node.definition->task_count)
        return nullptr;
    auto *child = reinterpret_cast<__gm__ const SchedulerGraphChildPlan *>(
                      graph.scheduler_state_address + node.definition->child_plans_offset
                  ) +
                  node.binding->source_index;
    if (child->reserved != 0 || child->tensor_count > MAX_TENSOR_ARGS || child->scalar_count > MAX_SCALAR_ARGS ||
        child->tensors_offset % 64 != 0 || child->scalars_offset % 64 != 0 ||
        child->tensors_offset > node.definition->args_bytes ||
        child->tensor_count > (node.definition->args_bytes - child->tensors_offset) / TASKPAYLOAD_TENSOR_STRIDE ||
        child->scalars_offset > node.definition->args_bytes ||
        child->scalar_count > (node.definition->args_bytes - child->scalars_offset) / sizeof(uint64_t) ||
        (child->predicate_offset != SCHEDULER_NATIVE_NO_PREDICATE &&
         (child->predicate_offset % 64 != 0 || child->predicate_offset > node.definition->args_bytes ||
          node.definition->args_bytes - child->predicate_offset < 64)))
        return nullptr;
    return child;
}

inline __aicore__ SchedulerGraphResult
scheduler_node_identity(const SchedulerGraphAccess &graph, int64_t ref, uint64_t *identity) {
    if (identity == nullptr || !scheduler_task_ref_valid(graph, ref)) return SchedulerGraphResult::INVALID_TASK_ID;
    if (graph.native_header_offset == 0) {
        *identity = static_cast<uint64_t>(ref);
        return SchedulerGraphResult::OK;
    }
    SchedulerNativeNode node{};
    if (!scheduler_native_node_checked(graph, ref, &node)) return SchedulerGraphResult::INVALID_TASK_ID;
    if (node.binding->kind == SchedulerNodeKind::CHILD_KERNEL ||
        (node.binding->kind == SchedulerNodeKind::CONTROL_DUMMY && node.instance != nullptr)) {
        *identity = (static_cast<uint64_t>(TaskId::Space::SUB_TASK) << TaskId::SPACE_SHIFT) |
                    (static_cast<uint64_t>(node.instance->original_shell_id) << TaskId::PARENT_SHIFT) |
                    node.binding->source_index;
    } else {
        *identity = node.binding->source_index;
    }
    return SchedulerGraphResult::OK;
}

inline __aicore__ __gm__ SchedulerDispatchPredicate *
scheduler_node_predicate(const SchedulerGraphAccess &graph, int64_t task_ref) {
    SchedulerGraphView storage_view = graph;
    int64_t storage_ref = task_ref;
    if (graph.native_header_offset != 0) {
        SchedulerNativeNode node{};
        if (!scheduler_native_node_checked(graph, task_ref, &node)) return nullptr;
        if (node.binding->kind == SchedulerNodeKind::CHILD_KERNEL) {
            auto *child = scheduler_native_child_plan_checked(graph, node);
            if (child == nullptr || child->predicate_offset == SCHEDULER_NATIVE_NO_PREDICATE ||
                node.instance->args_base == 0 || node.instance->args_base % 64 != 0 ||
                node.definition->args_bytes > UINT64_MAX - node.instance->args_base)
                return nullptr;
            return reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(
                node.instance->args_base + child->predicate_offset
            );
        }
        if (node.binding->kind != SchedulerNodeKind::TOP_KERNEL) return nullptr;
        storage_view = scheduler_original_storage_view(graph);
        storage_ref = node.binding->source_index;
    }
    auto *payload = scheduler_graph_payload(storage_view, storage_ref);
    return payload == nullptr ?
               nullptr :
               reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(payload + SCHEDULER_GRAPH_PREDICATE_OFFSET);
}

inline __aicore__ bool scheduler_native_child_fanin_row(
    const SchedulerGraphView &graph, const SchedulerNativeNode &node, uint32_t *begin, uint32_t *end
) {
    if (node.definition == nullptr || node.instance == nullptr ||
        node.binding->source_index >= node.definition->task_count)
        return false;
    auto *rows =
        reinterpret_cast<__gm__ const int32_t *>(graph.scheduler_state_address + node.definition->fanin_offsets_offset);
    const auto start = rows[node.binding->source_index];
    const auto stop = rows[node.binding->source_index + 1];
    if (start < 0 || stop < start || static_cast<uint64_t>(stop) > node.definition->edge_count) return false;
    *begin = static_cast<uint32_t>(start);
    *end = static_cast<uint32_t>(stop);
    return true;
}

// A view borrows immutable topology for one READY publication. Source indices
// are checked before completion-map lookup; resulting refs precede the consumer.
struct SchedulerFaninView {
    __gm__ const int32_t *global_refs{nullptr};
    __gm__ const uint16_t *local_refs{nullptr};
    __gm__ const int64_t *completion_refs{nullptr};
    int64_t base_ref{0};
    uint32_t count{0};
    uint32_t source_limit{0};
};

inline __aicore__ SchedulerGraphResult
scheduler_node_fanin_view(const SchedulerGraphAccess &graph, int64_t task_ref, SchedulerFaninView *view) {
    if (view == nullptr || !scheduler_graph_view_format_valid(graph)) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.storage_address == 0 || !scheduler_task_ref_valid(graph, task_ref))
        return SchedulerGraphResult::INVALID_TASK_COUNT;
    *view = {};
    __gm__ uint8_t *payload = nullptr;
    if (graph.native_header_offset != 0) {
        SchedulerNativeNode node{};
        if (!scheduler_native_node_checked(graph, task_ref, &node)) return SchedulerGraphResult::INVALID_ARGUMENTS;
        const uint64_t base = graph.scheduler_state_address;
        const auto kind = node.binding->kind;
        if (kind == SchedulerNodeKind::GRAPH_EXIT) {
            view->local_refs = reinterpret_cast<__gm__ const uint16_t *>(base + node.definition->sinks_offset);
            view->base_ref = node.instance->child_begin;
            view->source_limit = node.instance->child_count;
            view->count = static_cast<uint32_t>(node.definition->sink_count);
            return SchedulerGraphResult::OK;
        }
        if (kind == SchedulerNodeKind::CHILD_KERNEL ||
            (kind == SchedulerNodeKind::CONTROL_DUMMY && node.instance != nullptr)) {
            uint32_t begin = 0, end = 0;
            if (!scheduler_native_child_fanin_row(graph, node, &begin, &end))
                return SchedulerGraphResult::INVALID_FANIN_COUNT;
            if (begin == end) {
                view->count = 1;
                view->base_ref = node.instance->enter_ref;
            } else {
                view->local_refs =
                    reinterpret_cast<__gm__ const uint16_t *>(base + node.definition->fanin_indices_offset) + begin;
                view->base_ref = node.instance->child_begin;
                view->source_limit = node.binding->source_index;
                view->count = end - begin;
            }
            return SchedulerGraphResult::OK;
        }
        const auto original = scheduler_original_storage_view(graph);
        payload = scheduler_graph_payload(original, node.binding->source_index);
        view->completion_refs = reinterpret_cast<__gm__ const int64_t *>(base + node.header->completion_refs_offset);
        view->source_limit = node.binding->source_index;
    } else {
        payload = scheduler_graph_payload(graph, task_ref);
        view->source_limit = static_cast<uint32_t>(task_ref);
    }
    const int32_t count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    if (count < 0 || count > SCHEDULER_GRAPH_MAX_FANIN) return SchedulerGraphResult::INVALID_FANIN_COUNT;
    auto *field = payload + TASKPAYLOAD_FANIN_DELTA_OFFSET;
    const int32_t delta = *reinterpret_cast<__gm__ int32_t *>(field);
    if (count > 0 && delta == 0) return SchedulerGraphResult::INVALID_FANIN_ID;
    view->global_refs = reinterpret_cast<__gm__ const int32_t *>(field + delta);
    view->count = static_cast<uint32_t>(count);
    return SchedulerGraphResult::OK;
}

inline __aicore__ SchedulerGraphResult
scheduler_fanin_ref(const SchedulerFaninView &view, int64_t task_ref, uint32_t index, int64_t *producer) {
    if (producer == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (index >= view.count) return SchedulerGraphResult::INVALID_FANIN_ID;
    int64_t value = view.base_ref;
    if (view.local_refs != nullptr || view.global_refs != nullptr) {
        const int64_t source = view.local_refs != nullptr ? view.local_refs[index] : view.global_refs[index];
        if (source < 0 || static_cast<uint64_t>(source) >= view.source_limit)
            return SchedulerGraphResult::INVALID_FANIN_ID;
        value = view.completion_refs != nullptr ? view.completion_refs[source] : view.base_ref + source;
    }
    if (!scheduler_task_ref_precedes(value, task_ref)) return SchedulerGraphResult::INVALID_FANIN_ID;
    *producer = value;
    return SchedulerGraphResult::OK;
}

inline __aicore__ SchedulerGraphResult
scheduler_node_fanin_count(const SchedulerGraphAccess &graph, int64_t task_ref, uint32_t *count) {
    if (count == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    SchedulerFaninView view{};
    const auto status = scheduler_node_fanin_view(graph, task_ref, &view);
    if (status == SchedulerGraphResult::OK) *count = view.count;
    return status;
}

inline __aicore__ SchedulerGraphResult
scheduler_node_fanin_ref(const SchedulerGraphAccess &graph, int64_t task_ref, uint32_t index, int64_t *producer) {
    if (producer == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    SchedulerFaninView view{};
    const auto status = scheduler_node_fanin_view(graph, task_ref, &view);
    return status == SchedulerGraphResult::OK ? scheduler_fanin_ref(view, task_ref, index, producer) : status;
}

inline __aicore__ SchedulerGraphResult
scheduler_classify_task_shape(const SchedulerGraphAccess &graph, int64_t task_ref, SchedulerTaskShape *shape) {
    if (shape == nullptr) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
    *shape = {
        -1,
        {SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID, SCHEDULER_GRAPH_INVALID_KERNEL_ID},
        LaneMask::none(),
        {0, 0, 0}
    };
    if (graph.task_count == 0) return SchedulerGraphResult::EMPTY;
    if (graph.reserved != 0) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.storage_address == 0 || !scheduler_task_ref_valid(graph, task_ref)) {
        return SchedulerGraphResult::INVALID_TASK_COUNT;
    }

    if (!scheduler_graph_view_valid(graph)) return SchedulerGraphResult::INVALID_ARGUMENTS;

    SchedulerGraphView storage_view = graph;
    int64_t storage_ref = task_ref;
    if (graph.native_header_offset != 0) {
        SchedulerNativeNode node{};
        if (!scheduler_native_node_checked(graph, task_ref, &node)) return SchedulerGraphResult::INVALID_ARGUMENTS;
        if (node.binding->kind == SchedulerNodeKind::CHILD_KERNEL) {
            auto *child = scheduler_native_child_plan_checked(graph, node);
            if (child == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
            shape->active_mask = scheduler_lane_mask_load(child->metadata.active_mask);
            if (shape->active_mask.is_empty() || !shape->active_mask.is_valid())
                return SchedulerGraphResult::UNSUPPORTED_SHAPE;
            for (uint32_t slot = 0; slot < 3; ++slot) {
                if (shape->active_mask.has(static_cast<SubtaskSlot>(slot)))
                    shape->kernel_ids[slot] = child->metadata.kernel_ids[slot];
            }
            shape->task_ref = task_ref;
            return SchedulerGraphResult::OK;
        }
        if (node.binding->kind != SchedulerNodeKind::TOP_KERNEL) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
        storage_view = scheduler_original_storage_view(graph);
        storage_ref = node.binding->source_index;
    }
    __gm__ uint8_t *descriptor = scheduler_graph_descriptor(storage_view, storage_ref);
    uint64_t stored_task_ref = *reinterpret_cast<__gm__ uint64_t *>(descriptor + SCHEDULER_GRAPH_TASK_ID_OFFSET);
    if (stored_task_ref != static_cast<uint64_t>(storage_ref)) return SchedulerGraphResult::INVALID_TASK_ID;

    SchedulerFaninView fanin{};
    const auto fanin_status = scheduler_node_fanin_view(graph, task_ref, &fanin);
    if (fanin_status != SchedulerGraphResult::OK) return fanin_status;
    for (uint32_t i = 0; i < fanin.count; ++i) {
        int64_t producer = -1;
        if (scheduler_fanin_ref(fanin, task_ref, i, &producer) != SchedulerGraphResult::OK)
            return SchedulerGraphResult::INVALID_FANIN_ID;
        if (graph.native_header_offset == 0) {
            for (uint32_t prior = 0; prior < i; ++prior) {
                if (fanin.global_refs[prior] == producer) return SchedulerGraphResult::INVALID_FANIN_ID;
            }
            continue;
        }
        for (uint32_t prior = 0; prior < i; ++prior) {
            int64_t previous = -1;
            if (scheduler_fanin_ref(fanin, task_ref, prior, &previous) != SchedulerGraphResult::OK ||
                previous == producer) {
                return SchedulerGraphResult::INVALID_FANIN_ID;
            }
        }
    }

    __gm__ int32_t *kernel_ids = reinterpret_cast<__gm__ int32_t *>(descriptor + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET);
    for (int32_t slot = 0; slot < 3; ++slot) {
        if (kernel_ids[slot] == SCHEDULER_GRAPH_INVALID_KERNEL_ID) continue;
        if (kernel_ids[slot] < 0) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
        shape->kernel_ids[slot] = kernel_ids[slot];
    }
    shape->active_mask = LaneMask(
        shape->kernel_ids[0] != SCHEDULER_GRAPH_INVALID_KERNEL_ID,
        shape->kernel_ids[1] != SCHEDULER_GRAPH_INVALID_KERNEL_ID,
        shape->kernel_ids[2] != SCHEDULER_GRAPH_INVALID_KERNEL_ID
    );
    if (shape->active_mask.is_empty()) return SchedulerGraphResult::UNSUPPORTED_SHAPE;

    shape->task_ref = task_ref;
    return SchedulerGraphResult::OK;
}

struct SchedulerTaskPayloadArguments {
    __gm__ uint8_t *tensors{nullptr};
    __gm__ uint64_t *scalars{nullptr};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
};

inline __aicore__ SchedulerGraphResult scheduler_parse_task_payload_arguments(
    const SchedulerGraphAccess &graph, int64_t task_ref, SchedulerTaskPayloadArguments *arguments
) {
    if (arguments == nullptr) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.reserved != 0) return SchedulerGraphResult::INVALID_ARGUMENTS;
    if (graph.storage_address == 0 || !scheduler_task_ref_valid(graph, task_ref)) {
        return SchedulerGraphResult::INVALID_TASK_COUNT;
    }
    if (!scheduler_graph_view_valid(graph)) return SchedulerGraphResult::INVALID_ARGUMENTS;
    SchedulerGraphView storage_view = graph;
    int64_t storage_ref = task_ref;
    if (graph.native_header_offset != 0) {
        SchedulerNativeNode node{};
        if (!scheduler_native_node_checked(graph, task_ref, &node)) return SchedulerGraphResult::INVALID_ARGUMENTS;
        if (node.binding->kind == SchedulerNodeKind::CHILD_KERNEL) {
            auto *child = scheduler_native_child_plan_checked(graph, node);
            if (child == nullptr || node.instance->args_base % 64 != 0 ||
                (node.definition->args_bytes != 0 && node.instance->args_base == 0) ||
                node.definition->args_bytes > UINT64_MAX - node.instance->args_base)
                return SchedulerGraphResult::INVALID_ARGUMENTS;
            arguments->tensor_count = static_cast<int32_t>(child->tensor_count);
            arguments->scalar_count = static_cast<int32_t>(child->scalar_count);
            arguments->tensors =
                child->tensor_count == 0 ?
                    nullptr :
                    reinterpret_cast<__gm__ uint8_t *>(node.instance->args_base + child->tensors_offset);
            arguments->scalars =
                child->scalar_count == 0 ?
                    nullptr :
                    reinterpret_cast<__gm__ uint64_t *>(node.instance->args_base + child->scalars_offset);
            return SchedulerGraphResult::OK;
        }
        if (node.binding->kind != SchedulerNodeKind::TOP_KERNEL) return SchedulerGraphResult::UNSUPPORTED_SHAPE;
        storage_view = scheduler_original_storage_view(graph);
        storage_ref = node.binding->source_index;
    }
    __gm__ uint8_t *payload = scheduler_graph_payload(storage_view, storage_ref);
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
