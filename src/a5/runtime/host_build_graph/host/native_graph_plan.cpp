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

#include "native_graph_plan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/types.h"
#include "scheduler/scheduler_graph.h"
#include "task_interface/callable_limits.h"

namespace {
bool fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

struct Region {
    uint32_t begin;
    uint64_t end;
};

template <typename T>
bool section(const GraphDefinition &def, uint32_t offset, int32_t count, std::vector<Region> &regions) {
    if (count < 0) return false;
    if (count == 0) return offset == 0;
    if (offset < sizeof(GraphDefinition) || offset % alignof(T) != 0 || offset > def.total_bytes ||
        static_cast<uint64_t>(count) > (def.total_bytes - offset) / sizeof(T))
        return false;
    regions.push_back({offset, offset + static_cast<uint64_t>(count) * sizeof(T)});
    return true;
}

template <typename T>
T element(const std::vector<std::byte> &image, uint32_t offset, uint32_t index) {
    T value;
    std::memcpy(&value, image.data() + offset + static_cast<size_t>(index) * sizeof(T), sizeof(T));
    return value;
}

bool tensor_record_valid(const simpler::hbg::TensorData &tensor, const GraphDefinition &def, int32_t consumer) {
    const TaskId owner = tensor.owner_task_id;
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS ||
        (tensor.address_space != AddressSpace::HOST && tensor.address_space != AddressSpace::DEVICE))
        return false;
    if (owner.space() == TaskId::Space::PARAM) {
        return owner.local_id() >= 0 && owner.local_id() < def.boundary_tensor_count &&
               owner == TaskId::make_param(owner.local_id()) && tensor.buffer.addr == 0;
    }
    return owner.space() == TaskId::Space::SUB_TASK && owner.local_id() >= 0 && owner.local_id() <= consumer &&
           tensor.buffer.addr <= def.required_heap && tensor.buffer.size <= def.required_heap - tensor.buffer.addr;
}
}  // namespace

bool native_graph_plan_definition(
    const void *image, size_t bytes, uint64_t graph_key, NativeGraphDefinitionPlan *out, std::string *error
) {
    if (out == nullptr || image == nullptr || bytes < sizeof(GraphDefinition) || bytes > UINT32_MAX)
        return fail(error, "invalid Definition extent");
    NativeGraphDefinitionPlan next;
    std::memcpy(&next.definition, image, sizeof(GraphDefinition));
    const auto &def = next.definition;
    if (def.total_bytes != bytes || def.graph_key != graph_key || def.task_count <= 0 ||
        def.task_count > SUB_TASK_MAX_NUM || def.edge_count < 0 ||
        def.edge_count > def.task_count * (def.task_count - 1) / 2 || def.root_count <= 0 ||
        def.root_count > def.task_count || def.boundary_tensor_count < 0 ||
        def.boundary_tensor_count > GRAPH_MAX_TENSOR_ARGS || def.boundary_scalar_count < 0 ||
        def.boundary_scalar_count > GRAPH_MAX_SCALAR_ARGS || def.tensor_arg_count < 0 ||
        def.tensor_arg_count > def.task_count * MAX_TENSOR_ARGS || def.scalar_arg_count < 0 ||
        def.scalar_arg_count > def.task_count * MAX_SCALAR_ARGS || def.predicate_count < 0 ||
        def.predicate_count > def.task_count)
        return fail(error, "invalid Definition header counts");
    GraphExecutionStorageLayout legacy{};
    if (!graph_execution_storage_layout(
            def.task_count, def.tensor_arg_count, def.scalar_arg_count, def.edge_count, &legacy
        ) ||
        legacy.total_bytes != def.execution_storage_bytes)
        return fail(error, "invalid reserved execution storage");
    std::vector<Region> regions;
    if (!section<int32_t>(def, def.off_fanout_offsets, def.task_count + 1, regions) ||
        !section<uint16_t>(def, def.off_fanout_indices, def.edge_count, regions) ||
        !section<int32_t>(def, def.off_fanin_offsets, def.task_count + 1, regions) ||
        !section<uint16_t>(def, def.off_fanin_indices, def.edge_count, regions) ||
        !section<uint16_t>(def, def.off_root_indices, def.root_count, regions) ||
        !section<uint64_t>(def, def.off_sub_task_offsets, def.task_count, regions) ||
        !section<SubTaskDefinition>(def, def.off_sub_tasks, def.task_count, regions) ||
        !section<simpler::hbg::TensorData>(def, def.off_tensors, def.tensor_arg_count, regions) ||
        !section<uint64_t>(def, def.off_scalars, def.scalar_arg_count, regions) ||
        !section<GraphScalarInheritance>(def, def.off_scalar_inheritance, def.scalar_arg_count, regions) ||
        !section<GraphPredicate>(def, def.off_predicates, def.predicate_count, regions))
        return fail(error, "invalid Definition section extent or alignment");
    std::sort(regions.begin(), regions.end(), [](const auto &a, const auto &b) {
        return a.begin < b.begin;
    });
    for (size_t i = 1; i < regions.size(); ++i)
        if (regions[i].begin < regions[i - 1].end) return fail(error, "overlapping Definition sections");
    next.image.resize(bytes);
    std::memcpy(next.image.data(), image, bytes);
    const auto in = [&](int32_t i) {
        return element<int32_t>(next.image, def.off_fanin_offsets, i);
    };
    const auto out_row = [&](int32_t i) {
        return element<int32_t>(next.image, def.off_fanout_offsets, i);
    };
    if (in(0) != 0 || out_row(0) != 0 || in(def.task_count) != def.edge_count ||
        out_row(def.task_count) != def.edge_count)
        return fail(error, "invalid CSR endpoints");
    // A transpose built from fanin is compared with each fanout row using stamps.
    // This detects duplicates and asymmetric edge sets in O(nodes + edges).
    std::vector<std::vector<uint16_t>> transpose(static_cast<size_t>(def.task_count));
    std::vector<int32_t> seen(static_cast<size_t>(def.task_count), -1);
    std::vector<uint16_t> roots;
    for (int32_t consumer = 0; consumer < def.task_count; ++consumer) {
        const auto begin = in(consumer), end = in(consumer + 1);
        if (begin < 0 || begin > end || end > def.edge_count) return fail(error, "invalid fanin CSR row");
        if (begin == end) roots.push_back(static_cast<uint16_t>(consumer));
        for (int32_t j = begin; j < end; ++j) {
            const auto producer = element<uint16_t>(next.image, def.off_fanin_indices, j);
            if (producer >= consumer || seen[producer] == consumer)
                return fail(error, "duplicate or nonpreceding fanin");
            seen[producer] = consumer;
            transpose[producer].push_back(static_cast<uint16_t>(consumer));
        }
    }
    if (roots.size() != static_cast<size_t>(def.root_count)) return fail(error, "incomplete root set");
    std::fill(seen.begin(), seen.end(), -1);
    for (int32_t i = 0; i < def.root_count; ++i) {
        const auto root = element<uint16_t>(next.image, def.off_root_indices, i);
        if (root >= def.task_count || in(root) != in(root + 1) || seen[root] != -1)
            return fail(error, "invalid or duplicate root");
        seen[root] = 0;
    }
    std::fill(seen.begin(), seen.end(), -1);
    for (int32_t producer = 0; producer < def.task_count; ++producer) {
        const auto begin = out_row(producer), end = out_row(producer + 1);
        if (begin < 0 || begin > end || end > def.edge_count ||
            static_cast<size_t>(end - begin) != transpose[producer].size())
            return fail(error, "inconsistent fanout CSR row");
        if (begin == end) next.sinks.push_back(static_cast<uint16_t>(producer));
        for (auto consumer : transpose[producer])
            seen[consumer] = producer;
        for (int32_t j = begin; j < end; ++j) {
            const auto consumer = element<uint16_t>(next.image, def.off_fanout_indices, j);
            if (consumer <= producer || consumer >= def.task_count || seen[consumer] != producer)
                return fail(error, "duplicate or asymmetric fanout");
            seen[consumer] = -1;
        }
    }

    next.children.resize(static_cast<size_t>(def.task_count));
    uint64_t required_heap = 0;
    for (int32_t i = 0; i < def.task_count; ++i) {
        const auto task = element<SubTaskDefinition>(next.image, def.off_sub_tasks, i);
        const TaskAttrs attrs(task.task_attrs);
        const auto mask = task.active_mask;
        const auto active = static_cast<uint32_t>(mask.count());
        if (!mask.is_valid() || task.reserved != 0 || (task.ed_flags & ~(ED_FLAG_CANDIDATE | ED_FLAG_TRACKED)) != 0 ||
            task.logical_block_num <= 0 || task.tensor_count < 0 || task.tensor_count > MAX_TENSOR_ARGS ||
            task.scalar_count < 0 || task.scalar_count > MAX_SCALAR_ARGS || task.tensor_offset < 0 ||
            task.tensor_offset > def.tensor_arg_count ||
            task.tensor_count > def.tensor_arg_count - task.tensor_offset || task.scalar_offset < 0 ||
            task.scalar_offset > def.scalar_arg_count ||
            task.scalar_count > def.scalar_arg_count - task.scalar_offset || task.total_output_size < 0 ||
            task.total_required_subtasks != task.logical_block_num * active ||
            attrs.has_predicate() != (task.predicate_slot != 0) || task.predicate_slot > def.predicate_count ||
            (attrs.has_predicate() && (active != 1 || task.logical_block_num != 1)) ||
            (attrs.requires_sync_start() && task.logical_block_num == 1) ||
            (active == 0 && (task.logical_block_num != 1 || attrs.has_predicate() || attrs.requires_sync_start())) ||
            ((task.ed_flags & ED_FLAG_CANDIDATE) != 0 && (in(i) == in(i + 1) || attrs.has_predicate() || active == 0)))
            return fail(error, "invalid child shape or argument span");
        const auto offset = element<uint64_t>(next.image, def.off_sub_task_offsets, i);
        const uint64_t output_bytes = CHIP_ALIGN_UP(static_cast<uint64_t>(task.total_output_size), CHIP_ALIGN_SIZE);
        if (offset != required_heap || required_heap > def.required_heap ||
            output_bytes > def.required_heap - required_heap)
            return fail(error, "invalid child heap extent");
        required_heap += output_bytes;
        auto &child = next.children[i];
        child.predicate_offset = SCHEDULER_NATIVE_NO_PREDICATE;
        child.execution_prefix = next.execution_count;
        child.metadata.active_mask = mask;
        child.metadata.flags =
            active == 0 ?
                SCHEDULER_TASK_HAS_FANIN :
                scheduler_task_metadata_flags_from_submit_state(mask, attrs, task.logical_block_num, true, false);
        child.metadata.logical_block_num = static_cast<uint16_t>(task.logical_block_num);
        child.metadata.timing_slot = static_cast<int16_t>(attrs.timing_slot());
        for (int32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            const bool enabled = mask.has(static_cast<SubtaskSlot>(slot));
            if ((!enabled && task.kernel_id[slot] != INVALID_KERNEL_ID) ||
                (enabled && (task.kernel_id[slot] < 0 || task.kernel_id[slot] >= CHIP_MAX_FUNC_ID)))
                return fail(error, "invalid child callable");
            child.metadata.kernel_ids[slot] = enabled ? static_cast<uint16_t>(task.kernel_id[slot]) : UINT16_MAX;
        }
        if (active != 0) next.execution_count += static_cast<uint64_t>(task.total_required_subtasks);
        child.tensor_count = static_cast<uint32_t>(task.tensor_count);
        child.scalar_count = static_cast<uint32_t>(task.scalar_count);
        uint64_t scalar_bytes = 0;
        if (!scheduler_layout_checked_align(sizeof(uint64_t) * child.scalar_count, 64, &scalar_bytes) ||
            !scheduler_layout_reserve(
                &next.args_bytes, sizeof(simpler::hbg::Tensor) * child.tensor_count, 64, &child.tensors_offset
            ) ||
            !scheduler_layout_reserve(&next.args_bytes, scalar_bytes, 64, &child.scalars_offset) ||
            (attrs.has_predicate() && !scheduler_layout_reserve(&next.args_bytes, 64, 64, &child.predicate_offset)))
            return fail(error, "native argument layout overflow");
        for (int32_t j = 0; j < task.tensor_count; ++j) {
            const auto tensor = element<simpler::hbg::TensorData>(next.image, def.off_tensors, task.tensor_offset + j);
            if (!tensor_record_valid(tensor, def, i)) return fail(error, "invalid tensor relocation");
        }
        for (int32_t j = 0; j < task.scalar_count; ++j) {
            const auto ref =
                element<GraphScalarInheritance>(next.image, def.off_scalar_inheritance, task.scalar_offset + j);
            if (!ref.valid() || (ref.inherited() && ref.boundary_index() >= def.boundary_scalar_count))
                return fail(error, "invalid inherited scalar index");
        }
        if (attrs.has_predicate()) {
            const auto predicate = element<GraphPredicate>(next.image, def.off_predicates, task.predicate_slot - 1);
            const auto owner = predicate.operand.owner_task_id;
            const auto width = get_element_size(predicate.operand.dtype);
            if (!tensor_record_valid(predicate.operand, def, i) ||
                (owner.space() == TaskId::Space::SUB_TASK && owner.local_id() == i) || predicate.op < 1 ||
                predicate.op > 6 || predicate.elem_size != width ||
                (width != 1 && width != 2 && width != 4 && width != 8) ||
                predicate.elem_offset >= predicate.operand.extent_elem_cache ||
                std::any_of(std::begin(predicate.reserved), std::end(predicate.reserved), [](uint8_t v) {
                    return v != 0;
                }))
                return fail(error, "invalid child predicate");
        }
    }
    if (required_heap != def.required_heap) return fail(error, "incomplete child heap coverage");
    if (next.args_bytes > def.execution_storage_bytes) next.resident_supported = false;
    if (error != nullptr) error->clear();
    *out = std::move(next);
    return true;
}

bool native_graph_plan_nodes(
    const std::vector<uint32_t> &definition_indices, NativeGraphHostPlan *plan, std::string *error
) {
    if (plan == nullptr || definition_indices.size() > static_cast<size_t>(TaskId::GLOBAL_TASK_MAX_NUM))
        return fail(error, "invalid top-level task count");
    uint64_t nodes = definition_indices.size();
    for (auto index : definition_indices) {
        if (index == SCHEDULER_NATIVE_NO_INSTANCE) continue;
        if (index >= plan->definitions.size()) return fail(error, "unknown Definition index");
        const auto count = plan->definitions[index].definition.task_count;
        if (count <= 0 || count > SUB_TASK_MAX_NUM ||
            plan->definitions[index].children.size() != static_cast<size_t>(count) ||
            nodes > INT32_MAX - static_cast<uint64_t>(count) - 1)
            return fail(error, "expanded node count overflow");
        nodes += count + 1;
    }
    std::vector<SchedulerNodeBinding> bindings;
    std::vector<int64_t> completion_refs;
    std::vector<SchedulerGraphInstance> instances;
    bindings.reserve(static_cast<size_t>(nodes));
    completion_refs.reserve(definition_indices.size());
    for (uint32_t id = 0; id < definition_indices.size(); ++id) {
        const auto index = definition_indices[id];
        if (index == SCHEDULER_NATIVE_NO_INSTANCE) {
            completion_refs.push_back(static_cast<int64_t>(bindings.size()));
            bindings.push_back({id, SCHEDULER_NATIVE_NO_INSTANCE, SchedulerNodeKind::TOP_KERNEL, {}});
            continue;
        }
        const auto &definition = plan->definitions[index];
        SchedulerGraphInstance instance{};
        instance.definition_index = index;
        instance.original_shell_id = id;
        instance.enter_ref = static_cast<uint32_t>(bindings.size());
        instance.child_begin = instance.enter_ref + 1;
        instance.child_count = static_cast<uint32_t>(definition.definition.task_count);
        instance.exit_ref = instance.child_begin + instance.child_count;
        instance.boundary_tensor_count = static_cast<uint32_t>(definition.definition.boundary_tensor_count);
        instance.boundary_scalar_count = static_cast<uint32_t>(definition.definition.boundary_scalar_count);
        const auto invocation = static_cast<uint32_t>(instances.size());
        bindings.push_back({id, invocation, SchedulerNodeKind::GRAPH_ENTER, {}});
        for (uint32_t child = 0; child < instance.child_count; ++child) {
            const auto kind = definition.children[child].metadata.active_mask.is_empty() ?
                                  SchedulerNodeKind::CONTROL_DUMMY :
                                  SchedulerNodeKind::CHILD_KERNEL;
            bindings.push_back({child, invocation, kind, {}});
        }
        bindings.push_back({id, invocation, SchedulerNodeKind::GRAPH_EXIT, {}});
        completion_refs.push_back(instance.exit_ref);
        instances.push_back(instance);
    }
    plan->bindings = std::move(bindings);
    plan->completion_refs = std::move(completion_refs);
    plan->instances = std::move(instances);
    if (error != nullptr) error->clear();
    return true;
}

bool native_graph_plan_readonly_image(
    const NativeGraphHostPlan &plan, NativeGraphReadonlyImage *out, std::string *error
) {
    if (out == nullptr || plan.definitions.empty() || plan.instances.empty() || plan.bindings.empty() ||
        plan.bindings.size() > INT32_MAX || plan.completion_refs.size() > TaskId::GLOBAL_TASK_MAX_NUM)
        return fail(error, "invalid native image population");
    NativeGraphReadonlyImage next;
    SchedulerNativeGraphHeader header{};
    header.magic = SCHEDULER_NATIVE_GRAPH_MAGIC;
    header.version = SCHEDULER_NATIVE_GRAPH_VERSION;
    header.header_bytes = sizeof(header);
    header.node_count = plan.bindings.size();
    header.original_task_count = plan.completion_refs.size();
    header.definition_count = plan.definitions.size();
    header.instance_count = plan.instances.size();
    uint64_t cursor = sizeof(SchedulerRunControl);
    auto reserve = [&](uint64_t count, uint64_t stride, uint64_t alignment, uint64_t *offset) {
        uint64_t bytes = 0;
        return scheduler_layout_checked_mul(count, stride, &bytes) &&
               scheduler_layout_reserve(&cursor, bytes, alignment, offset);
    };
    if (!reserve(1, sizeof(header), alignof(SchedulerNativeGraphHeader), &next.header_offset) ||
        !reserve(
            header.node_count, sizeof(SchedulerNodeBinding), alignof(SchedulerNodeBinding), &header.bindings_offset
        ) ||
        !reserve(header.original_task_count, sizeof(int64_t), alignof(int64_t), &header.completion_refs_offset) ||
        !reserve(
            header.definition_count, sizeof(SchedulerGraphTemplate), alignof(SchedulerGraphTemplate),
            &header.templates_offset
        ) ||
        !reserve(
            header.instance_count, sizeof(SchedulerGraphInstance), alignof(SchedulerGraphInstance),
            &header.instances_offset
        ))
        return fail(error, "native readonly image overflow");
    std::vector<SchedulerGraphTemplate> templates(plan.definitions.size());
    for (size_t i = 0; i < plan.definitions.size(); ++i) {
        const auto &source = plan.definitions[i];
        const auto &def = source.definition;
        if (def.total_bytes != source.image.size() || source.children.size() != static_cast<size_t>(def.task_count) ||
            source.sinks.empty())
            return fail(error, "incomplete Definition plan");
        auto &entry = templates[i];
        entry.definition_bytes = source.image.size();
        entry.task_count = def.task_count;
        entry.edge_count = def.edge_count;
        entry.sink_count = source.sinks.size();
        entry.args_bytes = source.args_bytes;
        entry.execution_count = source.execution_count;
        entry.required_heap = def.required_heap;
        entry.legacy_execution_storage_bytes = def.execution_storage_bytes;
        entry.graph_key = def.graph_key;
        if (!reserve(entry.definition_bytes, 1, GRAPH_DEFINITION_OBJECT_ALIGN, &entry.definition_offset) ||
            !reserve(
                entry.task_count, sizeof(SchedulerGraphChildPlan), alignof(SchedulerGraphChildPlan),
                &entry.child_plans_offset
            ) ||
            !reserve(entry.sink_count, sizeof(uint16_t), alignof(uint16_t), &entry.sinks_offset))
            return fail(error, "native Definition image overflow");
        entry.fanin_offsets_offset = entry.definition_offset + def.off_fanin_offsets;
        entry.fanin_indices_offset = def.edge_count == 0 ? 0 : entry.definition_offset + def.off_fanin_indices;
    }
    if (!scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &header.readonly_end_offset) ||
        header.readonly_end_offset > std::numeric_limits<size_t>::max())
        return fail(error, "native readonly alignment overflow");
    next.bytes.resize(static_cast<size_t>(header.readonly_end_offset));
    const auto copy = [&](uint64_t offset, const auto &values) {
        if (!values.empty()) std::memcpy(next.bytes.data() + offset, values.data(), values.size() * sizeof(values[0]));
    };
    std::memcpy(next.bytes.data() + next.header_offset, &header, sizeof(header));
    copy(header.bindings_offset, plan.bindings);
    copy(header.completion_refs_offset, plan.completion_refs);
    copy(header.templates_offset, templates);
    copy(header.instances_offset, plan.instances);
    for (size_t i = 0; i < templates.size(); ++i) {
        copy(templates[i].definition_offset, plan.definitions[i].image);
        copy(templates[i].child_plans_offset, plan.definitions[i].children);
        copy(templates[i].sinks_offset, plan.definitions[i].sinks);
    }
    if (error != nullptr) error->clear();
    *out = std::move(next);
    return true;
}

bool native_graph_plan_worker_capacity(const NativeGraphHostPlan &plan, uint64_t workers, std::string *error) {
    if (workers == 0 || workers > PLATFORM_MAX_CORES || workers % PLATFORM_CORES_PER_BLOCKDIM != 0)
        return fail(error, "invalid native worker topology");
    for (const auto &definition : plan.definitions) {
        for (const auto &child : definition.children) {
            const auto &metadata = child.metadata;
            if (!scheduler_task_requires_sync_start(metadata.flags)) continue;
            SchedulerMinimumWorkerDemand demand{};
            if (!scheduler_resident_worker_demand_checked(
                    metadata.active_mask, metadata.logical_block_num, true, workers, &demand
                ))
                return fail(error, "Graph child exceeds sync-start worker capacity");
        }
    }
    return true;
}
