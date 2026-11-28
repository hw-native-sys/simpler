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

#include "graph_validation.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "host_build_graph/constants.h"
#include "host_build_graph/graph.h"
#include "host_build_graph/types.h"
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
bool section(const Graph &def, uint32_t offset, int32_t count, std::vector<Region> &regions) {
    if (count < 0) return false;
    if (count == 0) return offset == 0;
    if (offset < sizeof(Graph) || offset % alignof(T) != 0 || offset > def.total_bytes ||
        static_cast<uint64_t>(count) > (def.total_bytes - offset) / sizeof(T))
        return false;
    regions.push_back({offset, offset + static_cast<uint64_t>(count) * sizeof(T)});
    return true;
}

template <typename T>
T element(const std::byte *image, uint32_t offset, uint32_t index) {
    T value;
    std::memcpy(&value, image + offset + static_cast<size_t>(index) * sizeof(T), sizeof(T));
    return value;
}

bool tensor_record_valid(const simpler::hbg::TensorData &tensor, const Graph &def, int32_t consumer) {
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

bool validate_graph_image(const void *image, size_t bytes, uint64_t graph_key, std::string *error) {
    if (image == nullptr || bytes < sizeof(Graph) || bytes > UINT32_MAX)
        return fail(error, "invalid Definition extent");
    Graph def{};
    std::memcpy(&def, image, sizeof(def));
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
    const auto *source = static_cast<const std::byte *>(image);
    const auto in = [&](int32_t i) {
        return element<int32_t>(source, def.off_fanin_offsets, i);
    };
    const auto out_row = [&](int32_t i) {
        return element<int32_t>(source, def.off_fanout_offsets, i);
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
            const auto producer = element<uint16_t>(source, def.off_fanin_indices, j);
            if (producer >= consumer || seen[producer] == consumer)
                return fail(error, "duplicate or nonpreceding fanin");
            seen[producer] = consumer;
            transpose[producer].push_back(static_cast<uint16_t>(consumer));
        }
    }
    if (roots.size() != static_cast<size_t>(def.root_count)) return fail(error, "incomplete root set");
    std::fill(seen.begin(), seen.end(), -1);
    for (int32_t i = 0; i < def.root_count; ++i) {
        const auto root = element<uint16_t>(source, def.off_root_indices, i);
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
        for (auto consumer : transpose[producer])
            seen[consumer] = producer;
        for (int32_t j = begin; j < end; ++j) {
            const auto consumer = element<uint16_t>(source, def.off_fanout_indices, j);
            if (consumer <= producer || consumer >= def.task_count || seen[consumer] != producer)
                return fail(error, "duplicate or asymmetric fanout");
            seen[consumer] = -1;
        }
    }

    uint64_t required_heap = 0;
    int32_t tensor_cursor = 0;
    int32_t scalar_cursor = 0;
    for (int32_t i = 0; i < def.task_count; ++i) {
        const auto task = element<SubTaskDefinition>(source, def.off_sub_tasks, i);
        const TaskAttrs attrs(task.task_attrs);
        const auto mask = task.active_mask;
        const auto active = static_cast<uint32_t>(mask.count());
        if (!mask.is_valid() || task.reserved != 0 ||
            (task.ed_flags &
             ~(simpler::hbg::recorded_dispatch::candidate | simpler::hbg::recorded_dispatch::tracked)) != 0 ||
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
            ((task.ed_flags & simpler::hbg::recorded_dispatch::candidate) != 0 &&
             (in(i) == in(i + 1) || attrs.has_predicate() || active == 0)))
            return fail(error, "invalid child shape or argument span");
        if (task.tensor_offset != tensor_cursor || task.scalar_offset != scalar_cursor)
            return fail(error, "noncanonical child argument span");
        tensor_cursor += task.tensor_count;
        scalar_cursor += task.scalar_count;
        const auto offset = element<uint64_t>(source, def.off_sub_task_offsets, i);
        const uint64_t output_bytes = CHIP_ALIGN_UP(static_cast<uint64_t>(task.total_output_size), CHIP_ALIGN_SIZE);
        if (offset != required_heap || required_heap > def.required_heap ||
            output_bytes > def.required_heap - required_heap)
            return fail(error, "invalid child heap extent");
        required_heap += output_bytes;
        for (int32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            const bool enabled = mask.has(static_cast<SubtaskSlot>(slot));
            if ((!enabled && task.kernel_id[slot] != INVALID_KERNEL_ID) ||
                (enabled && (task.kernel_id[slot] < 0 || task.kernel_id[slot] >= CHIP_MAX_FUNC_ID)))
                return fail(error, "invalid child callable");
        }
        for (int32_t j = 0; j < task.tensor_count; ++j) {
            const auto tensor = element<simpler::hbg::TensorData>(source, def.off_tensors, task.tensor_offset + j);
            if (!tensor_record_valid(tensor, def, i)) return fail(error, "invalid tensor relocation");
        }
        for (int32_t j = 0; j < task.scalar_count; ++j) {
            const auto ref =
                element<GraphScalarInheritance>(source, def.off_scalar_inheritance, task.scalar_offset + j);
            if (!ref.valid() || (ref.inherited() && ref.boundary_index() >= def.boundary_scalar_count))
                return fail(error, "invalid inherited scalar index");
        }
        if (attrs.has_predicate()) {
            const auto predicate = element<GraphPredicate>(source, def.off_predicates, task.predicate_slot - 1);
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
    if (tensor_cursor != def.tensor_arg_count || scalar_cursor != def.scalar_arg_count)
        return fail(error, "incomplete child argument pool coverage");
    if (required_heap != def.required_heap) return fail(error, "incomplete child heap coverage");
    if (error != nullptr) error->clear();
    return true;
}
