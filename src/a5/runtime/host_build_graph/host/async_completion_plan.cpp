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

#include "async_completion_plan.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

namespace {

using namespace pto2::resident;

bool reject(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

template <typename T>
bool reserve_array(uint64_t *cursor, uint64_t count, uint64_t *offset) {
    uint64_t bytes = 0;
    return scheduler_layout_checked_mul(count, sizeof(T), &bytes) &&
           scheduler_layout_reserve(cursor, bytes, alignof(T), offset);
}

}  // namespace

bool plan_async_completion(
    uint64_t begin_offset, uint64_t run_epoch, uint32_t task_count, const std::vector<AsyncTaskPlanInput> &inputs,
    AsyncCompletionHostPlan *out, std::string *error, uint32_t sdma_guard_count
) {
    if (out == nullptr || run_epoch == 0 || begin_offset < sizeof(SchedulerRunControl) ||
        begin_offset % ASYNC_RECORD_ALIGNMENT != 0 || task_count > INT32_MAX || inputs.size() > task_count)
        return reject(error, "invalid async image identity or bounds");
    AsyncCompletionHostPlan next;
    next.begin_offset = begin_offset;
    next.total_bytes = begin_offset;
    next.initialized_bytes = begin_offset;
    auto &layout = next.layout;
    layout.run_epoch = run_epoch;
    layout.task_count = task_count;
    const bool increasing = std::adjacent_find(inputs.begin(), inputs.end(), [](const auto &left, const auto &right) {
                                return left.task_ref >= right.task_ref;
                            }) == inputs.end();
    std::unordered_set<int64_t> task_refs;
    if (!increasing) task_refs.reserve(inputs.size());
    next.tasks.reserve(inputs.size());
    for (const auto &input : inputs) {
        if (input.task_ref < 0 || static_cast<uint64_t>(input.task_ref) >= task_count ||
            (!increasing && !task_refs.insert(input.task_ref).second) || input.block_count == 0 ||
            input.block_count > SCHEDULER_ASYNC_JOIN_COUNT_MAX || !input.active_mask.is_valid())
            return reject(error, "invalid or duplicate async task shape");
        LaneMask mask{};
        uint32_t lane_count = 0;
        if (input.block_count * static_cast<uint32_t>(input.active_mask.count()) > SCHEDULER_ASYNC_JOIN_COUNT_MAX)
            return reject(error, "task physical execution count exceeds resident capacity");
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!input.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            const uint16_t capacity = input.capacities[lane];
            if (capacity > MAX_COMPLETIONS_PER_TASK) return reject(error, "invalid completion capacity");
            if (capacity == 0) continue;
            mask.activate(static_cast<SubtaskSlot>(lane));
            ++lane_count;
        }
        if (mask.is_empty()) continue;
        const uint32_t groups = input.block_count * lane_count;
        if (groups > SCHEDULER_ASYNC_JOIN_COUNT_MAX || groups > UINT32_MAX - layout.group_count)
            return reject(error, "async group count exceeds join or index capacity");
        AsyncTaskHostPlan task;
        task.data = {input.task_ref, layout.group_count, static_cast<uint16_t>(input.block_count), mask};
        task.capacities = input.capacities;
        task.urma_single_aiv = input.block_count == 1 && input.active_mask.count() == 1 &&
                               !input.active_mask.has(static_cast<SubtaskSlot>(0));
        next.tasks.push_back(task);
        layout.group_count += groups;
    }
    if (next.tasks.empty()) {
        *out = std::move(next);
        return true;
    }
    layout.async_task_count = static_cast<uint32_t>(next.tasks.size());
    layout.sdma_guard_count = sdma_guard_count;
    uint64_t cursor = begin_offset;
    if (!reserve_array<SchedulerAsyncLayout>(&cursor, 1, &next.layout_offset) ||
        !reserve_array<SchedulerAsyncActivity>(&cursor, 1, &layout.activity_offset) ||
        !reserve_array<uint32_t>(&cursor, task_count, &layout.task_map_offset) ||
        !reserve_array<SchedulerAsyncTaskData>(&cursor, layout.async_task_count, &layout.task_data_offset) ||
        !reserve_array<SchedulerAsyncTaskJoin>(&cursor, layout.async_task_count, &layout.joins_offset) ||
        !reserve_array<SchedulerAsyncDirectoryWord>(
            &cursor, (static_cast<uint64_t>(layout.group_count) + 63) / 64, &layout.check_directory_offset
        ) ||
        !reserve_array<SchedulerAsyncDirectoryWord>(
            &cursor, (static_cast<uint64_t>(layout.async_task_count) + 63) / 64, &layout.finish_directory_offset
        ))
        return reject(error, "async control storage overflows");
    if (sdma_guard_count != 0 &&
        !reserve_array<AsyncSdmaSubmissionGuard>(&cursor, sdma_guard_count, &layout.sdma_guards_offset))
        return reject(error, "SDMA submission guards overflow");
    if (!reserve_array<AsyncGroupData>(&cursor, layout.group_count, &layout.groups_offset) ||
        !scheduler_layout_checked_align(cursor, ASYNC_RECORD_ALIGNMENT, &next.initialized_bytes))
        return reject(error, "header storage overflows");
    cursor = next.initialized_bytes;
    if (!reserve_array<SchedulerAsyncLease>(&cursor, layout.group_count, &layout.group_leases_offset) ||
        !reserve_array<SchedulerAsyncGroupProgress>(&cursor, layout.group_count, &layout.group_progress_offset) ||
        !reserve_array<SchedulerAsyncLease>(&cursor, layout.async_task_count, &layout.finish_leases_offset))
        return reject(error, "deferred control storage overflows");
    for (auto &task : next.tasks) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (task.data.async_mask.has(static_cast<SubtaskSlot>(lane)))
                task.block_entry_bytes += async_group_entry_bytes(task.capacities[lane]);
        }
        uint64_t task_entry_bytes = 0;
        if (!scheduler_layout_checked_mul(task.data.block_count, task.block_entry_bytes, &task_entry_bytes) ||
            !scheduler_layout_reserve(&cursor, task_entry_bytes, ASYNC_RECORD_ALIGNMENT, &task.entries_offset))
            return reject(error, "async publication storage overflows");
    }
    if (!scheduler_layout_checked_align(cursor, ASYNC_RECORD_ALIGNMENT, &next.total_bytes) ||
        next.total_bytes > std::numeric_limits<size_t>::max())
        return reject(error, "async image size cannot be represented");
    layout.image_bytes = next.total_bytes;
    *out = std::move(next);
    return true;
}

bool initialize_async_completion(void *image, uint64_t image_bytes, const AsyncCompletionHostPlan &plan) {
    if (plan.layout_offset == 0)
        return plan.tasks.empty() && plan.layout.group_count == 0 && plan.total_bytes == plan.begin_offset &&
               plan.initialized_bytes == plan.begin_offset;
    if (image == nullptr || reinterpret_cast<uintptr_t>(image) % ASYNC_RECORD_ALIGNMENT != 0 ||
        image_bytes < plan.total_bytes || plan.begin_offset >= plan.initialized_bytes ||
        plan.initialized_bytes > plan.total_bytes || plan.layout.image_bytes != plan.total_bytes ||
        plan.layout.async_task_count != plan.tasks.size())
        return false;
    uint64_t expected_groups = 0;
    for (const auto &planned : plan.tasks) {
        const auto &task = planned.data;
        if (task.group_base != expected_groups || task.block_count == 0 || !task.async_mask.is_valid() ||
            task.async_mask.is_empty())
            return false;
        expected_groups += static_cast<uint64_t>(task.block_count) * task.async_mask.count();
    }
    if (expected_groups != plan.layout.group_count) return false;
    *scheduler_state_at<SchedulerAsyncLayout>(image, plan.layout_offset) = plan.layout;
    if (plan.layout.sdma_guard_count != 0) {
        auto *guards = scheduler_state_at<AsyncSdmaSubmissionGuard>(image, plan.layout.sdma_guards_offset);
        for (uint32_t index = 0; index < plan.layout.sdma_guard_count; ++index)
            guards[index].word = 0;
    }
    auto *map = scheduler_state_at<uint32_t>(image, plan.layout.task_map_offset);
    scheduler_state_at<SchedulerAsyncActivity>(image, plan.layout.activity_offset)->activated = 0;
    for (uint32_t ref = 0; ref < plan.layout.task_count; ++ref)
        map[ref] = ASYNC_INVALID_INDEX;
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(image, plan.layout.task_data_offset);
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(image, plan.layout.joins_offset);
    for (uint32_t index = 0; index < plan.tasks.size(); ++index) {
        const auto &task = plan.tasks[index].data;
        tasks[index] = task;
        map[task.task_ref] = index;
        joins[index].word = 0;
    }
    auto *checks = scheduler_state_at<SchedulerAsyncDirectoryWord>(image, plan.layout.check_directory_offset);
    auto *finishes = scheduler_state_at<SchedulerAsyncDirectoryWord>(image, plan.layout.finish_directory_offset);
    for (uint64_t word = 0; word < (static_cast<uint64_t>(plan.layout.group_count) + 63) / 64; ++word)
        checks[word].bits = 0;
    for (uint64_t word = 0; word < (static_cast<uint64_t>(plan.layout.async_task_count) + 63) / 64; ++word)
        finishes[word].bits = 0;
    AsyncGroupData header{};
    header.run_epoch = plan.layout.run_epoch;
    header.sdma_guards_offset = plan.layout.sdma_guards_offset;
    header.sdma_guard_count = plan.layout.sdma_guard_count;
    uint32_t group_index = 0;
    for (const auto &planned : plan.tasks) {
        const auto &task = planned.data;
        header.task_ref = task.task_ref;
        header.urma_single_aiv = planned.urma_single_aiv ? 1 : 0;
        uint64_t entries_offset = planned.entries_offset;
        for (uint32_t block = 0; block < task.block_count; ++block) {
            header.block_index = block;
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!task.async_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                header.group_index = group_index;
                header.subtask_slot = lane;
                header.capacity = planned.capacities[lane];
                header.entries_bytes = async_group_entry_bytes(header.capacity);
                header.entries_offset = entries_offset;
                const uint64_t offset =
                    plan.layout.groups_offset + static_cast<uint64_t>(group_index) * sizeof(AsyncGroupData);
                *scheduler_state_at<AsyncGroupData>(image, offset) = header;
                entries_offset += header.entries_bytes;
                ++group_index;
            }
        }
    }
    return true;
}
