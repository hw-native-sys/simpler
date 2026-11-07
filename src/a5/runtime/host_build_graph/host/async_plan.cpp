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

#include "async_plan.h"

#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {
bool fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

template <typename T>
void write(std::byte *image, uint64_t offset, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    // A host image need not have the 128-byte alignment of device records.
    std::memcpy(image + offset, &value, sizeof(T));
}

bool reserve(uint64_t count, uint64_t stride, uint64_t &cursor, uint64_t &offset) {
    uint64_t bytes = 0;
    return scheduler_layout_checked_mul(count, stride, &bytes) &&
           scheduler_layout_reserve(&cursor, bytes, 128, &offset);
}

bool plan_async_layout(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary, SchedulerAsyncLayout *out,
    std::string *error
) {
    if (out == nullptr || run_epoch == 0 || metadata.size() != capabilities.size() || metadata.size() > INT32_MAX)
        return fail(error, "async plan requires a live run epoch and matching capability rows");
    if (auxiliary.sdma_guard_count > ASYNC_SDMA_CHANNELS ||
        auxiliary.execution_contract > HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY)
        return fail(error, "async auxiliary configuration is invalid");
    SchedulerAsyncLayout layout{};
    layout.run_epoch = run_epoch;
    layout.task_count = metadata.size();
    layout.execution_contract = auxiliary.execution_contract;
    uint64_t token_bytes = 0;
    for (size_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        uint32_t executions = 0;
        const uint32_t lanes = meta.active_mask.count();
        if (!meta.active_mask.is_valid() ||
            !scheduler_subtask_count_checked(meta.logical_block_num, lanes, &executions) ||
            scheduler_task_is_mix(meta.flags) != (lanes > 1) ||
            scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
            (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) ||
            (scheduler_task_has_predicate(meta.flags) && (meta.logical_block_num != 1 || lanes != 1)))
            return fail(error, "async plan task exceeds the physical execution contract");
        SchedulerAsyncTaskMap map;
        map.block_count = meta.logical_block_num;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            const auto cap = core_async_effective_capability(capabilities[task][lane]);
            if (!core_async_capability_valid(cap))
                return fail(error, "async plan requires a valid callable capability");
            if (cap.mode != CoreAsyncMode::DEFERRED) continue;
            map.async_lane_mask |= 1u << lane;
            ++map.groups_per_block;
            uint64_t bytes = 0, aligned = 0;
            if (!scheduler_layout_checked_mul(cap.max_conditions, sizeof(DeferredCompletionEntry), &bytes) ||
                !scheduler_layout_checked_align(bytes, 128, &aligned) ||
                !scheduler_layout_checked_mul(aligned, meta.logical_block_num, &bytes) ||
                !scheduler_layout_checked_add(token_bytes, bytes, &token_bytes))
                return fail(error, "async completion-entry storage overflows");
            if (!scheduler_layout_checked_mul(cap.max_conditions, meta.logical_block_num, &bytes) ||
                !scheduler_layout_checked_add(layout.condition_capacity, bytes, &layout.condition_capacity))
                return fail(error, "async condition capacity overflows");
        }
        if (map.groups_per_block == 0) continue;
        map.group_base = layout.group_count;
        map.group_count = map.groups_per_block * meta.logical_block_num;
        map.finish_index = layout.finish_count++;
        if (!scheduler_layout_checked_add(layout.group_count, map.group_count, &layout.group_count))
            return fail(error, "async group count overflows");
    }
    if (layout.group_count == 0) {
        if (auxiliary.backend != 0 || auxiliary.sdma_guard_count != 0)
            return fail(error, "synchronous plan cannot carry unowned async auxiliary storage");
        *out = layout;
        return true;
    }
    layout.check_directory_words = (layout.group_count + 63) / 64;
    layout.finish_directory_words = (layout.finish_count + 63) / 64;
    layout.entries_bytes = token_bytes;
    layout.backend_bytes = auxiliary.backend;
    layout.sdma_guard_count = auxiliary.sdma_guard_count;
    uint64_t cursor = sizeof(SchedulerAsyncLayout);
    if (!reserve(layout.task_count, sizeof(SchedulerAsyncTaskMap), cursor, layout.task_map_offset) ||
        !reserve(layout.group_count, sizeof(SchedulerAsyncGroupIdentity), cursor, layout.group_identity_offset) ||
        !reserve(layout.group_count, sizeof(SchedulerAsyncGroupWriter), cursor, layout.group_writer_offset) ||
        !reserve(layout.group_count, sizeof(SchedulerAsyncLease), cursor, layout.group_lease_offset) ||
        !reserve(layout.group_count, sizeof(SchedulerAsyncGroupProgress), cursor, layout.group_progress_offset) ||
        !reserve(layout.finish_count, sizeof(SchedulerAsyncFinishIdentity), cursor, layout.finish_identity_offset) ||
        !reserve(layout.finish_count, sizeof(SchedulerAsyncLease), cursor, layout.finish_lease_offset) ||
        !reserve(layout.finish_count, sizeof(SchedulerAsyncTaskJoin), cursor, layout.finish_join_offset) ||
        !reserve(layout.finish_count, sizeof(SchedulerAsyncFinishProgress), cursor, layout.finish_progress_offset) ||
        !reserve(
            layout.check_directory_words, sizeof(SchedulerAsyncDirectoryWord), cursor, layout.check_directory_offset
        ) ||
        !reserve(
            layout.finish_directory_words, sizeof(SchedulerAsyncDirectoryWord), cursor, layout.finish_directory_offset
        ) ||
        !reserve(token_bytes, 1, cursor, layout.entries_offset) ||
        !reserve(layout.backend_bytes, 1, cursor, layout.backend_offset) ||
        (layout.sdma_guard_count != 0 &&
         !reserve(layout.sdma_guard_count + 1, 128, cursor, layout.sdma_guard_context_offset)) ||
        !scheduler_layout_checked_align(cursor, 128, &layout.total_bytes) || layout.total_bytes > byte_budget ||
        layout.total_bytes > std::numeric_limits<size_t>::max())
        return fail(error, "async state exceeds the byte budget or addressable storage");
    *out = layout;
    return true;
}

bool materialize_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const SchedulerAsyncLayout &layout, std::byte *image, std::string *error
) {
    std::memset(image, 0, static_cast<size_t>(layout.total_bytes));
    write(image, 0, layout);
    if (layout.sdma_guard_count != 0) {
        AsyncSdmaGuardContext context{};
        context.run_epoch = layout.run_epoch;
        context.guard_count = layout.sdma_guard_count;
        context.bytes = (layout.sdma_guard_count + 1) * 128;
        write(image, layout.sdma_guard_context_offset, context);
    }
    uint64_t group_base = 0, finish_index = 0;
    uint64_t entry_cursor = layout.entries_offset;
    for (size_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        SchedulerAsyncTaskMap map;
        if (scheduler_task_is_executable(meta.flags) && !scheduler_task_is_inline(meta.flags)) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (meta.active_mask.has(static_cast<SubtaskSlot>(lane)) &&
                    core_async_effective_capability(capabilities[task][lane]).mode == CoreAsyncMode::DEFERRED) {
                    map.async_lane_mask |= 1u << lane;
                    ++map.groups_per_block;
                }
            }
            if (map.groups_per_block != 0) {
                map.block_count = meta.logical_block_num;
                map.group_base = group_base;
                map.group_count = map.groups_per_block * map.block_count;
                map.finish_index = finish_index++;
                group_base += map.group_count;
            }
        }
        write(image, layout.task_map_offset + task * sizeof(map), map);
        if (map.finish_index == SCHEDULER_ASYNC_INDEX_NONE) continue;
        SchedulerAsyncFinishIdentity finish;
        finish.run_epoch = layout.run_epoch;
        finish.task_ref = static_cast<int64_t>(task);
        finish.group_base = map.group_base;
        finish.group_count = map.group_count;
        write(image, layout.finish_identity_offset + map.finish_index * sizeof(finish), finish);
        uint64_t initial_word = 0;
        if (!scheduler_async_join_initial_word(map.block_count, initial_word))
            return fail(error, "async block join cannot represent task shape");
        write(
            image,
            layout.finish_join_offset + map.finish_index * sizeof(SchedulerAsyncTaskJoin) +
                offsetof(SchedulerAsyncTaskJoin, word),
            initial_word
        );
        const uint64_t lease_word = SchedulerAsyncLease{}.word;
        write(
            image,
            layout.finish_lease_offset + map.finish_index * sizeof(SchedulerAsyncLease) +
                offsetof(SchedulerAsyncLease, word),
            lease_word
        );
        write(
            image, layout.finish_progress_offset + map.finish_index * sizeof(SchedulerAsyncFinishProgress),
            SchedulerAsyncFinishProgress{}
        );
        for (uint32_t block = 0; block < map.block_count; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                const uint64_t index = scheduler_async_group_index(map, block, lane);
                if (index == SCHEDULER_ASYNC_INDEX_NONE) continue;
                const auto cap = core_async_effective_capability(capabilities[task][lane]);
                SchedulerAsyncGroupIdentity identity;
                identity.run_epoch = layout.run_epoch;
                identity.task_ref = static_cast<int64_t>(task);
                identity.finish_index = map.finish_index;
                identity.entries_offset = entry_cursor;
                identity.block_idx = block;
                identity.subtask_slot = lane;
                identity.capacity = cap.max_conditions;
                identity.condition_types = cap.condition_types;
                write(image, layout.group_identity_offset + index * sizeof(identity), identity);
                write(
                    image,
                    layout.group_lease_offset + index * sizeof(SchedulerAsyncLease) +
                        offsetof(SchedulerAsyncLease, word),
                    lease_word
                );
                write(
                    image, layout.group_progress_offset + index * sizeof(SchedulerAsyncGroupProgress),
                    SchedulerAsyncGroupProgress{}
                );
                uint64_t stride = 0;
                scheduler_layout_checked_align(cap.max_conditions * sizeof(DeferredCompletionEntry), 128, &stride);
                entry_cursor += stride;
            }
        }
    }
    if (entry_cursor != layout.entries_offset + layout.entries_bytes || group_base != layout.group_count ||
        finish_index != layout.finish_count)
        return fail(error, "async image materialization does not match its size plan");
    return true;
}
}  // namespace

bool scheduler_plan_async_layout(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary, SchedulerAsyncLayout *out,
    std::string *error
) {
    return plan_async_layout(metadata, capabilities, run_epoch, byte_budget, auxiliary, out, error);
}

bool scheduler_append_async_layout(
    uint64_t prefix_bytes, const SchedulerAsyncLayout &layout, uint64_t byte_budget, uint64_t *async_offset,
    uint64_t *total_bytes, std::string *error
) {
    if (async_offset == nullptr || total_bytes == nullptr || layout.version != SCHEDULER_ASYNC_LAYOUT_VERSION ||
        (layout.group_count == 0) != (layout.total_bytes == 0))
        return fail(error, "invalid async layout placement");
    uint64_t cursor = prefix_bytes, offset = 0;
    if (layout.total_bytes != 0 &&
        (prefix_bytes == 0 || !scheduler_layout_reserve(&cursor, layout.total_bytes, 128, &offset)))
        return fail(error, "async scheduler backing size overflows");
    if (cursor > byte_budget || cursor > std::numeric_limits<size_t>::max())
        return fail(error, "async scheduler backing exceeds the total byte budget");
    *async_offset = offset;
    *total_bytes = cursor;
    return true;
}

bool scheduler_initialize_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const SchedulerAsyncLayout &layout, void *region, uint64_t region_bytes, std::string *error
) {
    SchedulerAsyncLayout expected{};
    const SchedulerAsyncAuxiliaryBytes auxiliary{
        layout.backend_bytes, layout.sdma_guard_count, layout.execution_contract
    };
    if (!plan_async_layout(metadata, capabilities, layout.run_epoch, layout.total_bytes, auxiliary, &expected, error))
        return false;
    if (std::memcmp(&expected, &layout, sizeof(layout)) != 0 || region_bytes < layout.total_bytes)
        return fail(error, "async initialization does not match its validated layout");
    if (layout.total_bytes == 0) return true;
    if (region == nullptr || layout.total_bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(region))
        return fail(error, "async initialization destination is invalid");
    return materialize_async_state(metadata, capabilities, layout, static_cast<std::byte *>(region), error);
}

bool scheduler_plan_async_state(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    uint64_t run_epoch, uint64_t byte_budget, const SchedulerAsyncAuxiliaryBytes &auxiliary,
    SchedulerAsyncHostPlan *out, std::string *error
) {
    if (out == nullptr) return fail(error, "async plan requires an output");
    try {
        SchedulerAsyncHostPlan next;
        if (!plan_async_layout(metadata, capabilities, run_epoch, byte_budget, auxiliary, &next.layout, error))
            return false;
        next.image.resize(static_cast<size_t>(next.layout.total_bytes));
        if (!scheduler_initialize_async_state(
                metadata, capabilities, next.layout, next.image.data(), next.image.size(), error
            ))
            return false;
        *out = std::move(next);
        return true;
    } catch (const std::bad_alloc &) {
        return fail(error, "cannot allocate the planned async state");
    } catch (const std::length_error &) {
        return fail(error, "planned async state exceeds container capacity");
    }
}
