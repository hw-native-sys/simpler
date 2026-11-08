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

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_async_check.h"

// This is retained Host provenance, not a device image or public kernel argument.
struct SchedulerProtocolAsyncPlan {
    uint64_t offset{0};
    SchedulerAsyncLayout layout{};
};

namespace scheduler_async_protocol_export {

inline void require(bool valid, const std::string &message) {
    if (!valid) throw std::invalid_argument("async protocol: " + message);
}

inline bool within(uint64_t offset, uint64_t count, uint64_t stride, uint64_t begin, uint64_t bytes) {
    return stride != 0 && offset >= begin && offset - begin <= bytes && count <= (bytes - (offset - begin)) / stride;
}

inline void stamp_shape(const SchedulerProtocolStamp &stamp, const std::string &path) {
    require(stamp.sequence != 0 || stamp.cycles == 0, path + " has cycles without a recorded sequence");
}

template <typename Object>
inline void
point(Object &parent, const char *name, const SchedulerProtocolStamp &stamp, uint64_t epoch, uint64_t worker) {
    stamp_shape(stamp, name);
    auto &out = parent.field(name);
    if (stamp.sequence == 0) {
        out << "null";
        return;
    }
    Object row(out);
    row.number("run_epoch", epoch);
    row.number("worker_index", worker);
    row.stamp("at", stamp);
}

template <typename Object, typename Release>
inline void interval(Object &parent, const Release &record, uint64_t epoch) {
    stamp_shape(record.before, "release.before");
    stamp_shape(record.after, "release.after");
    require(record.before.sequence != 0 || record.after.sequence == 0, "release.after exists without release.before");
    auto &out = parent.field("publication");
    if (record.before.sequence == 0) {
        out << "null";
        return;
    }
    Object row(out);
    row.number("run_epoch", epoch);
    row.number("worker_index", record.worker_index);
    row.stamp("before", record.before);
    if (record.after.sequence != 0) row.stamp("after", record.after);
    else row.field("after") << "null";
}

template <typename Image>
class View {
public:
    View(const Image &backing, const SchedulerProtocolAsyncPlan &frozen, uint64_t epoch, uint64_t tasks) :
        layout(frozen.layout),
        image(
            backing.template array<uint8_t>(checked_offset(frozen, epoch, tasks), layout.total_bytes),
            layout.total_bytes
        ) {
        const auto &wire = *image.template array<SchedulerAsyncLayout>(0, 1);
        require(std::memcmp(&wire, &layout, sizeof(layout)) == 0, "layout disagrees with retained Host plan");
        require(
            layout.group_trace_offset != 0 && layout.finish_trace_offset != 0, "protocol diagnostics were not allocated"
        );
        // The immutable layout comes from the validated Host plan. Image::array
        // checks each actual read; the wire copy must match that plan exactly.
    }

    template <typename T>
    const T &at(uint64_t offset, uint64_t index = 0) const {
        require(index < UINT64_MAX, "record index overflows");
        return image.template array<T>(offset, index + 1)[index];
    }

    template <typename Tasks>
    void validate_tasks(const Tasks &tasks) const {
        uint64_t groups = 0, finishes = 0, capacity = 0;
        uint64_t entries = layout.entries_offset;
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const auto &map = at<SchedulerAsyncTaskMap>(layout.task_map_offset, task);
            if (map.finish_index == SCHEDULER_ASYNC_INDEX_NONE) {
                require(
                    map.group_base == SCHEDULER_ASYNC_INDEX_NONE && map.block_count == 0 && map.groups_per_block == 0 &&
                        map.async_lane_mask == 0 && map.group_count == 0,
                    "inactive task map contains groups"
                );
                continue;
            }
            const auto &meta = tasks[task].metadata;
            require(
                scheduler_task_is_executable(meta.flags) && !scheduler_task_is_inline(meta.flags) &&
                    map.block_count == meta.logical_block_num && map.block_count != 0 && map.async_lane_mask != 0 &&
                    (map.async_lane_mask & ~meta.active_mask.raw()) == 0 &&
                    map.groups_per_block == static_cast<uint32_t>(__builtin_popcount(map.async_lane_mask)) &&
                    static_cast<uint64_t>(map.block_count) * map.groups_per_block == map.group_count &&
                    map.group_base == groups && map.group_base <= layout.group_count &&
                    map.group_count <= layout.group_count - map.group_base && map.finish_index == finishes &&
                    map.finish_index < layout.finish_count,
                "task map disagrees with frozen task shape or dense layout"
            );
            const auto &finish = at<SchedulerAsyncFinishIdentity>(layout.finish_identity_offset, finishes);
            require(
                finish.run_epoch == layout.run_epoch && finish.task_ref == static_cast<int64_t>(task) &&
                    finish.group_base == groups && finish.group_count == map.group_count,
                "FINISH identity disagrees with its task map"
            );
            for (uint32_t block = 0; block < map.block_count; ++block) {
                for (uint32_t slot = 0; slot < 3; ++slot) {
                    if ((map.async_lane_mask & (1u << slot)) == 0) continue;
                    const uint64_t index = scheduler_async_group_index(map, block, slot);
                    const auto &identity = at<SchedulerAsyncGroupIdentity>(layout.group_identity_offset, index);
                    if (index != map.group_base + static_cast<uint64_t>(block) * map.groups_per_block) {
                        const auto &release = at<SchedulerAsyncGroupTrace>(layout.group_trace_offset, index).release;
                        require(
                            release.before.cycles == 0 && release.before.sequence == 0 && release.after.cycles == 0 &&
                                release.after.sequence == 0,
                            "release evidence appears on a non-owning group"
                        );
                    }
                    require(
                        identity.run_epoch == layout.run_epoch && identity.task_ref == static_cast<int64_t>(task) &&
                            identity.finish_index == finishes && identity.block_idx == block &&
                            identity.subtask_slot == slot && identity.capacity > 0 && identity.capacity <= 64 &&
                            identity.condition_types != 0 &&
                            (identity.condition_types & ~CORE_ASYNC_CONDITION_TYPES) == 0,
                        "group identity disagrees with its task map"
                    );
                    require(
                        identity.entries_offset == entries &&
                            within(
                                entries, identity.capacity, sizeof(DeferredCompletionEntry), layout.entries_offset,
                                layout.entries_bytes
                            ),
                        "group conditions leave their planned sections or overlap another group"
                    );
                    entries += aligned_bytes(identity.capacity, sizeof(DeferredCompletionEntry));
                    require(capacity <= UINT64_MAX - identity.capacity, "condition capacity overflows");
                    capacity += identity.capacity;
                }
            }
            groups += map.group_count;
            ++finishes;
        }
        require(
            groups == layout.group_count && finishes == layout.finish_count && capacity == layout.condition_capacity &&
                entries - layout.entries_offset == layout.entries_bytes,
            "task maps do not cover the immutable async allocation"
        );
    }

    const SchedulerAsyncLayout &layout;
    Image image;

private:
    static uint64_t checked_offset(const SchedulerProtocolAsyncPlan &frozen, uint64_t epoch, uint64_t tasks) {
        require(
            frozen.offset != 0 && frozen.offset % 128 == 0 && frozen.layout.version == SCHEDULER_ASYNC_LAYOUT_VERSION &&
                frozen.layout.run_epoch == epoch && frozen.layout.task_count == tasks &&
                frozen.layout.group_count != 0 && frozen.layout.finish_count != 0 &&
                frozen.layout.total_bytes >= sizeof(SchedulerAsyncLayout) && frozen.layout.total_bytes % 128 == 0,
            "invalid retained Host async plan"
        );
        return frozen.offset;
    }
    static uint64_t aligned_bytes(uint32_t capacity, uint64_t stride) {
        return (static_cast<uint64_t>(capacity) * stride + 127) & ~UINT64_C(127);
    }
};

inline const char *
group_state(const SchedulerAsyncGroupWriter &writer, const SchedulerAsyncGroupProgress &progress, uint64_t lease_word) {
    require(writer.reservation.sealed <= 1 && writer.reservation.reserved == 0, "invalid group seal flags");
    require(progress.phase <= static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED), "invalid CHECK progress phase");
    if (writer.error_code != 0 || progress.phase == static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED) ||
        scheduler_async_lease_phase(lease_word) == SchedulerAsyncPhase::FAILED)
        return "FAILED";
    if (writer.reservation.sealed == 0) return "UNSEALED";
    if (writer.count == 0 && lease_word == scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT)) return "EMPTY";
    if (writer.count != 0 && progress.phase == static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED) &&
        scheduler_async_lease_phase(lease_word) == SchedulerAsyncPhase::TERMINAL)
        return "READY";
    return "PENDING";
}

inline const char *finish_state(const SchedulerAsyncFinishProgress &progress, uint64_t lease_word) {
    require(
        progress.completion.publication.phase <= static_cast<uint64_t>(SchedulerCompletionPhase::FAILED),
        "invalid FINISH progress phase"
    );
    if (progress.completion.publication.phase == static_cast<uint64_t>(SchedulerCompletionPhase::FAILED) ||
        scheduler_async_lease_phase(lease_word) == SchedulerAsyncPhase::FAILED)
        return "FAILED";
    return progress.contributed ? "DRAINED" : "PENDING";
}

template <typename Metadata>
inline uint64_t execution_index(const SchedulerAsyncGroupIdentity &identity, const Metadata &task, uint64_t count) {
    const auto slot = static_cast<SubtaskSlot>(identity.subtask_slot);
    require(
        identity.subtask_slot < 3 && task.active_mask.has(slot) && identity.block_idx < task.logical_block_num,
        "invalid group execution coordinates"
    );
    const uint64_t index = static_cast<uint64_t>(task.trace_index_base) +
                           static_cast<uint64_t>(identity.block_idx) * task.active_mask.count() +
                           task.active_mask.rank_of(slot);
    require(index < count, "group execution index leaves the protocol table");
    return index;
}

inline void execution_identity(
    const SchedulerAsyncGroupIdentity &identity, const SchedulerProtocolExecutionTrace &execution,
    const SchedulerProtocolStamp &sealed, uint64_t workers
) {
    if (execution.publication_count != 0 || execution.permit.sequence != 0 || execution.entry.sequence != 0 ||
        sealed.sequence != 0) {
        require(
            execution.run_epoch == identity.run_epoch && execution.task_ref == identity.task_ref &&
                execution.block_idx == identity.block_idx && execution.subtask_slot == identity.subtask_slot &&
                execution.generation != 0 && execution.worker_index < workers,
            "group disagrees with its recorded execution identity"
        );
    }
    require(sealed.sequence == 0 || execution.entry.sequence != 0, "recorded group seal has no actual execution entry");
}

template <typename Object, typename Image, typename Tasks>
inline void groups(
    Object &root, const View<Image> &view, const Tasks &tasks, const SchedulerProtocolExecutionTrace *executions,
    uint64_t trace_count, uint64_t workers
) {
    auto &out = root.field("groups");
    out << '[';
    for (uint64_t index = 0; index < view.layout.group_count; ++index) {
        if (index != 0) out << ',';
        const auto &identity = view.template at<SchedulerAsyncGroupIdentity>(view.layout.group_identity_offset, index);
        const auto &writer = view.template at<SchedulerAsyncGroupWriter>(view.layout.group_writer_offset, index);
        const auto &progress = view.template at<SchedulerAsyncGroupProgress>(view.layout.group_progress_offset, index);
        const auto &trace_record = view.template at<SchedulerAsyncGroupTrace>(view.layout.group_trace_offset, index);
        const uint64_t lease = view.template at<SchedulerAsyncLease>(view.layout.group_lease_offset, index).word;
        const uint64_t trace = execution_index(identity, tasks[identity.task_ref].metadata, trace_count);
        const auto &execution = executions[trace];
        execution_identity(identity, execution, trace_record.sealed_at, workers);
        require(
            writer.count <= identity.capacity, "groups[" + std::to_string(index) + "].registered_count exceeds capacity"
        );
        Object row(out);
        row.number("index", index);
        row.number("run_epoch", identity.run_epoch);
        row.number("task_ref", identity.task_ref);
        row.number("block_idx", identity.block_idx);
        row.number("subtask_slot", identity.subtask_slot);
        row.number("execution_index", trace);
        row.number("dispatch_generation", execution.generation);
        row.number("lease_word", lease);
        row.number("capacity", identity.capacity);
        row.number("registered_count", writer.count);
        row.field("conditions") << '[';
        for (uint32_t condition = 0; condition < writer.count; ++condition) {
            if (condition != 0) out << ',';
            const auto &entry = view.template at<DeferredCompletionEntry>(identity.entries_offset, condition);
            Object item(out);
            item.number("index", condition);
            item.number("type", entry.completion_type);
        }
        out << ']';
        row.string("state", group_state(writer, progress, lease));
        point(row, "sealed", trace_record.sealed_at, identity.run_epoch, execution.worker_index);
        point(row, "terminal", trace_record.terminal_at, identity.run_epoch, trace_record.terminal_worker_index);
        row.number("observations", trace_record.check.observation_count);
        row.number("pending_count", trace_record.check.pending_slice_count);
        row.number("slice_count", trace_record.check.slice_count);
        row.number("active_cycles", trace_record.check.active_cycles);
        row.number(
            "failure_code", writer.error_code != 0 ? static_cast<int64_t>(writer.error_code) : progress.failure_code
        );
    }
    out << ']';
}

template <typename Object, typename Image>
inline void releases(Object &root, const View<Image> &view) {
    auto &out = root.field("releases");
    out << '[';
    bool first = true;
    for (uint64_t task = 0; task < view.layout.task_count; ++task) {
        const auto &map = view.template at<SchedulerAsyncTaskMap>(view.layout.task_map_offset, task);
        if (map.finish_index == SCHEDULER_ASYNC_INDEX_NONE) continue;
        for (uint32_t block = 0; block < map.block_count; ++block) {
            if (!first) out << ',';
            first = false;
            const uint64_t index = map.group_base + static_cast<uint64_t>(block) * map.groups_per_block;
            const auto &trace_record =
                view.template at<SchedulerAsyncGroupTrace>(view.layout.group_trace_offset, index);
            Object row(out);
            row.number("task_ref", task);
            row.number("block_idx", block);
            row.number("run_epoch", view.layout.run_epoch);
            interval(row, trace_record.release, view.layout.run_epoch);
        }
    }
    out << ']';
}

template <typename Object, typename Image>
inline void finishes(Object &root, const View<Image> &view) {
    auto &out = root.field("finishes");
    out << '[';
    for (uint64_t index = 0; index < view.layout.finish_count; ++index) {
        if (index != 0) out << ',';
        const auto &identity =
            view.template at<SchedulerAsyncFinishIdentity>(view.layout.finish_identity_offset, index);
        const auto &progress =
            view.template at<SchedulerAsyncFinishProgress>(view.layout.finish_progress_offset, index);
        const auto &trace_record = view.template at<SchedulerAsyncFinishTrace>(view.layout.finish_trace_offset, index);
        const uint64_t lease = view.template at<SchedulerAsyncLease>(view.layout.finish_lease_offset, index).word;
        require(
            progress.completion.publication.pending_count <= 8,
            "finishes[" + std::to_string(index) +
                "].pending_count out of range: " + std::to_string(progress.completion.publication.pending_count)
        );
        Object row(out);
        row.number("index", index);
        row.number("run_epoch", identity.run_epoch);
        row.number("task_ref", identity.task_ref);
        row.number("lease_word", lease);
        row.string("state", finish_state(progress, lease));
        row.field("group_indices") << '[';
        for (uint64_t group = 0; group < identity.group_count; ++group) {
            if (group != 0) out << ',';
            out << identity.group_base + group;
        }
        out << ']';
        point(row, "drained", trace_record.drained_at, identity.run_epoch, trace_record.drained_worker_index);
        row.number("waiter_count", progress.completion.visited_count);
        row.number("pending_count", progress.completion.publication.pending_count);
        row.number("slice_count", trace_record.finish.slice_count);
        row.number("active_cycles", trace_record.finish.active_slice_cycles);
        row.number("failure_code", progress.failure_code);
    }
    out << ']';
}

template <typename Object, typename Image, typename Tasks>
inline void emit(
    Object &parent, const Image &image, const SchedulerProtocolAsyncPlan &frozen, uint64_t epoch, const Tasks &tasks,
    const SchedulerProtocolExecutionTrace *executions, uint64_t trace_count, uint64_t workers
) {
    const View<Image> view(image, frozen, epoch, tasks.size());
    view.validate_tasks(tasks);
    Object root(parent.field("async"));
    root.number("group_count", view.layout.group_count);
    root.number("finish_count", view.layout.finish_count);
    groups(root, view, tasks, executions, trace_count, workers);
    releases(root, view);
    finishes(root, view);
}

}  // namespace scheduler_async_protocol_export
