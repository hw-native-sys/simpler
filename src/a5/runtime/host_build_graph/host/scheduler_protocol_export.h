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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

struct SchedulerProtocolTaskSpec {
    SchedulerTaskMetadata metadata;
    std::vector<int32_t> dependencies;
};

inline std::vector<SchedulerProtocolTaskSpec>
scheduler_protocol_freeze_graph(const SchedulerGraphView &graph, const std::vector<SchedulerTaskMetadata> &metadata) {
    if (graph.task_count != metadata.size()) throw std::invalid_argument("protocol graph task count mismatch");
    std::vector<SchedulerProtocolTaskSpec> result;
    result.reserve(metadata.size());
    for (size_t task = 0; task < metadata.size(); ++task) {
        SchedulerTaskShape shape{};
        const auto status = scheduler_classify_task_shape(graph, static_cast<int64_t>(task), &shape);
        if (status != SchedulerGraphResult::OK &&
            !(status == SchedulerGraphResult::UNSUPPORTED_SHAPE &&
              (metadata[task].active_mask.is_empty() || scheduler_task_is_inline(metadata[task].flags))))
            throw std::invalid_argument("invalid frozen protocol graph");
        const auto *payload = scheduler_graph_payload(graph, static_cast<int64_t>(task));
        const auto count = *reinterpret_cast<const int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
        SchedulerProtocolTaskSpec spec{metadata[task], {}};
        spec.dependencies.reserve(static_cast<size_t>(count));
        for (int32_t index = 0; index < count; ++index)
            spec.dependencies.push_back(scheduler_graph_fanin_id(graph, static_cast<int64_t>(task), index));
        result.push_back(std::move(spec));
    }
    return result;
}

namespace scheduler_protocol_export {

// Field names and string values are schema literals. Device values are numeric only.
class JsonObject {
public:
    explicit JsonObject(std::ostream &stream) :
        stream_(stream) {
        stream_ << '{';
    }
    ~JsonObject() { stream_ << '}'; }
    JsonObject(const JsonObject &) = delete;
    JsonObject &operator=(const JsonObject &) = delete;

    std::ostream &field(const char *name) {
        if (!first_) stream_ << ',';
        first_ = false;
        return stream_ << '"' << name << "\":";
    }
    template <class T>
    void number(const char *name, T value) {
        field(name) << +value;
    }
    // The audit schema carries a lane set as the byte LaneMask serializes to.
    void lane_mask(const char *name, LaneMask value) { field(name) << +value.raw(); }
    void string(const char *name, const char *value) { field(name) << '"' << value << '"'; }
    void stamp(const char *name, const SchedulerProtocolStamp &value) {
        field(name) << '[' << value.cycles << ',' << value.sequence << ']';
    }
    template <class T>
    void numbers(const char *name, const T *values, size_t count) {
        field(name) << '[';
        for (size_t index = 0; index < count; ++index) {
            if (index != 0) stream_ << ',';
            stream_ << +values[index];
        }
        stream_ << ']';
    }

private:
    std::ostream &stream_;
    bool first_{true};
};

class Image {
public:
    Image(const void *base, size_t size) :
        base_(static_cast<const uint8_t *>(base)),
        size_(size) {
        if (base == nullptr || reinterpret_cast<uintptr_t>(base) % SCHEDULER_STATE_ALIGNMENT != 0)
            throw std::invalid_argument("invalid protocol image base");
    }
    template <class T>
    const T *array(uint64_t offset, uint64_t count) const {
        if (count == 0) return nullptr;
        if (offset % alignof(T) != 0 || offset > size_ || count > (size_ - offset) / sizeof(T))
            throw std::invalid_argument("protocol image region out of bounds");
        return reinterpret_cast<const T *>(base_ + offset);
    }

private:
    const uint8_t *base_;
    size_t size_;
};

template <class T>
inline bool populated(const T &record) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&record);
    return std::any_of(bytes, bytes + sizeof(T), [](uint8_t value) {
        return value != 0;
    });
}

template <class T, class Emit>
inline void sparse(JsonObject &root, const char *name, const T *records, uint64_t count, Emit emit) {
    auto &out = root.field(name);
    out << '[';
    bool first = true;
    for (uint64_t index = 0; index < count; ++index) {
        if (!populated(records[index])) continue;
        if (!first) out << ',';
        first = false;
        JsonObject row(out);
        row.number("index", index);
        emit(row, records[index]);
    }
    out << ']';
}

inline void actor(JsonObject &parent, const char *name, const SchedulerProtocolTaskActor &record) {
    auto &out = parent.field(name);
    if (!populated(record)) {
        out << "null";
        return;
    }
    JsonObject row(out);
    row.stamp("before", record.before);
    row.stamp("after", record.after);
    row.number("run_epoch", record.run_epoch);
    row.number("task_ref", record.task_ref);
    row.number("order_key", record.order_key);
    row.number("worker_index", record.worker_index);
    row.number("flags", record.flags);
}

inline void block_owner(JsonObject &parent, const SchedulerProtocolBlockOwner &record) {
    auto &out = parent.field("owner");
    if (!populated(record)) {
        out << "null";
        return;
    }
    JsonObject row(out);
    row.stamp("claim_before", record.claim_before);
    row.stamp("claim_after", record.claim_after);
    row.stamp("publish_before", record.publish_before);
    row.stamp("publish_after", record.publish_after);
    row.stamp("completion_before", record.completion_before);
    row.stamp("completion_after", record.completion_after);
    row.stamp("releases_observed", record.releases_observed);
    row.number("run_epoch", record.run_epoch);
    row.number("task_ref", record.task_ref);
    row.number("order_key", record.order_key);
    row.number("group_seq", record.group_seq);
    row.number("completion_ticket", record.completion_ticket);
    row.number("worker_index", record.worker_index);
    row.number("block_idx", record.block_idx);
    row.number("cluster_index", record.cluster_index);
    row.number("block_num", record.logical_block_num);
    row.lane_mask("active_mask", record.active_mask);
    row.number("flags", record.flags);
    row.numbers("released_seq", record.released_seq, 3);
}

inline void lane(JsonObject &row, const SchedulerProtocolLaneTrace &record) {
    row.stamp("descriptor_ready", record.descriptor_ready);
    row.stamp("arrive_before", record.arrive_before);
    row.stamp("arrive_after", record.arrive_after);
    row.stamp("arrivals_observed", record.arrivals_observed);
    row.stamp("release_before", record.release_before);
    row.stamp("release_after", record.release_after);
    row.number("run_epoch", record.run_epoch);
    row.number("task_ref", record.task_ref);
    row.number("order_key", record.order_key);
    row.number("group_seq", record.group_seq);
    row.number("worker_index", record.worker_index);
    row.number("block_idx", record.block_idx);
    row.number("cluster_index", record.cluster_index);
    row.number("lane", record.lane);
    row.lane_mask("active_mask", record.active_mask);
    row.number("flags", record.flags);
    row.number("block_num", record.logical_block_num);
    row.lane_mask("resource_mask", record.resource_mask);
    row.numbers("arrived_seq", record.arrived_seq, 3);
    row.numbers("lane_subtasks", record.lane_subtasks, 3);
}

inline void execution(JsonObject &row, const SchedulerProtocolExecutionTrace &record) {
    row.stamp("entry", record.entry);
    row.stamp("returned", record.returned);
    row.stamp("visible", record.visible);
    row.stamp("completion_consumed", record.completion_consumed);
    row.stamp("permit", record.permit);
    row.stamp("trace_published", record.trace_published);
    row.number("run_epoch", record.run_epoch);
    row.number("task_ref", record.task_ref);
    row.number("order_key", record.order_key);
    row.number("permit_tail", record.permit_tail);
    row.number("worker_index", record.worker_index);
    row.number("block_idx", record.block_idx);
    row.number("generation", record.generation);
    row.number("pending_slot", record.pending_slot);
    row.number("subtask_slot", record.subtask_slot);
    row.number("flags", record.flags);
    row.number("ready_source", record.ready_source);
    row.number("group_seq", record.group_seq);
    row.number("block_num", record.logical_block_num);
    row.number("cluster_index", record.cluster_index);
    row.number("cluster_lane", record.cluster_lane);
    row.number("publication_mode", record.publication_mode);
    row.number("publication_before_cycles", record.publication_before_cycles);
    row.number("publication_after_cycles", record.publication_after_cycles);
    row.number("publication_count", record.publication_count);
}

inline void validate_layout(const Image &image, const AicoreSchedulerLayout &layout, uint64_t epoch) {
    const auto &wire = *image.array<SchedulerProtocolLayout>(layout.protocol_layout_offset, 1);
    const auto &plan = layout.protocol;
    if (wire.magic != SCHEDULER_PROTOCOL_MAGIC || wire.run_epoch != epoch ||
        wire.storage_end_offset != plan.storage_end_offset || wire.storage_end_offset > layout.total_size ||
        wire.task_count != plan.task_count || wire.worker_count != plan.worker_count ||
        wire.trace_count != plan.trace_count || wire.cooperative_task_count != plan.cooperative_task_count ||
        wire.member_count != plan.member_count || wire.tasks_offset != plan.tasks_offset ||
        wire.members_offset != plan.members_offset || wire.blocks_offset != plan.blocks_offset ||
        wire.executions_offset != plan.executions_offset || wire.workers_offset != plan.workers_offset ||
        std::any_of(std::begin(wire.reserved), std::end(wire.reserved), [](uint64_t value) {
            return value != 0;
        }))
        throw std::invalid_argument("protocol header disagrees with Host plan or run epoch");
    const auto &summary = *image.array<SchedulerGraphSummary>(layout.graph_summary_offset, 1);
    if (summary.run_epoch != epoch || summary.protocol_layout_offset != layout.protocol_layout_offset ||
        summary.worker_count != layout.spmd.worker_count || summary.cluster_count != layout.spmd.cluster_count ||
        summary.cooperative_task_count != layout.spmd.cooperative_task_count ||
        summary.mix_task_count != layout.spmd.mix_task_count ||
        summary.admission_member_word_count != layout.admission_member_word_count ||
        summary.admission_records_offset != layout.admission_records_offset ||
        summary.admission_record_stride != layout.admission_record_stride ||
        summary.admission_log_offset != layout.admission_log_offset ||
        summary.admission_control_offset != layout.admission_control_offset ||
        summary.group_mailboxes_offset != layout.group_mailboxes_offset ||
        summary.mix_close_seals_offset != layout.mix_close_seals_offset)
        throw std::invalid_argument("protocol graph summary disagrees with Host plan");
}

inline void admissions(JsonObject &root, const Image &image, const AicoreSchedulerLayout &layout) {
    const auto count = layout.spmd.cooperative_task_count;
    const auto *control = image.array<SchedulerAdmissionControl>(layout.admission_control_offset, count != 0);
    root.number("commit_tail", control == nullptr ? 0 : control->commit_tail);
    root.number("commit_lock", control == nullptr ? 0 : control->commit_lock);
    root.numbers("admission_log", image.array<int64_t>(layout.admission_log_offset, count), count);
    auto &out = root.field("admissions");
    out << '[';
    for (uint64_t task = 0; count != 0 && task < layout.task_count; ++task) {
        const uint64_t offset = layout.admission_records_offset + task * layout.admission_record_stride;
        const auto &record = *image.array<SchedulerAdmissionRecord>(offset, 1);
        if (record.member_words_offset != offset + sizeof(SchedulerAdmissionRecord))
            throw std::invalid_argument("protocol member bitmap outside its task record");
        const uint64_t *seals = nullptr;
        if (record.mix_close_seals_offset != 0) {
            const uint64_t first = layout.mix_close_seals_offset;
            const uint64_t stride = layout.spmd.cluster_count * sizeof(uint64_t);
            if (stride == 0 || record.mix_close_seals_offset < first ||
                (record.mix_close_seals_offset - first) % stride != 0 ||
                (record.mix_close_seals_offset - first) / stride >= layout.spmd.mix_task_count)
                throw std::invalid_argument("protocol close seal outside Host planned region");
            seals = image.array<uint64_t>(record.mix_close_seals_offset, layout.spmd.cluster_count);
        }
        if (task != 0) out << ',';
        JsonObject row(out);
        row.number("index", task);
        row.number("run_epoch", record.run_epoch);
        row.number("task_ref", record.task_ref);
        row.number("order_key", record.order_key);
        row.number("block_num", record.logical_block_num);
        row.number("member_count", record.member_count);
        row.lane_mask("active_mask", record.active_mask);
        row.number("flags", record.flags);
        row.number("member_words_offset", record.member_words_offset);
        row.number("close_seals_offset", record.mix_close_seals_offset);
        row.numbers(
            "member_words", image.array<uint64_t>(record.member_words_offset, layout.admission_member_word_count),
            layout.admission_member_word_count
        );
        row.numbers("close_seals", seals, seals == nullptr ? 0 : layout.spmd.cluster_count);
    }
    out << ']';
}

}  // namespace scheduler_protocol_export

inline std::string scheduler_export_protocol(
    const void *base, size_t size, const AicoreSchedulerLayout &layout, uint64_t epoch,
    const std::vector<SchedulerProtocolTaskSpec> &tasks
) {
    using namespace scheduler_protocol_export;
    if (size != layout.total_size || tasks.size() != layout.task_count || layout.protocol_layout_offset == 0)
        throw std::invalid_argument("protocol image size or frozen task count mismatch");
    const Image image(base, size);
    validate_layout(image, layout, epoch);
    const auto &plan = layout.protocol;
    std::ostringstream out;
    {
        JsonObject root(out);
        // v5 records transient claim ownership separately from the published phase.
        root.string("schema", "hbg-native-protocol-v5");
        root.string("producer", "aicore");
        root.number("run_epoch", epoch);
        root.number("task_count", plan.task_count);
        root.number("worker_count", plan.worker_count);
        root.number("trace_count", plan.trace_count);
        root.number("member_count", plan.member_count);
        root.number("cooperative_task_count", plan.cooperative_task_count);
        root.number("cluster_count", layout.spmd.cluster_count);
        root.number("mix_task_count", layout.spmd.mix_task_count);
        const auto &run = *image.array<SchedulerRunControl>(layout.run_control_offset, 1);
        root.number("expected_task_count", run.expected_task_count);
        root.number("inline_completed_count", run.inline_completed_count);
        root.number("resolved_task_count", run.resolved_task_count);
        root.number("executed_task_count", run.executed_task_count);
        root.number("active_worker_count", run.active_worker_count);
        root.number("scheduler_count", run.scheduler_count);
        root.number("scheduler_error", run.scheduler_error);
        root.number("error_claimed", run.error_claimed);
        {
            JsonObject error(root.field("first_error"));
            error.number("task_ref", run.error_task_ref);
            error.number("core_id", run.error_core_id);
            error.number("core_type", run.error_core_type);
            error.number("site", run.error_site);
            error.number("run_epoch", run.error_run_epoch);
            error.number("order_key", run.error_order_key);
            error.number("block_idx", run.error_block_idx);
            error.number("group_seq", run.error_group_seq);
            error.number("worker_index", run.error_worker_index);
            error.number("cluster_lane", run.error_cluster_lane);
            error.number("identity_fields", run.error_identity_fields);
        }
        root.field("tasks") << '[';
        for (size_t index = 0; index < tasks.size(); ++index) {
            if (index != 0) out << ',';
            JsonObject row(out);
            const auto &task = tasks[index];
            row.number("task_ref", index);
            row.number("flags", task.metadata.flags);
            row.lane_mask("active_mask", task.metadata.active_mask);
            row.number("block_num", task.metadata.logical_block_num);
            row.number("trace_base", task.metadata.trace_index_base);
            row.numbers("kernel_ids", task.metadata.kernel_ids, 3);
            row.numbers("dependencies", task.dependencies.data(), task.dependencies.size());
        }
        out << ']';
        const auto *contexts = image.array<SchedulerWorkerContext>(layout.worker_contexts_offset, plan.worker_count);
        const auto *workers = image.array<SchedulerProtocolWorkerTrace>(plan.workers_offset, plan.worker_count);
        root.field("workers") << '[';
        for (uint64_t index = 0; index < plan.worker_count; ++index) {
            if (index != 0) out << ',';
            JsonObject row(out);
            const auto &context = contexts[index];
            const auto &worker = workers[index];
            row.number("index", index);
            row.number("worker_index", worker.worker_index);
            row.number("context_worker_index", context.worker_index);
            row.number("run_epoch", worker.run_epoch);
            row.number("next_sequence", worker.next_sequence);
            row.number("error_flags", worker.error_flags);
            row.number("finished", worker.finished);
            row.number("core_id", context.physical_core_id);
            row.number("core_type", context.core_type);
            row.number("cluster_index", context.cluster_index);
            row.numbers("cluster_workers", context.cluster_worker_ids, 3);
        }
        out << ']';
        admissions(root, image, layout);
        const auto *controls = image.array<SchedulerTaskControl>(layout.task_controls_offset, plan.task_count);
        root.field("task_controls") << '[';
        for (uint64_t task = 0; task < plan.task_count; ++task) {
            if (task != 0) out << ',';
            JsonObject row(out);
            row.number("task_ref", task);
            row.number("state", controls[task].state);
            row.number("next_block", controls[task].next_block);
            row.number("claim_tickets", controls[task].claim_tickets);
            row.number("completed_blocks", controls[task].completed_blocks);
            row.number("admission_state", controls[task].admission_state);
            row.number("admission_claim_lock", controls[task].admission_claim_lock);
        }
        out << ']';
        sparse(
            root, "task_events", image.array<SchedulerProtocolTaskTrace>(plan.tasks_offset, plan.task_count),
            plan.task_count, [](JsonObject &row, const auto &record) {
                actor(row, "admission", record.admission);
                actor(row, "completion", record.completion);
            }
        );
        sparse(
            root, "members", image.array<SchedulerProtocolMemberTrace>(plan.members_offset, plan.member_count),
            plan.member_count, [](JsonObject &row, const auto &record) {
                row.stamp("observed", record.observed);
                row.stamp("closed", record.closed);
                row.number("run_epoch", record.run_epoch);
                row.number("task_ref", record.task_ref);
                row.number("order_key", record.order_key);
                row.number("observed_tail", record.observed_tail);
                row.number("closed_cursor", record.closed_cursor);
                row.number("close_seal", record.close_seal);
                row.number("last_group_seq", record.last_group_seq);
                row.number("worker_index", record.worker_index);
                row.number("flags", record.flags);
                row.number("pending_mask", record.pending_mask);
            }
        );
        sparse(
            root, "blocks", image.array<SchedulerProtocolBlockTrace>(plan.blocks_offset, plan.trace_count),
            plan.trace_count, [](JsonObject &row, const auto &record) {
                block_owner(row, record.owner);
                sparse(row, "lanes", record.lanes, 3, lane);
            }
        );
        sparse(
            root, "executions", image.array<SchedulerProtocolExecutionTrace>(plan.executions_offset, plan.trace_count),
            plan.trace_count, execution
        );
    }
    return out.str();
}
