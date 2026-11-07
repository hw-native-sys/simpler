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
#include <cinttypes>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "aicore_scheduler_state.h"
#include "aicpu/cache_maintenance.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_ready_pool.h"

// AICPU cold-path diagnostics only. AICores may still be writing: these are
// bounded live samples of last published checkpoints, never a coherent snapshot
// or proof of completion. No RMW, cache writeback, device query, or payload read.
namespace scheduler_timeout_diagnostics {
struct View {
    const uint8_t *base{nullptr};
    uint64_t bytes{0};

    bool valid() const { return base != nullptr && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(base); }
    bool region(uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment, uint64_t minimum = 0) const {
        return valid() && alignment != 0 && offset >= minimum && offset % alignment == 0 && offset <= bytes &&
               stride != 0 && count <= (bytes - offset) / stride;
    }
    template <typename T>
    bool read(uint64_t offset, T *out, uint64_t index = 0) const {
        static_assert(std::is_trivially_copyable_v<T>);
        if (out == nullptr || index > (UINT64_MAX - offset) / sizeof(T)) return false;
        offset += index * sizeof(T);
        if (!region(offset, 1, sizeof(T), alignof(T))) return false;
        const void *source = base + offset;
        // A5 is coherent; the common helper is a no-op there. The read barrier
        // remains explicit, and a field can still change during this copy.
        cache_invalidate_range(source, sizeof(T));
        rmb();
        std::memcpy(out, source, sizeof(T));
        rmb();
        return true;
    }
};

inline void tasks(const View &view, const SchedulerWorkerContext &config) {
    const uint64_t count = config.graph_task_count;
    if (!view.region(config.task_controls_offset, count, sizeof(SchedulerTaskControl), alignof(SchedulerTaskControl)) ||
        !view.region(
            config.task_metadata_offset, count, sizeof(SchedulerTaskMetadata), alignof(SchedulerTaskMetadata)
        )) {
        LOG_ERROR("A5 HBG live sample: invalid task-table extent");
        return;
    }
    uint64_t printed = 0;
    const uint64_t scanned = std::min<uint64_t>(count, 64);
    for (uint64_t index = 0; index < scanned && printed < 8; ++index) {
        SchedulerTaskControl control{};
        SchedulerTaskMetadata meta{};
        if (!view.read(config.task_controls_offset, &control, index) ||
            !view.read(config.task_metadata_offset, &meta, index))
            return;
        if (control.state == static_cast<int64_t>(SchedulerTaskState::DONE)) continue;
        ++printed;
        LOG_ERROR(
            "A5 HBG live sample task=%" PRIu64 " state=%" PRId64 " claims=%" PRIu64 " next_block=%" PRIu64
            " completed_blocks=%" PRIu64 " wake=%" PRId64 " link=%" PRId64 " fanin=%d advisory_producer=%d"
            " trace_base=%u blocks=%u lanes=0x%x kernels=%u,%u,%u",
            index, control.state, control.claim_tickets, control.next_block, control.completed_blocks,
            control.wake_list_head, control.next_waiter, control.next_fanin_index, control.waiting_producer,
            meta.trace_index_base, static_cast<unsigned>(meta.logical_block_num),
            static_cast<unsigned>(meta.active_mask.raw()), static_cast<unsigned>(meta.kernel_ids[0]),
            static_cast<unsigned>(meta.kernel_ids[1]), static_cast<unsigned>(meta.kernel_ids[2])
        );
    }
    LOG_ERROR(
        "A5 HBG live sample tasks: scanned=%" PRIu64 "/%" PRIu64 " reported=%" PRIu64 " limit=8", scanned, count,
        printed
    );
}

inline void async_checkpoints(const View &backing, const SchedulerGraphSummary &summary, uint64_t task_count) {
    if (summary.async_layout_offset == 0) {
        LOG_ERROR("A5 HBG live sample: no async locator");
        return;
    }
    SchedulerAsyncLayout layout{};
    if (summary.async_layout_offset % 128 != 0 || !backing.read(summary.async_layout_offset, &layout) ||
        layout.version != SCHEDULER_ASYNC_LAYOUT_VERSION || layout.run_epoch != summary.run_epoch ||
        layout.task_count != task_count || layout.total_bytes < sizeof(layout) ||
        !backing.region(summary.async_layout_offset, layout.total_bytes, 1, 128)) {
        LOG_ERROR("A5 HBG live sample: invalid async header/epoch/extent");
        return;
    }
    const View view{backing.base + summary.async_layout_offset, layout.total_bytes};
    const auto rows = [&](uint64_t offset, uint64_t count, uint64_t stride) {
        return view.region(offset, count, stride, 128, sizeof(SchedulerAsyncLayout));
    };
    if (!rows(layout.group_identity_offset, layout.group_count, sizeof(SchedulerAsyncGroupIdentity)) ||
        !rows(layout.group_writer_offset, layout.group_count, sizeof(SchedulerAsyncGroupWriter)) ||
        !rows(layout.group_progress_offset, layout.group_count, sizeof(SchedulerAsyncGroupProgress)) ||
        !rows(layout.group_lease_offset, layout.group_count, sizeof(SchedulerAsyncLease)) ||
        !rows(layout.finish_identity_offset, layout.finish_count, sizeof(SchedulerAsyncFinishIdentity)) ||
        !rows(layout.finish_progress_offset, layout.finish_count, sizeof(SchedulerAsyncFinishProgress)) ||
        !rows(layout.finish_lease_offset, layout.finish_count, sizeof(SchedulerAsyncLease)) ||
        !rows(layout.finish_join_offset, layout.finish_count, sizeof(SchedulerAsyncTaskJoin)) ||
        layout.check_directory_words != layout.group_count / 64 + (layout.group_count % 64 != 0) ||
        layout.finish_directory_words != layout.finish_count / 64 + (layout.finish_count % 64 != 0) ||
        !rows(layout.check_directory_offset, layout.check_directory_words, sizeof(SchedulerAsyncDirectoryWord)) ||
        !rows(layout.finish_directory_offset, layout.finish_directory_words, sizeof(SchedulerAsyncDirectoryWord))) {
        LOG_ERROR("A5 HBG live sample: invalid async table extent");
        return;
    }
    LOG_ERROR(
        "A5 HBG live sample async: groups=%" PRIu64 " finishes=%" PRIu64 " per_table_limit=8", layout.group_count,
        layout.finish_count
    );
    for (uint64_t index = 0; index < std::min<uint64_t>(layout.group_count, 8); ++index) {
        SchedulerAsyncGroupIdentity identity{};
        SchedulerAsyncGroupWriter writer{};
        SchedulerAsyncGroupProgress progress{};
        SchedulerAsyncLease lease{};
        SchedulerAsyncDirectoryWord directory{};
        if (!view.read(layout.group_identity_offset, &identity, index) ||
            !view.read(layout.group_writer_offset, &writer, index) ||
            !view.read(layout.group_progress_offset, &progress, index) ||
            !view.read(layout.group_lease_offset, &lease, index) ||
            !view.read(layout.check_directory_offset, &directory, index / 64))
            return;
        LOG_ERROR(
            "A5 HBG live sample CHECK[%" PRIu64 "] epoch=%" PRIu64 " task=%" PRId64 " block=%u lane=%u"
            " finish=%" PRIu64 " lease=0x%" PRIx64 " directory[%" PRIu64 "]=0x%" PRIx64 " bit=%u"
            " sealed=%u count=%u writer_error=%d phase=%" PRIu64 " remaining=0x%" PRIx64 " next=%" PRIu64
            " observations=%" PRIu64 " slices=%" PRIu64 " failure=%" PRId64,
            index, identity.run_epoch, identity.task_ref, identity.block_idx, identity.subtask_slot,
            identity.finish_index, lease.word, index / 64, directory.bits,
            static_cast<unsigned>((directory.bits >> (index % 64)) & 1), writer.reservation.sealed, writer.count,
            writer.error_code, progress.phase, progress.remaining_mask, progress.next_condition,
            progress.observation_count, progress.slice_count, progress.failure_code
        );
    }
    for (uint64_t index = 0; index < std::min<uint64_t>(layout.finish_count, 8); ++index) {
        SchedulerAsyncFinishIdentity identity{};
        SchedulerAsyncFinishProgress progress{};
        SchedulerAsyncLease lease{};
        SchedulerAsyncTaskJoin join{};
        SchedulerAsyncDirectoryWord directory{};
        if (!view.read(layout.finish_identity_offset, &identity, index) ||
            !view.read(layout.finish_progress_offset, &progress, index) ||
            !view.read(layout.finish_lease_offset, &lease, index) ||
            !view.read(layout.finish_join_offset, &join, index) ||
            !view.read(layout.finish_directory_offset, &directory, index / 64))
            return;
        LOG_ERROR(
            "A5 HBG live sample FINISH[%" PRIu64 "] epoch=%" PRIu64 " task=%" PRId64 " lease=0x%" PRIx64
            " join=0x%" PRIx64 " directory[%" PRIu64 "]=0x%" PRIx64 " bit=%u phase=%" PRIu64 " waiter=%" PRId64
            " detached=%" PRId64 " visited=%" PRIu64 " pending=%" PRIu64 " kind=%" PRIu64 " queue=%" PRIu64
            " target=%" PRIu64 " resume=%" PRIu64 " resolve_task=%" PRId64 " resolve_producer=%" PRId64
            " resolve_fanin=%u/%u resolve_phase=%u slices=%" PRIu64 " contributed=%u",
            index, identity.run_epoch, identity.task_ref, lease.word, join.word, index / 64, directory.bits,
            static_cast<unsigned>((directory.bits >> (index % 64)) & 1), progress.completion.phase,
            progress.completion.current_waiter, progress.completion.detached_next, progress.completion.visited_count,
            progress.completion.pending_count, progress.completion.pending_kind, progress.completion.pending_queue,
            progress.completion.pending_target, progress.completion.resume_phase, progress.completion.resolve.task_ref,
            progress.completion.resolve.producer_ref, progress.completion.resolve.next_fanin,
            progress.completion.resolve.fanin_count, static_cast<unsigned>(progress.completion.resolve.phase),
            progress.slice_count, static_cast<unsigned>(progress.contributed)
        );
    }
}

inline void tickets(const View &backing, const SchedulerGraphSummary &summary, uint64_t task_count) {
    if (summary.ready_pool_offset == 0) return;
    SchedulerReadyPoolHeader pool{};
    if (!backing.read(summary.ready_pool_offset, &pool) || pool.version != SCHEDULER_READY_POOL_VERSION ||
        pool.task_count != task_count || pool.storage_end_offset > backing.bytes ||
        pool.storage_end_offset < summary.ready_pool_offset + sizeof(pool)) {
        LOG_ERROR("A5 HBG live sample: invalid ready-pool header/extent");
        return;
    }
    LOG_ERROR(
        "A5 HBG live sample ready pool: kind=%u queues=%u limit=16 entries_per_queue=4",
        static_cast<unsigned>(pool.kind), pool.queue_count
    );
    if (pool.kind != SchedulerReadyPoolKind::TICKET) return;
    const View view{backing.base, pool.storage_end_offset};
    if (!view.region(
            pool.descriptors_offset, pool.queue_count, sizeof(SchedulerReadyPoolDescriptor), 128,
            summary.ready_pool_offset + sizeof(pool)
        ) ||
        !view.region(
            pool.controls_offset, pool.queue_count, sizeof(SchedulerTicketQueue), 128,
            pool.descriptors_offset + pool.queue_count * sizeof(SchedulerReadyPoolDescriptor)
        )) {
        LOG_ERROR("A5 HBG live sample: invalid Ticket table extent");
        return;
    }
    const uint64_t entries_minimum = pool.controls_offset + pool.queue_count * sizeof(SchedulerTicketQueue);
    for (uint64_t index = 0; index < std::min<uint64_t>(pool.queue_count, 16); ++index) {
        SchedulerReadyPoolDescriptor descriptor{};
        SchedulerTicketQueue queue{};
        if (!view.read(pool.descriptors_offset, &descriptor, index) ||
            !view.read(pool.controls_offset, &queue, index) || descriptor.capacity > task_count ||
            !view.region(descriptor.entries_offset, descriptor.capacity, sizeof(int64_t), 128, entries_minimum)) {
            LOG_ERROR("A5 HBG live sample: invalid Ticket descriptor=%" PRIu64, index);
            continue;
        }
        const uint64_t head = queue.head, tail = queue.reserved_tail;
        const uint64_t begin = head <= descriptor.capacity ? (head == 0 ? 0 : head - 1) : 0;
        const uint64_t sampled = std::min<uint64_t>(descriptor.capacity - begin, 4);
        int64_t entries[4]{-1, -1, -1, -1};
        for (uint64_t i = 0; i < sampled; ++i)
            if (!view.read(descriptor.entries_offset, &entries[i], begin + i)) return;
        LOG_ERROR(
            "A5 HBG live sample TICKET[%" PRIu64 "] capacity=%" PRIu64 " head=%" PRIu64 " reserved_tail=%" PRIu64
            " sample_begin=%" PRIu64 " count=%" PRIu64 " entries=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
            index, descriptor.capacity, head, tail, begin, sampled, entries[0], entries[1], entries[2], entries[3]
        );
    }
}

inline void protocol(const View &backing, const SchedulerGraphSummary &summary, uint64_t task_count) {
    if (summary.protocol_layout_offset == 0) {
        LOG_ERROR("A5 HBG live sample: protocol recording disabled");
        return;
    }
    SchedulerProtocolLayout layout{};
    if (!backing.read(summary.protocol_layout_offset, &layout) || layout.magic != SCHEDULER_PROTOCOL_MAGIC ||
        layout.run_epoch != summary.run_epoch || layout.task_count != task_count ||
        layout.worker_count != summary.worker_count || layout.storage_end_offset > backing.bytes ||
        layout.storage_end_offset < summary.protocol_layout_offset + sizeof(layout)) {
        LOG_ERROR("A5 HBG live sample: invalid protocol header/extent");
        return;
    }
    const View view{backing.base, layout.storage_end_offset};
    if (!view.region(
            layout.executions_offset, layout.trace_count, sizeof(SchedulerProtocolExecutionTrace),
            alignof(SchedulerProtocolExecutionTrace), summary.protocol_layout_offset + sizeof(layout)
        )) {
        LOG_ERROR("A5 HBG live sample: invalid protocol execution extent");
        return;
    }
    LOG_ERROR(
        "A5 HBG live sample protocol: executions=%" PRIu64 " limit=16 (last published checkpoints)", layout.trace_count
    );
    for (uint64_t index = 0; index < std::min<uint64_t>(layout.trace_count, 16); ++index) {
        SchedulerProtocolExecutionTrace execution{};
        if (!view.read(layout.executions_offset, &execution, index)) return;
        LOG_ERROR(
            "A5 HBG live sample EXEC[%" PRIu64 "] epoch=%" PRIu64 " task=%" PRId64 " block=%u lane=%u flags=0x%x"
            " worker=%u slot=%u generation=%u publications=%u permit=%" PRIu64 "/%" PRIu64 " entry=%" PRIu64 "/%" PRIu64
            " returned=%" PRIu64 "/%" PRIu64 " visible=%" PRIu64 "/%" PRIu64 " consumed=%" PRIu64 "/%" PRIu64
            " trace_published=%" PRIu64 "/%" PRIu64,
            index, execution.run_epoch, execution.task_ref, execution.block_idx,
            static_cast<unsigned>(execution.subtask_slot), execution.flags, execution.worker_index,
            static_cast<unsigned>(execution.pending_slot), execution.generation, execution.publication_count,
            execution.permit.cycles, execution.permit.sequence, execution.entry.cycles, execution.entry.sequence,
            execution.returned.cycles, execution.returned.sequence, execution.visible.cycles,
            execution.visible.sequence, execution.completion_consumed.cycles, execution.completion_consumed.sequence,
            execution.trace_published.cycles, execution.trace_published.sequence
        );
    }
}
}  // namespace scheduler_timeout_diagnostics

inline void log_aicore_scheduler_timeout_checkpoints(Runtime *runtime, uint64_t expected_epoch) {
    using namespace scheduler_timeout_diagnostics;
    const auto *published = aicore_scheduler_bootstrap_context(runtime);
    if (published == nullptr || expected_epoch == 0) return;
    // The bootstrap context is the existing validated launch root. Read its
    // immutable prefix only; all subsequent addressing is bounded by that run.
    constexpr size_t prefix = offsetof(SchedulerWorkerContext, scheduler_state_bytes) + sizeof(uint64_t);
    SchedulerWorkerContext config{};
    cache_invalidate_range(published, prefix);
    rmb();
    std::memcpy(&config, published, prefix);
    rmb();
    const View view{
        reinterpret_cast<const uint8_t *>(config.scheduler_state_base_address), config.scheduler_state_bytes
    };
    SchedulerRunControl run{};
    SchedulerGraphSummary summary{};
    if (!view.valid() || config.scheduler_state_base_address % 128 != 0 || config.graph_task_count > INT32_MAX ||
        !view.read(config.run_control_offset, &run) || run.graph_summary_offset != config.graph_summary_offset ||
        !view.read(config.graph_summary_offset, &summary) || summary.run_epoch != expected_epoch ||
        summary.worker_count != config.runtime_worker_count) {
        LOG_ERROR("A5 HBG live sample: invalid current-run scheduler root/extent/epoch");
        return;
    }
    LOG_ERROR(
        "A5 HBG live sample epoch=%" PRIu64 " bytes=%" PRIu64
        " (not a consistent snapshot; last published checkpoints; no raw payload/local state)",
        expected_epoch, view.bytes
    );
    tasks(view, config);
    async_checkpoints(view, summary, config.graph_task_count);
    tickets(view, summary, config.graph_task_count);
    protocol(view, summary, config.graph_task_count);
}
