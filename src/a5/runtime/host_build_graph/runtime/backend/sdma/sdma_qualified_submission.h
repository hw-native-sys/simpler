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

#include "sdma_metadata_handoff.h"
#include "async_completion_writer.h"
#include "backend/async_resource_binding.h"
#include "scheduler/scheduler_submission_provenance.h"

// A receipt copies only the public backend event's address and full-width
// threshold. It is not permission to submit, a scheduler token, or a READY fact.
struct SdmaCompletionReceiptRecord {
    uint64_t address{0};
    uint64_t expected{0};
};
struct SdmaIgnoreCompletionRecords {
    inline __aicore__ void operator()(uint32_t, uint32_t, const SdmaCompletionReceiptRecord &) const {}
};

namespace pto2::detail {

// Local state produced by the qualified image reader. This carries no owner and
// must not outlive the current kernel call. Public SDK entry points construct it
// only after checking the canonical frame, immutable use map, and active grant.
struct SdmaQualifiedAccess {
    SchedulerSubmissionExecutionIdentity identity{};
    SchedulerSubmissionWorkspace workspace{};
    SchedulerSubmissionGrant grant{};
    uint64_t use_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t workspace_index{SCHEDULER_ASYNC_INDEX_NONE};
    __gm__ SchedulerSubmissionProvenance *provenance{nullptr};
    uint32_t provenance_capacity{0};
};

inline __aicore__ bool
sdma_qualified_scratch_valid(uint64_t address, uint64_t bytes, uint32_t alignment, uint32_t sync_id) {
    return address != 0 && alignment != 0 && (alignment & (alignment - 1)) == 0 && address % alignment == 0 &&
           bytes >= sizeof(uint64_t) && bytes <= UINT32_MAX && sync_id <= 7;
}

// Each interval is the exact contiguous view published by Host. Retention
// covers its cache-line envelope; it never authorizes the rest of that owner.
inline __aicore__ bool sdma_qualified_range(
    const AsyncResourceTables &tables, uint64_t index, uint64_t access, uint64_t address, uint64_t bytes
) {
    using namespace async_resource_detail;
    if (index >= tables.header.range_count || address == 0 || bytes > UINT64_MAX - address) return false;
    const auto range = load(tables.ranges + index);
    if (range.span_index >= tables.header.span_count || range.bytes == 0 || (range.access & ~UINT64_C(15)) != 0 ||
        (range.access & access) != access)
        return false;
    const auto owner = load(tables.spans + range.span_index);
    if (owner.address == 0 || owner.bytes > UINT64_MAX - owner.address || range.byte_offset > owner.bytes ||
        range.bytes > owner.bytes - range.byte_offset)
        return false;
    const uint64_t start = owner.address + range.byte_offset;
    if (!contains(start, range.bytes, address, bytes) || start > UINT64_MAX - 63 ||
        range.bytes > UINT64_MAX - start - 63)
        return false;
    const uint64_t first = start & ~UINT64_C(63);
    const uint64_t end = (start + range.bytes + 63) & ~UINT64_C(63);
    return contains(owner.address, owner.bytes, first, end - first);
}

inline __aicore__ bool sdma_qualified_select_use(
    const AsyncSubmissionContextHeader &h, int64_t task_ref, const AsyncResourceTables &tables, uint64_t workspace,
    uint64_t source, uint64_t destination, uint64_t bytes, uint64_t *selected
) {
    using namespace async_resource_detail;
    if (selected == nullptr) return false;
    *selected = SCHEDULER_ASYNC_INDEX_NONE;
    if (tables.header.version != ASYNC_RESOURCE_VERSION || tables.header.run_epoch != h.run_epoch ||
        tables.header.execution_contract != ASYNC_EXECUTION_UNQUALIFIED || tables.conditions == nullptr ||
        tables.ranges == nullptr || tables.spans == nullptr || h.use_count == 0 ||
        h.use_begin > tables.header.condition_count || h.use_count > tables.header.condition_count - h.use_begin ||
        workspace == 0)
        return false;
    uint64_t match = SCHEDULER_ASYNC_INDEX_NONE;
    for (uint64_t ordinal = 0; ordinal < h.use_count; ++ordinal) {
        const uint64_t index = h.use_begin + ordinal;
        const auto use = load(tables.conditions + index);
        if (use.task_ref != task_ref || use.block_idx != h.block_idx || use.subtask_slot != h.subtask_slot ||
            use.use_index != ordinal)
            return false;
        if (use.completion_type == COMPLETION_TYPE_COUNTER) continue;
        if (use.completion_type != COMPLETION_TYPE_SDMA_EVENT_RECORD || use.payload_none != 0 ||
            use.record_range != ASYNC_RESOURCE_NONE || use.record_count != 0 || use.record_stride != 0 ||
            use.payload_count != 1 || use.sdma_workspace_address == 0 || (use.observer_cores & ~UINT32_C(3)) != 0 ||
            use.observer_cores == 0)
            return false;
        if (use.sdma_workspace_address != workspace ||
            !sdma_qualified_range(tables, use.source_range, ASYNC_RANGE_READ, source, bytes) ||
            !sdma_qualified_range(tables, use.payload_range, ASYNC_RANGE_READ | ASYNC_RANGE_WRITE, destination, bytes))
            continue;
        if (match != SCHEDULER_ASYNC_INDEX_NONE) return false;
        match = index;
    }
    *selected = match;
    return match != SCHEDULER_ASYNC_INDEX_NONE;
}

inline __aicore__ bool sdma_qualified_writer(__gm__ int64_t *args, AsyncCtx *writer_out) {
    if (args == nullptr || writer_out == nullptr || reinterpret_cast<uint64_t>(args) < offsetof(DispatchPayload, args))
        return false;
    const uint64_t payload_address = reinterpret_cast<uint64_t>(args) - offsetof(DispatchPayload, args);
    if (payload_address % alignof(DispatchPayload) != 0) return false;
    const auto *frame = reinterpret_cast<__gm__ const DispatchPayload *>(payload_address);
    const auto &writer = frame->local_context.async_ctx;
    // The current core may already have registered COUNTERs. Do not invalidate
    // its own dirty writer as get_async_ctx's initial-load helper would do.
    writer_out->completion_count = writer.completion_count;
    writer_out->completion_error_code = writer.completion_error_code;
    writer_out->completion_entries = writer.completion_entries;
    writer_out->completion_capacity = writer.completion_capacity;
    TaskId::assign(writer_out->task_token, writer.task_token);
    if (frame->args[SPMD_LOCAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&frame->local_context) ||
        frame->args[SPMD_GLOBAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&frame->global_context))
        return false;
    return true;
}

inline __aicore__ bool sdma_qualified_writer_matches(
    __gm__ const uint8_t *base, const SchedulerAsyncLayout &layout, const SchedulerSubmissionExecution &execution,
    const AsyncSubmissionContextHeader &h, const AsyncCtx &writer
) {
    using namespace async_resource_detail;
    if (h.completion_mode == CoreAsyncMode::SYNCHRONOUS)
        return !writer.task_token.is_valid() && writer.completion_count == nullptr &&
               writer.completion_error_code == nullptr && writer.completion_entries == nullptr &&
               writer.completion_capacity == 0;
    if (execution.task_ref < 0 || static_cast<uint64_t>(execution.task_ref) >= layout.task_count ||
        layout.task_count > UINT64_MAX / sizeof(SchedulerAsyncTaskMap) ||
        layout.group_count > UINT64_MAX / sizeof(SchedulerAsyncGroupIdentity) ||
        !section(layout.total_bytes, layout.task_map_offset, layout.task_count * sizeof(SchedulerAsyncTaskMap)) ||
        !section(
            layout.total_bytes, layout.group_identity_offset, layout.group_count * sizeof(SchedulerAsyncGroupIdentity)
        ) ||
        !section(
            layout.total_bytes, layout.group_writer_offset, layout.group_count * sizeof(SchedulerAsyncGroupWriter)
        ))
        return false;
    const auto map = load(
        reinterpret_cast<__gm__ const SchedulerAsyncTaskMap *>(base + layout.task_map_offset) + execution.task_ref
    );
    if (map.async_lane_mask == 0 || (map.async_lane_mask & ~7u) != 0 ||
        map.groups_per_block != static_cast<uint32_t>(__builtin_popcount(map.async_lane_mask)) ||
        map.group_count != static_cast<uint64_t>(map.block_count) * map.groups_per_block ||
        map.group_base > layout.group_count || map.group_count > layout.group_count - map.group_base)
        return false;
    const uint64_t group_index = scheduler_async_group_index(map, h.block_idx, h.subtask_slot);
    if (group_index == SCHEDULER_ASYNC_INDEX_NONE || group_index >= layout.group_count) return false;
    SchedulerLocalStorage<SchedulerAsyncGroupIdentity> group_storage;
    load_into(
        reinterpret_cast<__gm__ const SchedulerAsyncGroupIdentity *>(base + layout.group_identity_offset) + group_index,
        &group_storage.get()
    );
    const auto &group = group_storage.get();
    if (group.run_epoch != h.run_epoch || group.task_ref != execution.task_ref || group.block_idx != h.block_idx ||
        group.subtask_slot != h.subtask_slot || group.capacity != execution.condition_capacity ||
        group.condition_types != execution.condition_types || group.entries_offset % 128 != 0 ||
        !section(layout.total_bytes, layout.entries_offset, layout.entries_bytes) ||
        !contains(
            layout.entries_offset, layout.entries_bytes, group.entries_offset,
            static_cast<uint64_t>(group.capacity) * sizeof(DeferredCompletionEntry)
        ))
        return false;
    const auto *canonical =
        reinterpret_cast<__gm__ const SchedulerAsyncGroupWriter *>(base + layout.group_writer_offset) + group_index;
    return TaskId::to_uint64(writer.task_token) == h.task_token && writer.completion_count == &canonical->count &&
           writer.completion_error_code == &canonical->error_code && writer.completion_capacity == group.capacity &&
           writer.completion_entries ==
               reinterpret_cast<__gm__ const DeferredCompletionEntry *>(base + group.entries_offset);
}

struct SdmaQualifiedFrame {
    __gm__ AsyncSubmissionContext *context{nullptr};
    AsyncCtx writer{};
    AsyncSubmissionContextHeader header{};
    SchedulerAsyncLayout layout{};
    SchedulerSubmissionProvenanceView proof{};
    SchedulerSubmissionExecution execution{};
};

// Establish a canonical error destination before inspecting transfer shape or
// resource permissions. A malformed writer never escapes this function; if
// only that writer fails, the independently verified context can record error.
inline __aicore__ bool sdma_qualified_frame(__gm__ int64_t *args, SdmaQualifiedFrame *out) {
    if (out == nullptr) return false;
    scheduler_reset_local(out);
    auto *context = get_async_submission_context(args);
    if (context == nullptr) return false;
    const auto &h = context->header;
    if (h.region_bytes < sizeof(SchedulerAsyncLayout) || h.subtask_slot == 0) return false;
    auto *base = reinterpret_cast<__gm__ uint8_t *>(h.region_address);
    out->layout = async_resource_detail::load(reinterpret_cast<__gm__ const SchedulerAsyncLayout *>(base));
    if (out->layout.total_bytes != h.region_bytes || out->layout.run_epoch != h.run_epoch ||
        !scheduler_submission_provenance_mount(base, out->layout, &out->proof) ||
        !scheduler_submission_context_header(out->proof, h.execution_index, &out->header) ||
        scheduler_submission_canonical_context(out->proof, h.execution_index) != context ||
        !async_submission_headers_equal(out->header, h) ||
        !scheduler_submission_detail::execution(out->proof.submissions, h.execution_index, &out->execution) ||
        out->execution.task_ref != context->evidence.task_ref)
        return false;
    out->context = context;
    AsyncCtx writer{};
    if (!sdma_qualified_writer(args, &writer) ||
        !sdma_qualified_writer_matches(base, out->layout, out->execution, out->header, writer))
        return false;
    out->writer = writer;
    return true;
}

inline __aicore__ bool sdma_qualified_access_from_frame(
    const SdmaQualifiedFrame &frame, uint64_t workspace_address, uint64_t source, uint64_t destination, uint64_t bytes,
    SdmaQualifiedAccess *out
) {
    if (out == nullptr || frame.context == nullptr) return false;
    *out = {};
    const auto &h = frame.header;
    const auto &execution = frame.execution;
    const auto &proof = frame.proof;
    auto *base = reinterpret_cast<__gm__ uint8_t *>(h.region_address);
    SchedulerLocalStorage<AsyncResourceTables> resources_storage;
    auto &resources = resources_storage.get();
    if (!AsyncResourceTables::mount(base, frame.layout, &resources)) return false;
    uint64_t use_index = SCHEDULER_ASYNC_INDEX_NONE;
    if (!sdma_qualified_select_use(
            h, execution.task_ref, resources, workspace_address, source, destination, bytes, &use_index
        ))
        return false;
    bool found = false;
    for (uint64_t ordinal = 0; ordinal < execution.grant_count; ++ordinal) {
        SchedulerSubmissionGrant grant{};
        SchedulerSubmissionWorkspace workspace{};
        uint64_t grant_index = 0;
        if (!scheduler_submission_lookup_grant(
                proof.submissions, h.execution_index, ordinal, &grant, &workspace, &grant_index
            ))
            return false;
        if (workspace.address != workspace_address) continue;
        if (found) return false;
        found = true;
        out->grant = grant;
        out->workspace = workspace;
        out->workspace_index = grant.workspace_index;
    }
    if (!found) return false;
    out->identity = {h.run_epoch,           execution.task_ref, h.execution_index,
                     h.dispatch_generation, h.block_idx,        h.subtask_slot};
    out->use_index = use_index;
    out->provenance_capacity = static_cast<uint32_t>(h.provenance_capacity);
    if (h.provenance_capacity != 0)
        out->provenance = reinterpret_cast<__gm__ SchedulerSubmissionProvenance *>(base + h.provenance_offset);
    return true;
}

// Read-only entry for callers needing the fully authorized view. Failure never
// exports a writer or context that a caller could mistake for write permission.
inline __aicore__ bool sdma_qualified_access(
    __gm__ int64_t *args, uint64_t workspace_address, uint64_t source, uint64_t destination, uint64_t bytes,
    __gm__ AsyncSubmissionContext **context_out, AsyncCtx *writer_out, SdmaQualifiedAccess *out
) {
    if (context_out == nullptr || writer_out == nullptr || out == nullptr) return false;
    *context_out = nullptr;
    *writer_out = {};
    *out = {};
    SchedulerLocalStorage<SdmaQualifiedFrame> frame_storage;
    auto &frame = frame_storage.get();
    SdmaQualifiedAccess access{};
    if (!sdma_qualified_frame(args, &frame) ||
        !sdma_qualified_access_from_frame(frame, workspace_address, source, destination, bytes, &access))
        return false;
    *context_out = frame.context;
    *writer_out = frame.writer;
    *out = access;
    return true;
}

template <typename Cache>
inline __aicore__ bool sdma_qualified_error(
    __gm__ AsyncSubmissionContext &context, AsyncCtx &writer, Cache &cache,
    int32_t error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
) {
    async_submission_error(context, error);
    async_completion_error(writer, error);
    async_completion_flush(writer, cache);
    cache.flush_range(&context.evidence, sizeof(context.evidence));
    cache.finish();
    return false;
}

// Backend builds a real session, posts, and exports the public event records.
// No default backend or successful publisher exists. Injected CPU backends test
// this protocol; they do not qualify a DMA engine or cache visibility.
template <typename Backend, typename Cache, typename Sink>
inline __aicore__ bool sdma_qualified_submit(
    __gm__ AsyncSubmissionContext &context, AsyncCtx &writer, const SdmaQualifiedAccess &access,
    uint64_t transfer_bytes, const SdmaMetadataLayout &layout, Backend &backend, Cache &cache, Sink &sink
) {
    const auto &h = context.header;
    const bool deferred = h.completion_mode == CoreAsyncMode::DEFERRED;
    const auto &identity = access.identity;
    if (!async_submission_context_valid(&context) || context.evidence.error_code != 0 ||
        (context.evidence.flags & ASYNC_SUBMISSION_SEALED) != 0 || identity.run_epoch != h.run_epoch ||
        identity.task_ref != context.evidence.task_ref || identity.execution_index != h.execution_index ||
        identity.dispatch_generation != h.dispatch_generation || identity.block_idx != h.block_idx ||
        identity.subtask_slot != h.subtask_slot || access.use_index < h.use_begin ||
        access.use_index - h.use_begin >= h.use_count ||
        (deferred ? !async_completion_writer_valid(writer) ||
                        access.provenance_capacity != writer.completion_capacity || access.provenance == nullptr :
                    writer.task_token.is_valid() || access.provenance_capacity != 0 || access.provenance != nullptr))
        return sdma_qualified_error(context, writer, cache);
    // Grant-derived geometry is checked before BuildSession, which itself has
    // cache/metadata effects. The actual built session is checked again below.
    const SdmaMetadataSession planned{
        access.workspace.address, access.grant.group_index, access.workspace.queue_num, true
    };
    SdmaMetadataHandoffRange metadata{};
    if (!sdma_metadata_handoff_preflight(
            layout, access.workspace_index, access.workspace, access.grant, identity, planned, &metadata
        ) ||
        !backend.preflight())
        return sdma_qualified_error(context, writer, cache);
    if (transfer_bytes == 0) return true;
    AsyncCompletionAllowance allowance{};
    if (deferred && !async_completion_reserve(writer, access.workspace.queue_num, &allowance))
        return sdma_qualified_error(
            context, writer, cache,
            writer.completion_error_code != nullptr ? *writer.completion_error_code :
                                                      SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
        );
    if (!async_submission_begin(context)) return sdma_qualified_error(context, writer, cache);
    if (!backend.build_session(access.workspace.queue_num, static_cast<uint32_t>(access.grant.group_index)))
        return sdma_qualified_error(context, writer, cache);
    const auto session = backend.metadata_session();
    SdmaMetadataHandoffPublisher<Cache> publisher(
        layout, access.workspace_index, access.workspace, access.grant, identity, session, cache
    );
    if (!publisher.preflight() || !backend.submit() || !backend.event_valid(transfer_bytes))
        return sdma_qualified_error(context, writer, cache);
    const uint32_t count = backend.record_count();
    if (count == 0 || count > access.workspace.queue_num ||
        (deferred && !async_completion_accept_count(writer, allowance, count)))
        return sdma_qualified_error(context, writer, cache);
    SdmaCompletionReceiptRecord records[SCHEDULER_SUBMISSION_SDMA_CHANNELS]{};
    for (uint32_t i = 0; i < count; ++i) {
        auto &record = records[i];
        if (!backend.record(i, &record) || record.address == 0 || record.address % alignof(uint64_t) != 0 ||
            record.expected == 0 || record.address < access.workspace.address ||
            record.address - access.workspace.address > access.workspace.bytes ||
            UINT64_C(64) > access.workspace.bytes - ((record.address & ~UINT64_C(63)) - access.workspace.address))
            return sdma_qualified_error(context, writer, cache);
        for (uint32_t j = 0; j < i; ++j)
            if (records[j].address == record.address) return sdma_qualified_error(context, writer, cache);
    }
    if (deferred) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t slot = allowance.begin + i;
            const auto &record = records[i];
            const CompletionToken token{
                record.address, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, record.expected
            };
            if (!async_completion_append(writer, token)) return sdma_qualified_error(context, writer, cache);
            auto &provenance = access.provenance[slot];
            provenance.version = 1;
            provenance.completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
            provenance.address = record.address;
            provenance.backend_cookie = record.expected;
            provenance.use_index = access.use_index;
            provenance.workspace_index = access.workspace_index;
            provenance.execution_index = identity.execution_index;
            provenance.channel_group = access.grant.group_index;
            provenance.reserved = 0;
            cache.flush_range(&provenance, sizeof(provenance));
        }
        async_completion_flush(writer, cache);
    } else if (!backend.wait()) {
        return sdma_qualified_error(context, writer, cache);
    }
    if (!async_submission_commit(context, publisher)) return sdma_qualified_error(context, writer, cache);
    for (uint32_t i = 0; i < count; ++i)
        sink(i, count, records[i]);
    return true;
}

// Shared by the public PTO overload and CPU image tests: shape/context early
// failures cannot write or flush pointers copied from an unverified frame.
template <typename Backend, typename Cache, typename Sink>
inline __aicore__ bool sdma_qualified_entry(
    __gm__ int64_t *args, bool valid_shape, uint64_t workspace, uint64_t source, uint64_t destination, uint64_t bytes,
    const SdmaMetadataLayout &layout, Backend &backend, Cache &cache, Sink &sink
) {
    SchedulerLocalStorage<SdmaQualifiedFrame> frame_storage;
    auto &frame = frame_storage.get();
    if (!sdma_qualified_frame(args, &frame))
        return frame.context == nullptr ? false : sdma_qualified_error(*frame.context, frame.writer, cache);
    SdmaQualifiedAccess access{};
    if (!valid_shape || !sdma_qualified_access_from_frame(frame, workspace, source, destination, bytes, &access))
        return sdma_qualified_error(*frame.context, frame.writer, cache);
    return sdma_qualified_submit(*frame.context, frame.writer, access, bytes, layout, backend, cache, sink);
}

template <typename Cache>
inline __aicore__ bool sdma_qualified_reject(__gm__ int64_t *args, Cache &cache) {
    SchedulerLocalStorage<SdmaQualifiedFrame> frame_storage;
    auto &frame = frame_storage.get();
    (void)sdma_qualified_frame(args, &frame);
    return frame.context == nullptr ? false : sdma_qualified_error(*frame.context, frame.writer, cache);
}

}  // namespace pto2::detail
