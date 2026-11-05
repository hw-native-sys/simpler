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

#include <cstdint>
#include <type_traits>

#include "async_capability.h"
#include "dispatch_payload.h"
#include "host_build_graph/runtime_status.h"

inline constexpr uint64_t ASYNC_SUBMISSION_VERSION = 1;
inline constexpr uint32_t ASYNC_SUBMISSION_METADATA_PUBLISHED = 1;
inline constexpr uint32_t ASYNC_SUBMISSION_SEALED = 2;
inline constexpr uint32_t ASYNC_SUBMISSION_FLAGS = ASYNC_SUBMISSION_METADATA_PUBLISHED | ASYNC_SUBMISSION_SEALED;

// The original executing core owns this row. The SDK publishes the attempted
// and committed submission facts; only the runtime seals a returned execution.
// It never shares a cache line with a channel-pool atomic or a bundle lease.
struct alignas(128) AsyncSubmissionEvidence {
    uint64_t version{ASYNC_SUBMISSION_VERSION};
    uint64_t run_epoch{0};
    int64_t task_ref{-1};
    uint64_t execution_index{UINT64_MAX};
    uint64_t dispatch_generation{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    volatile uint64_t attempted{0};
    volatile uint64_t committed{0};
    volatile int32_t error_code{0};
    volatile uint32_t flags{ASYNC_SUBMISSION_METADATA_PUBLISHED};
    uint64_t reserved[7]{};
};

// Relative offsets name tables in the retained run image. Submission layout
// interpretation belongs to the backend; this header supplies execution identity
// and bounds without overloading a completion token or a TaskId.
struct alignas(128) AsyncSubmissionContextHeader {
    uint64_t version{ASYNC_SUBMISSION_VERSION};
    uint64_t context_bytes{256};
    uint64_t run_epoch{0};
    uint64_t task_token{0};
    uint64_t region_address{0};
    uint64_t region_bytes{0};
    uint64_t submission_layout_offset{0};
    uint64_t execution_index{UINT64_MAX};
    uint64_t use_begin{0};
    uint64_t use_count{0};
    uint64_t provenance_offset{0};
    uint64_t provenance_capacity{0};
    uint64_t dispatch_generation{0};
    uint64_t frame_address{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t frame_sub_block_id{0};
    CoreAsyncMode completion_mode{CoreAsyncMode::UNKNOWN};
};

struct alignas(128) AsyncSubmissionContext {
    AsyncSubmissionContextHeader header{};
    AsyncSubmissionEvidence evidence{};
};

static_assert(sizeof(AsyncSubmissionEvidence) == 128 && sizeof(AsyncSubmissionContextHeader) == 128);
static_assert(sizeof(AsyncSubmissionContext) == 256 && offsetof(AsyncSubmissionContext, evidence) == 128);
static_assert(
    std::is_standard_layout_v<AsyncSubmissionContext> && std::is_trivially_copyable_v<AsyncSubmissionContext>
);

inline __aicore__ bool async_submission_context_valid(__gm__ const AsyncSubmissionContext *context) {
    if (context == nullptr || reinterpret_cast<uint64_t>(context) % alignof(AsyncSubmissionContext) != 0) return false;
    const auto &h = context->header;
    const auto &e = context->evidence;
    const uint64_t address = reinterpret_cast<uint64_t>(context);
    if (h.version != ASYNC_SUBMISSION_VERSION || h.context_bytes != sizeof(*context) || h.run_epoch == 0 ||
        h.dispatch_generation == 0 || h.execution_index == UINT64_MAX || h.subtask_slot >= 3 || h.frame_address == 0 ||
        h.frame_address % alignof(DispatchPayload) != 0 || h.frame_address > UINT64_MAX - sizeof(DispatchPayload) ||
        h.frame_sub_block_id > 1 || h.region_address == 0 || h.region_address % 128 != 0 ||
        h.region_bytes > UINT64_MAX - h.region_address || address < h.region_address ||
        address - h.region_address > h.region_bytes ||
        sizeof(*context) > h.region_bytes - (address - h.region_address) || h.submission_layout_offset == 0 ||
        h.submission_layout_offset >= h.region_bytes || h.submission_layout_offset % 128 != 0 ||
        (h.completion_mode != CoreAsyncMode::SYNCHRONOUS && h.completion_mode != CoreAsyncMode::DEFERRED) ||
        e.version != h.version || e.run_epoch != h.run_epoch || e.task_ref < 0 ||
        e.execution_index != h.execution_index || e.dispatch_generation != h.dispatch_generation ||
        e.block_idx != h.block_idx || e.subtask_slot != h.subtask_slot || (e.flags & ~ASYNC_SUBMISSION_FLAGS) != 0)
        return false;
    for (uint64_t reserved : e.reserved)
        if (reserved != 0) return false;
    return true;
}

// Only a scheduler-published pointer can provide this context. Old/synchronous
// dispatch without a qualified submission contract has zero in the named field.
// The caller still validates its requested resource and granted channel before
// touching a backend; obtaining this identity view is not a submit permission.
inline __aicore__ __gm__ AsyncSubmissionContext *get_async_submission_context(__gm__ int64_t *args) {
    const uint64_t argument_address = reinterpret_cast<uint64_t>(args);
    if (args == nullptr || argument_address < offsetof(DispatchPayload, args)) return nullptr;
    const uint64_t payload_address = argument_address - offsetof(DispatchPayload, args);
    if (payload_address % alignof(DispatchPayload) != 0) return nullptr;
    const auto *payload = reinterpret_cast<__gm__ const DispatchPayload *>(payload_address);
    if (payload->args[SPMD_LOCAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&payload->local_context) ||
        payload->args[SPMD_GLOBAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&payload->global_context))
        return nullptr;
    auto *context = reinterpret_cast<__gm__ AsyncSubmissionContext *>(payload->async_submission_context);
    if (!async_submission_context_valid(context)) return nullptr;
    const auto &h = context->header;
    if ((context->evidence.flags & ASYNC_SUBMISSION_SEALED) != 0 || h.frame_address != payload_address ||
        h.frame_sub_block_id != payload->global_context.sub_block_id ||
        h.block_idx != static_cast<uint32_t>(payload->local_context.block_idx))
        return nullptr;
    const auto token = TaskId::to_uint64(payload->local_context.async_ctx.task_token);
    const bool valid_token = token != TaskId::to_uint64(TaskId::invalid());
    if (h.completion_mode == CoreAsyncMode::DEFERRED) {
        if (!valid_token || token != h.task_token) return nullptr;
    } else if (valid_token) return nullptr;
    return context;
}

inline __aicore__ void async_submission_error(__gm__ AsyncSubmissionContext &context, int32_t error) {
    if (context.evidence.error_code == 0)
        context.evidence.error_code = error != 0 ? error : SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
}

// Called only after resource/channel/shape/capacity preflight succeeds and just
// before the backend can have side effects. A failed or uncommitted attempt is
// sticky evidence; it is never rolled back into a fabricated no-submit state.
inline __aicore__ bool async_submission_begin(__gm__ AsyncSubmissionContext &context) {
    auto &e = context.evidence;
    if (!async_submission_context_valid(&context) || (e.flags & ASYNC_SUBMISSION_SEALED) != 0) return false;
    if (e.error_code != 0 || e.flags != ASYNC_SUBMISSION_METADATA_PUBLISHED || e.attempted != e.committed ||
        e.attempted == UINT64_MAX) {
        async_submission_error(context, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    e.flags = 0;
    e.attempted = e.attempted + 1;
    return true;
}

// Publisher completes the actual backend queue/cursor/committed-post stores
// before this SDK fact is published. Token writeback alone is not this proof.
template <typename Publisher>
inline __aicore__ bool async_submission_commit(__gm__ AsyncSubmissionContext &context, Publisher &publisher) {
    auto &e = context.evidence;
    if (!async_submission_context_valid(&context) || (e.flags & ASYNC_SUBMISSION_SEALED) != 0) return false;
    if (e.error_code != 0 || e.flags != 0 || e.committed == UINT64_MAX || e.attempted != e.committed + 1) {
        async_submission_error(context, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    publisher.publish_backend_metadata();
    e.committed = e.committed + 1;
    e.flags = ASYNC_SUBMISSION_METADATA_PUBLISHED;
    publisher.flush_evidence(&e, sizeof(e));
    publisher.finish();
    return true;
}
