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

#include "backend/async_backend.h"
#include "sdma_submission_contract.h"
#include "sdma_submission_guard.h"

// A receipt is a copy of the SDK's public event record, not a completion fact.
struct SdmaCompletionReceiptRecord {
    uint64_t address{0};
    uint64_t expected{0};
};
struct SdmaIgnoreCompletionRecords {
    inline __aicore__ void operator()(uint32_t, uint32_t, const SdmaCompletionReceiptRecord &) const {}
};

namespace pto2::detail {
inline __aicore__ bool sdma_scratch_valid(uint64_t address, uint64_t bytes, uint32_t alignment, uint32_t sync_id) {
    // UB addresses are offsets; zero is a valid scratch allocation.
    return alignment != 0 && (alignment & (alignment - 1)) == 0 && address % alignment == 0 &&
           bytes >= sizeof(uint64_t) && bytes <= UINT32_MAX && sync_id <= 7;
}

template <typename Cache>
inline __aicore__ bool sdma_submit_error(AsyncCtx &writer, Cache &cache) {
    async_completion_error(writer, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    async_completion_flush(writer, cache);
    return false;
}

// Both public entry points use this transaction. The backend exposes the real
// session/event; CPU tests inject their boundary behavior, never DMA completion.
template <typename Backend, typename Cache, typename Sink, typename Metadata = SdmaSubmissionMetadata>
inline __aicore__ bool sdma_submit(
    AsyncCtx &writer, SdmaOp operation, bool valid_shape, uint64_t workspace, uint64_t source, uint64_t destination,
    uint64_t bytes, uint32_t physical_cluster, Backend &backend, Cache &cache, Sink &sink, Metadata metadata = {},
    uint32_t queues = 1
) {
    const bool deferred = writer.task_token.is_valid();
    // Nonempty deferred submissions validate writer state when reserving capacity.
    // Zero-work calls have no reservation, but must still reject an invalid writer.
    if ((deferred && bytes == 0 && !async_completion_open(writer)) ||
        (writer.completion_error_code != nullptr && *writer.completion_error_code != 0))
        return sdma_submit_error(writer, cache);
    uint64_t guard_address = 0;
    if (!valid_shape || (operation != SdmaOp::TGET && operation != SdmaOp::TPUT) || workspace == 0 ||
        workspace % 64 != 0 || source == 0 || destination == 0 || bytes > UINT64_MAX - source ||
        bytes > UINT64_MAX - destination || !async_sdma_guard_address(writer, &guard_address) ||
        (writer.execution_context != nullptr && writer.execution_context->subtask_slot == 0))
        return sdma_submit_error(writer, cache);
    if (queues == 0 || queues > ASYNC_SDMA_CHANNELS || !backend.preflight()) return sdma_submit_error(writer, cache);
    if (bytes == 0) return true;
    const uint32_t group = physical_cluster % (ASYNC_SDMA_CHANNELS / queues);
    AsyncCompletionReservation reservation{};
    if (deferred && !async_completion_reserve(writer, queues, &reservation)) return sdma_submit_error(writer, cache);
    SdmaSubmissionScope submission;
    if (!submission.acquire(writer, guard_address, group * queues, queues) || !backend.build_session(queues, group) ||
        !backend.session_valid(queues, group) || !backend.submit() || !backend.event_valid(bytes))
        return sdma_submit_error(writer, cache);
    const bool has_records = backend.has_records();
    const uint32_t count = has_records ? backend.record_count() : 0;
    if ((has_records && (count == 0 || count > queues)) ||
        (deferred && !async_completion_accept_count(writer, reservation, count)))
        return sdma_submit_error(writer, cache);
    // Snapshot under the queue guards: another submit can advance SDK metadata
    // as soon as they are released, including before a synchronous Wait returns.
    SdmaCompletionReceiptRecord records[ASYNC_SDMA_CHANNELS]{};
    for (uint32_t i = 0; i < count; ++i) {
        auto &record = records[i];
        if (!backend.record(i, &record) || record.address == 0 || record.address % alignof(uint64_t) != 0 ||
            record.expected == 0)
            return sdma_submit_error(writer, cache);
        for (uint32_t j = 0; j < i; ++j)
            if (records[j].address == record.address) return sdma_submit_error(writer, cache);
    }
    if (deferred) {
        for (uint32_t i = 0; i < count; ++i) {
            const auto &record = records[i];
            const CompletionToken token{
                record.address, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, record.expected
            };
            if (!async_completion_commit(writer, reservation, token)) return sdma_submit_error(writer, cache);
        }
        if (!async_completion_finish(writer, reservation, count)) return sdma_submit_error(writer, cache);
        async_completion_flush(writer, cache);
    }
    if (!submission.release(true, reinterpret_cast<__gm__ uint8_t *>(workspace), metadata))
        return sdma_submit_error(writer, cache);
    if (!deferred && !backend.wait()) return sdma_submit_error(writer, cache);
    for (uint32_t i = 0; i < count; ++i)
        sink(i, count, records[i]);
    return true;
}
}  // namespace pto2::detail
