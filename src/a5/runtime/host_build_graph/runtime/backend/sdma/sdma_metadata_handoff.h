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

#include <cstddef>
#include <cstdint>

#include "async_submission_context.h"
#include "scheduler/scheduler_submission_types.h"

inline constexpr char SDMA_METADATA_PTO_PIN[] = "5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f";
inline constexpr uint64_t SDMA_METADATA_LINE_BYTES = 64;

// Local inputs for bounded metadata maintenance. The PTO bridge derives
// geometry from public sizeof/offsetof and reads the actual session.
struct SdmaMetadataLayout {
    uint32_t header_bytes{0};
    uint32_t channel_bytes{0};
    uint32_t tail_offset{0};
};

struct SdmaMetadataSession {
    uint64_t workspace_address{0};
    uint64_t group_index{0};
    uint32_t queue_num{0};
    bool valid_sdma{false};
};

struct SdmaMetadataHandoffRange {
    uint64_t address{0};
    uint32_t bytes{0};
    uint32_t lines{0};
};

// The caller supplies rows from its active grant and actual built session.
// This proves bounds/identity only. The retained owner, fixed qNum, supported
// pin and hardware-qualified 64B line isolation remain SDK preflight duties.
inline __aicore__ bool sdma_metadata_handoff_preflight(
    const SdmaMetadataLayout &layout, uint64_t workspace_index, const SchedulerSubmissionWorkspace &workspace,
    const SchedulerSubmissionGrant &grant, const SchedulerSubmissionExecutionIdentity &identity,
    const SdmaMetadataSession &session, SdmaMetadataHandoffRange *out
) {
    if (out == nullptr) return false;
    *out = {};
    if (layout.header_bytes != SDMA_METADATA_LINE_BYTES || layout.channel_bytes != SDMA_METADATA_LINE_BYTES ||
        layout.tail_offset != sizeof(uint32_t) || workspace_index == SCHEDULER_ASYNC_INDEX_NONE ||
        identity.run_epoch == 0 || identity.task_ref < 0 || identity.execution_index == SCHEDULER_ASYNC_INDEX_NONE ||
        identity.dispatch_generation == 0 || identity.subtask_slot == 0 || identity.subtask_slot >= 3 ||
        workspace.qualification != SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN ||
        workspace.qualification_epoch != identity.run_epoch || workspace.reserved0 != 0 || workspace.address == 0 ||
        workspace.address % SDMA_METADATA_LINE_BYTES != 0 || workspace.bytes == 0 ||
        workspace.bytes > UINT64_MAX - workspace.address || workspace.physical_channels == 0 ||
        workspace.physical_channels > SCHEDULER_SUBMISSION_SDMA_CHANNELS || workspace.queue_num == 0 ||
        workspace.queue_num > workspace.physical_channels || grant.reserved != 0 ||
        grant.workspace_index != workspace_index || grant.execution_index != identity.execution_index ||
        grant.group_index >= workspace.physical_channels / workspace.queue_num || !session.valid_sdma ||
        session.workspace_address != workspace.address || session.queue_num != workspace.queue_num ||
        session.group_index != grant.group_index)
        return false;
    for (uint64_t reserved : workspace.reserved)
        if (reserved != 0) return false;
    const uint64_t header_bytes = layout.header_bytes;
    const uint64_t channel_bytes = layout.channel_bytes;
    // Channel/group limits above bound both products before any address is formed.
    const uint64_t array_bytes = static_cast<uint64_t>(workspace.physical_channels) * channel_bytes;
    if (header_bytes > workspace.bytes || array_bytes > workspace.bytes - header_bytes) return false;
    const uint64_t first_channel = grant.group_index * workspace.queue_num;
    const uint64_t offset = header_bytes + first_channel * channel_bytes;
    const uint64_t bytes = static_cast<uint64_t>(workspace.queue_num) * channel_bytes;
    if (offset % SDMA_METADATA_LINE_BYTES != 0 || bytes > workspace.bytes - offset || bytes > UINT32_MAX) return false;
    const uint64_t member = workspace.address + offset + layout.tail_offset;
    out->address = member - member % SDMA_METADATA_LINE_BYTES;
    out->bytes = static_cast<uint32_t>(bytes);
    out->lines = workspace.queue_num;
    return true;
}

// Same DCCI/DDR/PIPE sequence as the existing completion-cache publication
// primitives. Metadata is cleaned and invalidated before another core owns it.
// CPU/sim maintenance is not a hardware cache-coherence qualification.
struct SdmaMetadataHandoffCache {
    inline __aicore__ void flush_range(volatile __gm__ void *address, uint32_t bytes) {
#if defined(__CCE_AICORE__)
        auto *base = static_cast<volatile __gm__ uint8_t *>(address);
        for (uint32_t offset = 0; offset < bytes; offset += SDMA_METADATA_LINE_BYTES)
            dcci(const_cast<__gm__ uint8_t *>(base + offset), SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
        (void)address;
        (void)bytes;
#endif
    }
    inline __aicore__ void finish() {
#if defined(__CCE_AICORE__)
        dsb(DSB_DDR);
        pipe_barrier(PIPE_ALL);
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }
};

// No default publisher: a submission supplies its actual workspace/grant/session
// and a cache implementation. The normal SDK uses SdmaMetadataHandoffCache;
// injected caches are for observing publication order in CPU tests.
template <typename Cache>
class SdmaMetadataHandoffPublisher {
public:
    inline __aicore__ SdmaMetadataHandoffPublisher(
        const SdmaMetadataLayout &layout, uint64_t workspace_index, const SchedulerSubmissionWorkspace &workspace,
        const SchedulerSubmissionGrant &grant, const SchedulerSubmissionExecutionIdentity &identity,
        const SdmaMetadataSession &session, Cache &cache
    ) :
        layout_(layout),
        workspace_index_(workspace_index),
        workspace_(workspace),
        grant_(grant),
        identity_(identity),
        session_(session),
        cache_(cache) {}

    // Must also succeed before async_submission_begin and the hardware call.
    inline __aicore__ bool preflight() const {
        SdmaMetadataHandoffRange range{};
        return sdma_metadata_handoff_preflight(
            layout_, workspace_index_, workspace_, grant_, identity_, session_, &range
        );
    }

    inline __aicore__ bool publish_backend_metadata() {
        SdmaMetadataHandoffRange range{};
        if (!sdma_metadata_handoff_preflight(
                layout_, workspace_index_, workspace_, grant_, identity_, session_, &range
            ))
            return false;
        cache_.flush_range(reinterpret_cast<volatile __gm__ void *>(range.address), range.bytes);
        cache_.finish();
        return true;
    }

    inline __aicore__ void flush_evidence(volatile __gm__ void *address, uint32_t bytes) {
        cache_.flush_range(address, bytes);
    }
    inline __aicore__ void finish() { cache_.finish(); }

private:
    SdmaMetadataLayout layout_;
    uint64_t workspace_index_;
    const SchedulerSubmissionWorkspace &workspace_;
    const SchedulerSubmissionGrant &grant_;
    const SchedulerSubmissionExecutionIdentity &identity_;
    SdmaMetadataSession session_;
    Cache &cache_;
};
