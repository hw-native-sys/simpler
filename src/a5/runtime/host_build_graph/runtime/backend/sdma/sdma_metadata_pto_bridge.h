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

#include <pto/comm/async/sdma/sdma_types.hpp>

#include "sdma_metadata_handoff.h"

// Public types, but a layout contract specific to SDMA_METADATA_PTO_PIN:
// InitializeRuntimeCtx places group g at workspace + sizeof(FlagInfo) +
// g*qNum*sizeof(ChannelInfo). PersistSqTails dirties every qNum channel row.
// This bridge never derives a completion-record address.
inline constexpr SdmaMetadataLayout SDMA_PTO_METADATA_LAYOUT{
    sizeof(pto::comm::sdma::BatchWriteFlagInfo), sizeof(pto::comm::sdma::BatchWriteChannelInfo),
    offsetof(pto::comm::sdma::BatchWriteChannelInfo, sq_tail)
};
static_assert(SDMA_PTO_METADATA_LAYOUT.header_bytes == SDMA_METADATA_LINE_BYTES);
static_assert(SDMA_PTO_METADATA_LAYOUT.channel_bytes == SDMA_METADATA_LINE_BYTES);
static_assert(SDMA_PTO_METADATA_LAYOUT.tail_offset == sizeof(uint32_t));
static_assert(offsetof(pto::comm::sdma::BatchWriteChannelInfo, sq_head) == 0);
static_assert(pto::comm::sdma::kSdmaMaxChannel == SCHEDULER_SUBMISSION_SDMA_CHANNELS);

// The SDK fixes cfg/contextGm at session construction; only rt changes on post.
inline __aicore__ SdmaMetadataSession sdma_metadata_session(const pto::comm::AsyncSession &session) {
    if (!session.valid || session.engine != pto::comm::DmaEngine::SDMA) return {};
    return {
        reinterpret_cast<uint64_t>(session.contextGm), session.cfg.sdma.channelGroupIdx, session.cfg.sdma.queueNum, true
    };
}

inline __aicore__ bool sdma_metadata_handoff_preflight(
    uint64_t workspace_index, const SchedulerSubmissionWorkspace &workspace, const SchedulerSubmissionGrant &grant,
    const SchedulerSubmissionExecutionIdentity &identity, const pto::comm::AsyncSession &session,
    SdmaMetadataHandoffRange *out
) {
    return sdma_metadata_handoff_preflight(
        SDMA_PTO_METADATA_LAYOUT, workspace_index, workspace, grant, identity, sdma_metadata_session(session), out
    );
}
