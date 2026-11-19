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

namespace urma_test {
constexpr uint32_t kMaxJetties = 8;
constexpr uint32_t kRecordWords = 112;
constexpr uint32_t kObservationWords = 32;

struct QueueRecord {
    int64_t row;
    int64_t sq_depth;
    int64_t cq_depth;
    int64_t head_before;
    int64_t bb_tail_before;
    int64_t submitted_before;
    int64_t cqe_tail_before;
    int64_t head_after;
    int64_t submitted_after;
    int64_t target_bb;
    int64_t target_cqe;
    int64_t completion_count;
};

struct TransferRecord {
    int64_t status;
    int64_t sender;
    int64_t jetties;
    int64_t peer;
    int64_t rows_per_peer;
    int64_t completion_count;
    int64_t begin_cycles;
    int64_t post_cycles;
    int64_t cycle_hz;
    int64_t workspace_bytes;
    int64_t queue_rows;
    int64_t mode;
    int64_t chunks;
    int64_t repeats;
    int64_t logical_sender;
    int64_t error;
    QueueRecord queues[kMaxJetties];
};

struct QueueTails {
    int64_t completed_bb;
    int64_t completed_cqe;
};

struct ObservationRecord {
    int64_t begin_cycles;
    int64_t status;
    QueueTails queues[kMaxJetties];
    int64_t reserved[14];
};

static_assert(sizeof(TransferRecord) == kRecordWords * sizeof(int64_t));
static_assert(sizeof(ObservationRecord) == kObservationWords * sizeof(int64_t));
static_assert(std::is_trivially_copyable_v<TransferRecord> && std::is_standard_layout_v<TransferRecord>);
static_assert(std::is_trivially_copyable_v<ObservationRecord> && std::is_standard_layout_v<ObservationRecord>);
}  // namespace urma_test
