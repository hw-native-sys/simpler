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

// Host-only configuration of the existing workspace owner. SDK_ONLY is an
// explicit native/compiler access contract; it is not inferred from run mode.
enum class CommSdmaAccess : uint32_t { UNKNOWN = 0, LEGACY = 1, SDK_ONLY = 2 };

inline bool comm_sdma_request_valid(CommSdmaAccess access, uint32_t queue_num) {
    return (access == CommSdmaAccess::LEGACY && queue_num == 0) ||
           (access == CommSdmaAccess::SDK_ONLY && queue_num >= 1 && queue_num <= 48);
}

// Lives on CommHandle_, not on domains or runs. Releasing a domain must not
// reset this partition: several domains share the same persistent metadata.
struct CommSdmaOwnerConfiguration {
    CommSdmaAccess access{CommSdmaAccess::UNKNOWN};
    uint32_t queue_num{0};

    bool accepts(CommSdmaAccess requested, uint32_t queues) const {
        return comm_sdma_request_valid(requested, queues) &&
               (access == CommSdmaAccess::UNKNOWN || (access == requested && queue_num == queues));
    }
    bool bind(CommSdmaAccess requested, uint32_t queues) {
        if (!accepts(requested, queues)) return false;
        access = requested;
        queue_num = queues;
        return true;
    }
};

// Explicit trailer of an SDMA-bearing domain's existing IPC announcement.
// Rejected participants still announce and join the same peer exchange, so
// another participant never waits at p2p_ready for a newly rejected owner.
struct CommSdmaAdmissionV1 {
    static constexpr uint64_t MAGIC = 0x53444d4141444d31ULL;
    uint64_t magic{MAGIC};
    uint32_t version{1};
    CommSdmaAccess access{CommSdmaAccess::UNKNOWN};
    uint32_t queue_num{0};
    int32_t status{0};
};

inline bool comm_sdma_admissions_agree(const CommSdmaAdmissionV1 &a, const CommSdmaAdmissionV1 &b) {
    return a.magic == CommSdmaAdmissionV1::MAGIC && b.magic == CommSdmaAdmissionV1::MAGIC && a.version == 1 &&
           b.version == 1 && a.status == 0 && b.status == 0 && comm_sdma_request_valid(a.access, a.queue_num) &&
           a.access == b.access && a.queue_num == b.queue_num;
}

static_assert(sizeof(CommSdmaOwnerConfiguration) == 8);
static_assert(sizeof(CommSdmaAdmissionV1) == 24);
