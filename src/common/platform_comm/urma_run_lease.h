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

#include <memory>
#include <vector>
#include "platform_comm/urma_run_binding.h"
#include "platform_comm/urma_launch_permission.h"

enum class UrmaWait { Poll, Wait };

// An exclusive, stopped Run keeps the validated targets until all three queue writes succeed.
struct UrmaQueueReceipt {
    uint64_t cq_tail = 0;
    uint64_t cq_doorbell = 0;
    uint64_t sq_tail = 0;
    uint32_t target_cqe = 0;
    uint32_t target_bb = 0;
    bool cq_published = false;
    bool doorbell_published = false;
};

struct UrmaDomainLease {
    UrmaRunBinding binding{};
    std::shared_ptr<void> owner;
    std::shared_ptr<void> queue_claim;
    UrmaQueueState (*drain_stopped_queues)(void *, UrmaQueueReceipt &, UrmaWait, const UrmaLaunchPermission &){};
    UrmaQueueReceipt pending_receipt{};
};

class UrmaStoppedQueueScan {
public:
    const std::shared_ptr<UrmaScanThread> thread = std::make_shared<UrmaScanThread>();
    virtual ~UrmaStoppedQueueScan() = default;
    virtual UrmaQueueState
    check(const std::vector<UrmaDomainLease> &domains, UrmaWait mode, const UrmaLaunchPermission &permission) = 0;
    virtual void retire() = 0;
};

// Release the Run's operands and domain owners only after every matched workspace drains.
struct UrmaRunLease {
    std::vector<UrmaDomainLease> domains;
    std::shared_ptr<UrmaStoppedQueueScan> stopped_scan{};
    UrmaLaunchPermission launch_permission;

    UrmaQueueState check_stopped_queues(UrmaWait mode) noexcept try {
        if (stopped_scan) return stopped_scan->check(domains, mode, launch_permission);
        for (auto &domain : domains) {
            if (!domain.queue_claim) continue;
            const auto state =
                domain.drain_stopped_queues(domain.owner.get(), domain.pending_receipt, mode, launch_permission);
            if (state != UrmaQueueState::READY) return state;
        }
        return UrmaQueueState::READY;
    } catch (...) {
        // Failure to start or service a Host check leaves the original Run live.
        return UrmaQueueState::PENDING;
    }

    // The chip stopped posting and its queues drained. Late Host copy-back still owns the backing.
    void retire_queues() {
        if (stopped_scan) stopped_scan->retire();
        for (auto &domain : domains)
            domain.queue_claim.reset();
    }
};
