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

#include "execution_record.h"
#include "scheduler/scheduler_error.h"

static inline __aicore__ void error_identity_probe(__gm__ int64_t *args, uint32_t lane) {
    const uint32_t scenario = static_cast<uint32_t>(args[5]);
    const uint32_t block = scenario == 5 ? lane : static_cast<uint32_t>(get_block_idx(args));
    const uint32_t blocks = static_cast<uint32_t>(args[2]);
    const bool shared = args[3] != 0;
    const uint64_t epoch = static_cast<uint64_t>(args[4]);
    if (block >= blocks) return;
    auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(tensor_data(args[0]) + (shared ? 0 : block) * 64);
    auto *checks = reinterpret_cast<__gm__ ExecutionRecord *>(tensor_data(args[1]));
    auto *check = &checks[block];
    scheduler_gm_fetch_add(check->visits, UINT64_C(1));
    uint64_t errors = 0;
    if (scenario == 5) {
        // One GM Mix group binds these two participants to different physical
        // lanes. This probe-owned word follows, and is not part of, RunControl.
        auto &arrived = *reinterpret_cast<__gm__ uint64_t *>(tensor_data(args[0]) + 48);
        scheduler_gm_fetch_add(arrived, UINT64_C(1));
        uint64_t observed = 0;
        for (uint32_t attempt = 0; attempt < 1000000; ++attempt) {
            observed = scheduler_gm_fetch_add(arrived, UINT64_C(0));
            if (observed == 2) break;
        }
        if (observed != 2) errors |= 128;
    }

    // This synthetic identity belongs to the probe's private error arena, not
    // to the scheduler that executes the probe kernel.
    SchedulerErrorIdentity identity{};
    identity.worker_index = block % 96;
    identity.cluster_lane = lane;
    identity.valid_fields = SCHEDULER_ERROR_HAS_WORKER | SCHEDULER_ERROR_HAS_LANE;
    if (scenario != 1 && scenario != 3) {
        identity.run_epoch = epoch;
        identity.order_key = block * 2;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_EPOCH | SCHEDULER_ERROR_HAS_ORDER;
    }
    if (scenario == 0 || scenario == 1 || scenario == 4 || scenario == 5) {
        identity.block_idx = scenario == 1 ? 0 : block;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK;
    }
    if (scenario == 0 || scenario == 5) {
        identity.group_seq = block + 7;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_GROUP;
    }
    scheduler_record_error(run, block, SchedulerGraphResult::OK, nullptr, nullptr);
    scheduler_record_error(
        run, block, scenario == 1 ? SchedulerGraphResult::INVALID_CALLABLE : SchedulerGraphResult::INVALID_ARGUMENTS,
        nullptr, nullptr,
        scenario == 1 ? SchedulerErrorSite::DISPATCH_INVALID_CALLABLE : SchedulerErrorSite::GROUP_INVALID_STATE,
        &identity
    );
    identity.run_epoch += 1000;
    scheduler_record_error(
        run, block + 100000, SchedulerGraphResult::TIMEOUT, nullptr, nullptr, SchedulerErrorSite::EXIT_WAIT_TIMEOUT,
        &identity
    );

    uint64_t status = 0;
    for (uint32_t attempt = 0; attempt < 1000000; ++attempt) {
        status = scheduler_gm_fetch_add(run->scheduler_error, UINT64_C(0));
        if (status != 0) break;
    }
    const uint64_t winner = scheduler_gm_fetch_add(run->error_task_ref, UINT64_C(0));
    if (winner >= blocks || status != (scenario == 1 ? 7 : 6)) errors |= 1;
    const uint64_t expected_epoch = scenario == 1 || scenario == 3 ? UINT64_MAX : epoch;
    const uint64_t expected_order = scenario == 1 || scenario == 3 ? UINT64_MAX : winner * 2;
    const uint64_t expected_block = scenario == 2 || scenario == 3 ? UINT64_MAX : scenario == 1 ? 0 : winner;
    const uint64_t expected_group = scenario == 0 || scenario == 5 ? winner + 7 : UINT64_MAX;
    const uint64_t expected_fields = static_cast<uint64_t>(args[6]);
    if (scheduler_gm_fetch_add(run->error_run_epoch, UINT64_C(0)) != expected_epoch) errors |= 2;
    if (scheduler_gm_fetch_add(run->error_order_key, UINT64_C(0)) != expected_order) errors |= 4;
    if (scheduler_gm_fetch_add(run->error_block_idx, UINT64_C(0)) != expected_block) errors |= 8;
    if (scheduler_gm_fetch_add(run->error_group_seq, UINT64_C(0)) != expected_group) errors |= 16;
    if (scheduler_gm_fetch_add(run->error_worker_index, UINT64_C(0)) != winner % 96) errors |= 32;
    auto &lane_fields = *reinterpret_cast<__gm__ volatile uint64_t *>(&run->error_cluster_lane);
    const uint64_t expected_lane = scenario == 5 ? winner : lane;
    if (scheduler_gm_fetch_add(lane_fields, UINT64_C(0)) != (expected_fields << 32 | expected_lane)) errors |= 64;
    check->block_plus_one = static_cast<int64_t>(errors);
    check->block_num = static_cast<int64_t>(winner);
    scheduler_publish_cache_line(&check->block_plus_one);
}
