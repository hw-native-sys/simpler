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

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "runtime.h"

// Called once by callback 0 after the whole entry cohort rejected execution.
// No executor has initialized, and affinity TLS may be unavailable. Discover
// only this launch's reports and retire without publishing tasks or READY.
// A missing report/ACK leaves that core to host device recovery; returning an
// error is never proof that its storage may be reused.
inline int32_t abort_aicore_startup(Runtime *runtime, uint64_t regs_base, uint64_t epoch) {
    const int32_t total = runtime->dev.worker_count;
    if (total < 0 || total > PLATFORM_MAX_CORES || regs_base == 0) return -1;
    auto *regs = reinterpret_cast<const uint64_t *>(regs_base);
    auto *gates = runtime->get_teardown_gates();
    for (int32_t i = 0; i < total; ++i)
        gates[i] = {};
    wmb();

    AicoreExitTarget targets[PLATFORM_MAX_CORES];
    int32_t ids[PLATFORM_MAX_CORES];
    bool observed[PLATFORM_MAX_CORES]{};
    size_t count = 0;
    int32_t remaining = total;
    const uint64_t discovery_deadline = platform_aicore_exit_deadline();
    do {
        for (int32_t i = 0; i < total; ++i) {
            auto *report = &runtime->dev.workers[i];
            if (observed[i] || !aicore_report_accepted(report, epoch)) continue;
            rmb();
            observed[i] = true;
            --remaining;
            const uint32_t physical = report->physical_core_id;
            if (physical >= platform_get_physical_cores_count() || regs[physical] == 0) continue;
            // A corrupt duplicate report must not give two retirement owners
            // the same physical register window.
            bool duplicate = false;
            for (size_t j = 0; j < count; ++j)
                duplicate |= targets[j].reg_addr == regs[physical];
            if (duplicate) continue;
            // `i` indexes both runtime->dev.workers and teardown_gates. The
            // core-side exit path acknowledges through teardown_gates[block_idx],
            // so this path relies on the launch block index and the worker report
            // index being the same, as the normal retirement path does.
            targets[count] = {regs[physical], &gates[i]};
            ids[count++] = i;
        }
    } while (remaining != 0 && get_sys_cnt_aicpu() <= discovery_deadline);

    // Unlike normal initialization, this must not publish IDLE: the core must
    // see EXIT before touching any uninitialized task/profiling state.
    for (size_t i = 0; i < count; ++i)
        platform_open_aicore_exit_window(targets[i].reg_addr);
    bool released[PLATFORM_MAX_CORES];
    const int32_t retire_rc = platform_retire_aicore_group(targets, count, platform_aicore_exit_deadline(), released);
    for (size_t i = 0; i < count; ++i) {
        if (!released[i]) continue;
        auto *report = &runtime->dev.workers[ids[i]];
        report->aicore_done = 0;
        report->aicpu_ready = 0;
        cache_flush_range(report, sizeof(*report));
    }
    wmb();
    const int32_t rc = count == static_cast<size_t>(total) && retire_rc == 0 ? 0 : -1;
    if (rc != 0)
        LOG_ERROR("AICore startup abort incomplete: discovered=%zu expected=%d; host recovery required", count, total);
    return rc;
}
