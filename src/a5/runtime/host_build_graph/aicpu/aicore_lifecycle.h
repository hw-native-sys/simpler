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

#include <atomic>
#include <cstdint>

#include "common/core_type.h"
#include "common/device_phase.h"
#include "common/platform_config.h"

class Runtime;
class AicoreLifecycleTestPeer;

class AicoreLifecycle {
public:
    int32_t pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base);
    void handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);
    int32_t validate_discovery();
    int32_t initialize_pmu(Runtime *runtime);
    void publish_lifecycle_start() const;
    int32_t release_partition(Runtime *runtime, int32_t thread_idx, bool start_execution);
    void signal_shutdown_partition(int32_t thread_idx);
    int32_t finish_shutdown_partition(int32_t thread_idx, Runtime *runtime);
    /**
     * Settle every enabled owner's run data, once, if this run proved its cores
     * stopped.
     *
     * Called by the participant that closes this run's completion gate. That
     * arrival says every participant returned from its shutdown partition, not
     * that every partition succeeded, so the proof this reads is the count of
     * cores that acknowledged exit — a run short of it settles nothing and
     * reports unknown.
     *
     * Returns whether this run's owners were settled. False is not an error:
     * it is the run saying its accounting is unknown, which the host reads
     * from the unsettled producers themselves.
     */
    bool settle_run_diagnostics(Runtime *runtime, bool graph_completed);
    /**
     * Whether every core this run brought up has acknowledged exit.
     *
     * The one fact that licenses reading what a core wrote or writing the
     * registers it counts with. Arrival at the run's completion gate does not
     * establish it: a partition whose `platform_retire_aicore_group` reported
     * an unreleased core returns that failure and still arrives.
     */
    bool cores_retired() const {
        return core_count_ > 0 && retired_core_count_.load(std::memory_order_acquire) == core_count_;
    }
    // Called after all shutdown partitions arrive; consumes their saved reports.
    bool retired_work_window(AicoreWorkWindow &out, bool allow_idle) const;
    // This run's discovered cores, for a scheduler mode that configures them
    // itself. Valid once every thread has finished handshake_partition.
    int32_t core_count() const { return core_count_; }
    CoreType core_type(int32_t worker) const { return cores_[worker].core_type; }
    // Whether `worker` reported to this run: only an accepted report fills its
    // register address, so a silent worker must not be told READY.
    bool core_reported(int32_t worker) const { return cores_[worker].reg_addr != 0; }
    uint64_t core_reg_base(int32_t worker) const { return cores_[worker].reg_addr; }
    void retain_peer_dmb_group() { peer_dmb_group_ = true; }
    void deinit();

private:
    friend class AicoreLifecycleTestPeer;
    static constexpr int32_t kMaxWorkers = 108;

    struct CoreState {
        uint64_t reg_addr;
        uint32_t physical_core_id;
        CoreType core_type;
    };

    struct alignas(64) ThreadWorkWindow {
        AicoreWorkWindowAccumulator summary;
    };

    CoreState cores_[kMaxWorkers]{};
    uint32_t physical_core_ids_[kMaxWorkers]{};
    ThreadWorkWindow thread_work_windows_[PLATFORM_MAX_AICPU_THREADS]{};
    std::atomic<bool> handshake_failed_{false};
    // Cores this run proved are no longer executing, counted only after their
    // acknowledged release. A missing startup report supplies no such proof.
    std::atomic<int32_t> retired_core_count_{0};
    bool pmu_initialized_{false};
    bool peer_dmb_group_{false};
    int32_t core_count_{0};
    int32_t aicpu_thread_num_{0};
    uint64_t regs_base_{0};
    volatile uint32_t *lifecycle_notify_{nullptr};
    uint32_t lifecycle_start_{0};
    uint32_t *rejected_exit_{nullptr};
};
