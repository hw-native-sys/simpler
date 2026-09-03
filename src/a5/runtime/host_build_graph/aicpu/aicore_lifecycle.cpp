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

#include "aicore_lifecycle.h"
#include "aicore_scheduler_error.h"
#include "aicore_scheduler_state.h"

#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "runtime.h"
#include "spin_hint.h"

namespace {

uint64_t resident_scheduler_timeout_cycles() {
    const int32_t timeout_ms = get_scheduler_timeout_ms();
    return timeout_ms > 0 ? static_cast<uint64_t>(timeout_ms) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000) :
                            SCHEDULER_TIMEOUT_CYCLES;
}

bool lifecycle_timing_enabled() { return get_chip_swimlane_level() >= ChipSwimlaneLevel::SCHEDULE_TIMING; }

void record_lifecycle_timeout(Runtime *runtime, SchedulerErrorSite error_site) {
    SchedulerWorkerContext *context = aicore_scheduler_bootstrap_context(runtime);
    if (context == nullptr) return;
    cache_invalidate_range(context, 128);
    SchedulerRunControl *run_control = aicore_scheduler_run_control(context);
    if (run_control == nullptr ||
        !record_aicore_scheduler_runtime_error(run_control, SchedulerGraphResult::TIMEOUT, error_site))
        return;
    cache_flush_range(reinterpret_cast<uint8_t *>(run_control) + offsetof(SchedulerRunControl, error_claimed), 128);
    wmb();
}

}  // namespace

int32_t AicoreLifecycle::pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base) {
    if (runtime == nullptr || runtime->worker_count <= 0 || runtime->worker_count > kMaxWorkers) {
        LOG_ERROR("Invalid AICore lifecycle worker count");
        return -1;
    }

    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    core_count_ = runtime->worker_count;
    aicpu_thread_num_ = aicpu_thread_num;
    regs_base_ = regs_base;
    handshake_failed_.store(false, std::memory_order_release);

    const bool chip_swimlane_enabled = is_chip_swimlane_enabled();
    if (is_pmu_enabled() || is_dump_args_enabled()) {
        LOG_WARN(
            "A5 HBG AICore Scheduler PMU/argument diagnostics are best-effort: artifacts may be absent or incomplete"
        );
    }
    if (chip_swimlane_enabled) chip_swimlane_aicpu_init(core_count_);
    return 0;
}

void AicoreLifecycle::handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    Handshake *handshakes = runtime->workers;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(tidx) * core_count_) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(tidx + 1) * core_count_) / nthreads);
    const uint32_t physical_core_count = platform_get_physical_cores_count();
    uint64_t *regs = reinterpret_cast<uint64_t *>(regs_base_);

    struct ReadyCore {
        int32_t worker_id;
        uint32_t physical_core_id;
        uint64_t reg_addr;
        CoreType core_type;
        uint64_t handshake_observed_cycles;
    };
    ReadyCore ready[kMaxWorkers]{};
    bool observed[kMaxWorkers]{};
    int32_t ready_count = 0;
    const uint64_t wait_start = get_sys_cnt_aicpu();
    const uint64_t timeout_cycles = resident_scheduler_timeout_cycles();

    for (int32_t remaining = hi - lo; remaining > 0;) {
        for (int32_t i = lo; i < hi; ++i) {
            if (observed[i]) continue;
            Handshake *handshake = &handshakes[i];
            cache_invalidate_range(handshake, sizeof(*handshake));
            if (handshake->aicore_done == 0) {
                SPIN_WAIT_HINT();
                continue;
            }
            observed[i] = true;
            --remaining;
            uint32_t physical_core_id = handshake->physical_core_id;
            if (physical_core_id >= physical_core_count) {
                LOG_ERROR(
                    "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                    physical_core_count
                );
                handshake_failed_.store(true, std::memory_order_release);
                continue;
            }
            ready[ready_count++] = {
                i, physical_core_id, regs[physical_core_id], handshake->core_type,
                lifecycle_timing_enabled() ? get_sys_cnt_aicpu() : 0
            };
        }
        if (scheduler_watchdog_expired(wait_start, get_sys_cnt_aicpu(), timeout_cycles)) {
            LOG_ERROR("A5 HBG AICore Scheduler handshake timeout thread=%d remaining=%d", tidx, remaining);
            record_lifecycle_timeout(runtime, SchedulerErrorSite::AICPU_HANDSHAKE_TIMEOUT);
            handshake_failed_.store(true, std::memory_order_release);
            return;
        }
    }

    for (int32_t i = 0; i < ready_count; ++i) {
        const ReadyCore &core = ready[i];
        cores_[core.worker_id] = {core.reg_addr, core.physical_core_id,          core.core_type,
                                  nullptr,       core.handshake_observed_cycles, 0};
        physical_core_ids_[core.worker_id] = core.physical_core_id;
    }
    const uint64_t partition_complete_cycles = lifecycle_timing_enabled() ? get_sys_cnt_aicpu() : 0;
    for (int32_t i = lo; i < hi; ++i)
        cores_[i].handshake_partition_complete_cycles = partition_complete_cycles;
}

int32_t AicoreLifecycle::post_handshake_init(Runtime *runtime) {
    SchedulerWorkerContext *bootstrap_context = aicore_scheduler_bootstrap_context(runtime);
    if (bootstrap_context == nullptr || runtime->host_total_tasks < 0 ||
        handshake_failed_.load(std::memory_order_acquire))
        return -1;
    cache_invalidate_range(bootstrap_context, 256);
    void *scheduler_state_base = aicore_scheduler_state_base(bootstrap_context);
    const bool record_lifecycle_timing = lifecycle_timing_enabled();
    const uint64_t config_start_cycles = record_lifecycle_timing ? get_sys_cnt_aicpu() : 0;

    auto *run_control = aicore_scheduler_run_control(bootstrap_context);
    if (scheduler_state_base == nullptr || run_control == nullptr) return -1;
    auto *contexts =
        scheduler_state_at<SchedulerWorkerContext>(scheduler_state_base, bootstrap_context->worker_contexts_offset);
    auto *lifecycle_traces = scheduler_state_at<AicpuCoreLifecycleTrace>(
        scheduler_state_base, bootstrap_context->aicpu_lifecycle_traces_offset
    );
    cache_invalidate_range(run_control, sizeof(*run_control));
    cache_invalidate_range(contexts, static_cast<size_t>(core_count_) * sizeof(*contexts));
    cache_invalidate_range(lifecycle_traces, static_cast<size_t>(core_count_) * sizeof(*lifecycle_traces));

    int32_t aic_count = 0;
    int32_t aiv_count = 0;
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].core_type == CoreType::AIC) {
            contexts[i].type_rank = aic_count++;
        } else if (cores_[i].core_type == CoreType::AIV) {
            contexts[i].type_rank = aiv_count++;
        } else {
            LOG_ERROR("Core %d reported invalid core type", i);
            return -1;
        }
        contexts[i].core_type = static_cast<int32_t>(cores_[i].core_type);
        contexts[i].physical_core_id = static_cast<int32_t>(cores_[i].physical_core_id);
        contexts[i].active = 0;
        cores_[i].trace = &lifecycle_traces[i];
        lifecycle_traces[i].worker_id = static_cast<uint64_t>(i);
        lifecycle_traces[i].core_type = static_cast<uint64_t>(cores_[i].core_type);
        lifecycle_traces[i].physical_core_id = static_cast<uint64_t>(cores_[i].physical_core_id);
        lifecycle_traces[i].handshake_observed_cycles = cores_[i].handshake_observed_cycles;
        lifecycle_traces[i].handshake_partition_complete_cycles = cores_[i].handshake_partition_complete_cycles;
        lifecycle_traces[i].config_start_cycles = config_start_cycles;
    }
    LOG_INFO("Core discovery complete: %d AIC, %d AIV", aic_count, aiv_count);

    if (aic_count <= 0 || aic_count > static_cast<int32_t>(SCHEDULER_CLUSTER_CAPACITY) ||
        aiv_count != aic_count * PLATFORM_AIV_CORES_PER_BLOCKDIM) {
        LOG_ERROR("A5 HBG AICore scheduler: incomplete cluster topology AIC=%d AIV=%d", aic_count, aiv_count);
        return -1;
    }
    int32_t cluster_workers[SCHEDULER_CLUSTER_CAPACITY][PLATFORM_CORES_PER_BLOCKDIM];
    for (auto &cluster : cluster_workers)
        for (int32_t &worker : cluster)
            worker = -1;
    for (int32_t worker = 0; worker < core_count_; ++worker) {
        const int32_t physical_core_id = static_cast<int32_t>(cores_[worker].physical_core_id);
        SchedulerClusterCoordinate coordinate{-1, -1};
        const bool mapped = scheduler_cluster_coordinate_from_worker(
            worker, cores_[worker].core_type == CoreType::AIC, aic_count, PLATFORM_AIV_CORES_PER_BLOCKDIM, &coordinate
        );
        const int32_t cluster = coordinate.cluster_index;
        const int32_t lane = coordinate.cluster_lane;
        if (!mapped || lane >= PLATFORM_CORES_PER_BLOCKDIM || cluster_workers[cluster][lane] != -1) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: invalid discovered topology worker=%d physical=%d cluster=%d lane=%d", worker,
                physical_core_id, cluster, lane
            );
            return -1;
        }
        cluster_workers[cluster][lane] = worker;
    }
    static_assert(PLATFORM_CORES_PER_BLOCKDIM == 3, "Scheduler selection assumes one AIC and two AIV lanes");
    static_assert(PLATFORM_AIV_CORES_PER_BLOCKDIM == 2, "Scheduler selection assumes two AIV lanes per cluster");
    for (int32_t cluster = 0; cluster < aic_count; ++cluster) {
        for (int32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if (cluster_workers[cluster][lane] < 0) {
                LOG_ERROR("A5 HBG AICore scheduler: cluster %d is missing lane %d", cluster, lane);
                return -1;
            }
        }
    }

    const uint64_t executable_task_count = run_control->expected_task_count - run_control->inline_completed_count;
    const int32_t requested_aic = static_cast<int32_t>(
        std::min<uint64_t>(aic_count, std::max(run_control->aic_worker_demand, run_control->aic_task_count))
    );
    const int32_t requested_aiv = static_cast<int32_t>(
        std::min<uint64_t>(aiv_count, std::max(run_control->aiv_worker_demand, run_control->aiv_task_count))
    );
    uint64_t required_clusters = std::max<uint64_t>(
        requested_aic,
        (static_cast<uint64_t>(requested_aiv) + PLATFORM_AIV_CORES_PER_BLOCKDIM - 1) / PLATFORM_AIV_CORES_PER_BLOCKDIM
    );
    if (executable_task_count != 0) required_clusters = std::max<uint64_t>(required_clusters, 1);
    const int32_t active_clusters = static_cast<int32_t>(std::min<uint64_t>(required_clusters, aic_count));
    if (run_control->aic_worker_demand > static_cast<uint64_t>(aic_count) ||
        run_control->aiv_worker_demand > static_cast<uint64_t>(aiv_count) ||
        (executable_task_count != 0 && active_clusters == 0)) {
        LOG_ERROR(
            "A5 HBG AICore scheduler: topology cannot execute graph (demand AIC=%" PRIu64 " AIV=%" PRIu64
            ", cores AIC=%d AIV=%d)",
            run_control->aic_worker_demand, run_control->aiv_worker_demand, aic_count, aiv_count
        );
        return -1;
    }

    for (int32_t cluster = 0; cluster < aic_count; ++cluster) {
        const int32_t aiv0_worker = cluster_workers[cluster][1];
        const int32_t aiv1_worker = cluster_workers[cluster][2];
        // Do not bind Scheduler ownership to a runtime worker rank. Pick the
        // lower physical AIV in each discovered hardware Cluster.
        const uint64_t scheduler_worker = static_cast<uint64_t>(
            cores_[aiv0_worker].physical_core_id <= cores_[aiv1_worker].physical_core_id ? aiv0_worker : aiv1_worker
        );
        for (int32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            const int32_t worker = cluster_workers[cluster][lane];
            const bool scheduler_lane = static_cast<uint64_t>(worker) == scheduler_worker;
            const bool additional_aiv_lane =
                lane != 0 && !scheduler_lane && cluster * PLATFORM_AIV_CORES_PER_BLOCKDIM + 1 < requested_aiv;
            const bool active_lane = cluster < active_clusters &&
                                     (scheduler_lane || (lane == 0 && cluster < requested_aic) || additional_aiv_lane);
            contexts[worker].active = active_lane ? 1 : 0;
            contexts[worker].cluster_count = static_cast<uint64_t>(active_clusters);
            contexts[worker].cluster_index = static_cast<uint64_t>(cluster);
            contexts[worker].scheduler_index = cluster < active_clusters ? static_cast<uint64_t>(cluster) : UINT64_MAX;
            contexts[worker].scheduler_worker_id = scheduler_worker;
            contexts[worker].is_scheduler = scheduler_lane && cluster < active_clusters ? 1 : 0;
            contexts[worker].inbox_index =
                contexts[worker].is_scheduler != 0 ? static_cast<uint64_t>(cluster) : UINT64_MAX;
            contexts[worker].scheduler_count = static_cast<uint64_t>(active_clusters);
            for (int32_t member = 0; member < PLATFORM_CORES_PER_BLOCKDIM; ++member)
                contexts[worker].cluster_worker_ids[member] = static_cast<uint64_t>(cluster_workers[cluster][member]);
        }
    }
    int32_t active_aic = 0;
    int32_t active_aiv = 0;
    for (int32_t worker = 0; worker < core_count_; ++worker) {
        if (contexts[worker].active == 0) continue;
        if (cores_[worker].core_type == CoreType::AIC) ++active_aic;
        else ++active_aiv;
    }
    run_control->active_worker_count = static_cast<uint64_t>(active_aic) + static_cast<uint64_t>(active_aiv);
    run_control->aic_active_worker_count = static_cast<uint64_t>(active_aic);
    run_control->aiv_active_worker_count = static_cast<uint64_t>(active_aiv);
    run_control->scheduler_count = static_cast<uint64_t>(active_clusters);
    run_control->scheduler_timeout_cycles = resident_scheduler_timeout_cycles();
    if (executable_task_count == 0) {
        run_control->bootstrap_scan_arrived_count = static_cast<uint64_t>(active_clusters);
        run_control->bootstrap_scan_complete = 1;
        run_control->bootstrap_arrived_count = static_cast<uint64_t>(active_clusters);
        run_control->bootstrap_complete = 1;
    }

    const uint64_t topology_complete_cycles = record_lifecycle_timing ? get_sys_cnt_aicpu() : 0;
    for (int32_t i = 0; i < core_count_; ++i)
        lifecycle_traces[i].topology_complete_cycles = topology_complete_cycles;

    if (is_pmu_enabled()) pmu_aicpu_init(physical_core_ids_, core_count_);
    cache_flush_range(contexts, static_cast<size_t>(core_count_) * sizeof(*contexts));
    cache_flush_range(run_control, sizeof(*run_control));
    cache_flush_range(lifecycle_traces, static_cast<size_t>(core_count_) * sizeof(*lifecycle_traces));
    wmb();
    return 0;
}

void AicoreLifecycle::publish_context_partition(Runtime *runtime, int32_t thread_idx) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    Handshake *handshakes = runtime->workers;
    SchedulerWorkerContext *bootstrap_context = aicore_scheduler_bootstrap_context(runtime);
    if (bootstrap_context == nullptr) return;
    cache_invalidate_range(bootstrap_context, 256);
    void *scheduler_state_base = aicore_scheduler_state_base(bootstrap_context);
    for (int32_t i = lo; i < hi; ++i) {
        handshakes[i].task = reinterpret_cast<uint64_t>(scheduler_state_at<SchedulerWorkerContext>(
            scheduler_state_base,
            bootstrap_context->worker_contexts_offset + static_cast<uint64_t>(i) * sizeof(SchedulerWorkerContext)
        ));
        handshakes[i].aicpu_ready = SCHEDULER_RUNTIME_MODE_RESIDENT_READY;
    }
    if (hi > lo) cache_flush_range(&handshakes[lo], static_cast<size_t>(hi - lo) * sizeof(Handshake));
    wmb();
    const uint64_t publish_complete_cycles = lifecycle_timing_enabled() ? get_sys_cnt_aicpu() : 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].trace != nullptr) cores_[i].trace->context_publish_complete_cycles = publish_complete_cycles;
    }
}

int32_t AicoreLifecycle::wait_bootstrap_complete(Runtime *runtime) {
    SchedulerWorkerContext *context = aicore_scheduler_bootstrap_context(runtime);
    if (context == nullptr) return -1;
    cache_invalidate_range(context, 128);
    auto *run_control = aicore_scheduler_run_control(context);
    if (run_control == nullptr) return -1;
    const bool record_lifecycle_timing = lifecycle_timing_enabled();
    const uint64_t wait_start_cycles = record_lifecycle_timing ? get_sys_cnt_aicpu() : 0;
    const uint64_t watchdog_start = get_sys_cnt_aicpu();
    const uint64_t timeout_cycles = resident_scheduler_timeout_cycles();
    uint32_t error_poll_count = 0;
    while (true) {
        cache_invalidate_range(
            reinterpret_cast<uint8_t *>(run_control) + offsetof(SchedulerRunControl, executed_task_count), 128
        );
        if (run_control->bootstrap_complete != 0) break;
        if (++error_poll_count == 64) {
            error_poll_count = 0;
            cache_invalidate_range(
                reinterpret_cast<uint8_t *>(run_control) + offsetof(SchedulerRunControl, error_claimed), 128
            );
            if (run_control->scheduler_error != 0) return -1;
        }
        if (scheduler_watchdog_expired(watchdog_start, get_sys_cnt_aicpu(), timeout_cycles)) {
            LOG_ERROR("%s", "A5 HBG AICore Scheduler bootstrap timeout");
            record_lifecycle_timeout(runtime, SchedulerErrorSite::BOOTSTRAP_COMPLETE_TIMEOUT);
            return -1;
        }
        SPIN_WAIT_HINT();
    }
    const uint64_t complete_cycles = record_lifecycle_timing ? get_sys_cnt_aicpu() : 0;
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].trace == nullptr) continue;
        cores_[i].trace->bootstrap_wait_start_cycles = wait_start_cycles;
        cores_[i].trace->bootstrap_complete_cycles = complete_cycles;
    }
    return 0;
}

int32_t AicoreLifecycle::release_partition(int32_t thread_idx, bool start_execution) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    int32_t rc = 0;
    wmb();
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (start_execution) {
            if (cores_[i].trace != nullptr) {
                cores_[i].trace->aicpu_thread_id = static_cast<uint64_t>(thread_idx);
                cores_[i].trace->register_release_cycles = get_sys_cnt_aicpu();
            }
            platform_init_aicore_regs(cores_[i].reg_addr);
        } else {
            if (platform_deinit_aicore_regs(cores_[i].reg_addr) != 0) rc = -1;
        }
    }
    return rc;
}

void AicoreLifecycle::signal_shutdown_partition(int32_t thread_idx) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (cores_[i].trace != nullptr) {
            cores_[i].trace->aicpu_thread_id = static_cast<uint64_t>(thread_idx);
            cores_[i].trace->exit_signal_cycles = get_sys_cnt_aicpu();
        }
        write_reg(cores_[i].reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    }
}

int32_t AicoreLifecycle::finish_shutdown_partition(int32_t thread_idx, Runtime *runtime) {
    (void)runtime;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    int32_t rc = 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (platform_deinit_aicore_regs(cores_[i].reg_addr) != 0) {
            rc = -1;
        } else if (cores_[i].trace != nullptr) {
            cores_[i].trace->exit_ack_cycles = get_sys_cnt_aicpu();
        }
    }
    rmb();
    if (hi > lo && cores_[lo].trace != nullptr) {
        cache_flush_range(cores_[lo].trace, static_cast<size_t>(hi - lo) * sizeof(AicpuCoreLifecycleTrace));
    }

    int32_t core_ids[kMaxWorkers]{};
    int32_t count = 0;

    for (int32_t i = lo; i < hi; ++i)
        core_ids[count++] = i;

    if (is_chip_swimlane_enabled()) chip_swimlane_aicpu_flush(thread_idx, core_ids, count);
    if (is_pmu_enabled()) pmu_aicpu_finalize(core_ids, count);
    return rc;
}

void AicoreLifecycle::deinit() {
    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    handshake_failed_.store(false, std::memory_order_release);
    core_count_ = 0;
    aicpu_thread_num_ = 0;
    regs_base_ = 0;
}
