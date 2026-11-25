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
#include "profiling_config.h"
#if SIMPLER_DFX
#include "calendar_pmu.h"
#include "scheduler/profiling_types.h"
#endif
#include "aicore_scheduler_error.h"

#include "scheduler/scheduler_watchdog.h"
#include "scheduler/scheduler_types.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "runtime.h"
#include "host_build_graph/runtime_status.h"
#include "task_image_status.h"
#include "spin_hint.h"

namespace {

uint64_t scheduler_timeout_cycles() {
    const int32_t timeout_ms = get_scheduler_timeout_ms();
    return timeout_ms > 0 ? static_cast<uint64_t>(timeout_ms) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000) :
                            SCHEDULER_TIMEOUT_CYCLES;
}

// One AICPU thread consolidates diagnostics for all cores after retirement.
constexpr int32_t kDiagnosticDumpLane = 0;

#if SIMPLER_DFX
// The staging buffers belong to one execution. Publish before the next replay overwrites them.
bool publish_calendar_swimlane(Runtime *runtime, int32_t worker_count) {
    using simpler::percounter::PERCOUNTER_ORDER_BLOCK_IDX_SHIFT;
    using simpler::percounter::PercounterControl;
    using simpler::percounter::PercounterProfileBuffer;
    using simpler::percounter::PercounterTaskTiming;
    const auto *control =
        reinterpret_cast<const PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    if (control == nullptr || worker_count <= 0 || worker_count > PLATFORM_MAX_CORES) return false;
    cache_invalidate_range(control, sizeof(*control));
    rmb();
    const auto *image = reinterpret_cast<const uint8_t *>(control);
    const auto fits = [bytes = control->image_bytes](uint64_t offset, uint64_t size) {
        return offset >= sizeof(PercounterControl) && offset <= bytes && size <= bytes - offset;
    };
    const uint64_t offset = control->profile_buffers_offset;
    const uint64_t profiles_bytes = static_cast<uint64_t>(worker_count) * sizeof(PercounterProfileBuffer);
    if (offset % alignof(PercounterProfileBuffer) != 0 || !fits(offset, profiles_bytes)) return false;
    const auto *profiles = reinterpret_cast<const PercounterProfileBuffer *>(image + offset);
    cache_invalidate_range(profiles, profiles_bytes);
    rmb();

    bool complete = true;
    int core_ids[PLATFORM_MAX_CORES];
    for (int32_t worker = 0; worker < worker_count; ++worker) {
        core_ids[worker] = worker;
        const auto &profile = profiles[worker];
        const uint32_t count = profile.task_committed;
        const uint64_t bytes = uint64_t{count} * sizeof(PercounterTaskTiming);
        if (profile.worker_id != static_cast<uint32_t>(worker) || profile.physical_core_id < 0 ||
            count > profile.task_capacity || profile.task_records_offset % alignof(PercounterTaskTiming) != 0 ||
            !fits(profile.task_records_offset, bytes)) {
            complete = false;
            continue;
        }
        if (profile.task_dropped != 0) complete = false;
        if (count == 0) continue;
        const auto *tasks = reinterpret_cast<const PercounterTaskTiming *>(image + profile.task_records_offset);
        cache_invalidate_range(tasks, bytes);
        rmb();
        for (uint32_t i = 0; i < count; ++i) {
            const auto &task = tasks[i];
            if (!task.task_token.is_valid() || task.dispatch == 0 || task.start < task.dispatch ||
                task.kernel_end < task.start || task.finish < task.kernel_end) {
                complete = false;
                continue;
            }
            ChipSwimlaneAicoreTaskRecord record{task.start, task.kernel_end, task.task_token, i + 1, {0}};
            record.percounter_block_shape = static_cast<uint32_t>(task.entry >> PERCOUNTER_ORDER_BLOCK_IDX_SHIFT);
            if (!chip_swimlane_aicpu_record_staged_task(
                    worker, kDiagnosticDumpLane, record, task.dispatch, task.finish
                ))
                complete = false;
        }
    }
    chip_swimlane_aicpu_flush_staged(kDiagnosticDumpLane, core_ids, worker_count);
    return complete;
}
#endif

void record_lifecycle_timeout(Runtime *runtime) {
    if (runtime == nullptr || runtime->get_gm_sm_ptr() == nullptr) return;
    auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
    (void)latch_aicore_scheduler_runtime_error(error_code, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
}

}  // namespace

int32_t AicoreLifecycle::pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base) {
    core_count_ = 0;
    peer_dmb_group_ = false;
    aicpu_thread_num_ = aicpu_thread_num;
    regs_base_ = regs_base;
    pmu_initialized_ = false;
    handshake_failed_.store(false, std::memory_order_release);
    retired_core_count_.store(0, std::memory_order_release);
    if (runtime == nullptr || runtime->dev.worker_count <= 0 || runtime->dev.worker_count > kMaxWorkers) {
        LOG_ERROR("Invalid AICore lifecycle worker count");
        return -1;
    }

    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    for (auto &window : thread_work_windows_)
        window = {};
    core_count_ = runtime->dev.worker_count;
    lifecycle_notify_ =
        reinterpret_cast<volatile uint32_t *>(runtime->dev.scheduler_bootstrap.lifecycle_notify_address);
    rejected_exit_ = reinterpret_cast<uint32_t *>(runtime->dev.scheduler_bootstrap.rejected_exit_address);
    if (lifecycle_notify_ != nullptr) {
        const uint32_t prior = __atomic_load_n(lifecycle_notify_, __ATOMIC_RELAXED);
        lifecycle_start_ = (prior + 2U) & AICORE_LIFECYCLE_GENERATION_MASK;
    }
    // The descriptor's gate tail is device-owned and is not copied from the host.
    // Clear prior-run releases before any partition can open a register window.
    std::memset(runtime->get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * core_count_);
    wmb();

    const bool chip_swimlane_enabled = is_chip_swimlane_enabled();
    if (is_dump_args_enabled()) {
        if (get_platform_dump_base() == 0) return -1;
        dump_args_init(aicpu_thread_num, static_cast<DumpArgsLevel>(runtime->dev.dump_args_level));
    }
    if (chip_swimlane_enabled) {
        chip_swimlane_aicpu_init(core_count_, static_cast<ChipSwimlaneLevel>(runtime->dev.chip_swimlane_level));
    }
    return 0;
}

void AicoreLifecycle::handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    if (regs_base_ == 0) {
        handshake_failed_.store(true, std::memory_order_release);
        return;
    }
    Handshake *handshakes = runtime->get_workers();
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(tidx) * core_count_) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(tidx + 1) * core_count_) / nthreads);
    const uint32_t physical_core_count = platform_get_physical_cores_count();
    uint64_t *regs = reinterpret_cast<uint64_t *>(regs_base_);
    // This run's identity, latched from KernelArgs at AICPU entry. Non-zero on a
    // native program launch, so only a report stamped with it is this run's;
    // zero on a kernel/persistent launch, which keeps the aicore_done predicate.
    const uint64_t report_epoch = get_platform_run_result_epoch();

    bool observed[kMaxWorkers]{};
    const uint64_t wait_start = get_sys_cnt_aicpu();
    const uint64_t timeout_cycles = scheduler_timeout_cycles();

    for (int32_t remaining = hi - lo; remaining > 0;) {
        for (int32_t i = lo; i < hi; ++i) {
            if (observed[i]) continue;
            Handshake *handshake = &handshakes[i];
            if (!aicore_report_accepted(handshake, report_epoch)) {
                SPIN_WAIT_HINT();
                continue;
            }
            observed[i] = true;
            --remaining;
            // The accepted epoch/marker load is not an acquire, and the payload
            // loads have no address or data dependency on it, so they need this
            // load-load barrier before reading the report.
            rmb();
            uint32_t physical_core_id = handshake->physical_core_id;
            if (physical_core_id >= physical_core_count) {
                LOG_ERROR(
                    "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                    physical_core_count
                );
                handshake_failed_.store(true, std::memory_order_release);
                continue;
            }
            // Discovery also owns failure cleanup, even if a peer never reports.
            cores_[i] = {regs[physical_core_id], physical_core_id, handshake->core_type};
            physical_core_ids_[i] = physical_core_id;
        }
        if (scheduler_watchdog_expired(wait_start, get_sys_cnt_aicpu(), timeout_cycles)) {
            LOG_ERROR("A5 HBG AICore Scheduler handshake timeout thread=%d remaining=%d", tidx, remaining);
            record_lifecycle_timeout(runtime);
            handshake_failed_.store(true, std::memory_order_release);
            return;
        }
    }
}

int32_t AicoreLifecycle::validate_discovery() {
    bool valid = core_count_ > 0 && !handshake_failed_.load(std::memory_order_acquire);
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].reg_addr == 0) {
            valid = false;
            continue;
        }
        for (int32_t previous = 0; previous < i; ++previous) {
            if (cores_[previous].reg_addr != cores_[i].reg_addr) continue;
            // A register window has one retirement owner. The duplicate report
            // remains unacknowledged and cannot contribute retirement proof.
            LOG_ERROR("Duplicate AICore register window reported by workers %d and %d", previous, i);
            cores_[i].reg_addr = 0;
            valid = false;
            break;
        }
    }
    if (!valid) {
        // Discovery has joined. These workers have no register window and will
        // never receive a COND nonce. Publish their request before broadcast EXIT.
        if (rejected_exit_ != nullptr) {
            for (int32_t i = 0; i < core_count_; ++i) {
                if (!core_reported(i))
                    __atomic_store_n(
                        &rejected_exit_[i], lifecycle_start_ | AICORE_LIFECYCLE_EXIT_BIT, __ATOMIC_RELEASE
                    );
            }
            cache_flush_range(rejected_exit_, core_count_ * sizeof(*rejected_exit_));
            wmb();
        }
        handshake_failed_.store(true, std::memory_order_release);
    }
    return valid ? 0 : -SIMPLER_ERROR_INVALID_ARGS;
}

int32_t AicoreLifecycle::initialize_pmu(Runtime *runtime) {
    if (!is_pmu_enabled()) return 0;
    if (get_platform_pmu_base() == 0) return -SIMPLER_ERROR_INVALID_ARGS;
    pmu_aicpu_init(physical_core_ids_, core_count_, static_cast<PmuEventType>(runtime->dev.pmu_event_type));
    pmu_initialized_ = true;
    return 0;
}

// The calendar leader publishes this after validating all reports and making
// the topology and reset dependency counters visible. Only this leader writes START.
void AicoreLifecycle::publish_lifecycle_start() const {
    publish_aicore_lifecycle_word(lifecycle_notify_, lifecycle_start_);
}

int32_t AicoreLifecycle::release_partition(Runtime *runtime, int32_t thread_idx, bool start_execution) {
    if (core_count_ <= 0 || aicpu_thread_num_ <= 0) return -SIMPLER_ERROR_INVALID_ARGS;
    if (!start_execution) {
        if (lifecycle_notify_ != nullptr) signal_shutdown_partition(thread_idx);
        return finish_shutdown_partition(thread_idx, runtime);
    }
    if (lifecycle_notify_ != nullptr) return 0;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    wmb();
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr != 0) platform_init_aicore_regs(cores_[i].reg_addr);
    }
    return 0;
}

void AicoreLifecycle::signal_shutdown_partition(int32_t thread_idx) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    if (lifecycle_notify_ != nullptr) {
        publish_aicore_lifecycle_word(lifecycle_notify_, lifecycle_start_ | AICORE_LIFECYCLE_EXIT_BIT);
    } else {
        for (int32_t i = lo; i < hi; ++i) {
            if (cores_[i].reg_addr != 0) platform_signal_aicore_exit(cores_[i].reg_addr);
        }
    }
    wmb();
}

int32_t AicoreLifecycle::finish_shutdown_partition(int32_t thread_idx, Runtime *runtime) {
    if (core_count_ <= 0 || aicpu_thread_num_ <= 0 || thread_idx < 0 || thread_idx >= aicpu_thread_num_ ||
        thread_idx >= PLATFORM_MAX_AICPU_THREADS)
        return -SIMPLER_ERROR_INVALID_ARGS;
    // A target can retire its last kernel before another partition has issued
    // its final DMB write. Release the peer group only after every sender ACK.
    if (peer_dmb_group_ && thread_idx != 0) return 0;
    const int32_t lo = peer_dmb_group_ ?
                           0 :
                           static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi =
        peer_dmb_group_ ?
            core_count_ :
            static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    AicoreExitTarget targets[kMaxWorkers];
    int32_t core_ids[kMaxWorkers];
    size_t count = 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        targets[count] = {
            cores_[i].reg_addr, &runtime->get_teardown_gates()[i], lifecycle_notify_ == nullptr, peer_dmb_group_
        };
        core_ids[count] = i;
        ++count;
    }
    bool released[kMaxWorkers]{};
    int32_t rc =
        count == 0 ? 0 : platform_retire_aicore_group(targets, count, platform_aicore_exit_deadline(), released);
    // A missing report cannot prove that its launched core stopped polling GM.
    // Each core contributes retirement only after its own acknowledged release.
    int32_t proved = 0;
    AicoreWorkWindowAccumulator work_window;
    if (count != static_cast<size_t>(hi - lo) && rc == 0) {
        record_lifecycle_timeout(runtime);
        rc = -SIMPLER_ERROR_SCHEDULER_TIMEOUT;
    }
    for (size_t i = 0; i < count; ++i) {
        if (released[i]) {
            Handshake *report = &runtime->get_workers()[core_ids[i]];
            cache_invalidate_range(report, sizeof(*report));
            rmb();
            // This line is already needed for retirement. Save its endpoints
            // here so the finalizer does not fetch every report a second time.
            work_window.add(report->work_start_cycle, report->work_end_cycle);
            aicore_invalidate_retired_report(report);
            cache_flush_range(report, sizeof(*report));
            ++proved;
        } else {
            LOG_ERROR("AICore retirement: core %d not released", core_ids[i]);
            if (rc == 0) rc = -SIMPLER_ERROR_SCHEDULER_TIMEOUT;
        }
    }
    thread_work_windows_[thread_idx].summary = work_window;
    wmb();
    retired_core_count_.fetch_add(proved, std::memory_order_acq_rel);

    int32_t partition_core_ids[kMaxWorkers];
    int32_t partition_count = 0;
    for (int32_t i = lo; i < hi; ++i)
        partition_core_ids[partition_count++] = i;
    if (is_chip_swimlane_enabled()) chip_swimlane_aicpu_flush(thread_idx, partition_core_ids, partition_count);
    return rc;
}

bool AicoreLifecycle::retired_work_window(AicoreWorkWindow &out, bool allow_idle) const {
    out = {};
    if (!cores_retired() || aicpu_thread_num_ <= 0 || aicpu_thread_num_ > PLATFORM_MAX_AICPU_THREADS) return false;
    AicoreWorkWindowAccumulator summary;
    for (int32_t thread = 0; thread < aicpu_thread_num_; ++thread)
        summary.merge(thread_work_windows_[thread].summary);
    return summary.finish(out, allow_idle);
}

bool AicoreLifecycle::settle_run_diagnostics(Runtime *runtime, bool graph_completed) {
    // Arrival at the completion gate says every participant returned from its
    // shutdown partition, not that every partition succeeded: a partition whose
    // `platform_retire_aicore_group` reported an unreleased core returns that
    // failure and still arrives. A core that has not acknowledged exit may
    // still be executing, so nothing below may read what it writes or restore
    // the registers it is counting with. The run's accounting then stays
    // unsettled, which is what the host reads as unknown and withholds for.
    const int32_t retired = retired_core_count_.load(std::memory_order_acquire);
    if (!cores_retired()) {
        LOG_ERROR(
            "A5 HBG AICore Scheduler: %d of %d cores proved retirement; this run's diagnostics are left unsettled "
            "rather than read against a core that may still be executing",
            retired, core_count_
        );
        return false;
    }
    if (is_dump_args_enabled()) {
        for (int32_t thread = 0; thread < aicpu_thread_num_; ++thread) {
            dump_args_flush(thread);
            if (graph_completed) dump_args_close_run(thread);
        }
    }
#if SIMPLER_DFX
    if (graph_completed && is_chip_swimlane_recorded_window() &&
        get_chip_swimlane_level() >= ChipSwimlaneLevel::SCHEDULE_TIMING &&
        !publish_calendar_swimlane(runtime, core_count_)) {
        LOG_ERROR("A5 HBG recorded Swimlane task publication was incomplete");
        return false;
    }
#endif
    if (!is_pmu_enabled()) return true;
    if (!pmu_initialized_) return false;

    // Every core, from the one participant that reaches here: the records are
    // written against each executing core's own buffer, so the publication and
    // the settlement cover the whole run rather than one thread's slice of it.
    int32_t core_ids[kMaxWorkers];
    for (int32_t i = 0; i < core_count_; ++i)
        core_ids[i] = i;

    uint64_t missing = 0;
    bool complete = false;
#if SIMPLER_DFX
    complete = simpler::percounter::collect_calendar_pmu_records(runtime, kDiagnosticDumpLane, &missing);
#else
    (void)runtime;
#endif
    // Records still in a core's active buffer are published before any
    // accounting is settled: without this they would be counted as live at
    // close and lost with the buffer.
    pmu_aicpu_flush_buffers(kDiagnosticDumpLane, core_ids, core_count_);

    // Settling is a claim that what this run published is all of it, and that
    // is a question about the data. A run that did not complete its graph
    // cannot say which subtasks were owed a measurement, and a completed one
    // short of a staged cell lost that measurement — in both cases the
    // producers stay unsettled and the host reports this run's PMU accounting
    // unknown rather than publishing the smaller set that did arrive.
    const bool settled = graph_completed && complete;
    if (settled) {
        pmu_aicpu_close_run_cores(core_ids, core_count_);
    } else {
        LOG_ERROR(
            "PMU: run left unsettled (graph completed=%d, missing staged subtask measurements=%llu); its accounting "
            "is reported unknown rather than published short",
            graph_completed ? 1 : 0, static_cast<unsigned long long>(missing)
        );
    }

    // The register restore is a separate obligation from the accounting, and it
    // is reached on either answer. Its gate is the stop proof above and nothing
    // else: a stopped core's counters must be put back whatever this run could
    // say about its data, or the next run inherits a programmed PMU it never
    // asked for. A core that did not prove it stopped is never restored — that
    // return is above, before anything touches a register.
    pmu_aicpu_finalize(core_ids, core_count_);
    return settled;
}

void AicoreLifecycle::deinit() {
    pmu_initialized_ = false;
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    handshake_failed_.store(false, std::memory_order_release);
    retired_core_count_.store(0, std::memory_order_release);
    core_count_ = 0;
    aicpu_thread_num_ = 0;
    regs_base_ = 0;
    lifecycle_notify_ = nullptr;
    rejected_exit_ = nullptr;
}
