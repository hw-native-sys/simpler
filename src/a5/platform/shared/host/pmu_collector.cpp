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

/**
 * @file pmu_collector.cpp
 * @brief Host-side PMU collector. The mgmt-thread + buffer-pool machinery
 *        lives in profiling_common::BufferPoolManager parameterized by
 *        PmuModule (host/pmu_collector.h); this file owns the per-buffer
 *        on_buffer_collected callback (CSV output) and the device-side
 *        cross-check. The poll loop itself lives in
 *        profiling_common::ProfilerBase.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. Each
 * PmuBuffer's contents are pulled from device on demand inside
 * ProfilerAlgorithms::process_entry, so on_buffer_collected can read
 * `count` and `records[]` directly off the host shadow.
 */

#include "host/pmu_collector.h"

#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "../../../../common/worker/runtime_c_api.h"

namespace {

int owner_recycled_shard_for_core(int core_index, int thread_count) {
    int cluster_index = core_index / PLATFORM_CORES_PER_BLOCKDIM;
    return cluster_index % thread_count;
}

bool recycled_seed_capacity_is_sufficient(int num_cores, int thread_count, int surplus_per_core, size_t capacity) {
    if (surplus_per_core <= 0) return true;
    std::array<int, PLATFORM_MAX_AICPU_THREADS> per_shard{};
    for (int core = 0; core < num_cores; core++) {
        per_shard[static_cast<size_t>(owner_recycled_shard_for_core(core, thread_count))] += surplus_per_core;
    }

    bool ok = true;
    for (int shard = 0; shard < thread_count; shard++) {
        if (static_cast<size_t>(per_shard[static_cast<size_t>(shard)]) <= capacity) continue;
        LOG_ERROR(
            "PMU recycled seed exceeds lane capacity: shard=%d need=%d capacity=%zu "
            "(num_cores=%d thread_count=%d)",
            shard, per_shard[static_cast<size_t>(shard)], capacity, num_cores, thread_count
        );
        ok = false;
    }
    return ok;
}

}  // namespace

PmuCollector::~PmuCollector() {
    // The writer before the collector threads: it is the one that asks them for
    // a reference release, so joining it second could leave it waiting on
    // threads that are already gone.
    retained_runs_.stop_writer();
    stop();
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int PmuCollector::init(
    int num_cores, int num_threads, const PmuAllocCallback &alloc_cb, PmuRegisterCallback register_cb,
    const PmuFreeCallback &free_cb, int device_id
) {
    if (num_cores <= 0 || num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("PmuCollector::init: invalid arguments");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (num_cores > PLATFORM_MAX_CORES || num_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "PmuCollector::init: dimensions out of range (cores=%d/%d threads=%d/%d)", num_cores, PLATFORM_MAX_CORES,
            num_threads, PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (initialized_) {
        // Already holding this run's device resources. They are not per-run:
        // configuration arrives via begin_run() and the layout is fixed at
        // compile time, so there is nothing here left to re-apply.
        return 0;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_threads);

    num_cores_ = num_cores;
    num_threads_ = num_threads;

    reset_collector_shards();
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
    constexpr int kPmuSurplusPerCore = (PLATFORM_PMU_BUFFERS_PER_CORE > PLATFORM_PMU_SLOT_COUNT) ?
                                           (PLATFORM_PMU_BUFFERS_PER_CORE - PLATFORM_PMU_SLOT_COUNT) :
                                           0;
    if (!recycled_seed_capacity_is_sufficient(
            num_cores, num_threads, kPmuSurplusPerCore, decltype(manager_)::kRecycledQueueCapacity
        )) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Stash callbacks on the base up-front so alloc_paired_buffer sees
    // consistent values during init. shm_host_ stays nullptr until the shm
    // allocation succeeds — start(tf) gates on shm_host_.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_for_ops, profiling_copy_from_device_for_ops,
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: any early return after this point releases the shm
    // region + ring address table + per-core PmuAicoreRing + per-core
    // PmuBuffers. Per-core rings are tracked separately via
    // `guard.add_direct_ptr()` because they're plain alloc_cb allocations
    // (no host shadow / not in dev_to_host_). `guard.commit()` runs on the
    // success path before the trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    // ---- Allocate shared header + buffer-state region ----
    size_t shm_size = calc_pmu_data_size(num_cores);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_paired_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("PmuCollector: failed to allocate PMU shared memory (%zu bytes)", shm_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::memset(shm_host_local, 0, shm_size);
    PmuDataHeader *hdr = get_pmu_header(shm_host_local);
    hdr->event_type = static_cast<uint32_t>(event_type_);
    hdr->num_cores = static_cast<uint32_t>(num_cores);

    // ---- Allocate the per-core ring-address table for AICore. The ring
    // table is filled fully by the host. AICore resolves its own PMU MMIO
    // base at kernel entry from `KernelArgs::regs[get_physical_core_id()]`,
    // so no separate PMU reg-address table is needed.
    // Held in locals and published to the aicore_rings_dev_ / aicore_ring_addrs_*
    // members only after guard.commit() (see end of this function). The table
    // alloc registers in the rollback guard and each ring is added via
    // add_direct_ptr, so a later init failure frees them; assigning the members
    // here would leave them dangling at freed memory.
    std::vector<void *> aicore_rings_dev_local(num_cores, nullptr);
    size_t table_size = static_cast<size_t>(num_cores) * sizeof(uint64_t);
    void *ring_addrs_host_local = nullptr;
    void *ring_addrs_dev_local = alloc_paired_buffer(table_size, &ring_addrs_host_local);
    if (ring_addrs_dev_local == nullptr) {
        LOG_ERROR("PmuCollector: failed to allocate aicore ring address table (%zu bytes)", table_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    std::memset(ring_addrs_host_local, 0, table_size);

    // ---- Allocate per-core PmuBuffers; populate free_queues + recycled pool ----
    const size_t buf_size = sizeof(PmuBuffer);

    for (int c = 0; c < num_cores; c++) {
        PmuBufferState *state = get_pmu_buffer_state(shm_host_local, c);

        // Allocate the per-core stable PmuAicoreRing (no host shadow needed).
        void *ring_dev = alloc_cb(sizeof(PmuAicoreRing));
        if (ring_dev == nullptr) {
            LOG_ERROR("PmuCollector: failed to allocate PmuAicoreRing for core %d", c);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        aicore_rings_dev_local[c] = ring_dev;
        guard.add_direct_ptr(ring_dev);
        state->aicore_ring_ptr = reinterpret_cast<uint64_t>(ring_dev);
        reinterpret_cast<uint64_t *>(ring_addrs_host_local)[c] = reinterpret_cast<uint64_t>(ring_dev);

        for (int b = 0; b < PLATFORM_PMU_BUFFERS_PER_CORE; b++) {
            void *host_ptr = nullptr;
            void *dev_ptr = alloc_paired_buffer(buf_size, &host_ptr);
            if (dev_ptr == nullptr) {
                LOG_ERROR("PmuCollector: failed to allocate PmuBuffer c=%d b=%d", c, b);
                return PTO_RUNTIME_ERR_INTERNAL;
            }

            if (b < PLATFORM_PMU_SLOT_COUNT) {
                uint32_t tail = state->free_queue.tail;
                assert(tail - state->free_queue.head < PLATFORM_PMU_SLOT_COUNT && "free_queue overflow on init");
                state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                state->free_queue.tail = tail + 1;
            } else {
                int shard = owner_recycled_shard_for_core(c, num_threads);
                if (!manager_.push_recycled(0, dev_ptr, shard)) {
                    (void)manager_.retire_unqueued_buffer(0, dev_ptr, shard);
                }
            }
        }
    }

    // Push the populated ring address table to device.
    profiling_copy_to_device(ring_addrs_dev_local, ring_addrs_host_local, table_size);

    // Push the entire initialized shm region (header + BufferStates +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    rebuild_csv_header();

    LOG_INFO(
        "PMU collector initialized: %d cores, %d threads, SHM=0x%lx, CSV=%s (opened on first record)", num_cores,
        num_threads, reinterpret_cast<unsigned long>(shm_dev_local), csv_path_.c_str()
    );
    guard.commit();
    // Publish device-buffer members, the shm/memory context, and the
    // initialized_ flag only after the rollback guard is disarmed. On a failed
    // init they stay at their defaults — initialized_ stays false, so
    // is_initialized() reports false and finalize() never runs against buffers
    // the guard already freed. set_memory_context also publishes shm_host_;
    // start(tf) gates on it, so this is the moment the collector becomes
    // startable. initialized_ is published last, once everything else is set.
    aicore_rings_dev_ = std::move(aicore_rings_dev_local);
    aicore_ring_addrs_dev_ = ring_addrs_dev_local;
    aicore_ring_addrs_host_ = ring_addrs_host_local;
    shm_dev_ = shm_dev_local;
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_for_ops, profiling_copy_from_device_for_ops,
        shm_dev_local, shm_host_local, shm_size, device_id
    );
    initialized_ = true;
    return 0;
}

void PmuCollector::start(const profiling_common::ThreadFactory &thread_factory) {
    if (shm_host_ == nullptr) return;
    // A retaining collector owns no single-run shard state: its files, counters
    // and header belong to an epoch, and dropping them here would reset what a
    // predecessor is still publishing into.
    if (!retains_runs()) {
        reset_collector_shards();
    }
    profiling_common::ProfilerBase<PmuCollector, PmuModule>::start(thread_factory);
}

// ---------------------------------------------------------------------------
// CSV writing
// ---------------------------------------------------------------------------

size_t PmuCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = csv_shard_files_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

bool PmuCollector::close_csv_shards() {
    bool success = true;
    for (size_t shard = 0; shard < csv_shard_files_.size(); shard++) {
        auto &file = csv_shard_files_[shard];
        if (file.is_open()) {
            file.flush();
            if (!file.good()) {
                LOG_ERROR("PmuCollector: failed to flush CSV shard file: %s", csv_shard_paths_[shard].c_str());
                success = false;
            }
            file.close();
            if (file.fail()) {
                LOG_ERROR("PmuCollector: failed to close CSV shard file: %s", csv_shard_paths_[shard].c_str());
                success = false;
            }
        }
    }
    if (!success) {
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
    }
    return success;
}

void PmuCollector::cleanup_csv_shards() {
    for (const auto &path : csv_shard_paths_) {
        if (path.empty()) continue;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
}

std::string PmuCollector::build_csv_header(const PmuEventConfig *events) const {
    std::string header = "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles";
    const PmuEventConfig *evt = events != nullptr ? events : &PMU_EVENTS_A5_PIPE_UTIL;
    for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
        const char *name = evt->counter_names[i];
        if (name == nullptr || name[0] == '\0') {
            continue;
        }
        header += ',';
        header += name;
    }
    header += ",event_type,run_epoch\n";
    return header;
}

std::string PmuCollector::build_csv_header(const FrozenRunConfig &frozen) const {
    return build_csv_header(frozen.events);
}

PmuCollector::FrozenRunConfig PmuCollector::freeze_run_config(PmuEventType event_type) const {
    FrozenRunConfig frozen;
    frozen.event_type = event_type;
    frozen.events = pmu_resolve_event_config_a5(event_type);
    if (frozen.events == nullptr) frozen.events = &PMU_EVENTS_A5_PIPE_UTIL;
    return frozen;
}

void PmuCollector::rebuild_csv_header() {
    // Columns are named by the event config, so this is per-run state: a run
    // that selects a different event type needs different column names.
    csv_header_ = build_csv_header(pmu_resolve_event_config_a5(event_type_));
}

void PmuCollector::begin_run(const std::string &csv_path, PmuEventType event_type) {
    // Before reset_collector_shards(): it rebuilds the shard paths from csv_path_.
    csv_path_ = csv_path;
    event_type_ = event_type;

    reset_collector_shards();
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
    execution_complete_.store(false, std::memory_order_release);
    rebuild_csv_header();
    // The single-run path keeps its behaviour: a field the device would not
    // take is logged by `publish_field` and the run proceeds, exactly as it did
    // before retention existed. Only a retained admission treats that as a
    // refusal, because only it promises that the frozen host configuration and
    // the device's agree.
    (void)publish_run_config(event_type);
}

bool PmuCollector::publish_run_config(PmuEventType event_type) {
    event_type_ = event_type;
    // Before the first init() there is no region; init() writes the event type
    // from the member just set. Afterwards the device needs the new value by
    // another route — one narrow field, not a bulk write-back, so it cannot race
    // the AICPU's own header fields.
    if (shm_host_ == nullptr) return true;
    bool published = true;
    PmuDataHeader *hdr = get_pmu_header(shm_host_);
    hdr->event_type = static_cast<uint32_t>(event_type_);
    wmb();
    // The device reads the event type out of this header to program its
    // counters, so a copy the device did not take leaves it measuring the
    // previous run's event group while the host names this run's columns.
    // `publish_field` returns false in exactly that case.
    published = publish_field(&hdr->event_type, sizeof(hdr->event_type), "event_type") && published;

    // The per-core record counters are producer-side and never reset by the
    // device, so they carry the previous run's totals into this run's
    // reconcile. The three are adjacent, so one write-back per core covers
    // them and leaves the device-owned fields beside them alone.
    //
    // A retaining collector reaches here through an admission, which the
    // predecessor's close precedes: those totals have already been snapshotted
    // into that run's own epoch, so resetting them here destroys nothing.
    static_assert(
        offsetof(PmuBufferState, mismatch_record_count) ==
            offsetof(PmuBufferState, total_record_count) + 2 * sizeof(uint32_t),
        "the three counters must stay adjacent for this single write-back to cover them"
    );
    // The region holds num_cores_ states (calc_pmu_data_size), so that is
    // the bound — a wider loop writes past its end. The runner rebuilds this
    // collector when a run's core count changes, so a resident one is never
    // asked to reset a state it does not own.
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = get_pmu_buffer_state(shm_host_, c);
        state->total_record_count = 0;
        state->dropped_record_count = 0;
        state->mismatch_record_count = 0;
        wmb();
        // A reset the device did not take carries the predecessor's totals into
        // this run's reconcile, so its completeness comparison would be against
        // somebody else's producer count.
        published = publish_field(&state->total_record_count, 3 * sizeof(uint32_t), "record counters") && published;
    }
    return published;
}

void PmuCollector::reset_collector_shards() {
    (void)close_csv_shards();
    cleanup_csv_shards();

    const size_t shard_count = static_cast<size_t>(manager_.shard_count());
    csv_shard_paths_.clear();
    csv_shard_paths_.reserve(shard_count);
    for (size_t shard = 0; shard < shard_count; shard++) {
        csv_shard_paths_.push_back(csv_path_ + ".shard" + std::to_string(shard) + ".tmp");
    }
    csv_shard_files_.clear();
    csv_shard_files_.resize(shard_count);
    collector_counters_.assign(shard_count, {});
    total_collected_ = 0;
    csv_shard_io_failed_.store(false, std::memory_order_relaxed);
    csv_shards_finalized_ = false;
}

bool PmuCollector::ensure_csv_open() {
    if (csv_file_.is_open()) return csv_file_.good();
    csv_file_.clear();
    csv_file_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV file: %s", csv_path_.c_str());
        return false;
    }
    csv_file_ << csv_header_;
    if (!csv_file_.good()) {
        LOG_ERROR("PmuCollector: failed to write CSV header: %s", csv_path_.c_str());
        return false;
    }
    return true;
}

bool PmuCollector::ensure_csv_shard_open(size_t shard) {
    if (shard >= csv_shard_files_.size() || shard >= csv_shard_paths_.size()) {
        LOG_ERROR("PmuCollector: invalid CSV shard index %zu", shard);
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return false;
    }
    auto &file = csv_shard_files_[shard];
    if (file.is_open()) return true;
    file.open(csv_shard_paths_[shard], std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV shard file: %s", csv_shard_paths_[shard].c_str());
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void PmuCollector::append_buffer_to_csv_shard(
    int core_id, int thread_idx, const void *buf_host_ptr, int collector_shard
) {
    const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    const size_t shard = normalize_collector_shard(collector_shard);
    if (!ensure_csv_shard_open(shard)) return;

    const PmuEventConfig *evt = pmu_resolve_event_config_a5(event_type_);
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A5_PIPE_UTIL;
    }

    const uint64_t run_epoch = buf->run_epoch;

    auto &rows = csv_shard_files_[shard];
    for (uint32_t i = 0; i < n; i++) {
        const PmuRecord &r = buf->records[i];
        rows << thread_idx << ',' << core_id << ',';
        rows << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.task_id << std::dec << std::setfill(' ');
        rows << ',' << r.func_id << ',' << static_cast<int>(r.core_type) << ',' << r.pmu_total_cycles;
        for (int k = 0; k < PMU_COUNTER_COUNT_A5; k++) {
            const char *name = evt->counter_names[k];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            rows << ',' << r.pmu_counters[k];
        }
        rows << ',' << static_cast<uint32_t>(event_type_) << ',' << run_epoch << '\n';
    }
    if (!rows.good()) {
        LOG_ERROR("PmuCollector: failed to write CSV shard file: %s", csv_shard_paths_[shard].c_str());
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return;
    }
    collector_counters_[shard].total_collected += n;
}

bool PmuCollector::flush_collector_shards_to_csv() {
    if (csv_shards_finalized_) {
        return true;
    }

    const bool shards_closed = close_csv_shards();
    if (!shards_closed || csv_shard_io_failed_.load(std::memory_order_relaxed)) {
        LOG_ERROR("PmuCollector: CSV shard output failed; preserving shard files for diagnosis");
        return false;
    }

    uint64_t collected_candidate = 0;
    bool has_rows = false;
    for (size_t shard = 0; shard < collector_counters_.size(); shard++) {
        collected_candidate += collector_counters_[shard].total_collected;
        has_rows = has_rows || collector_counters_[shard].total_collected != 0;
    }

    if (has_rows) {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        if (!ensure_csv_open()) {
            if (csv_file_.is_open()) {
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
            }
            return false;
        }
        for (size_t shard = 0; shard < csv_shard_paths_.size(); shard++) {
            if (collector_counters_[shard].total_collected == 0) continue;
            std::ifstream shard_file(csv_shard_paths_[shard], std::ios::binary);
            if (!shard_file.is_open()) {
                LOG_ERROR("PmuCollector: failed to read CSV shard file: %s", csv_shard_paths_[shard].c_str());
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
                return false;
            }
            csv_file_ << shard_file.rdbuf();
            if (shard_file.bad() || !csv_file_.good()) {
                LOG_ERROR("PmuCollector: failed to merge CSV shard file: %s", csv_shard_paths_[shard].c_str());
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
                return false;
            }
        }
        csv_file_.flush();
        if (!csv_file_.good()) {
            LOG_ERROR("PmuCollector: failed to flush merged CSV file: %s", csv_path_.c_str());
            csv_file_.close();
            std::error_code ec;
            std::filesystem::remove(csv_path_, ec);
            return false;
        }
    }
    total_collected_ = collected_candidate;
    cleanup_csv_shards();
    csv_shards_finalized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// ProfilerBase callback
// ---------------------------------------------------------------------------

void PmuCollector::on_buffer_collected(const PmuReadyBufferInfo &info, int collector_shard) {
    // A retaining collector routes by the buffer's own run identity, so a
    // predecessor's late buffer lands in that run's file with that run's
    // columns rather than in whatever run is currently admitting.
    if (retains_runs()) {
        retained_runs_.route_buffer(
            info.host_buffer_ptr, static_cast<int>(info.core_index), static_cast<int>(info.thread_index),
            collector_shard
        );
        return;
    }
    append_buffer_to_csv_shard(
        static_cast<int>(info.core_index), static_cast<int>(info.thread_index), info.host_buffer_ptr, collector_shard
    );
}

// ---------------------------------------------------------------------------
// reconcile_counters: passive sanity-check + device-side cross-check
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device
// flush (pmu_aicpu_flush_buffers) is the only data path: a flush failure
// bumps dropped_record_count and zeroes the buffer's count, but the buffer
// stays the core's — AICPU consumes the free queue and never produces into
// it, so reuse by the next run's init is its only return. Host's job here is
// purely accounting + sanity assertion — recovering would mask AICPU flush
// bugs.

void PmuCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;
    // A retaining collector reconciles per epoch, at its close and in its
    // writer: this run's counters are snapshotted there, its rows are its own
    // shard files, and the drain path's retirement count is consumed once at
    // teardown rather than per run. Reaching here would read a successor's
    // counters and merge in place under a path an epoch owns.
    if (retains_runs()) return;
    report_drain_drops();

    // Pull the latest BufferStates (current_buf_ptr, total/dropped/mismatch
    // counters) before the per-core sanity loop so the cross-check sees
    // post-stop() device state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();
    flush_collector_shards_to_csv();

    // After stop(), a buffer the core still holds must hold no records. Two
    // outcomes leave the pointer set and both are legitimate: a core with
    // nothing to publish, and one whose enqueue failed (which charges dropped
    // and zeroes the count first). A non-zero pointer with a non-zero count is
    // the bug — records AICPU neither delivered nor accounted for.
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = pmu_state(c);
        uint64_t buf_dev = state->current_buf_ptr;
        if (buf_dev == 0) continue;

        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr == nullptr) continue;

        profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(PmuBuffer));
        uint32_t count = reinterpret_cast<const PmuBuffer *>(host_ptr)->count;
        if (count == 0) continue;

        LOG_ERROR(
            "PMU reconcile: core %d has un-flushed buffer (current_buf_ptr=0x%lx, count=%u) after "
            "stop() — device flush failed",
            c, static_cast<unsigned long>(buf_dev), count
        );
    }

    // Cross-check device-side totals against host CSV.  PMU is single-kind
    // (one per-core pool), so reconcile_one is invoked once; the lambda
    // shape matches ChipSwimlaneCollector::reconcile_counters so the two
    // single-arch implementations stay diff-able.
    auto reconcile_one = [&](int unit_count, auto get_state, uint64_t collected, bool optional) {
        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        uint64_t mismatch_device = 0;
        for (int i = 0; i < unit_count; i++) {
            PmuBufferState *state = get_state(i);
            total_device += state->total_record_count;
            dropped_device += state->dropped_record_count;
            mismatch_device += state->mismatch_record_count;
        }

        if (optional && total_device == 0 && collected == 0 && dropped_device == 0 && mismatch_device == 0) {
            return;
        }

        if (dropped_device > 0) {
            LOG_WARN(
                "PMU reconcile: %lu records dropped on device side (free_queue empty or ready_queue full). "
                "Increase PLATFORM_PMU_BUFFERS_PER_CORE / PLATFORM_PMU_READYQUEUE_SIZE if this is frequent.",
                static_cast<unsigned long>(dropped_device)
            );
        }
        if (mismatch_device > 0) {
            LOG_ERROR(
                "PMU reconcile: %lu records lost to AICore staging-slot task_id mismatch — "
                "completion-before-dispatch invariant violated",
                static_cast<unsigned long>(mismatch_device)
            );
        }
        uint64_t accounted = collected + dropped_device + mismatch_device;
        if (accounted != total_device) {
            LOG_WARN(
                "PMU reconcile: record count mismatch (collected=%lu + dropped=%lu + mismatch=%lu != "
                "device_total=%lu, silent_loss=%ld) — AICore/AICPU race",
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(mismatch_device), static_cast<unsigned long>(total_device),
                static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO(
                "PMU reconcile: record counts match (collected=%lu, dropped=%lu, mismatch=%lu, device_total=%lu)",
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(mismatch_device), static_cast<unsigned long>(total_device)
            );
        }
    };

    reconcile_one(
        num_cores_,
        [this](int c) {
            return pmu_state(c);
        },
        total_collected_, /*optional=*/false
    );
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void PmuCollector::finalize(PmuUnregisterCallback unregister_cb, const PmuFreeCallback &free_cb) {
    if (!initialized_) return;

    // Publish whatever the writer can still publish, for a caller that never
    // reached `finish_retained_runs()`. Idempotent, and it reports nothing:
    // the sticky summary is what carries a failure past this point.
    retained_runs_.finish();
    // Then the writer, before the threads it asks for a reference release.
    retained_runs_.stop_writer();

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();
    // Only now: the shards whose references a quarantined epoch could not prove
    // released are joined, so its storage is unreachable by any reader. Its
    // files stay on disk as evidence.
    retained_runs_.release_resources();
    retained_runs_.release_quarantined();
    // A retaining collector has no single-run shard state to merge; the
    // non-retained path's merge is what this is.
    if (!retains_runs()) {
        flush_collector_shards_to_csv();
    }

    if (csv_file_.is_open()) {
        csv_file_.close();
    }

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    };

    // Free buffers still parked in per-core free_queues / current_buf_ptr.
    // Release the device pointer only — the paired host shadow stays in
    // dev_to_host_ and is freed by clear_mappings() below (single source of
    // truth for shadow lifetime, no double-free).
    if (shm_host_ != nullptr) {
        for (int c = 0; c < num_cores_; c++) {
            PmuBufferState *state = pmu_state(c);

            release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
            state->current_buf_ptr = 0;

            rmb();
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_PMU_SLOT_COUNT) queued = PLATFORM_PMU_SLOT_COUNT;
            for (uint32_t i = 0; i < queued; i++) {
                uint32_t slot = (head + i) % PLATFORM_PMU_SLOT_COUNT;
                release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }

    // Release framework-owned device allocations (recycled pool,
    // ready_queue, done_queue). Host shadows are freed by clear_mappings().
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
    });

    // Free per-core PmuAicoreRings (no host shadow paired). The rings were
    // allocated directly via alloc_cb (not alloc_paired_buffer), so no entry
    // exists in dev_to_host_ for them.
    for (auto *ring_dev : aicore_rings_dev_) {
        if (ring_dev != nullptr) {
            release_dev(ring_dev);
        }
    }
    aicore_rings_dev_.clear();

    // Free the per-core ring-address table (device side; host shadow lives
    // in dev_to_host_ and is freed by clear_mappings below).
    if (aicore_ring_addrs_dev_ != nullptr) {
        release_dev(aicore_ring_addrs_dev_);
        aicore_ring_addrs_dev_ = nullptr;
    }
    aicore_ring_addrs_host_ = nullptr;

    // Free shared header region (device only — shadow stays in
    // dev_to_host_ until clear_mappings).
    if (shm_dev_ != nullptr) {
        release_dev(shm_dev_);
        shm_dev_ = nullptr;
    }

    // Free remaining host shadows (per-state buffers + shm region).
    manager_.clear_mappings();

    initialized_ = false;
    (void)close_csv_shards();
    if (csv_shards_finalized_) {
        cleanup_csv_shards();
    }
    csv_shard_paths_.clear();
    csv_shard_paths_.shrink_to_fit();
    csv_shard_files_.clear();
    csv_shard_files_.shrink_to_fit();
    collector_counters_.clear();
    collector_counters_.shrink_to_fit();
    csv_shards_finalized_ = false;
    clear_memory_context();
}

// ---------------------------------------------------------------------------
// Cross-run retention hooks
// ---------------------------------------------------------------------------
//
// The sequencing lives in simpler::dfx::pmu::RetainedRuns; these are the
// arch-specific halves it calls. The device stays serial — what overlaps a
// successor's execution is this collector's host-side receive and the
// background merge of a predecessor's rows.

namespace {

/**
 * The device address of one field inside core `core`'s buffer state.
 *
 * Narrow by construction: a5's host side of the region is a shadow the
 * framework mirrors per tick, and the free-queue cursors beside these counters
 * are written *by the host* drain and replenish threads. Refreshing the whole
 * state — or the whole region — would overwrite their work, so each field this
 * snapshot reads is fetched on its own into scratch.
 */
const void *pmu_dev_field(void *shm_dev, int core, size_t field_offset) {
    return reinterpret_cast<const char *>(shm_dev) + sizeof(PmuDataHeader) +
           static_cast<size_t>(core) * sizeof(PmuBufferState) + field_offset;
}

}  // namespace

simpler::dfx::pmu::RecordProofs PmuCollector::snapshot_run_records(bool device_execution_complete) const {
    simpler::dfx::pmu::RecordProofs proofs;
    proofs.device_execution_complete = device_execution_complete;
    if (shm_host_ == nullptr) {
        // No region to read, so this run's completeness is unknowable rather
        // than zero.
        return proofs;
    }
    // a5 is not SVM: `shm_host_` is a shadow, and a host fence over it says
    // nothing about the device's own counters. So the refresh is a real D2H
    // read, and its absence is an unreadable snapshot rather than a zero one —
    // the same rule `reconcile_counters` has always followed before its
    // cross-check, narrowed to the fields this reads.
    void *shm_dev = manager_.shared_mem_dev();
    if (shm_dev == nullptr || !copy_from_device_) {
        LOG_ERROR("PmuCollector: no device-to-host read for the run close snapshot; counts are unknown");
        return proofs;
    }
    // The three counters are adjacent, so one narrow read per core covers them
    // and touches nothing the host owns. Order matters: the scratch below is
    // indexed by it.
    static_assert(
        offsetof(PmuBufferState, mismatch_record_count) ==
            offsetof(PmuBufferState, total_record_count) + 2 * sizeof(uint32_t),
        "the three counters must stay adjacent for this single narrow read to cover them"
    );
    proofs.snapshot_readable = true;
    for (int c = 0; c < num_cores_; c++) {
        uint32_t counters[3] = {0, 0, 0};
        uint64_t buf_dev = 0;
        if (copy_from_device_(
                counters, pmu_dev_field(shm_dev, c, offsetof(PmuBufferState, total_record_count)), sizeof(counters)
            ) != 0 ||
            copy_from_device_(
                &buf_dev, pmu_dev_field(shm_dev, c, offsetof(PmuBufferState, current_buf_ptr)), sizeof(buf_dev)
            ) != 0) {
            // A read that failed leaves this run's accounting unprovable. It is
            // not folded into the totals: a partial sum would compare as a
            // balanced equation against rows that are not all there.
            LOG_ERROR("PmuCollector: could not read core %d's record counters for the run close snapshot", c);
            proofs.snapshot_readable = false;
            return proofs;
        }
        proofs.total_device += counters[0];
        proofs.dropped_device += counters[1];
        proofs.mismatch_device += counters[2];
        if (buf_dev == 0) continue;
        // Validate the pointer against the manager before handing it to a
        // device read: a buffer this collector does not own is not one to copy
        // from, and its contents are then unknowable.
        if (manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev)) == nullptr) {
            proofs.live_buffer_unreadable = true;
            continue;
        }
        uint32_t count = 0;
        const void *dev_count = reinterpret_cast<const char *>(buf_dev) + offsetof(PmuBuffer, count);
        if (copy_from_device_(&count, dev_count, sizeof(count)) != 0) {
            // The buffer a core still holds cannot be read, so whether it held
            // records is unknowable — which is not the same as zero.
            proofs.live_buffer_unreadable = true;
            continue;
        }
        if (count != 0) proofs.unflushed_records += count;
    }
    return proofs;
}

uint64_t PmuCollector::buffer_run_epoch(const void *buf_host_ptr) const {
    return reinterpret_cast<const PmuBuffer *>(buf_host_ptr)->run_epoch;
}

uint64_t PmuCollector::buffer_record_count(const void *buf_host_ptr) const {
    uint32_t n = reinterpret_cast<const PmuBuffer *>(buf_host_ptr)->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
    }
    return n;
}

uint64_t PmuCollector::write_buffer_rows(
    std::ofstream &out, const FrozenRunConfig &frozen, int core_id, int thread_idx, const void *buf_host_ptr,
    uint64_t buffer_epoch, bool *clamped
) {
    const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
        if (clamped != nullptr) *clamped = true;
    }
    if (n == 0) return 0;
    // Frozen at admission, so a successor's event type cannot reach these rows.
    const PmuEventConfig *evt = frozen.events;
    const uint32_t event_type = static_cast<uint32_t>(frozen.event_type);
    for (uint32_t i = 0; i < n; i++) {
        const PmuRecord &r = buf->records[i];
        out << thread_idx << ',' << core_id << ',';
        out << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.task_id << std::dec << std::setfill(' ');
        out << ',' << r.func_id << ',' << static_cast<int>(r.core_type) << ',' << r.pmu_total_cycles;
        for (int k = 0; k < PMU_COUNTER_COUNT_A5; k++) {
            const char *name = evt->counter_names[k];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            out << ',' << r.pmu_counters[k];
        }
        out << ',' << event_type << ',' << buffer_epoch << '\n';
    }
    return n;
}

void PmuCollector::install_paired_caps() {
    // Bound pool *growth* only: a device buffer and its non-SVM host shadow are
    // allocated together, so the cap is a paired figure and not per side. An
    // occupancy already above it is not reclaimed — `charge_paired` refuses the
    // next block, and the device then drops and charges as it does when a pool
    // runs dry.
    for (int kind = 0; kind < PmuModule::kBufferKinds; kind++) {
        const size_t seeded = manager_.paired_initial(kind);
        const size_t baseline = seeded > sizeof(PmuBuffer) ? seeded : sizeof(PmuBuffer);
        const size_t cap = baseline > SIZE_MAX / 2 ? SIZE_MAX : baseline * 2;
        manager_.set_paired_cap(kind, cap);
    }
}

void PmuCollector::release_paired_caps() {
    for (int kind = 0; kind < PmuModule::kBufferKinds; kind++) {
        manager_.set_paired_cap(kind, 0);
    }
}
