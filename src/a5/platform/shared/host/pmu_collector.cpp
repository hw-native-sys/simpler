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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <ios>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"

PmuCollector::~PmuCollector() { stop(); }

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int PmuCollector::init(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type,
    const PmuAllocCallback &alloc_cb, PmuRegisterCallback register_cb, const PmuFreeCallback &free_cb, int device_id
) {
    if (num_cores <= 0 || num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("PmuCollector::init: invalid arguments");
        return -1;
    }
    if (initialized_) {
        LOG_ERROR("PmuCollector already initialized");
        return -1;
    }

    num_cores_ = num_cores;
    num_threads_ = num_threads;
    event_type_ = event_type;
    csv_path_ = csv_path;

    total_collected_ = 0;
    if (csv_file_.is_open()) {
        csv_file_.close();
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
        return -1;
    }

    std::memset(shm_host_local, 0, shm_size);
    PmuDataHeader *hdr = get_pmu_header(shm_host_local);
    hdr->event_type = static_cast<uint32_t>(event_type);
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
        return -1;
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
            return -1;
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
                return -1;
            }

            if (b < PLATFORM_PMU_SLOT_COUNT) {
                uint32_t tail = state->free_queue.tail;
                assert(tail - state->free_queue.head < PLATFORM_PMU_SLOT_COUNT && "free_queue overflow on init");
                state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                state->free_queue.tail = tail + 1;
            } else {
                manager_.push_recycled(0, dev_ptr);
            }
        }
    }

    // Push the populated ring address table to device.
    profiling_copy_to_device(ring_addrs_dev_local, ring_addrs_host_local, table_size);

    // Push the entire initialized shm region (header + BufferStates +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    // ---- Build CSV header string (file is opened lazily on first record) ----
    {
        std::string header = "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles";
        const PmuEventConfig *evt = pmu_resolve_event_config_a5(event_type);
        if (evt == nullptr) {
            evt = &PMU_EVENTS_A5_PIPE_UTIL;
        }
        for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
            const char *name = evt->counter_names[i];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            header += ',';
            header += name;
        }
        header += ",event_type\n";
        csv_header_ = std::move(header);
    }

    LOG_INFO_V0(
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

// ---------------------------------------------------------------------------
// CSV writing
// ---------------------------------------------------------------------------

void PmuCollector::ensure_csv_open_unlocked() {
    if (csv_file_.is_open()) return;
    csv_file_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV file: %s", csv_path_.c_str());
        return;
    }
    csv_file_ << csv_header_;
}

void PmuCollector::write_buffer_to_csv(int core_id, int thread_idx, const void *buf_host_ptr) {
    const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    std::scoped_lock<std::mutex> lock(csv_mutex_);
    ensure_csv_open_unlocked();
    if (!csv_file_.is_open()) return;
    total_collected_ += n;

    const PmuEventConfig *evt = pmu_resolve_event_config_a5(event_type_);
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A5_PIPE_UTIL;
    }
    for (uint32_t i = 0; i < n; i++) {
        const PmuRecord &r = buf->records[i];
        csv_file_ << thread_idx << ',' << core_id << ',';
        csv_file_ << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.task_id << std::dec
                  << std::setfill(' ');
        csv_file_ << ',' << r.func_id << ',' << static_cast<int>(r.core_type) << ',' << r.pmu_total_cycles;
        for (int k = 0; k < PMU_COUNTER_COUNT_A5; k++) {
            const char *name = evt->counter_names[k];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            csv_file_ << ',' << r.pmu_counters[k];
        }
        csv_file_ << ',' << static_cast<uint32_t>(event_type_) << '\n';
    }
    csv_file_.flush();
}

// ---------------------------------------------------------------------------
// ProfilerBase callback
// ---------------------------------------------------------------------------

void PmuCollector::on_buffer_collected(const PmuReadyBufferInfo &info) {
    write_buffer_to_csv(static_cast<int>(info.core_index), static_cast<int>(info.thread_index), info.host_buffer_ptr);
}

// ---------------------------------------------------------------------------
// reconcile_counters: passive sanity-check + device-side cross-check
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device
// flush (pmu_aicpu_flush_buffers) is the only data path: a flush failure
// must bump dropped_record_count and clear current_buf_ptr on the device
// side. Host's job here is purely accounting + sanity assertion —
// recovering would mask AICPU flush bugs.

void PmuCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;

    // Pull the latest BufferStates (current_buf_ptr, total/dropped/mismatch
    // counters) before the per-core sanity loop so the cross-check sees
    // post-stop() device state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    // After stop(), pmu_aicpu_flush_buffers should have either enqueued the
    // active buffer (success → current_buf_ptr=0) or counted it as dropped
    // and cleared it. A non-zero pointer with non-zero count means records
    // AICPU neither delivered nor accounted for — a device-side flush bug.
    int leftover_active = 0;
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
        leftover_active++;
    }

    if (leftover_active > 0) {
        LOG_ERROR("PMU reconcile: %d core(s) had un-cleared current_buf_ptr — see prior errors", leftover_active);
    }

    // Cross-check device-side totals against host CSV.  PMU is single-kind
    // (one per-core pool), so reconcile_one is invoked once; the lambda
    // shape matches L2SwimlaneCollector::reconcile_counters so the two
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
            LOG_INFO_V0(
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

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

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

    // Release framework-owned buffers (recycled pool, ready_queue,
    // done_queue). release_owned_buffers also frees their host shadows.
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
    clear_memory_context();
    LOG_INFO_V0("PMU collector finalized");
}
