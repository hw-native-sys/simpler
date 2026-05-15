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
 * @file l2_perf_collector.cpp
 * @brief Performance data collector implementation. The mgmt-thread +
 *        buffer-pool machinery lives in profiling_common::BufferPoolManager
 *        parameterized by L2PerfModule (host/l2_perf_collector.h); the
 *        poll loop lives in profiling_common::ProfilerBase. This file
 *        owns the per-buffer on_buffer_collected callback and the export
 *        logic.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework's mgmt loop mirrors the shm region per tick; per-buffer
 * payloads (L2PerfBuffer / PhaseBuffer) are pulled on demand inside
 * ProfilerAlgorithms.
 */

#include "host/l2_perf_collector.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"

// =============================================================================
// L2PerfCollector Implementation
// =============================================================================

/**
 * Check if a phase ID belongs to a scheduler phase (vs orchestrator phase).
 * Scheduler phases: SCHED_COMPLETE(0), SCHED_DISPATCH(1), SCHED_SCAN(2), SCHED_IDLE_WAIT(3)
 * Orchestrator phases: ORCH_SYNC(16) through ORCH_SCOPE_END(24)
 */
static bool is_scheduler_phase(AicpuPhaseId id) {
    return static_cast<uint32_t>(id) < static_cast<uint32_t>(AicpuPhaseId::SCHED_PHASE_COUNT);
}

L2PerfCollector::~L2PerfCollector() {
    stop();
    if (shm_host_ != nullptr) {
        LOG_WARN("L2PerfCollector destroyed without finalize()");
    }
}

void *L2PerfCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate buffer (%zu bytes)", size);
        if (host_ptr_out) *host_ptr_out = nullptr;
        return nullptr;
    }

    void *host_ptr = nullptr;
    if (register_cb_ != nullptr) {
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            free_cb_(dev_ptr, user_data_);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
    } else {
        // a5 default: malloc + zero + push zeros to device.
        host_ptr = std::malloc(size);
        if (host_ptr == nullptr) {
            LOG_ERROR("Host shadow alloc failed for %zu bytes", size);
            free_cb_(dev_ptr, user_data_);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
        std::memset(host_ptr, 0, size);
        profiling_copy_to_device(dev_ptr, host_ptr, size);
    }

    if (host_ptr_out) *host_ptr_out = host_ptr;
    // Track dev→host so the framework can resolve_host_ptr() at recycle time.
    manager_.register_mapping(dev_ptr, host_ptr);
    return dev_ptr;
}

int L2PerfCollector::initialize(
    int num_aicore, int device_id, L2PerfAllocCallback alloc_cb, L2PerfRegisterCallback register_cb,
    L2PerfFreeCallback free_cb, void *user_data, const std::string &output_prefix
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("L2PerfCollector already initialized");
        return -1;
    }

    LOG_INFO_V0("Initializing performance profiling");

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }

    num_aicore_ = num_aicore;
    output_prefix_ = output_prefix;
    total_perf_collected_ = 0;
    total_phase_collected_ = 0;

    // Stash the memory context on the base up-front so alloc_single_buffer
    // (which reads alloc_cb_/register_cb_/free_cb_/user_data_/device_id_)
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — that nullptr guard makes a post-failure
    // start(tf) a no-op.
    set_memory_context(
        alloc_cb, register_cb, free_cb, user_data, /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // Step 1: Calculate shared memory size (slot arrays only, no actual buffers)
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:        %d", num_aicore);
    LOG_DEBUG("  Header size:            %zu bytes", sizeof(L2PerfDataHeader));
    LOG_DEBUG("  L2PerfBufferState size: %zu bytes each", sizeof(L2PerfBufferState));
    LOG_DEBUG("  PhaseBufferState size:  %zu bytes each", sizeof(PhaseBufferState));
    LOG_DEBUG("  Total shared memory:    %zu bytes (%zu KB)", total_size, total_size / 1024);

    // Step 2: Allocate shared memory + paired host shadow
    void *perf_host_ptr = nullptr;
    void *perf_dev_ptr = alloc_single_buffer(total_size, &perf_host_ptr);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: dev=%p host=%p", perf_dev_ptr, perf_host_ptr);

    // Step 3: Initialize header on host shadow
    std::memset(perf_host_ptr, 0, total_size);
    L2PerfDataHeader *header = get_l2_perf_header(perf_host_ptr);
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }
    header->num_cores = num_aicore;

    LOG_DEBUG("Initialized L2PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 4: Allocate per-core stable L2PerfAicoreRings + the address-table
    // buffer. Rings are allocated once and never rotated; AICore writes into
    // them at task time, AICPU reads at FIN time. The address-table mirrors
    // each ring's device pointer so the AICore-side `KernelArgs` machinery
    // can index by `block_idx` without needing to walk SHM.
    aicore_rings_dev_.assign(num_aicore, nullptr);
    void *table_host_ptr = nullptr;
    size_t table_size = static_cast<size_t>(num_aicore) * sizeof(uint64_t);
    void *table_dev_ptr = alloc_single_buffer(table_size, &table_host_ptr);
    if (table_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate L2Perf aicore ring address table (%zu bytes)", table_size);
        return -1;
    }
    std::memset(table_host_ptr, 0, table_size);
    aicore_ring_addrs_dev_ = table_dev_ptr;
    aicore_ring_addrs_host_ = table_host_ptr;

    // Step 4b: Initialize L2PerfBufferStates — 1 buffer/core in free_queue, rest to recycled pool.
    for (int i = 0; i < num_aicore; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(perf_host_ptr, i);
        std::memset(state, 0, sizeof(L2PerfBufferState));

        // Allocate the per-core staging ring (no host shadow needed: AICore
        // writes, AICPU reads — host never touches the ring directly).
        void *ring_dev = alloc_cb(sizeof(L2PerfAicoreRing), user_data);
        if (ring_dev == nullptr) {
            LOG_ERROR("Failed to allocate L2PerfAicoreRing for core %d", i);
            return -1;
        }
        aicore_rings_dev_[i] = ring_dev;
        state->aicore_ring_ptr = reinterpret_cast<uint64_t>(ring_dev);
        reinterpret_cast<uint64_t *>(table_host_ptr)[i] = reinterpret_cast<uint64_t>(ring_dev);

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2PerfBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2PerfBuffer for core %d, buffer %d", i, s);
                return -1;
            }

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::PERF_RECORD), dev_buf_ptr);
            }
        }
        state->free_queue.tail = 1;
    }
    LOG_DEBUG(
        "Initialized %d L2PerfBufferStates: 1 buffer/core, %d in recycled pool", num_aicore,
        num_aicore * (PLATFORM_PROF_BUFFERS_PER_CORE - 1)
    );

    // Push the populated address table to device.
    profiling_copy_to_device(table_dev_ptr, table_host_ptr, table_size);

    // Step 5: Initialize PhaseBufferStates — 1 buffer/thread in free_queue, rest to recycled pool.
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
        std::memset(state, 0, sizeof(PhaseBufferState));

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_THREAD; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(PhaseBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PhaseBuffer for thread %d, buffer %d", t, s);
                return -1;
            }

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::PHASE), dev_buf_ptr);
            }
        }
        state->free_queue.tail = 1;
    }
    LOG_DEBUG(
        "Initialized %d PhaseBufferStates: 1 buffer/thread, %d in recycled pool", num_phase_threads,
        num_phase_threads * (PLATFORM_PROF_BUFFERS_PER_THREAD - 1)
    );

    // Step 6: Push the initialized shm region (header + BufferStates +
    // free_queue contents) to device.
    profiling_copy_to_device(perf_dev_ptr, perf_host_ptr, total_size);

    // Step 7: Publish shm pointers on the base now that the region is ready.
    perf_shared_mem_dev_ = perf_dev_ptr;
    set_memory_context(alloc_cb, register_cb, free_cb, user_data, perf_dev_ptr, perf_host_ptr, total_size, device_id);

    collected_perf_records_.assign(num_aicore_, {});
    collected_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});

    LOG_DEBUG("L2 perf device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));
    LOG_INFO_V0("Performance profiling initialized (dynamic buffer mode)");
    return 0;
}

// ---------------------------------------------------------------------------
// ProfilerBase callbacks
// ---------------------------------------------------------------------------

void L2PerfCollector::copy_perf_buffer(const ReadyBufferInfo &info) {
    L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > PLATFORM_PROF_BUFFER_SIZE) {
        count = PLATFORM_PROF_BUFFER_SIZE;
    }
    uint32_t core_index = info.index;
    if (core_index < static_cast<uint32_t>(num_aicore_)) {
        for (uint32_t i = 0; i < count; i++) {
            collected_perf_records_[core_index].push_back(buf->records[i]);
        }
        total_perf_collected_ += count;
    }
}

void L2PerfCollector::copy_phase_buffer(const ReadyBufferInfo &info) {
    PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(info.host_buffer_ptr);
    rmb();
    uint32_t count = buf->count;
    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
    }
    uint32_t tidx = info.index;
    if (tidx < collected_phase_records_.size()) {
        for (uint32_t i = 0; i < count; i++) {
            collected_phase_records_[tidx].push_back(buf->records[i]);
        }
        total_phase_collected_ += count;
        if (count > 0) {
            has_phase_data_ = true;
        }
    }
}

void L2PerfCollector::on_buffer_collected(const ReadyBufferInfo &info) {
    if (info.type == ProfBufferType::PERF_RECORD) {
        copy_perf_buffer(info);
    } else {
        copy_phase_buffer(info);
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters / read_phase_header_metadata
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device flush
// is the only data path: a flush failure must bump dropped_record_count and
// clear current_buf_ptr on the device side. Host's job here is purely
// accounting + sanity check.
//
// L2PerfBufferState now tracks total / dropped / mismatch counters — same
// three-bucket accounting as PMU and a2a3. The cross-check equation
// (collected + dropped + mismatch == device_total) is enforced per pool
// (PERF + PHASE). Empty PHASE pools (runtime emits no phase records) are
// skipped via the `optional` flag.

void L2PerfCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;

    // Pull the latest BufferStates (current_buf_ptr) before the per-unit
    // sanity loop so leftovers reflect post-stop() device state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    // After stop(), AICPU's per-thread flush hooks
    // (l2_perf_aicpu_flush_buffers / l2_perf_aicpu_flush_phase_buffers)
    // should have either enqueued the active buffer (success →
    // current_buf_ptr=0) or cleared it on enqueue failure. A non-zero
    // pointer with non-zero count means records AICPU neither delivered
    // nor cleared — a device-side flush bug. Empty buffers (count=0,
    // never written) are fine; AICPU's flush legitimately skips them.
    int leftover_active = 0;
    for (int i = 0; i < num_aicore_; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(shm_host_, i);
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) continue;
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
        if (host_ptr == nullptr) continue;
        profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_ptr), sizeof(L2PerfBuffer));
        uint32_t count = reinterpret_cast<L2PerfBuffer *>(host_ptr)->count;
        if (count == 0) continue;
        LOG_ERROR(
            "L2Perf reconcile: core %d has un-flushed PERF buffer (current_buf_ptr=0x%lx, count=%u) "
            "after stop() — device flush failed",
            i, static_cast<unsigned long>(buf_ptr), count
        );
        leftover_active++;
    }

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        PhaseBufferState *state = get_phase_buffer_state(shm_host_, num_aicore_, t);
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) continue;
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
        if (host_ptr == nullptr) continue;
        profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_ptr), sizeof(PhaseBuffer));
        uint32_t count = reinterpret_cast<PhaseBuffer *>(host_ptr)->count;
        if (count == 0) continue;
        LOG_ERROR(
            "L2Perf reconcile: thread %d has un-flushed PHASE buffer (current_buf_ptr=0x%lx, count=%u) "
            "after stop() — device flush failed",
            t, static_cast<unsigned long>(buf_ptr), count
        );
        leftover_active++;
    }

    if (leftover_active > 0) {
        LOG_ERROR("L2Perf reconcile: %d unit(s) had un-cleared current_buf_ptr — see prior errors", leftover_active);
    }

    // Cross-check device-side totals against host CSV.  PERF and PHASE
    // each have their own pool of buffer states with the same accounting
    // shape: total_record_count = collected + dropped + mismatch.
    auto reconcile_one = [&](const char *kind, const char *unit_name, int unit_count, auto get_state,
                             uint64_t collected, bool optional) {
        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        uint64_t mismatch_device = 0;
        for (int i = 0; i < unit_count; i++) {
            L2PerfBufferState *state = get_state(i);
            total_device += state->total_record_count;
            dropped_device += state->dropped_record_count;
            mismatch_device += state->mismatch_record_count;
        }

        if (optional && total_device == 0 && collected == 0 && dropped_device == 0 && mismatch_device == 0) {
            return;
        }

        if (dropped_device > 0) {
            LOG_WARN(
                "L2Perf reconcile: %lu %s records dropped on device side (buffer full / "
                "ready_queue full / late FIN after flush).",
                static_cast<unsigned long>(dropped_device), kind
            );
        }
        if (mismatch_device > 0) {
            LOG_ERROR(
                "L2Perf reconcile: %lu %s records lost to AICore staging-slot task_id mismatch — "
                "completion-before-dispatch invariant violated",
                static_cast<unsigned long>(mismatch_device), kind
            );
        }
        uint64_t accounted = collected + dropped_device + mismatch_device;
        if (accounted != total_device) {
            LOG_WARN(
                "L2Perf reconcile: %s count mismatch (collected=%lu + dropped=%lu + mismatch=%lu != "
                "device_total=%lu, silent_loss=%ld)",
                kind, static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(mismatch_device), static_cast<unsigned long>(total_device),
                static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO_V0(
                "L2Perf reconcile: %s counts match (collected=%lu, dropped=%lu, mismatch=%lu, device_total=%lu)", kind,
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(mismatch_device), static_cast<unsigned long>(total_device)
            );
        }
        (void)unit_name;
    };

    reconcile_one(
        "PERF", "core", num_aicore_,
        [this](int i) {
            return get_perf_buffer_state(shm_host_, i);
        },
        total_perf_collected_, /*optional=*/false
    );
    reconcile_one(
        "PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int i) {
            return get_phase_buffer_state(shm_host_, num_aicore_, i);
        },
        total_phase_collected_, /*optional=*/true
    );
}

void L2PerfCollector::read_phase_header_metadata() {
    if (shm_host_ == nullptr) return;

    // Pull the AicpuPhaseHeader portion from device (the mgmt loop's final
    // mirror covered it, but stop() may have raced with a final orch_summary
    // write — re-mirror to be safe).
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    AicpuPhaseHeader *phase_header = get_phase_header(shm_host_, num_aicore_);

    if (phase_header->magic != AICPU_PHASE_MAGIC) {
        LOG_INFO_V0(
            "No phase profiling data found (magic mismatch: 0x%x vs 0x%x)", phase_header->magic, AICPU_PHASE_MAGIC
        );
        return;
    }

    int num_sched_threads = phase_header->num_sched_threads;
    if (num_sched_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Invalid num_sched_threads %d from shared memory (max=%d)", num_sched_threads, PLATFORM_MAX_AICPU_THREADS
        );
        return;
    }
    LOG_INFO_V0("Collecting phase metadata: %d scheduler threads", num_sched_threads);

    // Per-thread breakdown of records already collected via the buffer pipeline.
    for (size_t t = 0; t < collected_phase_records_.size(); t++) {
        if (!collected_phase_records_[t].empty()) {
            size_t sched_count = 0, orch_count = 0;
            for (const auto &r : collected_phase_records_[t]) {
                if (is_scheduler_phase(r.phase_id)) sched_count++;
                else orch_count++;
            }
            LOG_INFO_V0(
                "  Thread %zu: %zu records (sched=%zu, orch=%zu)", t, collected_phase_records_[t].size(), sched_count,
                orch_count
            );
        }
    }

    collected_orch_summary_ = phase_header->orch_summary;
    bool orch_valid = (collected_orch_summary_.magic == AICPU_PHASE_MAGIC);
    if (orch_valid) {
        LOG_INFO_V0(
            "  Orchestrator: %" PRId64 " tasks, %.3fus", static_cast<int64_t>(collected_orch_summary_.submit_count),
            cycles_to_us(collected_orch_summary_.end_time - collected_orch_summary_.start_time)
        );
    } else {
        LOG_INFO_V0("  Orchestrator: no summary data");
    }

    bool has_accumulated = has_phase_data_;
    if (!has_accumulated) {
        for (const auto &v : collected_phase_records_) {
            if (!v.empty()) {
                has_accumulated = true;
                break;
            }
        }
    }
    has_phase_data_ = (orch_valid || has_accumulated);

    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread, phase_header->core_to_thread + num_cores);
        LOG_INFO_V0("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO_V0("Phase metadata collection complete: orch_summary=%s", orch_valid ? "yes" : "no");
}

// ---------------------------------------------------------------------------
// export_swimlane_json
// ---------------------------------------------------------------------------

int L2PerfCollector::export_swimlane_json() {
    bool has_any_records = false;
    for (const auto &core_records : collected_perf_records_) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records) {
        LOG_WARN("Warning: No performance data to export.");
        return -1;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return -1;
    }

    struct TaggedRecord {
        const L2PerfRecord *record;
        uint32_t core_id;
    };
    std::vector<TaggedRecord> tagged_records;
    size_t total_records = 0;
    for (const auto &core_records : collected_perf_records_) {
        total_records += core_records.size();
    }
    tagged_records.reserve(total_records);
    for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
        for (const auto &record : collected_perf_records_[core_idx]) {
            tagged_records.push_back({&record, static_cast<uint32_t>(core_idx)});
        }
    }

    std::sort(tagged_records.begin(), tagged_records.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
        return a.record->task_id < b.record->task_id;
    });

    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto &tagged : tagged_records) {
        if (tagged.record->start_time < base_time_cycles) {
            base_time_cycles = tagged.record->start_time;
        }
        if (tagged.record->dispatch_time > 0 && tagged.record->dispatch_time < base_time_cycles) {
            base_time_cycles = tagged.record->dispatch_time;
        }
    }

    if (has_phase_data_) {
        for (const auto &thread_records : collected_phase_records_) {
            for (const auto &pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC && collected_orch_summary_.start_time > 0 &&
            collected_orch_summary_.start_time < base_time_cycles) {
            base_time_cycles = collected_orch_summary_.start_time;
        }
    }

    std::string filepath = output_prefix_ + "/l2_perf_records.json";

    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    int version = has_phase_data_ ? 2 : 1;
    outfile << "{\n";
    outfile << "  \"version\": " << version << ",\n";
    outfile << "  \"tasks\": [\n";

    for (size_t i = 0; i < tagged_records.size(); ++i) {
        const auto &tagged = tagged_records[i];
        const auto &record = *tagged.record;

        double start_us = cycles_to_us(record.start_time - base_time_cycles);
        double end_us = cycles_to_us(record.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        const char *core_type_str = (record.core_type == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << record.task_id << ",\n";
        outfile << "      \"func_id\": " << record.func_id << ",\n";
        outfile << "      \"core_id\": " << tagged.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"ring_id\": " << static_cast<int>(record.task_id >> 32) << ",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << ",\n";
        outfile << "      \"fanout\": [";
        int safe_fanout_count =
            (record.fanout_count >= 0 && record.fanout_count <= RUNTIME_MAX_FANOUT) ? record.fanout_count : 0;
        for (int j = 0; j < safe_fanout_count; ++j) {
            outfile << record.fanout[j];
            if (j < safe_fanout_count - 1) {
                outfile << ", ";
            }
        }
        outfile << "],\n";
        outfile << "      \"fanout_count\": " << record.fanout_count << "\n";
        outfile << "    }";
        if (i < tagged_records.size() - 1) {
            outfile << ",";
        }
        outfile << "\n";
    }
    outfile << "  ]";

    if (has_phase_data_) {
        auto sched_phase_name = [](AicpuPhaseId id) -> const char * {
            switch (id) {
            case AicpuPhaseId::SCHED_COMPLETE:
                return "complete";
            case AicpuPhaseId::SCHED_DISPATCH:
                return "dispatch";
            case AicpuPhaseId::SCHED_SCAN:
                return "scan";
            case AicpuPhaseId::SCHED_IDLE_WAIT:
                return "idle";
            default:
                return "unknown";
            }
        };

        auto orch_phase_name = [](AicpuPhaseId id) -> const char * {
            switch (id) {
            case AicpuPhaseId::ORCH_SYNC:
                return "orch_sync";
            case AicpuPhaseId::ORCH_ALLOC:
                return "orch_alloc";
            case AicpuPhaseId::ORCH_PARAMS:
                return "orch_params";
            case AicpuPhaseId::ORCH_LOOKUP:
                return "orch_lookup";
            case AicpuPhaseId::ORCH_HEAP:
                return "orch_heap";
            case AicpuPhaseId::ORCH_INSERT:
                return "orch_insert";
            case AicpuPhaseId::ORCH_FANIN:
                return "orch_fanin";
            case AicpuPhaseId::ORCH_FINALIZE:
                return "orch_finalize";
            case AicpuPhaseId::ORCH_SCOPE_END:
                return "orch_scope_end";
            default:
                return "unknown";
            }
        };

        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_phase_records_.size(); t++) {
            outfile << "    [\n";
            bool first = true;
            for (const auto &pr : collected_phase_records_[t]) {
                if (!is_scheduler_phase(pr.phase_id)) continue;
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                if (!first) outfile << ",\n";
                outfile << "      {\"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ", \"phase\": \""
                        << sched_phase_name(pr.phase_id) << "\""
                        << ", \"loop_iter\": " << pr.loop_iter << ", \"tasks_processed\": " << pr.tasks_processed
                        << "}";
                first = false;
            }
            if (!first) outfile << "\n";
            outfile << "    ]";
            if (t < collected_phase_records_.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]";

        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC) {
            double orch_start_us = cycles_to_us(collected_orch_summary_.start_time - base_time_cycles);
            double orch_end_us = cycles_to_us(collected_orch_summary_.end_time - base_time_cycles);

            outfile << ",\n  \"aicpu_orchestrator\": {\n";
            outfile << "    \"start_time_us\": " << std::fixed << std::setprecision(3) << orch_start_us << ",\n";
            outfile << "    \"end_time_us\": " << std::fixed << std::setprecision(3) << orch_end_us << ",\n";
            outfile << "    \"submit_count\": " << collected_orch_summary_.submit_count << ",\n";
            outfile << "    \"phase_us\": {\n";
            outfile << "      \"sync\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.sync_cycle) << ",\n";
            outfile << "      \"alloc\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.alloc_cycle) << ",\n";
            outfile << "      \"params\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.args_cycle) << ",\n";
            outfile << "      \"lookup\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.lookup_cycle) << ",\n";
            outfile << "      \"heap\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.heap_cycle) << ",\n";
            outfile << "      \"insert\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.insert_cycle) << ",\n";
            outfile << "      \"fanin\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.fanin_cycle) << ",\n";
            outfile << "      \"scope_end\": " << std::fixed << std::setprecision(3)
                    << cycles_to_us(collected_orch_summary_.scope_end_cycle) << "\n";
            outfile << "    }\n";
            outfile << "  }";
        }

        bool has_orch_phases = false;
        for (const auto &v : collected_phase_records_) {
            for (const auto &r : v) {
                if (!is_scheduler_phase(r.phase_id)) {
                    has_orch_phases = true;
                    break;
                }
            }
            if (has_orch_phases) break;
        }
        if (has_orch_phases) {
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < collected_phase_records_.size(); t++) {
                outfile << "    [\n";
                bool first = true;
                for (const auto &pr : collected_phase_records_[t]) {
                    if (is_scheduler_phase(pr.phase_id)) continue;
                    double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                    double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                    if (!first) outfile << ",\n";
                    outfile << "      {\"phase\": \"" << orch_phase_name(pr.phase_id) << "\""
                            << ", \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                            << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                            << ", \"submit_idx\": " << pr.loop_iter << ", \"task_id\": " << pr.task_id << "}";
                    first = false;
                }
                if (!first) outfile << "\n";
                outfile << "    ]";
                if (t < collected_phase_records_.size() - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
    }

    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO_V0("=== JSON Export Complete ===");
    LOG_INFO_V0("File: %s", filepath.c_str());
    LOG_INFO_V0("Records: %u", record_count);

    return 0;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

int L2PerfCollector::finalize(L2PerfUnregisterCallback unregister_cb, L2PerfFreeCallback free_cb, void *user_data) {
    if (shm_host_ == nullptr) return 0;

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    LOG_DEBUG("Cleaning up performance profiling resources");

    auto release_dev = [&](void *p) {
        if (p == nullptr) return;
        if (unregister_cb != nullptr) {
            unregister_cb(p, device_id_);
        }
        if (free_cb != nullptr) {
            free_cb(p, user_data);
        }
    };

    // Free buffers still parked in per-core / per-thread free_queues and as
    // current_buf_ptr — these are owned by the AICPU side, not the
    // framework. Only release the device pointer here; the paired host
    // shadow stays in dev_to_host_ and is freed by clear_mappings() below
    // (single source of truth for shadow lifetime, no double-free risk).
    for (int i = 0; i < num_aicore_; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(shm_host_, i);

        release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(shm_host_, num_aicore_, t);

        release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Release framework-owned buffers (recycled pools, ready_queue,
    // done_queue). release_owned_buffers frees both dev + host shadow and
    // erases mappings for them.
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
    });

    // Free per-core L2PerfAicoreRings (no host shadow paired). The rings
    // were allocated directly via alloc_cb (not alloc_single_buffer), so no
    // entry exists in dev_to_host_ for them.
    for (auto *ring_dev : aicore_rings_dev_) {
        if (ring_dev != nullptr) {
            release_dev(ring_dev);
        }
    }
    aicore_rings_dev_.clear();

    // Free address table (device + host shadow via clear_mappings below).
    if (aicore_ring_addrs_dev_ != nullptr) {
        release_dev(aicore_ring_addrs_dev_);
        aicore_ring_addrs_dev_ = nullptr;
    }
    aicore_ring_addrs_host_ = nullptr;

    // Free shared memory region (device only — shadow stays in
    // dev_to_host_ until clear_mappings).
    if (perf_shared_mem_dev_ != nullptr) {
        release_dev(perf_shared_mem_dev_);
        perf_shared_mem_dev_ = nullptr;
    }

    // Free remaining host shadows: per-state buffers + the shm region.
    manager_.clear_mappings();

    collected_perf_records_.clear();
    collected_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    total_phase_collected_ = 0;
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
