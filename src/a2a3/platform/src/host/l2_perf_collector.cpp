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
 * @brief Performance data collector implementation. The mgmt-thread + buffer-pool
 *        machinery lives in profiling_common::BufferPoolManager parameterized by
 *        L2PerfModule (host/l2_perf_collector.h); the poll loop lives in
 *        profiling_common::ProfilerBase. This file owns the per-buffer
 *        on_buffer_collected callback and the export logic.
 */

#include "host/l2_perf_collector.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

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
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void *host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        *host_ptr_out = dev_ptr;
    }

    // Register mapping so the BufferPoolManager can resolve dev→host
    manager_.register_mapping(dev_ptr, *host_ptr_out);
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

    // Stash the memory context on the base up-front so alloc_single_buffer
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — the nullptr guard makes a post-failure
    // start(tf) a no-op.
    set_memory_context(alloc_cb, register_cb, free_cb, user_data, /*shm_host=*/nullptr, device_id);

    // Step 1: Calculate shared memory size (slot arrays only, no actual buffers)
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(L2PerfDataHeader));
    LOG_DEBUG("  L2PerfBufferState size: %zu bytes each", sizeof(L2PerfBufferState));
    LOG_DEBUG("  PhaseBufferState size:%zu bytes each", sizeof(PhaseBufferState));
    LOG_DEBUG("  Total shared memory:  %zu bytes (%zu KB)", total_size, total_size / 1024);

    // Step 2: Allocate shared memory for slot arrays
    void *perf_dev_ptr = alloc_cb(total_size, user_data);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (optional)
    void *perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, &perf_host_ptr);
        if (rc != 0) {
            LOG_ERROR("Memory registration failed: %d", rc);
            return rc;
        }
        was_registered_ = true;
        if (perf_host_ptr == nullptr) {
            LOG_ERROR("register_cb succeeded but returned null host_ptr");
            return -1;
        }
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else {
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize header
    L2PerfDataHeader *header = get_l2_perf_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;

    LOG_DEBUG("Initialized L2PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize L2PerfBufferStates — 1 buffer per core in free_queue, rest to recycled pool
    for (int i = 0; i < num_aicore; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(L2PerfBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(L2PerfBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate L2PerfBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            L2PerfBuffer *buf = reinterpret_cast<L2PerfBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(L2PerfBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::PERF_RECORD), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized %d L2PerfBufferStates: 1 buffer/core, %d in recycled pool", num_aicore,
        num_aicore * (PLATFORM_PROF_BUFFERS_PER_CORE - 1)
    );

    // Step 6: Initialize PhaseBufferStates — 1 buffer per thread in free_queue, rest to recycled pool
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
        memset(state, 0, sizeof(PhaseBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_THREAD; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(PhaseBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PhaseBuffer for thread %d, buffer %d", t, s);
                return -1;
            }
            PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(host_buf_ptr);
            memset(buf, 0, sizeof(PhaseBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                manager_.push_recycled(static_cast<int>(ProfBufferType::PHASE), dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized %d PhaseBufferStates: 1 buffer/thread, %d in recycled pool", num_phase_threads,
        num_phase_threads * (PLATFORM_PROF_BUFFERS_PER_THREAD - 1)
    );

    wmb();

    // Step 7: Stash device pointer for the caller to publish via
    // kernel_args.l2_perf_data_base (read back via get_l2_perf_setup_device_ptr()).
    LOG_DEBUG("L2 perf device base = 0x%lx", reinterpret_cast<uint64_t>(perf_dev_ptr));

    perf_shared_mem_dev_ = perf_dev_ptr;
    shm_host_ = perf_host_ptr;

    collected_perf_records_.assign(num_aicore_, {});
    collected_phase_records_.assign(PLATFORM_MAX_AICPU_THREADS, {});

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

void L2PerfCollector::reconcile_counters() {
    if (shm_host_ == nullptr) {
        return;
    }

    rmb();

    // Three-bucket invariant for one pool: every commit attempt bumps
    // total_record_count; buffer-full, overwrite, and flush-enqueue failures
    // bump dropped_record_count. So
    //   silent_loss = device_total - (collected + dropped)
    // and any non-zero silent loss flags an AICore/AICPU WIP-mismatch race
    // (a different fault class from capacity-driven drops).
    //
    // Sanity sub-check: after stop(), any active buffer with records must
    // have been flushed by AICPU (success → current_buf_ptr=0; failure →
    // bump dropped, clear count + current_buf_ptr). A non-zero pointer with
    // non-zero count means records AICPU neither delivered nor accounted
    // for — i.e. a device-side flush bug. Empty buffers (count=0, never
    // written) are fine; AICPU's flush legitimately skips them.
    auto reconcile_one = [&](const char *kind, const char *unit_name, int unit_count, auto get_state,
                             auto read_buf_count, uint64_t collected, bool optional) {
        int leftover_active = 0;
        for (int i = 0; i < unit_count; i++) {
            L2PerfBufferState *state = get_state(i);
            uint64_t buf_ptr = state->current_buf_ptr;
            if (buf_ptr == 0) continue;
            void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
            if (host_ptr == nullptr) continue;
            uint32_t count = read_buf_count(host_ptr);
            if (count == 0) continue;
            LOG_ERROR(
                "L2Perf reconcile: %s %d has un-flushed %s buffer (current_buf_ptr=0x%lx, count=%u) "
                "after stop() — device flush failed",
                unit_name, i, kind, static_cast<unsigned long>(buf_ptr), count
            );
            leftover_active++;
        }

        uint64_t total_device = 0;
        uint64_t dropped_device = 0;
        for (int i = 0; i < unit_count; i++) {
            L2PerfBufferState *state = get_state(i);
            total_device += state->total_record_count;
            dropped_device += state->dropped_record_count;
        }

        // PHASE counters are populated only by runtimes that actually emit
        // phase records; skip the comparison entirely when nothing happened.
        if (optional && total_device == 0 && collected == 0 && dropped_device == 0) {
            return;
        }

        if (dropped_device > 0) {
            LOG_WARN(
                "L2Perf reconcile: %lu %s records dropped on device side (buffer full / "
                "ready_queue full).",
                static_cast<unsigned long>(dropped_device), kind
            );
        }
        uint64_t accounted = collected + dropped_device;
        if (accounted != total_device) {
            LOG_WARN(
                "L2Perf reconcile: %s count mismatch (collected=%lu + dropped=%lu != "
                "device_total=%lu, silent_loss=%ld) — AICore/AICPU race",
                kind, static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device), static_cast<long>(total_device) - static_cast<long>(accounted)
            );
        } else {
            LOG_INFO_V0(
                "L2Perf reconcile: %s counts match (collected=%lu, dropped=%lu, device_total=%lu)", kind,
                static_cast<unsigned long>(collected), static_cast<unsigned long>(dropped_device),
                static_cast<unsigned long>(total_device)
            );
        }

        if (leftover_active > 0) {
            LOG_ERROR(
                "L2Perf reconcile: %d %s(s) had un-cleared %s current_buf_ptr — see prior errors", leftover_active,
                unit_name, kind
            );
        }
    };

    reconcile_one(
        "PERF", "core", num_aicore_,
        [this](int core_index) {
            return get_perf_buffer_state(shm_host_, core_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<L2PerfBuffer *>(host_ptr)->count;
        },
        total_perf_collected_, /*optional=*/false
    );

    reconcile_one(
        "PHASE", "thread", PLATFORM_MAX_AICPU_THREADS,
        [this](int thread_index) {
            return get_phase_buffer_state(shm_host_, num_aicore_, thread_index);
        },
        [](void *host_ptr) {
            return reinterpret_cast<PhaseBuffer *>(host_ptr)->count;
        },
        total_phase_collected_, /*optional=*/true
    );
}

void L2PerfCollector::read_phase_header_metadata() {
    if (shm_host_ == nullptr) {
        return;
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

    // Orchestrator summary (header-resident; not buffered).
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

    // Core-to-thread mapping (header-resident; not buffered).
    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread, phase_header->core_to_thread + num_cores);
        LOG_INFO_V0("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO_V0("Phase metadata collection complete: orch_summary=%s", orch_valid ? "yes" : "no");
}

int L2PerfCollector::export_swimlane_json() {
    // Step 1: Validate collected data
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

    // Step 2: Create output directory (recursively — parent `outputs/` may not
    // yet exist on a clean checkout / standalone run). `output_prefix_` was
    // captured at initialize() time.
    std::error_code ec;
    std::filesystem::create_directories(output_prefix_, ec);
    if (ec) {
        LOG_ERROR("Error: Failed to create output directory %s: %s", output_prefix_.c_str(), ec.message().c_str());
        return -1;
    }

    // Step 3: Flatten per-core vectors into tagged records with core_id derived from index
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

    // Sort by canonical task_id (64-bit PTO2 raw)
    std::sort(tagged_records.begin(), tagged_records.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
        return a.record->task_id < b.record->task_id;
    });

    // Step 4: Calculate base time (minimum timestamp across all records)
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto &tagged : tagged_records) {
        if (tagged.record->start_time < base_time_cycles) {
            base_time_cycles = tagged.record->start_time;
        }
        if (tagged.record->dispatch_time > 0 && tagged.record->dispatch_time < base_time_cycles) {
            base_time_cycles = tagged.record->dispatch_time;
        }
    }

    // Include phase record timestamps in base_time calculation
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

    // Step 5: Compose output path. Filename is fixed (no timestamp) — the
    // caller-provided directory is the per-task uniqueness boundary.
    std::string filepath = output_prefix_ + "/l2_perf_records.json";

    // Step 6: Open JSON file for writing
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 7: Write JSON data
    int version = has_phase_data_ ? 2 : 1;
    outfile << "{\n";
    outfile << "  \"version\": " << version << ",\n";
    outfile << "  \"tasks\": [\n";

    for (size_t i = 0; i < tagged_records.size(); ++i) {
        const auto &tagged = tagged_records[i];
        const auto &record = *tagged.record;

        // Convert times to microseconds
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

    // Step 8: Write phase profiling data (version 2)
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

        // AICPU scheduler phases (filtered from unified collected_phase_records_)
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

        // AICPU orchestrator summary
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

        // Per-task orchestrator phase records (filtered from unified collected_phase_records_)
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

    // Core-to-thread mapping
    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";

    // Step 9: Close file
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO_V0("=== JSON Export Complete ===");
    LOG_INFO_V0("File: %s", filepath.c_str());
    LOG_INFO_V0("Records: %u", record_count);

    return 0;
}

int L2PerfCollector::finalize(L2PerfUnregisterCallback unregister_cb, L2PerfFreeCallback free_cb, void *user_data) {
    if (shm_host_ == nullptr) {
        return 0;
    }

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Release framework-owned buffers (recycled pools, done_queue, ready_queue).
    // These were not consumed via the AICPU side, so the per-pool free_queue
    // and current_buf_ptr remain populated and are handled in the loops below.
    if (free_cb != nullptr) {
        manager_.release_owned_buffers([free_cb, user_data](void *p) {
            free_cb(p, user_data);
        });
    }

    // Free buffers still parked in per-core / per-thread free_queues and as
    // current_buf_ptr — these were owned by the AICPU side, not the framework.
    for (int i = 0; i < num_aicore_; i++) {
        L2PerfBufferState *state = get_perf_buffer_state(shm_host_, i);

        // Free current buffer if any
        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            free_cb(reinterpret_cast<void *>(state->current_buf_ptr), user_data);
        }
        state->current_buf_ptr = 0;

        // Free all buffers remaining in free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[slot];
            if (buf_ptr != 0 && free_cb != nullptr) {
                free_cb(reinterpret_cast<void *>(buf_ptr), user_data);
            }
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(shm_host_, num_aicore_, t);

        // Free current buffer if any
        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            free_cb(reinterpret_cast<void *>(state->current_buf_ptr), user_data);
        }
        state->current_buf_ptr = 0;

        // Free all buffers remaining in free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_PROF_SLOT_COUNT) {
            queued = PLATFORM_PROF_SLOT_COUNT;
        }
        for (uint32_t k = 0; k < queued; k++) {
            uint32_t slot = (head + k) % PLATFORM_PROF_SLOT_COUNT;
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[slot];
            if (buf_ptr != 0 && free_cb != nullptr) {
                free_cb(reinterpret_cast<void *>(buf_ptr), user_data);
            }
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Unregister host mapping (optional)
    if (unregister_cb != nullptr && was_registered_) {
        int rc = unregister_cb(perf_shared_mem_dev_, device_id_);
        if (rc != 0) {
            LOG_ERROR("halHostUnregister failed: %d", rc);
            return rc;
        }
        LOG_DEBUG("Host mapping unregistered");
    }

    // Free shared memory (slot arrays)
    if (free_cb != nullptr && perf_shared_mem_dev_ != nullptr) {
        free_cb(perf_shared_mem_dev_, user_data);
        LOG_DEBUG("Shared memory freed");
    }

    perf_shared_mem_dev_ = nullptr;
    was_registered_ = false;
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    total_perf_collected_ = 0;
    clear_memory_context();

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
