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
 * @brief Host-side performance data collector (memcpy-based) implementation
 */

#include "host/l2_perf_collector.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/unified_log.h"

// =============================================================================
// Helpers
// =============================================================================

/**
 * Check if a phase ID belongs to a scheduler phase (vs orchestrator phase).
 * Scheduler phases: SCHED_COMPLETE(0), SCHED_DISPATCH(1), SCHED_SCAN(2), SCHED_IDLE_WAIT(3)
 * Orchestrator phases: ORCH_SYNC(16) through ORCH_SCOPE_END(24)
 */
static bool is_scheduler_phase(AicpuPhaseId id) {
    return static_cast<uint32_t>(id) < static_cast<uint32_t>(AicpuPhaseId::SCHED_PHASE_COUNT);
}

// =============================================================================
// L2PerfCollector Implementation
// =============================================================================

L2PerfCollector::~L2PerfCollector() {
    if (setup_header_dev_ != nullptr) {
        LOG_WARN("L2PerfCollector destroyed without finalize()");
    }
}

int L2PerfCollector::initialize(
    Runtime &runtime, int num_aicore, int device_id, L2PerfAllocCallback alloc_cb, L2PerfFreeCallback free_cb,
    L2PerfCopyToDeviceCallback copy_to_dev_cb, L2PerfCopyFromDeviceCallback copy_from_dev_cb
) {
    if (setup_header_dev_ != nullptr) {
        LOG_ERROR("L2PerfCollector already initialized");
        return -1;
    }

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }
    if (alloc_cb == nullptr || free_cb == nullptr || copy_to_dev_cb == nullptr || copy_from_dev_cb == nullptr) {
        LOG_ERROR("L2PerfCollector::initialize: null callback");
        return -1;
    }

    LOG_INFO("Initializing performance profiling (memcpy-based)");

    device_id_ = device_id;
    num_aicore_ = num_aicore;
    num_phase_threads_ = PLATFORM_MAX_AICPU_THREADS;
    alloc_cb_ = alloc_cb;
    free_cb_ = free_cb;
    copy_to_dev_cb_ = copy_to_dev_cb;
    copy_from_dev_cb_ = copy_from_dev_cb;

    l2_perf_buffer_bytes_ = calc_l2_perf_buffer_size(PLATFORM_PROF_BUFFER_SIZE);
    phase_buffer_bytes_ = calc_phase_buffer_size(PLATFORM_PHASE_RECORDS_PER_THREAD);

    LOG_DEBUG("  L2PerfSetupHeader size: %zu bytes", calc_l2_perf_setup_size());
    LOG_DEBUG("  L2PerfBuffer size:      %zu bytes (capacity=%d)", l2_perf_buffer_bytes_, PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG(
        "  PhaseBuffer size:     %zu bytes (capacity=%d)", phase_buffer_bytes_, PLATFORM_PHASE_RECORDS_PER_THREAD
    );
    LOG_DEBUG("  num_aicore:           %d", num_aicore_);
    LOG_DEBUG("  num_phase_threads:    %d", num_phase_threads_);

    // Step 1: Allocate L2PerfSetupHeader on device
    setup_header_dev_ = alloc_cb_(calc_l2_perf_setup_size());
    if (setup_header_dev_ == nullptr) {
        LOG_ERROR("Failed to allocate L2PerfSetupHeader (%zu bytes)", calc_l2_perf_setup_size());
        return -1;
    }

    // Step 2: Allocate one L2PerfBuffer per core on device
    core_buffers_dev_.assign(num_aicore_, nullptr);
    for (int i = 0; i < num_aicore_; i++) {
        void *buf = alloc_cb_(l2_perf_buffer_bytes_);
        if (buf == nullptr) {
            LOG_ERROR("Failed to allocate L2PerfBuffer for core %d (%zu bytes)", i, l2_perf_buffer_bytes_);
            finalize();
            return -1;
        }
        core_buffers_dev_[i] = buf;
    }

    // Step 3: Allocate one PhaseBuffer per AICPU thread on device
    phase_buffers_dev_.assign(num_phase_threads_, nullptr);
    for (int t = 0; t < num_phase_threads_; t++) {
        void *buf = alloc_cb_(phase_buffer_bytes_);
        if (buf == nullptr) {
            LOG_ERROR("Failed to allocate PhaseBuffer for thread %d (%zu bytes)", t, phase_buffer_bytes_);
            finalize();
            return -1;
        }
        phase_buffers_dev_[t] = buf;
    }

    // Step 4: Build L2PerfSetupHeader on host and copy to device
    L2PerfSetupHeader host_header;
    memset(&host_header, 0, sizeof(host_header));
    host_header.num_cores = static_cast<uint32_t>(num_aicore_);
    host_header.num_phase_threads = static_cast<uint32_t>(num_phase_threads_);
    host_header.total_tasks = 0;
    for (int i = 0; i < num_aicore_; i++) {
        host_header.core_buffer_ptrs[i] = reinterpret_cast<uint64_t>(core_buffers_dev_[i]);
    }
    for (int t = 0; t < num_phase_threads_; t++) {
        host_header.phase_buffer_ptrs[t] = reinterpret_cast<uint64_t>(phase_buffers_dev_[t]);
    }
    // phase_header is zero-initialized; AICPU sets magic during init.

    int rc = copy_to_dev_cb_(setup_header_dev_, &host_header, sizeof(host_header));
    if (rc != 0) {
        LOG_ERROR("Failed to copy L2PerfSetupHeader to device: %d", rc);
        finalize();
        return rc;
    }

    // Step 5: Publish the device-side header pointer via runtime.l2_perf_data_base.
    // AICPU reads this on init_profiling to discover per-core / per-thread buffer pointers.
    runtime.l2_perf_data_base = reinterpret_cast<uint64_t>(setup_header_dev_);
    LOG_DEBUG("runtime.l2_perf_data_base = 0x%lx", runtime.l2_perf_data_base);

    LOG_INFO(
        "Performance profiling initialized: %d cores × %zuB L2PerfBuffer, %d threads × %zuB PhaseBuffer", num_aicore_,
        l2_perf_buffer_bytes_, num_phase_threads_, phase_buffer_bytes_
    );
    return 0;
}

int L2PerfCollector::collect_all() {
    if (setup_header_dev_ == nullptr) {
        LOG_ERROR("L2PerfCollector::collect_all called before initialize");
        return -1;
    }

    LOG_INFO("Collecting performance data via device→host memcpy");

    // Step 1: Copy back L2PerfSetupHeader (contains total_tasks and phase_header)
    L2PerfSetupHeader host_header;
    memset(&host_header, 0, sizeof(host_header));
    int rc = copy_from_dev_cb_(&host_header, setup_header_dev_, sizeof(host_header));
    if (rc != 0) {
        LOG_ERROR("Failed to copy L2PerfSetupHeader from device: %d", rc);
        return rc;
    }

    uint32_t total_tasks = host_header.total_tasks;
    LOG_DEBUG("L2PerfSetupHeader: total_tasks=%u", total_tasks);

    // Step 2: Prepare host-side storage
    collected_perf_records_.clear();
    collected_perf_records_.resize(num_aicore_);
    collected_phase_records_.clear();
    collected_phase_records_.resize(num_phase_threads_);

    // Step 3: Two-step copy each L2PerfBuffer back.
    //   - First copy 64B header → read count
    //   - Then copy count * sizeof(L2PerfRecord) of actual data
    uint64_t total_perf_records = 0;
    {
        // Reusable header buffer (aligned to 64B to match L2PerfBuffer layout)
        alignas(64) unsigned char header_buf[sizeof(L2PerfBuffer)];
        for (int i = 0; i < num_aicore_; i++) {
            void *dev_ptr = core_buffers_dev_[i];
            if (dev_ptr == nullptr) continue;

            rc = copy_from_dev_cb_(header_buf, dev_ptr, sizeof(L2PerfBuffer));
            if (rc != 0) {
                LOG_ERROR("Failed to copy L2PerfBuffer header for core %d: %d", i, rc);
                continue;
            }

            uint32_t count = reinterpret_cast<L2PerfBuffer *>(header_buf)->count;
            if (count > static_cast<uint32_t>(PLATFORM_PROF_BUFFER_SIZE)) {
                LOG_WARN(
                    "Core %d: L2PerfBuffer count=%u exceeds capacity=%d, clamping", i, count, PLATFORM_PROF_BUFFER_SIZE
                );
                count = PLATFORM_PROF_BUFFER_SIZE;
            }
            if (count == 0) {
                LOG_DEBUG("Core %d: empty L2PerfBuffer", i);
                continue;
            }

            collected_perf_records_[i].resize(count);
            size_t records_bytes = static_cast<size_t>(count) * sizeof(L2PerfRecord);
            void *dev_records = static_cast<unsigned char *>(dev_ptr) + sizeof(L2PerfBuffer);
            rc = copy_from_dev_cb_(collected_perf_records_[i].data(), dev_records, records_bytes);
            if (rc != 0) {
                LOG_ERROR("Failed to copy L2PerfBuffer records for core %d: %d", i, rc);
                collected_perf_records_[i].clear();
                continue;
            }
            total_perf_records += count;
            LOG_DEBUG("Core %d: collected %u perf records", i, count);
        }
    }

    // Step 4: Two-step copy each PhaseBuffer back.
    uint64_t total_phase_records = 0;
    {
        alignas(64) unsigned char header_buf[sizeof(PhaseBuffer)];
        for (int t = 0; t < num_phase_threads_; t++) {
            void *dev_ptr = phase_buffers_dev_[t];
            if (dev_ptr == nullptr) continue;

            rc = copy_from_dev_cb_(header_buf, dev_ptr, sizeof(PhaseBuffer));
            if (rc != 0) {
                LOG_ERROR("Failed to copy PhaseBuffer header for thread %d: %d", t, rc);
                continue;
            }

            uint32_t count = reinterpret_cast<PhaseBuffer *>(header_buf)->count;
            if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                LOG_WARN(
                    "Thread %d: PhaseBuffer count=%u exceeds capacity=%d, clamping", t, count,
                    PLATFORM_PHASE_RECORDS_PER_THREAD
                );
                count = PLATFORM_PHASE_RECORDS_PER_THREAD;
            }
            if (count == 0) {
                continue;
            }

            collected_phase_records_[t].resize(count);
            size_t records_bytes = static_cast<size_t>(count) * sizeof(AicpuPhaseRecord);
            void *dev_records = static_cast<unsigned char *>(dev_ptr) + sizeof(PhaseBuffer);
            rc = copy_from_dev_cb_(collected_phase_records_[t].data(), dev_records, records_bytes);
            if (rc != 0) {
                LOG_ERROR("Failed to copy PhaseBuffer records for thread %d: %d", t, rc);
                collected_phase_records_[t].clear();
                continue;
            }
            total_phase_records += count;
        }
    }

    // Step 5: Extract phase header fields (orch summary + core-to-thread mapping)
    const AicpuPhaseHeader &phase_header = host_header.phase_header;
    bool phase_header_valid = (phase_header.magic == AICPU_PHASE_MAGIC);

    if (phase_header_valid) {
        collected_orch_summary_ = phase_header.orch_summary;
        int num_cores_mapping = static_cast<int>(phase_header.num_cores);
        if (num_cores_mapping > 0 && num_cores_mapping <= PLATFORM_MAX_CORES) {
            core_to_thread_.assign(phase_header.core_to_thread, phase_header.core_to_thread + num_cores_mapping);
            LOG_DEBUG("Core-to-thread mapping: %d cores", num_cores_mapping);
        }
    } else {
        memset(&collected_orch_summary_, 0, sizeof(collected_orch_summary_));
    }

    bool orch_valid = (collected_orch_summary_.magic == AICPU_PHASE_MAGIC);
    has_phase_data_ = (total_phase_records > 0) || orch_valid;

    // Step 6: Log per-thread totals (sched vs orch breakdown)
    if (has_phase_data_) {
        for (size_t t = 0; t < collected_phase_records_.size(); t++) {
            if (collected_phase_records_[t].empty()) continue;
            size_t sched_count = 0, orch_count = 0;
            for (const auto &r : collected_phase_records_[t]) {
                if (is_scheduler_phase(r.phase_id)) sched_count++;
                else orch_count++;
            }
            LOG_INFO(
                "  Thread %zu: %zu phase records (sched=%zu, orch=%zu)", t, collected_phase_records_[t].size(),
                sched_count, orch_count
            );
        }
        if (orch_valid) {
            LOG_INFO(
                "  Orchestrator: %" PRId64 " tasks, %.3fus", static_cast<int64_t>(collected_orch_summary_.submit_count),
                cycles_to_us(collected_orch_summary_.end_time - collected_orch_summary_.start_time)
            );
        }
    }

    LOG_INFO(
        "Collection complete: %" PRIu64 " perf records, %" PRIu64 " phase records, orch_summary=%s", total_perf_records,
        total_phase_records, orch_valid ? "yes" : "no"
    );

    if (total_tasks > 0 && total_perf_records < total_tasks) {
        LOG_WARN(
            "Incomplete collection: %" PRIu64 " / %u records (some cores may have filled their L2PerfBuffer)",
            total_perf_records, total_tasks
        );
    }

    return 0;
}

int L2PerfCollector::export_swimlane_json(const std::string &output_path_arg) {
    // Step 0: Resolve effective output directory. SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR (when set)
    // overrides the caller-supplied path so the parallel test orchestrator can
    // give each subprocess its own directory — avoids filename collisions when
    // two concurrent runs produce a l2_perf_records_*.json with the same
    // second-precision timestamp. Empty env var is treated as unset.
    const char *env_dir = std::getenv("SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR");
    const std::string output_path = (env_dir != nullptr && env_dir[0] != '\0') ? std::string(env_dir) : output_path_arg;

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

    // Step 2: Create output directory if it doesn't exist
    struct stat st;
    if (stat(output_path.c_str(), &st) == -1) {
        if (mkdir(output_path.c_str(), 0755) != 0) {
            LOG_ERROR("Error: Failed to create output directory.");
            return -1;
        }
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

    // Step 5: Generate filename with timestamp (YYYYMMDD_HHMMSS)
    std::time_t now = time(nullptr);
    std::tm *timeinfo = std::localtime(&now);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);
    std::string filepath = output_path + "/l2_perf_records_" + std::string(time_buffer) + ".json";

    // Step 6: Open JSON file for writing
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 7: Write JSON data
    // Build producer→consumers map from fanin records (PTO2 runtimes).
    // Fanout records (host_build_graph) contribute directly per-record.
    std::unordered_map<uint64_t, std::vector<uint64_t>> producer_to_consumers;
    producer_to_consumers.reserve(tagged_records.size());
    for (const auto &tagged : tagged_records) {
        const auto &rec = *tagged.record;
        if (!rec.is_fanin) continue;
        int safe_count =
            (rec.fan_tasks_count >= 0 && rec.fan_tasks_count <= RUNTIME_MAX_FAN_TASKS) ? rec.fan_tasks_count : 0;
        for (int k = 0; k < safe_count; k++) {
            producer_to_consumers[rec.fan_tasks[k]].push_back(rec.task_id);
        }
    }

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
        int fanout_emitted = 0;
        if (!record.is_fanin) {
            int safe_count = (record.fan_tasks_count >= 0 && record.fan_tasks_count <= RUNTIME_MAX_FAN_TASKS) ?
                                 record.fan_tasks_count :
                                 0;
            for (int j = 0; j < safe_count; ++j) {
                if (j > 0) {
                    outfile << ", ";
                }
                outfile << record.fan_tasks[j];
                fanout_emitted++;
            }
        } else {
            const auto it = producer_to_consumers.find(record.task_id);
            if (it != producer_to_consumers.end()) {
                const auto &successors = it->second;
                for (size_t j = 0; j < successors.size(); ++j) {
                    if (j > 0) {
                        outfile << ", ";
                    }
                    outfile << successors[j];
                    fanout_emitted++;
                }
            }
        }
        outfile << "],\n";
        outfile << "      \"fanout_count\": " << fanout_emitted << "\n";
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
    LOG_INFO("=== JSON Export Complete ===");
    LOG_INFO("File: %s", filepath.c_str());
    LOG_INFO("Records: %u", record_count);

    return 0;
}

int L2PerfCollector::finalize() {
    if (setup_header_dev_ == nullptr && core_buffers_dev_.empty() && phase_buffers_dev_.empty()) {
        return 0;
    }

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Free per-core L2PerfBuffers
    if (free_cb_ != nullptr) {
        for (void *ptr : core_buffers_dev_) {
            if (ptr != nullptr) {
                free_cb_(ptr);
            }
        }
    }
    core_buffers_dev_.clear();

    // Free per-thread PhaseBuffers
    if (free_cb_ != nullptr) {
        for (void *ptr : phase_buffers_dev_) {
            if (ptr != nullptr) {
                free_cb_(ptr);
            }
        }
    }
    phase_buffers_dev_.clear();

    // Free L2PerfSetupHeader
    if (free_cb_ != nullptr && setup_header_dev_ != nullptr) {
        free_cb_(setup_header_dev_);
    }
    setup_header_dev_ = nullptr;

    // Clear host-side state
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    memset(&collected_orch_summary_, 0, sizeof(collected_orch_summary_));
    core_to_thread_.clear();
    has_phase_data_ = false;

    num_aicore_ = 0;
    num_phase_threads_ = 0;
    device_id_ = -1;
    l2_perf_buffer_bytes_ = 0;
    phase_buffer_bytes_ = 0;
    alloc_cb_ = nullptr;
    free_cb_ = nullptr;
    copy_to_dev_cb_ = nullptr;
    copy_from_dev_cb_ = nullptr;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
