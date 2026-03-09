/**
 * @file performance_collector.cpp
 * @brief Platform-agnostic performance data collector implementation
 */

#include "host/performance_collector.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

PerformanceCollector::~PerformanceCollector() {
    if (perf_shared_mem_host_ != nullptr) {
        LOG_WARN("PerformanceCollector destroyed without finalize()");
    }
}

int PerformanceCollector::initialize(Runtime& runtime,
                                      int num_aicore,
                                      int device_id,
                                      PerfAllocCallback alloc_cb,
                                      PerfRegisterCallback register_cb,
                                      void* user_data) {
    if (perf_shared_mem_host_ != nullptr) {
        LOG_ERROR("PerformanceCollector already initialized");
        return -1;
    }

    LOG_INFO("Initializing performance profiling");

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }

    device_id_ = device_id;
    num_aicore_ = num_aicore;

    // Step 1: Calculate total memory size (with phase profiling region)
    // All PLATFORM_MAX_AICPU_THREADS slots: scheduler threads + orchestrator thread
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);
    size_t header_size = sizeof(PerfDataHeader);
    size_t single_db_size = sizeof(DoubleBuffer);
    size_t buffers_size = num_aicore * single_db_size;

    LOG_DEBUG("Memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", header_size);
    LOG_DEBUG("  Ready queue entries:  %d", PLATFORM_PROF_READYQUEUE_SIZE);
    LOG_DEBUG("  Single DoubleBuffer:  %zu bytes", single_db_size);
    LOG_DEBUG("  All DoubleBuffers:    %zu bytes", buffers_size);
    LOG_DEBUG("  Total size:           %zu bytes (%zu KB, %zu MB)",
              total_size, total_size / 1024, total_size / (1024 * 1024));

    // Step 2: Allocate device memory via callback
    void* perf_dev_ptr = alloc_cb(total_size, user_data);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate device memory for profiling (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated device memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (optional, can be nullptr for simulation)
    void* perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, user_data, &perf_host_ptr);
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
        // Simulation mode: both pointers point to same memory
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize header
    PerfDataHeader* header = get_perf_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->total_tasks = 0;

    LOG_DEBUG("Initialized PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);
    LOG_DEBUG("  num threads:      %d", PLATFORM_MAX_AICPU_THREADS);

    // Step 5: Initialize all DoubleBuffers
    DoubleBuffer* buffers = get_double_buffers(perf_host_ptr);
    for (int i = 0; i < num_aicore; i++) {
        DoubleBuffer* db = &buffers[i];
        memset(&db->buffer1, 0, sizeof(PerfBuffer));
        db->buffer1.count = 0;
        db->buffer1_status = BufferStatus::IDLE;
        memset(&db->buffer2, 0, sizeof(PerfBuffer));
        db->buffer2.count = 0;
        db->buffer2_status = BufferStatus::IDLE;
    }
    LOG_DEBUG("Initialized %d DoubleBuffers (all status=0, idle)", num_aicore);

    wmb();

    // Step 6: Pass to Runtime
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;
    LOG_DEBUG("Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("Performance profiling initialized");
    return 0;
}

void PerformanceCollector::poll_and_collect(int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    LOG_INFO("Collecting performance data");

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);
    DoubleBuffer* buffers = get_double_buffers(perf_shared_mem_host_);
    int num_aicore = num_aicore_;

    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);
    std::optional<std::chrono::steady_clock::time_point> idle_start;

    if (expected_tasks <= 0) {
        LOG_INFO("Waiting for AICPU to write total_tasks in PerfDataHeader...");
        idle_start = std::chrono::steady_clock::now();

        while (true) {
            rmb();
            uint32_t raw_total_tasks = header->total_tasks;

            if (raw_total_tasks > 0) {
                expected_tasks = static_cast<int>(raw_total_tasks);
                LOG_INFO("AICPU reported task count: %d", expected_tasks);
                break;
            }

            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR("Timeout waiting for AICPU task count after %ld seconds",
                         std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                LOG_INFO("AICPU finally reported task count: %d", raw_total_tasks);
                return;
            }
        }
    }

    LOG_DEBUG("Initial expected tasks: %d", expected_tasks);

    int total_records_collected = 0;
    int buffers_processed = 0;

    collected_perf_records_.clear();
    collected_perf_records_.resize(num_aicore);
    idle_start.reset();
    int empty_poll_count = 0;

    // Pre-allocate phase record storage for double-buffer collection during polling
    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);
    int num_sched_for_poll = 0;
    if (phase_header->magic == AICPU_PHASE_MAGIC) {
        num_sched_for_poll = phase_header->num_sched_threads;
        if (num_sched_for_poll > PLATFORM_MAX_AICPU_THREADS) {
            LOG_WARN("num_sched_threads %d capped to %d", num_sched_for_poll, PLATFORM_MAX_AICPU_THREADS);
            num_sched_for_poll = PLATFORM_MAX_AICPU_THREADS;
        }
        collected_phase_records_.clear();
        collected_phase_records_.resize(num_sched_for_poll);
        collected_orch_phase_records_.clear();
    }

    int current_thread = 0;

    while (total_records_collected < expected_tasks) {
        rmb();

        uint32_t head = header->queue_heads[current_thread];
        uint32_t tail = header->queue_tails[current_thread];

        if (head == tail) {
            current_thread = (current_thread + 1) % PLATFORM_MAX_AICPU_THREADS;

            if (current_thread == 0) {
                if (!idle_start.has_value()) {
                    idle_start = std::chrono::steady_clock::now();
                }

                empty_poll_count++;
                if (empty_poll_count >= PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM) {
                    empty_poll_count = 0;
                    auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
                    if (elapsed >= timeout_duration) {
                        LOG_ERROR("Performance data collection idle timeout after %ld seconds",
                                 std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                        LOG_ERROR("Collected %d / %d records before timeout",
                                 total_records_collected, expected_tasks);
                        break;
                    }
                }
            }
            continue;
        }

        idle_start.reset();
        empty_poll_count = 0;

        ReadyQueueEntry entry = header->queues[current_thread][head];
        uint32_t raw_buffer_id = entry.buffer_id;
        bool is_phase = (raw_buffer_id & PHASE_BUFFER_FLAG) != 0;
        uint32_t buffer_id = raw_buffer_id & ~PHASE_BUFFER_FLAG;

        if (is_phase) {
            // Phase buffer entry: core_index stores thread_idx, buffer_id is 1-based ring index
            uint32_t thread_idx = entry.core_index;
            if (thread_idx >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
                LOG_ERROR("Invalid phase thread_idx %u (max=%d)", thread_idx, PLATFORM_MAX_AICPU_THREADS);
                header->queue_heads[current_thread] = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
                wmb();
                continue;
            }
            uint32_t ring_idx = buffer_id - 1;  // Convert 1-based to 0-based
            if (ring_idx >= static_cast<uint32_t>(PLATFORM_PHASE_RING_DEPTH)) {
                LOG_ERROR("Invalid phase ring_idx %u for thread %u (buffer_id=%u)",
                         ring_idx, thread_idx, buffer_id);
                header->queue_heads[current_thread] = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
                wmb();
                continue;
            }
            PhaseRingBuffer* ring = get_phase_ring_buffer(perf_shared_mem_host_, num_aicore_, thread_idx);
            PhaseBuffer* pbuf = nullptr;
            volatile BufferStatus* pstatus = nullptr;
            get_phase_buffer_by_idx(ring, ring_idx, &pbuf, &pstatus);

            rmb();
            uint32_t count = pbuf->count;
            if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                count = PLATFORM_PHASE_RECORDS_PER_THREAD;
            }

            LOG_DEBUG("Processing phase: thread=%u, buffer=%u, count=%u", thread_idx, buffer_id, count);

            // Store phase records: scheduler threads → collected_phase_records_,
            // orchestrator thread → collected_orch_phase_records_
            if (thread_idx < static_cast<uint32_t>(num_sched_for_poll)) {
                for (uint32_t i = 0; i < count; i++) {
                    collected_phase_records_[thread_idx].push_back(pbuf->records[i]);
                }
            } else {
                for (uint32_t i = 0; i < count; i++) {
                    collected_orch_phase_records_.push_back(pbuf->records[i]);
                }
            }

            pbuf->count = 0;
            *pstatus = BufferStatus::IDLE;
            // Phase records don't count toward total_records_collected
        } else {
            // PerfRecord buffer entry (original logic)
            uint32_t core_index = entry.core_index;

            if (core_index >= static_cast<uint32_t>(num_aicore)) {
                LOG_ERROR("Invalid core_index %u (max=%d)", core_index, num_aicore);
                header->queue_heads[current_thread] = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
                wmb();
                continue;
            }

            LOG_DEBUG("Processing: thread=%d, core=%u, buffer=%u", current_thread, core_index, buffer_id);

            DoubleBuffer* db = &buffers[core_index];
            PerfBuffer* buf = nullptr;
            volatile BufferStatus* status = nullptr;
            get_buffer_and_status(db, buffer_id, &buf, &status);

            rmb();
            uint32_t count = buf->count;
            LOG_DEBUG("  Records in buffer: %u", count);

            for (uint32_t i = 0; i < count && i < PLATFORM_PROF_BUFFER_SIZE; i++) {
                collected_perf_records_[core_index].push_back(buf->records[i]);
                total_records_collected++;
            }

            buf->count = 0;
            *status = BufferStatus::IDLE;
        }

        header->queue_heads[current_thread] = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
        wmb();

        buffers_processed++;

        current_thread = (current_thread + 1) % PLATFORM_MAX_AICPU_THREADS;
    }

    LOG_INFO("Total buffers processed: %d", buffers_processed);
    LOG_INFO("Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("Performance data collection complete");
}

void PerformanceCollector::collect_phase_data() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    rmb();

    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);

    // Validate magic
    if (phase_header->magic != AICPU_PHASE_MAGIC) {
        LOG_INFO("No phase profiling data found (magic mismatch: 0x%x vs 0x%x)",
                 phase_header->magic, AICPU_PHASE_MAGIC);
        return;
    }

    int num_sched_threads = phase_header->num_sched_threads;
    if (num_sched_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid num_sched_threads %d from shared memory (max=%d)",
                  num_sched_threads, PLATFORM_MAX_AICPU_THREADS);
        return;
    }
    LOG_INFO("Collecting remaining phase data: %d scheduler threads", num_sched_threads);

    // Ensure collected_phase_records_ is allocated (may already have data from poll_and_collect)
    int total_slots = (num_sched_threads < PLATFORM_MAX_AICPU_THREADS)
                      ? num_sched_threads + 1 : num_sched_threads;
    if (collected_phase_records_.size() < static_cast<size_t>(num_sched_threads)) {
        collected_phase_records_.resize(num_sched_threads);
    }

    // Scan remaining PhaseRingBuffers (READY or WRITING with data)
    int total_phase_records = 0;
    for (int t = 0; t < total_slots; t++) {
        PhaseRingBuffer* ring = get_phase_ring_buffer(perf_shared_mem_host_, num_aicore_, t);

        for (int idx = 0; idx < PLATFORM_PHASE_RING_DEPTH; idx++) {
            PhaseBuffer* pbuf = nullptr;
            volatile BufferStatus* pstatus = nullptr;
            get_phase_buffer_by_idx(ring, idx, &pbuf, &pstatus);

            rmb();
            BufferStatus st = *pstatus;
            if ((st == BufferStatus::READY || st == BufferStatus::WRITING) && pbuf->count > 0) {
                uint32_t count = pbuf->count;
                if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                    count = PLATFORM_PHASE_RECORDS_PER_THREAD;
                }

                // Scheduler threads go to collected_phase_records_, orchestrator to orch
                if (t < num_sched_threads) {
                    for (uint32_t i = 0; i < count; i++) {
                        collected_phase_records_[t].push_back(pbuf->records[i]);
                    }
                } else {
                    for (uint32_t i = 0; i < count; i++) {
                        collected_orch_phase_records_.push_back(pbuf->records[i]);
                    }
                }
                total_phase_records += count;
            }
        }
    }

    // Log per-thread totals
    for (int t = 0; t < num_sched_threads; t++) {
        LOG_INFO("  Thread %d: %zu phase records", t, collected_phase_records_[t].size());
    }
    if (!collected_orch_phase_records_.empty()) {
        LOG_INFO("  Orchestrator: %zu per-task phase records", collected_orch_phase_records_.size());
    }

    // Read orchestrator summary
    collected_orch_summary_ = phase_header->orch_summary;
    bool orch_valid = (collected_orch_summary_.magic == AICPU_PHASE_MAGIC);

    if (orch_valid) {
        LOG_INFO("  Orchestrator: %lld tasks, %.3fus",
                 (long long)collected_orch_summary_.submit_count,
                 cycles_to_us(collected_orch_summary_.end_time - collected_orch_summary_.start_time));
    } else {
        LOG_INFO("  Orchestrator: no summary data");
    }

    has_phase_data_ = (total_phase_records > 0 || orch_valid);

    // Read core-to-thread mapping
    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread,
                                phase_header->core_to_thread + num_cores);
        LOG_INFO("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO("Phase data collection complete: %d remaining records, orch_summary=%s",
             total_phase_records, orch_valid ? "yes" : "no");
}

int PerformanceCollector::export_swimlane_json(const std::string& output_path) {
    // Step 1: Validate collected data
    bool has_any_records = false;
    for (const auto& core_records : collected_perf_records_) {
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
        const PerfRecord* record;
        uint32_t core_id;
    };
    std::vector<TaggedRecord> tagged_records;
    size_t total_records = 0;
    for (const auto& core_records : collected_perf_records_) {
        total_records += core_records.size();
    }
    tagged_records.reserve(total_records);
    for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
        for (const auto& record : collected_perf_records_[core_idx]) {
            tagged_records.push_back({&record, static_cast<uint32_t>(core_idx)});
        }
    }

    // Sort by task_id
    std::sort(tagged_records.begin(), tagged_records.end(),
              [](const TaggedRecord& a, const TaggedRecord& b) {
                  return a.record->task_id < b.record->task_id;
              });

    // Step 4: Calculate base time (minimum kernel_ready_time, including phase timestamps)
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto& tagged : tagged_records) {
        if (tagged.record->kernel_ready_time < base_time_cycles) {
            base_time_cycles = tagged.record->kernel_ready_time;
        }
        if (tagged.record->dispatch_time < base_time_cycles && tagged.record->dispatch_time > 0) {
            base_time_cycles = tagged.record->dispatch_time;
            LOG_WARN("Timestamp violation: dispatch_time (%lu) < base_time (%lu) for task %u, using dispatch_time as new base_time",
                        tagged.record->dispatch_time, base_time_cycles, tagged.record->task_id);
        }
    }

    // Include phase record timestamps in base_time calculation
    if (has_phase_data_) {
        for (const auto& thread_records : collected_phase_records_) {
            for (const auto& pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
        for (const auto& pr : collected_orch_phase_records_) {
            if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                base_time_cycles = pr.start_time;
            }
        }
        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC &&
            collected_orch_summary_.start_time > 0 &&
            collected_orch_summary_.start_time < base_time_cycles) {
            base_time_cycles = collected_orch_summary_.start_time;
        }
    }

    // Step 5: Generate filename with timestamp (YYYYMMDD_HHMMSS)
    std::time_t now = time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);
    std::string filepath = output_path + "/perf_swimlane_"
                          + std::string(time_buffer) + ".json";

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
        const auto& tagged = tagged_records[i];
        const auto& record = *tagged.record;

        // Convert times to microseconds
        double start_us = cycles_to_us(record.start_time - base_time_cycles);
        double end_us = cycles_to_us(record.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double kernel_ready_us = cycles_to_us(record.kernel_ready_time - base_time_cycles);
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        const char* core_type_str = (record.core_type == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << record.task_id << ",\n";
        outfile << "      \"func_id\": " << record.func_id << ",\n";
        outfile << "      \"core_id\": " << tagged.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"kernel_ready_time_us\": " << std::fixed << std::setprecision(3) << kernel_ready_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << ",\n";
        outfile << "      \"fanout\": [";
        int safe_fanout_count = (record.fanout_count >= 0 && record.fanout_count <= RUNTIME_MAX_FANOUT)
                                ? record.fanout_count : 0;
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
        // AICPU scheduler phases
        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_phase_records_.size(); t++) {
            outfile << "    [\n";
            const auto& thread_records = collected_phase_records_[t];
            for (size_t r = 0; r < thread_records.size(); r++) {
                const auto& pr = thread_records[r];
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);

                const char* phase_name = "unknown";
                switch (pr.phase_id) {
                    case AicpuPhaseId::SCHED_COMPLETE:    phase_name = "complete"; break;
                    case AicpuPhaseId::SCHED_DISPATCH:    phase_name = "dispatch"; break;
                    case AicpuPhaseId::SCHED_SCAN:        phase_name = "scan"; break;
                    case AicpuPhaseId::SCHED_IDLE_WAIT:   phase_name = "idle"; break;
                    default: break;
                }

                outfile << "      {\"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                        << ", \"phase\": \"" << phase_name << "\""
                        << ", \"loop_iter\": " << pr.loop_iter
                        << ", \"tasks_processed\": " << pr.tasks_processed
                        << "}";
                if (r < thread_records.size() - 1) outfile << ",";
                outfile << "\n";
            }
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
            outfile << "      \"sync\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.sync_cycle) << ",\n";
            outfile << "      \"alloc\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.alloc_cycle) << ",\n";
            outfile << "      \"params\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.params_cycle) << ",\n";
            outfile << "      \"lookup\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.lookup_cycle) << ",\n";
            outfile << "      \"heap\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.heap_cycle) << ",\n";
            outfile << "      \"insert\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.insert_cycle) << ",\n";
            outfile << "      \"fanin\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.fanin_cycle) << ",\n";
            outfile << "      \"scope_end\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.scope_end_cycle) << "\n";
            outfile << "    }\n";
            outfile << "  }";
        }

        // Per-task orchestrator phase records
        if (!collected_orch_phase_records_.empty()) {
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";

            // Map orchestrator phase IDs to names
            auto orch_phase_name = [](AicpuPhaseId id) -> const char* {
                switch (id) {
                    case AicpuPhaseId::ORCH_SYNC:      return "orch_sync";
                    case AicpuPhaseId::ORCH_ALLOC:     return "orch_alloc";
                    case AicpuPhaseId::ORCH_PARAMS:    return "orch_params";
                    case AicpuPhaseId::ORCH_LOOKUP:    return "orch_lookup";
                    case AicpuPhaseId::ORCH_HEAP:      return "orch_heap";
                    case AicpuPhaseId::ORCH_INSERT:    return "orch_insert";
                    case AicpuPhaseId::ORCH_FANIN:     return "orch_fanin";
                    case AicpuPhaseId::ORCH_FINALIZE:  return "orch_finalize";
                    case AicpuPhaseId::ORCH_SCOPE_END: return "orch_scope_end";
                    default: return "unknown";
                }
            };

            for (size_t r = 0; r < collected_orch_phase_records_.size(); r++) {
                const auto& pr = collected_orch_phase_records_[r];
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);

                outfile << "    {\"phase\": \"" << orch_phase_name(pr.phase_id) << "\""
                        << ", \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                        << ", \"submit_idx\": " << pr.loop_iter
                        << ", \"task_id\": " << static_cast<int32_t>(pr.tasks_processed)
                        << "}";
                if (r < collected_orch_phase_records_.size() - 1) outfile << ",";
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

int PerformanceCollector::finalize(PerfUnregisterCallback unregister_cb,
                                    PerfFreeCallback free_cb,
                                    void* user_data) {
    if (perf_shared_mem_host_ == nullptr) {
        return 0;
    }

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Unregister host mapping (optional)
    if (unregister_cb != nullptr && was_registered_) {
        int rc = unregister_cb(perf_shared_mem_dev_, device_id_, user_data);
        if (rc != 0) {
            LOG_ERROR("halHostUnregister failed: %d", rc);
            return rc;
        }
        LOG_DEBUG("Host mapping unregistered");
    }

    // Free device memory
    if (free_cb != nullptr && perf_shared_mem_dev_ != nullptr) {
        free_cb(perf_shared_mem_dev_, user_data);
        LOG_DEBUG("Device memory freed");
    }

    perf_shared_mem_dev_ = nullptr;
    perf_shared_mem_host_ = nullptr;
    was_registered_ = false;
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    collected_orch_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    device_id_ = -1;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
