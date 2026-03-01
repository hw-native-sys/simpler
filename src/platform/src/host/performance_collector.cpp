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

    // Step 1: Calculate total memory size
    size_t total_size = calc_perf_data_size(num_aicore);
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
    int last_logged_expected = -1;

    int current_thread = 0;

    while (total_records_collected < expected_tasks) {
        rmb();

        int current_expected = static_cast<int>(header->total_tasks);
        if (current_expected > expected_tasks) {
            expected_tasks = current_expected;
            if (last_logged_expected < 0) {
                LOG_INFO("Updated expected_tasks to %d (orchestrator progress)", expected_tasks);
                last_logged_expected = expected_tasks;
            }
        }

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
        uint32_t core_index = entry.core_index;
        uint32_t buffer_id = entry.buffer_id;

        if (core_index >= static_cast<uint32_t>(num_aicore)) {
            LOG_ERROR("Invalid core_index %u (max=%d)", core_index, num_aicore);
            break;
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
        header->queue_heads[current_thread] = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
        wmb();

        buffers_processed++;

        current_thread = (current_thread + 1) % PLATFORM_MAX_AICPU_THREADS;
    }

    if (last_logged_expected >= 0 && expected_tasks != last_logged_expected) {
        LOG_INFO("Final expected_tasks: %d (orchestration complete)", expected_tasks);
    }

    LOG_INFO("Total buffers processed: %d", buffers_processed);
    LOG_INFO("Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("Performance data collection complete");
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

    // Step 4: Calculate base time (minimum kernel_ready_time)
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
    outfile << "{\n";
    outfile << "  \"version\": 1,\n";
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
    outfile << "  ]\n";
    outfile << "}\n";

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
    device_id_ = -1;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
