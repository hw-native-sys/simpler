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
    // Destructor should not call finalize() because callbacks are not available
    // User must explicitly call finalize() before destruction
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

    device_id_ = device_id;

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
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else {
        // Simulation mode: both pointers point to same memory
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize fixed header
    PerfDataHeader* header = get_perf_header(perf_host_ptr);
    memset(header->queue, 0, sizeof(header->queue));
    header->queue_head = 0;
    header->queue_tail = 0;
    header->num_cores = num_aicore;

    LOG_DEBUG("Initialized PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);

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

    // Write memory barrier
    wmb();

    // Step 6: Pass to Runtime
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;
    LOG_DEBUG("Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    // Save pointers
    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("Performance profiling initialized");
    return 0;
}

void PerformanceCollector::poll_and_collect(int num_aicore, int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    LOG_INFO("Collecting performance data");

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);
    DoubleBuffer* buffers = get_double_buffers(perf_shared_mem_host_);

    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);
    std::optional<std::chrono::steady_clock::time_point> idle_start;

    // Poll for total_tasks if not provided
    if (expected_tasks <= 0) {
        LOG_INFO("Waiting for AICPU to write total_tasks to PerfDataHeader...");
        idle_start = std::chrono::steady_clock::now();

        while (true) {
            rmb();
            expected_tasks = static_cast<int>(header->total_tasks);

            if (expected_tasks > 0) {
                LOG_INFO("Task count read from PerfDataHeader: %d", expected_tasks);
                break;
            }

            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR("Timeout waiting for total_tasks from AICPU after %ld seconds",
                         std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                LOG_ERROR("AICPU may not have initialized performance profiling");
                return;
            }
        }
    }

    LOG_DEBUG("Expected tasks: %d", expected_tasks);

    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;
    int total_records_collected = 0;
    int buffers_processed = 0;

    collected_perf_records_.clear();
    idle_start.reset();
    int empty_poll_count = 0;

    // Poll the ready queue
    while (total_records_collected < expected_tasks) {
        rmb();
        uint32_t head = header->queue_head;
        uint32_t tail = header->queue_tail;

        if (head == tail) {
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }

            empty_poll_count++;
            if (empty_poll_count >= PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM) {
                empty_poll_count = 0;
                auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
                if (elapsed >= timeout_duration) {
                    LOG_WARN("Performance data collection idle timeout after %ld seconds",
                             std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                    LOG_WARN("Collected %d / %d records before timeout",
                             total_records_collected, expected_tasks);
                    break;
                }
            }
            continue;
        }

        idle_start.reset();
        empty_poll_count = 0;

        ReadyQueueEntry entry = header->queue[head];
        uint32_t core_index = entry.core_index;
        uint32_t buffer_id = entry.buffer_id;

        if (core_index >= static_cast<uint32_t>(num_aicore)) {
            LOG_ERROR("Invalid core_index %u (max=%d)", core_index, num_aicore);
            break;
        }

        LOG_DEBUG("Processing: core=%u, buffer=%u", core_index, buffer_id);

        DoubleBuffer* db = &buffers[core_index];
        PerfBuffer* buf = nullptr;
        volatile BufferStatus* status = nullptr;
        get_buffer_and_status(db, buffer_id, &buf, &status);

        rmb();
        uint32_t count = buf->count;
        LOG_DEBUG("  Records in buffer: %u", count);

        for (uint32_t i = 0; i < count && i < PLATFORM_PROF_BUFFER_SIZE; i++) {
            collected_perf_records_.push_back(buf->records[i]);
            total_records_collected++;
        }

        buf->count = 0;
        *status = BufferStatus::IDLE;
        wmb();

        header->queue_head = (head + 1) % capacity;
        wmb();

        buffers_processed++;
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
    if (collected_perf_records_.empty()) {
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

    // Step 3: Sort records by task_id
    std::vector<PerfRecord> sorted_records = collected_perf_records_;
    std::sort(sorted_records.begin(), sorted_records.end(),
              [](const PerfRecord& a, const PerfRecord& b) {
                  return a.task_id < b.task_id;
              });

    // Step 4: Calculate base time (minimum kernel_ready_time)
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto& record : sorted_records) {
        if (record.kernel_ready_time < base_time_cycles) {
            base_time_cycles = record.kernel_ready_time;
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

    for (size_t i = 0; i < sorted_records.size(); ++i) {
        const auto& record = sorted_records[i];

        // Convert times to microseconds
        double start_us = cycles_to_us(record.start_time - base_time_cycles);
        double end_us = cycles_to_us(record.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double kernel_ready_us = cycles_to_us(record.kernel_ready_time - base_time_cycles);
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        // Determine core type string
        const char* core_type_str = (record.core_type == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << record.task_id << ",\n";
        outfile << "      \"func_id\": " << record.func_id << ",\n";
        outfile << "      \"core_id\": " << record.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"kernel_ready_time_us\": " << std::fixed << std::setprecision(3) << kernel_ready_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << ",\n";
        outfile << "      \"fanout\": [";
        for (int j = 0; j < record.fanout_count; ++j) {
            outfile << record.fanout[j];
            if (j < record.fanout_count - 1) {
                outfile << ", ";
            }
        }
        outfile << "],\n";
        outfile << "      \"fanout_count\": " << record.fanout_count << "\n";
        outfile << "    }";
        if (i < sorted_records.size() - 1) {
            outfile << ",";
        }
        outfile << "\n";
    }
    outfile << "  ]\n";
    outfile << "}\n";

    // Step 9: Close file
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(sorted_records.size());
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
    if (unregister_cb != nullptr && perf_shared_mem_host_ != perf_shared_mem_dev_) {
        unregister_cb(perf_shared_mem_host_, device_id_, user_data);
        LOG_DEBUG("Host mapping unregistered");
    }

    // Free device memory
    if (free_cb != nullptr && perf_shared_mem_dev_ != nullptr) {
        free_cb(perf_shared_mem_dev_, user_data);
        LOG_DEBUG("Device memory freed");
    }

    perf_shared_mem_dev_ = nullptr;
    perf_shared_mem_host_ = nullptr;
    collected_perf_records_.clear();
    device_id_ = -1;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
