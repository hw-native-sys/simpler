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
 * @file tensor_dump_collector.cpp
 * @brief Host-side tensor dump collector implementation (memcpy-based)
 *
 * Mirrors l2_perf_collector.cpp patterns:
 * - Allocate device buffers, copy header to device
 * - After stream sync, two-step copy (header then data)
 * - Export to JSON manifest + binary payload
 */

#include "host/tensor_dump_collector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

// =============================================================================
// TensorDumpCollector
// =============================================================================

TensorDumpCollector::~TensorDumpCollector() {
    if (is_initialized()) {
        LOG_ERROR("TensorDumpCollector destroyed without finalize()");
    }
}

int TensorDumpCollector::initialize(
    int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpFreeCallback free_cb,
    DumpCopyToDeviceCallback copy_to_dev_cb, DumpCopyFromDeviceCallback copy_from_dev_cb
) {
    if (is_initialized()) {
        LOG_ERROR("TensorDumpCollector already initialized");
        return -1;
    }

    num_dump_threads_ = num_dump_threads;
    device_id_ = device_id;
    alloc_cb_ = alloc_cb;
    free_cb_ = free_cb;
    copy_to_dev_cb_ = copy_to_dev_cb;
    copy_from_dev_cb_ = copy_from_dev_cb;

    int capacity = PLATFORM_DUMP_RECORDS_PER_BUFFER;
    dump_buffer_bytes_ = calc_dump_buffer_size(capacity);
    uint64_t arena_size = calc_dump_arena_size();

    LOG_INFO(
        "Initializing tensor dump: %d threads, %d records/buffer, %zu bytes/buffer, %lu bytes/arena", num_dump_threads,
        capacity, dump_buffer_bytes_, arena_size
    );

    // Allocate DumpSetupHeader on device
    setup_header_dev_ = alloc_cb_(sizeof(DumpSetupHeader));
    if (setup_header_dev_ == nullptr) {
        LOG_ERROR("Failed to allocate DumpSetupHeader (%zu bytes)", sizeof(DumpSetupHeader));
        return -1;
    }

    // Build host-side setup header
    DumpSetupHeader host_header = {};
    host_header.num_dump_threads = static_cast<uint32_t>(num_dump_threads);
    host_header.records_per_buffer = static_cast<uint32_t>(capacity);
    host_header.magic = TENSOR_DUMP_MAGIC;

    // Allocate per-thread buffers and arenas
    dump_buffers_dev_.resize(num_dump_threads, nullptr);
    arena_headers_dev_.resize(num_dump_threads, nullptr);
    arena_data_dev_.resize(num_dump_threads, nullptr);

    for (int t = 0; t < num_dump_threads; t++) {
        // Allocate DumpBuffer
        void *buf = alloc_cb_(dump_buffer_bytes_);
        if (buf == nullptr) {
            LOG_ERROR("Failed to allocate DumpBuffer for thread %d (%zu bytes)", t, dump_buffer_bytes_);
            finalize();
            return -1;
        }
        dump_buffers_dev_[t] = buf;

        // Initialize DumpBuffer on host then copy to device
        std::vector<uint8_t> buf_init(dump_buffer_bytes_, 0);
        DumpBuffer *buf_host = reinterpret_cast<DumpBuffer *>(buf_init.data());
        buf_host->count = 0;
        buf_host->capacity = static_cast<uint32_t>(capacity);
        buf_host->dropped_count = 0;
        int rc = copy_to_dev_cb_(buf, buf_init.data(), dump_buffer_bytes_);
        if (rc != 0) {
            LOG_ERROR("Failed to initialize DumpBuffer for thread %d: %d", t, rc);
            finalize();
            return rc;
        }

        // Allocate DumpArenaHeader
        void *arena_hdr = alloc_cb_(sizeof(DumpArenaHeader));
        if (arena_hdr == nullptr) {
            LOG_ERROR("Failed to allocate DumpArenaHeader for thread %d", t);
            finalize();
            return -1;
        }
        arena_headers_dev_[t] = arena_hdr;

        // Initialize arena header
        DumpArenaHeader host_arena_hdr = {};
        host_arena_hdr.write_offset = 0;
        host_arena_hdr.arena_size = arena_size;
        rc = copy_to_dev_cb_(arena_hdr, &host_arena_hdr, sizeof(DumpArenaHeader));
        if (rc != 0) {
            LOG_ERROR("Failed to initialize DumpArenaHeader for thread %d: %d", t, rc);
            finalize();
            return rc;
        }

        // Allocate arena data
        void *arena = alloc_cb_(static_cast<size_t>(arena_size));
        if (arena == nullptr) {
            LOG_ERROR("Failed to allocate arena for thread %d (%lu bytes)", t, arena_size);
            finalize();
            return -1;
        }
        arena_data_dev_[t] = arena;

        // Fill setup header pointers
        host_header.dump_buffer_ptrs[t] = reinterpret_cast<uint64_t>(buf);
        host_header.arena_header_ptrs[t] = reinterpret_cast<uint64_t>(arena_hdr);
        host_header.arena_data_ptrs[t] = reinterpret_cast<uint64_t>(arena);
        host_header.arena_sizes[t] = arena_size;
    }

    // Copy setup header to device
    int rc = copy_to_dev_cb_(setup_header_dev_, &host_header, sizeof(DumpSetupHeader));
    if (rc != 0) {
        LOG_ERROR("Failed to copy DumpSetupHeader to device: %d", rc);
        finalize();
        return rc;
    }

    LOG_INFO("Tensor dump initialized: %d threads, header at %p", num_dump_threads, setup_header_dev_);
    return 0;
}

int TensorDumpCollector::collect_all() {
    if (!is_initialized()) {
        return -1;
    }

    LOG_INFO("Collecting tensor dump data from %d threads...", num_dump_threads_);

    uint64_t arena_size = calc_dump_arena_size();

    for (int t = 0; t < num_dump_threads_; t++) {
        // Step 1: Copy back DumpBuffer header (64 bytes) to read count
        DumpBuffer host_buf_header = {};
        int rc = copy_from_dev_cb_(&host_buf_header, dump_buffers_dev_[t], sizeof(DumpBuffer));
        if (rc != 0) {
            LOG_ERROR("Thread %d: failed to copy DumpBuffer header: %d", t, rc);
            continue;
        }

        uint32_t count = host_buf_header.count;
        uint32_t dropped = host_buf_header.dropped_count;
        total_dropped_count_ += dropped;

        if (count == 0) {
            LOG_DEBUG("Thread %d: no dump records", t);
            continue;
        }

        // Step 2: Copy back the actual records (count * sizeof(TensorDumpRecord))
        size_t records_bytes = static_cast<size_t>(count) * sizeof(TensorDumpRecord);
        std::vector<uint8_t> records_buf(records_bytes);
        void *dev_records = reinterpret_cast<char *>(dump_buffers_dev_[t]) + sizeof(DumpBuffer);
        rc = copy_from_dev_cb_(records_buf.data(), dev_records, records_bytes);
        if (rc != 0) {
            LOG_ERROR("Thread %d: failed to copy %u dump records: %d", t, count, rc);
            continue;
        }

        // Step 3: Copy back arena header
        DumpArenaHeader host_arena_hdr = {};
        rc = copy_from_dev_cb_(&host_arena_hdr, arena_headers_dev_[t], sizeof(DumpArenaHeader));
        if (rc != 0) {
            LOG_ERROR("Thread %d: failed to copy arena header: %d", t, rc);
            continue;
        }

        // Step 4: Copy back arena data (only up to min(write_offset, arena_size))
        uint64_t arena_bytes_to_copy = host_arena_hdr.write_offset;
        if (arena_bytes_to_copy > arena_size) {
            arena_bytes_to_copy = arena_size;  // Circular wraparound — copy entire arena
        }
        std::vector<uint8_t> arena_buf(static_cast<size_t>(arena_bytes_to_copy));
        if (arena_bytes_to_copy > 0) {
            rc = copy_from_dev_cb_(arena_buf.data(), arena_data_dev_[t], static_cast<size_t>(arena_bytes_to_copy));
            if (rc != 0) {
                LOG_ERROR("Thread %d: failed to copy arena data (%lu bytes): %d", t, arena_bytes_to_copy, rc);
                continue;
            }
        }

        // Step 5: Reconstruct DumpedTensor entries
        const TensorDumpRecord *records = reinterpret_cast<const TensorDumpRecord *>(records_buf.data());
        for (uint32_t i = 0; i < count; i++) {
            const TensorDumpRecord &rec = records[i];

            DumpedTensor dt = {};
            dt.task_id = rec.task_id;
            dt.subtask_id = rec.subtask_id;
            dt.func_id = rec.func_id;
            dt.arg_index = rec.arg_index;
            dt.role = static_cast<TensorDumpRole>(rec.role);
            dt.stage = static_cast<TensorDumpStage>(rec.stage);
            dt.dtype = rec.dtype;
            dt.ndims = rec.ndims;
            dt.is_contiguous = (rec.is_contiguous != 0);
            dt.truncated = (rec.truncated != 0);
            dt.payload_size = rec.payload_size;

            for (int d = 0; d < rec.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
                dt.shapes[d] = rec.shapes[d];
                dt.offsets[d] = rec.offsets[d];
                dt.raw_shapes[d] = rec.raw_shapes[d];
            }

            if (dt.truncated) {
                total_truncated_count_++;
            }

            // Check for arena overwrite
            bool overwritten = false;
            if (host_arena_hdr.write_offset > arena_size) {
                uint64_t oldest_valid = host_arena_hdr.write_offset - arena_size;
                if (rec.payload_offset < oldest_valid) {
                    overwritten = true;
                    total_overwrite_count_++;
                }
            }
            dt.overwritten = overwritten;

            // Extract payload from arena
            if (rec.payload_size > 0 && !overwritten && arena_bytes_to_copy > 0) {
                uint64_t arena_sz = arena_size;
                uint64_t pos = rec.payload_offset % arena_sz;
                uint64_t sz = rec.payload_size;
                dt.bytes.resize(static_cast<size_t>(sz));

                if (pos + sz <= arena_sz) {
                    memcpy(dt.bytes.data(), arena_buf.data() + pos, static_cast<size_t>(sz));
                } else {
                    // Circular wraparound read
                    uint64_t first = arena_sz - pos;
                    memcpy(dt.bytes.data(), arena_buf.data() + pos, static_cast<size_t>(first));
                    memcpy(dt.bytes.data() + first, arena_buf.data(), static_cast<size_t>(sz - first));
                }
            }

            collected_.push_back(std::move(dt));
        }

        LOG_INFO("Thread %d: collected %u records (dropped=%u)", t, count, dropped);
    }

    LOG_INFO("Tensor dump collection complete: %zu tensors total", collected_.size());
    return 0;
}

// =============================================================================
// Export Helpers
// =============================================================================

static const char *tensor_dump_role_name(TensorDumpRole role) {
    switch (role) {
    case TensorDumpRole::INPUT:
        return "input";
    case TensorDumpRole::OUTPUT:
        return "output";
    case TensorDumpRole::INOUT:
        return "inout";
    }
    return "unknown";
}

static const char *tensor_dump_stage_name(TensorDumpStage stage) {
    switch (stage) {
    case TensorDumpStage::BEFORE_DISPATCH:
        return "before_dispatch";
    case TensorDumpStage::AFTER_COMPLETION:
        return "after_completion";
    }
    return "unknown";
}

static std::string dims_to_string(const uint32_t dims[], int ndims) {
    std::ostringstream ss;
    ss << "[";
    for (int d = 0; d < ndims; d++) {
        if (d > 0) {
            ss << ", ";
        }
        ss << dims[d];
    }
    ss << "]";
    return ss.str();
}

static std::string get_dtype_name_from_raw(uint8_t dtype) { return get_dtype_name(static_cast<DataType>(dtype)); }

static uint64_t get_num_elements(const DumpedTensor &dt) {
    uint64_t numel = 1;
    for (int d = 0; d < dt.ndims; d++) {
        numel *= dt.shapes[d];
    }
    return (dt.ndims == 0) ? 1 : numel;
}

int TensorDumpCollector::export_dump_files(const std::string &output_path) {
    if (collected_.empty()) {
        LOG_WARN("No tensor dump data to export");
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    // Sort by task_id then subtask_id then func_id
    std::sort(collected_.begin(), collected_.end(), [](const DumpedTensor &a, const DumpedTensor &b) {
        if (a.task_id != b.task_id) return a.task_id < b.task_id;
        if (a.subtask_id != b.subtask_id) return a.subtask_id < b.subtask_id;
        if (a.func_id != b.func_id) return a.func_id < b.func_id;
        if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
        if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
        return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
    });

    // Create timestamped output directory
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = {};
    localtime_r(&time_t, &tm);
    std::ostringstream ts;
    ts << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = ts.str();

    std::filesystem::path run_dir = std::filesystem::path(output_path) / ("tensor_dump_" + timestamp);
    std::filesystem::create_directories(run_dir);

    std::string base_name = run_dir.filename().string();

    // Write binary payload file
    std::ofstream bin_file(run_dir / (base_name + ".bin"), std::ios::binary);
    uint64_t bin_offset = 0;
    for (auto &dt : collected_) {
        dt.bin_offset = bin_offset;
        if (!dt.bytes.empty()) {
            bin_file.write(
                reinterpret_cast<const char *>(dt.bytes.data()), static_cast<std::streamsize>(dt.bytes.size())
            );
            bin_offset += dt.bytes.size();
        }
        dt.bytes.clear();  // Free memory after writing
        dt.bytes.shrink_to_fit();
    }
    bin_file.close();

    // Count stats
    uint32_t num_before_dispatch = 0;
    uint32_t num_after_completion = 0;
    uint32_t num_input_tensors = 0;
    uint32_t num_output_tensors = 0;
    uint32_t num_inout_tensors = 0;
    for (const auto &dt : collected_) {
        if (dt.stage == TensorDumpStage::BEFORE_DISPATCH) {
            num_before_dispatch++;
        } else {
            num_after_completion++;
        }
        switch (dt.role) {
        case TensorDumpRole::INPUT:
            num_input_tensors++;
            break;
        case TensorDumpRole::OUTPUT:
            num_output_tensors++;
            break;
        case TensorDumpRole::INOUT:
            num_inout_tensors++;
            break;
        }
    }

    // Write JSON manifest
    LOG_INFO("Writing JSON manifest for %zu tensors...", collected_.size());
    std::ofstream json(run_dir / (base_name + ".json"));
    json << "{\n";
    json << "  \"timestamp\": \"" << timestamp << "\",\n";
    json << "  \"run_dir\": \"" << base_name << "\",\n";
    json << "  \"bin_format\": {\n";
    json << "    \"type\": \"logical_contiguous\",\n";
    json << "    \"byte_order\": \"little_endian\"\n";
    json << "  },\n";
    json << "  \"total_tensors\": " << collected_.size() << ",\n";
    json << "  \"before_dispatch\": " << num_before_dispatch << ",\n";
    json << "  \"after_completion\": " << num_after_completion << ",\n";
    json << "  \"input_tensors\": " << num_input_tensors << ",\n";
    json << "  \"output_tensors\": " << num_output_tensors << ",\n";
    json << "  \"inout_tensors\": " << num_inout_tensors << ",\n";
    json << "  \"truncated_tensors\": " << total_truncated_count_ << ",\n";
    json << "  \"dropped_records\": " << total_dropped_count_ << ",\n";
    json << "  \"dropped_overwrite\": " << total_overwrite_count_ << ",\n";
    json << "  \"bin_file\": \"" << base_name << ".bin\",\n";
    json << "  \"tensors\": [\n";

    bool first_entry = true;
    for (size_t i = 0; i < collected_.size(); i++) {
        const DumpedTensor &dt = collected_[i];
        std::string dtype_name = get_dtype_name_from_raw(dt.dtype);
        uint64_t numel = get_num_elements(dt);

        std::string shape_str = dims_to_string(dt.shapes, dt.ndims);
        std::string raw_shape_str = dims_to_string(dt.raw_shapes, dt.ndims);
        std::string offsets_str = dims_to_string(dt.offsets, dt.ndims);

        if (!first_entry) json << ",\n";
        first_entry = false;

        json << "    {\"task_id\": \"0x" << std::hex << std::setfill('0') << std::setw(16) << dt.task_id << std::dec
             << "\", \"subtask_id\": " << static_cast<uint32_t>(dt.subtask_id) << ", \"func_id\": " << dt.func_id
             << ", \"role\": \"" << tensor_dump_role_name(dt.role) << "\", \"stage\": \""
             << tensor_dump_stage_name(dt.stage) << "\", \"arg_index\": " << dt.arg_index << ", \"dtype\": \""
             << dtype_name << "\", \"is_contiguous\": " << (dt.is_contiguous ? "true" : "false")
             << ", \"shape\": " << shape_str << ", \"raw_shape\": " << raw_shape_str << ", \"offsets\": " << offsets_str
             << ", \"numel\": " << numel << ", \"bin_offset\": " << dt.bin_offset
             << ", \"bin_size\": " << dt.payload_size << ", \"truncated\": " << (dt.truncated ? "true" : "false")
             << ", \"overwritten\": " << (dt.overwritten ? "true" : "false") << "}";
    }

    json << "\n  ]\n}\n";
    json.close();

    auto export_end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(export_end - export_start).count();
    LOG_INFO(
        "Wrote dump files (%zu tensors, %lu bytes payload) to %s (%ldms)", collected_.size(), bin_offset,
        run_dir.c_str(), total_ms
    );

    if (total_truncated_count_ > 0 || total_dropped_count_ > 0 || total_overwrite_count_ > 0) {
        LOG_WARN(
            "Tensor dump anomalies: truncated=%u, dropped_records=%u, overwritten=%u", total_truncated_count_,
            total_dropped_count_, total_overwrite_count_
        );
    }

    // Clear state for potential subsequent runs
    collected_.clear();
    total_dropped_count_ = 0;
    total_truncated_count_ = 0;
    total_overwrite_count_ = 0;
    return 0;
}

int TensorDumpCollector::finalize() {
    if (!is_initialized()) {
        return 0;
    }

    // Free per-thread arena data
    for (auto *ptr : arena_data_dev_) {
        if (ptr != nullptr && free_cb_ != nullptr) {
            free_cb_(ptr);
        }
    }
    arena_data_dev_.clear();

    // Free per-thread arena headers
    for (auto *ptr : arena_headers_dev_) {
        if (ptr != nullptr && free_cb_ != nullptr) {
            free_cb_(ptr);
        }
    }
    arena_headers_dev_.clear();

    // Free per-thread DumpBuffers
    for (auto *ptr : dump_buffers_dev_) {
        if (ptr != nullptr && free_cb_ != nullptr) {
            free_cb_(ptr);
        }
    }
    dump_buffers_dev_.clear();

    // Free setup header
    if (setup_header_dev_ != nullptr && free_cb_ != nullptr) {
        free_cb_(setup_header_dev_);
    }
    setup_header_dev_ = nullptr;

    collected_.clear();
    num_dump_threads_ = 0;
    device_id_ = -1;

    LOG_INFO("TensorDumpCollector finalized");
    return 0;
}
