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
 * @brief Host-side tensor dump collector implementation. The mgmt-thread +
 *        buffer-pool machinery lives in profiling_common::BufferPoolManager
 *        parameterized by DumpModule (host/tensor_dump_collector.h); the
 *        poll loop lives in profiling_common::ProfilerBase. This file owns
 *        the per-buffer on_buffer_collected callback, arena reads, and disk
 *        export.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework's mgmt loop mirrors the shm region per tick and pulls each
 * popped DumpMetaBuffer's contents on demand. on_buffer_collected pulls
 * the relevant portion of the originating thread's arena before reading
 * tensor records (arena buffers live outside the shm region).
 */

#include "host/tensor_dump_collector.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
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

TensorDumpCollector::~TensorDumpCollector() { stop(); }

void *TensorDumpCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        if (host_ptr_out) *host_ptr_out = nullptr;
        return nullptr;
    }

    void *host_ptr = nullptr;
    if (register_cb_ != nullptr) {
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("TensorDumpCollector: register failed: %d", rc);
            free_cb_(dev_ptr, user_data_);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
    } else {
        // a5 default: malloc a paired shadow, zero it, push zeros to device
        // so the device buffer starts in a known state.
        host_ptr = std::malloc(size);
        if (host_ptr == nullptr) {
            LOG_ERROR("TensorDumpCollector: host shadow alloc failed for %zu bytes", size);
            free_cb_(dev_ptr, user_data_);
            if (host_ptr_out) *host_ptr_out = nullptr;
            return nullptr;
        }
        std::memset(host_ptr, 0, size);
        profiling_copy_to_device(dev_ptr, host_ptr, size);
    }

    if (host_ptr_out) *host_ptr_out = host_ptr;
    return dev_ptr;
}

int TensorDumpCollector::initialize(
    int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
    DumpFreeCallback free_cb, void *user_data, const std::string &output_prefix
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("TensorDumpCollector already initialized");
        return -1;
    }

    num_dump_threads_ = num_dump_threads;
    output_prefix_ = output_prefix;

    // Stash the memory context on the base up-front so alloc_single_buffer
    // (which reads alloc_cb_/register_cb_/free_cb_/user_data_/device_id_)
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — that nullptr guard makes a post-failure
    // start(tf) a no-op without further bookkeeping.
    set_memory_context(
        alloc_cb, register_cb, free_cb, user_data, /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // Allocate dump shared memory (header + buffer states)
    size_t shm_size = calc_dump_data_size(num_dump_threads);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_single_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("Failed to allocate dump shared memory (%zu bytes)", shm_size);
        return -1;
    }

    // Initialize header on host shadow
    std::memset(shm_host_local, 0, shm_size);
    DumpDataHeader *header = get_dump_header(shm_host_local);
    header->magic = TENSOR_DUMP_MAGIC;
    header->num_dump_threads = static_cast<uint32_t>(num_dump_threads);
    header->records_per_buffer = PLATFORM_DUMP_RECORDS_PER_BUFFER;

    uint64_t arena_size = calc_dump_arena_size();
    header->arena_size_per_thread = arena_size;

    // Allocate per-thread arenas (device + host shadow). Track the dev↔host
    // mapping so on_buffer_collected can pull arena bytes via the framework.
    arenas_.resize(num_dump_threads);
    for (int t = 0; t < num_dump_threads; t++) {
        ArenaInfo &ai = arenas_[t];
        ai.size = arena_size;
        ai.dev_ptr = alloc_single_buffer(arena_size, &ai.host_ptr);
        if (ai.dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate dump arena for thread %d (%lu bytes)", t, arena_size);
            return -1;
        }

        DumpBufferState *state = get_dump_buffer_state(shm_host_local, t);
        state->arena_base = reinterpret_cast<uint64_t>(ai.dev_ptr);
        state->arena_size = arena_size;
        state->arena_write_offset = 0;
        state->dropped_record_count = 0;

        LOG_INFO_V0(
            "Thread %d: dump arena allocated (dev=%p, host=%p, size=%lu MB)", t, ai.dev_ptr, ai.host_ptr,
            arena_size / (1024 * 1024)
        );
    }

    // Allocate initial DumpMetaBuffers and push into free_queues
    for (int t = 0; t < num_dump_threads; t++) {
        DumpBufferState *state = get_dump_buffer_state(shm_host_local, t);

        for (int b = 0; b < PLATFORM_DUMP_BUFFERS_PER_THREAD; b++) {
            void *host_ptr = nullptr;
            void *dev_ptr = alloc_single_buffer(sizeof(DumpMetaBuffer), &host_ptr);
            if (dev_ptr == nullptr) {
                LOG_ERROR("Failed to allocate dump meta buffer %d for thread %d", b, t);
                return -1;
            }

            manager_.register_mapping(dev_ptr, host_ptr);

            if (b < PLATFORM_DUMP_SLOT_COUNT) {
                uint32_t tail = state->free_queue.tail;
                state->free_queue.buffer_ptrs[tail % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                state->free_queue.tail = tail + 1;
            } else {
                manager_.push_recycled(0, dev_ptr);
            }
        }
    }

    // Push the entire initialized shm region (header + BufferStates +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    // Publish shm pointers on the base now that the region is ready. start(tf)
    // gates on shm_host_ being non-null, so this re-set_memory_context call
    // is the moment the collector becomes startable.
    dump_shared_mem_dev_ = shm_dev_local;
    set_memory_context(alloc_cb, register_cb, free_cb, user_data, shm_dev_local, shm_host_local, shm_size, device_id);

    LOG_INFO_V0(
        "Tensor dump initialized: %d threads, arena=%lu MB/thread, %d buffers/thread", num_dump_threads,
        arena_size / (1024 * 1024), PLATFORM_DUMP_BUFFERS_PER_THREAD
    );

    return 0;
}

void TensorDumpCollector::start_writer_thread_once() {
    if (writer_started_) return;
    writer_started_ = true;

    // `output_prefix_` is captured at initialize() time and is the per-task
    // uniqueness boundary; the dump dir name is fixed (`<prefix>/tensor_dump`).
    std::string base_name = "tensor_dump";
    run_dir_ = std::filesystem::path(output_prefix_) / base_name;
    std::filesystem::create_directories(run_dir_);
    bin_file_.open(run_dir_ / (base_name + ".bin"), std::ios::binary);
    next_bin_offset_ = 0;

    writer_done_.store(false);
    bytes_written_.store(0);
    run_start_time_ = std::chrono::steady_clock::now();
    last_progress_time_ = run_start_time_;
    buffers_collected_ = 0;

    writer_thread_ = std::thread(&TensorDumpCollector::writer_loop, this);
}

void TensorDumpCollector::process_dump_buffer(const DumpReadyBufferInfo &info) {
    DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(info.host_buffer_ptr);
    uint32_t count = buf->count;

    if (count == 0) return;

    if (count > PLATFORM_DUMP_RECORDS_PER_BUFFER) {
        LOG_ERROR(
            "Dump collector: invalid record count %u in buffer (thread=%u, seq=%u, max=%d), skipping", count,
            info.thread_index, info.buffer_seq, PLATFORM_DUMP_RECORDS_PER_BUFFER
        );
        return;
    }

    // a5: pull the relevant portion of the originating thread's arena from
    // device. arena_write_offset was mirrored into shm_host_ at the top of
    // the mgmt tick that produced this entry, so it is safe to read here.
    int thread_idx = static_cast<int>(info.thread_index);
    if (thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
        ArenaInfo &ai = arenas_[thread_idx];
        DumpBufferState *state = get_dump_buffer_state(shm_host_, thread_idx);
        uint64_t write_offset = state->arena_write_offset;
        uint64_t bytes_to_copy = (write_offset < ai.size) ? write_offset : ai.size;
        if (bytes_to_copy > 0) {
            profiling_copy_from_device(ai.host_ptr, ai.dev_ptr, bytes_to_copy);
        }
    }

    for (uint32_t i = 0; i < count; i++) {
        const TensorDumpRecord &rec = buf->records[i];

        DumpedTensor dt;
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
        dt.overwritten = false;
        if (dt.truncated && ++total_truncated_count_ == 1) {
            LOG_WARN("Tensor dump truncation detected. Increase PLATFORM_DUMP_AVG_TENSOR_BYTES.");
        }
        std::memcpy(dt.raw_shapes, rec.raw_shapes, sizeof(dt.raw_shapes));
        std::memcpy(dt.shapes, rec.shapes, sizeof(dt.shapes));
        std::memcpy(dt.offsets, rec.offsets, sizeof(dt.offsets));

        if (thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
            ArenaInfo &ai = arenas_[thread_idx];
            char *arena_host = reinterpret_cast<char *>(ai.host_ptr);
            uint64_t arena_sz = ai.size;

            uint64_t high_water = ai.high_water;
            if (high_water > arena_sz && rec.payload_offset < high_water - arena_sz) {
                dt.overwritten = true;
                if (++total_overwrite_count_ == 1) {
                    LOG_WARN(
                        "Tensor dump overwrite detected: host drain was slower than arena reuse. "
                        "Increase PLATFORM_DUMP_BUFFERS_PER_THREAD."
                    );
                }
            } else {
                dt.overwritten = false;
            }

            if (!dt.overwritten && rec.payload_size > 0) {
                dt.bytes.resize(rec.payload_size);
                uint64_t pos = rec.payload_offset % arena_sz;
                if (pos + rec.payload_size <= arena_sz) {
                    std::memcpy(dt.bytes.data(), arena_host + pos, rec.payload_size);
                } else {
                    uint64_t first = arena_sz - pos;
                    std::memcpy(dt.bytes.data(), arena_host + pos, first);
                    std::memcpy(dt.bytes.data() + first, arena_host, rec.payload_size - first);
                }
            }

            uint64_t end_offset = rec.payload_offset + rec.payload_size;
            if (end_offset > ai.high_water) {
                ai.high_water = end_offset;
            }
        }

        dt.payload_size = dt.bytes.size();

        bool has_payload = !dt.overwritten && !dt.bytes.empty();
        dt.bin_offset = has_payload ? next_bin_offset_ : 0;
        if (has_payload) {
            next_bin_offset_ += dt.payload_size;
        }

        // Store metadata-only copy in collected_ (no payload bytes)
        DumpedTensor meta = dt;
        meta.bytes.clear();
        {
            std::lock_guard<std::mutex> lock(collected_mutex_);
            collected_.push_back(std::move(meta));
        }

        // Enqueue full tensor (with payload) to writer thread
        if (has_payload) {
            {
                std::lock_guard<std::mutex> lock(write_mutex_);
                write_queue_.push(std::move(dt));
            }
            write_cv_.notify_one();
        }
    }
}

void TensorDumpCollector::on_buffer_collected(const DumpReadyBufferInfo &info) {
    start_writer_thread_once();
    process_dump_buffer(info);
    buffers_collected_++;

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_time_).count() >= 5) {
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_).count();
        LOG_INFO_V0(
            "Collecting: %zu tensors, %.1f GB written (%lds)", collected_.size(), bytes_written_.load() / 1e9, elapsed_s
        );
        last_progress_time_ = now;
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters: passive sanity-check + dropped accounting
// ---------------------------------------------------------------------------

void TensorDumpCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;

    // Pull the latest BufferStates (current_buf_ptr, dropped_record_count)
    // before the per-thread loop so leftovers reflect post-stop() device
    // state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();

    uint32_t dropped_total = 0;
    int leftover_active = 0;
    // After stop(), dump_tensor_flush should have either enqueued the
    // active buffer (success → current_buf_ptr=0) or counted it as dropped
    // and cleared it. A non-zero pointer with non-zero count means records
    // AICPU neither delivered nor accounted for — a device-side flush bug.
    for (int t = 0; t < num_dump_threads_; t++) {
        DumpBufferState *state = get_dump_buffer_state(shm_host_, t);

        total_dropped_record_count_ += state->dropped_record_count;
        dropped_total += state->dropped_record_count;

        uint64_t cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) continue;

        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(cur_ptr));
        if (host_ptr == nullptr) continue;

        profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(cur_ptr), sizeof(DumpMetaBuffer));
        uint32_t count = reinterpret_cast<DumpMetaBuffer *>(host_ptr)->count;
        if (count == 0) continue;

        LOG_ERROR(
            "Dump reconcile: thread %d has un-flushed buffer (current_buf_ptr=0x%lx, count=%u) after "
            "stop() — device flush failed",
            t, static_cast<unsigned long>(cur_ptr), count
        );
        leftover_active++;
    }

    if (dropped_total > 0) {
        LOG_WARN(
            "Dump reconcile: %u records dropped on device side. "
            "Increase PLATFORM_DUMP_BUFFERS_PER_THREAD or PLATFORM_DUMP_READYQUEUE_SIZE.",
            dropped_total
        );
    }
    if (leftover_active > 0) {
        LOG_ERROR("Dump reconcile: %d thread(s) had un-cleared current_buf_ptr — see prior errors", leftover_active);
    }
}

// ---------------------------------------------------------------------------
// Writer thread + export
// ---------------------------------------------------------------------------

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
        if (d > 0) ss << ", ";
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

void TensorDumpCollector::writer_loop() {
    while (true) {
        DumpedTensor dt;
        {
            std::unique_lock<std::mutex> lock(write_mutex_);
            write_cv_.wait(lock, [this] {
                return !write_queue_.empty() || writer_done_.load();
            });
            if (write_queue_.empty() && writer_done_.load()) {
                break;
            }
            dt = std::move(write_queue_.front());
            write_queue_.pop();
        }

        if (!dt.bytes.empty()) {
            bin_file_.write(
                reinterpret_cast<const char *>(dt.bytes.data()), static_cast<std::streamsize>(dt.bytes.size())
            );
        }

        bytes_written_ += dt.bytes.size();
    }
}

int TensorDumpCollector::export_dump_files() {
    // Stop the writer thread (started lazily in on_buffer_collected). Safe
    // to skip when writer_started_ is false (collector ran but produced no
    // buffers, or never started at all).
    if (writer_started_) {
        writer_done_.store(true);
        write_cv_.notify_one();
        while (writer_thread_.joinable()) {
            if (write_queue_.empty()) {
                writer_thread_.join();
                break;
            }
            auto elapsed_s =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - run_start_time_)
                    .count();
            LOG_INFO_V0(
                "Writing to disk: %.1f GB written, %zu tensors remaining (%lds)", bytes_written_.load() / 1e9,
                write_queue_.size(), elapsed_s
            );
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        bin_file_.close();

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - run_start_time_)
                .count();
        LOG_INFO_V0(
            "Collected %zu tensors, wrote %.1f GB to disk (%.1fs)", collected_.size(), bytes_written_.load() / 1e9,
            elapsed_ms / 1000.0
        );
    }

    if (collected_.empty()) {
        LOG_WARN("No tensor dump data to export");
        writer_started_ = false;
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    std::sort(collected_.begin(), collected_.end(), [](const DumpedTensor &a, const DumpedTensor &b) {
        if (a.task_id != b.task_id) return a.task_id < b.task_id;
        if (a.subtask_id != b.subtask_id) return a.subtask_id < b.subtask_id;
        if (a.func_id != b.func_id) return a.func_id < b.func_id;
        if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
        if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
        return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
    });

    LOG_INFO_V0("Writing JSON manifest for %zu tensors...", collected_.size());

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

    std::string base_name = run_dir_.filename().string();
    std::ofstream json(run_dir_ / (base_name + ".json"));
    json << "{\n";
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
    json << "  \"dropped_records\": " << total_dropped_record_count_ << ",\n";
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
    LOG_INFO_V0("Wrote JSON manifest (%zu tensors) to %s (%ldms)", collected_.size(), run_dir_.c_str(), total_ms);

    if (total_truncated_count_ > 0 || total_dropped_record_count_ > 0 || total_overwrite_count_ > 0) {
        LOG_WARN(
            "Tensor dump anomalies: truncated=%u, dropped_records=%u, overwritten=%u", total_truncated_count_,
            total_dropped_record_count_, total_overwrite_count_
        );
    }

    // Clear state so subsequent runs don't accumulate data from previous runs
    collected_.clear();
    total_dropped_record_count_ = 0;
    total_truncated_count_ = 0;
    total_overwrite_count_ = 0;
    writer_started_ = false;
    for (auto &ai : arenas_) {
        ai.high_water = 0;
    }
    return 0;
}

int TensorDumpCollector::finalize(DumpUnregisterCallback unregister_cb, DumpFreeCallback free_cb, void *user_data) {
    if (shm_host_ == nullptr) return 0;

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    auto release_dev = [&](void *p) {
        if (p == nullptr) return;
        if (unregister_cb != nullptr) {
            unregister_cb(p, device_id_);
        }
        if (free_cb != nullptr) {
            free_cb(p, user_data);
        }
    };

    // Free DumpMetaBuffers still in per-thread free_queues / current_buf_ptr.
    // These are owned by AICPU at runtime; the framework tracks them via
    // dev_to_host_ but doesn't enumerate them in release_owned_buffers.
    // Release the device pointer only — the paired host shadow stays in
    // dev_to_host_ and is freed by clear_mappings() below.
    if (shm_host_ != nullptr) {
        for (int t = 0; t < num_dump_threads_; t++) {
            DumpBufferState *state = get_dump_buffer_state(shm_host_, t);

            release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
            state->current_buf_ptr = 0;

            rmb();
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_DUMP_SLOT_COUNT) {
                queued = PLATFORM_DUMP_SLOT_COUNT;
            }
            for (uint32_t i = 0; i < queued; i++) {
                uint32_t slot = (head + i) % PLATFORM_DUMP_SLOT_COUNT;
                release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }

    // Release framework-owned buffers (recycled pools, ready_queue,
    // done_queue). release_owned_buffers also frees the paired host shadows
    // for these (and erases their mappings).
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
    });

    // Free arenas (device only — shadows tracked in dev_to_host_).
    for (auto &ai : arenas_) {
        if (ai.dev_ptr != nullptr) {
            release_dev(ai.dev_ptr);
            ai.dev_ptr = nullptr;
            ai.host_ptr = nullptr;
        }
    }
    arenas_.clear();

    // Free shared memory region (device only — shadow stays in
    // dev_to_host_ until clear_mappings).
    if (dump_shared_mem_dev_ != nullptr) {
        release_dev(dump_shared_mem_dev_);
        dump_shared_mem_dev_ = nullptr;
    }

    // Free remaining host shadows: per-state buffers + arenas + shm region.
    manager_.clear_mappings();

    // Reset state
    num_dump_threads_ = 0;
    collected_.clear();
    total_dropped_record_count_ = 0;
    total_truncated_count_ = 0;
    total_overwrite_count_ = 0;
    writer_started_ = false;
    clear_memory_context();

    return 0;
}
