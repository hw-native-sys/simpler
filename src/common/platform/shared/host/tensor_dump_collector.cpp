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

#include "data_type.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

int TensorDumpCollector::initialize(
    int num_dump_threads, int device_id, const DumpAllocCallback &alloc_cb, DumpRegisterCallback register_cb,
    const DumpFreeCallback &free_cb, const std::string &output_prefix, DumpTensorLevel dump_tensor_level
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("TensorDumpCollector already initialized");
        return -1;
    }

    num_dump_threads_ = num_dump_threads;
    output_prefix_ = output_prefix;
    dump_tensor_level_ = dump_tensor_level;

    // Stash the memory context on the base up-front so alloc_paired_buffer
    // (which reads alloc_cb_/register_cb_/free_cb_/device_id_)
    // sees consistent values during init. shm_host_ stays nullptr until the
    // shm allocation succeeds — that nullptr guard makes a post-failure
    // start(tf) a no-op without further bookkeeping.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: any early return after this point releases the shm
    // region + per-thread arenas + DumpMetaBuffers through the framework's
    // dev→host map. `guard.commit()` runs on the success path before the
    // trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    // Allocate dump shared memory (header + buffer states)
    size_t shm_size = calc_dump_data_size(num_dump_threads);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_paired_buffer(shm_size, &shm_host_local);
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
    header->dump_tensor_level = static_cast<uint32_t>(dump_tensor_level);

    uint64_t arena_size = calc_dump_arena_size();
    header->arena_size_per_thread = arena_size;

    // Allocate per-thread arenas (device + host shadow). Track the dev↔host
    // mapping so on_buffer_collected can pull arena bytes via the framework.
    arenas_.resize(num_dump_threads);
    for (int t = 0; t < num_dump_threads; t++) {
        ArenaInfo &ai = arenas_[t];
        ai.size = arena_size;
        ai.dev_ptr = alloc_paired_buffer(arena_size, &ai.host_ptr);
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
            void *dev_ptr = alloc_paired_buffer(sizeof(DumpMetaBuffer), &host_ptr);
            if (dev_ptr == nullptr) {
                LOG_ERROR("Failed to allocate dump meta buffer %d for thread %d", b, t);
                return -1;
            }
            // alloc_paired_buffer already registered dev→host via the manager.

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
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        shm_dev_local, shm_host_local, shm_size, device_id
    );

    LOG_INFO_V0(
        "Tensor dump initialized: %d threads, arena=%lu MB/thread, %d buffers/thread", num_dump_threads,
        arena_size / (1024 * 1024), PLATFORM_DUMP_BUFFERS_PER_THREAD
    );

    guard.commit();
    return 0;
}

void TensorDumpCollector::start_writer_thread_once() {
    if (writer_started_) return;
    writer_started_ = true;

    // `output_prefix_` is captured at initialize() time and is the per-task
    // uniqueness boundary; the dump dir name is fixed (`<prefix>/args_dump`).
    std::string run_dir_name = "args_dump";
    run_dir_ = std::filesystem::path(output_prefix_) / run_dir_name;
    std::filesystem::create_directories(run_dir_);
    // FULL_JSON_ONLY captures no payload (device sets payload_size == 0), so
    // there is nothing to stream — skip the .bin file rather than leaving a
    // 0-byte artifact next to the manifest.
    if (dump_tensor_level_ != DumpTensorLevel::FULL_JSON_ONLY) {
        bin_file_.open(run_dir_ / "args.bin", std::ios::binary);
    }
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
        // rec is read from device shared memory (untrusted): clamp func_count so a
        // corrupt oversized value can't drive an out-of-bounds read of the
        // fixed-size dt.func_ids[] when the record is serialized later.
        uint8_t func_count = rec.func_count;
        if (func_count > TENSOR_DUMP_MAX_FUNC_IDS) {
            LOG_WARN(
                "Dump collector: func_count %u exceeds max %d (corrupt record?), clamping", func_count,
                TENSOR_DUMP_MAX_FUNC_IDS
            );
            func_count = TENSOR_DUMP_MAX_FUNC_IDS;
        }
        dt.func_count = func_count;
        for (uint8_t f = 0; f < func_count; f++) {
            dt.func_ids[f] = (rec.func_ids[f] == 0xFFFF) ? -1 : static_cast<int32_t>(rec.func_ids[f]);
        }
        dt.arg_index = rec.arg_index;
        dt.role = static_cast<TensorDumpRole>(rec.role);
        dt.stage = static_cast<TensorDumpStage>(rec.stage);
        dt.dtype = rec.dtype;
        dt.ndims = rec.ndims;
        dt.flags = rec.flags;
        dt.kind = static_cast<TensorDumpKind>(rec.kind);
        dt.scalar_value = rec.scalar_value;
        dt.is_contiguous = (rec.is_contiguous != 0);
        dt.truncated = (rec.truncated != 0);
        dt.overwritten = false;
        dt.start_offset = rec.start_offset;
        std::memcpy(dt.shapes, rec.shapes, sizeof(dt.shapes));
        std::memcpy(dt.strides, rec.strides, sizeof(dt.strides));

        if (dt.truncated && ++total_truncated_count_ == 1) {
            LOG_WARN("Tensor dump truncation detected. Increase PLATFORM_DUMP_AVG_TENSOR_BYTES.");
        }

        if (dt.kind == TensorDumpKind::TENSOR && thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
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

        bool has_payload = dt.kind == TensorDumpKind::TENSOR && !dt.overwritten && !dt.bytes.empty();
        dt.bin_offset = has_payload ? next_bin_offset_ : 0;
        if (has_payload) {
            next_bin_offset_ += dt.payload_size;
        }

        // Store metadata-only copy in collected_ (no payload bytes)
        DumpedTensor meta = dt;
        meta.bytes.clear();
        {
            std::scoped_lock<std::mutex> lock(collected_mutex_);
            collected_.push_back(std::move(meta));
        }

        // Enqueue full tensor (with payload) to writer thread
        if (has_payload) {
            {
                std::scoped_lock<std::mutex> lock(write_mutex_);
                write_queue_.push(std::move(dt));
            }
            write_cv_.notify_one();
        }
    }
}

void TensorDumpCollector::on_buffer_collected(const DumpReadyBufferInfo &info) {
    std::scoped_lock<std::mutex> lock(collector_state_mutex_);
    start_writer_thread_once();
    process_dump_buffer(info);
    buffers_collected_++;

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_time_).count() >= 5) {
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_).count();
        LOG_INFO_V0(
            "Collecting: %zu args, %.1f GB written (%lds)", collected_.size(), bytes_written_.load() / 1e9, elapsed_s
        );
        last_progress_time_ = now;
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters: recover un-flushed current buffers + dropped accounting
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
    int recovered_threads = 0;
    // After stop(), a non-zero current_buf_ptr with records means the device
    // never ran dump_tensor_flush for that thread. The common cause is a hang:
    // the AICPU op is reaped by the hardware op-execution timeout (507xxx)
    // before its graceful scheduler-timeout shutdown can flush. The host still
    // holds the buffer and the originating arena, so recover the records here
    // (the same path the poll thread uses for a ready buffer) instead of
    // dropping them — export_dump_files() then writes them like any normally
    // collected buffer, so a hung run still yields its dumped inputs/outputs.
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

        DumpReadyBufferInfo info;
        info.thread_index = static_cast<uint32_t>(t);
        info.dev_buffer_ptr = reinterpret_cast<void *>(cur_ptr);
        info.host_buffer_ptr = host_ptr;
        info.buffer_seq = state->current_buf_seq;
        on_buffer_collected(info);
        recovered_threads++;
        LOG_WARN(
            "Dump reconcile: thread %d had an un-flushed buffer (count=%u) — device flush did not run "
            "(AICPU likely reaped on a hang); recovered the records host-side",
            t, count
        );
    }

    if (dropped_total > 0) {
        LOG_WARN(
            "Dump reconcile: %u records dropped on device side. "
            "Increase PLATFORM_DUMP_BUFFERS_PER_THREAD or PLATFORM_DUMP_READYQUEUE_SIZE.",
            dropped_total
        );
    }
    if (recovered_threads > 0) {
        LOG_WARN(
            "Dump reconcile: recovered un-flushed buffers from %d thread(s) (device-side flush was skipped, "
            "typically an AICPU hang reap)",
            recovered_threads
        );
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

static const char *tensor_dump_kind_name(TensorDumpKind kind) {
    switch (kind) {
    case TensorDumpKind::TENSOR:
        return "tensor";
    case TensorDumpKind::SCALAR:
        return "scalar";
    }
    return "unknown";
}

static void write_scalar_json_value(std::ofstream &json, const DumpedTensor &dt) {
    uint64_t raw = dt.scalar_value;
    if (dt.dtype == static_cast<uint8_t>(DataType::FLOAT32)) {
        float f;
        memcpy(&f, &raw, sizeof(float));
        if (std::isnan(f)) {
            json << ", \"value\": null";
        } else if (std::isinf(f)) {
            json << ", \"value\": " << (f < 0 ? "\"-$Inf\"" : "\"$Inf\"");
        } else {
            std::ostringstream val_ss;
            val_ss << f;
            std::string val_str = val_ss.str();
            if (val_str.find('.') == std::string::npos && val_str.find('e') == std::string::npos) {
                val_str += ".0";
            }
            json << ", \"value\": " << val_str;
        }
    } else if (dt.dtype == static_cast<uint8_t>(DataType::INT32)) {
        int32_t val;
        memcpy(&val, &raw, sizeof(int32_t));
        json << ", \"value\": " << val;
    } else if (dt.dtype == static_cast<uint8_t>(DataType::UINT32)) {
        uint32_t val;
        memcpy(&val, &raw, sizeof(uint32_t));
        json << ", \"value\": " << val;
    } else if (dt.dtype == static_cast<uint8_t>(DataType::BOOL)) {
        json << ", \"value\": " << (raw != 0 ? "true" : "false");
    } else if (dt.dtype == static_cast<uint8_t>(DataType::INT64)) {
        int64_t val;
        memcpy(&val, &raw, sizeof(int64_t));
        json << ", \"value\": " << val;
    } else {
        json << ", \"value\": " << raw;
    }
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
                "Writing to disk: %.1f GB written, %zu args remaining (%lds)", bytes_written_.load() / 1e9,
                write_queue_.size(), elapsed_s
            );
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (bin_file_.is_open()) {
            bin_file_.close();
        }

        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - run_start_time_)
                .count();
        LOG_INFO_V0(
            "Collected %zu args, wrote %.1f GB to disk (%.1fs)", collected_.size(), bytes_written_.load() / 1e9,
            elapsed_ms / 1000.0
        );
    }

    if (collected_.empty()) {
        LOG_WARN("No args dump data to export");
        writer_started_ = false;
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    std::sort(collected_.begin(), collected_.end(), [](const DumpedTensor &a, const DumpedTensor &b) {
        if (a.task_id != b.task_id) return a.task_id < b.task_id;
        if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
        if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
        return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
    });

    LOG_INFO_V0("Writing JSON manifest for %zu args...", collected_.size());

    uint32_t num_before_dispatch = 0;
    uint32_t num_after_completion = 0;
    uint32_t num_input_args = 0;
    uint32_t num_output_args = 0;
    uint32_t num_inout_args = 0;
    for (const auto &dt : collected_) {
        if (dt.stage == TensorDumpStage::BEFORE_DISPATCH) {
            num_before_dispatch++;
        } else {
            num_after_completion++;
        }
        switch (dt.role) {
        case TensorDumpRole::INPUT:
            num_input_args++;
            break;
        case TensorDumpRole::OUTPUT:
            num_output_args++;
            break;
        case TensorDumpRole::INOUT:
            num_inout_args++;
            break;
        }
    }

    std::string run_dir_name = run_dir_.filename().string();
    std::ofstream json(run_dir_ / "args_dump.json");
    json << "{\n";
    json << "  \"run_dir\": \"" << run_dir_name << "\",\n";
    json << "  \"bin_format\": {\n";
    json << "    \"type\": \"logical_contiguous\",\n";
    json << "    \"byte_order\": \"little_endian\"\n";
    json << "  },\n";
    json << "  \"total_args\": " << collected_.size() << ",\n";
    json << "  \"before_dispatch\": " << num_before_dispatch << ",\n";
    json << "  \"after_completion\": " << num_after_completion << ",\n";
    json << "  \"input_args\": " << num_input_args << ",\n";
    json << "  \"output_args\": " << num_output_args << ",\n";
    json << "  \"inout_args\": " << num_inout_args << ",\n";
    json << "  \"truncated_args\": " << total_truncated_count_ << ",\n";
    json << "  \"dropped_records\": " << total_dropped_record_count_ << ",\n";
    json << "  \"dropped_overwrite\": " << total_overwrite_count_ << ",\n";
    if (dump_tensor_level_ == DumpTensorLevel::FULL_JSON_ONLY) {
        json << "  \"bin_file\": null,\n";
    } else {
        json << "  \"bin_file\": \"args.bin\",\n";
    }
    json << "  \"args\": [\n";

    bool first_entry = true;

    for (size_t i = 0; i < collected_.size(); i++) {
        const DumpedTensor &dt = collected_[i];
        std::string dtype_name = get_dtype_name_from_raw(dt.dtype);
        uint64_t numel = get_num_elements(dt);

        std::string shape_str = dims_to_string(dt.shapes, dt.ndims);
        std::string strides_str = dims_to_string(dt.strides, dt.ndims);

        if (!first_entry) json << ",\n";
        first_entry = false;

        json << "    {\"task_id\": \"0x" << std::hex << std::setfill('0') << std::setw(16) << dt.task_id << std::dec
             << "\"";
        json << ", \"func_id\": [";
        for (int32_t f = 0; f < dt.func_count; f++) {
            if (f) json << ", ";
            json << dt.func_ids[f];
        }
        json << "]";
        json << ", \"arg_index\": " << dt.arg_index << ", \"role\": \"" << tensor_dump_role_name(dt.role)
             << "\", \"stage\": \"" << tensor_dump_stage_name(dt.stage) << "\", \"kind\": \""
             << tensor_dump_kind_name(dt.kind) << "\", \"dtype\": \"" << dtype_name << "\"";
        if (dt.kind == TensorDumpKind::SCALAR) {
            write_scalar_json_value(json, dt);
        }
        json << ", \"is_contiguous\": " << (dt.is_contiguous ? "true" : "false") << ", \"shape\": " << shape_str
             << ", \"strides\": " << strides_str << ", \"start_offset\": " << dt.start_offset
             << ", \"numel\": " << numel;
        if ((dt.flags & TENSOR_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS) != 0) {
            json << ", \"arg_index_ambiguous\": true";
        }
        json << ", \"bin_offset\": " << dt.bin_offset << ", \"bin_size\": " << dt.payload_size
             << ", \"truncated\": " << (dt.truncated ? "true" : "false")
             << ", \"overwritten\": " << (dt.overwritten ? "true" : "false") << "}";
    }

    json << "\n  ]\n}\n";
    json.close();

    auto export_end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(export_end - export_start).count();
    LOG_INFO_V0("Wrote JSON manifest (%zu args) to %s (%ldms)", collected_.size(), run_dir_.c_str(), total_ms);

    if (total_truncated_count_ > 0 || total_dropped_record_count_ > 0 || total_overwrite_count_ > 0) {
        LOG_WARN(
            "Args dump anomalies: truncated=%u, dropped_records=%u, overwritten=%u", total_truncated_count_,
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

int TensorDumpCollector::finalize(DumpUnregisterCallback unregister_cb, const DumpFreeCallback &free_cb) {
    if (shm_host_ == nullptr) return 0;

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();

    // ProfilerBase::stop() only joins the mgmt + poll threads. The writer
    // thread is otherwise torn down solely by export_dump_files(), so any path
    // that skips export — e.g. run() bailing on a device error before its
    // collector-teardown block — would leak it: left blocked on write_cv_ with
    // writer_done_ == false while writer_thread_ stays joinable, which trips
    // std::terminate when the collector is destroyed or re-run. finalize() is
    // reached via run()'s perf_cleanup guard on every exit path, so join the
    // writer here too. Idempotent: export_dump_files() clears writer_started_
    // on the success path, making this a no-op.
    if (writer_started_ && writer_thread_.joinable()) {
        writer_done_.store(true);
        write_cv_.notify_one();
        writer_thread_.join();
    }

    // The writer thread opens bin_file_ in start_writer_thread_once() and it is
    // otherwise closed only by export_dump_files(). Close it here too so an
    // export-skipping path does not leave it open — a stale-open stream makes
    // the next run's bin_file_.open() set failbit. Guarded for idempotency.
    if (bin_file_.is_open()) {
        bin_file_.close();
    }

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
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
