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
 * @file args_dump_collector.cpp
 * @brief Host-side args dump collector implementation. The mgmt-thread +
 *        buffer-pool machinery lives in profiling_common::BufferPoolManager
 *        parameterized by DumpModule (host/args_dump_collector.h); the
 *        poll loop lives in profiling_common::ProfilerBase. This file owns
 *        the per-buffer on_buffer_collected callback, arena reads, and disk
 *        export.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. The
 * framework's mgmt loop mirrors the shm region per tick and pulls each
 * popped DumpMetaBuffer's contents on demand. on_buffer_collected pulls
 * the relevant portion of the originating thread's arena before reading
 * args records (arena buffers live outside the shm region).
 */

#include "host/args_dump_collector.h"

#include "data_type.h"

#include <algorithm>
#include <cassert>
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
// ArgsDumpCollector
// =============================================================================

ArgsDumpCollector::~ArgsDumpCollector() { stop(); }

static int64_t steady_clock_ms(std::chrono::steady_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
}

size_t ArgsDumpCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = collected_by_collector_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

void ArgsDumpCollector::reset_collector_shards() {
    const size_t shard_count = static_cast<size_t>(manager_.shard_count());
    collected_.clear();
    collected_by_collector_.assign(shard_count, {});
    collector_counters_.assign(shard_count, {});
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
}

void ArgsDumpCollector::merge_collector_shards() {
    if (collector_shards_merged_) {
        return;
    }

    size_t total_records = 0;
    for (const auto &shard_records : collected_by_collector_) {
        total_records += shard_records.size();
    }

    collected_.clear();
    collected_.reserve(total_records);
    for (const auto &shard_records : collected_by_collector_) {
        collected_.insert(collected_.end(), shard_records.begin(), shard_records.end());
    }
    collector_shards_merged_ = true;
}

int ArgsDumpCollector::initialize(
    int num_dump_threads, int device_id, const DumpAllocCallback &alloc_cb, DumpRegisterCallback register_cb,
    const DumpFreeCallback &free_cb, const std::string &output_prefix, DumpArgsLevel dump_args_level
) {
    if (shm_host_ != nullptr) {
        LOG_ERROR("ArgsDumpCollector already initialized");
        return -1;
    }
    if (num_dump_threads <= 0 || num_dump_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "ArgsDumpCollector::initialize: invalid num_dump_threads=%d (valid range: 1-%d)", num_dump_threads,
            PLATFORM_MAX_AICPU_THREADS
        );
        return -1;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_dump_threads);

    num_dump_threads_ = num_dump_threads;
    output_prefix_ = output_prefix;
    dump_args_level_ = dump_args_level;
    reset_collector_shards();
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    total_overwrite_count_.store(0, std::memory_order_relaxed);
    last_progress_ms_.store(0, std::memory_order_relaxed);

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
    header->magic = ARGS_DUMP_MAGIC;
    header->num_dump_threads = static_cast<uint32_t>(num_dump_threads);
    header->records_per_buffer = PLATFORM_DUMP_RECORDS_PER_BUFFER;
    header->dump_args_level = static_cast<uint32_t>(dump_args_level);

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
                if (!manager_.push_recycled(0, dev_ptr, t)) {
                    (void)manager_.retire_unqueued_buffer(0, dev_ptr, t);
                }
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
        "Args dump initialized: %d threads, arena=%lu MB/thread, %d buffers/thread", num_dump_threads,
        arena_size / (1024 * 1024), PLATFORM_DUMP_BUFFERS_PER_THREAD
    );

    guard.commit();
    return 0;
}

void ArgsDumpCollector::start(const profiling_common::ThreadFactory &thread_factory) {
    if (shm_host_ == nullptr) return;
    reset_collector_shards();
    profiling_common::ProfilerBase<ArgsDumpCollector, DumpModule>::start(thread_factory);
}

void ArgsDumpCollector::start_writer_thread_once() {
    std::scoped_lock<std::mutex> lock(writer_start_mutex_);
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
    if (dump_args_level_ != DumpArgsLevel::FULL_JSON_ONLY) {
        bin_file_.open(run_dir_ / "args.bin", std::ios::binary);
    }
    next_bin_offset_ = 0;

    writer_done_.store(false);
    bytes_written_.store(0);
    run_start_time_ = std::chrono::steady_clock::now();
    last_progress_ms_.store(steady_clock_ms(run_start_time_), std::memory_order_relaxed);

    writer_thread_ = std::thread(&ArgsDumpCollector::writer_loop, this);
}

void ArgsDumpCollector::process_dump_buffer(const DumpReadyBufferInfo &info, int collector_shard) {
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

    const size_t shard = normalize_collector_shard(collector_shard);
    uint64_t records_appended = 0;

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
        const ArgsDumpRecord &rec = buf->records[i];

        DumpedArg dt{};
        dt.task_id = rec.task_id;
        // rec is read from device shared memory (untrusted): clamp func_count so a
        // corrupt oversized value can't drive an out-of-bounds read of the
        // fixed-size dt.func_ids[] when the record is serialized later.
        uint8_t func_count = rec.func_count;
        if (func_count > ARGS_DUMP_MAX_FUNC_IDS) {
            LOG_WARN(
                "Dump collector: func_count %u exceeds max %d (corrupt record?), clamping", func_count,
                ARGS_DUMP_MAX_FUNC_IDS
            );
            func_count = ARGS_DUMP_MAX_FUNC_IDS;
        }
        dt.func_count = func_count;
        for (uint8_t f = 0; f < func_count; f++) {
            dt.func_ids[f] = (rec.func_ids[f] == 0xFFFF) ? -1 : static_cast<int32_t>(rec.func_ids[f]);
        }
        dt.arg_index = rec.arg_index;
        dt.role = static_cast<ArgsDumpRole>(rec.role);
        dt.stage = static_cast<ArgsDumpStage>(rec.stage);
        dt.dtype = rec.dtype;
        uint8_t ndims = rec.ndims;
        if (ndims > PLATFORM_DUMP_MAX_DIMS) {
            LOG_ERROR(
                "Dump collector: ndims %u exceeds max %u (corrupt record?), clamping", static_cast<unsigned>(ndims),
                static_cast<unsigned>(PLATFORM_DUMP_MAX_DIMS)
            );
            ndims = PLATFORM_DUMP_MAX_DIMS;
        }
        dt.ndims = ndims;
        dt.flags = rec.flags;
        dt.kind = static_cast<ArgsDumpKind>(rec.kind);
        dt.scalar_value = rec.scalar_value;
        dt.is_contiguous = (rec.is_contiguous != 0);
        dt.truncated = (rec.truncated != 0);
        dt.overwritten = false;
        dt.start_offset = rec.start_offset;
        std::memcpy(dt.shapes, rec.shapes, sizeof(dt.shapes));
        std::memcpy(dt.strides, rec.strides, sizeof(dt.strides));

        if (dt.truncated && total_truncated_count_.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_WARN("Args dump truncation detected. Increase PLATFORM_DUMP_AVG_TENSOR_BYTES.");
        }

        if (dt.kind == ArgsDumpKind::TENSOR && thread_idx >= 0 && thread_idx < static_cast<int>(arenas_.size())) {
            ArenaInfo &ai = arenas_[thread_idx];
            char *arena_host = reinterpret_cast<char *>(ai.host_ptr);
            uint64_t arena_sz = ai.size;

            uint64_t high_water = ai.high_water;
            if (high_water > arena_sz && rec.payload_offset < high_water - arena_sz) {
                dt.overwritten = true;
                if (total_overwrite_count_.fetch_add(1, std::memory_order_relaxed) == 0) {
                    LOG_WARN(
                        "Args dump overwrite detected: host drain was slower than arena reuse. "
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

        bool has_payload = dt.kind == ArgsDumpKind::TENSOR && !dt.overwritten && !dt.bytes.empty();
        if (has_payload) {
            std::vector<uint8_t> payload = std::move(dt.bytes);
            DumpedArg writer_item = dt;
            writer_item.bytes = std::move(payload);
            {
                std::scoped_lock<std::mutex> lock(write_mutex_);
                dt.bin_offset = next_bin_offset_;
                writer_item.bin_offset = next_bin_offset_;
                next_bin_offset_ += dt.payload_size;
                write_queue_.push(std::move(writer_item));
            }
            collected_by_collector_[shard].push_back(std::move(dt));
            write_cv_.notify_one();
        } else {
            dt.bin_offset = 0;
            dt.bytes.clear();
            collected_by_collector_[shard].push_back(std::move(dt));
        }
        records_appended++;
    }

    if (records_appended > 0) {
        collector_counters_[shard].total_collected += records_appended;
        total_metadata_collected_.fetch_add(records_appended, std::memory_order_relaxed);
    }
}

void ArgsDumpCollector::on_buffer_collected(const DumpReadyBufferInfo &info, int collector_shard) {
    start_writer_thread_once();
    process_dump_buffer(info, collector_shard);

    auto now = std::chrono::steady_clock::now();
    int64_t now_ms = steady_clock_ms(now);
    int64_t last_ms = last_progress_ms_.load(std::memory_order_relaxed);
    if (now_ms - last_ms >= 5000 &&
        last_progress_ms_.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_).count();
        LOG_INFO_V0(
            "Collecting: %lu args, %.1f GB written (%lds)",
            static_cast<unsigned long>(total_metadata_collected_.load(std::memory_order_relaxed)),
            bytes_written_.load() / 1e9, elapsed_s
        );
    }
}

// ---------------------------------------------------------------------------
// reconcile_counters: recover un-flushed current buffers + dropped accounting
// ---------------------------------------------------------------------------

void ArgsDumpCollector::reconcile_counters() {
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
    // never ran dump_args_flush for that thread. The common cause is a hang:
    // the AICPU op is reaped by the hardware op-execution timeout (507xxx)
    // before its graceful scheduler-timeout shutdown can flush. The host still
    // holds the buffer and the originating arena, so recover the records here
    // (the same path the poll thread uses for a ready buffer) instead of
    // dropping them — export_dump_files() then writes them like any normally
    // collected buffer, so a hung run still yields its dumped inputs/outputs.
    for (int t = 0; t < num_dump_threads_; t++) {
        DumpBufferState *state = get_dump_buffer_state(shm_host_, t);

        total_dropped_record_count_.fetch_add(state->dropped_record_count, std::memory_order_relaxed);
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
        on_buffer_collected(info, static_cast<int>(t));
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

static const char *args_dump_role_name(ArgsDumpRole role) {
    switch (role) {
    case ArgsDumpRole::INPUT:
        return "input";
    case ArgsDumpRole::OUTPUT:
        return "output";
    case ArgsDumpRole::INOUT:
        return "inout";
    }
    return "unknown";
}

static const char *args_dump_stage_name(ArgsDumpStage stage) {
    switch (stage) {
    case ArgsDumpStage::BEFORE_DISPATCH:
        return "before_dispatch";
    case ArgsDumpStage::AFTER_COMPLETION:
        return "after_completion";
    }
    return "unknown";
}

static const char *args_dump_kind_name(ArgsDumpKind kind) {
    switch (kind) {
    case ArgsDumpKind::TENSOR:
        return "tensor";
    case ArgsDumpKind::SCALAR:
        return "scalar";
    }
    return "unknown";
}

static void write_scalar_json_value(std::ofstream &json, const DumpedArg &dt) {
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

static uint64_t get_num_elements(const DumpedArg &dt) {
    uint64_t numel = 1;
    for (int d = 0; d < dt.ndims; d++) {
        numel *= dt.shapes[d];
    }
    return (dt.ndims == 0) ? 1 : numel;
}

void ArgsDumpCollector::writer_loop() {
    while (true) {
        DumpedArg dt;
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

int ArgsDumpCollector::export_dump_files() {
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
            "Collected %lu args, wrote %.1f GB to disk (%.1fs)",
            static_cast<unsigned long>(total_metadata_collected_.load(std::memory_order_relaxed)),
            bytes_written_.load() / 1e9, elapsed_ms / 1000.0
        );
    }

    merge_collector_shards();
    if (collected_.empty()) {
        LOG_WARN("No args dump data to export");
        reset_collector_shards();
        total_dropped_record_count_.store(0, std::memory_order_relaxed);
        total_truncated_count_.store(0, std::memory_order_relaxed);
        total_overwrite_count_.store(0, std::memory_order_relaxed);
        writer_started_ = false;
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    std::sort(collected_.begin(), collected_.end(), [](const DumpedArg &a, const DumpedArg &b) {
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
        if (dt.stage == ArgsDumpStage::BEFORE_DISPATCH) {
            num_before_dispatch++;
        } else {
            num_after_completion++;
        }
        switch (dt.role) {
        case ArgsDumpRole::INPUT:
            num_input_args++;
            break;
        case ArgsDumpRole::OUTPUT:
            num_output_args++;
            break;
        case ArgsDumpRole::INOUT:
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
    json << "  \"truncated_args\": " << total_truncated_count_.load(std::memory_order_relaxed) << ",\n";
    json << "  \"dropped_records\": " << total_dropped_record_count_.load(std::memory_order_relaxed) << ",\n";
    json << "  \"dropped_overwrite\": " << total_overwrite_count_.load(std::memory_order_relaxed) << ",\n";
    if (dump_args_level_ == DumpArgsLevel::FULL_JSON_ONLY) {
        json << "  \"bin_file\": null,\n";
    } else {
        json << "  \"bin_file\": \"args.bin\",\n";
    }
    json << "  \"args\": [\n";

    bool first_entry = true;

    for (size_t i = 0; i < collected_.size(); i++) {
        const DumpedArg &dt = collected_[i];
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
        json << ", \"arg_index\": " << dt.arg_index << ", \"role\": \"" << args_dump_role_name(dt.role)
             << "\", \"stage\": \"" << args_dump_stage_name(dt.stage) << "\", \"kind\": \""
             << args_dump_kind_name(dt.kind) << "\", \"dtype\": \"" << dtype_name << "\"";
        if (dt.kind == ArgsDumpKind::SCALAR) {
            write_scalar_json_value(json, dt);
        }
        json << ", \"is_contiguous\": " << (dt.is_contiguous ? "true" : "false") << ", \"shape\": " << shape_str
             << ", \"strides\": " << strides_str << ", \"start_offset\": " << dt.start_offset
             << ", \"numel\": " << numel;
        if ((dt.flags & ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS) != 0) {
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

    uint32_t truncated = total_truncated_count_.load(std::memory_order_relaxed);
    uint32_t dropped = total_dropped_record_count_.load(std::memory_order_relaxed);
    uint32_t overwritten = total_overwrite_count_.load(std::memory_order_relaxed);
    if (truncated > 0 || dropped > 0 || overwritten > 0) {
        LOG_WARN(
            "Args dump anomalies: truncated=%u, dropped_records=%u, overwritten=%u", truncated, dropped, overwritten
        );
    }

    // Clear state so subsequent runs don't accumulate data from previous runs
    collected_.clear();
    collected_by_collector_.clear();
    collector_counters_.clear();
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    total_overwrite_count_.store(0, std::memory_order_relaxed);
    writer_started_ = false;
    for (auto &ai : arenas_) {
        ai.high_water = 0;
    }
    return 0;
}

int ArgsDumpCollector::finalize(DumpUnregisterCallback unregister_cb, const DumpFreeCallback &free_cb) {
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

    // Release framework-owned device allocations (recycled pools,
    // ready_queue, done_queue). Host shadows are freed by clear_mappings().
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
    collected_by_collector_.clear();
    collector_counters_.clear();
    collector_shards_merged_ = false;
    total_metadata_collected_.store(0, std::memory_order_relaxed);
    total_dropped_record_count_.store(0, std::memory_order_relaxed);
    total_truncated_count_.store(0, std::memory_order_relaxed);
    total_overwrite_count_.store(0, std::memory_order_relaxed);
    writer_started_ = false;
    clear_memory_context();

    return 0;
}
