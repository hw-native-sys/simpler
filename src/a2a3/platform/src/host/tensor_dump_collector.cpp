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
 * @brief Host-side tensor dump collector implementation
 *
 * - DumpMemoryManager: background thread polling dump ready queues
 * - TensorDumpCollector: lifecycle management, arena reads, file export
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
// DumpMemoryManager
// =============================================================================

DumpMemoryManager::~DumpMemoryManager() {
    if (running_.load()) {
        stop();
    }
}

void DumpMemoryManager::register_mapping(void *dev_ptr, void *host_ptr) { dev_to_host_[dev_ptr] = host_ptr; }

void *DumpMemoryManager::resolve_host_ptr(void *dev_ptr) {
    auto it = dev_to_host_.find(dev_ptr);
    if (it != dev_to_host_.end()) {
        return it->second;
    }
    // Simulation mode: dev == host
    return dev_ptr;
}

void *DumpMemoryManager::alloc_and_register(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        return nullptr;
    }

    void *host_ptr = dev_ptr;  // Default for simulation
    if (register_cb_ != nullptr) {
        int rc = register_cb_(dev_ptr, size, device_id_, user_data_, &host_ptr);
        if (rc != 0) {
            free_cb_(dev_ptr, user_data_);
            return nullptr;
        }
    }

    dev_to_host_[dev_ptr] = host_ptr;
    if (host_ptr_out) {
        *host_ptr_out = host_ptr;
    }
    return dev_ptr;
}

void DumpMemoryManager::free_buffer(void *dev_ptr) {
    if (dev_ptr && free_cb_) {
        free_cb_(dev_ptr, user_data_);
    }
    dev_to_host_.erase(dev_ptr);
}

void DumpMemoryManager::process_dump_entry(
    DumpDataHeader * /*header*/, int thread_idx, const DumpReadyQueueEntry &entry
) {
    void *dev_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
    void *host_ptr = resolve_host_ptr(dev_ptr);

    DumpReadyBufferInfo info;
    info.thread_index = entry.thread_index;
    info.dev_buffer_ptr = dev_ptr;
    info.host_buffer_ptr = host_ptr;
    info.buffer_seq = entry.buffer_seq;

    {
        std::lock_guard<std::mutex> lock(ready_mutex_);
        ready_queue_.push(info);
    }
    ready_cv_.notify_one();

    // Replenish: fill free_queue to capacity
    DumpBufferState *state = get_dump_buffer_state(shared_mem_host_, thread_idx);
    rmb();
    uint32_t fq_head = state->free_queue.head;
    uint32_t fq_tail = state->free_queue.tail;
    uint32_t fq_used = fq_tail - fq_head;

    while (fq_used < PLATFORM_DUMP_SLOT_COUNT) {
        void *new_dev = nullptr;
        if (!recycled_dump_buffers_.empty()) {
            new_dev = recycled_dump_buffers_.back();
            recycled_dump_buffers_.pop_back();
        } else {
            // Batch-allocate to refill recycled pool, avoiding per-buffer alloc overhead
            int batch = PLATFORM_DUMP_BUFFERS_PER_THREAD - PLATFORM_DUMP_SLOT_COUNT;
            if (batch < 1) {
                batch = 1;
            }
            for (int i = 0; i < batch; i++) {
                void *host = nullptr;
                void *dev = alloc_and_register(sizeof(DumpMetaBuffer), &host);
                if (dev == nullptr) {
                    break;
                }
                recycled_dump_buffers_.push_back(dev);
            }
            if (!recycled_dump_buffers_.empty()) {
                new_dev = recycled_dump_buffers_.back();
                recycled_dump_buffers_.pop_back();
            }
        }
        if (new_dev == nullptr) {
            break;
        }
        state->free_queue.buffer_ptrs[fq_tail % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(new_dev);
        wmb();
        fq_tail++;
        state->free_queue.tail = fq_tail;
        wmb();
        fq_used++;
    }
}

void DumpMemoryManager::mgmt_loop() {
    if (set_device_cb_ != nullptr) {
        set_device_cb_(device_id_, user_data_);
    }

    DumpDataHeader *header = get_dump_header(shared_mem_host_);
    uint64_t total_entries_processed = 0;
    uint64_t total_replenished = 0;

    while (running_.load()) {
        bool did_work = false;

        // Poll all threads' ready queues
        for (int t = 0; t < num_dump_threads_; t++) {
            rmb();
            uint32_t head = header->queue_heads[t];
            uint32_t tail = header->queue_tails[t];

            while (head != tail) {
                DumpReadyQueueEntry &entry = header->queues[t][head];
                process_dump_entry(header, t, entry);
                head = (head + 1) % PLATFORM_DUMP_READYQUEUE_SIZE;
                header->queue_heads[t] = head;
                wmb();
                did_work = true;
                total_entries_processed++;
                tail = header->queue_tails[t];
            }
        }

        // Proactively replenish free queues from recycled pool even when no new ready entries
        {
            std::lock_guard<std::mutex> lock(done_mutex_);
            while (!done_queue_.empty()) {
                void *dev_ptr = done_queue_.front();
                done_queue_.pop();
                recycled_dump_buffers_.push_back(dev_ptr);
                did_work = true;
            }
        }

        // Push recycled buffers into free queues that have space
        for (int t = 0; t < num_dump_threads_ && !recycled_dump_buffers_.empty(); t++) {
            DumpBufferState *state = get_dump_buffer_state(shared_mem_host_, t);
            rmb();
            uint32_t fq_head = state->free_queue.head;
            uint32_t fq_tail = state->free_queue.tail;
            uint32_t fq_used = fq_tail - fq_head;

            while (fq_used < PLATFORM_DUMP_SLOT_COUNT && !recycled_dump_buffers_.empty()) {
                void *new_dev = recycled_dump_buffers_.back();
                recycled_dump_buffers_.pop_back();
                state->free_queue.buffer_ptrs[fq_tail % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(new_dev);
                wmb();
                fq_tail++;
                state->free_queue.tail = fq_tail;
                wmb();
                fq_used++;
                total_replenished++;
                did_work = true;
            }
        }

        if (!did_work) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    LOG_DEBUG("Dump memory manager: %lu ready entries, %lu replenished", total_entries_processed, total_replenished);
}

void DumpMemoryManager::start(
    void *shared_mem_host, int num_dump_threads, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
    DumpFreeCallback free_cb, void *user_data, int device_id, DumpSetDeviceCallback set_device_cb
) {
    shared_mem_host_ = shared_mem_host;
    num_dump_threads_ = num_dump_threads;
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    user_data_ = user_data;
    device_id_ = device_id;
    set_device_cb_ = set_device_cb;

    LOG_INFO("Starting dump memory manager (device=%d, threads=%d)", device_id, num_dump_threads);
    running_.store(true);
    mgmt_thread_ = std::thread(&DumpMemoryManager::mgmt_loop, this);
}

void DumpMemoryManager::stop() {
    running_.store(false);
    if (mgmt_thread_.joinable()) {
        mgmt_thread_.join();
    }
}

bool DumpMemoryManager::try_pop_ready(DumpReadyBufferInfo &info) {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    if (ready_queue_.empty()) {
        return false;
    }
    info = ready_queue_.front();
    ready_queue_.pop();
    return true;
}

bool DumpMemoryManager::wait_pop_ready(DumpReadyBufferInfo &info, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    if (ready_cv_.wait_for(lock, timeout, [this] {
            return !ready_queue_.empty();
        })) {
        info = ready_queue_.front();
        ready_queue_.pop();
        return true;
    }
    return false;
}

void DumpMemoryManager::notify_copy_done(void *dev_buffer_ptr) {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_queue_.push(dev_buffer_ptr);
}

// =============================================================================
// TensorDumpCollector
// =============================================================================

TensorDumpCollector::~TensorDumpCollector() {
    if (memory_manager_.is_running()) {
        memory_manager_.stop();
    }
}

void *TensorDumpCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        return nullptr;
    }

    void *host_ptr = dev_ptr;
    if (register_cb_ != nullptr) {
        int rc = register_cb_(dev_ptr, size, device_id_, user_data_, &host_ptr);
        if (rc != 0) {
            free_cb_(dev_ptr, user_data_);
            return nullptr;
        }
    }

    if (host_ptr_out) {
        *host_ptr_out = host_ptr;
    }
    return dev_ptr;
}

int TensorDumpCollector::initialize(
    int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpRegisterCallback register_cb,
    DumpFreeCallback free_cb, void *user_data, DumpSetDeviceCallback set_device_cb
) {
    num_dump_threads_ = num_dump_threads;
    device_id_ = device_id;
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    user_data_ = user_data;
    set_device_cb_ = set_device_cb;

    // Allocate dump shared memory (header + buffer states)
    size_t shm_size = calc_dump_data_size(num_dump_threads);
    dump_shared_mem_dev_ = alloc_single_buffer(shm_size, &dump_shared_mem_host_);
    if (dump_shared_mem_dev_ == nullptr) {
        LOG_ERROR("Failed to allocate dump shared memory (%zu bytes)", shm_size);
        return -1;
    }
    was_registered_ = (register_cb != nullptr);

    // Initialize header
    memset(dump_shared_mem_host_, 0, shm_size);
    DumpDataHeader *header = get_dump_header(dump_shared_mem_host_);
    header->magic = TENSOR_DUMP_MAGIC;
    header->num_dump_threads = static_cast<uint32_t>(num_dump_threads);
    header->records_per_buffer = PLATFORM_DUMP_RECORDS_PER_BUFFER;

    uint64_t arena_size = calc_dump_arena_size();
    header->arena_size_per_thread = arena_size;

    // Allocate per-thread arenas
    arenas_.resize(num_dump_threads);
    for (int t = 0; t < num_dump_threads; t++) {
        ArenaInfo &ai = arenas_[t];
        ai.size = arena_size;
        ai.dev_ptr = alloc_single_buffer(arena_size, &ai.host_ptr);
        if (ai.dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate dump arena for thread %d (%lu bytes)", t, arena_size);
            return -1;
        }

        // Set arena info in buffer state
        DumpBufferState *state = get_dump_buffer_state(dump_shared_mem_host_, t);
        state->arena_base = reinterpret_cast<uint64_t>(ai.dev_ptr);
        state->arena_size = arena_size;
        state->arena_write_offset = 0;
        state->dropped_record_count = 0;

        LOG_INFO(
            "Thread %d: dump arena allocated (dev=%p, host=%p, size=%lu MB)", t, ai.dev_ptr, ai.host_ptr,
            arena_size / (1024 * 1024)
        );
    }

    // Allocate initial DumpMetaBuffers and push into free_queues
    for (int t = 0; t < num_dump_threads; t++) {
        DumpBufferState *state = get_dump_buffer_state(dump_shared_mem_host_, t);

        for (int b = 0; b < PLATFORM_DUMP_BUFFERS_PER_THREAD; b++) {
            void *host_ptr = nullptr;
            void *dev_ptr = alloc_single_buffer(sizeof(DumpMetaBuffer), &host_ptr);
            if (dev_ptr == nullptr) {
                LOG_ERROR("Failed to allocate dump meta buffer %d for thread %d", b, t);
                return -1;
            }

            memory_manager_.register_mapping(dev_ptr, host_ptr);

            if (b < PLATFORM_DUMP_SLOT_COUNT) {
                // Push into SPSC free_queue
                uint32_t tail = state->free_queue.tail;
                state->free_queue.buffer_ptrs[tail % PLATFORM_DUMP_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                state->free_queue.tail = tail + 1;
            } else {
                // Remaining go to recycled pool
                memory_manager_.recycled_dump_buffers_.push_back(dev_ptr);
            }
        }
    }

    LOG_INFO(
        "Tensor dump initialized: %d threads, arena=%lu MB/thread, %d buffers/thread", num_dump_threads,
        arena_size / (1024 * 1024), PLATFORM_DUMP_BUFFERS_PER_THREAD
    );

    return 0;
}

void TensorDumpCollector::start_memory_manager() {
    execution_complete_.store(false);
    memory_manager_.start(
        dump_shared_mem_host_, num_dump_threads_, alloc_cb_, register_cb_, free_cb_, user_data_, device_id_,
        set_device_cb_
    );
}

void TensorDumpCollector::process_dump_buffer(const DumpReadyBufferInfo &info) {
    // Track processed buffer pointers to prevent double-processing
    // (flush + drain can deliver a buffer that scan_remaining also sees)
    if (processed_buffers_.count(info.dev_buffer_ptr)) {
        return;
    }
    processed_buffers_.insert(info.dev_buffer_ptr);

    DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(info.host_buffer_ptr);
    uint32_t count = buf->count;

    if (count == 0) {
        return;
    }

    if (count > PLATFORM_DUMP_RECORDS_PER_BUFFER) {
        LOG_ERROR(
            "Dump collector: invalid record count %u in buffer (thread=%u, seq=%u, max=%d), skipping", count,
            info.thread_index, info.buffer_seq, PLATFORM_DUMP_RECORDS_PER_BUFFER
        );
        return;
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
        memcpy(dt.raw_shapes, rec.raw_shapes, sizeof(dt.raw_shapes));
        memcpy(dt.shapes, rec.shapes, sizeof(dt.shapes));
        memcpy(dt.offsets, rec.offsets, sizeof(dt.offsets));

        // Read tensor data from arena
        int thread_idx = static_cast<int>(info.thread_index);
        if (thread_idx < static_cast<int>(arenas_.size())) {
            ArenaInfo &ai = arenas_[thread_idx];
            char *arena_host = reinterpret_cast<char *>(ai.host_ptr);
            uint64_t arena_sz = ai.size;

            // Check if data was overwritten (offset too old)
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
                    memcpy(dt.bytes.data(), arena_host + pos, rec.payload_size);
                } else {
                    // Wraparound read
                    uint64_t first = arena_sz - pos;
                    memcpy(dt.bytes.data(), arena_host + pos, first);
                    memcpy(dt.bytes.data() + first, arena_host, rec.payload_size - first);
                }
            }

            // Update high-water mark
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

void TensorDumpCollector::poll_and_collect() {
    const auto wait_timeout = std::chrono::milliseconds(100);
    const auto idle_timeout = std::chrono::seconds(PLATFORM_DUMP_TIMEOUT_SECONDS);
    uint64_t buffers_collected = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_progress_time = start_time;
    bool idle_timer_started = false;
    std::chrono::steady_clock::time_point idle_start;

    // Create output directory and start writer thread
    auto now_wall = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now_wall);
    struct tm tm_now;
    localtime_r(&time_t_now, &tm_now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_now);

    std::string base_name = std::string("tensor_dump_") + ts;
    run_dir_ = std::filesystem::path("outputs") / base_name;
    std::filesystem::create_directories(run_dir_);
    bin_file_.open(run_dir_ / (base_name + ".bin"), std::ios::binary);
    next_bin_offset_ = 0;

    writer_done_.store(false);
    bytes_written_.store(0);
    writer_thread_ = std::thread(&TensorDumpCollector::writer_loop, this);

    while (true) {
        DumpReadyBufferInfo info;
        if (memory_manager_.try_pop_ready(info)) {
            process_dump_buffer(info);
            memory_manager_.notify_copy_done(info.dev_buffer_ptr);
            buffers_collected++;
            idle_timer_started = false;

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_time).count() >= 5) {
                auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                LOG_INFO(
                    "Collecting: %zu tensors, %.1f GB written (%lds)", collected_.size(), bytes_written_.load() / 1e9,
                    elapsed_s
                );
                last_progress_time = now;
            }
        } else {
            if (!memory_manager_.wait_pop_ready(info, wait_timeout)) {
                if (execution_complete_.load()) {
                    DumpReadyBufferInfo drain_info;
                    while (memory_manager_.try_pop_ready(drain_info)) {
                        process_dump_buffer(drain_info);
                        memory_manager_.notify_copy_done(drain_info.dev_buffer_ptr);
                        buffers_collected++;
                    }
                    break;
                }

                if (!idle_timer_started) {
                    idle_start = std::chrono::steady_clock::now();
                    idle_timer_started = true;
                }
                auto idle_elapsed = std::chrono::steady_clock::now() - idle_start;
                if (idle_elapsed >= idle_timeout) {
                    LOG_ERROR(
                        "Tensor dump collection idle timeout after %ld seconds",
                        std::chrono::duration_cast<std::chrono::seconds>(idle_elapsed).count()
                    );
                    LOG_ERROR(
                        "Collected %lu buffers and %zu tensors before timeout", buffers_collected, collected_.size()
                    );
                    break;
                }
                continue;
            }
            process_dump_buffer(info);
            memory_manager_.notify_copy_done(info.dev_buffer_ptr);
            buffers_collected++;
            idle_timer_started = false;
        }
    }

    // Stop writer thread and wait for it to drain, reporting progress
    writer_done_.store(true);
    write_cv_.notify_one();
    while (writer_thread_.joinable()) {
        if (write_queue_.empty()) {
            writer_thread_.join();
            break;
        }
        auto elapsed_s =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count();
        LOG_INFO(
            "Writing to disk: %.1f GB written, %zu tensors remaining (%lds)", bytes_written_.load() / 1e9,
            write_queue_.size(), elapsed_s
        );
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    bin_file_.close();

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
    LOG_INFO(
        "Collected %zu tensors, wrote %.1f GB to disk (%.1fs)", collected_.size(), bytes_written_.load() / 1e9,
        elapsed_ms / 1000.0
    );
}

void TensorDumpCollector::signal_execution_complete() { execution_complete_.store(true); }

void TensorDumpCollector::stop_memory_manager() { memory_manager_.stop(); }

void TensorDumpCollector::drain_remaining_buffers() {
    DumpReadyBufferInfo info;
    while (memory_manager_.try_pop_ready(info)) {
        process_dump_buffer(info);
        memory_manager_.notify_copy_done(info.dev_buffer_ptr);
    }
}

void TensorDumpCollector::scan_remaining_dump_buffers() {
    uint32_t dropped_total = 0;
    // Scan current_buf_ptr for each thread for partial buffers not yet enqueued
    for (int t = 0; t < num_dump_threads_; t++) {
        DumpBufferState *state = get_dump_buffer_state(dump_shared_mem_host_, t);

        // Accumulate dropped-record counts regardless of buffer state.
        total_dropped_record_count_ += state->dropped_record_count;
        dropped_total += state->dropped_record_count;

        uint64_t cur_ptr = state->current_buf_ptr;
        if (cur_ptr == 0) {
            continue;
        }

        void *dev_ptr = reinterpret_cast<void *>(cur_ptr);
        void *host_ptr = memory_manager_.resolve_host_ptr(dev_ptr);

        DumpMetaBuffer *buf = reinterpret_cast<DumpMetaBuffer *>(host_ptr);
        if (buf->count > 0) {
            DumpReadyBufferInfo info;
            info.thread_index = static_cast<uint32_t>(t);
            info.dev_buffer_ptr = dev_ptr;
            info.host_buffer_ptr = host_ptr;
            info.buffer_seq = state->current_buf_seq;
            process_dump_buffer(info);
        }
    }
    if (dropped_total > 0) {
        LOG_WARN(
            "Dump collector: %u records dropped on device side. "
            "Increase PLATFORM_DUMP_BUFFERS_PER_THREAD or PLATFORM_DUMP_READYQUEUE_SIZE.",
            dropped_total
        );
    }
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

int TensorDumpCollector::export_dump_files(const std::string & /*output_path*/) {
    if (collected_.empty()) {
        LOG_WARN("No tensor dump data to export");
        return 0;
    }
    auto export_start = std::chrono::steady_clock::now();

    // Sort by task_id then subtask_id then func_id.
    std::sort(collected_.begin(), collected_.end(), [](const DumpedTensor &a, const DumpedTensor &b) {
        if (a.task_id != b.task_id) return a.task_id < b.task_id;
        if (a.subtask_id != b.subtask_id) return a.subtask_id < b.subtask_id;
        if (a.func_id != b.func_id) return a.func_id < b.func_id;
        if (a.stage != b.stage) return static_cast<uint8_t>(a.stage) < static_cast<uint8_t>(b.stage);
        if (a.arg_index != b.arg_index) return a.arg_index < b.arg_index;
        return static_cast<uint8_t>(a.role) < static_cast<uint8_t>(b.role);
    });

    LOG_INFO("Writing JSON manifest for %zu tensors...", collected_.size());

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

    // Write JSON manifest (txt/bin files already written by writer thread)
    std::string base_name = run_dir_.filename().string();
    std::ofstream json(run_dir_ / (base_name + ".json"));
    json << "{\n";
    json << "  \"timestamp\": \"" << base_name.substr(sizeof("tensor_dump_") - 1) << "\",\n";
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
    LOG_INFO("Wrote JSON manifest (%zu tensors) to %s (%ldms)", collected_.size(), run_dir_.c_str(), total_ms);

    if (total_truncated_count_ > 0 || total_dropped_record_count_ > 0 || total_overwrite_count_ > 0) {
        LOG_WARN(
            "Tensor dump anomalies: truncated=%u, dropped_records=%u, overwritten=%u", total_truncated_count_,
            total_dropped_record_count_, total_overwrite_count_
        );
    }

    // Clear state so subsequent runs don't accumulate data from previous runs
    collected_.clear();
    processed_buffers_.clear();
    total_dropped_record_count_ = 0;
    total_truncated_count_ = 0;
    total_overwrite_count_ = 0;
    for (auto &ai : arenas_) {
        ai.high_water = 0;
    }
    return 0;
}

int TensorDumpCollector::finalize(DumpUnregisterCallback unregister_cb, DumpFreeCallback free_cb, void *user_data) {
    // Stop memory manager if still running
    if (memory_manager_.is_running()) {
        memory_manager_.stop();
    }

    std::unordered_set<void *> released_meta_buffers;
    auto release_meta_buffer = [&](void *ptr) {
        if (ptr == nullptr || !released_meta_buffers.insert(ptr).second) {
            return;
        }
        if (was_registered_ && unregister_cb) {
            unregister_cb(ptr, device_id_, user_data);
        }
        if (free_cb) {
            free_cb(ptr, user_data);
        }
    };

    // Free DumpMetaBuffers still in free_queues and current_buf_ptr
    if (dump_shared_mem_host_) {
        for (int t = 0; t < num_dump_threads_; t++) {
            DumpBufferState *state = get_dump_buffer_state(dump_shared_mem_host_, t);

            // Free current buffer if any
            release_meta_buffer(reinterpret_cast<void *>(state->current_buf_ptr));
            state->current_buf_ptr = 0;

            // Free all buffers remaining in free_queue
            rmb();
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_DUMP_SLOT_COUNT) {
                queued = PLATFORM_DUMP_SLOT_COUNT;
            }
            for (uint32_t i = 0; i < queued; i++) {
                uint32_t slot = (head + i) % PLATFORM_DUMP_SLOT_COUNT;
                release_meta_buffer(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }

    // Free buffers still queued for host processing
    {
        std::lock_guard<std::mutex> lock(memory_manager_.ready_mutex_);
        while (!memory_manager_.ready_queue_.empty()) {
            release_meta_buffer(memory_manager_.ready_queue_.front().dev_buffer_ptr);
            memory_manager_.ready_queue_.pop();
        }
    }

    // Free buffers held by memory manager (done_queue + recycled pool)
    {
        std::lock_guard<std::mutex> lock(memory_manager_.done_mutex_);
        while (!memory_manager_.done_queue_.empty()) {
            void *ptr = memory_manager_.done_queue_.front();
            memory_manager_.done_queue_.pop();
            release_meta_buffer(ptr);
        }
    }
    for (void *ptr : memory_manager_.recycled_dump_buffers_) {
        release_meta_buffer(ptr);
    }
    memory_manager_.recycled_dump_buffers_.clear();
    memory_manager_.dev_to_host_.clear();

    // Free arenas
    for (auto &ai : arenas_) {
        if (ai.dev_ptr) {
            if (unregister_cb) {
                unregister_cb(ai.dev_ptr, device_id_, user_data);
            }
            if (free_cb) {
                free_cb(ai.dev_ptr, user_data);
            }
            ai.dev_ptr = nullptr;
            ai.host_ptr = nullptr;
        }
    }
    arenas_.clear();

    // Free shared memory
    if (dump_shared_mem_dev_) {
        if (was_registered_ && unregister_cb) {
            unregister_cb(dump_shared_mem_dev_, device_id_, user_data);
        }
        if (free_cb) {
            free_cb(dump_shared_mem_dev_, user_data);
        }
        dump_shared_mem_dev_ = nullptr;
        dump_shared_mem_host_ = nullptr;
    }

    // Reset state
    num_dump_threads_ = 0;
    execution_complete_.store(false);
    collected_.clear();
    processed_buffers_.clear();
    total_dropped_record_count_ = 0;
    total_truncated_count_ = 0;
    total_overwrite_count_ = 0;

    return 0;
}
