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

#include "host/pmu_collector.h"

#include <cassert>
#include <cstring>
#include <iomanip>
#include <ios>
#include <thread>

#include "common/unified_log.h"

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

PmuCollector::~PmuCollector() = default;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int PmuCollector::init(
    int num_cores, int num_threads, uint64_t *kernel_args_pmu_data_base, const std::string &csv_path,
    PmuEventType event_type, PmuAllocCallback alloc_cb, PmuFreeCallback free_cb, void *user_data, int device_id
) {
    if (num_cores <= 0 || num_threads <= 0 || kernel_args_pmu_data_base == nullptr || alloc_cb == nullptr ||
        free_cb == nullptr) {
        LOG_ERROR("PmuCollector::init: invalid arguments");
        return -1;
    }

    num_cores_ = num_cores;
    num_threads_ = num_threads;
    device_id_ = device_id;
    event_type_ = event_type;
    alloc_cb_ = alloc_cb;
    free_cb_ = free_cb;
    user_data_ = user_data;
    csv_path_ = csv_path;

    // Reset per-run accumulators
    total_collected_ = 0;
    drained_bufs_.clear();
    execution_complete_.store(false, std::memory_order_release);
    if (csv_file_.is_open()) {
        csv_file_.close();
    }

    // ---- Allocate shared header + buffer-state region ----
    shm_size_ = calc_pmu_data_size(num_cores);
    shm_dev_ = alloc_cb(shm_size_, user_data);
    if (shm_dev_ == nullptr) {
        LOG_ERROR("PmuCollector: failed to allocate PMU shared memory (%zu bytes)", shm_size_);
        return -1;
    }

    // Allocate host shadow
    shm_host_ = std::malloc(shm_size_);
    if (shm_host_ == nullptr) {
        LOG_ERROR("PmuCollector: failed to allocate host shadow (%zu bytes)", shm_size_);
        free_cb(shm_dev_, user_data);
        shm_dev_ = nullptr;
        return -1;
    }
    std::memset(shm_host_, 0, shm_size_);

    // Write event_type into header so AICPU can read it from SHM
    get_pmu_header(shm_host_)->event_type = static_cast<uint32_t>(event_type);

    // Publish device address to KernelArgs
    *kernel_args_pmu_data_base = reinterpret_cast<uint64_t>(shm_dev_);

    // ---- Allocate per-core PmuBuffers and populate free_queues ----
    const size_t buf_size = sizeof(PmuBuffer);
    int total_bufs = num_cores * PLATFORM_PMU_BUFFERS_PER_CORE;
    buf_pool_.resize(total_bufs);

    for (int c = 0; c < num_cores; c++) {
        PmuBufferState *state = pmu_state_host(c);

        for (int b = 0; b < PLATFORM_PMU_BUFFERS_PER_CORE; b++) {
            int idx = c * PLATFORM_PMU_BUFFERS_PER_CORE + b;
            void *dev_ptr = alloc_cb(buf_size, user_data);
            if (dev_ptr == nullptr) {
                LOG_ERROR("PmuCollector: failed to allocate PmuBuffer c=%d b=%d", c, b);
                return -1;
            }

            void *host_ptr = std::malloc(buf_size);
            if (host_ptr == nullptr) {
                LOG_ERROR("PmuCollector: failed to allocate host shadow for PmuBuffer c=%d b=%d", c, b);
                free_cb(dev_ptr, user_data);
                return -1;
            }
            std::memset(host_ptr, 0, buf_size);

            // Zero the device buffer
            pmu_platform_copy_to_device(dev_ptr, host_ptr, buf_size);

            buf_pool_[idx] = {dev_ptr, host_ptr};

            // Push into free_queue (host is producer, device is consumer)
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_PMU_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            __sync_synchronize();
            state->free_queue.tail = tail + 1;
            __sync_synchronize();
        }
    }

    // Copy the entire host shadow (header + buffer states + free_queue contents) to device
    pmu_platform_copy_to_device(shm_dev_, shm_host_, shm_size_);

    // ---- Build CSV header string (file is opened lazily on first record) ----
    {
        std::string header = "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles";
        const PmuEventConfig *evt = pmu_resolve_event_config_a5(event_type);
        if (evt == nullptr) {
            evt = &PMU_EVENTS_A5_PIPE_UTIL;
        }
        for (int i = 0; i < PMU_COUNTER_COUNT_A5; i++) {
            const char *name = evt->counter_names[i];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            header += ',';
            header += name;
        }
        header += ",event_type\n";
        csv_header_ = std::move(header);
    }

    initialized_ = true;
    LOG_INFO(
        "PMU collector initialized: %d cores, %d threads, SHM=0x%lx, CSV=%s (opened on first record)", num_cores,
        num_threads, static_cast<unsigned long>(*kernel_args_pmu_data_base), csv_path_.c_str()
    );
    return 0;
}

// ---------------------------------------------------------------------------
// Collector lifecycle
// ---------------------------------------------------------------------------

void PmuCollector::signal_execution_complete() { execution_complete_.store(true, std::memory_order_release); }

// ---------------------------------------------------------------------------
// write_buffer_to_csv
// ---------------------------------------------------------------------------

void PmuCollector::ensure_csv_open_unlocked() {
    if (csv_file_.is_open()) {
        return;
    }
    csv_file_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV file: %s", csv_path_.c_str());
        return;
    }
    csv_file_ << csv_header_;
}

void PmuCollector::write_buffer_to_csv(int core_id, int thread_idx, const void *buf_host_ptr) {
    const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
    }
    if (n == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(csv_mutex_);
    ensure_csv_open_unlocked();
    if (!csv_file_.is_open()) {
        return;
    }
    total_collected_ += n;
    const PmuEventConfig *evt = pmu_resolve_event_config_a5(event_type_);
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A5_PIPE_UTIL;
    }
    for (uint32_t i = 0; i < n; i++) {
        const PmuRecord &r = buf->records[i];
        csv_file_ << thread_idx << ',' << core_id << ',';
        csv_file_ << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.task_id << std::dec
                  << std::setfill(' ');
        csv_file_ << ',' << r.func_id << ',' << static_cast<int>(r.core_type) << ',' << r.pmu_total_cycles;
        for (int k = 0; k < PMU_COUNTER_COUNT_A5; k++) {
            const char *name = evt->counter_names[k];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            csv_file_ << ',' << r.pmu_counters[k];
        }
        csv_file_ << ',' << static_cast<uint32_t>(event_type_) << '\n';
    }
    csv_file_.flush();
}

// ---------------------------------------------------------------------------
// resolve_host_ptr: device address -> host shadow
// ---------------------------------------------------------------------------

void *PmuCollector::resolve_host_ptr(uint64_t dev_addr) {
    for (auto &e : buf_pool_) {
        if (reinterpret_cast<uint64_t>(e.dev_ptr) == dev_addr) {
            return e.host_ptr;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// push_to_free_queue: recycle a buffer back to a core's free_queue
// ---------------------------------------------------------------------------

void PmuCollector::push_to_free_queue(int core_id, uint64_t buf_dev_addr) {
    // Sync host shadow of this core's buffer state from device
    pmu_platform_copy_from_device(pmu_state_host(core_id), pmu_state_dev(core_id), sizeof(PmuBufferState));

    PmuBufferState *state = pmu_state_host(core_id);

    // Zero the count on the host shadow buffer
    void *buf_host = resolve_host_ptr(buf_dev_addr);
    if (buf_host != nullptr) {
        PmuBuffer *buf = reinterpret_cast<PmuBuffer *>(buf_host);
        buf->count = 0;
        // Zero count on device too
        uint32_t zero = 0;
        uintptr_t dev_count_addr = static_cast<uintptr_t>(buf_dev_addr) + offsetof(PmuBuffer, count);
        pmu_platform_copy_to_device(reinterpret_cast<void *>(dev_count_addr), &zero, sizeof(uint32_t));
    }

    // Write the buffer address into the free_queue slot on host shadow
    uint32_t tail = state->free_queue.tail;
    state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT] = buf_dev_addr;
    __sync_synchronize();
    state->free_queue.tail = tail + 1;
    __sync_synchronize();

    // Copy updated free_queue slot and tail back to device
    PmuBufferState *dev_state = pmu_state_dev(core_id);
    uint64_t slot_val = buf_dev_addr;
    pmu_platform_copy_to_device(
        &dev_state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT], &slot_val, sizeof(uint64_t)
    );
    uint32_t new_tail = tail + 1;
    pmu_platform_copy_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
}

// ---------------------------------------------------------------------------
// poll_and_collect: collector thread body
// ---------------------------------------------------------------------------

void PmuCollector::poll_and_collect() {
    if (shm_host_ == nullptr) {
        return;
    }

    PmuDataHeader *hdr = pmu_header_host();
    PmuDataHeader *dev_hdr = pmu_header_dev();
    constexpr int kIdleTimeoutMs = PLATFORM_PMU_TIMEOUT_SECONDS * 1000;
    constexpr int kPollIntervalUs = 100;
    int idle_ms = 0;

    while (true) {
        bool found_any = false;

        // Read queue_tails from device
        pmu_platform_copy_from_device(hdr->queue_tails, dev_hdr->queue_tails, sizeof(hdr->queue_tails));

        for (int t = 0; t < num_threads_; t++) {
            __sync_synchronize();
            uint32_t head = hdr->queue_heads[t];
            uint32_t tail = hdr->queue_tails[t];

            while (head != tail) {
                // Read the ready queue entry from device
                PmuReadyQueueEntry entry;
                pmu_platform_copy_from_device(
                    &entry, &dev_hdr->queues[t][head % PLATFORM_PMU_READYQUEUE_SIZE], sizeof(PmuReadyQueueEntry)
                );

                uint32_t core_id = entry.core_index;
                uint64_t buf_dev = entry.buffer_ptr;

                // Copy buffer from device to host shadow
                void *buf_host = resolve_host_ptr(buf_dev);
                if (buf_host != nullptr) {
                    pmu_platform_copy_from_device(buf_host, reinterpret_cast<void *>(buf_dev), sizeof(PmuBuffer));
                    drained_bufs_.insert(buf_dev);
                    write_buffer_to_csv(static_cast<int>(core_id), t, buf_host);
                    push_to_free_queue(static_cast<int>(core_id), buf_dev);
                } else {
                    LOG_WARN("PMU collector: unknown buffer 0x%lx from core %u", buf_dev, core_id);
                }

                head = (head + 1) % PLATFORM_PMU_READYQUEUE_SIZE;
                hdr->queue_heads[t] = head;

                // Write updated head back to device
                pmu_platform_copy_to_device(&dev_hdr->queue_heads[t], &head, sizeof(uint32_t));

                found_any = true;
            }
        }

        if (!found_any) {
            if (execution_complete_.load(std::memory_order_acquire)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(kPollIntervalUs));
            idle_ms += kPollIntervalUs / 1000;
            if (idle_ms >= kIdleTimeoutMs) {
                LOG_WARN("PMU collector: idle timeout (%d ms), stopping", kIdleTimeoutMs);
                break;
            }
        } else {
            idle_ms = 0;
        }
    }

    LOG_INFO("PMU collector thread exiting");
}

// ---------------------------------------------------------------------------
// drain_remaining_buffers: after collector thread exits, pick up leftovers
// ---------------------------------------------------------------------------

void PmuCollector::drain_remaining_buffers() {
    if (shm_host_ == nullptr) {
        return;
    }

    // Sync entire shared memory region from device
    pmu_platform_copy_from_device(shm_host_, shm_dev_, shm_size_);

    PmuDataHeader *hdr = pmu_header_host();

    // Drain any ready_queue entries the collector thread may have missed
    for (int t = 0; t < num_threads_; t++) {
        __sync_synchronize();
        uint32_t head = hdr->queue_heads[t];
        uint32_t tail = hdr->queue_tails[t];

        while (head != tail) {
            const PmuReadyQueueEntry &entry = hdr->queues[t][head % PLATFORM_PMU_READYQUEUE_SIZE];
            uint32_t core_id = entry.core_index;
            uint64_t buf_dev = entry.buffer_ptr;

            if (drained_bufs_.find(buf_dev) == drained_bufs_.end()) {
                void *buf_host = resolve_host_ptr(buf_dev);
                if (buf_host != nullptr) {
                    pmu_platform_copy_from_device(buf_host, reinterpret_cast<void *>(buf_dev), sizeof(PmuBuffer));
                    write_buffer_to_csv(static_cast<int>(core_id), t, buf_host);
                    drained_bufs_.insert(buf_dev);
                }
            }

            head = (head + 1) % PLATFORM_PMU_READYQUEUE_SIZE;
            hdr->queue_heads[t] = head;
        }
    }

    // Also check current_buf_ptr for each core (AICPU may have flushed but
    // not enqueued if the ready_queue was observed full during flush)
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = pmu_state_host(c);
        __sync_synchronize();
        uint64_t buf_dev = state->current_buf_ptr;
        if (buf_dev == 0 || drained_bufs_.count(buf_dev) > 0) {
            continue;
        }
        void *buf_host = resolve_host_ptr(buf_dev);
        if (buf_host != nullptr) {
            pmu_platform_copy_from_device(buf_host, reinterpret_cast<void *>(buf_dev), sizeof(PmuBuffer));
            const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host);
            if (buf->count > 0) {
                write_buffer_to_csv(c, -1, buf_host);
                drained_bufs_.insert(buf_dev);
            }
        }
    }

    if (csv_file_.is_open()) {
        csv_file_.flush();
    }

    // Cross-check device-side totals against what we wrote to CSV
    uint64_t total_device = 0;
    uint64_t dropped_device = 0;
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = pmu_state_host(c);
        __sync_synchronize();
        total_device += state->total_record_count;
        dropped_device += state->dropped_record_count;
    }

    if (dropped_device > 0) {
        LOG_WARN(
            "PMU collector: %lu records dropped on device side (free_queue empty or ready_queue full). "
            "Increase PLATFORM_PMU_BUFFERS_PER_CORE / PLATFORM_PMU_READYQUEUE_SIZE if this is frequent.",
            static_cast<unsigned long>(dropped_device)
        );
    }
    if (total_collected_ + dropped_device != total_device) {
        LOG_WARN(
            "PMU collector: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
    } else {
        LOG_INFO(
            "PMU collector: record counts match (collected=%lu, dropped=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
    }

    LOG_INFO("PMU collector: drain_remaining_buffers complete");
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void PmuCollector::finalize(PmuUnregisterCallback /*unregister_cb*/, PmuFreeCallback free_cb, void *user_data) {
    if (!initialized_) {
        return;
    }

    if (csv_file_.is_open()) {
        csv_file_.close();
    }

    PmuFreeCallback cb = free_cb != nullptr ? free_cb : free_cb_;
    void *ud = free_cb != nullptr ? user_data : user_data_;

    // Free individual PmuBuffers
    for (auto &entry : buf_pool_) {
        if (entry.dev_ptr != nullptr && cb != nullptr) {
            cb(entry.dev_ptr, ud);
        }
        if (entry.host_ptr != nullptr) {
            std::free(entry.host_ptr);
        }
    }
    buf_pool_.clear();

    // Free shared header region
    if (shm_dev_ != nullptr && cb != nullptr) {
        cb(shm_dev_, ud);
    }
    if (shm_host_ != nullptr) {
        std::free(shm_host_);
    }
    shm_dev_ = nullptr;
    shm_host_ = nullptr;

    initialized_ = false;
    LOG_INFO("PMU collector finalized");
}
