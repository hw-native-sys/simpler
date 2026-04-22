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
 * @file performance_collector.cpp
 * @brief Platform-agnostic performance data collector implementation
 *
 * Implements ProfMemoryManager (dynamic buffer management thread) and
 * PerformanceCollector (data collection and export).
 *
 * A5 specifics:
 * - In memcpy mode (onboard): ReadyQueue/FreeQueue fields and buffer contents
 *   are accessed via rtMemcpy D2H/H2D through copy callbacks.
 * - In sim mode: dev_ptr == host_ptr, direct pointer access (no callbacks).
 */

#include "host/performance_collector.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <vector>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

// =============================================================================
// ProfMemoryManager Implementation
// =============================================================================

ProfMemoryManager::~ProfMemoryManager() {
    if (running_.load()) {
        stop();
    }
}

void ProfMemoryManager::start(
    void *shared_mem_host, void *shared_mem_dev, int num_cores, int num_phase_threads, PerfAllocCallback alloc_cb,
    PerfRegisterCallback register_cb, PerfFreeCallback free_cb, PerfCopyToDeviceCallback copy_to_dev_cb,
    PerfCopyFromDeviceCallback copy_from_dev_cb, int device_id, const ThreadFactory &thread_factory
) {
    shared_mem_host_ = shared_mem_host;
    shared_mem_dev_ = shared_mem_dev;
    num_cores_ = num_cores;
    num_phase_threads_ = num_phase_threads;
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    copy_to_dev_cb_ = copy_to_dev_cb;
    copy_from_dev_cb_ = copy_from_dev_cb;
    device_id_ = device_id;

    running_.store(true);
    if (thread_factory) {
        mgmt_thread_ = thread_factory([this]() {
            mgmt_loop();
        });
    } else {
        mgmt_thread_ = std::thread(&ProfMemoryManager::mgmt_loop, this);
    }

    LOG_INFO("ProfMemoryManager started: %d cores, %d phase threads", num_cores, num_phase_threads);
}

void ProfMemoryManager::stop() {
    running_.store(false);
    if (mgmt_thread_.joinable()) {
        mgmt_thread_.join();
    }

    // Drain remaining done_queue and free buffers
    {
        std::scoped_lock<std::mutex> lock(done_mutex_);
        while (!done_queue_.empty()) {
            CopyDoneInfo info = done_queue_.front();
            done_queue_.pop();
            free_buffer(info.dev_buffer_ptr);
        }
    }

    // Free recycled buffers
    for (void *ptr : recycled_perf_buffers_) {
        free_buffer(ptr);
    }
    recycled_perf_buffers_.clear();
    for (void *ptr : recycled_phase_buffers_) {
        free_buffer(ptr);
    }
    recycled_phase_buffers_.clear();

    LOG_INFO("ProfMemoryManager stopped");
}

bool ProfMemoryManager::try_pop_ready(ReadyBufferInfo &info) {
    std::scoped_lock<std::mutex> lock(ready_mutex_);
    if (ready_queue_.empty()) {
        return false;
    }
    info = ready_queue_.front();
    ready_queue_.pop();
    return true;
}

bool ProfMemoryManager::wait_pop_ready(ReadyBufferInfo &info, std::chrono::milliseconds timeout) {
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

void ProfMemoryManager::notify_copy_done(const CopyDoneInfo &info) {
    std::scoped_lock<std::mutex> lock(done_mutex_);
    done_queue_.push(info);
}

int ProfMemoryManager::sync_from_device(volatile void *host_dst, const volatile void *dev_src, size_t size) {
    if (copy_from_dev_cb_ != nullptr) {
        return copy_from_dev_cb_(const_cast<void *>(host_dst), const_cast<const void *>(dev_src), size);
    }
    return 0;  // sim mode: already in sync
}

int ProfMemoryManager::sync_to_device(volatile void *dev_dst, const void *host_src, size_t size) {
    if (copy_to_dev_cb_ != nullptr) {
        return copy_to_dev_cb_(const_cast<void *>(dev_dst), host_src, size);
    }
    return 0;  // sim mode: already in sync
}

void *ProfMemoryManager::alloc_and_register(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size);
    if (dev_ptr == nullptr) {
        const char *hint = (size == sizeof(PerfBuffer)) ?
                               "increase PLATFORM_PROF_BUFFERS_PER_CORE to reduce profiling data loss" :
                               "increase PLATFORM_PROF_BUFFERS_PER_THREAD to reduce profiling data loss";
        LOG_WARN("ProfMemoryManager: alloc failed for %zu bytes, %s", size, hint);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        // SVM mode (not used on A5, kept for interface parity)
        void *host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("ProfMemoryManager: register failed: %d", rc);
            free_buffer(dev_ptr);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else if (is_memcpy_mode()) {
        // Memcpy mode: allocate host shadow buffer
        void *host_ptr = std::malloc(size);
        if (host_ptr == nullptr) {
            LOG_ERROR("ProfMemoryManager: host shadow alloc failed for %zu bytes", size);
            free_buffer(dev_ptr);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        std::memset(host_ptr, 0, size);
        // Zero the device buffer too
        sync_to_device(dev_ptr, host_ptr, size);
        *host_ptr_out = host_ptr;
    } else {
        // Simulation mode: dev_ptr == host_ptr
        *host_ptr_out = dev_ptr;
    }

    dev_to_host_[dev_ptr] = *host_ptr_out;
    return dev_ptr;
}

void ProfMemoryManager::free_buffer(void *dev_ptr) {
    if (dev_ptr == nullptr) return;

    auto it = dev_to_host_.find(dev_ptr);
    if (it != dev_to_host_.end()) {
        // In memcpy mode, free the host shadow if it differs from dev_ptr
        if (is_memcpy_mode() && it->second != dev_ptr) {
            std::free(it->second);
        }
        dev_to_host_.erase(it);
    }

    if (free_cb_ != nullptr) {
        free_cb_(dev_ptr);
    }
}

void *ProfMemoryManager::resolve_host_ptr(void *dev_ptr) {
    if (!is_memcpy_mode() && register_cb_ == nullptr) {
        return dev_ptr;  // Simulation mode: dev_ptr == host_ptr
    }
    auto it = dev_to_host_.find(dev_ptr);
    if (it != dev_to_host_.end()) {
        return it->second;
    }
    LOG_ERROR("ProfMemoryManager: no host mapping for dev_ptr=%p", dev_ptr);
    return nullptr;
}

void ProfMemoryManager::register_mapping(void *dev_ptr, void *host_ptr) { dev_to_host_[dev_ptr] = host_ptr; }

void ProfMemoryManager::process_ready_entry(
    PerfDataHeader * /*header*/, int /*thread_idx*/, const ReadyQueueEntry &entry
) {
    bool is_phase = (entry.is_phase != 0);
    uint64_t old_dev_ptr = entry.buffer_ptr;
    uint32_t seq = entry.buffer_seq;

    // In memcpy mode, copy the full buffer from device to its host shadow
    if (is_memcpy_mode()) {
        void *old_host_ptr = resolve_host_ptr(reinterpret_cast<void *>(old_dev_ptr));
        if (old_host_ptr != nullptr) {
            size_t buf_size = is_phase ? sizeof(PhaseBuffer) : sizeof(PerfBuffer);
            sync_from_device(old_host_ptr, reinterpret_cast<void *>(old_dev_ptr), buf_size);
        }
    }

    if (is_phase) {
        uint32_t tidx = entry.core_index;
        if (tidx >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
            LOG_ERROR("ProfMemoryManager: invalid phase entry: thread=%u", tidx);
            return;
        }

        PhaseBufferState *state = get_phase_buffer_state(shared_mem_host_, num_cores_, tidx);

        // In memcpy mode, sync PerfBufferState from device
        if (is_memcpy_mode()) {
            PhaseBufferState *dev_state = get_phase_buffer_state(shared_mem_dev_, num_cores_, tidx);
            sync_from_device(state, dev_state, sizeof(PhaseBufferState));
        }

        // Replenish free_queue
        rmb();
        uint32_t head_val = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t available = tail - head_val;

        int to_push = PLATFORM_PROF_SLOT_COUNT;
        for (int p = 0; p < to_push && available + p < static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT); p++) {
            void *host_ptr = nullptr;
            void *new_dev_ptr = nullptr;

            if (!recycled_phase_buffers_.empty()) {
                new_dev_ptr = recycled_phase_buffers_.back();
                recycled_phase_buffers_.pop_back();
                host_ptr = resolve_host_ptr(new_dev_ptr);
            }
            if (new_dev_ptr == nullptr) {
                std::scoped_lock<std::mutex> lock(done_mutex_);
                while (!done_queue_.empty()) {
                    CopyDoneInfo dinfo = done_queue_.front();
                    done_queue_.pop();
                    if (dinfo.type == ProfBufferType::PERF_RECORD)
                        recycled_perf_buffers_.push_back(dinfo.dev_buffer_ptr);
                    else recycled_phase_buffers_.push_back(dinfo.dev_buffer_ptr);
                }
            }
            if (new_dev_ptr == nullptr && !recycled_phase_buffers_.empty()) {
                new_dev_ptr = recycled_phase_buffers_.back();
                recycled_phase_buffers_.pop_back();
                host_ptr = resolve_host_ptr(new_dev_ptr);
            }
            if (new_dev_ptr == nullptr) {
                new_dev_ptr = alloc_and_register(sizeof(PhaseBuffer), &host_ptr);
            }
            if (new_dev_ptr == nullptr) break;

            reinterpret_cast<PhaseBuffer *>(host_ptr)->count = 0;
            if (is_memcpy_mode()) {
                // Zero the device buffer count
                uint32_t zero = 0;
                sync_to_device(
                    reinterpret_cast<char *>(new_dev_ptr) + offsetof(PhaseBuffer, count), &zero, sizeof(uint32_t)
                );
            }

            uint32_t cur_tail = tail + p;
            state->free_queue.buffer_ptrs[cur_tail % PLATFORM_PROF_SLOT_COUNT] =
                reinterpret_cast<uint64_t>(new_dev_ptr);
            wmb();
            state->free_queue.tail = cur_tail + 1;
            wmb();

            // In memcpy mode, write the slot and tail back to device
            if (is_memcpy_mode()) {
                PhaseBufferState *dev_state = get_phase_buffer_state(shared_mem_dev_, num_cores_, tidx);
                uint64_t slot_val = reinterpret_cast<uint64_t>(new_dev_ptr);
                sync_to_device(
                    &dev_state->free_queue.buffer_ptrs[cur_tail % PLATFORM_PROF_SLOT_COUNT], &slot_val, sizeof(uint64_t)
                );
                uint32_t new_tail = cur_tail + 1;
                sync_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
            }
        }

        // Resolve host pointer of old buffer
        void *old_host_ptr = resolve_host_ptr(reinterpret_cast<void *>(old_dev_ptr));
        if (old_host_ptr == nullptr) {
            LOG_ERROR(
                "ProfMemoryManager: cannot resolve host ptr for phase buffer dev=%p",
                reinterpret_cast<void *>(old_dev_ptr)
            );
            return;
        }

        // Push old buffer to ready queue for main thread to copy
        ReadyBufferInfo info;
        info.type = ProfBufferType::PHASE;
        info.index = tidx;
        info.slot_idx = 0;
        info.dev_buffer_ptr = reinterpret_cast<void *>(old_dev_ptr);
        info.host_buffer_ptr = old_host_ptr;
        info.buffer_seq = seq;

        {
            std::scoped_lock<std::mutex> lock(ready_mutex_);
            ready_queue_.push(info);
        }
        ready_cv_.notify_one();

    } else {
        uint32_t core_index = entry.core_index;
        if (core_index >= static_cast<uint32_t>(num_cores_)) {
            LOG_ERROR("ProfMemoryManager: invalid perf entry: core=%u", core_index);
            return;
        }

        PerfBufferState *state = get_perf_buffer_state(shared_mem_host_, core_index);

        // In memcpy mode, sync PerfBufferState from device
        if (is_memcpy_mode()) {
            PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, core_index);
            sync_from_device(state, dev_state, sizeof(PerfBufferState));
        }

        // Replenish free_queue
        rmb();
        uint32_t head_val = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t available = tail - head_val;

        int to_push = PLATFORM_PROF_SLOT_COUNT;
        for (int p = 0; p < to_push && available + p < static_cast<uint32_t>(PLATFORM_PROF_SLOT_COUNT); p++) {
            void *host_ptr = nullptr;
            void *new_dev_ptr = nullptr;

            if (!recycled_perf_buffers_.empty()) {
                new_dev_ptr = recycled_perf_buffers_.back();
                recycled_perf_buffers_.pop_back();
                host_ptr = resolve_host_ptr(new_dev_ptr);
            }
            if (new_dev_ptr == nullptr) {
                std::scoped_lock<std::mutex> lock(done_mutex_);
                while (!done_queue_.empty()) {
                    CopyDoneInfo dinfo = done_queue_.front();
                    done_queue_.pop();
                    if (dinfo.type == ProfBufferType::PERF_RECORD)
                        recycled_perf_buffers_.push_back(dinfo.dev_buffer_ptr);
                    else recycled_phase_buffers_.push_back(dinfo.dev_buffer_ptr);
                }
            }
            if (new_dev_ptr == nullptr && !recycled_perf_buffers_.empty()) {
                new_dev_ptr = recycled_perf_buffers_.back();
                recycled_perf_buffers_.pop_back();
                host_ptr = resolve_host_ptr(new_dev_ptr);
            }
            if (new_dev_ptr == nullptr) {
                new_dev_ptr = alloc_and_register(sizeof(PerfBuffer), &host_ptr);
            }
            if (new_dev_ptr == nullptr) break;

            reinterpret_cast<PerfBuffer *>(host_ptr)->count = 0;
            if (is_memcpy_mode()) {
                uint32_t zero = 0;
                sync_to_device(
                    reinterpret_cast<char *>(new_dev_ptr) + offsetof(PerfBuffer, count), &zero, sizeof(uint32_t)
                );
            }

            uint32_t cur_tail = tail + p;
            state->free_queue.buffer_ptrs[cur_tail % PLATFORM_PROF_SLOT_COUNT] =
                reinterpret_cast<uint64_t>(new_dev_ptr);
            wmb();
            state->free_queue.tail = cur_tail + 1;
            wmb();

            // In memcpy mode, write the slot and tail back to device
            if (is_memcpy_mode()) {
                PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, core_index);
                uint64_t slot_val = reinterpret_cast<uint64_t>(new_dev_ptr);
                sync_to_device(
                    &dev_state->free_queue.buffer_ptrs[cur_tail % PLATFORM_PROF_SLOT_COUNT], &slot_val, sizeof(uint64_t)
                );
                uint32_t new_tail = cur_tail + 1;
                sync_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
            }
        }

        void *old_host_ptr = resolve_host_ptr(reinterpret_cast<void *>(old_dev_ptr));
        if (old_host_ptr == nullptr) {
            LOG_ERROR(
                "ProfMemoryManager: cannot resolve host ptr for perf buffer dev=%p",
                reinterpret_cast<void *>(old_dev_ptr)
            );
            return;
        }

        ReadyBufferInfo info;
        info.type = ProfBufferType::PERF_RECORD;
        info.index = core_index;
        info.slot_idx = 0;
        info.dev_buffer_ptr = reinterpret_cast<void *>(old_dev_ptr);
        info.host_buffer_ptr = old_host_ptr;
        info.buffer_seq = seq;

        {
            std::scoped_lock<std::mutex> lock(ready_mutex_);
            ready_queue_.push(info);
        }
        ready_cv_.notify_one();
    }
}

void ProfMemoryManager::mgmt_loop() {
    PerfDataHeader *header = get_perf_header(shared_mem_host_);

    while (running_.load()) {
        // 1. Recycle done queue: move completed buffers to recycled pools for reuse
        {
            std::scoped_lock<std::mutex> lock(done_mutex_);
            while (!done_queue_.empty()) {
                CopyDoneInfo info = done_queue_.front();
                done_queue_.pop();
                if (info.type == ProfBufferType::PERF_RECORD) {
                    recycled_perf_buffers_.push_back(info.dev_buffer_ptr);
                } else {
                    recycled_phase_buffers_.push_back(info.dev_buffer_ptr);
                }
            }
        }

        // 2. Poll ReadyQueues from all AICPU threads
        // In memcpy mode, sync queue_tails from device first
        if (is_memcpy_mode()) {
            PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
            sync_from_device(header->queue_tails, dev_header->queue_tails, sizeof(header->queue_tails));
        }

        bool found_any = false;
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            rmb();
            uint32_t head = header->queue_heads[t];
            uint32_t tail = header->queue_tails[t];

            // Validate indices to prevent OOB access from corrupted shared memory
            if (head >= PLATFORM_PROF_READYQUEUE_SIZE || tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                LOG_ERROR(
                    "mgmt_loop: invalid queue indices for thread %d: head=%u tail=%u (max=%d)", t, head, tail,
                    PLATFORM_PROF_READYQUEUE_SIZE
                );
                continue;
            }

            while (head != tail) {
                ReadyQueueEntry entry;
                if (is_memcpy_mode()) {
                    // Read the specific entry from device
                    PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
                    sync_from_device(&entry, &dev_header->queues[t][head], sizeof(ReadyQueueEntry));
                } else {
                    entry = header->queues[t][head];
                }

                process_ready_entry(header, t, entry);

                head = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
                header->queue_heads[t] = head;
                wmb();

                // In memcpy mode, write updated head back to device
                if (is_memcpy_mode()) {
                    PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
                    sync_to_device(&dev_header->queue_heads[t], &head, sizeof(uint32_t));
                }

                found_any = true;

                // Re-read tail in case more entries arrived
                if (is_memcpy_mode()) {
                    PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
                    sync_from_device(&header->queue_tails[t], &dev_header->queue_tails[t], sizeof(uint32_t));
                }
                rmb();
                tail = header->queue_tails[t];
                if (tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                    LOG_ERROR("mgmt_loop: invalid tail for thread %d: %u", t, tail);
                    break;
                }
            }
        }

        // 3. Proactive replenishment: push buffers to cores/threads whose free_queue
        //    is completely empty (avail == 0). Try recycled pool first, alloc as fallback.
        if (!recycled_perf_buffers_.empty() || !recycled_phase_buffers_.empty()) {
            for (int i = 0; i < num_cores_ && !recycled_perf_buffers_.empty(); i++) {
                PerfBufferState *state = get_perf_buffer_state(shared_mem_host_, i);
                if (is_memcpy_mode()) {
                    PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, i);
                    sync_from_device(state, dev_state, sizeof(PerfBufferState));
                }
                rmb();
                uint32_t avail = state->free_queue.tail - state->free_queue.head;
                if (avail == 0) {
                    void *dev_ptr = recycled_perf_buffers_.back();
                    recycled_perf_buffers_.pop_back();
                    void *host_ptr = resolve_host_ptr(dev_ptr);
                    if (host_ptr != nullptr) {
                        reinterpret_cast<PerfBuffer *>(host_ptr)->count = 0;
                        if (is_memcpy_mode()) {
                            uint32_t zero = 0;
                            sync_to_device(
                                reinterpret_cast<char *>(dev_ptr) + offsetof(PerfBuffer, count), &zero, sizeof(uint32_t)
                            );
                        }
                        uint32_t t_val = state->free_queue.tail;
                        state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT] =
                            reinterpret_cast<uint64_t>(dev_ptr);
                        wmb();
                        state->free_queue.tail = t_val + 1;
                        wmb();
                        if (is_memcpy_mode()) {
                            PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, i);
                            uint64_t slot_val = reinterpret_cast<uint64_t>(dev_ptr);
                            sync_to_device(
                                &dev_state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT], &slot_val,
                                sizeof(uint64_t)
                            );
                            uint32_t new_tail = t_val + 1;
                            sync_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
                        }
                    }
                }
            }
            for (int t = 0; t < num_phase_threads_ && !recycled_phase_buffers_.empty(); t++) {
                PhaseBufferState *state = get_phase_buffer_state(shared_mem_host_, num_cores_, t);
                if (is_memcpy_mode()) {
                    PhaseBufferState *dev_state = get_phase_buffer_state(shared_mem_dev_, num_cores_, t);
                    sync_from_device(state, dev_state, sizeof(PhaseBufferState));
                }
                rmb();
                uint32_t avail = state->free_queue.tail - state->free_queue.head;
                if (avail == 0) {
                    void *dev_ptr = recycled_phase_buffers_.back();
                    recycled_phase_buffers_.pop_back();
                    void *host_ptr = resolve_host_ptr(dev_ptr);
                    if (host_ptr != nullptr) {
                        reinterpret_cast<PhaseBuffer *>(host_ptr)->count = 0;
                        if (is_memcpy_mode()) {
                            uint32_t zero = 0;
                            sync_to_device(
                                reinterpret_cast<char *>(dev_ptr) + offsetof(PhaseBuffer, count), &zero,
                                sizeof(uint32_t)
                            );
                        }
                        uint32_t t_val = state->free_queue.tail;
                        state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT] =
                            reinterpret_cast<uint64_t>(dev_ptr);
                        wmb();
                        state->free_queue.tail = t_val + 1;
                        wmb();
                        if (is_memcpy_mode()) {
                            PhaseBufferState *dev_state = get_phase_buffer_state(shared_mem_dev_, num_cores_, t);
                            uint64_t slot_val = reinterpret_cast<uint64_t>(dev_ptr);
                            sync_to_device(
                                &dev_state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT], &slot_val,
                                sizeof(uint64_t)
                            );
                            uint32_t new_tail = t_val + 1;
                            sync_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
                        }
                    }
                }
            }
        }
        // Alloc fallback: if recycled pools are both empty, scan for depleted cores and alloc.
        if (recycled_perf_buffers_.empty() && recycled_phase_buffers_.empty()) {
            for (int i = 0; i < num_cores_; i++) {
                PerfBufferState *state = get_perf_buffer_state(shared_mem_host_, i);
                if (is_memcpy_mode()) {
                    PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, i);
                    sync_from_device(state, dev_state, sizeof(PerfBufferState));
                }
                rmb();
                if (state->free_queue.tail - state->free_queue.head == 0) {
                    void *host_ptr = nullptr;
                    void *dev_ptr = alloc_and_register(sizeof(PerfBuffer), &host_ptr);
                    if (dev_ptr == nullptr) break;  // HBM exhausted
                    reinterpret_cast<PerfBuffer *>(host_ptr)->count = 0;
                    if (is_memcpy_mode()) {
                        uint32_t zero = 0;
                        sync_to_device(
                            reinterpret_cast<char *>(dev_ptr) + offsetof(PerfBuffer, count), &zero, sizeof(uint32_t)
                        );
                    }
                    uint32_t t_val = state->free_queue.tail;
                    state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT] =
                        reinterpret_cast<uint64_t>(dev_ptr);
                    wmb();
                    state->free_queue.tail = t_val + 1;
                    wmb();
                    if (is_memcpy_mode()) {
                        PerfBufferState *dev_state = get_perf_buffer_state(shared_mem_dev_, i);
                        uint64_t slot_val = reinterpret_cast<uint64_t>(dev_ptr);
                        sync_to_device(
                            &dev_state->free_queue.buffer_ptrs[t_val % PLATFORM_PROF_SLOT_COUNT], &slot_val,
                            sizeof(uint64_t)
                        );
                        uint32_t new_tail = t_val + 1;
                        sync_to_device(&dev_state->free_queue.tail, &new_tail, sizeof(uint32_t));
                    }
                    break;  // One alloc per iteration
                }
            }
        }

        // 4. If nothing found, yield briefly to avoid busy-spinning
        if (!found_any) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Final drain: process any remaining entries
    if (is_memcpy_mode()) {
        PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
        sync_from_device(header->queue_tails, dev_header->queue_tails, sizeof(header->queue_tails));
    }
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        rmb();
        uint32_t head = header->queue_heads[t];
        uint32_t tail = header->queue_tails[t];
        if (head >= PLATFORM_PROF_READYQUEUE_SIZE || tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
            LOG_ERROR("mgmt_loop drain: invalid queue indices for thread %d: head=%u tail=%u", t, head, tail);
            continue;
        }
        while (head != tail) {
            ReadyQueueEntry entry;
            if (is_memcpy_mode()) {
                PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
                sync_from_device(&entry, &dev_header->queues[t][head], sizeof(ReadyQueueEntry));
            } else {
                entry = header->queues[t][head];
            }
            process_ready_entry(header, t, entry);
            head = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
            header->queue_heads[t] = head;
            wmb();
            if (is_memcpy_mode()) {
                PerfDataHeader *dev_header = get_perf_header(shared_mem_dev_);
                sync_to_device(&dev_header->queue_heads[t], &head, sizeof(uint32_t));
                sync_from_device(&header->queue_tails[t], &dev_header->queue_tails[t], sizeof(uint32_t));
            }
            rmb();
            tail = header->queue_tails[t];
            if (tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                LOG_ERROR("mgmt_loop drain: invalid tail for thread %d: %u", t, tail);
                break;
            }
        }
    }
}

// =============================================================================
// PerformanceCollector Implementation
// =============================================================================

static bool is_scheduler_phase(AicpuPhaseId id) {
    return static_cast<uint32_t>(id) < static_cast<uint32_t>(AicpuPhaseId::SCHED_PHASE_COUNT);
}

PerformanceCollector::~PerformanceCollector() {
    if (perf_shared_mem_host_ != nullptr) {
        LOG_WARN("PerformanceCollector destroyed without finalize()");
    }
}

void *PerformanceCollector::alloc_single_buffer(size_t size, void **host_ptr_out) {
    void *dev_ptr = alloc_cb_(size);
    if (dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate buffer (%zu bytes)", size);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void *host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else if (is_memcpy_mode()) {
        // Memcpy mode: allocate host shadow
        void *host_ptr = std::malloc(size);
        if (host_ptr == nullptr) {
            LOG_ERROR("Host shadow alloc failed for %zu bytes", size);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        *host_ptr_out = dev_ptr;
    }

    // Register mapping so ProfMemoryManager can resolve dev→host
    memory_manager_.register_mapping(dev_ptr, *host_ptr_out);
    return dev_ptr;
}

int PerformanceCollector::initialize(
    Runtime &runtime, int num_aicore, int device_id, PerfAllocCallback alloc_cb, PerfRegisterCallback register_cb,
    PerfFreeCallback free_cb, PerfCopyToDeviceCallback copy_to_dev_cb, PerfCopyFromDeviceCallback copy_from_dev_cb
) {
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
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    copy_to_dev_cb_ = copy_to_dev_cb;
    copy_from_dev_cb_ = copy_from_dev_cb;

    // Step 1: Calculate shared memory size (slot arrays only, no actual buffers)
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(PerfDataHeader));
    LOG_DEBUG("  PerfBufferState size: %zu bytes each", sizeof(PerfBufferState));
    LOG_DEBUG("  PhaseBufferState size:%zu bytes each", sizeof(PhaseBufferState));
    LOG_DEBUG("  Total shared memory:  %zu bytes (%zu KB)", total_size, total_size / 1024);

    // Step 2: Allocate shared memory for slot arrays
    void *perf_dev_ptr = alloc_cb(total_size);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: %p", perf_dev_ptr);

    // Step 3: Get or create host-side pointer
    void *perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, &perf_host_ptr);
        if (rc != 0 || perf_host_ptr == nullptr) {
            LOG_ERROR("Memory registration failed: %d", rc);
            return rc;
        }
        was_registered_ = true;
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else if (is_memcpy_mode()) {
        // Memcpy mode: allocate host shadow for the shared memory region
        perf_host_ptr = std::malloc(total_size);
        if (perf_host_ptr == nullptr) {
            LOG_ERROR("Host shadow alloc failed for shared memory (%zu bytes)", total_size);
            return -1;
        }
        LOG_DEBUG("Allocated host shadow: %p", perf_host_ptr);
    } else {
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize header
    PerfDataHeader *header = get_perf_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        std::memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->total_tasks = 0;

    LOG_DEBUG("Initialized PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize PerfBufferStates — 1 buffer per core in free_queue, rest to recycled pool
    for (int i = 0; i < num_aicore; i++) {
        PerfBufferState *state = get_perf_buffer_state(perf_host_ptr, i);
        std::memset(state, 0, sizeof(PerfBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_CORE; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(PerfBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PerfBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(host_buf_ptr);
            std::memset(buf, 0, sizeof(PerfBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                memory_manager_.recycled_perf_buffers_.push_back(dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized %d PerfBufferStates: 1 buffer/core, %d in recycled pool", num_aicore,
        num_aicore * (PLATFORM_PROF_BUFFERS_PER_CORE - 1)
    );

    // Step 6: Initialize PhaseBufferStates — 1 buffer per thread in free_queue, rest to recycled pool
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
        std::memset(state, 0, sizeof(PhaseBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        for (int s = 0; s < PLATFORM_PROF_BUFFERS_PER_THREAD; s++) {
            void *host_buf_ptr = nullptr;
            void *dev_buf_ptr = alloc_single_buffer(sizeof(PhaseBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PhaseBuffer for thread %d, buffer %d", t, s);
                return -1;
            }
            PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(host_buf_ptr);
            std::memset(buf, 0, sizeof(PhaseBuffer));
            buf->count = 0;

            if (s == 0) {
                state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(dev_buf_ptr);
            } else {
                memory_manager_.recycled_phase_buffers_.push_back(dev_buf_ptr);
            }
        }
        wmb();
        state->free_queue.tail = 1;
        wmb();
    }
    LOG_DEBUG(
        "Initialized %d PhaseBufferStates: 1 buffer/thread, %d in recycled pool", num_phase_threads,
        num_phase_threads * (PLATFORM_PROF_BUFFERS_PER_THREAD - 1)
    );

    // In memcpy mode, copy the entire initialized shared memory to device
    if (is_memcpy_mode()) {
        int rc = copy_to_dev_cb_(perf_dev_ptr, perf_host_ptr, total_size);
        if (rc != 0) {
            LOG_ERROR("Failed to copy shared memory to device: %d", rc);
            return rc;
        }
        // Also copy each initial buffer (zero'd count) to device
        for (int i = 0; i < num_aicore; i++) {
            PerfBufferState *state = get_perf_buffer_state(perf_host_ptr, i);
            void *dev_buf = reinterpret_cast<void *>(state->free_queue.buffer_ptrs[0]);
            void *host_buf = memory_manager_.resolve_host_ptr(dev_buf);
            if (host_buf != nullptr) {
                copy_to_dev_cb_(dev_buf, host_buf, sizeof(PerfBuffer));
            }
        }
        for (int t = 0; t < num_phase_threads; t++) {
            PhaseBufferState *state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
            void *dev_buf = reinterpret_cast<void *>(state->free_queue.buffer_ptrs[0]);
            void *host_buf = memory_manager_.resolve_host_ptr(dev_buf);
            if (host_buf != nullptr) {
                copy_to_dev_cb_(dev_buf, host_buf, sizeof(PhaseBuffer));
            }
        }
    }

    wmb();

    // Step 7: Pass base address to Runtime
    runtime.perf_data_base = reinterpret_cast<uint64_t>(perf_dev_ptr);
    LOG_DEBUG("Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("Performance profiling initialized (dynamic buffer mode)");
    return 0;
}

void PerformanceCollector::start_memory_manager(const ThreadFactory &thread_factory) {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    memory_manager_.start(
        perf_shared_mem_host_, perf_shared_mem_dev_, num_aicore_, PLATFORM_MAX_AICPU_THREADS, alloc_cb_, register_cb_,
        free_cb_, copy_to_dev_cb_, copy_from_dev_cb_, device_id_, thread_factory
    );
}

void PerformanceCollector::stop_memory_manager() {
    if (memory_manager_.is_running()) {
        memory_manager_.stop();
    }
}

void PerformanceCollector::signal_execution_complete() { execution_complete_.store(true); }

void PerformanceCollector::poll_and_collect(int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    execution_complete_.store(false);

    LOG_INFO("Collecting performance data");

    PerfDataHeader *header = get_perf_header(perf_shared_mem_host_);

    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);
    std::optional<std::chrono::steady_clock::time_point> idle_start;

    int total_records_collected = 0;
    int buffers_processed = 0;

    collected_perf_records_.clear();
    collected_perf_records_.resize(num_aicore_);
    collected_phase_records_.clear();
    collected_phase_records_.resize(PLATFORM_MAX_AICPU_THREADS);

    // Helper lambda: read total_tasks from device (memcpy mode) or host shadow
    auto read_total_tasks = [&]() -> uint32_t {
        if (is_memcpy_mode()) {
            PerfDataHeader *dev_header = get_perf_header(perf_shared_mem_dev_);
            uint32_t val = 0;
            copy_from_dev_cb_(
                &val, const_cast<const void *>(static_cast<volatile void *>(&dev_header->total_tasks)), sizeof(uint32_t)
            );
            header->total_tasks = val;
        }
        rmb();
        return header->total_tasks;
    };

    if (expected_tasks <= 0) {
        LOG_INFO("Waiting for AICPU to write total_tasks in PerfDataHeader...");
        idle_start = std::chrono::steady_clock::now();

        while (true) {
            uint32_t raw_total_tasks = read_total_tasks();

            if (raw_total_tasks > 0) {
                expected_tasks = static_cast<int>(raw_total_tasks);
                LOG_INFO("AICPU reported task count: %d", expected_tasks);
                break;
            }

            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR(
                    "Timeout waiting for AICPU task count after %ld seconds",
                    std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                );
                return;
            }

            // Process ready buffers while waiting
            ReadyBufferInfo info;
            if (memory_manager_.try_pop_ready(info)) {
                if (info.type == ProfBufferType::PERF_RECORD) {
                    PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(info.host_buffer_ptr);
                    rmb();
                    uint32_t count = buf->count;
                    if (count > PLATFORM_PROF_BUFFER_SIZE) count = PLATFORM_PROF_BUFFER_SIZE;
                    uint32_t core_index = info.index;
                    if (core_index < static_cast<uint32_t>(num_aicore_)) {
                        for (uint32_t i = 0; i < count; i++) {
                            collected_perf_records_[core_index].push_back(buf->records[i]);
                        }
                        total_records_collected += count;
                    }
                } else {
                    PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(info.host_buffer_ptr);
                    rmb();
                    uint32_t count = buf->count;
                    if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD))
                        count = PLATFORM_PHASE_RECORDS_PER_THREAD;
                    uint32_t tidx = info.index;
                    for (uint32_t i = 0; i < count; i++) {
                        collected_phase_records_[tidx].push_back(buf->records[i]);
                    }
                }
                memory_manager_.notify_copy_done({info.dev_buffer_ptr, info.type});
                buffers_processed++;
            }
        }
    }

    LOG_DEBUG("Initial expected tasks: %d", expected_tasks);

    idle_start.reset();
    int last_logged_expected = -1;

    while (total_records_collected < expected_tasks) {
        // Check for updated expected_tasks
        int current_expected = static_cast<int>(read_total_tasks());
        if (current_expected > expected_tasks) {
            expected_tasks = current_expected;
            if (last_logged_expected < 0) {
                LOG_INFO("Updated expected_tasks to %d (orchestrator progress)", expected_tasks);
                last_logged_expected = expected_tasks;
            }
        }

        ReadyBufferInfo info;
        if (memory_manager_.wait_pop_ready(info, std::chrono::milliseconds(100))) {
            idle_start.reset();

            if (info.type == ProfBufferType::PERF_RECORD) {
                PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(info.host_buffer_ptr);
                rmb();
                uint32_t count = buf->count;
                if (count > PLATFORM_PROF_BUFFER_SIZE) count = PLATFORM_PROF_BUFFER_SIZE;

                uint32_t core_index = info.index;
                if (core_index < static_cast<uint32_t>(num_aicore_)) {
                    for (uint32_t i = 0; i < count; i++) {
                        collected_perf_records_[core_index].push_back(buf->records[i]);
                    }
                    total_records_collected += count;
                }

                LOG_DEBUG(
                    "Collected %u perf records from core %u (total: %d/%d)", count, core_index, total_records_collected,
                    expected_tasks
                );

            } else {
                PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(info.host_buffer_ptr);
                rmb();
                uint32_t count = buf->count;
                if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD))
                    count = PLATFORM_PHASE_RECORDS_PER_THREAD;

                uint32_t tidx = info.index;
                for (uint32_t i = 0; i < count; i++) {
                    collected_phase_records_[tidx].push_back(buf->records[i]);
                }

                LOG_DEBUG("Collected %u phase records from thread %u", count, tidx);
            }

            memory_manager_.notify_copy_done({info.dev_buffer_ptr, info.type});
            buffers_processed++;

        } else {
            if (execution_complete_.load()) {
                ReadyBufferInfo drain_info;
                while (memory_manager_.try_pop_ready(drain_info)) {
                    if (drain_info.type == ProfBufferType::PERF_RECORD) {
                        PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(drain_info.host_buffer_ptr);
                        rmb();
                        uint32_t count = buf->count;
                        if (count > PLATFORM_PROF_BUFFER_SIZE) count = PLATFORM_PROF_BUFFER_SIZE;
                        uint32_t ci = drain_info.index;
                        if (ci < static_cast<uint32_t>(num_aicore_)) {
                            for (uint32_t i = 0; i < count; i++) {
                                collected_perf_records_[ci].push_back(buf->records[i]);
                            }
                            total_records_collected += count;
                        }
                    } else {
                        PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(drain_info.host_buffer_ptr);
                        rmb();
                        uint32_t count = buf->count;
                        if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD))
                            count = PLATFORM_PHASE_RECORDS_PER_THREAD;
                        uint32_t tidx = drain_info.index;
                        for (uint32_t i = 0; i < count; i++) {
                            collected_phase_records_[tidx].push_back(buf->records[i]);
                        }
                    }
                    memory_manager_.notify_copy_done({drain_info.dev_buffer_ptr, drain_info.type});
                    buffers_processed++;
                }
                LOG_INFO(
                    "Execution complete signal received, exiting with %d/%d records", total_records_collected,
                    expected_tasks
                );
                break;
            }
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }
            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR(
                    "Performance data collection idle timeout after %ld seconds",
                    std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                );
                LOG_ERROR("Collected %d / %d records before timeout", total_records_collected, expected_tasks);
                break;
            }
        }
    }

    LOG_INFO("Total buffers processed: %d", buffers_processed);
    LOG_INFO("Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("Incomplete collection (%d / %d records)", total_records_collected, expected_tasks);
    }

    LOG_INFO("Performance data collection complete");
}

void PerformanceCollector::drain_remaining_buffers() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    int drained_perf = 0;
    int drained_phase = 0;

    ReadyBufferInfo info;
    while (memory_manager_.try_pop_ready(info)) {
        if (info.type == ProfBufferType::PERF_RECORD) {
            PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(info.host_buffer_ptr);
            rmb();
            uint32_t count = buf->count;
            if (count > PLATFORM_PROF_BUFFER_SIZE) count = PLATFORM_PROF_BUFFER_SIZE;
            uint32_t core_index = info.index;
            if (core_index < static_cast<uint32_t>(num_aicore_)) {
                for (uint32_t i = 0; i < count; i++) {
                    collected_perf_records_[core_index].push_back(buf->records[i]);
                }
                drained_perf += count;
            }
        } else {
            PhaseBuffer *buf = reinterpret_cast<PhaseBuffer *>(info.host_buffer_ptr);
            rmb();
            uint32_t count = buf->count;
            if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD))
                count = PLATFORM_PHASE_RECORDS_PER_THREAD;
            uint32_t tidx = info.index;
            for (uint32_t i = 0; i < count; i++) {
                collected_phase_records_[tidx].push_back(buf->records[i]);
            }
            drained_phase += count;
        }

        memory_manager_.notify_copy_done({info.dev_buffer_ptr, info.type});
    }

    if (drained_perf > 0 || drained_phase > 0) {
        LOG_INFO("Drained remaining buffers: %d perf records, %d phase records", drained_perf, drained_phase);
    }

    if (drained_phase > 0) {
        has_phase_data_ = true;
    }
}

void PerformanceCollector::scan_remaining_perf_buffers() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    // In memcpy mode, sync all PerfBufferStates from device
    if (is_memcpy_mode()) {
        for (int i = 0; i < num_aicore_; i++) {
            PerfBufferState *host_state = get_perf_buffer_state(perf_shared_mem_host_, i);
            PerfBufferState *dev_state = get_perf_buffer_state(perf_shared_mem_dev_, i);
            copy_from_dev_cb_(host_state, dev_state, sizeof(PerfBufferState));
        }
    }
    rmb();

    int total_recovered = 0;

    for (int core_index = 0; core_index < num_aicore_; core_index++) {
        PerfBufferState *state = get_perf_buffer_state(perf_shared_mem_host_, core_index);

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) {
            continue;
        }

        void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
        if (host_ptr == nullptr) {
            LOG_ERROR(
                "scan_remaining_perf_buffers: no host mapping for dev_ptr=%p (core %d)",
                reinterpret_cast<void *>(buf_ptr), core_index
            );
            continue;
        }

        // In memcpy mode, sync the buffer from device
        if (is_memcpy_mode()) {
            copy_from_dev_cb_(host_ptr, reinterpret_cast<void *>(buf_ptr), sizeof(PerfBuffer));
        }

        PerfBuffer *buf = reinterpret_cast<PerfBuffer *>(host_ptr);
        uint32_t count = buf->count;
        if (count == 0) {
            continue;
        }
        if (count > PLATFORM_PROF_BUFFER_SIZE) {
            count = PLATFORM_PROF_BUFFER_SIZE;
        }

        for (uint32_t i = 0; i < count; i++) {
            collected_perf_records_[core_index].push_back(buf->records[i]);
        }
        total_recovered += count;
    }

    if (total_recovered > 0) {
        LOG_INFO("scan_remaining_perf_buffers: recovered %d records from active buffers", total_recovered);
    }
}

void PerformanceCollector::collect_phase_data() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    // In memcpy mode, sync AicpuPhaseHeader and PhaseBufferStates from device
    if (is_memcpy_mode()) {
        AicpuPhaseHeader *host_ph = get_phase_header(perf_shared_mem_host_, num_aicore_);
        AicpuPhaseHeader *dev_ph = get_phase_header(perf_shared_mem_dev_, num_aicore_);
        copy_from_dev_cb_(host_ph, dev_ph, sizeof(AicpuPhaseHeader));
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            PhaseBufferState *host_state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);
            PhaseBufferState *dev_state = get_phase_buffer_state(perf_shared_mem_dev_, num_aicore_, t);
            copy_from_dev_cb_(host_state, dev_state, sizeof(PhaseBufferState));
        }
    }
    rmb();

    AicpuPhaseHeader *phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);

    if (phase_header->magic != AICPU_PHASE_MAGIC) {
        LOG_INFO(
            "No phase profiling data found (magic mismatch: 0x%x vs 0x%x)", phase_header->magic, AICPU_PHASE_MAGIC
        );
        return;
    }

    int num_sched_threads = phase_header->num_sched_threads;
    if (num_sched_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Invalid num_sched_threads %d from shared memory (max=%d)", num_sched_threads, PLATFORM_MAX_AICPU_THREADS
        );
        return;
    }
    LOG_INFO("Collecting remaining phase data: %d scheduler threads", num_sched_threads);

    int total_slots = PLATFORM_MAX_AICPU_THREADS;

    int total_phase_records = 0;
    for (int t = 0; t < total_slots; t++) {
        PhaseBufferState *state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr != 0) {
            void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
            if (host_ptr == nullptr) {
                LOG_ERROR(
                    "collect_phase_data: no host mapping for dev_ptr=%p (thread %d)", reinterpret_cast<void *>(buf_ptr),
                    t
                );
                continue;
            }
            // In memcpy mode, sync the buffer from device
            if (is_memcpy_mode()) {
                copy_from_dev_cb_(host_ptr, reinterpret_cast<void *>(buf_ptr), sizeof(PhaseBuffer));
            }
            PhaseBuffer *pbuf = reinterpret_cast<PhaseBuffer *>(host_ptr);
            if (pbuf->count > 0) {
                uint32_t count = pbuf->count;
                if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD))
                    count = PLATFORM_PHASE_RECORDS_PER_THREAD;

                for (uint32_t i = 0; i < count; i++) {
                    collected_phase_records_[t].push_back(pbuf->records[i]);
                }
                total_phase_records += count;
            }
        }
    }

    // Log per-thread totals
    for (size_t t = 0; t < collected_phase_records_.size(); t++) {
        if (!collected_phase_records_[t].empty()) {
            size_t sched_count = 0, orch_count = 0;
            for (const auto &r : collected_phase_records_[t]) {
                if (is_scheduler_phase(r.phase_id)) sched_count++;
                else orch_count++;
            }
            LOG_INFO(
                "  Thread %zu: %zu records (sched=%zu, orch=%zu)", t, collected_phase_records_[t].size(), sched_count,
                orch_count
            );
        }
    }

    // Read orchestrator summary
    collected_orch_summary_ = phase_header->orch_summary;
    bool orch_valid = (collected_orch_summary_.magic == AICPU_PHASE_MAGIC);

    if (orch_valid) {
        LOG_INFO(
            "  Orchestrator: %" PRId64 " tasks, %.3fus", static_cast<int64_t>(collected_orch_summary_.submit_count),
            cycles_to_us(collected_orch_summary_.end_time - collected_orch_summary_.start_time)
        );
    } else {
        LOG_INFO("  Orchestrator: no summary data");
    }

    bool has_accumulated = has_phase_data_;
    if (!has_accumulated) {
        for (const auto &v : collected_phase_records_) {
            if (!v.empty()) {
                has_accumulated = true;
                break;
            }
        }
    }
    has_phase_data_ = (total_phase_records > 0 || orch_valid || has_accumulated);

    // Read core-to-thread mapping
    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread, phase_header->core_to_thread + num_cores);
        LOG_INFO("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO(
        "Phase data collection complete: %d remaining records, orch_summary=%s", total_phase_records,
        orch_valid ? "yes" : "no"
    );
}

// export_swimlane_json and finalize are identical to a2a3 — copy verbatim

int PerformanceCollector::export_swimlane_json(const std::string &output_path_arg) {
    const char *env_dir = std::getenv("SIMPLER_PERF_OUTPUT_DIR");
    const std::string output_path = (env_dir != nullptr && env_dir[0] != '\0') ? std::string(env_dir) : output_path_arg;

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

    struct stat st;
    if (stat(output_path.c_str(), &st) == -1) {
        if (mkdir(output_path.c_str(), 0755) != 0) {
            LOG_ERROR("Error: Failed to create output directory.");
            return -1;
        }
    }

    struct TaggedRecord {
        const PerfRecord *record;
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

    std::sort(tagged_records.begin(), tagged_records.end(), [](const TaggedRecord &a, const TaggedRecord &b) {
        return a.record->task_id < b.record->task_id;
    });

    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto &tagged : tagged_records) {
        if (tagged.record->start_time < base_time_cycles) base_time_cycles = tagged.record->start_time;
        if (tagged.record->dispatch_time > 0 && tagged.record->dispatch_time < base_time_cycles)
            base_time_cycles = tagged.record->dispatch_time;
    }
    if (has_phase_data_) {
        for (const auto &thread_records : collected_phase_records_) {
            for (const auto &pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) base_time_cycles = pr.start_time;
            }
        }
        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC && collected_orch_summary_.start_time > 0 &&
            collected_orch_summary_.start_time < base_time_cycles) {
            base_time_cycles = collected_orch_summary_.start_time;
        }
    }

    std::time_t now = time(nullptr);
    std::tm *timeinfo = std::localtime(&now);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);
    std::string filepath = output_path + "/perf_swimlane_" + std::string(time_buffer) + ".json";

    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    int version = has_phase_data_ ? 2 : 1;
    outfile << "{\n";
    outfile << "  \"version\": " << version << ",\n";
    outfile << "  \"tasks\": [\n";

    for (size_t i = 0; i < tagged_records.size(); ++i) {
        const auto &tagged = tagged_records[i];
        const auto &record = *tagged.record;

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
        int safe_fanout_count =
            (record.fanout_count >= 0 && record.fanout_count <= RUNTIME_MAX_FANOUT) ? record.fanout_count : 0;
        for (int j = 0; j < safe_fanout_count; ++j) {
            outfile << record.fanout[j];
            if (j < safe_fanout_count - 1) outfile << ", ";
        }
        outfile << "],\n";
        outfile << "      \"fanout_count\": " << record.fanout_count << "\n";
        outfile << "    }";
        if (i < tagged_records.size() - 1) outfile << ",";
        outfile << "\n";
    }
    outfile << "  ]";

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

    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO("=== JSON Export Complete ===");
    LOG_INFO("File: %s", filepath.c_str());
    LOG_INFO("Records: %u", record_count);

    return 0;
}

int PerformanceCollector::finalize(PerfUnregisterCallback unregister_cb, PerfFreeCallback free_cb) {
    if (perf_shared_mem_host_ == nullptr) {
        return 0;
    }

    stop_memory_manager();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // In memcpy mode, sync all buffer states from device for accurate free_queue reading
    if (is_memcpy_mode()) {
        for (int i = 0; i < num_aicore_; i++) {
            PerfBufferState *host_state = get_perf_buffer_state(perf_shared_mem_host_, i);
            PerfBufferState *dev_state = get_perf_buffer_state(perf_shared_mem_dev_, i);
            copy_from_dev_cb_(host_state, dev_state, sizeof(PerfBufferState));
        }
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            PhaseBufferState *host_state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);
            PhaseBufferState *dev_state = get_phase_buffer_state(perf_shared_mem_dev_, num_aicore_, t);
            copy_from_dev_cb_(host_state, dev_state, sizeof(PhaseBufferState));
        }
    }

    // Free buffers in free_queues and current_buf_ptr
    for (int i = 0; i < num_aicore_; i++) {
        PerfBufferState *state = get_perf_buffer_state(perf_shared_mem_host_, i);

        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            // Free host shadow if in memcpy mode
            if (is_memcpy_mode()) {
                void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(state->current_buf_ptr));
                if (host_ptr != nullptr && host_ptr != reinterpret_cast<void *>(state->current_buf_ptr)) {
                    std::free(host_ptr);
                }
                memory_manager_.dev_to_host_.erase(reinterpret_cast<void *>(state->current_buf_ptr));
            }
            free_cb(reinterpret_cast<void *>(state->current_buf_ptr));
        }

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t max_iters = PLATFORM_PROF_SLOT_COUNT;
        while (head != tail && max_iters-- > 0) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            if (buf_ptr != 0 && free_cb != nullptr) {
                if (is_memcpy_mode()) {
                    void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
                    if (host_ptr != nullptr && host_ptr != reinterpret_cast<void *>(buf_ptr)) {
                        std::free(host_ptr);
                    }
                    memory_manager_.dev_to_host_.erase(reinterpret_cast<void *>(buf_ptr));
                }
                free_cb(reinterpret_cast<void *>(buf_ptr));
            }
            head++;
        }
    }

    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState *state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);

        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            if (is_memcpy_mode()) {
                void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(state->current_buf_ptr));
                if (host_ptr != nullptr && host_ptr != reinterpret_cast<void *>(state->current_buf_ptr)) {
                    std::free(host_ptr);
                }
                memory_manager_.dev_to_host_.erase(reinterpret_cast<void *>(state->current_buf_ptr));
            }
            free_cb(reinterpret_cast<void *>(state->current_buf_ptr));
        }

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t max_iters = PLATFORM_PROF_SLOT_COUNT;
        while (head != tail && max_iters-- > 0) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            if (buf_ptr != 0 && free_cb != nullptr) {
                if (is_memcpy_mode()) {
                    void *host_ptr = memory_manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_ptr));
                    if (host_ptr != nullptr && host_ptr != reinterpret_cast<void *>(buf_ptr)) {
                        std::free(host_ptr);
                    }
                    memory_manager_.dev_to_host_.erase(reinterpret_cast<void *>(buf_ptr));
                }
                free_cb(reinterpret_cast<void *>(buf_ptr));
            }
            head++;
        }
    }

    if (unregister_cb != nullptr && was_registered_) {
        int rc = unregister_cb(perf_shared_mem_dev_, device_id_);
        if (rc != 0) {
            LOG_ERROR("halHostUnregister failed: %d", rc);
            return rc;
        }
    }

    // Free host shadow of shared memory
    if (is_memcpy_mode() && perf_shared_mem_host_ != nullptr && perf_shared_mem_host_ != perf_shared_mem_dev_) {
        std::free(perf_shared_mem_host_);
    }

    if (free_cb != nullptr && perf_shared_mem_dev_ != nullptr) {
        free_cb(perf_shared_mem_dev_);
    }

    perf_shared_mem_dev_ = nullptr;
    perf_shared_mem_host_ = nullptr;
    was_registered_ = false;
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    device_id_ = -1;
    alloc_cb_ = nullptr;
    register_cb_ = nullptr;
    free_cb_ = nullptr;
    copy_to_dev_cb_ = nullptr;
    copy_from_dev_cb_ = nullptr;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
