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

#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <pthread.h>

#include "aicpu/platform_aicpu_affinity.h"

namespace simpler {
// One per-thread affinity binding. The record, not the bound cell, owns the
// identity: invalidating a stale binding only clears thread-exclusive state, so
// it never writes a cell that a later cohort has bound to another thread.
// Records are trivially copyable with no destructor, and the key registers no
// thread-exit callback.
struct AffinityTlsRecord {
    int32_t *cell{nullptr};  // cell updated by publish(); nullptr while unbound
    int32_t value{-1};       // identity returned while cell is bound
};

// Two records per gate participant. MAX_GATE_THREADS covers the launch budget;
// the second half leaves room for host-bound threads and for a thread that exits
// without returning its record.
inline constexpr int kAffinityRecordPoolSize = 2 * MAX_GATE_THREADS;

// One module-owned key holding a per-thread record from a fixed pool. Failed
// creation, pool exhaustion (ENOSPC) and pthread_setspecific failures are
// reported to the caller. A null bind invalidates the record and returns it to
// the pool; a thread that exits without unbinding keeps its own record checked
// out.
class __attribute__((visibility("hidden"))) AffinityTlsKey {
public:
    ~AffinityTlsKey() {
        if (ready_.load(std::memory_order_relaxed)) pthread_key_delete(key_);
    }

    // Attaches `cell` to this thread's record and snapshots its value. Every
    // call performs a checked pthread_setspecific so binding failures stay
    // observable per launch.
    int bind(int32_t *cell) {
        if (cell == nullptr) return unbind();
        const int key_error = ensure_key();
        if (key_error != 0) return key_error;
        AffinityTlsRecord *record = record_ptr();
        bool fresh = false;
        if (record == nullptr) {
            record = checkout_record();
            if (record == nullptr) return ENOSPC;
            fresh = true;
        }
        const int set_error = pthread_setspecific(key_, record);
        if (set_error != 0) {
            if (fresh) return_record(record);
            return set_error;
        }
        record->cell = cell;
        record->value = *cell;
        return 0;
    }

    // Clears this thread's identity without touching the bound cell.
    void invalidate() {
        AffinityTlsRecord *record = record_ptr();
        if (record == nullptr) return;
        record->cell = nullptr;
        record->value = -1;
    }

    int32_t value() const {
        const AffinityTlsRecord *record = record_ptr();
        return record != nullptr && record->cell != nullptr ? record->value : -1;
    }

    // Writes the resolved identity to the record and its bound cell.
    bool publish(int32_t value) {
        AffinityTlsRecord *record = record_ptr();
        if (record == nullptr || record->cell == nullptr) return false;
        record->value = value;
        *record->cell = value;
        return true;
    }

private:
    int unbind() {
        // A gate entry unbinds before it knows whether this thread ever bound,
        // so the checked pthread_setspecific runs on the fresh-thread path too.
        const int key_error = ensure_key();
        if (key_error != 0) return key_error;
        AffinityTlsRecord *record = record_ptr();
        if (record != nullptr) {
            record->cell = nullptr;
            record->value = -1;
        }
        const int error = pthread_setspecific(key_, nullptr);
        if (error == 0 && record != nullptr) return_record(record);
        return error;
    }

    AffinityTlsRecord *record_ptr() const {
        if (!ready_.load(std::memory_order_acquire)) return nullptr;
        return static_cast<AffinityTlsRecord *>(pthread_getspecific(key_));
    }

    int ensure_key() {
        if (ready_.load(std::memory_order_acquire)) return 0;
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_.load(std::memory_order_relaxed)) return 0;
        const int error = pthread_key_create(&key_, nullptr);
        if (error != 0) return error;
        ready_.store(true, std::memory_order_release);
        return 0;
    }

    AffinityTlsRecord *checkout_record() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < kAffinityRecordPoolSize; ++i) {
            if (!in_use_[i]) {
                in_use_[i] = true;
                records_[i] = AffinityTlsRecord{};
                return &records_[i];
            }
        }
        return nullptr;
    }

    void return_record(AffinityTlsRecord *record) {
        if (record < records_ || record >= records_ + kAffinityRecordPoolSize) return;
        std::lock_guard<std::mutex> lock(mutex_);
        in_use_[static_cast<int>(record - records_)] = false;
    }

    pthread_key_t key_{};
    std::atomic<bool> ready_{false};
    std::mutex mutex_;
    AffinityTlsRecord records_[kAffinityRecordPoolSize]{};
    bool in_use_[kAffinityRecordPoolSize]{};
};

// Backend classification remains separate. Every advertised participant must
// copy this result before the backend's last-reader cleanup resets it. A new
// invocation must wait until all previous executors stop reading binding records
// and their cells.
class AffinityBindingCohort {
public:
    void record(int error) {
        if (error == 0) return;
        int expected = 0;
        error_.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
    }
    int arrive_and_read(int participants) {
        bound_.fetch_add(1, std::memory_order_release);
        while (bound_.load(std::memory_order_acquire) < participants) {}
        return error_.load(std::memory_order_acquire);
    }
    void reset_after_readers() {
        error_.store(0, std::memory_order_release);
        bound_.store(0, std::memory_order_release);
    }

private:
    std::atomic<int> error_{0};
    std::atomic<int> bound_{0};
};
}  // namespace simpler
