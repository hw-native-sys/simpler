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
#include <array>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <memory>
#include <pthread.h>
#include <system_error>
#include <type_traits>
#include <stdexcept>

namespace wse::temporary {
// Shared operation storage belongs to one Core's active Block, never to the process-global PTO table.
class BlockOperationStorage {
public:
    // Host and kernel DSOs may use different C++ standard libraries.
    struct Slot {
        explicit Slot(size_t size) :
            data(std::calloc(1, size)),
            bytes(size) {
            if (!data) throw std::bad_alloc();
        }
        Slot(const Slot &) = delete;
        Slot &operator=(const Slot &) = delete;
        ~Slot() {
            pthread_cond_destroy(&changed);
            pthread_mutex_destroy(&mutex);
            std::free(data);
        }
        void lock() {
            if (int error = pthread_mutex_lock(&mutex)) throw std::system_error(error, std::generic_category());
        }
        void unlock() noexcept { pthread_mutex_unlock(&mutex); }
        void notify_all() noexcept { pthread_cond_broadcast(&changed); }

        void *data;
        size_t bytes;
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
        bool initialized{false};
        bool cancelled{false};

        template <class Predicate>
        void wait(Predicate ready) {
            while (!cancelled && !ready()) {
                if (int error = pthread_cond_wait(&changed, &mutex))
                    throw std::system_error(error, std::generic_category());
            }
            if (cancelled) throw std::runtime_error("WSE shared operation cancelled");
        }
    };
    static_assert(std::is_standard_layout_v<Slot>);
    static void *get(void *context, uint32_t slot, size_t bytes) {
        auto &storage = *static_cast<BlockOperationStorage *>(context);
        std::lock_guard lock(storage.mutex_);
        if (storage.cancelled_) throw std::runtime_error("WSE shared operation cancelled");
        if (slot >= storage.slots_.size() || !bytes) throw std::invalid_argument("invalid WSE shared storage slot");
        auto &entry = storage.slots_[slot];
        if (entry && entry->bytes != bytes) throw std::invalid_argument("WSE shared storage shape mismatch");
        if (!entry) entry = std::make_shared<Slot>(bytes);
        return entry.get();
    }
    // All Block participants must have returned before their slots are destroyed.
    void clear() {
        std::lock_guard lock(mutex_);
        for (auto &entry : slots_)
            entry.reset();
    }

    void cancel() {
        decltype(slots_) slots;
        {
            std::lock_guard lock(mutex_);
            cancelled_ = true;
            slots = slots_;
        }
        // An operation trace callback may request another slot while holding its slot's mutex.
        // Keep the slots alive without holding the storage mutex while waking participants.
        for (const auto &slot : slots) {
            if (!slot) continue;
            std::lock_guard slot_lock(*slot);
            slot->cancelled = true;
            slot->notify_all();
        }
    }

private:
    std::mutex mutex_;
    std::array<std::shared_ptr<Slot>, 64> slots_{};
    bool cancelled_{false};
};
}  // namespace wse::temporary
