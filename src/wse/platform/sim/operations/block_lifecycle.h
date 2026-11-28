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
#include <condition_variable>
#include <stdexcept>

namespace wse::temporary {
// Shared operation storage belongs to one Core's active Block, never to the process-global PTO table.
class BlockOperationStorage {
public:
    struct Slot {
        explicit Slot(size_t size) :
            data(std::calloc(1, size), &std::free),
            bytes(size) {
            if (!data) throw std::bad_alloc();
        }
        std::unique_ptr<void, decltype(&std::free)> data;
        size_t bytes;
        std::mutex mutex;
        std::condition_variable changed;
        bool initialized{false};
        bool cancelled{false};

        template <class Predicate>
        void wait(std::unique_lock<std::mutex> &lock, Predicate ready) {
            changed.wait(lock, [&] {
                return cancelled || ready();
            });
            if (cancelled) throw std::runtime_error("WSE shared operation cancelled");
        }
    };
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
            std::lock_guard slot_lock(slot->mutex);
            slot->cancelled = true;
            slot->changed.notify_all();
        }
    }

private:
    std::mutex mutex_;
    std::array<std::shared_ptr<Slot>, 64> slots_{};
    bool cancelled_{false};
};
}  // namespace wse::temporary
