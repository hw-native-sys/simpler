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
    };
    static void *get(void *context, uint32_t slot, size_t bytes) {
        auto &storage = *static_cast<BlockOperationStorage *>(context);
        std::lock_guard lock(storage.mutex_);
        if (slot >= storage.slots_.size() || !bytes) throw std::invalid_argument("invalid WSE shared storage slot");
        auto &entry = storage.slots_[slot];
        if (entry && entry->bytes != bytes) throw std::invalid_argument("WSE shared storage shape mismatch");
        if (!entry) entry = std::make_unique<Slot>(bytes);
        return entry.get();
    }
    // All Block participants must have returned before their slots are destroyed.
    void clear() {
        std::lock_guard lock(mutex_);
        for (auto &entry : slots_)
            entry.reset();
    }

private:
    std::mutex mutex_;
    std::array<std::unique_ptr<Slot>, 64> slots_{};
};
}  // namespace wse::temporary
