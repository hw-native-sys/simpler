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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <system_error>
#include <type_traits>
#include <pthread.h>

namespace simpler {

// Small trivially-copyable selectors live in pthread's scalar slots, not in
// heap cells or C++ ELF TLS. The module owns the keys; no thread-exit callback
// can retain or call into the module. Users must quiesce before module unload.
template <typename T>
class __attribute__((visibility("hidden"))) ModuleThreadValue {
    static_assert(std::is_trivially_copyable_v<T>);
    static constexpr size_t count = (sizeof(T) + sizeof(uintptr_t) - 1) / sizeof(uintptr_t);
    using Words = std::array<uintptr_t, count>;

public:
    ModuleThreadValue() noexcept :
        ModuleThreadValue(T{}) {}
    explicit ModuleThreadValue(T initial) noexcept :
        initial_(initial) {
        for (; owned_ < count; ++owned_) {
            error_ = pthread_key_create(&keys_[owned_], nullptr);
            if (error_ != 0) {
                release();
                break;
            }
        }
    }
    ~ModuleThreadValue() { release(); }
    ModuleThreadValue(const ModuleThreadValue &) = delete;
    ModuleThreadValue &operator=(const ModuleThreadValue &) = delete;
    int error() const noexcept { return error_; }
    bool available() const noexcept { return error_ == 0 && !cache_disabled_.load(std::memory_order_relaxed); }
    // Optional diagnostic caches stop selecting old values after a failed update.
    bool cache(const T &value) noexcept {
        if (!available()) return false;
        Words words{}, initial{};
        std::memcpy(words.data(), &value, sizeof(T));
        std::memcpy(initial.data(), &initial_, sizeof(T));
        for (size_t i = 0; i < count; ++i) {
            if (pthread_setspecific(keys_[i], reinterpret_cast<void *>(words[i] ^ initial[i])) == 0) continue;
            // No borrowed lifetime depends on an optional cache. Disable its
            // reads instead of risking another failure while rolling it back.
            cache_disabled_.store(true, std::memory_order_relaxed);
            return false;
        }
        return true;
    }
    T get() const noexcept {
        if (!available()) return initial_;
        Words words{};
        std::memcpy(words.data(), &initial_, sizeof(T));
        for (size_t i = 0; i < count; ++i)
            words[i] ^= reinterpret_cast<uintptr_t>(pthread_getspecific(keys_[i]));
        T value = initial_;
        std::memcpy(&value, words.data(), sizeof(T));
        return value;
    }
    int try_set(const T &value) noexcept {
        if (error_ != 0) return error_;
        Words words{}, initial{};
        std::memcpy(words.data(), &value, sizeof(T));
        std::memcpy(initial.data(), &initial_, sizeof(T));
        std::array<void *, count> previous{};
        for (size_t i = 0; i < count; ++i)
            previous[i] = pthread_getspecific(keys_[i]);
        for (size_t i = 0; i < count; ++i) {
            const int rc = pthread_setspecific(keys_[i], reinterpret_cast<void *>(words[i] ^ initial[i]));
            if (rc == 0) continue;
            while (i != 0) {
                --i;
                if (pthread_setspecific(keys_[i], previous[i]) != 0) std::terminate();
            }
            return rc;
        }
        return 0;
    }
    void set(const T &value) {
        if (const int rc = try_set(value)) throw std::system_error(rc, std::generic_category(), "module thread state");
    }
    // Used when borrowed storage is about to die: a failed clear/restore must
    // not return with a selector pointing at released storage.
    void restore(const T &value) noexcept {
        if (try_set(value) != 0) std::terminate();
    }

private:
    void release() noexcept {
        while (owned_ != 0)
            pthread_key_delete(keys_[--owned_]);
    }
    T initial_;
    std::array<pthread_key_t, count> keys_{};
    size_t owned_{0};
    int error_{0};
    std::atomic<bool> cache_disabled_{false};
};

}  // namespace simpler
