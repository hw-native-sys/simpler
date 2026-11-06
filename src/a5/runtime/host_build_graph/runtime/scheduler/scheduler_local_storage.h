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

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace scheduler_local_storage_detail {
struct PlacementTag {};
}  // namespace scheduler_local_storage_detail

// CCEC cannot call the standard library's host-only placement-new overload.
// This private tagged overload only returns caller-owned local storage: it
// never allocates, frees, or addresses GM. The new-expression starts T's C++17
// lifetime after the caller has established its actual address alignment.
inline __aicore__ void *
operator new(std::size_t, scheduler_local_storage_detail::PlacementTag, void *address) noexcept {
    return address;
}
inline __aicore__ void operator delete(void *, scheduler_local_storage_detail::PlacementTag, void *) noexcept {}

// The address must refer to correctly aligned, caller-owned LOCAL storage.
// It may start a first lifetime or replace an old trivial lifetime without an
// aggregate temporary (unlike `*object = T{}`). Never use this on atomic GM rows.
template <typename T, typename... Args>
inline __aicore__ T &scheduler_reset_local(T *object, Args &&...args) {
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>);
    return *::new (scheduler_local_storage_detail::PlacementTag{}, static_cast<void *>(object))
        T(static_cast<Args &&>(args)...);
}

// Keep the carrier itself at ordinary scalar alignment. CCEC O3 can lower a
// declared alignas(128) automatic object's frame offset to a non-aligned address
// while retaining OR-based field addressing that assumes 128-byte alignment.
// Round the address of a sufficiently large byte buffer at runtime instead;
// wire types keep their original alignment, size and field offsets unchanged.
template <typename T>
class alignas(8) SchedulerLocalStorage {
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>);

    uint8_t storage_[sizeof(T) + alignof(T) - 1];
    T *object_;

public:
    template <typename... Args>
    inline __aicore__ explicit SchedulerLocalStorage(Args &&...args) {
        const uintptr_t address =
            (reinterpret_cast<uintptr_t>(storage_) + alignof(T) - 1) & ~(uintptr_t(alignof(T)) - 1);
        object_ = &scheduler_reset_local(reinterpret_cast<T *>(address), static_cast<Args &&>(args)...);
    }

    SchedulerLocalStorage(const SchedulerLocalStorage &) = delete;
    SchedulerLocalStorage &operator=(const SchedulerLocalStorage &) = delete;
    SchedulerLocalStorage(SchedulerLocalStorage &&) = delete;
    SchedulerLocalStorage &operator=(SchedulerLocalStorage &&) = delete;

    inline __aicore__ T &get() { return *object_; }
    inline __aicore__ const T &get() const { return *object_; }
    inline __aicore__ T *operator->() { return object_; }
    inline __aicore__ const T *operator->() const { return object_; }

    template <typename... Args>
    inline __aicore__ void reset(Args &&...args) {
        object_ = &scheduler_reset_local(object_, static_cast<Args &&>(args)...);
    }
};
