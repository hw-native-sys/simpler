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
#include <memory>
#include <new>
#include <utility>

/**
 * One pipeline slot's retained pair of blocks for device-side scheduler state:
 * the device block the scheduler reads and the host block a bind builds that
 * state in before the single H2D that ships it.
 *
 * Grow-only and per slot. A request that fits the retained capacity is answered
 * with no allocation.
 *
 * Neither block is cleared here or on handover. The caller writes every byte of
 * the length it asks for before shipping it, so a retained block carries
 * nothing between runs. The host side is a raw `new[]` of a trivially-typed
 * array rather than a container for the same reason: a container's growth would
 * zero a capacity the caller overwrites and copy bytes that mean nothing
 * between runs.
 *
 * Both sides hold two addresses that are equal only when the allocator returns
 * an already-aligned block: the raw allocation, which is what a free takes, and
 * the aligned base inside it, which is what a caller is handed and what has
 * `bytes` behind it.
 *
 * Failure rules:
 *
 *   - growth prepares the host side first, the side that can fail without
 *     touching the device, so a host failure leaves the device plan as the last
 *     successful acquire left it;
 *   - a failed device allocation keeps the previous device block, its capacity
 *     and the aligned base already handed out;
 *   - the replacement is recorded before the predecessor is released, so no
 *     window has the slot naming a block that is already gone;
 *   - a release that fails leaves that one block in `held_after_failed_release`
 *     and the slot refuses every later growth, which bounds the slot at its
 *     current block plus that one and keeps the record from being overwritten.
 *     The failed block keeps whatever owner the caller's allocator gives it.
 *
 * Reports failure by return value and logs nothing: this header is included by
 * translation units that do not share one logging backend, and the caller has
 * the slot index its message needs.
 *
 * The device allocate and free are supplied per call, so this class has no
 * platform dependency and every failure path above is reachable in a test.
 * `device_alloc(bytes)` returns the raw block or nullptr; `device_free(ptr)`
 * takes a raw block and returns 0 on success.
 */
class RetainedSchedulerStorage {
public:
    enum class Status : uint32_t {
        Ok = 0,
        // Zero length, a non-power-of-two alignment, or a length whose
        // alignment padding would overflow.
        InvalidRequest = 1,
        HostUnavailable = 2,
        // The device block could not be replaced. The previous one, if any, is
        // still recorded and still the one handed out.
        DeviceUnavailable = 3,
        // A previous release failed and that block is still held, so this slot
        // grows no further.
        GrowthRefused = 4,
    };

    RetainedSchedulerStorage() = default;
    RetainedSchedulerStorage(const RetainedSchedulerStorage &) = delete;
    RetainedSchedulerStorage &operator=(const RetainedSchedulerStorage &) = delete;

    /**
     * Hand out both aligned bases, growing either side only if it is too small
     * or wrongly aligned for this request.
     */
    template <class DeviceAlloc, class DeviceFree>
    Status acquire(
        size_t bytes, size_t alignment, DeviceAlloc device_alloc, DeviceFree device_free, void **device_out,
        void **host_out
    ) {
        if (device_out != nullptr) *device_out = nullptr;
        if (host_out != nullptr) *host_out = nullptr;
        if (device_out == nullptr || host_out == nullptr || bytes == 0 || alignment == 0 ||
            (alignment & (alignment - 1)) != 0 || bytes > SIZE_MAX - (alignment - 1)) {
            return Status::InvalidRequest;
        }
        const size_t raw_bytes = bytes + alignment - 1;

        if (!fits(host_addr_, host_capacity_, bytes, alignment)) {
            std::unique_ptr<std::byte[]> storage(new (std::nothrow) std::byte[raw_bytes]);
            if (storage == nullptr) return Status::HostUnavailable;
            host_addr_ = align_up(storage.get(), alignment);
            host_storage_ = std::move(storage);
            host_capacity_ = bytes;
        }

        if (!fits(device_addr_, device_capacity_, bytes, alignment)) {
            if (failed_release_ != nullptr) return Status::GrowthRefused;
            void *allocation = device_alloc(raw_bytes);
            if (allocation == nullptr) return Status::DeviceUnavailable;
            void *previous = device_allocation_;
            device_allocation_ = allocation;
            device_addr_ = align_up(allocation, alignment);
            device_capacity_ = bytes;
            if (previous != nullptr && device_free(previous) != 0) {
                failed_release_ = previous;
            }
        }

        *device_out = device_addr_;
        *host_out = host_addr_;
        return Status::Ok;
    }

    /**
     * Release both device blocks this slot holds and forget the pair.
     *
     * Every block is attempted, so one failure cannot strand the other, and the
     * first failing code is returned. The entry is cleared either way: a failed
     * free leaves the address with the caller's allocator, and clearing here is
     * what makes a second release a no-op rather than a second free.
     */
    template <class DeviceFree>
    int release(DeviceFree device_free) {
        int first_error = 0;
        for (void *allocation : {device_allocation_, failed_release_}) {
            if (allocation == nullptr) continue;
            const int rc = device_free(allocation);
            if (rc != 0 && first_error == 0) first_error = rc;
        }
        forget();
        return first_error;
    }

    /** Forget the pair without any device call, after a force reset. */
    void abandon() { forget(); }

    /** The aligned base handed out, not the allocation it sits in. */
    void *device_addr() const { return device_addr_; }
    void *host_addr() const { return host_addr_; }
    size_t device_capacity() const { return device_capacity_; }
    size_t host_capacity() const { return host_capacity_; }
    /**
     * The raw allocation whose release failed, or null. Refuses growth while
     * set, and is a block a free takes rather than a base anything reads.
     */
    void *held_after_failed_release() const { return failed_release_; }

private:
    static void *align_up(void *p, size_t alignment) {
        const uintptr_t raw = reinterpret_cast<uintptr_t>(p);
        return reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
    }

    static bool fits(const void *addr, size_t capacity, size_t bytes, size_t alignment) {
        return addr != nullptr && capacity >= bytes && reinterpret_cast<uintptr_t>(addr) % alignment == 0;
    }

    void forget() {
        device_allocation_ = nullptr;
        device_addr_ = nullptr;
        device_capacity_ = 0;
        failed_release_ = nullptr;
        host_storage_.reset();
        host_addr_ = nullptr;
        host_capacity_ = 0;
    }

    void *device_allocation_{nullptr};
    void *device_addr_{nullptr};
    size_t device_capacity_{0};
    void *failed_release_{nullptr};
    std::unique_ptr<std::byte[]> host_storage_;
    void *host_addr_{nullptr};
    size_t host_capacity_{0};
};
