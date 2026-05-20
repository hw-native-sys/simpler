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

#ifndef SRC_COMMON_PTO_RUNTIME2_DEVICE_POOL_H_
#define SRC_COMMON_PTO_RUNTIME2_DEVICE_POOL_H_

#include <cstddef>
#include <functional>

namespace pto_runtime2 {

/**
 * Single-slot pooled device buffer. Holds at most one allocation at a time.
 *
 * `acquire(size)` returns the current pointer if it is already large enough,
 * otherwise frees the existing buffer and allocates a fresh, larger one.
 * Designed for buffers whose lifetime should span many `worker.run()` calls
 * (e.g. PTO2 GM heap, PTO2 shared memory) — placing them in a DevicePool
 * lets the owner (per-Worker DeviceRunner) reuse the same device allocation
 * across runs and free it once at session teardown.
 *
 * The constructor takes raw `alloc(size_t) -> void*` / `free(void*)`
 * callables so unit tests can inject mock allocators without linking against
 * any real device runtime.
 */
class DevicePool {
public:
    using AllocFn = std::function<void *(size_t)>;
    using FreeFn = std::function<void(void *)>;

    DevicePool(AllocFn alloc_fn, FreeFn free_fn) :
        alloc_(std::move(alloc_fn)),
        free_(std::move(free_fn)) {}

    ~DevicePool() { release(); }

    DevicePool(const DevicePool &) = delete;
    DevicePool &operator=(const DevicePool &) = delete;

    void *acquire(size_t need) {
        if (need == 0) return nullptr;
        if (ptr_ != nullptr && need <= size_) return ptr_;
        if (ptr_ != nullptr) free_(ptr_);
        ptr_ = alloc_(need);
        size_ = (ptr_ != nullptr) ? need : 0;
        return ptr_;
    }

    void release() {
        if (ptr_ != nullptr) {
            free_(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void *ptr() const { return ptr_; }
    size_t size() const { return size_; }

private:
    AllocFn alloc_;
    FreeFn free_;
    void *ptr_{nullptr};
    size_t size_{0};
};

}  // namespace pto_runtime2

#endif  // SRC_COMMON_PTO_RUNTIME2_DEVICE_POOL_H_
