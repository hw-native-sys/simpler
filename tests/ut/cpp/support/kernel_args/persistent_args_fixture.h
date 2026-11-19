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
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "kernel_persistent_args.h"

namespace persistent_args_test {
constexpr size_t kBlockAlign = 64;
constexpr int kInjectedRc = -1701;
constexpr uint64_t kDeviceId = 3;

/**
 * Records every allocation, release and host-to-device copy, and can fail the
 * n-th of any of them. Blocks are 64-byte aligned so a recorded device image
 * can be read back through the type that was copied into it.
 *
 * Its fill_arch_fields draws the register table from its own alloc, which is
 * the obligation the real arch implementations carry too — otherwise the
 * balance assertions below would not hold.
 */
struct FakeArgsOps {
    struct Copy {
        void *dst;
        size_t dst_bytes;
        size_t src_bytes;
    };

    ~FakeArgsOps() {
        for (void *block : live)
            ::free(block);
    }

    static FakeArgsOps *self(void *context) { return static_cast<FakeArgsOps *>(context); }

    static void *alloc(void *context, size_t bytes) {
        FakeArgsOps *ops = self(context);
        ops->calls.push_back("alloc");
        ++ops->alloc_calls;
        if (ops->alloc_calls == ops->fail_alloc_on) return nullptr;
        const size_t rounded = ((bytes + kBlockAlign - 1) / kBlockAlign) * kBlockAlign;
        void *block = ::aligned_alloc(kBlockAlign, rounded);
        if (block != nullptr) {
            // Every block starts at `fill_byte` (0 unless a test asks otherwise).
            // A test that needs to see what a copy did NOT overwrite sets this
            // before preparing, so the mark is already in place when the real
            // copy runs — writing it afterwards would paint over the evidence.
            std::memset(block, ops->fill_byte, rounded);
            ops->live.insert(block);
            ops->sizes[block] = bytes;
        }
        return block;
    }

    static int free_(void *context, void *ptr) {
        FakeArgsOps *ops = self(context);
        ops->calls.push_back("free");
        ++ops->free_calls;
        ops->free_order.push_back(ptr);
        if (ops->free_calls == ops->fail_free_on) return kInjectedRc;
        if (ops->live.erase(ptr) == 0) {
            ADD_FAILURE() << "release of a block this table does not own";
            return kInjectedRc;
        }
        ops->sizes.erase(ptr);
        ::free(ptr);
        return 0;
    }

    static int copy_h2d(void *context, void *dst, size_t dst_bytes, const void *src, size_t src_bytes) {
        FakeArgsOps *ops = self(context);
        ops->calls.push_back("copy");
        ++ops->copy_calls;
        ops->copies.push_back(Copy{dst, dst_bytes, src_bytes});
        if (ops->copy_calls == ops->fail_copy_on) return kInjectedRc;
        size_t bytes = src_bytes < dst_bytes ? src_bytes : dst_bytes;
        // A copy longer than its destination block is the defect this fake exists
        // to catch, so report it rather than committing it: an unclamped memcpy
        // would run past the allocation and make a negative control undefined
        // instead of diagnostic.
        const auto it = ops->sizes.find(dst);
        if (it != ops->sizes.end() && bytes > it->second) {
            ADD_FAILURE() << "copy of " << bytes << " bytes into a block of " << it->second;
            bytes = it->second;
        }
        std::memcpy(dst, src, bytes);
        return 0;
    }

    static int fill_arch_fields(void *context, KernelArgs *args, uint64_t device_id) {
        FakeArgsOps *ops = self(context);
        ops->calls.push_back("arch");
        ++ops->fill_calls;
        ops->last_device_id = device_id;
        if (ops->fill_calls == ops->fail_fill_on) return kInjectedRc;
        void *block = alloc(context, kBlockAlign);
        if (block == nullptr) return kInjectedRc;
        args->regs = reinterpret_cast<uint64_t>(block);
        return 0;
    }

    PersistentArgsOps table() { return PersistentArgsOps{this, &alloc, &free_, &copy_h2d, &fill_arch_fields}; }

    size_t live_blocks() const { return live.size(); }
    size_t block_size(void *block) const {
        const auto it = sizes.find(block);
        return it == sizes.end() ? 0 : it->second;
    }

    int alloc_calls = 0;
    int free_calls = 0;
    int copy_calls = 0;
    int fill_calls = 0;
    // Byte every fresh block is filled with. Stands in for what the device left
    // behind, so a test can tell "this range was not copied into" from "this
    // range happened to be zero".
    unsigned char fill_byte = 0;
    int fail_alloc_on = 0;
    int fail_free_on = 0;
    int fail_copy_on = 0;
    int fail_fill_on = 0;
    uint64_t last_device_id = 0;
    std::vector<Copy> copies;
    std::vector<void *> free_order;
    std::set<void *> live;
    std::map<void *, size_t> sizes;
    std::vector<std::string> calls;
};

}  // namespace persistent_args_test
