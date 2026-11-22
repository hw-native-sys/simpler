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
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace launch_entry_args_test {
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int copy_rc = 0;
    int copies = 0;
    int shared_initializations = 0;
    bool recording = false;
    bool allow_sync_initialization = true;
    LaunchProgressSink sink;
    uint64_t last_copy_bytes = 0;
    // Bytes to write before reporting `copy_rc`. Zero is the all-or-nothing
    // failure the other cases use; a positive value is the partial write a
    // real interrupted copy can leave in the block.
    uint64_t partial_bytes_before_failure = 0;
};
extern RtsState rts;

struct CaptureAnswer {
    aclError rc = ACL_SUCCESS;
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    int queries = 0;
    const void *last_stream = nullptr;
};
extern CaptureAnswer capture;

struct RtsLaunchCopy {
    struct alignas(64) Block {
        unsigned char bytes[64];
    };
    std::vector<Block> storage;
    size_t bytes = 0;
    void *data() { return storage.data(); }
    const void *data() const { return storage.data(); }
    KernelArgs header() const {
        KernelArgs value{};
        std::memcpy(&value, data(), sizeof(value));
        return value;
    }
};

class LaunchEntryArgs : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
    void TearDown() override { EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0); }

    RtsLaunchCopy copy_launch() {
        RtsLaunchCopy copy;
        copy.bytes = helper.launch_payload_bytes();
        copy.storage.resize((copy.bytes + 63) / 64);
        std::memcpy(copy.data(), helper.launch_payload(), copy.bytes);
        return copy;
    }

    int prepare_kernel() {
        runtime.set_active_callable_id(3);
        helper.args.run_result_epoch = 19;
        const int rc = helper.prepare_runtime_args(runtime, allocator, slot, true);
        if (rc != 0) return rc;
        const auto plan = runtime_launch_entry_args_plan(runtime);
        return plan.supported ? helper.prepare_kernel_invocation({3, 7, plan.tensor_count, plan.scalar_count}) : 0;
    }

    int prepare() { return helper.prepare_runtime_args(runtime, allocator, slot); }
    int publish(bool permitted) { return helper.publish_runtime_args(permitted, this, rts.sink); }

    // One prepare+publish, with the launch route permitted or not.
    int run_once(bool permitted) {
        const int rc = prepare();
        if (rc != 0) return rc;
        return publish(permitted);
    }

    // The published descriptor, as the device would read it.
    const DeviceRuntimeLaunchDesc &device_descriptor() const {
        return *reinterpret_cast<const DeviceRuntimeLaunchDesc *>(slot.runtime_args);
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;
};
}  // namespace launch_entry_args_test
