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
#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace launch_entry_args_test {
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int copy_rc = 0;
    int copies = 0;
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

class LaunchEntryArgs : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
    void TearDown() override { EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0); }

    int prepare() { return helper.prepare_runtime_args(runtime, allocator, slot); }
    int publish(bool permitted) { return helper.publish_runtime_args(permitted); }

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
