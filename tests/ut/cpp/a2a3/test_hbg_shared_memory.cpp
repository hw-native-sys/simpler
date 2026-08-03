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
/**
 * Unit tests for the shuffle_higher_bits invariant in
 * host_build_graph's pto_shared_memory.cpp: flag_index() shifts by
 * shuffle_higher_bits, so init must reject a task_window_size that would
 * make that shift amount negative rather than let it become undefined
 * behavior.
 */

#include <gtest/gtest.h>

#include <vector>

#include "pto_shared_memory.h"

namespace {

TEST(HbgSharedMemoryShuffleBits, CreateAndInitDefaultProducesNonNegativeShuffleHigherBits) {
    DeviceArena arena;
    PTO2SharedMemoryHandle *handle = PTO2SharedMemoryHandle::create_and_init_default(arena);
    ASSERT_NE(handle, nullptr);
    EXPECT_GE(handle->header->ring.shuffle_higher_bits, 0);
}

TEST(HbgSharedMemoryShuffleBits, InitPerRingRejectsTaskWindowBelowShuffleFloor) {
    // ctzll(8) == 3 < shuffle_lower_bits (4), so shuffle_higher_bits would be -1.
    constexpr uint64_t kTaskWindowSize = 8;
    static_assert(kTaskWindowSize < (uint64_t{1} << PTO2SharedMemoryRingHeader::shuffle_lower_bits), "");

    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = kTaskWindowSize;
        heap_sizes[r] = 4096;
    }

    const uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(task_window_sizes);
    std::vector<char> buf(sm_size);

    PTO2SharedMemoryHandle handle{};
    EXPECT_FALSE(handle.init_per_ring(buf.data(), sm_size, task_window_sizes, heap_sizes));
}

}  // namespace
