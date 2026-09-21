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

#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include "call_config.h"

// Wire contract: parent and forked child move CallConfig with one memcpy.
TEST(CallConfig, WireLayoutMatchesConstant) {
    EXPECT_EQ(sizeof(RuntimeEnv), RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t));
    EXPECT_EQ(sizeof(CallConfig), 7 * sizeof(int32_t) + RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t) + 1024);
}

TEST(CallConfig, RuntimeEnvDefaultsAreUnset) {
    CallConfig cfg;
    for (int r = 0; r < RUNTIME_ENV_RING_COUNT; ++r) {
        EXPECT_EQ(cfg.runtime_env.ring_task_window[r], 0u);
        EXPECT_EQ(cfg.runtime_env.ring_heap[r], 0u);
        EXPECT_EQ(cfg.runtime_env.ring_dep_pool[r], 0u);
    }
    EXPECT_FALSE(cfg.runtime_env.any());
    EXPECT_NO_THROW(cfg.validate());
}

TEST(CallConfig, ValidRuntimeEnvPasses) {
    CallConfig cfg;
    // A broadcast scalar arrives here as every entry set to the same value...
    for (int r = 0; r < RUNTIME_ENV_RING_COUNT; ++r) {
        cfg.runtime_env.ring_task_window[r] = 64;
        cfg.runtime_env.ring_heap[r] = 2621440;
        cfg.runtime_env.ring_dep_pool[r] = 256;
    }
    // ...and a per-ring override sets a single ring independently.
    cfg.runtime_env.ring_task_window[2] = 1024;
    cfg.runtime_env.ring_heap[2] = 1536 * 1024 * 1024ull;
    cfg.runtime_env.ring_dep_pool[2] = 1024;
    EXPECT_TRUE(cfg.runtime_env.any());
    EXPECT_NO_THROW(cfg.validate());
}

TEST(CallConfig, RejectsRingTaskWindowBelowMinOrNonPow2) {
    CallConfig cfg;
    cfg.runtime_env.ring_task_window[0] = 3;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_task_window[0] = 48;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_task_window[0] = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(CallConfig, RejectsRingHeapBelowMin) {
    CallConfig cfg;
    cfg.runtime_env.ring_heap[0] = 512;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(CallConfig, RejectsRingDepPoolOutOfRange) {
    CallConfig cfg;
    cfg.runtime_env.ring_dep_pool[0] = 3;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_dep_pool[0] = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// Constraints are enforced on every ring index, not just ring 0.
TEST(CallConfig, RejectsInvalidValuesAtNonZeroRing) {
    CallConfig cfg;
    cfg.runtime_env.ring_task_window[1] = 48;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg = CallConfig{};
    cfg.runtime_env.ring_heap[1] = 512;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg = CallConfig{};
    cfg.runtime_env.ring_dep_pool[1] = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}
