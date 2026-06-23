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
TEST(CallConfig, WireLayoutUnchanged) {
    EXPECT_EQ(sizeof(RuntimeEnv), RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t));
    EXPECT_EQ(sizeof(CallConfig), 7 * sizeof(int32_t) + RUNTIME_ENV_UINT64_FIELD_COUNT * sizeof(uint64_t) + 1024);
}

TEST(CallConfig, RuntimeEnvDefaultsAreUnset) {
    CallConfig cfg;
    EXPECT_EQ(cfg.runtime_env.ring_task_window, 0u);
    EXPECT_EQ(cfg.runtime_env.ring_heap, 0u);
    EXPECT_EQ(cfg.runtime_env.ring_dep_pool, 0u);
    for (int r = 0; r < RUNTIME_ENV_RING_COUNT; ++r) {
        EXPECT_EQ(cfg.runtime_env.ring_task_windows[r], 0u);
        EXPECT_EQ(cfg.runtime_env.ring_heaps[r], 0u);
        EXPECT_EQ(cfg.runtime_env.ring_dep_pools[r], 0u);
    }
    EXPECT_FALSE(cfg.runtime_env.per_ring_any());
    EXPECT_FALSE(cfg.runtime_env.any());
    EXPECT_NO_THROW(cfg.validate());
}

TEST(CallConfig, ValidRuntimeEnvPasses) {
    CallConfig cfg;
    cfg.runtime_env.ring_task_window = 64;
    cfg.runtime_env.ring_heap = 2621440;
    cfg.runtime_env.ring_dep_pool = 256;
    cfg.runtime_env.ring_task_windows[2] = 1024;
    cfg.runtime_env.ring_heaps[2] = 1536 * 1024 * 1024ull;
    cfg.runtime_env.ring_dep_pools[2] = 1024;
    EXPECT_TRUE(cfg.runtime_env.any());
    EXPECT_TRUE(cfg.runtime_env.per_ring_any());
    EXPECT_NO_THROW(cfg.validate());
}

TEST(CallConfig, RejectsRingTaskWindowBelowMinOrNonPow2) {
    CallConfig cfg;
    cfg.runtime_env.ring_task_window = 3;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_task_window = 48;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_task_window = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(CallConfig, RejectsRingHeapBelowMin) {
    CallConfig cfg;
    cfg.runtime_env.ring_heap = 512;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(CallConfig, RejectsRingDepPoolOutOfRange) {
    CallConfig cfg;
    cfg.runtime_env.ring_dep_pool = 3;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.runtime_env.ring_dep_pool = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(CallConfig, RejectsPerRingRuntimeEnvValues) {
    CallConfig cfg;
    cfg.runtime_env.ring_task_windows[1] = 48;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg = CallConfig{};
    cfg.runtime_env.ring_heaps[1] = 512;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    cfg = CallConfig{};
    cfg.runtime_env.ring_dep_pools[1] = static_cast<uint64_t>(INT32_MAX) + 1;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}
