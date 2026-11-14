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

#include <gtest/gtest.h>

#include "host/notification_context_registry.h"

TEST(NotificationContextRegistry, ReadersKeepHostSnapshotsWithoutBlockingAllocationRemoval) {
    NotificationContextRegistry registry;
    HostNotificationContext snapshot;
    snapshot.bootstrap_token = 41;
    snapshot.allocation_id = 5;
    snapshot.context_address = 0x8000;
    snapshot.window_bytes = 4096;
    snapshot.local_rank = 0;
    snapshot.rank_count = 2;
    snapshot.rank_ids[0] = 3;
    snapshot.rank_ids[1] = 7;
    snapshot.windows_in[0] = 0x10000;
    snapshot.windows_in[1] = 0x20000;
    ASSERT_TRUE(registry.publish(2, snapshot));
    snapshot.context_address = 0x9000;
    snapshot.allocation_id = 6;
    ASSERT_TRUE(registry.publish(2, snapshot));
    EXPECT_FALSE(registry.acquire(3, 0x8000));
    EXPECT_FALSE(registry.acquire(2, 0x1234));
    auto reader = registry.acquire(2, 0x8000);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->bootstrap_token, 41u);
    EXPECT_EQ(reader->allocation_id, 5u);
    EXPECT_EQ(reader->rank_ids[1], 7u);
    registry.erase({2, 0x8000});
    EXPECT_FALSE(registry.acquire(2, 0x8000));
    EXPECT_NE(registry.acquire(2, 0x9000), nullptr);
    EXPECT_EQ(reader->allocation_id, 5u);
    EXPECT_FALSE(registry.publish(2, snapshot));
    registry.erase({2, 0x9000});
    EXPECT_FALSE(registry.acquire(2, 0x9000));
    snapshot.context_address = 0x8000;
    snapshot.allocation_id = 7;
    ASSERT_TRUE(registry.publish(2, snapshot));
    EXPECT_EQ(registry.acquire(2, 0x8000)->allocation_id, 7u);
    EXPECT_EQ(reader->allocation_id, 5u);
}
