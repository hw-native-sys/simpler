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

#include "scheduler/scheduler_types.h"

TEST(HostBuildGraphCoreTrackerTest, MixPlacementSelectionPreservesClustersAboveBit63) {
    CoreTracker tracker;
    tracker.init(CoreTracker::MAX_CLUSTERS);
    for (int32_t cluster = 0; cluster < CoreTracker::MAX_CLUSTERS; ++cluster) {
        int32_t offset = cluster * 3;
        tracker.set_cluster(cluster, offset, offset + 1, offset + 2);
    }

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0 | SUBTASK_MASK_AIV1;
    int32_t high_offset = (CoreTracker::MAX_CLUSTERS - 1) * 3;
    auto running = tracker.get_mix_cluster_offset_states(used_mask, CoreTracker::MixPlacement::RUNNING);
    EXPECT_EQ(running.count(), CoreTracker::MAX_CLUSTERS);
    EXPECT_TRUE((running & CoreTracker::BitStates::bit(high_offset)).has_value());

    for (int32_t offset = 0; offset < CoreTracker::MAX_CLUSTERS * 3; ++offset) {
        tracker.change_core_state(offset);
    }
    auto pending = tracker.get_mix_cluster_offset_states(used_mask, CoreTracker::MixPlacement::PENDING);
    EXPECT_EQ(pending.count(), CoreTracker::MAX_CLUSTERS);
    EXPECT_TRUE((pending & CoreTracker::BitStates::bit(high_offset)).has_value());
}
