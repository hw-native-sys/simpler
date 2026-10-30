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

#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include "host/runtime_launch_metadata.h"

namespace {
int copy_bytes(void *target, const void *source, size_t bytes) {
    std::memcpy(target, source, bytes);
    return 0;
}
}  // namespace

TEST(RuntimeLaunchMetadata, MaterializedSourcesSurviveBuilderReuse) {
    std::array<unsigned, 2> inputs{23, 47};
    std::array<unsigned, 2> targets{};
    std::array<unsigned, 2> pinned{};
    unsigned replacement_target = 0;
    RuntimeLaunchMetadata::Publication publication;
    {
        RuntimeLaunchMetadata metadata;
        HostMetadataRegion regions[] = {
            {&targets[0], &inputs[0], sizeof(unsigned)}, {&targets[1], &inputs[1], sizeof(unsigned)}
        };
        ASSERT_EQ(metadata.prepare(regions, 2), 0);
        inputs.fill(99);
        ASSERT_EQ(metadata.materialize(pinned.data(), sizeof(pinned), publication), 0);
        EXPECT_EQ(metadata.size(), 0U);
        metadata.clear();
        const HostMetadataRegion replacement{&replacement_target, &inputs[0], sizeof(unsigned)};
        ASSERT_EQ(metadata.prepare(&replacement, 1), 0);
    }
    ASSERT_EQ(publication.enqueue(copy_bytes), 0);
    EXPECT_EQ(targets, (std::array<unsigned, 2>{23, 47}));
    targets.fill(0);
    ASSERT_EQ(publication.enqueue(copy_bytes), 0);
    EXPECT_EQ(targets, (std::array<unsigned, 2>{23, 47}));
    EXPECT_EQ(pinned, targets);
    EXPECT_EQ(replacement_target, 0U);
}

TEST(RuntimeLaunchMetadata, InsufficientCapacityChangesNeitherSourceNorBuilder) {
    RuntimeLaunchMetadata metadata;
    unsigned input = 23, target = 0, pinned = 99;
    const HostMetadataRegion region{&target, &input, sizeof(input)};
    ASSERT_EQ(metadata.prepare(&region, 1), 0);
    RuntimeLaunchMetadata::Publication publication;
    EXPECT_EQ(metadata.materialize(&pinned, sizeof(pinned) - 1, publication), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(pinned, 99U);
    EXPECT_EQ(metadata.size(), sizeof(input));
    EXPECT_EQ(publication.enqueue(copy_bytes), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(metadata.materialize(&pinned, sizeof(pinned), publication), 0);
    ASSERT_EQ(publication.enqueue(copy_bytes), 0);
    EXPECT_EQ(target, 23U);
}

TEST(RuntimeLaunchMetadata, AnExistingPublicationCannotBeOverwritten) {
    RuntimeLaunchMetadata metadata;
    unsigned input = 23, target = 0, pinned = 99;
    const HostMetadataRegion region{&target, &input, sizeof(input)};
    ASSERT_EQ(metadata.prepare(&region, 1), 0);
    RuntimeLaunchMetadata::Publication publication;
    ASSERT_EQ(metadata.materialize(&pinned, sizeof(pinned), publication), 0);
    input = 47;
    ASSERT_EQ(metadata.prepare(&region, 1), 0);
    EXPECT_EQ(metadata.materialize(&pinned, sizeof(pinned), publication), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(pinned, 23U);
    EXPECT_EQ(metadata.size(), sizeof(input));
    ASSERT_EQ(publication.enqueue(copy_bytes), 0);
    EXPECT_EQ(target, 23U);
}

TEST(RuntimeLaunchMetadata, MoveTransfersOffsetsAndInvalidatesTheOldRecord) {
    RuntimeLaunchMetadata metadata;
    std::array<unsigned, 2> input{23, 47}, target{}, pinned{};
    HostMetadataRegion regions[] = {
        {&target[0], &input[0], sizeof(unsigned)}, {&target[1], &input[1], sizeof(unsigned)}
    };
    ASSERT_EQ(metadata.prepare(regions, 2), 0);
    RuntimeLaunchMetadata::Publication original;
    ASSERT_EQ(metadata.materialize(pinned.data(), sizeof(pinned), original), 0);
    auto moved = std::move(original);
    EXPECT_EQ(original.enqueue(copy_bytes), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(moved.enqueue(copy_bytes), 0);
    EXPECT_EQ(target, input);
}

TEST(RuntimeLaunchMetadata, PartialEnqueueFailureKeepsTheWholePublication) {
    RuntimeLaunchMetadata metadata;
    std::array<unsigned, 3> input{23, 47, 71}, target{}, pinned{};
    HostMetadataRegion regions[] = {
        {&target[0], &input[0], sizeof(unsigned)},
        {&target[1], &input[1], sizeof(unsigned)},
        {&target[2], &input[2], sizeof(unsigned)}
    };
    ASSERT_EQ(metadata.prepare(regions, 3), 0);
    RuntimeLaunchMetadata::Publication publication;
    ASSERT_EQ(metadata.materialize(pinned.data(), sizeof(pinned), publication), 0);
    int calls = 0;
    EXPECT_EQ(
        publication.enqueue([&](void *dst, const void *src, size_t bytes) {
            if (++calls == 2) return -91;
            return copy_bytes(dst, src, bytes);
        }),
        -91
    );
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(target, (std::array<unsigned, 3>{23, 0, 0}));
    EXPECT_EQ(pinned, input);
    // The test copies synchronously; no accepted reader remains at this point.
    ASSERT_EQ(publication.enqueue(copy_bytes), 0);
    EXPECT_EQ(target, input);
}

TEST(RuntimeLaunchMetadata, EmptyPublicationNeedsNoSourceAndQueuesNoCopy) {
    RuntimeLaunchMetadata metadata;
    RuntimeLaunchMetadata::Publication publication;
    ASSERT_EQ(metadata.materialize(nullptr, 0, publication), 0);
    int calls = 0;
    EXPECT_EQ(
        publication.enqueue([&](void *, const void *, size_t) {
            return ++calls;
        }),
        0
    );
    EXPECT_EQ(calls, 0);
}

TEST(RuntimeLaunchMetadata, OneShotPublicationConsumesOnCopyFailure) {
    RuntimeLaunchMetadata metadata;
    unsigned input = 23, target = 0, pinned = 0;
    const HostMetadataRegion region{&target, &input, sizeof(input)};
    ASSERT_EQ(metadata.prepare(&region, 1), 0);
    EXPECT_EQ(
        metadata.publish(
            &pinned, sizeof(pinned),
            [](void *, const void *, size_t) {
                return -91;
            }
        ),
        -91
    );
    EXPECT_EQ(metadata.size(), 0U);
    EXPECT_EQ(pinned, 23U);
}

TEST(RuntimeLaunchMetadata, InvalidRegionsLeaveThePreviousSnapshotIntact) {
    RuntimeLaunchMetadata metadata;
    unsigned input = 23, target = 0, pinned = 0;
    const HostMetadataRegion region{&target, &input, sizeof(input)};
    ASSERT_EQ(metadata.prepare(&region, 1), 0);
    const HostMetadataRegion invalid[] = {region, {&target, &input, std::numeric_limits<size_t>::max()}};
    EXPECT_EQ(metadata.prepare(invalid, 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    ASSERT_EQ(metadata.publish(&pinned, sizeof(pinned), copy_bytes), 0);
    EXPECT_EQ(target, 23U);
}
