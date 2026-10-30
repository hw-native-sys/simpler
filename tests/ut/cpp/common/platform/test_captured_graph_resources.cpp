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

#include <memory>
#include <vector>

#include "host/captured_graph_resources.h"

namespace {
struct Notifications {
    struct Entry {
        void (*callback)(void *);
        void *data;
    };
    std::vector<Entry> entries;
    int rc{0};
    int calls{0};
    int operator()(void *, void (*callback)(void *), void *data) {
        ++calls;
        if (rc == 0) entries.push_back({callback, data});
        return rc;
    }
    void destroy(size_t i) { entries.at(i).callback(entries.at(i).data); }
};
}  // namespace

TEST(CapturedGraphResources, NotificationDoesNotReleaseResourcesOrProveDrain) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto resource = std::make_shared<int>(23);
    std::weak_ptr<int> observed = resource;
    int model = 0;
    ASSERT_EQ(owner.retain(&model, {resource}, notifications), 0);
    resource.reset();
    EXPECT_FALSE(observed.expired());
    EXPECT_EQ(owner.retire_after_drain(), PTO_RUNTIME_ERR_INVALID_STATE);
    notifications.destroy(0);
    EXPECT_FALSE(owner.has_live_model());
    EXPECT_FALSE(observed.expired());
    ASSERT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(observed.expired());
    EXPECT_TRUE(owner.empty());
}

TEST(CapturedGraphResources, NodesShareOneCallbackAndRetainTheirActualOwners) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto first = std::make_shared<int>(23), second = std::make_shared<int>(47);
    std::weak_ptr<int> first_ref = first, second_ref = second;
    int model = 0;
    ASSERT_EQ(owner.retain(&model, {first}, notifications), 0);
    ASSERT_EQ(owner.retain(&model, {first, second}, notifications), 0);
    EXPECT_EQ(notifications.calls, 1);
    EXPECT_EQ(first.use_count(), 2);
    first.reset();
    second.reset();
    EXPECT_FALSE(first_ref.expired());
    EXPECT_FALSE(second_ref.expired());
    notifications.destroy(0);
    ASSERT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(first_ref.expired());
    EXPECT_TRUE(second_ref.expired());
}

TEST(CapturedGraphResources, RecycledModelAddressHasIndependentNotificationStorage) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto resource = std::make_shared<int>(23);
    std::weak_ptr<int> observed = resource;
    int reused_model = 0;
    ASSERT_EQ(owner.retain(&reused_model, {resource}, notifications), 0);
    notifications.destroy(0);
    ASSERT_EQ(owner.retain(&reused_model, {resource}, notifications), 0);
    resource.reset();
    EXPECT_EQ(notifications.calls, 2);
    EXPECT_NE(notifications.entries[0].data, notifications.entries[1].data);
    EXPECT_TRUE(owner.has_live_model());
    EXPECT_EQ(owner.retire_after_drain(), PTO_RUNTIME_ERR_INVALID_STATE);
    notifications.destroy(1);
    ASSERT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(observed.expired());
}

TEST(CapturedGraphResources, RegistrationFailureLeavesExistingGraphProtected) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto first = std::make_shared<int>(23), second = std::make_shared<int>(47);
    std::weak_ptr<int> first_ref = first, second_ref = second;
    int model1 = 0, model2 = 0;
    ASSERT_EQ(owner.retain(&model1, {first}, notifications), 0);
    notifications.rc = -91;
    EXPECT_EQ(owner.retain(&model2, {second}, notifications), -91);
    first.reset();
    second.reset();
    EXPECT_FALSE(first_ref.expired());
    EXPECT_TRUE(second_ref.expired());
    EXPECT_TRUE(owner.has_live_model());
    notifications.destroy(0);
    ASSERT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(first_ref.expired());
}

TEST(CapturedGraphResources, RejectsMissingResourcesBeforeCallbackRegistration) {
    CapturedGraphResources owner;
    Notifications notifications;
    int model = 0;
    EXPECT_EQ(owner.retain(nullptr, {}, notifications), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(owner.retain(&model, {nullptr}, notifications), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(notifications.calls, 0);
    EXPECT_TRUE(owner.empty());
}

TEST(CapturedGraphResources, DestructionDuringRegistrationCannotAdmitRecording) {
    CapturedGraphResources owner;
    auto resource = std::make_shared<int>(23);
    std::weak_ptr<int> observed = resource;
    int model = 0;
    EXPECT_EQ(
        owner.retain(
            &model, {resource},
            [](void *, void (*notify)(void *), void *data) {
                notify(data);
                return 0;
            }
        ),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    resource.reset();
    EXPECT_FALSE(observed.expired());
    EXPECT_FALSE(owner.has_live_model());
    ASSERT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(observed.expired());
}

namespace {
struct FallibleResource {
    int rc{0};
    int releases{0};
    static int release(void *data) {
        auto &resource = *static_cast<FallibleResource *>(data);
        ++resource.releases;
        return resource.rc;
    }
};
}  // namespace

TEST(CapturedGraphResources, FailedRetirementKeepsOwnersAndDoesNotRepeatCompletedCleanup) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto first = std::make_shared<FallibleResource>();
    auto second = std::make_shared<FallibleResource>();
    int model = 0;
    const auto release = &FallibleResource::release;
    ASSERT_EQ(owner.retain(&model, {}, notifications, nullptr, {}, {{first, release}, {second, release}}), 0);
    ASSERT_EQ(owner.retain(&model, {}, notifications, nullptr, {}, {{first, release}}), 0);
    EXPECT_EQ(owner.retire_after_drain(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(first->releases, 0);
    notifications.destroy(0);
    second->rc = -79;
    EXPECT_EQ(owner.retire_after_drain(), -79);
    EXPECT_FALSE(owner.empty());
    EXPECT_EQ(first->releases, 1);
    EXPECT_EQ(second->releases, 1);
    EXPECT_EQ(first.use_count(), 2);
    second->rc = 0;
    EXPECT_EQ(owner.retire_after_drain(), 0);
    EXPECT_TRUE(owner.empty());
    EXPECT_EQ(first->releases, 1);
    EXPECT_EQ(second->releases, 2);
}

TEST(CapturedGraphResources, RefusedRegistrationLeavesFallibleResourceWithItsCaller) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto resource = std::make_shared<FallibleResource>();
    int model = 0;
    notifications.rc = -91;
    EXPECT_EQ(owner.retain(&model, {}, notifications, nullptr, {}, {{resource, &FallibleResource::release}}), -91);
    EXPECT_TRUE(owner.empty());
    EXPECT_EQ(resource->releases, 0);
    EXPECT_EQ(resource.use_count(), 1);
    EXPECT_EQ(
        owner.retain(&model, {}, notifications, nullptr, {}, {{resource, nullptr}}), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(CapturedGraphResources, RefusedAdditionalWorkspaceClaimRollsBackOwnersAndCallerStreams) {
    CapturedGraphResources owner;
    Notifications notifications;
    WorkspaceManager workspace;
    auto first = std::make_shared<int>(23), refused = std::make_shared<int>(47);
    int model = 0, first_stream = 0, refused_stream = 0, unknown = 0;
    ASSERT_EQ(owner.retain(&model, {first}, notifications, nullptr, {}, {}, {&first_stream}), 0);
    EXPECT_EQ(
        owner.retain(&model, {refused}, notifications, &workspace, {&unknown}, {}, {&refused_stream}),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(refused.use_count(), 1);
    notifications.destroy(0);
    std::vector<void *> drained;
    EXPECT_EQ(
        owner.drain_and_retire([&](void *stream) {
            drained.push_back(stream);
            return 0;
        }),
        0
    );
    ASSERT_EQ(drained.size(), 1u);
    EXPECT_EQ(drained[0], &first_stream);
}

TEST(CapturedGraphResources, FailedCallerDrainKeepsAllResourcesBeforeCleanup) {
    CapturedGraphResources owner;
    Notifications notifications;
    auto resource = std::make_shared<FallibleResource>();
    int model = 0, stream = 0;
    ASSERT_EQ(
        owner.retain(&model, {}, notifications, nullptr, {}, {{resource, &FallibleResource::release}}, {&stream}), 0
    );
    int calls = 0;
    EXPECT_EQ(
        owner.drain_and_retire([&](void *) {
            ++calls;
            return 0;
        }),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(calls, 0);
    notifications.destroy(0);
    EXPECT_EQ(
        owner.drain_and_retire([](void *) {
            return -55;
        }),
        -55
    );
    EXPECT_EQ(resource->releases, 0);
    EXPECT_FALSE(owner.empty());
    EXPECT_EQ(
        owner.drain_and_retire([](void *) {
            return 0;
        }),
        0
    );
    EXPECT_EQ(resource->releases, 1);
}

TEST(CapturedGraphResources, CallableProtectionTracksAllModelsThroughFailedDrain) {
    CapturedGraphResources owner;
    Notifications notifications;
    int models[2]{}, stream{};
    ASSERT_EQ(owner.retain(&models[0], {}, notifications, nullptr, {}, {}, {&stream}, {3, 5}), 0);
    ASSERT_EQ(owner.retain(&models[1], {}, notifications, nullptr, {}, {}, {&stream}, {5}), 0);
    EXPECT_TRUE(owner.holds_callable(3));
    EXPECT_TRUE(owner.holds_callable(5));
    EXPECT_FALSE(owner.holds_callable(7));
    notifications.destroy(0);
    EXPECT_EQ(owner.retire_after_drain(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(owner.holds_callable(3));
    notifications.destroy(1);
    EXPECT_EQ(
        owner.drain_and_retire([](void *) {
            return -73;
        }),
        -73
    );
    EXPECT_TRUE(owner.holds_callable(3));
    EXPECT_TRUE(owner.holds_callable(5));
    EXPECT_EQ(
        owner.drain_and_retire([](void *) {
            return 0;
        }),
        0
    );
    EXPECT_FALSE(owner.holds_callable(3));
    EXPECT_FALSE(owner.holds_callable(5));
}

TEST(CapturedGraphResources, FailedCallbackRegistrationDoesNotProtectCallable) {
    CapturedGraphResources owner;
    Notifications notifications;
    notifications.rc = -73;
    int model{};
    EXPECT_EQ(owner.retain(&model, {}, notifications, nullptr, {}, {}, {}, {3}), -73);
    EXPECT_FALSE(owner.holds_callable(3));
}
