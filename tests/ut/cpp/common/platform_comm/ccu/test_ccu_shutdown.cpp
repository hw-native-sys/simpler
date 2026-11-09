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

#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>
#include "platform_comm/ccu/ccu_shutdown.h"

namespace {
struct Service {
    std::atomic<unsigned> active_runs{0};
    bool closed{false};
    bool close_confirmed{true};
    unsigned close_calls{0};
    std::shared_ptr<int> parent = std::make_shared<int>(1);
};

bool close_service(Service &service) {
    ++service.close_calls;
    if (!service.close_confirmed) return false;
    service.parent.reset();
    return true;
}

TEST(CcuShutdown, UnconfirmedReaderPreventsEveryCloseAndKeepsOwners) {
    const auto first = std::make_shared<Service>();
    const auto second = std::make_shared<Service>();
    const std::vector services{first, second};
    std::weak_ptr<int> first_owner = first->parent;
    std::weak_ptr<int> second_owner = second->parent;
    second->active_runs = 1;
    EXPECT_FALSE(simpler::ccu::close_idle_services(services, close_service));
    EXPECT_EQ(first->close_calls, 0U);
    EXPECT_EQ(second->close_calls, 0U);
    EXPECT_FALSE(first_owner.expired());
    EXPECT_FALSE(second_owner.expired());
    // Only an independently proved last-use release changes the admission.
    second->active_runs = 0;
    EXPECT_TRUE(simpler::ccu::close_idle_services(services, close_service));
    EXPECT_TRUE(first_owner.expired());
    EXPECT_TRUE(second_owner.expired());
}

TEST(CcuShutdown, FailedCloseRetainsOwnersAndRetriesOnlyUnclosedServices) {
    const auto first = std::make_shared<Service>();
    const auto second = std::make_shared<Service>();
    const auto third = std::make_shared<Service>();
    const std::vector services{first, second, third};
    std::weak_ptr<int> second_owner = second->parent;
    std::weak_ptr<int> third_owner = third->parent;
    second->close_confirmed = false;
    EXPECT_FALSE(simpler::ccu::close_idle_services(services, close_service));
    EXPECT_TRUE(first->closed);
    EXPECT_FALSE(second->closed);
    EXPECT_FALSE(third->closed);
    EXPECT_EQ(third->close_calls, 0U);
    EXPECT_FALSE(second_owner.expired());
    EXPECT_FALSE(third_owner.expired());
    second->close_confirmed = true;
    EXPECT_TRUE(simpler::ccu::close_idle_services(services, close_service));
    EXPECT_EQ(first->close_calls, 1U);
    EXPECT_EQ(second->close_calls, 2U);
    EXPECT_EQ(third->close_calls, 1U);
    EXPECT_TRUE(second_owner.expired());
    EXPECT_TRUE(third_owner.expired());
    EXPECT_TRUE(simpler::ccu::close_idle_services(services, close_service));
    EXPECT_EQ(second->close_calls, 2U);
}

TEST(CcuShutdown, ThrowingCloseCannotPublishStoppedOrReleaseItsOwner) {
    const auto service = std::make_shared<Service>();
    const std::vector services{service};
    const auto throwing_close = [](Service &) -> bool {
        throw std::runtime_error("SDK close failed");
    };
    EXPECT_THROW(simpler::ccu::close_idle_services(services, throwing_close), std::runtime_error);
    EXPECT_FALSE(service->closed);
    EXPECT_NE(service->parent, nullptr);
}

TEST(CcuShutdown, NoServicesNeedsNoSdkOperation) {
    const std::vector<std::shared_ptr<Service>> services;
    bool called = false;
    EXPECT_TRUE(simpler::ccu::close_idle_services(services, [&](Service &) {
        called = true;
        return false;
    }));
    EXPECT_FALSE(called);
}

TEST(CcuShutdown, ClosedServiceDoesNotAuthorizeResetOfItsLiveParentDomain) {
    const auto service = std::make_shared<Service>();
    std::vector services{service};
    service->active_runs = 1;
    EXPECT_FALSE(simpler::ccu::close_services_before_device_teardown(services, false, close_service));
    EXPECT_EQ(service->close_calls, 0U);
    service->active_runs = 0;
    service->close_confirmed = false;
    EXPECT_FALSE(simpler::ccu::close_services_before_device_teardown(services, false, close_service));
    service->close_confirmed = true;
    EXPECT_FALSE(simpler::ccu::close_services_before_device_teardown(services, false, close_service));
    EXPECT_TRUE(service->closed);
    EXPECT_FALSE(simpler::ccu::close_services_before_device_teardown(services, false, close_service));
    EXPECT_EQ(service->close_calls, 2U);
    // Domain cleanup alone cannot retire the parent communicator. Its own
    // successful destruction is the first reset-authorizing result.
    services.clear();
    EXPECT_FALSE(simpler::ccu::close_services_before_device_teardown(services, false, close_service));
    EXPECT_TRUE(simpler::ccu::close_services_before_device_teardown(services, true, close_service));
}
}  // namespace
