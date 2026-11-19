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

#include "platform_comm/ccu/ccu_domain_owner.h"

namespace {
using simpler::ccu::DomainCcuOwner;
using simpler::ccu::DomainResourceBackend;
using simpler::ccu::MissionRunResources;
using simpler::ccu::RunLifecycleSdk;
using simpler::ccu::StreamState;

class FakeBackend final : public DomainResourceBackend {
public:
    bool drained() const override { return is_drained; }
    bool close() override {
        ++*close_calls;
        return close_succeeds;
    }

    bool is_drained{true};
    bool close_succeeds{true};
    std::shared_ptr<int> close_calls{std::make_shared<int>(0)};
};

class FakeRunSdk final : public RunLifecycleSdk {
public:
    int launch(uint64_t, const uint64_t *, uint32_t, uint64_t) override { return 0; }
    int poll_ready_event(uint64_t, uint64_t, bool *ready) override {
        *ready = true;
        return 0;
    }
    int query_stream(uint64_t, StreamState *) override { return 0; }
    int abort_stream(uint64_t) override { return 0; }
};

MissionRunResources mission(uint64_t) {
    MissionRunResources value{};
    value.mission_id = 1;
    value.program_handle = 101;
    value.stream_handle = 11;
    value.launch_arguments = {0x1000, 0x2000};
    value.ready_event = {0x3000, SIMPLER_CCU_EVENT_MASK};
    return value;
}

TEST(CcuDomainOwner, ClosesOnlyAfterBackendAndLifecycleAreDrained) {
    auto backend = std::make_unique<FakeBackend>();
    FakeBackend *observed = backend.get();
    const auto close_calls = observed->close_calls;
    DomainCcuOwner owner(std::move(backend));
    owner.set_registered_kernels({{1, 7, 101}});
    FakeRunSdk sdk;
    auto run = owner.begin_run(8, {mission(8)}, sdk);

    EXPECT_FALSE(owner.drained());
    EXPECT_FALSE(owner.close());
    EXPECT_EQ(*close_calls, 0);
    run.reset();
    EXPECT_TRUE(owner.drained());
    EXPECT_TRUE(owner.close());
    EXPECT_TRUE(owner.closed());
    EXPECT_EQ(*close_calls, 1);
}

TEST(CcuDomainOwner, RetainsResourcesWhenBackendCloseCannotBeConfirmed) {
    auto backend = std::make_unique<FakeBackend>();
    FakeBackend *observed = backend.get();
    const auto close_calls = observed->close_calls;
    observed->close_succeeds = false;
    DomainCcuOwner owner(std::move(backend));
    owner.set_registered_kernels({{1, 7, 101}});

    EXPECT_FALSE(owner.close());
    EXPECT_FALSE(owner.closed());
    EXPECT_EQ(owner.registered_kernels().size(), 1U);
    observed->close_succeeds = true;
    EXPECT_TRUE(owner.close());
    EXPECT_TRUE(owner.registered_kernels().empty());
    EXPECT_EQ(*close_calls, 2);
}

TEST(CcuDomainOwner, UnconfirmedRunPermanentlyBlocksDomainReclamation) {
    auto backend = std::make_unique<FakeBackend>();
    FakeBackend *observed = backend.get();
    const auto close_calls = observed->close_calls;
    DomainCcuOwner owner(std::move(backend));
    owner.set_registered_kernels({{1, 7, 101}});
    FakeRunSdk sdk;
    auto run = owner.begin_run(9, {mission(9)}, sdk);
    run->launch_all();
    run.reset();

    EXPECT_FALSE(owner.lifecycle().admission_open());
    EXPECT_TRUE(owner.lifecycle().resources_retained());
    EXPECT_FALSE(owner.close());
    EXPECT_EQ(*close_calls, 0);
}

}  // namespace
