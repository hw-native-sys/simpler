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

#include "platform_comm/ccu/ccu_registry.h"

namespace {
using namespace simpler::ccu;

struct Backend {
    int builds = 0;
    int drains = 0;
    int closes = 0;
    int fail_at = 0;
    bool stopped = true;
    bool closed = true;
};

StopResult drain(void *handle) {
    auto &backend = *static_cast<Backend *>(handle);
    ++backend.drains;
    return backend.stopped ? StopResult::CONFIRMED : StopResult::UNCONFIRMED;
}

StopResult close_program(void *handle) {
    auto &backend = *static_cast<Backend *>(handle);
    ++backend.closes;
    return backend.closed ? StopResult::CONFIRMED : StopResult::UNCONFIRMED;
}

PollResult poll(void *handle) {
    return static_cast<Backend *>(handle)->stopped ? PollResult::COMPLETE : PollResult::PENDING;
}

int32_t build(const BuildContext *context, Program *program) {
    auto &backend = *static_cast<Backend *>(context->platform_context);
    ++backend.builds;
    *program = {ABI_VERSION, sizeof(Program), &backend, static_cast<uint64_t>(backend.builds),
                drain,       close_program,   poll};
    return backend.builds == backend.fail_at ? -1 : 0;
}

CallableIdentity identity(uint8_t id) {
    CallableIdentity result{};
    result[0] = id;
    return result;
}

Domain domain(const std::shared_ptr<Backend> &backend, uint64_t id = 1, uint64_t generation = 1) {
    return {{10, id, generation}, 0, {{0x1000, 0x100, 0}, {0x2000, 0x100, 1}}, backend, backend.get()};
}

std::vector<FunctionDefinition> functions(const std::shared_ptr<void> &module, size_t count = 1) {
    return std::vector<FunctionDefinition>(count, {module, build});
}

TEST(CcuRegistry, PreparationIsCachedPerCallableDomainAndGeneration) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    auto module = std::make_shared<int>(42);
    registry.register_callable(identity(1), functions(module, 2));
    registry.register_callable(identity(2), functions(module));
    const auto first_domain = domain(backend);
    auto first = registry.prepare(identity(1), first_domain);
    EXPECT_EQ(registry.prepare(identity(1), first_domain), first);
    EXPECT_EQ(backend->builds, 2);
    EXPECT_EQ(first->device_program(0), 1);
    EXPECT_EQ(first->device_program(1), 2);
    EXPECT_THROW(first->device_program(2), std::out_of_range);
    EXPECT_NE(registry.prepare(identity(2), first_domain), first);
    EXPECT_EQ(backend->builds, 3);
    EXPECT_TRUE(registry.close_domain(first_domain.identity));
    EXPECT_THROW(registry.prepare(identity(1), first_domain), std::runtime_error);
    auto replacement = registry.prepare(identity(1), domain(backend, 1, 2));
    EXPECT_EQ(replacement->device_program(0), 4);
    EXPECT_THROW(first->device_program(0), std::runtime_error);
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, ZeroAndMaximumDomainIdsRemainDistinctAcrossReleaseAndRecreation) {
    Registry registry;
    auto first_backend = std::make_shared<Backend>();
    auto second_backend = std::make_shared<Backend>();
    registry.register_callable(identity(1), functions(std::make_shared<int>(42)));
    const auto first_domain = domain(first_backend, 0);
    auto first = registry.prepare(identity(1), first_domain);
    auto second = registry.prepare(identity(1), domain(second_backend, UINT64_MAX));
    EXPECT_EQ(registry.prepare(identity(1), first_domain), first);
    EXPECT_NE(first, second);
    EXPECT_TRUE(registry.close_domain(first_domain.identity));
    EXPECT_EQ(first_backend->closes, 1);
    EXPECT_EQ(second_backend->closes, 0);
    EXPECT_EQ(second->device_program(0), 1);
    EXPECT_THROW(registry.prepare(identity(1), first_domain), std::runtime_error);
    auto replacement = registry.prepare(identity(1), domain(first_backend, 0, 2));
    EXPECT_NE(replacement, first);
    EXPECT_EQ(replacement->device_program(0), 2);
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, DomainReleaseDoesNotCloseOtherDomain) {
    Registry registry;
    auto first_backend = std::make_shared<Backend>();
    auto second_backend = std::make_shared<Backend>();
    registry.register_callable(identity(1), functions(std::make_shared<int>(42)));
    registry.prepare(identity(1), domain(first_backend, 1));
    auto second = registry.prepare(identity(1), domain(second_backend, 2));
    EXPECT_TRUE(registry.close_domain(domain(first_backend).identity));
    EXPECT_EQ(first_backend->closes, 1);
    EXPECT_EQ(second_backend->closes, 0);
    EXPECT_EQ(second->device_program(0), 1);
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, AcceptedRunHoldsModuleAndWindowUntilReleased) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    auto module = std::make_shared<int>(42);
    std::weak_ptr<int> weak_module = module;
    std::weak_ptr<Backend> weak_domain = backend;
    registry.register_callable(identity(1), functions(module));
    auto instance = registry.prepare(identity(1), domain(backend));
    instance->retain_run();
    module.reset();
    backend.reset();
    EXPECT_FALSE(registry.unregister_callable(identity(1)));
    EXPECT_FALSE(weak_module.expired());
    EXPECT_FALSE(weak_domain.expired());
    EXPECT_TRUE(instance->drain_run_completion(true));
    instance->release_run(true);
    EXPECT_TRUE(registry.unregister_callable(identity(1)));
    EXPECT_TRUE(weak_module.expired());
    EXPECT_TRUE(weak_domain.expired());
}

TEST(CcuRegistry, UnconfirmedDrainDisablesReuseAndRetainsResources) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    backend->stopped = false;
    auto module = std::make_shared<int>(42);
    std::weak_ptr<int> weak_module = module;
    registry.register_callable(identity(1), functions(module));
    registry.prepare(identity(1), domain(backend));
    module.reset();
    EXPECT_FALSE(registry.unregister_callable(identity(1)));
    EXPECT_FALSE(weak_module.expired());
    EXPECT_EQ(backend->closes, 0);
    EXPECT_THROW(registry.prepare(identity(1), domain(backend, 2)), std::runtime_error);
    backend->stopped = true;
    EXPECT_TRUE(registry.unregister_callable(identity(1)));
    EXPECT_TRUE(weak_module.expired());
}

TEST(CcuRegistry, FailedBuilderRollsBackEveryReturnedHandle) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    backend->fail_at = 2;
    registry.register_callable(identity(1), functions(std::make_shared<int>(42), 3));
    EXPECT_THROW(registry.prepare(identity(1), domain(backend)), std::runtime_error);
    EXPECT_EQ(backend->builds, 2);
    EXPECT_EQ(backend->closes, 2);
    backend->fail_at = 0;
    auto instance = registry.prepare(identity(1), domain(backend));
    EXPECT_EQ(instance->device_program(2), 5);
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, FailedRollbackCannotBeReused) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    backend->fail_at = 1;
    backend->closed = false;
    registry.register_callable(identity(1), functions(std::make_shared<int>(42)));
    EXPECT_THROW(registry.prepare(identity(1), domain(backend)), std::runtime_error);
    backend->fail_at = 0;
    EXPECT_THROW(registry.prepare(identity(1), domain(backend)), std::runtime_error);
    EXPECT_EQ(backend->builds, 1);
    backend->closed = true;
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, MutatedCredentialsAndInvalidRangesAreRejected) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    registry.register_callable(identity(1), functions(std::make_shared<int>(42)));
    const auto original = domain(backend);
    registry.prepare(identity(1), original);
    auto changed = original;
    changed.windows[0].credential = 3;
    EXPECT_THROW(registry.prepare(identity(1), changed), std::invalid_argument);
    changed = domain(backend, 2);
    changed.windows[0] = {UINT64_MAX - 2, 8, 1};
    EXPECT_THROW(registry.prepare(identity(1), changed), std::invalid_argument);
    EXPECT_EQ(backend->builds, 1);
    EXPECT_TRUE(registry.close());
}

TEST(CcuRegistry, FailedRunIsNotReusableAndCloseIsTerminal) {
    Registry registry;
    auto backend = std::make_shared<Backend>();
    registry.register_callable(identity(1), functions(std::make_shared<int>(42)));
    auto instance = registry.prepare(identity(1), domain(backend));
    instance->retain_run();
    EXPECT_TRUE(instance->drain_run_completion(true));
    instance->release_run(false);
    EXPECT_THROW(instance->retain_run(), std::runtime_error);
    EXPECT_THROW(registry.prepare(identity(1), domain(backend)), std::runtime_error);
    EXPECT_TRUE(registry.close());
    EXPECT_THROW(registry.register_callable(identity(2), functions(std::make_shared<int>(42))), std::runtime_error);
}
}  // namespace
