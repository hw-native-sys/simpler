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

#include <cerrno>
#include <cstdint>
#include <dlfcn.h>
#include <pthread.h>
#include <vector>

namespace {

class ReservedKeys {
public:
    ~ReservedKeys() { release_to(0); }
    bool acquire(size_t count) {
        while (keys_.size() < count) {
            pthread_key_t key;
            if (pthread_key_create(&key, nullptr) != 0) return false;
            keys_.push_back(key);
        }
        return true;
    }
    size_t exhaust() {
        pthread_key_t key;
        int result = 0;
        while ((result = pthread_key_create(&key, nullptr)) == 0)
            keys_.push_back(key);
        EXPECT_EQ(result, EAGAIN);
        return keys_.size();
    }
    void release_to(size_t count) {
        while (keys_.size() > count) {
            EXPECT_EQ(pthread_key_delete(keys_.back()), 0);
            keys_.pop_back();
        }
    }
    pthread_key_t first() const { return keys_.front(); }

private:
    std::vector<pthread_key_t> keys_;
};

size_t available_keys() {
    ReservedKeys probe;
    return probe.exhaust();
}

class WrapperModule {
public:
    WrapperModule() :
        handle_(dlopen(SIM_TLS_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL)) {}
    ~WrapperModule() {
        if (handle_ != nullptr) EXPECT_EQ(dlclose(handle_), 0);
    }
    bool valid() const { return handle_ != nullptr; }
    uint64_t run(uint64_t epoch, pthread_key_t protected_key) const {
        using Probe = uint64_t (*)(uint64_t, pthread_key_t);
        auto probe = reinterpret_cast<Probe>(dlsym(handle_, "run_sim_tls_wrapper_probe"));
        EXPECT_NE(probe, nullptr);
        return probe != nullptr ? probe(epoch, protected_key) : UINT64_MAX;
    }
    int initialize() const {
        using Initialize = int (*)();
        auto initialize = reinterpret_cast<Initialize>(dlsym(handle_, "initialize_aicore_sim_tls"));
        EXPECT_NE(initialize, nullptr);
        return initialize != nullptr ? initialize() : -1;
    }

private:
    void *handle_;
};

TEST(SimAicoreTls, LoadingWithoutCallingWrapperDoesNotAllocateKeys) {
    const size_t baseline = available_keys();
    {
        WrapperModule module;
        ASSERT_TRUE(module.valid()) << dlerror();
        EXPECT_EQ(available_keys(), baseline);
    }
    EXPECT_EQ(available_keys(), baseline);
}

TEST(SimAicoreTls, RealWrapperAllocatesOnceAndUnloadReturnsEveryKey) {
    ReservedKeys guard;
    ASSERT_TRUE(guard.acquire(1));
    const size_t baseline = available_keys();
    ASSERT_GT(baseline, 10u);
    {
        WrapperModule module;
        ASSERT_TRUE(module.valid()) << dlerror();
        EXPECT_EQ(module.run(101, guard.first()), UINT64_C(6) << 32);
        EXPECT_EQ(available_keys(), baseline - 10);
        EXPECT_EQ(module.run(UINT64_MAX, guard.first()), UINT64_C(6) << 32);
        EXPECT_EQ(available_keys(), baseline - 10);
    }
    EXPECT_EQ(available_keys(), baseline);
}

TEST(SimAicoreTls, MoreThanOneHundredRealWrapperReloadsKeepTheirIdentityAndKeyBudget) {
    ReservedKeys guard;
    ASSERT_TRUE(guard.acquire(1));
    const size_t baseline = available_keys();
    for (uint64_t epoch = 1; epoch <= 128; ++epoch) {
        SCOPED_TRACE(epoch);
        {
            WrapperModule module;
            ASSERT_TRUE(module.valid()) << dlerror();
            ASSERT_EQ(module.run(epoch, guard.first()), UINT64_C(6) << 32);
        }
        ASSERT_EQ(available_keys(), baseline);
    }
}

class SimAicoreTlsExhaustion : public ::testing::TestWithParam<size_t> {};

TEST_P(SimAicoreTlsExhaustion, PartialCreationRollsBackWithoutCallingExecutorOrUsingOtherKeys) {
    ReservedKeys guard;
    ASSERT_TRUE(guard.acquire(1));
    const size_t baseline = available_keys();
    ASSERT_GE(baseline, 10u);
    ReservedKeys held;
    ASSERT_TRUE(held.acquire(baseline - GetParam()));
    {
        WrapperModule module;
        ASSERT_TRUE(module.valid()) << dlerror();
        EXPECT_EQ(module.run(101, guard.first()), 0u);
        EXPECT_EQ(available_keys(), GetParam());
        EXPECT_EQ(module.initialize(), EAGAIN);
        held.release_to(0);
        EXPECT_EQ(module.initialize(), EAGAIN);
        EXPECT_EQ(module.run(102, guard.first()), 0u);
    }
    EXPECT_EQ(available_keys(), baseline);
    WrapperModule retry;
    ASSERT_TRUE(retry.valid()) << dlerror();
    EXPECT_EQ(retry.initialize(), 0);
    EXPECT_EQ(retry.run(103, guard.first()), UINT64_C(6) << 32);
}

TEST_P(SimAicoreTlsExhaustion, LoaderPreflightReportsExhaustionBeforeAnyWrapperStarts) {
    ReservedKeys guard;
    ASSERT_TRUE(guard.acquire(1));
    const size_t baseline = available_keys();
    ASSERT_GE(baseline, 10u);
    ReservedKeys held;
    ASSERT_TRUE(held.acquire(baseline - GetParam()));
    WrapperModule module;
    ASSERT_TRUE(module.valid()) << dlerror();
    EXPECT_EQ(module.initialize(), EAGAIN);
    EXPECT_EQ(available_keys(), GetParam());
    EXPECT_EQ(module.run(101, guard.first()), 0u);
}

INSTANTIATE_TEST_SUITE_P(EachPartialKeySet, SimAicoreTlsExhaustion, ::testing::Range<size_t>(0, 10));

}  // namespace
