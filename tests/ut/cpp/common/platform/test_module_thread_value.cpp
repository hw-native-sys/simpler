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

#include <atomic>
#include <cerrno>
#include <thread>
#include <unistd.h>
#include <gtest/gtest.h>
#include "common/module_thread_value.h"
#include "host/diagnostic_credit_signal.h"

namespace {
int fail_create = 0;
int fail_set = 0;
bool persist_set_failure = false;
int live_keys = 0;
struct Pair {
    uintptr_t a;
    uintptr_t b;
};
}  // namespace
extern "C" int __real_pthread_key_create(pthread_key_t *, void (*)(void *));
extern "C" int __real_pthread_key_delete(pthread_key_t);
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    if (destructor != nullptr) return __real_pthread_key_create(key, destructor);
    if (fail_create != 0 && --fail_create == 0) return EAGAIN;
    int rc = __real_pthread_key_create(key, destructor);
    if (rc == 0) ++live_keys;
    return rc;
}
extern "C" int __wrap_pthread_key_delete(pthread_key_t key) {
    int rc = __real_pthread_key_delete(key);
    if (rc == 0) --live_keys;
    return rc;
}
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (persist_set_failure && fail_set == 0) return ENOMEM;
    if (fail_set != 0 && --fail_set == 0) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
TEST(ModuleThreadValue, IsolatedDefaultsAndValues) {
    simpler::ModuleThreadValue<int> value(-1);
    EXPECT_EQ(value.get(), -1);
    value.set(13);
    std::thread other([&] {
        EXPECT_EQ(value.get(), -1);
        value.set(7);
        EXPECT_EQ(value.get(), 7);
    });
    other.join();
    EXPECT_EQ(value.get(), 13);
}
TEST(ModuleThreadValue, PartialCreationRollsBackWithoutDeletingReusedKeys) {
    const int before = live_keys;
    for (int failed : {1, 2}) {
        pthread_key_t replacement;
        {
            fail_create = failed;
            simpler::ModuleThreadValue<Pair> value;
            EXPECT_EQ(value.error(), EAGAIN);
            EXPECT_EQ(live_keys, before);
            EXPECT_EQ(value.try_set({3, 4}), EAGAIN);
            ASSERT_EQ(__wrap_pthread_key_create(&replacement, nullptr), 0);
        }
        int marker;
        EXPECT_EQ(__wrap_pthread_setspecific(replacement, &marker), 0);
        EXPECT_EQ(pthread_getspecific(replacement), &marker);
        EXPECT_EQ(__wrap_pthread_key_delete(replacement), 0);
    }
    EXPECT_EQ(live_keys, before);
}
TEST(ModuleThreadValue, PartialBindingRestoresWholeValue) {
    simpler::ModuleThreadValue<Pair> value;
    value.set({11, 22});
    for (int failed : {1, 2}) {
        fail_set = failed;
        EXPECT_EQ(value.try_set({33, 44}), ENOMEM);
        EXPECT_EQ(value.get().a, 11u);
        EXPECT_EQ(value.get().b, 22u);
    }
    value.set({55, 66});
    EXPECT_EQ(value.get().b, 66u);
}
TEST(ModuleThreadValue, RetiresKeysWhileCallingThreadRemainsAlive) {
    const int before = live_keys;
    for (int round = 0; round < 100; ++round) {
        {
            simpler::ModuleThreadValue<Pair> value;
            value.set({1, 2});
        }
        EXPECT_EQ(live_keys, before);
    }
}
TEST(ModuleThreadValueDeathTest, FailedRestoreCannotLeaveBorrowedState) {
    ASSERT_DEATH(
        {
            simpler::ModuleThreadValue<void *> value;
            int marker;
            value.set(&marker);
            fail_set = 1;
            value.restore(nullptr);
        },
        "terminate"
    );
}

TEST(ModuleThreadValueDeathTest, FailedRollbackCannotExposePartialValue) {
    ASSERT_DEATH(
        {
            simpler::ModuleThreadValue<Pair> value;
            value.set({11, 22});
            fail_set = 2;
            persist_set_failure = true;
            value.try_set({33, 44});
        },
        "terminate"
    );
}

TEST(ModuleThreadValue, RequiredUpdateThrowsAndPreservesPreviousValue) {
    simpler::ModuleThreadValue<int> value(17);
    value.set(29);
    fail_set = 1;
    EXPECT_THROW(value.set(41), std::system_error);
    EXPECT_EQ(value.get(), 29);
}

TEST(ModuleThreadValue, FailedOptionalCacheNeverSelectsStaleState) {
    simpler::ModuleThreadValue<Pair> value;
    value.set({11, 22});
    fail_set = 2;
    EXPECT_FALSE(value.cache({33, 44}));
    EXPECT_FALSE(value.available());
    EXPECT_EQ(value.get().a, 0u);
    EXPECT_EQ(value.get().b, 0u);
    EXPECT_FALSE(value.cache({55, 66}));
}
TEST(ModuleThreadValue, SuppressionFailureDoesNotInterruptResourceRollback) {
    simpler::dfx::credit::DiagnosticCreditSignal signal;
    auto before = signal.generation();
    fail_set = 1;
    EXPECT_NO_THROW({
        simpler::dfx::credit::DiagnosticCreditSignal::SuppressSelfNotify suppression;
        signal.advance();
    });
    EXPECT_EQ(signal.generation(), before);
}

TEST(ModuleThreadValue, ExplicitInitialValueNeedsNoDefaultConstructor) {
    struct Value {
        Value() = delete;
        explicit Value(int number) :
            number(number) {}
        int number;
    };
    static_assert(std::is_trivially_copyable_v<Value>);
    simpler::ModuleThreadValue<Value> value(Value(17));
    EXPECT_EQ(value.get().number, 17);
    value.set(Value(29));
    EXPECT_EQ(value.get().number, 29);
}

TEST(ModuleThreadValueDeathTest, CreationFailureKeepsRollbackQuietAndOtherThreadsObservable) {
    ASSERT_EXIT(
        {
            using namespace simpler::dfx::credit;
            DiagnosticCreditSignal signal;
            fail_create = 1;
            for (int attempt = 0; attempt < 32; ++attempt) {
                auto before = signal.generation();
                {
                    DiagnosticCreditSignal::SuppressSelfNotify outer;
                    {
                        DiagnosticCreditSignal::SuppressSelfNotify inner;
                        signal.advance();
                    }
                    signal.advance();
                    if (signal.wait_for(before, 0) != CreditWait::TimedOut) _exit(1);
                    std::thread other([&] {
                        {
                            DiagnosticCreditSignal::SuppressSelfNotify quiet;
                            signal.advance();
                        }
                        signal.advance();
                    });
                    other.join();
                    if (signal.wait_for(before, 0) != CreditWait::Advanced || signal.generation() != before + 1)
                        _exit(2);
                }
                auto restored = signal.generation();
                signal.advance();
                if (signal.generation() == restored) _exit(3);
            }
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(ModuleThreadValue, PersistentBindingFailureDoesNotManufactureCredit) {
    using namespace simpler::dfx::credit;
    DiagnosticCreditSignal signal;
    // Initialize the key before failures affect its per-thread binding.
    EXPECT_FALSE(DiagnosticCreditSignal::SuppressSelfNotify::active());
    persist_set_failure = true;
    fail_set = 0;
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto before = signal.generation();
        {
            DiagnosticCreditSignal::SuppressSelfNotify outer;
            {
                DiagnosticCreditSignal::SuppressSelfNotify inner;
                signal.advance();
            }
            signal.advance();
            EXPECT_EQ(signal.wait_for(before, 0), CreditWait::TimedOut);
            std::thread other([&] {
                {
                    DiagnosticCreditSignal::SuppressSelfNotify quiet;
                    signal.advance();
                }
                signal.advance();
            });
            other.join();
            EXPECT_EQ(signal.wait_for(before, 0), CreditWait::Advanced);
            EXPECT_EQ(signal.generation(), before + 1);
        }
        auto restored = signal.generation();
        signal.advance();
        EXPECT_GT(signal.generation(), restored);
    }
    persist_set_failure = false;
}
