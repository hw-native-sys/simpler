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
#include <cerrno>
#include <atomic>
#include <dlfcn.h>
#include <thread>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include <type_traits>
#include "common/core_type.h"
#include "aicore/sim_core_abi.h"
#include "runtime.h"
#include "host/sim_thread_start_gate.h"
#include "aicore/aicore_profiling_state.h"

extern "C" int initialize_aicore_sim_tls();
// The production wrapper is linked against a stub executor.
// Only the wrapper and its inline worker-count accessor run in this fixture.
Runtime::Runtime() {}
static std::atomic<int> reject_binding_after{-1};
static std::atomic<int> executor_calls{0};
static std::atomic<int> pthread_writes{0};
struct UnbindProbe {
    int fail_position;
    std::vector<pthread_key_t> bound;
    std::vector<pthread_key_t> cleared;
};
// Keep injection local to the wrapper thread; its failed borrowed slot is
// explicitly cleared before that thread leaves the test.
static thread_local UnbindProbe *unbind_probe = nullptr;
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    ++pthread_writes;
    if (unbind_probe != nullptr) {
        if (value != nullptr) {
            unbind_probe->bound.push_back(key);
        } else {
            const int position = static_cast<int>(unbind_probe->cleared.size());
            unbind_probe->cleared.push_back(key);
            if (position == unbind_probe->fail_position) return ENOMEM;
        }
    }
    if (value != nullptr && reject_binding_after.fetch_sub(1) == 0) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
void aicore_execute(Runtime *, int, CoreType) {
    ++executor_calls;
    const int before = pthread_writes.load();
    set_aicore_profiling_flag(17);
    EXPECT_EQ(get_aicore_profiling_flag(), 17u);
    set_aicore_report_epoch(73);
    EXPECT_EQ(get_aicore_report_epoch(), 73u);
    EXPECT_EQ(pthread_writes.load(), before) << "running setters must not write pthread TLS";
}
#if TEST_TLS_KEY_COUNT == 6
extern "C" std::remove_pointer_t<SimA2a3CoreEntryFn> aicore_execute_wrapper_checked;
#else
extern "C" std::remove_pointer_t<SimA5CoreEntryFn> aicore_execute_wrapper_checked;
#endif
static int invoke_wrapper(Runtime &runtime, bool (*ready)(void *, int), void *context) {
#if TEST_TLS_KEY_COUNT == 6
    return aicore_execute_wrapper_checked(&runtime, 0, CoreType::AIC, 0, 0, 0, 0, 0, ready, context);
#else
    return aicore_execute_wrapper_checked(&runtime, 0, CoreType::AIC, 0, 0, 0, 0, 0, 0, 0, ready, context);
#endif
}
TEST(AicoreTlsBinding, FailedBindingNeverEntersExecutor) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    for (int position = 0; position < TEST_TLS_KEY_COUNT; ++position) {
        SCOPED_TRACE(position);
        executor_calls = 0;
        reject_binding_after = position;
        int observed = 0;
        auto ready = [](void *context, int error) {
            *static_cast<int *>(context) = error;
            return error == 0;
        };
        EXPECT_EQ(invoke_wrapper(runtime, ready, &observed), ENOMEM);
        EXPECT_EQ(observed, ENOMEM);
        EXPECT_EQ(executor_calls.load(), 0);
    }
    reject_binding_after = -1;
}
TEST(AicoreTlsBinding, OneFailedCoreCancelsAllExecutorsBeforeAdmission) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    SimThreadStartGate startup;
    executor_calls = 0;
    reject_binding_after = 0;
    std::thread first([&] {
        invoke_wrapper(runtime, SimThreadStartGate::report_ready_and_wait, &startup);
    });
    std::thread second([&] {
        invoke_wrapper(runtime, SimThreadStartGate::report_ready_and_wait, &startup);
    });
    const int error = startup.wait_for_arrivals_or_error(2);
    startup.cancel();
    first.join();
    second.join();
    EXPECT_EQ(error, ENOMEM);
    EXPECT_EQ(executor_calls.load(), 0);
    reject_binding_after = -1;
}
TEST(AicoreTlsBinding, SuccessfulBindingWaitsForAdmissionAndSettersCannotFail) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    SimThreadStartGate startup;
    executor_calls = 0;
    reject_binding_after = -1;
    int rc = -1;
    std::thread core([&] {
        rc = invoke_wrapper(runtime, SimThreadStartGate::report_ready_and_wait, &startup);
    });
    const int error = startup.wait_for_arrivals_or_error(1);
    EXPECT_EQ(executor_calls.load(), 0);
    // Reject any new non-null pthread write after the gate. Setters use cells.
    reject_binding_after = 0;
    startup.release();
    core.join();
    EXPECT_EQ(error, 0);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(executor_calls.load(), 1);
    reject_binding_after = -1;
}
TEST(AicoreTlsBinding, UnbindFailureIsReturnedAndRemainingKeysAreCleared) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    reject_binding_after = -1;
    for (bool admitted : {false, true}) {
        SCOPED_TRACE(admitted);
        for (int position = 0; position < TEST_TLS_KEY_COUNT; ++position) {
            SCOPED_TRACE(position);
            executor_calls = 0;
            std::thread core([&] {
                UnbindProbe probe{position, {}, {}};
                unbind_probe = &probe;
                const int rc = invoke_wrapper(
                    runtime,
                    [](void *context, int error) {
                        EXPECT_EQ(error, 0);
                        EXPECT_EQ(unbind_probe->bound.size(), TEST_TLS_KEY_COUNT);
                        EXPECT_TRUE(unbind_probe->cleared.empty());
                        return *static_cast<bool *>(context);
                    },
                    &admitted
                );
                unbind_probe = nullptr;
                EXPECT_EQ(rc, ENOMEM);
                EXPECT_EQ(probe.cleared.size(), TEST_TLS_KEY_COUNT);
                for (size_t i = 0; i < probe.bound.size(); ++i) {
                    const pthread_key_t key = probe.bound[probe.bound.size() - 1 - i];
                    if (i < probe.cleared.size()) EXPECT_EQ(probe.cleared[i], key);
                    if (i == static_cast<size_t>(position)) {
                        // Inspect only the pointer, never the expired stack cell.
                        EXPECT_NE(pthread_getspecific(key), nullptr);
                    } else {
                        EXPECT_EQ(pthread_getspecific(key), nullptr);
                    }
                    // A failed clear retains its pointer. Remove every fixture
                    // binding even if an assertion found skipped cleanup.
                    EXPECT_EQ(__real_pthread_setspecific(key, nullptr), 0);
                }
            });
            core.join();
            EXPECT_EQ(executor_calls.load(), admitted ? 1 : 0);
        }
    }
}

TEST(AicoreTlsBinding, CancelDoesNotWaitForThreadsThatWereNeverCreated) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    SimThreadStartGate startup;
    executor_calls = 0;
    std::thread core([&] {
        invoke_wrapper(runtime, SimThreadStartGate::report_ready_and_wait, &startup);
    });
    startup.cancel();
    core.join();
    EXPECT_EQ(executor_calls.load(), 0);
}

namespace {
std::vector<pthread_key_t> reserve_keys() {
    std::vector<pthread_key_t> keys;
    pthread_key_t key;
    while (pthread_key_create(&key, nullptr) == 0)
        keys.push_back(key);
    return keys;
}
size_t available_keys() {
    auto keys = reserve_keys();
    for (auto key : keys)
        pthread_key_delete(key);
    return keys.size();
}
int partial_failure(size_t free_slots) {
    auto held = reserve_keys();
    if (held.size() < free_slots) return 1;
    for (size_t i = 0; i < free_slots; ++i) {
        pthread_key_delete(held.back());
        held.pop_back();
    }
    const int error = initialize_aicore_sim_tls();
    if (error != (free_slots < TEST_TLS_KEY_COUNT ? EAGAIN : 0)) return 4;
    if (error != 0 && (error != EAGAIN || available_keys() != free_slots)) return 2;
    const size_t after = available_keys();
    // A completed pthread_once must preserve both the failure and key budget.
    if (initialize_aicore_sim_tls() != error || available_keys() != after) return 3;
    for (auto key : held)
        pthread_key_delete(key);
    return 0;
}
}  // namespace

TEST(AicoreTlsKeysDeathTest, PartialCreationRollsBackBeforeModuleUnload) {
    // Covers every failure position in both the six-key and ten-key owners.
    for (size_t free_slots = 0; free_slots < TEST_TLS_KEY_COUNT; ++free_slots) {
        SCOPED_TRACE(free_slots);
        ASSERT_EXIT(_exit(partial_failure(free_slots)), ::testing::ExitedWithCode(0), "");
    }
}

TEST(AicoreTlsKeysDeathTest, RollbackThenUnloadPreservesReusedKeys) {
    ASSERT_EXIT(
        {
            void *module = dlopen(TEST_TLS_MODULE_PATH, RTLD_NOW | RTLD_LOCAL);
            if (!module) _exit(1);
            auto initialize = reinterpret_cast<int (*)()>(dlsym(module, "initialize_aicore_sim_tls"));
            if (!initialize) _exit(2);
            auto held = reserve_keys();
            if (held.size() < 3) _exit(3);
            for (int i = 0; i < 3; ++i) {
                pthread_key_delete(held.back());
                held.pop_back();
            }
            if (initialize() != EAGAIN) _exit(4);
            auto reused = reserve_keys();
            if (reused.size() != 3) _exit(5);
            int sentinel = 17;
            for (auto key : reused)
                if (pthread_setspecific(key, &sentinel) != 0) _exit(6);
            if (dlclose(module) != 0) _exit(7);
            Dl_info info{};
            if (dladdr(reinterpret_cast<void *>(initialize), &info) != 0) _exit(8);
            for (auto key : reused) {
                if (pthread_getspecific(key) != &sentinel || pthread_setspecific(key, nullptr) != 0) _exit(9);
                pthread_key_delete(key);
            }
            for (auto key : held)
                pthread_key_delete(key);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicoreTlsKeysDeathTest, RepeatedModuleUnloadRestoresKeyBudget) {
    ASSERT_EXIT(
        {
            const size_t before = available_keys();
            for (int round = 0; round < 180; ++round) {
                void *module = dlopen(TEST_TLS_MODULE_PATH, RTLD_NOW | RTLD_LOCAL);
                if (!module) _exit(1);
                auto initialize = reinterpret_cast<int (*)()>(dlsym(module, "initialize_aicore_sim_tls"));
                if (!initialize || initialize() != 0) _exit(2);
                if (dlclose(module) != 0) _exit(3);
                Dl_info info{};
                if (dladdr(reinterpret_cast<void *>(initialize), &info) != 0) _exit(4);
                if (available_keys() != before) _exit(5);
            }
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicoreTlsKeysDeathTest, ConcurrentInitializationCreatesOnlyOneKeySet) {
    ASSERT_EXIT(
        {
            const size_t before = available_keys();
            std::vector<int> errors(8);
            std::vector<std::thread> threads;
            for (size_t i = 0; i < errors.size(); ++i)
                threads.emplace_back([&, i] {
                    errors[i] = initialize_aicore_sim_tls();
                });
            for (auto &thread : threads)
                thread.join();
            for (int error : errors)
                if (error != 0) _exit(1);
            const size_t after = available_keys();
            if (before - after != TEST_TLS_KEY_COUNT) _exit(2);
            if (initialize_aicore_sim_tls() != 0 || available_keys() != after) _exit(3);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicoreTlsBinding, IdentityBindingFailureIsReportedBeforeExecutorAdmission) {
    ASSERT_EQ(initialize_aicore_sim_tls(), 0);
    Runtime runtime;
    runtime.set_worker_count(3);
    auto fail_identity = +[](uint32_t) {
        return ENOMEM;
    };
    sim_install_core_identity_setters(fail_identity, nullptr);
    executor_calls = 0;
    int observed = 0;
    const int rc = invoke_wrapper(
        runtime,
        [](void *context, int error) {
            *static_cast<int *>(context) = error;
            return error == 0;
        },
        &observed
    );
    sim_install_core_identity_setters(nullptr, nullptr);
    EXPECT_EQ(rc, ENOMEM);
    EXPECT_EQ(observed, ENOMEM);
    EXPECT_EQ(executor_calls.load(), 0);
}
