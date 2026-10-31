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
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <string>
#include <thread>

#include "common/host_api.h"
#include "orchestration_api.h"
#include "host/host_phase_records.h"
#include "host_build_graph/host/host_phase_trace.h"

namespace {
std::atomic<unsigned> created_keys{0};
std::atomic<unsigned> set_calls{0};
OrchContext *bound_context = nullptr;
extern "C" OrchContext *framework_current_orch_context() { return bound_context; }
extern "C" void framework_bind_orch_context(OrchContext *ctx) { bound_context = ctx; }
std::atomic<bool> fail_key{false};
std::atomic<bool> fail_set{false};
std::atomic<int> fail_set_after{0};
std::mutex failure_mutex;
std::condition_variable failure_cv;
bool pause_failure = false;
bool failure_entered = false;
bool release_failure = false;

void record() { host_phase_record(1, 2, static_cast<uint32_t>(HostPhaseKind::OrchRecordSubTask), 1, 0); }

void check_export(simpler::dfx::HostPhaseRecordStore &store, uint64_t dropped, size_t stored) {
    EXPECT_EQ(store.dropped_records(), dropped);
    EXPECT_EQ(store.records().size(), stored);
    char path[] = "/tmp/simpler-host-phase-failure-XXXXXX";
    const int fd = mkstemp(path);
    ASSERT_GE(fd, 0);
    close(fd);
    unlink(path);
    EXPECT_EQ(store.write_records_jsonl(path), 0);
    std::ifstream input(path);
    EXPECT_TRUE(input.good());
    std::string line;
    std::getline(input, line);
    EXPECT_NE(line.find("\"dropped\": " + std::to_string(dropped) + ","), std::string::npos);
    if (stored == 0) EXPECT_NE(line.find("\"records\": []"), std::string::npos);
    EXPECT_FALSE(std::getline(input, line));
    unlink(path);
}

void failure_export(bool key_creation, bool existing_record, bool reopen = false) {
    setenv("SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE", "1", 1);
    simpler::dfx::HostPhaseRecordStore store;
    HostApiOps ops{};
    ops.host_phase_pool_arm = [](void *ctx, uint32_t, int enabled) -> void * {
        return static_cast<simpler::dfx::HostPhaseRecordStore *>(ctx)->arm(enabled);
    };
    ops.host_phase_pool_finish = [](void *ctx, uint32_t, uint64_t tasks, uint64_t inv) {
        static_cast<simpler::dfx::HostPhaseRecordStore *>(ctx)->finish(tasks, inv);
    };
    HostApi api(&store, 0, 0, 1, &ops);
    host_phase_trace_begin(&api);
    if (existing_record) record();
    fail_key.store(key_creation);
    fail_set.store(!key_creation);
    pause_failure = reopen;
    // A fresh producer must bind its lane even if the main thread already has one.
    std::thread producer([reopen] {
        record();
        if (reopen) return;
        host_phase_record_bind(static_cast<uint32_t>(HostPhaseKind::BindArgs), 3, nullptr, 0, 4);
    });
    if (reopen) {
        {
            std::unique_lock lock(failure_mutex);
            failure_cv.wait(lock, [] {
                return failure_entered;
            });
        }
        host_phase_trace_end(&api);
        check_export(store, 0, 1);
        host_phase_trace_begin(&api);
        {
            std::lock_guard lock(failure_mutex);
            release_failure = true;
        }
        failure_cv.notify_one();
    }
    producer.join();
    EXPECT_FALSE(fail_key.load());
    EXPECT_FALSE(fail_set.load());
    // A cache binding failure disables the module's cache, including other threads.
    record();
    host_phase_trace_end(&api);
    host_phase_trace_end(&api);
    check_export(store, reopen ? 1 : 3, existing_record && !reopen ? 1 : 0);

    // Closing twice must not merge twice; a new pass must start its own ledger.
    host_phase_trace_begin(&api);
    record();
    host_phase_trace_end(&api);
    check_export(store, 1, 0);
}

// Each subprocess starts with fresh function-local caches; failures deliberately
// disable an optional cache for the remaining lifetime of the module.
TEST(HostPhaseTraceFailureDeathTest, KeyCreationFailureExportsEmptyPassWithDrops) {
    ASSERT_EXIT(
        {
            failure_export(true, false);
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(HostPhaseTraceFailureDeathTest, FirstBindingFailureExportsEmptyPassWithDrops) {
    ASSERT_EXIT(
        {
            failure_export(false, false);
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(HostPhaseTraceFailureDeathTest, LaterBindingFailurePreservesRecordsAndExportsDrops) {
    ASSERT_EXIT(
        {
            failure_export(false, true);
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(HostPhaseTraceFailureDeathTest, FailureReturningAfterReopenCannotPolluteNewPass) {
    ASSERT_EXIT(
        {
            failure_export(false, true, true);
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
void check_orchestration_budget() {
    OrchOps ops{};
    ops.is_fatal = [](OrchContext *) {
        return false;
    };
    ops.graph_begin = [](OrchContext *, uint64_t, const GraphTaskArgs &) {
        return GraphScopeResult{};
    };
    ops.orchestration_done = [](OrchContext *) {};
    OrchContext ctx{&ops};
    framework_bind_orch_context(&ctx);
    const auto before = created_keys.load();
    const auto before_set = set_calls.load();
    GraphTaskArgs args;
    for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(rt_submit_graph_impl(1, args, [](const GraphTaskArgs &) {}).has_error);
        rt_orchestration_done();
    }
    EXPECT_EQ(created_keys.load(), before);
    EXPECT_EQ(set_calls.load(), before_set);
    ops.record_orch_phase = [](uint32_t, uint64_t, uint64_t, uint64_t) {};
    EXPECT_FALSE(rt_submit_graph_impl(1, args, [](const GraphTaskArgs &) {}).has_error);
    EXPECT_EQ(created_keys.load(), before + 2);
    EXPECT_EQ(rt_submit_phase_state().get().count, 1u);
    rt_orchestration_done();
    EXPECT_EQ(rt_submit_phase_state().get().count, 0u);
    EXPECT_EQ(rt_submit_phase_state().get().prev_exit_ns, 0u);
}

TEST(HostPhaseTraceFailureDeathTest, DisabledOrchestrationUsesNoTimingKeysOrWrites) {
    ASSERT_EXIT(
        {
            check_orchestration_budget();
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

void check_pool_cache_writes() {
    simpler::dfx::HostPhaseRecordStore store;
    auto *pool = store.arm(true);
    const auto before = created_keys.load();
    const auto before_set = set_calls.load();
    const auto count = PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER + 3;
    for (int i = 0; i < count; ++i)
        host_phase_pool_append(pool, HostPhaseKind::OrchRecordSubTask, i, i + 1, i, i);
    store.finish(count, 1);
    EXPECT_EQ(created_keys.load(), before + 2);
    EXPECT_EQ(set_calls.load(), before_set + 4);
    EXPECT_EQ(store.records().size(), static_cast<size_t>(count));
    EXPECT_EQ(store.dropped_records(), 0u);
    const auto records = store.records();
    for (size_t i = 0; i < records.size(); ++i)
        EXPECT_EQ(records[i].payload, i);
}

TEST(HostPhaseTraceFailureDeathTest, PoolUpdatesKeysOnlyWhenClaimingBuffer) {
    ASSERT_EXIT(
        {
            check_pool_cache_writes();
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
void check_pool_binding_failure() {
    simpler::dfx::HostPhaseRecordStore store;
    auto *pool = store.arm(true);
    fail_set_after = 2;
    host_phase_pool_append(pool, HostPhaseKind::OrchRecordSubTask, 1, 2, 17, 0);
    EXPECT_EQ(fail_set_after.load(), 0);
    host_phase_pool_append(pool, HostPhaseKind::OrchRecordSubTask, 2, 3, 18, 0);
    store.finish(0, 1);
    ASSERT_EQ(store.records().size(), 1u);
    EXPECT_EQ(store.records()[0].payload, 17u);
    EXPECT_EQ(store.dropped_records(), 1u);
}

TEST(HostPhaseTraceFailureDeathTest, PartialPoolBindingKeepsStoredEventAndCountsLaterDrop) {
    ASSERT_EXIT(
        {
            check_pool_binding_failure();
            std::exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
}  // namespace

extern "C" int __real_pthread_key_create(pthread_key_t *, void (*)(void *));
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    if (fail_key.exchange(false)) return EAGAIN;
    const int rc = __real_pthread_key_create(key, destructor);
    if (rc == 0) ++created_keys;
    return rc;
}
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    ++set_calls;
    if (fail_set_after.load() > 0 && fail_set_after.fetch_sub(1) == 1) return ENOMEM;
    if (fail_set.exchange(false)) {
        std::unique_lock lock(failure_mutex);
        if (pause_failure) {
            failure_entered = true;
            failure_cv.notify_one();
            failure_cv.wait(lock, [] {
                return release_failure;
            });
        }
        return ENOMEM;
    }
    return __real_pthread_setspecific(key, value);
}
