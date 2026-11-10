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
#include <vector>
#include <unistd.h>
#include <sched.h>
#include <gtest/gtest.h>
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/affinity_thread_state.h"
#include "aicpu/native_invocation_entry.h"
#include "aicpu/device_run_result_aicpu.h"
#include "common/native_invocation_evidence.h"
#include <pthread.h>
namespace {
std::atomic<bool> fail_binding{false};
std::atomic<bool> fail_creation{false};
std::atomic<bool> fail_nonnull_binding{false};
// Resident-pool regression: a participant publishes its gate claim through the
// null rebind that follows the claim, and the round owner releases the next
// participant only after observing that event.
constexpr int kResidentRounds = 3;
constexpr int kResidentPool = 4;
thread_local int test_round = -1;
std::atomic<bool> claim_seen[kResidentRounds]{};
}  // namespace
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __real_pthread_key_create(pthread_key_t *, void (*)(void *));
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (fail_binding.exchange(false) || (value != nullptr && fail_nonnull_binding.exchange(false))) return ENOMEM;
    const int error = __real_pthread_setspecific(key, value);
    if (error == 0 && value == nullptr && test_round >= 0)
        claim_seen[test_round].store(true, std::memory_order_release);
    return error;
}
extern "C" int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    if (fail_creation.exchange(false)) return EAGAIN;
    return __real_pthread_key_create(key, destructor);
}
TEST(AicpuAffinityTls, BoundCellKeepsThreadsIndependent) {
    int32_t index = -1;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&index), 0);
    platform_aicpu_affinity_set_thread_idx(3);
    std::thread other([] {
        EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
        int32_t local = 7;
        ASSERT_EQ(platform_aicpu_affinity_bind_thread(&local), 0);
        EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 7);
        platform_aicpu_affinity_set_thread_idx(8);
        EXPECT_EQ(local, 8);
        platform_aicpu_affinity_bind_thread(nullptr);
    });
    other.join();
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 3);
    EXPECT_EQ(index, 3);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}
TEST(AicpuAffinityTlsDeathTest, BindingFailureRejectsWholeCohortWithoutHanging) {
    ASSERT_EXIT(
        {
            fail_binding.store(true);
            int32_t allowed[2];
            allowed[0] = sched_getcpu();
            allowed[1] = allowed[0];
            std::atomic<int> admitted{0};
            int errors[2]{};
            std::thread a([&] {
                admitted += platform_aicpu_affinity_gate_filter(allowed, 2, 2, nullptr, &errors[0]);
                platform_aicpu_affinity_bind_thread(nullptr);
            });
            std::thread b([&] {
                admitted += platform_aicpu_affinity_gate_filter(allowed, 2, 2, nullptr, &errors[1]);
                platform_aicpu_affinity_bind_thread(nullptr);
            });
            a.join();
            b.join();
            _exit(admitted == 0 && errors[0] == ENOMEM && errors[1] == ENOMEM ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(AicpuAffinityTlsDeathTest, KeyFailureIsReportedWithoutInvalidKeyUse) {
    ASSERT_EXIT(
        {
            fail_creation.store(true);
            int32_t cell = -1;
            _exit(platform_aicpu_affinity_bind_thread(&cell) == EAGAIN ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(AicpuAffinityTls, RepeatedCohortsKeepDistinctSlots) {
    const int32_t allowed[] = {sched_getcpu(), sched_getcpu()};
    for (int round = 0; round < 20; ++round) {
        std::atomic<int> mask{0};
        std::thread a([&] {
            if (platform_aicpu_affinity_gate_filter(allowed, 2, 2))
                mask.fetch_or(1 << platform_aicpu_affinity_thread_idx());
            platform_aicpu_affinity_bind_thread(nullptr);
        });
        std::thread b([&] {
            if (platform_aicpu_affinity_gate_filter(allowed, 2, 2))
                mask.fetch_or(1 << platform_aicpu_affinity_thread_idx());
            platform_aicpu_affinity_bind_thread(nullptr);
        });
        a.join();
        b.join();
        EXPECT_EQ(mask.load(), 3);
    }
}

TEST(AicpuAffinityTls, ResidentThreadsKeepOwnedSlotsAcrossRotatingCohorts) {
    // Unmatched cpu ids keep the onboard classifier on its report-order fill
    // path, so both backends assign slots in claim order.
    const int32_t allowed[] = {-1, -1};
    // Thread 0 owns slot 0 in round 0 and thread 2 owns slot 0 in round 1; both
    // re-enter together in round 2 holding a stale binding to that slot.
    constexpr int members[kResidentRounds][2] = {{0, 1}, {2, 3}, {0, 2}};
    struct Result {
        bool accepted{false};
        int32_t claim{-1};
        int32_t identity{-1};
    };
    Result results[kResidentRounds][kResidentPool];
    std::atomic<bool> go[kResidentRounds][2]{};
    std::atomic<int> completed{0};
    for (int round = 0; round < kResidentRounds; ++round)
        claim_seen[round].store(false);
    auto worker = [&](int id) {
        for (int round = 0; round < kResidentRounds; ++round) {
            const int seat = id == members[round][0] ? 0 : (id == members[round][1] ? 1 : -1);
            if (seat < 0) continue;
            while (!go[round][seat].load(std::memory_order_acquire)) {}
            test_round = round;
            const auto entry = platform_aicpu_entry_gate_filter(
                allowed, 2, 2,
                {[](void *) -> int32_t {
                     return 0;
                 },
                 nullptr}
            );
            results[round][id] = {entry.execute, entry.callback_index, platform_aicpu_affinity_thread_idx()};
            test_round = -1;
            completed.fetch_add(1, std::memory_order_release);
        }
        platform_aicpu_affinity_bind_thread(nullptr);
    };
    std::thread workers[kResidentPool];
    for (int id = 0; id < kResidentPool; ++id)
        workers[id] = std::thread(worker, id);
    for (int round = 0; round < kResidentRounds; ++round) {
        go[round][0].store(true, std::memory_order_release);
        if (round + 1 < kResidentRounds) {
            // The held participant claims before the second participant enters.
            while (!claim_seen[round].load(std::memory_order_acquire)) {}
        }
        go[round][1].store(true, std::memory_order_release);
        while (completed.load(std::memory_order_acquire) < 2 * (round + 1)) {}
    }
    for (auto &worker_thread : workers)
        worker_thread.join();
    for (int round = 0; round < kResidentRounds; ++round) {
        for (int seat = 0; seat < 2; ++seat) {
            const int id = members[round][seat];
            ASSERT_TRUE(results[round][id].accepted);
            EXPECT_EQ(results[round][id].identity, results[round][id].claim);
            EXPECT_GE(results[round][id].identity, 0);
            if (round + 1 < kResidentRounds) EXPECT_EQ(results[round][id].claim, seat);
        }
        EXPECT_NE(results[round][members[round][0]].identity, results[round][members[round][1]].identity);
    }
}

TEST(AicpuAffinityTls, RejectionInvalidatesBindingWithoutWritingTheBoundCell) {
    const int32_t allowed[] = {sched_getcpu()};
    int32_t cell = 7;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&cell), 0);
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 7);
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(allowed, 0, 1));
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
    EXPECT_EQ(cell, 7);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}

TEST(AicpuAffinityTlsDeathTest, KeyCreationFailureCanRetryInSameModule) {
    ASSERT_EXIT(
        {
            int32_t cell = 7;
            fail_creation = true;
            if (platform_aicpu_affinity_bind_thread(&cell) != EAGAIN) _exit(1);
            if (platform_aicpu_affinity_bind_thread(&cell) != 0) _exit(2);
            if (platform_aicpu_affinity_thread_idx() != 7) _exit(3);
            _exit(platform_aicpu_affinity_bind_thread(nullptr) == 0 ? 0 : 4);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicpuAffinityTlsDeathTest, FailedCohortDoesNotPoisonNextLaunch) {
    ASSERT_EXIT(
        {
            const int32_t allowed[] = {sched_getcpu()};
            fail_binding = true;
            if (platform_aicpu_affinity_gate_filter(allowed, 1, 1)) _exit(1);
            if (!platform_aicpu_affinity_gate_filter(allowed, 1, 1)) _exit(2);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicpuAffinityTlsDeathTest, UnboundSetterExplainsFatalContractViolation) {
    ASSERT_DEATH(platform_aicpu_affinity_set_thread_idx(3), "requires a bound cell");
}

TEST(AicpuAffinityTls, NativeFailurePreservesCallbackIdentityAndNextLaunchRecovers) {
    const int32_t allowed[] = {sched_getcpu(), sched_getcpu()};
    for (int round = 0; round < 3; ++round) {
        std::atomic<int> initializations{0};
        AicpuEntryGateResult results[2];
        int32_t indices[2]{};
        fail_binding = round == 0;
        fail_nonnull_binding = round == 1;
        auto enter = [&](int i) {
            results[i] = platform_aicpu_entry_gate_filter(
                allowed, 2, 2,
                {[](void *context) -> int32_t {
                     ++*static_cast<std::atomic<int> *>(context);
                     return 0;
                 },
                 &initializations}
            );
            indices[i] = platform_aicpu_affinity_thread_idx();
            platform_aicpu_affinity_bind_thread(nullptr);
        };
        std::thread a(enter, 0), b(enter, 1);
        a.join();
        b.join();
        EXPECT_EQ(initializations, 1);
        for (int32_t idx : indices) {
            if (round < 2) EXPECT_EQ(idx, -1);
            else EXPECT_GE(idx, 0);
        }
        EXPECT_NE(results[0].callback_index, results[1].callback_index);
        for (const auto &result : results) {
            EXPECT_GE(result.callback_index, 0);
            EXPECT_LT(result.callback_index, 2);
            EXPECT_EQ(result.status, round < 2 ? -1 : 0);
            EXPECT_EQ(result.execute, round == 2);
        }
    }
}

TEST(AicpuAffinityTls, ParameterRejectionInvalidatesPreviousIdentity) {
    const int32_t allowed[] = {sched_getcpu()};
    ASSERT_TRUE(platform_aicpu_affinity_gate_filter(allowed, 1, 1));
    ASSERT_EQ(platform_aicpu_affinity_thread_idx(), 0);
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(allowed, 0, 1));
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
    ASSERT_TRUE(platform_aicpu_affinity_gate_filter(allowed, 1, 1));
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(allowed, 1, 0));
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}

TEST(AicpuAffinityTls, ExcessParticipantInvalidatesPreviousIdentity) {
    const int32_t allowed[] = {sched_getcpu()};
    bool checked = false;
    auto result = platform_aicpu_entry_gate_filter(
        allowed, 1, 1,
        {[](void *context) -> int32_t {
             int32_t previous = 7;
             if (platform_aicpu_affinity_bind_thread(&previous) != 0) return -1;
             const int32_t cpus[] = {sched_getcpu()};
             int error = 0;
             const bool accepted = platform_aicpu_affinity_gate_filter(cpus, 1, 1, nullptr, &error);
             *static_cast<bool *>(context) = !accepted && error == EINVAL && platform_aicpu_affinity_thread_idx() == -1;
             // The temporary cell must not outlive this initializer.
             return platform_aicpu_affinity_bind_thread(nullptr);
         },
         &checked}
    );
    EXPECT_TRUE(checked);
    EXPECT_TRUE(result.execute);
    EXPECT_EQ(result.status, 0);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}

class AffinityInvocationRelease : public ::testing::TestWithParam<int32_t> {};

TEST_P(AffinityInvocationRelease, RotatingCallbacksReturnRecordsBeforePublishingCompletion) {
    const int32_t expected_status = GetParam();
    for (int round = 0; round < 4 * MAX_GATE_THREADS; ++round) {
        std::thread worker([&] {
            NativeInvocationResultStorage storage;
            NativeInvocationEntry invocation{1,     reinterpret_cast<uint64_t>(&storage),
                                             7,     aicpu_begin_native_invocation,
                                             [] {}, aicpu_finish_native_invocation};
            struct Context {
                NativeInvocationEntry *invocation;
                bool reject;
            } context{&invocation, expected_status == -51};
            const int32_t allowed[] = {sched_getcpu()};
            const auto entry = platform_aicpu_entry_gate_filter(
                allowed, 1, 1,
                {[](void *opaque) -> int32_t {
                     auto &ctx = *static_cast<Context *>(opaque);
                     const int32_t rc = NativeInvocationEntry::initialize(ctx.invocation);
                     return rc != 0 ? rc : (ctx.reject ? -51 : 0);
                 },
                 &context}
            );
            EXPECT_EQ(entry.status, context.reject ? -51 : 0) << round;
            EXPECT_EQ(entry.execute, !context.reject) << round;
            bool work_finished = false;
            bool rejection_finished = false;
            const int32_t rc = invocation.complete(
                entry,
                [&] {
                    struct Cleanup {
                        bool &finished;
                        ~Cleanup() {
                            EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 0);
                            finished = true;
                        }
                    } cleanup{work_finished};
                    const RunTerminalSelection terminal =
                        expected_status == 0 ?
                            RunTerminalSelection{DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None} :
                            RunTerminalSelection{DeviceRunVerdict::Error, expected_status, DeviceRunCodeSource::Header};
                    EXPECT_TRUE(aicpu_accept_runtime_terminal(terminal, nullptr, 0));
                    return expected_status;
                },
                [&] {
                    rejection_finished = true;
                },
                [&] {
                    EXPECT_EQ(storage.result.published, 0u);
                    EXPECT_TRUE(context.reject ? rejection_finished : work_finished);
                    return platform_aicpu_affinity_bind_thread(nullptr);
                }
            );
            EXPECT_EQ(rc, expected_status) << round;
            EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1) << round;
            EXPECT_EQ(storage.result.published, 7u) << round;
            EXPECT_EQ(device_run_result_terminal(storage.result, 7).code, expected_status) << round;
        });
        worker.join();
    }
}

INSTANTIATE_TEST_SUITE_P(SuccessErrorAndRejection, AffinityInvocationRelease, ::testing::Values(0, -73, -51));

TEST(AicpuAffinityTls, ReleaseFailureCannotPublishSuccessOrReplaceWorkError) {
    for (const int32_t work_status : {0, -73}) {
        NativeInvocationResultStorage storage;
        NativeInvocationEntry invocation{1,     reinterpret_cast<uint64_t>(&storage), 9, aicpu_begin_native_invocation,
                                         [] {}, aicpu_finish_native_invocation};
        const int32_t allowed[] = {sched_getcpu()};
        const auto entry = platform_aicpu_entry_gate_filter(allowed, 1, 1, invocation.initializer());
        ASSERT_TRUE(entry.execute);
        const int rc = invocation.complete(
            entry,
            [&] {
                EXPECT_TRUE(
                    aicpu_accept_runtime_terminal({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0)
                );
                fail_binding.store(true);
                return work_status;
            },
            [] {},
            [] {
                return platform_aicpu_affinity_bind_thread(nullptr);
            }
        );
        EXPECT_FALSE(fail_binding.load());
        EXPECT_EQ(rc, work_status == 0 ? -1 : work_status);
        EXPECT_EQ(device_run_result_terminal(storage.result, 9).code, rc);
        EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
        EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
    }
}

TEST(AicpuAffinityTls, InvalidCallbackStillReleasesItsRecordWithoutFinishing) {
    int32_t cell = 7;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&cell), 0);
    NativeInvocationEntry invocation{};
    EXPECT_EQ(
        invocation.complete(
            {-1, -51, false},
            [] {
                ADD_FAILURE();
                return 0;
            },
            [] {
                ADD_FAILURE();
            },
            [] {
                return platform_aicpu_affinity_bind_thread(nullptr);
            }
        ),
        -51
    );
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
}

TEST(AicpuAffinityTls, RecordPoolExhaustionIsReportedAndRecovers) {
    // The pool is process-wide per provider. Release any record this thread
    // still holds, then fill every slot with a distinct resident thread.
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
    constexpr int kPool = simpler::kAffinityRecordPoolSize;
    std::vector<int32_t> cells(kPool, 0);
    std::atomic<int> attempted{0};
    std::atomic<int> bound{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> holders;
    holders.reserve(kPool);
    for (int i = 0; i < kPool; ++i) {
        holders.emplace_back([&, i] {
            if (platform_aicpu_affinity_bind_thread(&cells[i]) == 0) bound.fetch_add(1, std::memory_order_release);
            attempted.fetch_add(1, std::memory_order_release);
            while (!release.load(std::memory_order_acquire)) {}
            platform_aicpu_affinity_bind_thread(nullptr);
        });
    }
    while (attempted.load(std::memory_order_acquire) < kPool) {}
    const int bound_count = bound.load(std::memory_order_acquire);
    int32_t extra = 0;
    const int exhausted = platform_aicpu_affinity_bind_thread(&extra);
    release.store(true, std::memory_order_release);
    for (auto &holder : holders)
        holder.join();
    EXPECT_EQ(bound_count, kPool);
    EXPECT_EQ(exhausted, ENOSPC);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(&extra), 0);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}
