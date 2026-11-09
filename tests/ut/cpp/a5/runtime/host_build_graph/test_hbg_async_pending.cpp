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
#include <functional>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

#include "scheduler/scheduler_async_pending.h"

namespace {

namespace pending = scheduler_async_pending;
using Result = pending::Result;

struct Word {
    uint64_t value{0};
    uint32_t operations{0};
    uint64_t fetch_add(uint64_t delta) {
        const uint64_t previous = value;
        value += delta;
        ++operations;
        return previous;
    }
};

TEST(HbgAsyncPending, NoRegistrationUsesSynchronousCompletion) {
    pending::Join join{};
    EXPECT_FALSE(pending::ready_to_finish(join.word));
    EXPECT_EQ(pending::close_computation(join).result, Result::SYNCHRONOUS);
    EXPECT_TRUE(pending::closed(join.word));
    EXPECT_FALSE(pending::ready_to_finish(join.word));
}

TEST(HbgAsyncPending, AnEarlyZeroObservationCannotAuthorizeCompletion) {
    pending::Join join{};
    ASSERT_EQ(join.word, 0u);
    ASSERT_EQ(pending::register_group(join).result, Result::PENDING);
    EXPECT_EQ(pending::close_computation(join).result, Result::PENDING);
    EXPECT_EQ(pending::complete_group(join).result, Result::FINISH_OWNER);
}

TEST(HbgAsyncPending, EnumeratesPublicationReleaseCheckAndCloseInterleavings) {
    struct State {
        Word word{};
        uint32_t registered{0};
        uint32_t published{0};
        uint32_t released{0};
        uint32_t completed{0};
        bool closed{false};
        uint32_t finish_owners{0};
        uint32_t synchronous_owners{0};
    };
    using Key = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, uint32_t>;
    for (uint32_t groups = 1; groups <= 3; ++groups) {
        const uint32_t all = (1u << groups) - 1;
        for (uint32_t active = 0; active <= all; ++active) {
            std::set<Key> visited;
            uint32_t terminal_states = 0;
            std::function<void(State)> visit;
            visit = [&](State s) {
                ASSERT_TRUE(pending::valid(s.word.value));
                ASSERT_EQ(pending::registered(s.word.value), static_cast<uint32_t>(__builtin_popcount(s.registered)));
                ASSERT_EQ(
                    pending::pending(s.word.value),
                    static_cast<uint32_t>(__builtin_popcount(s.registered & ~s.completed))
                );
                ASSERT_EQ(pending::closed(s.word.value), s.closed);
                ASSERT_LE(s.finish_owners + s.synchronous_owners, 1u);
                if (s.finish_owners || s.synchronous_owners) {
                    ASSERT_EQ(s.released, all);
                    ASSERT_EQ(s.completed, active);
                    ASSERT_TRUE(s.closed);
                }
                if (!visited
                         .emplace(
                             s.word.value, s.registered, s.published, s.released, s.completed, s.closed,
                             s.finish_owners, s.synchronous_owners
                         )
                         .second)
                    return;
                if (s.closed && s.completed == active) {
                    EXPECT_EQ(s.finish_owners, active == 0 ? 0u : 1u);
                    EXPECT_EQ(s.synchronous_owners, active == 0 ? 1u : 0u);
                    ++terminal_states;
                    return;
                }
                auto apply = [&](State next, pending::Outcome outcome) {
                    EXPECT_EQ(next.word.operations, s.word.operations + 1);
                    if (outcome.result == Result::FINISH_OWNER) ++next.finish_owners;
                    else if (outcome.result == Result::SYNCHRONOUS) ++next.synchronous_owners;
                    else EXPECT_EQ(outcome.result, Result::PENDING);
                    visit(next);
                };
                for (uint32_t group = 0; group < groups; ++group) {
                    const uint32_t bit = 1u << group;
                    if (!(active & bit)) {
                        if (!(s.released & bit)) {
                            auto next = s;
                            next.released |= bit;
                            visit(next);
                        }
                        continue;
                    }
                    if (!(s.registered & bit)) {
                        auto next = s;
                        next.registered |= bit;
                        const auto result = pending::detail::register_group(next.word);
                        apply(next, result);
                    } else if (!(s.published & bit)) {
                        auto next = s;
                        next.published |= bit;
                        visit(next);
                    } else {
                        if (!(s.released & bit)) {
                            auto next = s;
                            next.released |= bit;
                            visit(next);
                        }
                        if (!(s.completed & bit)) {
                            auto next = s;
                            next.completed |= bit;
                            const auto result = pending::detail::complete_group(next.word);
                            apply(next, result);
                        }
                    }
                }
                if (s.released == all && !s.closed) {
                    auto next = s;
                    next.closed = true;
                    const auto result = pending::detail::close_computation(next.word);
                    apply(next, result);
                }
            };
            visit(State{});
            EXPECT_GT(terminal_states, 0u);
        }
    }
}

TEST(HbgAsyncPending, InvalidAndDuplicateTransitionsNeverGrantCompletion) {
    for (uint64_t word : {UINT64_C(1), UINT64_C(2) | pending::REGISTERED_UNIT, UINT64_C(1) << 63}) {
        Word registration{word}, completion{word}, close{word};
        EXPECT_EQ(pending::detail::register_group(registration).result, Result::INVALID_WORD);
        EXPECT_EQ(pending::detail::complete_group(completion).result, Result::INVALID_WORD);
        EXPECT_EQ(pending::detail::close_computation(close).result, Result::INVALID_WORD);
    }
    Word late{pending::COMPUTATION_CLOSED}, duplicate{pending::COMPUTATION_CLOSED}, empty{};
    EXPECT_EQ(pending::detail::register_group(late).result, Result::INVALID_TRANSITION);
    EXPECT_EQ(pending::detail::close_computation(duplicate).result, Result::INVALID_TRANSITION);
    EXPECT_EQ(pending::detail::complete_group(empty).result, Result::COUNT_UNDERFLOW);
    Word full{pending::COUNT_MAX * pending::REGISTERED_UNIT + pending::COUNT_MAX};
    EXPECT_EQ(pending::detail::register_group(full).result, Result::COUNT_OVERFLOW);
}

TEST(HbgAsyncPending, RealThreadsRaceTheLastCheckAgainstComputationClose) {
    for (uint32_t round = 0; round < 64; ++round) {
        pending::Join join{};
        std::atomic<uint32_t> released{0}, owners{0}, errors{0};
        std::vector<std::thread> threads;
        auto finish = [&]() {
            const auto result = pending::complete_group(join).result;
            if (result == Result::FINISH_OWNER) ++owners;
            else if (result != Result::PENDING) ++errors;
        };
        for (uint32_t index = 0; index < 8; ++index) {
            threads.emplace_back([&, index]() {
                if (pending::register_group(join).result != Result::PENDING) ++errors;
                if ((index + round) % 2 == 0) finish();
                released.fetch_add(1, std::memory_order_release);
                if ((index + round) % 2 != 0) finish();
            });
        }
        while (released.load(std::memory_order_acquire) != 8) {}
        const auto result = pending::close_computation(join).result;
        if (result == Result::FINISH_OWNER) ++owners;
        else if (result != Result::PENDING) ++errors;
        for (auto &thread : threads)
            thread.join();
        EXPECT_EQ(errors, 0u);
        EXPECT_EQ(owners, 1u);
        EXPECT_TRUE(pending::ready_to_finish(join.word));
    }
}

}  // namespace
