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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "scheduler/scheduler_async_runtime.h"
#include "scheduler/scheduler_local_storage.h"

namespace {
using FinishStorage = SchedulerLocalStorage<SchedulerAsyncFinishProgress>;
static_assert(alignof(SchedulerAsyncFinishProgress) == 8);
static_assert(alignof(FinishStorage) <= 8);
static_assert(!std::is_copy_constructible_v<FinishStorage>);
static_assert(!std::is_copy_assignable_v<FinishStorage>);
static_assert(!std::is_move_constructible_v<FinishStorage>);
static_assert(!std::is_move_assignable_v<FinishStorage>);

// Placement construction starts both the carrier's and its contained value's
// lifetimes. Each scalar-aligned carrier address is exercised independently;
// no test creates an over-aligned T by merely casting a misaligned pointer.
template <typename T>
class GuardedCarrier {
    using Carrier = SchedulerLocalStorage<T>;
    static constexpr unsigned char kGuard = 0xa5;
    alignas(128) std::array<unsigned char, sizeof(Carrier) + 3 * 128> bytes_;
    size_t begin_;
    Carrier *carrier_;

public:
    template <typename... Args>
    explicit GuardedCarrier(size_t displacement, Args &&...args) :
        begin_(128 + displacement) {
        bytes_.fill(kGuard);
        carrier_ = ::new (static_cast<void *>(bytes_.data() + begin_)) Carrier(std::forward<Args>(args)...);
    }
    ~GuardedCarrier() {
        carrier_->get().~T();
        carrier_->~Carrier();
    }
    GuardedCarrier(const GuardedCarrier &) = delete;
    GuardedCarrier &operator=(const GuardedCarrier &) = delete;

    Carrier &carrier() { return *carrier_; }
    uintptr_t address() const { return reinterpret_cast<uintptr_t>(carrier_); }
    bool guards_intact() const {
        const auto guard = [](unsigned char value) {
            return value == kGuard;
        };
        return std::all_of(bytes_.begin(), bytes_.begin() + begin_, guard) &&
               std::all_of(bytes_.begin() + begin_ + sizeof(Carrier), bytes_.end(), guard);
    }
};

void fill_finish(SchedulerAsyncFinishProgress &value) {
    value.completion.resolve.task_ref = 11;
    value.completion.resolve.producer_ref = 13;
    value.completion.resolve.next_fanin = 17;
    value.completion.resolve.fanin_count = 19;
    value.completion.resolve.phase = SchedulerResolvePhase::LINK;
    value.completion.resolve.validate_current_state = false;
    value.completion.current_waiter = 23;
    value.completion.detached_next = 29;
    value.completion.publication.pending_count = 8;
    value.completion.publication.pending_queue = 31;
    value.completion.publication.pending_target = 37;
    value.completion.publication.pending_kind = 41;
    value.completion.publication.resume_phase = 43;
    value.completion.publication.pending_first = 47;
    value.completion.publication.pending_last = 53;
    value.completion.publication.pending_link_count = 59;
    value.completion.visited_count = 61;
    for (size_t i = 0; i < 8; ++i)
        value.completion.publication.pending_tasks[i] = 100 + i;
    // Writing the outer FINISH phase must preserve the nested resolver cursor
    // and both cache lines of the pending publication.
    value.completion.publication.phase = 7;
}

void expect_finish_defaults(const SchedulerAsyncFinishProgress &value) {
    EXPECT_EQ(value.completion.resolve.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.resolve.producer_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.resolve.phase, SchedulerResolvePhase::INITIALIZE);
    EXPECT_TRUE(value.completion.resolve.validate_current_state);
    EXPECT_EQ(value.completion.resolve.next_fanin, 0u);
    EXPECT_EQ(value.completion.resolve.fanin_count, 0u);
    EXPECT_EQ(value.completion.current_waiter, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.detached_next, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.publication.pending_first, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.publication.pending_last, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(value.completion.publication.phase, 0u);
    EXPECT_EQ(value.completion.publication.pending_count, 0u);
    EXPECT_EQ(value.completion.publication.pending_queue, 0u);
    EXPECT_EQ(value.completion.publication.pending_target, 0u);
    EXPECT_EQ(value.completion.publication.pending_kind, 0u);
    EXPECT_EQ(value.completion.publication.resume_phase, 0u);
    EXPECT_EQ(value.completion.publication.pending_link_count, 0u);
    EXPECT_EQ(value.completion.visited_count, 0u);
    for (const auto task : value.completion.publication.pending_tasks)
        EXPECT_EQ(task, 0);
}

struct alignas(128) ConstructedValue {
    uint64_t first{19};
    uint64_t middle[14]{};
    uint64_t last{23};

    ConstructedValue() = default;
    ConstructedValue(uint64_t value, uint64_t &calls, std::unique_ptr<uint64_t> tail) :
        first(value),
        last(*tail) {
        ++calls;
        middle[7] = value + *tail;
    }
};
static_assert(sizeof(ConstructedValue) == 128 && std::is_trivially_destructible_v<ConstructedValue>);
static_assert(alignof(SchedulerLocalStorage<ConstructedValue>) <= 8);
}  // namespace

TEST(HbgLocalStorage, EveryScalarAlignedBasePreservesFinishFieldsAndGuards) {
    for (size_t displacement = 0; displacement < 128; displacement += 8) {
        SCOPED_TRACE(displacement);
        GuardedCarrier<SchedulerAsyncFinishProgress> placed(displacement);
        auto &carrier = placed.carrier();
        auto &value = carrier.get();
        const auto object = reinterpret_cast<uintptr_t>(&value);
        ASSERT_EQ(placed.address() % 128, displacement);
        ASSERT_EQ(object % alignof(SchedulerAsyncFinishProgress), 0u);
        ASSERT_GE(object, placed.address());
        ASSERT_LE(object + sizeof(value), placed.address() + sizeof(carrier));

        fill_finish(value);
        const auto &observed = std::as_const(carrier).get();
        EXPECT_EQ(std::as_const(carrier).operator->(), &observed);
        EXPECT_EQ(carrier.operator->(), &value);
        EXPECT_EQ(observed.completion.resolve.task_ref, 11);
        EXPECT_EQ(observed.completion.resolve.producer_ref, 13);
        EXPECT_EQ(observed.completion.resolve.next_fanin, 17u);
        EXPECT_EQ(observed.completion.resolve.fanin_count, 19u);
        EXPECT_EQ(observed.completion.resolve.phase, SchedulerResolvePhase::LINK);
        EXPECT_FALSE(observed.completion.resolve.validate_current_state);
        EXPECT_EQ(observed.completion.current_waiter, 23);
        EXPECT_EQ(observed.completion.detached_next, 29);
        EXPECT_EQ(observed.completion.publication.phase, 7u);
        EXPECT_EQ(observed.completion.publication.pending_count, 8u);
        EXPECT_EQ(observed.completion.publication.pending_queue, 31u);
        EXPECT_EQ(observed.completion.publication.pending_target, 37u);
        EXPECT_EQ(observed.completion.publication.pending_kind, 41u);
        EXPECT_EQ(observed.completion.publication.resume_phase, 43u);
        EXPECT_EQ(observed.completion.publication.pending_first, 47);
        EXPECT_EQ(observed.completion.publication.pending_last, 53);
        EXPECT_EQ(observed.completion.publication.pending_link_count, 59u);
        EXPECT_EQ(observed.completion.visited_count, 61u);
        for (size_t i = 0; i < 8; ++i)
            EXPECT_EQ(observed.completion.publication.pending_tasks[i], 100 + i);
        EXPECT_TRUE(placed.guards_intact());
    }
}

TEST(HbgLocalStorage, ResetRestoresNonzeroSentinelsAtEveryScalarAlignedBase) {
    for (size_t displacement = 0; displacement < 128; displacement += 8) {
        SCOPED_TRACE(displacement);
        GuardedCarrier<SchedulerAsyncFinishProgress> placed(displacement);
        auto &carrier = placed.carrier();
        expect_finish_defaults(carrier.get());
        const auto address = reinterpret_cast<uintptr_t>(&carrier.get());
        for (unsigned iteration = 0; iteration < 3; ++iteration) {
            fill_finish(carrier.get());
            carrier.reset();
            EXPECT_EQ(reinterpret_cast<uintptr_t>(&carrier.get()), address);
            expect_finish_defaults(carrier.get());
            EXPECT_TRUE(placed.guards_intact());
        }
    }
}

TEST(HbgLocalStorage, ConstructionAndResetForwardReferencesAndMoveOnlyArguments) {
    for (size_t displacement = 0; displacement < 128; displacement += 8) {
        SCOPED_TRACE(displacement);
        uint64_t calls = 0;
        auto first_tail = std::make_unique<uint64_t>(101);
        GuardedCarrier<ConstructedValue> placed(displacement, 31, calls, std::move(first_tail));
        auto &carrier = placed.carrier();
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&carrier.get()) % 128, 0u);
        EXPECT_EQ(first_tail, nullptr);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(carrier->first, 31u);
        EXPECT_EQ(carrier->middle[7], 132u);
        EXPECT_EQ(carrier->last, 101u);

        auto second_tail = std::make_unique<uint64_t>(211);
        carrier.reset(37, calls, std::move(second_tail));
        EXPECT_EQ(second_tail, nullptr);
        EXPECT_EQ(calls, 2u);
        EXPECT_EQ(carrier->first, 37u);
        EXPECT_EQ(carrier->middle[7], 248u);
        EXPECT_EQ(carrier->last, 211u);

        carrier.reset();
        EXPECT_EQ(calls, 2u);
        EXPECT_EQ(carrier->first, 19u);
        for (const auto word : carrier->middle)
            EXPECT_EQ(word, 0u);
        EXPECT_EQ(carrier->last, 23u);
        EXPECT_TRUE(placed.guards_intact());
    }
}
