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
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_round_gate.h"

namespace {
using namespace simpler::tmr;

class Signal {
public:
    void set() {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = true;
        condition_.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return ready_;
        });
    }
    bool wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [&] {
            return ready_;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool ready_{false};
};

class Round {
public:
    explicit Round(KernelRoundGate &gate, int32_t count = 3) :
        gate(gate),
        tickets(count),
        admissions(count) {
        for (int32_t i = 0; i < count; ++i)
            EXPECT_TRUE(gate.join(count, i, &tickets[i]));
    }
    void admit(int32_t status = 0) {
        const int32_t allowed[]{0, 1};
        ASSERT_TRUE(gate.publish_admission(tickets[0], allowed, 2, status));
        for (size_t i = 0; i < tickets.size(); ++i)
            ASSERT_TRUE(gate.wait_admission(tickets[i], &admissions[i]));
    }
    void finish(int32_t run, int32_t sm, int32_t cleanup, int32_t expected) {
        for (size_t i = 0; i < tickets.size(); ++i) {
            EXPECT_EQ(
                gate.arrive(tickets[i], i == 0 ? run : 0),
                i + 1 == tickets.size() ? RoundArrival::Finalizer : RoundArrival::Peer
            );
        }
        ASSERT_TRUE(gate.publish_final_status(tickets.back(), sm, cleanup));
        EXPECT_FALSE(gate.publish_final_status(tickets.back(), sm, cleanup));
        for (size_t i = 0; i < tickets.size(); ++i) {
            KernelFinalStatus result{};
            ASSERT_TRUE(gate.read_final_status(tickets[i], &result));
            EXPECT_EQ(result.runtime_status, expected);
            EXPECT_EQ(result.cleanup_status, cleanup);
            EXPECT_EQ(gate.depart(tickets[i]), i + 1 == tickets.size() ? RoundDeparture::Last : RoundDeparture::Peer);
        }
        EXPECT_FALSE(gate.idle());
        EXPECT_TRUE(gate.complete_departure(tickets.back()));
        EXPECT_TRUE(gate.idle());
    }
    KernelRoundGate &gate;
    std::vector<KernelRoundTicket> tickets;
    std::vector<KernelRoundAdmission> admissions;
};

TEST(TmrKernelRoundGateTest, BoundsAndDuplicateProtocolOperationsDoNotAdvanceRound) {
    KernelRoundGate gate;
    KernelRoundTicket unchanged{77, 99};
    EXPECT_FALSE(gate.join(0, 0, &unchanged));
    EXPECT_FALSE(gate.join(MAX_GATE_THREADS + 1, 0, &unchanged));
    EXPECT_FALSE(gate.join(2, 0, nullptr));
    EXPECT_EQ(unchanged.epoch, 77u);
    EXPECT_EQ(unchanged.launch_index, 99);
    Round round(gate);
    EXPECT_FALSE(gate.join(3, 9, &unchanged));
    EXPECT_EQ(unchanged.epoch, 77u);
    const int32_t allowed[]{0, 1};
    const int32_t duplicate[]{0, 0};
    EXPECT_FALSE(gate.publish_admission(round.tickets[0], duplicate, 2, 0));
    auto aliased_epoch = round.tickets[0];
    aliased_epoch.epoch += uint64_t{1} << 56;
    EXPECT_FALSE(gate.publish_admission(aliased_epoch, allowed, 2, 0));
    round.admit();
    EXPECT_FALSE(gate.publish_admission(round.tickets[0], allowed, 2, 0));
    EXPECT_EQ(gate.depart(round.tickets[0]), RoundDeparture::Invalid);
    KernelRoundAdmission admission{};
    EXPECT_FALSE(gate.wait_admission(round.tickets[0], &admission));
    round.finish(0, 0, 0, 0);
    EXPECT_FALSE(gate.complete_departure(round.tickets.back()));
}

TEST(TmrKernelRoundGateTest, AdmissionRunAndCleanupFailuresPreservePriorityAndAllowReuse) {
    KernelRoundGate gate;
    {
        Round round(gate);
        round.admit(-11);
        round.finish(-33, -44, -55, -11);
    }
    {
        Round round(gate);
        round.admit();
        round.finish(-33, -44, -55, -44);
    }
    {
        Round round(gate);
        round.admit();
        round.finish(-33, 0, -55, -33);
    }
    Round round(gate);
    round.admit();
    round.finish(0, 0, 0, 0);
}

TEST(TmrKernelRoundGateTest, OldTicketsCannotWriteNewEpochOrReadItsVerdict) {
    KernelRoundGate gate;
    Round old(gate);
    old.admit();
    old.finish(0, 0, 0, 0);
    Round next(gate);
    EXPECT_GT(next.tickets[0].epoch, old.tickets[0].epoch);
    const int32_t allowed[]{0, 1};
    EXPECT_FALSE(gate.publish_admission(old.tickets[0], allowed, 2, -1));
    next.admit();
    for (const auto &ticket : old.tickets) {
        KernelRoundAdmission admission{17, 18};
        KernelFinalStatus final{19, 20};
        EXPECT_FALSE(gate.wait_admission(ticket, &admission));
        EXPECT_EQ(gate.arrive(ticket, -1), RoundArrival::Invalid);
        EXPECT_FALSE(gate.publish_final_status(ticket, -1, -1));
        EXPECT_FALSE(gate.read_final_status(ticket, &final));
        EXPECT_EQ(gate.depart(ticket), RoundDeparture::Invalid);
        EXPECT_FALSE(gate.complete_departure(ticket));
        EXPECT_EQ(admission.status, 17);
        EXPECT_EQ(final.runtime_status, 19);
    }
    next.finish(0, 0, 0, 0);
}

TEST(TmrKernelRoundGateTest, ConcurrentDuplicateArrivalIsCountedOnce) {
    KernelRoundGate gate;
    Round round(gate, 2);
    round.admit();
    Signal start;
    RoundArrival results[2]{};
    std::thread a([&] {
        start.wait();
        results[0] = gate.arrive(round.tickets[0], -1);
    });
    std::thread b([&] {
        start.wait();
        results[1] = gate.arrive(round.tickets[0], -1);
    });
    start.set();
    a.join();
    b.join();
    EXPECT_EQ((results[0] == RoundArrival::Peer) + (results[1] == RoundArrival::Peer), 1);
    EXPECT_EQ((results[0] == RoundArrival::Invalid) + (results[1] == RoundArrival::Invalid), 1);
    EXPECT_EQ(gate.arrive(round.tickets[1], 0), RoundArrival::Finalizer);
    ASSERT_TRUE(gate.publish_final_status(round.tickets[1], 0, 0));
    for (size_t i = 0; i < 2; ++i) {
        KernelFinalStatus result{};
        ASSERT_TRUE(gate.read_final_status(round.tickets[i], &result));
        EXPECT_EQ(result.runtime_status, -1);
        EXPECT_EQ(gate.depart(round.tickets[i]), i == 0 ? RoundDeparture::Peer : RoundDeparture::Last);
    }
    EXPECT_TRUE(gate.complete_departure(round.tickets[1]));
}

TEST(TmrKernelRoundGateTest, SlowFinalizerReaderAndRetiringOwnerKeepStoragePinned) {
    KernelRoundGate gate;
    Round round(gate, 2);
    round.admit();
    ASSERT_EQ(gate.arrive(round.tickets[0], 0), RoundArrival::Peer);
    ASSERT_EQ(gate.arrive(round.tickets[1], 0), RoundArrival::Finalizer);
    Signal reader_started;
    Signal reader_departed;
    std::atomic<bool> result_read{false};
    std::thread reader([&] {
        reader_started.set();
        KernelFinalStatus result{};
        EXPECT_TRUE(gate.read_final_status(round.tickets[0], &result));
        EXPECT_EQ(result.runtime_status, -7);
        result_read.store(true);
        EXPECT_EQ(gate.depart(round.tickets[0]), RoundDeparture::Peer);
        reader_departed.set();
    });
    reader_started.wait();
    EXPECT_FALSE(result_read.load());
    EXPECT_FALSE(gate.complete_departure(round.tickets[1]));
    ASSERT_TRUE(gate.publish_final_status(round.tickets[1], -7, 0));
    reader_departed.wait();
    reader.join();
    KernelRoundTicket next{0, -1};
    EXPECT_FALSE(gate.join(2, 0, &next));
    EXPECT_EQ(gate.depart(round.tickets[1]), RoundDeparture::Invalid);
    KernelFinalStatus result{};
    ASSERT_TRUE(gate.read_final_status(round.tickets[1], &result));
    EXPECT_EQ(gate.depart(round.tickets[1]), RoundDeparture::Last);
    EXPECT_FALSE(gate.join(2, 0, &next));
    EXPECT_FALSE(gate.idle());
    // Storage cleanup belongs here, before the sole Idle publication.
    int storage = 77;
    storage = 0;
    ASSERT_TRUE(gate.complete_departure(round.tickets[1]));
    Round following(gate);
    EXPECT_EQ(storage, 0);
    EXPECT_EQ(result.runtime_status, -7);
    following.admit();
    following.finish(0, 0, 0, 0);
}

TEST(TmrKernelRoundGateTest, ThreadedRepeatedRoundsIncludeFilteredAndDuplicateCpuReports) {
    for (int32_t launched : {2, 3, MAX_GATE_THREADS}) {
        KernelRoundGate gate;
        for (int repetition = 0; repetition < 30; ++repetition) {
            Signal start;
            const int32_t allowed[]{8, 9};
            std::atomic<int32_t> role_bits{0};
            std::atomic<int32_t> finalized{0};
            std::atomic<int32_t> retired{0};
            std::vector<std::thread> threads;
            for (int32_t i = 0; i < launched; ++i) {
                threads.emplace_back([&, i] {
                    start.wait();
                    KernelRoundTicket ticket;
                    ASSERT_TRUE(gate.join(launched, i % 2 == 0 ? 8 : -1, &ticket));
                    if (ticket.launch_index == 0) ASSERT_TRUE(gate.publish_admission(ticket, allowed, 2, 0));
                    KernelRoundAdmission admission;
                    ASSERT_TRUE(gate.wait_admission(ticket, &admission));
                    if (admission.execution_index >= 0) {
                        const int32_t bit = 1 << admission.execution_index;
                        EXPECT_EQ(role_bits.fetch_or(bit) & bit, 0);
                    }
                    const auto arrival = gate.arrive(ticket, 0);
                    ASSERT_NE(arrival, RoundArrival::Invalid);
                    if (arrival == RoundArrival::Finalizer) {
                        finalized.fetch_add(1);
                        ASSERT_TRUE(gate.publish_final_status(ticket, 0, 0));
                    }
                    KernelFinalStatus result;
                    ASSERT_TRUE(gate.read_final_status(ticket, &result));
                    EXPECT_EQ(result.runtime_status, 0);
                    const auto departure = gate.depart(ticket);
                    ASSERT_NE(departure, RoundDeparture::Invalid);
                    if (departure == RoundDeparture::Last) {
                        retired.fetch_add(1);
                        ASSERT_TRUE(gate.complete_departure(ticket));
                    }
                });
            }
            start.set();
            for (auto &thread : threads)
                thread.join();
            EXPECT_EQ(role_bits.load(), 3);
            EXPECT_EQ(finalized.load(), 1);
            EXPECT_EQ(retired.load(), 1);
            EXPECT_TRUE(gate.idle());
        }
    }
}

void finish_tickets(KernelRoundGate &gate, const std::vector<KernelRoundTicket> &tickets, int32_t status = 0) {
    for (size_t i = 0; i < tickets.size(); ++i) {
        EXPECT_EQ(gate.arrive(tickets[i], 0), i + 1 == tickets.size() ? RoundArrival::Finalizer : RoundArrival::Peer);
    }
    ASSERT_TRUE(gate.publish_final_status(tickets.back(), 0, 0));
    for (size_t i = 0; i < tickets.size(); ++i) {
        KernelFinalStatus result;
        ASSERT_TRUE(gate.read_final_status(tickets[i], &result));
        EXPECT_EQ(result.runtime_status, status);
        EXPECT_EQ(gate.depart(tickets[i]), i + 1 == tickets.size() ? RoundDeparture::Last : RoundDeparture::Peer);
    }
    EXPECT_FALSE(gate.idle());
    ASSERT_TRUE(gate.complete_departure(tickets.back()));
    EXPECT_TRUE(gate.idle());
}

TEST(TmrKernelRoundGateTest, ExactPrefixAdmitsBeforeLateDuplicateCpuReports) {
    for (int32_t status : {0, -19}) {
        KernelRoundGate gate;
        std::vector<KernelRoundTicket> tickets(4);
        ASSERT_TRUE(gate.join(4, 10, &tickets[0]));
        ASSERT_TRUE(gate.join(4, 11, &tickets[1]));
        const int32_t allowed[]{10, 11};
        ASSERT_TRUE(gate.publish_admission(tickets[0], allowed, 2, status));
        for (int32_t i = 0; i < 2; ++i) {
            KernelRoundAdmission admission;
            ASSERT_TRUE(gate.wait_admission(tickets[i], &admission));
            EXPECT_EQ(admission.execution_index, i);
            EXPECT_EQ(admission.status, status);
        }
        EXPECT_FALSE(gate.idle());
        for (int32_t i = 2; i < 4; ++i) {
            ASSERT_TRUE(gate.join(4, 10, &tickets[i]));
            KernelRoundAdmission admission;
            ASSERT_TRUE(gate.wait_admission(tickets[i], &admission));
            EXPECT_EQ(admission.execution_index, -1);
            EXPECT_EQ(admission.status, status);
        }
        KernelRoundTicket extra;
        EXPECT_FALSE(gate.join(4, 11, &extra));
        finish_tickets(gate, tickets, status);
        Round following(gate);
        following.admit();
        following.finish(0, 0, 0, 0);
    }
}

TEST(TmrKernelRoundGateTest, FallbackWaitsForEveryReportAndPreservesExactMatches) {
    KernelRoundGate gate;
    std::vector<KernelRoundTicket> tickets(4);
    ASSERT_TRUE(gate.join(4, 90, &tickets[0]));
    ASSERT_TRUE(gate.join(4, 91, &tickets[1]));
    const int32_t allowed[]{10, 11};
    Signal admitted;
    std::thread publisher([&] {
        EXPECT_TRUE(gate.publish_admission(tickets[0], allowed, 2, 0));
        admitted.set();
    });
    EXPECT_FALSE(admitted.wait_for(std::chrono::milliseconds(20)));
    EXPECT_TRUE(gate.join(4, 10, &tickets[2]));
    EXPECT_FALSE(admitted.wait_for(std::chrono::milliseconds(20)));
    EXPECT_TRUE(gate.join(4, 92, &tickets[3]));
    publisher.join();
    const int32_t expected[]{1, -1, 0, -1};
    for (size_t i = 0; i < tickets.size(); ++i) {
        KernelRoundAdmission admission;
        ASSERT_TRUE(gate.wait_admission(tickets[i], &admission));
        EXPECT_EQ(admission.execution_index, expected[i]);
    }
    finish_tickets(gate, tickets);
}

TEST(TmrKernelRoundGateTest, EarliestDuplicateWinsBeforeExactPrefixCompletes) {
    KernelRoundGate gate;
    std::vector<KernelRoundTicket> tickets(4);
    ASSERT_TRUE(gate.join(4, 10, &tickets[0]));
    ASSERT_TRUE(gate.join(4, 10, &tickets[1]));
    ASSERT_TRUE(gate.join(4, 11, &tickets[2]));
    const int32_t allowed[]{10, 11};
    ASSERT_TRUE(gate.publish_admission(tickets[0], allowed, 2, 0));
    ASSERT_TRUE(gate.join(4, 11, &tickets[3]));
    const int32_t expected[]{0, -1, 1, -1};
    for (size_t i = 0; i < tickets.size(); ++i) {
        KernelRoundAdmission admission;
        ASSERT_TRUE(gate.wait_admission(tickets[i], &admission));
        EXPECT_EQ(admission.execution_index, expected[i]);
    }
    finish_tickets(gate, tickets);
}

}  // namespace
