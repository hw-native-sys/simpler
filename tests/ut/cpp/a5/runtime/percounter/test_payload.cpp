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

#include "inner_kernel.h"
#include "runtime_types.h"

uint64_t get_aicore_report_epoch();
#include "lane_loop.h"
#include "region_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace {
using namespace simpler::percounter;

alignas(64) std::array<uint8_t, 4096> registers{};
std::function<void(int64_t *)> execute;
std::function<void()> on_poll;
void kernel(int64_t *args) { execute(args); }
constexpr uint64_t budget = 5'000'000'000ULL;

struct Fixture {
    PercounterControl control{};
    std::array<ChipTaskStorage, 3> storage{};
    std::array<uint64_t, 3> scalars{11, 22, 33};
    std::array<PercounterFanin, 3> fanin{};
    std::array<PercounterCounter, 3> counters{};
    uint64_t producer{0};
    uint64_t producer_address{reinterpret_cast<uint64_t>(&producer)};
    std::array<uint64_t, 2> callable{0, reinterpret_cast<uint64_t>(&kernel)};
    std::array<std::array<DispatchPayload, PERCOUNTER_PAYLOAD_SLOT_COUNT>, 6> payloads{};
    std::array<PercounterMixSignal, 2> signals{};
    std::array<PercounterTicket, 2> mix_tickets{};
    std::array<PercounterTicket, 2> tickets{};
    std::array<uint64_t, 3> order{};
    std::array<uint64_t, 2> mix_order{};
    std::array<uint64_t, 4> sync_pending{};
    PercounterSyncQueue queue{};
    std::array<LaneConfig, 6> cfg{};
    std::array<PercounterLaneState, 6> state{};

    Fixture() {
        registers.fill(0);
        execute = [](int64_t *) {};
        on_poll = {};
        queue.open_task_id = -1;
        for (int t = 0; t < 3; ++t) {
            storage[t].task.task_id = TaskId::make_global(t);
            storage[t].payload.scalar_count = 1;
            storage[t].payload.scalars.set(&scalars[t]);
            set_task(t, 1, 1);
            order[t] = percounter_order_entry(t, 0, 1);
        }
        fanin[0].count = 1;
        for (int i = 0; i < 6; ++i) {
            state[i] = {0, 0, {-1, 0, 1, false, -1}, {0, 0, -1, -1, 0, 0, 0}};
            auto &c = cfg[i];
            c.control = &control;
            c.graph = scheduler_ordinary_graph_view(reinterpret_cast<uint64_t>(storage.data()), 3, 2);
            c.callable_addresses = callable.data();
            c.callable_count = callable.size();
            c.payloads = payloads[i].data();
            for (auto &p : payloads[i]) {
                p.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&p.local_context);
                p.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&p.global_context);
            }
            c.task_count = c.table_task_count = 3;
            c.fanin = fanin.data();
            c.fanin_addresses = &producer_address;
            c.counters = counters.data();
            c.tickets = tickets.data();
            c.ticket_group_count = 1;
            c.order = order.data();
            c.cluster_index = i / 3;
            c.cluster_lane = i % 3;
            c.mix_signal = &signals[i / 3];
            c.mix_order = mix_order.data();
            c.mix_tickets = mix_tickets.data();
            c.sync_queue = &queue;
            c.sync_pending = sync_pending.data();
        }
    }

    void set_task(int task, uint32_t mask, int blocks) {
        for (int lane = 0; lane < 3; ++lane)
            storage[task].task.kernel_id[lane] = (mask & (1U << lane)) ? 1 : -1;
        counters[task].value = percounter_counter_preset(blocks * __builtin_popcount(mask));
    }

    PercounterTask resolve(int task) {
        PercounterTask result{};
        EXPECT_TRUE(percounter_resolve(cfg[0], task, &result));
        return result;
    }

    void enable_mix() {
        set_task(0, 7, 1);
        mix_order[0] = percounter_order_entry(0, 0, 1);
        for (auto &c : cfg)
            c.mix_order_count = c.mix_rendezvous_count = 1;
        percounter_mix_arm(cfg[1], &state[1].claim);
    }

    bool checkpoint(int lane) {
        bool served = false;
        EXPECT_TRUE(percounter_checkpoint_serve(cfg[lane], &state[lane], budget, &served));
        return served;
    }
};

TEST(PercounterPayload, OrdinaryCandidatePreparedBeforeProducerCompletes) {
    Fixture f;
    f.cfg[0].order_count = 3;
    std::vector<int64_t> calls;
    bool prepared_before_ready = false;
    on_poll = [&] {
        if (prepared_before_ready || f.payloads[0][0].function_bin_addr == 0) return;
        EXPECT_EQ(f.producer, 0U);
        EXPECT_TRUE(calls.empty());
        EXPECT_EQ(f.payloads[0][0].args[0], 11U);
        prepared_before_ready = true;
        // Model a producer on another lane completing after preparation.
        f.producer = 1;
    };
    execute = [&](int64_t *args) {
        calls.push_back(args[0]);
        EXPECT_TRUE(prepared_before_ready);
    };
    percounter_lane_loop(f.cfg[0], budget);
    EXPECT_TRUE(prepared_before_ready);
    EXPECT_EQ(calls, (std::vector<int64_t>{11, 22, 33}));
    EXPECT_EQ(f.control.lane_error, 0U);
}

TEST(PercounterPayload, MixNonCoordinatorsPrepareWhileDependencyIsBlocked) {
    Fixture f;
    f.enable_mix();
    ASSERT_EQ(f.signals[0].armed_generation, 0U);
    for (int lane = 0; lane < 3; ++lane) {
        EXPECT_FALSE(f.checkpoint(lane));
        EXPECT_EQ(f.state[lane].prepared_mix_task, 0);
        EXPECT_EQ(f.payloads[lane][PERCOUNTER_MIX_PAYLOAD_SLOT].args[0], 11U);
        EXPECT_EQ(f.signals[0].arrived[lane], 0U);
    }
    // An ordinary task executes in another slot while MIX still waits.
    auto ordinary = f.resolve(1);
    ASSERT_TRUE(percounter_prepare_task(f.cfg[2], ordinary, 0, 1, &f.payloads[2][0]));
    ASSERT_TRUE(percounter_run_prepared(f.cfg[2], ordinary, &f.payloads[2][0]));
    EXPECT_EQ(f.payloads[2][PERCOUNTER_MIX_PAYLOAD_SLOT].args[0], 11U);
    EXPECT_EQ(f.producer, 0U);
    EXPECT_EQ(f.signals[0].active_mask, 0U);
}

TEST(PercounterPayload, PreparationRecordCanBeReusedAfterAnUnassignedLaneReadsIt) {
    Fixture f;
    f.set_task(0, 3, 1);
    f.set_task(1, 6, 4);
    f.fanin[1].count = 1;
    for (auto &cfg : f.cfg)
        cfg.mix_order_count = 2;

    auto &signal = f.signals[0];
    percounter_mix_signal_clear(&signal);
    signal.armed_generation = 2;
    signal.prepare_task_id = 0;
    signal.prepare_block_idx = 0;
    signal.prepare_block_num = 1;
    signal.prepare_mask = 3;
    percounter_ssbuf_store_release(&signal.prepare_generation, 2);

    std::atomic<bool> read_finished{false};
    bool read_ok = false;
    PercounterMixClaim next{1, 2, 4, false, -1};
    std::thread publisher([&] {
        // This handoff orders test steps without synchronizing the record's accesses.
        while (!read_finished.load(std::memory_order_relaxed)) {}
        percounter_mix_arm(f.cfg[1], &next);
    });
    std::thread reader([&] {
        read_ok = percounter_mix_prepare(f.cfg[2], &f.state[2]);
        read_finished.store(true, std::memory_order_relaxed);
    });
    reader.join();
    publisher.join();

    ASSERT_TRUE(read_ok);
    EXPECT_EQ(f.state[2].prepared_mix_generation, 2U);
    EXPECT_EQ(f.state[2].prepared_mix_task, -1);
    ASSERT_EQ(percounter_ssbuf_load_acquire(&signal.prepare_generation), 4U);
    ASSERT_TRUE(percounter_mix_prepare(f.cfg[2], &f.state[2]));
    EXPECT_EQ(f.state[2].prepared_mix_task, 1);
    const auto &payload = f.payloads[2][PERCOUNTER_MIX_PAYLOAD_SLOT];
    EXPECT_EQ(payload.args[0], 22U);
    EXPECT_EQ(payload.local_context.block_idx, 2);
    EXPECT_EQ(payload.local_context.block_num, 4);
    EXPECT_EQ(signal.armed_generation, 2U);
    EXPECT_EQ(f.producer, 0U);
    EXPECT_EQ(f.control.lane_error, 0U);
}

TEST(PercounterPayload, SpmdCandidatesKeepDistinctBlockContexts) {
    Fixture f;
    f.set_task(0, 1, 2);
    f.cfg[0].order_count = 3;
    f.order = {percounter_order_entry(0, 0, 2), percounter_order_entry(0, 1, 2), percounter_order_entry(1, 0, 1)};
    std::vector<int> blocks;
    const uint32_t prepared_count = std::min(PERCOUNTER_PENDING_CAPACITY, uint32_t{2});
    on_poll = [&] {
        if (f.producer != 0) return;
        for (uint32_t slot = 0; slot < prepared_count; ++slot)
            if (f.payloads[0][slot].function_bin_addr == 0) return;
        for (uint32_t slot = 0; slot < prepared_count; ++slot)
            EXPECT_EQ(f.payloads[0][slot].local_context.block_idx, static_cast<int32_t>(slot));
        f.producer = 1;
    };
    execute = [&](int64_t *args) {
        if (args[0] == 22) return;
        auto *context = reinterpret_cast<LocalContext *>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]);
        EXPECT_EQ(context->block_num, 2);
        blocks.push_back(context->block_idx);
    };
    percounter_lane_loop(f.cfg[0], budget);
    EXPECT_EQ(blocks, (std::vector<int>{0, 1}));
    EXPECT_EQ(f.control.lane_error, 0U);
}

TEST(PercounterPayload, GraphBodyPublishedAfterClaimUsesPreparedPayload) {
    Fixture f;
    f.cfg[0].table_task_count = 2;
    f.cfg[0].order_count = 2;
    f.order[0] = 0;
    std::vector<int64_t> calls;
    bool published_after_claim = false;
    on_poll = [&] {
        if (published_after_claim || f.tickets[0].next == 0) return;
        EXPECT_EQ(f.payloads[0][0].function_bin_addr, 0U);
        f.counters[2].storage_address = reinterpret_cast<uint64_t>(&f.storage[2]);
        f.counters[2].body_index = 0;
        f.counters[2].shell_task_id = 0;
        f.order[0] = percounter_order_entry(2, 0, 1);
        published_after_claim = true;
    };
    execute = [&](int64_t *args) {
        EXPECT_TRUE(published_after_claim);
        calls.push_back(args[0]);
    };
    percounter_lane_loop(f.cfg[0], budget);
    EXPECT_TRUE(published_after_claim);
    EXPECT_EQ(calls, (std::vector<int64_t>{33, 22}));
    EXPECT_EQ(f.control.lane_error, 0U);
}

TEST(PercounterPayload, WaitingMixDoesNotBlockCohortOrLoseItsPayload) {
    Fixture f;
    f.enable_mix();
    for (int lane = 0; lane < 3; ++lane)
        ASSERT_FALSE(f.checkpoint(lane));
    f.set_task(1, 2, 4);
    for (auto &c : f.cfg) {
        c.sync_cohort_count = 2;
        c.sync_block_total = 4;
    }
    auto task = f.resolve(1);
    percounter_sync_enqueue(f.cfg[0], task, 4);
    for (int lane : {1, 2, 4, 5}) {
        bool served = false;
        ASSERT_TRUE(percounter_sync_take(f.cfg[lane], &f.state[lane], budget, &served));
    }
    for (int lane : {1, 2, 4})
        ASSERT_TRUE(f.checkpoint(lane));
    EXPECT_EQ(f.queue.completed, 4U);
    EXPECT_EQ(f.payloads[1][PERCOUNTER_MIX_PAYLOAD_SLOT].args[0], 11U);
    EXPECT_EQ(f.payloads[2][PERCOUNTER_MIX_PAYLOAD_SLOT].args[0], 11U);

    f.producer = 1;
    percounter_mix_arm(f.cfg[1], &f.state[1].claim);
    std::atomic<int> calls{0};
    execute = [&](int64_t *args) {
        EXPECT_EQ(args[0], 11);
        ++calls;
    };
    std::vector<std::thread> lanes;
    for (int lane = 0; lane < 3; ++lane)
        lanes.emplace_back([&, lane] {
            EXPECT_TRUE(f.checkpoint(lane));
        });
    for (auto &lane : lanes)
        lane.join();
    EXPECT_EQ(calls.load(), 3);
    EXPECT_EQ(f.mix_tickets[PERCOUNTER_MIX_TICKET_DONE].next, 1U);
    EXPECT_EQ(f.control.lane_error, 0U);
}

TEST(PercounterPayload, PredicateIsEvaluatedAtExecution) {
    Fixture f;
    int64_t value = 0;
    auto &predicate = f.storage[0].payload.predicate;
    predicate.addr = reinterpret_cast<uint64_t>(&value);
    predicate.target = 1;
    predicate.elem_size = sizeof(value);
    predicate.op = static_cast<PredicateOp>(1);
    f.cfg[0].has_predicates = true;
    auto task = f.resolve(0);
    ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
    };
    value = 1;
    f.producer = 1;
    ASSERT_TRUE(percounter_run_prepared(f.cfg[0], task, &f.payloads[0][0]));
    EXPECT_EQ(calls, 1);
    value = 0;
    ASSERT_TRUE(percounter_run_prepared(f.cfg[0], task, &f.payloads[0][0]));
    EXPECT_EQ(calls, 1);
}

TEST(PercounterPayload, LayoutReservesEveryLaneSlot) {
    PercounterLayout layout{};
    const uint64_t counts[2]{1, 0};
    ASSERT_TRUE(plan_layout(1, 1, 0, counts, &layout));
    EXPECT_EQ(
        layout.counters_offset - layout.dispatch_payloads_offset,
        SCHEDULER_WORKER_CAPACITY * PERCOUNTER_PAYLOAD_SLOT_COUNT * sizeof(DispatchPayload)
    );
}
}  // namespace

volatile uint8_t *sim_get_reg_base() {
    if (on_poll) on_poll();
    return registers.data();
}
uint32_t sim_get_physical_core_id() { return 0; }
