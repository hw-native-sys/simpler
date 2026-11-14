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
#include <memory>
#include <thread>
#include <vector>

namespace {
using namespace simpler::percounter;

TEST(PercounterScalarStorage, PreservesMixedIntervalsWithoutLogicalCount) {
    std::array<uint64_t, 8> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 100, 117);
    trace.write({200, uint64_t{UINT32_MAX} + 300, 42, static_cast<uint32_t>(ProfilePhase::Prepare), 7});
    trace.append_memory(ProfilePhase::DsbAll, 500, 500);

    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 100);
    EXPECT_EQ(record.end_time, 117);
    EXPECT_EQ(record.task_ref, -1);
    EXPECT_EQ(record.count, 1);
    EXPECT_EQ(record.kind, static_cast<uint32_t>(ProfilePhase::SsbufLoad) | (3U << 16));
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 200);
    EXPECT_EQ(record.end_time, uint64_t{UINT32_MAX} + 300);
    EXPECT_EQ(record.task_ref, 42);
    EXPECT_EQ(record.count, 7);
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 500);
    EXPECT_EQ(record.end_time, 500);
    EXPECT_EQ(cursor, trace.storage_words);
    EXPECT_EQ(trace.dropped, 0);
}

TEST(PercounterScalarStorage, RejectsWholeExtendedRecordAtCapacityBoundary) {
    constexpr uint32_t capacity = PERCOUNTER_SCALAR_WORD_CAPACITY;
    std::vector<uint64_t> slots(capacity + 1);
    slots.back() = 0xfeed;
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.storage_words = capacity - 1;
    trace.write({100, 200, 42, static_cast<uint32_t>(ProfilePhase::Prepare), 1});
    EXPECT_EQ(trace.storage_words, capacity - 1);
    EXPECT_EQ(trace.dropped, 1);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 0);
    trace.append_memory(ProfilePhase::SsbufLoad, 300, 400);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    trace.append_memory(ProfilePhase::SsbufLoad, 500, 600);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(trace.dropped, 2);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    EXPECT_EQ(slots.back(), 0xfeed);
}

TEST(PercounterScalarStorage, InvalidIntervalsNeverConsumeStorage) {
    std::array<uint64_t, 4> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.storage_words, 0);
    EXPECT_EQ(trace.dropped, 1);
    trace.dropped = UINT32_MAX;
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.dropped, UINT32_MAX);
}

TEST(PercounterScalarStorage, CompactWordsPreserveTicksAndUseFullWidthForLongIntervals) {
    std::array<uint64_t, 24> words{};
    PercounterScalarTrace trace{};
    trace.records = words.data();
    trace.window_start = 1000;
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 1007, 1012);
    EXPECT_EQ(words[0], UINT64_C(0x031c000500000007));
    trace.append_memory(ProfilePhase::SsbufLoad, 2000, 2000 + uint64_t{UINT16_MAX} + 1);
    trace.append_memory(ProfilePhase::SsbufLoad, uint64_t{UINT32_MAX} + 1001, uint64_t{UINT32_MAX} + 1002);
    trace.depth = 127;
    trace.append_memory(ProfilePhase::SsbufLoad, 3000, 3010);
    EXPECT_EQ(trace.storage_words, 16);
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, 1007);
    EXPECT_EQ(record.end_time, 1012);
    EXPECT_EQ(record.kind >> 16, 3);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.end_time, 2000 + uint64_t{UINT16_MAX} + 1);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, uint64_t{UINT32_MAX} + 1001);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.kind >> 16, 128);
    EXPECT_EQ(cursor, trace.storage_words);
}

TEST(PercounterScalarStorage, RejectsPartialExtendedRowsAndOverflowingAbsoluteTime) {
    const std::array<uint64_t, 4> partial{PERCOUNTER_SCALAR_EXTENDED, 1, 2, 3};
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    EXPECT_FALSE(percounter_decode_scalar_storage(partial.data(), partial.size(), cursor, 0, record));
    const uint64_t offset = 1;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&offset, 1, cursor, UINT64_MAX, record));
    const uint64_t duration = UINT64_C(1) << 32;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&duration, 1, cursor, UINT64_MAX, record));
}

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
    std::array<pto2::resident::AsyncGroupData, 6> async_groups{};
    std::array<std::array<DeferredCompletionEntry, PERCOUNTER_ASYNC_CAPACITY>, 6> async_entries{};
    std::array<PercounterAsyncCompletion, 16> async_completions{};
    std::array<uint64_t, 3> parked_next{};

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
            c.async_group = &async_groups[i];
            c.async_entries = async_entries[i].data();
            c.async_completions = async_completions.data();
            c.async_completion_capacity = async_completions.size();
            c.parked_next = parked_next.data();
            c.async_group->capacity = PERCOUNTER_ASYNC_CAPACITY;
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

    bool run(int lane, const PercounterTask &task, DispatchPayload *payload) {
        bool deferred = false;
        return percounter_run_prepared(cfg[lane], &state[lane], task, payload, &deferred);
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

TEST(PercounterPayload, PreparationBindsCurrentImageWithoutReplacingContextPointers) {
    Fixture f;
    auto task = f.resolve(0);
    auto &payload = f.payloads[0][0];
    const auto local = payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX];
    const auto global = payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX];
    ASSERT_TRUE(percounter_prepare_subtask(f.cfg[0], task, 0, 1, 3, 5, &payload));
    EXPECT_EQ(payload.function_bin_addr, reinterpret_cast<uint64_t>(&kernel));
    EXPECT_EQ(payload.args[0], f.scalars[0]);
    EXPECT_EQ(payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX], local);
    EXPECT_EQ(payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX], global);
    EXPECT_EQ(payload.local_context.block_idx, 3);
    EXPECT_EQ(payload.local_context.block_num, 5);
    task.graph.view_version = 0;
    EXPECT_FALSE(percounter_prepare_subtask(f.cfg[0], task, 0, 1, 0, 1, &payload));
    task = f.resolve(0);
    f.storage[0].payload.scalar_count = 1;
    f.storage[0].payload.scalars.set(nullptr);
    EXPECT_FALSE(percounter_prepare_subtask(f.cfg[0], task, 0, 1, 0, 1, &payload));
}

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
    ASSERT_TRUE(f.run(2, ordinary, &f.payloads[2][0]));
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
    ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
    percounter_commit_task_timing(f.cfg[0]);
    EXPECT_EQ(calls, 1);
    value = 0;
    ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
    percounter_commit_task_timing(f.cfg[0]);
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

TEST(PercounterPayload, TaskTimingRecordsIdentityAndBatchesCommittedCount) {
    Fixture f;
    auto buffer = std::make_unique<ChipSwimlaneAicoreTaskBuffer>();
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(buffer.get());
    ChipSwimlaneAicoreLocalState profile{};
    f.cfg[0].profile_head = &head;
    f.cfg[0].profile = &profile;
    auto task = f.resolve(0);
    ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
    };
    for (uint32_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
        percounter_commit_task_timing(f.cfg[0]);
    }
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(buffer->count, 0U);
    percounter_flush_stream_records(&profile);
    EXPECT_EQ(buffer->count, 3U);
    EXPECT_EQ(head.total_record_count, 0U);
    for (uint32_t i = 0; i < 3; ++i) {
        const auto &record = buffer->records[i];
        EXPECT_EQ(TaskId::to_uint64(record.task_token), TaskId::to_uint64(TaskId::make_global(0)));
        EXPECT_EQ(record.reg_task_id, i + 1);
        EXPECT_GT(record.start_time, 0U);
        EXPECT_GE(record.end_time, record.start_time);
    }
}

TEST(PercounterPayload, TaskTimingUsesGraphBodyIdentityAndResetsEachRun) {
    Fixture f;
    auto buffer = std::make_unique<ChipSwimlaneAicoreTaskBuffer>();
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(buffer.get());
    auto task = f.resolve(0);
    task.shell = 7;
    task.local = 2;
    for (int run = 0; run < 2; ++run) {
        buffer->count = 0;
        ChipSwimlaneAicoreLocalState profile{};
        f.cfg[0].profile_head = &head;
        f.cfg[0].profile = &profile;
        ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
        ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
        percounter_commit_task_timing(f.cfg[0]);
        percounter_flush_stream_records(&profile);
        EXPECT_EQ(buffer->count, 1U);
        EXPECT_EQ(buffer->records[0].reg_task_id, 1U);
        EXPECT_EQ(TaskId::to_uint64(buffer->records[0].task_token), TaskId::to_uint64(TaskId::make_sub_task(7, 2)));
    }
}

TEST(PercounterPayload, FullTimingBufferIsPublishedOnceAndReplacementKeepsIdentity) {
    Fixture f;
    auto first = std::make_unique<ChipSwimlaneAicoreTaskBuffer>();
    auto second = std::make_unique<ChipSwimlaneAicoreTaskBuffer>();
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(first.get());
    ChipSwimlaneAicoreLocalState profile{};
    f.cfg[0].profile_head = &head;
    f.cfg[0].profile = &profile;
    auto task = f.resolve(0);
    ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
    for (int i = 0; i < PLATFORM_AICORE_BUFFER_SIZE; ++i) {
        EXPECT_EQ(first->count, 0U);
        ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
        percounter_commit_task_timing(f.cfg[0]);
    }
    EXPECT_EQ(first->count, PLATFORM_AICORE_BUFFER_SIZE);

    // Model rotation and Host reuse of the published buffer. Lane exit must
    // not write its stale cached pointer after ownership was transferred.
    head.current_buf_ptr = reinterpret_cast<uint64_t>(second.get());
    head.current_buf_seq = 1;
    first->count = 17;
    percounter_flush_stream_records(&profile);
    EXPECT_EQ(first->count, 17U);

    ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
    percounter_commit_task_timing(f.cfg[0]);
    EXPECT_EQ(second->count, 0U);
    EXPECT_EQ(second->records[0].reg_task_id, PLATFORM_AICORE_BUFFER_SIZE + 1U);
    EXPECT_EQ(TaskId::to_uint64(second->records[0].task_token), TaskId::to_uint64(TaskId::make_global(0)));
    percounter_flush_stream_records(&profile);
    EXPECT_EQ(second->count, 1U);
    EXPECT_EQ(first->count, 17U);
}
TEST(PercounterPayload, TerminalDeferralKeepsLaneLiveUntilConditionHolds) {
    Fixture f;
    int32_t signal = 0;
    f.fanin[0].count = 0;
    f.counters[0].value = percounter_counter_preset(1);
    auto task = f.resolve(0);
    ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
    execute = [&](int64_t *) {
        f.async_entries[0][0] = {
            reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
        };
        f.async_groups[0].count = 1;
    };
    bool deferred = false;
    ASSERT_TRUE(percounter_run_prepared(f.cfg[0], &f.state[0], task, &f.payloads[0][0], &deferred));
    ASSERT_TRUE(deferred);
    uint64_t progress = 0;
    EXPECT_FALSE(percounter_lane_may_exit(f.cfg[0], &f.state[0], &progress));
    bool settled = false;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_FALSE(settled);
    EXPECT_FALSE(percounter_counter_published(f.counters[0].value));
    signal = 1;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_TRUE(settled);
    EXPECT_TRUE(percounter_counter_published(f.counters[0].value));
    EXPECT_TRUE(percounter_lane_may_exit(f.cfg[0], &f.state[0], &progress));
}

TEST(PercounterPayload, IndependentConditionsSettleOutOfRegistrationOrder) {
    Fixture f;
    std::array<int32_t, 3> signals{};
    int current = 0;
    execute = [&](int64_t *) {
        f.async_entries[0][0] = {
            reinterpret_cast<uint64_t>(&signals[current]), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
        };
        f.async_groups[0].count = 1;
    };
    for (; current < 3; ++current) {
        f.fanin[current].count = 0;
        f.counters[current].value = percounter_counter_preset(1);
        auto task = f.resolve(current);
        ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, &f.payloads[0][0]));
        ASSERT_TRUE(f.run(0, task, &f.payloads[0][0]));
    }
    signals[1] = 1;
    bool settled = false;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    ASSERT_TRUE(settled);
    EXPECT_FALSE(percounter_counter_published(f.counters[0].value));
    EXPECT_TRUE(percounter_counter_published(f.counters[1].value));
    EXPECT_FALSE(percounter_counter_published(f.counters[2].value));
    signals[0] = signals[2] = 1;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_EQ(f.state[0].async_head, 0U);
    EXPECT_EQ(f.control.async_registered, 3U);
    EXPECT_EQ(f.control.async_settled, 3U);
}

}  // namespace

volatile uint8_t *sim_get_reg_base() {
    if (on_poll) on_poll();
    return registers.data();
}
uint32_t sim_get_physical_core_id() { return 0; }

TEST(PercounterPayload, ProfilingLevelsReserveOnlyTheirRecordTails) {
    const uint64_t counts[2]{16, 0};
    std::array<PercounterLayout, 5> layouts{};
    for (uint32_t level = 0; level < layouts.size(); ++level)
        ASSERT_TRUE(plan_layout(1, 16, 0, counts, &layouts[level], 0, 0, {}, 0, level));
    EXPECT_EQ(layouts[0].total_size, layouts[1].total_size);
    EXPECT_EQ(layouts[1].profile_buffers_offset, 0);
    EXPECT_NE(layouts[2].profile_buffers_offset, 0);
    EXPECT_NE(layouts[2].task_records_offset, 0);
    EXPECT_EQ(layouts[2].scalar_records_offset, 0);
    EXPECT_EQ(layouts[3].upload_size, layouts[2].upload_size);
    EXPECT_EQ(layouts[3].total_size, layouts[4].total_size);
    EXPECT_GT(layouts[3].scalar_records_offset, layouts[3].upload_size);
    EXPECT_EQ(
        layouts[3].total_size - layouts[3].scalar_records_offset,
        SCHEDULER_WORKER_CAPACITY * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord)
    );
}

TEST(PercounterPayload, TaskTimingPartitionsOwnWholeCacheLines) {
    PercounterLayout layout{};
    const uint64_t counts[2]{3, 0};
    ASSERT_TRUE(plan_layout(1, 3, 0, counts, &layout, 0, 0, {}, 0, 2));
    std::vector<uint8_t> storage(layout.upload_size + 127);
    const auto address = (reinterpret_cast<uintptr_t>(storage.data()) + 127) & ~uintptr_t{127};
    auto *base = reinterpret_cast<void *>(address);
    TaskInput task{};
    task.block_num = 3;
    task.subtask_count = 1;
    ASSERT_TRUE(build_tables(&task, 1, layout, base, address, {}));
    const auto *buffers = region_at<PercounterProfileBuffer>(base, layout.profile_buffers_offset);
    for (uint32_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        EXPECT_EQ(buffers[worker].task_records_offset % 64, 0) << worker;
        if (worker + 1 < SCHEDULER_WORKER_CAPACITY)
            EXPECT_LE(
                buffers[worker].task_records_offset + 3 * sizeof(PercounterTaskTiming),
                buffers[worker + 1].task_records_offset
            );
    }
}

TEST(PercounterPayload, DeferredTimingDoesNotNeedStreamingBuffers) {
    Fixture f;
    f.cfg[0].order_count = 1;
    f.fanin[0].count = 0;
    PercounterProfileBuffer header{};
    header.task_capacity = 1;
    std::array<PercounterTaskTiming, 1> records{};
    uint32_t count = 0, dropped = 0;
    f.cfg[0].profile_buffer = &header;
    f.cfg[0].task_records = records.data();
    f.cfg[0].profile_task_count = &count;
    f.cfg[0].task_dropped = &dropped;
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
        EXPECT_EQ(count, 0U);
    };
    percounter_lane_loop(f.cfg[0], budget);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(percounter_counter_published(f.counters[0].value));
    EXPECT_EQ(f.control.lane_error, 0U);
    EXPECT_EQ(count, 1U);
    EXPECT_EQ(dropped, 0U);
    EXPECT_GT(records[0].dispatch, 0U);
    EXPECT_GE(records[0].finish, records[0].dispatch);
}

TEST(PercounterPayload, ExhaustedTimingStorageDoesNotBlockCompletion) {
    Fixture f;
    f.cfg[0].order_count = 1;
    f.fanin[0].count = 0;
    PercounterProfileBuffer header{};
    uint32_t count = 0, dropped = 0;
    f.cfg[0].profile_buffer = &header;
    f.cfg[0].profile_task_count = &count;
    f.cfg[0].task_dropped = &dropped;
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
    };
    percounter_lane_loop(f.cfg[0], budget);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(percounter_counter_published(f.counters[0].value));
    EXPECT_EQ(f.control.lane_error, 0U);
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(dropped, 1U);
}

TEST(PercounterPayload, TimingCommitFollowsCompletionAndKeepsCapturedIdentity) {
    Fixture f;
    PercounterProfileBuffer header{};
    header.task_capacity = 1;
    std::array<PercounterTaskTiming, 1> records{};
    uint32_t count = 0, dropped = 0;
    f.cfg[0].profile_buffer = &header;
    f.cfg[0].task_records = records.data();
    f.cfg[0].profile_task_count = &count;
    f.cfg[0].task_dropped = &dropped;
    auto task = f.resolve(0);
    task.shell = 7;
    task.local = 2;
    auto *payload = &f.payloads[0][0];
    ASSERT_TRUE(percounter_prepare_task(f.cfg[0], task, 0, 1, payload));
    ASSERT_TRUE(f.run(0, task, payload));
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(records[0].dispatch, 0U);
    percounter_publish_completion(f.cfg[0], task, 1);
    EXPECT_TRUE(percounter_counter_published(f.counters[0].value));
    EXPECT_EQ(count, 0U);
    task.id = 99;
    task.shell = -1;
    percounter_commit_task_timing(f.cfg[0]);
    EXPECT_EQ(count, 1U);
    EXPECT_EQ(TaskId::to_uint64(records[0].task_token), TaskId::to_uint64(TaskId::make_sub_task(7, 2)));
    EXPECT_LE(records[0].dispatch, records[0].start);
    EXPECT_LE(records[0].start, records[0].finish);
    percounter_commit_task_timing(f.cfg[0]);
    EXPECT_EQ(count, 1U);
}
