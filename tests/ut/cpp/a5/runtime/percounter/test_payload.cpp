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
#include "graph_template.h"
#include "scheduler/argument_metadata.h"

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
alignas(64) std::array<uint8_t, 4096> registers{};
std::function<void(int64_t *)> execute;
std::function<void()> on_poll;
thread_local void *current_context = nullptr;
int bind_context(void *context) {
    current_context = context;
    return 0;
}
void *read_context() { return current_context; }
alignas(64) std::array<uint8_t, PLATFORM_SSBUF_SIZE> startup_ssbuf{};
void kernel(int64_t *args) { execute(args); }

TaskInput task(uint32_t mask, int blocks = 1, bool sync = false) {
    TaskInput result{};
    result.active_mask = mask;
    result.is_mix = __builtin_popcount(mask) > 1;
    result.core_type_index = mask == 1 ? 0 : 1;
    result.subtask_count = __builtin_popcount(mask);
    result.block_num = blocks;
    result.sync_start = sync;
    return result;
}

struct Fixture {
    std::vector<TaskInput> inputs;
    std::vector<std::vector<int32_t>> folded_rows;
    std::vector<ChipTaskStorage> storage;
    std::vector<uint64_t> scalars;
    std::vector<uint8_t> region;
    PercounterLayout layout{};
    PercounterControl *control;
    PercounterCounter *counters;
    std::array<uint64_t, 2> callable{0, reinterpret_cast<uint64_t>(&kernel)};
    std::vector<DispatchPayload> arguments;
    std::array<PercounterSsbufRegion, 2> signals{};
    std::array<LaneConfig, 6> cfg{};
    std::array<PercounterLaneState, 6> state{};
    std::array<pto2::resident::AsyncGroupData, 6> async_groups{};
    std::array<std::array<DeferredCompletionEntry, PERCOUNTER_ASYNC_CAPACITY>, 6> async_entries{};
    std::array<PercounterAsyncCompletion, 128> async_completions{};
    std::vector<std::array<int64_t, 3>> calls;
    int running_lane{-1};

    explicit Fixture(
        std::vector<TaskInput> in = {task(1), task(1), task(1)}, uint32_t clusters = 2,
        const ExternalAddresses &external = {}, bool fold_dependencies = false
    ) :
        inputs(std::move(in)),
        storage(inputs.size()),
        scalars(inputs.size()),
        arguments(inputs.size()) {
        simpler_sim_install_invocation_context_hooks(bind_context, read_context);
        registers.fill(0);
        on_poll = {};
        if (fold_dependencies) {
            std::vector<int64_t> folded;
            EXPECT_TRUE(percounter_fold_dependency_tasks(inputs, folded_rows, folded));
        }
        uint64_t counts[2]{};
        uint64_t mix = 0, spmd = 0, sync = 0, blocks = 0, edges = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto &t = inputs[i];
            blocks += t.block_num;
            edges += t.fanin_count;
            if (!t.inline_completed && !t.published_off_lane) {
                if (t.is_mix) ++mix;
                ++counts[t.is_mix ? 1 : t.core_type_index];
                spmd += t.block_num > 1 || t.is_mix;
                sync += t.sync_start;
            }
            scalars[i] = i;
            storage[i].task.task_id = TaskId::make_global(i);
            storage[i].payload.scalar_count = 1;
            storage[i].payload.scalars.set(&scalars[i]);
            for (int lane = 0; lane < 3; ++lane)
                storage[i].task.kernel_id[lane] = (t.active_mask & (1U << lane)) ? 1 : -1;
        }
        EXPECT_TRUE(plan_layout(inputs.size(), blocks, edges, counts, &layout, mix, spmd, sync));
        region.resize(layout.total_size + 128);
        auto *base = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(region.data()) + 127) & ~uintptr_t{127});
        EXPECT_TRUE(build_tables(
            inputs.data(), inputs.size(), layout, base, reinterpret_cast<uint64_t>(base), clusters, external
        ));
        control = reinterpret_cast<PercounterControl *>(base);
        counters = region_at<PercounterCounter>(base, layout.counters_offset);
        for (size_t i = 0; i < inputs.size(); ++i)
            EXPECT_EQ(publish_arguments(i), SchedulerGraphResult::OK);
        for (int i = 0; i < 6; ++i) {
            auto &c = cfg[i];
            c.control = control;
            c.storage_address = reinterpret_cast<uint64_t>(storage.data());
            c.arguments = arguments.data();
            c.async_group = &async_groups[i];
            c.async_entries = async_entries[i].data();
            c.async_completions = async_completions.data();
            c.async_completion_capacity = async_completions.size();
            c.async_group->capacity = PERCOUNTER_ASYNC_CAPACITY;
            c.async_group->run_epoch = control->async_run_epoch;
            c.task_count = c.table_task_count = inputs.size();
            c.fanin = region_at<PercounterFanin>(base, layout.fanin_offset);
            c.fanin_addresses = region_at<uint64_t>(base, layout.fanin_addr_offset);
            c.counters = counters;
            c.tickets = region_at<PercounterTicket>(base, layout.tickets_offset);
            c.metadata = region_at<PercounterTaskMetadata>(base, layout.metadata_offset);
            c.metadata_ready = region_at<uint64_t>(base, layout.metadata_ready_offset);
            c.segments = region_at<PercounterSegment>(base, layout.segments_offset);
            c.task_claims = region_at<PercounterTaskClaim>(base, layout.task_claims_offset);
            c.claim_summary = region_at<PercounterClaimSummary>(base, layout.claim_summary_offset);
            c.sync_arrivals = region_at<PercounterStartBarrier>(base, layout.sync_arrivals_offset);
            c.cluster_index = i / 3;
            c.cluster_lane = i % 3;
            c.cluster_count = clusters;
            c.aiv_per_cluster = 2;
            const int type = c.cluster_lane == 0 ? 0 : 1;
            c.order = region_at<uint64_t>(base, layout.orders_offset[type]);
            c.ssbuf = &signals[i / 3];
            c.mix_task_count = layout.mix_task_count;
            for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
                c.segment_begin[l] = control->segment_begin[l];
                c.segment_count[l] = control->segment_count[l];
            }
            percounter_initialize_boundaries(c, &state[i].source);
        }
        execute = [this](int64_t *args) {
            calls.push_back({args[0], get_block_idx(args), running_lane});
        };
    }
    SchedulerGraphResult publish_arguments(size_t id) {
        const auto &payload = storage[id].payload;
        uint64_t functions[3]{};
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const auto kernel_id = storage[id].task.kernel_id[lane];
            if (kernel_id >= 0 && static_cast<size_t>(kernel_id) < callable.size())
                functions[lane] = callable[kernel_id];
        }
        const auto status = percounter_prepare_argument_metadata(
            counters[id], functions, inputs[id].active_mask, reinterpret_cast<uint64_t>(&arguments[id]),
            payload.tensor_count, payload.scalar_count
        );
        if (status != SchedulerGraphResult::OK) return status;
        for (int32_t i = 0; i < payload.tensor_count; ++i)
            arguments[id].args[i] = reinterpret_cast<uint64_t>(&payload.tensor_data()[i]);
        for (int32_t i = 0; i < payload.scalar_count; ++i)
            arguments[id].args[payload.tensor_count + i] = payload.scalar_data()[i];
        return status;
    }
    uint64_t claim_total(uint32_t index) const {
        uint64_t total = 0;
        for (const auto &sector : cfg[0].task_claims[index].sectors)
            total += sector.next;
        return total;
    }
    bool completed(size_t id) const {
        if (counters[id].completion_notify_address != 0)
            return *reinterpret_cast<const uint32_t *>(counters[id].completion_notify_address) >=
                   counters[id].completion_notify_target;
        return percounter_counter_published(counters[id].value);
    }
    void expect_completed(size_t id) const {
        if (counters[id].completion_notify_address != 0) {
            EXPECT_EQ(
                *reinterpret_cast<const uint32_t *>(counters[id].completion_notify_address),
                counters[id].completion_notify_target
            ) << id;
            return;
        }
        EXPECT_EQ(counters[id].value, PERCOUNTER_COUNTER_PUBLISHED) << id;
    }
    bool run(int lane, const PercounterTask &task, int block = 0) {
        running_lane = lane;
        percounter_bind_sim_context(&signals[lane / 3].contexts[lane % 3].value);
        bool deferred = false;
        return percounter_run_task(
            cfg[lane], &state[lane], task, block, inputs[task.id].block_num, 0,
            percounter_completion_address(cfg[lane], task.id), &deferred,
            percounter_order_entry(task.id, block, inputs[task.id].block_num)
        );
    }
    PercounterTask resolve(int id) {
        PercounterTask result{};
        EXPECT_TRUE(percounter_resolve_dispatch(cfg[0], id, cfg[0].metadata[id].active_mask, &result));
        return result;
    }
    void step(int lane) {
        running_lane = lane;
        percounter_bind_sim_context(&signals[lane / 3].contexts[lane % 3].value);
        bool executed;
        ASSERT_TRUE(percounter_step(cfg[lane], &state[lane], &executed));
        ASSERT_EQ(control->lane_error, 0U);
    }
    // What the AICPU supervisor watches: every top-level sink published.
    bool sinks_complete() const {
        const auto *sinks =
            reinterpret_cast<const uint64_t *>(reinterpret_cast<const uint8_t *>(control) + layout.sinks_offset);
        for (uint64_t i = 0; i < control->sink_count; ++i)
            if (!completed(sinks[i])) return false;
        return true;
    }
    // Steps every lane until `done`, which is the event the case is about:
    // whole-graph completion now waits on deferred settlement, so a case that
    // leaves a condition unmet asks for the physical work it does expect.
    template <typename Predicate>
    void drain_until(Predicate done, const char *expected) {
        for (int round = 0; round < 1000; ++round) {
            for (int lane = 0; lane < 6; ++lane)
                step(lane);
            if (done()) return;
        }
        FAIL() << "finite protocol did not reach " << expected;
    }
    void drain() {
        drain_until(
            [this] {
                return sinks_complete();
            },
            "every top-level sink complete"
        );
    }
};

TEST(PercounterResourceOrder, NarrowReadySyncUsesCapacityWhileWiderSyncWaits) {
    std::vector<ResourceOrderTask> tasks(3);
    for (auto &node : tasks)
        node.active_mask = 2;
    tasks[0].blocks = 2;
    tasks[1].blocks = 3;
    tasks[1].sync = true;
    tasks[2].blocks = 2;
    tasks[2].sync = true;
    const auto plan = make_resource_order(tasks, 2);
    ASSERT_TRUE(plan.has_value());
    const std::vector<uint32_t> order{0, 2, 1};
    EXPECT_EQ(plan->tasks, order);
    EXPECT_EQ(plan->first_start[2], 0U);
    EXPECT_EQ(plan->first_start[1], 1U);
    EXPECT_EQ(plan->duration, 2U);
}

TEST(PercounterResourceOrder, MixUsesOnlyItsActualParticipants) {
    for (uint8_t mix : {uint8_t{5}, uint8_t{6}}) {
        std::vector<ResourceOrderTask> tasks(2);
        tasks[0].active_mask = mix == 5 ? 2 : 1;
        tasks[1].active_mask = mix;
        const auto plan = make_resource_order(tasks, 1);
        ASSERT_TRUE(plan.has_value());
        EXPECT_EQ(plan->first_start[0], 0U);
        EXPECT_EQ(plan->first_start[1], 0U);
        EXPECT_EQ(plan->duration, 1U);
    }
}

TEST(PercounterResourceOrder, GraphEnvelopeReservesUsedTypesAndReleasesSuccessors) {
    std::vector<ResourceOrderTask> tasks(3);
    tasks[0].active_mask = 1;
    tasks[1].active_mask = 2;
    tasks[1].reserve_types = true;
    tasks[1].duration = tasks[1].depth = 3;
    tasks[2].active_mask = 1;
    tasks[2].predecessors = {1};
    const auto vector_body = make_resource_order(tasks, 2);
    ASSERT_TRUE(vector_body.has_value());
    EXPECT_EQ(vector_body->first_start[0], 0U);
    EXPECT_EQ(vector_body->first_start[1], 0U);
    EXPECT_EQ(vector_body->first_start[2], 3U);
    EXPECT_EQ(vector_body->duration, 4U);
    tasks[1].active_mask = 3;
    const auto mixed_body = make_resource_order(tasks, 2);
    ASSERT_TRUE(mixed_body.has_value());
    EXPECT_EQ(mixed_body->first_start[0], 3U);
    EXPECT_EQ(mixed_body->first_start[1], 0U);
    EXPECT_EQ(mixed_body->first_start[2], 3U);
    EXPECT_EQ(mixed_body->duration, 4U);
}

TEST(PercounterResourceOrder, DeclaredTopologyControlsVirtualCapacity) {
    std::vector<ResourceOrderTask> tasks(2);
    tasks[0].active_mask = tasks[1].active_mask = 1;
    tasks[0].blocks = 2;
    for (uint32_t clusters = 1; clusters <= 3; ++clusters) {
        const auto plan = make_resource_order(tasks, clusters);
        ASSERT_TRUE(plan.has_value());
        EXPECT_EQ(plan->first_start[1], 3U - clusters);
    }
}

TEST(PercounterResourceOrder, RejectsInvalidTopologyEdgesAndTimeOverflow) {
    std::vector<ResourceOrderTask> tasks(2);
    tasks[0].active_mask = tasks[1].active_mask = 1;
    EXPECT_FALSE(make_resource_order(tasks, 0).has_value());
    tasks[0].predecessors = {1};
    EXPECT_FALSE(make_resource_order(tasks, 2).has_value());
    tasks[0].predecessors.clear();
    tasks[0].sync = true;
    tasks[0].blocks = 3;
    EXPECT_FALSE(make_resource_order(tasks, 2).has_value());
    tasks[0].sync = false;
    tasks[0].blocks = 1;
    tasks[0].reserve_types = true;
    tasks[0].duration = UINT64_MAX;
    tasks[1].predecessors = {0};
    EXPECT_FALSE(make_resource_order(tasks, 2).has_value());
}

TEST(PercounterResourceOrder, EverySmallDagKeepsEdgesAndWholeSyncStarts) {
    const std::array<uint8_t, 6> masks{{1, 2, 3, 5, 6, 7}};
    for (uint32_t clusters : {1U, 2U}) {
        for (uint32_t graph = 0; graph < 64; ++graph) {
            for (uint32_t shape = 0; shape < masks.size(); ++shape) {
                std::vector<ResourceOrderTask> tasks(4);
                uint32_t edge = 0;
                for (uint32_t id = 0; id < tasks.size(); ++id) {
                    tasks[id].active_mask = masks[(shape + id) % masks.size()];
                    tasks[id].blocks = id % 2 + 1;
                    tasks[id].sync = clusters == 2 && id % 2 != 0;
                    for (uint32_t producer = 0; producer < id; ++producer, ++edge)
                        if (graph & (1U << edge)) tasks[id].predecessors.push_back(producer);
                }
                const auto plan = make_resource_order(tasks, clusters);
                ASSERT_TRUE(plan.has_value());
                ASSERT_EQ(plan->tasks.size(), tasks.size());
                std::array<uint32_t, 4> rank{};
                uint32_t seen = 0;
                for (uint32_t position = 0; position < plan->tasks.size(); ++position) {
                    ASSERT_LT(plan->tasks[position], rank.size());
                    ASSERT_EQ(seen & (1U << plan->tasks[position]), 0U);
                    seen |= 1U << plan->tasks[position];
                    rank[plan->tasks[position]] = position;
                }
                EXPECT_EQ(seen, 15U);
                for (uint32_t id = 0; id < tasks.size(); ++id) {
                    EXPECT_GT(plan->finish[id], plan->first_start[id]);
                    if (tasks[id].sync) EXPECT_EQ(plan->finish[id], plan->first_start[id] + 1);
                    for (uint32_t producer : tasks[id].predecessors) {
                        EXPECT_LT(rank[producer], rank[id]);
                        EXPECT_LE(plan->finish[producer], plan->first_start[id]);
                    }
                }
            }
        }
    }
}

TEST(PercounterAcquire, PartialMixClaimsCoverEveryBlockOnceAcrossActualTopologies) {
    for (uint32_t width = 2; width <= 32; ++width) {
        for (uint32_t blocks = 2; blocks <= width; ++blocks) {
            for (uint32_t rank : {0U, width - 1, UINT32_MAX}) {
                Fixture f({task(7, blocks, true)}, width);
                f.cfg[1].metadata[0].global_order_id = rank;
                std::vector<PercounterLaneState> states(width);
                std::vector<PercounterSsbufRegion> signals(width);
                std::vector<uint32_t> owners(blocks);
                auto cfg = f.cfg[1];
                for (uint32_t cluster = width; cluster-- > 0;) {
                    cfg.cluster_index = cluster;
                    cfg.ssbuf = &signals[cluster];
                    percounter_claim(cfg, &states[cluster]);
                    const uint32_t owned = states[cluster].mix_owned;
                    EXPECT_LE(__builtin_popcount(owned), 1);
                    if (owned == 0) continue;
                    const uint32_t slot = __builtin_ctz(owned);
                    const uint32_t block = percounter_order_block_idx(signals[cluster].mix[slot].entry);
                    ASSERT_LT(block, blocks);
                    ++owners[block];
                    percounter_claim(cfg, &states[cluster]);
                    EXPECT_EQ(states[cluster].mix_owned, owned);
                }
                for (uint32_t count : owners)
                    EXPECT_EQ(count, 1U);
                // Tickets account for every block the ticket path allocates. A
                // sync boundary sized to its owner count allocates by rank and
                // draws none, and records no exhaustion either: there the mask
                // could only say the highest ranks had claimed, which is not
                // the same as the boundary being finished.
                const bool statically_assigned = blocks == width;
                EXPECT_EQ(f.claim_total(0), statically_assigned ? 0U : blocks);
                EXPECT_EQ(percounter_claim_exhausted(percounter_claim_closed(f.cfg[1], 0)), !statically_assigned);
            }
        }
    }
}

TEST(PercounterAcquire, ClaimPartitionsCoverRaggedTasksAndCloseOnce) {
    for (uint32_t blocks = 1; blocks <= 2 * PERCOUNTER_CLAIM_SECTOR_COUNT + 1; ++blocks) {
        for (uint32_t owners : {1U, 2U, 3U, 5U}) {
            Fixture f({task(7, blocks)});
            auto cfg = f.cfg[1];
            const uint32_t segment = cfg.segment_begin[1];
            const uint64_t initial_closed = cfg.claim_summary[segment / 64].closed;
            std::vector<uint32_t> seen(blocks);
            uint32_t claimed = 0;
            for (uint32_t round = 0; round <= blocks; ++round) {
                for (uint32_t owner = 0; owner < owners; ++owner) {
                    cfg.cluster_index = owner;
                    const uint32_t block = percounter_claim_block(cfg, 0, blocks);
                    if (block < blocks) {
                        EXPECT_EQ(++seen[block], 1U);
                        ++claimed;
                    }
                    // Every lane skipping a closed boundary offers to publish
                    // it; repeating that must still add exactly one bit.
                    const uint64_t closed = percounter_claim_closed(cfg, 0);
                    if (percounter_claim_exhausted(closed)) percounter_publish_segment(cfg, 0, closed, segment);
                }
            }
            EXPECT_EQ(claimed, blocks);
            EXPECT_TRUE(percounter_claim_exhausted(percounter_claim_closed(cfg, 0)));
            EXPECT_EQ(
                cfg.task_claims[0].closed_sectors, PERCOUNTER_CLAIM_SECTOR_MASK | PERCOUNTER_CLAIM_SUMMARY_PUBLISHED
            );
            EXPECT_EQ(cfg.claim_summary[segment / 64].closed, initial_closed | (uint64_t{1} << (segment % 64)));
        }
    }
}

TEST(PercounterAcquire, FaninObserverDoesNotWaitForPeerDiscoveryButEntryDoes) {
    Fixture f({task(7)});
    f.step(1);
    auto &record = f.signals[0].mix[0];
    ASSERT_EQ(record.generation, 1U);
    EXPECT_EQ(record.fanin_ready, 0U);
    f.step(0);
    EXPECT_EQ(record.fanin_ready, 1U);
    EXPECT_EQ(record.enter_ready[0], 1U);
    EXPECT_EQ(record.enter_ready[2], 0U);
    EXPECT_TRUE(f.calls.empty());
    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    f.step(2);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(record.enter_idle[2], 1U);
    EXPECT_EQ(record.enter_idle[0], 0U);
    f.step(0);
    f.step(1);
    EXPECT_EQ(f.calls.size(), 1U);
    f.drain();
    EXPECT_EQ(f.calls.size(), 3U);
    f.expect_completed(0);
}

TEST(PercounterAcquire, RuntimeFaninSkipsOnlyHostCompletedProducers) {
    auto host_completed = task(0);
    host_completed.inline_completed = true;
    host_completed.subtask_count = 1;
    auto off_lane = task(1);
    off_lane.published_off_lane = true;
    const int32_t producers[]{0, 1};
    auto mixed_consumer = task(2);
    mixed_consumer.fanin_ids = producers;
    mixed_consumer.fanin_count = 2;
    auto ready_consumer = task(1);
    ready_consumer.fanin_ids = producers;
    ready_consumer.fanin_count = 1;
    Fixture f({host_completed, off_lane, mixed_consumer, ready_consumer});

    const auto remaining = percounter_fanin_view(f.cfg[1], 2);
    ASSERT_EQ(remaining.count, 1);
    EXPECT_EQ(remaining.addresses[0], reinterpret_cast<uint64_t>(&f.counters[1].value));
    EXPECT_EQ(percounter_fanin_view(f.cfg[0], 3).count, 0);

    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    f.step(0);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls[0][0], 3);
    f.counters[1].value = PERCOUNTER_COUNTER_PUBLISHED;
    f.drain();
    ASSERT_EQ(f.calls.size(), 2U);
    EXPECT_EQ(f.calls[1][0], 2);
    f.expect_completed(2);
    f.expect_completed(3);
}

TEST(PercounterAcquire, FaninCursorWaitsForEachProducerBeforePublishingReady) {
    for (const uint32_t mask : {1U, 6U, 7U}) {
        auto producer = task(1);
        producer.published_off_lane = true;
        auto consumer = task(mask);
        const int32_t roots[] = {0, 1};
        consumer.fanin_count = 2;
        consumer.fanin_ids = roots;
        Fixture f({producer, producer, consumer});
        for (int round = 0; round < 3; ++round)
            for (int lane = 0; lane < 6; ++lane)
                f.step(lane);
        EXPECT_TRUE(f.calls.empty());
        f.counters[0].value = 1;
        for (int round = 0; round < 3; ++round)
            for (int lane = 0; lane < 6; ++lane)
                f.step(lane);
        EXPECT_TRUE(f.calls.empty());
        f.counters[1].value = 1;
        f.drain();
        EXPECT_EQ(f.calls.size(), static_cast<size_t>(__builtin_popcount(mask)));
        f.expect_completed(2);
    }
}

TEST(PercounterAcquire, MixCannotStartWhileCommittedPeerRunsAnOrdinaryKernel) {
    for (bool sync : {false, true}) {
        Fixture f({task(7, sync ? 2 : 1, sync), task(2)});
        percounter_claim(f.cfg[2], &f.state[2]);
        bool ordinary_running = false;
        bool served_ordinary = false;
        const auto record_call = execute;
        execute = [&](int64_t *args) {
            if (args[0] == 0 && f.running_lane < 3) EXPECT_FALSE(ordinary_running);
            record_call(args);
            if (args[0] != 1) return;
            served_ordinary = true;
            ordinary_running = true;
            const int owner = f.running_lane;
            // The remaining peers can become ready while this kernel still
            // owns its UB and pipelines. ENTER_READY alone cannot release them.
            f.step(1);
            f.step(0);
            if (sync) {
                f.step(4);
                f.step(3);
                f.step(4);
                f.step(5);
                f.step(3);
                f.step(1);
                f.step(0);
            }
            f.running_lane = owner;
            ordinary_running = false;
        };
        f.step(1);
        f.step(0);
        for (int pass = 0; pass < 64 && !served_ordinary; ++pass)
            f.step(2);
        EXPECT_TRUE(served_ordinary);
        f.drain();
        EXPECT_EQ(f.calls.size(), sync ? 7U : 4U);
        f.expect_completed(0);
    }
}

TEST(PercounterAcquire, LargerOrdinaryPassesUnreadyCohort) {
    int32_t producer = 0;
    auto external = task(1);
    external.published_off_lane = true;
    auto mix = task(7);
    mix.fanin_count = 1;
    mix.fanin_ids = &producer;
    Fixture f({external, mix, task(7), task(2)});
    for (int r = 0; r < 8; ++r)
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls[0][0], 3);
    f.counters[0].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 7U);
}

TEST(PercounterAcquire, SmallerReadyOrdinaryPrecedesNewCohortCommitment) {
    Fixture f({task(1), task(7)});
    percounter_claim(f.cfg[0], &f.state[0]);
    f.step(1);
    f.step(0);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls.front()[0], 0);
    EXPECT_EQ(f.signals[0].mix[0].enter_ready[0], 0U);
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, CommittedCohortRunsWhenGoEvenIfEarlierOrdinaryBecomesReady) {
    auto producer = task(2);
    producer.published_off_lane = true;
    auto ordinary = task(1);
    const int32_t root = 0;
    ordinary.fanin_count = 1;
    ordinary.fanin_ids = &root;
    Fixture f({producer, ordinary, task(7)});
    f.step(0);
    f.step(1);
    f.step(0);
    auto &record = f.signals[0].mix[0];
    for (int retry = 0; retry < 3 && record.enter_ready[0] == 0; ++retry)
        f.step(0);
    ASSERT_EQ(record.enter_ready[0], 1U);
    for (int retry = 0; retry < 3 && record.enter_ready[2] == 0; ++retry)
        f.step(2);
    ASSERT_EQ(record.enter_ready[2], 1U);
    f.step(1);
    f.step(2);
    f.counters[0].value = 1;
    const size_t before = f.calls.size();
    f.step(0);
    ASSERT_EQ(f.calls.size(), before + 1);
    EXPECT_EQ(f.calls.back()[0], 2);
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, CompletedEvenBlocksDoNotReleaseConsumerBeforeOddBlock) {
    int32_t producer = 0;
    auto consumer = task(2);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({task(1, 3), consumer});
    percounter_claim(f.cfg[0], &f.state[0]);
    percounter_claim(f.cfg[3], &f.state[3]);
    f.step(0);
    f.step(0);
    ASSERT_EQ(f.calls.size(), 2U);
    EXPECT_EQ(f.calls[0][1], 0);
    EXPECT_EQ(f.calls[1][1], 2);
    EXPECT_EQ(f.counters[0].value, 0U);
    for (int tick = 0; tick < 4; ++tick)
        f.step(1);
    EXPECT_EQ(f.calls.size(), 2U);
    EXPECT_EQ(f.counters[1].value, percounter_counter_preset(1));
    f.step(3);
    ASSERT_EQ(f.calls.size(), 3U);
    EXPECT_EQ(f.calls.back()[1], 1);
    f.step(1);
    ASSERT_EQ(f.calls.size(), 4U);
    EXPECT_EQ(f.calls.back()[0], 1);
    EXPECT_EQ(f.counters[1].value, PERCOUNTER_COUNTER_PUBLISHED);
}

TEST(PercounterAcquire, SingleGmCounterReleasesFaninAfterEveryBlock) {
    constexpr uint32_t blocks = 5;
    Fixture f({task(1, blocks)});
    const auto producer = f.resolve(0);
    const uint64_t reference = reinterpret_cast<uint64_t>(&f.counters[0].value);
    uint64_t pending = 0, target = 0;
    EXPECT_FALSE(percounter_probe_completion(reference, &pending, &target));
    EXPECT_EQ(pending, reference);
    for (uint32_t block = 0; block < blocks; ++block) {
        percounter_publish_completion(
            f.cfg[0], producer, percounter_completion_address(f.cfg[0], 0), 1, percounter_order_entry(0, block, blocks)
        );
        EXPECT_EQ(percounter_probe_completion(0, &pending, &target), block + 1 == blocks);
    }
    EXPECT_EQ(pending, 0U);
    EXPECT_EQ(target, 0U);
    f.expect_completed(0);
}

TEST(PercounterAcquire, CompletedOddBlockDoesNotReleaseConsumerBeforeEvenBlocks) {
    int32_t producer = 0;
    auto consumer = task(2);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({task(1, 3), consumer});
    percounter_claim(f.cfg[0], &f.state[0]);
    percounter_claim(f.cfg[3], &f.state[3]);
    f.step(3);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls.back()[1], 1);
    EXPECT_EQ(f.counters[0].value, percounter_counter_preset(2));
    for (int tick = 0; tick < 4; ++tick)
        f.step(1);
    EXPECT_EQ(f.calls.size(), 1U);
    f.step(0);
    f.step(1);
    EXPECT_EQ(f.calls.size(), 2U);
    f.step(0);
    f.step(1);
    ASSERT_EQ(f.calls.size(), 4U);
    EXPECT_EQ(f.calls.back()[0], 1);
    f.expect_completed(0);
    f.expect_completed(1);
}

TEST(PercounterAcquire, DeferredOddBlockRetainsItsCounterAddressAfterPhysicalRetirement) {
    int32_t producer = 0;
    auto consumer = task(2);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({task(1, 2), consumer, task(1)});
    int32_t peer_signal = 0;
    execute = [&](int64_t *args) {
        const auto *local = &percounter_current_context()->local;
        f.calls.push_back({args[0], local->block_idx, f.running_lane});
        if (args[0] == 0 && local->block_idx == 1) {
            const int lane = f.running_lane;
            f.async_entries[lane][0] = {
                reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            f.async_groups[lane].count = 1;
        }
    };
    percounter_claim(f.cfg[0], &f.state[0]);
    percounter_claim(f.cfg[3], &f.state[3]);
    f.step(0);
    f.step(3);
    EXPECT_EQ(f.state[3].occupied, 0U);
    ASSERT_NE(f.state[3].async_head, 0U);
    EXPECT_EQ(
        f.async_completions[f.state[3].async_head - 1].completion_address,
        reinterpret_cast<uint64_t>(&f.counters[0].value)
    );
    EXPECT_EQ(f.counters[0].value, 0U);
    f.step(3);
    f.step(1);
    ASSERT_EQ(f.calls.size(), 3U);
    EXPECT_EQ(f.calls.back()[0], 2);
    EXPECT_FALSE(f.completed(0));
    peer_signal = 1;
    f.step(3);
    f.step(1);
    ASSERT_EQ(f.calls.size(), 4U);
    EXPECT_EQ(f.calls.back()[0], 1);
    f.expect_completed(0);
    EXPECT_EQ(f.control->async_registered, f.control->async_settled);
}

TEST(PercounterAcquire, DeferredOddMixCompletionOutlivesSharedRecordReuse) {
    for (uint32_t mask : {3U, 5U, 6U, 7U}) {
        Fixture f({task(mask, 3), task(mask, 3)});
        int32_t peer_signal = 0;
        execute = [&](int64_t *args) {
            const auto *local = &percounter_current_context()->local;
            f.calls.push_back({args[0], local->block_idx, f.running_lane});
            if (args[0] == 0 && local->block_idx == 1 && f.running_lane % 3 == __builtin_ctz(mask)) {
                const int lane = f.running_lane;
                f.async_entries[lane][0] = {
                    reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                };
                f.async_groups[lane].count = 1;
            }
        };
        const size_t executions = 6U * __builtin_popcount(mask);
        // Task 1 completes while task 0 still lacks its deferred credit.
        f.drain_until(
            [&] {
                return f.completed(1) && f.counters[0].value == 0U;
            },
            "the undeferred sink and all immediate credits"
        );
        EXPECT_EQ(f.calls.size(), executions);
        EXPECT_EQ(f.counters[0].value, 0U);
        EXPECT_FALSE(f.completed(0)) << mask;
        f.expect_completed(1);
        peer_signal = 1;
        for (int lane = 0; lane < 6; ++lane)
            f.step(lane);
        f.expect_completed(0);
        EXPECT_EQ(f.control->async_registered, f.control->async_settled);
    }
}

TEST(PercounterAcquire, CntNotifyDoesNotReleaseFaninBeforeTheDeferredSubtask) {
    for (const uint32_t mask : {2U, 3U, 5U, 6U, 7U}) {
        alignas(32) uint32_t completion = 0;
        const uint64_t address = reinterpret_cast<uint64_t>(&completion);
        const int32_t producer = 0;
        auto consumer = task(2);
        consumer.fanin_count = 1;
        consumer.fanin_ids = &producer;
        std::vector<TaskInput> inputs{task(mask, 3), consumer};
        SchedulerCompletionCounterPlan plan;
        std::string error;
        ASSERT_TRUE(plan_percounter_completion_counters(inputs, &plan, &error)) << error;
        ExternalAddresses external{};
        external.completion_plan = &plan;
        external.completion_notify = &address;
        external.completion_notify_count = 1;
        Fixture f(inputs, 2, external);
        const uint32_t expected = 3U * __builtin_popcount(mask);
        ASSERT_EQ(f.counters[0].completion_notify_target, expected);
        ASSERT_EQ(f.cfg[1].fanin_addresses[0] & PERCOUNTER_COMPLETION_NOTIFY, PERCOUNTER_COMPLETION_NOTIFY);
        int32_t peer_signal = 0;
        execute = [&](int64_t *args) {
            const auto *local = &percounter_current_context()->local;
            f.calls.push_back({args[0], local->block_idx, f.running_lane});
            const bool deferring_lane = mask == 2U || f.running_lane % 3 == __builtin_ctz(mask);
            if (args[0] == 0 && local->block_idx == 1 && deferring_lane) {
                const int lane = f.running_lane;
                f.async_entries[lane][0] = {
                    reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                };
                f.async_groups[lane].count = 1;
            }
        };
        f.drain_until(
            [&] {
                return completion == expected - 1;
            },
            "all immediate subtask credits"
        );
        for (int lane = 0; lane < 6; ++lane)
            f.step(lane);
        EXPECT_EQ(completion, expected - 1) << mask;
        EXPECT_FALSE(f.completed(1)) << mask;
        peer_signal = 1;
        f.drain();
        f.expect_completed(0);
        f.expect_completed(1);
        EXPECT_EQ(f.calls.size(), expected + 1) << mask;
        EXPECT_EQ(f.control->async_registered, 1U);
        EXPECT_EQ(f.control->async_settled, 1U);
    }
}

TEST(PercounterAcquire, SharedCntNotifyRetainsEarlierCompletionAndUnmappedTasksUseGm) {
    alignas(32) uint32_t completion = 0;
    const uint64_t address = reinterpret_cast<uint64_t>(&completion);
    const int32_t first = 0;
    const int32_t predecessors[]{0, 1};
    auto successor = task(2, 3);
    successor.fanin_count = 1;
    successor.fanin_ids = &first;
    auto consumer = task(1);
    consumer.fanin_count = 2;
    consumer.fanin_ids = predecessors;
    std::vector<TaskInput> inputs{task(2, 2), successor, task(2, 2), consumer};
    SchedulerCompletionCounterPlan plan;
    std::string error;
    ASSERT_TRUE(plan_percounter_completion_counters(inputs, &plan, &error)) << error;
    ASSERT_EQ(plan.chains.size(), 2U);
    ASSERT_EQ(plan.task_chains[0], plan.task_chains[1]);
    ASSERT_NE(plan.task_chains[0], plan.task_chains[2]);
    ExternalAddresses external{};
    external.completion_plan = &plan;
    external.completion_notify = &address;
    external.completion_notify_count = 1;
    Fixture f(inputs, 2, external);
    EXPECT_EQ(f.counters[0].completion_notify_address, address);
    EXPECT_EQ(f.counters[1].completion_notify_address, address);
    EXPECT_EQ(f.counters[0].completion_notify_target, 2U);
    EXPECT_EQ(f.counters[1].completion_notify_target, 5U);
    EXPECT_EQ(f.counters[2].completion_notify_address, 0U);
    EXPECT_TRUE(percounter_completion_notify_head(f.counters[0], f.counters[0].value));
    EXPECT_FALSE(percounter_completion_notify_head(f.counters[1], f.counters[1].value));

    completion = 2;
    uint32_t cursor = 0;
    uint64_t pending = 0, target = 0;
    ASSERT_FALSE(percounter_probe_fanin(f.cfg[0], f.resolve(3), &cursor, &pending, &target));
    EXPECT_EQ(cursor, 1U);
    EXPECT_EQ(target, 5U);
    completion = 0;

    f.drain();
    EXPECT_EQ(completion, 5U);
    EXPECT_TRUE(f.completed(0));
    EXPECT_TRUE(f.completed(1));
    f.expect_completed(2);
    EXPECT_EQ(f.calls.size(), 8U);
    cursor = 0;
    pending = 0;
    target = 0;
    ASSERT_TRUE(percounter_probe_fanin(f.cfg[0], f.resolve(3), &cursor, &pending, &target));
    EXPECT_EQ(cursor, 2U);
}

TEST(PercounterAcquire, HostCompletedFaninDoesNotImplyAnEarlierDeviceTaskHasCompleted) {
    const int32_t first = 0;
    const int32_t middle = 1;
    auto published = task(1);
    published.inline_completed = true;
    published.fanin_count = 1;
    published.fanin_ids = &first;
    auto successor = task(2, 2);
    successor.fanin_count = 1;
    successor.fanin_ids = &middle;
    SchedulerCompletionCounterPlan plan;
    std::string error;
    ASSERT_TRUE(plan_percounter_completion_counters({task(2, 2), published, successor}, &plan, &error)) << error;
    EXPECT_EQ(plan.chains.size(), 2U);
    EXPECT_NE(plan.task_chains[0], plan.task_chains[2]);
}

TEST(PercounterAcquire, GraphBodyFaninWaitsForBothBlocksAfterMetadataPublication) {
    auto shell = task(1);
    shell.inline_completed = true;
    int32_t producer = 1;
    auto consumer = task(2);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({shell, task(1, 2), consumer});
    f.counters[0].value = percounter_counter_preset(1);
    for (auto &cfg : f.cfg) {
        cfg.table_task_count = 1;
    }
    for (int id : {1, 2}) {
        f.counters[id].storage_address = reinterpret_cast<uint64_t>(&f.storage[1]);
        f.counters[id].body_index = id - 1;
        f.counters[id].shell_task_id = 0;
        f.cfg[0].metadata_ready[id] = 0;
    }
    f.counters[1].flags = PERCOUNTER_BODY_ROOT;
    f.counters[2].flags = PERCOUNTER_BODY_SINK;
    f.counters[2].fanin_count = 1;
    f.counters[2].fanin_offset = f.layout.fanin_addr_offset;
    f.cfg[0].fanin_addresses[0] = 0;  // First producer in the Graph body.
    percounter_claim(f.cfg[0], &f.state[0]);
    percounter_claim(f.cfg[3], &f.state[3]);
    f.step(0);
    f.step(3);
    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.claim_total(0), 2U);
    f.cfg[0].metadata_ready[1] = f.cfg[0].metadata_ready[2] = 1;
    f.step(0);
    EXPECT_EQ(f.counters[1].value, 0U);
    f.step(1);
    EXPECT_EQ(f.calls.size(), 1U);
    f.step(3);
    f.step(1);
    ASSERT_EQ(f.calls.size(), 3U);
    EXPECT_EQ(f.calls.back()[0], 2);
    f.expect_completed(0);
    f.expect_completed(1);
    f.expect_completed(2);
}

TEST(PercounterAcquire, ReusedSyncMixStillWaitsForEveryNewBlock) {
    const int32_t dependency = 0;
    auto successor = task(7, 2, true);
    successor.fanin_count = 1;
    successor.fanin_ids = &dependency;
    Fixture f({task(7, 2, true), successor});
    for (int pass = 0; pass < 100 && f.calls.size() < 6; ++pass)
        for (int lane = 0; lane < 6 && f.calls.size() < 6; ++lane)
            f.step(lane);
    ASSERT_EQ(f.calls.size(), 6U);
    // Drain physical completion on both clusters, then stop the second cluster
    // before it can arrive at the next task's start barrier.
    for (int lane : {0, 3})
        f.step(lane);
    for (int pass = 0; pass < 16; ++pass)
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
    EXPECT_EQ(f.calls.size(), 6U);
    f.drain();
    EXPECT_EQ(f.calls.size(), 12U);
    f.expect_completed(0);
    f.expect_completed(1);
}

TEST(PercounterAcquire, PartitionedStartWaitsForEveryUnevenBlockPartition) {
    for (uint32_t blocks : {1U, 2U, 3U, 4U, 17U, 24U, 32U}) {
        Fixture f({task(7, blocks < 2 ? 2 : blocks, true)}, 32);
        const PercounterSyncNotify gm{0, 0};
        for (uint32_t block = 0; block + 1 < blocks; ++block)
            percounter_start_arrive(f.cfg[0], gm, 0, block);
        for (uint32_t observer = 0; observer < 4; ++observer)
            EXPECT_FALSE(percounter_start_ready(f.cfg[0], gm, 0, blocks, observer));
        percounter_start_arrive(f.cfg[0], gm, 0, blocks - 1);
        for (uint32_t observer = 0; observer < 4; ++observer)
            EXPECT_TRUE(percounter_start_ready(f.cfg[0], gm, 0, blocks, observer));
    }
}

/**
 * One counter carries the whole cohort, so its arrival total is the block count
 * itself and every observer reads the same register. The GM sectors stay at
 * zero, which is what proves the two representations do not both run.
 */
TEST(PercounterAcquire, NotifiedStartCountsTheWholeCohortInOneRegister) {
    for (uint32_t blocks : {2U, 3U, 4U, 17U, 24U, 32U}) {
        Fixture f({task(7, blocks, true)}, 32);
        uint32_t counter = 0;
        const auto address = reinterpret_cast<uint64_t>(&counter);
        const PercounterSyncNotify notify{address, address};
        for (uint32_t block = 0; block + 1 < blocks; ++block)
            percounter_start_arrive(f.cfg[0], notify, 0, block);
        for (uint32_t observer = 0; observer < 4; ++observer)
            EXPECT_FALSE(percounter_start_ready(f.cfg[0], notify, 0, blocks, observer));
        percounter_start_arrive(f.cfg[0], notify, 0, blocks - 1);
        EXPECT_EQ(counter, blocks);
        for (uint32_t observer = 0; observer < 4; ++observer)
            EXPECT_TRUE(percounter_start_ready(f.cfg[0], notify, 0, blocks, observer));
        for (const auto &sector : f.cfg[0].sync_arrivals[0].sectors)
            EXPECT_EQ(sector.next, 0U);
    }
}

/**
 * Owner counts follow the reported topology, so the gate opens at exactly the
 * width a part of that size can fill: one owner per cluster for AIC and MIX,
 * one per AIV lane for ordinary vector work. A sync cohort is never wider --
 * `percounter_sync_widths_fit` rejects that at the AICPU handshake -- so the
 * equality selects the cohort that spans every owner rather than a width that
 * happens to match the part.
 */
TEST(PercounterAcquire, StaticAssignmentOpensAtEachDomainsFullWidth) {
    for (uint32_t clusters : {2U, 32U, 36U}) {
        LaneConfig cfg{};
        cfg.cluster_count = clusters;
        cfg.aiv_per_cluster = 2;
        const uint32_t cluster_wide = clusters;
        const uint32_t lane_wide = clusters * 2;
        EXPECT_TRUE(percounter_claim_is_static(cfg, 0, false, true, cluster_wide));
        EXPECT_TRUE(percounter_claim_is_static(cfg, 1, true, true, cluster_wide));
        EXPECT_TRUE(percounter_claim_is_static(cfg, 1, false, true, lane_wide));
        // Filling one domain does not fill the other.
        EXPECT_FALSE(percounter_claim_is_static(cfg, 1, false, true, cluster_wide));
        EXPECT_FALSE(percounter_claim_is_static(cfg, 0, false, true, lane_wide));
        // A narrower cohort keeps the ticket path, which is what places its
        // blocks on whichever owners reach the boundary first.
        for (uint32_t blocks : {1U, cluster_wide - 1, lane_wide - 1})
            EXPECT_FALSE(percounter_claim_is_static(cfg, 1, false, true, blocks));
        // Ordinary work draws tickets at every width.
        EXPECT_FALSE(percounter_claim_is_static(cfg, 0, false, false, cluster_wide));
        EXPECT_FALSE(percounter_claim_is_static(cfg, 1, false, false, lane_wide));
    }
}

/**
 * A full-width cohort's owners partition its blocks: every owner takes one,
 * every block is taken once, and the sectors end closed. The four highest ranks
 * carry the four sector bits and are distinct in their sector because four
 * consecutive ranks are; a sector no block reaches is seeded closed by the
 * Host, which has no owner to issue its bit.
 */
TEST(PercounterAcquire, FullWidthOwnersPartitionTheirCohortsBlocks) {
    struct Domain {
        uint32_t list;
        bool mix;
        uint32_t owners_per_cluster;
    };
    for (uint32_t clusters : {2U, 32U, 36U})
        for (const Domain domain : {Domain{0, false, 1}, Domain{1, true, 1}, Domain{1, false, 2}}) {
            const uint32_t blocks = clusters * domain.owners_per_cluster;
            std::vector<PercounterTaskClaim> claims(1);
            LaneConfig cfg{};
            cfg.cluster_count = clusters;
            cfg.aiv_per_cluster = 2;
            cfg.task_claims = claims.data();
            ASSERT_TRUE(percounter_claim_is_static(cfg, domain.list, domain.mix, true, blocks));
            std::vector<uint32_t> taken(blocks, 0);
            for (uint32_t cluster = 0; cluster < clusters; ++cluster)
                for (uint32_t lane = 0; lane < domain.owners_per_cluster; ++lane) {
                    cfg.cluster_index = static_cast<int32_t>(cluster);
                    cfg.cluster_lane = static_cast<int32_t>(lane + 1);
                    const uint32_t block = percounter_claim_static_block(cfg, domain.list, domain.mix, blocks);
                    ASSERT_LT(block, blocks);
                    ++taken[block];
                }
            for (uint32_t count : taken)
                EXPECT_EQ(count, 1U);
            // Assignment writes nothing. Exhaustion would claim the boundary was
            // finished once its highest ranks had claimed, which is false while a
            // lower rank still holds a reserved block, so no reader may ask it.
            EXPECT_EQ(percounter_claim_closed(cfg, 0), 0U);
            EXPECT_FALSE(percounter_claim_exhausted(percounter_claim_closed(cfg, 0)));
        }
}

/**
 * The publication floor is the rank of the oldest boundary this lane may still
 * claim from, and peers act on it: a lane enters a younger cohort only above
 * its own floor, and a MIX coordinator whose floor reaches the end publishes
 * `publication_closed`. A boundary this lane still owes a block to must hold
 * the floor down.
 *
 * Exhaustion cannot answer that on a statically assigned boundary: it is a
 * statement about other owners, and this lane's block is reserved for it alone.
 * The floor must ignore the mask there even when the mask is complete.
 */
TEST(PercounterAcquire, FloorHoldsAtAStaticBoundaryThisLaneStillOwes) {
    constexpr uint32_t kClusters = 8;
    Fixture f({task(1, kClusters, true)}, kClusters);
    auto &cfg = f.cfg[0];
    ASSERT_EQ(cfg.cluster_lane, 0);
    ASSERT_EQ(cfg.cluster_index, 0);
    const uint32_t segment = f.control->first_boundary_segment[0];
    ASSERT_NE(segment, PERCOUNTER_NO_BOUNDARY);
    ASSERT_TRUE(percounter_claim_is_static(cfg, 0, false, true, kClusters));
    const uint32_t boundary_rank =
        cfg.metadata[static_cast<uint32_t>(cfg.segments[segment].task_shape)].global_order_id;
    EXPECT_EQ(percounter_source_floor(cfg, f.state[0].source, false), boundary_rank);

    // Every other owner takes its reserved block. Assignment records no
    // exhaustion, so force the mask complete by hand: the floor must still hold,
    // because rank 0 has not scanned the boundary and block 0 is its alone.
    LaneConfig owner = cfg;
    for (uint32_t rank = 1; rank < kClusters; ++rank) {
        owner.cluster_index = static_cast<int32_t>(rank);
        EXPECT_EQ(percounter_claim_static_block(owner, 0, false, kClusters), rank);
    }
    EXPECT_EQ(percounter_claim_closed(cfg, segment), 0U);
    cfg.task_claims[segment].closed_sectors = PERCOUNTER_CLAIM_SECTOR_MASK;
    ASSERT_TRUE(percounter_claim_exhausted(percounter_claim_closed(cfg, segment)));

    EXPECT_EQ(percounter_source_floor(cfg, f.state[0].source, false), boundary_rank);

    // Claiming is what releases the floor, and the claim cursor passing the
    // boundary is what records it -- a lane that owns no block there passes it
    // the same way.
    f.state[0].source.segment = segment + 1;
    EXPECT_EQ(percounter_source_floor(cfg, f.state[0].source, false), UINT32_MAX);
}

TEST(PercounterPayload, NotifyTableCarriesOnlyTheMappedSetAddress) {
    const std::vector<TaskInput> inputs{task(1, 2, true), task(1, 2, true), task(1, 2, true)};
    const uint64_t counts[2]{inputs.size(), 0};
    PercounterLayout layout{};
    ASSERT_TRUE(plan_layout(inputs.size(), 6, 0, counts, &layout, 0, inputs.size(), inputs.size()));
    std::vector<uint8_t> storage(layout.upload_size + 127);
    const auto address = (reinterpret_cast<uintptr_t>(storage.data()) + 127) & ~uintptr_t{127};
    auto *base = reinterpret_cast<void *>(address);
    // The slot's ADD alias is a fixed displacement of its SET address, so an
    // address the alias rule rejects is an unmapped cohort and is published as
    // zero; the table never carries the alias itself.
    const uint64_t first = UINT64_C(0x8000'0000);
    const std::array<uint64_t, 3> sets{first, 0, first + 8};
    ExternalAddresses external{};
    external.sync_notify = sets.data();
    external.sync_notify_count = sets.size();
    ASSERT_TRUE(build_tables(inputs.data(), inputs.size(), layout, base, address, 2, external));
    const auto *table = region_at<uint64_t>(base, layout.sync_notify_offset);
    EXPECT_EQ(table[0], first);
    EXPECT_EQ(table[1], 0U);
    EXPECT_EQ(table[2], 0U);
    for (uint64_t entry : {table[0], table[1], table[2]})
        EXPECT_EQ(percounter_notify_add_address(entry), a5_cnt_notify_increment_address(entry));
}

TEST(PercounterAcquire, NotifySlotCostsOneLoadPerCohort) {
    Fixture f({task(1, 2, true)});
    auto &cfg = f.cfg[0];
    std::array<uint64_t, 3> table{UINT64_C(0x8000'0000), 0, UINT64_C(0x8000'0040)};
    cfg.sync_notify = table.data();
    for (uint32_t cohort = 0; cohort < table.size(); ++cohort) {
        const auto notify = percounter_sync_notify(cfg, &f.state[0], cohort);
        EXPECT_EQ(notify.read_address, table[cohort]);
        // An unmapped cohort keeps a zero pair, which is what selects the GM
        // arrival sectors.
        EXPECT_EQ(notify.add_address, table[cohort] == 0 ? 0 : table[cohort] + 8);
        EXPECT_EQ(f.state[0].notify_sync, cohort);
    }
    // A repeated probe of the cached cohort reads nothing: clearing the table
    // entry underneath it cannot change what it returns.
    table[2] = 0;
    EXPECT_EQ(percounter_sync_notify(cfg, &f.state[0], 2).add_address, UINT64_C(0x8000'0048));
    EXPECT_EQ(percounter_sync_notify(cfg, &f.state[0], PERCOUNTER_NO_SYNC).add_address, 0U);
}

TEST(PercounterPayload, GraphInvocationsKeepDeviceArgumentsDistinctAcrossBlocks) {
    auto shell = task(1);
    shell.inline_completed = true;
    Fixture f({shell, shell, task(1, 2), task(1, 2)});
    std::array<DispatchPayload, 2> graph_arguments{};
    for (auto &cfg : f.cfg) {
        cfg.table_task_count = 2;
    }
    const uint64_t functions[3]{reinterpret_cast<uint64_t>(&kernel), 0, 0};
    const uint64_t scalars[2]{71, 93};
    for (uint32_t invocation = 0; invocation < 2; ++invocation) {
        auto &counter = f.counters[2 + invocation];
        counter.storage_address = reinterpret_cast<uint64_t>(&f.storage[2 + invocation]);
        counter.body_index = 0;
        counter.shell_task_id = invocation;
        counter.flags = PERCOUNTER_BODY_ROOT;
        ASSERT_EQ(
            percounter_prepare_argument_metadata(
                counter, functions, 1, reinterpret_cast<uint64_t>(&graph_arguments[invocation]), 1, 1
            ),
            SchedulerGraphResult::OK
        );
        graph_arguments[invocation].args[0] = 0x1000U + invocation * 0x1000U;
        graph_arguments[invocation].args[1] = scalars[invocation];
        std::fill(std::begin(f.arguments[2 + invocation].args), std::end(f.arguments[2 + invocation].args), UINT64_MAX);
    }
    uint32_t invocation = 0;
    int32_t block = 0;
    execute = [&](int64_t *args) {
        EXPECT_EQ(args, reinterpret_cast<int64_t *>(graph_arguments[invocation].args));
        EXPECT_EQ(static_cast<uint64_t>(args[0]), 0x1000U + invocation * 0x1000U);
        EXPECT_EQ(static_cast<uint64_t>(args[1]), scalars[invocation]);
        EXPECT_EQ(get_block_idx(args), block);
        EXPECT_EQ(get_block_num(args), 2);
    };
    for (block = 0; block < 2; ++block)
        for (invocation = 0; invocation < 2; ++invocation) {
            const auto body = f.resolve(2 + invocation);
            EXPECT_EQ(body.shell, invocation);
            ASSERT_TRUE(f.run(0, body, block));
        }
    EXPECT_EQ(graph_arguments[0].args[1], scalars[0]);
    EXPECT_EQ(graph_arguments[1].args[1], scalars[1]);
}

TEST(PercounterAcquire, SyncClaimsAndStartArrivalsArePerTaskAndPerPhysicalLane) {
    Fixture f({task(1, 2, true), task(1, 2, true), task(2, 4, true)});
    f.drain();
    ASSERT_EQ(f.calls.size(), 8U);
    for (int id = 0; id < 3; ++id) {
        std::array<int, 6> per_lane{};
        const auto meta = f.cfg[0].metadata[id];
        for (uint32_t part = 0; part < PERCOUNTER_START_SECTOR_COUNT; ++part)
            EXPECT_EQ(
                f.cfg[0].sync_arrivals[meta.sync_index].sectors[part].next,
                (meta.block_num + PERCOUNTER_START_SECTOR_COUNT - 1 - part) / PERCOUNTER_START_SECTOR_COUNT
            );
        for (const auto &call : f.calls)
            if (call[0] == id) ++per_lane[call[2]];
        for (int count : per_lane)
            EXPECT_LE(count, 1);
        f.expect_completed(id);
    }
}

TEST(PercounterAcquire, MixMasksSpmdAndReuseRetireExactlyOnce) {
    Fixture f({task(3, 3), task(7, 2, true), task(6, 3), task(5, 3), task(7, 3)});
    f.drain();
    EXPECT_EQ(f.calls.size(), 33U);
    for (int id = 0; id < 5; ++id)
        f.expect_completed(id);
    for (const auto &call : f.calls)
        if (call[0] == 3) EXPECT_NE(call[2] % 3, 1);
}

TEST(PercounterAcquire, MetadataNotReadyCannotResolveOrExecute) {
    auto shell = task(1);
    shell.inline_completed = true;
    Fixture f({shell, task(7)});
    for (auto &cfg : f.cfg) {
        cfg.table_task_count = 1;
    }
    f.counters[1].storage_address = reinterpret_cast<uint64_t>(&f.storage[1]);
    f.counters[1].body_index = 0;
    f.counters[1].shell_task_id = 0;
    f.counters[1].flags = PERCOUNTER_BODY_ROOT;
    f.cfg[0].metadata_ready[1] = 0;
    for (int r = 0; r < 4; ++r)
        for (int lane = 0; lane < 6; ++lane)
            f.step(lane);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.signals[0].mix[0].fanin_ready, 0U);
    f.cfg[0].metadata_ready[1] = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 3U);
}

TEST(PercounterAcquire, MixTaskCountMustMatchMergedVectorList) {
    Fixture f({task(7)});
    auto layout = f.layout;
    layout.mix_task_count = 0;
    std::vector<uint8_t> backing(layout.total_size + 127);
    auto *bytes = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t{127});
    EXPECT_FALSE(build_tables(f.inputs.data(), f.inputs.size(), layout, bytes, 0, 2));
}

TEST(PercounterAcquire, RealTopologyRejectsOverwideSync) {
    const uint64_t widths[3]{2, 4, 2};
    EXPECT_TRUE(percounter_sync_widths_fit(widths, 2, 2));
    EXPECT_FALSE(percounter_sync_widths_fit(widths, 1, 2));
}

TEST(PercounterAcquire, RejectsUnrepresentableSpmdBlockCount) {
    const uint64_t counts[2]{1, 0};
    PercounterLayout layout;
    ASSERT_TRUE(plan_layout(1, 65536, 0, counts, &layout, 0, 1));
    std::vector<uint8_t> storage(layout.total_size + 127);
    auto *base = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(storage.data()) + 127) & ~uintptr_t{127});
    const auto input = task(1, 65536);
    BuildResult result;
    EXPECT_FALSE(build_tables(&input, 1, layout, base, 0, 2, {}, &result));
    EXPECT_EQ(result.status, BuildStatus::INVALID_ARGUMENT);
}

TEST(PercounterAcquire, SegmentIndicesMustFitTheDeviceCursor) {
    const uint64_t counts[2]{UINT32_MAX, 0};
    PercounterLayout layout;
    EXPECT_FALSE(plan_layout(UINT32_MAX, uint64_t{UINT32_MAX} * 2, 0, counts, &layout, 0, UINT32_MAX));
}

TEST(PercounterAcquire, ReadyFrontierKeepsBLevelPriorityAcrossBothTables) {
    int32_t producer = 1;
    auto consumer = task(7);
    consumer.fanin_count = 1;
    consumer.fanin_ids = &producer;
    Fixture f({task(1), task(2), consumer});
    EXPECT_EQ(f.cfg[0].metadata[1].global_order_id, 0U);
    EXPECT_EQ(f.cfg[0].metadata[0].global_order_id, 1U);
    EXPECT_EQ(f.cfg[0].metadata[2].global_order_id, 2U);
}

TEST(PercounterAcquire, IndependentReadyRootPrecedesUnreadyFullWidthConsumer) {
    const int32_t producer = 0;
    auto pending = task(1, 2);
    pending.fanin_count = 1;
    pending.fanin_ids = &producer;
    Fixture f({task(2), pending, task(1)});
    EXPECT_LT(f.cfg[0].metadata[2].global_order_id, f.cfg[0].metadata[1].global_order_id);
    EXPECT_EQ(percounter_order_task_id(f.cfg[0].order[0]), 2);
    f.step(0);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls[0][0], 2);
    EXPECT_EQ(f.counters[0].value, 0U);
    EXPECT_NE(f.counters[1].value, PERCOUNTER_COUNTER_PUBLISHED);
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, ReusedOrdinaryPositionWaitsForItsNewProducer) {
    auto first_producer = task(1);
    first_producer.published_off_lane = true;
    auto second_producer = first_producer;
    const int32_t first = 0, second = 1;
    auto first_consumer = task(2), second_consumer = task(2);
    first_consumer.fanin_count = second_consumer.fanin_count = 1;
    first_consumer.fanin_ids = &first;
    second_consumer.fanin_ids = &second;
    Fixture f({first_producer, second_producer, first_consumer, second_consumer});
    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    f.counters[0].value = PERCOUNTER_COUNTER_PUBLISHED;
    f.step(1);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls.front()[0], 2);
    f.step(1);
    EXPECT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.counters[3].value, 0U);
    f.counters[1].value = PERCOUNTER_COUNTER_PUBLISHED;
    f.step(1);
    ASSERT_EQ(f.calls.size(), 2U);
    EXPECT_EQ(f.calls.back()[0], 3);
}

TEST(PercounterAcquire, ReusedMixRecordWaitsForItsNewProducer) {
    auto first_producer = task(1);
    first_producer.published_off_lane = true;
    auto second_producer = first_producer;
    const int32_t first = 0, second = 1;
    auto first_consumer = task(7), second_consumer = task(7);
    first_consumer.fanin_count = second_consumer.fanin_count = 1;
    first_consumer.fanin_ids = &first;
    second_consumer.fanin_ids = &second;
    std::vector<TaskInput> inputs{first_producer, second_producer};
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i)
        inputs.push_back(first_consumer);
    inputs.push_back(second_consumer);
    const uint32_t last = static_cast<uint32_t>(inputs.size() - 1);
    Fixture f(std::move(inputs));
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i) {
        f.step(1);
        f.step(0);
        f.step(2);
    }
    auto &record = f.signals[0].mix[0];
    EXPECT_EQ(record.metadata_generation, record.generation);
    EXPECT_NE(record.fanin_ready, record.generation);
    f.counters[0].value = PERCOUNTER_COUNTER_PUBLISHED;
    for (uint32_t step = 0; step < 100 && !f.completed(2); ++step)
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
    ASSERT_EQ(f.counters[2].value, PERCOUNTER_COUNTER_PUBLISHED);
    for (uint32_t step = 0; step < 100; ++step) {
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
        if (percounter_order_task_id(record.entry) == last && record.metadata_generation == record.generation) break;
    }
    ASSERT_EQ(percounter_order_task_id(record.entry), last);
    ASSERT_EQ(record.metadata_generation, record.generation);
    EXPECT_NE(record.fanin_ready, record.generation);
    EXPECT_EQ(f.calls.size(), 3U * PERCOUNTER_PENDING_CAPACITY);
    f.counters[1].value = PERCOUNTER_COUNTER_PUBLISHED;
    for (uint32_t step = 0; step < 100 && !f.completed(last); ++step)
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
    EXPECT_EQ(f.counters[last].value, PERCOUNTER_COUNTER_PUBLISHED);
    EXPECT_EQ(f.calls.size(), 3U * (PERCOUNTER_PENDING_CAPACITY + 1));
}

TEST(PercounterAcquire, PublicationBoundariesKeepSeparateOrderedDomains) {
    Fixture f(
        {task(1, 4), task(2, 4), task(3, 2), task(2, 2, true), task(1, 2, true), task(5), task(2), task(6, 2, true),
         task(1, 2, true), task(2, 2, true)}
    );
    const std::array<uint32_t, 10> order{{0, 1, 3, 6, 2, 9, 4, 7, 5, 8}};
    for (uint32_t rank = 0; rank < order.size(); ++rank)
        EXPECT_EQ(f.cfg[0].metadata[order[rank]].global_order_id, rank);
    const std::array<std::vector<uint32_t>, 3> expected{{{4, 8}, {3, 9}, {2, 7, 5}}};
    for (uint32_t domain = 0; domain < expected.size(); ++domain) {
        std::vector<uint32_t> tasks;
        uint32_t segment = f.control->first_boundary_segment[domain];
        while (segment != PERCOUNTER_NO_BOUNDARY) {
            ASSERT_LT(tasks.size(), f.inputs.size());
            ASSERT_LT(segment, f.layout.boundary_task_count + PERCOUNTER_LIST_COUNT);
            const uint32_t id = static_cast<uint32_t>(f.cfg[0].segments[segment].task_shape);
            tasks.push_back(id);
            segment = f.cfg[0].metadata[id].next_boundary_segment;
        }
        EXPECT_EQ(tasks, expected[domain]);
    }
    const auto aic_floor = [&] {
        return percounter_source_floor(f.cfg[0], f.state[0].source, false);
    };
    const auto aiv_floor = [&] {
        return percounter_source_floor(f.cfg[1], f.state[1].source, false);
    };
    const auto mix_floor = [&] {
        return percounter_source_floor(f.cfg[1], f.state[1].source, true);
    };
    EXPECT_EQ(aic_floor(), 6U);
    EXPECT_EQ(aiv_floor(), 2U);
    EXPECT_EQ(mix_floor(), 4U);
    const auto close_boundary = [&](uint32_t task_id) {
        for (uint32_t list = 0; list < PERCOUNTER_LIST_COUNT; ++list) {
            for (uint32_t local = 0; local < f.control->segment_count[list]; ++local) {
                const uint32_t index = f.control->segment_begin[list] + local;
                const uint64_t shape = f.cfg[0].segments[index].task_shape;
                if (shape != 0 && static_cast<uint32_t>(shape) == task_id) {
                    auto &claim = f.cfg[0].task_claims[index - list];
                    for (uint32_t part = 0; part < PERCOUNTER_CLAIM_SECTOR_COUNT; ++part)
                        claim.sectors[part].next =
                            ((shape >> 48) + PERCOUNTER_CLAIM_SECTOR_COUNT - 1 - part) / PERCOUNTER_CLAIM_SECTOR_COUNT;
                    claim.closed_sectors = PERCOUNTER_CLAIM_SECTOR_MASK;
                    return;
                }
            }
        }
        FAIL() << "missing boundary task " << task_id;
    };
    // Exhaustion is a statement about every lane, and it releases a ticket
    // boundary outright. A statically assigned boundary reserves this lane's
    // own block, so only this lane's scan passing it releases the floor --
    // two of the three domains here are full-width and take that rule.
    const auto finish_scan = [&](uint32_t list) {
        f.state[list == 0 ? 0 : 1].source.segment = f.control->segment_count[list];
    };
    close_boundary(8);
    close_boundary(5);
    EXPECT_EQ(aic_floor(), 6U);
    EXPECT_EQ(mix_floor(), 4U);
    close_boundary(4);
    EXPECT_EQ(aic_floor(), 6U);
    finish_scan(0);
    EXPECT_EQ(aic_floor(), UINT32_MAX);
    close_boundary(3);
    EXPECT_EQ(aiv_floor(), 5U);
    close_boundary(2);
    EXPECT_EQ(mix_floor(), 7U);
    close_boundary(7);
    EXPECT_EQ(mix_floor(), 7U);
    finish_scan(1);
    EXPECT_EQ(mix_floor(), UINT32_MAX);
}

TEST(PercounterAcquire, GraphBodyRanksAndSyncBoundariesExistBeforeMaterialization) {
    const BodyTaskShape shapes[]{{1, false, false, 1, 1}, {7, true, false, 2, 6}};
    const int32_t offsets[]{0, 0, 1};
    const int32_t producers[]{0};
    HostGraphTemplate body;
    ASSERT_EQ(build_graph_template(shapes, 2, offsets, producers, 1, 6, &body).status, TemplateStatus::OK);
    std::array<TaskInput, 3> inputs{task(2), task(2), task(2)};
    inputs[1].graph_invocation = 0;
    const int32_t shell = 1;
    inputs[2].fanin_count = 1;
    inputs[2].fanin_ids = &shell;
    const GraphInvocationInput invocation{1, 0};
    const GraphPlanInput graph{&body, 1, &invocation, 1};
    const GraphLayoutInput graph_layout{2, 1, graph_template_bytes(body)};
    const uint64_t counts[]{1, 3};
    PercounterLayout layout;
    ASSERT_TRUE(plan_layout(3, 6, 1, counts, &layout, 1, 1, 1, graph_layout));
    std::vector<uint8_t> backing(layout.total_size + 127);
    auto *bytes = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t{127});
    ASSERT_TRUE(build_tables(inputs.data(), inputs.size(), layout, bytes, 0, 2, {}, nullptr, graph));
    const auto *metadata = region_at<PercounterTaskMetadata>(bytes, layout.metadata_offset);
    EXPECT_EQ(metadata[3].global_order_id, 0U);
    EXPECT_EQ(metadata[4].global_order_id, 1U);
    EXPECT_EQ(metadata[0].global_order_id, 2U);
    EXPECT_EQ(metadata[2].global_order_id, 3U);
    EXPECT_NE(metadata[4].sync_index, PERCOUNTER_NO_SYNC);
    const auto *control = region_at<PercounterControl>(bytes, layout.control_offset);
    EXPECT_EQ(control->max_sync_blocks[2], 2U);
    EXPECT_EQ(control->first_boundary_segment[0], PERCOUNTER_NO_BOUNDARY);
    EXPECT_EQ(control->first_boundary_segment[1], PERCOUNTER_NO_BOUNDARY);
    EXPECT_EQ(control->first_boundary_segment[2], 1U);
    EXPECT_EQ(metadata[4].next_boundary_segment, PERCOUNTER_NO_BOUNDARY);
    const auto *ready = region_at<uint64_t>(bytes, layout.metadata_ready_offset);
    EXPECT_EQ(ready[3], 0U);
    EXPECT_EQ(ready[4], 0U);
    EXPECT_LT(layout.metadata_ready_offset + 5 * sizeof(uint64_t), layout.counters_offset);
    const auto *mix = region_at<uint64_t>(bytes, layout.orders_offset[1]);
    EXPECT_EQ(percounter_order_task_id(mix[0]), 4);
    EXPECT_TRUE(percounter_order_is_spmd(mix[0]));
}

TEST(PercounterAcquire, HostSeedsGraphCompletionBeforeAnyBodyClaim) {
    const int32_t blocks = 5;
    const BodyTaskShape shapes[]{{1, false, false, blocks, blocks}, {2, false, false, 1, 1}};
    const int32_t offsets[]{0, 0, 1};
    const int32_t producers[]{0};
    HostGraphTemplate body;
    ASSERT_EQ(build_graph_template(shapes, 2, offsets, producers, 1, 6, &body).status, TemplateStatus::OK);
    auto shell = task(1);
    shell.graph_invocation = 0;
    const GraphInvocationInput invocation{0, 0};
    const GraphPlanInput graph{&body, 1, &invocation, 1};
    const GraphLayoutInput graph_layout{2, 1, graph_template_bytes(body)};
    const uint64_t counts[]{1, 1};
    PercounterLayout layout;
    ASSERT_TRUE(plan_layout(1, blocks + 1, 0, counts, &layout, 0, 1, 0, graph_layout));
    std::vector<uint8_t> bytes(layout.total_size + 128, 0xa5);
    auto *base = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(bytes.data()) + 127) & ~uintptr_t{127});
    ASSERT_TRUE(build_tables(&shell, 1, layout, base, reinterpret_cast<uint64_t>(base), 2, {}, nullptr, graph));
    const auto *counters = region_at<PercounterCounter>(base, layout.counters_offset);
    const auto *claims = region_at<PercounterTaskClaim>(base, layout.task_claims_offset);
    EXPECT_EQ(counters[1].value, percounter_counter_preset(blocks));
    EXPECT_EQ(layout.upload_size, layout.task_claims_offset + sizeof(PercounterTaskClaim));
    EXPECT_LT(layout.task_claims_offset, layout.upload_size);
    EXPECT_EQ(counters[1].arguments_address, 0U);
    LaneConfig cfg{};
    cfg.control = reinterpret_cast<PercounterControl *>(base);
    cfg.counters = region_at<PercounterCounter>(base, layout.counters_offset);
    cfg.metadata = region_at<PercounterTaskMetadata>(base, layout.metadata_offset);
    cfg.task_count = layout.task_count;
    cfg.task_claims = region_at<PercounterTaskClaim>(base, layout.task_claims_offset);
    cfg.segments = region_at<PercounterSegment>(base, layout.segments_offset);
    for (uint32_t list = 0; list < PERCOUNTER_LIST_COUNT; ++list)
        cfg.segment_count[list] = cfg.control->segment_count[list];
    EXPECT_EQ(claims[0].closed_sectors, 0U);
    for (const auto &sector : claims[0].sectors)
        EXPECT_EQ(sector.next, 0U);
    const auto *ready = region_at<uint64_t>(base, layout.metadata_ready_offset);
    EXPECT_EQ(ready[1], 0U);
    EXPECT_EQ(ready[2], 0U);
    const auto *header = region_at<PercounterGraphTemplate>(base, layout.graph_templates_offset);
    const auto *references = region_at<uint64_t>(base, header->fanin_indices_offset);
    EXPECT_EQ(references[0], 0U);
}

TEST(PercounterAcquire, TaskCountersDoNotCrossSyncBoundaryOrDuplicateBlocks) {
    Fixture f({task(1, 17), task(1, 2, true), task(1, 7)});
    f.drain();
    ASSERT_EQ(f.calls.size(), 26U);
    for (int id = 0; id < 3; ++id) {
        f.expect_completed(id);
        std::vector<int> counts(f.inputs[id].block_num);
        for (const auto &call : f.calls)
            if (call[0] == id) ++counts[call[1]];
        for (int count : counts)
            EXPECT_EQ(count, 1);
    }
}

TEST(PercounterAcquire, UnifiedVectorSlotsDoNotClaimLaterMixPastFullOrdinaryWindow) {
    auto external = task(1);
    external.published_off_lane = true;
    const int32_t dependency = 0;
    std::vector<TaskInput> tasks{external};
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY; ++i) {
        auto pending = task(2);
        pending.fanin_count = 1;
        pending.fanin_ids = &dependency;
        tasks.push_back(pending);
    }
    tasks.push_back(task(7));
    Fixture f(tasks);
    for (uint32_t i = 0; i < PERCOUNTER_PENDING_CAPACITY + 2; ++i)
        f.step(1);
    EXPECT_EQ(f.state[1].occupied, (1U << PERCOUNTER_PENDING_CAPACITY) - 1);
    EXPECT_EQ(f.state[1].mix_owned, 0U);
    f.counters[0].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), PERCOUNTER_PENDING_CAPACITY + 3);
}

TEST(PercounterAcquire, OrdinarySpanDoesNotCloseFollowingMixBoundary) {
    Fixture f({task(2), task(7)});
    const uint32_t index = f.cfg[1].segment_begin[1];
    const auto exhausted = [&] {
        return percounter_claim_exhausted(percounter_claim_closed(f.cfg[1], index - 1));
    };
    const auto summary_bit = [&] {
        return f.cfg[1].claim_summary[index / 64].closed & (uint64_t{1} << (index % 64));
    };
    percounter_claim(f.cfg[2], &f.state[2]);
    EXPECT_FALSE(exhausted());
    EXPECT_EQ(summary_bit(), 0U);
    percounter_claim(f.cfg[1], &f.state[1]);
    // Drawing the boundary's last block exhausts it. The summary bit is a skip
    // hint that a lane adds once it reaches the boundary and finds nothing to
    // draw, so it trails the exhaustion it reports.
    EXPECT_TRUE(exhausted());
    EXPECT_EQ(summary_bit(), 0U);
    EXPECT_FALSE(f.completed(1));
    EXPECT_TRUE(f.calls.empty());
    f.drain();
    EXPECT_NE(summary_bit(), 0U);
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, MissingClaimClosureHintsDoNotPreventProgress) {
    Fixture f({task(1, 7), task(7, 5), task(2, 9)});
    for (int round = 0; round < 1000 && !f.sinks_complete(); ++round) {
        for (int lane = 0; lane < 6; ++lane) {
            f.step(lane);
            f.cfg[lane].claim_summary[0].closed = 0;
        }
    }
    EXPECT_TRUE(f.sinks_complete());
    EXPECT_EQ(f.calls.size(), 31U);
    for (int task_id = 0; task_id < 3; ++task_id)
        f.expect_completed(task_id);
}

TEST(PercounterAcquire, SegmentClosurePublishesEachBitOnceAcrossSummaryWords) {
    std::vector<TaskInput> tasks(70, task(1, 2));
    Fixture f(tasks);
    // A summary bit is a skip hint added by a lane that revisits an exhausted
    // boundary, so the last boundary's bit trails the block that exhausted it.
    f.drain_until(
        [&f] {
            return f.sinks_complete() && f.cfg[0].claim_summary[1].closed == 255U;
        },
        "every boundary's summary bit published"
    );
    ASSERT_EQ(f.calls.size(), 140U);
    EXPECT_EQ(f.cfg[0].claim_summary[0].closed, UINT64_MAX);
    EXPECT_EQ(f.cfg[0].claim_summary[1].closed, 255U);
    for (size_t i = 0; i < tasks.size(); ++i)
        f.expect_completed(i);
}

TEST(PercounterAcquire, Aiv1SkipsMixWithoutConsumingItsTaskCounter) {
    Fixture f({task(7, 3), task(2, 6)});
    percounter_claim(f.cfg[2], &f.state[2]);
    EXPECT_EQ(f.claim_total(0), 0U);
    EXPECT_EQ(percounter_order_task_id(f.state[2].slots[0].entry), 1);
    percounter_claim(f.cfg[1], &f.state[1]);
    EXPECT_EQ(f.claim_total(0), 1U);
    EXPECT_EQ(f.state[1].mix_owned, 1U);
    EXPECT_EQ(percounter_order_task_id(f.signals[0].mix[0].entry), 0);
    f.drain();
    EXPECT_EQ(f.calls.size(), 15U);
}

TEST(PercounterAcquire, OrdinaryClaimCannotConsumeMixOrCrossItsBoundary) {
    Fixture f({task(2), task(7), task(2)});
    percounter_claim(f.cfg[2], &f.state[2]);
    EXPECT_EQ(f.state[2].occupied, 1U);
    EXPECT_EQ(percounter_order_task_id(f.state[2].slots[0].entry), 0);
    EXPECT_EQ(f.claim_total(0), 0U);
    percounter_claim(f.cfg[2], &f.state[2]);
    EXPECT_EQ(f.claim_total(0), 0U);
    percounter_claim(f.cfg[1], &f.state[1]);
    EXPECT_EQ(f.state[1].mix_owned, 1U);
    EXPECT_EQ(percounter_order_task_id(f.signals[0].mix[0].entry), 1);
    f.drain();
    EXPECT_EQ(f.calls.size(), 5U);
}

TEST(PercounterAcquire, GenericSlotsPreserveProgressInBothVectorDependencyDirections) {
    for (bool mix_producer : {false, true}) {
        const int32_t predecessor = 0;
        auto producer = task(mix_producer ? 7 : 2, mix_producer ? 2 : 4, true);
        auto consumer = task(mix_producer ? 2 : 7, 4);
        consumer.fanin_count = 1;
        consumer.fanin_ids = &predecessor;
        Fixture f({producer, consumer});
        for (int lane : {1, 4, 2, 5})
            f.step(lane);
        f.drain();
        f.expect_completed(0);
        f.expect_completed(1);
        EXPECT_EQ(f.calls.size(), mix_producer ? 10U : 16U);
    }
}

TEST(PercounterAcquire, HeldSpmdCanBeSkippedBeforeAllBlocksAreClaimedAndRevisited) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2) GTEST_SKIP() << "Requires two generic slots.";
    auto external = task(1);
    external.published_off_lane = true;
    const int32_t dependency = 0;
    auto pending = task(2, 9);
    pending.fanin_count = 1;
    pending.fanin_ids = &dependency;
    Fixture f({external, pending, task(2, 2)});
    for (int i = 0; i < 4; ++i)
        f.step(1);
    EXPECT_LT(f.claim_total(0), 9U);
    ASSERT_FALSE(f.calls.empty());
    EXPECT_EQ(f.calls.front()[0], 2);
    f.counters[0].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 11U);
    for (int block = 0; block < 9; ++block)
        EXPECT_EQ(
            std::count_if(
                f.calls.begin(), f.calls.end(),
                [=](const auto &call) {
                    return call[0] == 1 && call[1] == block;
                }
            ),
            1
        );
}

TEST(PercounterAcquire, NonparticipantSnapshotsRemainExcludedDuringConcurrentReuse) {
    Fixture f({task(6)});
    auto *record = &f.signals[0].mix[0];
    std::atomic<bool> stop{false};
    std::atomic<bool> admitted{false};
    std::atomic<uint64_t> observations{0};
    percounter_ssbuf_store_field(&record->active_mask, uint32_t{6});
    percounter_ssbuf_store_release(&record->generation, 1);
    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            PercounterCandidate candidate;
            if (percounter_shared_candidate(f.cfg[0], 0, &candidate)) admitted.store(true, std::memory_order_relaxed);
            observations.fetch_add(1, std::memory_order_relaxed);
        }
    });
    while (observations.load(std::memory_order_relaxed) == 0) {}
    for (uint32_t i = 1; i <= 20'000; ++i) {
        percounter_ssbuf_store_release(&record->generation, 2 * i);
        percounter_ssbuf_store_field(&record->entry, percounter_order_entry(i, 0, 1));
        percounter_ssbuf_store_field(&record->global_order_id, i);
        percounter_ssbuf_store_field(&record->active_mask, uint32_t{6});
        percounter_ssbuf_store_field(&record->sequence, i);
        percounter_ssbuf_store_field(&record->sync_index, PERCOUNTER_NO_SYNC);
        percounter_ssbuf_store_release(&record->generation, 2 * i + 1);
    }
    stop.store(true, std::memory_order_release);
    reader.join();
    EXPECT_GT(observations.load(), 0U);
    EXPECT_FALSE(admitted.load());
}

TEST(PercounterAcquire, OldGenerationTokensCannotAdmitReusedRecord) {
    Fixture f({task(7, 2)}, 1);
    for (int lane : {1, 0, 2, 1, 2, 0, 1, 2, 1})
        f.step(lane);
    auto &record = f.signals[0].mix[0];
    ASSERT_EQ(record.generation, 3U);
    EXPECT_EQ(record.fanin_ready, 1U);
    for (int lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(record.enter_ready[lane], 1U);
        EXPECT_EQ(record.enter_idle[lane], 1U);
    }
    PercounterCandidate candidate;
    ASSERT_TRUE(percounter_shared_candidate(f.cfg[0], 0, &candidate));
    EXPECT_FALSE(candidate.ready);
    EXPECT_FALSE(percounter_tokens_ready(record.enter_ready, 7, record.generation));
    EXPECT_FALSE(percounter_tokens_ready(record.enter_idle, 7, record.generation));
    for (int pass = 0; pass < 16 && !f.completed(0); ++pass)
        for (int lane : {0, 1, 2})
            f.step(lane);
    f.expect_completed(0);
    EXPECT_EQ(f.calls.size(), 6U);
}

TEST(PercounterAcquire, ReturnedPeerRejectsReusedRecordBeforeItsPublicationSequence) {
    Fixture f({task(7, 2)}, 1);
    for (int lane : {1, 0, 2, 1, 2, 0, 1, 2, 1})
        f.step(lane);
    auto &record = f.signals[0].mix[0];
    ASSERT_EQ(record.generation, 3U);
    const uint32_t published = f.signals[0].publish_sequence;
    ASSERT_GT(published, f.state[2].seen_sequence);
    // A peer can read the replacement descriptor before its publication-sequence load catches up.
    f.signals[0].publish_sequence = f.state[2].seen_sequence;
    f.step(2);
    EXPECT_EQ(record.enter_ready[2], 1U);
    EXPECT_EQ(f.calls.size(), 3U);
    f.signals[0].publish_sequence = published;
    for (int pass = 0; pass < 16 && !f.completed(0); ++pass)
        for (int lane : {0, 1, 2})
            f.step(lane);
    EXPECT_EQ(f.calls.size(), 6U);
    f.expect_completed(0);
}

TEST(PercounterAcquire, ConcurrentSpmdClaimsRetainOneBlockPerTaskAndCore) {
    for (const bool sync : {false, true}) {
        Fixture f({task(1, sync ? 32 : 64, sync), task(1, 32)}, 32);
        std::array<PercounterLaneState, 32> states{};
        std::atomic<uint32_t> arrived{0};
        std::vector<std::thread> threads;
        for (uint32_t core = 0; core < states.size(); ++core) {
            threads.emplace_back([&, core] {
                auto cfg = f.cfg[0];
                cfg.cluster_index = core;
                percounter_claim(cfg, &states[core]);
                arrived.fetch_add(1, std::memory_order_acq_rel);
                while (arrived.load(std::memory_order_acquire) != states.size()) {}
                for (uint32_t retry = 0; retry < 3; ++retry)
                    percounter_claim(cfg, &states[core]);
            });
        }
        for (auto &thread : threads)
            thread.join();
        std::array<uint32_t, 32> owners{};
        for (const auto &state : states) {
            uint32_t held = 0;
            for (uint32_t slot = 0; slot < PERCOUNTER_PENDING_CAPACITY; ++slot) {
                if (!(state.occupied & (1U << slot))) continue;
                const uint64_t entry = state.slots[slot].entry;
                if (percounter_order_task_id(entry) != 0) continue;
                ++held;
                const uint32_t block = percounter_order_block_idx(entry);
                ASSERT_LT(block, owners.size());
                ++owners[block];
            }
            EXPECT_EQ(held, 1U);
        }
        for (const uint32_t count : owners)
            EXPECT_EQ(count, 1U);
        // The 32-block sync task is sized to the 32 owners, so it is assigned by
        // rank and draws no ticket; the 64-block one still draws one per owner.
        EXPECT_EQ(f.claim_total(0), sync ? 0U : 32U);
        EXPECT_EQ(f.claim_total(1), PERCOUNTER_PENDING_CAPACITY > 1 ? 32U : 0U);
    }
}

TEST(PercounterAcquire, PartialSyncLeavesOtherOwnersAbleToClaimOrdinarySpmd) {
    auto producer = task(2);
    producer.published_off_lane = true;
    int32_t predecessor = 0;
    auto sync = task(1, 8, true);
    sync.fanin_count = 1;
    sync.fanin_ids = &predecessor;
    Fixture f({producer, sync, task(1, 20), task(1, 32)}, 32);
    std::array<PercounterLaneState, 32> states{};
    auto claim = [&](uint32_t core) {
        auto cfg = f.cfg[0];
        cfg.cluster_index = core;
        percounter_claim(cfg, &states[core]);
    };
    for (uint32_t core = 0; core < 8; ++core)
        claim(core);
    for (uint32_t core = 8; core < 32; ++core)
        claim(core);
    uint32_t earlier = 0;
    for (uint32_t core = 8; core < 32; ++core) {
        ASSERT_EQ(states[core].occupied, 1U);
        earlier += percounter_order_task_id(states[core].slots[0].entry) == 2;
    }
    EXPECT_EQ(earlier, 20U);
    // The sync task is narrower than its 32 owners, so it keeps the ticket path
    // and its eight blocks are drawn rather than assigned.
    EXPECT_EQ(f.claim_total(0), 8U);
    EXPECT_EQ(f.claim_total(1), 20U);
    for (const auto &sector : f.cfg[0].sync_arrivals[0].sectors)
        EXPECT_EQ(sector.next, 0U);
}

TEST(PercounterAcquire, HeldSpmdBlocksLeaveItsRemainingBlocksForOtherLanes) {
    auto producer = task(1);
    producer.published_off_lane = true;
    auto wave = task(2, 4);
    const int32_t root = 0;
    wave.fanin_count = 1;
    wave.fanin_ids = &root;
    auto next = wave;
    next.block_num = 2;
    Fixture f({producer, wave, next});
    for (int lane : {1, 2, 1, 2, 4, 5, 1, 2})
        f.step(lane);
    EXPECT_TRUE(f.calls.empty());
    f.counters[0].value = 1;
    f.drain();
    ASSERT_EQ(f.calls.size(), 6U);
    uint32_t executing_lanes = 0;
    for (const auto &call : f.calls)
        if (call[0] == 1) executing_lanes |= 1U << call[2];
    EXPECT_EQ(__builtin_popcount(executing_lanes), 4);
}

TEST(PercounterAcquire, PendingWaveAllowsImmediateClaimOfDistinctOrdinaryTask) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "This case holds one source slot and requires a second free slot.";
    auto producer = task(1);
    producer.published_off_lane = true;
    const int32_t root = 0;
    auto pending = task(2, 2);
    pending.fanin_count = 1;
    pending.fanin_ids = &root;
    Fixture f({producer, pending, task(2, 2)});
    percounter_claim(f.cfg[1], &f.state[1]);
    percounter_claim(f.cfg[2], &f.state[2]);
    for (int lane : {1, 2}) {
        f.step(lane);
        uint32_t tasks = 0;
        for (uint32_t slot = 0; slot < PERCOUNTER_PENDING_CAPACITY; ++slot)
            if (f.state[lane].occupied & (1U << slot))
                tasks |= 1U << percounter_order_task_id(f.state[lane].slots[slot].entry);
        EXPECT_EQ(tasks, (1U << 1) | (1U << 2));
    }
    f.step(1);
    f.step(2);
    EXPECT_EQ(f.calls.size(), 2U);
    for (const auto &call : f.calls)
        EXPECT_EQ(call[0], 2);
    EXPECT_FALSE(f.completed(1));
    f.counters[0].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, PendingOrdinaryDoesNotPreventNextSyncTaskClaim) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "This case holds one source slot and requires a second free slot.";
    auto producer = task(1);
    producer.published_off_lane = true;
    const int32_t root = 0;
    auto pending = task(1, 2);
    pending.fanin_count = 1;
    pending.fanin_ids = &root;
    Fixture f({producer, pending, task(1, 2, true)});
    for (int lane : {0, 3, 0, 3})
        f.step(lane);
    // Both owners claim the sync task, which is sized to them and therefore
    // assigned by rank rather than by ticket. No ticket is drawn and no
    // exhaustion is recorded; the calls below are what witness both claims.
    EXPECT_EQ(f.claim_total(1), 0U);
    EXPECT_FALSE(percounter_claim_exhausted(percounter_claim_closed(f.cfg[0], 1)));
    for (int pass = 0; pass < 4; ++pass)
        for (int lane : {0, 3})
            f.step(lane);
    EXPECT_EQ(f.calls.size(), 2U);
    for (const auto &call : f.calls)
        EXPECT_EQ(call[0], 2);
    EXPECT_FALSE(f.completed(1));
    f.counters[0].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
}

TEST(PercounterAcquire, PublisherReusesMixSlotWhilePreviousPeerKernelRuns) {
    Fixture f({task(7, 2)}, 1);
    bool reused_during_kernel = false;
    const auto original = execute;
    execute = [&](int64_t *args) {
        const int lane = f.running_lane;
        original(args);
        if (lane != 0 || f.calls.back()[1] != 0) return;
        f.step(1);
        f.step(1);
        EXPECT_EQ(f.claim_total(0), 2U);
        EXPECT_EQ(percounter_order_block_idx(f.signals[0].mix[0].entry), 1U);
        EXPECT_EQ(f.calls.size(), 2U);
        EXPECT_FALSE(f.completed(0));
        reused_during_kernel = true;
        f.running_lane = lane;
        percounter_bind_sim_context(&f.signals[0].contexts[lane].value);
    };
    // AIC supplies the last idle token and enters before AIV1 observes it.
    for (int lane : {1, 0, 2, 1, 2, 0})
        f.step(lane);
    ASSERT_TRUE(reused_during_kernel);
    for (int pass = 0; pass < 16 && !f.completed(0); ++pass)
        for (int lane : {0, 1, 2})
            f.step(lane);
    f.expect_completed(0);
    for (int block = 0; block < 2; ++block)
        for (int lane = 0; lane < 3; ++lane)
            EXPECT_EQ(std::count(f.calls.begin(), f.calls.end(), std::array<int64_t, 3>{0, block, lane}), 1);
}

TEST(PercounterAcquire, ReadyMixWaitingForPeerDoesNotBlockNextMixClaim) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "This case holds one source slot and requires a second free slot.";
    Fixture f({task(7), task(7)});
    f.step(1);
    f.step(0);
    f.step(1);
    EXPECT_EQ(__builtin_popcount(f.state[1].mix_owned), 2);
    EXPECT_TRUE(f.calls.empty());
    f.drain();
    EXPECT_EQ(f.calls.size(), 6U);
}

void expect_mix_revisit_before_successor(uint32_t mask, const std::array<int, 3> &peers, bool sync_successor) {
    const int32_t predecessor = 0;
    const uint32_t participants = static_cast<uint32_t>(__builtin_popcount(mask));
    const int clusters = sync_successor ? 2 : 1;
    const int predecessor_blocks = clusters + 1;
    auto successor = task(mask, clusters, sync_successor);
    successor.fanin_count = 1;
    successor.fanin_ids = &predecessor;
    Fixture f({task(mask, predecessor_blocks), successor}, clusters);
    const auto original = execute;
    execute = [&](int64_t *args) {
        if (args[0] == 1) EXPECT_TRUE(f.completed(0));
        original(args);
    };

    // Peers have not run yet, so the coordinator can claim both distinct tasks.
    f.step(1);
    f.step(1);
    ASSERT_TRUE(f.calls.empty());
    ASSERT_EQ(__builtin_popcount(f.state[1].mix_owned), 2);
    const uint32_t first_generation = f.signals[0].mix[0].generation;
    ASSERT_EQ(percounter_order_task_id(f.signals[0].mix[0].entry), 0);
    ASSERT_EQ(percounter_order_task_id(f.signals[0].mix[1].entry), 1);

    // Stop when the publisher can revisit the slot. Some peers may still hold the first launch.
    for (int step = 0; step < 96 && f.signals[0].mix[0].generation == first_generation; ++step)
        f.step(peers[step % 3]);
    ASSERT_NE(f.signals[0].mix[0].generation, first_generation);
    ASSERT_LE(f.calls.size(), participants);
    for (const auto &call : f.calls) {
        ASSERT_EQ(call[0], 0);
        ASSERT_EQ(call[1], 0);
    }
    ASSERT_FALSE(f.completed(0));

    const size_t returned = f.calls.size();
    if (!percounter_mix_armed(f.signals[0].mix[0].generation)) f.step(1);
    const auto &revisit = f.signals[0].mix[0];
    const auto &later = f.signals[0].mix[1];
    ASSERT_EQ(percounter_order_task_id(revisit.entry), 0);
    ASSERT_EQ(percounter_order_block_idx(revisit.entry), 1);
    ASSERT_LT(revisit.global_order_id, later.global_order_id);
    ASSERT_GT(revisit.sequence, later.sequence);
    ASSERT_NE(revisit.generation, first_generation);
    ASSERT_EQ(f.calls.size(), returned);

    for (int step = 0; step < 192 && !f.completed(1); ++step)
        f.step(peers[step % 3] + 3 * ((step / 3) % clusters));
    f.expect_completed(0);
    f.expect_completed(1);
    const int total_blocks = predecessor_blocks + clusters;
    EXPECT_EQ(f.calls.size(), total_blocks * participants);
    std::array<std::array<uint32_t, 3>, 5> invocations{};
    for (const auto &call : f.calls) {
        const int64_t invocation = call[0] == 0 ? call[1] : predecessor_blocks + call[1];
        ASSERT_GE(invocation, 0);
        ASSERT_LT(invocation, total_blocks);
        ASSERT_GE(call[2], 0);
        ASSERT_LT(call[2], 3 * clusters);
        ++invocations[invocation][call[2] % 3];
    }
    for (int block = 0; block < total_blocks; ++block)
        for (uint32_t lane = 0; lane < 3; ++lane)
            EXPECT_EQ(invocations[block][lane], (mask >> lane) & 1U);
}

TEST(PercounterAcquire, RetiredMixBlockRevisitPrecedesItsPublishedSuccessor) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "The successor must occupy a second position before the predecessor retires.";
    expect_mix_revisit_before_successor(7, {0, 1, 2}, false);
}

TEST(PercounterAcquire, MixRevisitPreservesOrderAcrossMasksPeersAndSyncSuccessors) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "The successor must occupy a second position before the predecessor retires.";
    for (uint32_t mask : {3U, 5U, 6U, 7U}) {
        for (bool sync : {false, true}) {
            std::array<int, 3> peers{0, 1, 2};
            do {
                SCOPED_TRACE(
                    ::testing::Message() << "mask=" << mask << " sync=" << sync << " peers=" << peers[0] << peers[1]
                                         << peers[2]
                );
                expect_mix_revisit_before_successor(mask, peers, sync);
                if (HasFatalFailure()) return;
            } while (std::next_permutation(peers.begin(), peers.end()));
        }
    }
}

TEST(PercounterAcquire, CommittedSyncWaitingForOtherBlockDoesNotDelayReadyOrdinary) {
    Fixture f({task(7, 2, true), task(2)});
    percounter_claim(f.cfg[2], &f.state[2]);
    for (int lane : {1, 0, 1})
        f.step(lane);
    ASSERT_TRUE(f.calls.empty());
    f.step(2);
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls.front()[0], 1);
    f.drain();
    EXPECT_EQ(f.calls.size(), 7U);
}

TEST(PercounterAcquire, OrdinaryClaimLeavesNextPositionForAnotherLane) {
    Fixture f({task(1), task(1), task(1), task(1), task(1)});
    percounter_claim(f.cfg[0], &f.state[0]);
    percounter_claim(f.cfg[3], &f.state[3]);

    EXPECT_EQ(__builtin_popcount(f.state[0].occupied), 1);
    EXPECT_EQ(__builtin_popcount(f.state[3].occupied), 1);
    EXPECT_EQ(percounter_order_task_id(f.state[0].slots[0].entry), 0);
    EXPECT_EQ(percounter_order_task_id(f.state[3].slots[0].entry), 1);
    f.drain();
    EXPECT_EQ(f.calls.size(), 5U);
}

TEST(PercounterAcquire, ReturnedPeerRunsOrdinaryTaskWhilePublisherRunsMix) {
    for (int peer : {0, 2}) {
        const int32_t producer = 0;
        auto external = task(1);
        external.published_off_lane = true;
        auto ordinary = task(peer == 0 ? 1 : 2);
        ordinary.fanin_count = 1;
        ordinary.fanin_ids = &producer;
        Fixture f({external, task(peer == 0 ? 3 : 6), ordinary}, 1);
        const auto original = execute;
        bool ran_ordinary = false;
        execute = [&](int64_t *args) {
            const int lane = f.running_lane;
            original(args);
            if (lane != 1 || args[0] != 1) return;
            ASSERT_EQ(f.signals[0].mix[0].generation, 1U);
            const std::array<int64_t, 3> mix_call{1, 0, peer};
            if (std::find(f.calls.begin(), f.calls.end(), mix_call) == f.calls.end()) f.step(peer);
            ASSERT_EQ(std::count(f.calls.begin(), f.calls.end(), mix_call), 1);
            const size_t returned = f.calls.size();
            f.counters[0].value = 1;
            f.step(peer);
            ASSERT_EQ(f.calls.size(), returned + 1);
            EXPECT_EQ(f.calls.back(), (std::array<int64_t, 3>{2, 0, peer}));
            EXPECT_FALSE(f.completed(1));
            ran_ordinary = true;
            f.running_lane = lane;
            percounter_bind_sim_context(&f.signals[0].contexts[lane].value);
        };
        for (int lane : {1, peer, 1, peer, 1})
            f.step(lane);
        ASSERT_TRUE(ran_ordinary);
        f.expect_completed(2);
        f.expect_completed(1);
        EXPECT_EQ(f.calls.size(), 3U);
    }
}

TEST(PercounterAcquire, WaitingTasksKeepIndependentImmutableArgumentVectors) {
    auto producer = task(1);
    producer.published_off_lane = true;
    const int32_t producer_id = 0;
    auto consumer = task(2);
    consumer.fanin_count = 1;
    consumer.fanin_ids = &producer_id;
    Fixture f({producer, consumer, consumer});
    f.step(1);
    f.step(1);
    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.arguments[1].args[0], 1);
    EXPECT_EQ(f.arguments[2].args[0], 2);
    std::vector<int64_t *> addresses;
    const auto original = execute;
    execute = [&](int64_t *args) {
        addresses.push_back(args);
        original(args);
    };
    f.counters[0].value = 1;
    f.step(1);
    f.step(1);
    ASSERT_EQ(addresses.size(), 2U);
    EXPECT_NE(addresses[0], addresses[1]);
    EXPECT_EQ(addresses[0], reinterpret_cast<int64_t *>(f.arguments[1].args));
    EXPECT_EQ(addresses[1], reinterpret_cast<int64_t *>(f.arguments[2].args));
    f.expect_completed(1);
    f.expect_completed(2);
}

TEST(PercounterAcquire, LongKernelLookaheadRevisitsRemainingEarlierBlocks) {
    auto producer = task(2);
    producer.published_off_lane = true;
    const int32_t root = 0;
    auto wave = task(1, 5);
    wave.fanin_count = 1;
    wave.fanin_ids = &root;
    const int32_t wave_id = 1;
    auto successor = task(1);
    successor.fanin_count = 1;
    successor.fanin_ids = &wave_id;
    Fixture f({producer, wave, successor, successor});
    f.step(3);
    f.step(3);
    f.step(0);
    f.step(0);
    f.counters[0].value = 1;
    f.step(0);
    f.step(3);
    EXPECT_EQ(f.calls.size(), 2U);
    EXPECT_FALSE(f.completed(1));
    f.drain();
    EXPECT_EQ(f.calls.size(), 7U);
    f.expect_completed(1);
}

TEST(PercounterAcquire, DependencyOnlyBlockWaitsForFaninWithoutCallingKernel) {
    auto producer = task(1);
    producer.published_off_lane = true;
    TaskInput dependency{};
    dependency.core_type_index = 1;
    dependency.subtask_count = 1;
    const int32_t producer_id = 0;
    dependency.fanin_count = 1;
    dependency.fanin_ids = &producer_id;
    auto consumer = task(1);
    const int32_t dependency_id = 1;
    consumer.fanin_count = 1;
    consumer.fanin_ids = &dependency_id;
    Fixture f({producer, dependency, consumer});
    for (int round = 0; round < 5; ++round)
        for (int lane = 0; lane < 6; ++lane)
            f.step(lane);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.counters[1].value, percounter_counter_preset(1));
    f.counters[0].value = 1;
    f.drain();
    ASSERT_EQ(f.calls.size(), 1U);
    EXPECT_EQ(f.calls.front()[0], 2);
    f.expect_completed(1);
    f.expect_completed(2);
}

TEST(PercounterAcquire, FoldedJoinStillWaitsForOffLaneCompletion) {
    auto remote = task(2);
    remote.published_off_lane = true;
    const int32_t prerequisites[] = {0, 1};
    auto join = task(0);
    join.subtask_count = 1;
    join.fanin_count = 2;
    join.fanin_ids = prerequisites;
    const int32_t join_id = 2;
    auto consumer = task(2);
    consumer.fanin_count = 1;
    consumer.fanin_ids = &join_id;
    Fixture f({remote, task(1), join, consumer}, 2, {}, true);

    ASSERT_TRUE(f.inputs[2].inline_completed);
    ASSERT_EQ(f.inputs[3].fanin_count, 2);
    EXPECT_EQ(f.inputs[3].fanin_ids[0], 0);
    EXPECT_EQ(f.inputs[3].fanin_ids[1], 1);
    EXPECT_EQ(f.control->order_count[1], 1U);
    f.drain_until(
        [&] {
            return f.completed(1);
        },
        "local predecessor"
    );
    f.step(1);
    f.step(2);
    EXPECT_FALSE(f.completed(3));
    f.counters[0].value = PERCOUNTER_COUNTER_PUBLISHED;
    f.drain();
    f.expect_completed(3);
    EXPECT_EQ(f.calls.size(), 2U);
}

TEST(PercounterAcquire, FoldedTerminalJoinRetainsUnsettledPredecessor) {
    auto remote = task(2);
    remote.published_off_lane = true;
    const int32_t producer = 0;
    auto join = task(0);
    join.subtask_count = 1;
    join.fanin_count = 1;
    join.fanin_ids = &producer;
    Fixture f({remote, join}, 2, {}, true);

    ASSERT_TRUE(f.inputs[1].inline_completed);
    EXPECT_EQ(f.control->order_count[1], 0U);
    EXPECT_FALSE(f.sinks_complete());
    f.counters[0].value = PERCOUNTER_COUNTER_PUBLISHED;
    EXPECT_TRUE(f.sinks_complete());
}

TEST(PercounterAcquire, DependencyJoinWithMultipleConsumersRemainsScheduled) {
    const int32_t producer = 0;
    auto join = task(0);
    join.subtask_count = 1;
    join.fanin_count = 1;
    join.fanin_ids = &producer;
    const int32_t join_id = 1;
    auto consumer = task(2);
    consumer.fanin_count = 1;
    consumer.fanin_ids = &join_id;
    Fixture f({task(1), join, consumer, consumer}, 2, {}, true);
    EXPECT_FALSE(f.inputs[1].inline_completed);
    EXPECT_EQ(f.control->order_count[1], 3U);
    f.drain();
    f.expect_completed(2);
    f.expect_completed(3);
}

TEST(PercounterAcquire, DependencyOnlySpmdRetiresAndRevisitsWithoutPayload) {
    auto dependency = task(0, 5);
    dependency.subtask_count = 1;
    Fixture f({dependency});
    f.drain();
    EXPECT_TRUE(f.calls.empty());
    f.expect_completed(0);
}

TEST(PercounterAcquire, ParticipatingPublisherRetiresWithoutRevisitingReturnedCube) {
    for (uint32_t mask : {3U, 7U}) {
        Fixture f({task(mask)}, 1);
        bool cube_returned = false;
        for (int round = 0; round < 24 && !f.completed(0); ++round) {
            if (!cube_returned) {
                f.step(0);
                cube_returned = std::any_of(f.calls.begin(), f.calls.end(), [](const auto &call) {
                    return call[2] == 0;
                });
            }
            f.step(1);
            if (mask & 4U) f.step(2);
        }
        ASSERT_TRUE(cube_returned);
        f.expect_completed(0);
        EXPECT_EQ(f.calls.size(), static_cast<size_t>(__builtin_popcount(mask)));
        EXPECT_EQ(f.state[1].mix_owned, 0U);
        EXPECT_EQ(f.signals[0].mix[0].generation, 2U);
    }
}

TEST(PercounterAcquire, NonparticipantCoordinatorIsNotNeededForExecutionOrCompletion) {
    Fixture f({task(5)});
    f.step(1);
    for (int round = 0; round < 5 && !f.completed(0); ++round) {
        f.step(0);
        f.step(2);
    }
    f.expect_completed(0);
    EXPECT_EQ(f.calls.size(), 2U);
    EXPECT_EQ(f.signals[0].mix[0].generation, 1U);
    f.step(1);
    EXPECT_EQ(f.signals[0].mix[0].generation, 2U);
}

TEST(PercounterAcquire, BlockIndexDoesNotImposeCohortOrder) {
    Fixture f({task(7, 2)});
    percounter_claim(f.cfg[1], &f.state[1]);
    percounter_claim(f.cfg[4], &f.state[4]);
    for (int round = 0; round < 4 && f.calls.empty(); ++round)
        for (int lane : {3, 4, 5})
            f.step(lane);
    ASSERT_FALSE(f.calls.empty());
    EXPECT_EQ(f.calls.front()[1], 1);
    f.drain();
    EXPECT_EQ(f.calls.size(), 6U);
    f.expect_completed(0);
}

TEST(PercounterAcquire, InvalidCallableCannotPublishEntryCommitment) {
    Fixture f({task(7)});
    f.step(1);
    f.counters[0].function_addresses[0] = 0;
    bool executed = false;
    EXPECT_FALSE(percounter_step(f.cfg[0], &f.state[0], &executed));
    EXPECT_EQ(f.control->lane_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(f.signals[0].mix[0].enter_ready[0], 0U);
    percounter_lane_loop<2>(f.cfg[2]);
    EXPECT_TRUE(f.calls.empty());
}

TEST(PercounterAcquire, ExitWhileWaitingForCohortDoesNotEnterKernel) {
    Fixture f({task(7)});
    f.step(1);
    bool exit_published = false;
    on_poll = [&] {
        if (f.signals[0].mix[0].enter_ready[0] == f.signals[0].mix[0].generation) {
            exit_published = true;
            on_poll = {};
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        }
    };
    percounter_lane_loop<0>(f.cfg[0]);
    EXPECT_TRUE(exit_published);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.control->lane_error, 0U);
    EXPECT_EQ(f.signals[0].mix[0].enter_ready[0], 1U);
    EXPECT_EQ(f.signals[0].mix[0].enter_idle[0], 0U);
    write_reg(RegId::DATA_MAIN_BASE, 0);
}

TEST(PercounterAcquire, QuiescentLaneWaitsForDelayedMixPublicationAndRetirement) {
    auto external = task(1);
    external.published_off_lane = true;
    const int32_t producer = 1;
    auto vector = task(2);
    vector.fanin_ids = &producer;
    vector.fanin_count = 1;
    Fixture f({task(7), external, vector});
    bool publisher_advanced = false;
    on_poll = [&] {
        publisher_advanced = true;
        f.step(1);
        f.step(2);
        f.running_lane = 0;
    };
    f.running_lane = 0;
    percounter_lane_loop<0>(f.cfg[0]);
    on_poll = {};
    ASSERT_EQ(f.control->lane_error, 0U);
    EXPECT_TRUE(publisher_advanced);
    EXPECT_EQ(f.signals[0].publication_closed, 1U);
    ASSERT_EQ(f.calls.size(), 3U);
    f.expect_completed(0);
    EXPECT_FALSE(f.completed(2));
    f.counters[1].value = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 4U);
    f.expect_completed(2);
}

TEST(PercounterAcquire, QuiescentLaneSettlesOwnDeferredRecordBeforeReturning) {
    const int32_t producer = 0;
    auto vector = task(2);
    vector.fanin_ids = &producer;
    vector.fanin_count = 1;
    Fixture f({task(1), vector});
    int32_t signal = 0;
    const auto original = execute;
    execute = [&](int64_t *args) {
        original(args);
        if (args[0] == 0) {
            f.async_entries[0][0] = {
                reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            f.async_groups[0].count = 1;
        }
    };
    bool pending_observed = false;
    on_poll = [&] {
        if (f.control->async_registered != 0) {
            EXPECT_EQ(f.control->async_settled, 0U);
            EXPECT_FALSE(f.completed(0));
            pending_observed = true;
            signal = 1;
            on_poll = {};
        }
    };
    f.running_lane = 0;
    percounter_lane_loop<0>(f.cfg[0]);
    on_poll = {};
    ASSERT_EQ(f.control->lane_error, 0U);
    EXPECT_TRUE(pending_observed);
    EXPECT_EQ(f.control->async_registered, 1U);
    EXPECT_EQ(f.control->async_settled, 1U);
    EXPECT_EQ(f.calls.size(), 1U);
    f.expect_completed(0);
    EXPECT_FALSE(f.completed(1));
    f.drain();
    EXPECT_EQ(f.calls.size(), 2U);
    f.expect_completed(1);
}

TEST(PercounterAcquire, SupervisorExitDoesNotSettlePendingDeferredCompletion) {
    const int32_t producer = 0;
    auto vector = task(2);
    vector.fanin_ids = &producer;
    vector.fanin_count = 1;
    Fixture f({task(1), vector});
    int32_t signal = 0;
    const auto original = execute;
    execute = [&](int64_t *args) {
        original(args);
        f.async_entries[0][0] = {
            reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
        };
        f.async_groups[0].count = 1;
    };
    bool pending_observed = false;
    on_poll = [&] {
        if (f.control->async_registered != 0) {
            EXPECT_EQ(f.control->async_settled, 0U);
            EXPECT_FALSE(f.completed(0));
            pending_observed = true;
            on_poll = {};
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        }
    };
    f.running_lane = 0;
    percounter_lane_loop<0>(f.cfg[0]);
    on_poll = {};
    write_reg(RegId::DATA_MAIN_BASE, 0);
    EXPECT_TRUE(pending_observed);
    EXPECT_EQ(f.control->lane_error, 0U);
    EXPECT_EQ(f.control->async_registered, 1U);
    EXPECT_EQ(f.control->async_settled, 0U);
    EXPECT_FALSE(f.completed(0));
    EXPECT_FALSE(f.completed(1));
    EXPECT_EQ(f.calls.size(), 1U);
}

TEST(PercounterAcquire, MixPublicationClosureIsResetForTheNextInvocation) {
    Fixture f({task(7)});
    f.signals[0].publication_closed = 1;
    uint64_t ready[2]{};
    f.cfg[1].mix_signal_ready = ready;
    EXPECT_TRUE(percounter_clear_mix_signal(f.cfg[1], 50'000'000ULL));
    EXPECT_EQ(f.signals[0].publication_closed, 0U);
    EXPECT_EQ(ready[0], 1U);
    f.drain();
    f.expect_completed(0);
}

TEST(PercounterAcquire, LocallyExhaustedLaneReturnsBeforeOtherCoreTypeFinishes) {
    const int32_t producer = 0;
    auto vector = task(2);
    vector.fanin_ids = &producer;
    vector.fanin_count = 1;
    Fixture f({task(1), vector});
    f.running_lane = 0;
    percounter_lane_loop<0>(f.cfg[0]);
    ASSERT_EQ(f.control->lane_error, 0U);
    ASSERT_EQ(f.calls.size(), 1U);
    f.expect_completed(0);
    EXPECT_FALSE(f.completed(1));
    f.drain();
    EXPECT_EQ(f.calls.size(), 2U);
    f.expect_completed(1);
}

TEST(PercounterAcquire, ExitBeforeEntryDoesNotClaimWork) {
    Fixture f;
    write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    percounter_lane_loop<0>(f.cfg[0]);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.cfg[0].tickets[0].next, 0U);
    write_reg(RegId::DATA_MAIN_BASE, 0);
}

TEST(PercounterAcquire, ExitAfterKernelStopsActiveLane) {
    Fixture f(std::vector<TaskInput>(256, task(1)));
    const auto original = execute;
    execute = [&](int64_t *args) {
        original(args);
        if (args[0] == 1) write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    };
    percounter_lane_loop<0>(f.cfg[0]);
    write_reg(RegId::DATA_MAIN_BASE, 0);
    ASSERT_GE(f.calls.size(), 2U);
    ASSERT_LT(f.calls.size(), 256U);
    EXPECT_EQ(f.calls[0][0], 0);
    EXPECT_EQ(f.calls[1][0], 1);
    f.expect_completed(0);
    f.expect_completed(1);
    EXPECT_FALSE(f.completed(255));
    EXPECT_EQ(f.control->lane_error, 0U);
}

TEST(PercounterAcquire, LaneErrorWhileWaitingForCohortStopsLane) {
    Fixture f({task(7)});
    f.step(1);
    bool error_published = false;
    on_poll = [&] {
        if (f.signals[0].mix[0].enter_ready[0] == f.signals[0].mix[0].generation) {
            error_published = true;
            on_poll = {};
            f.control->lane_error = static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS);
        }
    };
    percounter_lane_loop<0>(f.cfg[0]);
    EXPECT_TRUE(error_published);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(f.signals[0].mix[0].enter_idle[0], 0U);
    EXPECT_EQ(f.control->lane_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
}

TEST(PercounterAcquire, ExitAfterKernelStopsLaneWaitingForMixPeers) {
    const int32_t root = 0;
    const int32_t predecessor = 1;
    auto second = task(1);
    second.fanin_ids = &root;
    second.fanin_count = 1;
    auto successor = task(7);
    successor.fanin_ids = &predecessor;
    successor.fanin_count = 1;
    Fixture f({task(1), second, successor});
    const auto original = execute;
    execute = [&](int64_t *args) {
        original(args);
        if (args[0] == 1) {
            f.step(1);
            f.step(2);
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        }
    };
    percounter_lane_loop<0>(f.cfg[0]);
    write_reg(RegId::DATA_MAIN_BASE, 0);
    ASSERT_EQ(f.calls.size(), 2U);
    EXPECT_NE(f.signals[0].mix[0].generation, 0U);
    EXPECT_EQ(f.signals[0].mix[0].enter_idle[0], 0U);
    EXPECT_FALSE(f.completed(2));
    EXPECT_EQ(f.control->lane_error, 0U);
}

TEST(PercounterPayload, TaskTimingPreparesIdentityAndBatchesCommittedCount) {
    Fixture f;
    auto buffer = std::make_unique<ChipSwimlaneAicoreTaskBuffer>();
    ChipSwimlaneActiveHead head{};
    head.current_buf_ptr = reinterpret_cast<uint64_t>(buffer.get());
    ChipSwimlaneAicoreLocalState profile{};
    f.cfg[0].profile_head = &head;
    f.cfg[0].profile = &profile;
    auto task = f.resolve(0);
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
    };
    for (uint32_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(f.run(0, task));
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
        ASSERT_TRUE(f.run(0, task));
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
    for (int i = 0; i < PLATFORM_AICORE_BUFFER_SIZE; ++i) {
        EXPECT_EQ(first->count, 0U);
        ASSERT_TRUE(f.run(0, task));
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

    ASSERT_TRUE(f.run(0, task));

    percounter_commit_task_timing(f.cfg[0]);
    EXPECT_EQ(second->count, 0U);
    EXPECT_EQ(second->records[0].reg_task_id, PLATFORM_AICORE_BUFFER_SIZE + 1U);
    EXPECT_EQ(TaskId::to_uint64(second->records[0].task_token), TaskId::to_uint64(TaskId::make_global(0)));
    percounter_flush_stream_records(&profile);
    EXPECT_EQ(second->count, 1U);
    EXPECT_EQ(first->count, 17U);
}
TEST(PercounterAcquire, ReturnedDeferredBlockReleasesSlotBeforeLogicalCompletion) {
    for (uint32_t mask : {1U, 2U}) {
        Fixture f({task(mask), task(mask), task(mask), task(mask)});
        const int lane = mask == 1 ? 0 : 1;
        // The test driver supplies the peer completion; no local task signals it.
        int32_t signal = 0;
        std::vector<int64_t> ran;
        execute = [&](int64_t *args) {
            ran.push_back(args[0]);
            if (args[0] == 0) {
                f.async_entries[lane][0] = {
                    reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                };
                f.async_groups[lane].count = 1;
            }
        };
        for (int tick = 0; tick < 32; ++tick)
            f.step(lane);
        EXPECT_EQ(ran, (std::vector<int64_t>{0, 1, 2, 3}));
        EXPECT_EQ(f.state[lane].occupied, 0U);
        EXPECT_EQ(f.state[lane].committed, -1);
        EXPECT_NE(f.state[lane].async_head, 0U);
        EXPECT_FALSE(f.completed(0));
        for (int id = 1; id < 4; ++id)
            EXPECT_TRUE(f.completed(id));
        signal = 1;
        f.step(lane);
        EXPECT_EQ(f.state[lane].async_head, 0U);
        f.expect_completed(0);
        EXPECT_EQ(f.control->async_registered, f.control->async_settled);
        EXPECT_EQ(ran.size(), 4U);
    }
}

TEST(PercounterAcquire, ReturnedPeerWaitLeavesCapacityForIndependentSpmd) {
    for (bool sync : {false, true}) {
        Fixture f({task(1), task(1, 2, sync)});
        int32_t peer_signal = 0;
        const auto record_execution = execute;
        execute = [&](int64_t *args) {
            record_execution(args);
            if (args[0] == 0) {
                f.async_entries[f.running_lane][0] = {
                    reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                };
                f.async_groups[f.running_lane].count = 1;
            }
        };
        for (int tick = 0; tick < 64; ++tick) {
            f.step(0);
            f.step(3);
        }
        EXPECT_FALSE(f.completed(0));
        f.expect_completed(1);
        EXPECT_EQ(f.calls.size(), 3U);
        EXPECT_EQ(f.state[0].occupied | f.state[3].occupied, 0U);
        peer_signal = 1;
        f.drain();
        EXPECT_EQ(f.calls.size(), 3U);
        EXPECT_EQ(f.control->async_registered, f.control->async_settled);
    }
}

TEST(PercounterAcquire, PeerCompletionWaitDoesNotPreventIndependentSyncClaims) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "This case holds one source slot and requires a second free slot.";
    const int32_t producer = 0;
    auto consumer = task(1);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({task(1), consumer, consumer, task(1, 2, true)});
    int32_t peer_signal = 0;
    const auto record_execution = execute;
    execute = [&](int64_t *args) {
        record_execution(args);
        if (args[0] == 0) {
            f.async_entries[f.running_lane][0] = {
                reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            f.async_groups[f.running_lane].count = 1;
        }
    };
    for (int tick = 0; tick < 64; ++tick) {
        f.step(0);
        f.step(3);
    }
    EXPECT_FALSE(f.completed(0));
    EXPECT_FALSE(f.completed(1));
    EXPECT_FALSE(f.completed(2));
    f.expect_completed(3);
    EXPECT_EQ(f.calls.size(), 3U);
    // The test driver supplies a peer-card completion, not a local notifier task.
    peer_signal = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 5U);
    EXPECT_EQ(f.control->async_registered, f.control->async_settled);
}

TEST(PercounterAcquire, PeerCompletionWaitDoesNotPreventIndependentOrdinarySpmd) {
    if constexpr (PERCOUNTER_PENDING_CAPACITY < 2)
        GTEST_SKIP() << "This case holds one source slot and requires a second free slot.";
    const int32_t producer = 0;
    auto consumer = task(1, 2);
    consumer.fanin_ids = &producer;
    consumer.fanin_count = 1;
    Fixture f({task(1), consumer, task(1, 2)});
    int32_t peer_signal = 0;
    const auto record_execution = execute;
    execute = [&](int64_t *args) {
        record_execution(args);
        if (args[0] == 0) {
            f.async_entries[f.running_lane][0] = {
                reinterpret_cast<uint64_t>(&peer_signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            f.async_groups[f.running_lane].count = 1;
        }
    };
    for (int tick = 0; tick < 64; ++tick) {
        f.step(0);
        f.step(3);
    }
    EXPECT_FALSE(f.completed(0));
    EXPECT_FALSE(f.completed(1));
    f.expect_completed(2);
    EXPECT_EQ(f.calls.size(), 3U);
    peer_signal = 1;
    f.drain();
    EXPECT_EQ(f.calls.size(), 5U);
    EXPECT_EQ(f.control->async_registered, f.control->async_settled);
}

TEST(PercounterAcquire, DeferredMixShareOutlivesRecordReuseWithoutEarlyCompletion) {
    for (uint32_t mask : {5U, 6U, 7U}) {
        Fixture f({task(mask), task(mask)});
        int32_t signal = 0;
        const uint32_t observer = __builtin_ctz(mask);
        execute = [&](int64_t *args) {
            if (args[0] == 0 && static_cast<uint32_t>(f.running_lane) == observer) {
                f.async_entries[observer][0] = {
                    reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                };
                f.async_groups[observer].count = 1;
            }
        };
        for (int tick = 0; tick < 64; ++tick)
            for (int lane = 0; lane < 3; ++lane)
                f.step(lane);
        EXPECT_FALSE(f.completed(0)) << mask;
        EXPECT_TRUE(f.completed(1)) << mask;
        EXPECT_EQ(f.state[1].mix_owned, 0U);
        signal = 1;
        for (int lane = 0; lane < 3; ++lane)
            f.step(lane);
        EXPECT_TRUE(f.completed(0)) << mask;
        EXPECT_EQ(f.control->async_registered, f.control->async_settled);
    }
}

TEST(PercounterPayload, TerminalDeferralKeepsLaneLiveUntilConditionHolds) {
    Fixture f;
    int32_t signal = 0;
    f.counters[0].value = percounter_counter_preset(1);
    auto task = f.resolve(0);
    execute = [&](int64_t *) {
        f.async_entries[0][0] = {
            reinterpret_cast<uint64_t>(&signal), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
        };
        f.async_groups[0].count = 1;
    };
    bool deferred = false;
    ASSERT_TRUE(percounter_run_task(
        f.cfg[0], &f.state[0], task, 0, 1, 0, percounter_completion_address(f.cfg[0], task.id), &deferred,
        percounter_order_entry(task.id, 0, 1)
    ));
    ASSERT_TRUE(deferred);
    EXPECT_NE(f.control->async_registered, f.control->async_settled);
    bool settled = false;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_FALSE(settled);
    EXPECT_FALSE(f.completed(0));
    signal = 1;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_TRUE(settled);
    EXPECT_TRUE(f.completed(0));
    EXPECT_EQ(f.control->async_registered, f.control->async_settled);
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
        f.counters[current].value = percounter_counter_preset(1);
        auto task = f.resolve(current);
        ASSERT_TRUE(f.run(0, task));
        percounter_commit_task_timing(f.cfg[0]);
    }
    signals[1] = 1;
    bool settled = false;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    ASSERT_TRUE(settled);
    EXPECT_FALSE(f.completed(0));
    EXPECT_TRUE(f.completed(1));
    EXPECT_FALSE(f.completed(2));
    signals[0] = signals[2] = 1;
    ASSERT_TRUE(percounter_async_tick(f.cfg[0], &f.state[0], &settled));
    EXPECT_EQ(f.state[0].async_head, 0U);
    EXPECT_EQ(f.control->async_registered, 3U);
    EXPECT_EQ(f.control->async_settled, 3U);
}

}  // namespace

volatile uint8_t *sim_get_reg_base() {
    if (on_poll) on_poll();
    return registers.data();
}
uint32_t sim_get_physical_core_id() { return 0; }
uint64_t platform_ssbuf_hardware_base_address() { return reinterpret_cast<uint64_t>(startup_ssbuf.data()); }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }

TEST(PercounterScalarStorage, PreservesMixedIntervalsWithoutLogicalCount) {
    std::array<uint64_t, 8> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 100, 117);
    trace.write({200, uint64_t{UINT32_MAX} + 300, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 7});
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
    trace.write({100, 200, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 1});
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
    EXPECT_EQ(words[0], UINT64_C(0x031b000500000007));
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

TEST(PercounterPayload, ArgumentsAreSharedAcrossBlocksAndContextsFollowSelectedExecution) {
    Fixture f({task(1, 5)});
    auto task = f.resolve(0);
    auto *arguments = f.arguments[0].args;
    std::vector<int64_t *> addresses;
    execute = [&](int64_t *args) {
        addresses.push_back(args);
        EXPECT_EQ(get_block_idx(args), 3);
        EXPECT_EQ(get_block_num(args), 5);
        EXPECT_EQ(args[0], 0);
    };
    ASSERT_TRUE(f.run(0, task, 3));
    ASSERT_TRUE(f.run(3, task, 3));
    ASSERT_EQ(addresses.size(), 2U);
    EXPECT_EQ(addresses[0], reinterpret_cast<int64_t *>(arguments));
    EXPECT_EQ(addresses[0], addresses[1]);
    EXPECT_EQ(f.signals[0].contexts[0].value.local.block_idx, 3);
    EXPECT_EQ(f.signals[1].contexts[0].value.local.block_idx, 3);
    f.storage[0].payload.scalars.set(nullptr);
    ASSERT_TRUE(f.run(0, task, 3));
    EXPECT_EQ(addresses.back(), reinterpret_cast<int64_t *>(arguments));
}

TEST(PercounterPayload, WaitingCandidateCannotReplaceCurrentContext) {
    auto producer = task(1);
    producer.published_off_lane = true;
    const int32_t producer_id = 0;
    auto consumer = task(2, 3);
    consumer.fanin_count = 1;
    consumer.fanin_ids = &producer_id;
    Fixture f({producer, consumer});
    auto &context = f.signals[0].contexts[1].value;
    context.local = {-1, -1};
    context.global.sub_block_id = 0;
    context.global.async_run_epoch = 123;
    f.step(1);
    f.step(1);
    EXPECT_TRUE(f.calls.empty());
    EXPECT_EQ(context.local.block_idx, -1);
    EXPECT_EQ(context.local.block_num, -1);
    execute = [&](int64_t *args) {
        EXPECT_EQ(args, reinterpret_cast<int64_t *>(f.arguments[1].args));
        EXPECT_EQ(get_block_idx(args), 0);
        EXPECT_EQ(get_block_num(args), 3);
        EXPECT_EQ(load_global_context(args).async_run_epoch, 123U);
    };
    f.counters[0].value = 1;
    f.step(1);
    EXPECT_EQ(context.local.block_idx, 0);
    EXPECT_EQ(context.local.block_num, 3);
    EXPECT_EQ(context.global.async_run_epoch, 123U);
}

TEST(PercounterPayload, FullArgumentVectorPreservesTensorPointersAndScalarBits) {
    Fixture f({task(1)});
    std::array<uint64_t, MAX_SCALAR_ARGS> scalars{};
    for (uint32_t i = 0; i < scalars.size(); ++i)
        scalars[i] = UINT64_MAX - i;
    const uint64_t functions[3]{reinterpret_cast<uint64_t>(&kernel), 0, 0};
    ASSERT_EQ(
        percounter_prepare_argument_metadata(
            f.counters[0], functions, 1, reinterpret_cast<uint64_t>(&f.arguments[0]), MAX_TENSOR_ARGS, MAX_SCALAR_ARGS
        ),
        SchedulerGraphResult::OK
    );
    for (uint32_t i = 0; i < MAX_TENSOR_ARGS; ++i)
        f.arguments[0].args[i] = 0x1000U + i * TASK_IMAGE_TENSOR_STRIDE;
    for (uint32_t i = 0; i < MAX_SCALAR_ARGS; ++i)
        f.arguments[0].args[MAX_TENSOR_ARGS + i] = scalars[i];
    execute = [&](int64_t *args) {
        for (uint32_t i = 0; i < MAX_TENSOR_ARGS; ++i)
            EXPECT_EQ(static_cast<uint64_t>(args[i]), 0x1000U + i * TASK_IMAGE_TENSOR_STRIDE);
        for (uint32_t i = 0; i < MAX_SCALAR_ARGS; ++i)
            EXPECT_EQ(static_cast<uint64_t>(args[MAX_TENSOR_ARGS + i]), scalars[i]);
        EXPECT_EQ(get_block_idx(args), 0);
        EXPECT_EQ(get_block_num(args), 1);
    };
    ASSERT_TRUE(f.run(0, f.resolve(0)));
}

TEST(PercounterPayload, KernelWindowExcludesSkippedWorkAndSharesProfileEndpoints) {
    Fixture f({task(1)});
    int64_t value = 0;
    auto &predicate = f.storage[0].payload.predicate;
    predicate.addr = reinterpret_cast<uint64_t>(&value);
    predicate.target = 1;
    predicate.elem_size = sizeof(value);
    predicate.op = static_cast<PredicateOp>(1);
    auto &cfg = f.cfg[0];
    cfg.has_predicates = true;
    auto task = f.resolve(0);
    int calls = 0;
    uint64_t inside_kernel = 0;
    execute = [&](int64_t *) {
        ++calls;
        inside_kernel = get_sys_cnt_aicore();
    };

    ASSERT_TRUE(f.run(0, task));
    EXPECT_EQ(cfg.work_window.start_cycle, 0U);
    EXPECT_EQ(cfg.work_window.end_cycle, 0U);
    value = 1;
    const uint64_t callable = f.counters[0].function_addresses[0];
    f.counters[0].function_addresses[0] = 0;
    cfg.metadata[0].active_mask = 0;
    ASSERT_TRUE(f.run(0, f.resolve(0)));
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(cfg.work_window.start_cycle, 0U);
    EXPECT_EQ(cfg.work_window.end_cycle, 0U);

    f.counters[0].function_addresses[0] = callable;
    cfg.metadata[0].active_mask = 1;
    task = f.resolve(0);
    ASSERT_TRUE(f.run(0, task));
    const AicoreWorkWindow first = cfg.work_window;
    EXPECT_EQ(calls, 1);
    EXPECT_GT(first.start_cycle, 0U);
    EXPECT_LE(first.start_cycle, inside_kernel);
    EXPECT_GE(first.end_cycle, inside_kernel);
    EXPECT_FALSE(cfg.pending_task_valid);

    value = 0;
    ASSERT_TRUE(f.run(0, task));
    EXPECT_EQ(cfg.work_window.start_cycle, first.start_cycle);
    EXPECT_EQ(cfg.work_window.end_cycle, first.end_cycle);

    PercounterProfileBuffer profile{};
    cfg.profile_buffer = &profile;
    value = 1;
    ASSERT_TRUE(f.run(0, task));
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(cfg.work_window.start_cycle, first.start_cycle);
    EXPECT_GE(cfg.pending_task.start, first.end_cycle);
    EXPECT_LE(cfg.pending_task.start, inside_kernel);
    EXPECT_GE(cfg.work_window.end_cycle, inside_kernel);
    EXPECT_EQ(cfg.work_window.end_cycle, cfg.pending_task.finish);
    EXPECT_TRUE(cfg.pending_task_valid);
}

TEST(PercounterPayload, LaneEntryPublishesKernelWorkWindow) {
    Fixture f({task(1)}, 1);
    f.control->cluster_count = 1;
    f.control->aiv_per_cluster = 2;
    f.control->graph_storage_address = reinterpret_cast<uint64_t>(f.storage.data());
    f.control->graph_storage_count = f.storage.size();
    f.control->arguments_address = reinterpret_cast<uint64_t>(f.arguments.data());
    uint64_t inside_kernel = 0;
    execute = [&](int64_t *) {
        inside_kernel = get_sys_cnt_aicore();
        // The lane leaves on EXIT alone; the AICPU is what observes the sinks.
        write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    };
    const auto window = run_percounter_lane(f.control, 0, CoreType::AIC);
    write_reg(RegId::DATA_MAIN_BASE, 0);
    ASSERT_EQ(f.control->lane_error, 0U);
    ASSERT_GT(inside_kernel, 0U);
    EXPECT_GT(window.start_cycle, 0U);
    EXPECT_LE(window.start_cycle, inside_kernel);
    EXPECT_GE(window.end_cycle, inside_kernel);
}

TEST(PercounterPayload, ArgumentMetadataRejectsInvalidVectorsBeforePublication) {
    PercounterCounter metadata{};
    const uint64_t functions[3]{1, 2, 0};
    EXPECT_EQ(
        percounter_prepare_argument_metadata(metadata, functions, 1, 0, 0, 1), SchedulerGraphResult::INVALID_ARGUMENTS
    );
    EXPECT_EQ(
        percounter_prepare_argument_metadata(metadata, functions, 1, 0x1000, MAX_TENSOR_ARGS + 1, 0),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
    EXPECT_EQ(
        percounter_prepare_argument_metadata(metadata, functions, 1, UINT64_MAX - 63, 0, 2),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
    EXPECT_EQ(
        percounter_prepare_argument_metadata(metadata, functions, 7, 0x1000, 0, 0),
        SchedulerGraphResult::INVALID_CALLABLE
    );
    EXPECT_EQ(metadata.argument_counts, 0U);
    EXPECT_EQ(metadata.arguments_address, 0U);
    EXPECT_EQ(percounter_prepare_argument_metadata(metadata, functions, 3, 0x1000, 2, 1), SchedulerGraphResult::OK);
    EXPECT_EQ(metadata.arguments_address, 0x1000U);
    EXPECT_EQ(metadata.argument_counts, (1U << 16) | 2U);
}

TEST(PercounterPayload, ProfilingLevelsReserveOnlyTheirRecordTails) {
    const uint64_t counts[2]{1, 0};
    std::array<PercounterLayout, 5> layouts{};
    for (uint32_t level = 0; level < layouts.size(); ++level)
        ASSERT_TRUE(plan_layout(1, 16, 0, counts, &layouts[level], 0, 1, 0, {}, 0, level));
    EXPECT_EQ(layouts[0].total_size, layouts[1].total_size);
    EXPECT_EQ(layouts[1].profile_buffers_offset, 0);
    EXPECT_NE(layouts[2].profile_buffers_offset, 0);
    EXPECT_NE(layouts[2].task_records_offset, 0);
    EXPECT_EQ(layouts[1].event_records_offset, 0);
    EXPECT_GT(layouts[2].event_records_offset, layouts[2].task_records_offset);
    EXPECT_EQ(layouts[2].event_record_capacity, 16U * 12);
    EXPECT_EQ(layouts[3].event_record_capacity, 16U * 8);
    EXPECT_EQ(layouts[2].scalar_records_offset, 0);
    EXPECT_EQ(layouts[3].upload_size, layouts[2].upload_size);
    EXPECT_EQ(layouts[3].total_size, layouts[4].total_size);
    EXPECT_GT(layouts[3].scalar_records_offset, layouts[3].upload_size);
    EXPECT_EQ(
        layouts[3].event_records_offset - layouts[3].scalar_records_offset,
        SCHEDULER_WORKER_CAPACITY * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord)
    );
}

TEST(PercounterPayload, TaskTimingPartitionsOwnWholeCacheLines) {
    PercounterLayout layout{};
    const uint64_t counts[2]{1, 0};
    ASSERT_TRUE(plan_layout(1, 3, 0, counts, &layout, 0, 1, 0, {}, 0, 2));
    std::vector<uint8_t> storage(layout.upload_size + 127);
    const auto address = (reinterpret_cast<uintptr_t>(storage.data()) + 127) & ~uintptr_t{127};
    auto *base = reinterpret_cast<void *>(address);
    TaskInput task{};
    task.block_num = 3;
    task.subtask_count = 1;
    ASSERT_TRUE(build_tables(&task, 1, layout, base, address, 2, {}));
    const auto *buffers = region_at<PercounterProfileBuffer>(base, layout.profile_buffers_offset);
    for (uint32_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        EXPECT_EQ(buffers[worker].task_records_offset % 64, 0) << worker;
        EXPECT_EQ(buffers[worker].event_records_offset % 64, 0) << worker;
        if (worker + 1 < SCHEDULER_WORKER_CAPACITY)
            EXPECT_LE(
                buffers[worker].task_records_offset + 3 * sizeof(PercounterTaskTiming),
                buffers[worker + 1].task_records_offset
            );
    }
}

TEST(PercounterPayload, DeferredTimingDoesNotNeedStreamingBuffers) {
    Fixture f({task(1)});
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
        // The lane leaves on EXIT alone; the AICPU is what observes the sinks.
        write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    };
    percounter_lane_loop<0>(f.cfg[0]);
    write_reg(RegId::DATA_MAIN_BASE, 0);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(f.completed(0));
    EXPECT_EQ(f.control->lane_error, 0U);
    EXPECT_EQ(count, 1U);
    EXPECT_EQ(dropped, 0U);
    EXPECT_GT(records[0].dispatch, 0U);
    EXPECT_GE(records[0].finish, records[0].dispatch);
}

TEST(PercounterPayload, ExhaustedTimingStorageDoesNotBlockCompletion) {
    Fixture f({task(1)});
    PercounterProfileBuffer header{};
    uint32_t count = 0, dropped = 0;
    f.cfg[0].profile_buffer = &header;
    f.cfg[0].profile_task_count = &count;
    f.cfg[0].task_dropped = &dropped;
    int calls = 0;
    execute = [&](int64_t *) {
        ++calls;
        write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    };
    percounter_lane_loop<0>(f.cfg[0]);
    write_reg(RegId::DATA_MAIN_BASE, 0);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(f.completed(0));
    EXPECT_EQ(f.control->lane_error, 0U);
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
    ASSERT_TRUE(f.run(0, task));
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(records[0].dispatch, 0U);
    percounter_publish_completion(
        f.cfg[0], task, percounter_completion_address(f.cfg[0], task.id), 1, percounter_order_entry(task.id, 0, 1)
    );
    EXPECT_TRUE(f.completed(0));
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
