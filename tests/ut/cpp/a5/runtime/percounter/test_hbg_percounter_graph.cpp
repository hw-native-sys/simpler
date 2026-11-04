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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "percounter/percounter_graph.h"
#include "percounter/percounter_plan.h"
#include "scheduler/scheduler_layout.h"

namespace pc = simpler::hbg::percounter;

namespace {

constexpr uint64_t kDeviceBase = 0x7000'0000'0000ull;
constexpr uint8_t kAic = 0x1;
constexpr uint8_t kAiv = 0x2;
constexpr uint8_t kMix = 0x3;

pc::BodyTaskShape shape(uint8_t mask, int32_t blocks = 1) {
    return pc::BodyTaskShape{
        mask, /*sync_start=*/false, /*has_predicate=*/false, blocks, blocks * __builtin_popcount(mask)
    };
}

// A body as rows of producers, turned into the CSR a Definition carries.
struct Body {
    std::vector<pc::BodyTaskShape> shapes;
    std::vector<int32_t> offsets{0};
    std::vector<int32_t> indices;

    void add(pc::BodyTaskShape task, std::vector<int32_t> producers = {}) {
        shapes.push_back(task);
        indices.insert(indices.end(), producers.begin(), producers.end());
        offsets.push_back(static_cast<int32_t>(indices.size()));
    }
    pc::TemplateResult build(pc::HostGraphTemplate *out) const {
        return pc::build_graph_template(
            shapes.data(), static_cast<int32_t>(shapes.size()), offsets.data(), indices.data(),
            static_cast<int32_t>(indices.size()), out
        );
    }
};

// 0 (AIC) -> {1 (AIV), 2 (AIV, 2 blocks)} -> 3 (AIC): one root, one sink, depth 3.
Body diamond() {
    Body body;
    body.add(shape(kAic));
    body.add(shape(kAiv), {0});
    body.add(shape(kAiv, 2), {0});
    body.add(shape(kAic), {1, 2});
    return body;
}

std::vector<int64_t> ids_of(const std::vector<uint64_t> &entries) {
    std::vector<int64_t> ids;
    for (uint64_t entry : entries)
        ids.push_back(pc::percounter_order_task_id(entry));
    return ids;
}

class Region {
public:
    explicit Region(uint64_t size, uint8_t fill = 0) :
        storage_(static_cast<size_t>(size + SCHEDULER_STATE_ALIGNMENT), fill) {
        auto raw = reinterpret_cast<uintptr_t>(storage_.data());
        auto aligned = (raw + SCHEDULER_STATE_ALIGNMENT - 1) & ~(uintptr_t{SCHEDULER_STATE_ALIGNMENT} - 1);
        base_ = reinterpret_cast<void *>(aligned);
    }
    void *base() const { return base_; }

private:
    std::vector<uint8_t> storage_;
    void *base_ = nullptr;
};

}  // namespace

// --- templates ------------------------------------------------------------

TEST(PercounterGraphTemplate, MarksRootsAndSinksAndCountsTheShellPreset) {
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(diamond().build(&tmpl).status, pc::TemplateStatus::OK);
    ASSERT_EQ(tmpl.tasks.size(), 4u);
    EXPECT_EQ(tmpl.tasks[0].flags, pc::PERCOUNTER_BODY_ROOT);
    EXPECT_EQ(tmpl.tasks[1].flags, 0u);
    EXPECT_EQ(tmpl.tasks[2].flags, 0u);
    EXPECT_EQ(tmpl.tasks[3].flags, pc::PERCOUNTER_BODY_SINK);
    EXPECT_EQ(tmpl.tasks[2].subtask_total, 2u);
    // The shell is published by the sinks alone, one add per sink subtask.
    EXPECT_EQ(tmpl.sink_subtask_total, 1u);
    EXPECT_EQ(tmpl.executable_blocks, 5u);
    EXPECT_EQ(tmpl.depth, 3u);
    EXPECT_EQ(tmpl.fanin_offsets, (std::vector<int32_t>{0, 0, 1, 2, 4}));
    EXPECT_EQ(tmpl.fanin_indices, (std::vector<int32_t>{0, 0, 1, 2}));
}

// Inside the body the order is by level, so a root precedes everything it
// feeds, and an SPMD task takes one entry per block.
TEST(PercounterGraphTemplate, OrdersEachListByLevelInsideTheBody) {
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(diamond().build(&tmpl).status, pc::TemplateStatus::OK);
    EXPECT_EQ(ids_of(tmpl.orders[0]), (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(ids_of(tmpl.orders[1]), (std::vector<int64_t>{1, 2, 2}));
    EXPECT_TRUE(tmpl.orders[pc::PERCOUNTER_LIST_MIX].empty());
    EXPECT_EQ(pc::percounter_order_block_idx(tmpl.orders[1][2]), 1);
    EXPECT_EQ(pc::percounter_order_block_num(tmpl.orders[1][2]), 2);
}

TEST(PercounterGraphTemplate, PutsMixBodyTasksInTheMixListAndCountsTheirRendezvous) {
    Body body;
    body.add(shape(kMix, 3));
    body.add(shape(kAiv), {0});
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(body.build(&tmpl).status, pc::TemplateStatus::OK);
    EXPECT_EQ(tmpl.orders[pc::PERCOUNTER_LIST_MIX].size(), 3u);
    EXPECT_EQ(tmpl.mix_rendezvous_count, 3u);
    EXPECT_EQ(tmpl.tasks[0].subtask_total, 6u);
    EXPECT_TRUE(tmpl.orders[0].empty());
}

// The body shapes percounter does not run send the whole graph back to the
// default selection rather than failing the bind.
TEST(PercounterGraphTemplate, RefusesBodyShapesPercounterDoesNotRunAsUnsupported) {
    pc::HostGraphTemplate tmpl;
    {
        Body body;
        body.add(shape(0x4));
        EXPECT_EQ(body.build(&tmpl).status, pc::TemplateStatus::UNSUPPORTED);
    }
    {
        Body body;
        pc::BodyTaskShape predicated = shape(kAiv, 2);
        predicated.has_predicate = true;
        body.add(predicated);
        EXPECT_EQ(body.build(&tmpl).status, pc::TemplateStatus::UNSUPPORTED);
    }
}

// A sync-start body task holds one arming entry, as a table one does, and the
// template counts its cohort and blocks for the queue. A MIX one's rendezvous
// report to the cohort queue, not to the MIX list.
TEST(PercounterGraphTemplate, GivesASyncStartBodyTaskOneArmingEntry) {
    Body body;
    body.add(shape(kAic));
    pc::BodyTaskShape cohort = shape(kMix, 24);
    cohort.sync_start = true;
    body.add(cohort, {0});
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(body.build(&tmpl).status, pc::TemplateStatus::OK);
    ASSERT_EQ(tmpl.orders[pc::PERCOUNTER_LIST_MIX].size(), 1u);
    EXPECT_TRUE(pc::percounter_order_is_arming(tmpl.orders[pc::PERCOUNTER_LIST_MIX][0]));
    EXPECT_EQ(pc::percounter_order_block_num(tmpl.orders[pc::PERCOUNTER_LIST_MIX][0]), 24);
    EXPECT_EQ(tmpl.sync_cohort_count, 1u);
    EXPECT_EQ(tmpl.sync_block_total, 24u);
    EXPECT_EQ(tmpl.mix_rendezvous_count, 0u);
    EXPECT_EQ(tmpl.executable_blocks, 25u);
    EXPECT_EQ(tmpl.tasks[1].subtask_total, 48u);

    Body lonely;
    pc::BodyTaskShape one = shape(kAiv, 1);
    one.sync_start = true;
    lonely.add(one);
    EXPECT_EQ(lonely.build(&tmpl).status, pc::TemplateStatus::MALFORMED);
}

// Two invocations of a body with a cohort put two cohorts in the queue.
TEST(PercounterGraphTables, CountsEveryInvocationsCohorts) {
    Body body;
    pc::BodyTaskShape cohort = shape(kAiv, 4);
    cohort.sync_start = true;
    body.add(cohort);
    std::vector<pc::HostGraphTemplate> templates(1);
    ASSERT_EQ(body.build(&templates[0]).status, pc::TemplateStatus::OK);
    std::vector<pc::GraphInvocationInput> invocations{{0, 0}, {1, 0}};
    std::vector<pc::TaskInput> tasks{
        pc::TaskInput{0, false, false, 0, nullptr},
        pc::TaskInput{0, false, false, 0, nullptr},
    };
    tasks[0].graph_invocation = 0;
    tasks[1].graph_invocation = 1;
    const uint64_t order_count[2] = {0, 2};
    const pc::GraphLayoutInput graph{2, 2, pc::graph_template_bytes(templates[0])};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(2, 2 + 8, 0, order_count, &layout, 0, 2, graph));
    Region region(layout.total_size);
    ASSERT_TRUE(
        pc::build_tables(
            tasks.data(), 2, layout, region.base(), kDeviceBase, {}, nullptr,
            pc::GraphPlanInput{templates.data(), 1, invocations.data(), 2}
        )
    );
    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), layout.control_offset);
    EXPECT_EQ(control->sync_cohort_count, 2u);
    EXPECT_EQ(control->sync_block_total, 8u);
}

// A dependency-only task runs nothing but still orders its consumers after its
// producers, so it is one AIV entry worth one subtask rather than nothing.
TEST(PercounterGraphTemplate, PlacesADependencyOnlyTaskAsOneAivSubtask) {
    Body body;
    body.add(shape(kAic));
    body.add(pc::BodyTaskShape{0, false, false, 1, 0}, {0});
    body.add(shape(kAic), {1});
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(body.build(&tmpl).status, pc::TemplateStatus::OK);
    EXPECT_EQ(tmpl.tasks[1].subtask_total, 1u);
    EXPECT_EQ(ids_of(tmpl.orders[1]), (std::vector<int64_t>{1}));
    EXPECT_EQ(tmpl.executable_blocks, 3u);

    Body predicated;
    predicated.add(pc::BodyTaskShape{0, false, true, 1, 0});
    EXPECT_EQ(predicated.build(&tmpl).status, pc::TemplateStatus::MALFORMED);
}

TEST(PercounterGraphTemplate, AcceptsAPredicateOnASingleBlockTask) {
    Body body;
    pc::BodyTaskShape predicated = shape(kAiv);
    predicated.has_predicate = true;
    body.add(predicated);
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(body.build(&tmpl).status, pc::TemplateStatus::OK);
    EXPECT_TRUE(tmpl.has_predicates);
    EXPECT_NE(tmpl.tasks[0].flags & pc::PERCOUNTER_BODY_PREDICATE, 0u);
}

TEST(PercounterGraphTemplate, RefusesAContradictoryDefinitionAsMalformed) {
    pc::HostGraphTemplate tmpl;
    {
        Body body;
        body.add(shape(kAic));
        body.add(shape(kAic), {1});  // waits on itself
        EXPECT_EQ(body.build(&tmpl).status, pc::TemplateStatus::MALFORMED);
    }
    {
        Body body;
        pc::BodyTaskShape wrong = shape(kAiv, 2);
        wrong.total_required_subtasks = 3;
        body.add(wrong);
        EXPECT_EQ(body.build(&tmpl).status, pc::TemplateStatus::MALFORMED);
    }
}

// The template reader goes through upstream's decoders, so a hand-framed
// Definition checks the plumbing: framing, the sub-task array, and the fanin
// indices, which are 16-bit on the wire.
TEST(PercounterGraphTemplate, ReadsAFramedDefinition) {
    constexpr int32_t kTasks = 2;
    constexpr int32_t kEdges = 1;
    const uint32_t header_bytes = sizeof(GraphDefinitionHeader);
    uint32_t cursor = sizeof(GraphDefinition);
    const auto take = [&cursor](uint32_t bytes, uint32_t align) {
        cursor = (cursor + align - 1) & ~(align - 1);
        const uint32_t at = cursor;
        cursor += bytes;
        return at;
    };
    GraphDefinition definition{};
    definition.graph_key = 0x1234;
    definition.task_count = kTasks;
    definition.edge_count = kEdges;
    definition.off_sub_tasks = take(kTasks * sizeof(SubTaskDefinition), alignof(SubTaskDefinition));
    definition.off_fanin_offsets = take((kTasks + 1) * sizeof(int32_t), alignof(int32_t));
    definition.off_fanin_indices = take(kEdges * sizeof(uint16_t), alignof(uint16_t));
    definition.total_bytes = cursor;

    std::vector<std::byte> section(header_bytes + definition.total_bytes);
    GraphDefinitionHeader header{};
    header.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
    header.graph_key = definition.graph_key;
    header.definition_bytes = definition.total_bytes;
    std::memcpy(section.data(), &header, sizeof(header));
    std::byte *image = section.data() + header_bytes;
    std::memcpy(image, &definition, sizeof(definition));
    SubTaskDefinition tasks[kTasks]{};
    tasks[0].active_mask = kAic;
    tasks[0].logical_block_num = 1;
    tasks[0].total_required_subtasks = 1;
    tasks[1].active_mask = kAiv;
    tasks[1].logical_block_num = 4;
    tasks[1].total_required_subtasks = 4;
    std::memcpy(image + definition.off_sub_tasks, tasks, sizeof(tasks));
    const int32_t rows[kTasks + 1] = {0, 0, 1};
    std::memcpy(image + definition.off_fanin_offsets, rows, sizeof(rows));
    const uint16_t producers[kEdges] = {0};
    std::memcpy(image + definition.off_fanin_indices, producers, sizeof(producers));

    const GraphImageView view{section.data(), static_cast<uint32_t>(section.size())};
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(pc::read_graph_template(view, header_bytes, &tmpl).status, pc::TemplateStatus::OK);
    EXPECT_EQ(tmpl.tasks[0].flags, pc::PERCOUNTER_BODY_ROOT);
    EXPECT_EQ(tmpl.tasks[1].flags, pc::PERCOUNTER_BODY_SINK);
    EXPECT_EQ(tmpl.tasks[1].subtask_total, 4u);
    EXPECT_EQ(tmpl.fanin_indices, (std::vector<int32_t>{0}));
    EXPECT_EQ(tmpl.orders[1].size(), 4u);

    // An offset that names no framed object is refused, not decoded.
    EXPECT_EQ(pc::read_graph_template(view, header_bytes + 8, &tmpl).status, pc::TemplateStatus::MALFORMED);
}

// --- region ---------------------------------------------------------------

namespace {

// Table: 0 (AIC) -> 1 (GRAPH, diamond) -> 2 (AIV), and 3 (GRAPH, diamond) on
// its own. Two invocations share one template.
struct GraphTable {
    std::vector<pc::HostGraphTemplate> templates{1};
    std::vector<pc::GraphInvocationInput> invocations{{1, 0}, {3, 0}};
    std::vector<std::vector<int32_t>> fanin{{}, {0}, {1}, {}};
    std::vector<pc::TaskInput> tasks;
    pc::PercounterLayout layout{};

    GraphTable() {
        EXPECT_EQ(diamond().build(&templates[0]).status, pc::TemplateStatus::OK);
        for (size_t i = 0; i < fanin.size(); ++i) {
            pc::TaskInput task{
                static_cast<uint8_t>(i == 2 ? 1 : 0), false, false, static_cast<int32_t>(fanin[i].size()),
                fanin[i].empty() ? nullptr : fanin[i].data()
            };
            if (i == 1) task.graph_invocation = 0;
            if (i == 3) task.graph_invocation = 1;
            tasks.push_back(task);
        }
        const pc::HostGraphTemplate &tmpl = templates[0];
        // Task 0 and the two bodies' AIC entries; task 2 and the bodies' AIV ones.
        const uint64_t order_count[2] = {1 + 2 * tmpl.orders[0].size(), 1 + 2 * tmpl.orders[1].size()};
        pc::GraphLayoutInput graph{};
        graph.body_task_count = 2 * tmpl.tasks.size();
        graph.invocation_count = 2;
        graph.templates_bytes = pc::graph_template_bytes(tmpl);
        const uint64_t block_total = 4 + 2 * tmpl.executable_blocks;
        EXPECT_TRUE(pc::plan_layout(4, block_total, 2, order_count, &layout, 0, 0, graph));
    }
    pc::GraphPlanInput plan() const {
        return pc::GraphPlanInput{templates.data(), templates.size(), invocations.data(), invocations.size()};
    }
};

}  // namespace

TEST(PercounterGraphLayout, NumbersBodyTasksAfterTheTableAndPutsTheirCountersLast) {
    GraphTable graph;
    EXPECT_EQ(graph.layout.table_task_count, 4u);
    EXPECT_EQ(graph.layout.task_count, 12u);
    EXPECT_EQ(graph.layout.graph_templates_offset % alignof(pc::PercounterGraphTemplate), 0u);
    EXPECT_EQ(graph.layout.graph_invocations_offset % alignof(pc::PercounterGraphInvocation), 0u);
    // The counters close the region, and the upload stops at the table's.
    EXPECT_GT(graph.layout.counters_offset, graph.layout.dispatch_payloads_offset);
    EXPECT_EQ(graph.layout.upload_size, graph.layout.counters_offset + 4 * sizeof(pc::PercounterCounter));
    EXPECT_LE(graph.layout.counters_offset + 12 * sizeof(pc::PercounterCounter), graph.layout.total_size);
}

TEST(PercounterGraphLayout, WithoutGraphsTheUploadIsTheWholeRegionBeforePadding) {
    const uint64_t order_count[2] = {2, 2};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(4, 4, 0, order_count, &layout));
    EXPECT_EQ(layout.task_count, layout.table_task_count);
    EXPECT_EQ(layout.upload_size, layout.counters_offset + 4 * sizeof(pc::PercounterCounter));
    EXPECT_EQ(layout.graph_invocation_count, 0u);
}

TEST(PercounterGraphTables, RecordsEachInvocationAndLeavesItsRangesUnpublished) {
    GraphTable graph;
    Region region(graph.layout.total_size);
    pc::BuildResult result{};
    ASSERT_TRUE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, &result, graph.plan())
    ) << pc::build_status_name(result.status);

    const auto *invocations =
        pc::region_at<pc::PercounterGraphInvocation>(region.base(), graph.layout.graph_invocations_offset);
    EXPECT_EQ(invocations[0].shell_task_id, 1u);
    EXPECT_EQ(invocations[0].body_base, 4u);
    EXPECT_EQ(invocations[1].shell_task_id, 3u);
    EXPECT_EQ(invocations[1].body_base, 8u);
    EXPECT_EQ(invocations[0].template_offset, invocations[1].template_offset);
    EXPECT_EQ(invocations[0].storage_address, 0u);

    for (uint32_t l = 0; l < 2; ++l) {
        const auto *list = pc::region_at<uint64_t>(region.base(), graph.layout.orders_offset[l]);
        for (int k = 0; k < 2; ++k) {
            for (uint64_t i = 0; i < graph.templates[0].orders[l].size(); ++i) {
                const uint64_t entry = list[invocations[k].order_base[l] + i];
                EXPECT_EQ(pc::percounter_order_block_num(entry), 0)
                    << "list " << l << " invocation " << k << " position " << i;
            }
        }
    }
}

// The shell is preset against its sinks and runs nothing itself; its consumer
// waits on it like on any task.
TEST(PercounterGraphTables, PresetsTheShellAgainstItsSinksAndCountsTheBodiesAsWork) {
    GraphTable graph;
    Region region(graph.layout.total_size);
    ASSERT_TRUE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, nullptr, graph.plan())
    );
    const auto *counters = pc::region_at<pc::PercounterCounter>(region.base(), graph.layout.counters_offset);
    EXPECT_EQ(counters[1].value, pc::percounter_counter_preset(1));
    EXPECT_EQ(counters[3].value, pc::percounter_counter_preset(1));

    const auto *control = pc::region_at<pc::PercounterControl>(region.base(), graph.layout.control_offset);
    EXPECT_EQ(control->task_count, 12u);
    EXPECT_EQ(control->table_task_count, 4u);
    EXPECT_EQ(control->executable_task_count, 2u + 2 * 5u);
    EXPECT_EQ(control->graph_invocation_count, 2u);
    EXPECT_EQ(control->graph_invocations_offset, graph.layout.graph_invocations_offset);
}

TEST(PercounterGraphTables, CopiesTheTemplateIntoTheRegion) {
    GraphTable graph;
    Region region(graph.layout.total_size);
    ASSERT_TRUE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, nullptr, graph.plan())
    );
    const auto *invocation =
        pc::region_at<pc::PercounterGraphInvocation>(region.base(), graph.layout.graph_invocations_offset);
    const auto *header = pc::region_at<pc::PercounterGraphTemplate>(region.base(), invocation->template_offset);
    const pc::HostGraphTemplate &tmpl = graph.templates[0];
    EXPECT_EQ(header->task_count, 4u);
    EXPECT_EQ(header->edge_count, 4u);
    EXPECT_EQ(header->sink_subtask_total, 1u);
    EXPECT_EQ(header->depth, 3u);
    EXPECT_EQ(
        std::memcmp(
            pc::region_at<uint8_t>(region.base(), header->tasks_offset), tmpl.tasks.data(),
            tmpl.tasks.size() * sizeof(pc::PercounterBodyTask)
        ),
        0
    );
    EXPECT_EQ(
        std::memcmp(
            pc::region_at<uint8_t>(region.base(), header->fanin_offsets_offset), tmpl.fanin_offsets.data(),
            tmpl.fanin_offsets.size() * sizeof(int32_t)
        ),
        0
    );
    for (uint32_t l = 0; l < pc::PERCOUNTER_LIST_COUNT; ++l) {
        ASSERT_EQ(header->order_count[l], tmpl.orders[l].size());
        if (tmpl.orders[l].empty()) continue;
        EXPECT_EQ(header->orders_offset[l] % alignof(uint64_t), 0u);
        EXPECT_EQ(
            std::memcmp(
                pc::region_at<uint8_t>(region.base(), header->orders_offset[l]), tmpl.orders[l].data(),
                tmpl.orders[l].size() * sizeof(uint64_t)
            ),
            0
        );
    }
}

// Producer ahead of the body, body ahead of the consumer: the order the claim's
// deadlock argument needs across the boundary.
TEST(PercounterGraphTables, PlacesTheBodyBetweenItsProducerAndItsConsumer) {
    GraphTable graph;
    Region region(graph.layout.total_size);
    ASSERT_TRUE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, nullptr, graph.plan())
    );
    const auto *invocation =
        pc::region_at<pc::PercounterGraphInvocation>(region.base(), graph.layout.graph_invocations_offset);
    const auto *aic = pc::region_at<uint64_t>(region.base(), graph.layout.orders_offset[0]);
    const auto *aiv = pc::region_at<uint64_t>(region.base(), graph.layout.orders_offset[1]);
    uint64_t producer = UINT64_MAX;
    for (uint64_t p = 0; p < graph.layout.order_count[0]; ++p) {
        if (pc::percounter_order_block_num(aic[p]) != 0 && pc::percounter_order_task_id(aic[p]) == 0) producer = p;
    }
    uint64_t consumer = UINT64_MAX;
    for (uint64_t p = 0; p < graph.layout.order_count[1]; ++p) {
        if (pc::percounter_order_block_num(aiv[p]) != 0 && pc::percounter_order_task_id(aiv[p]) == 2) consumer = p;
    }
    ASSERT_NE(producer, UINT64_MAX);
    ASSERT_NE(consumer, UINT64_MAX);
    EXPECT_LT(producer, invocation[0].order_base[0]);
    EXPECT_GE(consumer, invocation[0].order_base[1] + graph.templates[0].orders[1].size());
}

// A GRAPH task weighs its body's depth: three tasks deep with nothing after
// it, it outranks a two-task chain that a unit weight would put first.
TEST(PercounterGraphTables, WeighsAGraphTaskByItsBodyDepth) {
    pc::HostGraphTemplate tmpl;
    ASSERT_EQ(diamond().build(&tmpl).status, pc::TemplateStatus::OK);
    std::vector<pc::HostGraphTemplate> templates{tmpl};
    std::vector<pc::GraphInvocationInput> invocations{{2, 0}};
    std::vector<int32_t> chain{0};
    std::vector<pc::TaskInput> tasks{
        pc::TaskInput{0, false, false, 0, nullptr},
        pc::TaskInput{0, false, false, 1, chain.data()},
        pc::TaskInput{0, false, false, 0, nullptr},
    };
    tasks[2].graph_invocation = 0;
    const uint64_t order_count[2] = {2 + tmpl.orders[0].size(), tmpl.orders[1].size()};
    pc::GraphLayoutInput graph{tmpl.tasks.size(), 1, pc::graph_template_bytes(tmpl)};
    pc::PercounterLayout layout{};
    ASSERT_TRUE(pc::plan_layout(3, 3 + tmpl.executable_blocks, 1, order_count, &layout, 0, 0, graph));
    Region region(layout.total_size);
    ASSERT_TRUE(
        pc::build_tables(
            tasks.data(), 3, layout, region.base(), kDeviceBase, {}, nullptr,
            pc::GraphPlanInput{templates.data(), 1, invocations.data(), 1}
        )
    );
    const auto *invocation =
        pc::region_at<pc::PercounterGraphInvocation>(region.base(), layout.graph_invocations_offset);
    EXPECT_EQ(invocation->order_base[0], 0u) << "the body outranks the chain";
}

// The body counters are the AICPU's: the host writes nothing past the upload.
TEST(PercounterGraphTables, LeavesTheBodyCountersUntouched) {
    GraphTable graph;
    Region region(graph.layout.total_size, 0xAB);
    ASSERT_TRUE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, nullptr, graph.plan())
    );
    const auto *tail = pc::region_at<uint8_t>(region.base(), graph.layout.upload_size);
    for (uint64_t i = 0; i < 8 * sizeof(pc::PercounterCounter); ++i)
        ASSERT_EQ(tail[i], 0xAB) << "byte " << i;
}

TEST(PercounterGraphTables, RejectsAnInvocationTheTasksDoNotName) {
    GraphTable graph;
    graph.tasks[3].graph_invocation = -1;
    graph.tasks[3].block_num = 1;
    Region region(graph.layout.total_size);
    pc::BuildResult result{};
    EXPECT_FALSE(
        pc::build_tables(graph.tasks.data(), 4, graph.layout, region.base(), kDeviceBase, {}, &result, graph.plan())
    );
}
