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

// One Graph invocation end to end: the host plans it from a framed Definition,
// and the AICPU half localizes, materializes and publishes it through upstream's
// own graph_execution functions.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "percounter/percounter_graph.h"
#include "percounter/percounter_graph_prepare.h"
#include "percounter/percounter_plan.h"
#include "scheduler/scheduler_layout.h"

namespace pc = simpler::hbg::percounter;

namespace {

template <typename T>
uint32_t append_section(std::vector<std::byte> &image, const std::vector<T> &values) {
    if (values.empty()) return 0;
    const size_t offset = CHIP_ALIGN_UP(image.size(), alignof(T));
    image.resize(offset + values.size() * sizeof(T));
    std::memcpy(image.data() + offset, values.data(), values.size() * sizeof(T));
    return static_cast<uint32_t>(offset);
}

simpler::hbg::TensorData heap_tensor(uint64_t offset) {
    simpler::hbg::TensorData tensor{};
    tensor.buffer.addr = offset;
    tensor.buffer.size = 32;
    tensor.owner_task_id = TaskId::make_sub_task(0, 0);
    tensor.extent_elem_cache = 1;
    tensor.shapes[0] = 1;
    tensor.strides[0] = 1;
    tensor.ndims = 1;
    tensor.dtype = DataType::FLOAT32;
    tensor.is_contiguous = true;
    return tensor;
}

// A three-task body: 0 (AIC) -> 1 (AIV) -> 2 (AIC), every tensor in the graph
// heap and no boundary arguments, so localize needs nothing but the heap.
std::vector<std::byte> make_definition(uint64_t graph_key) {
    std::vector<std::byte> image(sizeof(GraphDefinition));
    const std::vector<int32_t> fanin_offsets{0, 0, 1, 2};
    const std::vector<uint16_t> fanin_indices{0, 1};
    const std::vector<int32_t> fanout_offsets{0, 1, 2, 2};
    const std::vector<uint16_t> fanout_indices{1, 2};
    const std::vector<uint16_t> roots{0};
    const std::vector<uint64_t> sub_task_offsets{0, 64, 128};
    std::vector<SubTaskDefinition> tasks(3);
    for (size_t i = 0; i < tasks.size(); ++i) {
        SubTaskDefinition &task = tasks[i];
        std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
        const bool aiv = i == 1;
        task.kernel_id[aiv ? 1 : 0] = 42;
        task.active_mask = aiv ? 2 : 1;
        task.logical_block_num = 1;
        task.total_required_subtasks = 1;
        task.tensor_count = 1;
        task.tensor_offset = static_cast<int32_t>(i);
        task.total_output_size = 64;
    }
    const std::vector<simpler::hbg::TensorData> tensors{heap_tensor(0), heap_tensor(64), heap_tensor(128)};

    GraphDefinition definition{};
    definition.graph_key = graph_key;
    definition.required_heap = 192;
    definition.task_count = 3;
    definition.edge_count = 2;
    definition.root_count = 1;
    definition.tensor_arg_count = 3;
    definition.off_fanin_offsets = append_section(image, fanin_offsets);
    definition.off_fanin_indices = append_section(image, fanin_indices);
    definition.off_fanout_offsets = append_section(image, fanout_offsets);
    definition.off_fanout_indices = append_section(image, fanout_indices);
    definition.off_root_indices = append_section(image, roots);
    definition.off_sub_task_offsets = append_section(image, sub_task_offsets);
    definition.off_sub_tasks = append_section(image, tasks);
    definition.off_tensors = append_section(image, tensors);
    size_t execution_storage_bytes = 0;
    graph_execution_storage_bytes(
        definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
        &execution_storage_bytes
    );
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);
    definition.total_bytes = static_cast<uint32_t>(image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));
    return image;
}

template <typename T>
struct AlignedBuffer {
    explicit AlignedBuffer(size_t bytes) :
        bytes(bytes),
        data(::operator new(bytes, std::align_val_t(alignof(T)))) {
        std::memset(data, 0, bytes);
    }
    ~AlignedBuffer() { ::operator delete(data, std::align_val_t(alignof(T))); }
    uint8_t *base() const { return static_cast<uint8_t *>(data); }
    size_t bytes;
    void *data;
};

// The run as the device sees it: the Definition section, the shell's table
// entry and heap, and percounter's region built by the host planner.
struct GraphRun {
    std::vector<std::byte> image = make_definition(0x77);
    AlignedBuffer<GraphDefinitionHeader> section{sizeof(GraphDefinitionHeader) + image.size()};
    AlignedBuffer<ChipTaskStorage> heap;
    ChipTaskStorage table[1]{};
    pc::HostGraphTemplate tmpl;
    pc::PercounterLayout layout{};
    std::unique_ptr<AlignedBuffer<pc::PercounterControl>> region;

    GraphRun() :
        heap(
            static_cast<size_t>(
                reinterpret_cast<const GraphDefinition *>(image.data())->required_heap +
                reinterpret_cast<const GraphDefinition *>(image.data())->execution_storage_bytes
            )
        ) {
        auto *header = reinterpret_cast<GraphDefinitionHeader *>(section.base());
        header->magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        header->graph_key = 0x77;
        header->definition_bytes = static_cast<uint32_t>(image.size());
        std::memcpy(section.base() + sizeof(GraphDefinitionHeader), image.data(), image.size());

        ChipTaskStorage &shell = table[0];
        shell.task.task_id = TaskId::make_global(0);
        shell.task.packed_buffer_base = heap.base();
        shell.task.packed_buffer_end = heap.base() + heap.bytes;
        shell.slot.task_kind = TaskKind::GRAPH;
        shell.slot.to_descriptor().graph_definition_offset = image_offset();

        EXPECT_EQ(pc::read_graph_template(view(), image_offset(), &tmpl).status, pc::TemplateStatus::OK);
        const uint64_t order_count[2] = {tmpl.orders[0].size(), tmpl.orders[1].size()};
        const pc::GraphLayoutInput graph{tmpl.tasks.size(), 1, pc::graph_template_bytes(tmpl)};
        EXPECT_TRUE(pc::plan_layout(1, 1 + tmpl.executable_blocks, 0, order_count, &layout, 0, 0, graph));
        region = std::make_unique<AlignedBuffer<pc::PercounterControl>>(static_cast<size_t>(layout.total_size));
        pc::TaskInput input{0, false, false, 0, nullptr};
        input.graph_invocation = 0;
        const pc::GraphInvocationInput invocation{0, 0};
        pc::ExternalAddresses external{};
        external.graph_storage = reinterpret_cast<uint64_t>(&table[0]);
        external.graph_storage_count = 1;
        EXPECT_TRUE(
            pc::build_tables(
                &input, 1, layout, region->data, reinterpret_cast<uint64_t>(region->data), external, nullptr,
                pc::GraphPlanInput{&tmpl, 1, &invocation, 1}
            )
        );
    }

    static uint32_t image_offset() { return static_cast<uint32_t>(sizeof(GraphDefinitionHeader)); }
    GraphImageView view() const {
        return GraphImageView{
            reinterpret_cast<const std::byte *>(section.base()), static_cast<uint32_t>(section.bytes)
        };
    }
    pc::PercounterControl *control() const { return static_cast<pc::PercounterControl *>(region->data); }
    pc::PercounterGraphPrepare *prepare() const {
        return pc::percounter_region<pc::PercounterGraphPrepare>(control(), control()->graph_prepare_offset);
    }
    pc::PercounterGraphInvocation *invocation() const {
        return pc::percounter_region<pc::PercounterGraphInvocation>(control(), control()->graph_invocations_offset);
    }
    pc::PercounterCounter *counters() const {
        return pc::percounter_region<pc::PercounterCounter>(control(), control()->counters_offset);
    }
    const uint64_t *list(uint32_t l) const {
        return pc::percounter_region<uint64_t>(control(), control()->orders_offset[l]);
    }
};

}  // namespace

TEST(PercounterGraphPrepare, PublishesTheStorageAddressCountersAndEntries) {
    GraphRun run;
    pc::percounter_graph_prepare_init(run.control());
    ASSERT_TRUE(pc::percounter_prepare_invocation(run.control(), run.prepare(), 0, run.view()));
    EXPECT_EQ(run.prepare()->error, 0u);

    auto *execution = static_cast<GraphExecution *>(run.table[0].slot.graph_context);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(run.invocation()->storage_address, reinterpret_cast<uint64_t>(execution->task_storage));
    EXPECT_EQ(execution->materialized_tasks, 3);

    // Body task i is percounter id 1 + i, preset against its one subtask and
    // tagged with where it came from.
    for (uint32_t i = 0; i < 3; ++i) {
        const pc::PercounterCounter &counter = run.counters()[1 + i];
        EXPECT_EQ(counter.value, pc::percounter_counter_preset(1)) << "body task " << i;
        EXPECT_EQ(counter.invocation, 0u);
        EXPECT_EQ(counter.body_index, i);
        EXPECT_EQ(counter.storage_address, run.invocation()->storage_address);
        EXPECT_EQ(counter.shell_task_id, 0u);
    }
    // What a lane reads instead of the template: the root waits on its shell,
    // the others on one body producer each, and the last publishes the shell.
    EXPECT_EQ(run.counters()[1].flags, pc::PERCOUNTER_BODY_ROOT);
    EXPECT_EQ(run.counters()[2].flags, 0u);
    EXPECT_EQ(run.counters()[3].flags, pc::PERCOUNTER_BODY_SINK);
    for (uint32_t i = 1; i < 3; ++i) {
        const pc::PercounterCounter &counter = run.counters()[1 + i];
        ASSERT_EQ(counter.fanin_count, 1u);
        EXPECT_EQ(*pc::percounter_region<uint64_t>(run.control(), counter.fanin_offset), i - 1) << "body task " << i;
    }
    // The ranges now hold real entries, the template's carried to percounter ids.
    const uint64_t *aic = run.list(0) + run.invocation()->order_base[0];
    ASSERT_EQ(run.tmpl.orders[0].size(), 2u);
    EXPECT_EQ(pc::percounter_order_task_id(aic[0]), 1);
    EXPECT_EQ(pc::percounter_order_task_id(aic[1]), 3);
    EXPECT_EQ(pc::percounter_order_block_num(aic[0]), 1);
    const uint64_t *aiv = run.list(1) + run.invocation()->order_base[1];
    EXPECT_EQ(pc::percounter_order_task_id(aiv[0]), 2);
}

// The host's own counter -- the shell's -- is left as the host preset it: the
// sinks publish it, not the AICPU.
TEST(PercounterGraphPrepare, LeavesTheShellCounterToItsSinks) {
    GraphRun run;
    const uint64_t preset = run.counters()[0].value;
    ASSERT_TRUE(pc::percounter_prepare_invocation(run.control(), run.prepare(), 0, run.view()));
    EXPECT_EQ(run.counters()[0].value, preset);
    EXPECT_EQ(preset, pc::percounter_counter_preset(1));
}

// A Definition that does not decode leaves every range unpublished and records
// the failure for the supervisor.
TEST(PercounterGraphPrepare, RecordsAFailedLocalizeAndPublishesNothing) {
    GraphRun run;
    run.table[0].slot.to_descriptor().graph_definition_offset = GraphRun::image_offset() + 8;
    EXPECT_FALSE(pc::percounter_prepare_invocation(run.control(), run.prepare(), 0, run.view()));
    EXPECT_EQ(run.prepare()->error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
    EXPECT_EQ(run.prepare()->error_invocation, 0u);
    const uint64_t *aic = run.list(0) + run.invocation()->order_base[0];
    EXPECT_EQ(pc::percounter_order_block_num(aic[0]), 0);
    EXPECT_NE(pc::percounter_graph_prepare_error(run.control()), 0u);
}
