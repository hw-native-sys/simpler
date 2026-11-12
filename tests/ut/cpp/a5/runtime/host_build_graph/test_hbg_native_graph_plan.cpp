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

#include <cstring>
#include <atomic>
#include <numeric>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "host_build_graph/graph_execution.h"
#include "native_graph_plan.h"
#include "dependency_plan.h"
#include "ready_pool_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_graph_control.h"
#include "scheduler/scheduler_graph_binding.h"
#include "scheduler/scheduler_admission.h"
#include "scheduler/scheduler_dispatch.h"

namespace {
TEST(NativeGraphPlanTest, LegacyStorageFallbackDoesNotBypassSyncStartCapacity) {
    NativeGraphHostPlan plan;
    plan.definitions.resize(1);
    auto &definition = plan.definitions[0];
    definition.resident_supported = false;
    definition.children.resize(1);
    auto &metadata = definition.children[0].metadata;
    metadata.active_mask = LaneMask::all();
    metadata.logical_block_num = 2;
    metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    std::string error;
    EXPECT_FALSE(native_graph_plan_worker_capacity(plan, 3, &error));
    EXPECT_TRUE(native_graph_plan_worker_capacity(plan, 6, &error)) << error;
    metadata.flags &= ~SCHEDULER_TASK_SYNC_START;
    EXPECT_TRUE(native_graph_plan_worker_capacity(plan, 3, &error)) << error;
}

struct DefinitionFixture {
    GraphDefinition header{};
    std::vector<std::byte> bytes{sizeof(GraphDefinition)};

    template <typename T>
    uint32_t append(const std::vector<T> &values) {
        if (values.empty()) return 0;
        const size_t start = (bytes.size() + alignof(T) - 1) & ~(alignof(T) - 1);
        bytes.resize(start + values.size() * sizeof(T));
        std::memcpy(bytes.data() + start, values.data(), values.size() * sizeof(T));
        return static_cast<uint32_t>(start);
    }
    template <typename T>
    void put(uint32_t offset, uint32_t index, const T &value) {
        std::memcpy(bytes.data() + offset + sizeof(T) * index, &value, sizeof(T));
    }
    explicit DefinitionFixture(
        const std::vector<std::vector<uint16_t>> &fanin, bool dummy = false, bool scalar = false
    ) {
        header.graph_key = 77;
        header.task_count = static_cast<int32_t>(fanin.size());
        std::vector<int32_t> in_rows{0}, out_rows{0};
        std::vector<uint16_t> in_edges, out_edges, roots;
        std::vector<std::vector<uint16_t>> fanout(fanin.size());
        for (uint16_t child = 0; child < fanin.size(); ++child) {
            if (fanin[child].empty()) roots.push_back(child);
            for (auto producer : fanin[child]) {
                in_edges.push_back(producer);
                fanout[producer].push_back(child);
            }
            in_rows.push_back(static_cast<int32_t>(in_edges.size()));
        }
        for (const auto &row : fanout) {
            out_edges.insert(out_edges.end(), row.begin(), row.end());
            out_rows.push_back(static_cast<int32_t>(out_edges.size()));
        }
        header.edge_count = static_cast<int32_t>(in_edges.size());
        header.root_count = static_cast<int32_t>(roots.size());
        header.off_fanout_offsets = append(out_rows);
        header.off_fanout_indices = append(out_edges);
        header.off_fanin_offsets = append(in_rows);
        header.off_fanin_indices = append(in_edges);
        header.off_root_indices = append(roots);
        header.off_sub_task_offsets = append(std::vector<uint64_t>(fanin.size(), 0));
        std::vector<SubTaskDefinition> tasks(fanin.size());
        for (size_t i = 0; i < tasks.size(); ++i) {
            auto &task = tasks[i];
            task.kernel_id[0] = dummy ? -1 : 0;
            task.kernel_id[1] = task.kernel_id[2] = -1;
            task.active_mask = dummy ? LaneMask::none() : LaneMask::aic();
            task.logical_block_num = 1;
            task.total_required_subtasks = dummy ? 0 : 1;
            if (scalar) {
                task.scalar_count = 1;
                task.scalar_offset = static_cast<int32_t>(i);
            }
        }
        header.off_sub_tasks = append(tasks);
        if (scalar) {
            header.scalar_arg_count = header.task_count;
            header.off_scalars = append(std::vector<uint64_t>(fanin.size(), 42));
            header.off_scalar_inheritance =
                append(std::vector<GraphScalarInheritance>(fanin.size(), GraphScalarInheritance::self_value()));
        }
        GraphExecutionStorageLayout layout{};
        EXPECT_TRUE(
            graph_execution_storage_layout(header.task_count, 0, header.scalar_arg_count, header.edge_count, &layout)
        );
        header.execution_storage_bytes = static_cast<uint32_t>(layout.total_bytes);
        header.total_bytes = static_cast<uint32_t>(bytes.size());
        sync();
    }
    void sync() { std::memcpy(bytes.data(), &header, sizeof(header)); }
    bool build(NativeGraphDefinitionPlan &plan, std::string &error) const {
        return native_graph_plan_definition(bytes.data(), bytes.size(), header.graph_key, &plan, &error);
    }
};
}  // namespace

TEST(HbgNativeGraphPlan, FaninAndSinkCountsUseTheWholeDefinition) {
    for (int count : {1, 128, 129, 1023}) {
        std::vector<std::vector<uint16_t>> edges(static_cast<size_t>(count + 1));
        edges.back().resize(count);
        std::iota(edges.back().begin(), edges.back().end(), 0);
        DefinitionFixture fixture(edges);
        NativeGraphDefinitionPlan plan;
        std::string error;
        ASSERT_TRUE(fixture.build(plan, error)) << error;
        ASSERT_EQ(plan.sinks, std::vector<uint16_t>({static_cast<uint16_t>(count)}));
        EXPECT_EQ(plan.execution_count, count + 1);
        EXPECT_EQ(plan.definition.edge_count, count);
    }
    DefinitionFixture roots(std::vector<std::vector<uint16_t>>(1024), true);
    NativeGraphDefinitionPlan plan;
    std::string error;
    ASSERT_TRUE(roots.build(plan, error)) << error;
    EXPECT_EQ(plan.sinks.size(), 1024);
    EXPECT_EQ(plan.execution_count, 0);
    EXPECT_TRUE(plan.resident_supported);
}

TEST(HbgNativeGraphPlan, DenseRefsKeepOriginalShellIdentityAndShareDefinitions) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}, {0}});
    NativeGraphHostPlan plan;
    plan.definitions.resize(1);
    std::string error;
    ASSERT_TRUE(fixture.build(plan.definitions[0], error)) << error;
    ASSERT_TRUE(native_graph_plan_nodes({UINT32_MAX, 0, UINT32_MAX, 0}, &plan, &error)) << error;
    EXPECT_EQ(plan.completion_refs, std::vector<int64_t>({0, 4, 5, 9}));
    ASSERT_EQ(plan.bindings.size(), 10);
    ASSERT_EQ(plan.instances.size(), 2);
    EXPECT_EQ(plan.instances[0].original_shell_id, 1);
    EXPECT_EQ(plan.instances[1].original_shell_id, 3);
    EXPECT_EQ(plan.instances[1].definition_index, 0);
    EXPECT_EQ(plan.bindings[2].source_index, 0);
    EXPECT_EQ(plan.bindings[7].source_index, 0);
    EXPECT_EQ(plan.bindings[2].instance_index, 0);
    EXPECT_EQ(plan.bindings[7].instance_index, 1);
    EXPECT_EQ(plan.bindings[4].kind, SchedulerNodeKind::GRAPH_EXIT);
    EXPECT_EQ(plan.bindings[5].source_index, 2);
}

TEST(HbgNativeGraphPlan, OwnsBytesAndKeepsScalarWritersOnSeparateCacheLines) {
    DefinitionFixture fixture({{}, {0}, {1}}, false, true);
    NativeGraphDefinitionPlan plan;
    std::string error;
    ASSERT_TRUE(fixture.build(plan, error)) << error;
    EXPECT_EQ(plan.args_bytes, 192);
    for (size_t i = 0; i < plan.children.size(); ++i) {
        EXPECT_EQ(plan.children[i].scalars_offset, i * 64);
        EXPECT_EQ(plan.children[i].predicate_offset, UINT64_MAX);
        EXPECT_EQ(plan.children[i].execution_prefix, i);
    }
    const auto saved = plan.image;
    fixture.bytes.assign(fixture.bytes.size(), std::byte{0});
    EXPECT_EQ(plan.image, saved);
}

TEST(HbgNativeGraphPlan, RejectsDuplicateEdgesRootsAndAsymmetricCsr) {
    NativeGraphDefinitionPlan plan;
    std::string error;
    DefinitionFixture duplicate({{}, {0}, {0, 1}});
    duplicate.put<uint16_t>(duplicate.header.off_fanin_indices, 2, 0);
    EXPECT_FALSE(duplicate.build(plan, error));
    DefinitionFixture wrong_fanout({{}, {}, {0}, {1}});
    wrong_fanout.put<uint16_t>(wrong_fanout.header.off_fanout_indices, 0, 3);
    EXPECT_FALSE(wrong_fanout.build(plan, error));
    DefinitionFixture duplicate_root({{}, {}, {0, 1}});
    duplicate_root.put<uint16_t>(duplicate_root.header.off_root_indices, 1, 0);
    EXPECT_FALSE(duplicate_root.build(plan, error));
    DefinitionFixture self_edge(std::vector<std::vector<uint16_t>>{{}, {0}});
    self_edge.put<uint16_t>(self_edge.header.off_fanin_indices, 0, 1);
    EXPECT_FALSE(self_edge.build(plan, error));
}

TEST(HbgNativeGraphPlan, RejectsBadExtentsBeforeReadingAndLeavesOutputUnchanged) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}});
    NativeGraphDefinitionPlan plan;
    std::string error;
    ASSERT_TRUE(fixture.build(plan, error)) << error;
    const auto saved = plan.image;
    EXPECT_FALSE(native_graph_plan_definition(fixture.bytes.data(), sizeof(GraphDefinition) - 1, 77, &plan, &error));
    fixture.header.off_sub_tasks = UINT32_MAX;
    fixture.sync();
    EXPECT_FALSE(fixture.build(plan, error));
    EXPECT_EQ(plan.image, saved);
    fixture.header.task_count = INT32_MAX;
    fixture.sync();
    EXPECT_FALSE(fixture.build(plan, error));
    EXPECT_EQ(plan.image, saved);
}

TEST(HbgNativeGraphPlan, CooperativeShapeStillRejectsMalformedTopology) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}, {0}});
    SubTaskDefinition child{};
    std::memcpy(&child, fixture.bytes.data() + fixture.header.off_sub_tasks, sizeof(child));
    child.logical_block_num = 2;
    child.total_required_subtasks = 2;
    fixture.put(fixture.header.off_sub_tasks, 0, child);
    NativeGraphDefinitionPlan plan;
    std::string error;
    ASSERT_TRUE(fixture.build(plan, error)) << error;
    EXPECT_TRUE(plan.resident_supported);
    fixture.put<uint16_t>(fixture.header.off_fanin_indices, 0, 1);
    EXPECT_FALSE(fixture.build(plan, error));
}

namespace {
struct NativeViewFixture {
    scheduler_test::BasicGraphBuffer<8> original;
    NativeGraphHostPlan plan;
    NativeGraphReadonlyImage readonly_image;
    std::unique_ptr<void, decltype(&std::free)> backing{nullptr, &std::free};
    SchedulerGraphView graph{};
    AicoreSchedulerLayout layout{};
    SchedulerReadyPoolPlan ready_pool{};

    NativeViewFixture(
        const DefinitionFixture &definition, std::vector<uint32_t> indices, bool runtime_state = false,
        uint64_t workers = 3
    ) :
        NativeViewFixture(
            std::vector<const DefinitionFixture *>{&definition}, std::move(indices), runtime_state, workers
        ) {}

    NativeViewFixture(
        const std::vector<const DefinitionFixture *> &definitions, std::vector<uint32_t> indices,
        bool runtime_state = false, uint64_t workers = 3
    ) :
        original(indices.size()) {
        std::string error;
        plan.definitions.resize(definitions.size());
        for (size_t index = 0; index < definitions.size(); ++index)
            if (!definitions[index]->build(plan.definitions[index], error)) throw std::runtime_error(error);
        if (!native_graph_plan_nodes(indices, &plan, &error) ||
            !native_graph_plan_readonly_image(plan, &readonly_image, &error))
            throw std::runtime_error(error);
        SchedulerNativeGraphHeader header{};
        std::memcpy(&header, readonly_image.bytes.data() + readonly_image.header_offset, sizeof(header));
        header.control_inbox_count = workers;
        header.control_inboxes_offset = readonly_image.bytes.size();
        header.image_bytes = readonly_image.bytes.size() + workers * sizeof(SchedulerReadyInbox);
        header.run_epoch = 1;
        std::vector<SchedulerTaskMetadata> task_metadata(plan.bindings.size());
        if (runtime_state) {
            for (size_t ref = 0; ref < plan.bindings.size(); ++ref) {
                const auto &binding = plan.bindings[ref];
                if (binding.kind == SchedulerNodeKind::CHILD_KERNEL) {
                    const auto &instance = plan.instances[binding.instance_index];
                    task_metadata[ref] =
                        plan.definitions[instance.definition_index].children[binding.source_index].metadata;
                }
            }
            SchedulerSpmdRequirements requirements{workers, workers / 3, 0, 0};
            uint64_t trace_count = 0;
            for (const auto &instance : plan.instances) {
                const auto &definition_plan = plan.definitions[instance.definition_index];
                trace_count += definition_plan.execution_count;
                for (const auto &child : definition_plan.children) {
                    if (scheduler_task_requires_admission(child.metadata)) ++requirements.cooperative_task_count;
                    if (scheduler_task_is_mix(child.metadata.flags)) ++requirements.mix_task_count;
                }
            }
            if (!scheduler_plan_layout(
                    plan.bindings.size(), 0, 0, &layout, false, trace_count, requirements, false, false,
                    readonly_image.bytes.size()
                ))
                throw std::runtime_error("scheduler layout");
            if (!scheduler_plan_ready_pool(task_metadata, workers / 3, layout.total_size, &ready_pool))
                throw std::runtime_error("ready pool plan");
            layout.total_size = ready_pool.total_size;
            header.image_bytes = layout.total_size;
            header.control_inboxes_offset = layout.control_inboxes_offset;
        }
        backing.reset(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, header.image_bytes));
        if (!backing) throw std::bad_alloc();
        std::memset(backing.get(), 0, header.image_bytes);
        if (runtime_state &&
            !scheduler_init_data_from_layout(backing.get(), layout, reinterpret_cast<uint64_t>(backing.get())))
            throw std::runtime_error("scheduler init");
        std::memcpy(
            static_cast<std::byte *>(backing.get()) + sizeof(SchedulerRunControl),
            readonly_image.bytes.data() + sizeof(SchedulerRunControl),
            readonly_image.bytes.size() - sizeof(SchedulerRunControl)
        );
        std::memcpy(static_cast<std::byte *>(backing.get()) + readonly_image.header_offset, &header, sizeof(header));
        graph = original.graph();
        graph.task_count = plan.bindings.size();
        graph.scheduler_state_address = reinterpret_cast<uint64_t>(backing.get());
        graph.scheduler_state_bytes = header.image_bytes;
        graph.native_header_offset = readonly_image.header_offset;
        if (runtime_state) {
            std::copy(
                task_metadata.begin(), task_metadata.end(),
                scheduler_state_at<SchedulerTaskMetadata>(backing.get(), layout.task_metadata_offset)
            );
            scheduler_init_ready_pool(backing.get(), ready_pool);
            scheduler_state_at<SchedulerGraphSummary>(backing.get(), layout.graph_summary_offset)->ready_pool_offset =
                ready_pool.header_offset;
            if (!scheduler_init_spmd_task_records(backing.get(), layout, 1)) throw std::runtime_error("SPMD init");
        }
    }
    SchedulerNativeGraphHeader &header() {
        return *reinterpret_cast<SchedulerNativeGraphHeader *>(
            static_cast<std::byte *>(backing.get()) + readonly_image.header_offset
        );
    }
    SchedulerGraphTemplate &definition() {
        return *reinterpret_cast<SchedulerGraphTemplate *>(
            static_cast<std::byte *>(backing.get()) + header().templates_offset
        );
    }
};
}  // namespace

TEST(HbgNativeGraphPlan, DependencyCountersIncludeGraphEntryExitAndCrossInstanceEdges) {
    DefinitionFixture definition(std::vector<std::vector<uint16_t>>{{}, {}, {0, 1}});
    NativeViewFixture view(definition, {UINT32_MAX, 0, 0, UINT32_MAX});
    view.original.executable(0, 0);
    view.original.executable(1, 0, {0});
    view.original.executable(2, 0, {1});
    view.original.executable(3, 0, {2});
    SchedulerDependencyPlan plan;
    ASSERT_TRUE(scheduler_plan_dependencies(view.graph.task_count, 13, 128, &plan));
    std::unique_ptr<void, decltype(&std::free)> storage(
        std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, plan.total_size), &std::free
    );
    ASSERT_NE(storage, nullptr);
    ASSERT_TRUE(scheduler_init_dependencies(storage.get(), plan, view.graph, {0}));
    const auto *header = scheduler_state_at<SchedulerDependencyHeader>(storage.get(), plan.header_offset);
    EXPECT_EQ(header->edge_count, 12);
    std::vector<std::pair<int64_t, int64_t>> edges;
    for (int64_t producer = 0; producer < view.graph.task_count; ++producer) {
        for (uint32_t index = 0;; ++index) {
            int64_t successor;
            ASSERT_TRUE(scheduler_dependency_successor(storage.get(), *header, producer, index, &successor));
            if (successor == SCHEDULER_WAKE_LIST_OPEN) break;
            edges.emplace_back(producer, successor);
        }
    }
    EXPECT_EQ(
        edges, (std::vector<std::pair<int64_t, int64_t>>{
                   {1, 2}, {1, 3}, {2, 4}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {6, 8}, {7, 9}, {8, 9}, {9, 10}, {10, 11}
               })
    );
    const auto *remaining = scheduler_state_at<uint64_t>(storage.get(), header->remaining_offset);
    EXPECT_EQ(
        std::vector<uint64_t>(remaining, remaining + view.graph.task_count),
        (std::vector<uint64_t>{0, 0, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1})
    );
    ASSERT_TRUE(scheduler_plan_dependencies(view.graph.task_count, 11, 128, &plan));
    EXPECT_FALSE(scheduler_init_dependencies(storage.get(), plan, view.graph, {0}));
}

TEST(HbgNativeGraphPlan, AccessConstructionDoesNotObserveNativeBacking) {
    SchedulerGraphView wire = scheduler_ordinary_graph_view(4096, 1, 0);
    wire.scheduler_state_address = SCHEDULER_STATE_ALIGNMENT;
    wire.scheduler_state_bytes = 1024 * 1024;
    wire.native_header_offset = sizeof(SchedulerRunControl);
    ASSERT_TRUE(scheduler_native_header_extent_valid(
        wire.scheduler_state_address, wire.scheduler_state_bytes, wire.native_header_offset
    ));
    const SchedulerGraphAccess access(wire);
    EXPECT_EQ(access.scheduler_state_address, wire.scheduler_state_address);
    EXPECT_EQ(access.native_header_offset, wire.native_header_offset);
}

TEST(HbgNativeGraphPlan, AccessRevalidatesReusedBackingForEachRun) {
    DefinitionFixture body(std::vector<std::vector<uint16_t>>{{}});
    NativeViewFixture view(body, {0});
    {
        const SchedulerGraphAccess access(view.graph);
        SchedulerNativeNode node{};
        ASSERT_TRUE(scheduler_native_node_checked(access, 1, &node));
        EXPECT_EQ(node.header->run_epoch, 1u);
        EXPECT_EQ(node.definition->graph_key, body.header.graph_key);
        view.header().header_bytes = 0;
        const SchedulerGraphAccess next(access);
        EXPECT_FALSE(scheduler_graph_view_valid(next));
        view.header().header_bytes = sizeof(SchedulerNativeGraphHeader);
    }
    view.header().run_epoch = 2;
    view.header().header_bytes = 0;
    {
        const SchedulerGraphAccess access(view.graph);
        EXPECT_FALSE(scheduler_graph_view_valid(access));
        SchedulerNativeNode node{};
        EXPECT_FALSE(scheduler_native_node_checked(access, 1, &node));
    }
    view.header().header_bytes = sizeof(SchedulerNativeGraphHeader);
    const uint64_t task_count = view.definition().task_count;
    view.definition().task_count = 0;
    {
        const SchedulerGraphAccess access(view.graph);
        SchedulerNativeNode node{};
        EXPECT_FALSE(scheduler_native_node_checked(access, 1, &node));
    }
    view.definition().task_count = task_count;
    auto *definition = reinterpret_cast<GraphDefinition *>(
        static_cast<std::byte *>(view.backing.get()) + view.definition().definition_offset
    );
    ++definition->graph_key;
    view.definition().graph_key = definition->graph_key;
    {
        const SchedulerGraphAccess access(view.graph);
        SchedulerNativeNode node{};
        ASSERT_TRUE(scheduler_native_node_checked(access, 1, &node));
        EXPECT_EQ(node.header->run_epoch, 2u);
        EXPECT_EQ(node.definition->graph_key, body.header.graph_key + 1);
    }
}

TEST(HbgNativeGraphPlan, AccessAlternatesDefinitionsWithoutSkippingNodeBounds) {
    DefinitionFixture first(std::vector<std::vector<uint16_t>>{{}});
    DefinitionFixture second(std::vector<std::vector<uint16_t>>{{}});
    second.header.graph_key = first.header.graph_key + 1;
    second.sync();
    NativeViewFixture view(std::vector<const DefinitionFixture *>{&first, &second}, {0, 1, 0});
    const SchedulerGraphAccess access(view.graph);
    for (int64_t ref : {1, 4, 7, 4, 1}) {
        SchedulerNativeNode node{};
        ASSERT_TRUE(scheduler_native_node_checked(access, ref, &node));
        EXPECT_EQ(node.definition->graph_key, ref == 4 ? second.header.graph_key : first.header.graph_key);
    }
    SchedulerNativeNode node{};
    EXPECT_EQ(access.native_template(view.header().definition_count), nullptr);
    EXPECT_FALSE(scheduler_native_node_checked(access, -1, &node));
    EXPECT_FALSE(scheduler_native_node_checked(access, view.graph.task_count, &node));
    auto *bindings = reinterpret_cast<SchedulerNodeBinding *>(
        static_cast<std::byte *>(view.backing.get()) + view.header().bindings_offset
    );
    bindings[1].instance_index = view.header().instance_count;
    EXPECT_FALSE(scheduler_native_node_checked(access, 1, &node));
    bindings[1].instance_index = 0;
    auto *instances = reinterpret_cast<SchedulerGraphInstance *>(
        static_cast<std::byte *>(view.backing.get()) + view.header().instances_offset
    );
    instances[0].exit_ref = view.header().node_count;
    EXPECT_FALSE(scheduler_native_node_checked(access, 1, &node));
    instances[0].exit_ref = instances[0].child_begin + instances[0].child_count;
    auto *child = reinterpret_cast<SchedulerGraphChildPlan *>(
        static_cast<std::byte *>(view.backing.get()) + view.definition().child_plans_offset
    );
    child->tensor_count = MAX_TENSOR_ARGS + 1;
    SchedulerTaskPayloadArguments arguments{};
    EXPECT_EQ(scheduler_parse_task_payload_arguments(access, 1, &arguments), SchedulerGraphResult::INVALID_ARGUMENTS);
}

TEST(HbgNativeGraphPlan, AccessorsMapShellDependenciesToExitAndGateRootsByEnter) {
    DefinitionFixture body(std::vector<std::vector<uint16_t>>{{}, {0}});
    NativeViewFixture view(body, {UINT32_MAX, 0, UINT32_MAX, 0});
    view.original.executable(0, 0);
    view.original.executable(1, 0, {0});
    view.original.executable(2, 1, {1});
    view.original.executable(3, 0, {2});
    ASSERT_TRUE(scheduler_graph_view_valid(view.graph));
    EXPECT_GT(view.graph.task_count, view.graph.task_window_last_index + 1);
    for (int64_t ref = 1; ref < 10; ++ref) {
        uint32_t count = 0;
        int64_t producer = -1;
        ASSERT_EQ(scheduler_node_fanin_count(view.graph, ref, &count), SchedulerGraphResult::OK) << ref;
        ASSERT_EQ(count, 1) << ref;
        ASSERT_EQ(scheduler_node_fanin_ref(view.graph, ref, 0, &producer), SchedulerGraphResult::OK) << ref;
        EXPECT_EQ(producer, ref - 1) << ref;
        SchedulerFaninView fanin{};
        ASSERT_EQ(scheduler_node_fanin_view(view.graph, ref, &fanin), SchedulerGraphResult::OK) << ref;
        ASSERT_EQ(fanin.count, 1u);
        ASSERT_EQ(scheduler_fanin_ref(fanin, ref, 0, &producer), SchedulerGraphResult::OK) << ref;
        EXPECT_EQ(producer, ref - 1) << ref;
        EXPECT_EQ(scheduler_graph_storage(view.graph, ref), nullptr);
    }
    uint64_t identity = UINT64_MAX;
    ASSERT_EQ(scheduler_node_identity(view.graph, 7, &identity), SchedulerGraphResult::OK);
    EXPECT_EQ(identity, TaskId::to_uint64(TaskId::make_sub_task(3, 0)));
    ASSERT_EQ(scheduler_node_identity(view.graph, 5, &identity), SchedulerGraphResult::OK);
    EXPECT_EQ(identity, TaskId::to_uint64(TaskId::make_global(2)));
    SchedulerTaskShape shape{};
    ASSERT_EQ(scheduler_classify_task_shape(view.graph, 5, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(shape.active_mask, LaneMask::aiv0());
    ASSERT_EQ(scheduler_classify_task_shape(view.graph, 7, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(shape.active_mask, LaneMask::aic());
}

TEST(HbgNativeGraphPlan, NativeFaninAndExitReadersDoNotApplyTheTopLevel128Limit) {
    std::vector<std::vector<uint16_t>> rows(1024);
    rows.back().resize(1023);
    std::iota(rows.back().begin(), rows.back().end(), 0);
    DefinitionFixture body(rows);
    NativeViewFixture view(body, {0});
    uint32_t count = 0;
    ASSERT_EQ(scheduler_node_fanin_count(view.graph, 1024, &count), SchedulerGraphResult::OK);
    EXPECT_EQ(count, 1023);
    int64_t producer = -1;
    ASSERT_EQ(scheduler_node_fanin_ref(view.graph, 1024, 1022, &producer), SchedulerGraphResult::OK);
    EXPECT_EQ(producer, 1023);
    DefinitionFixture disconnected(std::vector<std::vector<uint16_t>>(1024), true);
    NativeViewFixture empty_body(disconnected, {0});
    ASSERT_EQ(scheduler_node_fanin_count(empty_body.graph, 1025, &count), SchedulerGraphResult::OK);
    EXPECT_EQ(count, 1024);
    ASSERT_EQ(scheduler_node_fanin_ref(empty_body.graph, 1025, 1023, &producer), SchedulerGraphResult::OK);
    EXPECT_EQ(producer, 1024);
    for (const auto &[graph, consumer, expected_count] :
         {std::tuple{view.graph, 1024, 1023}, std::tuple{empty_body.graph, 1025, 1024}}) {
        SchedulerFaninView fanin{};
        ASSERT_EQ(scheduler_node_fanin_view(graph, consumer, &fanin), SchedulerGraphResult::OK);
        ASSERT_EQ(fanin.count, expected_count);
        for (uint32_t index = 0; index < fanin.count; ++index) {
            ASSERT_EQ(scheduler_fanin_ref(fanin, consumer, index, &producer), SchedulerGraphResult::OK);
            EXPECT_EQ(producer, index + 1);
        }
    }
}

TEST(HbgNativeGraphPlan, ReadersRejectForgedHeaderRegionsAndInstanceBindings) {
    DefinitionFixture body(std::vector<std::vector<uint16_t>>{{}});
    NativeViewFixture view(body, {0});
    ASSERT_TRUE(scheduler_graph_view_valid(view.graph));
    auto graph = view.graph;
    graph.scheduler_state_bytes = view.readonly_image.header_offset;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    graph = view.graph;
    graph.native_header_offset = UINT64_MAX - 127;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    auto &header = view.header();
    header.image_bytes++;
    EXPECT_FALSE(scheduler_graph_view_valid(view.graph));
    header.image_bytes--;
    const auto bindings_offset = header.bindings_offset;
    header.bindings_offset = view.readonly_image.header_offset;
    EXPECT_FALSE(scheduler_graph_view_valid(view.graph));
    header.bindings_offset = bindings_offset;
    auto &definition = view.definition();
    definition.fanin_offsets_offset = header.readonly_end_offset;
    uint32_t count = 0;
    EXPECT_NE(scheduler_node_fanin_count(view.graph, 1, &count), SchedulerGraphResult::OK);
    definition.fanin_offsets_offset = definition.definition_offset + body.header.off_fanin_offsets;
    auto *bindings =
        reinterpret_cast<SchedulerNodeBinding *>(static_cast<std::byte *>(view.backing.get()) + header.bindings_offset);
    bindings[1].instance_index = 99;
    EXPECT_NE(scheduler_node_fanin_count(view.graph, 1, &count), SchedulerGraphResult::OK);
}

namespace {
DefinitionFixture ready_definition(uint32_t type, uint32_t count = 3, bool cooperative = false) {
    DefinitionFixture body{std::vector<std::vector<uint16_t>>(count)};
    for (uint32_t index = 0; index < count; ++index) {
        SubTaskDefinition task{};
        task.kernel_id[0] = type == 0 ? 0 : -1;
        task.kernel_id[1] = type == 1 ? 0 : -1;
        task.kernel_id[2] = -1;
        task.active_mask = type == 0 ? LaneMask::aic() : LaneMask::aiv0();
        task.logical_block_num = cooperative ? 2 : 1;
        task.total_required_subtasks = task.logical_block_num;
        body.put<SubTaskDefinition>(body.header.off_sub_tasks, index, task);
    }
    return body;
}

SchedulerLocalState ready_worker(NativeViewFixture &view, uint32_t type, uint64_t rank) {
    SchedulerLocalState local;
    auto &config = local.config;
    config.graph = &view.graph;
    config.task_controls_offset = view.layout.task_controls_offset;
    config.task_metadata_offset = view.layout.task_metadata_offset;
    config.ready_pool_offset = view.ready_pool.header_offset;
    config.runtime_worker_count = config.scheduler_count = 30;
    config.cluster_count = 10;
    config.self_lane = type == 0 ? 0 : 1 + rank % 2;
    config.self_core_type = type;
    const uint64_t cluster = type == 0 ? rank : rank / 2;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        config.worker_ids[lane] = cluster * 3 + lane;
    }
    config.scheduler_index = local.worker_id();
    return local;
}

void seed_native_ready(NativeViewFixture &view, SchedulerLocalState &local, int64_t ref) {
    SchedulerReadyPublication publication{};
    ASSERT_EQ(
        scheduler_ready_publication_append(view.graph, view.backing.get(), &local, ref, publication),
        SchedulerReadyAppendResult::ADDED
    );
    ASSERT_EQ(
        scheduler_ready_publication_step(view.graph, view.backing.get(), &local, publication),
        SchedulerReadyPublishResult::DONE
    );
}
}  // namespace

TEST(HbgNativeGraphPlan, AnyEligibleWorkerClaimsTheSharedQueue) {
    for (uint32_t type : {0u, 1u}) {
        for (uint32_t rank : {0u, 7u, 9u}) {
            NativeViewFixture view(ready_definition(type), {0}, true, 30);
            auto local = ready_worker(view, type, rank);
            auto *run = static_cast<SchedulerRunControl *>(view.backing.get());
            seed_native_ready(view, local, 1);
            SchedulerReadyClaim claim{};
            ASSERT_TRUE(scheduler_claim_ready_for_slot(view.graph, view.backing.get(), &local, run, 30, type, &claim));
            EXPECT_EQ(claim.task_ref, 1);
            EXPECT_EQ(claim.source, SchedulerReadySource::POOL);
            EXPECT_EQ(scheduler_read_error(run), 0u);
        }
    }
}

TEST(HbgNativeGraphPlan, NativeReadyClaimPreservesPublicationOrder) {
    NativeViewFixture view(ready_definition(0), {0}, true, 30);
    auto producer = ready_worker(view, 0, 9);
    auto consumer = ready_worker(view, 0, 0);
    auto *run = static_cast<SchedulerRunControl *>(view.backing.get());
    for (int64_t ref : {1, 2, 3})
        seed_native_ready(view, producer, ref);
    for (int64_t expected : {1, 2, 3}) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(scheduler_claim_ready_for_slot(view.graph, view.backing.get(), &consumer, run, 30, 0, &claim));
        EXPECT_EQ(claim.task_ref, expected);
    }
}

TEST(HbgNativeGraphPlan, ConcurrentConsumersClaimEachNativeTaskOnce) {
    constexpr uint32_t tasks = 64;
    NativeViewFixture view(ready_definition(0, tasks), {0}, true, 30);
    auto publisher = ready_worker(view, 0, 9);
    auto *base = view.backing.get();
    auto *run = static_cast<SchedulerRunControl *>(base);
    SchedulerReadyBatch batch{};
    for (int64_t ref = 1; ref <= tasks; ++ref)
        ASSERT_TRUE(scheduler_ready_batch_append(base, &publisher, ref, &batch));
    ASSERT_TRUE(scheduler_ready_batch_push(base, &publisher, 0, &batch));
    std::array<std::atomic<uint32_t>, tasks> claims{};
    std::atomic<bool> start{false};
    std::atomic<uint32_t> failures{0};
    std::vector<std::thread> thieves;
    for (uint32_t rank = 0; rank < 7; ++rank) {
        thieves.emplace_back([&, rank] {
            auto local = ready_worker(view, 0, rank);
            while (!start.load(std::memory_order_acquire)) {}
            for (uint32_t attempt = 0; attempt < tasks * 2; ++attempt) {
                SchedulerReadyClaim claim{};
                if (!scheduler_claim_ready_for_slot(view.graph, base, &local, run, 30, 0, &claim)) {
                    ++failures;
                    break;
                }
                if (scheduler_task_ref_is_null(claim.task_ref)) continue;
                if (claim.task_ref < 1 || claim.task_ref > tasks) {
                    ++failures;
                    break;
                }
                ++claims[claim.task_ref - 1];
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thief : thieves)
        thief.join();
    EXPECT_EQ(failures.load(), 0u);
    for (const auto &count : claims)
        EXPECT_EQ(count.load(), 1u);
    EXPECT_EQ(scheduler_read_error(run), 0u);
}

TEST(HbgNativeGraphPlan, OrderedIdleHintFindsSharedReadyWithoutClaimingIt) {
    for (uint32_t type : {0u, 1u}) {
        NativeViewFixture view(ready_definition(type, 1, true), {0}, true, 30);
        auto local = ready_worker(view, type, 0);
        SchedulerAdmissionLocalState admission{};
        admission.capacity = 1;
        admission.control_offset = view.layout.admission_control_offset;
        local.admission = &admission;
        bool empty = false;
        EXPECT_FALSE(scheduler_ordered_poll_may_progress(view.graph, view.backing.get(), &local, &empty));
        EXPECT_TRUE(empty);
        seed_native_ready(view, local, 1);
        EXPECT_TRUE(scheduler_ordered_poll_may_progress(view.graph, view.backing.get(), &local, &empty));
        EXPECT_FALSE(empty);
        const auto queue = scheduler_ready_pool_queue(view.backing.get(), view.ready_pool.header, type);
        EXPECT_EQ(queue.control->head, 0u);
        EXPECT_EQ(queue.control->reserved_tail, 1u);
        EXPECT_EQ(queue.entries[0].task_ref, 1);
        EXPECT_EQ(admission.scan_cursor, 0u);
        EXPECT_EQ(admission.current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

namespace {
struct ControlFixture : NativeViewFixture {
    std::array<SchedulerLocalState, 3> locals;
    SchedulerRunControl *run;
    explicit ControlFixture(const DefinitionFixture &definition) :
        NativeViewFixture(definition, {0}, true) {
        run = static_cast<SchedulerRunControl *>(backing.get());
        run->expected_task_count = graph.task_count;
        for (uint32_t worker = 0; worker < 3; ++worker) {
            auto &local = locals[worker];
            auto &config = local.config;
            config.graph = &graph;
            config.task_controls_offset = layout.task_controls_offset;
            config.task_metadata_offset = static_cast<uint32_t>(layout.task_metadata_offset);
            config.control_inboxes_offset = layout.control_inboxes_offset;
            config.ready_pool_offset = ready_pool.header_offset;
            config.runtime_worker_count = 3;
            config.scheduler_count = 3;
            config.cluster_count = 1;
            config.self_lane = static_cast<uint8_t>(worker);
            config.self_core_type = worker == 0 ? 0 : 1;
            config.worker_ids[0] = 0;
            config.worker_ids[1] = 1;
            config.worker_ids[2] = 2;

            local.control_peer_cursor = worker + 1;
            // No free payload slots: control readiness must remain independent.
            for (auto &slot : local.slots)
                slot.state = SchedulerDispatchSlotState::READY;
        }
        for (int64_t ref = 0; ref < static_cast<int64_t>(graph.task_count); ++ref) {
            const auto route = scheduler_bootstrap_route_task(graph, backing.get(), &locals[0], run, ref);
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                if (!scheduler_control_enqueue(graph, backing.get(), &locals[0], run, ref))
                    throw std::runtime_error("enqueue");
            } else if (route != SchedulerRouteResult::WAITING) {
                throw std::runtime_error("bootstrap route");
            }
        }
    }
    uint64_t completed() const {
        uint64_t value = 0;
        for (const auto &local : locals)
            value += local.pending_completed;
        return value;
    }
};
}  // namespace

TEST(HbgNativeGraphPlan, DummyChainProgressesWithFullKernelSlotsAfterBootstrapBarrier) {
    std::vector<std::vector<uint16_t>> rows(1024);
    for (uint16_t i = 1; i < rows.size(); ++i)
        rows[i].push_back(i - 1);
    DefinitionFixture body(rows, true);
    ControlFixture view(body);
    bool progress = true;
    ASSERT_TRUE(scheduler_service_controls(view.graph, view.backing.get(), &view.locals[0], view.run, 0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(view.completed(), 0);
    scheduler_gm_store(view.run->bootstrap_scan_complete, uint64_t{1});
    for (uint32_t step = 0; step < 512 && view.completed() < view.graph.task_count; ++step) {
        for (auto &local : view.locals) {
            const auto before = local.pending_completed;
            ASSERT_TRUE(scheduler_service_controls(view.graph, view.backing.get(), &local, view.run, 0, &progress));
            EXPECT_LE(local.pending_completed - before, SCHEDULER_CONTROL_NODE_BUDGET + 1);
        }
    }
    ASSERT_EQ(view.completed(), view.graph.task_count);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(view.backing.get(), view.layout.task_controls_offset);
    for (uint64_t i = 0; i < view.graph.task_count; ++i) {
        EXPECT_EQ(scheduler_gm_query(controls[i].state), static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(scheduler_gm_query(controls[i].wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    }
}

TEST(HbgNativeGraphPlan, SlicedFanoutCountsProducerOnlyAfterAllWaitersArePublished) {
    DefinitionFixture body(std::vector<std::vector<uint16_t>>(130), true);
    ControlFixture view(body);
    scheduler_gm_store(view.run->bootstrap_scan_complete, uint64_t{1});
    bool progress = false;
    ASSERT_TRUE(scheduler_service_controls(view.graph, view.backing.get(), &view.locals[0], view.run, 0, &progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(view.locals[0].control_resolve.producer_ref, 0);
    EXPECT_GE(view.locals[0].control_resolve.next_waiter, 0);
    EXPECT_EQ(view.completed(), 0);
    // Another worker can consume the published prefix while its producer's
    // continuation still owns the remaining wake chain.
    ASSERT_TRUE(scheduler_service_controls(view.graph, view.backing.get(), &view.locals[2], view.run, 0, &progress));
    EXPECT_GT(view.locals[2].pending_completed, 0);
    EXPECT_EQ(view.locals[0].pending_completed, 0);
    for (uint32_t step = 0; step < 128 && view.completed() < view.graph.task_count; ++step)
        for (auto &local : view.locals)
            ASSERT_TRUE(scheduler_service_controls(view.graph, view.backing.get(), &local, view.run, 0, &progress));
    EXPECT_EQ(view.completed(), view.graph.task_count);
    for (auto &local : view.locals)
        EXPECT_LT(local.control_resolve.producer_ref, 0);
}

namespace {
DefinitionFixture argument_definition(bool predicate) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}});
    auto &header = fixture.header;
    SubTaskDefinition task{};
    std::memcpy(&task, fixture.bytes.data() + header.off_sub_tasks, sizeof(task));
    task.tensor_count = 2;
    task.scalar_count = 2;
    task.total_output_size = CHIP_ALIGN_SIZE;
    if (predicate) {
        TaskAttrs attrs;
        attrs.set_predicate();
        task.task_attrs = attrs.raw();
        task.predicate_slot = 1;
    }
    fixture.put(header.off_sub_tasks, 0, task);
    simpler::hbg::TensorData parameter{};
    parameter.owner_task_id = TaskId::make_param(0);
    parameter.start_offset = 3;
    parameter.version = 19;
    parameter.ndims = 2;
    parameter.dtype = DataType::INT32;
    parameter.manual_dep = true;
    parameter.is_contiguous = false;
    parameter.address_space = AddressSpace::DEVICE;
    parameter.shapes[0] = 2;
    parameter.shapes[1] = 3;
    parameter.strides[0] = 7;
    parameter.strides[1] = 1;
    parameter.extent_elem_cache = 10;
    auto output = parameter;
    output.owner_task_id = TaskId::make_sub_task(13, 0);
    output.buffer.size = CHIP_ALIGN_SIZE;
    output.start_offset = 1;
    header.required_heap = CHIP_ALIGN_SIZE;
    header.boundary_tensor_count = 1;
    header.boundary_scalar_count = 1;
    header.tensor_arg_count = 2;
    header.scalar_arg_count = 2;
    header.off_tensors = fixture.append(std::vector{parameter, output});
    header.off_scalars = fixture.append(std::vector<uint64_t>{91, 23});
    header.off_scalar_inheritance =
        fixture.append(std::vector{GraphScalarInheritance::from_boundary(0), GraphScalarInheritance::self_value()});
    if (predicate) {
        GraphPredicate pred{};
        pred.operand = parameter;
        pred.elem_offset = 2;
        pred.elem_size = 4;
        pred.op = 1;
        pred.target = 17;
        header.predicate_count = 1;
        header.off_predicates = fixture.append(std::vector{pred});
    }
    GraphExecutionStorageLayout layout{};
    EXPECT_TRUE(graph_execution_storage_layout(1, 2, 2, 0, &layout));
    header.execution_storage_bytes = static_cast<uint32_t>(layout.total_bytes);
    header.total_bytes = static_cast<uint32_t>(fixture.bytes.size());
    fixture.sync();
    return fixture;
}

struct BoundArgumentsFixture {
    alignas(64) std::byte heap[8192]{};
    simpler::hbg::Tensor boundary{};
    uint64_t scalar{71};
    SchedulerGraphInstance *instance;
    explicit BoundArgumentsFixture(NativeViewFixture &view, uint32_t index) {
        instance = reinterpret_cast<SchedulerGraphInstance *>(
                       static_cast<std::byte *>(view.backing.get()) + view.header().instances_offset
                   ) +
                   index;
        boundary.buffer.addr = 0x100000 + index * 0x10000;
        boundary.buffer.size = 4096;
        boundary.start_offset = 7 + index;
        boundary.address_space = AddressSpace::HOST;
        scalar += index;
        instance->boundary_tensor_count = 1;
        instance->boundary_scalar_count = 1;
        instance->boundary_tensors_address = reinterpret_cast<uint64_t>(&boundary);
        instance->boundary_scalars_address = reinterpret_cast<uint64_t>(&scalar);
        instance->heap_base = reinterpret_cast<uint64_t>(heap);
        instance->args_base = instance->heap_base + CHIP_ALIGN_SIZE;
    }
};
}  // namespace

TEST(HbgNativeGraphPlan, BindingRelocatesAllFieldsAndKeepsInstancesIndependent) {
    auto body = argument_definition(true);
    NativeViewFixture view(body, {0, 0});
    BoundArgumentsFixture first(view, 0), second(view, 1);
    for (auto *bound : {&first, &second}) {
        const auto &instance = *bound->instance;
        ASSERT_EQ(scheduler_bind_graph_arguments(view.graph, instance.child_begin), SchedulerGraphBindingError::NONE);
        const auto &plan = view.plan.definitions[0].children[0];
        auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(instance.args_base + plan.tensors_offset);
        EXPECT_EQ(tensors[0].buffer.addr, bound->boundary.buffer.addr);
        EXPECT_EQ(tensors[0].buffer.size, 4096);
        EXPECT_EQ(tensors[0].start_offset, bound->boundary.start_offset + 3);
        EXPECT_EQ(tensors[0].owner_task_id, TaskId::make_param(0));
        EXPECT_EQ(tensors[0].address_space, AddressSpace::HOST);
        EXPECT_EQ(tensors[0].version, 19);
        EXPECT_EQ(tensors[0].ndims, 2);
        EXPECT_EQ(tensors[0].dtype, DataType::INT32);
        EXPECT_TRUE(tensors[0].manual_dep);
        EXPECT_FALSE(tensors[0].is_contiguous);
        EXPECT_EQ(tensors[0].shapes[0], 2);
        EXPECT_EQ(tensors[0].shapes[1], 3);
        EXPECT_EQ(tensors[0].strides[0], 7);
        EXPECT_EQ(tensors[0].strides[1], 1);
        EXPECT_EQ(tensors[0].extent_elem_cache, 10);
        EXPECT_EQ(tensors[1].buffer.addr, instance.heap_base);
        EXPECT_EQ(tensors[1].buffer.size, CHIP_ALIGN_SIZE);
        EXPECT_EQ(tensors[1].start_offset, 1);
        EXPECT_EQ(tensors[1].address_space, AddressSpace::DEVICE);
        EXPECT_EQ(tensors[1].owner_task_id, TaskId::make_sub_task(13, 0));
        auto *scalars = reinterpret_cast<uint64_t *>(instance.args_base + plan.scalars_offset);
        EXPECT_EQ(scalars[0], bound->scalar);
        EXPECT_EQ(scalars[1], 23);
        auto *predicate = reinterpret_cast<SchedulerDispatchPredicate *>(instance.args_base + plan.predicate_offset);
        EXPECT_EQ(predicate->addr, bound->boundary.buffer.addr + (bound->boundary.start_offset + 5) * 4);
        EXPECT_EQ(predicate->target, 17);
        EXPECT_EQ(predicate->elem_size, 4);
        EXPECT_EQ(predicate->op, 1);
        EXPECT_EQ(
            scheduler_observe_graph_arguments(view.graph, instance.child_begin), SchedulerGraphBindingError::NONE
        );
    }
}

TEST(HbgNativeGraphPlan, ParameterBindingRejectsViewsPastTheBoundaryBuffer) {
    auto body = argument_definition(false);
    NativeViewFixture view(body, {0});
    BoundArgumentsFixture bound(view, 0);
    const auto ref = bound.instance->child_begin;
    bound.boundary.buffer.size = 80;
    ASSERT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::NONE);
    bound.boundary.buffer.size = 79;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::TENSOR);
    bound.boundary.buffer.size = 80;
    bound.boundary.start_offset++;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::TENSOR);
}

TEST(HbgNativeGraphPlan, ParameterBindingRejectsInvalidElementSizeAndExtentOverflow) {
    for (bool invalid_dtype : {false, true}) {
        auto body = argument_definition(false);
        simpler::hbg::TensorData parameter{};
        std::memcpy(&parameter, body.bytes.data() + body.header.off_tensors, sizeof(parameter));
        if (invalid_dtype) parameter.dtype = DataType::DATA_TYPE_NUM;
        else parameter.extent_elem_cache = UINT64_MAX;
        body.put(body.header.off_tensors, 0, parameter);
        NativeViewFixture view(body, {0});
        BoundArgumentsFixture bound(view, 0);
        EXPECT_EQ(
            scheduler_bind_graph_arguments(view.graph, bound.instance->child_begin), SchedulerGraphBindingError::TENSOR
        );
    }
}

TEST(HbgNativeGraphPlan, BindingRejectsOverflowAndInvalidBoundaryBeforePublication) {
    auto body = argument_definition(true);
    NativeViewFixture view(body, {0});
    BoundArgumentsFixture bound(view, 0);
    const auto ref = bound.instance->child_begin;
    bound.boundary.start_offset = UINT64_MAX;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::TENSOR);
    bound.boundary.start_offset = 7;
    bound.boundary.buffer.size = 32;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::TENSOR);
    bound.boundary.buffer.size = 4096;
    bound.instance->boundary_scalars_address = 0;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::SCALAR);
    bound.instance->boundary_scalars_address = reinterpret_cast<uint64_t>(&bound.scalar);
    bound.instance->args_base++;
    EXPECT_EQ(scheduler_bind_graph_arguments(view.graph, ref), SchedulerGraphBindingError::INSTANCE);
}

TEST(HbgNativeGraphPlan, BootstrapBindsBlockedChildAndRefreshesReusedArgumentBacking) {
    auto body = argument_definition(true);
    NativeViewFixture view(body, {0}, true);
    BoundArgumentsFixture bound(view, 0);
    const int64_t ref = bound.instance->child_begin;
    auto *run = static_cast<SchedulerRunControl *>(view.backing.get());
    SchedulerLocalState local;
    local.config.graph = &view.graph;
    local.config.task_controls_offset = view.layout.task_controls_offset;
    local.config.task_metadata_offset = view.layout.task_metadata_offset;
    alignas(64) int32_t values[1024]{};
    bound.boundary.buffer.addr = reinterpret_cast<uint64_t>(values);
    const auto &plan = view.plan.definitions[0].children[0];
    auto *scalars = reinterpret_cast<uint64_t *>(bound.instance->args_base + plan.scalars_offset);
    auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(bound.instance->args_base + plan.tensors_offset);
    auto *control = scheduler_task_control_at(view.backing.get(), &local, ref);
    for (uint64_t epoch : {1u, 2u}) {
        view.header().run_epoch = epoch;
        bound.boundary.start_offset = epoch;
        bound.scalar = 70 + epoch;
        values[epoch + 5] = 0;
        const SchedulerGraphAccess access(view.graph);
        ASSERT_TRUE(scheduler_bootstrap_bind_arguments(access, ref, &local, run));
        EXPECT_EQ(run->bootstrap_scan_complete, 0u);
        EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
        EXPECT_EQ(scalars[0], 70 + epoch);
        EXPECT_EQ(tensors[0].start_offset, epoch + 3);
        // A producer writes the operand after descriptor binding. Its value is
        // observed by predicate evaluation, not frozen by bootstrap preparation.
        values[epoch + 5] = 17;
        ASSERT_EQ(scheduler_observe_graph_arguments(access, ref), SchedulerGraphBindingError::NONE);
        EXPECT_EQ(scheduler_evaluate_task_predicate(access, ref), SchedulerPredicateResult::PASS);
    }
}

namespace {
DefinitionFixture cooperative_argument_definition(bool mix, LaneMask mix_mask = LaneMask::all()) {
    auto body = argument_definition(false);
    SubTaskDefinition task{};
    std::memcpy(&task, body.bytes.data() + body.header.off_sub_tasks, sizeof(task));
    task.logical_block_num = 2;
    task.total_required_subtasks = mix ? 6 : 2;
    if (mix) {
        task.active_mask = mix_mask;
        task.total_required_subtasks = 2 * mix_mask.count();
        task.kernel_id[0] = mix_mask.has_aic() ? 0 : -1;
        task.kernel_id[1] = mix_mask.has_aiv0() ? 1 : -1;
        task.kernel_id[2] = mix_mask.has_aiv1() ? 2 : -1;
    }
    body.put(body.header.off_sub_tasks, 0, task);
    return body;
}
struct NativeAdmissionFixture : NativeViewFixture {
    std::array<SchedulerLocalState, 3> locals;
    std::array<SchedulerAdmissionLocalState, 3> admissions;
    BoundArgumentsFixture bound;
    SchedulerRunControl *run;
    explicit NativeAdmissionFixture(bool mix, LaneMask mix_mask = LaneMask::all()) :
        NativeViewFixture(cooperative_argument_definition(mix, mix_mask), {0}, true),
        bound(*this, 0) {
        run = static_cast<SchedulerRunControl *>(backing.get());
        run->expected_task_count = graph.task_count;
        auto *contexts = scheduler_state_at<SchedulerWorkerContext>(backing.get(), layout.worker_contexts_offset);
        for (uint32_t worker = 0; worker < 3; ++worker) {
            contexts[worker].worker_index = worker;
            contexts[worker].core_type = static_cast<int32_t>(worker == 0 ? CoreType::AIC : CoreType::AIV);
            contexts[worker].cluster_index = 0;
            for (uint32_t lane = 0; lane < 3; ++lane)
                contexts[worker].cluster_worker_ids[lane] = lane;
            auto &config = locals[worker].config;
            config.graph = &graph;
            config.shared_context = contexts + worker;
            config.graph_summary_offset = layout.graph_summary_offset;
            config.worker_contexts_offset = layout.worker_contexts_offset;
            config.task_controls_offset = layout.task_controls_offset;
            config.task_metadata_offset = layout.task_metadata_offset;
            config.ready_pool_offset = ready_pool.header_offset;
            config.runtime_worker_count = 3;
            config.cluster_count = 1;
            config.self_lane = worker;
            config.self_core_type = worker == 0 ? 0 : 1;
            config.worker_ids[0] = 0;
            config.worker_ids[1] = 1;
            config.worker_ids[2] = 2;
            locals[worker].admission = &admissions[worker];
        }
        for (uint32_t worker = 0; worker < 3; ++worker)
            if (!scheduler_initialize_admission(backing.get(), &locals[worker], &admissions[worker]))
                throw std::runtime_error("admission init");
    }
    int64_t child() const { return bound.instance->child_begin; }
    SchedulerAdmissionControl *control() {
        return scheduler_state_at<SchedulerAdmissionControl>(backing.get(), layout.admission_control_offset);
    }
};
}  // namespace

TEST(HbgNativeGraphPlan, BootstrapArgumentsSurviveAdmissionRetriesForSpmdAndMix) {
    for (bool mix : {false, true}) {
        NativeAdmissionFixture view(mix);
        auto *base = view.backing.get();
        auto *local = &view.locals[0];
        auto *admission = &view.admissions[0];
        ASSERT_TRUE(scheduler_bootstrap_bind_arguments(view.graph, view.child(), local, view.run));
        ASSERT_TRUE(scheduler_bootstrap_prepare_admission(view.graph, base, local, view.run, admission, view.child()));
        view.run->bootstrap_scan_complete = 1;
        view.bound.scalar = 999;
        SchedulerReadyClaim candidate{};
        candidate.task_ref = view.child();
        ASSERT_EQ(
            scheduler_try_claim_admission(view.graph, base, local, view.run, admission, candidate),
            SchedulerAdmissionResult::PROGRESS
        );
        view.control()->commit_lock = 3;
        ASSERT_TRUE(scheduler_prepare_admission_record(view.graph, base, local, view.run, admission));
        const auto &plan = view.plan.definitions[0].children[0];
        auto *scalars = reinterpret_cast<uint64_t *>(view.bound.instance->args_base + plan.scalars_offset);
        EXPECT_EQ(scalars[0], 71);
        EXPECT_EQ(
            scheduler_try_commit_admission(view.graph, base, local, view.run, admission),
            SchedulerAdmissionResult::RETRY
        );
        ASSERT_TRUE(scheduler_prepare_admission_record(view.graph, base, local, view.run, admission));
        EXPECT_EQ(scalars[0], 71) << "admission does not rewrite bootstrap's immutable argument publication";
        view.control()->commit_lock = 0;
        ASSERT_EQ(
            scheduler_try_commit_admission(view.graph, base, local, view.run, admission),
            SchedulerAdmissionResult::PROGRESS
        );
        EXPECT_EQ(view.control()->commit_tail, 1);
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(scheduler_refresh_local_obligation(
                view.graph, base, &view.locals[worker], view.run, &view.admissions[worker]
            ));
            EXPECT_EQ(
                view.admissions[worker].current_task_ref, mix || worker == 0 ? view.child() : SCHEDULER_TASK_REF_INVALID
            );
        }
        EXPECT_EQ(scalars[0], 71);
    }
}

TEST(HbgNativeGraphPlan, BootstrapBindingFailureCannotClaimOrPublishAdmission) {
    NativeAdmissionFixture view(true);
    auto *base = view.backing.get();
    SchedulerReadyClaim candidate{};
    candidate.task_ref = view.child();
    view.bound.instance->boundary_scalars_address = 0;
    EXPECT_FALSE(scheduler_bootstrap_bind_arguments(view.graph, view.child(), &view.locals[0], view.run));
    EXPECT_EQ(view.run->bootstrap_scan_complete, 0u);
    EXPECT_EQ(view.run->bootstrap_scan_arrived_count, 0u);
    EXPECT_EQ(
        scheduler_try_claim_admission(view.graph, base, &view.locals[0], view.run, &view.admissions[0], candidate),
        SchedulerAdmissionResult::ERROR
    );
    EXPECT_FALSE(scheduler_prepare_admission_record(view.graph, base, &view.locals[0], view.run, &view.admissions[0]));
    EXPECT_NE(scheduler_read_error(view.run), 0);
    EXPECT_EQ(view.run->error_site, static_cast<uint64_t>(SchedulerErrorSite::GRAPH_BIND_SCALAR));
    auto *control = scheduler_task_control_at(base, &view.locals[0], view.child());
    EXPECT_EQ(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    EXPECT_EQ(view.control()->commit_tail, 0);
    EXPECT_EQ(view.control()->commit_lock, 0);
}

TEST(HbgNativeGraphPlan, NativeMixAdmissionIncludesOnlyActiveLanesAndCoordinator) {
    for (const auto mask : {LaneMask{true, true, false}, LaneMask{true, false, true}, LaneMask::both_aiv()}) {
        NativeAdmissionFixture view(true, mask);
        auto *base = view.backing.get();
        auto *local = &view.locals[0];
        auto *admission = &view.admissions[0];
        ASSERT_TRUE(scheduler_bootstrap_bind_arguments(view.graph, view.child(), local, view.run));
        ASSERT_TRUE(scheduler_bootstrap_prepare_admission(view.graph, base, local, view.run, admission, view.child()));
        view.run->bootstrap_scan_complete = 1;
        SchedulerReadyClaim candidate{};
        candidate.task_ref = view.child();
        ASSERT_EQ(
            scheduler_try_claim_admission(view.graph, base, local, view.run, admission, candidate),
            SchedulerAdmissionResult::PROGRESS
        );
        ASSERT_TRUE(scheduler_prepare_admission_record(view.graph, base, local, view.run, admission));
        ASSERT_EQ(
            scheduler_try_commit_admission(view.graph, base, local, view.run, admission),
            SchedulerAdmissionResult::PROGRESS
        );
        const auto *record = scheduler_admission_record_at(base, admission, view.child());
        EXPECT_EQ(record->member_count, mask.has_aic() ? 2u : 3u);
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(scheduler_refresh_local_obligation(
                view.graph, base, &view.locals[worker], view.run, &view.admissions[worker]
            ));
            const bool participates = worker == 0 || mask.has(static_cast<SubtaskSlot>(worker));
            EXPECT_EQ(
                view.admissions[worker].current_task_ref, participates ? view.child() : SCHEDULER_TASK_REF_INVALID
            );
        }
    }
}
