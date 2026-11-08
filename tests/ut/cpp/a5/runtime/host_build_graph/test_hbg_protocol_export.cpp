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

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "async_plan.h"
#include "scheduler_protocol_export.h"
#include "scheduler_protocol_graph_source.h"

namespace {

struct ProtocolExportFixture : ::testing::Test {
    AicoreSchedulerLayout layout{};
    std::unique_ptr<scheduler_test::SchedulerStateBuffer> storage;
    std::vector<SchedulerProtocolTaskSpec> tasks;
    void SetUp() override {
        ASSERT_TRUE(scheduler_plan_layout(2, 1, 0, &layout, false, 6, {3, 1, 1, 1}, true));
        storage = std::make_unique<scheduler_test::SchedulerStateBuffer>(layout);
        auto *meta = at<SchedulerTaskMetadata>(layout.task_metadata_offset);
        meta[0].active_mask = LaneMask::all();
        meta[0].logical_block_num = 2;
        meta[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD;
        meta[0].timing_slot = -1;
        ASSERT_TRUE(scheduler_init_spmd_task_records(storage->base(), layout, 0));
        scheduler_test::BasicGraphBuffer<2> graph(2);
        graph.mixed(0, LaneMask::all());
        graph.payload(1).fanin_count = 1;
        graph.payload(1).fanin_data()[0] = 0;
        tasks = scheduler_protocol_freeze_graph(graph.graph(), {meta[0], meta[1]});
    }
    template <class T>
    T *at(uint64_t offset) {
        return scheduler_state_at<T>(storage->base(), offset);
    }
    std::string dump(uint64_t epoch = 0) {
        return scheduler_export_protocol(storage->base(), layout.total_size, layout, epoch, tasks);
    }
};

TEST_F(ProtocolExportFixture, PreservesFrozenGraphAndHostCompletedNodeWithoutFabricatingEvents) {
    at<SchedulerTaskMetadata>(layout.task_metadata_offset)[0].logical_block_num = 99;
    const auto json = dump();
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v8\""), std::string::npos);
    EXPECT_NE(json.find("\"close_seal_stride\":128"), std::string::npos);
    EXPECT_NE(json.find("\"claim_tickets\":0"), std::string::npos);
    EXPECT_NE(json.find("\"admission_claim_lock\":0"), std::string::npos);
    EXPECT_NE(json.find("\"block_num\":2"), std::string::npos);
    EXPECT_EQ(json.find("\"block_num\":99"), std::string::npos);
    EXPECT_NE(json.find("\"dependencies\":[0]"), std::string::npos);
    EXPECT_NE(json.find("\"task_events\":[]"), std::string::npos);
    EXPECT_NE(json.find("\"executions\":[]"), std::string::npos);
    EXPECT_NE(json.find("\"admission_log\":[-1]"), std::string::npos);
    EXPECT_EQ(json.find("protocol_verified"), std::string::npos);
}

TEST_F(ProtocolExportFixture, NativeSchemaCarriesBothIdentitiesAndIndependentSource) {
    tasks[0].kind = SchedulerNodeKind::CHILD_KERNEL;
    tasks[0].original_task_id = 7;
    tasks[0].local_child_id = 3;
    tasks[0].public_task_id = TaskId::to_uint64(TaskId::make_sub_task(7, 3));
    const auto json = scheduler_export_protocol(
        storage->base(), layout.total_size, layout, 0, tasks, "{\"tasks\":[],\"definitions\":[]}"
    );
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v9\""), std::string::npos);
    EXPECT_NE(json.find("\"original_task_id\":7,\"local_child_id\":3"), std::string::npos);
    EXPECT_NE(json.find("\"graph_source\":{\"tasks\":[],\"definitions\":[]}"), std::string::npos);
}

TEST(ProtocolGraphSourceTest, CopiesOriginalCsrAndDeduplicatesDefinitionsByKey) {
    scheduler_test::BasicGraphBuffer<2> original(2);
    original.payload(1).fanin_count = 1;
    original.payload(1).fanin_data()[0] = 0;
    GraphDefinition def{};
    def.graph_key = 31;
    def.task_count = 2;
    def.edge_count = 1;
    def.off_sub_tasks = sizeof(def);
    def.off_fanin_offsets = def.off_sub_tasks + 2 * sizeof(SubTaskDefinition);
    def.off_fanin_indices = def.off_fanin_offsets + 3 * sizeof(int32_t);
    def.total_bytes = def.off_fanin_indices + sizeof(uint16_t);
    std::vector<std::byte> bytes(def.total_bytes);
    std::memcpy(bytes.data(), &def, sizeof(def));
    const int32_t rows[]{0, 0, 1};
    std::memcpy(bytes.data() + def.off_fanin_offsets, rows, sizeof(rows));
    const auto json = scheduler_protocol_freeze_source(original.graph(), {{1, 0}, {1, 0}}, bytes);
    EXPECT_NE(json.find("\"dependencies\":[0]"), std::string::npos);
    const auto definitions = json.find("\"definitions\":[{\"graph_key\":31");
    ASSERT_NE(definitions, std::string::npos);
    const auto first_key = json.find("\"graph_key\":31", definitions);
    ASSERT_NE(first_key, std::string::npos);
    EXPECT_EQ(json.find("\"graph_key\":31", first_key + 1), std::string::npos);
    EXPECT_EQ(json.find("\"node_map\""), std::string::npos);
    bytes.resize(sizeof(GraphDefinition) - 1);
    EXPECT_THROW(scheduler_protocol_freeze_source(original.graph(), {{1, 0}, {1, 0}}, bytes), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, ZeroCycleAndMaximumEpochRemainLiteralValues) {
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage->base(), layout, UINT64_MAX));
    auto *events = at<SchedulerProtocolTaskTrace>(layout.protocol.tasks_offset);
    events[0].admission.before = {0, 1};
    events[0].admission.run_epoch = UINT64_MAX;
    const auto json = dump(UINT64_MAX);
    EXPECT_NE(json.find("\"before\":[0,1]"), std::string::npos);
    EXPECT_NE(json.find("\"run_epoch\":18446744073709551615"), std::string::npos);
    EXPECT_NE(json.find("\"after\":[0,0]"), std::string::npos);
}

TEST_F(ProtocolExportFixture, IncompleteAndErroneousRecordsAreExportedRatherThanDropped) {
    at<SchedulerProtocolWorkerTrace>(layout.protocol.workers_offset)[1].error_flags = SCHEDULER_PROTOCOL_DUPLICATE;
    auto *execution = at<SchedulerProtocolExecutionTrace>(layout.protocol.executions_offset);
    execution[4].publication_count = 2;
    execution[4].worker_index = 1;
    execution[4].flags = SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED;
    auto *block = at<SchedulerProtocolBlockTrace>(layout.protocol.blocks_offset);
    block[3].lanes[2].arrived_seq[1] = 99;
    const auto json = dump();
    EXPECT_NE(json.find("\"error_flags\":1"), std::string::npos);
    EXPECT_NE(json.find("\"publication_count\":2"), std::string::npos);
    EXPECT_NE(json.find("\"arrived_seq\":[0,99,0]"), std::string::npos);
    EXPECT_NE(json.find("\"index\":4"), std::string::npos);
    EXPECT_NE(json.find("\"entry\":[0,0]"), std::string::npos);
}

TEST_F(ProtocolExportFixture, RejectsStaleEpoch) { EXPECT_THROW(dump(1), std::invalid_argument); }

TEST_F(ProtocolExportFixture, RejectsCorruptHeaderBeforeFollowingItsOffsets) {
    at<SchedulerProtocolLayout>(layout.protocol_layout_offset)->executions_offset = UINT64_MAX;
    EXPECT_THROW(dump(), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, RejectsWrongSummaryEpoch) {
    at<SchedulerGraphSummary>(layout.graph_summary_offset)->run_epoch = 9;
    EXPECT_THROW(dump(), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, RejectsTruncatedOrMisalignedImageAndMissingFrozenTasks) {
    EXPECT_THROW(
        scheduler_export_protocol(storage->base(), layout.total_size - 1, layout, 0, tasks), std::invalid_argument
    );
    EXPECT_THROW(
        scheduler_export_protocol(
            static_cast<const uint8_t *>(storage->base()) + 1, layout.total_size, layout, 0, tasks
        ),
        std::invalid_argument
    );
    tasks.pop_back();
    EXPECT_THROW(dump(), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, RejectsRedirectedMemberBitmap) {
    at<SchedulerAdmissionRecord>(layout.admission_records_offset)->member_words_offset = layout.run_control_offset;
    EXPECT_THROW(dump(), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, RejectsOutOfRegionSeal) {
    at<SchedulerAdmissionRecord>(layout.admission_records_offset)->mix_close_seals_offset = layout.total_size - 8;
    EXPECT_THROW(dump(), std::invalid_argument);
}

TEST_F(ProtocolExportFixture, PreservesMemberWordsSealAndRawTopology) {
    auto *record = at<SchedulerAdmissionRecord>(layout.admission_records_offset);
    *at<uint64_t>(record->member_words_offset) = UINT64_C(0x8000000000000007);
    at<SchedulerMixCloseSeal>(record->mix_close_seals_offset)->value = 17;
    auto *contexts = at<SchedulerWorkerContext>(layout.worker_contexts_offset);
    contexts[1].physical_core_id = 65;
    contexts[1].cluster_index = 0;
    contexts[1].cluster_worker_ids[0] = 0;
    contexts[1].cluster_worker_ids[1] = 2;
    contexts[1].cluster_worker_ids[2] = 1;
    const auto json = dump();
    EXPECT_NE(json.find("\"member_words\":[9223372036854775815]"), std::string::npos);
    EXPECT_NE(json.find("\"close_seals\":[17]"), std::string::npos);
    EXPECT_NE(json.find("\"core_id\":65"), std::string::npos);
    EXPECT_NE(json.find("\"cluster_workers\":[0,2,1]"), std::string::npos);
}

TEST(ProtocolCloseSealExport, ReadsEveryClusterAndTaskValueWithoutInterpretingCellPadding) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(2, 2, 2, &layout, false, 4, {6, 2, 2, 2}, true));
    scheduler_test::SchedulerStateBuffer storage(layout);
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(storage.base(), layout.task_metadata_offset);
    scheduler_test::BasicGraphBuffer<2> graph(2);
    for (uint32_t task = 0; task < 2; ++task) {
        graph.mixed(task, LaneMask(true, true, false));
        metadata[task].active_mask = LaneMask(true, true, false);
        metadata[task].logical_block_num = 1;
        metadata[task].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
        metadata[task].trace_index_base = task * 2;
        metadata[task].timing_slot = -1;
        metadata[task].kernel_ids[0] = metadata[task].kernel_ids[1] = 1;
        metadata[task].kernel_ids[2] = UINT16_MAX;
    }
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, 91));
    const auto tasks = scheduler_protocol_freeze_graph(graph.graph(), {metadata[0], metadata[1]});
    const std::array<std::array<uint64_t, 2>, 2> expected{{{17, 29}, {41, 53}}};
    for (uint32_t task = 0; task < 2; ++task) {
        auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            storage.base(), layout.admission_records_offset + task * layout.admission_record_stride
        );
        auto *cells = scheduler_state_at<SchedulerMixCloseSeal>(storage.base(), record->mix_close_seals_offset);
        for (uint32_t cluster = 0; cluster < 2; ++cluster) {
            auto *bytes = reinterpret_cast<uint8_t *>(&cells[cluster]);
            std::memset(bytes + sizeof(uint64_t), 0xa5, sizeof(SchedulerMixCloseSeal) - sizeof(uint64_t));
            cells[cluster].value = expected[task][cluster];
        }
    }
    const auto dump = [&] {
        return scheduler_export_protocol(storage.base(), layout.total_size, layout, 91, tasks);
    };
    const auto json = dump();
    EXPECT_NE(json.find("\"close_seal_stride\":128"), std::string::npos);
    EXPECT_NE(json.find("\"close_seals\":[17,29]"), std::string::npos);
    EXPECT_NE(json.find("\"close_seals\":[41,53]"), std::string::npos);
    auto *record = scheduler_state_at<SchedulerAdmissionRecord>(storage.base(), layout.admission_records_offset);
    for (uint64_t delta : {UINT64_C(1), UINT64_C(8), UINT64_C(16), UINT64_C(128), UINT64_C(512)}) {
        SCOPED_TRACE(delta);
        record->mix_close_seals_offset = layout.mix_close_seals_offset + delta;
        EXPECT_THROW(dump(), std::invalid_argument);
    }
}

TEST(ProtocolCloseSealExport, RejectsARegionThatContainsTheValueButNotTheCompleteCell) {
    alignas(128) std::array<uint8_t, 2 * sizeof(SchedulerMixCloseSeal)> bytes{};
    const scheduler_protocol_export::Image image(bytes.data(), sizeof(SchedulerMixCloseSeal) + sizeof(uint64_t));
    EXPECT_NE(image.array<SchedulerMixCloseSeal>(0, 1), nullptr);
    EXPECT_THROW(image.array<SchedulerMixCloseSeal>(0, 2), std::invalid_argument);
    EXPECT_THROW(image.array<SchedulerMixCloseSeal>(sizeof(uint64_t), 1), std::invalid_argument);
}

TEST(ProtocolGraphSnapshot, OwnsMetadataAndFaninsAcrossHostGraphReuse) {
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    std::vector<SchedulerTaskMetadata> metadata(2);
    metadata[1].logical_block_num = 512;
    auto frozen = scheduler_protocol_freeze_graph(graph.graph(), metadata);
    graph.payload(1).fanin_data()[0] = 1;
    metadata[1].logical_block_num = 1;
    EXPECT_EQ(frozen[1].metadata.logical_block_num, 512);
    EXPECT_EQ(frozen[1].dependencies, std::vector<int32_t>({0}));
    EXPECT_THROW(scheduler_protocol_freeze_graph(graph.graph(), metadata), std::invalid_argument);
}

TEST(ProtocolGraphSnapshot, RejectsWrongTaskCountAndDuplicateDependencies) {
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 1, {0, 0});
    EXPECT_THROW(
        scheduler_protocol_freeze_graph(graph.graph(), std::vector<SchedulerTaskMetadata>(2)), std::invalid_argument
    );
    EXPECT_THROW(
        scheduler_protocol_freeze_graph(graph.graph(), std::vector<SchedulerTaskMetadata>(1)), std::invalid_argument
    );
}

TEST(ProtocolGraphSnapshot, DependencyOnlyServiceRetainsRoutingWithoutRequiringAKernel) {
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.payload(1).fanin_count = 1;
    graph.payload(1).fanin_data()[0] = 0;
    std::vector<SchedulerTaskMetadata> metadata(2);
    metadata[1].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE | SCHEDULER_TASK_HAS_FANIN;
    metadata[1].active_mask = LaneMask::aiv0();
    metadata[1].logical_block_num = 1;
    for (auto &kernel : metadata[1].kernel_ids)
        kernel = UINT16_MAX;
    const auto frozen = scheduler_protocol_freeze_graph(graph.graph(), metadata);
    EXPECT_TRUE(scheduler_task_is_inline(frozen[1].metadata.flags));
    EXPECT_EQ(frozen[1].metadata.active_mask, LaneMask::aiv0());
    EXPECT_EQ(frozen[1].metadata.kernel_ids[1], UINT16_MAX);
    EXPECT_EQ(frozen[1].dependencies, std::vector<int32_t>({0}));
}

struct AsyncProtocolExportFixture : ProtocolExportFixture {
    static constexpr uint64_t kEpoch = 17;
    SchedulerProtocolAsyncPlan frozen;

    void SetUp() override {
        ProtocolExportFixture::SetUp();
        std::vector<SchedulerTaskMetadata> metadata;
        for (const auto &task : tasks)
            metadata.push_back(task.metadata);
        std::vector<SchedulerAsyncCapabilities> capabilities(tasks.size());
        for (uint32_t lane = 0; lane < 3; ++lane)
            capabilities[0][lane] = {
                CORE_ASYNC_CAPABILITY_VERSION, lane == 1 ? CoreAsyncMode::SYNCHRONOUS : CoreAsyncMode::DEFERRED,
                lane == 1 ? 0u : 3u, lane == 1 ? 0u : CORE_ASYNC_CONDITION_TYPES
            };
        SchedulerAsyncHostPlan plan;
        std::string error;
        ASSERT_TRUE(scheduler_plan_async_state(metadata, capabilities, kEpoch, 1 << 20, {0, true}, &plan, &error))
            << error;
        std::vector<uint8_t> prefix(layout.total_size);
        std::memcpy(prefix.data(), storage->base(), prefix.size());
        uint64_t total = 0;
        ASSERT_TRUE(
            scheduler_append_async_layout(layout.total_size, plan.layout, 1 << 20, &frozen.offset, &total, &error)
        );
        frozen.layout = plan.layout;
        layout.total_size = total;
        storage = std::make_unique<scheduler_test::SchedulerStateBuffer>(layout);
        std::memcpy(storage->base(), prefix.data(), prefix.size());
        std::memcpy(static_cast<uint8_t *>(storage->base()) + frozen.offset, plan.image.data(), plan.image.size());
        ASSERT_TRUE(scheduler_init_spmd_task_records(storage->base(), layout, kEpoch));
        at<SchedulerGraphSummary>(layout.graph_summary_offset)->async_layout_offset = frozen.offset;
    }

    template <typename T>
    T &async_at(uint64_t offset, uint64_t index = 0) {
        return scheduler_state_at<T>(storage->base(), frozen.offset + offset)[index];
    }
    std::string dump_async(const std::string &source = {}) {
        return scheduler_export_protocol(storage->base(), layout.total_size, layout, kEpoch, tasks, source, frozen);
    }
    void seal(uint64_t index, uint32_t count = 0) {
        const auto &identity = async_at<SchedulerAsyncGroupIdentity>(frozen.layout.group_identity_offset, index);
        auto &writer = async_at<SchedulerAsyncGroupWriter>(frozen.layout.group_writer_offset, index);
        writer.count = count;
        writer.reservation.sealed = 1;
        async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset, index).sealed_at = {0, 2};
        const uint64_t trace = identity.block_idx * 3 + identity.subtask_slot;
        auto &execution = at<SchedulerProtocolExecutionTrace>(layout.protocol.executions_offset)[trace];
        execution.run_epoch = kEpoch;
        execution.task_ref = identity.task_ref;
        execution.block_idx = identity.block_idx;
        execution.subtask_slot = identity.subtask_slot;
        execution.worker_index = identity.subtask_slot;
        execution.generation = 9;
        execution.publication_count = 1;
        execution.entry = {0, 1};
    }
};

size_t count_text(const std::string &text, const std::string &needle) {
    size_t count = 0, cursor = 0;
    while ((cursor = text.find(needle, cursor)) != std::string::npos) {
        ++count;
        cursor += needle.size();
    }
    return count;
}

TEST_F(AsyncProtocolExportFixture, DenseNeverRunGroupsAndFinishesKeepLayoutCounts) {
    const auto json = dump_async();
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v12\""), std::string::npos);
    EXPECT_NE(json.find("\"async\":{\"group_count\":4,\"finish_count\":1"), std::string::npos);
    EXPECT_EQ(count_text(json, "\"state\":\"UNSEALED\""), 4u);
    EXPECT_EQ(count_text(json, "\"publication\":null"), 2u);
    EXPECT_EQ(count_text(json, "\"sealed\":null"), 4u);
    EXPECT_NE(json.find("\"group_indices\":[0,1,2,3]"), std::string::npos);
    EXPECT_NE(json.find("\"drained\":null"), std::string::npos);
}

TEST_F(AsyncProtocolExportFixture, EmptySealUsesActualExecutionIdentityAndLiteralZeroCycle) {
    seal(1);
    at<SchedulerTaskMetadata>(layout.task_metadata_offset)[0].trace_index_base = UINT32_MAX;
    const auto json = dump_async();
    EXPECT_NE(json.find("\"execution_index\":2,\"dispatch_generation\":9"), std::string::npos);
    EXPECT_NE(
        json.find("\"state\":\"EMPTY\",\"sealed\":{\"run_epoch\":17,\"worker_index\":2,\"at\":[0,2]}"),
        std::string::npos
    );
    EXPECT_NE(json.find("\"terminal\":null"), std::string::npos);
}

TEST_F(AsyncProtocolExportFixture, DynamicRegistrationKeepsProtocolEvidenceWithoutAnAllowedUseTable) {
    auto &identity = async_at<SchedulerAsyncGroupIdentity>(frozen.layout.group_identity_offset);
    seal(0, 1);
    const auto json = dump_async();
    EXPECT_EQ(json.find("binding_index"), std::string::npos);
    EXPECT_EQ(json.find("owner_generation"), std::string::npos);
    identity.capacity = 0;
    EXPECT_THROW(dump_async(), std::invalid_argument);
}

TEST_F(AsyncProtocolExportFixture, ConditionsKeepTypesWithoutInventingResourceBindingEvidence) {
    seal(0, 2);
    const auto &id = async_at<SchedulerAsyncGroupIdentity>(frozen.layout.group_identity_offset);
    for (uint32_t condition = 0; condition < 2; ++condition) {
        async_at<DeferredCompletionEntry>(id.entries_offset, condition).completion_type = condition;
    }
    auto &progress = async_at<SchedulerAsyncGroupProgress>(frozen.layout.group_progress_offset);
    progress.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED);
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).terminal_at = {8, 4};
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).terminal_worker_index = 1;
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).check.observation_count = 2;
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).check.slice_count = 3;
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).check.pending_slice_count = 2;
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).check.active_cycles = 11;
    async_at<SchedulerAsyncLease>(frozen.layout.group_lease_offset).word =
        scheduler_async_lease_word(7, SchedulerAsyncPhase::TERMINAL);
    const auto json = dump_async();
    EXPECT_NE(json.find("\"index\":0,\"type\":0"), std::string::npos);
    EXPECT_NE(json.find("\"index\":1,\"type\":1"), std::string::npos);
    EXPECT_NE(json.find("\"state\":\"READY\""), std::string::npos);
    EXPECT_NE(json.find("\"terminal\":{\"run_epoch\":17,\"worker_index\":1,\"at\":[8,4]}"), std::string::npos);
    EXPECT_NE(
        json.find("\"observations\":2,\"pending_count\":2,\"slice_count\":3,\"active_cycles\":11"), std::string::npos
    );
}

TEST_F(AsyncProtocolExportFixture, PartialLogicalBlockReleasePreservesItsBeforePointOnce) {
    auto &release = async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).release;
    release.before = {0, 5};
    release.worker_index = 0;
    const auto json = dump_async();
    EXPECT_EQ(
        count_text(json, "\"publication\":{\"run_epoch\":17,\"worker_index\":0,\"before\":[0,5],\"after\":null}"), 1u
    );
    EXPECT_EQ(count_text(json, "\"publication\":null"), 1u);
    release.after = {10, 6};
    EXPECT_NE(dump_async().find("\"before\":[0,5],\"after\":[10,6]"), std::string::npos);
}

TEST_F(AsyncProtocolExportFixture, DoesNotDiscardAnOrphanOrNonownerReleaseEvent) {
    auto &release = async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).release;
    release.after = {10, 6};
    EXPECT_THROW(dump_async(), std::invalid_argument);
    release.after = {};
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset, 1).release.before = {10, 6};
    EXPECT_THROW(dump_async(), std::invalid_argument);
}

TEST_F(AsyncProtocolExportFixture, FinishDrainRequiresContributionAndKeepsMissingEvidenceNull) {
    auto &progress = async_at<SchedulerAsyncFinishProgress>(frozen.layout.finish_progress_offset);
    progress.completion.publication.phase = static_cast<uint64_t>(SchedulerCompletionPhase::DRAINED);
    progress.completion.visited_count = 5;
    async_at<SchedulerAsyncFinishTrace>(frozen.layout.finish_trace_offset).finish.slice_count = 2;
    async_at<SchedulerAsyncFinishTrace>(frozen.layout.finish_trace_offset).finish.active_slice_cycles = 31;
    async_at<SchedulerAsyncLease>(frozen.layout.finish_lease_offset).word =
        scheduler_async_lease_word(3, SchedulerAsyncPhase::TERMINAL);
    auto json = dump_async();
    EXPECT_NE(json.find("\"state\":\"PENDING\",\"group_indices\":[0,1,2,3],\"drained\":null"), std::string::npos);
    progress.contributed = true;
    json = dump_async();
    EXPECT_NE(json.find("\"state\":\"DRAINED\",\"group_indices\":[0,1,2,3],\"drained\":null"), std::string::npos);
    async_at<SchedulerAsyncFinishTrace>(frozen.layout.finish_trace_offset).drained_at = {100, 7};
    async_at<SchedulerAsyncFinishTrace>(frozen.layout.finish_trace_offset).drained_worker_index = 2;
    json = dump_async();
    EXPECT_NE(json.find("\"drained\":{\"run_epoch\":17,\"worker_index\":2,\"at\":[100,7]}"), std::string::npos);
    EXPECT_NE(
        json.find("\"waiter_count\":5,\"pending_count\":0,\"slice_count\":2,\"active_cycles\":31"), std::string::npos
    );
}

TEST_F(AsyncProtocolExportFixture, ErrorEvidenceAndUnboundConditionsRemainReadable) {
    seal(0, 1);
    auto &writer = async_at<SchedulerAsyncGroupWriter>(frozen.layout.group_writer_offset);
    writer.error_code = -7;
    auto &progress = async_at<SchedulerAsyncGroupProgress>(frozen.layout.group_progress_offset);
    progress.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED);
    progress.failure_code = -2;
    auto &finish = async_at<SchedulerAsyncFinishProgress>(frozen.layout.finish_progress_offset);
    finish.completion.publication.phase = static_cast<uint64_t>(SchedulerCompletionPhase::FAILED);
    finish.failure_code = -19;
    at<SchedulerRunControl>(layout.run_control_offset)->scheduler_error = 7;
    const auto json = dump_async();
    EXPECT_EQ(count_text(json, "\"state\":\"FAILED\""), 2u);
    EXPECT_NE(json.find("\"failure_code\":-7"), std::string::npos);
    EXPECT_NE(json.find("\"failure_code\":-19"), std::string::npos);
    EXPECT_EQ(json.find("binding_index"), std::string::npos);
    EXPECT_EQ(json.find("owner_generation"), std::string::npos);
    EXPECT_EQ(count_text(json, "\"state\":\"UNSEALED\""), 3u);
}

TEST_F(AsyncProtocolExportFixture, GraphAsyncSchemaKeepsExistingNodeMapAndSource) {
    tasks[0].kind = SchedulerNodeKind::CHILD_KERNEL;
    tasks[0].original_task_id = 7;
    tasks[0].local_child_id = 3;
    tasks[0].public_task_id = TaskId::to_uint64(TaskId::make_sub_task(7, 3));
    const auto json = dump_async("{\"tasks\":[],\"definitions\":[]}");
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v13\""), std::string::npos);
    EXPECT_NE(json.find("\"original_task_id\":7,\"local_child_id\":3"), std::string::npos);
    EXPECT_NE(json.find("\"graph_source\":{\"tasks\":[],\"definitions\":[]}"), std::string::npos);
}

TEST_F(AsyncProtocolExportFixture, RejectsChangedImmutableCountsAndLocator) {
    auto &wire = async_at<SchedulerAsyncLayout>(0);
    --wire.group_count;
    EXPECT_THROW(dump_async(), std::invalid_argument);
    wire = frozen.layout;
    at<SchedulerGraphSummary>(layout.graph_summary_offset)->async_layout_offset += 128;
    EXPECT_THROW(dump_async(), std::invalid_argument);
}

TEST_F(AsyncProtocolExportFixture, RejectsConditionsRedirectedToAnotherSectionAndCapacityOverflow) {
    auto &identity = async_at<SchedulerAsyncGroupIdentity>(frozen.layout.group_identity_offset);
    const auto saved = identity;
    identity.entries_offset = frozen.layout.group_writer_offset;
    EXPECT_THROW(dump_async(), std::invalid_argument);
    identity = saved;
    async_at<SchedulerAsyncGroupWriter>(frozen.layout.group_writer_offset).count = identity.capacity + 1;
    EXPECT_THROW(dump_async(), std::invalid_argument);
}

TEST_F(AsyncProtocolExportFixture, RejectsWrongMapOwnershipAndRecordedSealWithoutExecution) {
    auto &map = async_at<SchedulerAsyncTaskMap>(frozen.layout.task_map_offset);
    const auto saved = map;
    map.group_base = 1;
    EXPECT_THROW(dump_async(), std::invalid_argument);
    map = saved;
    async_at<SchedulerAsyncGroupTrace>(frozen.layout.group_trace_offset).sealed_at = {10, 1};
    EXPECT_THROW(dump_async(), std::invalid_argument);
}

TEST_F(AsyncProtocolExportFixture, RejectsOversizedFinishBatchWithoutClamping) {
    auto &finish = async_at<SchedulerAsyncFinishProgress>(frozen.layout.finish_progress_offset);
    finish.completion.publication.pending_count = 8;
    EXPECT_NO_THROW(dump_async());
    finish.completion.publication.pending_count = 9;
    try {
        (void)dump_async();
        FAIL() << "oversized batch accepted";
    } catch (const std::invalid_argument &error) {
        EXPECT_NE(std::string(error.what()).find("finishes[0].pending_count out of range: 9"), std::string::npos);
    }
}

}  // namespace
