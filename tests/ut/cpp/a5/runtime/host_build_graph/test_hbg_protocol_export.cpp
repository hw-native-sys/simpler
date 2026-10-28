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

#include <string>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler_protocol_export.h"

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
    EXPECT_NE(json.find("\"schema\":\"hbg-native-protocol-v5\""), std::string::npos);
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
    *at<uint64_t>(record->mix_close_seals_offset) = 17;
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

}  // namespace
