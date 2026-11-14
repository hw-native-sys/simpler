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

#include "notification_metadata_plan.h"
#include "notification_context_plan.h"
#include "hbg_scheduler_test_support.h"

namespace {
using namespace simpler::hbg;

NotificationPlanStatus query(uint32_t mode, const NotificationPlanInput *, NotificationEventSink sink, void *state) {
    NotificationPlanEvent event{};
    event.ndims = 2;
    event.offsets[0] = 1;
    event.offsets[1] = mode == 2 ? 8 : 2;
    event.value = 9;
    if (mode == 3) {
        event.kind = NotificationEventKind::SET;
        event.peer = 1;
        event.comm_context = 8192;
    }
    if (!sink(state, &event)) return NotificationPlanStatus::INVALID;
    return mode == 1 ? NotificationPlanStatus::UNSUPPORTED : NotificationPlanStatus::COMPLETE;
}

TEST(HbgNotificationMetadata, BoundStridedCoordinatesRemainLocalAndPartialPlansAreRejected) {
    TensorData tensor{};
    tensor.buffer.addr = 4096;
    tensor.buffer.size = 1024;
    tensor.start_offset = 3;
    tensor.dtype = DataType::INT32;
    tensor.address_space = AddressSpace::DEVICE;
    tensor.ndims = 2;
    tensor.shapes[0] = 4;
    tensor.shapes[1] = 8;
    tensor.strides[0] = 16;
    tensor.strides[1] = 1;
    NotificationPlanInput input{};
    input.tensor_count = 1;
    input.tensors = &tensor;
    input.block_count = 2;
    input.block_index = 1;
    input.lane = 1;
    std::vector<SchedulerBoundNotificationEvent> events;
    std::string error;
    EXPECT_EQ(
        scheduler_collect_notification_metadata(query, 0, 7, input, &events, &error), NotificationPlanStatus::COMPLETE
    );
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].local_counter_address, 4180u);
    EXPECT_EQ(events[0].task_ref, 7);
    EXPECT_EQ(events[0].block_index, 1u);
    EXPECT_EQ(events[0].effect.peer, -1);
    EXPECT_EQ(events[0].effect.value, 9);
    EXPECT_EQ(
        scheduler_collect_notification_metadata(query, 1, 8, input, &events, &error),
        NotificationPlanStatus::UNSUPPORTED
    );
    EXPECT_EQ(events.size(), 1u);
    EXPECT_EQ(
        scheduler_collect_notification_metadata(query, 2, 8, input, &events, &error), NotificationPlanStatus::INVALID
    );
    EXPECT_EQ(events.size(), 1u);
}

TEST(HbgNotificationMetadata, OwnedCounterIdentityProducesOnlyItsMatchingWaitOrder) {
    scheduler_test::BasicGraphBuffer<8> graph(8);
    for (uint32_t ref = 0; ref < 8; ++ref)
        graph.executable(ref, 0);
    auto context = std::make_shared<HostNotificationContext>();
    context->bootstrap_token = 9;
    context->allocation_id = 4;
    context->context_address = 1024;
    context->rank_count = 1;
    context->window_bytes = 1024;
    context->windows_in[0] = 4096;
    SchedulerNotificationContexts contexts{{1024, context}};
    SchedulerBoundNotificationEvent wait{};
    wait.task_ref = 0;
    wait.local_counter_address = 4180;
    wait.effect.comm_context = 1024;
    wait.effect.value = 1;
    auto producer = wait;
    producer.task_ref = 7;
    producer.effect.kind = NotificationEventKind::SET;
    producer.effect.peer = 0;
    SchedulerNotificationMetadataPlan metadata;
    metadata.events = {wait, producer};
    std::vector<SchedulerTaskOrderEdge> edges;
    std::string error;
    ASSERT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error))
        << error;
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].predecessor, 7);
    EXPECT_EQ(edges[0].successor, 0);
    metadata.events[1].local_counter_address += 4;
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error));
    EXPECT_TRUE(edges.empty());
    metadata.events = {wait, producer};
    context->rank_count = 2;
    ASSERT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error))
        << error;
    EXPECT_EQ(edges.size(), 1u);
    metadata.events[1].effect.peer = 1;
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error));
    EXPECT_TRUE(edges.empty());
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, {}, 11, graph.graph(), {}, &edges, &error));
    EXPECT_TRUE(edges.empty());
    metadata.events = {wait, producer};
    metadata.unavailable.push_back({6, 0, 0, 0, NOTIFICATION_KERNEL_NOTIFY, NotificationPlanStatus::UNSUPPORTED});
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error));
    EXPECT_TRUE(edges.empty());
    metadata.unavailable.clear();
    producer.task_ref = 1;
    auto later = producer;
    later.task_ref = 3;
    later.effect.value = 2;
    wait.effect.value = 2;
    metadata.events = {wait, producer, later};
    graph.executable(2, 0, {1});
    graph.executable(3, 0, {2});
    ASSERT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {}, &edges, &error))
        << error;
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(edges[0].predecessor, 3);
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, contexts, 11, graph.graph(), {2}, &edges, &error));
    EXPECT_TRUE(edges.empty());
    metadata.events.clear();
    EXPECT_TRUE(scheduler_resolve_notification_order(metadata, {}, 11, graph.graph(), {}, &edges, &error));
    EXPECT_TRUE(edges.empty());
}

TEST(HbgNotificationMetadata, UnavailableWaitMetadataKeepsExistingBehaviorAndRemotePeersStayUnresolved) {
    TensorData tensor{};
    tensor.buffer.addr = 4096;
    tensor.buffer.size = 1024;
    tensor.dtype = DataType::INT32;
    tensor.address_space = AddressSpace::DEVICE;
    tensor.ndims = 2;
    tensor.shapes[0] = 4;
    tensor.shapes[1] = 8;
    tensor.strides[0] = 16;
    tensor.strides[1] = 1;
    NotificationPlanInput input{};
    input.tensor_count = 1;
    input.tensors = &tensor;
    SchedulerNotificationMetadataPlan plan;
    std::string error;
    EXPECT_TRUE(
        scheduler_collect_notification_invocation(nullptr, NOTIFICATION_KERNEL_OPAQUE, 0, 0, input, &plan, &error)
    );
    EXPECT_TRUE(plan.events.empty());
    EXPECT_TRUE(plan.unavailable.empty());
    EXPECT_TRUE(scheduler_collect_notification_invocation(query, NOTIFICATION_KERNEL_WAIT, 1, 1, input, &plan, &error));
    EXPECT_TRUE(plan.events.empty());
    ASSERT_EQ(plan.unavailable.size(), 1u);
    EXPECT_EQ(plan.unavailable[0].status, NotificationPlanStatus::UNSUPPORTED);
    EXPECT_TRUE(
        scheduler_collect_notification_invocation(query, NOTIFICATION_KERNEL_NOTIFY, 3, 2, input, &plan, &error)
    );
    ASSERT_EQ(plan.events.size(), 1u);
    EXPECT_EQ(plan.events[0].local_counter_address, 4168u);
    EXPECT_EQ(plan.events[0].effect.peer, 1);
    EXPECT_EQ(plan.events[0].effect.comm_context, 8192u);
}
}  // namespace
