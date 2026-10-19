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
#include <cstddef>
#include <cstring>
#include "launch_entry_args_fixture.h"

using namespace launch_entry_args_test;
// The other runtime's half of the seam: no launch route, and a launch payload
// that is still the header alone.
TEST_F(LaunchEntryArgs, ARuntimeWithoutALaunchRouteReportsSo) {
    EXPECT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    const auto plan = runtime_launch_entry_args_plan(runtime);
    EXPECT_EQ(plan.tensor_count, 0U);
    EXPECT_EQ(plan.scalar_count, 0U);
    EXPECT_EQ(plan.payload_bytes, 0U);

    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(helper.launch_payload(), static_cast<void *>(&helper.args));
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
    EXPECT_EQ(helper.args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::Descriptor));
    EXPECT_EQ(helper.args.entry_args_offset, 0U);
}

// Unsupported and invalid are separate answers, and a runtime that examines no
// counts reports them valid: a caller must not read "no launch route" as "these
// counts are wrong", nor the reverse.
TEST_F(LaunchEntryArgs, AnAbsentLaunchRouteIsNotAnInvalidCount) {
    const auto plan = runtime_launch_entry_args_plan(runtime);
    EXPECT_FALSE(plan.supported);
    EXPECT_TRUE(plan.counts_valid);
    EXPECT_EQ(prepare(), 0) << "nothing here fails a prepare";
}

// This runtime's other half of the same seam: the Graph section takes the
// package the entry region would have taken, and takes it only when the slot's
// staging and the run's descriptor name one length between them.
TEST_F(LaunchEntryArgs, AGraphSectionTravelsOnlyWhenBothSidesNameTheSameLength) {
    std::array<unsigned char, 128> section{};
    section.fill(0x3b);
    runtime.publish_graph_section(GraphSectionSource::LaunchEnvelope, 0, static_cast<uint32_t>(section.size()));
    ASSERT_EQ(stage_graph_section(slot, section.data(), section.size()), 0);

    ASSERT_EQ(run_once(false), 0);
    EXPECT_EQ(helper.args.graph_section_bytes, section.size());
    EXPECT_EQ(helper.args.graph_section_offset, static_cast<uint32_t>(LAUNCH_ENVELOPE_HEADER_BYTES));
    EXPECT_EQ(helper.args.graph_section_source, static_cast<uint32_t>(GraphSectionSource::LaunchEnvelope));
    const auto *package = static_cast<const unsigned char *>(helper.launch_payload());
    ASSERT_NE(package, nullptr);
    EXPECT_EQ(helper.launch_payload_bytes(), LAUNCH_ENVELOPE_HEADER_BYTES + section.size());
    EXPECT_EQ(std::memcmp(package + LAUNCH_ENVELOPE_HEADER_BYTES, section.data(), section.size()), 0)
        << "the bytes RTS would copy are the ones the bind staged";
    helper.release_run_view();

    // Half the length under the same descriptor: this run's own bind
    // disagreeing with itself, which fails the publication rather than naming a
    // length nothing staged.
    ASSERT_EQ(stage_graph_section(slot, section.data(), section.size() / 2), 0);
    ASSERT_EQ(prepare(), 0);
    EXPECT_NE(publish(false), 0);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr) << "a refused publication submits nothing";
    helper.release_run_view();
}

// HBG ignores the entry-route permit. After initialization, both permit values
// publish the same descriptor prefix with exactly one copy.
TEST_F(LaunchEntryArgs, EitherRouteSendsExactlyOneCopy) {
    // Take the first publication out of the way: it carries the whole
    // initialized prefix whichever route is permitted.
    ASSERT_EQ(run_once(false), 0);
    ASSERT_TRUE(slot.workers_initialized);
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(false), 0);
    EXPECT_EQ(rts.copies, 1);
    const uint64_t descriptor_route_bytes = rts.last_copy_bytes;
    EXPECT_EQ(descriptor_route_bytes, runtime_device_copy_size(runtime));
    EXPECT_EQ(descriptor_route_bytes, offsetof(DeviceRuntimeLaunchDesc, workers));
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "the entry-route permit does not suppress the descriptor copy";
    EXPECT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    EXPECT_EQ(rts.last_copy_bytes, descriptor_route_bytes) << "a runtime with no launch route ignores the permit";
}

// HBG initializes the full host-owned prefix on first publication, then
// publishes only its steady prefix on reuse. Both use the descriptor route.
TEST_F(LaunchEntryArgs, TheFirstPublicationOntoABlockStaysOnTheDescriptor) {
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime));
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
    EXPECT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    EXPECT_EQ(helper.args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::Descriptor));

    // Reuse sends only the steady prefix; HBG still uses the descriptor route.
    helper.release_run_view();
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_LT(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime));
}

// The production publication step, driven through `publish_for_launch` — the
// same function both arches' launch paths call — with the capture query
// answering each way it can. The launch header must retain the descriptor
// route for every answer because HBG has no launch-entry payload route.
TEST_F(LaunchEntryArgs, EveryCaptureAnswerRoutesThroughTheProductionPublishStep) {
    ASSERT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    struct Answer {
        const char *name;
        aclError rc;
        aclmdlRICaptureStatus status;
    };
    // Unknown status: a value this build's enum does not name, which a newer
    // CANN could return. It is not NONE, so it must not open the route.
    const auto unknown_status = static_cast<aclmdlRICaptureStatus>(99);
    const Answer answers[] = {
        {"none", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_NONE},
        {"active", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE},
        {"invalidated", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED},
        {"unknown status", ACL_SUCCESS, unknown_status},
        {"query error", -7, ACL_MODEL_RI_CAPTURE_STATUS_NONE},
    };

    int stream_storage = 0;
    auto *stream = static_cast<rtStream_t>(&stream_storage);

    // Initialize the block first: a first publication carries the whole prefix
    // and stays on the descriptor whatever the query says.
    capture = {};
    ASSERT_EQ(prepare(), 0);
    ASSERT_EQ(publish_for_launch(helper, stream, rts.sink), 0);
    ASSERT_TRUE(slot.workers_initialized);

    for (const Answer &answer : answers) {
        SCOPED_TRACE(answer.name);
        helper.release_run_view();
        rts = {};
        capture = {};
        capture.rc = answer.rc;
        capture.status = answer.status;

        ASSERT_EQ(prepare(), 0);
        ASSERT_EQ(publish_for_launch(helper, stream, rts.sink), 0);
        EXPECT_EQ(capture.queries, 1) << "the permit asks once, and never retries";
        EXPECT_EQ(capture.last_stream, stream) << "it must ask about the stream the launch will submit on";
        EXPECT_TRUE(helper.runtime_args_published());
        EXPECT_EQ(rts.copies, 1);

        EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
        EXPECT_EQ(helper.args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::Descriptor));
    }
}
