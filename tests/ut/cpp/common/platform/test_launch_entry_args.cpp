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
// Which route one run's entry arguments take to the device, driven through the
// real onboard helper and allocator with host-backed RTS storage.
//
// The route is chosen at publication, after prepare captured the values, so the
// questions here are about that seam: that the published bytes are the captured
// ones and not whatever the source holds by then, that either route sends
// exactly one copy of a range the block is not already known to hold, that a
// failed copy publishes nothing and keeps its own error, and that a runtime with
// no launch route is unaffected. Shared cases compile against both runtimes;
// backend-specific expectations live in the HBG and TMR case files.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "acl/acl.h"
#include "device_runner_helpers.h"
#include "launch_entry_args_fixture.h"

namespace launch_entry_args_test {
RtsState rts;
CaptureAnswer capture;
}  // namespace launch_entry_args_test
using namespace launch_entry_args_test;

extern "C" rtError_t rtMalloc(void **ptr, uint64_t bytes, uint32_t, uint16_t) {
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) return -1;
    std::memset(*ptr, kDeviceMark, bytes);
    return 0;
}
extern "C" rtError_t rtFree(void *ptr) {
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    ++rts.copies;
    rts.last_copy_bytes = bytes;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0) {
        if (rts.partial_bytes_before_failure > 0) {
            std::memcpy(dst, src, std::min(rts.partial_bytes_before_failure, bytes));
        }
        return rts.copy_rc;
    }
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }

// The capture-status query the production permit asks. A case chooses the
// answer; every other stub here stands in for the same CANN surface the
// KernelArgsHelper cases already replace.

extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI) {
    ++capture.queries;
    capture.last_stream = stream;
    if (modelRI != nullptr) *modelRI = nullptr;
    if (capture.rc != ACL_SUCCESS) return capture.rc;
    if (status != nullptr) *status = capture.status;
    return ACL_SUCCESS;
}

// A launch entry must admit a prepared run, because publication is the first
// thing the launch does; only a kernel submission needs Published.
TEST_F(LaunchEntryArgs, PreparedIsReachableAndPublishingIsWhatCompletesIt) {
    ASSERT_EQ(prepare(), 0);
    EXPECT_TRUE(helper.runtime_args_prepared());
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(rts.copies, 0) << "prepare must not copy: the route is not decided yet";

    ASSERT_EQ(publish(false), 0);
    EXPECT_FALSE(helper.runtime_args_prepared());
    EXPECT_TRUE(helper.runtime_args_published());
}

// A publication failure keeps its own error, publishes nothing, and leaves no
// payload a caller could submit a kernel with.
TEST_F(LaunchEntryArgs, AFailedPublicationLeavesNothingLaunchable) {
    ASSERT_EQ(prepare(), 0);
    rts.copy_rc = -91;
    EXPECT_EQ(publish(true), -91) << "the copy's own error, not a substituted one";
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_FALSE(helper.runtime_args_prepared());
    EXPECT_EQ(helper.launch_payload(), nullptr);
    EXPECT_EQ(helper.launch_payload_bytes(), 0U);
    EXPECT_FALSE(slot.workers_initialized);

    // The consumed snapshot cannot be published again; only a fresh prepare can.
    EXPECT_NE(publish(true), 0);
    EXPECT_EQ(rts.copies, 1) << "a rejected re-publish must not copy";

    rts.copy_rc = 0;
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime))
        << "the retry initializes, rather than assuming the failed copy landed";
}

// A second publish of one snapshot is refused whether or not the first
// succeeded: a run publishes once.
TEST_F(LaunchEntryArgs, ASnapshotIsPublishedAtMostOnce) {
    ASSERT_EQ(run_once(true), 0);
    const int copies_after_publish = rts.copies;
    EXPECT_NE(publish(true), 0);
    EXPECT_NE(publish(false), 0);
    EXPECT_EQ(rts.copies, copies_after_publish);
}

// A null stream cannot be asked, so it is not asked — and routes through the
// descriptor, which needs no query to be correct.
TEST_F(LaunchEntryArgs, ANullStreamIsNotQueriedAndTakesTheDescriptor) {
    capture = {};
    ASSERT_EQ(prepare(), 0);
    ASSERT_EQ(publish_for_launch(helper, nullptr), 0);
    EXPECT_EQ(capture.queries, 0);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
}

// What a launch entry admits, as both arches ask it. An unpublished, unprepared
// run is refused before anything is submitted; a failed publication returns to
// that state.
TEST_F(LaunchEntryArgs, LaunchAdmissionTracksThePublicationStates) {
    int stream_storage = 0;
    auto *stream = static_cast<rtStream_t>(&stream_storage);
    capture = {};

    EXPECT_FALSE(helper.launchable()) << "an empty helper names no descriptor to publish";

    ASSERT_EQ(prepare(), 0);
    EXPECT_TRUE(helper.launchable()) << "prepared is admitted: publication is what the launch does first";

    ASSERT_EQ(publish_for_launch(helper, stream), 0);
    EXPECT_TRUE(helper.launchable());
    EXPECT_TRUE(helper.runtime_args_published());

    // Already published: the step is idempotent and asks nothing again.
    const int queries_before = capture.queries;
    const int copies_before = rts.copies;
    EXPECT_EQ(publish_for_launch(helper, stream), 0);
    EXPECT_EQ(capture.queries, queries_before);
    EXPECT_EQ(rts.copies, copies_before);

    helper.release_run_view();
    EXPECT_FALSE(helper.launchable());
}

// A copy that fails takes the run out of every launchable state, keeps its own
// error, and leaves nothing that could be submitted as a kernel.
TEST_F(LaunchEntryArgs, AFailedProductionPublishReturnsItsOwnErrorAndNoPayload) {
    int stream_storage = 0;
    auto *stream = static_cast<rtStream_t>(&stream_storage);
    capture = {};
    rts.copy_rc = -73;

    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(publish_for_launch(helper, stream), -73) << "the rc the copy returned, unchanged";
    EXPECT_FALSE(helper.launchable()) << "an unpublished run must not reach a kernel submission";
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr);
    EXPECT_EQ(helper.launch_payload_bytes(), 0U);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.copies, 1) << "one attempt, not a retry";
}

// The routing table itself, stated once and asked directly, so a status this
// build does not name cannot drift into opening the route.
TEST_F(LaunchEntryArgs, OnlyASuccessfulNoCaptureAnswerOpensTheLaunchRoute) {
    EXPECT_TRUE(launch_route_permitted_by_capture(ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_NONE));
    EXPECT_FALSE(launch_route_permitted_by_capture(ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE));
    EXPECT_FALSE(launch_route_permitted_by_capture(ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED));
    EXPECT_FALSE(launch_route_permitted_by_capture(ACL_SUCCESS, 99));
    EXPECT_FALSE(launch_route_permitted_by_capture(-7, ACL_MODEL_RI_CAPTURE_STATUS_NONE));
}

// A Graph section is staged on the slot and named by the descriptor, and the
// publication takes both or neither. The scenario is a run whose prepare failed
// after staging: the length stays on the slot, and the next run on it names no
// section, so the two disagree and the staged bytes must not reach that run's
// header. Nothing in the section's own decoding can catch this — by then the
// header already names a predecessor's bytes.
TEST_F(LaunchEntryArgs, AStagedSectionNoDescriptorNamesIsNotPublished) {
    std::array<unsigned char, 96> section{};
    section.fill(0x7c);
    ASSERT_EQ(stage_graph_section(slot, section.data(), section.size()), 0);
    ASSERT_EQ(slot.graph_section_bytes, section.size());
    ASSERT_EQ(runtime_graph_section_bytes(runtime), 0U) << "this run submits no Graph task";

    ASSERT_EQ(run_once(false), 0);

    EXPECT_EQ(helper.args.graph_section_bytes, 0U);
    EXPECT_EQ(helper.args.graph_section_offset, 0U);
    EXPECT_EQ(helper.args.graph_section_source, static_cast<uint32_t>(GraphSectionSource::None));
    EXPECT_EQ(slot.graph_section_bytes, 0U) << "the staged length is consumed, not left for the run after this one";
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs)) << "the package the section sits in is not submitted";

    // And the run after it, which stages nothing, is unchanged by any of that.
    helper.release_run_view();
    ASSERT_EQ(run_once(false), 0);
    EXPECT_EQ(helper.args.graph_section_bytes, 0U);
    EXPECT_EQ(helper.args.graph_section_source, static_cast<uint32_t>(GraphSectionSource::None));
}
