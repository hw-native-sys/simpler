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
#include "aicpu/runtime_launch_view.h"

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
extern "C" rtError_t rtMallocHost(void **ptr, uint64_t bytes, uint16_t) {
    *ptr = std::malloc(bytes);
    return *ptr == nullptr ? -1 : 0;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    ++rts.shared_initializations;
    EXPECT_TRUE(rts.allow_sync_initialization) << "descriptor publication must be asynchronous";
    if (!rts.allow_sync_initialization) return -99;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t) {
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
    if (!rts.recording) std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }

// The capture-status query the production permit asks. A case chooses the
// answer; every other stub here stands in for the same CANN surface the
// KernelArgsHelper cases already replace.

extern "C" aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *) { return -1; }
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
    EXPECT_TRUE(slot.workers_initialized);

    // The consumed snapshot cannot be published again; only a fresh prepare can.
    EXPECT_NE(publish(true), 0);
    EXPECT_EQ(rts.copies, 1) << "a rejected re-publish must not copy";

    rts.copy_rc = 0;
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(
        rts.last_copy_bytes, runtime_launch_entry_args_plan(runtime).supported ?
                                 runtime_launch_entry_args_plan(runtime).descriptor_bytes_when_launched :
                                 runtime_device_copy_size(runtime)
    ) << "the descriptor is republished after the failure";
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

// Publication requires the stream that orders its asynchronous copy.
TEST_F(LaunchEntryArgs, ANullStreamIsNotQueriedAndCannotPublish) {
    capture = {};
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(publish_for_launch(helper, nullptr, rts.sink), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(capture.queries, 0);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_FALSE(rts.sink.submitted());
    EXPECT_EQ(rts.copies, 0);
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

    ASSERT_EQ(publish_for_launch(helper, stream, rts.sink), 0);
    EXPECT_TRUE(helper.launchable());
    EXPECT_TRUE(helper.runtime_args_published());

    // Already published: the step is idempotent and asks nothing again.
    const int queries_before = capture.queries;
    const int copies_before = rts.copies;
    EXPECT_EQ(publish_for_launch(helper, stream, rts.sink), 0);
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
    EXPECT_EQ(publish_for_launch(helper, stream, rts.sink), -73) << "the rc the copy returned, unchanged";
    EXPECT_FALSE(helper.launchable()) << "an unpublished run must not reach a kernel submission";
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr);
    EXPECT_EQ(helper.launch_payload_bytes(), 0U);
    EXPECT_TRUE(slot.workers_initialized);
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

TEST_F(LaunchEntryArgs, KernelDescriptorInRtsStorageBorrowsSharedState) {
    std::array<uint64_t, 4> borrowed{11, 22, 33, 44};
    runtime.set_worker_count(7);
    runtime.set_gm_sm_ptr(borrowed.data());
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_NE(slot.shared_state, nullptr);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
    EXPECT_EQ(allocator.committed_bytes(), sizeof(RuntimeSharedState));
    ASSERT_EQ(publish(true), 0);
    EXPECT_EQ(rts.copies, 0);
    auto copied = copy_launch();
    RuntimeLaunchView local;
    const auto *descriptor = local.bind(*static_cast<const KernelArgs *>(copied.data()));
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->dev.worker_count, 7);
    EXPECT_EQ(descriptor->dev.shared_state_addr, reinterpret_cast<uint64_t>(slot.shared_state));
    EXPECT_EQ(descriptor->dev.gm_sm_ptr_, borrowed.data());
    EXPECT_EQ(borrowed, (std::array<uint64_t, 4>{11, 22, 33, 44}));
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
}

TEST_F(LaunchEntryArgs, TwoRtsDescriptorCopiesKeepOneSharedStateWithoutOwningBorrowedBuffers) {
    std::array<uint64_t, 2> borrowed{11, 22};
    runtime.set_gm_sm_ptr(borrowed.data());
    runtime.set_worker_count(3);
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_EQ(helper.publish_runtime_args(true, this, rts.sink, true), 0);
    auto recorded = copy_launch();
    RuntimeLaunchView first_view;
    const auto *first = first_view.bind(*static_cast<const KernelArgs *>(recorded.data()));
    ASSERT_NE(first, nullptr);
    auto *shared = slot.shared_state;
    shared->workers[0].report_epoch = 99;
    helper.release_run_view();
    runtime.set_worker_count(5);
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_EQ(publish(true), 0);
    auto eager = copy_launch();
    RuntimeLaunchView second_view;
    const auto *second = second_view.bind(*static_cast<const KernelArgs *>(eager.data()));
    ASSERT_NE(second, nullptr);
    ASSERT_NE(recorded.data(), eager.data());
    first = first_view.bind(*static_cast<const KernelArgs *>(recorded.data()));
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->dev.worker_count, 3);
    EXPECT_EQ(second->dev.worker_count, 5);
    EXPECT_EQ(first->dev.shared_state_addr, second->dev.shared_state_addr);
    EXPECT_EQ(shared->workers[0].report_epoch, 99U);
    helper.release_run_view();
    first = first_view.bind(*static_cast<const KernelArgs *>(recorded.data()));
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->dev.worker_count, 3);
    EXPECT_EQ(second->dev.gm_sm_ptr_, borrowed.data());
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
    borrowed[0] = 77;
    EXPECT_EQ(borrowed[0], 77U);
    EXPECT_EQ(borrowed[1], 22U);
}

TEST_F(LaunchEntryArgs, KernelSnapshotSurvivesHelperMove) {
    runtime.set_worker_count(7);
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_EQ(publish(true), 0);
    KernelArgsHelper moved(std::move(helper));
    helper.release_run_view();
    EXPECT_FALSE(helper.launchable());
    ASSERT_TRUE(moved.runtime_args_published());
    EXPECT_EQ(moved.runtime_launch_bytes(), runtime_device_launch_size(runtime));
    DeviceRuntimeLaunchDesc snapshot{};
    std::memcpy(&snapshot, moved.runtime_launch_data(), moved.runtime_launch_bytes());
    EXPECT_EQ(snapshot.worker_count, 7);
    EXPECT_EQ(snapshot.shared_state_addr, reinterpret_cast<uint64_t>(slot.shared_state));
    EXPECT_NE(moved.launch_payload(), nullptr);
}

TEST_F(LaunchEntryArgs, MisalignedRtsDescriptorGetsAnAlignedPrivateViewWithoutRewritingItsSource) {
    runtime.set_worker_count(7);
    runtime.set_aicpu_thread_num(3);
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_EQ(publish(true), 0);
    const auto copy = copy_launch();
    std::vector<RtsLaunchCopy::Block> storage((copy.bytes + 8 + 63) / 64);
    auto *packet = reinterpret_cast<unsigned char *>(storage.data()) + 8;
    std::memcpy(packet, copy.data(), copy.bytes);
    const auto &header = *reinterpret_cast<const KernelArgs *>(packet);
    const auto *source = packet + header.runtime_section.descriptor_offset;
    ASSERT_EQ(reinterpret_cast<uintptr_t>(source) % 64, 8U);
    RuntimeLaunchView owner;
    Runtime *view = owner.bind(header);
    ASSERT_NE(view, nullptr);
    EXPECT_NE(static_cast<void *>(view), source);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view) % alignof(DeviceRuntimeLaunchDesc), 0U);
    EXPECT_EQ(view->get_worker_count(), 7);
    EXPECT_EQ(view->get_aicpu_thread_num(), 3);
    EXPECT_EQ(view->dev.shared_state_addr, reinterpret_cast<uint64_t>(slot.shared_state));
    EXPECT_EQ(view->get_workers(), slot.shared_state->workers);
    view->set_worker_count(11);
    EXPECT_EQ(std::memcmp(packet, copy.data(), copy.bytes), 0);
    EXPECT_EQ(runtime.get_worker_count(), 7);
}

TEST_F(LaunchEntryArgs, RuntimeViewRejectsInvalidSizesAndBorrowsProgramDescriptors) {
    RuntimeLaunchView owner;
    KernelArgs header{};
    EXPECT_EQ(owner.bind(header), nullptr);
    header.runtime_args = &runtime;
    for (uint32_t bytes :
         {uint32_t(RUNTIME_LAUNCH_DESCRIPTOR_BYTES - 1), uint32_t(RUNTIME_LAUNCH_DESCRIPTOR_BYTES + 1), UINT32_MAX}) {
        header.runtime_descriptor_bytes = bytes;
        EXPECT_EQ(owner.bind(header), nullptr);
    }
    header.runtime_descriptor_bytes = 0;
    EXPECT_EQ(owner.bind(header), &runtime);
    EXPECT_EQ(runtime.get_worker_count(), 0);
}

TEST_F(LaunchEntryArgs, KernelHeaderCarriesExactDescriptorLengthWhileProgramBorrowsItsAddress) {
    ASSERT_EQ(prepare_kernel(), 0);
    ASSERT_EQ(publish(true), 0);
    auto copied = copy_launch();
    EXPECT_EQ(copied.header().runtime_descriptor_bytes, RUNTIME_LAUNCH_DESCRIPTOR_BYTES);
    helper.release_run_view();
    ASSERT_EQ(release_slot_persistent_args(slot, allocator), 0);
    ASSERT_EQ(prepare(), 0);
    ASSERT_EQ(publish(false), 0);
    KernelArgs program_header{};
    std::memcpy(&program_header, helper.launch_payload(), sizeof(program_header));
    EXPECT_EQ(program_header.runtime_descriptor_bytes, 0U);
    EXPECT_EQ(program_header.runtime_args, slot.runtime_args);
}
