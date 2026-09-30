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
// no launch route is unaffected. The same source compiles against both runtimes;
// `LaunchRouteSupported` is what tells them apart, so neither needs a separate
// file.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace {
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int copy_rc = 0;
    int copies = 0;
    uint64_t last_copy_bytes = 0;
    // Bytes to write before reporting `copy_rc`. Zero is the all-or-nothing
    // failure the other cases use; a positive value is the partial write a
    // real interrupted copy can leave in the block.
    uint64_t partial_bytes_before_failure = 0;
} rts;

// Whether this build's runtime carries entry values as launch arguments. Read
// from the seam itself rather than from a build flag, so the two runtimes'
// expectations cannot drift from what their own plan reports.
bool launch_route_supported(const Runtime &runtime) { return runtime_launch_entry_args_plan(runtime).supported; }

class LaunchEntryArgs : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
    void TearDown() override { EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0); }

    int prepare() { return helper.prepare_runtime_args(runtime, allocator, slot); }
    int publish(bool permitted) { return helper.publish_runtime_args(permitted); }

    // One prepare+publish, with the launch route permitted or not.
    int run_once(bool permitted) {
        const int rc = prepare();
        if (rc != 0) return rc;
        return publish(permitted);
    }

    // The published descriptor, as the device would read it.
    const DeviceRuntimeLaunchDesc &device_descriptor() const {
        return *reinterpret_cast<const DeviceRuntimeLaunchDesc *>(slot.runtime_args);
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;
};
}  // namespace

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
namespace {
struct CaptureAnswer {
    aclError rc = ACL_SUCCESS;
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    int queries = 0;
    const void *last_stream = nullptr;
} capture;
}  // namespace

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

// Both routes send exactly one descriptor copy, and the launch route's is the
// shorter one — it stops where the entry storage starts. Every publication here
// is one no record covers: a first publication onto the block, a fallback, and
// the first launch-route publication.
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
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "the first launch-route publication onto a block has no record to match";
    if (launch_route_supported(runtime)) {
        EXPECT_EQ(rts.last_copy_bytes, runtime_launch_entry_args_plan(runtime).descriptor_bytes_when_launched);
        EXPECT_LT(rts.last_copy_bytes, descriptor_route_bytes);
    } else {
        EXPECT_EQ(rts.last_copy_bytes, descriptor_route_bytes) << "a runtime with no launch route ignores the permit";
    }
}

// The first publication onto a block sends the whole initialized prefix, entry
// storage included, so it stays on the descriptor route even when the launch
// route is permitted — the values are already in the copy.
TEST_F(LaunchEntryArgs, TheFirstPublicationOntoABlockStaysOnTheDescriptor) {
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime));
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
#ifdef SIMPLER_UT_TRB_RUNTIME
    EXPECT_EQ(device_descriptor().entry_args_source_, static_cast<uint32_t>(EntryArgsSource::Descriptor))
        << "values sent inside the prefix must not also be claimed as launch arguments";
#endif

    // Only a reuse takes the launch route, and it publishes less than the
    // initialization did.
    helper.release_run_view();
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_LT(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime));
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

// The production publication step, driven through `publish_for_launch` — the
// same function both arches' launch paths call — with the capture query
// answering each way it can. What the route is, is read back off the descriptor
// that publication actually wrote.
TEST_F(LaunchEntryArgs, EveryCaptureAnswerRoutesThroughTheProductionPublishStep) {
    struct Answer {
        const char *name;
        aclError rc;
        aclmdlRICaptureStatus status;
        bool expect_launch_route;
    };
    // Unknown status: a value this build's enum does not name, which a newer
    // CANN could return. It is not NONE, so it must not open the route.
    const auto unknown_status = static_cast<aclmdlRICaptureStatus>(99);
    const Answer answers[] = {
        {"none", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_NONE, true},
        {"active", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, false},
        {"invalidated", ACL_SUCCESS, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, false},
        {"unknown status", ACL_SUCCESS, unknown_status, false},
        {"query error", -7, ACL_MODEL_RI_CAPTURE_STATUS_NONE, false},
    };

    int stream_storage = 0;
    auto *stream = static_cast<rtStream_t>(&stream_storage);

    // Initialize the block first: a first publication carries the whole prefix
    // and stays on the descriptor whatever the query says.
    capture = {};
    ASSERT_EQ(prepare(), 0);
    ASSERT_EQ(publish_for_launch(helper, stream), 0);
    ASSERT_TRUE(slot.workers_initialized);

    for (const Answer &answer : answers) {
        SCOPED_TRACE(answer.name);
        helper.release_run_view();
        rts = {};
        capture = {};
        capture.rc = answer.rc;
        capture.status = answer.status;

        ASSERT_EQ(prepare(), 0);
        ASSERT_EQ(publish_for_launch(helper, stream), 0);
        EXPECT_EQ(capture.queries, 1) << "the permit asks once, and never retries";
        EXPECT_EQ(capture.last_stream, stream) << "it must ask about the stream the launch will submit on";
        EXPECT_TRUE(helper.runtime_args_published());
        EXPECT_EQ(rts.copies, 1);

        const bool launched = answer.expect_launch_route && launch_route_supported(runtime);
        if (launched) {
            EXPECT_GT(helper.launch_payload_bytes(), sizeof(KernelArgs));
        } else {
            EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
        }
        EXPECT_EQ(
            helper.args.entry_args_source,
            static_cast<uint32_t>(launched ? EntryArgsSource::LaunchEnvelope : EntryArgsSource::Descriptor)
        );
#ifdef SIMPLER_UT_TRB_RUNTIME
        EXPECT_EQ(device_descriptor().entry_args_source_, helper.args.entry_args_source)
            << "the published descriptor and the launch header must name one route";
#endif
    }
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

#ifdef SIMPLER_UT_TRB_RUNTIME
// Everything below is about values only this runtime's descriptor carries.
#include "tensormap_and_ringbuffer/entry_args.h"

namespace {
// A run's entry values, as a caller's ChipStorageTaskArgs would supply them.
ChipStorageTaskArgs entry_args(int32_t tensors, int32_t scalars, uint64_t scalar_seed) {
    ChipStorageTaskArgs args;
    for (int32_t i = 0; i < tensors; ++i) {
        ChipTensor t{};
        t.buffer.addr = 0x1000 + static_cast<uint64_t>(i);
        args.add_tensor(t);
    }
    for (int32_t i = 0; i < scalars; ++i)
        args.add_scalar(scalar_seed + static_cast<uint64_t>(i));
    return args;
}
}  // namespace

// The published values are the ones prepare captured. A successor's prepare
// refills the source between this run's prepare and its publication, which is
// exactly the overlap the split creates.
TEST_F(LaunchEntryArgs, PublicationSendsTheCapturedValuesNotTheSourcesCurrentOnes) {
    runtime.set_orch_args(entry_args(3, 2, 0x1111));
    ASSERT_EQ(run_once(false), 0);
    ASSERT_TRUE(slot.workers_initialized);
    helper.release_run_view();

    runtime.set_orch_args(entry_args(3, 2, 0x2222));
    ASSERT_EQ(prepare(), 0);
    // The caller moves on: a successor binds different values into the same
    // Runtime while this run is still between prepare and launch.
    runtime.set_orch_args(entry_args(7, 4, 0x3333));

    ASSERT_EQ(publish(false), 0);
    EXPECT_EQ(device_descriptor().orch_args_storage_.scalar(0), 0x2222U);
    EXPECT_EQ(device_descriptor().orch_args_storage_.tensor_count(), 3);
    EXPECT_EQ(device_descriptor().entry_tensor_count_, 3U);
    EXPECT_EQ(device_descriptor().entry_scalar_count_, 2U);
}

// The launch package and the descriptor come from one capture, so the counts
// the device compares agree and the values are the same run's.
TEST_F(LaunchEntryArgs, TheLaunchPackageCarriesTheSameCaptureAsItsDescriptor) {
    ASSERT_EQ(run_once(true), 0);  // initialize the block
    helper.release_run_view();

    runtime.set_orch_args(entry_args(3, 2, 0x4444));
    ASSERT_EQ(prepare(), 0);
    runtime.set_orch_args(entry_args(9, 1, 0x5555));
    ASSERT_EQ(publish(true), 0);

    const auto plan = runtime_launch_entry_args_plan(runtime);
    ASSERT_NE(helper.launch_payload(), nullptr);
    EXPECT_EQ(helper.launch_payload_bytes(), LAUNCH_ENVELOPE_HEADER_BYTES + (3 * sizeof(simpler::tmr::Tensor) + 2 * 8))
        << "the package is the header plus this run's values, not the storage's capacity";
    EXPECT_EQ(plan.supported, true);

    const auto *package = static_cast<const unsigned char *>(helper.launch_payload());
    KernelArgs header{};
    std::memcpy(&header, package, sizeof(header));
    EXPECT_EQ(header.entry_args_source, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope));
    EXPECT_EQ(header.entry_args_offset, LAUNCH_ENVELOPE_HEADER_BYTES);
    EXPECT_EQ(header.entry_tensor_count, 3U);
    EXPECT_EQ(header.entry_scalar_count, 2U);
    // The descriptor names the same counts and the same route, so the device's
    // comparison holds.
    EXPECT_EQ(device_descriptor().entry_tensor_count_, header.entry_tensor_count);
    EXPECT_EQ(device_descriptor().entry_scalar_count_, header.entry_scalar_count);
    EXPECT_EQ(device_descriptor().entry_args_source_, header.entry_args_source);

    // And the payload's scalars are the captured run's, reachable only by
    // copying them out: the region is raw bytes at an offset the package sets.
    uint64_t scalar = 0;
    std::memcpy(&scalar, package + LAUNCH_ENVELOPE_HEADER_BYTES + 3 * sizeof(simpler::tmr::Tensor), sizeof(scalar));
    EXPECT_EQ(scalar, 0x4444U);
}

// A decode reads the region as bytes, so a source the tensor type could not be
// addressed through decodes the same values.
TEST_F(LaunchEntryArgs, DecodingAcceptsASourceNoTensorPointerCouldAddress) {
    simpler::tmr::EntryArgsStorage source;
    for (int32_t i = 0; i < 3; ++i) {
        simpler::tmr::Tensor t{};
        t.buffer.addr = 0x7000 + static_cast<uint64_t>(i);
        source.add_tensor(t);
    }
    source.add_scalar(0x6666);

    // Pack the wire region one byte off a 64-byte boundary, which is what a
    // launch buffer whose base alignment is not the type's can produce.
    const size_t wire = simpler::tmr::EntryArgsStorage::wire_bytes(3, 1);
    std::vector<unsigned char> buffer(wire + 64, 0);
    unsigned char *misaligned = buffer.data() + 1;
    std::memcpy(misaligned, source.tensors_, 3 * sizeof(simpler::tmr::Tensor));
    std::memcpy(misaligned + 3 * sizeof(simpler::tmr::Tensor), source.scalars_, sizeof(uint64_t));
    ASSERT_NE(reinterpret_cast<uintptr_t>(misaligned) % alignof(simpler::tmr::Tensor), 0U);

    simpler::tmr::EntryArgsStorage decoded;
    ASSERT_TRUE(decoded.load_from_wire(misaligned, 3, 1));
    EXPECT_EQ(decoded.tensor_count(), 3);
    EXPECT_EQ(decoded.scalar_count(), 1);
    EXPECT_EQ(decoded.tensor(2).buffer.addr, 0x7002U);
    EXPECT_EQ(decoded.scalar(0), 0x6666U);
}

// Counts the storage cannot hold are refused, and a refusal writes nothing —
// a half-decoded entry would leave counts and values disagreeing.
TEST_F(LaunchEntryArgs, DecodingRefusesCountsPastCapacityAndWritesNothing) {
    simpler::tmr::EntryArgsStorage decoded;
    decoded.add_scalar(0x9999);
    std::vector<unsigned char> buffer(1024, 0);

    EXPECT_FALSE(decoded.load_from_wire(buffer.data(), CHIP_MAX_TENSOR_ARGS + 1, 0));
    EXPECT_FALSE(decoded.load_from_wire(buffer.data(), 0, CHIP_MAX_SCALAR_ARGS + 1));
    EXPECT_EQ(decoded.tensor_count(), 0);
    EXPECT_EQ(decoded.scalar_count(), 1) << "a refused decode left the storage as it was";
    EXPECT_EQ(decoded.scalar(0), 0x9999U);
}

// The device's own check: a launch package whose counts disagree with the
// descriptor is rejected rather than reconciled.
TEST_F(LaunchEntryArgs, AdoptionRejectsCountsTheDescriptorDoesNotName) {
    runtime.set_orch_args(entry_args(2, 1, 0x7777));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);

    DeviceRuntimeLaunchDesc &published = *reinterpret_cast<DeviceRuntimeLaunchDesc *>(slot.runtime_args);
    ASSERT_EQ(published.entry_tensor_count_, 2U);
    Runtime *device_view = reinterpret_cast<Runtime *>(slot.runtime_args);
    std::vector<unsigned char> payload(simpler::tmr::EntryArgsStorage::wire_bytes(2, 1), 0);

    EXPECT_FALSE(device_view->adopt_entry_args_from_launch(payload.data(), 3, 1)) << "tensor count disagrees";
    EXPECT_FALSE(device_view->adopt_entry_args_from_launch(payload.data(), 2, 2)) << "scalar count disagrees";
    EXPECT_TRUE(device_view->adopt_entry_args_from_launch(payload.data(), 2, 1));
}

// The descriptor's wire offsets, which all three programs decode by. A field
// added ahead of the entry storage moves the values under the AICPU.
TEST_F(LaunchEntryArgs, TheDescriptorsWireOffsetsAreFixed) {
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, entry_tensor_count_), 124U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, entry_scalar_count_), 128U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, entry_args_source_), 132U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_), 192U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, workers), 34048U)
        << "the handshake region's address is what the first publication's length and the device both depend on";
    EXPECT_EQ(sizeof(simpler::tmr::EntryArgsStorage), 33856U);
    EXPECT_EQ(
        offsetof(DeviceRuntimeLaunchDesc, entry_tensor_count_), runtime_launch_entry_args_plan(runtime).control_offset
    );
}

// A count outside capacity is not a routing decision: the run fails where it is
// found, before an allocation or a copy, rather than falling back to a
// descriptor publication whose counts a consumer would then trust.
TEST_F(LaunchEntryArgs, CountsPastCapacityFailPrepareWithoutAllocatingOrCopying) {
    // Reachable only by writing the descriptor past what both builders allow:
    // ChipStorageTaskArgs::add_tensor and EntryArgsStorage::add_tensor each
    // throw at capacity, so this stands in for a corrupted descriptor.
    runtime.dev.orch_args_storage_.tensor_count_ = CHIP_MAX_TENSOR_ARGS + 1;

    const auto plan = runtime_launch_entry_args_plan(runtime);
    EXPECT_TRUE(plan.supported) << "the runtime has a launch route; it is the counts that are wrong";
    EXPECT_FALSE(plan.counts_valid);

    EXPECT_EQ(prepare(), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(slot.runtime_args, nullptr) << "nothing was allocated for a run that cannot publish";
    EXPECT_FALSE(helper.runtime_args_prepared());
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr);

    runtime.dev.orch_args_storage_.tensor_count_ = 0;
    EXPECT_EQ(prepare(), 0) << "a valid descriptor still prepares on the same slot";
}

// The launch header the device reads is the one the submission hands to RTS, not
// the one publication left behind: the run's bases are armed in between.
TEST_F(LaunchEntryArgs, TheSubmittedHeaderCarriesFieldsArmedAfterPublication) {
    runtime.set_orch_args(entry_args(3, 2, 0x8888));
    ASSERT_EQ(run_once(true), 0);  // initialize the block
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    ASSERT_NE(helper.launch_payload(), nullptr);
    ASSERT_GT(helper.launch_payload_bytes(), sizeof(KernelArgs));

    // Exactly what arming does between publication and submission: write this
    // run's wall buffer, collector bases and terminal bank into `args`.
    helper.args.device_wall_data_base = 0xdead0000;
    helper.args.chip_swimlane_data_base = 0xbeef0000;
    helper.args.chip_swimlane_run_terminal_bank = 0xfeed0000;
    helper.args.enable_profiling_flag = 0x3;

    const int copies_before_submit = rts.copies;
    const auto *package = static_cast<const unsigned char *>(helper.launch_payload());
    KernelArgs submitted{};
    std::memcpy(&submitted, package, sizeof(submitted));
    EXPECT_EQ(submitted.device_wall_data_base, 0xdead0000U) << "the device would read a pre-arming wall buffer";
    EXPECT_EQ(submitted.chip_swimlane_data_base, 0xbeef0000U);
    EXPECT_EQ(submitted.chip_swimlane_run_terminal_bank, 0xfeed0000U);
    EXPECT_EQ(submitted.enable_profiling_flag, 0x3U);
    // And the route it names is still this run's.
    EXPECT_EQ(submitted.entry_args_source, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope));
    EXPECT_EQ(submitted.entry_tensor_count, 3U);
    EXPECT_EQ(submitted.entry_scalar_count, 2U);
    EXPECT_EQ(submitted.runtime_args, helper.args.runtime_args);

    // Refreshing the header leaves the entry region alone: those bytes are the
    // prepare snapshot's, and one descriptor copy is still all this run made.
    uint64_t scalar = 0;
    std::memcpy(&scalar, package + LAUNCH_ENVELOPE_HEADER_BYTES + 3 * sizeof(simpler::tmr::Tensor), sizeof(scalar));
    EXPECT_EQ(scalar, 0x8888U);
    EXPECT_EQ(rts.copies, copies_before_submit) << "taking the payload must not copy to the device again";
}

// The decode order, as the AICPU decides it: every rejection is named, and only
// Adopt licenses forming a payload address.
TEST_F(LaunchEntryArgs, TheLaunchHeaderIsClassifiedBeforeAnyPayloadAddressExists) {
    runtime.set_orch_args(entry_args(3, 2, 0x9999));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    const Runtime &published = *reinterpret_cast<const Runtime *>(slot.runtime_args);
    ASSERT_EQ(published.get_entry_args_source(), EntryArgsSource::LaunchEnvelope);

    constexpr uint32_t kEnvelope = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    constexpr uint32_t kDescriptor = static_cast<uint32_t>(EntryArgsSource::Descriptor);
    const uint32_t offset = LAUNCH_ENVELOPE_HEADER_BYTES;

    EXPECT_EQ(classify_launch_entry_args(published, kEnvelope, offset, 3, 2), LaunchEntryArgsVerdict::Adopt);
    // A value that is neither route is refused on its own, not by matching:
    // the descriptor holding the same unknown value must not license a read.
    EXPECT_EQ(classify_launch_entry_args(published, 2, offset, 3, 2), LaunchEntryArgsVerdict::UndefinedSource);
    EXPECT_EQ(classify_launch_entry_args(published, kDescriptor, offset, 3, 2), LaunchEntryArgsVerdict::SourceMismatch);
    EXPECT_EQ(
        classify_launch_entry_args(published, kEnvelope, offset + 1, 3, 2), LaunchEntryArgsVerdict::UnexpectedOffset
    );
    EXPECT_EQ(
        classify_launch_entry_args(published, kEnvelope, offset, CHIP_MAX_TENSOR_ARGS + 1, 2),
        LaunchEntryArgsVerdict::CountsPastCapacity
    );
    EXPECT_EQ(
        classify_launch_entry_args(published, kEnvelope, offset, 3, CHIP_MAX_SCALAR_ARGS + 1),
        LaunchEntryArgsVerdict::CountsPastCapacity
    );
    EXPECT_EQ(classify_launch_entry_args(published, kEnvelope, offset, 4, 2), LaunchEntryArgsVerdict::CountsMismatch);
    EXPECT_EQ(classify_launch_entry_args(published, kEnvelope, offset, 3, 1), LaunchEntryArgsVerdict::CountsMismatch);
}

// A descriptor-route run's header says so on both sides, so the AICPU adopts
// nothing and reads its values where they already are.
TEST_F(LaunchEntryArgs, ADescriptorRouteRunAdoptsNothing) {
    runtime.set_orch_args(entry_args(2, 1, 0xaaaa));
    ASSERT_EQ(run_once(false), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(false), 0);

    const Runtime &published = *reinterpret_cast<const Runtime *>(slot.runtime_args);
    EXPECT_EQ(published.get_entry_args_source(), EntryArgsSource::Descriptor);
    EXPECT_EQ(helper.args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::Descriptor));
    EXPECT_EQ(
        classify_launch_entry_args(published, static_cast<uint32_t>(EntryArgsSource::Descriptor), 0, 0, 0),
        LaunchEntryArgsVerdict::Descriptor
    );
    EXPECT_EQ(published.get_orch_args().scalar(0), 0xaaaaU) << "the values are in the descriptor this run published";
}

// A warm launch-route publication whose 192 bytes the block already holds sends
// no copy — and is as published as a copy would have made it, launch package
// included.
TEST_F(LaunchEntryArgs, AnIdenticalWarmPrefixIsNotRepublished) {
    runtime.set_orch_args(entry_args(3, 2, 0x1234));
    ASSERT_EQ(run_once(true), 0);  // first publication onto the block: full prefix
    ASSERT_TRUE(slot.workers_initialized);
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "the first warm publication has nothing to compare against";
    EXPECT_EQ(rts.last_copy_bytes, runtime_launch_entry_args_plan(runtime).descriptor_bytes_when_launched);
    EXPECT_EQ(slot.published_prefix_bytes, rts.last_copy_bytes);
    helper.release_run_view();

    // Same callable, same shape, same config: the prefix is byte-identical.
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 0) << "the block already holds these bytes";
    EXPECT_TRUE(helper.runtime_args_published()) << "a skip still publishes the run";
    EXPECT_GT(helper.launch_payload_bytes(), sizeof(KernelArgs)) << "the launch package is still built";
    EXPECT_EQ(slot.published_prefix_bytes, runtime_launch_entry_args_plan(runtime).descriptor_bytes_when_launched);

    // And the snapshot is still consumed exactly once.
    EXPECT_NE(publish(true), 0);
    EXPECT_EQ(rts.copies, 0);
}

// A prefix that moved is republished. The count words live inside the prefix, so
// a reshaped run is one of the things that moves it.
TEST_F(LaunchEntryArgs, AChangedWarmPrefixIsRepublished) {
    runtime.set_orch_args(entry_args(3, 2, 0x1234));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    ASSERT_EQ(rts.copies, 0) << "baseline: this run would have been skipped";
    helper.release_run_view();

    // A different tensor count changes `entry_tensor_count_`, which sits ahead
    // of the entry storage and so inside the shorter prefix.
    runtime.set_orch_args(entry_args(4, 2, 0x1234));
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "a moved prefix must reach the device";
    helper.release_run_view();

    // So does a different launch shape, with the entry args untouched.
    runtime.set_worker_count(runtime.get_worker_count() + 1);
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
}

// Same counts, different values: the prefix is identical and skipped, but the
// launch package RTS copies must carry this run's payload. Equal counts are not
// equal arguments.
TEST_F(LaunchEntryArgs, AnIdenticalPrefixStillDeliversAChangedPayload) {
    runtime.set_orch_args(entry_args(3, 2, 0xaaaa));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();

    runtime.set_orch_args(entry_args(3, 2, 0xbbbb));
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 0) << "the prefix did not move: same counts, same config";

    const auto *package = static_cast<const unsigned char *>(helper.launch_payload());
    ASSERT_NE(package, nullptr);
    uint64_t scalar = 0;
    std::memcpy(&scalar, package + LAUNCH_ENVELOPE_HEADER_BYTES + 3 * sizeof(simpler::tmr::Tensor), sizeof(scalar));
    EXPECT_EQ(scalar, 0xbbbbU) << "the skipped copy is the descriptor prefix, never the arguments";
    simpler::tmr::Tensor tensor{};
    std::memcpy(&tensor, package + LAUNCH_ENVELOPE_HEADER_BYTES, sizeof(tensor));
    EXPECT_EQ(tensor.buffer.addr, 0x1000U);
}

// The counterexample the record exists for: a copy that modifies part of the
// block and then fails leaves contents nothing may describe, so a later run
// whose bytes match the pre-failure record must still copy.
TEST_F(LaunchEntryArgs, APartialCopyFailureIsNotSkippedWhenTheOldBytesReturn) {
    runtime.set_orch_args(entry_args(3, 2, 0x5555));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);  // records the prefix
    const uint32_t recorded = slot.published_prefix_bytes;
    ASSERT_GT(recorded, 0U);
    // The block's contents as the record now describes them.
    std::array<unsigned char, LAUNCH_ROUTE_PREFIX_CACHE_BYTES> recorded_contents{};
    ASSERT_LE(recorded, recorded_contents.size());
    std::memcpy(recorded_contents.data(), slot.runtime_args, recorded);
    helper.release_run_view();

    // A reshaped run whose copy reaches through the one word that moved and then
    // fails: the count is on the device, the storage the shorter prefix ends at
    // is not, and the block holds a mixture no record describes. Reaching that
    // word is what makes this a partial write rather than a no-op — a failure
    // stopping short of it would leave the recorded bytes intact.
    constexpr uint64_t partial_bytes =
        offsetof(DeviceRuntimeLaunchDesc, entry_tensor_count_) + sizeof(DeviceRuntimeLaunchDesc::entry_tensor_count_);
    runtime.set_orch_args(entry_args(4, 2, 0x5555));
    rts = {};
    rts.copy_rc = -91;
    rts.partial_bytes_before_failure = partial_bytes;
    ASSERT_EQ(prepare(), 0);
    ASSERT_LT(partial_bytes, runtime_launch_entry_args_plan(runtime).descriptor_bytes_when_launched)
        << "a write that covers the whole prefix is not the case under test";
    EXPECT_EQ(publish(true), -91) << "the copy's own error";
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr) << "no payload a caller could launch with";
    EXPECT_EQ(slot.published_prefix_bytes, 0U) << "invalidated before the copy, not after it failed";
    EXPECT_NE(std::memcmp(slot.runtime_args, recorded_contents.data(), recorded), 0)
        << "the failed copy left the block holding bytes the record no longer describes";

    // Back to the shape that was recorded before the failure: its bytes equal
    // what the record held, and the block's do not.
    runtime.set_orch_args(entry_args(3, 2, 0x5555));
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "a partially written block must be rewritten";
    EXPECT_EQ(slot.published_prefix_bytes, recorded);
    EXPECT_EQ(std::memcmp(slot.runtime_args, recorded_contents.data(), recorded), 0)
        << "and the copy is what puts those bytes back";
}

// The descriptor fallback writes the same range without recording it, so a
// launch-route run after one cannot match its way out of copying.
TEST_F(LaunchEntryArgs, ADescriptorFallbackLeavesNoRecordToMatch) {
    runtime.set_orch_args(entry_args(2, 1, 0x7777));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    ASSERT_GT(slot.published_prefix_bytes, 0U);
    helper.release_run_view();

    // Capture says no: the fallback publishes the longer prefix.
    rts = {};
    ASSERT_EQ(run_once(false), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_copy_size(runtime));
    EXPECT_EQ(slot.published_prefix_bytes, 0U) << "a route that records nothing must invalidate";
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "the launch route re-records before it can skip";
    EXPECT_GT(slot.published_prefix_bytes, 0U);
}

// The record describes an allocation, not a slot: releasing the block takes it
// with it, and the replacement re-initializes before anything can be skipped.
TEST_F(LaunchEntryArgs, AReplacementAllocationInheritsNoRecord) {
    runtime.set_orch_args(entry_args(2, 1, 0x9999));
    ASSERT_EQ(run_once(true), 0);
    helper.release_run_view();
    ASSERT_EQ(run_once(true), 0);
    ASSERT_GT(slot.published_prefix_bytes, 0U);
    helper.release_run_view();

    ASSERT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.published_prefix_bytes, 0U) << "the record belonged to the freed block";

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime))
        << "a fresh block is initialized, not matched";
    EXPECT_EQ(slot.published_prefix_bytes, 0U) << "the initializing route records nothing";
}

#else
// The other runtime's half of the seam: no launch route, and a launch payload
// that is still the header alone.
TEST_F(LaunchEntryArgs, ARuntimeWithoutALaunchRouteReportsSo) {
    EXPECT_FALSE(launch_route_supported(runtime));
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
#endif
