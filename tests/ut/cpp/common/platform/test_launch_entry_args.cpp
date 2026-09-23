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
// exactly one copy, that a failed copy publishes nothing and keeps its own
// error, and that a runtime with no launch route is unaffected. The same source
// compiles against both runtimes; `LaunchRouteSupported` is what tells them
// apart, so neither needs a separate file.
#include <gtest/gtest.h>

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
    if (rts.copy_rc != 0) return rts.copy_rc;
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
// shorter one — it stops where the entry storage starts.
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
    EXPECT_EQ(rts.copies, 1) << "the launch route replaces bytes in the copy, not the copy";
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
#endif
