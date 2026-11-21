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
#include "aicpu/hbg_execution_image.h"
#include "aicpu/runtime_launch_view.h"
#include "launch_entry_args_fixture.h"

using namespace launch_entry_args_test;
TEST_F(LaunchEntryArgs, DescriptorLayoutKeepsSharedStateAsAnExternalAddress) {
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, ccu_workspace_addr), 80U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, sm_image_bytes), 240U);
    EXPECT_LT(offsetof(DeviceRuntimeLaunchDesc, shared_state_addr), sizeof(DeviceRuntimeLaunchDesc));
}

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

TEST_F(LaunchEntryArgs, KernelRuntimeDescriptorFollowsGraphDefinitionInTheSamePackage) {
    std::array<unsigned char, 128> definition{};
    definition.fill(0x3b);
    runtime.publish_graph_section(GraphSectionSource::LaunchEnvelope, 0, definition.size());
    ASSERT_EQ(stage_graph_section(slot, definition.data(), definition.size()), 0);
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    ASSERT_EQ(helper.publish_runtime_args(true, this, rts.sink, true), 0);
    auto copied = copy_launch();
    const auto header = copied.header();
    EXPECT_EQ(header.graph_section_bytes, definition.size());
    EXPECT_EQ(header.graph_section_source, static_cast<uint32_t>(GraphSectionSource::LaunchEnvelope));
    const auto *bytes = static_cast<const unsigned char *>(copied.data());
    EXPECT_EQ(std::memcmp(bytes + header.graph_section_offset, definition.data(), definition.size()), 0);
    RuntimeLaunchView view;
    const auto *local = view.bind(*static_cast<const KernelArgs *>(copied.data()));
    ASSERT_NE(local, nullptr);
    EXPECT_EQ(view.execution_image(), nullptr);
    const auto *desc = &local->dev;
    EXPECT_EQ(desc->graph_section_base, 0U);
    EXPECT_EQ(desc->graph_section_bytes, definition.size());
    EXPECT_GE(header.runtime_section.descriptor_offset, header.graph_section_offset + header.graph_section_bytes);
    EXPECT_EQ(desc->shared_state_addr, reinterpret_cast<uint64_t>(slot.shared_state));
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(rts.copies, 0);
    definition.fill(0);
    EXPECT_EQ(bytes[header.graph_section_offset], 0x3b);
}

TEST_F(LaunchEntryArgs, KernelExecutionImageTravelsInRtsStorageAndRestoresTheWorkingArena) {
    const std::array<unsigned char, 128> expected{0x32, 0x08, 0xd5};
    auto staging = expected;
    alignas(64) std::array<unsigned char, expected.size()> working{};
    const HostMetadataRegion image{working.data(), staging.data(), staging.size(), true};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&image, 1, true), 0);
    staging.fill(0xff);

    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    capture = {};
    capture.rc = -7;
    ASSERT_EQ(publish_for_launch(helper, this, rts.sink), 0);
    EXPECT_EQ(capture.queries, 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);

    auto captured = copy_launch();
    RuntimeLaunchView view;
    const auto *local = view.bind(*static_cast<const KernelArgs *>(captured.data()));
    ASSERT_NE(local, nullptr);
    ASSERT_NE(view.execution_image(), nullptr);
    const auto aicpu_shared_state = local->dev.shared_state_addr;
    AicoreLaunchArgs core{};
    fill_shared_launch_args(core, helper.args, helper.aicore_view());
    ASSERT_NE(slot.shared_state, nullptr);
    EXPECT_EQ(aicpu_shared_state, reinterpret_cast<uint64_t>(slot.shared_state));
    EXPECT_EQ(core.shared_state_addr, aicpu_shared_state);
    for (int replay = 0; replay < 3; ++replay) {
        working.fill(0xa5);
        ASSERT_EQ(
            restore_hbg_execution_image(
                view.execution_image(), view.execution_image_bytes(), [](const void *, size_t) {},
                [](const void *, size_t) {}
            ),
            0
        );
        EXPECT_EQ(working, expected);
    }
}

TEST_F(LaunchEntryArgs, ProgramDeliversDescriptorAndExecutionImageAlongsideGraphDefinition) {
    const std::array<unsigned char, 64> initial{0xfe, 0x11};
    alignas(64) std::array<unsigned char, initial.size()> working{};
    uint64_t tensor_address = 0;
    HbgImageReference ref{reinterpret_cast<uint64_t>(&tensor_address), 32};
    const HostMetadataRegion image{working.data(), initial.data(), initial.size(), true, 16, &ref, 1};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&image, 1, true), 0);
    ref.image_offset = 0;
    working.fill(0x55);
    const std::array<unsigned char, 32> definition{0x42};
    runtime.publish_graph_section(GraphSectionSource::LaunchEnvelope, 0, definition.size());
    ASSERT_EQ(stage_graph_section(slot, definition.data(), definition.size()), 0);
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
    EXPECT_EQ(rts.copies, 0);
    auto package = copy_launch();
    const auto header = package.header();
    RuntimeLaunchView view;
    ASSERT_NE(view.bind(*static_cast<const KernelArgs *>(package.data())), nullptr);
    ASSERT_NE(view.execution_image(), nullptr);
    ASSERT_EQ(header.runtime_descriptor_bytes, RUNTIME_LAUNCH_DESCRIPTOR_BYTES);
    ASSERT_EQ(header.graph_section_bytes, definition.size());
    EXPECT_EQ(
        std::memcmp(
            static_cast<const unsigned char *>(package.data()) + header.graph_section_offset, definition.data(),
            definition.size()
        ),
        0
    );
    ASSERT_EQ(
        restore_hbg_execution_image(
            view.execution_image(), view.execution_image_bytes(), [](const void *, size_t) {},
            [](const void *, size_t) {}
        ),
        0
    );
    EXPECT_EQ(std::memcmp(working.data(), initial.data(), 16), 0);
    EXPECT_EQ(working[16], 0x55);
    EXPECT_GT(package.bytes, 4096u);
    EXPECT_EQ(tensor_address, reinterpret_cast<uint64_t>(view.execution_image()) + sizeof(HbgExecutionImage) + 32);
}

TEST_F(LaunchEntryArgs, HbgPacketRejectsOverlappingAndTruncatedSectionsBeforeReplay) {
    alignas(64) std::array<unsigned char, 64> working{};
    std::array<unsigned char, 64> initial{};
    uint64_t field = 0;
    HbgImageReference ref{reinterpret_cast<uint64_t>(&field), 32};
    const HostMetadataRegion region{working.data(), initial.data(), initial.size(), true, 16, &ref, 1};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&region, 1, true), 0);
    ASSERT_EQ(run_once(true), 0);
    const auto published = copy_launch();
    auto moved = published;
    auto &header = *static_cast<KernelArgs *>(moved.data());
    const auto valid = header;
    RuntimeLaunchView view;
    ASSERT_NE(view.bind(header), nullptr);
    const auto *image = static_cast<const unsigned char *>(view.execution_image());
    ASSERT_EQ(image, static_cast<const unsigned char *>(moved.data()) + valid.image_section.offset);
    ASSERT_EQ(view.execution_image_bytes(), sizeof(HbgExecutionImage) + initial.size() + sizeof(ref));

    header.runtime_section.descriptor_offset = 0;
    EXPECT_EQ(view.bind(header), nullptr);
    header = valid;
    header.runtime_section.packet_bytes = valid.image_section.offset + valid.image_section.bytes - 1;
    EXPECT_EQ(view.bind(header), nullptr);
    header = valid;
    header.image_section.offset = valid.runtime_section.descriptor_offset;
    EXPECT_EQ(view.bind(header), nullptr);
    header = valid;
    header.image_section.bytes = UINT32_MAX;
    EXPECT_EQ(view.bind(header), nullptr);
    header = valid;
    ASSERT_NE(view.bind(header), nullptr);
    const auto coherent = [](const void *, size_t) {};
    EXPECT_NE(restore_hbg_execution_image(image, sizeof(HbgExecutionImage) - 1, coherent, coherent), 0);
    EXPECT_NE(restore_hbg_execution_image(image, view.execution_image_bytes() - 1, coherent, coherent), 0);
    EXPECT_EQ(field, 0u);

    // Move the same well-framed packet to an eight-byte-aligned RTS slot.
    std::vector<RtsLaunchCopy::Block> small_pool((moved.bytes + 8 + 63) / 64);
    auto *packet = reinterpret_cast<unsigned char *>(small_pool.data()) + 8;
    std::memcpy(packet, moved.data(), moved.bytes);
    const auto *unaligned_image = packet + valid.image_section.offset;
    EXPECT_NE(restore_hbg_execution_image(unaligned_image, valid.image_section.bytes, coherent, coherent), 0);
    EXPECT_EQ(field, 0u);
}

// HBG ignores the entry-route permit. After initialization, both permit values
// publish the same descriptor prefix with exactly one copy.
TEST_F(LaunchEntryArgs, EitherRouteSendsExactlyOneCopy) {
    // Establish initialized shared storage and a prior descriptor publication.
    ASSERT_EQ(run_once(false), 0);
    ASSERT_TRUE(slot.workers_initialized);
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(false), 0);
    EXPECT_EQ(rts.copies, 1);
    const uint64_t descriptor_route_bytes = rts.last_copy_bytes;
    EXPECT_EQ(descriptor_route_bytes, runtime_device_copy_size(runtime));
    EXPECT_EQ(descriptor_route_bytes, sizeof(DeviceRuntimeLaunchDesc));
    helper.release_run_view();

    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1) << "the entry-route permit does not suppress the descriptor copy";
    EXPECT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    EXPECT_EQ(rts.last_copy_bytes, descriptor_route_bytes) << "a runtime with no launch route ignores the permit";
}

// HBG Program publishes its descriptor independently of shared-state initialization.
TEST_F(LaunchEntryArgs, TheFirstPublicationOntoABlockStaysOnTheDescriptor) {
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_initialized_prefix_size(runtime));
    EXPECT_EQ(helper.launch_payload_bytes(), sizeof(KernelArgs));
    EXPECT_FALSE(runtime_launch_entry_args_plan(runtime).supported);
    EXPECT_EQ(helper.args.entry_args_source, static_cast<uint32_t>(EntryArgsSource::Descriptor));

    // HBG uses the same descriptor range on every publication.
    helper.release_run_view();
    rts = {};
    ASSERT_EQ(run_once(true), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(rts.last_copy_bytes, runtime_device_copy_size(runtime));
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

    // Establish a prior publication before varying the capture query.
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
