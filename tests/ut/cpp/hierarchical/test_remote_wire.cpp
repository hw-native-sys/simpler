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
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "remote_wire.h"

namespace {

ContinuousTensor metadata_tensor() {
    ContinuousTensor t{};
    t.data = 0;
    t.ndims = 1;
    t.shapes[0] = 4;
    t.dtype = DataType::UINT8;
    t.child_memory = 0;
    return t;
}

}  // namespace

TEST(RemoteWire, FrameRoundTripValidatesHeader) {
    std::vector<uint8_t> payload{1, 2, 3};
    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::TASK;
    header.session_id = 7;
    header.endpoint_id = 2;
    header.sequence = 9;

    auto encoded = remote_l3::encode_frame(header, payload);
    auto decoded = remote_l3::decode_frame(encoded);

    EXPECT_EQ(decoded.header.frame_type, remote_l3::FrameType::TASK);
    EXPECT_EQ(decoded.header.session_id, 7u);
    EXPECT_EQ(decoded.header.endpoint_id, 2);
    EXPECT_EQ(decoded.header.sequence, 9u);
    EXPECT_EQ(decoded.header.payload_bytes, payload.size());
    EXPECT_EQ(decoded.payload, payload);

    encoded[0] = 'X';
    EXPECT_THROW((void)remote_l3::decode_frame(encoded), std::runtime_error);
}

TEST(RemoteWire, FrameRejectsBadVersionFlagsAndUnknownType) {
    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::HEALTH;
    header.session_id = 7;
    header.endpoint_id = 2;
    header.sequence = 9;
    auto encoded = remote_l3::encode_frame(header, {});

    auto bad_version = encoded;
    bad_version[4] = 0xFF;
    EXPECT_THROW((void)remote_l3::decode_frame(bad_version), std::runtime_error);

    auto bad_type = encoded;
    bad_type[8] = 0xFF;
    EXPECT_THROW((void)remote_l3::decode_frame(bad_type), std::runtime_error);

    auto bad_flags = encoded;
    bad_flags[36] = 1;
    EXPECT_THROW((void)remote_l3::decode_frame(bad_flags), std::runtime_error);
}

TEST(RemoteWire, TaskPayloadRejectsNonZeroTensorData) {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest.fill(0xAB);
    payload.args.tensor_metadata.push_back(metadata_tensor());
    payload.args.scalars.push_back(0xCAFE);

    auto encoded = remote_l3::encode_task_payload(payload);
    auto decoded = remote_l3::decode_task_payload(encoded.data(), encoded.size());
    ASSERT_EQ(decoded.args.tensor_metadata.size(), 1u);
    EXPECT_EQ(decoded.callable_digest[0], 0xAB);
    EXPECT_EQ(decoded.args.scalars[0], 0xCAFEu);

    payload.args.tensor_metadata[0].data = 0x1234;
    EXPECT_THROW((void)remote_l3::encode_task_payload(payload), std::runtime_error);
}

TEST(RemoteWire, TaskPayloadPreservesScopeStatsCallConfig) {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest.fill(0xAB);
    payload.config.block_dim = 7;
    payload.config.aicpu_thread_num = 5;
    payload.config.enable_scope_stats = 1;
    const char *prefix = "/tmp/remote-scope";
    std::memcpy(payload.config.output_prefix, prefix, std::strlen(prefix));
    payload.args.tensor_metadata.push_back(metadata_tensor());

    auto encoded = remote_l3::encode_task_payload(payload);
    auto decoded = remote_l3::decode_task_payload(encoded.data(), encoded.size());

    EXPECT_EQ(decoded.config.block_dim, 7);
    EXPECT_EQ(decoded.config.aicpu_thread_num, 5);
    EXPECT_EQ(decoded.config.enable_scope_stats, 1);
    EXPECT_STREQ(decoded.config.output_prefix, prefix);
}

TEST(RemoteWire, TruncatedControlPayloadIsRejected) {
    std::vector<uint8_t> truncated{1, 0, 0};
    EXPECT_THROW((void)remote_l3::decode_control(truncated.data(), truncated.size()), std::runtime_error);
}

TEST(RemoteWire, HostInlineDescriptorBoundsAreChecked) {
    remote_l3::RemoteTaskArgsWire args;
    args.tensor_metadata.push_back(metadata_tensor());
    args.inline_payload = {1, 2, 3, 4};

    RemoteTensorSidecar sidecar;
    sidecar.present = true;
    sidecar.desc.address_space = RemoteAddressSpace::HOST_INLINE;
    sidecar.desc.owner_endpoint_id = 0;
    sidecar.desc.buffer_id = 0;
    sidecar.desc.offset = 0;
    sidecar.desc.nbytes = 2;
    sidecar.desc.generation = 0;
    sidecar.desc.inline_payload_offset = 1;
    sidecar.desc.inline_payload_len = 2;
    args.remote_desc.push_back(sidecar);

    auto encoded = remote_l3::encode_remote_task_args(args);
    auto decoded = remote_l3::decode_remote_task_args(encoded.data(), encoded.size());
    ASSERT_EQ(decoded.remote_desc.size(), 1u);
    EXPECT_TRUE(decoded.remote_desc[0].present);
    EXPECT_EQ(decoded.remote_desc[0].desc.inline_payload_offset, 1u);

    args.remote_desc[0].desc.inline_payload_offset = 3;
    EXPECT_THROW((void)remote_l3::encode_remote_task_args(args), std::runtime_error);
}

TEST(RemoteWire, NonHostInlineDescriptorRejectsInlinePayloadFields) {
    remote_l3::RemoteTaskArgsWire args;
    args.tensor_metadata.push_back(metadata_tensor());

    RemoteTensorSidecar sidecar;
    sidecar.present = true;
    sidecar.desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.desc.owner_endpoint_id = 3;
    sidecar.desc.buffer_id = 9;
    sidecar.desc.generation = 1;
    sidecar.desc.nbytes = 4;
    sidecar.desc.inline_payload_len = 1;
    args.remote_desc.push_back(sidecar);

    EXPECT_THROW((void)remote_l3::encode_remote_task_args(args), std::runtime_error);
}

TEST(RemoteWire, CompletionAndControlReplyMatchSequences) {
    remote_l3::CompletionPayload completion;
    completion.sequence = 42;
    completion.error_code = 1;
    completion.error_message = "remote failure";
    auto completion_bytes = remote_l3::encode_completion(completion);
    auto decoded_completion = remote_l3::decode_completion(completion_bytes.data(), completion_bytes.size(), 42);
    EXPECT_EQ(decoded_completion.error_code, 1);
    EXPECT_EQ(decoded_completion.error_message, "remote failure");
    EXPECT_THROW(
        (void)remote_l3::decode_completion(completion_bytes.data(), completion_bytes.size(), 43), std::runtime_error
    );

    remote_l3::ControlReplyPayload reply;
    reply.sequence = 8;
    reply.control_name = remote_l3::ControlName::PREPARE_REGISTER_CALLABLE;
    reply.control_version = 1;
    reply.result_bytes = {9, 9};
    auto reply_bytes = remote_l3::encode_control_reply(reply);
    auto decoded_reply = remote_l3::decode_control_reply(
        reply_bytes.data(), reply_bytes.size(), 8, remote_l3::ControlName::PREPARE_REGISTER_CALLABLE, 1
    );
    EXPECT_EQ(decoded_reply.result_bytes, reply.result_bytes);
    EXPECT_THROW(
        (void)remote_l3::decode_control_reply(
            reply_bytes.data(), reply_bytes.size(), 8, remote_l3::ControlName::COMMIT_REGISTER_CALLABLE, 1
        ),
        std::runtime_error
    );
}

TEST(RemoteWire, RemoteBufferExportImportControlsRoundTrip) {
    remote_l3::ExportBufferRequest export_request;
    export_request.owner_endpoint_id = 3;
    export_request.buffer_id = 11;
    export_request.generation = 2;
    export_request.offset = 16;
    export_request.nbytes = 64;
    export_request.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ_WRITE;
    export_request.transport_profile = "sim";

    auto export_request_bytes = remote_l3::encode_export_buffer_request(export_request);
    auto decoded_export_request =
        remote_l3::decode_export_buffer_request(export_request_bytes.data(), export_request_bytes.size());
    EXPECT_EQ(decoded_export_request.owner_endpoint_id, 3);
    EXPECT_EQ(decoded_export_request.offset, 16u);
    EXPECT_EQ(decoded_export_request.transport_profile, "sim");

    RemoteBufferExport export_result;
    export_result.owner_endpoint_id = 3;
    export_result.buffer_id = 11;
    export_result.generation = 2;
    export_result.address_space = RemoteAddressSpace::REMOTE_WINDOW;
    export_result.offset = 16;
    export_result.nbytes = 64;
    export_result.export_id = 5;
    export_result.remote_addr = 0x1000;
    export_result.rkey_or_token = 5;
    export_result.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ_WRITE;
    export_result.transport_profile = "sim";
    export_result.transport_descriptor = {'s', 'h', 'm'};

    auto export_result_bytes = remote_l3::encode_export_buffer_result(export_result);
    auto decoded_export_result =
        remote_l3::decode_export_buffer_result(export_result_bytes.data(), export_result_bytes.size());
    EXPECT_EQ(decoded_export_result.export_id, 5u);
    EXPECT_EQ(decoded_export_result.transport_descriptor, export_result.transport_descriptor);

    remote_l3::ImportBufferRequest import_request;
    import_request.importer_endpoint_id = 4;
    import_request.requested_access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ;
    import_request.export_desc = export_result;
    auto import_request_bytes = remote_l3::encode_import_buffer_request(import_request);
    auto decoded_import_request =
        remote_l3::decode_import_buffer_request(import_request_bytes.data(), import_request_bytes.size());
    EXPECT_EQ(decoded_import_request.importer_endpoint_id, 4);
    EXPECT_EQ(decoded_import_request.requested_access_flags, remote_l3::REMOTE_BUFFER_ACCESS_READ);
    EXPECT_EQ(decoded_import_request.export_desc.owner_endpoint_id, 3);

    RemoteBufferHandle import_result;
    import_result.endpoint_id = 4;
    import_result.owner_endpoint_id = 3;
    import_result.buffer_id = 11;
    import_result.generation = 2;
    import_result.import_id = 7;
    import_result.address_space = RemoteAddressSpace::REMOTE_WINDOW;
    import_result.offset = 16;
    import_result.nbytes = 64;
    import_result.rkey_or_token = 7;
    import_result.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ;
    auto import_result_bytes = remote_l3::encode_import_buffer_result(import_result);
    auto decoded_import_result =
        remote_l3::decode_import_buffer_result(import_result_bytes.data(), import_result_bytes.size());
    EXPECT_EQ(decoded_import_result.endpoint_id, 4);
    EXPECT_EQ(decoded_import_result.owner_endpoint_id, 3);
    EXPECT_EQ(decoded_import_result.import_id, 7u);

    remote_l3::ReleaseImportRequest release_request;
    release_request.importer_endpoint_id = 4;
    release_request.owner_endpoint_id = 3;
    release_request.buffer_id = 11;
    release_request.generation = 2;
    release_request.import_id = 7;
    auto release_bytes = remote_l3::encode_release_import_request(release_request);
    auto decoded_release = remote_l3::decode_release_import_request(release_bytes.data(), release_bytes.size());
    EXPECT_EQ(decoded_release.importer_endpoint_id, 4);
    EXPECT_EQ(decoded_release.import_id, 7u);
}

TEST(RemoteWire, RemoteBufferControlsRejectInvalidAccessAndReservedBytes) {
    remote_l3::ExportBufferRequest request;
    request.owner_endpoint_id = 3;
    request.buffer_id = 11;
    request.generation = 2;
    request.nbytes = 64;
    request.transport_profile = "sim";
    request.access_flags = 0;
    EXPECT_THROW((void)remote_l3::encode_export_buffer_request(request), std::runtime_error);

    request.access_flags = 8;
    EXPECT_THROW((void)remote_l3::encode_export_buffer_request(request), std::runtime_error);

    request.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ;
    auto bytes = remote_l3::encode_export_buffer_request(request);
    bytes.back() = 1;
    EXPECT_THROW((void)remote_l3::decode_export_buffer_request(bytes.data(), bytes.size()), std::runtime_error);
}

TEST(RemoteWire, OrderedCommandLaneIsSingleFlight) {
    remote_l3::OrderedCommandLane lane;
    uint64_t first = lane.begin_command();
    EXPECT_TRUE(lane.in_flight());
    EXPECT_THROW((void)lane.begin_command(), std::runtime_error);
    EXPECT_THROW(lane.finish_reply(first + 1), std::runtime_error);
    lane.finish_reply(first);
    EXPECT_FALSE(lane.in_flight());
    EXPECT_EQ(lane.begin_command(), first + 1);
}
