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

// A task arg bound for a remote worker: a REMOTE_SIDECAR placeholder naming remote buffer 9 on
// worker 3, with no local backing of its own. That is the only backend this wire accepts.
Tensor remote_arg_tensor() {
    Tensor tensor{};
    tensor.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    tensor.buffer.address_space = static_cast<uint8_t>(AddressSpace::HOST);
    tensor.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    tensor.buffer.backend_kind = static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR);
    tensor.buffer.identity.owner_instance_id[0] = 3;
    tensor.buffer.identity.buffer_id = 9;
    tensor.buffer.identity.generation = 1;
    tensor.buffer.nbytes = 4;
    tensor.ndims = 1;
    tensor.shapes[0] = 4;
    tensor.strides[0] = 1;
    tensor.dtype = DataType::UINT8;
    validate_tensor(tensor);
    return tensor;
}

}  // namespace

TEST(RemoteWire, FrameRoundTripValidatesHeader) {
    std::vector<uint8_t> payload{1, 2, 3};
    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::TASK;
    header.session_id = 7;
    header.worker_id = 2;
    header.sequence = 9;

    auto encoded = remote_l3::encode_frame(header, payload);
    auto decoded = remote_l3::decode_frame(encoded);

    EXPECT_EQ(decoded.header.frame_type, remote_l3::FrameType::TASK);
    EXPECT_EQ(decoded.header.session_id, 7u);
    EXPECT_EQ(decoded.header.worker_id, 2);
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
    header.worker_id = 2;
    header.sequence = 9;
    auto encoded = remote_l3::encode_frame(header, {});

    auto bad_version = encoded;
    bad_version[4] = 0xFF;
    EXPECT_THROW((void)remote_l3::decode_frame(bad_version), std::runtime_error);

    auto bad_type = encoded;
    bad_type[8] = 0xFF;
    EXPECT_THROW((void)remote_l3::decode_frame(bad_type), std::runtime_error);

    auto bad_flags = encoded;
    bad_flags[36] = static_cast<uint8_t>(remote_l3::FRAME_FLAGS_KNOWN + 1);
    EXPECT_THROW((void)remote_l3::decode_frame(bad_flags), std::runtime_error);

    auto group_flag = encoded;
    group_flag[36] = remote_l3::FRAME_FLAG_GROUP_TARGET;
    EXPECT_EQ(remote_l3::decode_frame(group_flag).header.flags, remote_l3::FRAME_FLAG_GROUP_TARGET);
}

TEST(RemoteWire, TaskPayloadRoundTripsTheArgTensorVerbatim) {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest.fill(0xAB);
    Tensor arg = remote_arg_tensor();
    // A strided view: strides are carried explicitly, so the wire does not flatten one.
    arg.buffer.nbytes = 16;
    arg.ndims = 2;
    arg.shapes[0] = 2;
    arg.shapes[1] = 2;
    arg.strides[0] = 8;
    arg.strides[1] = 1;
    validate_tensor(arg);
    payload.args.tensors.push_back(arg);
    payload.args.scalars.push_back(0xCAFE);

    auto encoded = remote_l3::encode_task_payload(payload);
    auto decoded = remote_l3::decode_task_payload(encoded.data(), encoded.size());
    ASSERT_EQ(decoded.args.tensors.size(), 1u);
    EXPECT_EQ(decoded.callable_digest[0], 0xAB);
    EXPECT_EQ(decoded.args.scalars[0], 0xCAFEu);

    const Tensor &back = decoded.args.tensors[0];
    EXPECT_TRUE(back.buffer == arg.buffer);
    EXPECT_EQ(back.byte_offset, arg.byte_offset);
    EXPECT_EQ(back.ndims, 2u);
    EXPECT_EQ(back.shapes[0], 2u);
    EXPECT_EQ(back.shapes[1], 2u);
    EXPECT_EQ(back.strides[0], 8u);
    EXPECT_EQ(back.strides[1], 1u);
    EXPECT_EQ(back.dtype, DataType::UINT8);
    // Slots past ndims never cross, so a decoded record compares clean against a fresh one.
    EXPECT_EQ(back.shapes[2], 0u);
    EXPECT_EQ(back.strides[2], 0u);
}

TEST(RemoteWire, TaskPayloadRejectsAnArgWithALocalBacking) {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest.fill(0xAB);
    Tensor arg = remote_arg_tensor();
    // A backing this endpoint could materialize locally is a second source of truth for a backing
    // the sidecar already names, so it never crosses this wire.
    arg.buffer.backend_kind = static_cast<uint8_t>(BackendKind::DEVICE_MALLOC);
    payload.args.tensors.push_back(arg);

    EXPECT_THROW((void)remote_l3::encode_task_payload(payload), std::runtime_error);
}

TEST(RemoteWire, TensorRecordCarriesWholeBackingAndByteOffset) {
    Tensor arg = remote_arg_tensor();
    arg.buffer.nbytes = 8;
    arg.byte_offset = 4;
    arg.shapes[0] = 4;
    validate_tensor(arg);
    auto encoded = remote_l3::encode_tensor(arg);
    // byte_offset follows the 39-byte descriptor head of a record whose backend body is empty:
    // identity nonce(8) + buffer_id(8) + generation(4) + address_space/access/backend(3) +
    // nbytes(8) + owner_worker_path_id(4) + body_len(4).
    constexpr size_t kByteOffsetPos = 39;
    ASSERT_GT(encoded.size(), kByteOffsetPos + sizeof(uint64_t));

    size_t offset = 0;
    const Tensor decoded = remote_l3::decode_tensor(encoded.data(), encoded.size(), offset);
    EXPECT_EQ(decoded.buffer.nbytes, 8u);
    EXPECT_EQ(decoded.byte_offset, 4u);

    // A malformed record whose offset is moved past the backing is still rejected by the common
    // Tensor validator, rather than by a remote-wire-only rule.
    encoded[kByteOffsetPos] = 8;
    offset = 0;
    EXPECT_THROW((void)remote_l3::decode_tensor(encoded.data(), encoded.size(), offset), std::runtime_error);
}

TEST(RemoteWire, TaskPayloadPreservesScopeStatsCallConfig) {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest.fill(0xAB);
    payload.config.aicpu_thread_num = 5;
    payload.config.enable_scope_stats = 1;
    const char *prefix = "/tmp/remote-scope";
    std::memcpy(payload.config.output_prefix, prefix, std::strlen(prefix));
    payload.args.tensors.push_back(remote_arg_tensor());

    auto encoded = remote_l3::encode_task_payload(payload);
    auto decoded = remote_l3::decode_task_payload(encoded.data(), encoded.size());

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
    args.tensors.push_back(remote_arg_tensor());
    args.inline_payload = {1, 2, 3, 4};

    RemoteTensorSidecar sidecar;
    sidecar.present = true;
    sidecar.desc.address_space = RemoteAddressSpace::HOST_INLINE;
    sidecar.desc.owner_worker_id = 0;
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
    args.tensors.push_back(remote_arg_tensor());

    RemoteTensorSidecar sidecar;
    sidecar.present = true;
    sidecar.desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.desc.owner_worker_id = 3;
    sidecar.desc.buffer_id = 9;
    sidecar.desc.generation = 1;
    sidecar.desc.nbytes = 4;
    sidecar.desc.inline_payload_len = 1;
    args.remote_desc.push_back(sidecar);

    EXPECT_THROW((void)remote_l3::encode_remote_task_args(args), std::runtime_error);
}

TEST(RemoteWire, NonHostInlineDescriptorRejectsMissingBufferIdentity) {
    remote_l3::RemoteTaskArgsWire args;
    args.tensors.push_back(remote_arg_tensor());

    RemoteTensorSidecar sidecar;
    sidecar.present = true;
    sidecar.desc.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    sidecar.desc.owner_worker_id = 3;
    sidecar.desc.buffer_id = 9;
    sidecar.desc.generation = 1;
    sidecar.desc.nbytes = 4;
    args.remote_desc.push_back(sidecar);

    args.remote_desc[0].desc.buffer_id = 0;
    EXPECT_THROW((void)remote_l3::encode_remote_task_args(args), std::runtime_error);

    args.remote_desc[0].desc.buffer_id = 9;
    args.remote_desc[0].desc.generation = 0;
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
    export_request.owner_worker_id = 3;
    export_request.buffer_id = 11;
    export_request.generation = 2;
    export_request.offset = 16;
    export_request.nbytes = 64;
    export_request.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ_WRITE;
    export_request.transport_profile = "sim";

    auto export_request_bytes = remote_l3::encode_export_buffer_request(export_request);
    auto decoded_export_request =
        remote_l3::decode_export_buffer_request(export_request_bytes.data(), export_request_bytes.size());
    EXPECT_EQ(decoded_export_request.owner_worker_id, 3);
    EXPECT_EQ(decoded_export_request.offset, 16u);
    EXPECT_EQ(decoded_export_request.transport_profile, "sim");

    RemoteBufferExport export_result;
    export_result.owner_worker_id = 3;
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
    import_request.importer_worker_id = 4;
    import_request.requested_access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ;
    import_request.export_desc = export_result;
    auto import_request_bytes = remote_l3::encode_import_buffer_request(import_request);
    auto decoded_import_request =
        remote_l3::decode_import_buffer_request(import_request_bytes.data(), import_request_bytes.size());
    EXPECT_EQ(decoded_import_request.importer_worker_id, 4);
    EXPECT_EQ(decoded_import_request.requested_access_flags, remote_l3::REMOTE_BUFFER_ACCESS_READ);
    EXPECT_EQ(decoded_import_request.export_desc.owner_worker_id, 3);

    RemoteBufferHandle import_result;
    import_result.worker_id = 4;
    import_result.owner_worker_id = 3;
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
    EXPECT_EQ(decoded_import_result.worker_id, 4);
    EXPECT_EQ(decoded_import_result.owner_worker_id, 3);
    EXPECT_EQ(decoded_import_result.import_id, 7u);

    remote_l3::ReleaseImportRequest release_request;
    release_request.importer_worker_id = 4;
    release_request.owner_worker_id = 3;
    release_request.buffer_id = 11;
    release_request.generation = 2;
    release_request.import_id = 7;
    auto release_bytes = remote_l3::encode_release_import_request(release_request);
    auto decoded_release = remote_l3::decode_release_import_request(release_bytes.data(), release_bytes.size());
    EXPECT_EQ(decoded_release.importer_worker_id, 4);
    EXPECT_EQ(decoded_release.import_id, 7u);
}

TEST(RemoteWire, RemoteBufferControlsRejectInvalidAccessAndReservedBytes) {
    remote_l3::ExportBufferRequest request;
    request.owner_worker_id = 3;
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
