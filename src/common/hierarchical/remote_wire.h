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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/tensor.h"
#include "types.h"

namespace remote_l3 {

static constexpr uint32_t PROTOCOL_VERSION = 1;
static constexpr uint32_t MAX_FRAME_PAYLOAD_BYTES = 16U * 1024U * 1024U;
static constexpr uint32_t MAX_STRING_BYTES = 1024U;
static constexpr uint32_t MAX_ERROR_BYTES = 4096U;
static constexpr uint32_t MAX_TENSORS = 4096U;
static constexpr uint32_t MAX_SCALARS = 4096U;
static constexpr uint32_t MAX_INLINE_PAYLOAD_BYTES = 1024U * 1024U;
static constexpr uint32_t MAX_TRANSPORT_PROFILE_BYTES = 128U;
static constexpr uint32_t MAX_TRANSPORT_DESCRIPTOR_BYTES = 4096U;
static constexpr uint32_t REMOTE_BUFFER_ACCESS_READ = 1U << 0;
static constexpr uint32_t REMOTE_BUFFER_ACCESS_WRITE = 1U << 1;
static constexpr uint32_t REMOTE_BUFFER_ACCESS_READ_WRITE = REMOTE_BUFFER_ACCESS_READ | REMOTE_BUFFER_ACCESS_WRITE;

enum class FrameType : uint32_t {
    HELLO = 1,
    TASK = 2,
    CONTROL = 3,
    CONTROL_REPLY = 4,
    COMPLETION = 5,
    HEALTH = 6,
    SHUTDOWN = 7,
};

enum class ControlName : uint32_t {
    UNREGISTER_CALLABLE = 1,
    PREPARE_REGISTER_CALLABLE = 2,
    COMMIT_REGISTER_CALLABLE = 3,
    ABORT_REGISTER_CALLABLE = 4,
    PREPARE_CALLABLE = 5,
    ALLOC_REMOTE_BUFFER = 6,
    FREE_REMOTE_BUFFER = 7,
    COPY_TO_REMOTE = 8,
    COPY_FROM_REMOTE = 9,
    EXPORT_BUFFER = 10,
    IMPORT_BUFFER = 11,
    RELEASE_IMPORT = 12,
    COMM_INIT = 13,
    ALLOC_DOMAIN = 14,
    RELEASE_DOMAIN = 15,
};

enum class ReadyState : uint32_t {
    NOT_READY = 0,
    READY = 1,
};

enum class RemoteRegistryTarget : uint32_t {
    REMOTE_TASK_DISPATCHER = 1,
    INNER_L3_WORKER = 2,
};

struct FrameHeader {
    FrameType frame_type{FrameType::HELLO};
    uint64_t session_id{0};
    int32_t endpoint_id{-1};
    uint64_t sequence{0};
    uint32_t payload_bytes{0};
    uint32_t flags{0};
};

struct DecodedFrame {
    FrameHeader header{};
    std::vector<uint8_t> payload;
};

struct HelloPayload {
    uint64_t session_id{0};
    int32_t endpoint_id{-1};
    uint32_t protocol_version{PROTOCOL_VERSION};
    std::string comm_profile;
    uint64_t feature_flags{0};
    ReadyState ready_state{ReadyState::NOT_READY};
};

struct RemoteTaskArgsWire {
    std::vector<Tensor> tensor_metadata;
    std::vector<RemoteTensorSidecar> remote_desc;
    std::vector<uint64_t> scalars;
    std::vector<uint8_t> inline_payload;
};

struct TaskPayloadWire {
    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> callable_digest{};
    CallConfig config{};
    RemoteTaskArgsWire args;
};

struct CompletionPayload {
    uint64_t sequence{0};
    int32_t error_code{0};
    std::string error_message;
};

struct ControlPayload {
    ControlName control_name{ControlName::PREPARE_CALLABLE};
    uint32_t control_version{1};
    std::vector<uint8_t> command_bytes;
};

struct ControlReplyPayload {
    uint64_t sequence{0};
    ControlName control_name{ControlName::PREPARE_CALLABLE};
    uint32_t control_version{1};
    int32_t error_code{0};
    std::string error_message;
    std::vector<uint8_t> result_bytes;
};

struct ExportBufferRequest {
    int32_t owner_endpoint_id{-1};
    uint64_t buffer_id{0};
    uint64_t generation{0};
    uint64_t offset{0};
    uint64_t nbytes{0};
    uint32_t access_flags{0};
    std::string transport_profile;
};

struct ImportBufferRequest {
    int32_t importer_endpoint_id{-1};
    uint32_t requested_access_flags{0};
    RemoteBufferExport export_desc{};
};

struct ReleaseImportRequest {
    int32_t importer_endpoint_id{-1};
    int32_t owner_endpoint_id{-1};
    uint64_t buffer_id{0};
    uint64_t generation{0};
    uint64_t import_id{0};
};

std::vector<uint8_t> encode_frame(const FrameHeader &header, const std::vector<uint8_t> &payload);
DecodedFrame decode_frame(const uint8_t *data, size_t size);
DecodedFrame decode_frame(const std::vector<uint8_t> &data);

std::vector<uint8_t> encode_hello(const HelloPayload &payload);
HelloPayload decode_hello(const uint8_t *data, size_t size);

std::vector<uint8_t> encode_call_config(const CallConfig &config);
CallConfig decode_call_config(const uint8_t *data, size_t size, size_t &offset);

std::vector<uint8_t> encode_tensor(const Tensor &tensor);
Tensor decode_tensor(const uint8_t *data, size_t size, size_t &offset, bool remote_task);

std::vector<uint8_t> encode_remote_tensor_desc(const RemoteTensorDesc &desc);
RemoteTensorDesc decode_remote_tensor_desc(const uint8_t *data, size_t size, size_t &offset);

std::vector<uint8_t> encode_remote_task_args(const RemoteTaskArgsWire &args);
RemoteTaskArgsWire decode_remote_task_args(const uint8_t *data, size_t size);

std::vector<uint8_t> encode_task_payload(const TaskPayloadWire &payload);
TaskPayloadWire decode_task_payload(const uint8_t *data, size_t size);

std::vector<uint8_t> encode_completion(const CompletionPayload &payload);
CompletionPayload decode_completion(const uint8_t *data, size_t size, uint64_t expected_sequence);

std::vector<uint8_t> encode_control(const ControlPayload &payload);
ControlPayload decode_control(const uint8_t *data, size_t size);

std::vector<uint8_t> encode_register_callable_command(
    RemoteRegistryTarget target_registry, CallableKind callable_kind,
    const std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> &digest, uint32_t payload_version,
    const std::vector<uint8_t> &payload
);
std::vector<uint8_t> encode_digest_callable_command(
    RemoteRegistryTarget target_registry, CallableKind callable_kind,
    const std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> &digest
);

std::vector<uint8_t> encode_export_buffer_request(const ExportBufferRequest &request);
ExportBufferRequest decode_export_buffer_request(const uint8_t *data, size_t size);
std::vector<uint8_t> encode_export_buffer_result(const RemoteBufferExport &result);
RemoteBufferExport decode_export_buffer_result(const uint8_t *data, size_t size);
std::vector<uint8_t> encode_import_buffer_request(const ImportBufferRequest &request);
ImportBufferRequest decode_import_buffer_request(const uint8_t *data, size_t size);
std::vector<uint8_t> encode_import_buffer_result(const RemoteBufferHandle &result);
RemoteBufferHandle decode_import_buffer_result(const uint8_t *data, size_t size);
std::vector<uint8_t> encode_release_import_request(const ReleaseImportRequest &request);
ReleaseImportRequest decode_release_import_request(const uint8_t *data, size_t size);

std::vector<uint8_t> encode_control_reply(const ControlReplyPayload &payload);
ControlReplyPayload decode_control_reply(
    const uint8_t *data, size_t size, uint64_t expected_sequence, ControlName expected_control_name,
    uint32_t expected_control_version
);

class OrderedCommandLane {
public:
    uint64_t begin_command();
    void finish_reply(uint64_t sequence);
    bool in_flight() const { return in_flight_; }

private:
    uint64_t next_sequence_{1};
    uint64_t in_flight_sequence_{0};
    bool in_flight_{false};
};

}  // namespace remote_l3
