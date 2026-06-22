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

#include "remote_wire.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace remote_l3 {
namespace {

static constexpr uint8_t MAGIC[4] = {'S', 'L', 'R', '3'};
static constexpr size_t FRAME_HEADER_BYTES = 40;

void ensure(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void ensure_available(size_t size, size_t offset, size_t n, const char *what) {
    if (offset > size || n > size - offset) {
        throw std::runtime_error(std::string("remote_wire: truncated ") + what);
    }
}

void put_u8(std::vector<uint8_t> &out, uint8_t v) { out.push_back(v); }

void put_u32(std::vector<uint8_t> &out, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffU));
}

void put_i32(std::vector<uint8_t> &out, int32_t v) { put_u32(out, static_cast<uint32_t>(v)); }

void put_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffU));
}

uint8_t get_u8(const uint8_t *data, size_t size, size_t &offset) {
    ensure_available(size, offset, 1, "uint8");
    return data[offset++];
}

uint32_t get_u32(const uint8_t *data, size_t size, size_t &offset) {
    ensure_available(size, offset, 4, "uint32");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<uint32_t>(data[offset++]) << (8 * i);
    return v;
}

int32_t get_i32(const uint8_t *data, size_t size, size_t &offset) {
    return static_cast<int32_t>(get_u32(data, size, offset));
}

uint64_t get_u64(const uint8_t *data, size_t size, size_t &offset) {
    ensure_available(size, offset, 8, "uint64");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(data[offset++]) << (8 * i);
    return v;
}

void put_bytes(std::vector<uint8_t> &out, const uint8_t *data, size_t n) {
    if (n == 0) return;
    out.insert(out.end(), data, data + n);
}

void put_string(std::vector<uint8_t> &out, const std::string &value, uint32_t max_bytes, const char *field_name) {
    ensure(value.size() <= max_bytes, std::string("remote_wire: ") + field_name + " exceeds max length");
    put_u32(out, static_cast<uint32_t>(value.size()));
    put_bytes(out, reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void put_blob(
    std::vector<uint8_t> &out, const std::vector<uint8_t> &value, uint32_t max_bytes, const char *field_name
) {
    ensure(value.size() <= max_bytes, std::string("remote_wire: ") + field_name + " exceeds max length");
    put_u32(out, static_cast<uint32_t>(value.size()));
    put_bytes(out, value.data(), value.size());
}

std::string get_string(const uint8_t *data, size_t size, size_t &offset, uint32_t max_bytes, const char *field_name) {
    uint32_t n = get_u32(data, size, offset);
    ensure(n <= max_bytes, std::string("remote_wire: ") + field_name + " exceeds max length");
    ensure_available(size, offset, n, field_name);
    std::string out(reinterpret_cast<const char *>(data + offset), n);
    offset += n;
    return out;
}

std::vector<uint8_t>
get_blob(const uint8_t *data, size_t size, size_t &offset, uint32_t max_bytes, const char *field_name) {
    uint32_t n = get_u32(data, size, offset);
    ensure(n <= max_bytes, std::string("remote_wire: ") + field_name + " exceeds max length");
    ensure_available(size, offset, n, field_name);
    std::vector<uint8_t> out(data + offset, data + offset + n);
    offset += n;
    return out;
}

void validate_access_flags(uint32_t flags, const char *field_name) {
    ensure(flags != 0, std::string("remote_wire: ") + field_name + " must be non-zero");
    ensure(
        (flags & ~REMOTE_BUFFER_ACCESS_READ_WRITE) == 0,
        std::string("remote_wire: ") + field_name + " contains unknown bits"
    );
}

bool valid_frame_type(uint32_t v) {
    switch (static_cast<FrameType>(v)) {
    case FrameType::HELLO:
    case FrameType::TASK:
    case FrameType::CONTROL:
    case FrameType::CONTROL_REPLY:
    case FrameType::COMPLETION:
    case FrameType::HEALTH:
    case FrameType::SHUTDOWN:
        return true;
    }
    return false;
}

bool valid_control_name(uint32_t v) {
    switch (static_cast<ControlName>(v)) {
    case ControlName::UNREGISTER_CALLABLE:
    case ControlName::PREPARE_REGISTER_CALLABLE:
    case ControlName::COMMIT_REGISTER_CALLABLE:
    case ControlName::ABORT_REGISTER_CALLABLE:
    case ControlName::PREPARE_CALLABLE:
    case ControlName::ALLOC_REMOTE_BUFFER:
    case ControlName::FREE_REMOTE_BUFFER:
    case ControlName::COPY_TO_REMOTE:
    case ControlName::COPY_FROM_REMOTE:
    case ControlName::EXPORT_BUFFER:
    case ControlName::IMPORT_BUFFER:
    case ControlName::RELEASE_IMPORT:
    case ControlName::COMM_INIT:
    case ControlName::ALLOC_DOMAIN:
    case ControlName::RELEASE_DOMAIN:
        return true;
    }
    return false;
}

bool valid_ready_state(uint32_t v) {
    switch (static_cast<ReadyState>(v)) {
    case ReadyState::NOT_READY:
    case ReadyState::READY:
        return true;
    }
    return false;
}

bool valid_remote_registry_target(uint32_t v) {
    switch (static_cast<RemoteRegistryTarget>(v)) {
    case RemoteRegistryTarget::REMOTE_TASK_DISPATCHER:
    case RemoteRegistryTarget::INNER_L3_WORKER:
        return true;
    }
    return false;
}

bool valid_callable_kind(uint32_t v) {
    switch (static_cast<CallableKind>(static_cast<int32_t>(v))) {
    case CallableKind::CHIP_CALLABLE:
    case CallableKind::PYTHON_SERIALIZED:
    case CallableKind::PYTHON_IMPORT:
        return true;
    }
    return false;
}

bool valid_address_space(uint32_t v) {
    switch (static_cast<RemoteAddressSpace>(v)) {
    case RemoteAddressSpace::HOST_INLINE:
    case RemoteAddressSpace::REMOTE_DEVICE:
    case RemoteAddressSpace::REMOTE_WINDOW:
    case RemoteAddressSpace::UB_LDST:
        return true;
    }
    return false;
}

bool valid_import_address_space(RemoteAddressSpace space) {
    return space == RemoteAddressSpace::REMOTE_WINDOW || space == RemoteAddressSpace::UB_LDST;
}

bool valid_dtype(uint32_t v) { return v < static_cast<uint32_t>(DataType::DATA_TYPE_NUM); }

std::string call_config_prefix(const CallConfig &config) {
    size_t n = 0;
    while (n < sizeof(config.output_prefix) && config.output_prefix[n] != '\0')
        ++n;
    return std::string(config.output_prefix, config.output_prefix + n);
}

void validate_desc_against_inline_payload(const RemoteTensorDesc &desc, size_t inline_payload_len) {
    ensure(desc.flags == 0, "remote_wire: RemoteTensorDesc flags are reserved in v1");
    ensure(
        desc.nbytes <= std::numeric_limits<uint64_t>::max() - desc.offset,
        "remote_wire: RemoteTensorDesc offset+nbytes overflows"
    );
    if (desc.address_space == RemoteAddressSpace::HOST_INLINE) {
        ensure(desc.owner_worker_id == 0, "remote_wire: HOST_INLINE owner_worker_id must be zero");
        ensure(desc.buffer_id == 0, "remote_wire: HOST_INLINE buffer_id must be zero");
        ensure(desc.remote_addr == 0, "remote_wire: HOST_INLINE remote_addr must be zero");
        ensure(desc.rkey_or_token == 0, "remote_wire: HOST_INLINE rkey_or_token must be zero");
        ensure(desc.generation == 0, "remote_wire: HOST_INLINE generation must be zero");
        ensure(desc.inline_payload_len == desc.nbytes, "remote_wire: HOST_INLINE inline length must equal nbytes");
        ensure(
            desc.inline_payload_offset <= static_cast<uint64_t>(inline_payload_len) &&
                desc.inline_payload_len <= static_cast<uint64_t>(inline_payload_len) - desc.inline_payload_offset,
            "remote_wire: HOST_INLINE payload range exceeds inline arena"
        );
    } else {
        ensure(desc.inline_payload_offset == 0, "remote_wire: non-HOST_INLINE inline offset must be zero");
        ensure(desc.inline_payload_len == 0, "remote_wire: non-HOST_INLINE inline length must be zero");
        ensure(desc.owner_worker_id >= 0, "remote_wire: remote descriptor owner worker must be non-negative");
        ensure(desc.buffer_id != 0, "remote_wire: remote descriptor buffer_id must be non-zero");
        ensure(desc.generation != 0, "remote_wire: remote descriptor generation must be non-zero");
    }
}

}  // namespace

std::vector<uint8_t> encode_frame(const FrameHeader &header, const std::vector<uint8_t> &payload) {
    ensure(payload.size() <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: frame payload exceeds maximum");
    ensure(header.flags == 0, "remote_wire: frame flags are reserved in v1");
    std::vector<uint8_t> out;
    out.reserve(FRAME_HEADER_BYTES + payload.size());
    put_bytes(out, MAGIC, sizeof(MAGIC));
    put_u32(out, PROTOCOL_VERSION);
    put_u32(out, static_cast<uint32_t>(header.frame_type));
    put_u64(out, header.session_id);
    put_i32(out, header.worker_id);
    put_u64(out, header.sequence);
    put_u32(out, static_cast<uint32_t>(payload.size()));
    put_u32(out, header.flags);
    put_bytes(out, payload.data(), payload.size());
    return out;
}

DecodedFrame decode_frame(const uint8_t *data, size_t size) {
    ensure(size >= FRAME_HEADER_BYTES, "remote_wire: truncated frame header");
    ensure(std::memcmp(data, MAGIC, sizeof(MAGIC)) == 0, "remote_wire: bad frame magic");
    size_t offset = sizeof(MAGIC);
    uint32_t version = get_u32(data, size, offset);
    ensure(version == PROTOCOL_VERSION, "remote_wire: unsupported frame version");
    uint32_t raw_type = get_u32(data, size, offset);
    ensure(valid_frame_type(raw_type), "remote_wire: unknown frame type");
    DecodedFrame frame;
    frame.header.frame_type = static_cast<FrameType>(raw_type);
    frame.header.session_id = get_u64(data, size, offset);
    frame.header.worker_id = get_i32(data, size, offset);
    frame.header.sequence = get_u64(data, size, offset);
    frame.header.payload_bytes = get_u32(data, size, offset);
    frame.header.flags = get_u32(data, size, offset);
    ensure(frame.header.flags == 0, "remote_wire: frame flags are reserved in v1");
    ensure(frame.header.payload_bytes <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: frame payload exceeds maximum");
    ensure(size - offset == frame.header.payload_bytes, "remote_wire: frame payload length mismatch");
    frame.payload.assign(data + offset, data + size);
    return frame;
}

DecodedFrame decode_frame(const std::vector<uint8_t> &data) { return decode_frame(data.data(), data.size()); }

std::vector<uint8_t> encode_hello(const HelloPayload &payload) {
    ensure(payload.session_id != 0, "remote_wire: HELLO session_id must be non-zero");
    ensure(payload.worker_id >= 0, "remote_wire: HELLO worker_id must be non-negative");
    ensure(payload.protocol_version == PROTOCOL_VERSION, "remote_wire: HELLO protocol version mismatch");
    std::vector<uint8_t> out;
    put_u64(out, payload.session_id);
    put_i32(out, payload.worker_id);
    put_u32(out, payload.protocol_version);
    put_string(out, payload.comm_profile, MAX_STRING_BYTES, "HELLO.comm_profile");
    put_u64(out, payload.feature_flags);
    put_u32(out, static_cast<uint32_t>(payload.ready_state));
    return out;
}

HelloPayload decode_hello(const uint8_t *data, size_t size) {
    size_t offset = 0;
    HelloPayload payload;
    payload.session_id = get_u64(data, size, offset);
    ensure(payload.session_id != 0, "remote_wire: HELLO session_id must be non-zero");
    payload.worker_id = get_i32(data, size, offset);
    ensure(payload.worker_id >= 0, "remote_wire: HELLO worker_id must be non-negative");
    payload.protocol_version = get_u32(data, size, offset);
    ensure(payload.protocol_version == PROTOCOL_VERSION, "remote_wire: HELLO protocol version mismatch");
    payload.comm_profile = get_string(data, size, offset, MAX_STRING_BYTES, "HELLO.comm_profile");
    payload.feature_flags = get_u64(data, size, offset);
    uint32_t ready = get_u32(data, size, offset);
    ensure(valid_ready_state(ready), "remote_wire: unknown HELLO ready_state");
    payload.ready_state = static_cast<ReadyState>(ready);
    ensure(offset == size, "remote_wire: trailing bytes after HELLO");
    return payload;
}

std::vector<uint8_t> encode_call_config(const CallConfig &config) {
    std::vector<uint8_t> out;
    put_i32(out, config.block_dim);
    put_i32(out, config.aicpu_thread_num);
    put_i32(out, config.enable_l2_swimlane);
    put_i32(out, config.enable_dump_tensor);
    put_i32(out, config.enable_pmu);
    put_i32(out, config.enable_dep_gen);
    put_i32(out, config.enable_scope_stats);
    put_string(out, call_config_prefix(config), MAX_STRING_BYTES, "CallConfig.output_prefix");
    return out;
}

CallConfig decode_call_config(const uint8_t *data, size_t size, size_t &offset) {
    CallConfig config{};
    config.block_dim = get_i32(data, size, offset);
    config.aicpu_thread_num = get_i32(data, size, offset);
    config.enable_l2_swimlane = get_i32(data, size, offset);
    config.enable_dump_tensor = get_i32(data, size, offset);
    config.enable_pmu = get_i32(data, size, offset);
    config.enable_dep_gen = get_i32(data, size, offset);
    config.enable_scope_stats = get_i32(data, size, offset);
    std::string prefix = get_string(data, size, offset, MAX_STRING_BYTES, "CallConfig.output_prefix");
    ensure(prefix.size() < sizeof(config.output_prefix), "remote_wire: CallConfig.output_prefix is too long");
    std::memset(config.output_prefix, 0, sizeof(config.output_prefix));
    std::memcpy(config.output_prefix, prefix.data(), prefix.size());
    config.validate();
    return config;
}

// Wire format carries only the contiguous-defining fields (addr, shapes, ndims,
// dtype, child_memory +
// 7 reserved bytes). The unified Tensor's derived state (strides / start_offset
// / is_contiguous) is recomputed as row-major on decode. The wire is therefore
// contiguous-only: a strided Tensor would be silently flattened. Guard against
// that here so the loss is loud, not silent — strided views only round-trip
// over the local fork/shm mailbox blob, never this remote wire.
std::vector<uint8_t> encode_tensor(const Tensor &tensor) {
    ensure(
        tensor.is_contiguous && tensor.start_offset == 0,
        "remote_wire: only contiguous, zero-offset tensors are supported on the wire"
    );
    std::vector<uint8_t> out;
    put_u64(out, tensor.buffer.addr);
    for (uint32_t shape : tensor.shapes)
        put_u32(out, shape);
    put_u32(out, tensor.ndims);
    put_u32(out, static_cast<uint32_t>(tensor.dtype));
    put_u8(out, tensor.child_memory);
    for (int i = 0; i < 7; ++i)
        put_u8(out, 0);
    return out;
}

Tensor decode_tensor(const uint8_t *data, size_t size, size_t &offset, bool remote_task) {
    uint64_t addr = get_u64(data, size, offset);
    if (remote_task) ensure(addr == 0, "remote_wire: remote TASK tensor data must be zero");
    uint32_t shapes[MAX_TENSOR_DIMS];
    for (uint32_t &shape : shapes)
        shape = get_u32(data, size, offset);
    uint32_t ndims = get_u32(data, size, offset);
    ensure(ndims > 0 && ndims <= MAX_TENSOR_DIMS, "remote_wire: tensor ndims out of range");
    uint32_t dtype = get_u32(data, size, offset);
    ensure(valid_dtype(dtype), "remote_wire: unknown tensor dtype");
    uint8_t child_memory = get_u8(data, size, offset);
    ensure(child_memory == 0 || child_memory == 1, "remote_wire: tensor child_memory must be 0 or 1");
    for (int i = 0; i < 7; ++i) {
        ensure(get_u8(data, size, offset) == 0, "remote_wire: Tensor reserved bytes must be zero");
    }
    // Reconstruct a contiguous external Tensor (owner=invalid, row-major strides).
    return make_tensor_external(
        reinterpret_cast<void *>(addr), shapes, ndims, static_cast<DataType>(dtype), /*manual_dep=*/false,
        /*version=*/0, child_memory
    );
}

std::vector<uint8_t> encode_remote_tensor_desc(const RemoteTensorDesc &desc) {
    std::vector<uint8_t> out;
    put_u32(out, static_cast<uint32_t>(desc.address_space));
    put_i32(out, desc.owner_worker_id);
    put_u64(out, desc.buffer_id);
    put_u64(out, desc.offset);
    put_u64(out, desc.nbytes);
    put_u64(out, desc.remote_addr);
    put_u64(out, desc.rkey_or_token);
    put_u64(out, desc.generation);
    put_u64(out, desc.inline_payload_offset);
    put_u64(out, desc.inline_payload_len);
    put_u64(out, desc.flags);
    return out;
}

RemoteTensorDesc decode_remote_tensor_desc(const uint8_t *data, size_t size, size_t &offset) {
    uint32_t raw_space = get_u32(data, size, offset);
    ensure(valid_address_space(raw_space), "remote_wire: unknown RemoteTensorDesc address_space");
    RemoteTensorDesc desc{};
    desc.address_space = static_cast<RemoteAddressSpace>(raw_space);
    desc.owner_worker_id = get_i32(data, size, offset);
    desc.buffer_id = get_u64(data, size, offset);
    desc.offset = get_u64(data, size, offset);
    desc.nbytes = get_u64(data, size, offset);
    desc.remote_addr = get_u64(data, size, offset);
    desc.rkey_or_token = get_u64(data, size, offset);
    desc.generation = get_u64(data, size, offset);
    desc.inline_payload_offset = get_u64(data, size, offset);
    desc.inline_payload_len = get_u64(data, size, offset);
    desc.flags = get_u64(data, size, offset);
    return desc;
}

std::vector<uint8_t> encode_remote_task_args(const RemoteTaskArgsWire &args) {
    ensure(args.tensor_metadata.size() <= MAX_TENSORS, "remote_wire: tensor count exceeds maximum");
    ensure(args.scalars.size() <= MAX_SCALARS, "remote_wire: scalar count exceeds maximum");
    ensure(args.inline_payload.size() <= MAX_INLINE_PAYLOAD_BYTES, "remote_wire: inline payload exceeds maximum");
    ensure(
        args.remote_desc.empty() || args.remote_desc.size() == args.tensor_metadata.size(),
        "remote_wire: remote descriptor count must match tensor count"
    );
    std::vector<uint8_t> out;
    put_u32(out, static_cast<uint32_t>(args.tensor_metadata.size()));
    put_u32(out, static_cast<uint32_t>(args.scalars.size()));
    for (const auto &tensor : args.tensor_metadata) {
        ensure(tensor.buffer.addr == 0, "remote_wire: remote TASK tensor data must be zero");
        auto encoded = encode_tensor(tensor);
        put_bytes(out, encoded.data(), encoded.size());
    }
    for (size_t i = 0; i < args.tensor_metadata.size(); ++i) {
        RemoteTensorSidecar sidecar{};
        if (!args.remote_desc.empty()) sidecar = args.remote_desc[i];
        put_u8(out, sidecar.present ? 1 : 0);
        if (sidecar.present) {
            validate_desc_against_inline_payload(sidecar.desc, args.inline_payload.size());
            auto encoded = encode_remote_tensor_desc(sidecar.desc);
            put_bytes(out, encoded.data(), encoded.size());
        }
    }
    for (uint64_t scalar : args.scalars)
        put_u64(out, scalar);
    put_u32(out, static_cast<uint32_t>(args.inline_payload.size()));
    put_bytes(out, args.inline_payload.data(), args.inline_payload.size());
    return out;
}

RemoteTaskArgsWire decode_remote_task_args(const uint8_t *data, size_t size) {
    size_t offset = 0;
    uint32_t tensor_count = get_u32(data, size, offset);
    uint32_t scalar_count = get_u32(data, size, offset);
    ensure(tensor_count <= MAX_TENSORS, "remote_wire: tensor count exceeds maximum");
    ensure(scalar_count <= MAX_SCALARS, "remote_wire: scalar count exceeds maximum");

    RemoteTaskArgsWire args;
    args.tensor_metadata.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i)
        args.tensor_metadata.push_back(decode_tensor(data, size, offset, true));

    args.remote_desc.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint8_t present = get_u8(data, size, offset);
        ensure(present == 0 || present == 1, "remote_wire: remote descriptor presence must be 0 or 1");
        RemoteTensorSidecar sidecar{};
        sidecar.present = present != 0;
        if (sidecar.present) sidecar.desc = decode_remote_tensor_desc(data, size, offset);
        args.remote_desc.push_back(sidecar);
    }

    args.scalars.reserve(scalar_count);
    for (uint32_t i = 0; i < scalar_count; ++i)
        args.scalars.push_back(get_u64(data, size, offset));

    uint32_t inline_len = get_u32(data, size, offset);
    ensure(inline_len <= MAX_INLINE_PAYLOAD_BYTES, "remote_wire: inline payload exceeds maximum");
    ensure_available(size, offset, inline_len, "inline payload");
    args.inline_payload.assign(data + offset, data + offset + inline_len);
    offset += inline_len;
    ensure(offset == size, "remote_wire: trailing bytes after RemoteTaskArgs");

    for (const auto &sidecar : args.remote_desc) {
        if (sidecar.present) validate_desc_against_inline_payload(sidecar.desc, args.inline_payload.size());
    }
    return args;
}

std::vector<uint8_t> encode_task_payload(const TaskPayloadWire &payload) {
    std::vector<uint8_t> out;
    put_bytes(out, payload.callable_digest.data(), payload.callable_digest.size());
    auto config = encode_call_config(payload.config);
    put_bytes(out, config.data(), config.size());
    auto args = encode_remote_task_args(payload.args);
    put_bytes(out, args.data(), args.size());
    return out;
}

TaskPayloadWire decode_task_payload(const uint8_t *data, size_t size) {
    size_t offset = 0;
    ensure_available(size, offset, CALLABLE_HASH_DIGEST_SIZE, "callable digest");
    TaskPayloadWire payload;
    std::memcpy(payload.callable_digest.data(), data + offset, CALLABLE_HASH_DIGEST_SIZE);
    offset += CALLABLE_HASH_DIGEST_SIZE;
    payload.config = decode_call_config(data, size, offset);
    payload.args = decode_remote_task_args(data + offset, size - offset);
    return payload;
}

std::vector<uint8_t> encode_completion(const CompletionPayload &payload) {
    ensure(payload.error_message.size() <= MAX_ERROR_BYTES, "remote_wire: completion error message too long");
    std::vector<uint8_t> out;
    put_u64(out, payload.sequence);
    put_i32(out, payload.error_code);
    put_string(out, payload.error_message, MAX_ERROR_BYTES, "completion.error_message");
    return out;
}

CompletionPayload decode_completion(const uint8_t *data, size_t size, uint64_t expected_sequence) {
    size_t offset = 0;
    CompletionPayload payload;
    payload.sequence = get_u64(data, size, offset);
    ensure(payload.sequence == expected_sequence, "remote_wire: completion sequence mismatch");
    payload.error_code = get_i32(data, size, offset);
    payload.error_message = get_string(data, size, offset, MAX_ERROR_BYTES, "completion.error_message");
    ensure(offset == size, "remote_wire: trailing bytes after completion");
    return payload;
}

std::vector<uint8_t> encode_control(const ControlPayload &payload) {
    ensure(valid_control_name(static_cast<uint32_t>(payload.control_name)), "remote_wire: unknown control name");
    ensure(payload.control_version != 0, "remote_wire: control version must be non-zero");
    ensure(payload.command_bytes.size() <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: control payload too large");
    std::vector<uint8_t> out;
    put_u32(out, static_cast<uint32_t>(payload.control_name));
    put_u32(out, payload.control_version);
    put_u32(out, static_cast<uint32_t>(payload.command_bytes.size()));
    put_bytes(out, payload.command_bytes.data(), payload.command_bytes.size());
    return out;
}

ControlPayload decode_control(const uint8_t *data, size_t size) {
    size_t offset = 0;
    uint32_t raw_control = get_u32(data, size, offset);
    ensure(valid_control_name(raw_control), "remote_wire: unknown control name");
    ControlPayload payload;
    payload.control_name = static_cast<ControlName>(raw_control);
    payload.control_version = get_u32(data, size, offset);
    ensure(payload.control_version != 0, "remote_wire: control version must be non-zero");
    uint32_t command_len = get_u32(data, size, offset);
    ensure(command_len <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: control payload too large");
    ensure_available(size, offset, command_len, "control payload");
    payload.command_bytes.assign(data + offset, data + offset + command_len);
    offset += command_len;
    ensure(offset == size, "remote_wire: trailing bytes after control");
    return payload;
}

std::vector<uint8_t> encode_register_callable_command(
    RemoteRegistryTarget target_registry, CallableKind callable_kind,
    const std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> &digest, uint32_t payload_version,
    const std::vector<uint8_t> &payload
) {
    ensure(
        valid_remote_registry_target(static_cast<uint32_t>(target_registry)), "remote_wire: unknown registry target"
    );
    ensure(valid_callable_kind(static_cast<uint32_t>(callable_kind)), "remote_wire: unknown callable kind");
    ensure(payload_version != 0, "remote_wire: callable payload version must be non-zero");
    ensure(payload.size() <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: callable payload too large");
    std::vector<uint8_t> out;
    put_u32(out, static_cast<uint32_t>(target_registry));
    put_u32(out, static_cast<uint32_t>(callable_kind));
    put_bytes(out, digest.data(), digest.size());
    put_u32(out, payload_version);
    put_u32(out, static_cast<uint32_t>(payload.size()));
    put_bytes(out, payload.data(), payload.size());
    return out;
}

std::vector<uint8_t> encode_digest_callable_command(
    RemoteRegistryTarget target_registry, CallableKind callable_kind,
    const std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> &digest
) {
    ensure(
        valid_remote_registry_target(static_cast<uint32_t>(target_registry)), "remote_wire: unknown registry target"
    );
    ensure(valid_callable_kind(static_cast<uint32_t>(callable_kind)), "remote_wire: unknown callable kind");
    std::vector<uint8_t> out;
    put_u32(out, static_cast<uint32_t>(target_registry));
    put_u32(out, static_cast<uint32_t>(callable_kind));
    put_bytes(out, digest.data(), digest.size());
    return out;
}

std::vector<uint8_t> encode_export_buffer_request(const ExportBufferRequest &request) {
    ensure(request.owner_worker_id >= 0, "remote_wire: EXPORT_BUFFER owner worker must be non-negative");
    ensure(request.buffer_id != 0, "remote_wire: EXPORT_BUFFER buffer_id must be non-zero");
    ensure(request.generation != 0, "remote_wire: EXPORT_BUFFER generation must be non-zero");
    ensure(request.nbytes != 0, "remote_wire: EXPORT_BUFFER nbytes must be non-zero");
    ensure(
        request.nbytes <= std::numeric_limits<uint64_t>::max() - request.offset,
        "remote_wire: EXPORT_BUFFER offset+nbytes overflows"
    );
    validate_access_flags(request.access_flags, "EXPORT_BUFFER access_flags");
    std::vector<uint8_t> out;
    put_i32(out, request.owner_worker_id);
    put_u64(out, request.buffer_id);
    put_u64(out, request.generation);
    put_u64(out, request.offset);
    put_u64(out, request.nbytes);
    put_u32(out, request.access_flags);
    put_string(out, request.transport_profile, MAX_TRANSPORT_PROFILE_BYTES, "EXPORT_BUFFER.transport_profile");
    put_u32(out, 0);
    return out;
}

ExportBufferRequest decode_export_buffer_request(const uint8_t *data, size_t size) {
    size_t offset = 0;
    ExportBufferRequest request;
    request.owner_worker_id = get_i32(data, size, offset);
    request.buffer_id = get_u64(data, size, offset);
    request.generation = get_u64(data, size, offset);
    request.offset = get_u64(data, size, offset);
    request.nbytes = get_u64(data, size, offset);
    request.access_flags = get_u32(data, size, offset);
    request.transport_profile =
        get_string(data, size, offset, MAX_TRANSPORT_PROFILE_BYTES, "EXPORT_BUFFER.transport_profile");
    ensure(get_u32(data, size, offset) == 0, "remote_wire: EXPORT_BUFFER reserved field must be zero");
    ensure(offset == size, "remote_wire: trailing bytes after EXPORT_BUFFER");
    ensure(request.owner_worker_id >= 0, "remote_wire: EXPORT_BUFFER owner worker must be non-negative");
    ensure(request.buffer_id != 0, "remote_wire: EXPORT_BUFFER buffer_id must be non-zero");
    ensure(request.generation != 0, "remote_wire: EXPORT_BUFFER generation must be non-zero");
    ensure(request.nbytes != 0, "remote_wire: EXPORT_BUFFER nbytes must be non-zero");
    ensure(
        request.nbytes <= std::numeric_limits<uint64_t>::max() - request.offset,
        "remote_wire: EXPORT_BUFFER offset+nbytes overflows"
    );
    validate_access_flags(request.access_flags, "EXPORT_BUFFER access_flags");
    return request;
}

std::vector<uint8_t> encode_export_buffer_result(const RemoteBufferExport &result) {
    ensure(result.owner_worker_id >= 0, "remote_wire: export result owner worker must be non-negative");
    ensure(result.buffer_id != 0, "remote_wire: export result buffer_id must be non-zero");
    ensure(result.generation != 0, "remote_wire: export result generation must be non-zero");
    ensure(result.nbytes != 0, "remote_wire: export result nbytes must be non-zero");
    ensure(result.export_id != 0, "remote_wire: export result export_id must be non-zero");
    ensure(valid_import_address_space(result.address_space), "remote_wire: export result address_space is invalid");
    validate_access_flags(result.access_flags, "export result access_flags");
    std::vector<uint8_t> out;
    put_i32(out, result.owner_worker_id);
    put_u64(out, result.buffer_id);
    put_u64(out, result.generation);
    put_u32(out, static_cast<uint32_t>(result.address_space));
    put_u64(out, result.offset);
    put_u64(out, result.nbytes);
    put_u64(out, result.export_id);
    put_u64(out, result.remote_addr);
    put_u64(out, result.rkey_or_token);
    put_u64(out, result.ub_ldst_va);
    put_u32(out, result.access_flags);
    put_string(out, result.transport_profile, MAX_TRANSPORT_PROFILE_BYTES, "export result transport_profile");
    put_blob(out, result.transport_descriptor, MAX_TRANSPORT_DESCRIPTOR_BYTES, "export result transport_descriptor");
    put_u32(out, 0);
    return out;
}

RemoteBufferExport decode_export_buffer_result(const uint8_t *data, size_t size) {
    size_t offset = 0;
    RemoteBufferExport result;
    result.owner_worker_id = get_i32(data, size, offset);
    result.buffer_id = get_u64(data, size, offset);
    result.generation = get_u64(data, size, offset);
    uint32_t raw_space = get_u32(data, size, offset);
    ensure(valid_address_space(raw_space), "remote_wire: export result unknown address_space");
    result.address_space = static_cast<RemoteAddressSpace>(raw_space);
    result.offset = get_u64(data, size, offset);
    result.nbytes = get_u64(data, size, offset);
    result.export_id = get_u64(data, size, offset);
    result.remote_addr = get_u64(data, size, offset);
    result.rkey_or_token = get_u64(data, size, offset);
    result.ub_ldst_va = get_u64(data, size, offset);
    result.access_flags = get_u32(data, size, offset);
    result.transport_profile =
        get_string(data, size, offset, MAX_TRANSPORT_PROFILE_BYTES, "export result transport_profile");
    result.transport_descriptor =
        get_blob(data, size, offset, MAX_TRANSPORT_DESCRIPTOR_BYTES, "export result transport_descriptor");
    ensure(get_u32(data, size, offset) == 0, "remote_wire: export result reserved field must be zero");
    ensure(offset == size, "remote_wire: trailing bytes after export result");
    ensure(result.owner_worker_id >= 0, "remote_wire: export result owner worker must be non-negative");
    ensure(result.buffer_id != 0, "remote_wire: export result buffer_id must be non-zero");
    ensure(result.generation != 0, "remote_wire: export result generation must be non-zero");
    ensure(result.nbytes != 0, "remote_wire: export result nbytes must be non-zero");
    ensure(result.export_id != 0, "remote_wire: export result export_id must be non-zero");
    ensure(valid_import_address_space(result.address_space), "remote_wire: export result address_space is invalid");
    validate_access_flags(result.access_flags, "export result access_flags");
    return result;
}

std::vector<uint8_t> encode_import_buffer_request(const ImportBufferRequest &request) {
    ensure(request.importer_worker_id >= 0, "remote_wire: IMPORT_BUFFER importer worker must be non-negative");
    validate_access_flags(request.requested_access_flags, "IMPORT_BUFFER requested_access_flags");
    ensure(
        (request.requested_access_flags & ~request.export_desc.access_flags) == 0,
        "remote_wire: IMPORT_BUFFER requested access is not a subset of export access"
    );
    std::vector<uint8_t> out;
    put_i32(out, request.importer_worker_id);
    put_u32(out, request.requested_access_flags);
    auto encoded_export = encode_export_buffer_result(request.export_desc);
    put_bytes(out, encoded_export.data(), encoded_export.size());
    put_u32(out, 0);
    return out;
}

ImportBufferRequest decode_import_buffer_request(const uint8_t *data, size_t size) {
    size_t offset = 0;
    ImportBufferRequest request;
    request.importer_worker_id = get_i32(data, size, offset);
    request.requested_access_flags = get_u32(data, size, offset);
    ensure(size >= offset + 4, "remote_wire: IMPORT_BUFFER payload is truncated");
    request.export_desc = decode_export_buffer_result(data + offset, size - offset - 4);
    offset = size - 4;
    ensure(get_u32(data, size, offset) == 0, "remote_wire: IMPORT_BUFFER reserved field must be zero");
    ensure(offset == size, "remote_wire: trailing bytes after IMPORT_BUFFER");
    ensure(request.importer_worker_id >= 0, "remote_wire: IMPORT_BUFFER importer worker must be non-negative");
    validate_access_flags(request.requested_access_flags, "IMPORT_BUFFER requested_access_flags");
    ensure(
        (request.requested_access_flags & ~request.export_desc.access_flags) == 0,
        "remote_wire: IMPORT_BUFFER requested access is not a subset of export access"
    );
    return request;
}

std::vector<uint8_t> encode_import_buffer_result(const RemoteBufferHandle &result) {
    ensure(result.worker_id >= 0, "remote_wire: import result importer worker must be non-negative");
    ensure(result.owner_worker_id >= 0, "remote_wire: import result owner worker must be non-negative");
    ensure(result.buffer_id != 0, "remote_wire: import result buffer_id must be non-zero");
    ensure(result.generation != 0, "remote_wire: import result generation must be non-zero");
    ensure(result.import_id != 0, "remote_wire: import result import_id must be non-zero");
    ensure(valid_import_address_space(result.address_space), "remote_wire: import result address_space is invalid");
    validate_access_flags(result.access_flags, "import result access_flags");
    std::vector<uint8_t> out;
    put_i32(out, result.worker_id);
    put_i32(out, result.owner_worker_id);
    put_u64(out, result.buffer_id);
    put_u64(out, result.generation);
    put_u64(out, result.import_id);
    put_u32(out, static_cast<uint32_t>(result.address_space));
    put_u64(out, result.offset);
    put_u64(out, result.nbytes);
    put_u64(out, result.remote_addr);
    put_u64(out, result.rkey_or_token);
    put_u64(out, result.ub_ldst_va);
    put_u32(out, result.access_flags);
    put_string(out, "", MAX_TRANSPORT_PROFILE_BYTES, "import result transport_profile");
    std::vector<uint8_t> empty_descriptor;
    put_blob(out, empty_descriptor, MAX_TRANSPORT_DESCRIPTOR_BYTES, "import result import_descriptor");
    put_u32(out, 0);
    return out;
}

RemoteBufferHandle decode_import_buffer_result(const uint8_t *data, size_t size) {
    size_t offset = 0;
    RemoteBufferHandle result;
    result.worker_id = get_i32(data, size, offset);
    result.owner_worker_id = get_i32(data, size, offset);
    result.buffer_id = get_u64(data, size, offset);
    result.generation = get_u64(data, size, offset);
    result.import_id = get_u64(data, size, offset);
    uint32_t raw_space = get_u32(data, size, offset);
    ensure(valid_address_space(raw_space), "remote_wire: import result unknown address_space");
    result.address_space = static_cast<RemoteAddressSpace>(raw_space);
    result.offset = get_u64(data, size, offset);
    result.nbytes = get_u64(data, size, offset);
    result.remote_addr = get_u64(data, size, offset);
    result.rkey_or_token = get_u64(data, size, offset);
    result.ub_ldst_va = get_u64(data, size, offset);
    result.access_flags = get_u32(data, size, offset);
    (void)get_string(data, size, offset, MAX_TRANSPORT_PROFILE_BYTES, "import result transport_profile");
    (void)get_blob(data, size, offset, MAX_TRANSPORT_DESCRIPTOR_BYTES, "import result import_descriptor");
    ensure(get_u32(data, size, offset) == 0, "remote_wire: import result reserved field must be zero");
    ensure(offset == size, "remote_wire: trailing bytes after import result");
    ensure(result.worker_id >= 0, "remote_wire: import result importer worker must be non-negative");
    ensure(result.owner_worker_id >= 0, "remote_wire: import result owner worker must be non-negative");
    ensure(result.buffer_id != 0, "remote_wire: import result buffer_id must be non-zero");
    ensure(result.generation != 0, "remote_wire: import result generation must be non-zero");
    ensure(result.import_id != 0, "remote_wire: import result import_id must be non-zero");
    ensure(valid_import_address_space(result.address_space), "remote_wire: import result address_space is invalid");
    validate_access_flags(result.access_flags, "import result access_flags");
    return result;
}

std::vector<uint8_t> encode_release_import_request(const ReleaseImportRequest &request) {
    ensure(request.importer_worker_id >= 0, "remote_wire: RELEASE_IMPORT importer worker must be non-negative");
    ensure(request.owner_worker_id >= 0, "remote_wire: RELEASE_IMPORT owner worker must be non-negative");
    ensure(request.buffer_id != 0, "remote_wire: RELEASE_IMPORT buffer_id must be non-zero");
    ensure(request.generation != 0, "remote_wire: RELEASE_IMPORT generation must be non-zero");
    ensure(request.import_id != 0, "remote_wire: RELEASE_IMPORT import_id must be non-zero");
    std::vector<uint8_t> out;
    put_i32(out, request.importer_worker_id);
    put_i32(out, request.owner_worker_id);
    put_u64(out, request.buffer_id);
    put_u64(out, request.generation);
    put_u64(out, request.import_id);
    put_u32(out, 0);
    return out;
}

ReleaseImportRequest decode_release_import_request(const uint8_t *data, size_t size) {
    size_t offset = 0;
    ReleaseImportRequest request;
    request.importer_worker_id = get_i32(data, size, offset);
    request.owner_worker_id = get_i32(data, size, offset);
    request.buffer_id = get_u64(data, size, offset);
    request.generation = get_u64(data, size, offset);
    request.import_id = get_u64(data, size, offset);
    ensure(get_u32(data, size, offset) == 0, "remote_wire: RELEASE_IMPORT reserved field must be zero");
    ensure(offset == size, "remote_wire: trailing bytes after RELEASE_IMPORT");
    ensure(request.importer_worker_id >= 0, "remote_wire: RELEASE_IMPORT importer worker must be non-negative");
    ensure(request.owner_worker_id >= 0, "remote_wire: RELEASE_IMPORT owner worker must be non-negative");
    ensure(request.buffer_id != 0, "remote_wire: RELEASE_IMPORT buffer_id must be non-zero");
    ensure(request.generation != 0, "remote_wire: RELEASE_IMPORT generation must be non-zero");
    ensure(request.import_id != 0, "remote_wire: RELEASE_IMPORT import_id must be non-zero");
    return request;
}

std::vector<uint8_t> encode_control_reply(const ControlReplyPayload &payload) {
    ensure(payload.error_message.size() <= MAX_ERROR_BYTES, "remote_wire: control reply error message too long");
    ensure(payload.result_bytes.size() <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: control reply result too large");
    std::vector<uint8_t> out;
    put_u64(out, payload.sequence);
    put_u32(out, static_cast<uint32_t>(payload.control_name));
    put_u32(out, payload.control_version);
    put_i32(out, payload.error_code);
    put_string(out, payload.error_message, MAX_ERROR_BYTES, "control_reply.error_message");
    put_u32(out, static_cast<uint32_t>(payload.result_bytes.size()));
    put_bytes(out, payload.result_bytes.data(), payload.result_bytes.size());
    return out;
}

ControlReplyPayload decode_control_reply(
    const uint8_t *data, size_t size, uint64_t expected_sequence, ControlName expected_control_name,
    uint32_t expected_control_version
) {
    size_t offset = 0;
    ControlReplyPayload payload;
    payload.sequence = get_u64(data, size, offset);
    ensure(payload.sequence == expected_sequence, "remote_wire: control reply sequence mismatch");
    uint32_t raw_control = get_u32(data, size, offset);
    ensure(valid_control_name(raw_control), "remote_wire: unknown control reply name");
    payload.control_name = static_cast<ControlName>(raw_control);
    ensure(payload.control_name == expected_control_name, "remote_wire: control reply name mismatch");
    payload.control_version = get_u32(data, size, offset);
    ensure(payload.control_version == expected_control_version, "remote_wire: control reply version mismatch");
    payload.error_code = get_i32(data, size, offset);
    payload.error_message = get_string(data, size, offset, MAX_ERROR_BYTES, "control_reply.error_message");
    uint32_t result_len = get_u32(data, size, offset);
    ensure(result_len <= MAX_FRAME_PAYLOAD_BYTES, "remote_wire: control reply result too large");
    ensure_available(size, offset, result_len, "control reply result");
    payload.result_bytes.assign(data + offset, data + offset + result_len);
    offset += result_len;
    ensure(offset == size, "remote_wire: trailing bytes after control reply");
    return payload;
}

uint64_t OrderedCommandLane::begin_command() {
    ensure(!in_flight_, "remote_wire: ordered command lane already has an in-flight command");
    in_flight_ = true;
    in_flight_sequence_ = next_sequence_++;
    return in_flight_sequence_;
}

void OrderedCommandLane::finish_reply(uint64_t sequence) {
    ensure(in_flight_, "remote_wire: ordered command lane has no in-flight command");
    ensure(sequence == in_flight_sequence_, "remote_wire: ordered command lane reply sequence mismatch");
    in_flight_ = false;
    in_flight_sequence_ = 0;
}

}  // namespace remote_l3
