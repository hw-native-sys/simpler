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

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "remote_endpoint.h"
#include "ring.h"

namespace {

class FakeRemoteTransport : public RemoteL3Transport {
public:
    int32_t next_error_code{0};
    std::string next_error_message;
    std::vector<uint8_t> last_frame;
    remote_l3::ControlName last_control_name{remote_l3::ControlName::PREPARE_CALLABLE};
    remote_l3::RemoteRegistryTarget last_target_registry{remote_l3::RemoteRegistryTarget::REMOTE_TASK_DISPATCHER};
    CallableKind last_callable_kind{CallableKind::PYTHON_IMPORT};

    void submit_frame(const std::vector<uint8_t> &frame) override { last_frame = frame; }

    std::vector<uint8_t> wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) override {
        auto submitted = remote_l3::decode_frame(last_frame);
        EXPECT_EQ(submitted.header.sequence, sequence);
        if (submitted.header.frame_type == remote_l3::FrameType::CONTROL) {
            EXPECT_EQ(frame_type, remote_l3::FrameType::CONTROL_REPLY);
            auto control = remote_l3::decode_control(submitted.payload.data(), submitted.payload.size());
            last_control_name = control.control_name;
            if (control.control_name == remote_l3::ControlName::PREPARE_REGISTER_CALLABLE) {
                EXPECT_GE(control.command_bytes.size(), 8u);
                uint32_t raw_target = static_cast<uint32_t>(control.command_bytes[0]) |
                                      (static_cast<uint32_t>(control.command_bytes[1]) << 8) |
                                      (static_cast<uint32_t>(control.command_bytes[2]) << 16) |
                                      (static_cast<uint32_t>(control.command_bytes[3]) << 24);
                uint32_t raw_kind = static_cast<uint32_t>(control.command_bytes[4]) |
                                    (static_cast<uint32_t>(control.command_bytes[5]) << 8) |
                                    (static_cast<uint32_t>(control.command_bytes[6]) << 16) |
                                    (static_cast<uint32_t>(control.command_bytes[7]) << 24);
                last_target_registry = static_cast<remote_l3::RemoteRegistryTarget>(raw_target);
                last_callable_kind = static_cast<CallableKind>(static_cast<int32_t>(raw_kind));
            }
            remote_l3::ControlReplyPayload payload;
            payload.sequence = sequence;
            payload.control_name = control.control_name;
            payload.control_version = control.control_version;
            remote_l3::FrameHeader header;
            header.frame_type = remote_l3::FrameType::CONTROL_REPLY;
            header.session_id = submitted.header.session_id;
            header.endpoint_id = submitted.header.endpoint_id;
            header.sequence = sequence;
            return remote_l3::encode_frame(header, remote_l3::encode_control_reply(payload));
        }

        EXPECT_EQ(frame_type, remote_l3::FrameType::COMPLETION);
        auto task = remote_l3::decode_task_payload(submitted.payload.data(), submitted.payload.size());
        EXPECT_EQ(task.callable_digest[0], 0x5A);

        remote_l3::CompletionPayload payload;
        payload.sequence = sequence;
        payload.error_code = next_error_code;
        payload.error_message = next_error_message;
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::COMPLETION;
        header.session_id = submitted.header.session_id;
        header.endpoint_id = submitted.header.endpoint_id;
        header.sequence = sequence;
        return remote_l3::encode_frame(header, remote_l3::encode_completion(payload));
    }
};

TaskSlot make_slot(Ring &ring, const TaskArgs &args) {
    AllocResult ar = ring.alloc(0, 0);
    if (ar.slot == INVALID_SLOT) throw std::runtime_error("alloc failed");
    TaskSlotState &s = *ring.slot_state(ar.slot);
    s.reset();
    s.callable.digest.fill(0x5A);
    s.worker_type = WorkerType::NEXT_LEVEL;
    s.task_args = args;
    s.is_group_ = false;
    s.state.store(TaskState::RUNNING);
    return ar.slot;
}

TaskArgs scalar_args() {
    TaskArgs args;
    args.add_scalar(7);
    return args;
}

TaskArgs bare_pointer_args() {
    TaskArgs args;
    ContinuousTensor tensor{};
    tensor.data = 0x1234;
    tensor.ndims = 1;
    tensor.shapes[0] = 1;
    tensor.dtype = DataType::UINT8;
    args.add_tensor(tensor, TensorArgType::INPUT);
    return args;
}

}  // namespace

TEST(RemoteEndpoint, SuccessCompletionMapsToSuccess) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, scalar_args());

    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    WorkerCompletion completion = endpoint.run(&ring, dispatch);

    EXPECT_EQ(completion.outcome, EndpointOutcome::SUCCESS);
    EXPECT_FALSE(transport->last_frame.empty());
    ring.shutdown();
}

TEST(RemoteEndpoint, RemoteTaskErrorMapsToTaskFailure) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, scalar_args());

    auto *transport = new FakeRemoteTransport();
    transport->next_error_code = 1;
    transport->next_error_message = "remote orch failed";
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    WorkerCompletion completion = endpoint.run(&ring, dispatch);

    EXPECT_EQ(completion.outcome, EndpointOutcome::TASK_FAILURE);
    EXPECT_EQ(completion.error_message, "remote orch failed");
    ring.shutdown();
}

TEST(RemoteEndpoint, ControlPrepareUsesTypedPrepareCallableFrame) {
    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    digest.fill(0x7B);

    endpoint.control_prepare(digest.data());

    EXPECT_EQ(transport->last_control_name, remote_l3::ControlName::PREPARE_CALLABLE);
}

TEST(RemoteEndpoint, RemoteRegisterPrepareCarriesRequestedRegistryTarget) {
    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    digest.fill(0x7B);
    std::vector<uint8_t> payload{'x'};

    endpoint.control_remote_prepare_register(
        remote_l3::RemoteRegistryTarget::INNER_L3_WORKER, CallableKind::CHIP_CALLABLE, digest.data(), payload.data(),
        payload.size()
    );

    EXPECT_EQ(transport->last_control_name, remote_l3::ControlName::PREPARE_REGISTER_CALLABLE);
    EXPECT_EQ(transport->last_target_registry, remote_l3::RemoteRegistryTarget::INNER_L3_WORKER);
    EXPECT_EQ(transport->last_callable_kind, CallableKind::CHIP_CALLABLE);
}

TEST(RemoteEndpoint, BareHostPointerWithoutSidecarIsEndpointFailure) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, bare_pointer_args());

    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    WorkerCompletion completion = endpoint.run(&ring, dispatch);

    EXPECT_EQ(completion.outcome, EndpointOutcome::ENDPOINT_FAILURE);
    EXPECT_NE(completion.error_message.find("bare host pointer"), std::string::npos);
    EXPECT_TRUE(transport->last_frame.empty());
    ring.shutdown();
}
