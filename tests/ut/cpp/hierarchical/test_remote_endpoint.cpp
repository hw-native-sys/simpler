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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "remote_endpoint.h"
#include "ring.h"

namespace {

volatile sig_atomic_t g_sigpipe_count = 0;

void count_sigpipe(int) { ++g_sigpipe_count; }

class ScopedSigpipeCounter {
public:
    ScopedSigpipeCounter() {
        struct sigaction action {};
        action.sa_handler = count_sigpipe;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGPIPE, &action, &old_action_) != 0) {
            throw std::runtime_error(std::string("sigaction failed: ") + std::strerror(errno));
        }
        g_sigpipe_count = 0;
    }

    ~ScopedSigpipeCounter() { (void)sigaction(SIGPIPE, &old_action_, nullptr); }

    ScopedSigpipeCounter(const ScopedSigpipeCounter &) = delete;
    ScopedSigpipeCounter &operator=(const ScopedSigpipeCounter &) = delete;

private:
    struct sigaction old_action_ {};
};

void append_i32(std::vector<uint8_t> &out, int32_t v) {
    uint32_t raw = static_cast<uint32_t>(v);
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>((raw >> (8 * i)) & 0xffU));
}

void append_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffU));
}

std::vector<uint8_t>
malloc_result(int32_t worker_id, uint64_t buffer_id, uint64_t generation, int32_t address_space, uint64_t nbytes) {
    std::vector<uint8_t> out;
    append_i32(out, worker_id);
    append_u64(out, buffer_id);
    append_u64(out, generation);
    append_i32(out, address_space);
    append_u64(out, nbytes);
    append_u64(out, 0x1000);
    append_u64(out, 0x2000);
    append_u64(out, 0x3000);
    return out;
}

uint16_t start_closing_server(std::thread &server_thread) {
    int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
    int one = 1;
    (void)::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(listener, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        int err = errno;
        ::close(listener);
        throw std::runtime_error(std::string("bind failed: ") + std::strerror(err));
    }
    if (::listen(listener, 1) != 0) {
        int err = errno;
        ::close(listener);
        throw std::runtime_error(std::string("listen failed: ") + std::strerror(err));
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listener, reinterpret_cast<sockaddr *>(&addr), &len) != 0) {
        int err = errno;
        ::close(listener);
        throw std::runtime_error(std::string("getsockname failed: ") + std::strerror(err));
    }
    server_thread = std::thread([listener]() {
        int fd = ::accept(listener, nullptr, nullptr);
        if (fd >= 0) {
            struct linger rst {};
            rst.l_onoff = 1;
            rst.l_linger = 0;
            (void)::setsockopt(fd, SOL_SOCKET, SO_LINGER, &rst, sizeof(rst));
            ::close(fd);
        }
        ::close(listener);
    });
    return ntohs(addr.sin_port);
}

class FakeRemoteTransport : public RemoteL3Transport {
public:
    int32_t next_error_code{0};
    std::string next_error_message;
    std::vector<uint8_t> next_control_result_bytes;
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
                if (control.command_bytes.size() < 8u) {
                    ADD_FAILURE() << "PREPARE_REGISTER_CALLABLE command is truncated";
                    return {};
                }
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
            payload.result_bytes = next_control_result_bytes;
            remote_l3::FrameHeader header;
            header.frame_type = remote_l3::FrameType::CONTROL_REPLY;
            header.session_id = submitted.header.session_id;
            header.worker_id = submitted.header.worker_id;
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
        header.worker_id = submitted.header.worker_id;
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
    Tensor tensor{};
    tensor.buffer.addr = 0x1234;
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

TEST(RemoteEndpoint, RemoteMallocAcceptsValidOwnerHandle) {
    auto *transport = new FakeRemoteTransport();
    transport->next_control_result_bytes =
        malloc_result(3, 9, 2, static_cast<int32_t>(RemoteAddressSpace::REMOTE_DEVICE), 64);
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    RemoteBufferHandle handle = endpoint.control_remote_malloc(64);

    EXPECT_EQ(handle.worker_id, 3);
    EXPECT_EQ(handle.owner_worker_id, 3);
    EXPECT_EQ(handle.buffer_id, 9u);
    EXPECT_EQ(handle.generation, 2u);
    EXPECT_EQ(handle.import_id, 0u);
    EXPECT_EQ(handle.address_space, RemoteAddressSpace::REMOTE_DEVICE);
    EXPECT_EQ(handle.nbytes, 64u);
}

TEST(RemoteEndpoint, RemoteMallocRejectsInvalidOwnerHandle) {
    auto expect_reject = [](std::vector<uint8_t> result_bytes, size_t requested_size) {
        auto *transport = new FakeRemoteTransport();
        transport->next_control_result_bytes = std::move(result_bytes);
        RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
        EXPECT_THROW((void)endpoint.control_remote_malloc(requested_size), std::runtime_error);
    };

    EXPECT_THROW(
        {
            auto *transport = new FakeRemoteTransport();
            RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
            (void)endpoint.control_remote_malloc(0);
        },
        std::invalid_argument
    );
    expect_reject(malloc_result(3, 0, 2, static_cast<int32_t>(RemoteAddressSpace::REMOTE_DEVICE), 64), 64);
    expect_reject(malloc_result(3, 9, 0, static_cast<int32_t>(RemoteAddressSpace::REMOTE_DEVICE), 64), 64);
    expect_reject(malloc_result(3, 9, 2, static_cast<int32_t>(RemoteAddressSpace::HOST_INLINE), 64), 64);
    expect_reject(malloc_result(3, 9, 2, 99, 64), 64);
    expect_reject(malloc_result(3, 9, 2, static_cast<int32_t>(RemoteAddressSpace::REMOTE_DEVICE), 32), 64);
}

TEST(RemoteEndpoint, RemoteBufferControlsRejectOutOfRangeSlices) {
    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
    RemoteBufferHandle handle;
    handle.worker_id = 3;
    handle.owner_worker_id = 3;
    handle.buffer_id = 9;
    handle.generation = 2;
    handle.address_space = RemoteAddressSpace::REMOTE_DEVICE;
    handle.nbytes = 64;
    handle.offset = 0;
    std::array<uint8_t, 8> bytes{};

    EXPECT_THROW(endpoint.control_remote_copy_to(handle, 64, bytes.data(), 1), std::out_of_range);
    EXPECT_THROW(endpoint.control_remote_copy_from(bytes.data(), handle, 63, 2), std::out_of_range);
    EXPECT_THROW(
        endpoint.control_remote_export(handle, 64, 1, remote_l3::REMOTE_BUFFER_ACCESS_READ, "tcp"), std::out_of_range
    );

    handle.offset = 16;
    EXPECT_THROW(
        endpoint.control_remote_export(handle, 48, 1, remote_l3::REMOTE_BUFFER_ACCESS_READ, "tcp"), std::out_of_range
    );

    handle.offset = 0;
    handle.nbytes = 0;
    EXPECT_THROW(endpoint.control_remote_copy_to(handle, 0, bytes.data(), 1), std::invalid_argument);
}

TEST(RemoteSocketTransport, ClosedPeerWriteDoesNotRaiseSigpipe) {
    std::thread server_thread;
    uint16_t port = start_closing_server(server_thread);
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, 1.0);
    server_thread.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ScopedSigpipeCounter sigpipe_counter;
    std::vector<uint8_t> frame(4096, 0x5A);
    bool saw_error = false;
    for (int i = 0; i < 3; ++i) {
        try {
            transport.submit_frame(frame);
        } catch (const std::runtime_error &) {
            saw_error = true;
            break;
        }
    }

    EXPECT_TRUE(saw_error);
    EXPECT_EQ(g_sigpipe_count, 0);
    transport.shutdown();
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
