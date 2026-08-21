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
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstring>
#include <future>
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

// Owns a helper server thread and the stop flag it polls, for the whole
// lifetime of a test body.
//
// The stop flag lives here rather than on the test's stack because
// start_stalling_server() captures it by reference: an unwind that destroyed
// the flag while the thread still polled it would be a use-after-free. Joining
// from the destructor covers the other unwind hazard — every socket test
// constructs a RemoteL3SocketTransport after starting the server, and that
// constructor throws on a connect or HELLO timeout, which a plain
// `std::thread` local would meet while still joinable (std::terminate).
//
// Declare before the transport so the transport is destroyed first.
class ScopedServerThread {
public:
    ScopedServerThread() = default;
    ~ScopedServerThread() { stop_and_join(); }

    ScopedServerThread(const ScopedServerThread &) = delete;
    ScopedServerThread &operator=(const ScopedServerThread &) = delete;

    std::thread &thread() { return thread_; }
    std::atomic<bool> &stop_flag() { return stop_; }

    // Idempotent: tests that need the server reaped mid-body call this, and the
    // destructor then finds nothing joinable.
    void stop_and_join() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

private:
    std::thread thread_;
    std::atomic<bool> stop_{false};
};

class ScopedSigpipeCounter {
public:
    ScopedSigpipeCounter() {
        struct sigaction action{};
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
    struct sigaction old_action_{};
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

// Accept one connection, giving up once `stop` is set.
//
// A bare blocking accept() cannot be reaped: when a client never connects — the
// case when a transport constructor throws before or during connect — the
// thread parks in accept() forever and stop_and_join() would hang. Polling in
// short slices bounds that. A connection already pending always wins over
// `stop`, because poll() runs before the flag is read: ClosedPeerWrite... calls
// stop_and_join() immediately after a successful connect and still needs that
// connection accepted and RST.
//
// Returns the accepted fd, or -1 on stop / error / the hard cap.
int accept_until_stop(int listener, std::atomic<bool> &stop) {
    constexpr int POLL_SLICE_MS = 20;
    constexpr int MAX_SLICES = 500;  // 10s cap, so a wedged test cannot hang the suite
    for (int i = 0; i < MAX_SLICES; ++i) {
        struct pollfd pfd{};
        pfd.fd = listener;
        pfd.events = POLLIN;
        int ready = ::poll(&pfd, 1, POLL_SLICE_MS);
        if (ready > 0) return ::accept(listener, nullptr, nullptr);
        if (ready < 0 && errno != EINTR) return -1;
        if (stop.load(std::memory_order_acquire)) return -1;
    }
    return -1;
}

uint16_t start_closing_server(std::thread &server_thread, std::atomic<bool> &stop) {
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
    server_thread = std::thread([listener, &stop]() {
        int fd = accept_until_stop(listener, stop);
        if (fd >= 0) {
            struct linger rst{};
            rst.l_onoff = 1;
            rst.l_linger = 0;
            (void)::setsockopt(fd, SOL_SOCKET, SO_LINGER, &rst, sizeof(rst));
            ::close(fd);
        }
        ::close(listener);
    });
    return ntohs(addr.sin_port);
}

int make_loopback_listener(uint16_t &port_out) {
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
    port_out = ntohs(addr.sin_port);
    return listener;
}

// Accept one connection and hold it open (sending nothing) until `stop` is set,
// so a client blocked in read_frame can only end by timing out on its own
// deadline — never on EOF. Holding well past any client timeout is what lets a
// test tell an attach-bounded read from a runtime-bounded one; a hard safety cap
// keeps a failing test from hanging the suite.
uint16_t start_stalling_server(std::thread &server_thread, std::atomic<bool> &stop) {
    uint16_t port = 0;
    int listener = make_loopback_listener(port);
    server_thread = std::thread([listener, &stop]() {
        int fd = accept_until_stop(listener, stop);
        for (int i = 0; i < 500 && !stop.load(std::memory_order_acquire); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (fd >= 0) ::close(fd);
        ::close(listener);
    });
    return port;
}

// Accept one connection, wait delay_ms, then send a single COMPLETION frame with
// the given sequence so a client's wait_for_reply(COMPLETION, seq) succeeds.
uint16_t
start_delayed_reply_server(std::thread &server_thread, std::atomic<bool> &stop, int delay_ms, uint64_t sequence) {
    uint16_t port = 0;
    int listener = make_loopback_listener(port);
    server_thread = std::thread([listener, &stop, delay_ms, sequence]() {
        int fd = accept_until_stop(listener, stop);
        if (fd >= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            remote_l3::FrameHeader header;
            header.frame_type = remote_l3::FrameType::COMPLETION;
            header.session_id = 1;
            header.worker_id = 0;
            header.sequence = sequence;
            std::vector<uint8_t> frame = remote_l3::encode_frame(header, {});
            size_t off = 0;
            while (off < frame.size()) {
                ssize_t n = ::send(fd, frame.data() + off, frame.size() - off, MSG_NOSIGNAL);
                if (n <= 0) break;
                off += static_cast<size_t>(n);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ::close(fd);
        }
        ::close(listener);
    });
    return port;
}

uint16_t start_split_header_reply_server(
    std::thread &server_thread, std::atomic<bool> &stop, std::atomic<bool> &prefix_sent, std::atomic<bool> &send_suffix,
    uint64_t sequence
) {
    uint16_t port = 0;
    int listener = make_loopback_listener(port);
    server_thread = std::thread([listener, &stop, &prefix_sent, &send_suffix, sequence]() {
        int fd = accept_until_stop(listener, stop);
        if (fd >= 0) {
            remote_l3::FrameHeader header;
            header.frame_type = remote_l3::FrameType::COMPLETION;
            header.session_id = 1;
            header.worker_id = 0;
            header.sequence = sequence;
            std::vector<uint8_t> frame = remote_l3::encode_frame(header, {});
            const size_t prefix_size = remote_l3::FRAME_HEADER_BYTES / 2;
            size_t offset = 0;
            while (offset < prefix_size) {
                ssize_t n = ::send(fd, frame.data() + offset, prefix_size - offset, MSG_NOSIGNAL);
                if (n <= 0) break;
                offset += static_cast<size_t>(n);
            }
            if (offset == prefix_size) {
                prefix_sent.store(true, std::memory_order_release);
                while (!send_suffix.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (!stop.load(std::memory_order_acquire)) {
                    while (offset < frame.size()) {
                        ssize_t n = ::send(fd, frame.data() + offset, frame.size() - offset, MSG_NOSIGNAL);
                        if (n <= 0) break;
                        offset += static_cast<size_t>(n);
                    }
                }
            }
            ::close(fd);
        }
        ::close(listener);
    });
    return port;
}

class FakeRemoteTransport : public RemoteL3Transport {
public:
    int32_t next_error_code{0};
    std::string next_error_message;
    std::vector<uint8_t> next_control_result_bytes;
    std::vector<uint8_t> last_frame;
    int progress_polls_before_ready{0};
    remote_l3::ControlName last_control_name{remote_l3::ControlName::PREPARE_CALLABLE};
    remote_l3::RemoteRegistryTarget last_target_registry{remote_l3::RemoteRegistryTarget::REMOTE_TASK_DISPATCHER};
    CallableKind last_callable_kind{CallableKind::PYTHON_IMPORT};
    int shutdown_calls{0};

    void submit_frame(const std::vector<uint8_t> &frame) override {
        if (shutdown_calls != 0) throw std::runtime_error("FakeRemoteTransport: submit after shutdown");
        last_frame = frame;
    }
    void submit_progress_frame(const std::vector<uint8_t> &frame) override { submit_frame(frame); }
    void shutdown() override { ++shutdown_calls; }

    bool poll_progress_reply(remote_l3::FrameType frame_type, uint64_t sequence, std::vector<uint8_t> &reply) override {
        if (progress_polls_before_ready > 0) {
            --progress_polls_before_ready;
            return false;
        }
        reply = wait_for_reply(frame_type, sequence);
        return true;
    }

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
    Tensor ref{};
    ref.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
    ref.buffer.nbytes = 1;  // a local backing without a remote sidecar -> rejected by the remote endpoint
    ref.buffer.identity.buffer_id = 0x1234;
    ref.ndims = 1;
    ref.shapes[0] = 1;
    ref.strides[0] = 1;
    ref.dtype = DataType::UINT8;
    args.add_tensor(ref, TensorArgType::INPUT);
    return args;
}

TaskArgs sidecar_free_placeholder_args() {
    TaskArgs args;
    Tensor ref{};
    ref.buffer.magic = BUFFER_DESCRIPTOR_MAGIC;
    ref.buffer.address_space = static_cast<uint8_t>(AddressSpace::HOST);
    ref.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
    ref.buffer.backend_kind = static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR);
    ref.buffer.identity.buffer_id = 0x1234;
    ref.buffer.identity.generation = 1;
    ref.buffer.nbytes = 1;
    ref.ndims = 1;
    ref.shapes[0] = 1;
    ref.strides[0] = 1;
    ref.dtype = DataType::UINT8;
    args.add_tensor(ref, TensorArgType::INPUT);
    return args;
}

}  // namespace

TEST(RemoteEndpoint, TaskDispatchUsesProgressSubmissionAndPolling) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, scalar_args());

    auto *transport = new FakeRemoteTransport();
    transport->progress_polls_before_ready = 1;
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    dispatch.dispatch_id = 7;
    endpoint.submit_progress(&ring, dispatch);

    WorkerEndpointProgress progress;
    EXPECT_FALSE(endpoint.poll_progress(progress));
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.kind, WorkerProgressKind::COMPLETED);
    EXPECT_EQ(progress.dispatch.dispatch_id, 7u);
    EXPECT_EQ(progress.completion.outcome, EndpointOutcome::SUCCESS);
    EXPECT_FALSE(transport->last_frame.empty());
    ring.shutdown();
}

TEST(RemoteEndpoint, ProgressStopReleasesWaitingControl) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, scalar_args());

    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));
    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    endpoint.submit_progress(&ring, dispatch);

    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest{};
    auto control = std::async(std::launch::async, [&] {
        endpoint.control_prepare(digest.data());
    });
    EXPECT_EQ(control.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    endpoint.request_progress_stop();
    ASSERT_EQ(control.wait_for(std::chrono::milliseconds(500)), std::future_status::ready);
    EXPECT_THROW(control.get(), std::runtime_error);
    ring.shutdown();
}

TEST(RemoteEndpoint, IdleProgressStopDefersTransportShutdownUntilLifecycleFrame) {
    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    endpoint.request_progress_stop();
    EXPECT_EQ(transport->shutdown_calls, 0);

    endpoint.shutdown_child();
    ASSERT_FALSE(transport->last_frame.empty());
    EXPECT_EQ(remote_l3::decode_frame(transport->last_frame).header.frame_type, remote_l3::FrameType::SHUTDOWN);
    EXPECT_EQ(transport->shutdown_calls, 1);
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
    endpoint.submit_progress(&ring, dispatch);

    WorkerEndpointProgress progress;
    ASSERT_TRUE(endpoint.poll_progress(progress));
    EXPECT_EQ(progress.completion.outcome, EndpointOutcome::TASK_FAILURE);
    EXPECT_EQ(progress.completion.error_message, "remote orch failed");
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
    ScopedServerThread server;
    uint16_t port = start_closing_server(server.thread(), server.stop_flag());
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, 1.0, 1.0);
    server.stop_and_join();
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

// Every socket test below starts a server thread and then constructs a
// RemoteL3SocketTransport, whose constructor throws on a connect or HELLO
// timeout — routine on a loaded box. With a bare `std::thread` local the unwind
// destroyed it while still joinable, and std::terminate aborted the whole
// binary mid-suite. Reaching the end of this test at all is the assertion: a
// regression turns it into an abort, not a failure.
//
// The server here never sees a client, so it also pins that a thread parked in
// accept() is still reapable — the join must not hang.
TEST(RemoteSocketTransport, ServerThreadIsJoinedWhenTestBodyUnwinds) {
    bool caught = false;
    auto t0 = std::chrono::steady_clock::now();
    try {
        ScopedServerThread server;
        (void)start_stalling_server(server.thread(), server.stop_flag());
        throw std::runtime_error("stands in for a transport constructor timeout");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    EXPECT_TRUE(caught);
    EXPECT_LT(elapsed, 5.0) << "destructor join should not wait out the server's own cap";
}

TEST(RemoteSocketTransport, CtorRejectsNonPositiveTimeouts) {
    // Validation runs before connect_socket(), so no server is needed.
    EXPECT_THROW(RemoteL3SocketTransport("127.0.0.1", 1, "127.0.0.1", 1, 0.0, 5.0), std::invalid_argument);
    EXPECT_THROW(RemoteL3SocketTransport("127.0.0.1", 1, "127.0.0.1", 1, -1.0, 5.0), std::invalid_argument);
    EXPECT_THROW(RemoteL3SocketTransport("127.0.0.1", 1, "127.0.0.1", 1, 5.0, 0.0), std::invalid_argument);
    EXPECT_THROW(RemoteL3SocketTransport("127.0.0.1", 1, "127.0.0.1", 1, 5.0, -1.0), std::invalid_argument);
}

TEST(RemoteSocketTransport, HelloReadBoundedByAttachTimeout) {
    // Server accepts the command connection but never sends HELLO, so the read
    // can only end by timing out. A small attach budget (0.2s) and a large
    // runtime budget (5.0s) tell the two apart: bounding the HELLO read by the
    // runtime timeout would take ~5s.
    ScopedServerThread server;
    uint16_t port = start_stalling_server(server.thread(), server.stop_flag());
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 0.2, /*runtime*/ 5.0);

    auto t0 = std::chrono::steady_clock::now();
    EXPECT_THROW(transport.expect_hello_ready(1, 0, "sim"), std::runtime_error);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(elapsed, 1.0);

    transport.shutdown();
    server.stop_and_join();
}

TEST(RemoteSocketTransport, RuntimeReadSurvivesElapsedAttachDeadline) {
    // The reply arrives after the attach deadline (0.3s) has elapsed. A runtime
    // read that (wrongly) reused the attach deadline would throw immediately; a
    // fresh runtime budget (2.0s) receives it. This proves the value split, not
    // just the path.
    ScopedServerThread server;
    uint16_t port = start_delayed_reply_server(server.thread(), server.stop_flag(), /*delay_ms=*/500, /*sequence=*/1);
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 0.3, /*runtime*/ 2.0);

    std::vector<uint8_t> probe(16, 0x11);
    transport.submit_frame(probe);
    EXPECT_NO_THROW({
        auto reply = transport.wait_for_reply(remote_l3::FrameType::COMPLETION, 1);
        EXPECT_FALSE(reply.empty());
    });

    transport.shutdown();
    server.stop_and_join();
}

TEST(RemoteSocketTransport, ProgressPollDoesNotWaitForDelayedReply) {
    ScopedServerThread server;
    uint16_t port = start_delayed_reply_server(server.thread(), server.stop_flag(), /*delay_ms=*/500, /*sequence=*/1);
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 1.0, /*runtime*/ 2.0);

    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::TASK;
    header.session_id = 1;
    header.worker_id = 0;
    header.sequence = 1;
    transport.submit_progress_frame(remote_l3::encode_frame(header, {}));

    std::vector<uint8_t> reply;
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_FALSE(transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 1, reply));
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(elapsed, 0.2);

    bool complete = false;
    for (int i = 0; i < 100 && !complete; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        complete = transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 1, reply);
    }
    EXPECT_TRUE(complete);
    EXPECT_FALSE(reply.empty());
    transport.shutdown();
    server.stop_and_join();
}

TEST(RemoteSocketTransport, ProgressPollResumesAfterPartialHeader) {
    std::atomic<bool> prefix_sent{false};
    std::atomic<bool> send_suffix{false};
    ScopedServerThread server;
    uint16_t port =
        start_split_header_reply_server(server.thread(), server.stop_flag(), prefix_sent, send_suffix, /*sequence=*/1);
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 1.0, /*runtime*/ 2.0);

    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::TASK;
    header.session_id = 1;
    header.worker_id = 0;
    header.sequence = 1;
    transport.submit_progress_frame(remote_l3::encode_frame(header, {}));

    for (int i = 0; i < 100 && !prefix_sent.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(prefix_sent.load(std::memory_order_acquire));
    std::vector<uint8_t> reply;
    EXPECT_FALSE(transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 1, reply));

    send_suffix.store(true, std::memory_order_release);
    bool complete = false;
    for (int i = 0; i < 100 && !complete; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        complete = transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 1, reply);
    }
    EXPECT_TRUE(complete);
    EXPECT_EQ(reply.size(), remote_l3::FRAME_HEADER_BYTES);
    transport.shutdown();
    server.stop_and_join();
}

TEST(RemoteSocketTransport, ProgressErrorClearsActiveCommand) {
    ScopedServerThread server;
    uint16_t port = start_closing_server(server.thread(), server.stop_flag());
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 1.0, /*runtime*/ 1.0);
    server.stop_and_join();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    remote_l3::FrameHeader header;
    header.frame_type = remote_l3::FrameType::TASK;
    header.session_id = 1;
    header.worker_id = 0;
    header.sequence = 1;
    std::vector<uint8_t> frame = remote_l3::encode_frame(header, {});
    transport.submit_progress_frame(frame);

    std::vector<uint8_t> reply;
    bool saw_error = false;
    for (int i = 0; i < 3 && !saw_error; ++i) {
        try {
            (void)transport.poll_progress_reply(remote_l3::FrameType::COMPLETION, 1, reply);
        } catch (const std::runtime_error &) {
            saw_error = true;
        }
    }
    ASSERT_TRUE(saw_error);
    EXPECT_NO_THROW(transport.submit_progress_frame(frame));
    transport.shutdown();
}

TEST(RemoteSocketTransport, RuntimeWriteToStalledReaderTimesOut) {
    // The peer accepts but never reads; a large frame overruns the socket buffers
    // so a single blocking send() would hang past the runtime deadline. The fd is
    // persistently O_NONBLOCK, so the write re-polls under the deadline and throws.
    ScopedServerThread server;
    uint16_t port = start_stalling_server(server.thread(), server.stop_flag());
    RemoteL3SocketTransport transport("127.0.0.1", port, "127.0.0.1", 1, /*attach*/ 1.0, /*runtime*/ 0.5);

    std::vector<uint8_t> big(16 * 1024 * 1024, 0x7E);
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_THROW(transport.submit_frame(big), std::runtime_error);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(elapsed, 3.0);

    transport.shutdown();
    server.stop_and_join();
}

TEST(RemoteEndpoint, BareHostPointerWithoutSidecarIsRejectedAtSubmission) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, bare_pointer_args());

    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    // Encoding happens while publishing, so an unsidecar'd local backing is
    // rejected at submission rather than reported as a completion. Match the
    // message: submit_progress has several other runtime_error paths, and a
    // bare EXPECT_THROW would pass on any of them.
    try {
        endpoint.submit_progress(&ring, dispatch);
        ADD_FAILURE() << "expected the local backing to be rejected";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("local backing"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(transport->last_frame.empty());
    ring.shutdown();
}

TEST(RemoteEndpoint, SidecarFreePlaceholderIsRejectedAtSubmission) {
    Ring ring;
    ring.init(1ULL << 20);
    TaskSlot slot = make_slot(ring, sidecar_free_placeholder_args());

    auto *transport = new FakeRemoteTransport();
    RemoteL3Endpoint endpoint(3, 99, "fake", std::unique_ptr<RemoteL3Transport>(transport));

    WorkerDispatch dispatch;
    dispatch.task_slot = slot;
    // A REMOTE_SIDECAR placeholder names its backing only through its sidecar, so one submitted
    // without a sidecar names nothing the runner could resolve. Match the message: submit_progress
    // has several other runtime_error paths.
    try {
        endpoint.submit_progress(&ring, dispatch);
        ADD_FAILURE() << "expected the sidecar-free placeholder to be rejected";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("without remote sidecar"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(transport->last_frame.empty());
    ring.shutdown();
}

namespace {

std::vector<uint8_t> ready_mpi_mailbox(int32_t world_size) {
    using namespace mpi_group_mailbox;
    std::vector<uint8_t> mailbox(MAILBOX_BYTES, 0);
    std::memcpy(mailbox.data() + OFF_MAGIC, MAGIC, sizeof(MAGIC));
    const uint32_t version = PROTOCOL_VERSION;
    const uint32_t header_bytes = HEADER_BYTES;
    const uint64_t mailbox_bytes = MAILBOX_BYTES;
    const uint32_t size = static_cast<uint32_t>(world_size);
    const int32_t ready = static_cast<int32_t>(GroupState::READY);
    const int32_t idle = static_cast<int32_t>(RequestState::IDLE);
    std::memcpy(mailbox.data() + OFF_PROTOCOL_VERSION, &version, sizeof(version));
    std::memcpy(mailbox.data() + OFF_HEADER_BYTES, &header_bytes, sizeof(header_bytes));
    std::memcpy(mailbox.data() + OFF_MAILBOX_BYTES, &mailbox_bytes, sizeof(mailbox_bytes));
    std::memcpy(mailbox.data() + OFF_WORLD_SIZE, &size, sizeof(size));
    std::memcpy(mailbox.data() + OFF_GROUP_STATE, &ready, sizeof(ready));
    std::memcpy(mailbox.data() + OFF_REQUEST_STATE, &idle, sizeof(idle));
    return mailbox;
}

int32_t mailbox_state(const std::vector<uint8_t> &mailbox, size_t offset) {
    int32_t value = 0;
    __atomic_load(reinterpret_cast<const int32_t *>(mailbox.data() + offset), &value, __ATOMIC_ACQUIRE);
    return value;
}

void set_mailbox_state(std::vector<uint8_t> &mailbox, size_t offset, int32_t value) {
    __atomic_store(reinterpret_cast<int32_t *>(mailbox.data() + offset), &value, __ATOMIC_RELEASE);
}

void respond_with_payloads(std::vector<uint8_t> &mailbox, const std::vector<std::vector<uint8_t>> &payloads) {
    using namespace mpi_group_mailbox;
    while (mailbox_state(mailbox, OFF_REQUEST_STATE) != static_cast<int32_t>(RequestState::REQUEST_READY)) {}
    const uint32_t count = static_cast<uint32_t>(payloads.size());
    std::memcpy(mailbox.data() + RESPONSE_OFFSET, &count, sizeof(count));
    size_t offset = RESPONSE_OFFSET + 4;
    size_t response_bytes = 4 + 4 * payloads.size();
    for (const auto &payload : payloads) {
        const uint32_t size = static_cast<uint32_t>(payload.size());
        std::memcpy(mailbox.data() + offset, &size, sizeof(size));
        offset += 4;
        response_bytes += payload.size();
    }
    for (const auto &payload : payloads) {
        if (!payload.empty()) std::memcpy(mailbox.data() + offset, payload.data(), payload.size());
        offset += payload.size();
    }
    std::memcpy(mailbox.data() + OFF_RESPONSE_COUNT, &count, sizeof(count));
    const uint32_t encoded_bytes = static_cast<uint32_t>(response_bytes);
    std::memcpy(mailbox.data() + OFF_RESPONSE_BYTES, &encoded_bytes, sizeof(encoded_bytes));
    set_mailbox_state(mailbox, OFF_REQUEST_STATE, static_cast<int32_t>(mpi_group_mailbox::RequestState::TASK_DONE));
}

}  // namespace

TEST(MpiGroupMailboxChannel, FullGroupTaskUsesOnePerRankEnvelope) {
    using namespace mpi_group_mailbox;
    auto mailbox = ready_mpi_mailbox(2);
    MpiGroupMailboxChannel channel(mailbox.data(), mailbox.size(), 2, -1, 2.0);
    std::vector<uint8_t> reply0;
    std::vector<uint8_t> reply1;
    std::thread rank0([&]() {
        reply0 = channel.exchange_group_task({0x10}, 0, 7, 2);
    });
    std::thread rank1([&]() {
        reply1 = channel.exchange_group_task({0x20}, 1, 7, 2);
    });

    while (mailbox_state(mailbox, OFF_REQUEST_STATE) != static_cast<int32_t>(RequestState::REQUEST_READY)) {}
    uint32_t target = 0;
    uint32_t request_count = 0;
    std::memcpy(&target, mailbox.data() + OFF_TARGET, sizeof(target));
    std::memcpy(&request_count, mailbox.data() + OFF_REQUEST_COUNT, sizeof(request_count));
    EXPECT_EQ(target, static_cast<uint32_t>(Target::PER_RANK));
    EXPECT_EQ(request_count, 2U);
    respond_with_payloads(mailbox, {{0xA0}, {0xA1}});

    rank0.join();
    rank1.join();
    EXPECT_EQ(reply0, std::vector<uint8_t>({0xA0}));
    EXPECT_EQ(reply1, std::vector<uint8_t>({0xA1}));
    EXPECT_EQ(mailbox_state(mailbox, OFF_REQUEST_STATE), static_cast<int32_t>(RequestState::IDLE));
}

TEST(MpiGroupMailboxChannel, RejectsNonZeroReservedHeaderBytes) {
    using namespace mpi_group_mailbox;
    auto mailbox = ready_mpi_mailbox(1);
    mailbox[RESERVED_OFFSET + 8] = 1;
    EXPECT_THROW(MpiGroupMailboxChannel(mailbox.data(), mailbox.size(), 1, -1, 1.0), std::invalid_argument);
}

TEST(MpiGroupMailboxChannel, TimeoutMakesGroupTerminal) {
    using namespace mpi_group_mailbox;
    auto mailbox = ready_mpi_mailbox(1);
    MpiGroupMailboxChannel channel(mailbox.data(), mailbox.size(), 1, -1, 0.01);
    EXPECT_THROW(channel.exchange_group_task({0x10}, 0, 9, 1), std::runtime_error);
    EXPECT_TRUE(channel.terminal());
    EXPECT_EQ(mailbox_state(mailbox, OFF_GROUP_STATE), static_cast<int32_t>(GroupState::TERMINAL));
}

TEST(MpiGroupMailboxChannel, GroupArrivalTimeoutFailsTaskWithoutKillingGroup) {
    using namespace mpi_group_mailbox;
    auto mailbox = ready_mpi_mailbox(2);
    MpiGroupMailboxChannel channel(mailbox.data(), mailbox.size(), 2, -1, 0.05);
    EXPECT_THROW(channel.exchange_group_task({0x10}, 0, 9, 2), std::runtime_error);
    EXPECT_FALSE(channel.terminal());
    EXPECT_EQ(mailbox_state(mailbox, OFF_GROUP_STATE), static_cast<int32_t>(GroupState::READY));
    // The withdrawn arrival leaves a clean slate: a later full batch succeeds.
    std::vector<uint8_t> reply0;
    std::vector<uint8_t> reply1;
    std::thread rank0([&]() {
        reply0 = channel.exchange_group_task({0x11}, 0, 10, 2);
    });
    std::thread rank1([&]() {
        reply1 = channel.exchange_group_task({0x21}, 1, 10, 2);
    });
    respond_with_payloads(mailbox, {{0xB0}, {0xB1}});
    rank0.join();
    rank1.join();
    EXPECT_EQ(reply0, std::vector<uint8_t>({0xB0}));
    EXPECT_EQ(reply1, std::vector<uint8_t>({0xB1}));
}

TEST(MpiGroupMailboxTransport, GroupProgressSubmitAndPollRoundTrip) {
    using namespace mpi_group_mailbox;
    auto mailbox = ready_mpi_mailbox(2);
    auto channel = std::make_shared<MpiGroupMailboxChannel>(mailbox.data(), mailbox.size(), 2, -1, 2.0);
    MpiGroupMailboxTransport rank0(channel, 0);
    MpiGroupMailboxTransport rank1(channel, 1);

    auto task_frame = [](int32_t worker_id, uint64_t sequence) {
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::TASK;
        header.session_id = 77;
        header.worker_id = worker_id;
        header.sequence = sequence;
        return remote_l3::encode_frame(header, {0x11});
    };
    auto completion_frame = [](int32_t worker_id, uint64_t sequence, uint8_t marker) {
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::COMPLETION;
        header.session_id = 77;
        header.worker_id = worker_id;
        header.sequence = sequence;
        return remote_l3::encode_frame(header, {marker});
    };

    rank0.submit_group_progress_frame(task_frame(10, 21), 7, 0, 2);
    std::vector<uint8_t> reply;
    EXPECT_FALSE(rank0.poll_progress_reply(remote_l3::FrameType::COMPLETION, 21, reply));
    rank1.submit_group_progress_frame(task_frame(11, 22), 7, 1, 2);

    respond_with_payloads(mailbox, {completion_frame(10, 21, 0xA0), completion_frame(11, 22, 0xA1)});

    while (!rank0.poll_progress_reply(remote_l3::FrameType::COMPLETION, 21, reply)) {}
    EXPECT_EQ(remote_l3::decode_frame(reply).payload, std::vector<uint8_t>({0xA0}));
    while (!rank1.poll_progress_reply(remote_l3::FrameType::COMPLETION, 22, reply)) {}
    EXPECT_EQ(remote_l3::decode_frame(reply).payload, std::vector<uint8_t>({0xA1}));
    EXPECT_EQ(mailbox_state(mailbox, OFF_REQUEST_STATE), static_cast<int32_t>(RequestState::IDLE));
}
