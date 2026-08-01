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

#include "remote_endpoint.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>

#include "ring.h"

namespace {

using Deadline = std::chrono::steady_clock::time_point;

struct TcpAddress {
    int family{AF_UNSPEC};
    int socktype{SOCK_STREAM};
    int protocol{0};
    sockaddr_storage addr{};
    socklen_t addrlen{0};
};

struct ResolveTcpResult {
    int error_code{0};
    std::string error_message;
    std::vector<TcpAddress> addresses;
};

uint32_t read_le_u32(const uint8_t *data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> digest_array(const uint8_t *digest) {
    if (digest == nullptr) throw std::invalid_argument("RemoteL3Endpoint: null callable digest");
    std::array<uint8_t, CALLABLE_HASH_DIGEST_SIZE> out{};
    std::memcpy(out.data(), digest, out.size());
    return out;
}

void put_u32(std::vector<uint8_t> &out, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffU));
}

void put_i32(std::vector<uint8_t> &out, int32_t v) { put_u32(out, static_cast<uint32_t>(v)); }

void put_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xffU));
}

uint64_t get_u64(const std::vector<uint8_t> &data, size_t &offset) {
    if (offset > data.size() || data.size() - offset < 8) {
        throw std::runtime_error("RemoteL3Endpoint: truncated uint64 result");
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(data[offset++]) << (8 * i);
    return v;
}

int32_t get_i32(const std::vector<uint8_t> &data, size_t &offset) {
    if (offset > data.size() || data.size() - offset < 4) {
        throw std::runtime_error("RemoteL3Endpoint: truncated int32 result");
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<uint32_t>(data[offset++]) << (8 * i);
    return static_cast<int32_t>(v);
}

double remaining_seconds(Deadline deadline, const std::string &message) {
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) throw std::runtime_error(message);
    return std::chrono::duration<double>(deadline - now).count();
}

int timeout_to_poll_ms(double timeout_s) {
    if (timeout_s <= 0.0) throw std::invalid_argument("RemoteL3SocketTransport: timeout must be positive");
    int timeout_ms = static_cast<int>(timeout_s * 1000.0);
    return timeout_ms > 0 ? timeout_ms : 1;
}

ResolveTcpResult resolve_tcp_addresses(const std::string &host, const std::string &port_s) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *results = nullptr;

    ResolveTcpResult result;
    int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &results);
    if (rc != 0) {
        result.error_code = rc;
        result.error_message = gai_strerror(rc);
        return result;
    }
    for (addrinfo *ai = results; ai != nullptr; ai = ai->ai_next) {
        if (ai->ai_addrlen > sizeof(sockaddr_storage)) continue;
        TcpAddress addr;
        addr.family = ai->ai_family;
        addr.socktype = ai->ai_socktype;
        addr.protocol = ai->ai_protocol;
        addr.addrlen = static_cast<socklen_t>(ai->ai_addrlen);
        std::memcpy(&addr.addr, ai->ai_addr, ai->ai_addrlen);
        result.addresses.push_back(addr);
    }
    freeaddrinfo(results);
    return result;
}

std::vector<TcpAddress> resolve_tcp_addresses_with_timeout(
    const std::string &host, const std::string &port_s, const std::string &label, Deadline deadline
) {
    auto promise = std::make_shared<std::promise<ResolveTcpResult>>();
    std::future<ResolveTcpResult> future = promise->get_future();
    std::thread([promise, host, port_s]() {
        try {
            promise->set_value(resolve_tcp_addresses(host, port_s));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }).detach();

    double remaining = remaining_seconds(deadline, label + ": timed out resolving address");
    if (future.wait_for(std::chrono::duration<double>(remaining)) != std::future_status::ready) {
        throw std::runtime_error(label + ": timed out resolving address");
    }
    ResolveTcpResult result = future.get();
    if (result.error_code != 0) {
        throw std::runtime_error(
            label + ": getaddrinfo failed for " + host + ":" + port_s + ": " + result.error_message
        );
    }
    if (result.addresses.empty()) {
        throw std::runtime_error(label + ": no TCP address found for " + host + ":" + port_s);
    }
    return result.addresses;
}

void configure_socket_no_sigpipe(int fd, const std::string &label) {
#if defined(SO_NOSIGPIPE)
    int one = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) {
        throw std::runtime_error(label + ": setsockopt(SO_NOSIGPIPE) failed: " + std::strerror(errno));
    }
#else
    (void)fd;
    (void)label;
#endif
}

ssize_t send_no_sigpipe(int fd, const uint8_t *data, size_t size) {
    // The fd is O_NONBLOCK, so a send never blocks: it writes what fits and
    // returns EAGAIN when the buffer is full, letting write_all re-poll under the
    // deadline. MSG_NOSIGNAL only suppresses SIGPIPE on a closed peer.
#if defined(MSG_NOSIGNAL)
    return ::send(fd, data, size, MSG_NOSIGNAL);
#else
    return ::send(fd, data, size, 0);
#endif
}

short poll_socket(
    int fd, short events, Deadline deadline, const std::string &timeout_message, const std::string &poll_error_context
) {
    while (true) {
        double remaining = remaining_seconds(deadline, timeout_message);
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = events;
        int rc = ::poll(&pfd, 1, timeout_to_poll_ms(std::min(0.2, remaining)));
        if (rc == 0) continue;
        if (rc < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(poll_error_context + ": " + std::strerror(errno));
        }
        if ((pfd.revents & POLLNVAL) != 0) {
            throw std::runtime_error(poll_error_context + ": invalid file descriptor");
        }
        if ((pfd.revents & (events | POLLERR | POLLHUP)) != 0) {
            return pfd.revents;
        }
    }
}

int wait_for_connect(int fd, Deadline deadline, const std::string &label) {
    while (true) {
        short revents =
            poll_socket(fd, POLLOUT, deadline, label + ": connect timed out", label + ": poll(connect) failed");
        int socket_error = 0;
        socklen_t len = sizeof(socket_error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &len) != 0) {
            return errno;
        }
        if (socket_error == 0 && (revents & (POLLERR | POLLHUP)) != 0) {
            return ECONNRESET;
        }
        return socket_error;
    }
}

void validate_remote_buffer_relative_range(
    const char *op_name, const RemoteBufferHandle &handle, uint64_t offset, uint64_t size
) {
    if (handle.nbytes == 0) {
        throw std::invalid_argument(std::string(op_name) + ": handle size must be non-zero");
    }
    if (offset > handle.nbytes || size > handle.nbytes - offset) {
        throw std::out_of_range(std::string(op_name) + ": range exceeds remote buffer");
    }
}

void validate_remote_buffer_export_range(
    const char *op_name, const RemoteBufferHandle &handle, uint64_t offset, uint64_t size
) {
    if (handle.nbytes == 0) {
        throw std::invalid_argument(std::string(op_name) + ": handle size must be non-zero");
    }
    if (handle.offset > handle.nbytes) {
        throw std::invalid_argument(std::string(op_name) + ": handle offset exceeds size");
    }
    uint64_t available = handle.nbytes - handle.offset;
    if (offset > available || size > available - offset) {
        throw std::out_of_range(std::string(op_name) + ": range exceeds remote buffer");
    }
}

std::chrono::steady_clock::time_point deadline_from_now(double timeout_s) {
    return std::chrono::steady_clock::now() +
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_s));
}

int connect_tcp_socket(const std::string &host, uint16_t port, const std::string &label, Deadline deadline) {
    // The caller's absolute deadline is used verbatim for resolution and every
    // per-address connect — never re-derived as now()+remaining, which would
    // extend the hard wall if this thread is descheduled before entry.
    std::string port_s = std::to_string(port);
    std::vector<TcpAddress> addresses = resolve_tcp_addresses_with_timeout(host, port_s, label, deadline);
    int fd = -1;
    int last_errno = 0;
    for (const TcpAddress &addr : addresses) {
        (void)remaining_seconds(deadline, label + ": connect timed out");
        int candidate = ::socket(addr.family, addr.socktype, addr.protocol);
        if (candidate < 0) {
            last_errno = errno;
            continue;
        }
        try {
            configure_socket_no_sigpipe(candidate, label);
            int flags = ::fcntl(candidate, F_GETFL, 0);
            if (flags < 0) {
                last_errno = errno;
                ::close(candidate);
                continue;
            }
            if (::fcntl(candidate, F_SETFL, flags | O_NONBLOCK) != 0) {
                last_errno = errno;
                ::close(candidate);
                continue;
            }
            int rc = ::connect(candidate, reinterpret_cast<const sockaddr *>(&addr.addr), addr.addrlen);
            if (rc == 0) {
                // Keep the fd O_NONBLOCK for its whole life: all frame I/O polls
                // for readiness under a deadline, so recv/send never block (a
                // blocking send of a large frame to a stalled reader would
                // outlast the deadline, and MSG_DONTWAIT is not portable).
                fd = candidate;
                break;
            }
            if (errno != EINPROGRESS) {
                last_errno = errno;
                ::close(candidate);
                continue;
            }
            int connect_error = wait_for_connect(candidate, deadline, label);
            if (connect_error == 0) {
                fd = candidate;
                break;
            }
            last_errno = connect_error;
            ::close(candidate);
        } catch (...) {
            ::close(candidate);
            throw;
        }
    }
    if (fd < 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(label + ": connect timed out to " + host + ":" + port_s);
        }
        throw std::runtime_error(
            label + ": connect failed to " + host + ":" + port_s + ": " + std::strerror(last_errno)
        );
    }
    return fd;
}

RemoteAddressSpace decode_remote_address_space(int32_t raw, const char *field_name) {
    switch (static_cast<RemoteAddressSpace>(raw)) {
    case RemoteAddressSpace::HOST_INLINE:
    case RemoteAddressSpace::REMOTE_DEVICE:
    case RemoteAddressSpace::REMOTE_WINDOW:
    case RemoteAddressSpace::UB_LDST:
        return static_cast<RemoteAddressSpace>(raw);
    default:
        throw std::runtime_error(std::string("RemoteL3Endpoint: unknown ") + field_name);
    }
}

void validate_owner_buffer_handle(const RemoteBufferHandle &handle, size_t requested_size) {
    if (handle.buffer_id == 0) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: buffer_id must be non-zero");
    }
    if (handle.generation == 0) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: generation must be non-zero");
    }
    if (handle.import_id != 0) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: import_id must be zero");
    }
    if (handle.address_space != RemoteAddressSpace::REMOTE_DEVICE) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: owner allocation must be REMOTE_DEVICE");
    }
    if (handle.nbytes == 0 || handle.nbytes != static_cast<uint64_t>(requested_size)) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: result size mismatch");
    }
    if (handle.offset != 0) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: owner allocation offset must be zero");
    }
}

}  // namespace

std::vector<uint8_t>
RemoteL3Transport::exchange_group_task(const std::vector<uint8_t> &frame, uint64_t, int32_t, int32_t) {
    const auto decoded = remote_l3::decode_frame(frame);
    submit_frame(frame);
    return wait_for_reply(remote_l3::FrameType::COMPLETION, decoded.header.sequence);
}

RemoteL3SocketTransport::RemoteL3SocketTransport(
    std::string host, uint16_t port, std::string health_host, uint16_t health_port, double attach_timeout_s,
    double runtime_timeout_s
) :
    host_(std::move(host)),
    port_(port),
    health_host_(std::move(health_host)),
    health_port_(health_port),
    attach_timeout_s_(attach_timeout_s),
    runtime_timeout_s_(runtime_timeout_s) {
    if (host_.empty()) throw std::invalid_argument("RemoteL3SocketTransport: host must be non-empty");
    if (port_ == 0) throw std::invalid_argument("RemoteL3SocketTransport: port must be non-zero");
    if (health_host_.empty()) throw std::invalid_argument("RemoteL3SocketTransport: health host must be non-empty");
    if (health_port_ == 0) throw std::invalid_argument("RemoteL3SocketTransport: health port must be non-zero");
    if (attach_timeout_s_ <= 0.0)
        throw std::invalid_argument("RemoteL3SocketTransport: attach timeout must be positive");
    if (runtime_timeout_s_ <= 0.0)
        throw std::invalid_argument("RemoteL3SocketTransport: runtime timeout must be positive");
    // One absolute deadline for the whole attach phase; command-connect, the
    // HELLO read, and health-connect all derive their remaining from it so the
    // attach cannot exceed the caller's startup-budget slice.
    attach_deadline_ =
        std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                               std::chrono::duration<double>(attach_timeout_s_)
                                           );
    connect_socket();
}

RemoteL3SocketTransport::~RemoteL3SocketTransport() { close_socket(); }

void RemoteL3SocketTransport::connect_socket() {
    fd_ = connect_tcp_socket(host_, port_, "RemoteL3SocketTransport(command)", attach_deadline_);
}

void RemoteL3SocketTransport::close_socket() {
    stop_health_monitor();
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_RDWR);
        ::close(fd_);
        fd_ = -1;
    }
}

void RemoteL3SocketTransport::mark_health_failed(const std::string &message) {
    std::lock_guard<std::mutex> lk(health_mu_);
    if (health_failed_.load(std::memory_order_acquire)) return;
    health_error_ = message;
    health_failed_.store(true, std::memory_order_release);
}

void RemoteL3SocketTransport::check_health() {
    if (!health_failed_.load(std::memory_order_acquire)) return;
    std::string message;
    {
        std::lock_guard<std::mutex> lk(health_mu_);
        message = health_error_;
    }
    throw std::runtime_error("RemoteL3SocketTransport: health lane failed: " + message);
}

void RemoteL3SocketTransport::start_health_monitor(uint64_t session_id, int32_t worker_id) {
    if (health_thread_.joinable()) return;
    // Health-connect is the last attach-phase op, so it shares attach_deadline_.
    health_fd_ = connect_tcp_socket(health_host_, health_port_, "RemoteL3SocketTransport(health)", attach_deadline_);
    health_stop_.store(false, std::memory_order_release);
    health_failed_.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(health_mu_);
        health_error_.clear();
    }
    int fd = health_fd_;
    // The health-monitor loop is a runtime lane; its per-frame read uses the
    // runtime timeout, not the (spent) attach budget.
    double timeout_s = runtime_timeout_s_;
    health_thread_ = std::thread([this, fd, session_id, worker_id, timeout_s]() {
        auto read_exact = [&](uint8_t *data, size_t size) -> bool {
            size_t off = 0;
            auto deadline =
                std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                       std::chrono::duration<double>(timeout_s)
                                                   );
            while (off < size) {
                if (health_stop_.load(std::memory_order_acquire)) return false;
                auto now = std::chrono::steady_clock::now();
                if (now >= deadline) throw std::runtime_error("timed out waiting for HEALTH frame");
                (void)poll_socket(fd, POLLIN, deadline, "timed out waiting for HEALTH frame", "poll failed");
                ssize_t n = ::recv(fd, data + off, size - off, 0);
                if (n < 0) {
                    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
                    throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
                }
                if (n == 0) throw std::runtime_error("health socket closed");
                off += static_cast<size_t>(n);
            }
            return true;
        };

        try {
            static constexpr size_t HEADER_BYTES = 40;
            while (!health_stop_.load(std::memory_order_acquire)) {
                std::vector<uint8_t> frame(HEADER_BYTES);
                if (!read_exact(frame.data(), HEADER_BYTES)) return;
                uint32_t payload_bytes = read_le_u32(frame.data() + 32);
                if (payload_bytes > remote_l3::MAX_FRAME_PAYLOAD_BYTES) {
                    throw std::runtime_error("HEALTH payload exceeds maximum");
                }
                frame.resize(HEADER_BYTES + payload_bytes);
                if (payload_bytes != 0 && !read_exact(frame.data() + HEADER_BYTES, payload_bytes)) return;
                auto decoded = remote_l3::decode_frame(frame);
                if (decoded.header.frame_type != remote_l3::FrameType::HEALTH) {
                    throw std::runtime_error("non-HEALTH frame on health lane");
                }
                if (decoded.header.session_id != session_id || decoded.header.worker_id != worker_id) {
                    throw std::runtime_error("HEALTH session or worker mismatch");
                }
            }
        } catch (const std::exception &e) {
            if (!health_stop_.load(std::memory_order_acquire)) mark_health_failed(e.what());
        }
    });
}

void RemoteL3SocketTransport::stop_health_monitor() {
    health_stop_.store(true, std::memory_order_release);
    if (health_fd_ >= 0) {
        ::shutdown(health_fd_, SHUT_RDWR);
    }
    if (health_thread_.joinable()) {
        health_thread_.join();
    }
    if (health_fd_ >= 0) {
        ::close(health_fd_);
        health_fd_ = -1;
    }
}

void RemoteL3SocketTransport::wait_readable(std::chrono::steady_clock::time_point deadline) {
    while (true) {
        check_health();
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) throw std::runtime_error("RemoteL3SocketTransport: timed out waiting for frame");
        (void)poll_socket(
            fd_, POLLIN, deadline, "RemoteL3SocketTransport: timed out waiting for frame",
            "RemoteL3SocketTransport: poll(read) failed"
        );
        return;
    }
}

void RemoteL3SocketTransport::wait_writable(std::chrono::steady_clock::time_point deadline) {
    while (true) {
        check_health();
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) throw std::runtime_error("RemoteL3SocketTransport: timed out writing frame");
        (void)poll_socket(
            fd_, POLLOUT, deadline, "RemoteL3SocketTransport: timed out writing frame",
            "RemoteL3SocketTransport: poll(write) failed"
        );
        return;
    }
}

void RemoteL3SocketTransport::write_all(
    const uint8_t *data, size_t size, std::chrono::steady_clock::time_point deadline
) {
    size_t off = 0;
    while (off < size) {
        wait_writable(deadline);
        ssize_t n = send_no_sigpipe(fd_, data + off, size - off);
        if (n < 0) {
            // EAGAIN/EWOULDBLOCK: the buffer filled (peer not draining) — re-poll
            // under the deadline, which throws if the write outlasts it.
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            throw std::runtime_error(std::string("RemoteL3SocketTransport: send failed: ") + std::strerror(errno));
        }
        if (n == 0) throw std::runtime_error("RemoteL3SocketTransport: socket closed while writing");
        off += static_cast<size_t>(n);
    }
}

std::vector<uint8_t> RemoteL3SocketTransport::read_frame(std::chrono::steady_clock::time_point deadline) {
    static constexpr size_t HEADER_BYTES = 40;
    std::vector<uint8_t> frame(HEADER_BYTES);
    size_t off = 0;
    while (off < HEADER_BYTES) {
        wait_readable(deadline);
        ssize_t n = ::recv(fd_, frame.data() + off, HEADER_BYTES - off, 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            throw std::runtime_error(
                std::string("RemoteL3SocketTransport: recv header failed: ") + std::strerror(errno)
            );
        }
        if (n == 0) throw std::runtime_error("RemoteL3SocketTransport: socket closed while reading header");
        off += static_cast<size_t>(n);
    }
    uint32_t payload_bytes = read_le_u32(frame.data() + 32);
    if (payload_bytes > remote_l3::MAX_FRAME_PAYLOAD_BYTES) {
        throw std::runtime_error("RemoteL3SocketTransport: frame payload exceeds maximum");
    }
    frame.resize(HEADER_BYTES + payload_bytes);
    off = HEADER_BYTES;
    while (off < frame.size()) {
        wait_readable(deadline);
        ssize_t n = ::recv(fd_, frame.data() + off, frame.size() - off, 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            throw std::runtime_error(
                std::string("RemoteL3SocketTransport: recv payload failed: ") + std::strerror(errno)
            );
        }
        if (n == 0) throw std::runtime_error("RemoteL3SocketTransport: socket closed while reading payload");
        off += static_cast<size_t>(n);
    }
    return frame;
}

void RemoteL3SocketTransport::expect_hello_ready(
    uint64_t session_id, int32_t worker_id, const std::string &comm_profile
) {
    // The HELLO read is an attach-phase op: it shares attach_deadline_ so the
    // whole (possibly multi-recv) read cannot outlast the startup-budget slice.
    auto frame = remote_l3::decode_frame(read_frame(attach_deadline_));
    if (frame.header.frame_type != remote_l3::FrameType::HELLO) {
        throw std::runtime_error("RemoteL3SocketTransport: expected HELLO frame");
    }
    auto hello = remote_l3::decode_hello(frame.payload.data(), frame.payload.size());
    if (hello.session_id != session_id || hello.worker_id != worker_id) {
        throw std::runtime_error("RemoteL3SocketTransport: HELLO session or worker mismatch");
    }
    if (hello.ready_state != remote_l3::ReadyState::READY) {
        throw std::runtime_error("RemoteL3SocketTransport: HELLO did not report READY");
    }
    if (hello.comm_profile != comm_profile) {
        throw std::runtime_error("RemoteL3SocketTransport: HELLO comm profile mismatch");
    }
    start_health_monitor(session_id, worker_id);
}

void RemoteL3SocketTransport::submit_frame(const std::vector<uint8_t> &frame) {
    if (fd_ < 0) throw std::runtime_error("RemoteL3SocketTransport: socket is closed");
    // Each runtime command gets a fresh runtime-timeout budget, independent of
    // the (already-spent) attach deadline.
    write_all(frame.data(), frame.size(), deadline_from_now(runtime_timeout_s_));
}

std::vector<uint8_t> RemoteL3SocketTransport::wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) {
    auto frame_bytes = read_frame(deadline_from_now(runtime_timeout_s_));
    auto frame = remote_l3::decode_frame(frame_bytes);
    if (frame.header.frame_type != frame_type || frame.header.sequence != sequence) {
        throw std::runtime_error("RemoteL3SocketTransport: reply frame type or sequence mismatch");
    }
    return frame_bytes;
}

void RemoteL3SocketTransport::shutdown() { close_socket(); }

MpiGroupMailboxChannel::MpiGroupMailboxChannel(
    void *mailbox, size_t mailbox_bytes, int32_t world_size, int mpirun_pid, double runtime_timeout_s
) :
    mailbox_(static_cast<uint8_t *>(mailbox)),
    mailbox_bytes_(mailbox_bytes),
    world_size_(world_size),
    mpirun_pid_(mpirun_pid),
    runtime_timeout_s_(runtime_timeout_s) {
    using namespace mpi_group_mailbox;
    if (mailbox_ == nullptr) throw std::invalid_argument("MpiGroupMailboxChannel: null mailbox");
    if (mailbox_bytes_ < MAILBOX_BYTES) throw std::invalid_argument("MpiGroupMailboxChannel: mailbox is too small");
    if (world_size_ <= 0) throw std::invalid_argument("MpiGroupMailboxChannel: world_size must be positive");
    if (!(runtime_timeout_s_ > 0.0)) {
        throw std::invalid_argument("MpiGroupMailboxChannel: runtime_timeout_s must be positive");
    }
    if (std::memcmp(mailbox_ + OFF_MAGIC, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::invalid_argument("MpiGroupMailboxChannel: mailbox magic mismatch");
    }
    if (read_u32(OFF_PROTOCOL_VERSION) != PROTOCOL_VERSION || read_u32(OFF_HEADER_BYTES) != HEADER_BYTES) {
        throw std::invalid_argument("MpiGroupMailboxChannel: mailbox protocol mismatch");
    }
    if (read_u64(OFF_MAILBOX_BYTES) != MAILBOX_BYTES ||
        read_u32(OFF_WORLD_SIZE) != static_cast<uint32_t>(world_size_)) {
        throw std::invalid_argument("MpiGroupMailboxChannel: mailbox layout or world size mismatch");
    }
}

int32_t MpiGroupMailboxChannel::load_i32(size_t offset) const {
    int32_t value = 0;
    __atomic_load(reinterpret_cast<const int32_t *>(mailbox_ + offset), &value, __ATOMIC_ACQUIRE);
    return value;
}

void MpiGroupMailboxChannel::store_i32(size_t offset, int32_t value) {
    __atomic_store(reinterpret_cast<int32_t *>(mailbox_ + offset), &value, __ATOMIC_RELEASE);
}

uint32_t MpiGroupMailboxChannel::read_u32(size_t offset) const {
    uint32_t value = 0;
    std::memcpy(&value, mailbox_ + offset, sizeof(value));
    return value;
}

uint64_t MpiGroupMailboxChannel::read_u64(size_t offset) const {
    uint64_t value = 0;
    std::memcpy(&value, mailbox_ + offset, sizeof(value));
    return value;
}

void MpiGroupMailboxChannel::write_u32(size_t offset, uint32_t value) {
    std::memcpy(mailbox_ + offset, &value, sizeof(value));
}

void MpiGroupMailboxChannel::write_u64(size_t offset, uint64_t value) {
    std::memcpy(mailbox_ + offset, &value, sizeof(value));
}

std::string MpiGroupMailboxChannel::terminal_reason() const {
    using namespace mpi_group_mailbox;
    const size_t size = std::min(static_cast<size_t>(read_u32(OFF_ERROR_BYTES)), ERROR_BYTES);
    return std::string(reinterpret_cast<const char *>(mailbox_ + ERROR_OFFSET), size);
}

void MpiGroupMailboxChannel::mark_terminal(const std::string &reason) {
    using namespace mpi_group_mailbox;
    const size_t size = std::min(reason.size(), ERROR_BYTES);
    if (size > 0) std::memcpy(mailbox_ + ERROR_OFFSET, reason.data(), size);
    write_u32(OFF_ERROR_BYTES, static_cast<uint32_t>(size));
    store_i32(OFF_GROUP_STATE, static_cast<int32_t>(GroupState::TERMINAL));
    store_i32(OFF_REQUEST_STATE, static_cast<int32_t>(RequestState::TASK_FAILED));
}

void MpiGroupMailboxChannel::kill_mpirun_group() const {
    if (mpirun_pid_ <= 0) return;
    (void)::kill(-mpirun_pid_, SIGKILL);
    (void)::kill(mpirun_pid_, SIGKILL);
}

bool MpiGroupMailboxChannel::terminal() const {
    return load_i32(mpi_group_mailbox::OFF_GROUP_STATE) ==
           static_cast<int32_t>(mpi_group_mailbox::GroupState::TERMINAL);
}

std::vector<std::vector<uint8_t>> MpiGroupMailboxChannel::run_exchange(
    const std::vector<std::vector<uint8_t>> &frames, mpi_group_mailbox::Opcode opcode, mpi_group_mailbox::Target target,
    int32_t target_rank
) {
    using namespace mpi_group_mailbox;
    if (frames.empty()) throw std::invalid_argument("MpiGroupMailboxChannel: request requires a payload");
    if (target == Target::PER_RANK && frames.size() != static_cast<size_t>(world_size_)) {
        throw std::invalid_argument("MpiGroupMailboxChannel: per-rank request must include every rank");
    }
    if (target != Target::PER_RANK && frames.size() != 1) {
        throw std::invalid_argument("MpiGroupMailboxChannel: rank/group request requires one payload");
    }
    size_t encoded_request_bytes = 4 + 4 * frames.size();
    for (const auto &frame : frames) {
        if (frame.size() > UINT32_MAX || frame.size() > PAYLOAD_BYTES ||
            encoded_request_bytes > PAYLOAD_BYTES - frame.size()) {
            throw std::invalid_argument("MpiGroupMailboxChannel: request frame vector exceeds mailbox capacity");
        }
        encoded_request_bytes += frame.size();
    }
    if (target == Target::RANK && (target_rank < 0 || target_rank >= world_size_)) {
        throw std::invalid_argument("MpiGroupMailboxChannel: target rank is outside the group");
    }
    const auto group_state = static_cast<GroupState>(load_i32(OFF_GROUP_STATE));
    if (group_state == GroupState::TERMINAL) {
        throw std::runtime_error("MpiGroupMailboxChannel: group is terminal: " + terminal_reason());
    }
    if (group_state != GroupState::READY) {
        throw std::runtime_error("MpiGroupMailboxChannel: group is not ready");
    }
    if (static_cast<RequestState>(load_i32(OFF_REQUEST_STATE)) != RequestState::IDLE) {
        throw std::runtime_error("MpiGroupMailboxChannel: request lane is not idle");
    }

    const uint64_t sequence = next_sequence_++;
    const uint32_t count = static_cast<uint32_t>(frames.size());
    std::memcpy(mailbox_ + REQUEST_OFFSET, &count, sizeof(count));
    size_t request_offset = REQUEST_OFFSET + 4;
    for (const auto &frame : frames) {
        const uint32_t frame_size = static_cast<uint32_t>(frame.size());
        std::memcpy(mailbox_ + request_offset, &frame_size, sizeof(frame_size));
        request_offset += 4;
    }
    for (const auto &frame : frames) {
        if (!frame.empty()) std::memcpy(mailbox_ + request_offset, frame.data(), frame.size());
        request_offset += frame.size();
    }
    write_u64(OFF_SEQUENCE_ID, sequence);
    write_u32(OFF_OPCODE, static_cast<uint32_t>(opcode));
    write_u32(OFF_TARGET, static_cast<uint32_t>(target));
    std::memcpy(mailbox_ + OFF_TARGET_RANK, &target_rank, sizeof(target_rank));
    write_u32(OFF_REQUEST_COUNT, count);
    write_u32(OFF_REQUEST_BYTES, static_cast<uint32_t>(encoded_request_bytes));
    write_u32(OFF_RESPONSE_COUNT, 0);
    write_u32(OFF_RESPONSE_BYTES, 0);
    write_u32(OFF_ERROR_BYTES, 0);
    const RequestState ready_state =
        opcode == Opcode::SHUTDOWN ? RequestState::SHUTDOWN_READY : RequestState::REQUEST_READY;
    store_i32(OFF_REQUEST_STATE, static_cast<int32_t>(ready_state));

    const Deadline deadline = deadline_from_now(runtime_timeout_s_);
    uint32_t poll_spins = 0;
    while (true) {
        const auto state = static_cast<RequestState>(load_i32(OFF_REQUEST_STATE));
        if (state == RequestState::TASK_DONE) {
            const uint32_t response_count = read_u32(OFF_RESPONSE_COUNT);
            const size_t response_bytes = read_u32(OFF_RESPONSE_BYTES);
            const uint32_t expected_count = target == Target::PER_RANK ? static_cast<uint32_t>(world_size_) : 1U;
            const size_t prefix_bytes = 4 + 4 * static_cast<size_t>(response_count);
            if (response_count != expected_count || response_bytes < prefix_bytes || response_bytes > PAYLOAD_BYTES) {
                mark_terminal("MPI group mailbox returned an invalid response vector");
                kill_mpirun_group();
                throw std::runtime_error("MpiGroupMailboxChannel: invalid response vector");
            }
            uint32_t encoded_count = 0;
            std::memcpy(&encoded_count, mailbox_ + RESPONSE_OFFSET, sizeof(encoded_count));
            if (encoded_count != response_count) {
                mark_terminal("MPI group mailbox response vector length mismatch");
                kill_mpirun_group();
                throw std::runtime_error("MpiGroupMailboxChannel: response vector length mismatch");
            }
            std::vector<uint32_t> payload_sizes(response_count);
            size_t total_payload_bytes = 0;
            for (uint32_t i = 0; i < response_count; ++i) {
                std::memcpy(
                    &payload_sizes[i], mailbox_ + RESPONSE_OFFSET + 4 + 4 * static_cast<size_t>(i),
                    sizeof(payload_sizes[i])
                );
                total_payload_bytes += payload_sizes[i];
            }
            if (prefix_bytes + total_payload_bytes != response_bytes) {
                mark_terminal("MPI group mailbox response vector length mismatch");
                kill_mpirun_group();
                throw std::runtime_error("MpiGroupMailboxChannel: response vector length mismatch");
            }
            std::vector<std::vector<uint8_t>> responses;
            responses.reserve(response_count);
            size_t response_offset = RESPONSE_OFFSET + prefix_bytes;
            for (uint32_t payload_size : payload_sizes) {
                std::vector<uint8_t> response(payload_size);
                if (payload_size > 0) {
                    std::memcpy(response.data(), mailbox_ + response_offset, payload_size);
                }
                response_offset += payload_size;
                responses.push_back(std::move(response));
            }
            store_i32(OFF_REQUEST_STATE, static_cast<int32_t>(RequestState::IDLE));
            return responses;
        }
        if (state == RequestState::SHUTDOWN_DONE) {
            store_i32(OFF_REQUEST_STATE, static_cast<int32_t>(RequestState::IDLE));
            return {{}};
        }
        if (state == RequestState::TASK_FAILED || terminal()) {
            const std::string reason = terminal_reason();
            if (!terminal()) store_i32(OFF_REQUEST_STATE, static_cast<int32_t>(RequestState::IDLE));
            throw std::runtime_error(
                "MpiGroupMailboxChannel: MPI group request failed" + (reason.empty() ? std::string() : ": " + reason)
            );
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            mark_terminal("MPI group mailbox request timed out at sequence " + std::to_string(sequence));
            kill_mpirun_group();
            throw std::runtime_error("MpiGroupMailboxChannel: request timed out");
        }
        if (++poll_spins >= 64) {
            std::this_thread::yield();
            poll_spins = 0;
        }
    }
}

std::vector<uint8_t> MpiGroupMailboxChannel::exchange(const std::vector<uint8_t> &frame, int32_t target_rank) {
    std::lock_guard<std::mutex> lock(lane_mu_);
    const auto decoded = remote_l3::decode_frame(frame);
    mpi_group_mailbox::Opcode opcode = mpi_group_mailbox::Opcode::CONTROL;
    mpi_group_mailbox::Target target = mpi_group_mailbox::Target::RANK;
    if (decoded.header.frame_type == remote_l3::FrameType::TASK) {
        opcode = mpi_group_mailbox::Opcode::TASK;
    } else if (decoded.header.frame_type == remote_l3::FrameType::HEALTH) {
        opcode = mpi_group_mailbox::Opcode::PING;
        target = mpi_group_mailbox::Target::GROUP;
        target_rank = -1;
    } else if (decoded.header.frame_type == remote_l3::FrameType::CONTROL) {
        const auto control = remote_l3::decode_control(decoded.payload.data(), decoded.payload.size());
        if (control.control_name == remote_l3::ControlName::ALLOC_DOMAIN ||
            control.control_name == remote_l3::ControlName::RELEASE_DOMAIN) {
            target = mpi_group_mailbox::Target::GROUP;
            target_rank = -1;
        }
    } else {
        throw std::runtime_error("MpiGroupMailboxChannel: unsupported request frame type");
    }
    auto responses = run_exchange({frame}, opcode, target, target_rank);
    return std::move(responses.front());
}

std::vector<uint8_t> MpiGroupMailboxChannel::exchange_group_task(
    const std::vector<uint8_t> &frame, int32_t target_rank, uint64_t task_slot, int32_t group_size
) {
    if (group_size != world_size_) {
        // Existing submit_next_level_group permits subsets. Keep that behavior
        // as ordered rank-targeted requests; only a full MPI group is batched
        // into one PER_RANK mailbox envelope.
        return exchange(frame, target_rank);
    }
    if (target_rank < 0 || target_rank >= world_size_) {
        throw std::invalid_argument("MpiGroupMailboxChannel: group task target rank is outside the group");
    }

    const Deadline deadline = deadline_from_now(runtime_timeout_s_);
    bool leader = false;
    std::vector<std::vector<uint8_t>> frames;
    {
        std::unique_lock<std::mutex> lock(group_mu_);
        while (group_active_ && group_task_slot_ != task_slot) {
            if (group_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                lock.unlock();
                mark_terminal("MPI group task batching timed out waiting for the prior task");
                kill_mpirun_group();
                throw std::runtime_error("MpiGroupMailboxChannel: group task batching timed out");
            }
        }
        if (!group_active_) {
            group_active_ = true;
            group_done_ = false;
            group_task_slot_ = task_slot;
            group_arrived_ = 0;
            group_departed_ = 0;
            group_frames_.assign(static_cast<size_t>(world_size_), {});
            group_replies_.clear();
            group_error_ = nullptr;
        }
        auto &rank_frame = group_frames_[static_cast<size_t>(target_rank)];
        if (!rank_frame.empty()) {
            throw std::runtime_error("MpiGroupMailboxChannel: duplicate rank in one MPI group task");
        }
        rank_frame = frame;
        ++group_arrived_;
        if (group_arrived_ == world_size_) {
            leader = true;
            frames = group_frames_;
        } else {
            while (!group_done_) {
                if (group_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                    group_error_ = std::make_exception_ptr(std::runtime_error("MPI group task batching timed out"));
                    group_done_ = true;
                    ++group_departed_;
                    lock.unlock();
                    mark_terminal("MPI group task batching timed out waiting for all rank payloads");
                    kill_mpirun_group();
                    group_cv_.notify_all();
                    throw std::runtime_error("MpiGroupMailboxChannel: group task batching timed out");
                }
            }
        }
    }

    if (leader) {
        try {
            std::lock_guard<std::mutex> lane_lock(lane_mu_);
            auto replies =
                run_exchange(frames, mpi_group_mailbox::Opcode::TASK, mpi_group_mailbox::Target::PER_RANK, -1);
            std::lock_guard<std::mutex> group_lock(group_mu_);
            group_replies_ = std::move(replies);
            group_done_ = true;
        } catch (...) {
            std::lock_guard<std::mutex> group_lock(group_mu_);
            group_error_ = std::current_exception();
            group_done_ = true;
        }
        group_cv_.notify_all();
    }

    std::unique_lock<std::mutex> lock(group_mu_);
    while (!group_done_) {
        if (group_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            group_error_ = std::make_exception_ptr(std::runtime_error("MPI group task dispatch timed out"));
            group_done_ = true;
            ++group_departed_;
            lock.unlock();
            mark_terminal("MPI group task dispatch timed out");
            kill_mpirun_group();
            group_cv_.notify_all();
            throw std::runtime_error("MpiGroupMailboxChannel: group task dispatch timed out");
        }
    }
    std::exception_ptr error = group_error_;
    std::vector<uint8_t> reply;
    if (error == nullptr) {
        if (group_replies_.size() != static_cast<size_t>(world_size_)) {
            error = std::make_exception_ptr(
                std::runtime_error("MpiGroupMailboxChannel: MPI group task reply count mismatch")
            );
        } else {
            reply = group_replies_[static_cast<size_t>(target_rank)];
        }
    }
    ++group_departed_;
    if (group_departed_ == world_size_) {
        group_active_ = false;
        group_done_ = false;
        group_frames_.clear();
        group_replies_.clear();
        group_error_ = nullptr;
        group_cv_.notify_all();
    }
    lock.unlock();
    if (error != nullptr) std::rethrow_exception(error);
    return reply;
}

void MpiGroupMailboxChannel::shutdown(const std::vector<uint8_t> &frame) {
    std::lock_guard<std::mutex> lock(lane_mu_);
    if (shutdown_sent_ || terminal()) return;
    shutdown_sent_ = true;
    (void)run_exchange({frame}, mpi_group_mailbox::Opcode::SHUTDOWN, mpi_group_mailbox::Target::GROUP, -1);
}

MpiGroupMailboxTransport::MpiGroupMailboxTransport(
    std::shared_ptr<MpiGroupMailboxChannel> channel, int32_t target_rank
) :
    channel_(std::move(channel)),
    target_rank_(target_rank) {
    if (!channel_) throw std::invalid_argument("MpiGroupMailboxTransport: null channel");
    if (target_rank_ < 0) throw std::invalid_argument("MpiGroupMailboxTransport: target rank must be non-negative");
}

void MpiGroupMailboxTransport::submit_frame(const std::vector<uint8_t> &frame) {
    if (pending_) throw std::runtime_error("MpiGroupMailboxTransport: a request is already pending");
    pending_frame_ = frame;
    pending_ = true;
}

std::vector<uint8_t> MpiGroupMailboxTransport::wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) {
    if (!pending_) throw std::runtime_error("MpiGroupMailboxTransport: no request is pending");
    auto frame = std::move(pending_frame_);
    pending_frame_.clear();
    pending_ = false;
    auto reply = channel_->exchange(frame, target_rank_);
    auto decoded = remote_l3::decode_frame(reply);
    if (decoded.header.frame_type != frame_type || decoded.header.sequence != sequence) {
        throw std::runtime_error("MpiGroupMailboxTransport: reply frame type or sequence mismatch");
    }
    return reply;
}

std::vector<uint8_t> MpiGroupMailboxTransport::exchange_group_task(
    const std::vector<uint8_t> &frame, uint64_t task_slot, int32_t, int32_t group_size
) {
    const auto request = remote_l3::decode_frame(frame);
    auto reply = channel_->exchange_group_task(frame, target_rank_, task_slot, group_size);
    const auto decoded = remote_l3::decode_frame(reply);
    if (decoded.header.frame_type != remote_l3::FrameType::COMPLETION ||
        decoded.header.sequence != request.header.sequence) {
        throw std::runtime_error("MpiGroupMailboxTransport: group task reply frame type or sequence mismatch");
    }
    return reply;
}

void MpiGroupMailboxTransport::shutdown() {
    if (!pending_) return;
    auto frame = std::move(pending_frame_);
    pending_frame_.clear();
    pending_ = false;
    channel_->shutdown(frame);
}

RemoteL3Endpoint::RemoteL3Endpoint(
    int32_t worker_id, uint64_t session_id, std::string transport_name, std::unique_ptr<RemoteL3Transport> transport,
    WorkerEndpointKind endpoint_kind
) :
    session_id_(session_id),
    transport_(std::move(transport)) {
    if (worker_id < 0) throw std::invalid_argument("RemoteL3Endpoint: worker_id must be non-negative");
    if (session_id == 0) throw std::invalid_argument("RemoteL3Endpoint: session_id must be non-zero");
    if (!transport_) throw std::invalid_argument("RemoteL3Endpoint: null transport");
    caps_.kind = endpoint_kind;
    caps_.worker_id = worker_id;
    caps_.remote = true;
    caps_.supports_task_dispatch = true;
    caps_.supports_control = true;
    caps_.transport = std::move(transport_name);
}

MpiGroupMailboxEndpoint::MpiGroupMailboxEndpoint(
    int32_t worker_id, uint64_t session_id, int32_t rank, std::shared_ptr<MpiGroupMailboxChannel> channel
) :
    RemoteL3Endpoint(
        worker_id, session_id, "mpi-group-mailbox",
        std::make_unique<MpiGroupMailboxTransport>(std::move(channel), rank), WorkerEndpointKind::MPI_GROUP_MAILBOX
    ) {}

remote_l3::TaskPayloadWire RemoteL3Endpoint::build_task_payload(const TaskSlotState &slot, int32_t group_index) const {
    remote_l3::TaskPayloadWire payload;
    payload.callable_digest = slot.callable.digest;
    payload.config = slot.config;

    TaskArgsView view = slot.args_view(group_index);
    const RemoteTaskArgsSidecar &sidecar = slot.remote_sidecar_for(group_index);
    if (!sidecar.tensors.empty() && sidecar.tensors.size() != static_cast<size_t>(view.tensor_count)) {
        throw std::runtime_error("RemoteL3Endpoint::run: remote sidecar tensor count does not match TaskArgs");
    }
    payload.args.inline_payload = sidecar.inline_payload;
    payload.args.tensor_metadata.reserve(static_cast<size_t>(view.tensor_count));
    payload.args.remote_desc.reserve(static_cast<size_t>(view.tensor_count));

    for (int32_t i = 0; i < view.tensor_count; ++i) {
        Tensor tensor = view.tensors(i);
        RemoteTensorSidecar tensor_sidecar{};
        if (!sidecar.tensors.empty()) tensor_sidecar = sidecar.tensors[static_cast<size_t>(i)];
        if (tensor.buffer.addr != 0 && !tensor_sidecar.present) {
            throw std::runtime_error("RemoteL3Endpoint::run: bare host pointer submitted without remote sidecar");
        }
        if (tensor.is_child_memory() && !tensor_sidecar.present) {
            throw std::runtime_error("RemoteL3Endpoint::run: child-memory tensor submitted without remote sidecar");
        }
        if (!tensor_sidecar.present && tensor.nbytes() != 0) {
            throw std::runtime_error("RemoteL3Endpoint::run: tensor payload submitted without remote sidecar");
        }
        tensor.buffer.addr = 0;
        payload.args.tensor_metadata.push_back(tensor);
        payload.args.remote_desc.push_back(tensor_sidecar);
    }
    payload.args.scalars.reserve(static_cast<size_t>(view.scalar_count));
    for (int32_t i = 0; i < view.scalar_count; ++i)
        payload.args.scalars.push_back(view.scalars[i]);
    return payload;
}

WorkerCompletion RemoteL3Endpoint::run(Ring *ring, const WorkerDispatch &dispatch) {
    if (ring == nullptr) throw std::invalid_argument("RemoteL3Endpoint::run: null ring");
    TaskSlotState &slot = *ring->slot_state(dispatch.task_slot);

    WorkerCompletion completion;
    completion.task_slot = dispatch.task_slot;
    completion.group_index = dispatch.group_index;

    uint64_t sequence = 0;
    std::unique_lock<std::mutex> command_lk(command_mu_);
    try {
        sequence = command_lane_.begin_command();
        auto payload = remote_l3::encode_task_payload(build_task_payload(slot, dispatch.group_index));
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::TASK;
        header.session_id = session_id_;
        header.worker_id = caps_.worker_id;
        header.sequence = sequence;
        auto frame = remote_l3::encode_frame(header, payload);
        std::vector<uint8_t> reply_bytes;
        if (slot.is_group() && transport_->supports_group_batch()) {
            reply_bytes = transport_->exchange_group_task(
                frame, static_cast<uint64_t>(dispatch.task_slot), dispatch.group_index, slot.group_size()
            );
        } else {
            transport_->submit_frame(frame);
            reply_bytes = transport_->wait_for_reply(remote_l3::FrameType::COMPLETION, sequence);
        }
        auto reply = remote_l3::decode_frame(reply_bytes);
        if (reply.header.frame_type != remote_l3::FrameType::COMPLETION) {
            throw std::runtime_error("RemoteL3Endpoint::run: expected COMPLETION reply");
        }
        if (reply.header.session_id != session_id_ || reply.header.worker_id != caps_.worker_id) {
            throw std::runtime_error("RemoteL3Endpoint::run: completion session or worker mismatch");
        }
        auto decoded = remote_l3::decode_completion(reply.payload.data(), reply.payload.size(), sequence);
        command_lane_.finish_reply(sequence);

        if (decoded.error_code == 0) {
            completion.outcome = EndpointOutcome::SUCCESS;
        } else {
            completion.outcome = EndpointOutcome::TASK_FAILURE;
            completion.error_message = decoded.error_message;
        }
    } catch (const std::exception &e) {
        if (sequence != 0 && command_lane_.in_flight()) {
            try {
                command_lane_.finish_reply(sequence);
            } catch (...) {}
        }
        completion.outcome = EndpointOutcome::ENDPOINT_FAILURE;
        completion.error_message =
            std::string("RemoteL3Endpoint::run(worker_id=") + std::to_string(caps_.worker_id) + "): " + e.what();
    }
    return completion;
}

remote_l3::ControlReplyPayload
RemoteL3Endpoint::run_control(remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes) {
    std::unique_lock<std::mutex> command_lk(command_mu_);
    uint64_t sequence = 0;
    try {
        sequence = command_lane_.begin_command();
        remote_l3::ControlPayload control;
        control.control_name = control_name;
        control.control_version = 1;
        control.command_bytes = command_bytes;
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::CONTROL;
        header.session_id = session_id_;
        header.worker_id = caps_.worker_id;
        header.sequence = sequence;
        transport_->submit_frame(remote_l3::encode_frame(header, remote_l3::encode_control(control)));

        auto reply_bytes = transport_->wait_for_reply(remote_l3::FrameType::CONTROL_REPLY, sequence);
        auto reply = remote_l3::decode_frame(reply_bytes);
        if (reply.header.session_id != session_id_ || reply.header.worker_id != caps_.worker_id) {
            throw std::runtime_error("RemoteL3Endpoint::control: reply session or worker mismatch");
        }
        auto decoded =
            remote_l3::decode_control_reply(reply.payload.data(), reply.payload.size(), sequence, control_name, 1);
        command_lane_.finish_reply(sequence);
        if (decoded.error_code != 0) {
            throw std::runtime_error(decoded.error_message);
        }
        return decoded;
    } catch (...) {
        if (sequence != 0 && command_lane_.in_flight()) {
            try {
                command_lane_.finish_reply(sequence);
            } catch (...) {}
        }
        throw;
    }
}

void RemoteL3Endpoint::control_remote_prepare_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
    const void *payload, size_t payload_size
) {
    if (payload == nullptr && payload_size != 0) {
        throw std::invalid_argument("RemoteL3Endpoint::control_remote_prepare_register: null payload");
    }
    std::vector<uint8_t> bytes;
    const auto *payload_bytes = static_cast<const uint8_t *>(payload);
    if (payload_size > 0) bytes.assign(payload_bytes, payload_bytes + payload_size);
    run_control(
        remote_l3::ControlName::PREPARE_REGISTER_CALLABLE,
        remote_l3::encode_register_callable_command(target_registry, callable_kind, digest_array(digest), 1, bytes)
    );
}

void RemoteL3Endpoint::control_prepare(const uint8_t *digest) {
    run_control(
        remote_l3::ControlName::PREPARE_CALLABLE,
        remote_l3::encode_digest_callable_command(
            remote_l3::RemoteRegistryTarget::REMOTE_TASK_DISPATCHER, CallableKind::PYTHON_IMPORT, digest_array(digest)
        )
    );
}

void RemoteL3Endpoint::control_remote_commit_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    run_control(
        remote_l3::ControlName::COMMIT_REGISTER_CALLABLE,
        remote_l3::encode_digest_callable_command(target_registry, callable_kind, digest_array(digest))
    );
}

void RemoteL3Endpoint::control_remote_abort_register(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    run_control(
        remote_l3::ControlName::ABORT_REGISTER_CALLABLE,
        remote_l3::encode_digest_callable_command(target_registry, callable_kind, digest_array(digest))
    );
}

void RemoteL3Endpoint::control_remote_unregister(
    remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
) {
    run_control(
        remote_l3::ControlName::UNREGISTER_CALLABLE,
        remote_l3::encode_digest_callable_command(target_registry, callable_kind, digest_array(digest))
    );
}

RemoteBufferHandle RemoteL3Endpoint::control_remote_malloc(size_t size) {
    if (size == 0) throw std::invalid_argument("RemoteL3Endpoint::control_remote_malloc: size must be non-zero");
    std::vector<uint8_t> command;
    put_u64(command, static_cast<uint64_t>(size));
    auto reply = run_control(remote_l3::ControlName::ALLOC_REMOTE_BUFFER, command);
    size_t offset = 0;
    RemoteBufferHandle handle;
    handle.worker_id = get_i32(reply.result_bytes, offset);
    handle.owner_worker_id = handle.worker_id;
    handle.buffer_id = get_u64(reply.result_bytes, offset);
    handle.generation = get_u64(reply.result_bytes, offset);
    handle.import_id = 0;
    handle.address_space =
        decode_remote_address_space(get_i32(reply.result_bytes, offset), "ALLOC_REMOTE_BUFFER address_space");
    handle.nbytes = get_u64(reply.result_bytes, offset);
    handle.offset = 0;
    handle.remote_addr = get_u64(reply.result_bytes, offset);
    handle.rkey_or_token = get_u64(reply.result_bytes, offset);
    handle.ub_ldst_va = get_u64(reply.result_bytes, offset);
    handle.access_flags = remote_l3::REMOTE_BUFFER_ACCESS_READ_WRITE;
    if (handle.worker_id != caps_.worker_id) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: worker mismatch in result");
    }
    if (offset != reply.result_bytes.size()) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_malloc: trailing bytes in result");
    }
    validate_owner_buffer_handle(handle, size);
    return handle;
}

void RemoteL3Endpoint::control_remote_free(const RemoteBufferHandle &handle) {
    std::vector<uint8_t> command;
    put_i32(command, handle.worker_id);
    put_u64(command, handle.buffer_id);
    put_u64(command, handle.generation);
    run_control(remote_l3::ControlName::FREE_REMOTE_BUFFER, command);
}

void RemoteL3Endpoint::control_remote_copy_to(
    const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size
) {
    if (src == nullptr && size != 0) throw std::invalid_argument("control_remote_copy_to: null src");
    validate_remote_buffer_relative_range("control_remote_copy_to", handle, offset, static_cast<uint64_t>(size));
    std::vector<uint8_t> command;
    put_i32(command, handle.worker_id);
    put_u64(command, handle.buffer_id);
    put_u64(command, handle.generation);
    put_u64(command, offset);
    put_u64(command, static_cast<uint64_t>(size));
    const auto *bytes = static_cast<const uint8_t *>(src);
    if (size > 0) command.insert(command.end(), bytes, bytes + size);
    run_control(remote_l3::ControlName::COPY_TO_REMOTE, command);
}

void RemoteL3Endpoint::control_remote_copy_from(
    void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size
) {
    if (dst == nullptr && size != 0) throw std::invalid_argument("control_remote_copy_from: null dst");
    validate_remote_buffer_relative_range("control_remote_copy_from", handle, offset, static_cast<uint64_t>(size));
    std::vector<uint8_t> command;
    put_i32(command, handle.worker_id);
    put_u64(command, handle.buffer_id);
    put_u64(command, handle.generation);
    put_u64(command, offset);
    put_u64(command, static_cast<uint64_t>(size));
    auto reply = run_control(remote_l3::ControlName::COPY_FROM_REMOTE, command);
    if (reply.result_bytes.size() != size) {
        throw std::runtime_error("control_remote_copy_from: result size mismatch");
    }
    if (size > 0) std::memcpy(dst, reply.result_bytes.data(), size);
}

RemoteBufferExport RemoteL3Endpoint::control_remote_export(
    const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
    const std::string &transport_profile
) {
    validate_remote_buffer_export_range("RemoteL3Endpoint::control_remote_export", handle, offset, size);
    const int32_t owner_worker_id = handle.owner_worker_id >= 0 ? handle.owner_worker_id : handle.worker_id;
    if (owner_worker_id != caps_.worker_id) {
        throw std::invalid_argument("RemoteL3Endpoint::control_remote_export: worker is not the owner");
    }
    remote_l3::ExportBufferRequest request;
    request.owner_worker_id = owner_worker_id;
    request.buffer_id = handle.buffer_id;
    request.generation = handle.generation;
    request.offset = handle.offset + offset;
    request.nbytes = size;
    request.access_flags = access_flags;
    request.transport_profile = transport_profile;
    auto reply = run_control(remote_l3::ControlName::EXPORT_BUFFER, remote_l3::encode_export_buffer_request(request));
    auto result = remote_l3::decode_export_buffer_result(reply.result_bytes.data(), reply.result_bytes.size());
    if (result.owner_worker_id != owner_worker_id) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_export: owner worker mismatch in result");
    }
    return result;
}

RemoteBufferHandle RemoteL3Endpoint::control_remote_import(
    int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
) {
    if (importer_worker_id != caps_.worker_id) {
        throw std::invalid_argument("RemoteL3Endpoint::control_remote_import: worker is not the importer");
    }
    remote_l3::ImportBufferRequest request;
    request.importer_worker_id = importer_worker_id;
    request.requested_access_flags = requested_access_flags;
    request.export_desc = export_desc;
    auto reply = run_control(remote_l3::ControlName::IMPORT_BUFFER, remote_l3::encode_import_buffer_request(request));
    auto handle = remote_l3::decode_import_buffer_result(reply.result_bytes.data(), reply.result_bytes.size());
    if (handle.worker_id != importer_worker_id) {
        throw std::runtime_error("RemoteL3Endpoint::control_remote_import: importer worker mismatch in result");
    }
    return handle;
}

void RemoteL3Endpoint::control_remote_release_import(const RemoteBufferHandle &handle) {
    if (handle.worker_id != caps_.worker_id) {
        throw std::invalid_argument("RemoteL3Endpoint::control_remote_release_import: worker is not the importer");
    }
    remote_l3::ReleaseImportRequest request;
    request.importer_worker_id = handle.worker_id;
    request.owner_worker_id = handle.owner_worker_id;
    request.buffer_id = handle.buffer_id;
    request.generation = handle.generation;
    request.import_id = handle.import_id;
    run_control(remote_l3::ControlName::RELEASE_IMPORT, remote_l3::encode_release_import_request(request));
}

std::vector<uint8_t> RemoteL3Endpoint::control_remote_domain(
    remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes
) {
    auto reply = run_control(control_name, command_bytes);
    return std::move(reply.result_bytes);
}

void RemoteL3Endpoint::shutdown_child() {
    if (!transport_) return;
    try {
        std::lock_guard<std::mutex> command_lk(command_mu_);
        uint64_t sequence = command_lane_.begin_command();
        remote_l3::FrameHeader header;
        header.frame_type = remote_l3::FrameType::SHUTDOWN;
        header.session_id = session_id_;
        header.worker_id = caps_.worker_id;
        header.sequence = sequence;
        transport_->submit_frame(remote_l3::encode_frame(header, {}));
        command_lane_.finish_reply(sequence);
        transport_->shutdown();
    } catch (...) {}
}
