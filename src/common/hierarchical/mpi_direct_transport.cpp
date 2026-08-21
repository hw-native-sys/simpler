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

#include "mpi_direct_transport.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

std::chrono::steady_clock::time_point deadline_from_now(double timeout_s) {
    return std::chrono::steady_clock::now() +
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_s));
}

MpiDirectTag outbound_tag(remote_l3::FrameType type) {
    switch (type) {
    case remote_l3::FrameType::TASK:
    case remote_l3::FrameType::CONTROL:
        return MpiDirectTag::COMMAND_REQUEST;
    case remote_l3::FrameType::SHUTDOWN:
        return MpiDirectTag::LIFECYCLE;
    default:
        throw std::runtime_error("MpiDirectTransport: invalid outbound SLR3 frame type");
    }
}

MpiDirectTag inbound_tag(remote_l3::FrameType type) {
    switch (type) {
    case remote_l3::FrameType::COMPLETION:
    case remote_l3::FrameType::CONTROL_REPLY:
        return MpiDirectTag::COMMAND_REPLY;
    case remote_l3::FrameType::HEALTH:
        return MpiDirectTag::HEALTH;
    case remote_l3::FrameType::HELLO:
        return MpiDirectTag::LIFECYCLE;
    default:
        throw std::runtime_error("MpiDirectTransportHub: invalid inbound SLR3 frame type");
    }
}

}  // namespace

MpiDirectTransportHub::MpiDirectTransportHub(size_t max_pending_frame_bytes) :
    max_pending_frame_bytes_(max_pending_frame_bytes) {
    const size_t max_frame_bytes = remote_l3::FRAME_HEADER_BYTES + remote_l3::MAX_FRAME_PAYLOAD_BYTES;
    if (max_pending_frame_bytes_ < max_frame_bytes) {
        throw std::invalid_argument("MpiDirectTransportHub: pending byte budget must fit one maximum SLR3 frame");
    }
}

void MpiDirectTransportHub::register_route(
    int32_t worker_id, int32_t mpi_rank, uint64_t session_id, const std::string &comm_profile
) {
    if (worker_id < 0) throw std::invalid_argument("MpiDirectTransportHub: worker id must be non-negative");
    if (mpi_rank <= 0) throw std::invalid_argument("MpiDirectTransportHub: executor rank must be positive");
    if (session_id == 0) throw std::invalid_argument("MpiDirectTransportHub: session id must be non-zero");
    if (comm_profile.empty()) throw std::invalid_argument("MpiDirectTransportHub: comm profile must be non-empty");

    std::lock_guard<std::mutex> lk(mu_);
    throw_if_terminal_locked();
    if (routes_by_worker_.count(worker_id) != 0 || worker_by_rank_.count(mpi_rank) != 0) {
        throw std::invalid_argument("MpiDirectTransportHub: duplicate worker id or MPI rank");
    }
    Route route;
    route.worker_id = worker_id;
    route.mpi_rank = mpi_rank;
    route.session_id = session_id;
    route.comm_profile = comm_profile;
    routes_by_worker_.emplace(worker_id, std::move(route));
    worker_by_rank_.emplace(mpi_rank, worker_id);
}

void MpiDirectTransportHub::throw_if_terminal_locked() const {
    if (!terminal_error_.empty()) throw std::runtime_error("MpiDirectTransportHub: " + terminal_error_);
    if (closed_) throw std::runtime_error("MpiDirectTransportHub: closed");
}

void MpiDirectTransportHub::throw_if_route_terminal_locked(const Route &route) const {
    throw_if_terminal_locked();
    if (!route.terminal_error.empty()) {
        throw std::runtime_error(
            "MpiDirectTransportHub: route worker " + std::to_string(route.worker_id) + ": " + route.terminal_error
        );
    }
}

void MpiDirectTransportHub::fail_locked(const std::string &message) {
    if (terminal_error_.empty()) terminal_error_ = message.empty() ? "terminal failure" : message;
    cv_.notify_all();
}

void MpiDirectTransportHub::fail(const std::string &message) {
    std::lock_guard<std::mutex> lk(mu_);
    fail_locked(message);
}

void MpiDirectTransportHub::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    cv_.notify_all();
}

void MpiDirectTransportHub::cancel_route(int32_t worker_id, const std::string &message) {
    std::lock_guard<std::mutex> lk(mu_);
    auto route_it = routes_by_worker_.find(worker_id);
    if (route_it == routes_by_worker_.end()) throw std::invalid_argument("MpiDirectTransportHub: unknown worker id");
    if (route_it->second.terminal_error.empty()) {
        route_it->second.terminal_error = message.empty() ? "cancelled" : message;
    }
    cv_.notify_all();
}

void MpiDirectTransportHub::enqueue(
    int32_t worker_id, MpiDirectTag tag, const std::vector<uint8_t> &frame, double timeout_s
) {
    if (!(timeout_s > 0.0) || !std::isfinite(timeout_s)) {
        throw std::invalid_argument("MpiDirectTransportHub: timeout must be positive and finite");
    }
    if (frame.size() > remote_l3::FRAME_HEADER_BYTES + remote_l3::MAX_FRAME_PAYLOAD_BYTES) {
        throw std::invalid_argument("MpiDirectTransportHub: frame exceeds maximum");
    }
    auto deadline = deadline_from_now(timeout_s);
    std::unique_lock<std::mutex> lk(mu_);
    auto route_it = routes_by_worker_.find(worker_id);
    if (route_it == routes_by_worker_.end()) throw std::invalid_argument("MpiDirectTransportHub: unknown worker id");
    while (frame.size() > max_pending_frame_bytes_ - pending_frame_bytes_) {
        throw_if_route_terminal_locked(route_it->second);
        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            fail_locked("timed out waiting for outbound frame credit");
            throw_if_terminal_locked();
        }
    }
    throw_if_route_terminal_locked(route_it->second);
    MpiDirectOutboundFrame outbound;
    outbound.ticket = next_ticket_++;
    outbound.target_rank = route_it->second.mpi_rank;
    outbound.tag = tag;
    outbound.frame = frame;
    pending_frame_bytes_ += frame.size();
    outbound_.push_back(std::move(outbound));
    cv_.notify_all();
}

std::optional<std::vector<uint8_t>>
MpiDirectTransportHub::poll_inbound(int32_t worker_id, remote_l3::FrameType frame_type, uint64_t sequence) {
    std::lock_guard<std::mutex> lk(mu_);
    auto route_it = routes_by_worker_.find(worker_id);
    if (route_it == routes_by_worker_.end()) throw std::invalid_argument("MpiDirectTransportHub: unknown worker id");
    throw_if_route_terminal_locked(route_it->second);
    std::deque<std::vector<uint8_t>> &queue =
        frame_type == remote_l3::FrameType::HELLO ? route_it->second.lifecycle : route_it->second.replies;
    if (queue.empty()) return std::nullopt;
    std::vector<uint8_t> frame = std::move(queue.front());
    queue.pop_front();
    auto decoded = remote_l3::decode_frame(frame);
    if (decoded.header.frame_type != frame_type || decoded.header.sequence != sequence) {
        fail_locked("inbound frame type or sequence mismatch");
        throw_if_terminal_locked();
    }
    return frame;
}

std::optional<MpiDirectOutboundFrame> MpiDirectTransportHub::poll_outbound(double timeout_s) {
    if (timeout_s < 0.0 || !std::isfinite(timeout_s)) {
        throw std::invalid_argument("MpiDirectTransportHub: poll timeout must be finite and non-negative");
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (outbound_.empty() && timeout_s > 0.0 && terminal_error_.empty() && !closed_) {
        cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [this] {
            return !outbound_.empty() || !terminal_error_.empty() || closed_;
        });
    }
    if (outbound_.empty()) {
        throw_if_terminal_locked();
        return std::nullopt;
    }
    MpiDirectOutboundFrame frame = std::move(outbound_.front());
    outbound_.pop_front();
    in_flight_bytes_.emplace(frame.ticket, frame.frame.size());
    return frame;
}

void MpiDirectTransportHub::complete_outbound(uint64_t ticket) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = in_flight_bytes_.find(ticket);
    if (it == in_flight_bytes_.end()) {
        fail_locked("unknown or already completed outbound ticket");
        throw_if_terminal_locked();
    }
    pending_frame_bytes_ -= it->second;
    in_flight_bytes_.erase(it);
    cv_.notify_all();
}

void MpiDirectTransportHub::deliver(int32_t source_rank, MpiDirectTag tag, const std::vector<uint8_t> &frame) {
    remote_l3::DecodedFrame decoded;
    try {
        decoded = remote_l3::decode_frame(frame);
    } catch (const std::exception &e) {
        fail(std::string("invalid inbound SLR3 frame: ") + e.what());
        throw;
    }

    std::lock_guard<std::mutex> lk(mu_);
    throw_if_terminal_locked();
    auto worker_it = worker_by_rank_.find(source_rank);
    if (worker_it == worker_by_rank_.end()) {
        fail_locked("frame arrived from an unknown MPI rank");
        throw_if_terminal_locked();
    }
    Route &route = routes_by_worker_.at(worker_it->second);
    if (!route.terminal_error.empty()) return;
    if (decoded.header.worker_id != route.worker_id || decoded.header.session_id != route.session_id) {
        fail_locked("inbound frame identity does not match manifest route");
        throw_if_terminal_locked();
    }
    MpiDirectTag expected;
    try {
        expected = inbound_tag(decoded.header.frame_type);
    } catch (const std::exception &e) {
        fail_locked(e.what());
        throw_if_terminal_locked();
    }
    if (tag != expected) {
        fail_locked("inbound MPI tag does not match SLR3 frame type");
        throw_if_terminal_locked();
    }
    if (tag == MpiDirectTag::HEALTH) {
        if (!decoded.payload.empty()) {
            fail_locked("HEALTH frame payload must be empty");
            throw_if_terminal_locked();
        }
        route.last_health = std::chrono::steady_clock::now();
    } else if (tag == MpiDirectTag::LIFECYCLE) {
        route.lifecycle.push_back(frame);
    } else {
        route.replies.push_back(frame);
    }
    cv_.notify_all();
}

std::vector<uint8_t> MpiDirectTransportHub::wait_inbound(
    int32_t worker_id, remote_l3::FrameType frame_type, uint64_t sequence, double timeout_s
) {
    auto deadline = deadline_from_now(timeout_s);
    std::unique_lock<std::mutex> lk(mu_);
    auto route_it = routes_by_worker_.find(worker_id);
    if (route_it == routes_by_worker_.end()) throw std::invalid_argument("MpiDirectTransportHub: unknown worker id");
    std::deque<std::vector<uint8_t>> &queue =
        frame_type == remote_l3::FrameType::HELLO ? route_it->second.lifecycle : route_it->second.replies;
    while (queue.empty()) {
        throw_if_route_terminal_locked(route_it->second);
        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            fail_locked("timed out waiting for inbound frame");
            throw_if_terminal_locked();
        }
    }
    throw_if_route_terminal_locked(route_it->second);
    std::vector<uint8_t> frame = std::move(queue.front());
    queue.pop_front();
    auto decoded = remote_l3::decode_frame(frame);
    if (decoded.header.frame_type != frame_type || decoded.header.sequence != sequence) {
        fail_locked("inbound frame type or sequence mismatch");
        throw_if_terminal_locked();
    }
    return frame;
}

void MpiDirectTransportHub::expect_hello_ready(int32_t worker_id, double timeout_s) {
    auto bytes = wait_inbound(worker_id, remote_l3::FrameType::HELLO, 0, timeout_s);
    auto frame = remote_l3::decode_frame(bytes);
    auto hello = remote_l3::decode_hello(frame.payload.data(), frame.payload.size());
    std::lock_guard<std::mutex> lk(mu_);
    Route &route = routes_by_worker_.at(worker_id);
    throw_if_route_terminal_locked(route);
    if (hello.session_id != route.session_id || hello.worker_id != worker_id ||
        hello.ready_state != remote_l3::ReadyState::READY || hello.comm_profile != route.comm_profile) {
        fail_locked("HELLO READY does not match manifest route");
        throw_if_terminal_locked();
    }
}

size_t MpiDirectTransportHub::pending_frame_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_frame_bytes_;
}

bool MpiDirectTransportHub::terminal() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !terminal_error_.empty();
}

std::string MpiDirectTransportHub::terminal_error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return terminal_error_;
}

MpiDirectTransport::MpiDirectTransport(
    std::shared_ptr<MpiDirectTransportHub> hub, int32_t worker_id, double attach_timeout_s, double runtime_timeout_s
) :
    hub_(std::move(hub)),
    worker_id_(worker_id),
    attach_timeout_s_(attach_timeout_s),
    runtime_timeout_s_(runtime_timeout_s) {
    if (!hub_) throw std::invalid_argument("MpiDirectTransport: null hub");
    if (worker_id_ < 0) throw std::invalid_argument("MpiDirectTransport: worker id must be non-negative");
    if (!(attach_timeout_s_ > 0.0) || !std::isfinite(attach_timeout_s_)) {
        throw std::invalid_argument("MpiDirectTransport: attach timeout must be positive and finite");
    }
    if (!(runtime_timeout_s_ > 0.0) || !std::isfinite(runtime_timeout_s_)) {
        throw std::invalid_argument("MpiDirectTransport: runtime timeout must be positive and finite");
    }
}

void MpiDirectTransport::expect_hello_ready() { hub_->expect_hello_ready(worker_id_, attach_timeout_s_); }

void MpiDirectTransport::submit_frame(const std::vector<uint8_t> &frame) {
    if (closed_.load(std::memory_order_acquire)) throw std::runtime_error("MpiDirectTransport: closed");
    if (progress_active_.load(std::memory_order_acquire)) {
        throw std::logic_error("MpiDirectTransport: progress command is active");
    }
    auto decoded = remote_l3::decode_frame(frame);
    hub_->enqueue(worker_id_, outbound_tag(decoded.header.frame_type), frame, runtime_timeout_s_);
}

std::vector<uint8_t> MpiDirectTransport::wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) {
    if (closed_.load(std::memory_order_acquire)) throw std::runtime_error("MpiDirectTransport: closed");
    if (progress_active_.load(std::memory_order_acquire)) {
        throw std::logic_error("MpiDirectTransport: progress command is active");
    }
    return hub_->wait_inbound(worker_id_, frame_type, sequence, runtime_timeout_s_);
}

void MpiDirectTransport::submit_progress_frame(const std::vector<uint8_t> &frame) {
    if (closed_.load(std::memory_order_acquire)) throw std::runtime_error("MpiDirectTransport: closed");
    if (progress_active_.load(std::memory_order_acquire)) {
        throw std::logic_error("MpiDirectTransport: progress command is already active");
    }
    auto decoded = remote_l3::decode_frame(frame);
    hub_->enqueue(worker_id_, outbound_tag(decoded.header.frame_type), frame, runtime_timeout_s_);
    progress_deadline_ = deadline_from_now(runtime_timeout_s_);
    progress_active_.store(true, std::memory_order_release);
}

bool MpiDirectTransport::poll_progress_reply(
    remote_l3::FrameType frame_type, uint64_t sequence, std::vector<uint8_t> &reply
) {
    if (closed_.load(std::memory_order_acquire)) throw std::runtime_error("MpiDirectTransport: closed");
    if (!progress_active_.load(std::memory_order_acquire)) {
        throw std::logic_error("MpiDirectTransport: no progress command is active");
    }
    if (std::chrono::steady_clock::now() >= progress_deadline_) {
        progress_active_.store(false, std::memory_order_release);
        hub_->fail("MpiDirectTransport: progress command timed out");
        throw std::runtime_error("MpiDirectTransport: progress command timed out");
    }
    auto result = hub_->poll_inbound(worker_id_, frame_type, sequence);
    if (!result.has_value()) return false;
    progress_active_.store(false, std::memory_order_release);
    reply = std::move(*result);
    return true;
}

void MpiDirectTransport::shutdown() {
    closed_.store(true, std::memory_order_release);
    progress_active_.store(false, std::memory_order_release);
    hub_->cancel_route(worker_id_, "transport shut down");
}
