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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "remote_endpoint.h"

enum class MpiDirectTag : int32_t {
    COMMAND_REQUEST = 1,
    COMMAND_REPLY = 2,
    HEALTH = 3,
    LIFECYCLE = 4,
};

struct MpiDirectOutboundFrame {
    uint64_t ticket{0};
    int32_t target_rank{-1};
    MpiDirectTag tag{MpiDirectTag::COMMAND_REQUEST};
    std::vector<uint8_t> frame;
};

class MpiDirectTransportHub {
public:
    explicit MpiDirectTransportHub(size_t max_pending_frame_bytes);

    void register_route(int32_t worker_id, int32_t mpi_rank, uint64_t session_id, const std::string &comm_profile);
    std::optional<MpiDirectOutboundFrame> poll_outbound(double timeout_s);
    void complete_outbound(uint64_t ticket);
    void deliver(int32_t source_rank, MpiDirectTag tag, const std::vector<uint8_t> &frame);
    void fail(const std::string &message);
    void close();

    size_t pending_frame_bytes() const;
    bool terminal() const;
    std::string terminal_error() const;

private:
    friend class MpiDirectTransport;

    struct Route {
        int32_t worker_id{-1};
        int32_t mpi_rank{-1};
        uint64_t session_id{0};
        std::string comm_profile;
        std::deque<std::vector<uint8_t>> replies;
        std::deque<std::vector<uint8_t>> lifecycle;
        std::chrono::steady_clock::time_point last_health{};
        std::string terminal_error;
    };

    void cancel_route(int32_t worker_id, const std::string &message);
    void enqueue(int32_t worker_id, MpiDirectTag tag, const std::vector<uint8_t> &frame, double timeout_s);
    std::optional<std::vector<uint8_t>>
    poll_inbound(int32_t worker_id, remote_l3::FrameType frame_type, uint64_t sequence);
    std::vector<uint8_t>
    wait_inbound(int32_t worker_id, remote_l3::FrameType frame_type, uint64_t sequence, double timeout_s);
    void expect_hello_ready(int32_t worker_id, double timeout_s);
    void throw_if_terminal_locked() const;
    void throw_if_route_terminal_locked(const Route &route) const;
    void fail_locked(const std::string &message);

    size_t max_pending_frame_bytes_{0};
    size_t pending_frame_bytes_{0};
    uint64_t next_ticket_{1};
    bool closed_{false};
    std::string terminal_error_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<MpiDirectOutboundFrame> outbound_;
    std::unordered_map<uint64_t, size_t> in_flight_bytes_;
    std::unordered_map<int32_t, Route> routes_by_worker_;
    std::unordered_map<int32_t, int32_t> worker_by_rank_;
};

class MpiDirectTransport : public RemoteL3Transport {
public:
    MpiDirectTransport(
        std::shared_ptr<MpiDirectTransportHub> hub, int32_t worker_id, double attach_timeout_s, double runtime_timeout_s
    );

    void expect_hello_ready();
    void submit_frame(const std::vector<uint8_t> &frame) override;
    std::vector<uint8_t> wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) override;
    void submit_progress_frame(const std::vector<uint8_t> &frame) override;
    bool poll_progress_reply(remote_l3::FrameType frame_type, uint64_t sequence, std::vector<uint8_t> &reply) override;
    void shutdown() override;

private:
    std::shared_ptr<MpiDirectTransportHub> hub_;
    int32_t worker_id_{-1};
    double attach_timeout_s_{0.0};
    double runtime_timeout_s_{0.0};
    std::chrono::steady_clock::time_point progress_deadline_{};
    std::atomic<bool> progress_active_{false};
    std::atomic<bool> closed_{false};
};
