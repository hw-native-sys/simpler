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

#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "remote_wire.h"
#include "worker_manager.h"

class RemoteL3Transport {
public:
    virtual ~RemoteL3Transport() = default;
    virtual void submit_frame(const std::vector<uint8_t> &frame) = 0;
    virtual std::vector<uint8_t> wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) = 0;
    virtual void shutdown() {}
};

class RemoteL3SocketTransport : public RemoteL3Transport {
public:
    RemoteL3SocketTransport(
        std::string host, uint16_t port, std::string health_host, uint16_t health_port, double attach_timeout_s,
        double runtime_timeout_s
    );
    ~RemoteL3SocketTransport() override;

    void expect_hello_ready(uint64_t session_id, int32_t worker_id, const std::string &comm_profile);
    void submit_frame(const std::vector<uint8_t> &frame) override;
    std::vector<uint8_t> wait_for_reply(remote_l3::FrameType frame_type, uint64_t sequence) override;
    void shutdown() override;

private:
    std::string host_;
    uint16_t port_{0};
    std::string health_host_;
    uint16_t health_port_{0};
    // Startup-budget clock for the attach phase (command-connect, HELLO read,
    // health-connect all share attach_deadline_) vs the per-command runtime
    // timeout for post-attach frame I/O and the health-monitor loop.
    double attach_timeout_s_{30.0};
    double runtime_timeout_s_{30.0};
    std::chrono::steady_clock::time_point attach_deadline_{};
    int fd_{-1};
    int health_fd_{-1};
    std::thread health_thread_;
    std::atomic<bool> health_stop_{false};
    std::atomic<bool> health_failed_{false};
    std::mutex health_mu_;
    std::string health_error_;

    void connect_socket();
    void close_socket();
    void start_health_monitor(uint64_t session_id, int32_t worker_id);
    void stop_health_monitor();
    void mark_health_failed(const std::string &message);
    void check_health();
    void wait_readable(std::chrono::steady_clock::time_point deadline);
    void wait_writable(std::chrono::steady_clock::time_point deadline);
    void write_all(const uint8_t *data, size_t size, std::chrono::steady_clock::time_point deadline);
    std::vector<uint8_t> read_frame(std::chrono::steady_clock::time_point deadline);
};

class RemoteL3Endpoint : public WorkerEndpoint {
public:
    RemoteL3Endpoint(
        int32_t worker_id, uint64_t session_id, std::string transport_name, std::unique_ptr<RemoteL3Transport> transport
    );

    const WorkerEndpointCaps &caps() const override { return caps_; }
    WorkerCompletion run(Ring *ring, const WorkerDispatch &dispatch) override;
    void shutdown_child() override;
    void control_prepare(const uint8_t *digest) override;
    void control_remote_prepare_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest,
        const void *payload, size_t payload_size
    ) override;
    void control_remote_commit_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    void control_remote_abort_register(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    void control_remote_unregister(
        remote_l3::RemoteRegistryTarget target_registry, CallableKind callable_kind, const uint8_t *digest
    ) override;
    RemoteBufferHandle control_remote_malloc(size_t size) override;
    void control_remote_free(const RemoteBufferHandle &handle) override;
    void
    control_remote_copy_to(const RemoteBufferHandle &handle, uint64_t offset, const void *src, size_t size) override;
    void control_remote_copy_from(void *dst, const RemoteBufferHandle &handle, uint64_t offset, size_t size) override;
    RemoteBufferExport control_remote_export(
        const RemoteBufferHandle &handle, uint64_t offset, uint64_t size, uint32_t access_flags,
        const std::string &transport_profile
    ) override;
    RemoteBufferHandle control_remote_import(
        int32_t importer_worker_id, const RemoteBufferExport &export_desc, uint32_t requested_access_flags
    ) override;
    void control_remote_release_import(const RemoteBufferHandle &handle) override;
    std::vector<uint8_t>
    control_remote_domain(remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes) override;

private:
    WorkerEndpointCaps caps_;
    uint64_t session_id_{0};
    std::unique_ptr<RemoteL3Transport> transport_;
    remote_l3::OrderedCommandLane command_lane_;
    std::mutex command_mu_;

    remote_l3::TaskPayloadWire build_task_payload(const TaskSlotState &slot, int32_t group_index) const;
    remote_l3::ControlReplyPayload
    run_control(remote_l3::ControlName control_name, const std::vector<uint8_t> &command_bytes);
};
