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
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/l0_profiling.h"
#include "worker/native_run_execution.h"

class L0CaptureSession {
public:
    using Alloc = std::function<void *(size_t)>;
    using Free = std::function<int(void *)>;

    L0CaptureSession() = default;
    ~L0CaptureSession();
    L0CaptureSession(const L0CaptureSession &) = delete;
    L0CaptureSession &operator=(const L0CaptureSession &) = delete;

    int arm(int device_id, int core_count, uint64_t reg_addr_table, uint64_t run_epoch, Alloc alloc, Free free);
    static constexpr uint32_t kRingLength = 32u << 20;
    static constexpr uint32_t kRingCapacity = kRingLength + L0_DFX_TAIL_HEADROOM_BYTES;
    const std::vector<void *> &buffers() const { return buffers_; }
    L0ConfigArgs config_args(uint64_t dfx_shm) const;
    int start_capture_channels();
    int read_terminal_snapshot();
    int verify_terminal_status() const;
    bool terminal_range(size_t core, L0RingRange *range) const;
    int stop_producer();
    // A poisoned device must be reset by the caller; true skips HAL stops and device frees.
    int reset(bool abandon_device_resources = false) noexcept;
    bool active() const { return hal_device_id_ >= 0; }
    bool channels_unresolved() const { return channels_unresolved_; }
    bool device_io_failed() const { return device_io_failed_; }
    uint64_t run_epoch() const { return run_epoch_; }
    uint64_t aic_frequency_hz() const { return aic_frequency_hz_; }
    uint64_t aiv_frequency_hz() const { return aiv_frequency_hz_; }

private:
    int start_channels();
    int stop_channels() noexcept;
    void release_channel_handles() noexcept;
    int release_buffers(bool abandon) noexcept;

    int hal_device_id_{-1};
    uint64_t run_epoch_{0};
    uint64_t aic_frequency_hz_{0};
    uint64_t aiv_frequency_hz_{0};
    uint64_t reg_addr_table_{0};
    Free free_;
    std::vector<void *> buffers_;
    std::vector<L0CoreRegSnapshot> configured_snapshots_;
    std::vector<L0CoreEndSnapshot> terminal_snapshots_;
    void *addr_table_{nullptr};
    void *snapshots_{nullptr};
    void *end_snapshots_{nullptr};
    void *hal_handle_{nullptr};
    int (*prof_start_)(unsigned int, unsigned int, struct prof_start_para *){nullptr};
    int (*prof_stop_)(unsigned int, unsigned int){nullptr};
    std::vector<unsigned int> channels_started_;
    bool channels_unresolved_{false};
    bool device_io_failed_{false};
};

// Capture resources belong to one run; GM is read after sampling stops.
class L0CaptureCollector {
public:
    using Alloc = L0CaptureSession::Alloc;
    using Free = L0CaptureSession::Free;

    L0CaptureCollector() = default;
    ~L0CaptureCollector() = default;
    L0CaptureCollector(const L0CaptureCollector &) = delete;
    L0CaptureCollector &operator=(const L0CaptureCollector &) = delete;

    int initialize(
        const NativeRunIdentity &identity, int device_id, int core_count, uint64_t register_table, Alloc alloc_cb,
        Free free_cb
    );
    int start_run(const NativeRunIdentity &identity, const std::string &output_prefix);
    int close_run(
        const NativeRunIdentity &identity, bool device_execution_complete, const std::function<int()> &synchronize
    );
    void finalize_capture(bool abandon_device_resources);
    int configure_run(const NativeRunIdentity &identity, const std::function<int(L0ConfigArgs &)> &launch);
    bool capture_active() const { return capture_active_.load(std::memory_order_acquire); }

    int begin_run(const std::filesystem::path &output_dir);
    bool device_io_failed() const { return device_io_failed_.load(std::memory_order_acquire); }

private:
    int init(Alloc alloc_cb);
    void copy_terminal_ranges();
    bool finish_output(bool terminal_status_ok);
    // Caller holds capture_mutex_; release follows producer completion or abandonment.
    int release_capture(bool abandon_device_resources);
    L0CaptureSession session_;
    Free free_;
    std::mutex capture_mutex_;
    std::optional<NativeRunIdentity> capture_owner_;
    std::atomic<bool> capture_active_{false};
    struct CoreOutput {
        std::ofstream raw;
        uint64_t bytes{0};
        bool complete{false};
    };

    int copy_to_device(volatile void *destination, const void *source, size_t bytes);
    int copy_from_device(volatile void *destination, const volatile void *source, size_t bytes);

    void *shm_dev_{nullptr};
    std::unique_ptr<uint8_t, decltype(&std::free)> shm_host_{nullptr, &std::free};
    size_t shm_size_{0};
    std::vector<CoreOutput> outputs_;
    bool output_failed_{false};
    std::filesystem::path output_dir_;
    bool initialized_{false};
    std::atomic<bool> device_io_failed_{false};
};
