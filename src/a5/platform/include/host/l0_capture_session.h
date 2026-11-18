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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/l0_profiling.h"
#include "host/profiler_base.h"

class L0CaptureSession {
public:
    using Alloc = profiling_common::ProfAllocCallback;
    using Free = profiling_common::ProfFreeCallback;

    L0CaptureSession() = default;
    ~L0CaptureSession();
    L0CaptureSession(const L0CaptureSession &) = delete;
    L0CaptureSession &operator=(const L0CaptureSession &) = delete;

    int
    arm(int device_id, int core_count, uint64_t reg_addr_table, uint64_t run_epoch, const std::string &output_prefix,
        Alloc alloc, Free free);
    static constexpr uint32_t kRingLength = 8u << 20;
    static constexpr uint32_t kRingCapacity = kRingLength + L0_DFX_TAIL_HEADROOM_BYTES;
    uint32_t ring_len() const { return kRingLength; }
    uint32_t ring_capacity() const { return kRingCapacity; }
    const std::vector<void *> &buffers() const { return buffers_; }
    L0ConfigArgs config_args(uint64_t dfx_shm) const;
    int start_capture_channels();
    int read_terminal_snapshot();
    int verify_terminal_status() const;
    int stop_producer();
    // A poisoned device must be reset by the caller; true skips HAL stops and device frees.
    void reset(bool abandon_device_resources = false) noexcept;
    void confirm_device_reset() noexcept {
        channels_unresolved_ = false;
        device_io_failed_ = false;
    }
    bool active() const { return device_id_ >= 0; }
    bool has_open_channels() const { return !channels_started_.empty(); }
    bool channels_unresolved() const { return channels_unresolved_; }
    bool device_io_failed() const { return device_io_failed_; }
    uint64_t run_epoch() const { return run_epoch_; }
    uint64_t aic_frequency_hz() const { return aic_frequency_hz_; }
    uint64_t aiv_frequency_hz() const { return aiv_frequency_hz_; }

private:
    int start_channels();
    int stop_channels() noexcept;
    void release_channel_handles() noexcept;
    void release_buffers(bool abandon) noexcept;

    int device_id_{-1};
    uint64_t run_epoch_{0};
    uint64_t aic_frequency_hz_{0};
    uint64_t aiv_frequency_hz_{0};
    uint64_t reg_addr_table_{0};
    std::string output_prefix_;
    Alloc alloc_;
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
