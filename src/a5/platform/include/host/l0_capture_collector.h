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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/l0_profiling.h"
#include "host/profiler_base.h"

// The AICPU checkpoint publishes sealed ring ranges to a per-thread ready
// queue; this collector drains those queues, copies each range straight out of
// the GM ring, writes raw output, and advances the per-core consumed cursor.
// There is no intermediate device-side buffer.
class L0CaptureCollector {
public:
    using Alloc = profiling_common::ProfAllocCallback;
    using Free = profiling_common::ProfFreeCallback;

    L0CaptureCollector() = default;
    ~L0CaptureCollector();
    L0CaptureCollector(const L0CaptureCollector &) = delete;
    L0CaptureCollector &operator=(const L0CaptureCollector &) = delete;

    int init(
        const std::vector<uint32_t> &physical_core_ids, int num_threads, const std::vector<void *> &ring_addrs,
        Alloc alloc_cb, Free free_cb, int device_id
    );
    int begin_run(
        uint64_t run_epoch, const std::filesystem::path &output_dir, uint32_t ring_len, uint32_t ring_capacity,
        uint64_t aic_frequency_hz = 0, uint64_t aiv_frequency_hz = 0
    );
    void start(const profiling_common::ThreadFactory &create_thread);
    void stop();
    /** One synchronous pass over every ready queue, for the terminal drain. */
    void drain_remaining();
    /** Copy every GM allocation after producer stop, in physical address order. */
    bool dump_gm_rings();
    bool reconcile(bool terminal_status_ok);
    void finalize(Free free_cb);
    void stop_for_device_failure();
    void abandon_after_device_reset();
    void *device_shm() const { return shm_dev_; }
    bool initialized() const { return initialized_; }
    bool device_io_failed() const { return device_io_failed_; }
    bool output_failed() const { return output_failed_.load(); }
    std::vector<uint32_t> physical_core_types() const;

private:
    struct CoreOutput {
        std::ofstream raw;
        std::ofstream index;
        uint64_t offset{0};
        uint64_t next_sequence{0};
        uint64_t gm_snapshot_bytes{0};
    };

    void drain_loop();
    bool drain_available();
    void collect_range(const L0RangeEntry &entry);
    bool verify_live_copy(const L0RangeEntry &entry);
    bool advance_consumed(uint32_t core_index, uint32_t end_wptr, uint32_t end_samp_wrt);

    void *shm_dev_{nullptr};
    std::vector<uint8_t> shm_host_;
    std::vector<uint8_t> copy_buffer_;
    size_t shm_size_{0};
    uint32_t num_threads_{0};
    std::vector<void *> ring_addrs_;
    std::vector<uint32_t> physical_core_ids_;
    std::vector<CoreOutput> outputs_;
    std::mutex output_mutex_;
    std::atomic<bool> output_failed_{false};
    std::atomic<bool> stop_{false};
    std::thread drain_thread_;
    uint64_t run_epoch_{0};
    uint64_t live_batches_{0};
    uint32_t ring_len_{0};
    uint32_t ring_capacity_{0};
    uint64_t aic_frequency_hz_{0};
    uint64_t aiv_frequency_hz_{0};
    std::filesystem::path output_dir_;
    bool initialized_{false};
    bool device_io_failed_{false};
    int device_id_{-1};
    Alloc alloc_;
    Free free_;
};
