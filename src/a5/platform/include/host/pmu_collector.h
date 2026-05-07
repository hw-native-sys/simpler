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

/**
 * @file pmu_collector.h
 * @brief Host-side PMU buffer allocation, streaming collection, and CSV export.
 *
 * Lifecycle mirrors a2a3 PmuCollector:
 *   init()                    — Allocate PmuDataHeader + PmuBufferState shared memory,
 *                               pre-allocate PmuBuffers and push into free_queues.
 *   [start collector thread]
 *   poll_and_collect()        — Poll ready_queues, recycle buffers, write CSV.
 *   signal_execution_complete() — Notify collector that device is done.
 *   [join collector thread]
 *   drain_remaining_buffers() — Scan any buffers still held by AICPU after execution.
 *   finalize()                — Free all device memory.
 *
 * Memory model (a5-specific):
 *   a5 has no halHostRegister, so host↔device SPSC fields are accessed
 *   through host shadow buffers + rtMemcpy (onboard) or memcpy (sim).
 *   Each device buffer has a paired host shadow in buf_pool_.
 *   The shared memory region (PmuDataHeader + PmuBufferState[]) also has
 *   separate device and host copies synchronized via copy hooks.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/platform_config.h"
#include "common/pmu_profiling.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"

// ---------------------------------------------------------------------------
// Memory operation callbacks (injected by DeviceRunner, same as a2a3)
// ---------------------------------------------------------------------------

/**
 * Allocate device memory. Returns nullptr on failure.
 */
using PmuAllocCallback = void *(*)(size_t size, void *user_data);

/**
 * Register device memory for host-visible access.
 * Unused on a5 (no halHostRegister); kept for interface parity with a2a3.
 */
using PmuRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void *user_data, void **host_ptr);

/**
 * Unregister previously registered host-visible device memory.
 * Unused on a5; kept for interface parity with a2a3.
 */
using PmuUnregisterCallback = int (*)(void *dev_ptr, int device_id, void *user_data);

/**
 * Free device memory.
 */
using PmuFreeCallback = int (*)(void *dev_ptr, void *user_data);

// ---------------------------------------------------------------------------
// PmuCollector
// ---------------------------------------------------------------------------

class PmuCollector {
public:
    PmuCollector() = default;
    ~PmuCollector();

    PmuCollector(const PmuCollector &) = delete;
    PmuCollector &operator=(const PmuCollector &) = delete;

    /**
     * Allocate PMU shared memory and pre-populate free_queues.
     *
     * @param num_cores               Number of AICore instances in use
     * @param num_threads             Number of AICPU scheduling threads
     * @param kernel_args_pmu_data_base  Out: device address of PmuDataHeader
     * @param csv_path                Output CSV file path
     * @param event_type              PmuEventType value (written to CSV rows)
     * @param alloc_cb / register_cb / free_cb  Memory operation callbacks
     * @param user_data               Opaque pointer forwarded to callbacks
     * @param device_id               Device ID (for interface parity)
     * @return 0 on success, non-zero on failure
     */
    int init(
        int num_cores, int num_threads, uint64_t *kernel_args_pmu_data_base, const std::string &csv_path,
        PmuEventType event_type, PmuAllocCallback alloc_cb, PmuFreeCallback free_cb, void *user_data, int device_id
    );

    /**
     * Main body of the collector thread.
     * Polls all per-thread ready_queues via rtMemcpy, appends records to CSV,
     * recycles buffers back into free_queues.
     */
    void poll_and_collect();

    /**
     * Signal that device execution has finished.
     * The collector thread will drain remaining entries then exit.
     */
    void signal_execution_complete();

    /**
     * After the collector thread exits, scan PmuBufferState.current_buf_ptr for
     * any remaining non-empty buffers that AICPU flushed but the collector
     * thread may not have consumed yet.
     */
    void drain_remaining_buffers();

    /**
     * Free all device/shared memory and unregister mapped regions.
     */
    void finalize(PmuUnregisterCallback unregister_cb, PmuFreeCallback free_cb, void *user_data);

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    int num_cores_ = 0;
    int num_threads_ = 0;
    int device_id_ = -1;
    PmuEventType event_type_ = PmuEventType::PIPE_UTILIZATION;

    // Shared memory region (PmuDataHeader + PmuBufferState[])
    void *shm_dev_ = nullptr;   // Device address
    void *shm_host_ = nullptr;  // Host shadow
    size_t shm_size_ = 0;

    // Pre-allocated PmuBuffers (one pool per core × BUFFERS_PER_CORE)
    struct BufEntry {
        void *dev_ptr = nullptr;
        void *host_ptr = nullptr;
    };
    std::vector<BufEntry> buf_pool_;

    PmuAllocCallback alloc_cb_ = nullptr;
    PmuFreeCallback free_cb_ = nullptr;
    void *user_data_ = nullptr;

    std::string csv_path_;
    std::string csv_header_;
    std::ofstream csv_file_;
    std::mutex csv_mutex_;

    std::atomic<bool> execution_complete_{false};
    uint64_t total_collected_ = 0;

    std::unordered_set<uint64_t> drained_bufs_;

    // Internal helpers
    PmuDataHeader *pmu_header_host() const { return get_pmu_header(shm_host_); }
    PmuBufferState *pmu_state_host(int core_id) const { return get_pmu_buffer_state(shm_host_, core_id); }
    PmuDataHeader *pmu_header_dev() const { return get_pmu_header(shm_dev_); }
    PmuBufferState *pmu_state_dev(int core_id) const { return get_pmu_buffer_state(shm_dev_, core_id); }

    void write_buffer_to_csv(int core_id, int thread_idx, const void *buf_host_ptr);
    void push_to_free_queue(int core_id, uint64_t buf_dev_addr);
    void ensure_csv_open_unlocked();

    void *resolve_host_ptr(uint64_t dev_addr);
};

// ---------------------------------------------------------------------------
// Utility: resolve PMU event type (env-var override)
// ---------------------------------------------------------------------------

inline PmuEventType resolve_pmu_event_type(int requested_event_type) {
    PmuEventType resolved = PmuEventType::PIPE_UTILIZATION;
    if (requested_event_type > 0 &&
        pmu_resolve_event_config_a5(static_cast<PmuEventType>(requested_event_type)) != nullptr) {
        resolved = static_cast<PmuEventType>(requested_event_type);
    } else if (requested_event_type != 0) {
        LOG_WARN(
            "Invalid PMU event type %u, using default (PIPE_UTILIZATION=%u)", requested_event_type,
            PMU_EVENT_TYPE_DEFAULT
        );
    }
    const char *pmu_env = std::getenv("SIMPLER_PMU_EVENT_TYPE");
    if (pmu_env == nullptr) {
        return resolved;
    }
    int val = std::atoi(pmu_env);
    if (val > 0 && pmu_resolve_event_config_a5(static_cast<PmuEventType>(val)) != nullptr) {
        resolved = static_cast<PmuEventType>(val);
        LOG_INFO_V0("PMU event type set to %u from SIMPLER_PMU_EVENT_TYPE", static_cast<uint32_t>(resolved));
        return resolved;
    }
    LOG_WARN("Invalid SIMPLER_PMU_EVENT_TYPE=%s, using default (PIPE_UTILIZATION=%u)", pmu_env, PMU_EVENT_TYPE_DEFAULT);
    return resolved;
}

inline std::string make_pmu_csv_path(const std::string &output_dir) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        LOG_WARN("Failed to create PMU output directory %s: %s", output_dir.c_str(), ec.message().c_str());
    }
    // Filename is fixed (no timestamp) — the caller-provided directory is the
    // per-task uniqueness boundary.
    return output_dir + "/pmu.csv";
}

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
