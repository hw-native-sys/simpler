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
 * @brief Host-side PMU data collector (memcpy-based)
 *
 * Design:
 *   1. Host pre-allocates one PmuBuffer per AICore on device, plus a single
 *      PmuSetupHeader that stores all buffer device pointers, core count,
 *      and the selected PMU event type.
 *   2. During execution, AICPU reads PMU MMIO counters after each task FIN
 *      and writes one PmuRecord into that core's PmuBuffer (silently
 *      incrementing PmuBufferState::dropped_record_count when the buffer is full).
 *   3. After stream sync, host copies the PmuBuffer header (to learn count)
 *      and then count*sizeof(PmuRecord) actual records back, per core.
 *   4. Host exports a LuoPan-compatible CSV under outputs/.
 *
 * halHostRegister is not supported on DAV_3510, so the collector uses
 * post-stream-sync memcpy rather than a shared-memory + SPSC-queue +
 * background-collector-thread streaming model.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "common/pmu_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"

// ---------------------------------------------------------------------------
// Memory operation callbacks (injected by DeviceRunner)
// ---------------------------------------------------------------------------

using PmuAllocCallback = void *(*)(size_t size);
using PmuFreeCallback = int (*)(void *dev_ptr);
using PmuCopyToDeviceCallback = int (*)(void *dev_dst, const void *host_src, size_t size);
using PmuCopyFromDeviceCallback = int (*)(void *host_dst, const void *dev_src, size_t size);

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
     * Allocate device-side PMU buffers and publish the setup header pointer.
     *
     * @param num_cores        Number of AICore instances to profile
     * @param event_type       PmuEventType value (stored in header, used by AICPU)
     * @param kernel_args_pmu_data_base Out: device address of PmuSetupHeader
     * @param alloc_cb         Device memory alloc
     * @param free_cb          Device memory free
     * @param copy_to_dev_cb   Host→device (for publishing header)
     * @param copy_from_dev_cb Device→host (for collect_all)
     * @return 0 on success
     */
    int initialize(
        int num_cores, uint32_t event_type, uint64_t *kernel_args_pmu_data_base, PmuAllocCallback alloc_cb,
        PmuFreeCallback free_cb, PmuCopyToDeviceCallback copy_to_dev_cb, PmuCopyFromDeviceCallback copy_from_dev_cb
    );

    /**
     * Copy all PMU buffers back from device. Fills collected_records_.
     * Must be called after the execution stream has been fully synchronized.
     */
    int collect_all();

    /**
     * Export collected records to a LuoPan-compatible CSV under output_dir/.
     */
    int export_csv(const std::string &output_dir = "outputs");

    /**
     * Free all device buffers and reset host state.
     */
    int finalize();

    bool is_initialized() const { return setup_header_dev_ != nullptr; }

    const std::vector<std::vector<PmuRecord>> &get_records() const { return collected_records_; }

private:
    void *setup_header_dev_{nullptr};
    std::vector<void *> core_buffers_dev_;  // PmuBuffer* per core (device)

    int num_cores_{0};
    uint32_t event_type_{0};
    size_t pmu_buffer_bytes_{0};
    size_t setup_region_bytes_{0};

    PmuAllocCallback alloc_cb_{nullptr};
    PmuFreeCallback free_cb_{nullptr};
    PmuCopyToDeviceCallback copy_to_dev_cb_{nullptr};
    PmuCopyFromDeviceCallback copy_from_dev_cb_{nullptr};

    // Host-side collected data (indexed by core id)
    std::vector<std::vector<PmuRecord>> collected_records_;
    std::vector<uint32_t> dropped_counts_;
    std::vector<uint32_t> total_counts_;
    std::vector<uint32_t> owning_thread_ids_;
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

inline uint32_t resolve_pmu_event_type(int requested_event_type) {
    uint32_t resolved = PMU_EVENT_TYPE_DEFAULT;
    if (requested_event_type > 0 &&
        pmu_resolve_event_config_a5(static_cast<uint32_t>(requested_event_type)) != nullptr) {
        resolved = static_cast<uint32_t>(requested_event_type);
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
    if (val > 0 && pmu_resolve_event_config_a5(static_cast<uint32_t>(val)) != nullptr) {
        resolved = static_cast<uint32_t>(val);
        LOG_INFO("PMU event type set to %u from SIMPLER_PMU_EVENT_TYPE", resolved);
        return resolved;
    }
    LOG_WARN("Invalid SIMPLER_PMU_EVENT_TYPE=%s, using default (PIPE_UTILIZATION=%u)", pmu_env, PMU_EVENT_TYPE_DEFAULT);
    return resolved;
}

inline std::string make_pmu_csv_path(const std::string &output_dir) {
    if (mkdir(output_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOG_WARN("Failed to create PMU output directory %s: errno=%d", output_dir.c_str(), errno);
    }
    char csv_name[128];
    auto now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm *tm_info = std::localtime(&t_now);
    if (tm_info != nullptr) {
        char base[96];
        std::strftime(base, sizeof(base), "pmu_%Y%m%d_%H%M%S", tm_info);
        std::snprintf(csv_name, sizeof(csv_name), "%s_%03ld.csv", base, static_cast<long>(ms));
    } else {
        std::snprintf(csv_name, sizeof(csv_name), "pmu_output.csv");
    }
    return output_dir + "/" + csv_name;
}

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_PMU_COLLECTOR_H_
