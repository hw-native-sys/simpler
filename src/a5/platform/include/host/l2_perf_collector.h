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
 * @file l2_perf_collector.h
 * @brief Host-side performance data collector (memcpy-based)
 *
 * Design:
 *   1. Host pre-allocates one L2PerfBuffer per core and one PhaseBuffer per
 *      AICPU thread on device, plus a single L2PerfSetupHeader that stores
 *      all buffer pointers, total_tasks, and the AicpuPhaseHeader.
 *   2. During execution, AICore writes timing into L2PerfBuffers and AICPU
 *      completes records + writes phase data directly into device memory.
 *      When a buffer fills up, records are silently dropped (AICPU-side
 *      early return).
 *   3. After stream sync, Host copies L2PerfSetupHeader, each L2PerfBuffer, and
 *      each PhaseBuffer back via rtMemcpy (two-step: 64B header → read
 *      count → copy count*sizeof(record) actual data).
 *
 * This replaces the previous shared-memory + ProfMemoryManager design that
 * depended on halHostRegister, which A5 hardware does not support.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"

/**
 * Device memory allocation callback.
 *
 * @param size      Memory size in bytes
 * @return Allocated device memory pointer, or nullptr on failure
 */
using L2PerfAllocCallback = void *(*)(size_t size);

/**
 * Device memory free callback.
 *
 * @param dev_ptr   Device memory pointer
 * @return 0 on success, error code on failure
 */
using L2PerfFreeCallback = int (*)(void *dev_ptr);

/**
 * Host -> Device copy callback (rtMemcpy HOST_TO_DEVICE / memcpy in sim).
 *
 * @param dev_dst   Device destination pointer
 * @param host_src  Host source pointer
 * @param size      Number of bytes to copy
 * @return 0 on success, error code on failure
 */
using L2PerfCopyToDeviceCallback = int (*)(void *dev_dst, const void *host_src, size_t size);

/**
 * Device -> Host copy callback (rtMemcpy DEVICE_TO_HOST / memcpy in sim).
 *
 * @param host_dst  Host destination pointer
 * @param dev_src   Device source pointer
 * @param size      Number of bytes to copy
 * @return 0 on success, error code on failure
 */
using L2PerfCopyFromDeviceCallback = int (*)(void *host_dst, const void *dev_src, size_t size);

/**
 * Host-side performance data collector.
 *
 * Lifecycle:
 *   1. initialize() — allocate L2PerfSetupHeader and all per-core/per-thread
 *      buffers on device, publish pointers into runtime.l2_perf_data_base
 *   2. (AICore/AICPU run, writing directly into device buffers)
 *   3. collect_all() — after stream sync, copy L2PerfSetupHeader back,
 *      then copy each L2PerfBuffer / PhaseBuffer back using two-step
 *      count-first memcpy. Fills collected_*_records_ vectors.
 *   4. export_swimlane_json() — serialize collected data to Chrome Trace
 *      Event Format JSON. Logic unchanged from previous design.
 *   5. finalize() — free all device buffers.
 */
class L2PerfCollector {
public:
    L2PerfCollector() = default;
    ~L2PerfCollector();

    L2PerfCollector(const L2PerfCollector &) = delete;
    L2PerfCollector &operator=(const L2PerfCollector &) = delete;

    /**
     * Allocate device buffers and publish L2PerfSetupHeader.
     *
     * @param runtime          Runtime to configure (sets runtime.l2_perf_data_base)
     * @param num_aicore       Number of AICore instances to profile
     * @param device_id        Device ID (stored for later callbacks)
     * @param alloc_cb         Device memory alloc
     * @param free_cb          Device memory free
     * @param copy_to_dev_cb   Host→device copy (used during init to publish header)
     * @param copy_from_dev_cb Device->host copy (used during collect_all)
     * @return 0 on success, error code on failure
     */
    int initialize(
        Runtime &runtime, int num_aicore, int device_id, L2PerfAllocCallback alloc_cb, L2PerfFreeCallback free_cb,
        L2PerfCopyToDeviceCallback copy_to_dev_cb, L2PerfCopyFromDeviceCallback copy_from_dev_cb
    );

    /**
     * Copy all profiling data back from device and parse it into
     * collected_perf_records_ / collected_phase_records_ /
     * collected_orch_summary_ / core_to_thread_.
     *
     * Must be called after the execution stream has been fully synchronized.
     *
     * @return 0 on success, error code on failure
     */
    int collect_all();

    /**
     * Export collected data to Chrome Trace Event Format JSON.
     *
     * @param output_path Output directory
     * @return 0 on success, -1 on failure
     */
    int export_swimlane_json(const std::string &output_path = "outputs");

    /**
     * Free all device buffers and clear host-side state.
     *
     * @return 0 on success, error code on failure
     */
    int finalize();

    /**
     * Check if the collector has been initialized.
     */
    bool is_initialized() const { return setup_header_dev_ != nullptr; }

    /**
     * Accessor used by tests.
     */
    const std::vector<std::vector<L2PerfRecord>> &get_records() const { return collected_perf_records_; }

private:
    // Device-side allocations
    void *setup_header_dev_{nullptr};
    std::vector<void *> core_buffers_dev_;
    std::vector<void *> phase_buffers_dev_;

    // Configuration
    int num_aicore_{0};
    int num_phase_threads_{0};
    int device_id_{-1};

    // Sizes (computed once in initialize)
    size_t l2_perf_buffer_bytes_{0};
    size_t phase_buffer_bytes_{0};

    // Callbacks
    L2PerfAllocCallback alloc_cb_{nullptr};
    L2PerfFreeCallback free_cb_{nullptr};
    L2PerfCopyToDeviceCallback copy_to_dev_cb_{nullptr};
    L2PerfCopyFromDeviceCallback copy_from_dev_cb_{nullptr};

    // Host-side collected data (indexed by core / thread)
    std::vector<std::vector<L2PerfRecord>> collected_perf_records_;
    std::vector<std::vector<AicpuPhaseRecord>> collected_phase_records_;
    AicpuOrchSummary collected_orch_summary_{};
    bool has_phase_data_{false};

    // Core-to-thread mapping (core_id → scheduler thread index, -1 = unassigned)
    std::vector<int8_t> core_to_thread_;
};

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_L2_PERF_COLLECTOR_H_
