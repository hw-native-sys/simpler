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
 * @file tensor_dump_collector.h
 * @brief Host-side tensor dump collector (memcpy-based)
 *
 * - Host allocates per-thread DumpBuffers + arenas on device
 * - AICPU writes records and payload during execution
 * - After stream sync, host copies everything back via rtMemcpy/memcpy
 * - Export dump files (JSON manifest + binary payload)
 *
 * No background threads, no SPSC queues — simple collect-after-sync.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "common/platform_config.h"
#include "common/tensor_dump.h"
#include "data_type.h"

/**
 * Device memory allocation callback.
 */
using DumpAllocCallback = void *(*)(size_t size);

/**
 * Device memory free callback.
 */
using DumpFreeCallback = int (*)(void *dev_ptr);

/**
 * Host -> Device copy callback.
 */
using DumpCopyToDeviceCallback = int (*)(void *dev_dst, const void *host_src, size_t size);

/**
 * Device -> Host copy callback.
 */
using DumpCopyFromDeviceCallback = int (*)(void *host_dst, const void *dev_src, size_t size);

// =============================================================================
// DumpedTensor - Collected tensor metadata + payload bytes
// =============================================================================

/**
 * Collected tensor metadata + payload bytes (identical to A2A3 DumpedTensor).
 */
struct DumpedTensor {
    uint64_t task_id;
    uint8_t subtask_id;
    uint32_t func_id;
    uint32_t arg_index;
    TensorDumpRole role;
    TensorDumpStage stage;
    uint8_t dtype;
    uint8_t ndims;
    uint32_t raw_shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t shapes[PLATFORM_DUMP_MAX_DIMS];
    uint32_t offsets[PLATFORM_DUMP_MAX_DIMS];
    bool is_contiguous;
    bool truncated;
    bool overwritten;
    uint64_t payload_size;
    uint64_t bin_offset;
    std::vector<uint8_t> bytes;
};

// =============================================================================
// TensorDumpCollector - Main Collector
// =============================================================================

/**
 * Host-side tensor dump collector.
 *
 * Lifecycle:
 *   1. initialize() — allocate DumpSetupHeader + per-thread DumpBuffers + arenas,
 *      caller reads get_dump_setup_device_ptr() and sets kernel_args.dump_data_base
 *   2. (AICPU execution writes records and payload data)
 *   3. collect_all() — after stream sync, copy header + buffers + arenas back
 *   4. export_dump_files() — write JSON manifest + binary payload
 *   5. finalize() — free all device allocations
 */
class TensorDumpCollector {
public:
    TensorDumpCollector() = default;
    ~TensorDumpCollector();

    TensorDumpCollector(const TensorDumpCollector &) = delete;
    TensorDumpCollector &operator=(const TensorDumpCollector &) = delete;

    /**
     * Initialize tensor dump device buffers.
     *
     * @param num_dump_threads Number of AICPU scheduling threads
     * @param device_id        Device ID
     * @param alloc_cb         Device memory alloc
     * @param free_cb          Device memory free
     * @param copy_to_dev_cb   Host->device copy
     * @param copy_from_dev_cb Device->host copy
     * @return 0 on success, error code on failure
     */
    int initialize(
        int num_dump_threads, int device_id, DumpAllocCallback alloc_cb, DumpFreeCallback free_cb,
        DumpCopyToDeviceCallback copy_to_dev_cb, DumpCopyFromDeviceCallback copy_from_dev_cb
    );

    /**
     * Copy all dump data back from device and parse into collected_ vector.
     * Must be called after execution stream has been fully synchronized.
     *
     * @return 0 on success, error code on failure
     */
    int collect_all();

    /**
     * Export collected data to dump files (JSON manifest + binary payload).
     *
     * @param output_path Output directory
     * @return 0 on success, -1 on failure
     */
    int export_dump_files(const std::string &output_path = "outputs");

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
     * Get the device pointer to the DumpSetupHeader.
     * Used to set kernel_args.dump_data_base.
     */
    void *get_dump_setup_device_ptr() const { return setup_header_dev_; }

private:
    // Device-side allocations
    void *setup_header_dev_{nullptr};
    std::vector<void *> dump_buffers_dev_;   // Per-thread DumpBuffer
    std::vector<void *> arena_headers_dev_;  // Per-thread DumpArenaHeader
    std::vector<void *> arena_data_dev_;     // Per-thread arena data

    // Configuration
    int num_dump_threads_{0};
    int device_id_{-1};
    size_t dump_buffer_bytes_{0};

    // Callbacks
    DumpAllocCallback alloc_cb_{nullptr};
    DumpFreeCallback free_cb_{nullptr};
    DumpCopyToDeviceCallback copy_to_dev_cb_{nullptr};
    DumpCopyFromDeviceCallback copy_from_dev_cb_{nullptr};

    // Collected data
    std::vector<DumpedTensor> collected_;

    // Stats
    uint32_t total_dropped_count_{0};
    uint32_t total_truncated_count_{0};
    uint32_t total_overwrite_count_{0};
};

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_TENSOR_DUMP_COLLECTOR_H_
