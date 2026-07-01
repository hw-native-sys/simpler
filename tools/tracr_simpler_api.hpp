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
 * TraCR API functions for Simpler A2A3, A2A3sim, A5, A5sim
 *
 * TODO: A5 not yet able to test
 */

#pragma once

#include <filesystem>  // C++17 or newer
#include <fstream>
#include <array>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <tracr/tracr.hpp>
#include <tracr_simpler_markers.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// TraCR profiling/benchmarking stuff
size_t getSampleID() {
    const auto env = std::getenv("PYPTO_RUN_SAMPLE_ID");
    return env ? std::stoul(env) : 0;
}
size_t sampleID = getSampleID();

std::string tracr_dir = "~/ascend/tracr/proc.1";

/**
 * A function for defining the path of the TraCR traces in home
 */
fs::path expand_user_path(const std::string &path) {
    if (!path.empty() && path[0] == '~') {
        const char *home = std::getenv("HOME");
        if (!home) throw std::runtime_error("HOME not set");

        std::string sub = path.substr(1);                        // remove ~
        if (!sub.empty() && sub[0] == '/') sub = sub.substr(1);  // remove leading slash

        return fs::path(home) / sub;
    }
    return fs::path(path);
}

/**
 *
 */
inline int TracrData2BTS(const TraCR::Payload *tracrData, const size_t *tracrDataSizes, const size_t num_threads) {
    fs::path base_dir = expand_user_path(tracr_dir);

    fs::create_directories(base_dir);

    for (uint32_t t = 0; t < num_threads; ++t) {
        size_t num_traces = tracrDataSizes[t];

        if (num_traces == 0) continue;

        if (num_traces > TraCR::CAPACITY) {
            LOG_ERROR("Thread %u exceeds CAPACITY", t);
            return -1;
        }

        fs::path thread_dir = base_dir / ("thread." + std::to_string(t + 1));

        fs::create_directories(thread_dir);

        fs::path file_path = thread_dir / "traces.bts";

        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            LOG_ERROR("Cannot open %s", file_path);
            return -1;
        }

        const TraCR::Payload *thread_ptr = tracrData + t * TraCR::CAPACITY;

        out.write(reinterpret_cast<const char *>(thread_ptr), num_traces * sizeof(TraCR::Payload));

        if (!out) {
            LOG_ERROR("Write failed for %s", file_path);
            return -1;
        }
    }
    return 0;
}

/**
 * A method for storing the TraCR metadata.json
 */
template <typename RuntimeT>
int StoreTracrMetaData(RuntimeT &runtime) {
    fs::path base_dir = expand_user_path(tracr_dir);

    // Add the metadata.json
    nlohmann::json metadata;

    // channel_names
    nlohmann::json channel_names = nlohmann::json::array();
    for (int i = 0; i < runtime.get_aicpu_thread_num(); ++i) {
        channel_names.push_back("AICPU_" + std::to_string(i));
    }
    for (int i = 0; i < int(runtime.get_worker_count() / 3); ++i) {
        channel_names.push_back("AICube_" + std::to_string(i));
    }
    for (int i = 0; i < int(2 * runtime.get_worker_count() / 3); ++i) {
        channel_names.push_back("AIVector_" + std::to_string(i));
    }
    channel_names.push_back("INVALID");

    metadata["channel_names"] = channel_names;
    metadata["num_channels"] = channel_names.size();

    // markerTypes
    metadata["markerTypes"] = nlohmann::json::object();

    for (int i = 0; i < MARKERTYPE_COUNT; ++i) {
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << (i + 1);
        metadata["markerTypes"][oss.str()] = MarkerTypeNames[i];
    }

    metadata["pid"] = 1;
    metadata["start_time"] = 0;
    metadata["tid"] = 0;

    fs::path metadata_dir = base_dir / ("metadata.json");

    std::ofstream file(metadata_dir);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing.\n");
        return -1;
    }

    // Dump JSON into file
    file << metadata.dump(4);

    // Close the file
    file.close();

    return 0;
}

/**
 * A function for extracting the TraCR data from the Device to Host
 */
template <typename DeviceRunnerT, typename RuntimeT>
int StoreTracrData(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    static_assert(
        std::is_trivially_copyable_v<TraCR::Payload>, "TraCR::Payload must be trivially copyable for raw binary dump"
    );

    if (runtime.get_tracr_data() == nullptr) {
        LOG_ERROR("runtime.tracrData_ is a nullptr");
        return -1;
    }

    if (runtime.get_tracr_data_sizes() == nullptr) {
        LOG_ERROR("runtime.tracrDataSizes_ is a nullptr");
        return -1;
    }

    if (runtime.get_aicpu_thread_num() <= 0) {
        LOG_ERROR("runtime.aicpu_thread_num is zero or negative: %d", runtime.get_aicpu_thread_num());
        return -1;
    }

    // Download the tracrData_ from Device to Host
    size_t size = sizeof(TraCR::Payload) * TraCR::CAPACITY * runtime.get_aicpu_thread_num();
    std::vector<TraCR::Payload> tracrData(TraCR::CAPACITY * runtime.get_aicpu_thread_num());
    int rc = device_runner->copy_from_device(
        reinterpret_cast<void *>(tracrData.data()), reinterpret_cast<void *>(runtime.get_tracr_data()), size
    );
    if (rc != 0) {
        LOG_ERROR("device_runner->copy_from_device 'tracrData' failed rc=%d", rc);
        return rc;
    }

    // Download the tracrDataSizes_ from Device to Host
    size = sizeof(size_t) * runtime.get_aicpu_thread_num();
    std::vector<size_t> tracrDataSizes(runtime.get_aicpu_thread_num());
    rc = device_runner->copy_from_device(
        reinterpret_cast<void *>(tracrDataSizes.data()), reinterpret_cast<void *>(runtime.get_tracr_data_sizes()), size
    );
    if (rc != 0) {
        LOG_ERROR("device_runner->copy_from_device 'tracrDataSizes' failed rc=%d", rc);
        return rc;
    }

    // Now, store the traces into '~/ascend/tracr/'
    tracr_dir =
        "~/ascend/tracr_" + std::to_string(sampleID++) + "/proc." + std::to_string(1000 + device_runner->device_id());
    rc = TracrData2BTS(tracrData.data(), tracrDataSizes.data(), runtime.get_aicpu_thread_num());
    if (rc != 0) {
        LOG_ERROR("TracrData2BTS() failed");
        return rc;
    }

    // Free device TraCR memory data placeholder
    device_runner->free_tensor(runtime.get_tracr_data());
    device_runner->free_tensor(runtime.get_tracr_data_sizes());

    rc = StoreTracrMetaData(runtime);
    if (rc != 0) {
        LOG_ERROR("StoreTracrMetaData failed: %d", rc);
        return rc;
    }

    return 0;
}

/**
 * A method for allocating memory on the device
 *
 * Polymorphic to A2A3 and A5 (should be)
 */
template <typename DeviceRunnerT, typename RuntimeT>
int DevAllocTraCR(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    const size_t size = sizeof(TraCR::Payload) * runtime.get_aicpu_thread_num() * TraCR::CAPACITY;
    // LOG_INFO_V9("Device alloc start of size=%u, %p", size, runtime.get_tracr_data());
    runtime.set_tracr_data(device_runner->allocate_tensor(size));
    if (runtime.get_tracr_data() == nullptr) {
        LOG_ERROR("runtime.tracrData_: alloc %zu bytes failed", size);
        return -1;
    }
    // LOG_INFO_V9("Device alloc start of size=%u, %p", size, runtime.get_tracr_data());
    runtime.set_tracr_data_sizes(device_runner->allocate_tensor(runtime.get_aicpu_thread_num() * sizeof(size_t)));
    if (runtime.get_tracr_data_sizes() == nullptr) {
        const size_t sizes_bytes = runtime.get_aicpu_thread_num() * sizeof(size_t);
        LOG_ERROR("runtime.tracrDataSizes_: alloc %zu bytes failed", sizes_bytes);
        device_runner->free_tensor(runtime.get_tracr_data());
        runtime.set_tracr_data(nullptr);
        return -1;
    }
    return 0;
}