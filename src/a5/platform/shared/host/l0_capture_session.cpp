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

#include "host/l0_capture_session.h"

#include <dlfcn.h>

#include <driver/ascend_hal_base.h>
#include <runtime/rt.h>

#include <array>
#include <utility>
#include <vector>

#include "common/unified_log.h"
#include "common/platform_config.h"
#include "common/l0_profiling.h"
#include "host/profiling_copy.h"

namespace {
// Keep enough per-core history to cover setup and short tasks before the host
// drains the ring. The extra 64 KiB is only allocation headroom for wrapping.
constexpr uint32_t kBufferLength = L0CaptureSession::kRingLength;
constexpr uint32_t kBufferCapacity = L0CaptureSession::kRingCapacity;
constexpr uint64_t kMaxTotalBufferBytes = 1ull << 30;
constexpr std::array<unsigned int, 6> kMonitoredClusters = {0, 9, 17, 18, 27, 35};
constexpr unsigned int kChannelFirst = 11;
constexpr unsigned int kSubCoresPerCluster = 3;

struct BiuPerfUserData {
    uint32_t header_size;
    uint32_t mode;
    uint32_t sub_core;
    uint32_t cluster;
};
static_assert(sizeof(BiuPerfUserData) == 16);

// halResMap also returns addresses for unavailable blocks. Their L0 registers
// read back as all ones, including the writable base/length/enable registers.
// A large WPTR alone is not enough to identify such a slot.
}  // namespace

L0CaptureSession::~L0CaptureSession() { reset(); }

int L0CaptureSession::start_channels() {
    channels_started_.reserve(kMonitoredClusters.size() * kSubCoresPerCluster);
    hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
    if (hal_handle_ == nullptr) {
        LOG_ERROR("L0: dlopen(libascend_hal.so) failed: %s", dlerror());
        return -1;
    }
    prof_start_ = reinterpret_cast<decltype(prof_start_)>(dlsym(hal_handle_, "prof_drv_start"));
    prof_stop_ = reinterpret_cast<decltype(prof_stop_)>(dlsym(hal_handle_, "prof_stop"));
    if (prof_start_ == nullptr || prof_stop_ == nullptr) {
        LOG_ERROR("L0: driver profiling symbols are unavailable");
        return -1;
    }
    // Match the CANN BIU parser: ordinary word deltas are AI core cycles,
    // while START_STAMP is in the system-counter domain. HAL reports rated
    // AI core frequencies in MHz. Missing metadata must not drop raw bytes.
    auto get_info = reinterpret_cast<decltype(&halGetDeviceInfo)>(dlsym(hal_handle_, "halGetDeviceInfo"));
    if (get_info != nullptr) {
        const auto frequency_hz = [&](int32_t module) -> uint64_t {
            int64_t mhz = 0;
            if (get_info(static_cast<uint32_t>(device_id_), module, INFO_TYPE_FREQUE, &mhz) != 0 || mhz < 100 ||
                mhz > 10000)
                return 0;
            return static_cast<uint64_t>(mhz) * 1000000ull;
        };
        aic_frequency_hz_ = frequency_hz(MODULE_TYPE_AICORE);
        aiv_frequency_hz_ = frequency_hz(MODULE_TYPE_VECTOR_CORE);
    }
    for (size_t group = 0; group < kMonitoredClusters.size(); ++group) {
        for (unsigned int sub = 0; sub < kSubCoresPerCluster; ++sub) {
            BiuPerfUserData user{sizeof(BiuPerfUserData), 0, sub, kMonitoredClusters[group]};
            prof_start_para para{};
            para.channel_type = static_cast<PROF_CHANNEL_TYPE>(0);
            para.sample_period = 0;
            para.real_time = 1;
            para.user_data = &user;
            para.user_data_size = sizeof(user);
            const unsigned int channel = kChannelFirst + group * kSubCoresPerCluster + sub;
            const int rc = prof_start_(static_cast<unsigned int>(device_id_), channel, &para);
            if (rc != 0) {
                LOG_ERROR("L0: prof_drv_start channel=%u failed: %d", channel, rc);
                return rc;
            }
            channels_started_.push_back(channel);
        }
    }
    return 0;
}

int L0CaptureSession::stop_channels() noexcept {
    int result = 0;
    if (prof_stop_ != nullptr) {
        for (auto it = channels_started_.rbegin(); it != channels_started_.rend(); ++it) {
            const int rc = prof_stop_(static_cast<unsigned int>(device_id_), *it);
            if (rc != 0) {
                LOG_ERROR("L0: prof_stop channel=%u failed: %d", *it, rc);
                result = -1;
                // A failed stop may mean the device is poisoned; leave the
                // remaining channels for the device reset.
                break;
            }
        }
    }
    release_channel_handles();
    if (result != 0) channels_unresolved_ = true;
    return result;
}

void L0CaptureSession::release_channel_handles() noexcept {
    channels_started_.clear();
    prof_start_ = nullptr;
    prof_stop_ = nullptr;
    if (hal_handle_ != nullptr) dlclose(hal_handle_);
    hal_handle_ = nullptr;
}

void L0CaptureSession::release_buffers(bool abandon) noexcept {
    if (!abandon && free_) {
        if (snapshots_ != nullptr) (void)free_(snapshots_);
        if (end_snapshots_ != nullptr) (void)free_(end_snapshots_);
        if (addr_table_ != nullptr) (void)free_(addr_table_);
        for (void *buffer : buffers_)
            if (buffer != nullptr) (void)free_(buffer);
    }
    snapshots_ = nullptr;
    end_snapshots_ = nullptr;
    addr_table_ = nullptr;
    buffers_.clear();
}

void L0CaptureSession::reset(bool abandon_device_resources) noexcept {
    if (!active()) return;
    if (channels_unresolved_) abandon_device_resources = true;
    // A poisoned device can block inside prof_stop. Its channels and buffers
    // remain owned by that device generation until the caller resets it.
    if (abandon_device_resources) {
        if (!channels_started_.empty()) channels_unresolved_ = true;
        release_channel_handles();
    } else if (stop_channels() != 0) {
        abandon_device_resources = true;
    }
    release_buffers(abandon_device_resources);
    configured_snapshots_.clear();
    terminal_snapshots_.clear();
    device_id_ = -1;
    run_epoch_ = 0;
    aic_frequency_hz_ = 0;
    aiv_frequency_hz_ = 0;
    reg_addr_table_ = 0;
    output_prefix_.clear();
    alloc_ = {};
    free_ = {};
}

int L0CaptureSession::arm(
    int device_id, int core_count, uint64_t reg_addr_table, uint64_t run_epoch, const std::string &output_prefix,
    Alloc alloc, Free free
) {
    if (active() || channels_unresolved_ || device_io_failed_ || device_id < 0 || core_count <= 0 ||
        core_count > PLATFORM_MAX_CORES || static_cast<uint64_t>(core_count) * kBufferCapacity > kMaxTotalBufferBytes ||
        reg_addr_table == 0 || output_prefix.empty() || !alloc || !free) {
        return -1;
    }
    device_id_ = device_id;
    run_epoch_ = run_epoch;
    reg_addr_table_ = reg_addr_table;
    output_prefix_ = output_prefix;
    alloc_ = std::move(alloc);
    free_ = std::move(free);
    buffers_.reserve(static_cast<size_t>(core_count));
    for (int i = 0; i < core_count; ++i) {
        void *buffer = alloc_(kBufferCapacity);
        if (buffer == nullptr) {
            LOG_ERROR("L0: failed to initialize GM buffer %d/%d", i, core_count);
            reset();
            return -1;
        }
        buffers_.push_back(buffer);
        if (const int rc = rtMemset(buffer, kBufferCapacity, 0xDD, kBufferCapacity); rc != 0) {
            LOG_ERROR("L0: rtMemset GM buffer %d/%d failed: %d", i, core_count, rc);
            device_io_failed_ = true;
            reset(/*abandon_device_resources=*/true);
            return rc;
        }
    }

    const size_t table_bytes = buffers_.size() * sizeof(uint64_t);
    addr_table_ = alloc_(table_bytes);
    snapshots_ = alloc_(buffers_.size() * sizeof(L0CoreRegSnapshot));
    end_snapshots_ = alloc_(buffers_.size() * sizeof(L0CoreEndSnapshot));
    if (addr_table_ == nullptr || snapshots_ == nullptr || end_snapshots_ == nullptr) {
        reset();
        return -1;
    }
    std::vector<uint64_t> addresses;
    addresses.reserve(buffers_.size());
    for (void *buffer : buffers_)
        addresses.push_back(reinterpret_cast<uint64_t>(buffer));
    int io_rc = profiling_copy_to_device(addr_table_, addresses.data(), table_bytes);
    if (io_rc == 0) {
        io_rc = rtMemset(
            snapshots_, buffers_.size() * sizeof(L0CoreRegSnapshot), 0, buffers_.size() * sizeof(L0CoreRegSnapshot)
        );
    }
    if (io_rc == 0) {
        io_rc = rtMemset(
            end_snapshots_, buffers_.size() * sizeof(L0CoreEndSnapshot), 0, buffers_.size() * sizeof(L0CoreEndSnapshot)
        );
    }
    if (io_rc != 0) {
        LOG_ERROR("L0: device initialization I/O failed: %d", io_rc);
        device_io_failed_ = true;
        reset(/*abandon_device_resources=*/true);
        return io_rc;
    }
    return 0;
}

int L0CaptureSession::start_capture_channels() {
    if (!active() || !channels_started_.empty() || hal_handle_ != nullptr) return -1;
    return start_channels();
}

L0ConfigArgs L0CaptureSession::config_args(uint64_t dfx_shm) const {
    return L0ConfigArgs{
        reg_addr_table_,
        reinterpret_cast<uint64_t>(addr_table_),
        reinterpret_cast<uint64_t>(snapshots_),
        reinterpret_cast<uint64_t>(end_snapshots_),
        dfx_shm,
        run_epoch_,
        static_cast<uint32_t>(buffers_.size()),
        kBufferLength,
        kBufferCapacity,
        0
    };
}

int L0CaptureSession::read_terminal_snapshot() {
    if (!active() || end_snapshots_ == nullptr || snapshots_ == nullptr) return -1;
    // The config snapshot is read back here, at terminal time, so the config
    // launch itself needs no sync: the config payload has long finished by now.
    configured_snapshots_.resize(buffers_.size());
    if (profiling_copy_from_device(
            configured_snapshots_.data(), snapshots_, configured_snapshots_.size() * sizeof(L0CoreRegSnapshot)
        ) != 0)
        return -1;
    terminal_snapshots_.resize(buffers_.size());
    return profiling_copy_from_device(
        terminal_snapshots_.data(), end_snapshots_, terminal_snapshots_.size() * sizeof(L0CoreEndSnapshot)
    );
}

int L0CaptureSession::verify_terminal_status() const {
    if (configured_snapshots_.size() != terminal_snapshots_.size()) return -1;
    int result = 0;
    for (size_t core = 0; core < configured_snapshots_.size(); ++core) {
        const uint32_t expected =
            l0_slot_unavailable(configured_snapshots_[core]) ? L0_DFX_STATUS_UNAVAILABLE : L0_DFX_STATUS_STAGED;
        if (terminal_snapshots_[core].reserved == expected) continue;
        LOG_ERROR("L0 terminal core=%zu status=%u expected=%u", core, terminal_snapshots_[core].reserved, expected);
        result = -1;
    }
    return result;
}

int L0CaptureSession::stop_producer() {
    if (!active()) return -1;
    if (channels_started_.empty()) return channels_unresolved_ ? -1 : 0;
    if (stop_channels() != 0) return -1;
    return 0;
}
