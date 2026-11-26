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

#include "host/l0_capture_collector.h"

#include <dlfcn.h>

#include <driver/ascend_hal_base.h>
#include <runtime/rt.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <utility>

#include "common/acl_hal_device.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "host/aligned_allocation.h"

// The extra 64 KiB is allocation headroom for records crossing the ring boundary.
static constexpr uint32_t kL0BufferLength = L0CaptureSession::kRingLength;
static constexpr uint32_t kL0BufferCapacity = L0CaptureSession::kRingCapacity;
static constexpr uint64_t kL0MaxTotalBufferBytes = 4ull << 30;
static constexpr std::array<unsigned int, 6> kL0MonitoredClusters = {0, 9, 17, 18, 27, 35};
static constexpr unsigned int kL0ChannelFirst = 11;
static constexpr unsigned int kL0SubCoresPerCluster = 3;

struct L0BiuPerfUserData {
    uint32_t header_size;
    uint32_t mode;
    uint32_t sub_core;
    uint32_t cluster;
};
static_assert(sizeof(L0BiuPerfUserData) == 16);

// halResMap also returns addresses for unavailable blocks. Their L0 registers
// read back as all ones, including the writable base/length/enable registers.
// A large WPTR alone is not enough to identify such a slot.

L0CaptureSession::~L0CaptureSession() { reset(); }

int L0CaptureSession::start_channels() {
    channels_started_.reserve(kL0MonitoredClusters.size() * kL0SubCoresPerCluster);
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
            if (get_info(static_cast<uint32_t>(hal_device_id_), module, INFO_TYPE_FREQUE, &mhz) != 0 || mhz < 100 ||
                mhz > 10000)
                return 0;
            return static_cast<uint64_t>(mhz) * 1000000ull;
        };
        aic_frequency_hz_ = frequency_hz(MODULE_TYPE_AICORE);
        aiv_frequency_hz_ = frequency_hz(MODULE_TYPE_VECTOR_CORE);
    }
    for (size_t group = 0; group < kL0MonitoredClusters.size(); ++group) {
        for (unsigned int sub = 0; sub < kL0SubCoresPerCluster; ++sub) {
            L0BiuPerfUserData user{sizeof(L0BiuPerfUserData), 0, sub, kL0MonitoredClusters[group]};
            prof_start_para para{};
            para.channel_type = static_cast<PROF_CHANNEL_TYPE>(0);
            para.sample_period = 0;
            para.real_time = 1;
            para.user_data = &user;
            para.user_data_size = sizeof(user);
            const unsigned int channel = kL0ChannelFirst + group * kL0SubCoresPerCluster + sub;
            const int rc = prof_start_(static_cast<unsigned int>(hal_device_id_), channel, &para);
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
            const int rc = prof_stop_(static_cast<unsigned int>(hal_device_id_), *it);
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

int L0CaptureSession::release_buffers(bool abandon) noexcept {
    int rc = 0;
    if (!abandon && free_) {
        const auto release = [&](void *ptr) {
            if (ptr != nullptr && rc == 0) rc = free_(ptr);
        };
        release(snapshots_);
        release(end_snapshots_);
        release(addr_table_);
        for (void *buffer : buffers_)
            release(buffer);
    }
    snapshots_ = nullptr;
    end_snapshots_ = nullptr;
    addr_table_ = nullptr;
    buffers_.clear();
    return rc;
}

int L0CaptureSession::reset(bool abandon_device_resources) noexcept {
    if (!active()) return 0;
    if (channels_unresolved_) abandon_device_resources = true;
    int rc = 0;
    // A poisoned device can block inside prof_stop. Its channels and buffers
    // remain owned by that device generation until the caller resets it.
    if (abandon_device_resources) {
        if (!channels_started_.empty()) channels_unresolved_ = true;
        release_channel_handles();
    } else {
        rc = stop_channels();
        if (rc != 0) abandon_device_resources = true;
    }
    const int free_rc = release_buffers(abandon_device_resources);
    if (free_rc != 0) device_io_failed_ = true;
    configured_snapshots_.clear();
    terminal_snapshots_.clear();
    hal_device_id_ = -1;
    run_epoch_ = 0;
    aic_frequency_hz_ = 0;
    aiv_frequency_hz_ = 0;
    reg_addr_table_ = 0;
    free_ = {};
    return rc != 0 ? rc : free_rc;
}

int L0CaptureSession::arm(
    int device_id, int core_count, uint64_t reg_addr_table, uint64_t run_epoch, Alloc alloc, Free free
) {
    if (active() || channels_unresolved_ || device_io_failed_ || device_id < 0 || core_count <= 0 ||
        core_count > PLATFORM_MAX_CORES ||
        static_cast<uint64_t>(core_count) * kL0BufferCapacity > kL0MaxTotalBufferBytes || reg_addr_table == 0 ||
        !alloc || !free) {
        return -1;
    }
    hal_device_id_ = static_cast<int>(pto::acl_to_hal_device_id(device_id));
    run_epoch_ = run_epoch;
    reg_addr_table_ = reg_addr_table;
    free_ = std::move(free);
    buffers_.reserve(static_cast<size_t>(core_count));
    for (int i = 0; i < core_count; ++i) {
        void *buffer = alloc(kL0BufferCapacity);
        if (buffer == nullptr) {
            LOG_ERROR("L0: failed to initialize GM buffer %d/%d", i, core_count);
            const int rc = reset();
            return rc != 0 ? rc : -1;
        }
        buffers_.push_back(buffer);
        if (const int rc = rtMemset(buffer, kL0BufferCapacity, 0xDD, kL0BufferCapacity); rc != 0) {
            LOG_ERROR("L0: rtMemset GM buffer %d/%d failed: %d", i, core_count, rc);
            device_io_failed_ = true;
            reset(/*abandon_device_resources=*/true);
            return rc;
        }
    }

    const size_t table_bytes = buffers_.size() * sizeof(uint64_t);
    addr_table_ = alloc(table_bytes);
    snapshots_ = alloc(buffers_.size() * sizeof(L0CoreRegSnapshot));
    end_snapshots_ = alloc(buffers_.size() * sizeof(L0CoreEndSnapshot));
    if (addr_table_ == nullptr || snapshots_ == nullptr || end_snapshots_ == nullptr) {
        const int rc = reset();
        return rc != 0 ? rc : -1;
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
        kL0BufferLength,
        kL0BufferCapacity,
        0
    };
}

int L0CaptureSession::read_terminal_snapshot() {
    if (!active() || device_io_failed_ || end_snapshots_ == nullptr || snapshots_ == nullptr) return -1;
    // The config snapshot is read back here, at terminal time, so the config
    // launch itself needs no sync: the config payload has long finished by now.
    configured_snapshots_.resize(buffers_.size());
    const int config_rc = profiling_copy_from_device(
        configured_snapshots_.data(), snapshots_, configured_snapshots_.size() * sizeof(L0CoreRegSnapshot)
    );
    if (config_rc != 0) {
        device_io_failed_ = true;
        return config_rc;
    }
    terminal_snapshots_.resize(buffers_.size());
    const int terminal_rc = profiling_copy_from_device(
        terminal_snapshots_.data(), end_snapshots_, terminal_snapshots_.size() * sizeof(L0CoreEndSnapshot)
    );
    if (terminal_rc != 0) device_io_failed_ = true;
    return terminal_rc;
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

bool L0CaptureSession::terminal_range(size_t core, L0RingRange *range) const {
    if (range == nullptr || core >= configured_snapshots_.size() || core >= terminal_snapshots_.size()) return false;
    const auto &start = configured_snapshots_[core];
    const auto &end = terminal_snapshots_[core];
    *range = {};
    if (l0_slot_unavailable(start)) return end.reserved == L0_DFX_STATUS_UNAVAILABLE;
    if (end.reserved != L0_DFX_STATUS_STAGED) return false;
    *range = l0_plan_ring_range(
        start.initial_wptr, start.initial_samp_wrt, end.samp_wrt, end.wptr, kRingLength, kRingCapacity
    );
    return range->trustworthy() && range->bytes < kRingLength;
}

int L0CaptureSession::stop_producer() {
    if (!active()) return -1;
    if (channels_started_.empty()) return channels_unresolved_ ? -1 : 0;
    if (stop_channels() != 0) return -1;
    return 0;
}

// Device address from a device base pointer plus a byte offset, without
// dereferencing device memory on the host.
static inline const volatile void *l0_device_offset(const void *base, size_t off) {
    return reinterpret_cast<const volatile void *>(reinterpret_cast<uintptr_t>(base) + off);
}

int L0CaptureCollector::initialize(
    const NativeRunIdentity &identity, int device_id, int core_count, uint64_t register_table, Alloc alloc_cb,
    Free free_cb
) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_owner_ || device_io_failed()) return PTO_RUNTIME_ERR_INVALID_STATE;
    capture_owner_ = identity;
    capture_active_.store(true, std::memory_order_release);
    free_ = std::move(free_cb);
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = session_.arm(device_id, core_count, register_table, identity.run_epoch, alloc_cb, free_);
        if (rc == 0) rc = init(std::move(alloc_cb));
        if (rc == 0) return 0;
    } catch (const std::exception &error) {
        LOG_WARN("L0 setup raised an exception: %s", error.what());
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    const int release_rc = release_capture(false);
    return release_rc != 0 ? release_rc : rc;
}

int L0CaptureCollector::configure_run(
    const NativeRunIdentity &identity, const std::function<int(L0ConfigArgs &)> &launch
) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!capture_owner_ || *capture_owner_ != identity) return PTO_RUNTIME_ERR_INVALID_STATE;
    auto config = session_.config_args(reinterpret_cast<uint64_t>(shm_dev_));
    return launch(config);
}

int L0CaptureCollector::start_run(const NativeRunIdentity &identity, const std::string &output_prefix) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!capture_owner_ || *capture_owner_ != identity) return PTO_RUNTIME_ERR_INVALID_STATE;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = begin_run(std::filesystem::path(output_prefix) / "l0_dfx_capture");
        if (rc == 0) rc = session_.start_capture_channels();
        if (rc == 0) return 0;
    } catch (const std::exception &error) {
        LOG_WARN("L0 start raised an exception: %s", error.what());
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    const int release_rc = release_capture(false);
    return release_rc != 0 ? release_rc : rc;
}

int L0CaptureCollector::close_run(
    const NativeRunIdentity &identity, bool device_execution_complete, const std::function<int()> &synchronize
) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (!capture_owner_ || *capture_owner_ != identity) return 0;
    if (!device_execution_complete || device_io_failed() || session_.device_io_failed()) {
        const bool abandon = device_io_failed() || session_.device_io_failed() || synchronize() != 0;
        const int rc = release_capture(abandon);
        if (rc != 0) return rc;
        return device_io_failed() ? PTO_RUNTIME_ERR_INVALID_STATE : 0;
    }
    const int rc = session_.stop_producer();
    if (rc != 0) {
        release_capture(true);
        return rc;
    }
    bool terminal_ok = false;
    try {
        if (session_.read_terminal_snapshot() == 0) {
            terminal_ok = session_.verify_terminal_status() == 0;
            copy_terminal_ranges();
        }
    } catch (const std::exception &error) {
        output_failed_ = true;
        LOG_WARN("L0 collection failed: %s", error.what());
    }
    if (device_io_failed() || session_.device_io_failed()) {
        release_capture(true);
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    try {
        if (!finish_output(terminal_ok))
            LOG_WARN("L0 capture incomplete for run %llu", static_cast<unsigned long long>(identity.run_epoch));
    } catch (const std::exception &error) {
        LOG_WARN("L0 output failed: %s", error.what());
    }
    return release_capture(false);
}

void L0CaptureCollector::finalize_capture(bool abandon_device_resources) {
    std::lock_guard<std::mutex> lock(capture_mutex_);
    if (capture_owner_) release_capture(abandon_device_resources);
}

int L0CaptureCollector::release_capture(bool abandon_device_resources) {
    bool abandon = abandon_device_resources || device_io_failed() || session_.device_io_failed();
    int rc = session_.reset(abandon);
    abandon = abandon || session_.channels_unresolved() || rc != 0;
    outputs_.clear();
    output_dir_.clear();
    if (shm_dev_ != nullptr && !abandon) rc = free_(shm_dev_);
    shm_dev_ = nullptr;
    shm_host_.reset();
    initialized_ = false;
    if (rc != 0 || abandon) device_io_failed_.store(true, std::memory_order_release);
    free_ = {};
    capture_owner_.reset();
    capture_active_.store(false, std::memory_order_release);
    return rc;
}

int L0CaptureCollector::copy_to_device(volatile void *destination, const void *source, size_t bytes) {
    if (device_io_failed()) return -1;
    const int rc = profiling_copy_to_device(destination, source, bytes);
    if (rc != 0) {
        device_io_failed_.store(true, std::memory_order_release);
        output_failed_ = true;
    }
    return rc;
}

int L0CaptureCollector::copy_from_device(volatile void *destination, const volatile void *source, size_t bytes) {
    if (device_io_failed()) return -1;
    const int rc = profiling_copy_from_device(destination, source, bytes);
    if (rc != 0) {
        device_io_failed_.store(true, std::memory_order_release);
        output_failed_ = true;
    }
    return rc;
}

int L0CaptureCollector::init(Alloc alloc_cb) {
    if (initialized_ || device_io_failed_) return -1;
    const auto num_cores = static_cast<uint32_t>(session_.buffers().size());
    shm_size_ = l0_dfx_data_size(num_cores);
    shm_host_.reset(static_cast<uint8_t *>(simpler::platform::aligned_host_alloc(shm_size_)));
    if (!shm_host_) return -1;
    shm_dev_ = alloc_cb(shm_size_);
    if (!shm_dev_) return -1;
    std::memset(shm_host_.get(), 0, shm_size_);
    auto *header = reinterpret_cast<L0DataHeader *>(shm_host_.get());
    header->num_cores = num_cores;
    for (uint32_t core = 0; core < num_cores; ++core)
        l0_dfx_state(shm_host_.get(), core)->physical_core_id = core;
    if (copy_to_device(shm_dev_, shm_host_.get(), shm_size_) != 0) return -1;
    initialized_ = true;
    return 0;
}

int L0CaptureCollector::begin_run(const std::filesystem::path &output_dir) {
    if (!initialized_ || !outputs_.empty()) return -1;
    output_dir_ = output_dir / "l0_raw" / ("run_" + std::to_string(session_.run_epoch()));
    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) return -1;
    std::filesystem::remove(output_dir_ / "manifest.json", ec);
    if (ec) return -1;
    output_failed_ = false;
    outputs_.resize(session_.buffers().size());
    for (size_t core = 0; core < outputs_.size(); ++core) {
        auto &out = outputs_[core];
        out.raw.open(output_dir_ / ("core_" + std::to_string(core) + ".bin"), std::ios::binary | std::ios::trunc);
        if (!out.raw) {
            output_failed_ = true;
            return -1;
        }
    }
    return 0;
}

void L0CaptureCollector::copy_terminal_ranges() {
    if (copy_from_device(shm_host_.get(), shm_dev_, shm_size_) != 0) return;
    std::vector<uint8_t> bytes;
    for (uint32_t core = 0; core < outputs_.size(); ++core) {
        const auto *state = l0_dfx_state(shm_host_.get(), core);
        L0RingRange range{};
        if (state->producer_stopped != 1 || state->capture_error != L0_DFX_CAPTURE_OK ||
            !session_.terminal_range(core, &range)) {
            output_failed_ = true;
            LOG_WARN(
                "L0 data missing for physical core %u: stopped=%u capture_error=%u", core, state->producer_stopped,
                state->capture_error
            );
            continue;
        }
        bytes.resize(range.bytes);
        const auto *ring = session_.buffers()[core];
        if (range.first_bytes != 0 &&
            copy_from_device(bytes.data(), l0_device_offset(ring, range.start), range.first_bytes) != 0)
            return;
        if (range.second_bytes != 0 &&
            copy_from_device(bytes.data() + range.first_bytes, l0_device_offset(ring, 0), range.second_bytes) != 0)
            return;
        auto &out = outputs_[core];
        if (range.bytes != 0)
            out.raw.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(range.bytes));
        if (out.raw) {
            out.bytes = range.bytes;
            out.complete = true;
        } else output_failed_ = true;
    }
}

bool L0CaptureCollector::finish_output(bool terminal_status_ok) {
    if (!initialized_ || outputs_.size() != session_.buffers().size()) return false;
    bool complete = !output_failed_ && terminal_status_ok;
    uint64_t raw_bytes = 0;
    for (size_t core = 0; core < outputs_.size(); ++core) {
        auto &out = outputs_[core];
        out.raw.close();
        out.complete = out.complete && out.raw.good();
        const auto *state = l0_dfx_state(shm_host_.get(), static_cast<uint32_t>(core));
        if (!out.complete || state->producer_stopped != 1 || state->capture_error != L0_DFX_CAPTURE_OK)
            complete = false;
        raw_bytes += out.bytes;
    }
    std::ofstream manifest(output_dir_ / "manifest.json", std::ios::trunc);
    manifest << "{\n  \"run_epoch\": " << session_.run_epoch()
             << ",\n  \"system_counter_frequency_hz\": " << PLATFORM_PROF_SYS_CNT_FREQ
             << ",\n  \"aic_frequency_hz\": " << session_.aic_frequency_hz()
             << ",\n  \"aiv_frequency_hz\": " << session_.aiv_frequency_hz() << ",\n  \"cores\": [\n";
    for (size_t core = 0; core < outputs_.size(); ++core) {
        const auto *state = l0_dfx_state(shm_host_.get(), static_cast<uint32_t>(core));
        if (core != 0) manifest << ",\n";
        manifest << "    {\"physical_core_id\": " << core << ", \"raw_bytes\": " << outputs_[core].bytes
                 << ", \"core_type\": " << state->core_type << ", \"producer_stopped\": " << state->producer_stopped
                 << ", \"capture_error\": " << state->capture_error
                 << ", \"capture_complete\": " << (outputs_[core].complete ? "true" : "false") << "}";
    }
    manifest << "\n  ],\n  \"capture_complete\": " << (complete && raw_bytes != 0 ? "true" : "false")
             << ",\n  \"raw_transport_complete\": " << (complete ? "true" : "false") << "\n}\n";
    manifest.flush();
    if (!manifest || !complete) output_failed_ = true;
    return !output_failed_;
}
