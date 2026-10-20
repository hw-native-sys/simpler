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

#include "aicore_pmu_lease.h"

#include <dlfcn.h>

#include <cstring>
#include <type_traits>
#include <vector>

#include "common/acl_hal_device.h"
#include "common/unified_log.h"

namespace {

// Driver prof-channel surface. These are driver HAL entry points, not part of
// the CANN toolkit API this runtime links, so they are resolved at run time and
// their absence is a state rather than a link error.
constexpr unsigned int kChannelFftsProfileTask = 53;

// prof_start_para.channel_type; the FFTS profile channel is served by TS.
constexpr std::uint32_t kProfTsType = 0;
// prof_start_para.real_time
constexpr std::uint32_t kProfRealTime = 1;

// Collection-object mask: 0 none, 1 AIC, 2 AIV, 3 both. Zero is what keeps the
// device from emitting records this process would never drain.
constexpr std::uint32_t kCfgModeNone = 0;
// aiEventCfg[].type bits: 0 task base, 1 sample base, 2 block task, 3 sub task.
// A zero type is rejected by START with -2.
constexpr std::uint32_t kAiTypeTaskBaseSubTask = 9;
constexpr std::uint32_t kPmuEventMaxNum = 8;
constexpr std::uint32_t kFftsProfModes = 2;

// The driver copies this blob into a PROF_HDC_CMD_START message of at most
// PROF_USER_DATA_LEN (128) bytes and forwards it to the device without reading
// any field, so the layout below only has to agree with the device firmware.
struct AiCoreProfileConfig {
    std::uint32_t type;
    std::uint32_t period;
    std::uint32_t core_mask;
    std::uint32_t event_num;
    std::uint16_t event[kPmuEventMaxNum];
};

struct FftsProfileConfig {
    std::uint32_t cfg_mode;
    AiCoreProfileConfig ai_event_cfg[kFftsProfModes];
    std::uint16_t tag;
    std::uint16_t aic_scale;
};

static_assert(
    std::is_trivially_copyable_v<FftsProfileConfig> && std::is_standard_layout_v<FftsProfileConfig>,
    "FftsProfileConfig is memcpy'd into a driver message"
);
static_assert(sizeof(FftsProfileConfig) == 72, "FftsProfileConfig must match the driver's wire size");

struct ProfStartPara {
    std::uint32_t channel_type;
    std::uint32_t sample_period;
    std::uint32_t real_time;
    void *user_data;
    std::uint32_t user_data_size;
};

using ProfDrvStartFn = int (*)(unsigned int, unsigned int, ProfStartPara *);
using ProfStopFn = int (*)(unsigned int, unsigned int);
using ProfDataFlushFn = int (*)(unsigned int, unsigned int, unsigned int *);
using ProfChannelReadFn = int (*)(unsigned int, unsigned int, char *, unsigned int);

struct ProfApi {
    ProfDrvStartFn start = nullptr;
    ProfStopFn stop = nullptr;
    ProfDataFlushFn flush = nullptr;
    ProfChannelReadFn read = nullptr;

    bool usable() const { return start != nullptr && stop != nullptr; }
};

/// Resolved once per process; an unresolved table stays unresolved.
const ProfApi &prof_api() {
    static const ProfApi api = [] {
        ProfApi resolved;
        if (dlsym(RTLD_DEFAULT, "prof_drv_start") == nullptr) {
            static const char *const kHalLibs[] = {
                "libascend_hal.so",
                "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
            };
            for (const char *path : kHalLibs) {
                if (dlopen(path, RTLD_LAZY | RTLD_GLOBAL) != nullptr) break;
            }
        }
        resolved.start = reinterpret_cast<ProfDrvStartFn>(dlsym(RTLD_DEFAULT, "prof_drv_start"));
        resolved.stop = reinterpret_cast<ProfStopFn>(dlsym(RTLD_DEFAULT, "prof_stop"));
        resolved.flush = reinterpret_cast<ProfDataFlushFn>(dlsym(RTLD_DEFAULT, "halProfDataFlush"));
        resolved.read = reinterpret_cast<ProfChannelReadFn>(dlsym(RTLD_DEFAULT, "prof_channel_read"));
        return resolved;
    }();
    return api;
}

// prof_drv_start's own documented kernel-side code for a channel another
// process holds.
constexpr int kProfChannelOccupied = -10;

void fill_config(FftsProfileConfig *cfg) {
    std::memset(cfg, 0, sizeof(*cfg));
    cfg->cfg_mode = kCfgModeNone;
    cfg->tag = 0;        // enable immediately
    cfg->aic_scale = 0;  // all cores
    for (std::uint32_t mode = 0; mode < kFftsProfModes; mode++) {
        cfg->ai_event_cfg[mode].type = kAiTypeTaskBaseSubTask;
        // `period` and `core_mask` are sample-base fields, and a mask paired
        // with cfg_mode=0 selects nothing in any case. START accepts this
        // config, the one naming every core with a real event table, and the
        // one differing only in event_num.
        cfg->ai_event_cfg[mode].period = 0;
        cfg->ai_event_cfg[mode].core_mask = 0;
        cfg->ai_event_cfg[mode].event_num = 0;
    }
}

/// Bytes the channel still held at release. Zero at cfgMode=0.
long drain_channel(const ProfApi &api, unsigned int device) {
    if (api.flush == nullptr || api.read == nullptr) return 0;
    unsigned int pending = 0;
    (void)api.flush(device, kChannelFftsProfileTask, &pending);
    std::vector<char> buffer(1U << 16);
    long drained = 0;
    // Bounded so a channel that keeps producing cannot hold up teardown.
    constexpr int kMaxReads = 256;
    for (int i = 0; i < kMaxReads; i++) {
        const int got =
            api.read(device, kChannelFftsProfileTask, buffer.data(), static_cast<unsigned int>(buffer.size()));
        if (got <= 0) break;
        drained += got;
    }
    return drained;
}

}  // namespace

AicorePmuLease::~AicorePmuLease() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!owners_.empty() && state_ == State::kHeld) {
        const ProfApi &api = prof_api();
        if (api.stop != nullptr) (void)api.stop(static_cast<unsigned int>(hal_device_id_), kChannelFftsProfileTask);
    }
    owners_.clear();
    hal_device_id_ = -1;
    state_ = State::kAbsent;
}

bool AicorePmuLease::acquire(int device_id, std::uint64_t run_epoch) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (device_id < 0) {
        state_ = State::kRefused;
        why_ = "no device bound";
        LOG_WARN("AicorePmuLease: %s; PMU collection runs unprotected", why_.c_str());
        return false;
    }
    if (!owners_.empty()) {
        if (device_id != device_id_) {
            LOG_ERROR("AicorePmuLease: held for device %d, refusing acquire for device %d", device_id_, device_id);
            return false;
        }
        owners_.insert(run_epoch);
        return state_ == State::kHeld;
    }

    const ProfApi &api = prof_api();
    if (!api.usable()) {
        state_ = State::kAbsent;
        why_ = "driver prof symbols unavailable";
        LOG_INFO("AicorePmuLease: %s; PMU collection runs unprotected", why_.c_str());
        return false;
    }

    FftsProfileConfig cfg;
    fill_config(&cfg);
    ProfStartPara para;
    std::memset(&para, 0, sizeof(para));
    para.channel_type = kProfTsType;
    para.sample_period = 0;
    para.real_time = kProfRealTime;
    para.user_data = &cfg;
    para.user_data_size = static_cast<std::uint32_t>(sizeof(cfg));

    // Every direct driver HAL call indexes the driver-visible space, which
    // ASCEND_RT_VISIBLE_DEVICES can remap away from the logical id the caller
    // named. Resolved once and stored: holding the channel for the wrong
    // device protects nothing while reporting success.
    const int hal_device = static_cast<int>(pto::acl_to_hal_device_id(device_id));
    const int rc = api.start(static_cast<unsigned int>(hal_device), kChannelFftsProfileTask, &para);
    if (rc == 0) {
        device_id_ = device_id;
        hal_device_id_ = hal_device;
        owners_.insert(run_epoch);
        state_ = State::kHeld;
        why_.clear();
        // Stated on every take, so a collection's protection is readable from
        // the run's own log rather than inferred from counters that did not
        // reset.
        LOG_INFO(
            "AicorePmuLease: holding prof channel 53 on device %d (hal %d) for PMU collection", device_id, hal_device
        );
        return true;
    }
    if (rc == kProfChannelOccupied) {
        state_ = State::kOccupied;
        why_ = "prof channel 53 is held by another process on this device";
        LOG_WARN(
            "AicorePmuLease: %s (device %d, hal %d); a profiler such as msprof holding it means PMU counters "
            "may be reset mid-collection",
            why_.c_str(), device_id, hal_device
        );
        return false;
    }
    state_ = State::kRefused;
    why_ = "prof_drv_start returned " + std::to_string(rc);
    LOG_WARN(
        "AicorePmuLease: %s (device %d, hal %d); PMU collection runs unprotected", why_.c_str(), device_id, hal_device
    );
    return false;
}

void AicorePmuLease::release(std::uint64_t run_epoch) {
    std::lock_guard<std::mutex> guard(mutex_);
    // Only an epoch that holds a reference drops one. Every other call is a
    // path that had nothing to give back: a run refused before the acquire
    // ran, a run whose acquire failed, or a second release of the same run.
    if (owners_.erase(run_epoch) == 0) return;
    if (!owners_.empty()) return;

    const int device = device_id_;
    const int hal_device = hal_device_id_;
    device_id_ = -1;
    hal_device_id_ = -1;
    if (state_ != State::kHeld) {
        state_ = State::kAbsent;
        return;
    }
    state_ = State::kAbsent;

    const ProfApi &api = prof_api();
    const long drained = drain_channel(api, static_cast<unsigned int>(hal_device));
    if (drained > 0) {
        LOG_WARN(
            "AicorePmuLease: drained %ld bytes from prof channel 53 on device %d (hal %d) at release; the lease "
            "is taken with cfgMode=0, which is expected to emit nothing, so this configuration no longer "
            "holds and the host ring may have been overwriting records",
            drained, device, hal_device
        );
    }
    if (api.stop != nullptr) {
        const int rc = api.stop(static_cast<unsigned int>(hal_device), kChannelFftsProfileTask);
        if (rc != 0) LOG_WARN("AicorePmuLease: prof_stop returned %d for device %d (hal %d)", rc, device, hal_device);
    }
}

AicorePmuLease::State AicorePmuLease::state() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_;
}

const char *AicorePmuLease::state_name() const {
    std::lock_guard<std::mutex> guard(mutex_);
    switch (state_) {
    case State::kHeld:
        return "held";
    case State::kOccupied:
        return "occupied";
    case State::kRefused:
        return "refused";
    case State::kAbsent:
    default:
        return "absent";
    }
}

std::string AicorePmuLease::why() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return why_;
}

bool AicorePmuLease::held() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_ == State::kHeld;
}
