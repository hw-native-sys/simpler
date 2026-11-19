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

#include <driver/ascend_hal.h>
#include <driver/ascend_hal_define.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "common/acl_hal_device.h"

namespace cann_examples {

class AicoreRegMapping {
public:
    AicoreRegMapping() = default;
    AicoreRegMapping(const AicoreRegMapping &) = delete;
    AicoreRegMapping &operator=(const AicoreRegMapping &) = delete;

    ~AicoreRegMapping() {
        if (unmap_ == nullptr) return;
        for (uint32_t core = 0; core < mapped_cores_; ++core) {
            res_map_info info{};
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = core;
            const int rc = unmap_(hal_device_id_, &info);
            if (rc != 0) std::fprintf(stderr, "halResUnmap core=%u rc=%d\n", core, rc);
        }
    }

    bool Init(uint32_t device_id) {
        hal_device_id_ = static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id));
        const char *arch = std::getenv("SIMPLER_SMOKE_ARCH");
        if (arch == nullptr || std::strcmp(arch, "a2a3") == 0) {
            return InitA2A3();
        }
        if (std::strcmp(arch, "a5") == 0) {
            return InitA5();
        }
        std::fprintf(stderr, "unsupported SIMPLER_SMOKE_ARCH=%s\n", arch);
        return false;
    }

    const std::vector<uint64_t> &Bases() const { return bases_; }
    uint32_t AicCoreCount() const { return mapped_cores_; }
    uint32_t CondOffset() const { return cond_offset_; }
    uint32_t DmbOffset() const { return dmb_offset_; }

private:
    using ResMapFn = int (*)(unsigned int, res_map_info *, unsigned long *, unsigned int *);
    using ResUnmapFn = int (*)(unsigned int, res_map_info *);

    bool InitA2A3() {
        auto *fn = reinterpret_cast<int (*)(int, void *, size_t, void *, size_t *)>(dlsym(nullptr, "halMemCtl"));
        if (fn == nullptr) {
            std::fprintf(stderr, "halMemCtl not found in symbol table\n");
            return false;
        }
        AddrMapInPara in{};
        in.devid = hal_device_id_;
        in.addr_type = ADDR_MAP_TYPE_REG_AIC_CTRL;
        AddrMapOutPara out{};
        const int rc = fn(0, &in, sizeof(in), &out, nullptr);
        if (rc != 0 || out.ptr == 0) {
            std::fprintf(stderr, "halMemCtl(AIC_CTRL) rc=%d ptr=0x%llx\n", rc, out.ptr);
            return false;
        }

        constexpr uint32_t kPhysicalCores = 25;
        constexpr uint64_t kCoreStride = 8ULL * 1024 * 1024;
        bases_.reserve(kPhysicalCores);
        for (uint32_t core = 0; core < kPhysicalCores; ++core) {
            bases_.push_back(out.ptr + static_cast<uint64_t>(core) * kCoreStride);
        }
        mapped_cores_ = kPhysicalCores;
        cond_offset_ = 0x4C8;
        dmb_offset_ = 0xA0;
        return true;
    }

    bool InitA5() {
        auto *map = reinterpret_cast<ResMapFn>(dlsym(nullptr, "halResMap"));
        unmap_ = reinterpret_cast<ResUnmapFn>(dlsym(nullptr, "halResUnmap"));
        if (map == nullptr || unmap_ == nullptr) {
            std::fprintf(stderr, "halResMap/halResUnmap not found in symbol table\n");
            unmap_ = nullptr;
            return false;
        }

        // Match src/a5/platform/onboard/host/host_regs.cpp: each die has
        // [18 AIC][36 AIV] slots. get_coreid() indexes this layout, while
        // halResMap takes a compact AIC resource id in [0, 36).
        constexpr uint32_t kAicPerDie = 18;
        constexpr uint32_t kSubcoresPerAic = 3;
        constexpr uint32_t kDieSlots = kAicPerDie * kSubcoresPerAic;
        constexpr uint32_t kPhysicalCores = 2 * kAicPerDie;
        constexpr uint64_t kAivFirstOffset = 0x100000;
        constexpr uint64_t kAivSecondOffset = 0x200000;
        constexpr uint32_t kMapBytes = 0x300000;
        cond_offset_ = 0x5108;
        dmb_offset_ = 0xD0;
        bases_.resize(2 * kDieSlots);
        for (uint32_t core = 0; core < kPhysicalCores; ++core) {
            res_map_info info{};
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = core;
            unsigned long va = 0;
            uint32_t len = kMapBytes;
            const int rc = map(hal_device_id_, &info, &va, &len);
            if (rc != 0) {
                std::fprintf(stderr, "halResMap core=%u rc=%d\n", core, rc);
                return false;
            }
            ++mapped_cores_;
            if (va == 0 || len < kAivSecondOffset + cond_offset_ + sizeof(uint32_t)) {
                std::fprintf(stderr, "halResMap core=%u returned va=0x%lx len=0x%x\n", core, va, len);
                return false;
            }
            const uint32_t die_base = (core / kAicPerDie) * kDieSlots;
            const uint32_t local_core = core % kAicPerDie;
            bases_[die_base + local_core] = va;
            const uint32_t aiv_slot = die_base + kAicPerDie + 2 * local_core;
            bases_[aiv_slot] = va + kAivFirstOffset;
            bases_[aiv_slot + 1] = va + kAivSecondOffset;
        }
        return true;
    }

    std::vector<uint64_t> bases_;
    ResUnmapFn unmap_ = nullptr;
    uint32_t hal_device_id_ = 0;
    uint32_t mapped_cores_ = 0;
    uint32_t cond_offset_ = 0;
    uint32_t dmb_offset_ = 0;
};

}  // namespace cann_examples
