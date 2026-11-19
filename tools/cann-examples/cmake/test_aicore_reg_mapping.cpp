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
#include "aicore_reg_mapping.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(expr)                                            \
    do {                                                       \
        if (!(expr)) {                                         \
            std::fprintf(stderr, "CHECK failed: %s\n", #expr); \
            std::abort();                                      \
        }                                                      \
    } while (0)

namespace {

constexpr uint64_t kFakeBase = 0x100000000ULL;
constexpr uint64_t kFakeStride = 0x400000ULL;
uint32_t mapped = 0;
uint32_t unmapped = 0;
uint32_t memctl_device = UINT32_MAX;

uint64_t Base(uint32_t resource) { return kFakeBase + resource * kFakeStride; }

}  // namespace

extern "C" drvError_t halResMap(unsigned int device, res_map_info *info, unsigned long *va, unsigned int *len) {
    CHECK(device == 2);
    CHECK(info->target_proc_type == PROCESS_CP1 && info->res_type == RES_AICORE);
    CHECK(info->res_id == mapped && mapped < 36);
    *va = Base(mapped++);
    *len = 0x300000;
    return static_cast<drvError_t>(0);
}

extern "C" drvError_t halResUnmap(unsigned int device, res_map_info *info) {
    CHECK(device == 2 && info->res_id == unmapped);
    ++unmapped;
    return static_cast<drvError_t>(0);
}

extern "C" drvError_t halMemCtl(int, void *input, size_t, void *output, size_t *) {
    const auto *in = static_cast<const AddrMapInPara *>(input);
    CHECK(in->addr_type == ADDR_MAP_TYPE_REG_AIC_CTRL);
    memctl_device = in->devid;
    static_cast<AddrMapOutPara *>(output)->ptr = kFakeBase;
    return static_cast<drvError_t>(0);
}

int main() {
    setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3", 1);
    setenv("SIMPLER_SMOKE_ARCH", "a5", 1);
    {
        cann_examples::AicoreRegMapping mapping;
        CHECK(mapping.Init(0));
        const auto &bases = mapping.Bases();
        CHECK(mapping.AicCoreCount() == 36 && bases.size() == 108);
        CHECK(bases[0] == Base(0) && bases[17] == Base(17));
        CHECK(bases[54] == Base(18) && bases[71] == Base(35));
        CHECK(bases[18] == Base(0) + 0x100000 && bases[53] == Base(17) + 0x200000);
        CHECK(bases[72] == Base(18) + 0x100000 && bases[107] == Base(35) + 0x200000);
    }
    CHECK(mapped == 36 && unmapped == 36);

    setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6", 1);
    setenv("SIMPLER_SMOKE_ARCH", "a2a3", 1);
    {
        cann_examples::AicoreRegMapping mapping;
        CHECK(mapping.Init(1));
        CHECK(memctl_device == 6);
        CHECK(mapping.AicCoreCount() == 25 && mapping.Bases().size() == 25);
        CHECK(mapping.Bases()[24] == kFakeBase + 24 * 8ULL * 1024 * 1024);
    }
    std::puts("AICore register layout and ACL-to-HAL remapping: OK");
}
