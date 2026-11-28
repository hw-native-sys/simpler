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

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

namespace {
enum class MaskMode { Normal, Count };
MaskMode mask_mode;
std::optional<MaskMode> executor_mask_mode;

void set_mask_norm() { mask_mode = MaskMode::Normal; }
int get_block_idx() { return 0; }
int get_subblockdim() { return 2; }
int get_subblockid() { return 0; }
int get_block_num() { return 1; }
void set_ffts_base_addr(uint64_t) {}
uintptr_t get_para_base() { return 0; }
}  // namespace

#define __DAV_VEC__ 1
#include "onboard/aicore/kernel.cpp"  // NOLINT(bugprone-suspicious-include): exercise the onboard AIV entry
#undef __DAV_VEC__

void aicore_execute(Runtime *, int, CoreType type) {
    EXPECT_EQ(type, CoreType::AIV);
    executor_mask_mode = mask_mode;
}

TEST(AivLaunch, EstablishesNormalMaskModeBeforeDispatch) {
    AicoreRuntimePacket packet{};
    mask_mode = MaskMode::Count;
    executor_mask_mode.reset();
    aicore_kernel_0_mix_aiv(reinterpret_cast<uint64_t>(packet.descriptor), 0, 0, 0, 0);
    EXPECT_EQ(executor_mask_mode, MaskMode::Normal);
}
