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

#include <cstdint>

enum PROF_CHANNEL_TYPE { PROF_CHANNEL_TEST = 0 };
struct prof_start_para {
    PROF_CHANNEL_TYPE channel_type;
    uint64_t sample_period;
    uint32_t real_time;
    void *user_data;
    uint32_t user_data_size;
};
constexpr int32_t MODULE_TYPE_AICORE = 0;
constexpr int32_t MODULE_TYPE_VECTOR_CORE = 1;
constexpr int32_t INFO_TYPE_FREQUE = 0;
extern "C" int halGetDeviceInfo(uint32_t device, int32_t module, int32_t info, int64_t *value);
