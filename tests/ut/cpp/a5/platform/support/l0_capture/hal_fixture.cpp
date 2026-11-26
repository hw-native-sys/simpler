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

#include "driver/ascend_hal_base.h"

static int open_channels = 0;

extern "C" int l0_test_hal_open_channels() { return open_channels; }

extern "C" int halGetDeviceInfo(uint32_t device, int32_t module, int32_t, int64_t *value) {
    if (device != 5) return -1;
    *value = module == MODULE_TYPE_AICORE ? 1300 : 1400;
    return 0;
}

extern "C" int prof_drv_start(unsigned int device, unsigned int, prof_start_para *) {
    if (device != 5) return -1;
    ++open_channels;
    return 0;
}

extern "C" int prof_stop(unsigned int device, unsigned int) {
    if (device != 5) return -1;
    --open_channels;
    return 0;
}
